"""Person 4: convolutional autoencoder (CAE) experiments.

Run from the repository root.  Choose the GPU explicitly; PCI_BUS_ID makes the
index match ``nvidia-smi``::

    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
    python scripts/run_person4.py --mode smoke                    # ~1 min, no results written
    python scripts/run_person4.py --mode pilot --d 32 --seed 0    # validation only
    python scripts/run_person4.py --mode freeze                   # record the protocol
    python scripts/run_person4.py --mode final --all              # 7 d x 5 seeds via evaluate()
    python scripts/run_person4.py --mode final --d 32 --seed 0    # one official run
    python scripts/run_person4.py --mode report                   # checks + figures (whole grid)
    python scripts/merge_results.py

Modes
  smoke   Small train/val subsets.  Checks shapes, ranges, finiteness, checkpoint
          round-trip, seed determinism and that a small subset can be overfit.
          Writes nothing under results/.
  pilot   Full 54K train with early stopping on the 6K validation split.  The
          official test arrays are never bound.  Outputs carry the protocol
          fingerprint in their names (results/cae_pilot/<run>_p<fp>.json,
          figures/cae_pilot_*_p<fp>.png, checkpoints/cae/pilot/), so pilots
          under different protocols never overwrite each other.
  freeze  Writes results/cae_protocol.json (protocol + fingerprint).  Requires a
          pilot produced under the same fingerprint.
  final   Official runs through the shared evaluate().  Requires the frozen
          protocol.  Completed compatible runs are skipped; incompatible or
          inconsistent existing results stop the sweep before any training.
  report  After final runs, always over the whole d x seed grid: checkpoint
          reload check, per-run diagnostics, seed summary, figures.  Exits
          non-zero if any verification fails.

Run-level commit protocol (final): evaluate() writes its row to a temporary
file; the runner saves the checkpoint, verifies a reload, exports Person 5's
artifact (d=32, seed=0), writes results/cae_runs/<run>.json and only then
appends the row to results/results_cae.csv.  A row in that CSV therefore means
the whole run finished; an interrupted run leaves no row and is simply rerun.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import config
import evaluate as shared_eval
from data import load_data, set_seed
from models.cae import (FIXED_PROTOCOL, CAEReducer, ConvAutoencoder,
                        expected_n_params, fp32_numerics)

METHOD = "CAE"
ARTIFACT_RUN = (32, 0)            # Person 5: t-SNE + reconstruction grid
MSE_TOL = 5e-7 + 1e-9             # half of the CSV's 1e-6 rounding unit (+ float slack)
RELOAD_RTOL = 1e-5                # reload re-encoding vs saved Z_test, relative to max|Z|
SOURCE_FILES = ("models/cae.py", "scripts/run_person4.py", "evaluate.py", "data.py", "config.py")
# Code that determines training or metrics; its hash is part of the fingerprint,
# so any edit (even a comment) requires a new pilot + freeze before final runs.
PROTOCOL_CODE = ("models/cae.py", "evaluate.py", "data.py")


class IntegrityError(RuntimeError):
    """Existing results are inconsistent with the current protocol."""


# ----------------------------------------------------------------- locations
def results_csv() -> Path:
    return Path(config.RESULTS_DIR) / f"results_{METHOD.lower()}.csv"


def runs_dir() -> Path:
    return Path(config.RESULTS_DIR) / "cae_runs"


def pilot_dir() -> Path:
    return Path(config.RESULTS_DIR) / "cae_pilot"


def protocol_file() -> Path:
    return Path(config.RESULTS_DIR) / "cae_protocol.json"


def checkpoint_dir() -> Path:
    return Path(config.ROOT) / "checkpoints" / "cae"


def figure_dir() -> Path:
    return Path(config.ROOT) / "figures"


def latent_path(d: int, seed: int) -> Path:
    return Path(config.LATENT_DIR) / f"{METHOD}_d{d}_s{seed}.npz"


def run_name(d: int, seed: int) -> str:
    return f"{METHOD}_d{d}_s{seed}"


# -------------------------------------------------------------- small utils
def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Rows of a results CSV; refuses a file whose header is not the shared schema."""
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != shared_eval.COLUMNS:
            raise IntegrityError(f"{path} header {reader.fieldnames} != {shared_eval.COLUMNS}")
        return list(reader)


def append_row(path: Path, row: dict[str, Any]) -> None:
    """Append one row exactly as evaluate() would (same columns, header, csv dialect)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    if not new_file and path.stat().st_size > 0:
        with open(path, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b"\n":
                raise IntegrityError(f"{path} does not end with a newline (interrupted write?)")
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=shared_eval.COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:  # git missing is not fatal for metadata
        return ""


def first_index_per_class(y: np.ndarray, n_classes: int = 10) -> list[int]:
    """Predetermined figure samples: the first image of each class in file order."""
    return [int(np.flatnonzero(y == c)[0]) for c in range(n_classes)]


# ------------------------------------------------------------- environment
def gpu_snapshot() -> list[dict[str, str]] | None:
    query = "index,uuid,pci.bus_id,name,driver_version,utilization.gpu,memory.used,memory.total"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return None
    keys = query.split(",")
    return [dict(zip(keys, (v.strip() for v in line.split(",")))) for line in out.splitlines()
            if line.strip()]


def environment() -> dict[str, Any]:
    import sklearn
    env: dict[str, Any] = {
        "python": platform.python_version(), "platform": platform.platform(),
        "hostname": socket.gethostname(), "cpu_count": os.cpu_count(),
        "load_avg_1_5_15": list(os.getloadavg()),
        "numpy": np.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_tracked_changes": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "source_sha256": {f: _sha256(ROOT / f) for f in SOURCE_FILES if (ROOT / f).exists()},
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        uuid = str(getattr(props, "uuid", ""))
        env["gpu"] = {"name": props.name, "capability": f"{props.major}.{props.minor}",
                      "total_memory_gib": round(props.total_memory / 2**30, 2), "uuid": uuid}
        snap = gpu_snapshot() or []
        match = [g for g in snap if uuid and uuid in g.get("uuid", "")]
        if match:
            env["gpu"].update(nvidia_smi_index=match[0]["index"], pci_bus_id=match[0]["pci.bus_id"],
                              driver=match[0]["driver_version"])
    return env


def warm_up_cuda() -> None:
    """Create the CUDA context and load cuDNN/cuBLAS kernels before any timed fit().

    Otherwise the first run in a process would carry ~1 s of one-off start-up
    cost in its fit_time_s.  Building the throwaway network draws from the CPU
    RNG; fork_rng restores it, and evaluate() re-seeds afterwards in any case.
    """
    if not torch.cuda.is_available():
        return
    with torch.random.fork_rng(devices=[]), fp32_numerics():
        net = ConvAutoencoder(8).cuda()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        x = torch.zeros(256, 1, 28, 28, device="cuda")
        for _ in range(3):
            loss = F.mse_loss(net(x), x)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.inference_mode():
            net.decoder(net.encoder(x[:100])).cpu()
        torch.cuda.synchronize()
    del net, opt, x
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- protocol
def architecture_spec() -> list[str]:
    """Layer list of ConvAutoencoder with the bottleneck width written as 'd'.

    Built from module attributes rather than repr(), so it does not change with
    the torch version.
    """
    with torch.random.fork_rng(devices=[]):
        net = ConvAutoencoder(1)
    spec = []
    for name, m in net.named_modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            extra = (f", output_padding={tuple(m.output_padding)}"
                     if isinstance(m, nn.ConvTranspose2d) else "")
            spec.append(f"{name}: {type(m).__name__}({m.in_channels}->{m.out_channels}, "
                        f"kernel={tuple(m.kernel_size)}, stride={tuple(m.stride)}, "
                        f"padding={tuple(m.padding)}{extra}, bias={m.bias is not None})")
        elif isinstance(m, nn.Linear):
            width = (lambda v: "d" if v == 1 else str(v))
            spec.append(f"{name}: Linear({width(m.in_features)}->{width(m.out_features)}, "
                        f"bias={m.bias is not None})")
        elif isinstance(m, nn.Unflatten):
            spec.append(f"{name}: Unflatten{tuple(m.unflattened_size)}")
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.Flatten)):
            spec.append(f"{name}: {type(m).__name__}")
    return spec


def sweep_protocol(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Everything that must be identical for CAE rows to be pooled.

    ``overrides`` (pilot-only) replaces CAEReducer defaults.  Hardware and
    library versions are recorded per run but are not part of the fingerprint;
    ``report`` flags runs whose environment differs.
    """
    return {
        "method": METHOD,
        "architecture": architecture_spec(),
        "n_params_formula": "11137 + 3137*d",
        "code_sha256": {f: _sha256(ROOT / f) for f in PROTOCOL_CODE},
        "training": CAEReducer(1, device="cpu", **(overrides or {})).get_params(),
        "fixed": dict(FIXED_PROTOCOL),
        "data": {"split_sha256": _sha256(Path(config.SPLIT_DIR) / "split_indices.npz"),
                 "VAL_SIZE": config.VAL_SIZE, "SPLIT_SEED": config.SPLIT_SEED},
        "evaluation": {"KNN_K": config.KNN_K, "SILHOUETTE_SAMPLE": config.SILHOUETTE_SAMPLE,
                       "ENCODE_BATCH": config.ENCODE_BATCH, "columns": list(shared_eval.COLUMNS)},
    }


def fingerprint(proto: dict[str, Any] | None = None) -> str:
    blob = json.dumps(proto or sweep_protocol(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def run_status(d: int, seed: int, fp: str) -> str:
    """'todo' or 'done'; raises IntegrityError for anything that must not be skipped
    or silently duplicated."""
    name = run_name(d, seed)
    meta_path = runs_dir() / f"{name}.json"
    try:
        rows = [r for r in _read_rows(results_csv())
                if r["method"] == METHOD and int(r["d"]) == d and int(r["seed"]) == seed]
        if not rows:
            return "todo"        # metadata without a row = run never committed; rerun it
        if len(rows) > 1:
            raise IntegrityError(f"{name}: {len(rows)} duplicate rows in {results_csv()}")
        if not meta_path.exists():
            raise IntegrityError(f"{name}: row in {results_csv()} has no metadata in "
                                 f"{runs_dir()}; it was not produced by this runner")
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") != fp:
            raise IntegrityError(f"{name}: existing row was produced under protocol "
                                 f"{meta.get('fingerprint')}, expected {fp}")
        want = {c: str(meta["row"][c]) for c in shared_eval.COLUMNS}
        if want != {c: rows[0][c] for c in shared_eval.COLUMNS}:
            raise IntegrityError(f"{name}: CSV row differs from the row recorded in {meta_path}")
        return "done"
    except IntegrityError:
        raise
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise IntegrityError(f"{name}: unreadable results or metadata "
                             f"({type(e).__name__}: {e})") from e


def load_frozen_protocol(require_current: bool = True) -> dict[str, Any]:
    """The frozen protocol.  final needs the current code to match it exactly;
    report only warns, because it re-checks runs against the frozen fingerprint."""
    path = protocol_file()
    if not path.exists():
        raise IntegrityError(f"{path} not found: run a pilot, review it, then --mode freeze")
    frozen = json.loads(path.read_text())
    current = fingerprint()
    if frozen.get("fingerprint") != current:
        msg = (f"current protocol {current} != frozen {frozen.get('fingerprint')} in {path}.  "
               "Settings or protocol code changed after the freeze.")
        if require_current:
            raise IntegrityError(msg)
        print(f"WARNING: {msg}")
    return frozen


@contextlib.contextmanager
def exclusive_lock() -> Iterator[None]:
    """One writer at a time for results_cae.csv (two sweeps would race)."""
    checkpoint_dir().mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir() / ".run_person4.lock", "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another run_person4.py final/report process holds the lock.")
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


# ------------------------------------------------- Person 5 artifact (NPZ)
def verify_reconstruction_artifact(path: Path, X_test: np.ndarray, csv_mse: float, d: int,
                                   n_train: int | None = None) -> dict[str, Any]:
    """Check keys, shapes, dtypes, finiteness, [0,1] range and MSE against the CSV."""
    with np.load(path) as f:
        keys = sorted(f.files)
        arrays = {k: f[k] for k in keys}
    if keys != ["X_hat", "Z_test", "Z_train"]:
        raise IntegrityError(f"{path}: keys {keys} != ['X_hat', 'Z_test', 'Z_train']")
    Z_train, Z_test, X_hat = arrays["Z_train"], arrays["Z_test"], arrays["X_hat"]
    n_test = X_test.shape[0]
    problems = []
    if Z_train.ndim != 2 or Z_train.shape[1] != d or (n_train and Z_train.shape[0] != n_train):
        problems.append(f"Z_train shape {Z_train.shape}")
    if Z_test.shape != (n_test, d):
        problems.append(f"Z_test shape {Z_test.shape} != {(n_test, d)}")
    if X_hat.shape != (n_test, X_test.shape[1]):
        problems.append(f"X_hat shape {X_hat.shape} != {X_test.shape}")
    for k, a in arrays.items():
        if a.dtype != np.float32:
            problems.append(f"{k} dtype {a.dtype}")
        if not np.isfinite(a).all():
            problems.append(f"{k} has non-finite values")
    if X_hat.size and (X_hat.min() < 0.0 or X_hat.max() > 1.0):
        problems.append(f"X_hat range [{X_hat.min()}, {X_hat.max()}] outside [0, 1]")
    if problems:
        raise IntegrityError(f"{path}: " + "; ".join(problems))
    mse = float(np.mean((X_test - X_hat) ** 2))
    if abs(mse - float(csv_mse)) > MSE_TOL:
        raise IntegrityError(f"{path}: MSE from X_hat {mse:.8f} != CSV mse {csv_mse}")
    return {"path": str(path), "keys": keys,
            "shapes": {k: list(a.shape) for k, a in arrays.items()},
            "dtypes": {k: str(a.dtype) for k, a in arrays.items()},
            "x_hat_min": float(X_hat.min()), "x_hat_max": float(X_hat.max()),
            "mse_from_x_hat": mse, "mse_csv": float(csv_mse),
            "abs_diff": abs(mse - float(csv_mse)), "size_bytes": path.stat().st_size,
            "sha256": _sha256(path)}


def export_reconstruction_artifact(model: CAEReducer, row: dict[str, Any], X_test: np.ndarray,
                                   path: Path, n_train: int | None = None) -> dict[str, Any]:
    """Add X_hat (official test reconstructions) to evaluate()'s latent file.

    X_hat is recomputed from the same trained instance evaluate() just scored,
    the Z_test that evaluate() saved, and evaluate()'s own 1000-row batching and
    clipping.  Inference is deterministic, so it equals the array evaluate()
    used for the CSV's mse; the MSE check (within the CSV's rounding) confirms
    it.  Z_train/Z_test are rewritten unchanged; the new file is verified
    before an atomic os.replace().
    """
    with np.load(path) as f:
        if not {"Z_train", "Z_test"} <= set(f.files):
            raise IntegrityError(f"{path}: expected Z_train and Z_test, found {f.files}")
        Z_train, Z_test = f["Z_train"], f["Z_test"]
    X_hat = np.clip(shared_eval._batched(model.inverse_transform, Z_test), 0.0, 1.0)
    X_hat = np.asarray(X_hat, dtype=np.float32)

    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, Z_train=Z_train, Z_test=Z_test, X_hat=X_hat)
            fh.flush()
            os.fsync(fh.fileno())
        with np.load(tmp) as f:
            for k, a in (("Z_train", Z_train), ("Z_test", Z_test), ("X_hat", X_hat)):
                if not np.array_equal(f[k], a):
                    raise IntegrityError(f"{tmp}: {k} did not round-trip")
        verify_reconstruction_artifact(tmp, X_test, row["mse"], model.d, n_train)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return verify_reconstruction_artifact(path, X_test, row["mse"], model.d, n_train)


# ------------------------------------------------------------------ figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_histories(histories: dict[str, list[dict[str, float]]], path: Path, title: str,
                   show_train: bool = True) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, (label, hist) in enumerate(histories.items()):
        ep = [h["epoch"] for h in hist]
        color = f"C{i % 10}"
        ax.plot(ep, [h["val_mse"] for h in hist], color=color, lw=1.6, label=f"{label} val")
        if show_train:
            ax.plot(ep, [h["train_mse"] for h in hist], color=color, lw=1.0, ls="--",
                    label=f"{label} train")
        best = min(hist, key=lambda h: h["val_mse"])
        ax.plot(best["epoch"], best["val_mse"], "o", color=color, ms=4)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE per pixel (log scale)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reconstructions(originals: np.ndarray, rows: list[tuple[str, np.ndarray]], path: Path,
                         title: str, col_labels: list[str] | None = None) -> None:
    plt = _plt()
    n_rows, n_cols = 1 + len(rows), originals.shape[0]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.2 * n_cols + 1.2, 1.25 * n_rows + 0.6),
                             squeeze=False)
    for r, (label, images) in enumerate([("original", originals)] + rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.imshow(images[c].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=8)
            if r == 0 and col_labels:
                ax.set_title(col_labels[c], fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------- modes
def cmd_smoke(d: int, seed: int) -> int:
    """Fast end-to-end checks on train/val subsets (no test data, no results)."""
    X_train, _, X_val, _ = load_data()[:4]
    Xs, Xv = X_train[:2048], X_val[:512]
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            failures.append(name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"smoke: d={d} seed={seed} device={device} train={Xs.shape} val={Xv.shape}")
    set_seed(seed)
    m = CAEReducer(d, X_val=Xv, max_epochs=3, device=device).fit(Xs)
    tr = [h["train_mse"] for h in m.history_]
    check("3 finite epochs", len(m.history_) == 3 and all(np.isfinite(
        [h["train_mse"] for h in m.history_] + [h["val_mse"] for h in m.history_])))
    check("train MSE decreases", tr[-1] < tr[0], f"{tr[0]:.5f} -> {tr[-1]:.5f}")
    check("n_params formula", m.n_params == expected_n_params(d), f"{m.n_params}")
    Z = m.transform(Xv)
    check("transform shape/dtype", Z.shape == (len(Xv), d) and Z.dtype == np.float32, f"{Z.shape}")
    Xh = m.inverse_transform(Z)
    check("inverse shape/range", Xh.shape == Xv.shape and Xh.min() >= 0 and Xh.max() <= 1,
          f"{Xh.shape} [{Xh.min():.3f}, {Xh.max():.3f}]")
    mse_np = float(np.mean((Xv - Xh) ** 2))
    check("numpy-path MSE == best val MSE", abs(mse_np - m.best_val_mse_) < 1e-6,
          f"{mse_np:.7f} vs {m.best_val_mse_:.7f}")
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "cae.pt")
        m.save(p)
        m2 = CAEReducer.load(p, device=device)
        check("checkpoint round-trip", np.array_equal(m2.transform(Xv), Z)
              and np.array_equal(m2.inverse_transform(Z), Xh))
    set_seed(seed)
    m3 = CAEReducer(d, X_val=Xv, max_epochs=3, device=device).fit(Xs)
    check("same seed -> identical latents", np.array_equal(m3.transform(Xv), Z))
    set_seed(seed + 1)
    m4 = CAEReducer(d, X_val=Xv, max_epochs=3, device=device).fit(Xs)
    check("different seed -> different latents", not np.array_equal(m4.transform(Xv), Z))

    # Overfitting sanity check: the network must be able to fit a tiny subset.
    tiny = X_train[:512]
    baseline = float(np.mean((tiny - tiny.mean(axis=0)) ** 2))   # predict the mean image
    set_seed(seed)
    m5 = CAEReducer(d, X_val=tiny, max_epochs=300, patience=300, device=device).fit(tiny)
    check("overfits 512 images", m5.best_val_mse_ < 0.35 * baseline,
          f"MSE {m5.best_val_mse_:.5f} vs mean-image baseline {baseline:.5f}")
    print(f"smoke: {'FAILED ' + ', '.join(failures) if failures else 'all checks passed'}")
    return 1 if failures else 0


def cmd_pilot(runs: list[tuple[int, int]], verbose: bool,
              overrides: dict[str, Any] | None = None) -> int:
    """Validation-only training runs under the current protocol (or pilot overrides)."""
    X_train, y_train, X_val, y_val = load_data()[:4]   # official test arrays not bound
    from sklearn.neighbors import KNeighborsClassifier
    overrides = overrides or {}
    proto = sweep_protocol(overrides)
    fp = fingerprint(proto)
    tag = f"_p{fp[:8]}"                     # pilots under different protocols never collide
    warm_up_cuda()
    for d, seed in runs:
        name = run_name(d, seed) + tag
        print(f"\n=== PILOT {name} (validation only, protocol {fp}, overrides {overrides}) ===",
              flush=True)
        set_seed(seed)                                   # same order as evaluate()
        model = CAEReducer(d, X_val=X_val, verbose=verbose, **overrides)
        t0 = time.perf_counter()
        model.fit(X_train)
        fit_s = time.perf_counter() - t0
        Z_train, Z_val = model.transform(X_train), model.transform(X_val)
        X_val_hat = np.clip(model.inverse_transform(Z_val), 0.0, 1.0)
        val_mse_np = float(np.mean((X_val - X_val_hat) ** 2))
        knn = KNeighborsClassifier(n_neighbors=config.KNN_K, n_jobs=-1).fit(Z_train, y_train)
        val_knn = float(knn.score(Z_val, y_val))
        budget_binding = not model.stopped_early_
        record = {
            "fingerprint": fp, "d": d, "seed": seed, "created": _now(),
            "protocol": proto, "overrides": overrides, "fit_time_s": fit_s, "n_params": model.n_params,
            "best_epoch": model.best_epoch_, "best_val_mse": model.best_val_mse_,
            "val_mse_numpy_path": val_mse_np, "epochs_run": model.epochs_run_,
            "stopped_early": model.stopped_early_, "epoch_budget_binding": budget_binding,
            "val_knn_acc_diagnostic": val_knn,
            "z_train_std_per_dim": [float(s) for s in Z_train.std(axis=0)],
            "history": model.history_, "environment": environment(),
        }
        _write_json_atomic(pilot_dir() / f"{name}.json", record)
        model.save(str(checkpoint_dir() / "pilot" / f"{name}.pt"),
                   extra={"fingerprint": fp, "seed": seed, "mode": "pilot"})
        plot_histories({f"d={d}": model.history_},
                       figure_dir() / f"cae_pilot_curves_d{d}_s{seed}{tag}.png",
                       f"CAE pilot d={d} seed={seed} (protocol {fp[:8]}): train/val MSE")
        idx = first_index_per_class(y_val)
        plot_reconstructions(X_val[idx], [(f"CAE d={d}", X_val_hat[idx])],
                             figure_dir() / f"cae_pilot_val_recon_d{d}_s{seed}{tag}.png",
                             f"Validation reconstructions (first val image of each class), d={d}",
                             col_labels=[f"val #{i}" for i in idx])
        print(f"{name}: best epoch {model.best_epoch_}/{model.epochs_run_} "
              f"(stopped early: {model.stopped_early_}), best val MSE {model.best_val_mse_:.6f}, "
              f"numpy-path val MSE {val_mse_np:.6f}, val kNN acc {val_knn:.4f} (diagnostic), "
              f"fit {fit_s:.1f}s" + ("  <- EPOCH CAP REACHED" if budget_binding else ""),
              flush=True)
    return 0


def cmd_freeze() -> int:
    proto = sweep_protocol()
    fp = fingerprint(proto)
    pilots = sorted(p.name for p in pilot_dir().glob("*.json")
                    if json.loads(p.read_text()).get("fingerprint") == fp)
    if not pilots:
        raise IntegrityError(f"no pilot under protocol {fp} in {pilot_dir()}; run --mode pilot first")
    path = protocol_file()
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("fingerprint") == fp:
            print(f"Protocol {fp} is already frozen in {path}")
            return 0
        raise IntegrityError(f"{path} already freezes protocol {old.get('fingerprint')}.  Changing "
                             "the frozen protocol needs team agreement; move that file aside "
                             "deliberately (and archive old CAE results) before re-freezing.")
    _write_json_atomic(path, {"fingerprint": fp, "frozen_at": _now(), "protocol": proto,
                              "pilot_evidence": pilots, "git_commit": _git("rev-parse", "HEAD")})
    print(f"Froze protocol {fp} -> {path} (pilots: {', '.join(pilots)})")
    return 0


def reload_relative_diff(ckpt: Path, X_test: np.ndarray, Z_test_saved: np.ndarray) -> float:
    """Re-encode the test set from a saved checkpoint with evaluate()'s own 1000-row
    batching and compare with evaluate()'s Z_test; returns max|diff| / max(1, max|Z|).
    0.0 means bitwise identical."""
    reloaded = CAEReducer.load(str(ckpt))
    Z = np.asarray(shared_eval._batched(reloaded.transform, X_test), dtype=np.float32)
    if Z.shape != Z_test_saved.shape:
        return float("inf")
    return float(np.max(np.abs(Z - Z_test_saved))) / max(1.0, float(np.max(np.abs(Z_test_saved))))


def run_final_one(d: int, seed: int, fp: str, X_test: np.ndarray, n_train: int,
                  verbose: bool) -> dict[str, Any]:
    name = run_name(d, seed)
    print(f"\n=== FINAL {name} (protocol {fp}) ===", flush=True)
    model = CAEReducer(d, verbose=verbose)       # protocol defaults; validation = shared split
    gpu_before, started = gpu_snapshot(), _now()
    with tempfile.TemporaryDirectory() as tmp:   # row is committed below, after all checks
        row = shared_eval.evaluate(model, METHOD, d=d, seed=seed,
                                   results_file=os.path.join(tmp, "staged.csv"), save_latents=True)
    gpu_after = gpu_snapshot()

    # ---- post-evaluation checks (outside evaluate()'s timers) ----
    if row["n_params"] != expected_n_params(d):
        raise IntegrityError(f"{name}: n_params {row['n_params']} != {expected_n_params(d)}")
    if model.val_source_ != "shared_split":
        raise IntegrityError(f"{name}: validation source {model.val_source_!r}")
    ckpt = checkpoint_dir() / f"{name}.pt"
    model.save(str(ckpt), extra={"fingerprint": fp, "seed": seed, "mode": "final"})
    with np.load(latent_path(d, seed)) as f:
        Z_test_saved = f["Z_test"]
    reload_diff = reload_relative_diff(ckpt, X_test, Z_test_saved)
    if reload_diff > RELOAD_RTOL:
        raise IntegrityError(f"{name}: reloaded checkpoint re-encodes the test set with relative "
                             f"max diff {reload_diff}")
    artifact = None
    if (d, seed) == ARTIFACT_RUN:
        artifact = export_reconstruction_artifact(model, row, X_test, latent_path(d, seed), n_train)
        print(f"  Person 5 artifact {artifact['path']}: {artifact['shapes']}, MSE from X_hat "
              f"{artifact['mse_from_x_hat']:.7f} vs CSV {artifact['mse_csv']}", flush=True)

    meta = {
        "fingerprint": fp, "run": name, "d": d, "seed": seed, "row": row,
        "started": started, "finished": _now(),
        "best_epoch": model.best_epoch_, "best_val_mse": model.best_val_mse_,
        "epochs_run": model.epochs_run_, "stopped_early": model.stopped_early_,
        "epoch_budget_binding": not model.stopped_early_,
        "val_source": model.val_source_, "history": model.history_,
        "checkpoint": str(ckpt.relative_to(config.ROOT)), "checkpoint_sha256": _sha256(ckpt),
        "latents": str(latent_path(d, seed).relative_to(config.ROOT)),
        "reload_relative_diff_z_test": reload_diff, "person5_artifact": artifact,
        "gpu_before": gpu_before, "gpu_after": gpu_after, "environment": environment(),
    }
    _write_json_atomic(runs_dir() / f"{name}.json", meta)
    append_row(results_csv(), row)              # commit point
    print(f"  committed {name}: best epoch {model.best_epoch_}/{model.epochs_run_}"
          f"{' (EPOCH CAP REACHED)' if not model.stopped_early_ else ''}, "
          f"reload diff {reload_diff:.1e}", flush=True)
    return row


def cmd_final(runs: list[tuple[int, int]], verbose: bool) -> int:
    if not torch.cuda.is_available():
        raise SystemExit("final mode requires CUDA (CAE timings are recorded on the GPU)")
    runs = list(dict.fromkeys(runs))            # a repeated --d/--seed value must not run twice
    bad = [(d, s) for d, s in runs if d not in config.D_GRID or s not in config.SEEDS]
    if bad:
        raise SystemExit(f"final runs must use config.D_GRID x config.SEEDS; got {bad}")
    fp = load_frozen_protocol(require_current=True)["fingerprint"]
    with exclusive_lock():
        status, problems = {}, []
        for run in runs:
            try:
                status[run] = run_status(*run, fp)
            except IntegrityError as e:
                problems.append(str(e))
        if problems:
            print("Refusing to run; existing CAE results are inconsistent:\n  " + "\n  ".join(problems))
            return 2
        todo = [r for r in runs if status[r] == "todo"]
        for r in runs:
            if status[r] == "done":
                print(f"Skipping completed {run_name(*r)} (protocol {fp})")
        print(f"{len(runs) - len(todo)} complete, {len(todo)} to run", flush=True)
        if not todo:
            return 0
        X_train, _, _, _, X_test, _ = load_data()
        warm_up_cuda()
        failures = []
        for d, seed in todo:
            try:
                if run_status(d, seed, fp) != "todo":     # re-check right before training
                    print(f"Skipping {run_name(d, seed)}: committed meanwhile")
                    continue
                run_final_one(d, seed, fp, X_test, len(X_train), verbose)
            except Exception:
                traceback.print_exc()
                failures.append(run_name(d, seed))
                print(f"  FAILED {run_name(d, seed)}; no row committed", flush=True)
        print(f"\nfinal: {len(todo) - len(failures)} succeeded, {len(failures)} failed"
              + (f": {', '.join(failures)}" if failures else ""))
        return 1 if failures else 0


def cmd_report() -> int:
    """Post-evaluation verification, summaries and figures over the whole grid.

    Runs are checked against the *frozen* fingerprint.  Returns 1 if any
    checkpoint fails its hash or reload check.
    """
    import pandas as pd
    runs = [(d, s) for s in config.SEEDS for d in config.D_GRID]
    fp = load_frozen_protocol(require_current=False)["fingerprint"]
    with exclusive_lock():
        done = [r for r in runs if run_status(*r, fp) == "done"]
        missing = [r for r in runs if r not in done]
        print(f"report: {len(done)} committed runs, {len(missing)} missing"
              + (f": {[run_name(*r) for r in missing]}" if missing else ""))
        if not done:
            return 1
        _, _, _, _, X_test, y_test = load_data()
        diag, histories0, envs = [], {}, set()
        for d, seed in done:
            meta = json.loads((runs_dir() / f"{run_name(d, seed)}.json").read_text())
            ckpt = Path(config.ROOT) / meta["checkpoint"]
            ok_sha = _sha256(ckpt) == meta["checkpoint_sha256"]
            with np.load(latent_path(d, seed)) as f:
                diff = reload_relative_diff(ckpt, X_test, f["Z_test"])
            env = meta["environment"]
            envs.add((env.get("gpu", {}).get("name"), env.get("torch"),
                      json.dumps(env.get("source_sha256", {}), sort_keys=True)))
            diag.append({"d": d, "seed": seed, "best_epoch": meta["best_epoch"],
                         "epochs_run": meta["epochs_run"], "stopped_early": meta["stopped_early"],
                         "best_val_mse": round(meta["best_val_mse"], 7),
                         "test_mse": meta["row"]["mse"], "knn_acc": meta["row"]["knn_acc"],
                         "silhouette": meta["row"]["silhouette"],
                         "fit_time_s": meta["row"]["fit_time_s"],
                         "encode_time_ms_per_img": meta["row"]["encode_time_ms_per_img"],
                         "checkpoint_sha256_ok": ok_sha, "reload_relative_diff": diff,
                         "gpu": env.get("gpu", {}).get("name")})
            if seed == config.SEEDS[0]:
                histories0[f"d={d}"] = meta["history"]
        table = pd.DataFrame(diag).sort_values(["d", "seed"])
        table.to_csv(Path(config.RESULTS_DIR) / "cae_run_diagnostics.csv", index=False)
        metrics = ["knn_acc", "silhouette", "test_mse", "fit_time_s", "encode_time_ms_per_img",
                   "best_epoch", "epochs_run"]
        summary = table.groupby("d")[metrics].agg(["mean", "std", "count"])
        summary.columns = [f"{m}_{s}" for m, s in summary.columns]
        summary.insert(0, "n_params", [expected_n_params(d) for d in summary.index])
        summary.round(6).to_csv(Path(config.RESULTS_DIR) / "cae_summary.csv")
        print(table.to_string(index=False))

        failed = table[~table["checkpoint_sha256_ok"] | (table["reload_relative_diff"] > RELOAD_RTOL)]
        capped = table[~table["stopped_early"].astype(bool)]
        if len(envs) > 1:
            print(f"WARNING: runs differ in GPU / torch / source files ({len(envs)} combinations)")
        if len(capped):
            print(f"WARNING: {len(capped)} run(s) reached the epoch cap without early stopping: "
                  f"{[run_name(int(r.d), int(r.seed)) for r in capped.itertuples()]}")
        if len(failed):
            print(f"ERROR: checkpoint verification failed for "
                  f"{[run_name(int(r.d), int(r.seed)) for r in failed.itertuples()]}")

        if histories0:
            plot_histories(histories0, figure_dir() / "cae_training_curves_s0.png",
                           f"CAE validation MSE by epoch, seed {config.SEEDS[0]} (best epoch marked)",
                           show_train=False)
        seed0 = sorted(d for d, s in done if s == config.SEEDS[0])
        if seed0:
            idx = first_index_per_class(y_test)
            rows = []
            for d in seed0:
                model = CAEReducer.load(str(checkpoint_dir() / f"{run_name(d, config.SEEDS[0])}.pt"))
                with np.load(latent_path(d, config.SEEDS[0])) as f:
                    rows.append((f"CAE d={d}",
                                 np.clip(model.inverse_transform(f["Z_test"][idx]), 0, 1)))
            plot_reconstructions(X_test[idx], rows, figure_dir() / "cae_test_reconstructions_s0.png",
                                 f"CAE test reconstructions, seed {config.SEEDS[0]} "
                                 "(first test image of each class)",
                                 col_labels=[f"test #{i}" for i in idx])
        print(f"wrote {Path(config.RESULTS_DIR) / 'cae_summary.csv'}, cae_run_diagnostics.csv, "
              "figures/cae_training_curves_s0.png, figures/cae_test_reconstructions_s0.png")
        return 1 if len(failed) else 0


# ---------------------------------------------------------------------- CLI
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=("smoke", "pilot", "freeze", "final", "report"))
    p.add_argument("--d", type=int, nargs="+", help="latent dimension(s) (default 32)")
    p.add_argument("--seed", type=int, nargs="+", help="seed(s) (default 0)")
    p.add_argument("--all", action="store_true", help="every d in D_GRID x every seed in SEEDS")
    p.add_argument("--quiet", action="store_true", help="no per-epoch lines")
    p.add_argument("--max-epochs", type=int, help="pilot only: epoch-budget study override")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    explicit = args.d is not None or args.seed is not None
    if args.max_epochs is not None and args.mode != "pilot":
        raise SystemExit("--max-epochs is a pilot-only override; other modes use the protocol")
    if args.all and explicit:
        raise SystemExit("--all cannot be combined with --d/--seed")
    if args.mode in ("freeze", "report") and explicit:
        raise SystemExit(f"--mode {args.mode} takes no --d/--seed ({args.mode} covers the protocol/grid)")
    d_list, seeds = args.d or [32], args.seed or [0]
    if args.all:   # seed-major order: the seed-0 sweep (and Person 5's run) finishes first
        runs = [(d, s) for s in config.SEEDS for d in config.D_GRID]
    else:
        runs = list(dict.fromkeys((d, s) for s in seeds for d in d_list))
    try:
        if args.mode == "smoke":
            if len(runs) != 1:
                raise SystemExit("smoke mode takes a single --d and --seed")
            return cmd_smoke(*runs[0])
        if args.mode == "pilot":
            overrides = {} if args.max_epochs is None else {"max_epochs": args.max_epochs}
            return cmd_pilot(runs, verbose=not args.quiet, overrides=overrides)
        if args.mode == "freeze":
            return cmd_freeze()
        if args.mode == "final":
            return cmd_final(runs, verbose=not args.quiet)
        return cmd_report()
    except IntegrityError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
