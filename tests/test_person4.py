"""Tests for Person 4's CAE (models/cae.py) and runner (scripts/run_person4.py).

Everything except ``test_person5_artifact_on_disk`` uses small synthetic arrays,
so the suite never needs the dataset and never touches the official test set.
GPU-only checks are skipped when CUDA is unavailable.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

import config
import evaluate as shared_eval
from data import set_seed
from models.cae import (CAEReducer, ConvAutoencoder, count_params, expected_n_params,
                        fp32_numerics)
from scripts import run_person4 as runner

CUDA = torch.cuda.is_available()


def _pixels(n: int, seed: int = 0) -> np.ndarray:
    """Synthetic images in [0, 1]: a few smooth blobs per image (learnable, not noise)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:28, 0:28]
    out = np.zeros((n, 28, 28), dtype=np.float32)
    for i in range(n):
        for _ in range(2):
            cy, cx = rng.uniform(6, 22, size=2)
            s = rng.uniform(2.5, 6.0)
            out[i] += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
    return np.clip(out, 0, 1).reshape(n, 784)


class _Poison:
    """Stands in for the official test arrays; any use fails the test."""

    def _fail(self, *a, **k):
        raise AssertionError("official test data was accessed")

    __array__ = __len__ = __getitem__ = __iter__ = _fail

    def __getattr__(self, name):
        self._fail()


class FastCAE(CAEReducer):
    """CPU, 2-epoch CAEReducer used to exercise the runner quickly (keeps .load())."""

    def __init__(self, d, **kw):
        kw.setdefault("max_epochs", 2)
        kw.setdefault("device", "cpu")
        super().__init__(d, **kw)


def _small_reducer(d: int = 4, **kw) -> CAEReducer:
    kw.setdefault("max_epochs", 2)
    kw.setdefault("device", "cpu")
    kw.setdefault("X_val", _pixels(64, seed=99))
    return CAEReducer(d, **kw)


# ------------------------------------------------------------ architecture
@pytest.mark.parametrize("d", config.D_GRID)
def test_encoder_decoder_shapes_for_every_d(d):
    net = ConvAutoencoder(d)
    h = torch.rand(3, 1, 28, 28)
    enc_shapes = []
    for layer in net.encoder:
        h = layer(h)
        enc_shapes.append(tuple(h.shape))
    assert enc_shapes == [(3, 16, 14, 14), (3, 16, 14, 14), (3, 32, 7, 7), (3, 32, 7, 7),
                          (3, 1568), (3, d)]
    dec_shapes = []
    for layer in net.decoder:
        h = layer(h)
        dec_shapes.append(tuple(h.shape))
    assert dec_shapes == [(3, 1568), (3, 32, 7, 7), (3, 16, 14, 14), (3, 16, 14, 14),
                          (3, 1, 28, 28), (3, 1, 28, 28)]


@pytest.mark.parametrize("d", config.D_GRID)
def test_parameter_count_matches_formula(d):
    net = ConvAutoencoder(d)
    manual = ((1 * 16 * 9 + 16) + (16 * 32 * 9 + 32) + (1568 * d + d)        # encoder
              + (d * 1568 + 1568) + (32 * 16 * 9 + 16) + (16 * 1 * 9 + 1))  # decoder
    assert count_params(net) == manual == expected_n_params(d) == 11137 + 3137 * d
    assert expected_n_params(32) == 111521


def test_output_range_and_finite_loss():
    net = ConvAutoencoder(8)
    x = torch.cat([torch.zeros(2, 1, 28, 28), torch.ones(2, 1, 28, 28), torch.rand(4, 1, 28, 28)])
    out = net(x)
    assert out.shape == x.shape
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert torch.isfinite(torch.nn.functional.mse_loss(out, x))


def test_gradient_reaches_every_encoder_and_decoder_parameter():
    torch.manual_seed(0)
    net = ConvAutoencoder(16)
    x = torch.from_numpy(_pixels(8)).view(-1, 1, 28, 28)
    torch.nn.functional.mse_loss(net(x), x).backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        assert float(p.grad.abs().sum()) > 0, name


def test_minimal_training_steps_reduce_loss():
    torch.manual_seed(0)
    net = ConvAutoencoder(8)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x = torch.from_numpy(_pixels(64)).view(-1, 1, 28, 28)
    losses = []
    for _ in range(40):
        loss = torch.nn.functional.mse_loss(net(x), x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < 0.5 * losses[0]


# ------------------------------------------------------- numpy wrapper API
def test_reducer_numpy_api_contract():
    X = _pixels(300)
    r = _small_reducer(d=8).fit(X)
    Z = r.transform(X[:37])
    assert Z.shape == (37, 8) and Z.dtype == np.float32 and np.isfinite(Z).all()
    X_hat = r.inverse_transform(Z)
    assert X_hat.shape == (37, 784) and X_hat.dtype == np.float32
    assert X_hat.min() >= 0.0 and X_hat.max() <= 1.0
    assert r.n_params == expected_n_params(8)
    assert r.transform(X[:0]).shape == (0, 8)
    with pytest.raises(ValueError):
        r.transform(X[:, :100])
    with pytest.raises(ValueError):
        r.inverse_transform(np.zeros((2, 5), dtype=np.float32))


def test_input_validation():
    r = _small_reducer()
    with pytest.raises(RuntimeError):
        r.transform(_pixels(2))                      # not fitted
    bad = _pixels(20)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        r.fit(bad)
    with pytest.raises(ValueError):
        r.fit(_pixels(20) * 2.0)                     # outside [0, 1]
    with pytest.raises(TypeError):
        r.fit((_pixels(20) * 255).astype(np.uint8))  # 0..255 integer pixels
    with pytest.raises(ValueError):
        _small_reducer(X_val=_pixels(0)).fit(_pixels(20))          # empty validation set
    with pytest.raises(ValueError):
        _small_reducer(X_val=_pixels(8) * 255).fit(_pixels(20))    # validation outside [0, 1]
    for bad_kw in ({"batch_size": 2.9}, {"max_epochs": True}, {"patience": 0}):
        with pytest.raises((TypeError, ValueError)):
            CAEReducer(4, **bad_kw)
    with pytest.raises(TypeError):
        CAEReducer(True)


def test_constructor_creates_no_weights_and_consumes_no_rng():
    state = torch.get_rng_state()
    r = CAEReducer(32)
    assert torch.equal(state, torch.get_rng_state())
    assert r.model_ is None and not hasattr(r, "n_params")


def test_seed_reproducibility_with_lazy_initialisation():
    """Mirrors evaluate(): object built first, unrelated RNG use, then set_seed, then fit."""
    X = _pixels(256)
    a = _small_reducer()
    torch.rand(1000)
    np.random.rand(1000)
    set_seed(3)
    a.fit(X)
    b = _small_reducer()
    set_seed(3)
    b.fit(X)
    c = _small_reducer()
    set_seed(4)
    c.fit(X)
    np.testing.assert_array_equal(a.transform(X), b.transform(X))
    def strip(hist):
        return [{k: v for k, v in h.items() if k != "elapsed_s"} for h in hist]
    assert strip(a.history_) == strip(b.history_)
    assert not np.array_equal(a.transform(X), c.transform(X))


@pytest.mark.skipif(not CUDA, reason="CUDA not available")
def test_seed_reproducibility_on_gpu_is_bitwise():
    X = _pixels(512)
    outs = []
    for _ in range(2):
        set_seed(0)
        r = _small_reducer(d=16, device="cuda", max_epochs=3).fit(X)
        outs.append(r.transform(X))
    np.testing.assert_array_equal(outs[0], outs[1])


@pytest.mark.skipif(not CUDA, reason="CUDA not available")
def test_gpu_and_cpu_training_agree():
    X = _pixels(256)
    res = {}
    for dev in ("cpu", "cuda"):
        set_seed(0)
        res[dev] = _small_reducer(device=dev, max_epochs=1).fit(X).transform(X)
    np.testing.assert_allclose(res["cpu"], res["cuda"], atol=1e-3)


def test_numerics_context_overrides_and_restores_global_flags(monkeypatch):
    # Start from the opposite of what the context sets, so no check is vacuous.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)

    def flags():
        return (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)

    with fp32_numerics():
        assert flags() == (False, False, True, False)
    assert flags() == (True, True, False, True)
    with pytest.raises(RuntimeError):
        with fp32_numerics():
            raise RuntimeError("boom")
    assert flags() == (True, True, False, True)


def test_transform_handles_inputs_larger_than_one_internal_batch():
    X = _pixels(5000)                                 # > EVAL_BATCH (4096)
    r = _small_reducer(max_epochs=1).fit(_pixels(128))
    Z = r.transform(X)
    assert Z.shape == (5000, 4)
    np.testing.assert_allclose(Z, np.concatenate([r.transform(X[:3000]), r.transform(X[3000:])]),
                               rtol=1e-5, atol=1e-6)
    assert r.inverse_transform(Z).shape == (5000, 784)


# ------------------------------------------------- training / checkpoints
def test_best_validation_epoch_is_restored_after_early_stopping():
    # Validation images unlike the training blobs make validation MSE turn upward
    # while training continues, so early stopping fires after a non-best epoch.
    X, X_val = _pixels(256), np.random.default_rng(7).random((64, 784), dtype=np.float32)
    set_seed(0)
    r = CAEReducer(4, X_val=X_val, lr=0.01, max_epochs=60, patience=2, device="cpu").fit(X)
    vals = [h["val_mse"] for h in r.history_]
    assert r.stopped_early_ and r.best_epoch_ < r.epochs_run_   # last epoch is not the best
    assert r.best_epoch_ == int(np.argmin(vals)) + 1
    recomputed = float(np.mean((X_val - r.inverse_transform(r.transform(X_val))) ** 2))
    assert recomputed == pytest.approx(r.best_val_mse_, abs=1e-6)
    assert abs(recomputed - vals[-1]) > 1e-7                    # differs from the final epoch


def test_non_finite_loss_is_reported():
    class NaNReducer(CAEReducer):
        def _build_network(self):
            net = super()._build_network()
            with torch.no_grad():
                net.decoder[0].bias.fill_(float("nan"))
            return net

    with pytest.raises(FloatingPointError):
        NaNReducer(4, X_val=_pixels(16), max_epochs=2, device="cpu").fit(_pixels(64))


def test_checkpoint_save_and_reload_are_identical(tmp_path):
    X = _pixels(200)
    set_seed(0)
    r = _small_reducer(d=8).fit(X)
    path = tmp_path / "m.pt"
    r.save(str(path), extra={"note": "x"})
    state = torch.get_rng_state()
    r2 = CAEReducer.load(str(path), device="cpu")
    assert torch.equal(state, torch.get_rng_state())
    Z = r.transform(X)
    np.testing.assert_array_equal(r2.transform(X), Z)
    np.testing.assert_array_equal(r2.inverse_transform(Z), r.inverse_transform(Z))
    assert (r2.n_params, r2.best_epoch_, r2.history_) == (r.n_params, r.best_epoch_, r.history_)
    assert not list(tmp_path.glob("*.tmp-*"))


def test_fit_uses_shared_validation_split_and_never_test(monkeypatch):
    X_tr, X_val = _pixels(200), _pixels(50, seed=5)
    labels = np.zeros(200, dtype=np.int64)
    import data
    monkeypatch.setattr(data, "load_data",
                        lambda: (X_tr, labels, X_val, labels[:50], _Poison(), _Poison()))
    set_seed(0)
    r = CAEReducer(4, max_epochs=2, device="cpu").fit(X_tr)
    assert r.val_source_ == "shared_split"
    recomputed = float(np.mean((X_val - r.inverse_transform(r.transform(X_val))) ** 2))
    assert recomputed == pytest.approx(r.best_val_mse_, abs=1e-6)


# ---------------------------------------------------- shared evaluator
def _patch_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(config, "LATENT_DIR", str(tmp_path / "latents"))
    monkeypatch.setattr(config, "ROOT", str(tmp_path))


def test_shared_evaluate_integration_and_schema(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)
    rng = np.random.default_rng(0)
    Xtr, Xv, Xte = _pixels(300, 1), _pixels(60, 2), _pixels(120, 3)
    ytr, yv, yte = (rng.integers(0, 10, n) for n in (300, 60, 120))
    monkeypatch.setattr(shared_eval, "load_data", lambda: (Xtr, ytr, Xv, yv, Xte, yte))
    model = CAEReducer(4, X_val=Xv, max_epochs=2, device="cpu")
    staged = tmp_path / "staged.csv"
    row = shared_eval.evaluate(model, "CAE", d=4, seed=0, results_file=str(staged), verbose=False)

    assert list(row) == shared_eval.COLUMNS
    assert row["n_params"] == expected_n_params(4)
    assert 0.0 <= row["mse"] <= 1.0 and 0.0 <= row["knn_acc"] <= 1.0
    with np.load(tmp_path / "latents" / "CAE_d4_s0.npz") as f:
        assert f["Z_train"].shape == (300, 4) and f["Z_test"].shape == (120, 4)
    # The runner's commit must write byte-identical CSV to evaluate()'s own writer.
    runner.append_row(runner.results_csv(), row)
    assert runner.results_csv().read_bytes() == staged.read_bytes()
    with open(runner.results_csv(), newline="") as fh:
        assert next(csv.reader(fh)) == shared_eval.COLUMNS


# ---------------------------------------------- resume / duplicate safety
def _commit(monkeypatch, d, seed, fp, row_overrides=None):
    row = {"method": "CAE", "d": d, "seed": seed, "knn_k": 9, "knn_acc": 0.8, "silhouette": 0.1,
           "mse": 0.012345, "fit_time_s": 1.5, "encode_time_ms_per_img": 0.001,
           "n_params": expected_n_params(d)}
    runner._write_json_atomic(runner.runs_dir() / f"CAE_d{d}_s{seed}.json",
                              {"fingerprint": fp, "row": row})
    runner.append_row(runner.results_csv(), dict(row, **(row_overrides or {})))


def test_run_status_skips_only_compatible_completed_runs(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)
    fp = "abc123"
    assert runner.run_status(32, 0, fp) == "todo"
    _commit(monkeypatch, 32, 0, fp)
    assert runner.run_status(32, 0, fp) == "done"
    with pytest.raises(runner.IntegrityError, match="protocol"):
        runner.run_status(32, 0, "different")
    # metadata without a committed row: interrupted run -> rerun
    runner._write_json_atomic(runner.runs_dir() / "CAE_d8_s1.json", {"fingerprint": fp, "row": {}})
    assert runner.run_status(8, 1, fp) == "todo"
    # duplicate row
    _commit(monkeypatch, 32, 0, fp)
    with pytest.raises(runner.IntegrityError, match="duplicate"):
        runner.run_status(32, 0, fp)


def test_run_status_rejects_foreign_or_edited_rows(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)
    fp = "abc123"
    _commit(monkeypatch, 16, 2, fp, row_overrides={"knn_acc": 0.99})
    with pytest.raises(runner.IntegrityError, match="differs"):
        runner.run_status(16, 2, fp)
    runner.append_row(runner.results_csv(), {c: 0 for c in shared_eval.COLUMNS} | {
        "method": "CAE", "d": 4, "seed": 0})
    with pytest.raises(runner.IntegrityError, match="no metadata"):
        runner.run_status(4, 0, fp)


def test_wrong_csv_header_is_refused(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)
    runner.results_csv().parent.mkdir(parents=True)
    runner.results_csv().write_text("method,d,seed\nCAE,32,0\n")
    with pytest.raises(runner.IntegrityError, match="header"):
        runner.run_status(32, 0, "x")


def test_fingerprint_tracks_protocol_changes(monkeypatch):
    base = runner.fingerprint()
    assert runner.fingerprint() == base
    monkeypatch.setattr(config, "KNN_K", config.KNN_K + 2)
    assert runner.fingerprint() != base
    monkeypatch.undo()
    monkeypatch.setitem(runner.FIXED_PROTOCOL, "optimizer", "SGD")
    assert runner.fingerprint() != base
    monkeypatch.undo()
    real_sha = runner._sha256                          # any edit to the training code
    monkeypatch.setattr(runner, "_sha256",
                        lambda path: "edited" if str(path).endswith("cae.py") else real_sha(path))
    assert runner.fingerprint() != base


def test_architecture_spec_is_portable_and_complete():
    spec = runner.architecture_spec()
    assert spec[0].startswith("encoder.0: Conv2d(1->16, kernel=(3, 3), stride=(2, 2), padding=(1, 1)")
    assert "encoder.5: Linear(1568->d, bias=True)" in spec
    assert "decoder.0: Linear(d->1568, bias=True)" in spec
    assert any("ConvTranspose2d(16->1" in line and "output_padding=(1, 1)" in line for line in spec)
    assert all("0x" not in line for line in spec)       # no object addresses / reprs


def test_final_mode_requires_matching_frozen_protocol(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)
    with pytest.raises(runner.IntegrityError, match="not found"):
        runner.load_frozen_protocol()
    runner._write_json_atomic(runner.protocol_file(), {"fingerprint": "stale"})
    with pytest.raises(runner.IntegrityError, match="changed after the freeze"):
        runner.load_frozen_protocol()
    runner._write_json_atomic(runner.protocol_file(), {"fingerprint": runner.fingerprint()})
    assert runner.load_frozen_protocol()["fingerprint"] == runner.fingerprint()


# -------------------------------------------------- Person 5 NPZ artifact
def _fake_official_run(tmp_path, d=4):
    X_tr, X_te = _pixels(200, 1), _pixels(90, 2)
    set_seed(0)
    model = _small_reducer(d=d).fit(X_tr)
    Z_tr = shared_eval._batched(model.transform, X_tr)
    Z_te = shared_eval._batched(model.transform, X_te)
    X_hat = np.clip(shared_eval._batched(model.inverse_transform, Z_te), 0.0, 1.0)
    row = {"mse": round(float(np.mean((X_te - X_hat) ** 2)), 6)}       # as evaluate() does
    path = tmp_path / f"CAE_d{d}_s0.npz"
    np.savez_compressed(path, Z_train=Z_tr, Z_test=Z_te)
    return model, row, X_te, path, Z_tr, Z_te


def test_reconstruction_artifact_export(tmp_path):
    model, row, X_te, path, Z_tr, Z_te = _fake_official_run(tmp_path)
    info = runner.export_reconstruction_artifact(model, row, X_te, path, n_train=200)
    with np.load(path) as f:
        assert sorted(f.files) == ["X_hat", "Z_test", "Z_train"]
        np.testing.assert_array_equal(f["Z_train"], Z_tr)       # existing arrays preserved
        np.testing.assert_array_equal(f["Z_test"], Z_te)
        X_hat = f["X_hat"]
    assert X_hat.shape == X_te.shape and X_hat.dtype == np.float32
    assert X_hat.min() >= 0.0 and X_hat.max() <= 1.0 and np.isfinite(X_hat).all()
    assert info["abs_diff"] <= runner.MSE_TOL
    assert info["shapes"] == {"X_hat": [90, 784], "Z_test": [90, 4], "Z_train": [200, 4]}
    assert not list(tmp_path.glob(".*tmp-*"))


def test_reconstruction_artifact_rejects_mse_mismatch_without_touching_file(tmp_path):
    model, row, X_te, path, _, _ = _fake_official_run(tmp_path)
    before = path.read_bytes()
    with pytest.raises(runner.IntegrityError, match="MSE"):
        runner.export_reconstruction_artifact(model, {"mse": row["mse"] + 1e-3}, X_te, path)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".*tmp-*"))


def test_reconstruction_artifact_rejects_wrong_test_order(tmp_path):
    model, row, X_te, path, _, _ = _fake_official_run(tmp_path)
    runner.export_reconstruction_artifact(model, row, X_te, path)
    with pytest.raises(runner.IntegrityError):
        runner.verify_reconstruction_artifact(path, X_te[::-1], row["mse"], d=4)


# ------------------------------------------------- development isolation
def test_pilot_mode_never_reads_test_data(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)
    rng = np.random.default_rng(0)
    Xtr, Xv = _pixels(200, 1), _pixels(50, 2)
    ytr, yv = rng.integers(0, 10, 200), rng.integers(0, 10, 50)
    monkeypatch.setattr(runner, "load_data", lambda: (Xtr, ytr, Xv, yv, _Poison(), _Poison()))
    monkeypatch.setattr(runner, "CAEReducer", FastCAE)
    monkeypatch.setattr(runner, "warm_up_cuda", lambda: None)
    monkeypatch.setattr(runner, "environment", lambda: {})
    assert runner.cmd_pilot([(4, 0)], verbose=False) == 0
    tag = f"_p{runner.fingerprint()[:8]}"
    record = json.loads((runner.pilot_dir() / f"CAE_d4_s0{tag}.json").read_text())
    assert record["best_epoch"] >= 1 and len(record["history"]) == record["epochs_run"]
    assert (tmp_path / "figures" / f"cae_pilot_curves_d4_s0{tag}.png").exists()


# --------------------------------------------- final-mode commit protocol
def _official_setup(monkeypatch, tmp_path):
    """Synthetic 'official' data + fast CPU CAE wired through the real runner/evaluate."""
    _patch_dirs(monkeypatch, tmp_path)
    rng = np.random.default_rng(0)
    data_tuple = (_pixels(240, 1), rng.integers(0, 10, 240), _pixels(60, 2),
                  rng.integers(0, 10, 60), _pixels(100, 3), np.repeat(np.arange(10), 10))
    import data
    for module in (data, shared_eval, runner):
        monkeypatch.setattr(module, "load_data", lambda: data_tuple)
    monkeypatch.setattr(runner, "CAEReducer", FastCAE)
    monkeypatch.setattr(runner, "ARTIFACT_RUN", (4, 0))
    monkeypatch.setattr(runner, "warm_up_cuda", lambda: None)
    monkeypatch.setattr(runner, "environment", lambda: {"torch": "test"})
    monkeypatch.setattr(runner, "gpu_snapshot", lambda: None)
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)   # device stays CPU
    runner._write_json_atomic(runner.protocol_file(), {"fingerprint": runner.fingerprint()})
    return data_tuple


def _cae_rows():
    with open(runner.results_csv(), newline="") as fh:
        return [(r["d"], r["seed"]) for r in csv.DictReader(fh)]


def test_final_run_commits_row_only_after_every_artifact(monkeypatch, tmp_path):
    data_tuple = _official_setup(monkeypatch, tmp_path)
    X_test = data_tuple[4]
    fp = runner.fingerprint()

    def crash(*a, **k):
        raise RuntimeError("simulated crash during artifact export")

    monkeypatch.setattr(runner, "export_reconstruction_artifact", crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        runner.run_final_one(4, 0, fp, X_test, n_train=240, verbose=False)
    assert not runner.results_csv().exists()               # nothing committed
    assert runner.run_status(4, 0, fp) == "todo"
    monkeypatch.undo()                                     # restore everything, then redo setup
    data_tuple = _official_setup(monkeypatch, tmp_path)

    row = runner.run_final_one(4, 0, fp, X_test, n_train=240, verbose=False)
    assert runner.run_status(4, 0, fp) == "done" and _cae_rows() == [("4", "0")]
    meta = json.loads((runner.runs_dir() / "CAE_d4_s0.json").read_text())
    assert meta["reload_relative_diff_z_test"] <= runner.RELOAD_RTOL
    assert meta["checkpoint_sha256"] == runner._sha256(tmp_path / meta["checkpoint"])
    info = runner.verify_reconstruction_artifact(runner.latent_path(4, 0), X_test, row["mse"], d=4,
                                                 n_train=240)
    assert info["keys"] == ["X_hat", "Z_test", "Z_train"]


def test_final_cli_dedupes_repeated_values_and_resumes(monkeypatch, tmp_path):
    _official_setup(monkeypatch, tmp_path)
    assert runner.main(["--mode", "final", "--d", "4", "4", "--seed", "1", "1", "--quiet"]) == 0
    assert _cae_rows() == [("4", "1")]
    assert runner.main(["--mode", "final", "--d", "4", "--seed", "1", "--quiet"]) == 0   # skip
    assert _cae_rows() == [("4", "1")]
    runner._write_json_atomic(runner.protocol_file(), {"fingerprint": "stale"})
    assert runner.main(["--mode", "final", "--d", "4", "--seed", "1", "--quiet"]) == 2


def test_report_covers_grid_and_respects_lock(monkeypatch, tmp_path):
    _official_setup(monkeypatch, tmp_path)
    assert runner.main(["--mode", "final", "--d", "4", "--seed", "1", "--quiet"]) == 0
    assert runner.main(["--mode", "report"]) == 0
    with open(Path(config.RESULTS_DIR) / "cae_summary.csv", newline="") as fh:
        summary = list(csv.DictReader(fh))
    assert [(r["d"], r["knn_acc_count"]) for r in summary] == [("4", "1")]
    with runner.exclusive_lock():
        with pytest.raises(SystemExit):
            runner.cmd_report()


@pytest.mark.parametrize("argv", [
    ["--mode", "smoke", "--max-epochs", "5"],
    ["--mode", "final", "--all", "--d", "32"],
    ["--mode", "report", "--d", "32"],
    ["--mode", "freeze", "--seed", "1"],
])
def test_cli_rejects_ambiguous_arguments(argv):
    with pytest.raises(SystemExit):
        runner.main(argv)


def test_append_row_refuses_a_torn_csv(tmp_path):
    path = tmp_path / "results_cae.csv"
    path.write_text(",".join(shared_eval.COLUMNS) + "\nCA")
    with pytest.raises(runner.IntegrityError, match="newline"):
        runner.append_row(path, {c: 0 for c in shared_eval.COLUMNS})


def test_run_status_turns_malformed_records_into_integrity_errors(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)
    runner.append_row(runner.results_csv(), {c: 0 for c in shared_eval.COLUMNS} | {
        "method": "CAE", "d": "32.0", "seed": 0})
    with pytest.raises(runner.IntegrityError, match="unreadable"):
        runner.run_status(32, 0, "fp")


# ------------------------------------------- real artifact (after final run)
ARTIFACT = Path(config.LATENT_DIR) / "CAE_d32_s0.npz"
RESULTS = Path(config.RESULTS_DIR) / "results_cae.csv"
DATA_PRESENT = Path(config.DATA_DIR, "t10k-images-idx3-ubyte.gz").exists()


@pytest.mark.skipif(not (ARTIFACT.exists() and RESULTS.exists() and DATA_PRESENT),
                    reason="official CAE d=32 seed=0 artifact not produced yet")
def test_person5_artifact_on_disk():
    from data import load_data
    rows = [r for r in csv.DictReader(open(RESULTS))
            if r["method"] == "CAE" and r["d"] == "32" and r["seed"] == "0"]
    assert len(rows) == 1
    X_train, _, _, _, X_test, _ = load_data()
    info = runner.verify_reconstruction_artifact(ARTIFACT, X_test, float(rows[0]["mse"]), d=32,
                                                 n_train=len(X_train))
    assert info["shapes"] == {"X_hat": [10000, 784], "Z_test": [10000, 32], "Z_train": [54000, 32]}
