"""Convolutional autoencoder (CAE) for the PCA / NMF / CAE comparison (Person 4).

Two pieces:

* ``ConvAutoencoder``: the PyTorch network (two strided convolutions down,
  two transposed convolutions up; as in the team proposal there is no
  activation between the decoder Linear and the first transposed convolution).
* ``CAEReducer``: an sklearn-style wrapper (``fit`` / ``transform`` /
  ``inverse_transform`` / ``n_params``) that the shared ``evaluate()`` calls
  exactly like PCA or NMF.

Architecture (identical for every d; only the two bottleneck Linear layers change)::

    encoder  (B,1,28,28) -Conv(1->16, k3,s2,p1) + ReLU->  (B,16,14,14)
                         -Conv(16->32,k3,s2,p1) + ReLU->  (B,32,7,7)
                         -flatten->                       (B,1568)
                         -Linear(1568->d)->               (B,d)   latent, no activation
    decoder  (B,d)       -Linear(d->1568)->               (B,1568)
                         -reshape->                       (B,32,7,7)
                         -ConvT(32->16,k3,s2,p1,op1) + ReLU->    (B,16,14,14)
                         -ConvT(16->1, k3,s2,p1,op1) + Sigmoid-> (B,1,28,28)

    n_params(d) = 11137 + 3137 * d        (d = 32  ->  111,521)

Seed handling
-------------
``evaluate()`` builds nothing itself: the caller constructs the model object,
then ``evaluate()`` calls ``set_seed(seed)`` and only then ``fit()``.  So
``CAEReducer.__init__`` creates no weights.  The network is created inside
``fit()`` on the CPU, from the generator that ``set_seed()`` has just seeded,
and then moved to the device; mini-batch order comes from the same CPU
generator.  Initial weights and batch order therefore depend only on the seed,
not on what ran earlier in the process or on the device.  No DataLoader (and
no worker processes) is used: the training set is held on the device and
indexed with ``torch.randperm``.

Numerics
--------
PyTorch enables TF32 for cuDNN convolutions by default on Ampere/Ada GPUs.
Every CAE call runs inside ``fp32_numerics()``, which turns TF32 off and
selects deterministic cuDNN kernels, then restores the caller's flags.
"""
from __future__ import annotations

import contextlib
import math
import os
import time
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

N_PIXELS = 784
IMG_SHAPE = (1, 28, 28)
FLAT_HIDDEN = 32 * 7 * 7          # 1568
CHECKPOINT_FORMAT = 1
EVAL_BATCH = 4096                 # internal batch for validation / transform (memory only)

# Choices that are fixed by design and are not constructor arguments.  They are
# part of protocol(), so editing one changes the run fingerprint.
FIXED_PROTOCOL: dict[str, Any] = {
    "loss": "pixelwise MSE, mean over batch and 784 pixels",
    "optimizer": "torch.optim.Adam(betas=(0.9, 0.999), eps=1e-8, weight_decay=0)",
    "lr_schedule": "constant",
    "selection": "restore epoch with the lowest validation MSE",
    "init": "PyTorch default layer init, built on CPU after set_seed()",
    "shuffle": "torch.randperm on the CPU generator, once per epoch, last partial batch kept",
    "numerics": "float32; TF32 disabled; cuDNN deterministic, benchmark off",
    "validation": "shared 6K split from data.load_data() unless X_val is passed explicitly",
}


def expected_n_params(d: int) -> int:
    """Closed-form trainable-parameter count of ``ConvAutoencoder(d)``.

    conv1 160 + conv2 4640 + enc Linear 1569d + dec Linear (1568d + 1568)
    + convT1 4624 + convT2 145  =  11137 + 3137 d.
    """
    return 11137 + 3137 * int(d)


def count_params(module: nn.Module) -> int:
    """Number of trainable scalars in ``module``."""
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


@contextlib.contextmanager
def fp32_numerics() -> Iterator[None]:
    """Run the enclosed CAE work in full float32 with deterministic cuDNN.

    The caller's global flags are restored on exit.
    """
    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.backends.cudnn.flags(enabled=True, benchmark=False,
                                        deterministic=True, allow_tf32=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32


class ConvAutoencoder(nn.Module):
    """Two-stage strided-conv encoder and transposed-conv decoder.

    The decoder is not an exact mirror: following the proposal, the decoder
    Linear feeds the first ConvTranspose2d directly (no ReLU in between), so the
    decoder has one hidden nonlinearity and the encoder two.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.d = _check_int(d, "latent dimension d")
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),    # 28 -> 14
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),   # 14 -> 7
            nn.ReLU(),
            nn.Flatten(),                                            # 32*7*7 = 1568
            nn.Linear(FLAT_HIDDEN, self.d),                          # latent (linear)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.d, FLAT_HIDDEN),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1,
                               output_padding=1),                    # 7 -> 14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1,
                               output_padding=1),                    # 14 -> 28
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def _check_int(value: Any, name: str, minimum: int = 1) -> int:
    """Reject bools, floats and values below ``minimum`` for integer settings."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _as_float32_2d(A: Any, n_cols: int, name: str) -> np.ndarray:
    """Validate a numpy-like input and return a C-contiguous float32 copy/view.

    Integer arrays are refused so that 0..255 pixel values cannot slip in.
    """
    if not np.issubdtype(np.asarray(A).dtype, np.floating):
        raise TypeError(f"{name} must be a floating-point array, got {np.asarray(A).dtype}")
    A = np.ascontiguousarray(A, dtype=np.float32)
    if A.ndim != 2 or A.shape[1] != n_cols:
        raise ValueError(f"{name} must have shape (n, {n_cols}), got {A.shape}")
    if not np.isfinite(A).all():
        raise ValueError(f"{name} contains NaN or inf")
    return A


def _reconstruction_mse(net: nn.Module, X: torch.Tensor) -> float:
    """Mean squared error over all pixels of X (shape (n,1,28,28)), float64 accumulation."""
    net.eval()
    sse = torch.zeros((), dtype=torch.float64, device=X.device)
    with torch.inference_mode():
        for s in range(0, X.shape[0], EVAL_BATCH):
            xb = X[s:s + EVAL_BATCH]
            sse += ((net(xb) - xb) ** 2).sum(dtype=torch.float64)
    return float(sse.item()) / X.numel()


class CAEReducer:
    """sklearn-style CAE wrapper for the shared ``evaluate()``.

    ``fit(X)`` trains on X (the 54K training partition when called by
    ``evaluate``) and early-stops on validation MSE.  Validation images are the
    shared 6K split from ``data.load_data()`` unless ``X_val`` is given.  The
    official test set is never read by this class.

    After ``fit``: ``n_params``, ``history_`` (per-epoch train/val MSE),
    ``best_epoch_``, ``best_val_mse_``, ``epochs_run_``, ``stopped_early_``.
    """

    # max_epochs: the proposal suggested 80.  Validation-only pilots (seed 0,
    # d = 2, 32, 128) showed that cap was binding (early stopping never fired);
    # with a 400 cap early stopping fired at epochs 194-217.  See docs/person4.md.
    def __init__(self, d: int, *, lr: float = 1e-3, batch_size: int = 256,
                 max_epochs: int = 400, patience: int = 10, min_delta: float = 0.0,
                 device: str | None = None, X_val: np.ndarray | None = None,
                 verbose: bool = False) -> None:
        self.d = _check_int(d, "d")
        self.batch_size = _check_int(batch_size, "batch_size")
        self.max_epochs = _check_int(max_epochs, "max_epochs")
        self.patience = _check_int(patience, "patience")
        if not (lr > 0 and min_delta >= 0):
            raise ValueError("lr > 0 and min_delta >= 0 are required")
        self.lr = float(lr)
        self.min_delta = float(min_delta)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.X_val = X_val
        self.verbose = verbose
        # Fitted state.  Deliberately no network and no ``n_params`` yet (see module doc).
        self.model_: ConvAutoencoder | None = None
        self.history_: list[dict[str, float]] = []
        self.best_epoch_: int | None = None
        self.best_val_mse_: float | None = None
        self.epochs_run_: int = 0
        self.stopped_early_: bool | None = None
        self.val_source_: str | None = None

    # ------------------------------------------------------------------ config
    def get_params(self) -> dict[str, Any]:
        """Training hyper-parameters (constructor arguments that affect results)."""
        return {"lr": self.lr, "batch_size": self.batch_size, "max_epochs": self.max_epochs,
                "patience": self.patience, "min_delta": self.min_delta}

    def _build_network(self) -> ConvAutoencoder:
        return ConvAutoencoder(self.d)

    def _validation_pixels(self) -> np.ndarray:
        if self.X_val is not None:
            source, X_val = "provided", self.X_val
        else:
            from data import load_data      # shared 54K/6K split; test arrays are not used
            source, X_val = "shared_split", load_data()[2]
        X_val = _as_float32_2d(X_val, N_PIXELS, "X_val")
        if X_val.shape[0] == 0 or X_val.min() < 0.0 or X_val.max() > 1.0:
            raise ValueError("X_val must be non-empty with pixel values in [0, 1]")
        self.val_source_ = source
        return X_val

    def _require_fitted(self) -> ConvAutoencoder:
        if self.model_ is None:
            raise RuntimeError("CAEReducer is not fitted; call fit() or CAEReducer.load()")
        return self.model_

    # ------------------------------------------------------------------- fit
    def fit(self, X: np.ndarray, y: Any = None) -> "CAEReducer":
        """Train a fresh network on X; restore the lowest-validation-MSE epoch."""
        X = _as_float32_2d(X, N_PIXELS, "X")
        if X.shape[0] == 0 or X.min() < 0.0 or X.max() > 1.0:
            raise ValueError("X must be non-empty with pixel values in [0, 1]")
        X_val = self._validation_pixels()
        device = torch.device(self.device)

        with fp32_numerics():
            net = self._build_network().to(device)        # weights drawn on CPU generator
            opt = torch.optim.Adam(net.parameters(), lr=self.lr)
            Xt = torch.from_numpy(X).to(device).view(-1, *IMG_SHAPE)
            Xv = torch.from_numpy(X_val).to(device).view(-1, *IMG_SHAPE)
            n, bs = Xt.shape[0], self.batch_size

            history: list[dict[str, float]] = []
            best_state: dict[str, torch.Tensor] | None = None
            best_val, best_epoch, since_best = math.inf, 0, 0
            stopped_early = False
            t_start = time.perf_counter()
            for epoch in range(1, self.max_epochs + 1):
                net.train()
                perm = torch.randperm(n).to(device)
                sse = torch.zeros((), dtype=torch.float64, device=device)
                for start in range(0, n, bs):
                    xb = Xt[perm[start:start + bs]]
                    loss = F.mse_loss(net(xb), xb)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    sse += loss.detach().double() * xb.shape[0]
                # Running mean over the epoch while the weights change; it is a
                # diagnostic, not directly comparable to the end-of-epoch val MSE.
                train_mse = float(sse.item()) / n
                val_mse = _reconstruction_mse(net, Xv)
                if not (math.isfinite(train_mse) and math.isfinite(val_mse)):
                    raise FloatingPointError(
                        f"non-finite loss at epoch {epoch}: train={train_mse}, val={val_mse}")
                history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse,
                                "elapsed_s": time.perf_counter() - t_start})
                if val_mse < best_val - self.min_delta:
                    best_val, best_epoch, since_best = val_mse, epoch, 0
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                else:
                    since_best += 1
                if self.verbose:
                    flag = " *" if best_epoch == epoch else ""
                    print(f"    epoch {epoch:3d}  train {train_mse:.6f}  val {val_mse:.6f}{flag}",
                          flush=True)
                if since_best >= self.patience:
                    stopped_early = True
                    break

            assert best_state is not None
            net.load_state_dict(best_state)
            net.eval()
            restored_val = _reconstruction_mse(net, Xv)
            if not math.isclose(restored_val, best_val, rel_tol=1e-6, abs_tol=1e-9):
                raise RuntimeError(f"best checkpoint did not restore: val {restored_val} "
                                   f"!= recorded {best_val}")
            del Xt, Xv, opt

        self.model_ = net
        self.history_ = history
        self.best_epoch_ = best_epoch
        self.best_val_mse_ = best_val
        self.epochs_run_ = len(history)
        self.stopped_early_ = stopped_early
        self.n_params = count_params(net)
        return self

    # ------------------------------------------------------------- inference
    def _apply(self, fn: nn.Module, A: np.ndarray, out_dim: int) -> np.ndarray:
        """Run ``fn`` over A in batches; returns float32 numpy, host-synchronised."""
        if A.shape[0] == 0:
            return np.empty((0, out_dim), dtype=np.float32)
        device = torch.device(self.device)
        chunks = []
        with fp32_numerics(), torch.inference_mode():
            for s in range(0, A.shape[0], EVAL_BATCH):
                xb = torch.from_numpy(A[s:s + EVAL_BATCH]).to(device)
                chunks.append(fn(xb).reshape(xb.shape[0], -1).cpu())
            if device.type == "cuda":
                # .cpu() above already blocks on the copy; this makes the guarantee
                # explicit, so evaluate()'s wall-clock timing covers all GPU work.
                torch.cuda.synchronize(device)
        out = torch.cat(chunks).numpy()
        if not np.isfinite(out).all():
            raise FloatingPointError("CAE produced non-finite output")
        return out

    def transform(self, X: np.ndarray) -> np.ndarray:
        """(n, 784) pixels -> (n, d) float32 latent codes."""
        net = self._require_fitted()
        X = _as_float32_2d(X, N_PIXELS, "X")
        return self._apply(net.encoder, X.reshape(-1, *IMG_SHAPE), self.d)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """(n, d) latent codes -> (n, 784) float32 reconstructions in [0, 1]."""
        net = self._require_fitted()
        Z = _as_float32_2d(Z, self.d, "Z")
        return self._apply(net.decoder, Z, N_PIXELS)

    # ----------------------------------------------------------- persistence
    def save(self, path: str, extra: dict[str, Any] | None = None) -> None:
        """Atomically write weights + configuration + training history."""
        net = self._require_fitted()
        payload = {
            "format": CHECKPOINT_FORMAT, "d": self.d, "params": self.get_params(),
            "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
            "history": self.history_, "best_epoch": self.best_epoch_,
            "best_val_mse": self.best_val_mse_, "epochs_run": self.epochs_run_,
            "stopped_early": self.stopped_early_, "n_params": self.n_params,
            "val_source": self.val_source_, "extra": extra or {},
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = f"{path}.tmp-{os.getpid()}"
        try:
            with open(tmp, "wb") as fh:
                torch.save(payload, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "CAEReducer":
        """Rebuild a fitted reducer from ``save()`` output (does not consume RNG)."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"unsupported checkpoint format {payload.get('format')!r}")
        reducer = cls(payload["d"], device=device, **payload["params"])
        with torch.random.fork_rng(devices=[]):
            net = ConvAutoencoder(payload["d"])
        net.load_state_dict(payload["state_dict"])
        reducer.model_ = net.to(torch.device(reducer.device)).eval()
        reducer.history_ = payload["history"]
        reducer.best_epoch_ = payload["best_epoch"]
        reducer.best_val_mse_ = payload["best_val_mse"]
        reducer.epochs_run_ = payload["epochs_run"]
        reducer.stopped_early_ = payload["stopped_early"]
        reducer.val_source_ = payload["val_source"]
        reducer.n_params = count_params(net)
        if reducer.n_params != payload["n_params"]:
            raise ValueError("checkpoint parameter count mismatch")
        return reducer
