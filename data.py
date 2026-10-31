"""
Data loading, the fixed train/val split, and set_seed().

    from data import load_data, set_seed
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

All X arrays are float32 in [0, 1] with shape (n, 784). Labels are int64 in 0..9.
"""
import gzip
import os
import random
import urllib.request

import numpy as np

import config

_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
_MIRRORS = [
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
    "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/",
]

_CACHE = None


def set_seed(seed):
    """Seed Python, NumPy and (if installed) PyTorch so a run can be repeated exactly."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _download():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    for fname in _FILES.values():
        path = os.path.join(config.DATA_DIR, fname)
        if os.path.exists(path):
            continue
        for base in _MIRRORS:
            try:
                print(f"Downloading {fname} ...")
                urllib.request.urlretrieve(base + fname, path)
                break
            except Exception as e:  # try the next mirror
                print(f"  failed from {base}: {e}")
        else:
            raise RuntimeError(f"Could not download {fname}. Put it in {config.DATA_DIR} manually.")


def _read_images(path):
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return data.reshape(-1, 784)


def _read_labels(path):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=8).astype(np.int64)


def _get_split_indices(n_train_full):
    """Shuffle the official 60K training set once with SPLIT_SEED and save the indices,
    so every group member gets exactly the same 54K/6K split."""
    os.makedirs(config.SPLIT_DIR, exist_ok=True)
    path = os.path.join(config.SPLIT_DIR, "split_indices.npz")
    if os.path.exists(path):
        s = np.load(path)
        return s["train_idx"], s["val_idx"]
    rng = np.random.default_rng(config.SPLIT_SEED)
    perm = rng.permutation(n_train_full)
    val_idx = np.sort(perm[: config.VAL_SIZE])
    train_idx = np.sort(perm[config.VAL_SIZE:])
    np.savez(path, train_idx=train_idx, val_idx=val_idx)
    return train_idx, val_idx


def load_data():
    """Return X_train, y_train, X_val, y_val, X_test, y_test (cached after the first call)."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    _download()
    d = config.DATA_DIR
    X_full = _read_images(os.path.join(d, _FILES["train_images"]))
    y_full = _read_labels(os.path.join(d, _FILES["train_labels"]))
    X_test = _read_images(os.path.join(d, _FILES["test_images"]))
    y_test = _read_labels(os.path.join(d, _FILES["test_labels"]))

    # scale to [0, 1]; already flattened to 784
    X_full = X_full.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    train_idx, val_idx = _get_split_indices(len(X_full))
    _CACHE = (X_full[train_idx], y_full[train_idx],
              X_full[val_idx], y_full[val_idx],
              X_test, y_test)
    return _CACHE
