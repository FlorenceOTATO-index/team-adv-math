"""
Shared evaluation: every method goes through evaluate() so all results are comparable.

A "model" is any object with sklearn-style methods that take/return numpy arrays:
    fit(X)                 -> learns from X_train (n, 784)
    transform(X)           -> returns Z (n, d)
    inverse_transform(Z)   -> returns reconstructed X (n, 784)   [optional; no MSE without it]
sklearn's PCA and NMF already work as-is. The CAE just needs a small wrapper class (CAEReducer).

Usage:
    from sklearn.decomposition import PCA
    from evaluate import evaluate
    row = evaluate(PCA(n_components=32, svd_solver="full"), "PCA", d=32, seed=0)
"""
import csv
import os
import time

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier

import config
from data import load_data, set_seed

COLUMNS = ["method", "d", "seed", "knn_k", "knn_acc", "silhouette", "mse",
           "fit_time_s", "encode_time_ms_per_img", "n_params"]


class IdentityReducer:
    """Raw-pixel baseline (d = 784): no reduction, MSE = 0. Used as the k-NN ceiling."""
    def fit(self, X):
        return self

    def transform(self, X):
        return X

    def inverse_transform(self, Z):
        return Z

    n_params = 0


def _batched(fn, X, batch=None):
    batch = batch or config.ENCODE_BATCH
    return np.concatenate([fn(X[i:i + batch]) for i in range(0, len(X), batch)], axis=0)


def _count_params(model):
    """Model size. Uses model.n_params if the wrapper sets it (e.g. the CAE),
    otherwise counts sklearn's learned arrays (components_, mean_)."""
    if hasattr(model, "n_params"):
        return int(model.n_params)
    total = 0
    for attr in ("components_", "mean_"):
        arr = getattr(model, attr, None)
        if arr is not None:
            total += np.asarray(arr).size
    return total


def evaluate(model, method_name, d, seed, results_file=None, save_latents=True, verbose=True):
    """Fit `model` on the 54K training set, score it on the official 10K test set,
    save the latents, and append one row to results/results_{method}.csv.
    Returns the row as a dict."""
    X_train, y_train, _, _, X_test, y_test = load_data()
    set_seed(seed)

    # ---- fit (timed) ----
    t0 = time.perf_counter()
    model.fit(X_train)
    fit_time = time.perf_counter() - t0

    # ---- encode (test set timed, reported per image) ----
    Z_train = _batched(model.transform, X_train)
    t0 = time.perf_counter()
    Z_test = _batched(model.transform, X_test)
    encode_ms = (time.perf_counter() - t0) / len(X_test) * 1000
    Z_train = np.asarray(Z_train, dtype=np.float32)
    Z_test = np.asarray(Z_test, dtype=np.float32)

    # ---- reconstruction MSE on test (clipped to the valid pixel range [0, 1]) ----
    if hasattr(model, "inverse_transform"):
        X_hat = _batched(model.inverse_transform, Z_test)
        X_hat = np.clip(X_hat, 0.0, 1.0)
        mse = float(np.mean((X_test - X_hat) ** 2))
    else:
        mse = float("nan")

    # ---- class separation ----
    # k-NN: train on train latents, predict test latents (val set is only for choosing k / CAE early stopping)
    knn = KNeighborsClassifier(n_neighbors=config.KNN_K, n_jobs=-1)
    knn.fit(Z_train, y_train)
    knn_acc = float(knn.score(Z_test, y_test))
    # silhouette: how well the 10 classes separate in the test latent space (-1 to 1, higher = better)
    sil = float(silhouette_score(Z_test, y_test, sample_size=config.SILHOUETTE_SAMPLE,
                                 random_state=seed if config.SILHOUETTE_SAMPLE else None))

    row = {
        "method": method_name, "d": d, "seed": seed, "knn_k": config.KNN_K,
        "knn_acc": round(knn_acc, 4), "silhouette": round(sil, 4),
        "mse": round(mse, 6) if not np.isnan(mse) else "",
        "fit_time_s": round(fit_time, 3), "encode_time_ms_per_img": round(encode_ms, 5),
        "n_params": _count_params(model),
    }

    # ---- save latents (for t-SNE plots / later analysis) ----
    if save_latents:
        os.makedirs(config.LATENT_DIR, exist_ok=True)
        np.savez_compressed(os.path.join(config.LATENT_DIR, f"{method_name}_d{d}_s{seed}.npz"),
                            Z_train=Z_train, Z_test=Z_test)

    # ---- append to this method's CSV (one file per method avoids GitHub merge conflicts) ----
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = results_file or os.path.join(config.RESULTS_DIR, f"results_{method_name.lower()}.csv")
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow(row)

    if verbose:
        print(f"{method_name:>6} d={d:<3} seed={seed} | kNN acc {row['knn_acc']:.4f} | "
              f"silhouette {row['silhouette']:+.4f} | MSE {mse:.5f} | fit {fit_time:.1f}s")
    return row


def load_all_results():
    """Merge every results_*.csv into one pandas DataFrame (for Person 5's plots)."""
    import glob
    import pandas as pd
    files = sorted(glob.glob(os.path.join(config.RESULTS_DIR, "results_*.csv")))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame(columns=COLUMNS)
