"""Person 5: t-SNE panels + reconstruction grid at one latent size (default d = 32).

Usage (from the repo root):
    python scripts/make_tsne_grid.py                       # looks for NMF/CAE .npz in config.LATENT_DIR
    python scripts/make_tsne_grid.py --latent-dir ~/Downloads/latents   # e.g. files downloaded from Drive

What it uses:
    Raw    : the test images themselves (load_data)
    PCA    : refit here (deterministic, seconds) -> identical to Person 2's run
    NMF    : NMF_d{d}_s0.npz if found, otherwise refit here (nndsvda init is deterministic;
             check NMF_KWARGS below matches Person 3's settings)
    CAE    : CAE_d{d}_s0.npz with Z_test (t-SNE) and X_hat (first 8 test images, grid)
A missing CAE file is skipped with a "pending" panel, so you can run this now and rerun later.

Outputs (figures/):
    tsne_d{d}.png            one panel per method, colored by class, silhouette + trustworthiness in corner
    recon_grid_d{d}.png      rows = Original, PCA, NMF, CAE; columns = the same 8 test images
    tsne_scores_d{d}.csv     silhouette / trustworthiness per panel
    splits/tsne_idx.npy      the fixed 5,000-image subsample (created once, reused forever)
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import config  # noqa: E402
from data import load_data  # noqa: E402

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
PER_CLASS = 500          # 500 x 10 = 5,000 points per t-SNE panel
N_GRID = 8               # first 8 test images, matching X_hat[:8] saved by evaluate()
SEED = 0
# Must match Person 3's NMF settings (see their run script / nmf_convergence.csv).
NMF_KWARGS = dict(init="nndsvda", random_state=0, max_iter=1000)


def balanced_subsample(y):
    path = ROOT / "splits" / "tsne_idx.npy"
    if path.exists():
        return np.load(path)
    rng = np.random.default_rng(SEED)
    idx = np.concatenate([rng.choice(np.where(y == c)[0], PER_CLASS, replace=False)
                          for c in range(10)])
    idx.sort()
    np.save(path, idx)
    print(f"Saved t-SNE subsample indices -> {path.relative_to(ROOT)} (commit this)")
    return idx


def run_tsne(Z):
    if Z.shape[1] > 50:  # standard practice: PCA to 50 dims before t-SNE (only matters for raw pixels)
        Z = PCA(n_components=50, random_state=SEED).fit_transform(Z)
    return TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED).fit_transform(Z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--latent-dir", default=config.LATENT_DIR)
    args = ap.parse_args()
    d, latent_dir = args.d, Path(os.path.expanduser(args.latent_dir))
    figs = ROOT / "figures"
    figs.mkdir(exist_ok=True)

    X_train, y_train, _, _, X_test, y_test = load_data()
    y_test = np.asarray(y_test)
    idx = balanced_subsample(y_test)

    # ---- gather latents (Z_test) and reconstructions (first 8 test images) ----
    Z, recon = {"Raw": X_test}, {}
    pca = PCA(n_components=d, svd_solver="full").fit(X_train)
    Z["PCA"] = pca.transform(X_test)
    recon["PCA"] = np.clip(pca.inverse_transform(Z["PCA"][:N_GRID]), 0, 1)

    for m in ("NMF", "CAE"):
        f = latent_dir / f"{m}_d{d}_s0.npz"
        if m == "NMF" and not f.exists():
            print(f"{f.name} not found - refitting NMF (d={d}) here, ~15-60 s ...")
            nmf = NMF(n_components=d, **NMF_KWARGS).fit(X_train)
            Z["NMF"] = nmf.transform(X_test)
            recon["NMF"] = np.clip(nmf.inverse_transform(Z["NMF"][:N_GRID]), 0, 1)
            mse = float(np.mean((X_test - np.clip(nmf.inverse_transform(Z["NMF"]), 0, 1)) ** 2))
            print(f"  refit NMF test MSE = {mse:.6f}  (should match results_nmf.csv for d={d}; "
                  f"0.017658 at d=32)")
            continue
        if not f.exists():
            print(f"[pending] {f} not found - {m} panel left blank")
            continue
        npz = np.load(f, allow_pickle=True)
        Z[m] = npz["Z_test"]
        if "X_hat" in npz.files and npz["X_hat"].shape != ():
            recon[m] = npz["X_hat"][:N_GRID]
        else:
            print(f"[note] {f.name} has no X_hat - {m} row of the grid left blank")

    # ---- t-SNE panels ----
    order = ["Raw", "PCA", "NMF", "CAE"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    cmap = plt.get_cmap("tab10")
    scores = []
    for ax, m in zip(axes, order):
        ax.set_xticks([]); ax.set_yticks([])
        title = "Raw pixels (784-d)" if m == "Raw" else f"{m} (d = {d})"
        ax.set_title(title)
        if m not in Z:
            ax.text(0.5, 0.5, "pending", ha="center", va="center",
                    transform=ax.transAxes, color="gray", fontsize=14)
            continue
        Zs = np.asarray(Z[m])[idx].astype(np.float32)
        print(f"t-SNE: {m} ...")
        Y = run_tsne(Zs)
        sil = silhouette_score(Zs, y_test[idx])
        tw = trustworthiness(Zs, Y, n_neighbors=10)
        scores.append({"method": m, "d": 784 if m == "Raw" else d,
                       "silhouette_latent": round(sil, 4), "trustworthiness": round(tw, 4)})
        for c in range(10):
            pts = y_test[idx] == c
            ax.scatter(Y[pts, 0], Y[pts, 1], s=2, color=cmap(c), label=CLASSES[c])
        ax.text(0.02, 0.02, f"silhouette {sil:+.3f}\ntrustworthiness {tw:.3f}",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=10, markerscale=5, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(figs / f"tsne_d{d}.png", dpi=200)
    plt.close(fig)
    pd.DataFrame(scores).to_csv(figs / f"tsne_scores_d{d}.csv", index=False)

    # ---- reconstruction grid ----
    rows = [("Original", X_test[:N_GRID])] + [(m, recon.get(m)) for m in ("PCA", "NMF", "CAE")]
    fig, axes = plt.subplots(len(rows), N_GRID, figsize=(N_GRID * 1.2, len(rows) * 1.35))
    for r, (name, imgs) in enumerate(rows):
        for c in range(N_GRID):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            if imgs is not None:
                ax.imshow(np.asarray(imgs[c]).reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            elif c == N_GRID // 2:
                ax.text(0, 0.5, "pending", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=10)
            if c == 0:
                ax.set_ylabel(name if name == "Original" else f"{name}\nd={d}", fontsize=9)
            if r == 0:
                ax.set_title(CLASSES[int(y_test[c])], fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / f"recon_grid_d{d}.png", dpi=200)
    plt.close(fig)

    print(pd.DataFrame(scores).to_string(index=False))
    print(f"\nWrote figures/tsne_d{d}.png, figures/recon_grid_d{d}.png, figures/tsne_scores_d{d}.csv")


if __name__ == "__main__":
    main()