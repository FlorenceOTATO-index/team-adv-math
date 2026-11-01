"""Person 5: build all summary figures from results/results_*.csv.

Usage (from the repo root):
    python scripts/make_figures.py

Merges every results/results_<method>.csv (the files evaluate() writes),
so new NMF / CAE files appear automatically. results.csv is ignored.
- Multiple seeds per (method, d) are averaged, with a ±1 std band.
- The "Raw" row (d = 784) is drawn as a dashed reference line, not a curve.
Writes PNGs into figures/.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGS = ROOT / "figures"

# Fixed colors so a method looks the same on every slide.
COLORS = {"PCA": "#1f77b4", "NMF": "#ff7f0e", "CAE": "#2ca02c", "FCAE": "#9467bd"}
MARKERS = {"PCA": "o", "NMF": "s", "CAE": "^", "FCAE": "D"}
REF_METHOD = "Raw"

METRICS = {
    "knn_acc": ("k-NN test accuracy", "knn_acc_vs_d.png"),
    "mse": ("Reconstruction MSE (test, [0,1] pixels)", "mse_vs_d.png"),
    "silhouette": ("Silhouette score (true labels)", "silhouette_vs_d.png"),
}


def load():
    files = sorted(RESULTS_DIR.glob("results_*.csv"))
    if not files:
        raise SystemExit(f"No results_*.csv files in {RESULTS_DIR}")
    print("Reading:", ", ".join(f.name for f in files))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["method"] = df["method"].replace({"RAW": "Raw", "raw": "Raw"})
    ref = df[df["method"] == REF_METHOD]
    runs = df[df["method"] != REF_METHOD]
    numeric = [c for c in runs.columns if c not in ("method", "d", "seed", "hardware")]
    agg = runs.groupby(["method", "d"])[numeric].agg(["mean", "std"]).reset_index()
    return agg, ref


def style_d_axis(ax, ds):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ds)
    ax.set_xticklabels([str(int(d)) for d in ds])
    ax.set_xlabel("Latent dimension d")
    ax.grid(True, alpha=0.3)


def plot_metric_vs_d(agg, ref, metric, ylabel, fname, logy=False, show_ref=True):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ds = sorted(agg["d"].unique())
    for method, g in agg.groupby("method"):
        g = g.sort_values("d")
        mean, std = g[(metric, "mean")], g[(metric, "std")].fillna(0)
        ax.plot(g["d"], mean, marker=MARKERS.get(method, "o"),
                color=COLORS.get(method), label=method)
        ax.fill_between(g["d"], mean - std, mean + std,
                        color=COLORS.get(method), alpha=0.15)
    if show_ref and metric != "mse" and not ref.empty:  # raw MSE is 0 by definition
        val = ref[metric].mean()
        ax.axhline(val, ls="--", color="gray", label=f"Raw pixels (784-d): {val:.3f}")
    style_d_axis(ax, ds)
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / fname, dpi=200)
    plt.close(fig)


def plot_acc_vs_cost(agg, ref):
    """Quality vs. cost. Points of one method are joined in order of d;
    only the smallest and largest d are labeled to avoid overlapping text."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, g in agg.groupby("method"):
        g = g.sort_values("d")
        x, y = g[("fit_time_s", "mean")].values, g[("knn_acc", "mean")].values
        ds = g["d"].values
        ax.plot(x, y, marker=MARKERS.get(method, "o"), color=COLORS.get(method),
                lw=1, alpha=0.9, label=method)
        for i in (0, len(ds) - 1):
            ax.annotate(f"d={int(ds[i])}", (x[i], y[i]), textcoords="offset points",
                        xytext=(8, -4), fontsize=8, color=COLORS.get(method),
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
    if not ref.empty:
        ax.axhline(ref["knn_acc"].mean(), ls="--", color="gray", label="Raw pixels")
    ax.set_xscale("log")
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi * 2.5)  # room for labels on the right
    ax.set_xlabel("Fit time (s, log scale)")
    ax.set_ylabel("k-NN test accuracy")
    ax.set_title("Accuracy vs. training cost (lines connect d = 2 \u2192 128)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "acc_vs_fit_time.png", dpi=200)
    plt.close(fig)


def summary_table(agg, ref):
    """Slide-ready table: mean (± std when several seeds) per method and d."""
    rows = []
    for _, r in agg.sort_values(["method", "d"]).iterrows():
        def fmt(col, nd):
            m, s = r[(col, "mean")], r[(col, "std")]
            return f"{m:.{nd}f}" if pd.isna(s) or s == 0 else f"{m:.{nd}f} ± {s:.{nd}f}"
        rows.append({"method": r["method"].iloc[0] if hasattr(r["method"], "iloc") else r["method"],
                     "d": int(r["d"].iloc[0] if hasattr(r["d"], "iloc") else r["d"]),
                     "kNN acc": fmt("knn_acc", 4), "MSE": fmt("mse", 5),
                     "silhouette": fmt("silhouette", 4), "fit (s)": fmt("fit_time_s", 2),
                     "params": int(r[("n_params", "mean")])})
    for _, r in ref.iterrows():
        rows.append({"method": "Raw", "d": 784, "kNN acc": f"{r['knn_acc']:.4f}",
                     "MSE": "0", "silhouette": f"{r['silhouette']:.4f}",
                     "fit (s)": "-", "params": 0})
    out = pd.DataFrame(rows)
    out.to_csv(FIGS / "summary_table.csv", index=False)
    return out


def plot_optional_extras():
    """k selection plot, if results/knn_k_selection.csv exists (column names guessed)."""
    # PCA explained variance is plotted by Person 2 (figures/pca_explained_variance.png);
    # not redrawn here so their figure is never overwritten.
    ks = ROOT / "results" / "knn_k_selection.csv"
    if ks.exists():
        df = pd.read_csv(ks)
        kcol = next((c for c in df.columns if c.lower() in ("k", "knn_k")), df.columns[0])
        acol = next((c for c in df.columns if "acc" in c.lower()), df.columns[-1])
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(df[kcol], df[acol], marker="o")
        best = df.loc[df[acol].idxmax()]
        ax.scatter([best[kcol]], [best[acol]], s=120, facecolors="none",
                   edgecolors="red", label=f"chosen k = {int(best[kcol])}")
        ax.set_xlabel("k")
        ax.set_ylabel("Validation accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIGS / "knn_k_selection.png", dpi=200)
        plt.close(fig)


def main():
    FIGS.mkdir(exist_ok=True)
    agg, ref = load()
    for metric, (ylabel, fname) in METRICS.items():
        plot_metric_vs_d(agg, ref, metric, ylabel, fname)
    plot_metric_vs_d(agg, ref, "fit_time_s", "Fit time (s, log scale)",
                     "fit_time_vs_d.png", logy=True, show_ref=False)
    plot_acc_vs_cost(agg, ref)
    print(summary_table(agg, ref).to_string(index=False))
    plot_optional_extras()
    print(f"\nFigures written to {FIGS}")


if __name__ == "__main__":
    main()