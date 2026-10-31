"""Create the cumulative explained-variance curve for PCA.

Examples (run from the repository root):
    python scripts/plot_pca_variance.py --max-components 128
    python scripts/plot_pca_variance.py --max-components 784
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import config
from data import load_data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-components",
        type=int,
        default=max(config.D_GRID),
        help="Largest component count plotted (default: max D_GRID).",
    )
    parser.add_argument(
        "--output-stem",
        default="pca_explained_variance",
        help="Output filename stem in figures/ and results/.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    X_train, _, _, _, _, _ = load_data()
    max_components = min(args.max_components, X_train.shape[1])

    reducer = PCA(n_components=max_components, svd_solver="full")
    reducer.fit(X_train)

    components = np.arange(1, max_components + 1)
    cumulative = np.cumsum(reducer.explained_variance_ratio_)
    table = pd.DataFrame(
        {
            "component": components,
            "explained_variance_ratio": reducer.explained_variance_ratio_,
            "cumulative_explained_variance": cumulative,
        }
    )

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    csv_path = os.path.join(config.RESULTS_DIR, f"{args.output_stem}.csv")
    figure_path = os.path.join(ROOT, "figures", f"{args.output_stem}.png")
    table.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(components, cumulative, color="#1f77b4", linewidth=2)
    project_d = [d for d in config.D_GRID if d <= max_components]
    if project_d:
        plt.scatter(
            project_d,
            cumulative[np.array(project_d) - 1],
            color="#d62728",
            zorder=3,
            label="Project d values",
        )
    plt.xlabel("Number of PCA components")
    plt.ylabel("Cumulative explained variance")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()

    print(f"Saved data to {csv_path}")
    print(f"Saved figure to {figure_path}")
    print(f"Cumulative variance at d=32: {cumulative[31]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
