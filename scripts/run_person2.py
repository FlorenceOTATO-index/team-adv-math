"""Run Person 2's raw-pixel baseline and PCA sweep.

Run from the repository root after Person 1 has set KNN_K in config.py:
    python scripts/run_person2.py

PCA uses the shared evaluate() function, full SVD, the shared official
Fashion-MNIST split, and the shared k-NN value.  PCA is deterministic, so the
default run uses seed 0.  Use --all-seeds only if the team wants duplicate
deterministic rows for all shared seeds.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.decomposition import PCA

import config
from evaluate import IdentityReducer, evaluate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help="Run raw/PCA for every value in config.SEEDS instead of seed 0.",
    )
    parser.add_argument(
        "--skip-raw",
        action="store_true",
        help="Skip the raw-pixel baseline.",
    )
    parser.add_argument(
        "--skip-pca",
        action="store_true",
        help="Skip the PCA sweep.",
    )
    return parser.parse_args()


def _completed(method: str, d: int, seed: int) -> bool:
    """Avoid accidental duplicate rows when a run is restarted."""
    path = os.path.join(config.RESULTS_DIR, f"results_{method.lower()}.csv")
    if not os.path.exists(path):
        return False
    with open(path, newline="") as handle:
        return any(
            row.get("method") == method
            and int(row.get("d", -1)) == d
            and int(row.get("seed", -1)) == seed
            for row in csv.DictReader(handle)
        )


def run_raw(seeds: list[int], force: bool = False) -> None:
    for seed in seeds:
        if not force and _completed("Raw", 784, seed):
            print(f"Skipping existing Raw d=784 seed={seed}")
            continue
        evaluate(
            IdentityReducer(),
            "Raw",
            d=784,
            seed=seed,
            save_latents=True,
        )


def run_pca(seeds: list[int], force: bool = False) -> None:
    for d in config.D_GRID:
        for seed in seeds:
            if not force and _completed("PCA", d, seed):
                print(f"Skipping existing PCA d={d} seed={seed}")
                continue
            evaluate(
                PCA(n_components=d, svd_solver="full"),
                "PCA",
                d=d,
                seed=seed,
                save_latents=True,
            )


def main() -> int:
    args = _parse_args()
    seeds = list(config.SEEDS) if args.all_seeds else [config.SEEDS[0]]

    if not args.skip_raw:
        run_raw(seeds)
    if not args.skip_pca:
        run_pca(seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
