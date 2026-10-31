"""Select the shared k for k-NN using validation data only.

The project compares representations with the same downstream classifier.
This script chooses k without looking at the official test set, then writes
the validation results to results/knn_k_selection.csv.  Person 1 should copy
the printed best k into config.py before the group runs the final sweeps.

Examples (run from the repository root):
    python scripts/select_knn.py --representation pca32
    python scripts/select_knn.py --representation raw
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

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import config
from data import load_data


def select_best_k(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_val: np.ndarray,
    y_val: np.ndarray,
    candidates: tuple[int, ...] = (1, 3, 5, 7, 9),
    n_jobs: int = -1,
) -> tuple[int, list[dict[str, float | int]]]:
    """Return the best k and validation results for the supplied representation."""
    rows: list[dict[str, float | int]] = []
    for k in candidates:
        classifier = KNeighborsClassifier(n_neighbors=k, n_jobs=n_jobs)
        classifier.fit(Z_train, y_train)
        accuracy = float(accuracy_score(y_val, classifier.predict(Z_val)))
        rows.append({"k": k, "validation_accuracy": accuracy})

    # max() is deterministic: if accuracies tie, the smaller candidate wins.
    best_k = max(rows, key=lambda row: (row["validation_accuracy"], -row["k"]))[
        "k"
    ]
    return int(best_k), rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--representation",
        choices=("raw", "pca32"),
        default="pca32",
        help="Representation used for k selection (default: pca32).",
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        type=int,
        default=[1, 3, 5, 7, 9],
        help="Candidate neighbor counts (default: 1 3 5 7 9).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    X_train, y_train, X_val, y_val, _, _ = load_data()

    if args.representation == "raw":
        Z_train, Z_val = X_train, X_val
    else:
        reducer = PCA(n_components=32, svd_solver="full")
        reducer.fit(X_train)
        Z_train = reducer.transform(X_train)
        Z_val = reducer.transform(X_val)

    best_k, rows = select_best_k(
        Z_train,
        y_train,
        Z_val,
        y_val,
        candidates=tuple(args.candidates),
    )

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(config.RESULTS_DIR, "knn_k_selection.csv")
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["representation", "k", "validation_accuracy"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "representation": args.representation,
                    "k": row["k"],
                    "validation_accuracy": f'{row["validation_accuracy"]:.6f}',
                }
            )

    print(f"Representation used for selection: {args.representation}")
    for row in rows:
        print(f'k={row["k"]:>2}: validation accuracy={row["validation_accuracy"]:.4f}')
    print(f"Best k: {best_k}")
    print(f"Saved validation results to {output_path}")
    print(f"Set KNN_K = {best_k} in config.py before running final evaluations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
