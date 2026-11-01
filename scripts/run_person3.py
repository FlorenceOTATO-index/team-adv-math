"""
Run Person 3's NMF sweep.

Run from the repository root, on one machine only (so all NMF timings are comparable):
    python scripts/run_person3.py

NMF uses the shared evaluate() function, the shared official Fashion-MNIST split,
and the shared k-NN value. init="nndsvda" starts NMF the same way every time, so
the default run uses seed 0. Use --all-seeds only if the team wants rows for every
shared seed.

max_iter = 1000 was chosen in explore_nmf.ipynb.

The d grid runs largest first, because large d takes longest. Runs already in
results/results_nmf.csv are skipped, so a stopped sweep can be restarted safely.

Outputs:
    results/results_nmf.csv       (written by evaluate())
    results/nmf_convergence.csv   (iterations used and whether each run converged)
    figures/nmf_basis_d32.png     (the 32 learned NMF parts)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")  # save figures without opening windows
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

import config
from evaluate import evaluate

MAX_ITER = 1000  # confirmed in explore_nmf.ipynb
INIT = "nndsvda"
BASIS_FIGURE_D = 32
FIGURE_DIR = ROOT / "figures"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all-seeds", action="store_true",
                        help="Run NMF for every value in config.SEEDS instead of seed 0.")
    parser.add_argument("--force", action="store_true",
                        help="Rerun even if a row already exists (adds a duplicate row).")
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


def _log_convergence(d: int, seed: int, n_iter: int, converged: bool, wall_min: float) -> None:
    """Append one row per run, for the slide's convergence note."""
    path = os.path.join(config.RESULTS_DIR, "nmf_convergence.csv")
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(["d", "seed", "init", "max_iter", "n_iter", "converged", "total_minutes"])
        writer.writerow([d, seed, INIT, MAX_ITER, n_iter, converged, round(wall_min, 2)])


def plot_basis(model: NMF, path: Path) -> None:
    """Show each learned NMF part as a 28x28 image."""
    n = model.components_.shape[0]
    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 1.6 * rows + 0.6))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            ax.imshow(model.components_[i].reshape(28, 28), cmap="gray")
            ax.set_title(f"part {i}", fontsize=8)
    fig.suptitle(f"NMF basis images (d = {n})", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_nmf(seeds: list[int], force: bool = False) -> None:
    FIGURE_DIR.mkdir(exist_ok=True)
    for d in sorted(config.D_GRID, reverse=True):
        for seed in seeds:
            if not force and _completed("NMF", d, seed):
                print(f"Skipping existing NMF d={d} seed={seed}")
                continue

            model = NMF(n_components=d, init=INIT, random_state=seed, max_iter=MAX_ITER)
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                evaluate(model, "NMF", d=d, seed=seed, save_latents=True)
            wall_min = (time.perf_counter() - start) / 60

            converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
            _log_convergence(d, seed, model.n_iter_, converged, wall_min)
            note = "" if converged else "  <- did NOT fully converge"
            print(f"       used {model.n_iter_}/{MAX_ITER} iterations, "
                  f"{wall_min:.1f} min total{note}", flush=True)

            if d == BASIS_FIGURE_D and seed == seeds[0]:
                out = FIGURE_DIR / f"nmf_basis_d{d}.png"
                plot_basis(model, out)
                print(f"       saved {out.relative_to(ROOT)}", flush=True)


def main() -> int:
    args = _parse_args()
    seeds = list(config.SEEDS) if args.all_seeds else [config.SEEDS[0]]
    run_nmf(seeds, force=args.force)
    print("\nDone. Next: python scripts/merge_results.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
