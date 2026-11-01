# PCA and raw-pixel baseline

## Scope

Person 2 owns the no-reduction baseline and the PCA sweep. Both use the
shared `evaluate()` function so that the train/test split, metrics, latent
files, and result schema match the other methods.

## Run order

From the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/select_knn.py --representation pca32
```

Copy the printed `Best k` into `config.py` as `KNN_K`. The selection script
uses only the 54K training and 6K validation split; it never uses the official
10K test labels.

Then run:

```bash
python scripts/run_person2.py
python scripts/plot_pca_variance.py --max-components 128
pytest -q
```

Use `--all-seeds` only if the team decides to record deterministic PCA/raw
rows for all five shared seeds. A seed-0 run is sufficient for PCA and raw
pixels because neither has stochastic model training in this project.

For a full 784-component variance curve, use:

```bash
python scripts/plot_pca_variance.py --max-components 784
```

## Outputs

- `results/knn_k_selection.csv`: validation-only k selection results
- `results/results_raw.csv`: raw-pixel baseline row
- `results/results_pca.csv`: PCA rows for the d grid
- `results/pca_explained_variance.csv`: variance-curve data
- `figures/pca_explained_variance.png`: presentation figure
- `latents/Raw_d784_s0.npz`: raw test/train latents for Person 5
- `latents/PCA_d{d}_s0.npz`: PCA latent files for Person 5

## Reproducibility rules

- Do not fit PCA on validation or test images.
- Do not choose `k` using test accuracy.
- Use `svd_solver="full"` exactly as specified by the task.
- Do not manually edit result values.
- If a run must be repeated after changing settings, remove only the affected
  method CSV and latent files, then rerun it. Do not delete the shared split.
