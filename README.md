# Group 5 - PCA vs NMF vs CAE on Fashion-MNIST

Shared pipeline so every method uses the **same split, seeds, and metrics**.

This repository is an adapted replication of Fournier and Aloise's study. The
paper compares dimensionality-reduction methods by training k-NN on each
learned projection. Our project keeps that classification protocol while
adding NMF, a convolutional autoencoder (CAE), reconstruction error, model
size, silhouette score, and t-SNE visualizations.

## Files
| File | What it does |
|---|---|
| `config.py` | All shared settings (seeds, d grid, k for k-NN, paths). Tell the group before changing anything. |
| `data.py` | Downloads Fashion-MNIST, scales to [0,1], flattens to 784, makes the fixed 54K/6K train/val split (seed 42, indices saved in `splits/`), and `set_seed()`. |
| `evaluate.py` | `evaluate(model, method_name, d, seed)` — fits, times, scores, saves latents, appends a row to `results/results_<method>.csv`. Also `IdentityReducer` (raw-pixel baseline) and `load_all_results()`. |
| `run_pca_example.py` | Example: raw pixels + PCA over the whole d grid. |
| `scripts/select_knn.py` | Selects k using validation accuracy only; does not inspect test labels. |
| `scripts/run_person2.py` | Runs the raw-pixel baseline and Person 2's PCA sweep. |
| `scripts/plot_pca_variance.py` | Creates the cumulative explained-variance table and figure. |
| `scripts/merge_results.py` | Merges per-method CSVs into `results/results.csv` for Person 5. |
| `docs/person2.md` | Person 2's exact run order, outputs, and reproducibility rules. |
| `tests/` | Lightweight tests for shared interfaces and Person 2's k-selection code. |

## Setup
- Official split: **54K train / 6K val** (from the 60K train set) and the **official 10K test**.
  This follows the paper's data organization. It is not the previously discussed
  custom 60/10/10/20 split; do not mix the two protocols without documenting a
  deliberate project change.
- Seeds: `[0, 1, 2, 3, 4]`. d grid: `[2, 4, 8, 16, 32, 64, 128]`.
- Data downloads automatically on the first run.

## How to use
```python
from sklearn.decomposition import PCA, NMF
from evaluate import evaluate

evaluate(PCA(n_components=32, svd_solver="full"), "PCA", d=32, seed=0)
evaluate(NMF(n_components=32, init="nndsvda", random_state=0, max_iter=500), "NMF", d=32, seed=0)
```
In Colab: clone the repo, `%cd` into it, then run the cells above.

## Person 2 commands

```bash
python -m pip install -r requirements.txt
python scripts/select_knn.py --representation pca32
# Copy the printed best k into config.py.
python scripts/run_person2.py
python scripts/plot_pca_variance.py --max-components 128
python scripts/merge_results.py
pytest -q
```

The current validation run selected `KNN_K = 9` using PCA at `d=32`.

**For the CAE:** wrap the model in a class with `fit(X)`, `transform(X)`, `inverse_transform(Z)`
that take and return **numpy** arrays of shape `(n, 784)` / `(n, d)`, and set `self.n_params`
(number of trainable parameters). Use the validation set from `load_data()` for early stopping inside `fit`.

## What gets recorded (one row per run)
`method, d, seed, knn_k, knn_acc, silhouette, mse, fit_time_s, encode_time_ms_per_img, n_params`

- **knn_acc**: k-NN trained on train latents, tested on test latents (class separation).
- **silhouette**: on the 10K test latents with true labels, −1 to 1, higher = classes better separated.
- **mse**: test reconstruction error, reconstructions clipped to [0, 1].
- Latents saved to `latents/<method>_d<d>_s<seed>.npz` (`Z_train`, `Z_test`) for the t-SNE plots.

Each method writes its own CSV to avoid merge conflicts; `load_all_results()`
combines them. Run `python scripts/merge_results.py` after all methods finish
to create the aggregate `results/results.csv` consumed by Person 5.
`data/` and `latents/` are git-ignored (too large).

## Still to set
- `KNN_K` in `config.py` is a placeholder (5). Person 2 picks the final k on the validation set.
