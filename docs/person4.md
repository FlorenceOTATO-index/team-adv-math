# Person 4: Convolutional autoencoder (CAE)

This page describes the CAE implementation, its frozen training protocol, how to
rerun it, and which files Person 5 needs. Results are summarised at the end. The
full numbers live in the result files, and the interpretation is in
`docs/mathematical_analysis.md` §5.

## Files

| File | Purpose |
|---|---|
| `models/cae.py` | `ConvAutoencoder` (PyTorch network) and `CAEReducer` (sklearn-style wrapper used by the shared `evaluate()`). |
| `scripts/run_person4.py` | Command-line runner: `smoke`, `pilot`, `freeze`, `final`, `report`. |
| `tests/test_person4.py` | Unit and integration tests. They use synthetic data, except one artifact check that runs only after the official d=32 seed=0 run exists. |
| `results/results_cae.csv` | Official CAE rows, shared schema. Read by `scripts/merge_results.py`. |
| `results/cae_protocol.json` | The frozen training protocol and its fingerprint. |
| `results/cae_runs/CAE_d{d}_s{seed}.json` | Per-run metadata: fingerprint, CSV row, epoch-by-epoch train/val MSE, best epoch, checkpoint hash, GPU and software versions. |
| `results/cae_pilot/CAE_d{d}_s{seed}_p{fingerprint}.json` | Validation-only pilot runs (development evidence). The suffix is the first 8 characters of the protocol fingerprint, so pilots under different protocols never overwrite each other. `p0149fbcb` is the 80-epoch protocol, `pd7dfdac2` is the 400-epoch study, and the pilot under the frozen code carries the frozen fingerprint. |
| `results/cae_summary.csv` | Mean, std and count over the five seeds for each d (written by `report`). |
| `results/cae_run_diagnostics.csv` | One line per run: best epoch, epochs run, checkpoint reload check (written by `report`). |
| `results/cae_vs_pca_mcnemar.csv` | Post-hoc paired test of CAE vs PCA k-NN predictions per run: 2×2 counts, exact McNemar p, Holm-adjusted p over the 35 runs (`docs/mathematical_analysis.md`, Appendix B). |
| `figures/cae_*.png` | Pilot curves and validation reconstructions, final training curves, final test reconstructions. |
| `checkpoints/cae/*.pt` | Best-validation weights for every run (git-ignored, about 0.1 to 1.6 MB each). |
| `latents/CAE_d{d}_s{seed}.npz` | `Z_train`, `Z_test` from `evaluate()` (git-ignored). `CAE_d32_s0.npz` also holds `X_hat`. |

No shared file (`config.py`, `data.py`, `evaluate.py`) was modified.

## Architecture

The architecture is the same for every d. Only the two bottleneck Linear layers change size.

| Stage | Layer | Output shape |
|---|---|---|
| input | | (B, 1, 28, 28) |
| encoder | Conv2d(1, 16, k=3, s=2, p=1) + ReLU | (B, 16, 14, 14) |
| | Conv2d(16, 32, k=3, s=2, p=1) + ReLU | (B, 32, 7, 7) |
| | Flatten | (B, 1568) |
| | Linear(1568, d), no activation | (B, d) |
| decoder | Linear(d, 1568) | (B, 1568) |
| | Reshape | (B, 32, 7, 7) |
| | ConvTranspose2d(32, 16, k=3, s=2, p=1, output_padding=1) + ReLU | (B, 16, 14, 14) |
| | ConvTranspose2d(16, 1, k=3, s=2, p=1, output_padding=1) + Sigmoid | (B, 1, 28, 28) |

Trainable parameters, counted by hand and checked against PyTorch for every d in the grid:

```
conv1 160 + conv2 4640 + encoder Linear 1569 d + decoder Linear (1568 d + 1568)
+ convT1 4624 + convT2 145  =  11137 + 3137 d          (d = 32: 111,521)
```

| d | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|
| n_params | 17,411 | 23,685 | 36,233 | 61,329 | 111,521 | 211,905 | 412,673 |

Unlike the ReLU after each encoder convolution, the proposal puts no activation
between the decoder Linear and the first transposed convolution. Those two layers
therefore compose into one linear map, and the decoder has one hidden
nonlinearity against the encoder's two. The code follows the proposal exactly.
Adding that ReLU would not change the parameter count. It is a possible variant
for the team to discuss, not something that was tested.

## Wrapper contract (`CAEReducer`)

* `fit(X)` takes `X` with shape (n, 784), float values in [0, 1]. It builds a **new** network and trains it on `X`. It early-stops on the shared 6K validation split, loaded via `data.load_data()`, or on `X_val` if one is passed. Afterwards it restores the epoch with the lowest validation MSE.
* `transform(X)` maps (n, 784) to (n, d) float32. `inverse_transform(Z)` maps (n, d) to (n, 784) float32 in [0, 1]. Both run in batches under `torch.inference_mode()`, check that outputs are finite, and return only after all GPU work has finished.
* `n_params` is the number of trainable PyTorch parameters. It exists only after `fit`, which is the only point at which `evaluate()` reads it.
* `save(path)` and `CAEReducer.load(path)` store and restore weights, hyper-parameters and the training history. Writes are atomic. Loading does not consume random numbers.

**Seeds.** `evaluate()` calls `set_seed(seed)` after the model object has been
built. For that reason `__init__` creates no weights. The network is built
inside `fit()` on the CPU from the freshly seeded generator, and then moved to
the GPU. Batch order comes from `torch.randperm` on the same generator. No
DataLoader or worker processes are used. Every CAE call runs with TF32 disabled
and deterministic cuDNN kernels, and the caller's global flags are restored
afterwards. On the same GPU and software stack, training is bitwise
reproducible, including across separate processes. This was verified in the tests and pilots.

## Training protocol (frozen)

| Setting | Value |
|---|---|
| Loss | pixel-wise MSE (mean over batch and 784 pixels) |
| Optimizer | Adam, lr 1e-3, betas (0.9, 0.999), eps 1e-8, no weight decay, constant lr |
| Batch size | 256, reshuffled every epoch, last partial batch kept |
| Epoch limit | 400 |
| Early stopping | stop after 10 epochs without a strict decrease in validation MSE |
| Checkpoint | weights from the epoch with the lowest validation MSE |
| Initialisation | PyTorch default layer initialisation |
| Validation data | shared 6K split (`splits/split_indices.npz`), never used for gradients |
| Numerics | float32, TF32 off, cuDNN deterministic |

The fingerprint is stored in `results/cae_protocol.json`. It is a hash of:
* the settings above;
* the layer list of the architecture;
* the SHA-256 of `models/cae.py`, `evaluate.py` and `data.py`;
* the split file;
* the shared evaluation settings (`KNN_K`, silhouette sample size, encode batch, CSV columns).

`final` refuses to run if the current code produces a different fingerprint. Any
edit to those three files, even a comment, therefore requires a new pilot and
freeze before more official runs. `report` re-checks committed runs against the
frozen fingerprint and only warns if the current code differs.

`train_mse` in the histories is the running mean over each epoch while the
weights change. It is a diagnostic and slightly overstates the end-of-epoch
training error, so it is not directly comparable with the validation MSE.

### Why the epoch limit is 400 rather than the proposed 80

The proposal suggested 80 epochs with a patience of 10. Validation-only pilots
at d = 2, 32 and 128 (seed 0) showed that the 80-epoch cap was the real stopping
rule. Validation MSE was still decreasing at epoch 80 for every d, early
stopping never fired, and the train/validation gap stayed small, so the network
was not overfitting.

Before the longer pilot was run, this decision rule was stated in the working
session: if training to the early-stopping point improved the best validation
MSE by more than 2 % at any pilot d, the cap would be raised so that early
stopping governs. The rule was not written to a repository file before the run,
so its timing cannot be checked from the repository alone.

| d | best val MSE, 80-epoch cap (epoch) | best val MSE, 400-epoch cap (epoch, stopped at) | improvement |
|---|---|---|---|
| 2 | 0.031505 (80) | 0.030457 (195, 205) | 3.33 % |
| 32 | 0.007571 (80) | 0.007234 (207, 217) | 4.46 % |
| 128 | 0.003405 (79) | 0.003235 (184, 194) | 5.01 % |

Measured against the literal epoch-80 value, the d = 128 improvement is 5.20 %.
The limit was raised to 400. The learning rate, batch size and patience were
not changed. No test data was used for this decision.

The decision rests on one seed and three of the seven d values. To guard
against that, every official run records whether it reached the cap
(`epoch_budget_binding` in its metadata), and `report` lists any run that did.
The evidence is in `results/cae_pilot/` and `figures/cae_pilot_curves_*`.

## How to run

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0   # pick an idle GPU; index as in nvidia-smi
python -m pytest -q tests/                                   # unit + integration tests
python scripts/run_person4.py --mode smoke                   # ~1 min sanity checks, writes no results
python scripts/run_person4.py --mode pilot --d 32 --seed 0   # validation only
python scripts/run_person4.py --mode freeze                  # once, after reviewing the pilot
python scripts/run_person4.py --mode final --all             # 7 d x 5 seeds through evaluate()
python scripts/run_person4.py --mode final --d 32 --seed 0   # a single official run
python scripts/run_person4.py --mode report                  # whole grid: reload checks, summary, figures
python scripts/merge_results.py                              # rebuild results/results.csv
```

* **Resuming.** Rerunning `final` skips any run whose row is already in `results_cae.csv` with matching metadata and the same protocol fingerprint. If a row is duplicated, has no metadata, differs from its metadata, or was produced under a different fingerprint, `final` stops before training anything and explains why.
* **Commit point.** A run's row is appended only after its checkpoint, reload check, metadata and (for d=32 seed=0) Person 5's file have been written. An interrupted run leaves no row and is simply rerun.
* **Changing the protocol.** Agree it with the team first. Then move `results/results_cae.csv`, `results/cae_runs/` and `results/cae_protocol.json` aside, pilot again, freeze again, and rerun the sweep. Old and new rows are never mixed.
* **Single writer.** Only one `final` or `report` process can run at a time. A lock file in `checkpoints/cae/` enforces this.
* **Repeated values.** A repeated `--d` or `--seed` value runs only once. Each run's status is checked again just before it trains.
* **Report scope.** `report` always summarises the whole 7 × 5 grid, so a partial invocation cannot overwrite the summary with a subset. It exits non-zero if any checkpoint fails its hash or reload check.

## Artifacts for Person 5

* **Every run.** Each run writes `latents/CAE_d{d}_s{seed}.npz` with `Z_train` of shape (54000, d) and `Z_test` of shape (10000, d), the same as PCA and NMF.
* **`latents/CAE_d32_s0.npz`.** This file also contains `X_hat` of shape (10000, 784), float32, in [0, 1], in the official test order. `X_hat` is recomputed from the same trained model instance and the same saved `Z_test` that `evaluate()` scored, with its 1000-image batching and clipping. Inference is deterministic, so this equals the array behind the CSV's `mse`. Before the file is replaced atomically, the MSE between `X_hat` and the test images is checked against the CSV value, within the CSV's 1e-6 rounding. `evaluate()` only returns the rounded MSE, so this check confirms the same rounding bucket rather than bitwise identity.
* **Not in the repo.** Latent files and checkpoints are git-ignored. Copy them from the GPU server with:

```bash
scp <user>@<gpu-server>:<repo-dir>/latents/CAE_d32_s0.npz .
scp '<user>@<gpu-server>:<repo-dir>/latents/CAE_d*_s0.npz' .   # whole seed-0 sweep
```

Person 5's `make_tsne_grid.py` was not in the repository when this was written,
so its expected keys, shapes and dtypes could not be checked against the script
itself. The file follows the requested specification.

## Fairness and limitations

* **Hardware.** CAE timings were measured on one GPU (an RTX 5000 Ada in a shared server, recorded in each run's metadata). PCA and NMF timings were measured on CPUs, on a machine the repository does not record. `fit_time_s` and `encode_time_ms_per_img` are therefore wall-clock costs on different hardware, not hardware-normalised comparisons. Ratios derived from them, such as "CAE training took 18–37 times PCA's fit time", hold only for these environments. Neither the CAE on a CPU nor PCA on a GPU was measured. Parameter and operation counts in `docs/mathematical_analysis.md` §3 are the hardware-independent comparison.
* **What fit time includes.** CAE `fit_time_s` covers the whole training procedure: moving data to the GPU, every epoch, the per-epoch validation pass that early stopping needs, and restoring the best weights. CUDA start-up is excluded by a warm-up before the first timed run. Its length depends on when early stopping fires, so it varies across seeds.
* **Encode time.** `encode_time_ms_per_img` includes copying data to and from the GPU for each 1,000-image chunk that `evaluate()` passes in. It also includes the wrapper's input validation (dtype, shape and finiteness checks), as sklearn's `transform` does for PCA and NMF. In a review measurement the wrapper path cost about 0.77 µs per image against 0.44 µs for the bare encoder.
* **Single measurements.** Timings are single measurements on a shared server, so other users' jobs can add noise.
* **Validation data.** The CAE uses the 6K validation images for early stopping. PCA and NMF use no validation data. None of the methods uses the test set for any decision.
* **n_params.** For the CAE this counts separate encoder and decoder weights. PCA counts `components_` and `mean_`, which are shared by encoding and decoding. NMF counts `components_`.
* **Seed spread.** For MSE and k-NN accuracy, the spread across seeds reflects training randomness: initialisation, batch order and the resulting early-stopping epoch. For silhouette it also includes the seed-dependent 5,000-image subsample that `evaluate()` scores. For fit and encode times it also includes system-level measurement variability. PCA, NMF and Raw were run with seed 0 only.
* **Hardware dependence.** Reproducibility is bitwise only on the same GPU model and software stack. A different GPU or library version can change the last digits.
* **Residual nondeterminism.** cuDNN runs in deterministic mode, but `torch.use_deterministic_algorithms` and `CUBLAS_WORKSPACE_CONFIG` are not set. Severe GPU memory pressure from other users could in principle change a kernel choice. All repeated runs so far have been bitwise identical.
* **Batch-size sensitivity.** GPU encodings differ in the last bits between batch sizes, for example 1,000 versus 4,096 images. All reload checks therefore use `evaluate()`'s own 1000-image batching.

## Official sweep: status and verification

All 35 official runs (7 d × 5 seeds) completed under frozen protocol
`271764b028689114`, on an NVIDIA RTX 5000 Ada (nvidia-smi GPU 0) with torch
2.5.1+cu124 and cuDNN 9.1. Every run passed these checks:

* **Early stopping.** It fired in every run, between epochs 102 and 334. No run reached the 400-epoch cap.
* **Reload.** Each saved checkpoint re-encodes the test set bitwise identically to `evaluate()`'s `Z_test` (relative difference 0.0), and every checkpoint matches its recorded SHA-256 (`results/cae_run_diagnostics.csv`).
* **Reproducibility.** The official d = 2, 32 and 128 seed-0 runs reproduce their validation-only pilots bitwise, epoch by epoch. This confirms that seeding through the shared `evaluate()` matches the pilot.
* **Shared results.** `scripts/merge_results.py` added exactly the 35 CAE rows to `results/results.csv`. The PCA, NMF and Raw rows are byte-identical to before.
* **Provenance.** Before publication, the server hostname, the GPU UUIDs and one absolute file path were replaced in the committed metadata with `"redacted"` or a repository-relative path. No setting, timestamp or measured value changed. The unredacted metadata is kept on the GPU server.
* **Code identity.** The local development history was then squashed into a single commit, so the `git_commit` values recorded in the metadata and in `results/cae_protocol.json` (`538ef1c…`) do not resolve in this repository. The experiment code is identified instead by the `source_sha256` values recorded in every official run's metadata and in the three frozen-protocol pilots. All 38 records are identical and match the files in this commit:

  | File | SHA-256 recorded at run time |
  |---|---|
  | `models/cae.py` | `f8622d35260b3c59d420197cd5d9cb2f3dd179865a42c20cbbcad2f6de9bb4bc` |
  | `scripts/run_person4.py` | `73e10a1aa1130a0e3161e7994003e9c7d81882b520fab701225c0ec384f627f6` |
  | `evaluate.py` | `e11741ec7713168cdc9120674f0997c2733f64c48fcc744f7e4cfc8062eba38f` |
  | `data.py` | `900ed3d36b6acb19626e4f6f47864a838b0a5528985c2deaa4483433653631d2` |
  | `config.py` | `adb5ff25cdeb7ea4ea7538344c9a57ab2754160369a2e0c4bcc5e0fc69e6ca0e` |

  To check them from the repository root:

  ```bash
  sha256sum models/cae.py scripts/run_person4.py evaluate.py data.py config.py
  ```

Each value below is the mean ± sample standard deviation (n − 1) over the five seeds,
all scored on the same test set. For MSE and k-NN accuracy the spread reflects training
randomness. For silhouette it also includes the seed-dependent test subsample. For fit
time it also includes system-level measurement variability. None of it reflects
test-set sampling. The full tables and the paired comparison with PCA (exact McNemar,
Holm-adjusted) are in `docs/mathematical_analysis.md` §5.

| d | test MSE | k-NN acc | silhouette | fit time s | best epoch |
|---|---|---|---|---|---|
| 2 | 0.03103 ± 0.00041 | 0.6517 ± 0.0117 | 0.0685 ± 0.0126 | 40.7 ± 10.1 | 161 ± 42 |
| 4 | 0.02105 ± 0.00006 | 0.7658 ± 0.0036 | 0.0912 ± 0.0039 | 32.0 ± 5.2 | 126 ± 23 |
| 8 | 0.01467 ± 0.00013 | 0.8370 ± 0.0010 | 0.1032 ± 0.0074 | 57.7 ± 6.6 | 234 ± 31 |
| 16 | 0.01049 ± 0.00010 | 0.8632 ± 0.0008 | 0.0890 ± 0.0040 | 54.6 ± 14.0 | 224 ± 62 |
| 32 | 0.00751 ± 0.00014 | 0.8741 ± 0.0005 | 0.0782 ± 0.0035 | 49.9 ± 6.8 | 202 ± 28 |
| 64 | 0.00528 ± 0.00006 | 0.8757 ± 0.0012 | 0.0646 ± 0.0031 | 59.0 ± 15.0 | 242 ± 60 |
| 128 | 0.00339 ± 0.00006 | 0.8695 ± 0.0002 | 0.0550 ± 0.0025 | 46.1 ± 7.4 | 189 ± 31 |

## Person 5 artifact (ready)

| Item | Value |
|---|---|
| Location | `latents/CAE_d32_s0.npz` in the repository clone on the GPU server (git-ignored; ask Person 4 for the server address and path) |
| `Z_train` | (54000, 32) float32 |
| `Z_test` | (10000, 32) float32 |
| `X_hat` | (10000, 784) float32, values 0.0000 to 0.9978, official test order |
| MSE(X_test, X_hat) | 0.0073476, against CSV `mse` 0.007348 (difference 4.0e-7, within the 6-decimal rounding) |
| Size | 37,366,138 bytes (35.6 MiB, compressed `.npz`) |
| SHA-256 | `ca4a5c761b6017e053e3f04c983fa3e282413da30ef0bfe67412470b4223ec99` |

The order check gives a mean per-image MSE of 0.0074 when aligned with the test set
and 0.166 when shifted by one image. `tests/test_person4.py::test_person5_artifact_on_disk`
re-verifies the file whenever it is present. `X_hat` rows reshape to 28×28 images
with `X_hat[i].reshape(28, 28)`.

To transfer and verify the file:

```bash
scp <user>@<gpu-server>:<repo-dir>/latents/CAE_d32_s0.npz .
sha256sum CAE_d32_s0.npz          # macOS: shasum -a 256 CAE_d32_s0.npz
# expected: ca4a5c761b6017e053e3f04c983fa3e282413da30ef0bfe67412470b4223ec99
python -c "import numpy as np; f = np.load('CAE_d32_s0.npz'); print({k: f[k].shape for k in f.files})"
# expected: {'Z_train': (54000, 32), 'Z_test': (10000, 32), 'X_hat': (10000, 784)}
```
