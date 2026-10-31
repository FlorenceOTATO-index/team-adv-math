"""
Shared settings for the DATA 220 Group 5 project (PCA vs NMF vs CAE on Fashion-MNIST).
Everyone imports from here so all runs use the same split, seeds, d grid and metrics.
Change a value here only after telling the group.
"""
import os

# ---- Reproducibility ----
SPLIT_SEED = 42                 # seed for the train/val split (never change after first run)
SEEDS = [0, 1, 2, 3, 4]         # experiment seeds (CAE runs all; PCA/NMF are deterministic)

# ---- Data split ----
VAL_SIZE = 6000                 # taken from the official 60K train set -> 54K train / 6K val
                                # the official 10K test set is never touched until evaluation

# ---- Experiment grid ----
D_GRID = [2, 4, 8, 16, 32, 64, 128]   # latent dimensions to sweep

# ---- Evaluation ----
KNN_K = 9                       # selected by Person 2 on PCA(d=32) validation accuracy
ENCODE_BATCH = 1000             # batch size for transform / inverse_transform
# Silhouette is quadratic in the number of points.  A fixed 5K sample keeps
# the shared evaluator practical; the test labels are still never used to
# choose model settings.
SILHOUETTE_SAMPLE = 5000

# ---- Paths (relative, so they work in Colab and on Tzu-Yang's server) ----
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")          # raw Fashion-MNIST files (downloaded automatically)
SPLIT_DIR = os.path.join(ROOT, "splits")       # saved train/val indices
LATENT_DIR = os.path.join(ROOT, "latents")     # saved Z_train / Z_test per run
RESULTS_DIR = os.path.join(ROOT, "results")    # one CSV per method, merged later
