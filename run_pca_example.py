"""Example / smoke test: run PCA and the raw-pixel baseline through evaluate().
Person 2 can copy this pattern; Person 3 swaps in NMF; Person 4 swaps in CAEReducer."""
from sklearn.decomposition import PCA

import config
from evaluate import IdentityReducer, evaluate, load_all_results

if __name__ == "__main__":
    evaluate(IdentityReducer(), "Raw", d=784, seed=0, save_latents=False)
    for d in config.D_GRID:
        evaluate(PCA(n_components=d, svd_solver="full"), "PCA", d=d, seed=0)
    print(load_all_results())
