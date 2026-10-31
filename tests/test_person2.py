import numpy as np

from evaluate import IdentityReducer
from scripts.select_knn import select_best_k


def test_identity_reducer_is_a_zero_error_baseline():
    X = np.arange(20, dtype=np.float32).reshape(5, 4)
    reducer = IdentityReducer()

    assert reducer.fit(X) is reducer
    np.testing.assert_array_equal(reducer.transform(X), X)
    np.testing.assert_array_equal(reducer.inverse_transform(X), X)
    assert reducer.n_params == 0


def test_select_best_k_uses_only_the_supplied_validation_arrays():
    rng = np.random.default_rng(0)
    Z_train = rng.normal(size=(40, 3))
    y_train = np.repeat(np.arange(4), 10)
    Z_val = rng.normal(size=(12, 3))
    y_val = np.repeat(np.arange(4), 3)

    best_k, rows = select_best_k(
        Z_train,
        y_train,
        Z_val,
        y_val,
        candidates=(1, 3, 5),
        n_jobs=1,
    )

    assert best_k in {1, 3, 5}
    assert [row["k"] for row in rows] == [1, 3, 5]
    assert all(0.0 <= row["validation_accuracy"] <= 1.0 for row in rows)
