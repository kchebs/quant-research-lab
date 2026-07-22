import numpy as np

from quantlab.learners import BaggedTrees, RegressionTree


def _dataset(n=400, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 3))
    y = 2 * X[:, 0] - X[:, 1] + rng.normal(0, 0.05, n)
    return X, y


def test_tree_fits_training_data():
    X, y = _dataset()
    tree = RegressionTree(leaf_size=1).fit(X, y)
    preds = tree.predict(X)
    rmse = np.sqrt(np.mean((preds - y) ** 2))
    assert rmse < 0.1


def test_tree_generalizes():
    X, y = _dataset()
    X_test, y_test = _dataset(seed=2)
    tree = RegressionTree(leaf_size=10).fit(X, y)
    preds = tree.predict(X_test)
    correlation = np.corrcoef(preds, y_test)[0, 1]
    assert correlation > 0.9


def test_constant_target_gives_leaf():
    X = np.random.default_rng(0).uniform(size=(50, 2))
    y = np.full(50, 7.0)
    tree = RegressionTree().fit(X, y)
    assert np.allclose(tree.predict(X), 7.0)


def test_bagging_beats_single_random_tree():
    X, y = _dataset()
    X_test, y_test = _dataset(seed=3)
    single = RegressionTree(leaf_size=5, random_split=True, rng=np.random.default_rng(0))
    single.fit(X, y)
    bag = BaggedTrees(n_estimators=30, leaf_size=5, seed=0).fit(X, y)
    rmse_single = np.sqrt(np.mean((single.predict(X_test) - y_test) ** 2))
    rmse_bag = np.sqrt(np.mean((bag.predict(X_test) - y_test) ** 2))
    assert rmse_bag < rmse_single


def test_predict_before_fit_raises():
    import pytest

    with pytest.raises(RuntimeError):
        RegressionTree().predict(np.zeros((1, 2)))
    with pytest.raises(RuntimeError):
        BaggedTrees().predict(np.zeros((1, 2)))
