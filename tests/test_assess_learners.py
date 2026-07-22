import numpy as np

from quantlab.experiments import (
    best_for_decision_tree,
    best_for_linear_regression,
    leaf_size_rmse_curve,
    rmse,
    train_test_split_shuffle,
)
from quantlab.learners import (
    InsaneLearner,
    LinearRegressionLearner,
    RegressionTree,
)


def test_linreg_fits_line():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 1.5
    model = LinearRegressionLearner().fit(X, y)
    preds = model.predict(X)
    assert rmse(y, preds) < 1e-8


def test_insane_learner_predicts():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 3))
    y = X.sum(axis=1)
    model = InsaneLearner(n_outer=3, n_inner=3, seed=0).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (80,)
    assert np.corrcoef(preds, y)[0, 1] > 0.9


def test_best4_linreg_beats_tree():
    X, y = best_for_linear_regression(seed=5)
    Xtr, Xte, ytr, yte = train_test_split_shuffle(X, y, seed=0)
    lr = LinearRegressionLearner().fit(Xtr, ytr)
    dt = RegressionTree(leaf_size=1).fit(Xtr, ytr)
    assert rmse(yte, lr.predict(Xte)) < rmse(yte, dt.predict(Xte))


def test_best4_tree_beats_linreg():
    X, y = best_for_decision_tree(seed=5)
    Xtr, Xte, ytr, yte = train_test_split_shuffle(X, y, seed=1)
    lr = LinearRegressionLearner().fit(Xtr, ytr)
    dt = RegressionTree(leaf_size=1).fit(Xtr, ytr)
    assert rmse(yte, dt.predict(Xte)) < rmse(yte, lr.predict(Xte))


def test_leaf_size_curve_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    y = X[:, 0] + 0.1 * rng.normal(size=120)
    Xtr, Xte, ytr, yte = train_test_split_shuffle(X, y, seed=2)
    curve = leaf_size_rmse_curve(Xtr, ytr, Xte, yte, leaf_sizes=[1, 10, 50])
    assert len(curve["leaf_size"]) == 3
    assert curve["in_rmse"][0] <= curve["in_rmse"][-1] + 1e-9
