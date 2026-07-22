"""Learner assessment helpers: RMSE curves by leaf size, bagging, DT vs RT."""

from __future__ import annotations

import time

import numpy as np

from quantlab.learners import BaggedTrees, RegressionTree


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def train_test_split_shuffle(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.6,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    cut = int(n * train_frac)
    train_idx, test_idx = idx[:cut], idx[cut:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def leaf_size_rmse_curve(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    leaf_sizes: range | list[int] | None = None,
    *,
    random_split: bool = False,
    bag: bool = False,
    n_estimators: int = 20,
    seed: int | None = 0,
) -> dict[str, list[float]]:
    """In-sample and out-of-sample RMSE vs leaf size."""
    if leaf_sizes is None:
        leaf_sizes = range(1, 101)
    sizes: list[int] = []
    in_rmses: list[float] = []
    out_rmses: list[float] = []
    for leaf in leaf_sizes:
        if bag:
            model = BaggedTrees(
                n_estimators=n_estimators,
                leaf_size=leaf,
                random_split=random_split,
                seed=seed,
            ).fit(X_train, y_train)
        else:
            model = RegressionTree(
                leaf_size=leaf,
                random_split=random_split,
                rng=np.random.default_rng(seed),
            ).fit(X_train, y_train)
        sizes.append(int(leaf))
        in_rmses.append(rmse(y_train, model.predict(X_train)))
        out_rmses.append(rmse(y_test, model.predict(X_test)))
    return {"leaf_size": sizes, "in_rmse": in_rmses, "out_rmse": out_rmses}


def compare_dt_rt_timing(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    leaf_sizes: range | list[int] | None = None,
    seed: int = 0,
) -> dict[str, list[float]]:
    """MAE and wall time for correlation-split DT vs random-split RT."""
    if leaf_sizes is None:
        leaf_sizes = range(1, 101, 5)
    out: dict[str, list[float]] = {
        "leaf_size": [],
        "dt_mae": [],
        "rt_mae": [],
        "dt_seconds": [],
        "rt_seconds": [],
    }
    for leaf in leaf_sizes:
        t0 = time.perf_counter()
        dt = RegressionTree(leaf_size=leaf, random_split=False).fit(X_train, y_train)
        dt_pred = dt.predict(X_test)
        dt_secs = time.perf_counter() - t0

        t0 = time.perf_counter()
        rt = RegressionTree(
            leaf_size=leaf, random_split=True, rng=np.random.default_rng(seed)
        ).fit(X_train, y_train)
        rt_pred = rt.predict(X_test)
        rt_secs = time.perf_counter() - t0

        out["leaf_size"].append(float(leaf))
        out["dt_mae"].append(mae(y_test, dt_pred))
        out["rt_mae"].append(mae(y_test, rt_pred))
        out["dt_seconds"].append(dt_secs)
        out["rt_seconds"].append(rt_secs)
    return out
