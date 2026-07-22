"""Synthetic datasets that favor one learner family over another."""

from __future__ import annotations

import numpy as np


def best_for_linear_regression(seed: int = 5, n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Linear target ``y = x0 + x1`` with ``x1 = 10 * x0`` — trees struggle more."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 2))
    X[:, 0] = rng.random(n) * 200 - 100
    X[:, 1] = 10 * X[:, 0]
    y = X[:, 0] + X[:, 1]
    return X, y


def best_for_decision_tree(seed: int = 5, n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned threshold rule — linear regression is a poor fit."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (X[:, 1] <= 0).astype(float)
    return X, y
