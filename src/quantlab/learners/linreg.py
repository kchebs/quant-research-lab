"""Ordinary least-squares linear regression (numpy only)."""

from __future__ import annotations

import numpy as np


class LinearRegressionLearner:
    """Fit ``y ~ X @ w + b`` via least squares."""

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionLearner":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        design = np.column_stack([X, np.ones(len(X))])
        params, *_ = np.linalg.lstsq(design, y, rcond=None)
        self.coef_ = params[:-1]
        self.intercept_ = float(params[-1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Learner has not been fitted")
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_
