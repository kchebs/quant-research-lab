"""InsaneLearner: bag of bags of linear regressors."""

from __future__ import annotations

import numpy as np

from quantlab.learners.linreg import LinearRegressionLearner


class InsaneLearner:
    """20 outer bags, each averaging 20 bootstrap linear regressions."""

    def __init__(
        self,
        n_outer: int = 20,
        n_inner: int = 20,
        seed: int | None = None,
    ):
        self.n_outer = n_outer
        self.n_inner = n_inner
        self.rng = np.random.default_rng(seed)
        self._bags: list[list[LinearRegressionLearner]] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "InsaneLearner":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)
        self._bags = []
        for _ in range(self.n_outer):
            inner: list[LinearRegressionLearner] = []
            for _ in range(self.n_inner):
                idx = self.rng.integers(0, n, size=n)
                learner = LinearRegressionLearner().fit(X[idx], y[idx])
                inner.append(learner)
            self._bags.append(inner)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._bags:
            raise RuntimeError("Learner has not been fitted")
        outer_preds = []
        for inner in self._bags:
            preds = np.mean([lr.predict(X) for lr in inner], axis=0)
            outer_preds.append(preds)
        return np.mean(outer_preds, axis=0)
