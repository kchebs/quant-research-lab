"""Bootstrap-aggregated ensemble of regression trees (from scratch)."""

from __future__ import annotations

import numpy as np

from quantlab.learners.trees import RegressionTree


class BaggedTrees:
    """Bag of random-split regression trees.

    Each tree is trained on a bootstrap sample (drawn with replacement, same
    size as the training set); predictions are the ensemble mean. Random
    split selection plus bagging trades a little bias for a large variance
    reduction, which is what makes this usable on noisy financial targets.
    """

    def __init__(
        self,
        n_estimators: int = 20,
        leaf_size: int = 5,
        random_split: bool = True,
        seed: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.leaf_size = leaf_size
        self.random_split = random_split
        self.rng = np.random.default_rng(seed)
        self._trees: list[RegressionTree] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaggedTrees:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)
        self._trees = []
        for _ in range(self.n_estimators):
            idx = self.rng.integers(0, n, size=n)
            tree = RegressionTree(
                leaf_size=self.leaf_size,
                random_split=self.random_split,
                rng=self.rng,
            )
            tree.fit(X[idx], y[idx])
            self._trees.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._trees:
            raise RuntimeError("Ensemble has not been fitted")
        return np.mean([tree.predict(X) for tree in self._trees], axis=0)
