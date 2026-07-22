"""From-scratch regression trees (numpy only).

Two split-selection modes:

- deterministic: split on the feature with the highest absolute Pearson
  correlation with the target (a cheap proxy for information gain);
- random: split on a randomly chosen feature, the classic building block for
  fast bagged ensembles (random trees are weak alone, strong in a bag).

Splits are at the median of the chosen feature. Leaves are created when the
node has ``leaf_size`` or fewer samples, when the target is constant, or when
no feature can split the data.
"""

from __future__ import annotations

import numpy as np


class RegressionTree:
    def __init__(
        self,
        leaf_size: int = 5,
        random_split: bool = False,
        rng: np.random.Generator | None = None,
    ):
        self.leaf_size = leaf_size
        self.random_split = random_split
        self.rng = rng if rng is not None else np.random.default_rng()
        self._tree: np.ndarray | None = None  # rows: [feature, split, left, right]

    _LEAF = -1

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegressionTree":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._tree = self._build(X, y)
        return self

    def _make_leaf(self, y: np.ndarray) -> np.ndarray:
        return np.array([[self._LEAF, float(np.mean(y)), np.nan, np.nan]])

    def _choose_feature(self, X: np.ndarray, y: np.ndarray) -> int:
        n_features = X.shape[1]
        if self.random_split:
            return int(self.rng.integers(n_features))
        correlations = np.zeros(n_features)
        for j in range(n_features):
            col = X[:, j]
            if np.std(col) == 0 or np.std(y) == 0:
                continue
            correlations[j] = abs(np.corrcoef(col, y)[0, 1])
        return int(np.nanargmax(correlations))

    def _build(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if len(y) <= self.leaf_size or np.all(y == y[0]):
            return self._make_leaf(y)

        feature = self._choose_feature(X, y)
        split_value = float(np.median(X[:, feature]))
        left_mask = X[:, feature] <= split_value

        # A degenerate split (all rows on one side) cannot partition the node.
        # Try the remaining features before giving up and making a leaf.
        if left_mask.all() or not left_mask.any():
            for feature in range(X.shape[1]):
                split_value = float(np.median(X[:, feature]))
                left_mask = X[:, feature] <= split_value
                if not (left_mask.all() or not left_mask.any()):
                    break
            else:
                return self._make_leaf(y)

        left = self._build(X[left_mask], y[left_mask])
        right = self._build(X[~left_mask], y[~left_mask])
        root = np.array([[feature, split_value, 1, len(left) + 1]])
        return np.vstack([root, left, right])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._tree is None:
            raise RuntimeError("Tree has not been fitted")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.array([self._predict_row(row) for row in X])

    def _predict_row(self, row: np.ndarray) -> float:
        node = 0
        while True:
            feature, value, left, right = self._tree[node]
            if feature == self._LEAF:
                return value
            node += int(left) if row[int(feature)] <= value else int(right)
