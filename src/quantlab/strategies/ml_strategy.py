"""ML-driven trading strategy: bagged random trees on indicator features.

Training: features are the standard indicator frame; the label is the
forward ``lookahead``-day return. The ensemble regresses future return, and
positions are taken when the predicted return clears a trade threshold
(set above expected round-trip transaction costs so the model only trades
when its edge should survive costs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.indicators import indicator_frame
from quantlab.learners import BaggedTrees


class MLTradingStrategy:
    def __init__(
        self,
        n_estimators: int = 25,
        leaf_size: int = 5,
        lookahead: int = 5,
        trade_threshold: float = 0.01,
        seed: int | None = 42,
    ):
        self.n_estimators = n_estimators
        self.leaf_size = leaf_size
        self.lookahead = lookahead
        self.trade_threshold = trade_threshold
        self.seed = seed
        self._model: BaggedTrees | None = None

    def _features(self, prices: pd.Series) -> pd.DataFrame:
        return indicator_frame(prices)

    def fit(self, prices: pd.Series) -> MLTradingStrategy:
        features = self._features(prices)
        future_return = prices.shift(-self.lookahead) / prices - 1
        data = features.assign(target=future_return).dropna()
        self._model = BaggedTrees(
            n_estimators=self.n_estimators,
            leaf_size=self.leaf_size,
            seed=self.seed,
        )
        self._model.fit(data.drop(columns="target").values, data["target"].values)
        return self

    def predicted_returns(self, prices: pd.Series) -> pd.Series:
        if self._model is None:
            raise RuntimeError("Strategy has not been fitted")
        features = self._features(prices).dropna()
        preds = self._model.predict(features.values)
        return pd.Series(preds, index=features.index)

    def positions(self, prices: pd.Series) -> pd.Series:
        """Target position in {-1, 0, +1} for each date in `prices`."""
        preds = self.predicted_returns(prices)
        positions = pd.Series(np.nan, index=prices.index)
        positions[preds.index[preds > self.trade_threshold]] = 1
        positions[preds.index[preds < -self.trade_threshold]] = -1
        # Hold the previous position between signals; flat before the first one.
        return positions.ffill().fillna(0)
