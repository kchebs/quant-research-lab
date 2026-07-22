"""Manual (human-designed) rule-based trading strategy.

Votes three technical indicators and takes a position when at least two
agree. The thresholds are deliberately simple and fixed in advance — this is
the interpretable baseline the ML and RL strategies must beat.
"""

from __future__ import annotations

import pandas as pd

from quantlab.indicators import indicator_frame


class ManualRuleStrategy:
    """Long/short/flat signal from Bollinger %B, momentum, and price/SMA.

    Buy votes  : %B < bb_low (oversold), momentum > +mom_threshold,
                 price/SMA < sma_low (below trend, mean-reversion entry).
    Sell votes : %B > bb_high (overbought), momentum < -mom_threshold,
                 price/SMA > sma_high (extended above trend).
    A position is taken when votes >= min_votes, else flat.
    """

    def __init__(
        self,
        bb_low: float = 0.2,
        bb_high: float = 0.8,
        mom_threshold: float = 0.03,
        sma_low: float = 0.97,
        sma_high: float = 1.03,
        min_votes: int = 2,
        sma_window: int = 20,
        bb_window: int = 20,
        momentum_window: int = 10,
    ):
        self.bb_low = bb_low
        self.bb_high = bb_high
        self.mom_threshold = mom_threshold
        self.sma_low = sma_low
        self.sma_high = sma_high
        self.min_votes = min_votes
        self.sma_window = sma_window
        self.bb_window = bb_window
        self.momentum_window = momentum_window

    def positions(self, prices: pd.Series) -> pd.Series:
        """Target position in {-1, 0, +1} for each date."""
        features = indicator_frame(
            prices,
            sma_window=self.sma_window,
            bb_window=self.bb_window,
            momentum_window=self.momentum_window,
        )
        buy_votes = (
            (features["bb_pct_b"] < self.bb_low).astype(int)
            + (features["momentum"] > self.mom_threshold).astype(int)
            + (features["price_to_sma"] < self.sma_low).astype(int)
        )
        sell_votes = (
            (features["bb_pct_b"] > self.bb_high).astype(int)
            + (features["momentum"] < -self.mom_threshold).astype(int)
            + (features["price_to_sma"] > self.sma_high).astype(int)
        )
        positions = pd.Series(0, index=prices.index, dtype=float)
        positions[buy_votes >= self.min_votes] = 1
        positions[sell_votes >= self.min_votes] = -1
        # Hold the previous position when no signal fires (0 = no new signal).
        positions = positions.replace(0, pd.NA).ffill().fillna(0).astype(float)
        # Indicators are undefined during the warm-up window: stay flat.
        warmup = features.isna().any(axis=1)
        positions[warmup] = 0
        return positions
