"""Theoretically optimal strategy: trade with perfect foresight of next close."""

from __future__ import annotations

import pandas as pd


def theoretically_optimal_trades(
    prices: pd.Series,
    max_shares: int = 1000,
) -> pd.Series:
    """Share deltas that keep holdings in ``{-max_shares, 0, +max_shares}``.

    On each day the target is long if tomorrow's close is higher, short if
    lower. Trades are the share deltas needed to move to that target.
    """
    prices = prices.astype(float)
    next_up = prices.shift(-1) > prices
    next_down = prices.shift(-1) < prices
    target = pd.Series(0.0, index=prices.index)
    target[next_up] = float(max_shares)
    target[next_down] = float(-max_shares)
    # Last day has no foresight; flatten.
    target.iloc[-1] = 0.0
    trades = target.diff()
    trades.iloc[0] = target.iloc[0]
    return trades
