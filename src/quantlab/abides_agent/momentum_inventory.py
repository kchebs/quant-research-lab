"""Momentum + inventory trading logic used by the ABIDES agent.

Pure functions so the strategy can be tested without the discrete-event kernel.
The ABIDES agent in ``third_party/abides/contributed_traders/momentum_inventory_agent``
applies the same rules against live order-book snapshots.
"""

from __future__ import annotations

import numpy as np


def ema_numpy(series: list[float] | np.ndarray, span: int) -> float | None:
    arr = np.asarray(series, dtype=float)
    if len(arr) <= span:
        return None
    alpha = 2.0 / (span + 1.0)
    value = arr[0]
    for x in arr[1:]:
        value = alpha * x + (1 - alpha) * value
    return float(value)


def momentum_signal(mid_prices: list[float], fast_span: int, slow_span: int) -> str:
    """Return ``buy``, ``sell``, or ``flat`` from fast vs slow mid-price EMAs."""
    fast = ema_numpy(mid_prices, fast_span)
    slow = ema_numpy(mid_prices, slow_span)
    if fast is None or slow is None:
        return "flat"
    if fast > slow:
        return "buy"
    if fast < slow:
        return "sell"
    return "flat"


def inventory_skewed_prices(
    mid: float,
    spread_std: float,
    cash: float,
    shares: float,
    ask: float,
    bid: float,
) -> tuple[int, int, int, int]:
    """Return ``(buy_qty, buy_px, sell_qty, sell_px)`` with inventory skew.

    Long inventory lowers asks; short/cash-heavy inventory raises bids.
    """
    vol_buy = int(max(0, cash // max(mid, 1e-9)))
    vol_sell = int(max(0, abs(shares))) if shares > 0 else int(max(0, cash // max(mid, 1e-9)))
    m_price = mid
    if vol_buy + vol_sell > 0:
        weight = vol_buy / (vol_buy + vol_sell)
        m_price = weight * spread_std / 7.0 - spread_std / 14.0 + mid
    buy_px = int(np.floor(min(mid - spread_std / 1.5, bid + 1)))
    sell_px = int(np.ceil(max(mid + spread_std / 1.5, ask - 1)))
    # Prefer inventory reduction: if long, emphasize sells; if flat/cash, buys.
    if shares > 0:
        vol_buy = 0
    return vol_buy, buy_px, vol_sell, sell_px
