"""Technical indicators, implemented from scratch on pandas Series.

All functions accept a price Series (or DataFrame of one column per symbol)
and return an object of the same shape. Windows are in trading days.
"""

from __future__ import annotations

import pandas as pd


def sma(prices: pd.Series | pd.DataFrame, window: int = 20):
    """Simple moving average."""
    return prices.rolling(window).mean()


def ema(prices: pd.Series | pd.DataFrame, span: int = 20):
    """Exponential moving average."""
    return prices.ewm(span=span, adjust=False).mean()


def price_to_sma(prices: pd.Series | pd.DataFrame, window: int = 20):
    """Price divided by its SMA. >1 means price is above trend."""
    return prices / sma(prices, window)


def bollinger_bands(
    prices: pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Middle/upper/lower Bollinger bands for a single price series."""
    middle = sma(prices, window)
    std = prices.rolling(window).std()
    return pd.DataFrame(
        {
            "middle": middle,
            "upper": middle + num_std * std,
            "lower": middle - num_std * std,
        }
    )


def bollinger_pct_b(
    prices: pd.Series | pd.DataFrame, window: int = 20, num_std: float = 2.0
):
    """%B: position of price within the Bollinger bands.

    0 = at lower band, 1 = at upper band; values outside [0, 1] mean the
    price has broken out of the bands.
    """
    middle = sma(prices, window)
    std = prices.rolling(window).std()
    lower = middle - num_std * std
    upper = middle + num_std * std
    return (prices - lower) / (upper - lower)


def momentum(prices: pd.Series | pd.DataFrame, window: int = 10):
    """Rate of change over `window` days: price_t / price_{t-window} - 1."""
    return prices / prices.shift(window) - 1


def rolling_volatility(prices: pd.Series | pd.DataFrame, window: int = 20):
    """Rolling standard deviation of daily returns."""
    return prices.pct_change().rolling(window).std()


def indicator_frame(
    prices: pd.Series,
    sma_window: int = 20,
    bb_window: int = 20,
    momentum_window: int = 10,
    vol_window: int = 20,
) -> pd.DataFrame:
    """Standard feature matrix for one symbol, used by the ML and RL strategies."""
    return pd.DataFrame(
        {
            "price_to_sma": price_to_sma(prices, sma_window),
            "bb_pct_b": bollinger_pct_b(prices, bb_window),
            "momentum": momentum(prices, momentum_window),
            "volatility": rolling_volatility(prices, vol_window),
        }
    )
