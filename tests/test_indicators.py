import numpy as np
import pandas as pd

from quantlab.indicators import (
    bollinger_pct_b,
    ema,
    indicator_frame,
    momentum,
    price_to_sma,
    sma,
)


def _price_series(n=100, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=dates)


def test_sma_matches_rolling_mean():
    prices = _price_series()
    expected = prices.rolling(10).mean()
    pd.testing.assert_series_equal(sma(prices, 10), expected)


def test_ema_converges_to_constant():
    prices = pd.Series([50.0] * 60, index=pd.bdate_range("2020-01-01", periods=60))
    assert np.isclose(ema(prices, span=10).iloc[-1], 50.0)


def test_momentum_on_linear_growth():
    prices = pd.Series(
        np.arange(1, 51, dtype=float), index=pd.bdate_range("2020-01-01", periods=50)
    )
    mom = momentum(prices, window=10)
    # price_t / price_{t-10} - 1 for t=10 (11 / 1 - 1 = 10)
    assert np.isclose(mom.iloc[10], 10.0)


def test_bollinger_pct_b_bounds_for_normal_moves():
    prices = _price_series(300)
    pct_b = bollinger_pct_b(prices).dropna()
    # With 2-sigma bands the vast majority of observations lie inside [0, 1].
    inside = ((pct_b >= 0) & (pct_b <= 1)).mean()
    assert inside > 0.85


def test_price_to_sma_above_one_in_uptrend():
    prices = pd.Series(
        np.linspace(100, 200, 60), index=pd.bdate_range("2020-01-01", periods=60)
    )
    ratio = price_to_sma(prices, 20).dropna()
    assert (ratio > 1).all()


def test_indicator_frame_columns_and_warmup():
    prices = _price_series(80)
    frame = indicator_frame(prices)
    assert list(frame.columns) == ["price_to_sma", "bb_pct_b", "momentum", "volatility"]
    assert frame.iloc[:9].isna().any(axis=1).all()  # warm-up rows have NaNs
    assert not frame.iloc[25:].isna().any().any()
