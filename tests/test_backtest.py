import numpy as np
import pandas as pd

from quantlab.backtest import affordable_shares, positions_to_trades, run_backtest
from quantlab.backtest.engine import buy_and_hold_benchmark


def _prices():
    dates = pd.bdate_range("2021-01-01", periods=5)
    return pd.Series([100.0, 110.0, 120.0, 90.0, 100.0], index=dates)


def test_no_trades_keeps_cash_flat():
    prices = _prices()
    trades = pd.Series(0.0, index=prices.index)
    result = run_backtest(trades, prices, start_cash=50_000)
    assert (result["value"] == 50_000).all()


def test_single_buy_marks_to_market():
    prices = _prices()
    trades = pd.Series(0.0, index=prices.index)
    trades.iloc[0] = 100  # buy 100 shares on day one
    result = run_backtest(trades, prices, start_cash=100_000, commission=0, impact=0)
    assert np.isclose(result["value"].iloc[0], 100_000)  # buy at close, no costs
    assert np.isclose(result["value"].iloc[2], 100_000 + 100 * 20)  # +$20/share


def test_costs_reduce_value():
    prices = _prices()
    trades = pd.Series(0.0, index=prices.index)
    trades.iloc[0] = 100
    free = run_backtest(trades, prices, commission=0, impact=0)
    costly = run_backtest(trades, prices, commission=9.95, impact=0.005)
    expected_cost = 9.95 + 100 * 100.0 * 0.005
    assert np.isclose(free["value"].iloc[-1] - costly["value"].iloc[-1], expected_cost)


def test_short_position_profits_from_decline():
    prices = _prices()
    trades = pd.Series(0.0, index=prices.index)
    trades.iloc[1] = -100  # short 100 at 110
    result = run_backtest(trades, prices, commission=0, impact=0)
    # Day 3: price fell 110 -> 90, short gains 100 * 20 = 2000
    assert np.isclose(result["value"].iloc[3], 100_000 + 2000)


def test_positions_to_trades_targets():
    dates = pd.bdate_range("2021-01-01", periods=4)
    positions = pd.Series([1, 1, -1, 0], index=dates, dtype=float)
    trades = positions_to_trades(positions, max_shares=1000)
    assert trades.tolist() == [1000.0, 0.0, -2000.0, 1000.0]


def test_affordable_shares_is_unlevered():
    prices = _prices()  # first price 100
    shares = affordable_shares(prices, 100_000)
    assert shares == 1000
    assert shares * prices.iloc[0] <= 100_000


def test_buy_and_hold_benchmark():
    prices = _prices()
    result = buy_and_hold_benchmark(prices, shares=100, commission=0, impact=0)
    assert np.isclose(result["value"].iloc[-1], 100_000)  # price round-trips to 100
    assert (result["shares"] == 100).all()
