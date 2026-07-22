import pandas as pd

from quantlab.backtest import run_backtest
from quantlab.strategies import theoretically_optimal_trades


def test_tos_beats_buy_and_hold_on_trending_series():
    prices = pd.Series(
        [10, 11, 10, 12, 11, 13, 12, 14],
        index=pd.date_range("2020-01-01", periods=8, freq="B"),
    )
    trades = theoretically_optimal_trades(prices, max_shares=10)
    tos = run_backtest(trades, prices, start_cash=10_000, commission=0, impact=0)
    buy = pd.Series(0.0, index=prices.index)
    buy.iloc[0] = 10
    bh = run_backtest(buy, prices, start_cash=10_000, commission=0, impact=0)
    assert tos["value"].iloc[-1] > bh["value"].iloc[-1]
    assert trades.abs().max() <= 20
