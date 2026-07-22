import numpy as np
import pandas as pd

from quantlab.backtest.metrics import (
    cumulative_return,
    max_drawdown,
    sharpe_ratio,
    summarize,
)


def test_cumulative_return():
    portvals = pd.Series([100.0, 110.0, 121.0])
    assert np.isclose(cumulative_return(portvals), 0.21)


def test_max_drawdown():
    portvals = pd.Series([100.0, 150.0, 75.0, 120.0])
    assert np.isclose(max_drawdown(portvals), -0.5)  # 150 -> 75


def test_sharpe_positive_for_steady_gains():
    portvals = pd.Series(np.linspace(100, 150, 100) + np.random.default_rng(0).normal(0, 0.1, 100))
    assert sharpe_ratio(portvals) > 1


def test_summarize_keys():
    portvals = pd.Series([100.0, 101.0, 102.0, 101.5])
    summary = summarize(portvals, name="test")
    assert summary.name == "test"
    for key in [
        "cumulative_return",
        "avg_daily_return",
        "daily_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "final_value",
    ]:
        assert key in summary.index
