"""Performance metrics for portfolio value series."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def cumulative_return(portvals: pd.Series) -> float:
    return portvals.iloc[-1] / portvals.iloc[0] - 1


def average_daily_return(portvals: pd.Series) -> float:
    return portvals.pct_change().iloc[1:].mean()


def daily_volatility(portvals: pd.Series) -> float:
    return portvals.pct_change().iloc[1:].std()


def sharpe_ratio(portvals: pd.Series, risk_free_daily: float = 0.0) -> float:
    """Annualized Sharpe ratio from a daily portfolio value series."""
    daily = portvals.pct_change().iloc[1:] - risk_free_daily
    if daily.std() == 0:
        return np.nan
    return np.sqrt(TRADING_DAYS_PER_YEAR) * daily.mean() / daily.std()


def max_drawdown(portvals: pd.Series) -> float:
    """Largest peak-to-trough decline, returned as a negative fraction."""
    running_max = portvals.cummax()
    drawdown = portvals / running_max - 1
    return drawdown.min()


def summarize(portvals: pd.Series, name: str = "strategy") -> pd.Series:
    """One-row metric summary, convenient for side-by-side scorecards."""
    return pd.Series(
        {
            "cumulative_return": cumulative_return(portvals),
            "avg_daily_return": average_daily_return(portvals),
            "daily_volatility": daily_volatility(portvals),
            "sharpe_ratio": sharpe_ratio(portvals),
            "max_drawdown": max_drawdown(portvals),
            "final_value": portvals.iloc[-1],
        },
        name=name,
    )
