"""Naive next-day price forecasts and error metrics.

These baselines (last close; close × rolling mean return) exist to validate a
forecast → MAE/RMSE path in CI. They are not a claimed trading edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def last_value_forecast(prices: pd.Series) -> pd.Series:
    """Predict next close as today's close (random-walk / persistence)."""
    return prices.copy().rename("last_value")


def mean_return_forecast(prices: pd.Series, window: int = 20) -> pd.Series:
    """Predict next close as price × (1 + rolling mean return)."""
    rets = prices.pct_change()
    mean_ret = rets.rolling(window, min_periods=max(2, window // 2)).mean()
    return (prices * (1.0 + mean_ret)).rename("mean_return")


def forecast_errors(
    prices: pd.Series,
    forecast: pd.Series,
) -> pd.Series:
    """Aligned errors: forecast_at_t vs actual close at t+1."""
    actual_next = prices.shift(-1)
    err = forecast - actual_next
    return err.dropna().rename("error")


def mae(errors: pd.Series) -> float:
    return float(np.mean(np.abs(errors.to_numpy(dtype=float))))


def rmse(errors: pd.Series) -> float:
    arr = errors.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(arr * arr)))
