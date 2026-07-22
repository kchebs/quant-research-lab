"""Sharpe-maximizing long-only portfolio allocation via SLSQP."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quantlab.backtest.metrics import sharpe_ratio


def portfolio_values(
    prices: pd.DataFrame, weights: np.ndarray | list[float], start_value: float = 1.0
) -> pd.Series:
    """Daily value of a buy-and-hold portfolio with the given weights."""
    normalized = prices / prices.iloc[0]
    return (normalized * np.asarray(weights)).sum(axis=1) * start_value


def max_sharpe_weights(prices: pd.DataFrame) -> np.ndarray:
    """Long-only weights (sum to 1) that maximize the annualized Sharpe ratio."""
    n = prices.shape[1]

    def neg_sharpe(weights: np.ndarray) -> float:
        return -sharpe_ratio(portfolio_values(prices, weights))

    result = minimize(
        neg_sharpe,
        x0=np.full(n, 1 / n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return result.x
