import numpy as np
import pandas as pd

from quantlab.portfolio import max_sharpe_weights, portfolio_values


def _prices():
    dates = pd.bdate_range("2020-01-01", periods=250)
    rng = np.random.default_rng(0)
    winner = 100 * np.exp(np.cumsum(rng.normal(0.002, 0.005, 250)))
    loser = 100 * np.exp(np.cumsum(rng.normal(-0.001, 0.02, 250)))
    return pd.DataFrame({"WIN": winner, "LOSE": loser}, index=dates)


def test_weights_sum_to_one_and_bounded():
    weights = max_sharpe_weights(_prices())
    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= -1e-9).all() and (weights <= 1 + 1e-9).all()


def test_optimizer_prefers_dominant_asset():
    weights = max_sharpe_weights(_prices())
    assert weights[0] > 0.9  # nearly everything in the high-Sharpe asset


def test_portfolio_values_start_at_start_value():
    prices = _prices()
    values = portfolio_values(prices, [0.5, 0.5], start_value=10_000)
    assert np.isclose(values.iloc[0], 10_000)
