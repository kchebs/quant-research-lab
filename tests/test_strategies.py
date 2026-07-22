import numpy as np
import pandas as pd

from quantlab.strategies import ManualRuleStrategy, MLTradingStrategy


def _prices(n=400, seed=5):
    dates = pd.bdate_range("2018-01-01", periods=n)
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, n))), index=dates)


def test_manual_strategy_positions_valid():
    prices = _prices()
    positions = ManualRuleStrategy().positions(prices)
    assert positions.index.equals(prices.index)
    assert set(positions.unique()).issubset({-1.0, 0.0, 1.0})
    # Flat during warm-up
    assert (positions.iloc[:10] == 0).all()


def test_ml_strategy_fit_predict_roundtrip():
    prices = _prices()
    strategy = MLTradingStrategy(n_estimators=10, seed=0)
    strategy.fit(prices.iloc[:250])
    positions = strategy.positions(prices.iloc[250:])
    assert set(positions.unique()).issubset({-1.0, 0.0, 1.0})
    assert positions.index.equals(prices.iloc[250:].index)


def test_ml_strategy_predictions_track_training_signal():
    # Deterministic-ish sinusoidal price: the learner should achieve positive
    # in-sample correlation between predicted and realized forward returns.
    dates = pd.bdate_range("2018-01-01", periods=500)
    t = np.arange(500)
    prices = pd.Series(100 + 10 * np.sin(t / 10), index=dates)
    strategy = MLTradingStrategy(n_estimators=20, lookahead=5, seed=1)
    strategy.fit(prices)
    preds = strategy.predicted_returns(prices)
    realized = (prices.shift(-5) / prices - 1).reindex(preds.index)
    both = pd.DataFrame({"p": preds, "r": realized}).dropna()
    assert np.corrcoef(both["p"], both["r"])[0, 1] > 0.5


def test_ml_strategy_requires_fit():
    import pytest

    with pytest.raises(RuntimeError):
        MLTradingStrategy().positions(_prices())
