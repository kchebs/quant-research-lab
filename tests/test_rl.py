import numpy as np
import pandas as pd

from quantlab.rl import QLearner, TradingEnvironment


def test_qlearner_learns_trivial_mdp():
    # Two states; action 1 always gives reward 1, action 0 gives 0.
    learner = QLearner(num_states=2, num_actions=2, epsilon=0.5, seed=0)
    rng = np.random.default_rng(0)
    state = 0
    for _ in range(500):
        action = learner.choose_action(state)
        reward = 1.0 if action == 1 else 0.0
        next_state = int(rng.integers(2))
        learner.update(state, action, reward, next_state)
        state = next_state
    assert learner.choose_action(0, explore=False) == 1
    assert learner.choose_action(1, explore=False) == 1


def test_epsilon_decays():
    learner = QLearner(num_states=2, num_actions=2, epsilon=0.5, epsilon_decay=0.9)
    for _ in range(10):
        learner.update(0, 0, 0.0, 1)
    assert learner.epsilon < 0.5 * 0.9**9 + 1e-9


def _trend_prices(n=300):
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(3)
    return pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, n))), index=dates
    )


def test_environment_state_space_and_rollout():
    prices = _trend_prices()
    env = TradingEnvironment(n_bins=4)
    env.fit_bins(prices)
    learner = QLearner(env.num_states, env.num_actions, seed=0)
    env.train(prices, learner, n_episodes=5)
    positions = env.positions(prices, learner)
    assert set(positions.unique()).issubset({-1.0, 0.0, 1.0})
    assert positions.index.equals(prices.index)


def test_discretize_requires_fit():
    import pytest

    env = TradingEnvironment()
    with pytest.raises(RuntimeError):
        env.discretize(_trend_prices())
