from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.backtest import run_orders
from quantlab.rl import GridWorld, QLearner, greedy_success, train_on_grid


def test_run_orders_buy_and_hold():
    prices = pd.DataFrame(
        {"AAA": [100.0, 101.0, 102.0]},
        index=pd.date_range("2020-01-01", periods=3, freq="B"),
    )
    orders = pd.DataFrame(
        [{"Symbol": "AAA", "Order": "BUY", "Shares": 10}],
        index=[prices.index[0]],
    )
    values = run_orders(orders, prices, start_cash=10_000, commission=0, impact=0)
    assert len(values) == 3
    assert values.iloc[-1] == 10_000 - 10 * 100 + 10 * 102


def test_gridworld_learns_simple_maze(tmp_path: Path):
    # Tiny open corridor: start left, goal right
    grid = np.array(
        [
            [0, 0, 0, 0, 0],
            [5, 0, 0, 0, 3],
            [0, 0, 0, 0, 0],
        ]
    )
    world = GridWorld(grid)
    learner = QLearner(
        num_states=world.n_states,
        num_actions=4,
        alpha=0.5,
        gamma=0.9,
        epsilon=0.3,
        epsilon_decay=0.995,
        seed=0,
    )
    train_on_grid(world, learner, episodes=300, max_steps=50)
    assert greedy_success(world, learner, max_steps=20)
