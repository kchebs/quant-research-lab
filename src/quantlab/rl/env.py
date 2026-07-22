"""Single-asset trading environment for the tabular Q-learner.

State: technical indicators discretized into quantile bins (bins are learned
on the training window only, then reused out-of-sample so no look-ahead
leaks in). The agent's current position is part of the state so it can learn
to consider switching costs.

Actions: 0 = short, 1 = flat, 2 = long.

Reward: next-day portfolio return of the held position, minus an impact
penalty when the position changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.indicators import indicator_frame
from quantlab.rl.qlearner import QLearner

ACTIONS = {0: -1, 1: 0, 2: 1}  # action id -> position


class TradingEnvironment:
    def __init__(self, n_bins: int = 4, impact: float = 0.001):
        self.n_bins = n_bins
        self.impact = impact
        self._bin_edges: dict[str, np.ndarray] | None = None

    @property
    def num_states(self) -> int:
        # bins per feature ^ n_features * 3 possible held positions
        return self.n_bins**4 * 3

    @property
    def num_actions(self) -> int:
        return len(ACTIONS)

    def fit_bins(self, prices: pd.Series) -> "TradingEnvironment":
        """Learn quantile bin edges from the training window."""
        features = indicator_frame(prices).dropna()
        self._bin_edges = {
            col: np.quantile(
                features[col], np.linspace(0, 1, self.n_bins + 1)[1:-1]
            )
            for col in features.columns
        }
        return self

    def discretize(self, prices: pd.Series) -> pd.Series:
        """Map each date to a feature-state id (before position encoding)."""
        if self._bin_edges is None:
            raise RuntimeError("Call fit_bins on the training window first")
        features = indicator_frame(prices).dropna()
        state = pd.Series(0, index=features.index)
        for col in features.columns:
            bins = np.searchsorted(self._bin_edges[col], features[col])
            state = state * self.n_bins + bins
        return state

    def _full_state(self, feature_state: int, position: int) -> int:
        return feature_state * 3 + (position + 1)

    def train(
        self,
        prices: pd.Series,
        learner: QLearner,
        n_episodes: int = 100,
        converge_patience: int = 5,
    ) -> QLearner:
        """Run Q-learning episodes over the training price window."""
        states = self.discretize(prices)
        rets = prices.pct_change().shift(-1).reindex(states.index)
        state_arr = states.values
        ret_arr = rets.values

        prev_reward = None
        stable = 0
        for _ in range(n_episodes):
            position = 0
            total_reward = 0.0
            for t in range(len(state_arr) - 1):
                s = self._full_state(state_arr[t], position)
                action = learner.choose_action(s)
                new_position = ACTIONS[action]
                cost = self.impact * abs(new_position - position)
                reward = new_position * ret_arr[t] - cost
                s_next = self._full_state(state_arr[t + 1], new_position)
                learner.update(s, action, reward, s_next)
                position = new_position
                total_reward += reward
            if prev_reward is not None and abs(total_reward - prev_reward) < 1e-6:
                stable += 1
                if stable >= converge_patience:
                    break
            else:
                stable = 0
            prev_reward = total_reward
        return learner

    def positions(self, prices: pd.Series, learner: QLearner) -> pd.Series:
        """Greedy (no-exploration) policy rollout: position in {-1, 0, +1}."""
        states = self.discretize(prices)
        positions = pd.Series(0.0, index=prices.index)
        position = 0
        for date, feature_state in states.items():
            s = self._full_state(feature_state, position)
            position = ACTIONS[learner.choose_action(s, explore=False)]
            positions.loc[date] = position
        return positions
