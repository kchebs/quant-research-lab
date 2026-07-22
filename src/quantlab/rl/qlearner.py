"""Tabular Q-learning agent (from scratch, numpy only)."""

from __future__ import annotations

import numpy as np


class QLearner:
    """Classic epsilon-greedy tabular Q-learner with optional Dyna-Q.

    Parameters
    ----------
    num_states, num_actions : size of the (discrete) state/action spaces.
    alpha : learning rate.
    gamma : discount factor.
    epsilon : initial exploration probability.
    epsilon_decay : multiplicative decay applied to epsilon after each update.
    dyna : number of hallucinated experience replays after each real update.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        alpha: float = 0.2,
        gamma: float = 0.9,
        epsilon: float = 0.5,
        epsilon_decay: float = 0.999,
        dyna: int = 0,
        seed: int | None = None,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.dyna = dyna
        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros((num_states, num_actions))
        self._memory: list[tuple[int, int, float, int]] = []

    def choose_action(self, state: int, explore: bool = True) -> int:
        """Epsilon-greedy action selection; greedy when explore=False."""
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.num_actions))
        return int(np.argmax(self.q_table[state]))

    def _td_update(self, state: int, action: int, reward: float, next_state: int) -> None:
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        self.q_table[state, action] += self.alpha * (
            td_target - self.q_table[state, action]
        )

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        """One Q-learning temporal-difference update (+ optional Dyna replays)."""
        self._td_update(state, action, reward, next_state)
        self._memory.append((state, action, reward, next_state))
        self.epsilon *= self.epsilon_decay
        if self.dyna > 0 and self._memory:
            for _ in range(self.dyna):
                s, a, r, sp = self._memory[int(self.rng.integers(len(self._memory)))]
                self._td_update(s, a, r, sp)
