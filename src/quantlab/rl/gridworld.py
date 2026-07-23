"""Grid-world environment for tabular Q-learning navigation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Cell codes used in the classic CSV mazes:
# 0 empty, 1 wall, 2 start (legacy), 3 goal, 5 start, 6 pit / obstacle
START_CODES = {2, 5}
GOAL_CODE = 3
WALL_CODES = {1}
PIT_CODES = {6}

# Actions: 0 N, 1 E, 2 S, 3 W
_DELTA = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}


class GridWorld:
    """Discrete maze; state id is ``row * ncols + col``."""

    def __init__(self, grid: np.ndarray):
        self.grid = np.asarray(grid, dtype=int)
        self.nrows, self.ncols = self.grid.shape
        starts = np.argwhere(np.isin(self.grid, list(START_CODES)))
        goals = np.argwhere(self.grid == GOAL_CODE)
        if len(starts) == 0 or len(goals) == 0:
            raise ValueError("Grid must contain a start and a goal cell")
        self.start = tuple(int(x) for x in starts[0])
        self.goal = tuple(int(x) for x in goals[0])
        self.pos = self.start

    @classmethod
    def from_csv(cls, path: str | Path) -> GridWorld:
        grid = np.loadtxt(path, delimiter=",", dtype=int)
        return cls(grid)

    @property
    def n_states(self) -> int:
        return self.nrows * self.ncols

    def state_id(self, pos: tuple[int, int] | None = None) -> int:
        r, c = self.pos if pos is None else pos
        return r * self.ncols + c

    def reset(self) -> int:
        self.pos = self.start
        return self.state_id()

    def step(self, action: int) -> tuple[int, float, bool]:
        """Take action; return ``(next_state, reward, done)``."""
        dr, dc = _DELTA[int(action)]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        if not (0 <= nr < self.nrows and 0 <= nc < self.ncols):
            return self.state_id(), -1.0, False
        cell = int(self.grid[nr, nc])
        if cell in WALL_CODES:
            return self.state_id(), -1.0, False
        self.pos = (nr, nc)
        if cell in PIT_CODES:
            reward, done = -100.0, True
            self.pos = self.start
            return self.state_id(), reward, done
        if (nr, nc) == self.goal:
            return self.state_id(), 1.0, True
        return self.state_id(), -1.0, False


def train_on_grid(
    world: GridWorld,
    learner,
    episodes: int = 500,
    max_steps: int = 1000,
) -> list[float]:
    """Train a QLearner-like agent; return per-episode cumulative rewards."""
    rewards: list[float] = []
    for _ in range(episodes):
        state = world.reset()
        action = learner.choose_action(state, explore=True)
        total = 0.0
        for _ in range(max_steps):
            next_state, reward, done = world.step(action)
            total += reward
            if done:
                learner.update(state, action, reward, next_state)
                break
            next_action = learner.choose_action(next_state, explore=True)
            learner.update(state, action, reward, next_state)
            state, action = next_state, next_action
        rewards.append(total)
    return rewards


def greedy_success(world: GridWorld, learner, max_steps: int = 1000) -> bool:
    """Whether a greedy policy reaches the goal from start."""
    state = world.reset()
    for _ in range(max_steps):
        action = learner.choose_action(state, explore=False)
        state, _, done = world.step(action)
        if done and world.pos == world.goal:
            return True
        if done:
            return False
    return False
