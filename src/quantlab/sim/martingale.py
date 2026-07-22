"""Martingale betting on American roulette (black).

Doubling-after-loss until a target bankroll gain is hit, optionally with a
finite bankroll cap that can force ruin before the target.
"""

from __future__ import annotations

import numpy as np

AMERICAN_ROULETTE_BLACK_PROB = 18.0 / 38.0


def martingale_episode(
    win_prob: float = AMERICAN_ROULETTE_BLACK_PROB,
    target: float = 80.0,
    max_bets: int = 1000,
    bankroll_cap: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate one martingale path; return length-``max_bets`` winnings series.

    Winnings freeze at the terminal value once the episode ends (target hit,
    bankroll exhausted, or ``max_bets`` reached).
    """
    rng = rng if rng is not None else np.random.default_rng()
    winnings = np.zeros(max_bets, dtype=float)
    equity = 0.0
    bet = 1.0
    finished = False

    for i in range(max_bets):
        if finished:
            winnings[i] = equity
            continue

        won = rng.random() <= win_prob
        if won:
            equity += bet
            bet = 1.0
        else:
            equity -= bet
            bet *= 2.0
            if bankroll_cap is not None:
                remaining = bankroll_cap + equity
                if remaining <= 0:
                    equity = -bankroll_cap
                    finished = True
                elif bet > remaining:
                    bet = remaining

        winnings[i] = equity
        if equity >= target:
            finished = True
            if i + 1 < max_bets:
                winnings[i + 1 :] = equity
            break

    return winnings


def run_martingale_experiment(
    n_sims: int = 1000,
    max_bets: int = 1000,
    target: float = 80.0,
    bankroll_cap: float | None = None,
    win_prob: float = AMERICAN_ROULETTE_BLACK_PROB,
    seed: int | None = None,
) -> np.ndarray:
    """Run ``n_sims`` episodes; return array shaped ``(n_sims, max_bets)``."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_sims, max_bets), dtype=float)
    for i in range(n_sims):
        out[i] = martingale_episode(
            win_prob=win_prob,
            target=target,
            max_bets=max_bets,
            bankroll_cap=bankroll_cap,
            rng=rng,
        )
    return out


def summarize_winnings(paths: np.ndarray, target: float = 80.0) -> dict[str, float]:
    """Summary stats from a ``(n_sims, max_bets)`` winnings matrix."""
    terminal = paths[:, -1]
    hit_rate = float(np.mean(terminal >= target - 1e-9))
    return {
        "hit_rate": hit_rate,
        "expected_terminal": float(np.mean(terminal)),
        "median_terminal": float(np.median(terminal)),
        "std_terminal": float(np.std(terminal)),
    }
