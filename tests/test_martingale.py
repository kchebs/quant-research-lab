import numpy as np

from quantlab.sim import (
    run_martingale_experiment,
    summarize_winnings,
)


def test_unlimited_bankroll_hits_target():
    paths = run_martingale_experiment(
        n_sims=50, max_bets=1000, bankroll_cap=None, seed=42
    )
    summary = summarize_winnings(paths, target=80)
    assert summary["hit_rate"] == 1.0
    assert summary["expected_terminal"] == 80.0


def test_capped_bankroll_can_fail():
    paths = run_martingale_experiment(
        n_sims=200, max_bets=1000, bankroll_cap=256.0, seed=7
    )
    summary = summarize_winnings(paths, target=80)
    assert 0.4 < summary["hit_rate"] < 0.9
    assert summary["expected_terminal"] < 80.0
