"""Generate figures for docs/RESEARCH_REPORT.md (deterministic seeds)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantlab.experiments import (
    best_for_decision_tree,
    best_for_linear_regression,
    leaf_size_rmse_curve,
    rmse,
    train_test_split_shuffle,
)
from quantlab.experiments.assess_learners import compare_dt_rt_timing
from quantlab.learners import LinearRegressionLearner, RegressionTree
from quantlab.sim import run_martingale_experiment, summarize_winnings

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img" / "report"
OUT.mkdir(parents=True, exist_ok=True)


def fig_martingale() -> dict:
    uncapped = run_martingale_experiment(n_sims=200, bankroll_cap=None, seed=42)
    capped = run_martingale_experiment(n_sims=200, bankroll_cap=256.0, seed=42)
    s1 = summarize_winnings(uncapped)
    s2 = summarize_winnings(capped)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i in range(min(10, len(uncapped))):
        axes[0].plot(uncapped[i, :300], alpha=0.7, lw=1)
    axes[0].set_title("First test style: unlimited bankroll")
    axes[0].set_xlabel("Bet #")
    axes[0].set_ylabel("Winnings")
    axes[0].axhline(80, color="k", ls="--", lw=0.8)

    mean = capped.mean(axis=0)
    std = capped.std(axis=0)
    x = np.arange(300)
    axes[1].plot(x, mean[:300], color="black", label="mean")
    axes[1].plot(x, mean[:300] + std[:300], color="red", alpha=0.5)
    axes[1].plot(x, mean[:300] - std[:300], color="red", alpha=0.5)
    axes[1].set_title("After cleanup: capped bankroll mean±SD")
    axes[1].set_xlabel("Bet #")
    fig.tight_layout()
    fig.savefig(OUT / "martingale_paths.png", dpi=120)
    plt.close(fig)
    return {"uncapped": s1, "capped": s2}


def fig_learners() -> dict:
    path = ROOT / "data" / "learners" / "Istanbul.csv"
    df = pd.read_csv(path)
    # Istanbul.csv historically has a date column then features; last col is target-ish
    data = df.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    X, y = data[:, :-1], data[:, -1]
    Xtr, Xte, ytr, yte = train_test_split_shuffle(X, y, seed=0)
    curve = leaf_size_rmse_curve(
        Xtr, ytr, Xte, yte, leaf_sizes=range(1, 81, 2), seed=0
    )
    bag = leaf_size_rmse_curve(
        Xtr, ytr, Xte, yte, leaf_sizes=range(1, 81, 2), bag=True, n_estimators=10, seed=0
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(curve["leaf_size"], curve["in_rmse"], label="in-sample")
    axes[0].plot(curve["leaf_size"], curve["out_rmse"], label="out-of-sample")
    axes[0].set_title("DT leaf-size RMSE (Istanbul)")
    axes[0].set_xlabel("leaf size")
    axes[0].legend()
    axes[1].plot(bag["leaf_size"], bag["in_rmse"], label="in-sample")
    axes[1].plot(bag["leaf_size"], bag["out_rmse"], label="out-of-sample")
    axes[1].set_title("Bagged DT leaf-size RMSE")
    axes[1].set_xlabel("leaf size")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "learners_leaf_rmse.png", dpi=120)
    plt.close(fig)

    timing = compare_dt_rt_timing(Xtr, ytr, Xte, yte, leaf_sizes=range(1, 51, 5), seed=0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(timing["leaf_size"], timing["dt_mae"], label="DT")
    axes[0].plot(timing["leaf_size"], timing["rt_mae"], label="RT")
    axes[0].set_title("Out-of-sample MAE")
    axes[0].legend()
    axes[1].plot(timing["leaf_size"], timing["dt_seconds"], label="DT")
    axes[1].plot(timing["leaf_size"], timing["rt_seconds"], label="RT")
    axes[1].set_title("Fit+predict time (s)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "learners_dt_vs_rt.png", dpi=120)
    plt.close(fig)

    Xl, yl = best_for_linear_regression()
    Xd, yd = best_for_decision_tree()
    results = {}
    for name, X, y in [("lin", Xl, yl), ("dt", Xd, yd)]:
        Xtr, Xte, ytr, yte = train_test_split_shuffle(X, y, seed=1)
        lr = LinearRegressionLearner().fit(Xtr, ytr)
        dt = RegressionTree(leaf_size=1).fit(Xtr, ytr)
        results[name] = {
            "lr_rmse": rmse(yte, lr.predict(Xte)),
            "dt_rmse": rmse(yte, dt.predict(Xte)),
        }
    return {"best4": results}


def fig_tos_synthetic() -> None:
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0005, 0.01, size=252)
    prices = pd.Series(100 * np.cumprod(1 + rets), index=pd.date_range("2020-01-01", periods=252, freq="B"))
    from quantlab.backtest import run_backtest
    from quantlab.strategies import theoretically_optimal_trades

    trades = theoretically_optimal_trades(prices, max_shares=100)
    tos = run_backtest(trades, prices, start_cash=100_000, commission=0, impact=0)["value"]
    bh_tr = pd.Series(0.0, index=prices.index)
    bh_tr.iloc[0] = 100
    bh = run_backtest(bh_tr, prices, start_cash=100_000, commission=0, impact=0)["value"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tos / tos.iloc[0], label="Theoretically optimal")
    ax.plot(bh / bh.iloc[0], label="Buy & hold")
    ax.set_title("TOS ceiling vs buy & hold (synthetic path)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "tos_ceiling.png", dpi=120)
    plt.close(fig)


def main() -> None:
    m = fig_martingale()
    l = fig_learners()
    fig_tos_synthetic()
    summary = OUT / "generation_summary.txt"
    summary.write_text(
        f"martingale uncapped={m['uncapped']}\nmartingale capped={m['capped']}\nbest4={l['best4']}\n"
    )
    print("Wrote figures to", OUT)
    print(summary.read_text())


if __name__ == "__main__":
    main()
