#!/usr/bin/env python3
"""Emit a reproducible scorecard metrics JSON for CI (no fabricated returns).

Uses synthetic prices so CI does not require network or cached market data.
Validates the backtest + metrics pipeline and writes artifacts/scorecard_smoke.json.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.backtest.engine import buy_and_hold_benchmark, positions_to_trades, run_backtest
from quantlab.backtest.metrics import summarize

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "scorecard_smoke.json"


def main() -> None:
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2020-01-02", periods=252)
    rets = rng.normal(0.0005, 0.01, size=len(idx))
    prices = pd.Series(100 * np.exp(np.cumsum(rets)), index=idx, name="close")

    z = (prices.pct_change() - prices.pct_change().rolling(20).mean()) / (
        prices.pct_change().rolling(20).std()
    )
    positions = pd.Series(0.0, index=idx)
    positions = positions.mask(z < -1, 1.0)
    positions = positions.mask(z > 1, -1.0)
    positions = positions.fillna(0.0)

    trades = positions_to_trades(positions, max_shares=100)
    port = run_backtest(trades, prices)
    bh = buy_and_hold_benchmark(prices, shares=100)
    payload = {
        "schema_version": 1,
        "mode": "ci_synthetic_smoke",
        "note": "Synthetic prices for CI only — not the research OOS scorecard.",
        "strategy": summarize(port["value"]).to_dict(),
        "buy_and_hold": summarize(bh["value"], name="buy_and_hold").to_dict(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
