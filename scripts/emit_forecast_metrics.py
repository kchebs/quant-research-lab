#!/usr/bin/env python3
"""Emit naive next-day forecast MAE/RMSE for CI (no network required).

Uses cached local prices under data/prices/ when present; otherwise synthetic
prices (same spirit as scorecard smoke). Writes artifacts/forecast_smoke.json.

Honest caveat: this is a pipeline / baseline demo, not live trading edge.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.forecasting.naive import (
    forecast_errors,
    last_value_forecast,
    mae,
    mean_return_forecast,
    rmse,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "forecast_smoke.json"
CACHE_DIR = ROOT / "data" / "prices"


def _load_cached_close() -> tuple[pd.Series, str] | None:
    if not CACHE_DIR.is_dir():
        return None
    csvs = sorted(CACHE_DIR.glob("*.csv"))
    if not csvs:
        return None
    path = csvs[0]
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    col = "Close" if "Close" in df.columns else df.columns[0]
    prices = df[col].dropna().astype(float)
    if len(prices) < 40:
        return None
    return prices.rename("close"), path.stem


def _synthetic_prices(n: int = 252, seed: int = 11) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=n)
    rets = rng.normal(0.0005, 0.01, size=n)
    return pd.Series(100 * np.exp(np.cumsum(rets)), index=idx, name="close")


def _metrics_for(prices: pd.Series, forecast: pd.Series) -> dict[str, float | int]:
    errors = forecast_errors(prices, forecast)
    return {
        "n": int(len(errors)),
        "mae": mae(errors),
        "rmse": rmse(errors),
    }


def main() -> None:
    loaded = _load_cached_close()
    if loaded is not None:
        prices, symbol = loaded
        mode = "cached_local_prices"
        note = (
            f"Cached local prices ({symbol}) — offline only; "
            "naive baselines are not a live trading edge."
        )
    else:
        prices = _synthetic_prices()
        symbol = "SYNTHETIC"
        mode = "ci_synthetic_smoke"
        note = (
            "Synthetic prices for CI only — not live market data and "
            "not a claimed trading edge."
        )

    last = last_value_forecast(prices)
    mean_ret = mean_return_forecast(prices, window=20)

    payload = {
        "schema_version": 1,
        "mode": mode,
        "symbol": symbol,
        "note": note,
        "models": {
            "last_value": _metrics_for(prices, last),
            "mean_return": _metrics_for(prices, mean_ret),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
