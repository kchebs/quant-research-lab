"""Adjusted daily price loading with a local CSV cache.

Prices come from Yahoo Finance via ``yfinance`` (auto-adjusted closes). Each
symbol's full downloaded history is cached under ``data/prices/`` so that
notebooks re-run deterministically and offline once the cache is warm.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "prices"

# Cache files hold history from this date onward.
_HISTORY_START = "2000-01-01"


def _cache_path(symbol: str, cache_dir: Path) -> Path:
    return cache_dir / f"{symbol.upper().replace('.', '-')}.csv"


def _download_history(symbol: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        symbol,
        start=_HISTORY_START,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )
    if raw is None or raw.empty:
        raise ValueError(f"No price data returned for symbol {symbol!r}")
    raw.index.name = "Date"
    return raw


def load_symbol_history(
    symbol: str,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return the full cached OHLCV history for one symbol (download on miss)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(symbol, cache_dir)
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col="Date", parse_dates=True)
    history = _download_history(symbol)
    history.to_csv(path)
    return history


def load_prices(
    symbols: list[str] | str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    field: str = "Close",
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    dropna: bool = True,
) -> pd.DataFrame:
    """Load a DataFrame of adjusted prices, one column per symbol.

    Parameters
    ----------
    symbols : ticker(s) to load.
    start, end : inclusive date window.
    field : OHLCV field to extract ("Close" is dividend/split adjusted).
    dropna : drop dates where any symbol is missing (aligned trading days).
    """
    if isinstance(symbols, str):
        symbols = [symbols]
    frames = {}
    for symbol in symbols:
        history = load_symbol_history(symbol, cache_dir=cache_dir)
        frames[symbol] = history[field]
    prices = pd.DataFrame(frames).loc[pd.Timestamp(start) : pd.Timestamp(end)]
    if dropna:
        prices = prices.dropna()
    return prices


def daily_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Simple daily returns; the first row is dropped."""
    return prices.pct_change().iloc[1:]
