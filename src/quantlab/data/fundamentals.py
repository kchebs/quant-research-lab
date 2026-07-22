"""Loading and cleaning of the 2020 fundamentals + Glassdoor-outlook snapshots.

The CSVs under ``data/`` were assembled in 2020 by merging a hand-built ticker
universe with fields scraped from Morningstar (profitability, valuation,
trailing returns) and Glassdoor ("Positive Business Outlook" and rating).
They are messy by nature: numbers with thousands separators, em-dash
placeholders, mixed yes/no/1/0 flags. This module centralizes the cleanup so
every notebook works from one canonical loading path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"

# Values scraped sites use as "no data" placeholders.
_NA_TOKENS = ["—", "-", "#DIV/0!", "nan", "N/A", ""]

_NUMERIC_COLUMNS = [
    "outlook",
    "rating",
    "revenue",
    "rev_change_percent",
    "profit",
    "profit_change_percent",
    "assets",
    "market_value",
    "net_income",
    "ft_employee",
    "div_yield",
    "one_year",
    "three_year",
    "five_year",
    "ten_year",
    "fifteen_year",
    "rd",
    "peg",
    "profit_margin",
    "roa",
    "roe",
]

_ESG_FLAGS = [
    "alcohol",
    "adult",
    "gambling",
    "tobacco",
    "controversial_weapons",
    "small_arms",
    "military_contracting",
    "coal",
]

_YES_NO_MAP = {"yes": 1.0, "y": 1.0, "1": 1.0, "no": 0.0, "n": 0.0, "mo": 0.0, "0": 0.0}

# "pass" was used in the source sheets to mean the trend check passed;
# "maybe" / "too early" are treated as unknown rather than a hard no.
_TREND_MAP = {"yes": 1.0, "pass": 1.0, "no": 0.0, "maybe": np.nan, "too early": np.nan}


def _to_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(",", "", regex=False)
        .replace(_NA_TOKENS, pd.NA)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _map_flag(series: pd.Series, mapping: dict[str, float]) -> pd.Series:
    return series.astype("string").str.strip().str.lower().map(mapping).astype(float)


def load_fundamentals_snapshot(path: str | Path) -> pd.DataFrame:
    """Load one monthly snapshot CSV and return a cleaned, typed DataFrame."""
    df = pd.read_csv(path)

    df.columns = [str(c).strip() for c in df.columns]
    drop_cols = [
        c
        for c in df.columns
        if c.startswith("Unnamed") or c in {"", "yahoo diagram", "rec", "avg"}
    ]
    df = df.drop(columns=drop_cols)
    # Snapshots merged twice carry sector_x/sector_y style duplicates.
    df = df.rename(columns={"sector_y": "sector", "industry_y": "industry"})

    for col in _NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = _to_numeric(df[col])

    for col in ["use", "trust", *_ESG_FLAGS]:
        if col in df.columns:
            df[col] = _map_flag(df[col], _YES_NO_MAP)
    if "upward_trend" in df.columns:
        df["upward_trend"] = _map_flag(df["upward_trend"], _TREND_MAP)

    df = df.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"], keep="first")

    # Net income is reported in millions USD in the source snapshots.
    if {"net_income", "ft_employee"}.issubset(df.columns):
        df["net_per_emp"] = df["net_income"] * 1e6 / df["ft_employee"]

    return df.reset_index(drop=True)


def load_research_universe(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Merge the 2020-09 and 2020-12 snapshots into one research table.

    The September snapshot contributes the universe, ESG exclusion flags,
    company size metrics (revenue / assets / market value) and employee
    counts; the December snapshot contributes fresher Morningstar
    profitability, valuation and trailing-return fields.
    """
    data_dir = Path(data_dir)
    sept = load_fundamentals_snapshot(data_dir / "2020_09.csv")
    dec = load_fundamentals_snapshot(data_dir / "2020_12.csv")

    dec_cols = [
        "ticker",
        "outlook",
        "rating",
        "use",
        "trust",
        "upward_trend",
        "net_income",
        "div_yield",
        "one_year",
        "three_year",
        "five_year",
        "ten_year",
        "fifteen_year",
        "rd",
        "peg",
        "profit_margin",
        "roa",
        "roe",
        "sector",
        "industry",
    ]
    merged = pd.merge(
        sept,
        dec[[c for c in dec_cols if c in dec.columns]],
        on="ticker",
        how="outer",
        suffixes=("_sept", ""),
    )
    # Prefer the December value, fall back to September, for overlapping fields.
    for col in [c[: -len("_sept")] for c in merged.columns if c.endswith("_sept")]:
        merged[col] = merged[col].fillna(merged[f"{col}_sept"])
    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_sept")])

    merged["net_per_emp"] = merged["net_income"] * 1e6 / merged["ft_employee"]
    return merged.reset_index(drop=True)
