from pathlib import Path

from quantlab.data import load_fundamentals_snapshot
from quantlab.data.fundamentals import load_research_universe

DATA = Path(__file__).resolve().parents[1] / "data"


def test_snapshot_loads_and_types_are_numeric():
    df = load_fundamentals_snapshot(DATA / "2020_12.csv")
    assert len(df) > 1000
    for col in ["outlook", "rating", "five_year", "peg", "profit_margin"]:
        assert df[col].dtype.kind in "if", f"{col} should be numeric"


def test_snapshot_flags_are_binary():
    df = load_fundamentals_snapshot(DATA / "2020_09.csv")
    for col in ["alcohol", "gambling", "tobacco", "coal"]:
        values = set(df[col].dropna().unique())
        assert values.issubset({0.0, 1.0}), f"{col} has values {values}"


def test_no_duplicate_tickers():
    df = load_fundamentals_snapshot(DATA / "2020_12.csv")
    assert df["ticker"].is_unique


def test_research_universe_merges_snapshots():
    df = load_research_universe(DATA)
    assert len(df) >= 1000
    # December profitability fields and September size fields both present
    for col in ["profit_margin", "peg", "revenue", "assets", "market_value", "net_per_emp"]:
        assert col in df.columns
    # Enough joint outlook / 5-year observations for the hypothesis test
    joint = df[["outlook", "five_year"]].dropna()
    assert len(joint) >= 352
