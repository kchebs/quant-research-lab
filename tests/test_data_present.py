from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def test_data_directory_has_csv_files():
    csvs = list(DATA.glob("*.csv"))
    assert csvs, "Expected research CSV files under data/"
    # Spot-check one file loads
    df = pd.read_csv(csvs[0])
    assert len(df.columns) >= 1
