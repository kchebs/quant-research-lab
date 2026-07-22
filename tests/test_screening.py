import numpy as np
import pandas as pd

from quantlab.screening import apply_rules, default_rules


def _universe():
    return pd.DataFrame(
        {
            "ticker": ["GOOD", "WEAK", "MISSING"],
            "net_per_emp": [50_000.0, 5_000.0, np.nan],
            "profit_margin": [15.0, 1.0, np.nan],
            "three_year": [20.0, 5.0, np.nan],
            "five_year": [25.0, 5.0, np.nan],
            "ten_year": [15.0, 5.0, np.nan],
            "one_year": [40.0, 2.0, np.nan],
            "upward_trend": [1.0, 0.0, np.nan],
            "outlook": [80.0, 30.0, np.nan],
            "rating": [4.2, 2.0, np.nan],
            "peg": [0.8, 3.0, np.nan],
        }
    )


def test_all_rules_pass_for_strong_company():
    rules = default_rules(sp500_1yr=18.0, sp500_5yr=15.0)
    result = apply_rules(_universe(), rules)
    good = result[result["ticker"] == "GOOD"].iloc[0]
    assert good["passes_all"]
    assert good["rules_passed"] == len(rules)


def test_weak_company_fails_all_rules():
    rules = default_rules(sp500_1yr=18.0, sp500_5yr=15.0)
    result = apply_rules(_universe(), rules)
    weak = result[result["ticker"] == "WEAK"].iloc[0]
    assert weak["rules_passed"] == 0


def test_missing_data_counts_as_fail_not_error():
    rules = default_rules(sp500_1yr=18.0, sp500_5yr=15.0)
    result = apply_rules(_universe(), rules)
    missing = result[result["ticker"] == "MISSING"].iloc[0]
    assert missing["rules_passed"] == 0
    assert missing["rules_evaluable"] == 0


def test_ranking_order():
    rules = default_rules(sp500_1yr=18.0, sp500_5yr=15.0)
    result = apply_rules(_universe(), rules)
    assert result["ticker"].iloc[0] == "GOOD"
    assert (result["rules_passed"].diff().dropna() <= 0).all()
