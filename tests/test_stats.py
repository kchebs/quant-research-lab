import numpy as np
import pandas as pd

from quantlab.stats import (
    regression_report,
    required_sample_size,
    significant_pairs,
    spearman_matrix,
)


def test_required_sample_size_matches_reference():
    # 4102 public companies, 95% confidence, 5% margin -> ~352 (Cochran w/ FPC)
    assert required_sample_size(4102) == 352


def test_required_sample_size_monotonic_in_margin():
    assert required_sample_size(4102, margin_of_error=0.01) > required_sample_size(
        4102, margin_of_error=0.05
    )


def test_spearman_detects_monotonic_relationship():
    rng = np.random.default_rng(0)
    x = rng.uniform(size=200)
    df = pd.DataFrame({"x": x, "y": x**3 + rng.normal(0, 0.01, 200), "noise": rng.uniform(size=200)})
    rho, pval = spearman_matrix(df)
    assert rho.loc["x", "y"] > 0.95
    assert pval.loc["x", "y"] < 0.001
    assert pval.loc["x", "noise"] > 0.01


def test_significant_pairs_filters():
    rng = np.random.default_rng(1)
    x = rng.uniform(size=300)
    df = pd.DataFrame({"a": x, "b": x + rng.normal(0, 0.05, 300), "c": rng.uniform(size=300)})
    rho, pval = spearman_matrix(df)
    pairs = significant_pairs(rho, pval)
    assert len(pairs) == 1
    assert set(pairs.iloc[0][["var_1", "var_2"]]) == {"a", "b"}


def test_regression_report_verdicts():
    rng = np.random.default_rng(2)
    x = pd.Series(rng.uniform(size=100))
    strong = regression_report(x, 3 * x + rng.normal(0, 0.1, 100))
    assert strong.reject_null
    assert "REJECT" in strong.summary()
    weak = regression_report(x, pd.Series(rng.uniform(size=100)))
    assert not weak.reject_null


def test_regression_report_handles_missing_values():
    x = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    y = pd.Series([2.0, 4.1, 6.0, np.nan, 10.2, 12.0])
    result = regression_report(x, y)
    assert result.n == 4
