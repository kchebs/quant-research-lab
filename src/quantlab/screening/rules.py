"""Rule-based investment screening engine.

Implements the eight screening rules that were documented (but never coded)
in the original research notebook. Each rule is an explicit, auditable
predicate over the cleaned fundamentals table; the engine reports per-rule
pass/fail plus an overall score so partial passes remain visible.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class Rule:
    name: str
    description: str
    predicate: Callable[[pd.DataFrame], pd.Series]
    required_columns: tuple[str, ...] = field(default_factory=tuple)

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        """Boolean pass/fail per row; missing inputs evaluate to False."""
        result = self.predicate(df)
        return result.fillna(False).astype(bool)


def default_rules(sp500_1yr: float, sp500_5yr: float) -> list[Rule]:
    """The eight documented screening rules.

    Parameters
    ----------
    sp500_1yr : S&P 500 trailing 1-year return (%) as of the snapshot date.
    sp500_5yr : S&P 500 trailing 5-year annualized return (%) as of the
        snapshot date. Both are used by the "beats the market" rule.
    """
    return [
        Rule(
            "net_income_per_employee",
            "Net income per employee >= $10,000",
            lambda df: df["net_per_emp"] >= 10_000,
            ("net_per_emp",),
        ),
        Rule(
            "profit_margin",
            "Net profit margin > 2.57%",
            lambda df: df["profit_margin"] > 2.57,
            ("profit_margin",),
        ),
        Rule(
            "long_term_roi",
            "3-, 5-, and 10-year trailing ROI all > 10.1%",
            lambda df: (df["three_year"] > 10.1)
            & (df["five_year"] > 10.1)
            & (df["ten_year"] > 10.1),
            ("three_year", "five_year", "ten_year"),
        ),
        Rule(
            "upward_trend",
            "Price shows an upward trend",
            lambda df: df["upward_trend"] == 1,
            ("upward_trend",),
        ),
        Rule(
            "business_outlook",
            "Glassdoor positive business outlook >= 60%",
            lambda df: df["outlook"] >= 60,
            ("outlook",),
        ),
        Rule(
            "employee_rating",
            "Glassdoor overall rating >= 3.5",
            lambda df: df["rating"] >= 3.5,
            ("rating",),
        ),
        Rule(
            "peg_undervalued",
            "PEG ratio between 0 and 1 (growth-adjusted undervaluation)",
            lambda df: (df["peg"] >= 0) & (df["peg"] <= 1),
            ("peg",),
        ),
        Rule(
            "beats_sp500",
            f"Beats S&P 500 over 1 year (> {sp500_1yr:.1f}%) and 5 years (> {sp500_5yr:.1f}%)",
            lambda df: (df["one_year"] > sp500_1yr) & (df["five_year"] > sp500_5yr),
            ("one_year", "five_year"),
        ),
    ]


def apply_rules(df: pd.DataFrame, rules: list[Rule]) -> pd.DataFrame:
    """Evaluate every rule and append pass flags, a score, and a rank.

    Returns the input frame with one boolean column per rule (prefixed
    ``rule_``), ``rules_passed`` (count), ``rules_evaluable`` (how many rules
    had the required data present), and ``passes_all`` (passed every rule).
    Sorted by rules passed, descending.
    """
    out = df.copy()
    rule_cols = []
    evaluable = pd.Series(0, index=df.index)
    for rule in rules:
        col = f"rule_{rule.name}"
        out[col] = rule.evaluate(df)
        rule_cols.append(col)
        has_data = pd.Series(True, index=df.index)
        for req in rule.required_columns:
            has_data &= df[req].notna() if req in df.columns else False
        evaluable += has_data.astype(int)
    out["rules_passed"] = out[rule_cols].sum(axis=1)
    out["rules_evaluable"] = evaluable
    out["passes_all"] = out["rules_passed"] == len(rules)
    return out.sort_values(
        ["rules_passed", "rules_evaluable"], ascending=False
    ).reset_index(drop=True)
