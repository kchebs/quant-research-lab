"""Statistical research utilities: sample sizing, rank correlation, regression."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def required_sample_size(
    population: int,
    confidence: float = 0.95,
    margin_of_error: float = 0.05,
    proportion: float = 0.5,
) -> int:
    """Minimum sample size for estimating a proportion in a finite population.

    Uses the standard Cochran formula with finite-population correction.
    ``proportion=0.5`` is the conservative (largest-sample) choice.
    """
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    numerator = population * z**2 * proportion * (1 - proportion)
    denominator = margin_of_error**2 * (population - 1) + z**2 * proportion * (1 - proportion)
    return math.ceil(numerator / denominator)


def spearman_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise Spearman rank correlations and two-sided p-values.

    Spearman is used (rather than Pearson) because the fundamentals are not
    normally distributed and contain outliers. Pairs are compared using all
    rows where both columns are present (pairwise-complete observations).
    """
    numeric = df.select_dtypes(include=[np.number])
    cols = numeric.columns
    rho = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    pval = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            pair = numeric[[a, b]].dropna()
            if len(pair) < 3 or pair[a].nunique() < 2 or pair[b].nunique() < 2:
                r, p = np.nan, np.nan
            else:
                r, p = stats.spearmanr(pair[a], pair[b])
            rho.loc[a, b] = rho.loc[b, a] = r
            pval.loc[a, b] = pval.loc[b, a] = p
    return rho, pval


def significant_pairs(
    rho: pd.DataFrame,
    pval: pd.DataFrame,
    alpha: float = 0.05,
    min_abs_rho: float = 0.5,
) -> pd.DataFrame:
    """Tidy table of variable pairs with p <= alpha and |rho| >= min_abs_rho."""
    rows = []
    cols = list(rho.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r, p = rho.loc[a, b], pval.loc[a, b]
            r_f = float(r)
            p_f = float(p)
            if pd.notna(p_f) and p_f <= alpha and abs(r_f) >= min_abs_rho:
                rows.append(
                    {"var_1": a, "var_2": b, "spearman_rho": r_f, "p_value": p_f}
                )
    result = pd.DataFrame(rows, columns=["var_1", "var_2", "spearman_rho", "p_value"])
    return result.sort_values("spearman_rho", ascending=False).reset_index(drop=True)


@dataclass
class RegressionResult:
    slope: float
    intercept: float
    r_value: float
    p_value: float
    std_err: float
    n: int
    alpha: float = 0.05

    @property
    def reject_null(self) -> bool:
        """True if the no-relationship null hypothesis is rejected at alpha."""
        return self.p_value <= self.alpha

    def summary(self, x_name: str = "x", y_name: str = "y") -> str:
        verdict = (
            f"REJECT the null hypothesis at alpha={self.alpha}: "
            f"the linear relationship between {x_name} and {y_name} is statistically significant."
            if self.reject_null
            else f"FAIL TO REJECT the null hypothesis at alpha={self.alpha}: "
            f"no statistically significant linear relationship between {x_name} and {y_name}."
        )
        return (
            f"{y_name} = {self.slope:.4f} * {x_name} + {self.intercept:.4f}\n"
            f"r = {self.r_value:.4f}, r^2 = {self.r_value ** 2:.4f}, "
            f"p-value = {self.p_value:.4g}, std_err = {self.std_err:.4f}, n = {self.n}\n"
            f"{verdict}"
        )


def regression_report(
    x: pd.Series, y: pd.Series, alpha: float = 0.05
) -> RegressionResult:
    """Ordinary least-squares fit of y on x with a significance verdict."""
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    slope, intercept, r_value, p_value, std_err = stats.linregress(pair["x"], pair["y"])
    return RegressionResult(
        slope=slope,
        intercept=intercept,
        r_value=r_value,
        p_value=p_value,
        std_err=std_err,
        n=len(pair),
        alpha=alpha,
    )
