"""Naive price forecasting helpers for CI / smoke demos."""

from quantlab.forecasting.naive import (
    forecast_errors,
    last_value_forecast,
    mae,
    mean_return_forecast,
    rmse,
)

__all__ = [
    "forecast_errors",
    "last_value_forecast",
    "mae",
    "mean_return_forecast",
    "rmse",
]
