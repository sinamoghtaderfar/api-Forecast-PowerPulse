"""
Rolling-origin (expanding-window) backtesting with genuine recursive forecasts.

At every origin the forecaster only receives raw rows up to and including the
origin year, so imputation, exogenous growth estimation, feature construction
and model fitting use exactly the information available at that time. The
future exogenous variables are projected, never taken from the actual data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

# Minimum number of training years before the first backtest origin.
MIN_TRAIN_YEARS = 12

RawForecaster = Callable[[pd.DataFrame, int], np.ndarray]


@dataclass(frozen=True)
class BacktestRecord:
    """One recursive forecast made at ``origin`` for ``origin + horizon``."""

    origin: int
    horizon: int
    year: int
    actual: float
    predicted: float


def backtest_origins(
    raw: pd.DataFrame,
    min_train_years: int = MIN_TRAIN_YEARS,
    first_origin: int | None = None,
) -> list[int]:
    """Origins from the first one with enough history to the second-to-last year."""
    years = raw["year"].astype(int).tolist()
    start = years[0] + min_train_years - 1
    if first_origin is not None:
        start = max(start, first_origin)
    return [year for year in years[:-1] if year >= start]


def rolling_origin_backtest(
    forecaster: RawForecaster,
    raw: pd.DataFrame,
    *,
    max_horizon: int | None = None,
    min_train_years: int = MIN_TRAIN_YEARS,
    first_origin: int | None = None,
) -> list[BacktestRecord]:
    """
    Run ``forecaster(raw_up_to_origin, horizon)`` at every origin and compare
    the recursive forecasts with the actual targets that followed.
    """
    actual = dict(zip(raw["year"].astype(int), raw["y"].astype(float)))
    last_year = max(actual)
    records = []
    for origin in backtest_origins(raw, min_train_years, first_origin):
        horizon = last_year - origin
        if max_horizon is not None:
            horizon = min(horizon, max_horizon)
        if horizon < 1:
            continue
        train = raw[raw["year"] <= origin].reset_index(drop=True)
        predictions = np.asarray(forecaster(train, horizon), dtype=float)
        if predictions.shape != (horizon,) or not np.isfinite(predictions).all():
            raise ValueError(f"Invalid backtest forecast at origin {origin}")
        for step, predicted in enumerate(predictions, start=1):
            year = origin + step
            records.append(
                BacktestRecord(origin, step, year, actual[year], float(predicted))
            )
    return records
