"""
Alternative projections of the exogenous variables for backtests.

Every projection receives only the raw rows up to the forecast origin. The
``oracle`` projection is the exception: it reads actual future values and
exists only as a diagnostic upper bound (it is not a valid forecast).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from app.services.features import (
    EXOG_COLUMNS,
    exogenous_growth_rates,
    exogenous_path,
    prepare_history,
)

Projection = Callable[[pd.DataFrame, int], pd.DataFrame]

LIMITS = {
    "population": (1.0, np.inf),
    "gdp": (1.0, np.inf),
    "renewables_share_energy": (0.0, 100.0),
}


def cagr_current(raw: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Production behaviour: 10-year CAGR applied from the (carried-forward) last row."""
    history = prepare_history(raw)
    return exogenous_path(history.iloc[-1], exogenous_growth_rates(raw), horizon)


def _last_observed(raw: pd.DataFrame, col: str) -> tuple[int, float]:
    observed = raw[["year", col]].dropna()
    return int(observed["year"].iloc[-1]), float(observed[col].iloc[-1])


def _frame(origin: int, horizon: int, values: dict[str, np.ndarray]) -> pd.DataFrame:
    years = np.arange(origin + 1, origin + horizon + 1)
    frame = pd.DataFrame({"year": years})
    for col in EXOG_COLUMNS:
        low, high = LIMITS[col]
        frame[col] = np.clip(values[col], low, high)
    return frame


def cagr(window: int, damping: float | None = None) -> Projection:
    """
    CAGR over ``window`` years, applied from the last *observed* value (so a
    variable missing for the last years keeps growing through the gap).
    With ``damping`` the growth rate of step s is multiplied by damping**s.
    """

    def project(raw: pd.DataFrame, horizon: int) -> pd.DataFrame:
        origin = int(raw["year"].iloc[-1])
        rates = exogenous_growth_rates(raw, window=window)
        values = {}
        for col in EXOG_COLUMNS:
            last_year, last_value = _last_observed(raw, col)
            steps = np.arange(origin + 1, origin + horizon + 1) - last_year
            if damping is None:
                values[col] = last_value * (1 + rates[col]) ** steps
            else:
                max_step = int(steps[-1])
                factors = np.cumprod(
                    1 + rates[col] * damping ** np.arange(1, max_step + 1)
                )
                values[col] = last_value * factors[steps - 1]
        return _frame(origin, horizon, values)

    return project


def flat(raw: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Last observed value of each variable."""
    origin = int(raw["year"].iloc[-1])
    values = {
        col: np.full(horizon, _last_observed(raw, col)[1]) for col in EXOG_COLUMNS
    }
    return _frame(origin, horizon, values)


def linear(window: int) -> Projection:
    """Last observed value plus the OLS slope of the trailing ``window`` years."""

    def project(raw: pd.DataFrame, horizon: int) -> pd.DataFrame:
        origin = int(raw["year"].iloc[-1])
        values = {}
        for col in EXOG_COLUMNS:
            observed = raw[raw["year"] > origin - window][["year", col]].dropna()
            last_year, last_value = _last_observed(raw, col)
            slope = (
                np.polyfit(observed["year"], observed[col], 1)[0]
                if len(observed) >= 2
                else 0.0
            )
            steps = np.arange(origin + 1, origin + horizon + 1) - last_year
            values[col] = last_value + slope * steps
        return _frame(origin, horizon, values)

    return project


def oracle(full_raw: pd.DataFrame) -> Projection:
    """DIAGNOSTIC ONLY: actual future exogenous values (carry-forward where missing)."""
    filled = full_raw.copy()
    filled[EXOG_COLUMNS] = filled[EXOG_COLUMNS].ffill()
    by_year = filled.set_index("year")

    def project(raw: pd.DataFrame, horizon: int) -> pd.DataFrame:
        origin = int(raw["year"].iloc[-1])
        years = list(range(origin + 1, origin + horizon + 1))
        frame = by_year.loc[years, EXOG_COLUMNS].reset_index()
        return frame.rename(columns={"index": "year"})

    return project


PROJECTIONS: dict[str, Projection] = {
    "cagr10": cagr_current,
    "cagr5": cagr(5),
    "cagr20": cagr(20),
    "cagr10_damped": cagr(10, damping=0.8),
    "flat": flat,
    "linear10": linear(10),
}


def with_projection(service, project: Projection):
    """Forecaster that uses ``service``'s model with a different exogenous projection."""

    def forecast(raw: pd.DataFrame, horizon: int) -> np.ndarray:
        history = prepare_history(raw)
        exog = project(raw, horizon)
        model = service._fit_model(history)  # pylint: disable=protected-access
        return service._predict_point(  # pylint: disable=protected-access
            model, history, exog
        )

    return forecast
