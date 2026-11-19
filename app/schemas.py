"""Pydantic response models of the public API."""

from __future__ import annotations

from typing import Annotated, Any, Optional

from pydantic import BaseModel, Field

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

TARGET = "electricity_generation"
UNIT = "TWh"


class ForecastPoint(BaseModel):
    """One forecast year. ``lower``/``upper`` are null where no calibrated interval exists."""

    year: int
    forecast: FiniteFloat
    lower: Optional[FiniteFloat] = None
    upper: Optional[FiniteFloat] = None


class RandomForestPoint(ForecastPoint):
    """Random Forest row (kept fields: width and growth vs. previous year in %)."""

    uncertainty_width: Optional[FiniteFloat] = None
    growth_rate: Optional[FiniteFloat] = None


class XGBoostPoint(ForecastPoint):
    """XGBoost row (growth vs. previous year in %)."""

    growth_from_previous: Optional[FiniteFloat] = None


class ModelSummary(BaseModel):
    """Growth summary over the forecast period (see README for definitions)."""

    model: str
    period: str
    start_value: FiniteFloat
    end_value: FiniteFloat
    growth_periods: int = Field(description="Annual steps between first and last year")
    total_growth_percent: Optional[FiniteFloat] = None
    avg_annual_growth: Optional[FiniteFloat] = Field(
        None, description="Compound annual growth rate in % (CAGR)"
    )
    avg_uncertainty_width: Optional[FiniteFloat] = None


class IntervalInfo(BaseModel):
    """How the prediction intervals were obtained."""

    method: str
    nominal_coverage: float
    max_calibrated_horizon: int
    residuals_per_horizon: dict[str, int]


class Units(BaseModel):
    """Units of the response values."""

    target: str = TARGET
    forecast: str = UNIT
    uncertainty: str = (
        "empirical prediction interval targeting 95% coverage from rolling-origin "
        "backtest residuals; backtested coverage was below nominal (see README)"
    )


class ForecastsByModel(BaseModel):
    """Forecast rows of each model (empty list if a model is unavailable)."""

    prophet: list[ForecastPoint]
    random_forest: list[RandomForestPoint]
    xgboost: list[XGBoostPoint]


class AllMetadata(BaseModel):
    """Metadata of ``/forecast/all``."""

    years_ahead: int
    models: list[str]
    units: Units = Field(default_factory=Units)
    intervals: dict[str, Optional[IntervalInfo]] = Field(default_factory=dict)
    unavailable_models: list[str] = Field(default_factory=list)


class AllForecastsResponse(BaseModel):
    """Response of ``/forecast/all``."""

    forecasts: ForecastsByModel
    summaries: dict[str, Optional[ModelSummary]]
    metadata: AllMetadata


class SingleMetadata(BaseModel):
    """Metadata of a single-model forecast."""

    years_ahead: int
    units: Units = Field(default_factory=Units)
    interval: Optional[IntervalInfo] = None


class SingleForecastResponse(BaseModel):
    """Response of ``/forecast/{model}``."""

    model: str
    forecast: list[dict[str, Any]]
    summary: Optional[ModelSummary]
    metadata: SingleMetadata


class HealthResponse(BaseModel):
    """Backwards-compatible health report."""

    status: str
    models_loaded: bool
    models: dict[str, bool]
    api_version: str
