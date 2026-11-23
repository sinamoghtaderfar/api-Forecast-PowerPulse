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


class CoverageStat(BaseModel):
    """Backtested (sequential, no look-ahead) interval coverage."""

    n: int
    coverage: float
    mean_width: float


class IntervalInfo(BaseModel):
    """How the prediction intervals were obtained and how they performed."""

    method: str
    nominal_coverage: float
    max_calibrated_horizon: int
    residuals_per_horizon: dict[str, int]
    scale_twh: Optional[float] = Field(
        None, description="Volatility scale (std of the last annual changes) in TWh"
    )
    scale_window_years: Optional[int] = None
    pooled_horizons: Optional[int] = None
    score_quantile: Optional[float] = None
    backtest_coverage: dict[str, CoverageStat] = Field(
        default_factory=dict,
        description="Coverage of intervals built without look-ahead, by horizon range",
    )
    validated_coverage: bool = Field(
        False, description="Never true: coverage is backtested, not guaranteed"
    )


class HorizonEvidence(BaseModel):
    """Backtest accuracy at one horizon (origins >= 2000)."""

    n: int
    mae: float
    bias: float = Field(description="Mean forecast - actual in TWh (>0 = too high)")
    relative_mae: Optional[float] = Field(
        None, description="MAE divided by the persistence benchmark's MAE"
    )


class ModelEvidence(BaseModel):
    """Out-of-sample backtest accuracy of one model."""

    first_origin: int
    by_horizon: dict[str, HorizonEvidence]
    mean_relative_mae_h1_10: Optional[float] = None


class HorizonGuidanceItem(BaseModel):
    """How much evidence supports forecasts in a horizon range."""

    years: str
    evidence: str


class Units(BaseModel):
    """Units of the response values."""

    target: str = TARGET
    forecast: str = UNIT
    uncertainty: str = (
        "empirical interval targeting 95% coverage from rolling-origin backtest "
        "residuals (volatility-scaled conformal, horizons 1-10 only); coverage is "
        "not guaranteed (see README)"
    )


class ForecastsByModel(BaseModel):
    """Forecast rows of each model (empty list if a model is unavailable)."""

    prophet: list[ForecastPoint]
    random_forest: list[RandomForestPoint]
    xgboost: list[XGBoostPoint]
    persistence: list[ForecastPoint] = Field(default_factory=list)


class AllMetadata(BaseModel):
    """Metadata of ``/forecast/all``."""

    years_ahead: int
    models: list[str]
    units: Units = Field(default_factory=Units)
    intervals: dict[str, Optional[IntervalInfo]] = Field(default_factory=dict)
    unavailable_models: list[str] = Field(default_factory=list)
    benchmark_model: str = "persistence"
    recommended_model: Optional[str] = Field(
        None,
        description=(
            "Lowest backtest MAE over horizons 1-10; the benchmark unless another "
            "model beats it by more than 2 %"
        ),
    )
    evidence: dict[str, Optional[ModelEvidence]] = Field(default_factory=dict)
    horizon_guidance: dict[str, HorizonGuidanceItem] = Field(default_factory=dict)
    assumptions: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


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
    evidence: Optional[ModelEvidence] = None
    horizon_guidance: dict[str, HorizonGuidanceItem] = Field(default_factory=dict)
    assumptions: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


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
