"""Forecast endpoints for the Germany Energy Forecast API."""

from __future__ import annotations

import logging
import math
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.dependencies import SERVICES
from app.schemas import (
    AllForecastsResponse,
    AllMetadata,
    ForecastPoint,
    RandomForestPoint,
    SingleForecastResponse,
    SingleMetadata,
    XGBoostPoint,
)
from app.services.base import MAX_YEARS_AHEAD, BaseForecastService, ForecastRows

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Forecast"])

YearsAhead = Query(10, ge=1, le=MAX_YEARS_AHEAD, description="Years to forecast")

ROW_MODELS = {
    "prophet": ForecastPoint,
    "random_forest": RandomForestPoint,
    "xgboost": XGBoostPoint,
}


def calculate_model_summary(
    forecast_data: ForecastRows, model_name: str
) -> dict[str, Any] | None:
    """
    Growth and uncertainty summary of one model's forecast.

    * ``growth_periods``: annual steps between the first and last forecast year
      (``len(forecast_data) - 1``).
    * ``total_growth_percent``: ``(end / start - 1) * 100`` over those steps.
    * ``avg_annual_growth``: compound annual growth rate
      ``((end / start) ** (1 / growth_periods) - 1) * 100``.

    Growth values are None when undefined: a one-year horizon (no period),
    or non-positive / non-finite start or end values.
    """
    if not forecast_data:
        return None

    first, last = forecast_data[0], forecast_data[-1]
    start, end = float(first["forecast"]), float(last["forecast"])
    periods = len(forecast_data) - 1

    total_growth = avg_growth = None
    valid = math.isfinite(start) and math.isfinite(end) and start > 0 and end > 0
    if periods > 0 and valid:
        ratio = end / start
        total_growth = round((ratio - 1) * 100, 2)
        avg_growth = round((ratio ** (1 / periods) - 1) * 100, 2)

    widths = [
        row["upper"] - row["lower"]
        for row in forecast_data
        if row.get("lower") is not None and row.get("upper") is not None
    ]
    return {
        "model": model_name,
        "period": f"{first['year']}-{last['year']}",
        "start_value": start,
        "end_value": end,
        "growth_periods": periods,
        "total_growth_percent": total_growth,
        "avg_annual_growth": avg_growth,
        "avg_uncertainty_width": (
            round(sum(widths) / len(widths), 2) if widths else None
        ),
    }


def _unavailable(service: BaseForecastService) -> HTTPException:
    return HTTPException(
        status_code=503, detail=f"{service.model_name} model is not available"
    )


def _run_forecast(service: BaseForecastService, years_ahead: int) -> ForecastRows:
    if not service.is_ready:
        raise _unavailable(service)
    try:
        return service.forecast(years_ahead=years_ahead)
    except ValueError as exc:
        logger.exception("%s forecast failed", service.model_name)
        raise HTTPException(
            status_code=500, detail=f"{service.model_name} produced an invalid forecast"
        ) from exc
    except Exception as exc:
        logger.exception("%s forecast failed", service.model_name)
        raise HTTPException(
            status_code=500, detail=f"Error getting {service.model_name} forecast"
        ) from exc


def _single_model_response(key: str, years_ahead: int) -> dict[str, Any]:
    service = SERVICES[key]
    forecast = _run_forecast(service, years_ahead)
    rows = [ROW_MODELS[key].model_validate(row).model_dump() for row in forecast]
    return {
        "model": key,
        "forecast": rows,
        "summary": calculate_model_summary(rows, service.model_name),
        "metadata": SingleMetadata(
            years_ahead=years_ahead, interval=service.uncertainty_info()
        ),
    }


# Endpoints are sync so that CPU-bound model work runs in FastAPI's threadpool
# instead of blocking the event loop.
@router.get("/all", response_model=AllForecastsResponse)
def get_all_forecasts(years_ahead: int = YearsAhead) -> dict[str, Any]:
    """Forecasts of all models; unavailable models return an empty list."""
    forecasts: dict[str, ForecastRows] = {}
    unavailable = []
    for key, service in SERVICES.items():
        try:
            forecasts[key] = _run_forecast(service, years_ahead)
        except HTTPException:
            forecasts[key] = []
            unavailable.append(key)
    if len(unavailable) == len(SERVICES):
        raise HTTPException(status_code=503, detail="No forecasting model is available")

    return {
        "forecasts": forecasts,
        "summaries": {
            key: calculate_model_summary(forecasts[key], service.model_name)
            for key, service in SERVICES.items()
        },
        "metadata": AllMetadata(
            years_ahead=years_ahead,
            models=list(SERVICES),
            intervals={key: s.uncertainty_info() for key, s in SERVICES.items()},
            unavailable_models=unavailable,
        ),
    }


@router.get("/prophet", response_model=SingleForecastResponse)
def get_prophet_forecast(years_ahead: int = YearsAhead) -> dict[str, Any]:
    """Prophet forecast."""
    return _single_model_response("prophet", years_ahead)


@router.get("/random-forest", response_model=SingleForecastResponse)
def get_random_forest_forecast(years_ahead: int = YearsAhead) -> dict[str, Any]:
    """Random Forest forecast."""
    return _single_model_response("random_forest", years_ahead)


@router.get("/xgboost", response_model=SingleForecastResponse)
def get_xgboost_forecast(years_ahead: int = YearsAhead) -> dict[str, Any]:
    """XGBoost forecast."""
    return _single_model_response("xgboost", years_ahead)
