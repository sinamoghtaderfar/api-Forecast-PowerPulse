"""Forecast endpoints for the Germany Energy Forecast API."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.dependencies import SERVICES
from app.services.base import MAX_YEARS_AHEAD, BaseForecastService, ForecastRows

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Forecast"])

YearsAhead = Query(10, ge=1, le=MAX_YEARS_AHEAD, description="Years to forecast")


def calculate_model_summary(
    forecast_data: ForecastRows, model_name: str
) -> dict[str, Any]:
    """Summary statistics (growth, uncertainty) for one model's forecast."""
    if not forecast_data:
        return {}

    first, last = forecast_data[0], forecast_data[-1]
    first_value, last_value = first["forecast"], last["forecast"]
    total_growth = ((last_value / first_value) - 1) * 100 if first_value else 0.0

    avg_uncertainty = None
    if "lower" in first and "upper" in first:
        widths = [item["upper"] - item["lower"] for item in forecast_data]
        avg_uncertainty = sum(widths) / len(widths)

    return {
        "model": model_name,
        "period": f"{first['year']}-{last['year']}",
        "start_value": first_value,
        "end_value": last_value,
        "total_growth_percent": round(total_growth, 2),
        "avg_annual_growth": round(total_growth / len(forecast_data), 2),
        "avg_uncertainty_width": (
            round(avg_uncertainty, 2) if avg_uncertainty is not None else None
        ),
    }


def _run_forecast(service: BaseForecastService, years_ahead: int) -> ForecastRows:
    try:
        return service.forecast(years_ahead=years_ahead)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("%s forecast failed", service.model_name)
        raise HTTPException(
            status_code=500, detail=f"Error getting {service.model_name} forecast"
        ) from exc


def _single_model_response(key: str, years_ahead: int) -> dict[str, Any]:
    service = SERVICES[key]
    forecast = _run_forecast(service, years_ahead)
    return {
        "model": key,
        "forecast": forecast,
        "summary": calculate_model_summary(forecast, service.model_name),
        "metadata": {"years_ahead": years_ahead},
    }


# Endpoints are sync so that CPU-bound model work runs in FastAPI's threadpool
# instead of blocking the event loop.
@router.get("/all")
def get_all_forecasts(years_ahead: int = YearsAhead) -> dict[str, Any]:
    """Get forecasts from all three models with confidence intervals."""
    forecasts = {
        key: _run_forecast(service, years_ahead) for key, service in SERVICES.items()
    }
    summaries = {
        key: calculate_model_summary(forecasts[key], service.model_name)
        for key, service in SERVICES.items()
    }
    return {
        "forecasts": forecasts,
        "summaries": summaries,
        "metadata": {
            "years_ahead": years_ahead,
            "models": list(SERVICES),
            "units": {
                "forecast": "electricity_generation",
                "uncertainty": "95% confidence interval",
            },
        },
    }


@router.get("/prophet")
def get_prophet_forecast(years_ahead: int = YearsAhead) -> dict[str, Any]:
    """Prophet forecast."""
    return _single_model_response("prophet", years_ahead)


@router.get("/random-forest")
def get_random_forest_forecast(years_ahead: int = YearsAhead) -> dict[str, Any]:
    """Random Forest forecast."""
    return _single_model_response("random_forest", years_ahead)


@router.get("/xgboost")
def get_xgboost_forecast(years_ahead: int = YearsAhead) -> dict[str, Any]:
    """XGBoost forecast."""
    return _single_model_response("xgboost", years_ahead)
