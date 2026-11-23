"""
Main application file for the Germany Energy Forecast API.

Initializes the FastAPI app, configures CORS, loads/trains the forecasting
models (Prophet, Random Forest, XGBoost and the persistence benchmark) on
startup and registers the routes.
A model that fails to load is reported as unavailable instead of preventing
the API from starting.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.controllers.energy_controller import router as energy_router

# Service instances are re-exported for backward compatibility.
from app.dependencies import (  # noqa: F401  # pylint: disable=unused-import
    SERVICES,
    persistence_service,
    prophet_service,
    random_forest_service,
    xgboost_service,
)
from app.schemas import HealthResponse

logger = logging.getLogger(__name__)

API_VERSION = "1.2.0"


def load_models() -> None:
    """Load (or train) all forecasting models; failures are logged per model."""
    for service in SERVICES.values():
        try:
            service.load_data()
            service.train_model()
            logger.info("%s model ready", service.model_name)
        except Exception:  # pylint: disable=broad-except
            logger.exception("%s model failed to load", service.model_name)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Prepare all models before the API starts serving requests."""
    if not get_settings().skip_startup_training:
        load_models()
    yield


app = FastAPI(
    title="Germany Energy Forecast API", version=API_VERSION, lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _model_status() -> dict[str, bool]:
    return {key: service.is_ready for key, service in SERVICES.items()}


@app.get("/")
def root() -> dict[str, Any]:
    """Basic information about the API."""
    return {
        "message": "Germany Energy Forecast API",
        "status": "running",
        "target": "electricity_generation (TWh)",
        "models": list(SERVICES),
        "endpoints": {
            "prophet": "/forecast/prophet",
            "random_forest": "/forecast/random-forest",
            "xgboost": "/forecast/xgboost",
            "persistence": "/forecast/persistence",
            "all_models": "/forecast/all",
        },
    }


@app.get("/health", response_model=HealthResponse)
def health_check() -> dict[str, Any]:
    """Report API status and whether all models are loaded (always HTTP 200)."""
    models = _model_status()
    return {
        "status": "healthy" if all(models.values()) else "degraded",
        "models_loaded": all(models.values()),
        "models": models,
        "api_version": API_VERSION,
    }


@app.get("/health/live")
def liveness() -> dict[str, str]:
    """Liveness probe: the process is up and serving requests."""
    return {"status": "alive"}


@app.get("/health/ready")
def readiness() -> JSONResponse:
    """Readiness probe: HTTP 200 only when every model is loaded, else 503."""
    models = _model_status()
    ready = all(models.values())
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ready" if ready else "not_ready", "models": models},
    )


app.include_router(energy_router, prefix="/forecast")
