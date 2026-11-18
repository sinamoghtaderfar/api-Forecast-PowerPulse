"""
Main application file for the Germany Energy Forecast API.

Initializes the FastAPI app, configures CORS, loads/trains the forecasting
models (Prophet, Random Forest, XGBoost) on startup and registers the routes.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.controllers.energy_controller import router as energy_router

# Service instances are re-exported for backward compatibility.
from app.dependencies import (  # noqa: F401  # pylint: disable=unused-import
    SERVICES,
    prophet_service,
    random_forest_service,
    xgboost_service,
)

logger = logging.getLogger(__name__)

DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"


def _cors_origins() -> list[str]:
    raw = os.getenv("POWERPULSE_CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def load_models() -> None:
    """Load (or train) all forecasting models."""
    for service in SERVICES.values():
        service.load_data()
        service.train_model()
        logger.info("%s model ready", service.model_name)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Prepare all models before the API starts serving requests."""
    if os.getenv("POWERPULSE_SKIP_STARTUP_TRAINING") != "1":
        load_models()
    yield


app = FastAPI(title="Germany Energy Forecast API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, Any]:
    """Basic information about the API."""
    return {
        "message": "Germany Energy Forecast API",
        "status": "running",
        "models": ["prophet", "random_forest", "xgboost"],
        "endpoints": {
            "prophet": "/forecast/prophet",
            "random_forest": "/forecast/random-forest",
            "xgboost": "/forecast/xgboost",
            "all_models": "/forecast/all",
        },
    }


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Report API status and whether all models are loaded."""
    models = {key: service.is_ready for key, service in SERVICES.items()}
    return {
        "status": "healthy" if all(models.values()) else "degraded",
        "models_loaded": all(models.values()),
        "models": models,
        "api_version": "1.0.0",
    }


app.include_router(energy_router, prefix="/forecast")
