"""Shared forecasting service instances used by the API."""

from app.services.persistence_service import PersistenceForecastService
from app.services.prophet_service import ProphetForecastService
from app.services.random_forest_service import RandomForestForecastService
from app.services.xgboost_service import XGBoostForecastService

prophet_service = ProphetForecastService()
random_forest_service = RandomForestForecastService()
xgboost_service = XGBoostForecastService()
persistence_service = PersistenceForecastService()

SERVICES = {
    "prophet": prophet_service,
    "random_forest": random_forest_service,
    "xgboost": xgboost_service,
    "persistence": persistence_service,
}
