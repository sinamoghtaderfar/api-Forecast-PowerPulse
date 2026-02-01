# app/controllers/energy_controller.py
from fastapi import APIRouter
from app.services.prophet_service import ProphetForecastService
from app.services.random_forest_service import RandomForestForecastService
from app.services.xgboost_service import XGBoostForecastService

router = APIRouter(prefix="/forecast", tags=["Forecast"])

# Initialize services
prophet_service = ProphetForecastService("app/data/germany-energy-clean.csv")
random_service = RandomForestForecastService("app/data/germany-energy-clean.csv")
xgboost_service = XGBoostForecastService("app/data/germany-energy-clean.csv")

# Load data once at startup
prophet_service.load_data()
random_service.load_data()
xgboost_service.load_data()

@router.get("/all")
def get_all_forecasts(years_ahead: int = 10):
    """Return forecasts from all models separately"""
    prophet = prophet_service.train_and_forecast(years_ahead)
    random_forest = random_service.train_and_forecast(years_ahead)
    xgboost = xgboost_service.train_and_forecast(years_ahead)
    return {
        "prophet": prophet,
        "random_forest": random_forest,
        "xgboost": xgboost
    }

@router.get("/combined")
def get_combined_forecast(years_ahead: int = 10):
    """average of all models"""
    prophet = prophet_service.train_and_forecast(years_ahead)
    random_forest = random_service.train_and_forecast(years_ahead)
    xgboost = xgboost_service.train_and_forecast(years_ahead)

    combined = []
    for i in range(years_ahead):
        year = prophet[i]["year"]
        avg_forecast = (prophet[i]["forecast"] + random_forest[i]["forecast"] + xgboost[i]["forecast"]) / 3
        combined.append({
            "year": year,
            "forecast": avg_forecast
        })

    return {"combined": combined}
