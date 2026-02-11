from fastapi import APIRouter, HTTPException
from typing import Dict, Any

router = APIRouter(tags=["Forecast"])


@router.get("/all")
async def get_all_forecasts(years_ahead: int = 10) -> Dict[str, Any]:
    """
    Get forecasts from all three models with confidence intervals
    """
    from app.main import prophet_service, random_forest_service, xgboost_service

    try:
        # Get forecasts from all models
        prophet_forecast = prophet_service.forecast(years_ahead=years_ahead)
        random_forest_forecast = random_forest_service.forecast(years_ahead=years_ahead)
        xgboost_forecast = xgboost_service.forecast(years_ahead=years_ahead)

        # Calculate summary statistics
        def calculate_model_summary(forecast_data, model_name):
            if not forecast_data:
                return {}

            first_year = forecast_data[0]["year"]
            last_year = forecast_data[-1]["year"]
            first_value = forecast_data[0]["forecast"]
            last_value = forecast_data[-1]["forecast"]
            total_growth = ((last_value / first_value) - 1) * 100

            # Calculate average uncertainty width (if available)
            avg_uncertainty = None
            if "lower" in forecast_data[0] and "upper" in forecast_data[0]:
                uncertainties = [
                    (item["upper"] - item["lower"]) for item in forecast_data
                ]
                avg_uncertainty = sum(uncertainties) / len(uncertainties)

            return {
                "model": model_name,
                "period": f"{first_year}-{last_year}",
                "start_value": first_value,
                "end_value": last_value,
                "total_growth_percent": round(total_growth, 2),
                "avg_annual_growth": round(total_growth / years_ahead, 2),
                "avg_uncertainty_width": (
                    round(avg_uncertainty, 2) if avg_uncertainty else None
                ),
            }

        summaries = {
            "prophet": calculate_model_summary(prophet_forecast, "Prophet"),
            "random_forest": calculate_model_summary(
                random_forest_forecast, "Random Forest"
            ),
            "xgboost": calculate_model_summary(xgboost_forecast, "XGBoost"),
        }

        return {
            "forecasts": {
                "prophet": prophet_forecast,
                "random_forest": random_forest_forecast,
                "xgboost": xgboost_forecast,
            },
            "summaries": summaries,
            "metadata": {
                "years_ahead": years_ahead,
                "models": ["prophet", "random_forest", "xgboost"],
                "units": {
                    "forecast": "electricity_generation",
                    "uncertainty": "95% confidence interval",
                },
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting forecasts: {str(e)}"
        )