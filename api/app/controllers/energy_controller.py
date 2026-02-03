from fastapi import APIRouter
from typing import Dict, List

router = APIRouter(tags=["Forecast"])

@router.get("/all")
async def get_all_forecasts(years_ahead: int = 10) -> Dict[str, List[Dict]]:

    from app.main import prophet_service, random_forest_service, xgboost_service

    return {
        "prophet": prophet_service.forecast(years_ahead),
        "random_forest": random_forest_service.forecast(years_ahead),
        "xgboost": xgboost_service.forecast(years_ahead),
    }


@router.get("/combined")
async def get_combined_forecast(years_ahead: int = 10) -> Dict[str, List[Dict]]:

    from app.main import prophet_service, random_forest_service, xgboost_service

    prophet_preds = prophet_service.forecast(years_ahead)
    rf_preds = random_forest_service.forecast(years_ahead)
    xgb_preds = xgboost_service.forecast(years_ahead)

    if len(prophet_preds) != years_ahead or len(rf_preds) != years_ahead or len(xgb_preds) != years_ahead:
        raise ValueError("تعداد پیش‌بینی‌های مدل‌ها با years_ahead مطابقت ندارد")

    combined = []
    for i in range(years_ahead):
        year = prophet_preds[i]["year"]
        avg_forecast = (
            prophet_preds[i]["forecast"] +
            rf_preds[i]["forecast"] +
            xgb_preds[i]["forecast"]
        ) / 3

        combined.append({
            "year": int(year),
            "forecast": float(avg_forecast)
        })

    return {"combined": combined}