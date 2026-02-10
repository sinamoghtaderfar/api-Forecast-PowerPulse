from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any

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
            
            first_year = forecast_data[0]['year']
            last_year = forecast_data[-1]['year']
            first_value = forecast_data[0]['forecast']
            last_value = forecast_data[-1]['forecast']
            total_growth = ((last_value / first_value) - 1) * 100
            
            # Calculate average uncertainty width (if available)
            avg_uncertainty = None
            if 'lower' in forecast_data[0] and 'upper' in forecast_data[0]:
                uncertainties = [(item['upper'] - item['lower']) for item in forecast_data]
                avg_uncertainty = sum(uncertainties) / len(uncertainties)
            
            return {
                "model": model_name,
                "period": f"{first_year}-{last_year}",
                "start_value": first_value,
                "end_value": last_value,
                "total_growth_percent": round(total_growth, 2),
                "avg_annual_growth": round(total_growth / years_ahead, 2),
                "avg_uncertainty_width": round(avg_uncertainty, 2) if avg_uncertainty else None
            }
        
        summaries = {
            "prophet": calculate_model_summary(prophet_forecast, "Prophet"),
            "random_forest": calculate_model_summary(random_forest_forecast, "Random Forest"),
            "xgboost": calculate_model_summary(xgboost_forecast, "XGBoost")
        }
        
        return {
            "forecasts": {
                "prophet": prophet_forecast,
                "random_forest": random_forest_forecast,
                "xgboost": xgboost_forecast
            },
            "summaries": summaries,
            "metadata": {
                "years_ahead": years_ahead,
                "models": ["prophet", "random_forest", "xgboost"],
                "units": {
                    "forecast": "electricity_generation",
                    "uncertainty": "95% confidence interval"
                }
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting forecasts: {str(e)}")

@router.get("/combined")
async def get_combined_forecast(years_ahead: int = 10) -> Dict[str, Any]:
    """
    Get ensemble forecast (weighted average of all models)
    """
    from app.main import prophet_service, random_forest_service, xgboost_service
    
    try:
        # Get forecasts
        prophet_preds = prophet_service.forecast(years_ahead=years_ahead)
        rf_preds = random_forest_service.forecast(years_ahead=years_ahead)
        xgb_preds = xgboost_service.forecast(years_ahead=years_ahead)
        
        # Validate data
        if not all(len(preds) == years_ahead for preds in [prophet_preds, rf_preds, xgb_preds]):
            raise ValueError(f"All forecasts must have exactly {years_ahead} predictions")
        
        combined_forecast = []
        uncertainty_ranges = []
        
        for i in range(years_ahead):
            year = prophet_preds[i]["year"]
            
            # Extract values (ensure they exist)
            p_forecast = prophet_preds[i].get("forecast", 0)
            p_lower = prophet_preds[i].get("lower", p_forecast)
            p_upper = prophet_preds[i].get("upper", p_forecast)
            
            rf_forecast = rf_preds[i].get("forecast", 0)
            rf_lower = rf_preds[i].get("lower", rf_forecast)
            rf_upper = rf_preds[i].get("upper", rf_forecast)
            
            xgb_forecast = xgb_preds[i].get("forecast", 0)
            xgb_lower = xgb_preds[i].get("lower", xgb_forecast)
            xgb_upper = xgb_preds[i].get("upper", xgb_forecast)
            
            # Weighted average (you can adjust weights based on model performance)
            weights = {"prophet": 0.4, "random_forest": 0.3, "xgboost": 0.3}
            
            avg_forecast = (
                weights["prophet"] * p_forecast +
                weights["random_forest"] * rf_forecast +
                weights["xgboost"] * xgb_forecast
            )
            
            # Combine uncertainty ranges
            avg_lower = (
                weights["prophet"] * p_lower +
                weights["random_forest"] * rf_lower +
                weights["xgboost"] * xgb_lower
            )
            
            avg_upper = (
                weights["prophet"] * p_upper +
                weights["random_forest"] * rf_upper +
                weights["xgboost"] * xgb_upper
            )
            
            combined_forecast.append({
                "year": int(year),
                "forecast": float(avg_forecast),
                "lower": float(avg_lower),
                "upper": float(avg_upper),
                "model_contributions": {
                    "prophet": float(p_forecast),
                    "random_forest": float(rf_forecast),
                    "xgboost": float(xgb_forecast)
                }
            })
            
            uncertainty_ranges.append(float(avg_upper - avg_lower))
        
        # Calculate ensemble statistics
        if combined_forecast:
            first_value = combined_forecast[0]['forecast']
            last_value = combined_forecast[-1]['forecast']
            total_growth = ((last_value / first_value) - 1) * 100
            avg_uncertainty = sum(uncertainty_ranges) / len(uncertainty_ranges)
        
        return {
            "ensemble_forecast": combined_forecast,
            "ensemble_statistics": {
                "years_ahead": years_ahead,
                "total_growth_percent": round(total_growth, 2),
                "avg_annual_growth": round(total_growth / years_ahead, 2),
                "avg_uncertainty_width": round(avg_uncertainty, 2),
                "model_weights": {
                    "prophet": 0.4,
                    "random_forest": 0.3,
                    "xgboost": 0.3
                }
            },
            "metadata": {
                "description": "Weighted ensemble of Prophet, Random Forest, and XGBoost models",
                "recommendation": "Use ensemble forecast for more robust predictions"
            }
        }
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating ensemble forecast: {str(e)}")

@router.get("/consensus")
async def get_consensus_forecast(years_ahead: int = 10) -> Dict[str, Any]:
    """
    Get consensus forecast (median of all models) - more robust than average
    """
    from app.main import prophet_service, random_forest_service, xgboost_service
    import numpy as np
    
    try:
        prophet_preds = prophet_service.forecast(years_ahead=years_ahead)
        rf_preds = random_forest_service.forecast(years_ahead=years_ahead)
        xgb_preds = xgboost_service.forecast(years_ahead=years_ahead)
        
        consensus_forecast = []
        
        for i in range(years_ahead):
            year = prophet_preds[i]["year"]
            
            # Get forecasts from all models
            forecasts = [
                prophet_preds[i].get("forecast", 0),
                rf_preds[i].get("forecast", 0),
                xgb_preds[i].get("forecast", 0)
            ]
            
            # Calculate median (more robust than mean)
            median_forecast = float(np.median(forecasts))
            
            # Get confidence intervals
            lower_bounds = [
                prophet_preds[i].get("lower", forecasts[0]),
                rf_preds[i].get("lower", forecasts[1]),
                xgb_preds[i].get("lower", forecasts[2])
            ]
            
            upper_bounds = [
                prophet_preds[i].get("upper", forecasts[0]),
                rf_preds[i].get("upper", forecasts[1]),
                xgb_preds[i].get("upper", forecasts[2])
            ]
            
            median_lower = float(np.median(lower_bounds))
            median_upper = float(np.median(upper_bounds))
            
            consensus_forecast.append({
                "year": int(year),
                "forecast": median_forecast,
                "lower": median_lower,
                "upper": median_upper,
                "model_values": {
                    "prophet": float(forecasts[0]),
                    "random_forest": float(forecasts[1]),
                    "xgboost": float(forecasts[2])
                }
            })
        
        return {
            "consensus_forecast": consensus_forecast,
            "method": "median_of_models",
            "advantages": [
                "Robust to outliers",
                "Less sensitive to extreme predictions",
                "Better statistical properties"
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating consensus forecast: {str(e)}")