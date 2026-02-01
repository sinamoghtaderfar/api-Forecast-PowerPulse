# run.py
from app.services.forecast_service import ProphetForecastService, RandomForestForecastService, XGBoostForecastService
def main():
    prophet_service = ProphetForecastService(data_path="app/data/germany-energy-clean.csv")
    random_service = RandomForestForecastService(data_path="app/data/germany-energy-clean.csv")
    xGBoost_service = XGBoostForecastService(data_path="app/data/germany-energy-clean.csv")
    
    prophet_service.load_data()
    random_service.load_data()
    xGBoost_service.load_data()
    
    prophet_forecast_json = prophet_service.train_and_forecast(years_ahead=10)
    random_forecast_service = random_service.train_and_forecast(years_ahead=10)
    xGBoost_forecast_service = xGBoost_service.train_and_forecast(years_ahead=10)
    

    

if __name__ == "__main__":
    main()
