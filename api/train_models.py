from app.services.forecast_service import ProphetForecastService, RandomForestForecastService, XGBoostForecastService

def main():
    prophet_service = ProphetForecastService("app/data/germany-energy-clean.csv")
    random_service = RandomForestForecastService("app/data/germany-energy-clean.csv")
    xGBoost_service = XGBoostForecastService("app/data/germany-energy-clean.csv")
    
    # Load data
    prophet_service.load_data()
    random_service.load_data()
    xGBoost_service.load_data()
    
    # Train and forecast
    prophet_forecast_json = prophet_service.train_and_forecast(years_ahead=10)
    random_forecast_json = random_service.train_and_forecast(years_ahead=10)
    xGBoost_forecast_json = xGBoost_service.train_and_forecast(years_ahead=10)
    
    print("\nProphet:", prophet_forecast_json)
    print("\nRandomForest:", random_forecast_json)
    print("\nXGBoost:", xGBoost_forecast_json)

if __name__ == "__main__":
    main()
