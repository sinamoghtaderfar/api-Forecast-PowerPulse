from app.services.forecast_service import ProphetForecastService, RandomForestForecastService, XGBoostForecastService

def main():
    prophet_service = ProphetForecastService()
    random_service = RandomForestForecastService()
    xgboost_service = XGBoostForecastService()
    
    # Load data
    prophet_service.load_data()
    random_service.load_data()
    xgboost_service.load_data()
    
    # Train and forecast
    prophet_forecast_json = prophet_service.train_and_forecast(years_ahead=10)
    random_forecast_json = random_service.train_and_forecast(years_ahead=10)
    xgboost_forecast_json = xgboost_service.train_and_forecast(years_ahead=10)
    
    print("\n" + "="*60)
    print("PROPHET FORECAST (10 years):")
    print("="*60)
    for item in prophet_forecast_json:
        print(f"  Year {item['year']}: {item['forecast']:,.0f} "
              f"(Range: {item.get('lower', 'N/A'):,.0f} - {item.get('upper', 'N/A'):,.0f})")
    
    print("\n" + "="*60)
    print("RANDOM FOREST FORECAST (10 years):")
    print("="*60)
    for item in random_forecast_json:
        print(f"  Year {item['year']}: {item['forecast']:,.0f} "
              f"(Range: {item.get('lower', 'N/A'):,.0f} - {item.get('upper', 'N/A'):,.0f})")
    
    print("\n" + "="*60)
    print("XGBOOST FORECAST (10 years):")
    print("="*60)
    for item in xgboost_forecast_json:
        print(f"  Year {item['year']}: {item['forecast']:,.0f} "
              f"(Range: {item.get('lower', 'N/A'):,.0f} - {item.get('upper', 'N/A'):,.0f})")
    
    print("\n" + "="*60)
    print("FORECAST SUMMARY")
    print("="*60)
    
    def calculate_summary(data, name):
        if data:
            first = data[0]['forecast']
            last = data[-1]['forecast']
            growth = ((last / first) - 1) * 100
            return f"{name}: {first:,.0f} → {last:,.0f} ({growth:+.1f}%)"
        return f"{name}: No data"
    
    print(calculate_summary(prophet_forecast_json, "Prophet"))
    print(calculate_summary(random_forecast_json, "Random Forest"))
    print(calculate_summary(xgboost_forecast_json, "XGBoost"))

if __name__ == "__main__":
    main()