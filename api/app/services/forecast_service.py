from app.services.prophet_service import ProphetForecastService
from app.services.random_forest_service import RandomForestForecastService
from app.services.xgboost_service import XGBoostForecastService

class EnergyForecastService:
    def __init__(self):
        self.prophet = ProphetForecastService()
        self.rf = RandomForestForecastService()
        self.xgb = XGBoostForecastService()
    
    def prophet_forecast(self, years_ahead=10):
        return self.prophet.train_and_forecast(years_ahead)
    
    def random_forest_forecast(self, years_ahead=10):
        self.rf.train_model()
        return self.rf.predict(years_ahead)
    
    def xgboost_forecast(self, years_ahead=10):
        self.xgb.train_model()
        return self.xgb.predict(years_ahead)
