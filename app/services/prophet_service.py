import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# === PROPHET ===
class ProphetForecastService:
    def __init__(self, data_path="app/data/germany-energy-clean.csv"):
        self.data_path = data_path
        self.df = None
        self.model = None
        self.last_year = None
        self.last_values = None
    
    def load_data(self):
        try:
            self.df = pd.read_csv(self.data_path)
            required = ['year', 'electricity_generation', 'population', 'gdp', 'renewables_share_energy']
            for col in required:
                if col not in self.df.columns:
                    raise ValueError(f"Missing column: {col}")
            
            for col in required:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            self.df = self.df.ffill()
            self.df['ds'] = pd.to_datetime(self.df['year'].astype(str) + '-01-01')
            self.df = self.df.rename(columns={'electricity_generation': 'y'})
            
            self.last_year = int(self.df['year'].max())
            self.last_values = self.df.iloc[-1][['population', 'gdp', 'renewables_share_energy']].to_dict()
            
            print(f"Prophet data loaded. Last year: {self.last_year}")
        except Exception as e:
            print(f"Error loading Prophet data: {e}")
            raise
    
    def train_model(self):
        if self.df is None:
            self.load_data()
        
        self.model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            growth='linear',
            interval_width=0.95
        )
        
        self.model.add_regressor('population')
        self.model.add_regressor('gdp')
        self.model.add_regressor('renewables_share_energy')
        
        train_df = self.df[['ds', 'y', 'population', 'gdp', 'renewables_share_energy']]
        self.model.fit(train_df)
        print("Prophet model trained")
    
    def forecast(self, years_ahead=10):
        if self.model is None:
            self.train_model()
        
        future_dates = pd.date_range(
            start=pd.Timestamp(f"{self.last_year + 1}-01-01"),
            periods=years_ahead,
            freq='YS'
        )
        
        future = pd.DataFrame({'ds': future_dates})
        years_from_last = np.arange(1, years_ahead + 1)
        
        future['population'] = self.last_values['population'] * (1 + 0.01) ** years_from_last
        future['gdp'] = self.last_values['gdp'] * (1 + 0.02) ** years_from_last
        renew_growth = self.last_values['renewables_share_energy'] * (1 + 0.03) ** years_from_last
        future['renewables_share_energy'] = np.minimum(renew_growth, 100.0)
        
        forecast_result = self.model.predict(future)
        
        forecast_json = []
        for _, row in forecast_result.iterrows():
            forecast_json.append({
                "year": int(row['ds'].year),
                "forecast": float(row['yhat']),
                "lower": float(row['yhat_lower']),
                "upper": float(row['yhat_upper'])
            })
        
        return forecast_json
    
    def train_and_forecast(self, years_ahead=10):
        self.load_data()
        self.train_model()
        return self.forecast(years_ahead=years_ahead)


