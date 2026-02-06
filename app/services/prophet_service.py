import pandas as pd
from prophet import Prophet
import numpy as np


class ProphetForecastService:
    def __init__(self, data_path="app/data/germany-energy-clean.csv"):
        self.data_path = data_path
        self.df = None
        self.model = None
        self.last_year = None
        self.last_values = None  

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        
        self.df['ds'] = pd.to_datetime(self.df['year'].astype(str) + '-01-01')
        self.df = self.df.rename(columns={'electricity_generation': 'y'})
        
        self.last_year = int(self.df['year'].max())
        self.last_values = self.df.iloc[-1][['population', 'gdp', 'renewables_share_energy']].to_dict()
        
        print("Prophet data loaded")
        print(f"Shape: {self.df.shape}")
        print(f"Last year: {self.last_year}")
        print(f"Last regressors: {self.last_values}")

    def train_model(self):
        if self.df is None:
            self.load_data()

        if self.model is not None:
            print("Prophet model already trained")
            return

        print("Training Prophet model...")
        self.model = Prophet(
            yearly_seasonality=False,   
            weekly_seasonality=False,
            daily_seasonality=False,
            growth='linear'
        )
        
        self.model.add_regressor('population')
        self.model.add_regressor('gdp')
        self.model.add_regressor('renewables_share_energy')

        self.model.fit(self.df[['ds', 'y', 'population', 'gdp', 'renewables_share_energy']])
        print("Prophet model trained successfully")

    def forecast(self, years_ahead=10):
        if self.model is None:
            raise RuntimeError("Prophet model is not trained yet. Call train_model() first.")
        if self.df is None:
            self.load_data()

        future_dates = pd.date_range(
            start=pd.Timestamp(f"{self.last_year + 1}-01-01"),
            periods=years_ahead,
            freq='YS'  
        )
        
        future = pd.DataFrame({'ds': future_dates})

        years_from_last = np.arange(1, years_ahead + 1)
        
        future['population'] = self.last_values['population'] * (1 + 0.01) ** years_from_last
        future['gdp']        = self.last_values['gdp']        * (1 + 0.02) ** years_from_last
        future['renewables_share_energy'] = self.last_values['renewables_share_energy'] * (1 + 0.05) ** years_from_last

        forecast = self.model.predict(future)

        forecast_json = []
        for _, row in forecast.iterrows():
            forecast_json.append({
                "year": int(row['ds'].year),
                "forecast": float(row['yhat']),
                "lower": float(row['yhat_lower']),
                "upper": float(row['yhat_upper'])
            })

        print(f"\nProphet forecast for {years_ahead} years ahead:")
        for item in forecast_json:
            print(item)

        return forecast_json