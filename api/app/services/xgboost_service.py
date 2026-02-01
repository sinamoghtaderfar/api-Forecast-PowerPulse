import pandas as pd
import numpy as np
from xgboost import XGBRegressor

class XGBoostForecastService:
    def __init__(self, data_path="app/data/germany-energy-clean.csv"):
        self.data_path = data_path
        self.df = None
        self.model = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.rename(columns={'electricity_generation': 'y'})

        # Lag features
        self.df['y_lag1'] = self.df['y'].shift(1)
        self.df['y_diff1'] = self.df['y'] - self.df['y_lag1']
        self.df['y_diff2'] = self.df['y_diff1'].diff()

        # Derivatives of exogenous variables
        self.df['gdp_growth'] = self.df['gdp'].diff()
        self.df['population_growth'] = self.df['population'].diff()
        self.df['renewables_growth'] = self.df['renewables_share_energy'].diff()

        # Clean NaNs
        self.df = self.df.bfill()

        print("XGBoost data loaded with derivative features")

    def train_and_forecast(self, years_ahead=10):
        features = [
            'year',
            'y_lag1',
            'y_diff1',
            'y_diff2',
            'gdp_growth',
            'population_growth',
            'renewables_growth'
        ]

        X = self.df[features]
        y = self.df['y_diff1']  # 🔥 Train on change, not level

        self.model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        self.model.fit(X, y)

        last = self.df.iloc[-1]
        current_y = last['y']

        forecasts = []

        for i in range(1, years_ahead + 1):
            future_year = int(last['year'] + i)

            X_future = pd.DataFrame([{
                'year': future_year,
                'y_lag1': current_y,
                'y_diff1': last['y_diff1'],
                'y_diff2': last['y_diff2'],
                'gdp_growth': last['gdp_growth'],
                'population_growth': last['population_growth'],
                'renewables_growth': last['renewables_growth']
            }])

            delta_y = self.model.predict(X_future)[0]
            current_y += delta_y

            forecasts.append({
                "year": future_year,
                "forecast": float(current_y)
            })

        print("\nXGBoost Forecast (Derivative-based):")
        for f in forecasts:
            print(f)

        return forecasts
