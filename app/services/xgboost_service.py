import pandas as pd
import numpy as np
from xgboost import XGBRegressor


class XGBoostForecastService:
    def __init__(self, data_path="app/data/germany-energy-clean.csv"):
        self.data_path = data_path
        self.df = None
        self.model = None
        self.last_row = None
        self.last_y = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.rename(columns={'electricity_generation': 'y'})

        self.df['y_lag1'] = self.df['y'].shift(1)
        self.df['y_diff1'] = self.df['y'] - self.df['y_lag1']
        self.df['y_diff2'] = self.df['y_diff1'].diff()

        self.df['gdp_growth'] = self.df['gdp'].diff()
        self.df['population_growth'] = self.df['population'].diff()
        self.df['renewables_growth'] = self.df['renewables_share_energy'].diff()

        self.df = self.df.ffill().bfill()

        self.last_row = self.df.iloc[-1].copy()
        self.last_y = float(self.last_row['y'])

        print("XGBoost data prepared")
        print(f"Shape: {self.df.shape}")
        print(f"Last year: {self.last_row['year']}")
        print(f"Last y: {self.last_y}")

    def train_model(self):
        """آموزش مدل – فقط یک بار در startup"""
        if self.df is None:
            self.load_data()

        if self.model is not None:
            print("XGBoost model already trained")
            return

        print("Training XGBoost model...")

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
        y = self.df['y_diff1']  

        self.model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            objective='reg:squarederror'
        )

        self.model.fit(X, y)

        print("XGBoost training completed")

    def forecast(self, years_ahead=10):
        if self.model is None:
            raise RuntimeError("XGBoost model not trained. Call train_model() first.")
        if self.df is None:
            self.load_data()

        current_y = self.last_y
        current_row = self.last_row.copy()

        forecasts = []

        for i in range(1, years_ahead + 1):
            future_year = int(current_row['year'] + i)

            X_future = pd.DataFrame([{
                'year': future_year,
                'y_lag1': current_y,
                'y_diff1': current_row['y_diff1'],
                'y_diff2': current_row['y_diff2'],
                'gdp_growth': current_row['gdp_growth'],
                'population_growth': current_row['population_growth'],
                'renewables_growth': current_row['renewables_growth']
            }])

            delta_y = self.model.predict(X_future)[0]
            current_y += delta_y

            # به‌روزرسانی برای گام بعدی
            current_row['y_diff2'] = current_row['y_diff1']
            current_row['y_diff1'] = delta_y
            current_row['year'] = future_year

            forecasts.append({
                "year": future_year,
                "forecast": float(current_y)
            })

        print("\nXGBoost Forecast:")
        for f in forecasts:
            print(f)

        return forecasts