import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np


class RandomForestForecastService:
    def __init__(self, data_path="app/data/germany-energy-clean.csv"):
        self.data_path = data_path
        self.df = None
        self.model = None
        self.last_row = None  

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        
        self.df = self.df.rename(columns={'electricity_generation': 'y'})
        
        self.df['y_diff'] = self.df['y'].diff().fillna(0)
        self.df['y_lag1'] = self.df['y'].shift(1).bfill()
        
        self.last_row = self.df.iloc[-1].copy()
        
        print("Random Forest data loaded successfully")
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()}")
        print(f"Last year: {self.last_row['year']}")
        print("\nFirst 5 rows:")
        print(self.df.head())

    def train_model(self):
        if self.df is None:
            self.load_data()

        if self.model is not None:
            print("Random Forest model already trained")
            return

        print("Training Random Forest model...")

        X = self.df[
            ['year', 'population', 'gdp', 'renewables_share_energy', 'y_diff', 'y_lag1']
        ]
        y = self.df['y']

        self.model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1   
        )
        
        self.model.fit(X, y)
        
        print(f"Random Forest trained successfully – {len(self.model.estimators_)} trees")

    def forecast(self, years_ahead=10):
        if self.model is None:
            raise RuntimeError("Model not trained yet. Call train_model() first.")
        if self.df is None:
            self.load_data()

        future_years = np.arange(
            self.last_row['year'] + 1,
            self.last_row['year'] + 1 + years_ahead
        )

        years_ahead_arr = np.arange(1, years_ahead + 1)
        
        population_future = self.last_row['population'] * (1 + 0.01) ** years_ahead_arr
        gdp_future = self.last_row['gdp'] * (1 + 0.02) ** years_ahead_arr
        renewables_future = self.last_row['renewables_share_energy'] * (1 + 0.05) ** years_ahead_arr

        y_diff_future = np.full(years_ahead, self.last_row['y_diff'])

        y_lag_future = np.zeros(years_ahead)
        y_lag_future[0] = self.last_row['y']
        for i in range(1, years_ahead):
            y_lag_future[i] = y_lag_future[i - 1] + y_diff_future[i - 1]

        X_future = pd.DataFrame({
            'year': future_years,
            'population': population_future,
            'gdp': gdp_future,
            'renewables_share_energy': renewables_future,
            'y_diff': y_diff_future,
            'y_lag1': y_lag_future
        })

        y_pred = self.model.predict(X_future)

        #JSON
        forecast_json = [
            {
                "year": int(year),
                "forecast": float(pred)
            }
            for year, pred in zip(future_years, y_pred)
        ]

        print("\nRandom Forest Forecast JSON Output:")
        for item in forecast_json:
            print(item)

        return forecast_json