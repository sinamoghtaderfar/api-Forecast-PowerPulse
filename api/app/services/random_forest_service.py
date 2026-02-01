import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class RandomForestForecastService:
    def __init__(self, data_path="app/data/germany-energy-clean.csv"):
        # Path to the cleaned energy dataset
        self.data_path = data_path
        self.df = None
        self.model = None

    def load_data(self):
        # Load CSV data
        self.df = pd.read_csv(self.data_path)

        # Rename target column to 'y'
        self.df = self.df.rename(columns={'electricity_generation': 'y'})

        # Create derivative-based features
        # Year-to-year change in electricity generation
        self.df['y_diff'] = self.df['y'].diff().fillna(0)

        # Lag feature: electricity generation of the previous year
        self.df['y_lag1'] = self.df['y'].shift(1).bfill()

        # Debug information
        print("Data loaded successfully")
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()}")
        print("\nFirst 5 rows:")
        print(self.df.head())

    def train_and_forecast(self, years_ahead=10):
        if self.df is None:
            raise ValueError("Data not loaded! Call load_data() first.")

        # Define features (X) and target variable (y)
        X = self.df[
            ['year', 'population', 'gdp', 'renewables_share_energy', 'y_diff', 'y_lag1']
        ]
        y = self.df['y']

        # Train Random Forest regression model
        self.model = RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )
        self.model.fit(X, y)

        # Generate future years
        last_row = self.df.iloc[-1]
        future_years = np.arange(
            last_row['year'] + 1,
            last_row['year'] + 1 + years_ahead
        )

        # Assume annual growth rates for external indicators
        population_future = last_row['population'] * (1 + 0.01) ** (future_years - last_row['year'])
        gdp_future = last_row['gdp'] * (1 + 0.02) ** (future_years - last_row['year'])
        renewables_future = last_row['renewables_share_energy'] * (1 + 0.05) ** (future_years - last_row['year'])

        # Use the last observed derivative for future differences
        y_diff_future = np.full(years_ahead, last_row['y_diff'])

        # Generate lagged values recursively
        y_lag_future = np.zeros(years_ahead)
        y_lag_future[0] = last_row['y']
        for i in range(1, years_ahead):
            y_lag_future[i] = y_lag_future[i - 1] + y_diff_future[i - 1]

        # Build future feature DataFrame
        X_future = pd.DataFrame({
            'year': future_years,
            'population': population_future,
            'gdp': gdp_future,
            'renewables_share_energy': renewables_future,
            'y_diff': y_diff_future,
            'y_lag1': y_lag_future
        })

        # Predict future electricity generation
        y_pred = self.model.predict(X_future)

        # Prepare JSON-style output
        forecast_json = []
        for year, pred in zip(future_years, y_pred):
            forecast_json.append({
                "year": int(year),
                "forecast": float(pred)
            })

        print("\nRandom Forest Forecast JSON Output:")
        for item in forecast_json:
            print(item)

        return forecast_json
