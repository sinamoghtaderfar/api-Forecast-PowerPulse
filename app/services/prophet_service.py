"""
ProphetForecastService module.

This module defines the ProphetForecastService class, which loads Germany
energy data, trains a Prophet time series model, and generates multi-year forecasts
including uncertainty intervals. 

It supports additional regressors: population, GDP, and renewables_share_energy.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
import warnings
import os
import json
import hashlib
from pathlib import Path
from functools import lru_cache
warnings.filterwarnings("ignore")


# === PROPHET ===
class ProphetForecastService:
    """
    Prophet-based forecasting service for Germany energy data.

    Attributes:
        data_path (str): Path to the CSV data file.
        df (pd.DataFrame): Loaded dataset with features.
        model (Prophet): Trained Prophet model.
        last_year (int): Last year in the dataset.
        last_values (dict): Last available values for regressors.
    """
    def __init__(self, data_path="app/data/germany-energy-clean.csv"):
        """
        Initialize the ProphetForecastService.

        Args:
            data_path (str): Path to the CSV file containing historical energy data.
        """
        self.data_path = data_path
        self.df = None
        self.model = None
        self.last_year = None
        self.last_values = None
        models_dir = Path(__file__).resolve().parent.parent / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = models_dir / "prophet_model.json"

    def compute_data_hash(self):
        with open(self.data_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def load_data(self):
        """
        Load and preprocess historical energy data for Prophet.

        - Loads CSV data into a pandas DataFrame.
        - Validates required columns.
        - Converts columns to numeric.
        - Fills missing values.
        - Prepares 'ds' column for Prophet.
        - Stores last year and last values for regressors.

        Raises:
            Exception: If data loading or processing fails.
        """
        try:
            self.df = pd.read_csv(self.data_path)
            required = [
                "year",
                "electricity_generation",
                "population",
                "gdp",
                "renewables_share_energy",
            ]
            for col in required:
                if col not in self.df.columns:
                    raise ValueError(f"Missing column: {col}")

            for col in required:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

            self.df = self.df.ffill()
            self.df["ds"] = pd.to_datetime(self.df["year"].astype(str) + "-01-01")
            self.df = self.df.rename(columns={"electricity_generation": "y"})

            self.last_year = int(self.df["year"].max())
            self.last_values = self.df.iloc[-1][
                ["population", "gdp", "renewables_share_energy"]
            ].to_dict()

            print(f"Prophet data loaded. Last year: {self.last_year}")
        except Exception as e:
            print(f"Error loading Prophet data: {e}")
            raise

    def train_model(self):
        """
        Train the Prophet model with additional regressors.

        - Adds population, GDP, and renewables_share_energy as external regressors.
        - Fits the Prophet model on the historical data.
        """
        if self.df is None:
            self.load_data()

        current_hash = self.compute_data_hash()
        print(f"Model path being used: {self.model_path}")
        print(f"Exists? {os.path.exists(self.model_path)}")
        if os.path.exists(self.model_path):
            print(f"File size: {os.path.getsize(self.model_path)} bytes")
            try:
                with open(self.model_path, 'r') as f:
                    saved = json.load(f)
                if saved['data_hash'] == current_hash:
                    self.model = model_from_json(saved['model_json'])
                    print("Loaded saved Prophet model")
                    return
            except Exception as e:
                print(f"Error loading saved Prophet model: {e}")

        self.model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            growth="linear",
            interval_width=0.95,
        )

        self.model.add_regressor("population")
        self.model.add_regressor("gdp")
        self.model.add_regressor("renewables_share_energy")

        train_df = self.df[["ds", "y", "population", "gdp", "renewables_share_energy"]]
        self.model.fit(train_df)
        print("Prophet model trained")

        # Save the model
        to_save = {
            'model_json': model_to_json(self.model),
            'data_hash': current_hash
        }
        with open(self.model_path, 'w') as f:
            json.dump(to_save, f)
        print("Saved trained Prophet model")
    @lru_cache(maxsize=32)
    def forecast(self, years_ahead=10):
        """
        Generate multi-year forecasts using the trained Prophet model.

        - Projects future dates.
        - Estimates future values for regressors using simple growth assumptions.
        - Returns predictions with uncertainty intervals.

        Args:
            years_ahead (int): Number of future years to forecast.

        Returns:
            list[dict]: Forecast results for each year, each containing:
                        'year', 'forecast', 'lower', 'upper'.
        """
        if self.model is None:
            self.train_model()

        future_dates = pd.date_range(
            start=pd.Timestamp(f"{self.last_year + 1}-01-01"),
            periods=years_ahead,
            freq="YS",
        )

        future = pd.DataFrame({"ds": future_dates})
        years_from_last = np.arange(1, years_ahead + 1)

        future["population"] = (
            self.last_values["population"] * (1 + 0.01) ** years_from_last
        )
        future["gdp"] = self.last_values["gdp"] * (1 + 0.02) ** years_from_last
        renew_growth = (
            self.last_values["renewables_share_energy"] * (1 + 0.03) ** years_from_last
        )
        future["renewables_share_energy"] = np.minimum(renew_growth, 100.0)

        forecast_result = self.model.predict(future)

        forecast_json = []
        for _, row in forecast_result.iterrows():
            forecast_json.append(
                {
                    "year": int(row["ds"].year),
                    "forecast": float(row["yhat"]),
                    "lower": float(row["yhat_lower"]),
                    "upper": float(row["yhat_upper"]),
                }
            )

        return forecast_json

    def train_and_forecast(self, years_ahead=10):
        """
        Convenience method to load data, train the model, and generate forecasts.

        Args:
            years_ahead (int): Number of future years to forecast.

        Returns:
            list[dict]: Forecast results for each year.
        """
        self.load_data()
        self.train_model()
        return self.forecast(years_ahead=years_ahead)