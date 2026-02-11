"""
XGBoostForecastService module.

This module defines the XGBoostForecastService class, which is responsible
for loading Germany energy data, training an XGBoost regression model, and
generating multi-year energy forecasts with uncertainty estimation.

The class uses advanced features such as lag features, rolling statistics,
interaction features, and Monte Carlo simulations to provide robust
forecasts.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")


class XGBoostForecastService:
    """
    XGBoost-based forecasting service for Germany energy data.

    Attributes:
        data_path (str): Path to the CSV data file.
        df (pd.DataFrame): Loaded dataset with features.
        model (XGBRegressor): Trained XGBoost regression model.
        last_row (pd.Series): Last row of the dataset.
        last_year (int): Last year in the dataset.
        feature_names (list): Names of features used for training.
        avg_growth_rates (dict): Average historical growth rates for features.
    """
    
    def __init__(self, data_path="app/data/germany-energy-clean.csv"):
        """
        Initialize the XGBoostForecastService.

        Args:
            data_path (str): Path to the CSV file containing historical energy data.
        """
        self.data_path = data_path
        self.df = None
        self.model = None
        self.last_row = None
        self.last_year = None
        self.feature_names = None
        self.avg_growth_rates = {}  # store average historical growth rates

    def load_data(self):
        """
        Load and preprocess the historical energy dataset.

        This method:
        - Loads CSV data into a pandas DataFrame.
        - Validates required columns.
        - Converts columns to numeric.
        - Fills missing values and renames the target column.
        - Calculates average historical growth rates.
        - Creates lag features, rolling statistics, and interaction features.
        - Stores the last row and year for future forecasts.

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
            self.df = self.df.rename(columns={"electricity_generation": "y"})

            # Calculate average historical growth rates
            self.avg_growth_rates = {
                "y_growth": self.df["y"].pct_change().mean(),
                "gdp_growth": self.df["gdp"].pct_change().mean(),
                "population_growth": self.df["population"].pct_change().mean(),
                "renewables_growth": self.df["renewables_share_energy"]
                .pct_change()
                .mean(),
            }

            print(f"Average growth rates: {self.avg_growth_rates}")

            # Enhanced features – using average growth rates
            for lag in [1, 2, 3]:
                self.df[f"y_lag{lag}"] = self.df["y"].shift(lag)

            self.df["y_diff1"] = self.df["y"].diff()

            # Use average historical growth instead of last observed growth
            self.df["gdp_growth_feature"] = self.avg_growth_rates["gdp_growth"]
            self.df["population_growth_feature"] = self.avg_growth_rates[
                "population_growth"
            ]
            self.df["renewables_growth_feature"] = self.avg_growth_rates[
                "renewables_growth"
            ]

            # Rolling statistics
            for window in [3, 5]:
                self.df[f"y_rolling_mean_{window}"] = (
                    self.df["y"].shift(1).rolling(window=window, min_periods=1).mean()
                )
                self.df[f"y_rolling_std_{window}"] = (
                    self.df["y"].shift(1).rolling(window=window, min_periods=1).std()
                )

            # Time-based features
            self.df["year_index"] = self.df["year"] - self.df["year"].min()
            self.df["year_squared"] = self.df["year_index"] ** 2

            # Interaction features
            self.df["gdp_per_capita"] = self.df["gdp"] / self.df["population"]
            self.df["energy_intensity"] = self.df["y"] / self.df["gdp"]

            self.df = self.df.ffill().bfill()
            self.last_row = self.df.iloc[-1]
            self.last_year = int(self.last_row["year"])

            print(f"XGBoost data loaded. Last year: {self.last_year}")
            print(f"Last y value: {self.last_row['y']:,.0f}")

        except Exception as e:
            print(f"Error loading XGBoost data: {e}")
            raise

    def train_model(self):
        """
        Generate a multi-year forecast using the trained XGBoost model.

        Uses Monte Carlo simulations with random noise to estimate prediction
        uncertainty. Returns a list of dictionaries containing yearly forecasts,
        lower and upper confidence bounds, and percentage growth from the previous year.

        Args:
            years_ahead (int): Number of future years to forecast.

        Returns:
            list[dict]: Forecast data for each year with 'year', 'forecast',
                        'lower', 'upper', and 'growth_from_previous'.
        """
        if self.df is None:
            self.load_data()

        # Improved feature set
        feature_cols = [
            "year",
            "population",
            "gdp",
            "renewables_share_energy",
            "y_lag1",
            "y_lag2",
            "y_lag3",
            "y_diff1",
            "y_rolling_mean_3",
            "y_rolling_std_3",
            "y_rolling_mean_5",
            "y_rolling_std_5",
            "gdp_growth_feature",
            "population_growth_feature",
            "renewables_growth_feature",
            "year_index",
            "year_squared",
            "gdp_per_capita",
            "energy_intensity",
        ]

        available_features = [col for col in feature_cols if col in self.df.columns]
        self.feature_names = available_features

        print(f"Training with {len(available_features)} features: {available_features}")

        # Prepare training data
        train_df = self.df.dropna(subset=available_features + ["y"])

        # Target: next year's value (y_t+1)
        X = train_df[available_features].iloc[:-1]
        y = self.df["y"].iloc[1:]
        y = y.iloc[: len(X)]

        print(f"Training samples: {len(X)}, Target samples: {len(y)}")

        # XGBoost model with tuned hyperparameters
        self.model = XGBRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror",
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="mae",
        )

        # Train the model
        self.model.fit(X, y)

        # Evaluate model on training data
        train_predictions = self.model.predict(X)
        mae = np.mean(np.abs(train_predictions - y))
        mape = np.mean(np.abs((train_predictions - y) / y)) * 100

        print(f"XGBoost model trained")
        print(f"Training MAE: {mae:,.0f}")
        print(f"Training MAPE: {mape:.1f}%")

        # Feature importance
        importance = pd.DataFrame(
            {
                "feature": available_features,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        print("\nTop 5 important features:")
        print(importance.head().to_string())

    def forecast(self, years_ahead=10):
        if self.model is None:
            self.train_model()

        print(f"\nGenerating {years_ahead}-year forecast with XGBoost...")

        # Store predictions from multiple runs for uncertainty estimation
        all_predictions = []

        # Monte Carlo simulation
        n_iterations = 100

        for iteration in range(n_iterations):
            if iteration % 20 == 0:
                print(f"  Running iteration {iteration + 1}/{n_iterations}")

            iteration_preds = []
            temp_data = self.df.copy()

            # Add random noise to growth rates
            pop_factor = np.random.uniform(0.95, 1.05)
            gdp_factor = np.random.uniform(0.95, 1.05)
            renew_factor = np.random.uniform(0.95, 1.05)

            for i in range(1, years_ahead + 1):
                target_year = self.last_year + i

                # Compute future auxiliary variables with realistic growth
                pop = (
                    self.last_row["population"]
                    * (1 + self.avg_growth_rates["population_growth"] * pop_factor) ** i
                )
                gdp = (
                    self.last_row["gdp"]
                    * (1 + self.avg_growth_rates["gdp_growth"] * gdp_factor) ** i
                )
                renew = (
                    self.last_row["renewables_share_energy"]
                    * (1 + self.avg_growth_rates["renewables_growth"] * renew_factor)
                    ** i
                )
                renew = min(renew, 100.0)

                # Create future row
                future_row = pd.DataFrame(
                    {
                        "year": [target_year],
                        "population": [pop],
                        "gdp": [gdp],
                        "renewables_share_energy": [renew],
                        "y": [np.nan],
                    }
                )

                # Combine with historical data
                temp_df = pd.concat([temp_data, future_row], ignore_index=True)

                # Generate features for combined data
                for lag in [1, 2, 3]:
                    temp_df[f"y_lag{lag}"] = temp_df["y"].shift(lag)

                temp_df["y_diff1"] = temp_df["y"].diff()

                # Use average historical growth rates
                temp_df["gdp_growth_feature"] = self.avg_growth_rates["gdp_growth"]
                temp_df["population_growth_feature"] = self.avg_growth_rates[
                    "population_growth"
                ]
                temp_df["renewables_growth_feature"] = self.avg_growth_rates[
                    "renewables_growth"
                ]

                for window in [3, 5]:
                    temp_df[f"y_rolling_mean_{window}"] = (
                        temp_df["y"]
                        .shift(1)
                        .rolling(window=window, min_periods=1)
                        .mean()
                    )
                    temp_df[f"y_rolling_std_{window}"] = (
                        temp_df["y"]
                        .shift(1)
                        .rolling(window=window, min_periods=1)
                        .std()
                    )

                temp_df["year_index"] = temp_df["year"] - temp_df["year"].min()
                temp_df["year_squared"] = temp_df["year_index"] ** 2
                temp_df["gdp_per_capita"] = temp_df["gdp"] / temp_df["population"]
                temp_df["energy_intensity"] = temp_df["y"] / temp_df["gdp"]

                temp_df = temp_df.ffill().bfill()

                # Extract future features
                future_features = temp_df.iloc[[-1]][self.feature_names]

                # Prediction
                pred = self.model.predict(future_features)[0]

                # Ensure positive prediction
                pred = max(pred, self.last_row["y"] * 0.5)

                iteration_preds.append(pred)

                # Update for next step
                future_row["y"] = pred
                temp_data = pd.concat([temp_data, future_row], ignore_index=True)

            all_predictions.append(iteration_preds)

        # Compute statistics
        all_predictions = np.array(all_predictions)
        mean_predictions = np.mean(all_predictions, axis=0)
        std_predictions = np.std(all_predictions, axis=0)

        # 95% confidence interval
        lower_bounds = mean_predictions - 1.96 * std_predictions
        upper_bounds = mean_predictions + 1.96 * std_predictions

        # Ensure non-negative bounds
        lower_bounds = np.maximum(lower_bounds, 0)

        # Future years
        forecast_years = np.arange(self.last_year + 1, self.last_year + 1 + years_ahead)

        # Build JSON output
        forecast_json = []
        for i, (year, mean, lower, upper) in enumerate(
            zip(forecast_years, mean_predictions, lower_bounds, upper_bounds)
        ):
            forecast_json.append(
                {
                    "year": int(year),
                    "forecast": float(mean),
                    "lower": float(lower),
                    "upper": float(upper),
                    "growth_from_previous": float(
                        (
                            (
                                mean / ( self.last_row["y"] if i == 0 else mean_predictions[i - 1]))- 1)* 100),
                }
            )

        total_growth = ((mean_predictions[-1] / mean_predictions[0]) - 1) * 100
        avg_uncertainty = np.mean(upper_bounds - lower_bounds)

        print(f"\nXGBoost Forecast Summary:")
        print(f"  First year ({forecast_years[0]}): {mean_predictions[0]:,.0f}")
        print(f"  Last year ({forecast_years[-1]}): {mean_predictions[-1]:,.0f}")
        print(f"  Total growth: {total_growth:.1f}%")
        print(f"  Avg uncertainty width: {avg_uncertainty:,.0f}")

        return forecast_json

    def train_and_forecast(self, years_ahead=10):
        """
        Load data, train the model, and generate forecasts in a single step.

        Args:
            years_ahead (int): Number of future years to forecast.

        Returns:
            list[dict]: Forecast results for each year.
        """
        self.load_data()
        self.train_model()
        return self.forecast(years_ahead=years_ahead)
