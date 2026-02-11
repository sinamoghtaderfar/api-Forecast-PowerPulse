
"""
RandomForestForecastService module.

This module defines the RandomForestForecastService class, which loads Germany
energy data, trains a Random Forest regression model, and generates multi-year
forecasts with uncertainty estimation.

The service supports advanced features including lag features, rolling statistics,
interaction features, trend and seasonality approximations, and time series
cross-validation.
"""


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import warnings
import os
import joblib
import hashlib
from pathlib import Path
from functools import lru_cache
warnings.filterwarnings("ignore")


class RandomForestForecastService:
    """
    Random Forest-based forecasting service for Germany energy data.

    Attributes:
        data_path (str): Path to the CSV data file.
        df (pd.DataFrame): Loaded dataset with features.
        model (RandomForestRegressor): Trained Random Forest model.
        last_row (pd.Series): Last row of the dataset.
        last_year (int): Last year in the dataset.
        feature_names (list): Names of features used for training.
        avg_growth_rates (dict): Average historical growth rates for features.
        cv_scores (list): Cross-validation MAE scores.
    """

    def __init__(self, data_path="app/data/germany-energy-clean.csv"):
        self.data_path = data_path
        self.df = None
        self.model = None
        self.last_row = None
        self.last_year = None
        self.feature_names = None
        self.avg_growth_rates = {}
        self.cv_scores = []
        models_dir = Path(__file__).resolve().parent.parent / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_path = models_dir / "random_forest_model.joblib"

    def compute_data_hash(self):
        with open(self.data_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def load_data(self):
        """
        Load and preprocess historical energy data.

        - Loads CSV data into a pandas DataFrame.
        - Validates required columns.
        - Converts columns to numeric.
        - Fills missing values and renames the target column.
        - Calculates average historical growth rates.
        - Creates lag, difference, rolling, trend, seasonal, and interaction features.
        - Stores last row and last year for forecasting.

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

            # Create advanced features
            # Lag features
            for lag in [1, 2, 3, 4, 5]:
                self.df[f"y_lag{lag}"] = self.df["y"].shift(lag)

            # Difference features
            self.df["y_diff1"] = self.df["y"].diff()
            self.df["y_diff2"] = self.df["y_diff1"].diff()

            # Rolling statistics
            for window in [3, 5, 7]:
                self.df[f"y_rolling_mean_{window}"] = (
                    self.df["y"].shift(1).rolling(window=window, min_periods=1).mean()
                )
                self.df[f"y_rolling_std_{window}"] = (
                    self.df["y"].shift(1).rolling(window=window, min_periods=1).std()
                )
                self.df[f"y_rolling_min_{window}"] = (
                    self.df["y"].shift(1).rolling(window=window, min_periods=1).min()
                )
                self.df[f"y_rolling_max_{window}"] = (
                    self.df["y"].shift(1).rolling(window=window, min_periods=1).max()
                )

            # Economic growth features (using average growth rates)
            self.df["gdp_growth_feature"] = self.avg_growth_rates["gdp_growth"]
            self.df["population_growth_feature"] = self.avg_growth_rates[
                "population_growth"
            ]
            self.df["renewables_growth_feature"] = self.avg_growth_rates[
                "renewables_growth"
            ]

            # Time features
            self.df["year_index"] = self.df["year"] - self.df["year"].min()
            self.df["year_squared"] = self.df["year_index"] ** 2
            self.df["year_cubic"] = self.df["year_index"] ** 3

            # Interaction features
            self.df["gdp_per_capita"] = self.df["gdp"] / self.df["population"]
            self.df["energy_intensity"] = self.df["y"] / self.df["gdp"]
            self.df["renewables_impact"] = (
                self.df["renewables_share_energy"] * self.df["gdp_per_capita"]
            )

            # Trend features
            self.df["trend_5yr"] = (
                self.df["y"]
                .shift(1)
                .rolling(window=5, min_periods=1)
                .apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0,
                    raw=False,
                )
            )

            # Seasonality approximation (for annual data)
            self.df["year_sin"] = np.sin(2 * np.pi * self.df["year_index"] / 10)
            self.df["year_cos"] = np.cos(2 * np.pi * self.df["year_index"] / 10)

            self.df = self.df.ffill().bfill()
            self.last_row = self.df.iloc[-1]
            self.last_year = int(self.last_row["year"])

            print(f"Random Forest data loaded. Last year: {self.last_year}")
            print(f"Last y value: {self.last_row['y']:,.0f}")
            print(f"Data shape: {self.df.shape}")

        except Exception as e:
            print(f"Error loading Random Forest data: {e}")
            raise

    def train_model(self, use_cv=True):
        """
        Train the Random Forest regression model.

        - Prepares training data using all available features.
        - Optionally performs time series cross-validation and reports MAE per fold.
        - Trains the final Random Forest model on the full dataset.
        - Evaluates feature importance and prints top features.
        - Reports training performance (MAE, MAPE).

        Args:
            use_cv (bool): Whether to perform time series cross-validation.
        """
        if self.df is None:
            self.load_data()

        # Set feature list (always after load_data)
        feature_cols = [
            # Time features
            "year",
            "year_index",
            "year_squared",
            "year_cubic",
            "year_sin",
            "year_cos",
            # Target history
            "y_lag1",
            "y_lag2",
            "y_lag3",
            "y_lag4",
            "y_lag5",
            "y_diff1",
            "y_diff2",
            "y_rolling_mean_3",
            "y_rolling_std_3",
            "y_rolling_mean_5",
            "y_rolling_std_5",
            "y_rolling_mean_7",
            "y_rolling_std_7",
            "y_rolling_min_3",
            "y_rolling_max_3",
            "trend_5yr",
            # Economic features
            "population",
            "gdp",
            "renewables_share_energy",
            "gdp_growth_feature",
            "population_growth_feature",
            "renewables_growth_feature",
            # Interaction features
            "gdp_per_capita",
            "energy_intensity",
            "renewables_impact",
        ]

        self.feature_names = [col for col in feature_cols if col in self.df.columns]

        print(f"Training with {len(self.feature_names)} features")
        print(f"Features: {self.feature_names}")

        current_hash = self.compute_data_hash()

        if os.path.exists(self.model_path):
            try:
                saved = joblib.load(self.model_path)
                if saved['data_hash'] == current_hash:
                    self.model = saved['model']
                    print("Loaded saved Random Forest model")
                    return
            except Exception as e:
                print(f"Error loading saved Random Forest model: {e}")

        # Prepare training data
        train_df = self.df.dropna(subset=self.feature_names + ["y"])
        X = train_df[self.feature_names]
        y = train_df["y"]

        print(f"Training samples: {len(X)}")

        # Time series cross-validation
        if use_cv and len(X) > 10:
            print("\nPerforming Time Series Cross-Validation...")
            tscv = TimeSeriesSplit(n_splits=min(5, len(X) // 2))

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                fold_model = RandomForestRegressor(
                    n_estimators=100, random_state=42, n_jobs=-1, max_depth=10
                )

                fold_model.fit(X_train, y_train)
                val_pred = fold_model.predict(X_val)
                mae = mean_absolute_error(y_val, val_pred)
                self.cv_scores.append(mae)

                print(f"  Fold {fold}: MAE = {mae:,.0f}")

            print(
                f"CV Average MAE: {np.mean(self.cv_scores):,.0f} "
                f"(+/- {np.std(self.cv_scores):,.0f})"
            )

        # Train final model
        print("\nTraining final Random Forest model...")
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            bootstrap=True,
            max_features="sqrt",
        )

        self.model.fit(X, y)

        # Feature importance
        importance = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        print(f"\nRandom Forest trained with {len(self.model.estimators_)} trees")
        print("\nTop 10 most important features:")
        print(importance.head(10).to_string())

        # Training performance evaluation
        train_pred = self.model.predict(X)
        train_mae = mean_absolute_error(y, train_pred)
        train_mape = np.mean(np.abs((train_pred - y) / y)) * 100

        print(f"\nTraining Performance:")
        print(f"  MAE: {train_mae:,.0f}")
        print(f"  MAPE: {train_mape:.1f}%")

        # Save the model
        to_save = {
            'model': self.model,
            'data_hash': current_hash
        }
        joblib.dump(to_save, self.model_path)
        print("Saved trained Random Forest model")
    @lru_cache(maxsize=32)
    def forecast(self, years_ahead=10):
        """
        Generate multi-year forecasts using the trained Random Forest model.

        - Generates predictions per tree to estimate uncertainty.
        - Uses average historical growth rates for auxiliary features.
        - Computes mean, standard deviation, and 95% confidence intervals.
        - Returns forecast results as a list of dictionaries including
          'year', 'forecast', 'lower', 'upper', 'uncertainty_width', and
          'growth_rate' from previous year.

        Args:
            years_ahead (int): Number of future years to forecast.

        Returns:
            list[dict]: Forecast results for each year.
        """
        if self.model is None:
            self.train_model()

        print(f"\nGenerating {years_ahead}-year forecast with Random Forest...")

        # Store predictions from individual trees
        all_tree_predictions = []

        # Generate predictions from each tree
        for tree_idx, tree in enumerate(self.model.estimators_):
            if tree_idx % 20 == 0:
                print(f"  Processing tree {tree_idx + 1}/{len(self.model.estimators_)}")

            tree_preds = []
            temp_data = self.df.copy()

            # Growth parameters with random variation per tree
            pop_factor = np.random.uniform(0.9, 1.1)
            gdp_factor = np.random.uniform(0.9, 1.1)
            renew_factor = np.random.uniform(0.9, 1.1)

            for i in range(1, years_ahead + 1):
                target_year = self.last_year + i

                # Compute future variables
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

                future_row = pd.DataFrame(
                    {
                        "year": [target_year],
                        "population": [pop],
                        "gdp": [gdp],
                        "renewables_share_energy": [renew],
                        "y": [np.nan],
                    }
                )

                temp_df = pd.concat([temp_data, future_row], ignore_index=True)

                # Generate all features for combined data
                for lag in [1, 2, 3, 4, 5]:
                    temp_df[f"y_lag{lag}"] = temp_df["y"].shift(lag)

                temp_df["y_diff1"] = temp_df["y"].diff()
                temp_df["y_diff2"] = temp_df["y_diff1"].diff()

                for window in [3, 5, 7]:
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
                    temp_df[f"y_rolling_min_{window}"] = (
                        temp_df["y"]
                        .shift(1)
                        .rolling(window=window, min_periods=1)
                        .min()
                    )
                    temp_df[f"y_rolling_max_{window}"] = (
                        temp_df["y"]
                        .shift(1)
                        .rolling(window=window, min_periods=1)
                        .max()
                    )

                temp_df["gdp_growth_feature"] = self.avg_growth_rates["gdp_growth"]
                temp_df["population_growth_feature"] = self.avg_growth_rates[
                    "population_growth"
                ]
                temp_df["renewables_growth_feature"] = self.avg_growth_rates[
                    "renewables_growth"
                ]

                temp_df["year_index"] = temp_df["year"] - temp_df["year"].min()
                temp_df["year_squared"] = temp_df["year_index"] ** 2
                temp_df["year_cubic"] = temp_df["year_index"] ** 3
                temp_df["year_sin"] = np.sin(2 * np.pi * temp_df["year_index"] / 10)
                temp_df["year_cos"] = np.cos(2 * np.pi * temp_df["year_index"] / 10)

                temp_df["gdp_per_capita"] = temp_df["gdp"] / temp_df["population"]
                temp_df["energy_intensity"] = temp_df["y"] / temp_df["gdp"]
                temp_df["renewables_impact"] = (
                    temp_df["renewables_share_energy"] * temp_df["gdp_per_capita"]
                )

                temp_df["trend_5yr"] = (
                    temp_df["y"]
                    .shift(1)
                    .rolling(window=5, min_periods=1)
                    .apply(
                        lambda x: (np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0 ),raw=False,)
                )

                temp_df = temp_df.ffill().bfill()

                # Extract future row features
                future_features = temp_df.iloc[[-1]][self.feature_names]

                # Prediction from this tree
                pred = tree.predict(future_features)[0]

                # Apply realistic constraints
                min_pred = self.last_row["y"] * 0.3
                max_pred = self.last_row["y"] * 2.0
                pred = np.clip(pred, min_pred, max_pred)

                tree_preds.append(pred)

                # Update for next iteration
                future_row["y"] = pred
                temp_data = pd.concat([temp_data, future_row], ignore_index=True)

            all_tree_predictions.append(tree_preds)

        # Compute statistics across all trees
        all_tree_predictions = np.array(all_tree_predictions)
        mean_predictions = np.mean(all_tree_predictions, axis=0)
        std_predictions = np.std(all_tree_predictions, axis=0)

        # 95% confidence interval
        lower_bounds = mean_predictions - 1.96 * std_predictions
        upper_bounds = mean_predictions + 1.96 * std_predictions

        # Ensure positive and reasonable values
        lower_bounds = np.maximum(lower_bounds, 0)

        # Future years
        forecast_years = np.arange(self.last_year + 1, self.last_year + 1 + years_ahead)

        # Build JSON output
        forecast_json = []
        for i, (year, mean, lower, upper) in enumerate(
            zip(forecast_years, mean_predictions, lower_bounds, upper_bounds)
        ):
           forecast_json.append({
            "year": int(year),
            "forecast": float(mean),
            "lower": float(lower),
            "upper": float(upper),
            "uncertainty_width": float(upper - lower),
            "growth_rate": float(((mean / (self.last_row["y"] if i == 0 else mean_predictions[i - 1])) - 1) * 100)
        })

        # Overall statistics
        total_growth = ((mean_predictions[-1] / mean_predictions[0]) - 1) * 100
        avg_annual_growth = total_growth / years_ahead
        avg_uncertainty = np.mean(upper_bounds - lower_bounds)

        print(f"\nRandom Forest Forecast Summary:")
        print(f"  Forecast period: {forecast_years[0]} to {forecast_years[-1]}")
        print(f"  First year: {mean_predictions[0]:,.0f}")
        print(f"  Last year: {mean_predictions[-1]:,.0f}")
        print(f"  Total growth: {total_growth:.1f}%")
        print(f"  Avg annual growth: {avg_annual_growth:.1f}%")
        print(f"  Avg uncertainty width: {avg_uncertainty:,.0f}")
        print(f"  Model trees used: {len(self.model.estimators_)}")

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