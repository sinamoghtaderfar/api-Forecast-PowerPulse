"""
RandomForestForecastService module.

Trains a Random Forest regression model on lag, rolling, trend, time and
economic features of Germany energy data, and generates recursive multi-year
forecasts with per-tree uncertainty estimation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from app.services.base import BaseForecastService, ForecastRows, growth_pct
from app.services.features import (
    build_training_frame,
    compute_features,
    project_exogenous,
)

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
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

RANDOM_STATE = 42
GROWTH_FACTOR_RANGE = (0.9, 1.1)


class RandomForestForecastService(BaseForecastService):
    """Random Forest-based forecasting service for Germany energy data."""

    model_name = "Random Forest"
    model_filename = "random_forest_model.joblib"
    model_version = "2"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.feature_names: list[str] = list(FEATURE_COLUMNS)
        self.cv_scores: list[float] = []
        self._use_cv = True

    def train_model(self, use_cv: bool = True) -> None:
        """Train (or load) the model, optionally reporting time-series CV MAE."""
        self._use_cv = use_cv
        super().train_model()

    def _library_versions(self) -> list[str]:
        return [f"sklearn={sklearn.__version__}"]

    def _fit(self) -> None:
        X, y = build_training_frame(self.df, self.avg_growth_rates, self.feature_names)
        logger.info("Random Forest training samples: %d", len(X))

        self.cv_scores = []
        if self._use_cv and len(X) > 10:
            tscv = TimeSeriesSplit(n_splits=min(5, len(X) // 2))
            for train_idx, val_idx in tscv.split(X):
                fold_model = RandomForestRegressor(
                    n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, max_depth=10
                )
                fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
                self.cv_scores.append(
                    float(
                        mean_absolute_error(
                            y.iloc[val_idx], fold_model.predict(X.iloc[val_idx])
                        )
                    )
                )
            logger.info(
                "Random Forest CV MAE: %.2f (+/- %.2f)",
                np.mean(self.cv_scores),
                np.std(self.cv_scores),
            )

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            bootstrap=True,
            max_features="sqrt",
        )
        model.fit(X, y)
        self.model = model

    def _predict(self, years_ahead: int) -> ForecastRows:
        trees = self.model.estimators_
        n_trees = len(trees)
        rng = np.random.default_rng(RANDOM_STATE)
        # Each tree follows its own growth scenario for the exogenous variables.
        factors = rng.uniform(*GROWTH_FACTOR_RANGE, size=(n_trees, 3))

        history = np.tile(self.df["y"].to_numpy(dtype=float), (n_trees, 1))
        year_min = int(self.df["year"].iloc[0])
        last_y = float(self.last_row["y"])
        prev_gdp = np.full(n_trees, float(self.last_row["gdp"]))
        predictions = np.empty((n_trees, years_ahead))

        for step in range(1, years_ahead + 1):
            exog = project_exogenous(
                self.last_row, self.avg_growth_rates, step, factors
            )
            X = compute_features(
                history,
                year=self.last_year + step,
                year_min=year_min,
                prev_gdp=prev_gdp,
                growth_rates=self.avg_growth_rates,
                **exog,
            )[self.feature_names].to_numpy()
            preds = np.array(
                [tree.predict(X[i : i + 1])[0] for i, tree in enumerate(trees)]
            )
            preds = np.clip(preds, last_y * 0.3, last_y * 2.0)
            predictions[:, step - 1] = preds
            history = np.column_stack([history, preds])
            prev_gdp = exog["gdp"]

        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        lower = np.maximum(mean - 1.96 * std, 0)
        upper = mean + 1.96 * std

        rows = []
        for i in range(years_ahead):
            previous = last_y if i == 0 else mean[i - 1]
            rows.append(
                {
                    "year": self.last_year + i + 1,
                    "forecast": float(mean[i]),
                    "lower": float(lower[i]),
                    "upper": float(upper[i]),
                    "uncertainty_width": float(upper[i] - lower[i]),
                    "growth_rate": growth_pct(mean[i], previous),
                }
            )
        return rows

    def _serialize(self) -> dict[str, Any]:
        return {"model": self.model, "feature_names": self.feature_names}

    def _deserialize(self, payload: dict[str, Any]) -> None:
        if payload.get("feature_names") != self.feature_names:
            raise ValueError("Saved model was trained with different features")
        self.model = payload["model"]
