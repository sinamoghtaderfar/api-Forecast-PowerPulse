"""
XGBoostForecastService module.

Trains an XGBoost regression model on lag, rolling, time and economic features
of Germany energy data, and generates recursive multi-year forecasts with
Monte Carlo uncertainty estimation over exogenous growth scenarios.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor

from app.services.base import BaseForecastService, ForecastRows, growth_pct
from app.services.features import (
    build_training_frame,
    compute_features,
    project_exogenous,
)

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
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

RANDOM_STATE = 42
N_SIMULATIONS = 100
GROWTH_FACTOR_RANGE = (0.95, 1.05)


class XGBoostForecastService(BaseForecastService):
    """XGBoost-based forecasting service for Germany energy data."""

    model_name = "XGBoost"
    model_filename = "xgboost_model.joblib"
    model_version = "2"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.feature_names: list[str] = list(FEATURE_COLUMNS)

    def _library_versions(self) -> list[str]:
        return [f"xgboost={xgboost.__version__}"]

    def _fit(self) -> None:
        # Features of year t only use targets up to t-1, so this is a
        # one-step-ahead model consistent with the recursive forecast.
        X, y = build_training_frame(self.df, self.avg_growth_rates, self.feature_names)
        logger.info("XGBoost training samples: %d", len(X))
        model = XGBRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            objective="reg:squarederror",
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="mae",
        )
        model.fit(X, y)
        self.model = model

    def _predict(self, years_ahead: int) -> ForecastRows:
        rng = np.random.default_rng(RANDOM_STATE)
        factors = rng.uniform(*GROWTH_FACTOR_RANGE, size=(N_SIMULATIONS, 3))

        history = np.tile(self.df["y"].to_numpy(dtype=float), (N_SIMULATIONS, 1))
        year_min = int(self.df["year"].iloc[0])
        last_y = float(self.last_row["y"])
        prev_gdp = np.full(N_SIMULATIONS, float(self.last_row["gdp"]))
        predictions = np.empty((N_SIMULATIONS, years_ahead))

        for step in range(1, years_ahead + 1):
            exog = project_exogenous(
                self.last_row, self.avg_growth_rates, step, factors
            )
            X: pd.DataFrame = compute_features(
                history,
                year=self.last_year + step,
                year_min=year_min,
                prev_gdp=prev_gdp,
                growth_rates=self.avg_growth_rates,
                **exog,
            )[self.feature_names]
            preds = np.maximum(self.model.predict(X), last_y * 0.5)
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
                    "growth_from_previous": growth_pct(mean[i], previous),
                }
            )
        return rows

    def _serialize(self) -> dict[str, Any]:
        return {"model": self.model, "feature_names": self.feature_names}

    def _deserialize(self, payload: dict[str, Any]) -> None:
        if payload.get("feature_names") != self.feature_names:
            raise ValueError("Saved model was trained with different features")
        self.model = payload["model"]
