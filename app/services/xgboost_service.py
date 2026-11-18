"""
XGBoostForecastService module.

XGBoost on lag, rolling, time and economic features of annual German
electricity generation (TWh) with recursive multi-step forecasting.
Prediction intervals come from rolling-origin backtest residuals. (Perturbing
exogenous growth scenarios, the previous approach, mostly left predictions in
the same tree leaves and produced zero-width, meaningless intervals.)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor

from app.services.base import BaseForecastService, growth_pct
from app.services.features import build_training_frame, recursive_forecast

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
    "year_index",
    "year_squared",
    "gdp_per_capita",
    "energy_intensity",
]

RANDOM_STATE = 42


class XGBoostForecastService(BaseForecastService):
    """XGBoost-based forecasting service for Germany energy data."""

    model_name = "XGBoost"
    model_filename = "xgboost_model.joblib"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.feature_names: list[str] = list(FEATURE_COLUMNS)

    def _library_versions(self) -> list[str]:
        return [f"xgboost={xgboost.__version__}"]

    def _fit_model(self, history: pd.DataFrame) -> XGBRegressor:
        X, y = build_training_frame(history, self.feature_names)
        model = XGBRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=1,
            objective="reg:squarederror",
            reg_alpha=0.1,
            reg_lambda=1.0,
        )
        return model.fit(X, y)

    def _predict_point(
        self, model: XGBRegressor, history: pd.DataFrame, exog: pd.DataFrame
    ) -> np.ndarray:
        return recursive_forecast(model.predict, history, exog, self.feature_names)

    def _row_extras(self, row: dict[str, Any], previous: float) -> dict[str, Any]:
        return {"growth_from_previous": growth_pct(row["forecast"], previous)}

    def _serialize(self) -> dict[str, Any]:
        return {"model": self.model, "feature_names": self.feature_names}

    def _deserialize(self, payload: dict[str, Any]) -> None:
        if payload.get("feature_names") != self.feature_names:
            raise ValueError("Saved model was trained with different features")
        self.model = payload["model"]
