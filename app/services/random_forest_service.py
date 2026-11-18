"""
RandomForestForecastService module.

Random Forest on lag, rolling, trend, time and economic features of annual
German electricity generation (TWh) with recursive multi-step forecasting.
Prediction intervals come from rolling-origin backtest residuals; the spread
between individual trees is not used because it measures model disagreement,
not forecast error.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor

from app.services.base import BaseForecastService, growth_pct
from app.services.features import build_training_frame, recursive_forecast

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
    # Interaction features
    "gdp_per_capita",
    "energy_intensity",
    "renewables_impact",
]

RANDOM_STATE = 42


class RandomForestForecastService(BaseForecastService):
    """Random Forest-based forecasting service for Germany energy data."""

    model_name = "Random Forest"
    model_filename = "random_forest_model.joblib"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.feature_names: list[str] = list(FEATURE_COLUMNS)

    def _library_versions(self) -> list[str]:
        return [f"sklearn={sklearn.__version__}"]

    def _fit_model(self, history: pd.DataFrame) -> RandomForestRegressor:
        X, y = build_training_frame(history, self.feature_names)
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
        # Single-threaded prediction keeps the tree-average summation order, and
        # therefore the output, bit-for-bit reproducible.
        return model.set_params(n_jobs=1)

    def _predict_point(
        self, model: RandomForestRegressor, history: pd.DataFrame, exog: pd.DataFrame
    ) -> np.ndarray:
        return recursive_forecast(model.predict, history, exog, self.feature_names)

    def _row_extras(self, row: dict[str, Any], previous: float) -> dict[str, Any]:
        width = (
            None
            if row["lower"] is None or row["upper"] is None
            else row["upper"] - row["lower"]
        )
        return {
            "uncertainty_width": width,
            "growth_rate": growth_pct(row["forecast"], previous),
        }

    def _serialize(self) -> dict[str, Any]:
        return {"model": self.model, "feature_names": self.feature_names}

    def _deserialize(self, payload: dict[str, Any]) -> None:
        if payload.get("feature_names") != self.feature_names:
            raise ValueError("Saved model was trained with different features")
        self.model = payload["model"]
