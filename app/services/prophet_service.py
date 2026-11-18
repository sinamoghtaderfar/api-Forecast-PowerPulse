"""
ProphetForecastService module.

Trains a Prophet time series model on Germany energy data with population, GDP
and renewables_share_energy as additional regressors, and generates multi-year
forecasts including uncertainty intervals.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
import prophet
from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json

from app.services.base import BaseForecastService, ForecastRows
from app.services.features import EXOG_COLUMNS

logger = logging.getLogger(__name__)

# Annual growth assumptions for the future regressor values.
POPULATION_GROWTH = 0.01
GDP_GROWTH = 0.02
RENEWABLES_GROWTH = 0.03


class ProphetForecastService(BaseForecastService):
    """Prophet-based forecasting service for Germany energy data."""

    model_name = "Prophet"
    model_filename = "prophet_model.json"
    model_version = "2"

    @property
    def last_values(self) -> dict[str, float] | None:
        """Last observed regressor values."""
        if self.last_row is None:
            return None
        return {col: float(self.last_row[col]) for col in EXOG_COLUMNS}

    def _library_versions(self) -> list[str]:
        return [f"prophet={prophet.__version__}"]

    def _training_frame(self) -> pd.DataFrame:
        df = self.df[["year", "y", *EXOG_COLUMNS]].copy()
        df["ds"] = pd.to_datetime(df["year"].astype(str) + "-01-01")
        return df[["ds", "y", *EXOG_COLUMNS]]

    def _fit(self) -> None:
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            growth="linear",
            interval_width=0.95,
        )
        for col in EXOG_COLUMNS:
            model.add_regressor(col)
        model.fit(self._training_frame())
        self.model = model

    def _serialize(self) -> dict[str, Any]:
        return {"model_json": model_to_json(self.model)}

    def _deserialize(self, payload: dict[str, Any]) -> None:
        self.model = model_from_json(payload["model_json"])

    def _write_payload(self, payload: dict[str, Any]) -> None:
        self.model_path.write_text(json.dumps(payload), encoding="utf-8")

    def _read_payload(self) -> dict[str, Any]:
        return json.loads(self.model_path.read_text(encoding="utf-8"))

    def _predict(self, years_ahead: int) -> ForecastRows:
        steps = np.arange(1, years_ahead + 1)
        last = self.last_values
        future = pd.DataFrame(
            {
                "ds": pd.date_range(
                    start=pd.Timestamp(f"{self.last_year + 1}-01-01"),
                    periods=years_ahead,
                    freq="YS",
                ),
                "population": last["population"] * (1 + POPULATION_GROWTH) ** steps,
                "gdp": last["gdp"] * (1 + GDP_GROWTH) ** steps,
                "renewables_share_energy": np.minimum(
                    last["renewables_share_energy"] * (1 + RENEWABLES_GROWTH) ** steps,
                    100.0,
                ),
            }
        )
        result = self.model.predict(future)
        return [
            {
                "year": int(ds.year),
                "forecast": float(yhat),
                "lower": float(lower),
                "upper": float(upper),
            }
            for ds, yhat, lower, upper in zip(
                result["ds"], result["yhat"], result["yhat_lower"], result["yhat_upper"]
            )
        ]
