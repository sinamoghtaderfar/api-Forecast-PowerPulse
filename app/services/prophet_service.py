"""
ProphetForecastService module.

Prophet model of annual German electricity generation (TWh) with population,
GDP and renewables share as additional regressors. Future regressor values are
projected with the shared exogenous growth assumptions; prediction intervals
come from rolling-origin backtest residuals (see ``app.services.uncertainty``)
rather than Prophet's own intervals, which ignore regressor uncertainty.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Prophet logs an ERROR at import when the optional plotly dependency is
# missing; plotting is not used by this service.
logging.getLogger("prophet.plot").setLevel(logging.CRITICAL)

# pylint: disable=wrong-import-position
import prophet
from cmdstanpy.utils import get_logger as get_cmdstanpy_logger
from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json

from app.services.base import BaseForecastService
from app.services.features import EXOG_COLUMNS

logger = logging.getLogger(__name__)
# cmdstanpy configures its logger lazily; initialise it, then silence INFO noise.
get_cmdstanpy_logger().setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


def _to_prophet_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["ds"] = pd.to_datetime(out["year"].astype(int).astype(str) + "-01-01")
    return out


class ProphetForecastService(BaseForecastService):
    """Prophet-based forecasting service for Germany energy data."""

    model_name = "Prophet"
    model_filename = "prophet_model.json"

    @property
    def last_values(self) -> dict[str, float] | None:
        """Last observed regressor values."""
        if self.last_row is None:
            return None
        return {col: float(self.last_row[col]) for col in EXOG_COLUMNS}

    def _library_versions(self) -> list[str]:
        return [f"prophet={prophet.__version__}"]

    def _fit_model(self, history: pd.DataFrame) -> Prophet:
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            growth="linear",
            # Intervals are calibrated empirically; skip Prophet's sampling.
            uncertainty_samples=0,
        )
        for col in EXOG_COLUMNS:
            model.add_regressor(col)
        model.fit(_to_prophet_frame(history)[["ds", "y", *EXOG_COLUMNS]])
        return model

    def _predict_point(
        self, model: Prophet, history: pd.DataFrame, exog: pd.DataFrame
    ) -> np.ndarray:
        del history
        result = model.predict(_to_prophet_frame(exog)[["ds", *EXOG_COLUMNS]])
        return result["yhat"].to_numpy(dtype=float)

    def _serialize(self) -> dict[str, Any]:
        return {"model_json": model_to_json(self.model)}

    def _deserialize(self, payload: dict[str, Any]) -> None:
        self.model = model_from_json(payload["model_json"])

    def _write_payload(self, payload: dict[str, Any], path: Path) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _read_payload(self) -> dict[str, Any]:
        return json.loads(self.model_path.read_text(encoding="utf-8"))
