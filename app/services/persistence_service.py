"""
PersistenceForecastService module.

Naive benchmark: every future year equals the last observed annual
electricity generation (TWh). It has no fitted parameters and ignores the
exogenous projections. In the rolling-origin backtests (origins 2000-2024)
none of Prophet, Random Forest or XGBoost beat it reliably at any horizon, so
it is served as the reference forecast (see docs/FORECAST_ACCURACY_REPORT.md).
Prediction intervals use the same backtest-calibrated method as the other
models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.services.base import BaseForecastService


class PersistenceForecastService(BaseForecastService):
    """Last-value (random-walk) benchmark forecast."""

    model_name = "Persistence"
    model_filename = "persistence_model.json"
    uses_exogenous = False

    def _fit_model(self, history: pd.DataFrame) -> dict[str, float]:
        return {
            "last_year": int(history["year"].iloc[-1]),
            "last_value": float(history["y"].iloc[-1]),
        }

    def _predict_point(
        self, model: dict[str, float], history: pd.DataFrame, exog: pd.DataFrame
    ) -> np.ndarray:
        del history
        return np.full(len(exog), float(model["last_value"]))

    def _write_payload(self, payload: dict[str, Any], path: Path) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _read_payload(self) -> dict[str, Any]:
        return json.loads(self.model_path.read_text(encoding="utf-8"))
