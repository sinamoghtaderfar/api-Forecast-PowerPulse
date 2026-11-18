"""
Common infrastructure for the forecasting services: data loading, the shared
fit/forecast pipeline (identical in production and backtesting), empirical
interval calibration, model persistence and a thread-safe forecast cache.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import os
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.services import backtesting, features
from app.services.features import (
    exogenous_growth_rates,
    exogenous_path,
    prepare_history,
    read_energy_csv,
)
from app.services.uncertainty import DEFAULT_LEVEL, ResidualCalibration

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = APP_DIR / "data" / "germany-energy-clean.csv"
DEFAULT_MODELS_DIR = Path(os.getenv("POWERPULSE_MODELS_DIR", APP_DIR / "models"))
MAX_YEARS_AHEAD = 50

ForecastRows = list[dict[str, Any]]


def validate_years_ahead(years_ahead: int) -> int:
    """Return ``years_ahead`` if it is an int in [1, MAX_YEARS_AHEAD], else raise."""
    if isinstance(years_ahead, bool) or not isinstance(years_ahead, int):
        raise ValueError("years_ahead must be an integer")
    if not 1 <= years_ahead <= MAX_YEARS_AHEAD:
        raise ValueError(f"years_ahead must be between 1 and {MAX_YEARS_AHEAD}")
    return years_ahead


def growth_pct(current: float, previous: float) -> float | None:
    """Percentage change; None when undefined (non-positive or non-finite base)."""
    if not (np.isfinite(current) and np.isfinite(previous)) or previous <= 0:
        return None
    return float((current / previous - 1) * 100)


class BaseForecastService(ABC):
    """
    Base class for forecasting services.

    Subclasses implement :meth:`_fit_model` and :meth:`_predict_point`. The
    same two methods are used for production forecasts and for the
    rolling-origin backtests that calibrate the prediction intervals.
    """

    model_name: str = ""
    model_filename: str = ""
    # Bump when features/hyperparameters change so persisted models are retrained.
    model_version: str = "3"

    def __init__(
        self,
        data_path: str | Path | None = None,
        models_dir: str | Path | None = None,
        *,
        calibrate: bool = True,
        interval_level: float = DEFAULT_LEVEL,
    ) -> None:
        self.data_path = Path(data_path) if data_path else DEFAULT_DATA_PATH
        self.models_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
        self.model_path = self.models_dir / self.model_filename
        self.calibrate = calibrate
        self.interval_level = interval_level
        self.raw: pd.DataFrame | None = None
        self.df: pd.DataFrame | None = None
        self.model: Any = None
        self.calibration: ResidualCalibration | None = None
        self.last_row: pd.Series | None = None
        self.last_year: int | None = None
        self.growth_rates: dict[str, float] = {}
        self._forecast_cache: dict[int, ForecastRows] = {}
        self._lock = threading.RLock()

    @property
    def is_ready(self) -> bool:
        """Whether a trained model is available."""
        return self.model is not None

    def compute_data_hash(self) -> str:
        """Fingerprint of dataset (line-ending independent) and model configuration."""
        content = self.data_path.read_bytes().replace(b"\r\n", b"\n")
        digest = hashlib.sha256(content)
        config = (
            f"{type(self).__name__}:{self.model_version}:calibrate={self.calibrate}:"
            f"level={self.interval_level}:min_train={backtesting.MIN_TRAIN_YEARS}:"
            f"growth_window={features.GROWTH_WINDOW_YEARS}"
        )
        digest.update(config.encode())
        for part in self._library_versions():
            digest.update(part.encode())
        return digest.hexdigest()

    def _library_versions(self) -> list[str]:
        return []

    def load_data(self) -> pd.DataFrame:
        """Load and validate the historical dataset."""
        raw = read_energy_csv(self.data_path)
        df = prepare_history(raw)
        with self._lock:
            self.raw = raw
            self.df = df
            self.last_row = df.iloc[-1]
            self.last_year = int(self.last_row["year"])
            self.growth_rates = exogenous_growth_rates(raw)
            self._forecast_cache.clear()
        logger.info("%s data loaded. Last year: %s", self.model_name, self.last_year)
        return df

    def train_model(self) -> None:
        """Load a persisted model for the current data, or train and persist one."""
        with self._lock:
            if self.df is None:
                self.load_data()
            self._forecast_cache.clear()
            data_hash = self.compute_data_hash()
            if self._load_persisted(data_hash):
                logger.info("Loaded saved %s model", self.model_name)
                return
            self.model = self._fit_model(self.df)
            self.calibration = self._calibrate() if self.calibrate else None
            logger.info("%s model trained", self.model_name)
            self._persist(data_hash)

    def forecast_from_raw(self, raw: pd.DataFrame, horizon: int) -> np.ndarray:
        """
        Fit on ``raw`` (rows up to a forecast origin) and return a recursive
        point forecast for the following ``horizon`` years. Used by backtests.
        """
        history = prepare_history(raw)
        exog = exogenous_path(history.iloc[-1], exogenous_growth_rates(raw), horizon)
        model = self._fit_model(history)
        return self._predict_point(model, history, exog)

    def _calibrate(self) -> ResidualCalibration:
        records = backtesting.rolling_origin_backtest(self.forecast_from_raw, self.raw)
        return ResidualCalibration.from_records(records, self.interval_level)

    def uncertainty_info(self) -> dict[str, Any] | None:
        """Description of the interval method and its evidence."""
        return self.calibration.summary() if self.calibration else None

    def forecast(self, years_ahead: int = 10) -> ForecastRows:
        """Return a ``years_ahead``-year forecast (cached per horizon)."""
        years_ahead = validate_years_ahead(years_ahead)
        with self._lock:
            if self.model is None:
                self.train_model()
            if years_ahead not in self._forecast_cache:
                self._forecast_cache[years_ahead] = self._predict(years_ahead)
            return copy.deepcopy(self._forecast_cache[years_ahead])

    def train_and_forecast(self, years_ahead: int = 10) -> ForecastRows:
        """Load data, train (or load) the model and generate a forecast."""
        self.load_data()
        self.train_model()
        return self.forecast(years_ahead=years_ahead)

    def _predict(self, years_ahead: int) -> ForecastRows:
        exog = exogenous_path(self.last_row, self.growth_rates, years_ahead)
        point = self._predict_point(self.model, self.df, exog)
        if point.shape != (years_ahead,) or not np.isfinite(point).all():
            raise ValueError(f"{self.model_name} produced invalid predictions")
        widths = (
            self.calibration.half_widths(years_ahead)
            if self.calibration
            else [None] * years_ahead
        )
        rows = []
        previous = float(self.last_row["y"])
        for year, value, width in zip(exog["year"], point, widths):
            row = {
                "year": int(year),
                "forecast": float(value),
                "lower": None if width is None else float(max(value - width, 0.0)),
                "upper": None if width is None else float(value + width),
            }
            row.update(self._row_extras(row, previous))
            rows.append(row)
            previous = float(value)
        return rows

    def _row_extras(self, row: dict[str, Any], previous: float) -> dict[str, Any]:
        """Model-specific per-row fields (kept for API compatibility)."""
        del row, previous
        return {}

    # --- persistence -------------------------------------------------------

    def _serialize(self) -> dict[str, Any]:
        return {"model": self.model}

    def _deserialize(self, payload: dict[str, Any]) -> None:
        self.model = payload["model"]

    def _write_payload(self, payload: dict[str, Any], path: Path) -> None:
        joblib.dump(payload, path)

    def _read_payload(self) -> dict[str, Any]:
        # Only load artifacts written by this service into a trusted directory:
        # joblib/pickle files can execute code when loaded.
        return joblib.load(self.model_path)

    def _load_persisted(self, data_hash: str) -> bool:
        if not self.model_path.exists():
            return False
        try:
            payload = self._read_payload()
            if not isinstance(payload, dict) or payload.get("data_hash") != data_hash:
                return False
            self._deserialize(payload)
            calibration = payload.get("calibration")
            self.calibration = (
                ResidualCalibration.from_dict(calibration) if calibration else None
            )
            return True
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                "Could not load saved %s model; retraining",
                self.model_name,
                exc_info=True,
            )
            self.model = None
            return False

    def _persist(self, data_hash: str) -> None:
        payload = {
            **self._serialize(),
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "data_hash": data_hash,
        }
        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            # Write atomically so concurrent readers never see a partial file.
            tmp_path = self.model_path.with_name(self.model_path.name + ".tmp")
            self._write_payload(payload, tmp_path)
            os.replace(tmp_path, self.model_path)
        except OSError:
            logger.warning("Could not save %s model", self.model_name, exc_info=True)

    # --- model specific ----------------------------------------------------

    @abstractmethod
    def _fit_model(self, history: pd.DataFrame) -> Any:
        """Fit and return a model on ``history`` (prepared, imputed frame)."""

    @abstractmethod
    def _predict_point(
        self, model: Any, history: pd.DataFrame, exog: pd.DataFrame
    ) -> np.ndarray:
        """Point forecasts for the years in ``exog`` following ``history``."""
