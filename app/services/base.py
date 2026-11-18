"""
Common infrastructure for the forecasting services: data paths, model
persistence keyed by a data/model fingerprint, and a thread-safe forecast cache.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.services.features import average_growth_rates, load_energy_data

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = APP_DIR / "data" / "germany-energy-clean.csv"
DEFAULT_MODELS_DIR = APP_DIR / "models"
MAX_YEARS_AHEAD = 50

ForecastRows = list[dict[str, Any]]


def validate_years_ahead(years_ahead: int) -> int:
    """Return ``years_ahead`` if it is an int in [1, MAX_YEARS_AHEAD], else raise."""
    if isinstance(years_ahead, bool) or not isinstance(years_ahead, int):
        raise ValueError("years_ahead must be an integer")
    if not 1 <= years_ahead <= MAX_YEARS_AHEAD:
        raise ValueError(f"years_ahead must be between 1 and {MAX_YEARS_AHEAD}")
    return years_ahead


class BaseForecastService(ABC):
    """
    Base class for forecasting services.

    Subclasses implement :meth:`_fit` and :meth:`_predict`; this class handles
    data loading, persisted-model reuse and caching of forecast results.
    """

    model_name: str = ""
    model_filename: str = ""
    # Bump when features/hyperparameters change so persisted models are retrained.
    model_version: str = "1"

    def __init__(
        self,
        data_path: str | Path | None = None,
        models_dir: str | Path | None = None,
    ) -> None:
        self.data_path = Path(data_path) if data_path else DEFAULT_DATA_PATH
        self.models_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
        self.model_path = self.models_dir / self.model_filename
        self.df: pd.DataFrame | None = None
        self.model: Any = None
        self.last_row: pd.Series | None = None
        self.last_year: int | None = None
        self.avg_growth_rates: dict[str, float] = {}
        self._forecast_cache: dict[int, ForecastRows] = {}
        self._lock = threading.RLock()

    @property
    def is_ready(self) -> bool:
        """Whether a trained model is available."""
        return self.model is not None

    def compute_data_hash(self) -> str:
        """Fingerprint of the dataset (line-ending independent) and model version."""
        content = self.data_path.read_bytes().replace(b"\r\n", b"\n")
        digest = hashlib.sha256(content)
        digest.update(f"{type(self).__name__}:{self.model_version}".encode())
        for part in self._library_versions():
            digest.update(part.encode())
        return digest.hexdigest()

    def _library_versions(self) -> list[str]:
        return []

    def load_data(self) -> pd.DataFrame:
        """Load and validate the historical dataset."""
        df = load_energy_data(self.data_path)
        with self._lock:
            self.df = df
            self.last_row = df.iloc[-1]
            self.last_year = int(self.last_row["year"])
            self.avg_growth_rates = average_growth_rates(df)
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
            self._fit()
            logger.info("%s model trained", self.model_name)
            self._persist(data_hash)

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

    # --- persistence -------------------------------------------------------

    def _serialize(self) -> dict[str, Any]:
        return {"model": self.model}

    def _deserialize(self, payload: dict[str, Any]) -> None:
        self.model = payload["model"]

    def _write_payload(self, payload: dict[str, Any]) -> None:
        joblib.dump(payload, self.model_path)

    def _read_payload(self) -> dict[str, Any]:
        return joblib.load(self.model_path)

    def _load_persisted(self, data_hash: str) -> bool:
        if not self.model_path.exists():
            return False
        try:
            payload = self._read_payload()
            if not isinstance(payload, dict) or payload.get("data_hash") != data_hash:
                return False
            self._deserialize(payload)
            return True
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                "Could not load saved %s model; retraining",
                self.model_name,
                exc_info=True,
            )
            return False

    def _persist(self, data_hash: str) -> None:
        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            self._write_payload({**self._serialize(), "data_hash": data_hash})
        except OSError:
            logger.warning("Could not save %s model", self.model_name, exc_info=True)

    # --- model specific ----------------------------------------------------

    @abstractmethod
    def _fit(self) -> None:
        """Train ``self.model`` on ``self.df``."""

    @abstractmethod
    def _predict(self, years_ahead: int) -> ForecastRows:
        """Generate forecast rows for the next ``years_ahead`` years."""


def growth_pct(current: float, previous: float) -> float:
    """Percentage change, 0.0 when the previous value is zero."""
    return float((current / previous - 1) * 100) if previous else 0.0
