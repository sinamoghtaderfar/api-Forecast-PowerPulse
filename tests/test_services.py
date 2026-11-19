import shutil

import numpy as np
import pytest

from app.services.base import DEFAULT_DATA_PATH, MAX_YEARS_AHEAD
from app.services.prophet_service import ProphetForecastService
from app.services.random_forest_service import RandomForestForecastService
from app.services.xgboost_service import XGBoostForecastService

SERVICE_CLASSES = [
    ProphetForecastService,
    RandomForestForecastService,
    XGBoostForecastService,
]
KEYS = {
    ProphetForecastService: "prophet",
    RandomForestForecastService: "random_forest",
    XGBoostForecastService: "xgboost",
}


@pytest.mark.parametrize("key", ["prophet", "random_forest", "xgboost"])
def test_forecast_rows_valid(trained_services, key):
    service = trained_services[key]
    max_h = service.calibration.max_calibrated_horizon
    assert max_h >= 10
    rows = service.forecast(max_h + 2)
    assert [r["year"] for r in rows] == list(range(2026, 2026 + max_h + 2))
    widths = []
    for row in rows[:max_h]:
        assert np.isfinite([row["forecast"], row["lower"], row["upper"]]).all()
        assert 0 <= row["lower"] <= row["forecast"] <= row["upper"]
        widths.append(row["upper"] - row["forecast"])
    assert (np.diff(widths) >= -1e-9).all()
    assert widths[0] > 0
    for row in rows[max_h:]:
        assert row["lower"] is None and row["upper"] is None


@pytest.mark.parametrize("cls", SERVICE_CLASSES)
def test_persisted_model_is_reused(trained_services, models_dir, cls, monkeypatch):
    reference = trained_services[KEYS[cls]]
    service = cls(models_dir=models_dir)
    monkeypatch.setattr(
        service, "_fit_model", lambda *_: pytest.fail("model should not be retrained")
    )
    service.load_data()
    service.train_model()
    assert service.calibration == reference.calibration
    assert service.forecast(3) == reference.forecast(3)


@pytest.mark.parametrize("cls", SERVICE_CLASSES)
def test_fit_and_forecast_deterministic(cls, raw):
    train = raw[raw["year"] <= 2015].reset_index(drop=True)
    first = cls(calibrate=False).forecast_from_raw(train, 4)
    second = cls(calibrate=False).forecast_from_raw(train, 4)
    np.testing.assert_array_equal(first, second)


def test_uncalibrated_service_returns_null_bounds(tmp_path):
    service = XGBoostForecastService(models_dir=tmp_path, calibrate=False)
    rows = service.train_and_forecast(2)
    assert all(r["lower"] is None and r["upper"] is None for r in rows)
    assert service.uncertainty_info() is None


def test_cache_invalidation_on_data_and_config_change(tmp_path):
    data = tmp_path / "data.csv"
    shutil.copy(DEFAULT_DATA_PATH, data)
    service = XGBoostForecastService(
        data_path=data, models_dir=tmp_path, calibrate=False
    )
    service.train_and_forecast(2)
    old_hash = service.compute_data_hash()
    calibrated = XGBoostForecastService(data_path=data, models_dir=tmp_path)
    assert calibrated.compute_data_hash() != old_hash

    with open(data, "a", encoding="utf-8") as fh:
        fh.write("\n2026,500.0,84000000.0,,\n")
    calls = []
    original = service._fit_model
    service._fit_model = lambda history: calls.append(1) or original(history)
    service.load_data()
    service.train_model()
    assert calls and service.compute_data_hash() != old_hash
    assert service.forecast(1)[0]["year"] == 2027


def test_corrupt_artifact_triggers_retrain(tmp_path):
    service = XGBoostForecastService(models_dir=tmp_path, calibrate=False)
    service.model_path.write_bytes(b"not a model")
    service.load_data()
    service.train_model()
    assert service.is_ready


def test_hash_ignores_line_endings(tmp_path):
    crlf = tmp_path / "crlf.csv"
    lf = DEFAULT_DATA_PATH.read_bytes().replace(b"\r\n", b"\n")
    crlf.write_bytes(lf.replace(b"\n", b"\r\n"))
    a = XGBoostForecastService(models_dir=tmp_path).compute_data_hash()
    b = XGBoostForecastService(data_path=crlf, models_dir=tmp_path).compute_data_hash()
    assert a == b


@pytest.mark.parametrize("years", [0, -1, MAX_YEARS_AHEAD + 1, 2.5, True])
def test_invalid_years_ahead(years, tmp_path):
    service = XGBoostForecastService(models_dir=tmp_path)
    with pytest.raises(ValueError):
        service.forecast(years)
