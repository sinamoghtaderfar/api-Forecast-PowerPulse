import shutil

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


@pytest.mark.parametrize("cls", SERVICE_CLASSES)
def test_forecast_persist_and_reload(cls, tmp_path):
    service = cls(models_dir=tmp_path)
    result = service.train_and_forecast(years_ahead=3)
    assert [row["year"] for row in result] == [
        service.last_year + 1,
        service.last_year + 2,
        service.last_year + 3,
    ]
    for row in result:
        assert row["lower"] <= row["forecast"] <= row["upper"]
    assert service.model_path.exists()

    reloaded = cls(models_dir=tmp_path)
    reloaded.load_data()
    reloaded.train_model()
    assert reloaded.is_ready

    # Cached results are copies, not shared mutable state.
    result[0]["forecast"] = -1
    assert service.forecast(3)[0]["forecast"] != -1


@pytest.mark.parametrize("cls", [RandomForestForecastService, XGBoostForecastService])
def test_tree_forecasts_deterministic(cls, tmp_path):
    first = cls(models_dir=tmp_path).train_and_forecast(5)
    second = cls(models_dir=tmp_path / "other").train_and_forecast(5)
    assert first == second


def test_retrains_when_data_changes(tmp_path):
    data = tmp_path / "data.csv"
    shutil.copy(DEFAULT_DATA_PATH, data)
    service = XGBoostForecastService(data_path=data, models_dir=tmp_path)
    service.train_and_forecast(2)
    old_hash = service.compute_data_hash()
    with open(data, "a", encoding="utf-8") as fh:
        fh.write("2026,500.0,84000000.0,3909612753939.0,24.5\n")
    service.load_data()
    service.train_model()
    assert service.compute_data_hash() != old_hash
    assert service.forecast(1)[0]["year"] == 2027


def test_hash_ignores_line_endings(tmp_path):
    crlf = tmp_path / "crlf.csv"
    crlf.write_bytes(
        DEFAULT_DATA_PATH.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    )
    a = XGBoostForecastService(models_dir=tmp_path).compute_data_hash()
    b = XGBoostForecastService(data_path=crlf, models_dir=tmp_path).compute_data_hash()
    assert a == b


@pytest.mark.parametrize("years", [0, -1, MAX_YEARS_AHEAD + 1, 2.5, True])
def test_invalid_years_ahead(years, tmp_path):
    service = XGBoostForecastService(models_dir=tmp_path)
    with pytest.raises(ValueError):
        service.forecast(years)
