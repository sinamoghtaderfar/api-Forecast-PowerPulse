"""Tests of the persistence benchmark service, evidence metadata and v4 caching."""

import json

import numpy as np
import pytest

from app.evaluation import persistence_forecast
from app.services.backtesting import rolling_origin_backtest
from app.services.evidence import (
    BENCHMARK_MODEL,
    HORIZON_GUIDANCE,
    model_evidence,
    recommended_model,
)
from app.services.persistence_service import PersistenceForecastService
from app.services.xgboost_service import XGBoostForecastService


def test_persistence_forecast_is_last_observed_value(trained_services, raw):
    service = trained_services["persistence"]
    rows = service.forecast(12)
    last = float(raw["y"].iloc[-1])
    assert [r["year"] for r in rows] == list(range(2026, 2038))
    assert all(r["forecast"] == last for r in rows)
    assert service.uses_exogenous is False
    for row in rows[:10]:
        assert 0 <= row["lower"] < row["forecast"] < row["upper"]
    assert all(r["lower"] is None and r["upper"] is None for r in rows[10:])


def test_persistence_backtest_equals_evaluation_baseline(raw):
    service = PersistenceForecastService(calibrate=False)
    assert rolling_origin_backtest(
        service.forecast_from_raw, raw
    ) == rolling_origin_backtest(persistence_forecast, raw)


def test_persistence_artifact_is_json_and_reused(trained_services, models_dir):
    path = models_dir / PersistenceForecastService.model_filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["calibration"]["kind"] == "scaled"
    service = PersistenceForecastService(models_dir=models_dir)
    service.load_data()
    service.train_model()
    assert service.calibration == trained_services["persistence"].calibration


@pytest.mark.parametrize("key", ["prophet", "random_forest", "xgboost", "persistence"])
def test_long_horizons_are_finite_and_non_negative(trained_services, key):
    rows = trained_services[key].forecast(50)
    values = np.array([r["forecast"] for r in rows])
    assert len(rows) == 50 and np.isfinite(values).all() and (values >= 0).all()


def test_calibration_matches_scaled_method(trained_services):
    for service in trained_services.values():
        info = service.uncertainty_info()
        assert info["max_calibrated_horizon"] == 10
        assert info["validated_coverage"] is False
        assert 0 <= info["backtest_coverage"]["all"]["coverage"] <= 1
        widths = service.calibration.half_widths(10)
        assert np.allclose(
            np.array(widths) / np.sqrt(np.arange(1, 11)), widths[0], rtol=1e-12
        )


def test_evidence_is_relative_to_benchmark(trained_services):
    evidence = model_evidence(trained_services)
    bench = evidence[BENCHMARK_MODEL]
    assert bench["mean_relative_mae_h1_10"] == pytest.approx(1.0)
    xgb = evidence["xgboost"]["by_horizon"]
    assert xgb["1"]["relative_mae"] == pytest.approx(
        xgb["1"]["mae"] / bench["by_horizon"]["1"]["mae"]
    )
    # Origins 2000-2024: 25 one-year-ahead and 16 ten-year-ahead forecasts.
    assert bench["by_horizon"]["1"]["n"] == 25
    assert bench["by_horizon"]["10"]["n"] == 16


def test_recommendation_rule_prefers_benchmark_within_margin():
    def ev(score):
        return {"mean_relative_mae_h1_10": score}

    assert recommended_model({"persistence": ev(1.0), "x": ev(0.99)}) == "persistence"
    assert recommended_model({"persistence": ev(1.0), "x": ev(0.95)}) == "x"
    assert recommended_model({"x": ev(1.3), "y": ev(1.1)}) == "y"
    assert recommended_model({"x": None}) is None


def test_hash_changes_with_model_version(tmp_path, monkeypatch):
    service = XGBoostForecastService(models_dir=tmp_path)
    before = service.compute_data_hash()
    monkeypatch.setattr(XGBoostForecastService, "model_version", "3")
    assert service.compute_data_hash() != before


def test_legacy_calibration_artifact_triggers_retrain(tmp_path):
    service = PersistenceForecastService(models_dir=tmp_path)
    service.load_data()
    service.train_model()
    payload = json.loads(service.model_path.read_text(encoding="utf-8"))
    payload["calibration"] = {"level": 0.95, "residuals": {"1": [1.0] * 30}}
    service.model_path.write_text(json.dumps(payload), encoding="utf-8")

    fresh = PersistenceForecastService(models_dir=tmp_path)
    calls = []
    original = fresh._fit_model
    fresh._fit_model = lambda history: calls.append(1) or original(history)
    fresh.load_data()
    fresh.train_model()
    assert calls, "a legacy-format artifact must not be reused"
    assert fresh.calibration.max_calibrated_horizon == 10


def test_metadata_in_api(client):
    body = client.get("/forecast/all", params={"years_ahead": 12}).json()
    meta = body["metadata"]
    assert meta["benchmark_model"] == "persistence"
    assert meta["recommended_model"] in body["forecasts"]
    assert set(meta["evidence"]) == set(body["forecasts"])
    assert set(meta["horizon_guidance"]) == set(HORIZON_GUIDANCE)
    assert meta["assumptions"]["carried_forward_years"] == {
        "gdp": [2023, 2024, 2025],
        "renewables_share_energy": [2025],
    }
    for info in meta["intervals"].values():
        assert info["method"] and info["nominal_coverage"] == 0.95
        assert info["max_calibrated_horizon"] == 10
        assert set(info["residuals_per_horizon"]) >= {str(h) for h in range(1, 11)}
        assert info["validated_coverage"] is False
    for rows in body["forecasts"].values():
        assert all(r["lower"] is not None for r in rows[:10])
        assert all(r["lower"] is None for r in rows[10:])
    assert len(meta["units"]["uncertainty"]) <= 500


def test_persistence_endpoint(client):
    body = client.get("/forecast/persistence", params={"years_ahead": 4}).json()
    assert body["model"] == "persistence"
    assert len(body["forecast"]) == 4
    assert body["summary"]["total_growth_percent"] == 0.0
    assert body["metadata"]["evidence"]["mean_relative_mae_h1_10"] == pytest.approx(1)
    assert body["metadata"]["assumptions"] == {}
    xgb = client.get("/forecast/xgboost", params={"years_ahead": 4}).json()
    assert xgb["metadata"]["assumptions"]["models_using_exogenous"]


def test_recommendation_ignores_unavailable_models(client, monkeypatch):
    from app.dependencies import SERVICES  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(SERVICES["persistence"], "model", None)
    meta = client.get("/forecast/all", params={"years_ahead": 2}).json()["metadata"]
    assert meta["unavailable_models"] == ["persistence"]
    assert meta["recommended_model"] != "persistence"


def test_physical_floor_is_reported(client, trained_services):
    rows = trained_services["prophet"].forecast(50)
    floored = [r["year"] for r in rows if r["forecast"] == 0.0]
    meta = client.get("/forecast/all", params={"years_ahead": 50}).json()["metadata"]
    if floored:
        assert any(w.startswith("prophet:") for w in meta["warnings"])
    short = client.get("/forecast/all", params={"years_ahead": 5}).json()["metadata"]
    assert short["warnings"] == []
