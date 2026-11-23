import math

import pytest

from app.dependencies import SERVICES

ROW_FIELDS = {
    "prophet": {"year", "forecast", "lower", "upper"},
    "random_forest": {
        "year",
        "forecast",
        "lower",
        "upper",
        "uncertainty_width",
        "growth_rate",
    },
    "xgboost": {"year", "forecast", "lower", "upper", "growth_from_previous"},
    # Added in API 1.2 (additive): the persistence benchmark.
    "persistence": {"year", "forecast", "lower", "upper"},
}


def test_root(client):
    body = client.get("/").json()
    assert body["status"] == "running"
    for path in body["endpoints"].values():
        assert client.get(path, params={"years_ahead": 2}).status_code == 200


def test_health_endpoints(client):
    body = client.get("/health").json()
    assert body["models_loaded"] is True
    assert body["status"] == "healthy"
    assert client.get("/health/live").json() == {"status": "alive"}
    ready = client.get("/health/ready")
    assert ready.status_code == 200 and ready.json()["status"] == "ready"


@pytest.mark.parametrize("years", [1, 3, 10])
def test_all_forecasts(client, years):
    response = client.get("/forecast/all", params={"years_ahead": years})
    assert response.status_code == 200
    body = response.json()
    assert set(body["forecasts"]) == set(ROW_FIELDS)
    for key, rows in body["forecasts"].items():
        assert len(rows) == years
        assert [r["year"] for r in rows] == list(range(2026, 2026 + years))
        for row in rows:
            assert set(row) == ROW_FIELDS[key]
            assert math.isfinite(row["forecast"])
            assert row["lower"] <= row["forecast"] <= row["upper"]
        summary = body["summaries"][key]
        assert summary["period"] == f"{rows[0]['year']}-{rows[-1]['year']}"
        assert summary["growth_periods"] == years - 1
        assert (summary["avg_annual_growth"] is None) == (years == 1)
    meta = body["metadata"]
    assert meta["years_ahead"] == years
    assert meta["units"]["target"] == "electricity_generation"
    assert meta["units"]["forecast"] == "TWh"
    assert meta["unavailable_models"] == []
    assert meta["intervals"]["xgboost"]["nominal_coverage"] == 0.95


def test_default_horizon(client):
    body = client.get("/forecast/all").json()
    assert all(len(rows) == 10 for rows in body["forecasts"].values())


def test_beyond_calibrated_horizon_bounds_are_null(client):
    body = client.get("/forecast/xgboost", params={"years_ahead": 15}).json()
    max_h = body["metadata"]["interval"]["max_calibrated_horizon"]
    assert all(r["lower"] is not None for r in body["forecast"][:max_h])
    assert all(
        r["lower"] is None and r["upper"] is None for r in body["forecast"][max_h:]
    )


@pytest.mark.parametrize("value", ["0", "-3", "51", "abc", "2.5"])
def test_invalid_years_ahead(client, value):
    assert client.get("/forecast/all", params={"years_ahead": value}).status_code == 422
    assert (
        client.get("/forecast/prophet", params={"years_ahead": value}).status_code
        == 422
    )


@pytest.mark.parametrize(
    "path, key",
    [
        ("prophet", "prophet"),
        ("random-forest", "random_forest"),
        ("xgboost", "xgboost"),
    ],
)
def test_single_model(client, path, key):
    body = client.get(f"/forecast/{path}", params={"years_ahead": 3}).json()
    assert body["model"] == key
    assert len(body["forecast"]) == 3
    assert set(body["forecast"][0]) == ROW_FIELDS[key]
    assert body["summary"]["growth_periods"] == 2
    assert body["metadata"]["units"]["forecast"] == "TWh"


def test_unavailable_model(client, monkeypatch):
    monkeypatch.setattr(SERVICES["random_forest"], "model", None)
    assert client.get("/forecast/random-forest").status_code == 503
    body = client.get("/forecast/all", params={"years_ahead": 2}).json()
    assert body["forecasts"]["random_forest"] == []
    assert body["summaries"]["random_forest"] is None
    assert body["metadata"]["unavailable_models"] == ["random_forest"]
    assert len(body["forecasts"]["prophet"]) == 2
    assert client.get("/health").json()["status"] == "degraded"
    assert client.get("/health/ready").status_code == 503
    assert client.get("/health/live").status_code == 200


def test_all_models_unavailable(client, monkeypatch):
    for service in SERVICES.values():
        monkeypatch.setattr(service, "model", None)
    response = client.get("/forecast/all")
    assert response.status_code == 503


def test_internal_errors_are_not_leaked(client, monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("secret internal path /srv/x")

    monkeypatch.setattr(SERVICES["xgboost"], "forecast", boom)
    response = client.get("/forecast/xgboost")
    assert response.status_code == 500
    assert "secret" not in response.text
