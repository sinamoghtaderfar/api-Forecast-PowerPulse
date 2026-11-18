import pytest


def test_root(client):
    body = client.get("/").json()
    assert body["status"] == "running"
    for path in body["endpoints"].values():
        assert client.get(path, params={"years_ahead": 2}).status_code == 200


def test_health(client):
    body = client.get("/health").json()
    assert body["models_loaded"] is True
    assert body["status"] == "healthy"


def test_all_forecasts(client):
    response = client.get("/forecast/all", params={"years_ahead": 4})
    assert response.status_code == 200
    body = response.json()
    assert set(body["forecasts"]) == {"prophet", "random_forest", "xgboost"}
    for key, rows in body["forecasts"].items():
        assert len(rows) == 4
        assert {"year", "forecast", "lower", "upper"} <= rows[0].keys()
        summary = body["summaries"][key]
        assert summary["period"] == f"{rows[0]['year']}-{rows[-1]['year']}"
    assert body["metadata"]["years_ahead"] == 4


def test_default_horizon(client):
    body = client.get("/forecast/all").json()
    assert all(len(rows) == 10 for rows in body["forecasts"].values())


@pytest.mark.parametrize("value", ["0", "-3", "51", "abc"])
def test_invalid_years_ahead(client, value):
    response = client.get("/forecast/all", params={"years_ahead": value})
    assert response.status_code == 422


def test_single_model(client):
    body = client.get("/forecast/xgboost", params={"years_ahead": 3}).json()
    assert body["model"] == "xgboost"
    assert len(body["forecast"]) == 3
