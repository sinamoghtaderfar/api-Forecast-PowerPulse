import numpy as np
import pandas as pd
import pytest

from app.services.base import DEFAULT_DATA_PATH
from app.services.features import (
    EXOG_COLUMNS,
    build_training_frame,
    carried_forward_years,
    compute_features,
    exogenous_growth_rates,
    exogenous_path,
    load_energy_data,
    prepare_history,
    recursive_forecast,
)
from app.services.random_forest_service import FEATURE_COLUMNS as RF_FEATURES


@pytest.fixture(scope="module")
def df():
    return load_energy_data(DEFAULT_DATA_PATH)


def _write(tmp_path, frame):
    path = tmp_path / "data.csv"
    frame.rename(columns={"y": "electricity_generation"}).to_csv(path, index=False)
    return path


def test_dataset_facts(raw):
    assert raw["year"].iloc[0] == 1985 and raw["year"].iloc[-1] == 2025
    assert (raw["year"].diff().dropna() == 1).all()
    # Values absent from the OWID source are kept missing, not invented.
    assert carried_forward_years(raw) == {
        "gdp": [2023, 2024, 2025],
        "renewables_share_energy": [2025],
    }


def test_load_data_clean(df):
    assert df["year"].is_monotonic_increasing
    assert not df.isna().any().any()
    assert (
        df.loc[df["year"] == 2025, "gdp"].item()
        == df.loc[df["year"] == 2022, "gdp"].item()
    )


def test_unsorted_input_is_sorted(tmp_path, df):
    loaded = load_energy_data(_write(tmp_path, df.iloc[::-1]))
    pd.testing.assert_frame_equal(loaded, df)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda f: pd.concat([f, f.iloc[[0]]]), "Duplicate"),
        (lambda f: f.drop(index=10), "consecutive"),
        (lambda f: f.assign(y=f["y"].where(f.index != 3, -1.0)), "positive"),
        (lambda f: f.assign(y=f["y"].where(f.index != 3, np.nan)), "positive"),
        (lambda f: f.drop(columns="gdp"), "Missing column"),
        (lambda f: f.iloc[:5], "Not enough"),
    ],
)
def test_malformed_input_rejected(tmp_path, df, mutate, message):
    with pytest.raises(ValueError, match=message):
        load_energy_data(_write(tmp_path, mutate(df.copy())))


def test_invalid_exogenous_values_are_treated_as_missing(tmp_path, df):
    bad = df.copy()
    bad.loc[5, "gdp"] = -1.0
    bad.loc[6, "renewables_share_energy"] = 150.0
    bad.loc[7, "population"] = np.inf
    loaded = load_energy_data(_write(tmp_path, bad))
    assert loaded.loc[5, "gdp"] == df.loc[4, "gdp"]
    assert (
        loaded.loc[6, "renewables_share_energy"] == df.loc[5, "renewables_share_energy"]
    )
    assert loaded.loc[7, "population"] == df.loc[6, "population"]


def test_leading_missing_exogenous_rows_are_dropped_not_backfilled(raw):
    shifted = raw.copy()
    shifted.loc[:1, "gdp"] = np.nan
    history = prepare_history(shifted)
    assert history["year"].iloc[0] == 1987


def test_growth_rates_use_only_observed_recent_values(raw):
    rates = exogenous_growth_rates(raw, window=10)
    observed = raw[(raw["year"] > 2015)].dropna(subset=["gdp"])
    expected = (observed["gdp"].iloc[-1] / observed["gdp"].iloc[0]) ** (1 / 6) - 1
    assert rates["gdp"] == pytest.approx(expected)
    at_2010 = exogenous_growth_rates(raw[raw["year"] <= 2010])
    window = raw[(raw["year"] > 2000) & (raw["year"] <= 2010)]
    assert at_2010["population"] == pytest.approx(
        (window["population"].iloc[-1] / window["population"].iloc[0]) ** (1 / 9) - 1
    )
    assert exogenous_growth_rates(raw.iloc[:1])["gdp"] == 0.0


def test_exogenous_projection(df):
    path = exogenous_path(df.iloc[-1], {c: 0.5 for c in EXOG_COLUMNS}, 3)
    assert path["year"].tolist() == [2026, 2027, 2028]
    assert (path["renewables_share_energy"] <= 100).all()
    assert path["population"].iloc[0] == pytest.approx(df["population"].iloc[-1] * 1.5)


def test_features_do_not_use_current_target(df):
    """Changing y_t must not change the features of year t (no leakage)."""
    X, y = build_training_frame(df, RF_FEATURES)
    modified = df.copy()
    modified.loc[modified.index[-1], "y"] *= 10
    X_mod, y_mod = build_training_frame(modified, RF_FEATURES)
    pd.testing.assert_frame_equal(X, X_mod)
    assert y.iloc[-1] != y_mod.iloc[-1]


def test_training_rows_have_full_history(df):
    X, y = build_training_frame(df, RF_FEATURES)
    assert len(X) == len(y) == len(df) - 5
    assert np.isfinite(X.to_numpy()).all()


def test_compute_features_slope_and_safe_division(df):
    hist = df["y"].to_numpy()
    kwargs = dict(
        year=2026, year_min=1985, population=8.4e7, renewables_share_energy=25.0
    )
    row = compute_features(hist, gdp=4e12, prev_gdp=3.9e12, **kwargs)
    assert row["trend_5yr"].iloc[0] == pytest.approx(
        np.polyfit(range(5), hist[-5:], 1)[0]
    )
    zero = compute_features(
        hist, gdp=4e12, prev_gdp=0.0, **{**kwargs, "population": 0.0}
    )
    assert np.isnan(zero["energy_intensity"].iloc[0])
    assert np.isnan(zero["gdp_per_capita"].iloc[0])


def test_recursive_forecast_feeds_predictions_back(df):
    exog = exogenous_path(df.iloc[-1], {c: 0.0 for c in EXOG_COLUMNS}, 3)
    seen = []

    def predict(features):
        seen.append(features["y_lag1"].iloc[0])
        return np.array([seen[-1] + 1.0])

    result = recursive_forecast(predict, df, exog, ["y_lag1"])
    last = df["y"].iloc[-1]
    assert seen == [last, last + 1, last + 2]
    assert result.tolist() == [last + 1, last + 2, last + 3]


def test_recursive_forecast_rejects_non_finite(df):
    exog = exogenous_path(df.iloc[-1], {c: 0.0 for c in EXOG_COLUMNS}, 1)
    with pytest.raises(ValueError, match="Non-finite prediction"):
        recursive_forecast(lambda f: np.array([np.nan]), df, exog, ["y_lag1"])
    exog.loc[0, "population"] = 0.0
    with pytest.raises(ValueError, match="Non-finite features"):
        recursive_forecast(lambda f: np.array([1.0]), df, exog, ["gdp_per_capita"])
