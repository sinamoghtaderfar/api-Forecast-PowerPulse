import numpy as np
import pandas as pd
import pytest

from app.services.base import DEFAULT_DATA_PATH
from app.services.features import (
    average_growth_rates,
    build_training_frame,
    compute_features,
    load_energy_data,
)
from app.services.random_forest_service import FEATURE_COLUMNS as RF_FEATURES


@pytest.fixture(scope="module")
def df():
    return load_energy_data(DEFAULT_DATA_PATH)


def test_load_data_sorted_clean(df):
    assert df["year"].is_monotonic_increasing
    assert not df.isna().any().any()
    assert "y" in df.columns


def test_load_data_sorts_and_dedupes(tmp_path, df):
    raw = df.rename(columns={"y": "electricity_generation"})
    shuffled = pd.concat([raw.iloc[::-1], raw.iloc[[0]]])
    path = tmp_path / "data.csv"
    shuffled.to_csv(path, index=False)
    loaded = load_energy_data(path)
    pd.testing.assert_frame_equal(loaded, df)


def test_load_data_missing_column(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"year": [2000], "gdp": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing column"):
        load_energy_data(path)


def test_features_do_not_use_current_target(df):
    """Changing y_t must not change the features of year t (no leakage)."""
    rates = average_growth_rates(df)
    X, y = build_training_frame(df, rates, RF_FEATURES)
    modified = df.copy()
    modified.loc[modified.index[-1], "y"] *= 10
    X_mod, y_mod = build_training_frame(modified, rates, RF_FEATURES)
    pd.testing.assert_frame_equal(X, X_mod)
    assert y.iloc[-1] != y_mod.iloc[-1]


def test_training_rows_have_full_history(df):
    rates = average_growth_rates(df)
    X, y = build_training_frame(df, rates, RF_FEATURES)
    assert len(X) == len(y) == len(df) - 5
    assert not X.isna().any().any()


def test_compute_features_vectorised_matches_single(df):
    rates = average_growth_rates(df)
    hist = df["y"].to_numpy()
    kwargs = dict(
        year=2026,
        year_min=1985,
        population=8.4e7,
        gdp=4e12,
        renewables_share_energy=25.0,
        prev_gdp=3.9e12,
        growth_rates=rates,
    )
    single = compute_features(hist, **kwargs)
    multi = compute_features(np.vstack([hist, hist]), **kwargs)
    pd.testing.assert_frame_equal(multi.iloc[[1]].reset_index(drop=True), single)
    expected_slope = np.polyfit(range(5), hist[-5:], 1)[0]
    assert single["trend_5yr"].iloc[0] == pytest.approx(expected_slope)
