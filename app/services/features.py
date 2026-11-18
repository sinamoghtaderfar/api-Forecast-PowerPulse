"""
Shared data loading and leakage-free feature engineering for the forecasting services.

Features for year ``t`` are computed only from target values strictly before ``t``
plus the exogenous variables (population, GDP, renewables share) of year ``t``.
The same function is used for training and for recursive multi-step forecasting,
so both always see identically constructed features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COLUMN = "electricity_generation"
EXOG_COLUMNS = ["population", "gdp", "renewables_share_energy"]
REQUIRED_COLUMNS = ["year", TARGET_COLUMN, *EXOG_COLUMNS]

LAGS = (1, 2, 3, 4, 5)
ROLLING_WINDOWS = (3, 5, 7)
TREND_WINDOW = 5
SEASONAL_PERIOD = 10


def load_energy_data(data_path: str | Path) -> pd.DataFrame:
    """
    Load, validate and clean the historical dataset.

    Returns a frame sorted by year with columns ``year``, ``y`` (target) and the
    exogenous columns, without missing values.
    """
    df = pd.read_csv(data_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(missing)}")

    df = df[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["year", TARGET_COLUMN])
    df["year"] = df["year"].astype(int)
    df = (
        df.sort_values("year")
        .drop_duplicates(subset="year", keep="last")
        .reset_index(drop=True)
    )
    df[EXOG_COLUMNS] = df[EXOG_COLUMNS].ffill().bfill()
    df = df.dropna().rename(columns={TARGET_COLUMN: "y"})

    if len(df) < max(LAGS) + 3:
        raise ValueError(
            f"Not enough usable rows in dataset ({len(df)}); "
            f"at least {max(LAGS) + 3} are required"
        )
    return df


def average_growth_rates(df: pd.DataFrame) -> dict[str, float]:
    """Mean historical year-over-year growth rates of target and exogenous columns."""
    return {
        "y_growth": float(df["y"].pct_change().mean()),
        "gdp_growth": float(df["gdp"].pct_change().mean()),
        "population_growth": float(df["population"].pct_change().mean()),
        "renewables_growth": float(df["renewables_share_energy"].pct_change().mean()),
    }


def _window_slope(window: np.ndarray) -> np.ndarray:
    """Least-squares slope per row (equivalent to ``np.polyfit(x, y, 1)[0]``)."""
    n = window.shape[1]
    if n < 2:
        return np.zeros(window.shape[0])
    x = np.arange(n, dtype=float)
    x_centered = x - x.mean()
    return (
        (window - window.mean(axis=1, keepdims=True))
        @ x_centered
        / (x_centered**2).sum()
    )


def compute_features(
    y_history: np.ndarray,
    *,
    year: int,
    year_min: int,
    population: np.ndarray | float,
    gdp: np.ndarray | float,
    renewables_share_energy: np.ndarray | float,
    prev_gdp: np.ndarray | float,
    growth_rates: dict[str, float],
) -> pd.DataFrame:
    """
    Build the feature rows for ``year`` for one or more independent series.

    Args:
        y_history: Array of shape (n_series, n_past) with target values of all
            years before ``year`` (oldest first).
        year: Year the features are built for.
        year_min: First year of the training data (anchor for time features).
        population, gdp, renewables_share_energy: Exogenous values for ``year``.
        prev_gdp: GDP of the previous year (used for lagged energy intensity).
        growth_rates: Output of :func:`average_growth_rates`.

    Returns:
        DataFrame with one row per series. Features that need more history than
        available are NaN.
    """
    y_history = np.atleast_2d(np.asarray(y_history, dtype=float))
    n_series, n_past = y_history.shape
    nan = np.full(n_series, np.nan)

    def full(value) -> np.ndarray:
        return np.broadcast_to(np.asarray(value, dtype=float), (n_series,)).copy()

    def lag(k: int) -> np.ndarray:
        return y_history[:, -k].copy() if n_past >= k else nan.copy()

    year_index = float(year - year_min)
    feats: dict[str, np.ndarray] = {
        "year": full(year),
        "year_index": full(year_index),
        "year_squared": full(year_index**2),
        "year_cubic": full(year_index**3),
        "year_sin": full(np.sin(2 * np.pi * year_index / SEASONAL_PERIOD)),
        "year_cos": full(np.cos(2 * np.pi * year_index / SEASONAL_PERIOD)),
    }

    for k in LAGS:
        feats[f"y_lag{k}"] = lag(k)
    feats["y_diff1"] = feats["y_lag1"] - feats["y_lag2"]
    feats["y_diff2"] = feats["y_lag1"] - 2 * feats["y_lag2"] + feats["y_lag3"]

    for window in ROLLING_WINDOWS:
        values = y_history[:, -window:] if n_past else None
        if values is None:
            mean = std = vmin = vmax = nan
        else:
            mean = values.mean(axis=1)
            std = values.std(axis=1, ddof=1) if values.shape[1] > 1 else nan
            vmin, vmax = values.min(axis=1), values.max(axis=1)
        feats[f"y_rolling_mean_{window}"] = mean
        feats[f"y_rolling_std_{window}"] = std
        feats[f"y_rolling_min_{window}"] = vmin
        feats[f"y_rolling_max_{window}"] = vmax

    feats["trend_5yr"] = (
        _window_slope(y_history[:, -TREND_WINDOW:]) if n_past else nan.copy()
    )

    population = full(population)
    gdp = full(gdp)
    renewables = full(renewables_share_energy)
    feats["population"] = population
    feats["gdp"] = gdp
    feats["renewables_share_energy"] = renewables
    feats["gdp_growth_feature"] = full(growth_rates["gdp_growth"])
    feats["population_growth_feature"] = full(growth_rates["population_growth"])
    feats["renewables_growth_feature"] = full(growth_rates["renewables_growth"])
    feats["gdp_per_capita"] = gdp / population
    # Last observed intensity (previous year) to avoid leaking the current target.
    feats["energy_intensity"] = feats["y_lag1"] / full(prev_gdp)
    feats["renewables_impact"] = renewables * feats["gdp_per_capita"]

    return pd.DataFrame(feats)


def build_training_frame(
    df: pd.DataFrame, growth_rates: dict[str, float], feature_names: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the supervised (features at t -> target y_t) training set."""
    y_values = df["y"].to_numpy(dtype=float)
    year_min = int(df["year"].iloc[0])
    rows = []
    for t in range(len(df)):
        row = df.iloc[t]
        rows.append(
            compute_features(
                y_values[:t].reshape(1, -1),
                year=int(row["year"]),
                year_min=year_min,
                population=row["population"],
                gdp=row["gdp"],
                renewables_share_energy=row["renewables_share_energy"],
                prev_gdp=df["gdp"].iloc[t - 1] if t > 0 else np.nan,
                growth_rates=growth_rates,
            )
        )
    features = pd.concat(rows, ignore_index=True)[feature_names]
    valid = features.notna().all(axis=1).to_numpy()
    return (
        features[valid].reset_index(drop=True),
        df["y"][valid].reset_index(drop=True),
    )


def project_exogenous(
    last_row: pd.Series,
    growth_rates: dict[str, float],
    steps_ahead: int,
    factors: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Project exogenous variables ``steps_ahead`` years past ``last_row``.

    ``factors`` has shape (n_series, 3) and scales the population, GDP and
    renewables growth rates per series.
    """
    pop = (
        last_row["population"]
        * (1 + growth_rates["population_growth"] * factors[:, 0]) ** steps_ahead
    )
    gdp = (
        last_row["gdp"]
        * (1 + growth_rates["gdp_growth"] * factors[:, 1]) ** steps_ahead
    )
    renew = (
        last_row["renewables_share_energy"]
        * (1 + growth_rates["renewables_growth"] * factors[:, 2]) ** steps_ahead
    )
    return {
        "population": pop,
        "gdp": gdp,
        "renewables_share_energy": np.minimum(renew, 100.0),
    }
