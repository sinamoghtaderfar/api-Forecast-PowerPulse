"""
Shared data loading, exogenous projection and leakage-free feature engineering.

Target: annual German ``electricity_generation`` in TWh (Our World in Data
energy dataset). Exogenous variables: ``population`` (people), ``gdp``
(OWID GDP, constant international-$) and ``renewables_share_energy``
(% of primary energy).

Rules that keep training, backtesting and production inference consistent:

* Features of year ``t`` use target values strictly before ``t`` plus the
  exogenous values of year ``t``.
* Missing exogenous observations are carried forward from the last observed
  value only (never backward-filled from the future).
* Future exogenous values are projected with growth rates estimated from
  observations available at the forecast origin.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET_COLUMN = "electricity_generation"
EXOG_COLUMNS = ["population", "gdp", "renewables_share_energy"]
REQUIRED_COLUMNS = ["year", TARGET_COLUMN, *EXOG_COLUMNS]

LAGS = (1, 2, 3, 4, 5)
ROLLING_WINDOWS = (3, 5, 7)
TREND_WINDOW = 5
SEASONAL_PERIOD = 10
MIN_HISTORY_ROWS = max(LAGS) + 3
# Trailing window (years) for estimating exogenous growth at an origin.
GROWTH_WINDOW_YEARS = 10


def read_energy_csv(data_path: str | Path) -> pd.DataFrame:
    """
    Read and validate the raw dataset without imputing anything.

    Returns a frame sorted by year with columns ``year``, ``y`` and the
    exogenous columns. Missing or out-of-domain exogenous values are NaN.

    Raises:
        ValueError: missing columns, duplicate/non-consecutive years, or
            missing/non-positive target values.
    """
    df = pd.read_csv(data_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(missing)}")

    df = df[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    if df["year"].isna().any() or (df["year"] % 1 != 0).any():
        raise ValueError("Column 'year' must contain integer years")
    df["year"] = df["year"].astype(int)
    if df["year"].duplicated().any():
        raise ValueError("Duplicate years in dataset")
    df = df.sort_values("year").reset_index(drop=True)
    if (df["year"].diff().dropna() != 1).any():
        raise ValueError("Years must be consecutive (lag features assume annual steps)")

    target = df[TARGET_COLUMN]
    if target.isna().any() or (target <= 0).any():
        raise ValueError(
            f"'{TARGET_COLUMN}' must be present and positive for every year"
        )

    invalid = {
        "population": df["population"] <= 0,
        "gdp": df["gdp"] <= 0,
        "renewables_share_energy": ~df["renewables_share_energy"].between(0, 100),
    }
    for col, mask in invalid.items():
        mask = mask & df[col].notna()
        if mask.any():
            logger.warning("Ignoring %d out-of-range value(s) in '%s'", mask.sum(), col)
            df.loc[mask, col] = np.nan

    return df.rename(columns={TARGET_COLUMN: "y"})


def prepare_history(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Impute exogenous gaps by carrying the last observation forward.

    Leading rows without any prior exogenous observation are dropped instead
    of being back-filled, so no value is taken from the future.
    """
    df = raw.copy()
    df[EXOG_COLUMNS] = df[EXOG_COLUMNS].ffill()
    df = df.dropna().reset_index(drop=True)
    if len(df) < MIN_HISTORY_ROWS:
        raise ValueError(
            f"Not enough usable rows in dataset ({len(df)}); "
            f"at least {MIN_HISTORY_ROWS} are required"
        )
    return df


def carried_forward_years(raw: pd.DataFrame) -> dict[str, list[int]]:
    """Years whose exogenous values are missing and carried forward, per column."""
    observed_before = raw[EXOG_COLUMNS].ffill().notna()
    missing = raw[EXOG_COLUMNS].isna() & observed_before
    return {
        col: raw.loc[missing[col], "year"].astype(int).tolist()
        for col in EXOG_COLUMNS
        if missing[col].any()
    }


def load_energy_data(data_path: str | Path) -> pd.DataFrame:
    """Read the dataset and return the imputed history used for modelling."""
    return prepare_history(read_energy_csv(data_path))


def exogenous_growth_rates(
    raw: pd.DataFrame, window: int = GROWTH_WINDOW_YEARS
) -> dict[str, float]:
    """
    Compound annual growth rate of each exogenous variable over the trailing
    ``window`` years, using observed (non-imputed) values only.

    Returns 0.0 for a variable without two positive observations in the window.
    """
    last_year = int(raw["year"].iloc[-1])
    recent = raw[raw["year"] > last_year - window]
    rates = {}
    for col in EXOG_COLUMNS:
        observed = recent[["year", col]].dropna()
        observed = observed[observed[col] > 0]
        if len(observed) < 2:
            rates[col] = 0.0
            continue
        first, last = observed.iloc[0], observed.iloc[-1]
        years = float(last["year"] - first["year"])
        rates[col] = float((last[col] / first[col]) ** (1.0 / years) - 1.0)
    return rates


def project_exogenous(
    last_row: pd.Series, growth_rates: dict[str, float], steps_ahead: int
) -> dict[str, float]:
    """Project exogenous variables ``steps_ahead`` years past ``last_row``."""
    projected = {
        col: float(last_row[col] * (1.0 + growth_rates[col]) ** steps_ahead)
        for col in EXOG_COLUMNS
    }
    projected["renewables_share_energy"] = min(
        projected["renewables_share_energy"], 100.0
    )
    return projected


def exogenous_path(
    last_row: pd.Series, growth_rates: dict[str, float], horizon: int
) -> pd.DataFrame:
    """Projected exogenous values for the next ``horizon`` years."""
    last_year = int(last_row["year"])
    rows = [
        {"year": last_year + step, **project_exogenous(last_row, growth_rates, step)}
        for step in range(1, horizon + 1)
    ]
    return pd.DataFrame(rows, columns=["year", *EXOG_COLUMNS])


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


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    return np.where(np.isfinite(result) & (denominator > 0), result, np.nan)


def compute_features(
    y_history: np.ndarray,
    *,
    year: int,
    year_min: int,
    population: float,
    gdp: float,
    renewables_share_energy: float,
    prev_gdp: float,
) -> pd.DataFrame:
    """
    Build the feature row for ``year``.

    Args:
        y_history: Target values of all years before ``year`` (oldest first),
            shape (n_past,) or (1, n_past).
        year: Year the features are built for.
        year_min: First year of the training data (anchor for time features).
        population, gdp, renewables_share_energy: Exogenous values for ``year``.
        prev_gdp: GDP of the previous year (used for lagged energy intensity).

    Returns:
        Single-row DataFrame. Features needing unavailable history are NaN.
    """
    y_history = np.asarray(y_history, dtype=float).reshape(1, -1)
    n_past = y_history.shape[1]
    nan = np.array([np.nan])

    def lag(k: int) -> np.ndarray:
        return y_history[:, -k].copy() if n_past >= k else nan.copy()

    year_index = float(year - year_min)
    feats: dict[str, np.ndarray] = {
        "year": np.array([float(year)]),
        "year_index": np.array([year_index]),
        "year_squared": np.array([year_index**2]),
        "year_cubic": np.array([year_index**3]),
        "year_sin": np.array([np.sin(2 * np.pi * year_index / SEASONAL_PERIOD)]),
        "year_cos": np.array([np.cos(2 * np.pi * year_index / SEASONAL_PERIOD)]),
    }

    for k in LAGS:
        feats[f"y_lag{k}"] = lag(k)
    feats["y_diff1"] = feats["y_lag1"] - feats["y_lag2"]
    feats["y_diff2"] = feats["y_lag1"] - 2 * feats["y_lag2"] + feats["y_lag3"]

    for window in ROLLING_WINDOWS:
        if n_past == 0:
            mean = std = vmin = vmax = nan
        else:
            values = y_history[:, -window:]
            mean = values.mean(axis=1)
            std = values.std(axis=1, ddof=1) if values.shape[1] > 1 else nan
            vmin, vmax = values.min(axis=1), values.max(axis=1)
        feats[f"y_rolling_mean_{window}"] = mean
        feats[f"y_rolling_std_{window}"] = std
        feats[f"y_rolling_min_{window}"] = vmin
        feats[f"y_rolling_max_{window}"] = vmax

    feats["trend_5yr"] = _window_slope(y_history[:, -TREND_WINDOW:]) if n_past else nan

    population_arr = np.array([float(population)])
    gdp_arr = np.array([float(gdp)])
    renewables = np.array([float(renewables_share_energy)])
    feats["population"] = population_arr
    feats["gdp"] = gdp_arr
    feats["renewables_share_energy"] = renewables
    feats["gdp_per_capita"] = _safe_divide(gdp_arr, population_arr)
    # Last observed intensity (previous year) to avoid leaking the current target.
    feats["energy_intensity"] = _safe_divide(
        feats["y_lag1"], np.array([float(prev_gdp)])
    )
    feats["renewables_impact"] = renewables * feats["gdp_per_capita"]

    return pd.DataFrame(feats)


def build_training_frame(
    df: pd.DataFrame, feature_names: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the supervised (features at t -> target y_t) training set."""
    y_values = df["y"].to_numpy(dtype=float)
    year_min = int(df["year"].iloc[0])
    rows = [
        compute_features(
            y_values[:t],
            year=int(df["year"].iloc[t]),
            year_min=year_min,
            population=df["population"].iloc[t],
            gdp=df["gdp"].iloc[t],
            renewables_share_energy=df["renewables_share_energy"].iloc[t],
            prev_gdp=df["gdp"].iloc[t - 1] if t > 0 else np.nan,
        )
        for t in range(len(df))
    ]
    features = pd.concat(rows, ignore_index=True)[feature_names]
    valid = features.notna().all(axis=1).to_numpy()
    return (
        features[valid].reset_index(drop=True),
        df["y"][valid].reset_index(drop=True),
    )


def recursive_forecast(
    predict: Callable[[pd.DataFrame], np.ndarray],
    history: pd.DataFrame,
    exog: pd.DataFrame,
    feature_names: list[str],
) -> np.ndarray:
    """
    Recursive multi-step forecast: each predicted year is appended to the
    target history used for the next year's features.
    """
    y_history = history["y"].to_numpy(dtype=float).tolist()
    year_min = int(history["year"].iloc[0])
    prev_gdp = float(history["gdp"].iloc[-1])
    predictions = []
    for row in exog.itertuples(index=False):
        features = compute_features(
            np.asarray(y_history),
            year=int(row.year),
            year_min=year_min,
            population=row.population,
            gdp=row.gdp,
            renewables_share_energy=row.renewables_share_energy,
            prev_gdp=prev_gdp,
        )[feature_names]
        if not np.isfinite(features.to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite features for year {row.year}")
        prediction = float(predict(features)[0])
        if not np.isfinite(prediction):
            raise ValueError(f"Non-finite prediction for year {row.year}")
        predictions.append(prediction)
        y_history.append(prediction)
        prev_gdp = float(row.gdp)
    return np.asarray(predictions)
