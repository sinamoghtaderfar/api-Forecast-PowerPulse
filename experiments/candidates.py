"""
Candidate forecasters for the accuracy experiments.

Every forecaster has the signature ``f(raw_up_to_origin, horizon) -> ndarray``
and may only use the rows it receives (all with ``year <= origin``). Model
selection inside a forecaster (ARIMA order, ridge penalty, tree grid) uses
only those rows, so the selection is nested inside each backtest origin.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from app.services.features import EXOG_COLUMNS, prepare_history

Forecaster = Callable[[pd.DataFrame, int], np.ndarray]
SEED = 42
BASE_COLUMNS = ["year", "y", *EXOG_COLUMNS]


# --------------------------------------------------------------------------
# Naive baselines
# --------------------------------------------------------------------------


def persistence(raw: pd.DataFrame, horizon: int) -> np.ndarray:
    """Last observed value."""
    return np.full(horizon, float(raw["y"].iloc[-1]))


def drift(raw: pd.DataFrame, horizon: int) -> np.ndarray:
    """Last value plus the mean annual change over the whole history."""
    y = raw["y"].to_numpy(dtype=float)
    return y[-1] + (y[-1] - y[0]) / (len(y) - 1) * np.arange(1, horizon + 1)


def drift_recent(window: int) -> Forecaster:
    """Last value plus the mean annual change over the last ``window`` years."""

    def forecast(raw: pd.DataFrame, horizon: int) -> np.ndarray:
        y = raw["y"].to_numpy(dtype=float)[-(window + 1) :]
        slope = (y[-1] - y[0]) / (len(y) - 1)
        return y[-1] + slope * np.arange(1, horizon + 1)

    return forecast


# --------------------------------------------------------------------------
# Exponential smoothing (ETS with additive errors), estimated by SSE
# --------------------------------------------------------------------------


def _ets_filter(
    y, alpha, beta, phi, level0, trend0
):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    level, trend = level0, trend0
    sse = 0.0
    for value in y:
        fitted = level + phi * trend
        error = value - fitted
        sse += error * error
        level = fitted + alpha * error
        trend = phi * trend + beta * error
    return sse, level, trend


def fit_ets(y: np.ndarray, trend: bool, damped_trend: bool) -> dict:
    """
    Fit SES (``trend=False``), Holt or damped-trend ETS by least squares.

    Parameters: alpha in (0, 1), beta = alpha * beta_star (beta_star in (0, 1)),
    phi in [0.8, 0.98] for the damped model (1 for Holt), initial level and
    trend. Several deterministic starting points guard against local minima.
    """
    y = np.asarray(y, dtype=float)
    scale = max(np.std(y), 1.0)
    b_init = float(np.mean(np.diff(y[: min(len(y), 6)]))) if trend else 0.0

    def unpack(theta):  # pylint: disable=redefined-outer-name
        alpha = theta[0]
        beta = alpha * theta[1] if trend else 0.0
        phi = theta[2] if damped_trend else (1.0 if trend else 0.0)
        level0 = theta[3]
        trend0 = theta[4] if trend else 0.0
        return alpha, beta, phi, level0, trend0

    def objective(theta):  # pylint: disable=redefined-outer-name
        return _ets_filter(y, *unpack(theta))[0] / scale**2

    bounds = [
        (1e-4, 0.9999),
        (1e-4, 0.9999),
        (0.8, 0.98),
        (None, None),
        (None, None),
    ]
    best = None
    for alpha0 in (0.1, 0.3, 0.5, 0.7, 0.9):
        for beta0 in (0.01, 0.1, 0.3) if trend else (0.05,):
            for phi0 in (0.85, 0.95) if damped_trend else (0.9,):
                x0 = [alpha0, beta0, phi0, y[0], b_init]
                res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
                if best is None or res.fun < best.fun:
                    best = res
    alpha, beta, phi, level0, trend0 = unpack(best.x)
    sse, level, last_trend = _ets_filter(y, alpha, beta, phi, level0, trend0)
    return {
        "alpha": alpha,
        "beta": beta,
        "phi": phi,
        "level": level,
        "trend": last_trend,
        "sse": sse,
        "n": len(y),
        "trend_component": trend,
        "damped": damped_trend,
    }


def ets_forecast(fit: dict, horizon: int) -> np.ndarray:
    """Point forecasts of a fitted ETS model."""
    steps = np.arange(1, horizon + 1)
    if not fit["trend_component"]:
        return np.full(horizon, fit["level"])
    phi = fit["phi"]
    if fit["damped"]:
        multiplier = np.cumsum(phi**steps)
    else:
        multiplier = steps.astype(float)
    return fit["level"] + multiplier * fit["trend"]


def ses(raw: pd.DataFrame, horizon: int) -> np.ndarray:
    """Simple exponential smoothing."""
    return ets_forecast(fit_ets(raw["y"].to_numpy(), False, False), horizon)


def holt(raw: pd.DataFrame, horizon: int) -> np.ndarray:
    """Holt's linear trend."""
    return ets_forecast(fit_ets(raw["y"].to_numpy(), True, False), horizon)


def damped(raw: pd.DataFrame, horizon: int) -> np.ndarray:
    """Additive damped trend (ETS A,Ad,N)."""
    return ets_forecast(fit_ets(raw["y"].to_numpy(), True, True), horizon)


def theta(raw: pd.DataFrame, horizon: int) -> np.ndarray:
    """
    Standard Theta method (theta = 2), Hyndman & Billah (2003) form:
    SES forecast plus half the OLS slope of y on time with the SES drift term.
    """
    y = raw["y"].to_numpy(dtype=float)
    n = len(y)
    fit = fit_ets(y, False, False)
    alpha = fit["alpha"]
    slope = np.polyfit(np.arange(n), y, 1)[0]
    steps = np.arange(1, horizon + 1)
    drift_term = 0.5 * slope * (steps - 1 + 1 / alpha - (1 - alpha) ** n / alpha)
    return fit["level"] + drift_term


def comb_shd(raw: pd.DataFrame, horizon: int) -> np.ndarray:
    """Equal-weight combination of SES, Holt and damped trend (M3 Comb S-H-D)."""
    return (ses(raw, horizon) + holt(raw, horizon) + damped(raw, horizon)) / 3.0


# --------------------------------------------------------------------------
# ARIMA with nested AICc order selection (statsmodels, experiments only)
# --------------------------------------------------------------------------


def arima_auto(raw: pd.DataFrame, horizon: int) -> np.ndarray:
    """ARIMA(p,1,q) +/- drift, order chosen by AICc on the training rows only."""
    from statsmodels.tsa.arima.model import (  # pylint: disable=import-outside-toplevel
        ARIMA,
    )

    y = raw["y"].to_numpy(dtype=float)
    best_aicc, best_res = np.inf, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in (0, 1, 2):
            for q in (0, 1):
                for trend in ("n", "t"):
                    try:
                        res = ARIMA(y, order=(p, 1, q), trend=trend).fit()
                    except Exception:  # pylint: disable=broad-except
                        continue
                    if np.isfinite(res.aicc) and res.aicc < best_aicc:
                        best_aicc, best_res = res.aicc, res
    if best_res is None:
        return persistence(raw, horizon)
    return np.asarray(best_res.forecast(horizon), dtype=float)


# --------------------------------------------------------------------------
# Linear ML
# --------------------------------------------------------------------------

RIDGE_ALPHAS = np.logspace(-2, 3, 11)


def _chronological_ridge(X: np.ndarray, y: np.ndarray, min_train: int = 6):
    """Ridge with the penalty chosen by expanding-window one-step CV (no shuffling)."""
    n = len(y)
    best_alpha, best_err = RIDGE_ALPHAS[-1], np.inf
    folds = list(range(max(min_train, n - 6), n))
    if len(folds) >= 2:
        for alpha in RIDGE_ALPHAS:
            errors = []
            for t in folds:
                model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                model.fit(X[:t], y[:t])
                errors.append(abs(model.predict(X[t : t + 1])[0] - y[t]))
            err = float(np.mean(errors))
            if err < best_err:
                best_alpha, best_err = alpha, err
    model = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
    return model.fit(X, y)


AR_LAGS = 3


def ridge_ar_diff(raw: pd.DataFrame, horizon: int) -> np.ndarray:
    """Recursive ridge AR(3) on annual changes."""
    y = raw["y"].to_numpy(dtype=float)
    d = np.diff(y)
    X = np.column_stack(
        [d[AR_LAGS - k - 1 : len(d) - k - 1] for k in range(AR_LAGS)]
    )  # lag1, lag2, lag3
    target = d[AR_LAGS:]
    model = _chronological_ridge(X, target)
    history = list(d)
    level = y[-1]
    out = []
    for _ in range(horizon):
        features = np.array([[history[-1 - k] for k in range(AR_LAGS)]])
        change = float(model.predict(features)[0])
        level += change
        history.append(change)
        out.append(level)
    return np.asarray(out)


MIN_DIRECT_ROWS = 8


def direct_ridge(raw: pd.DataFrame, horizon: int) -> np.ndarray:
    """
    Direct multi-horizon ridge: one model per h for y_{t+h} - y_t on
    (dy_t, dy_{t-1}, dy_{t-2}). Horizons with fewer than 8 training rows reuse
    the forecast change of the longest estimable horizon.
    """
    y = raw["y"].to_numpy(dtype=float)
    n = len(y)
    d = np.concatenate([[np.nan], np.diff(y)])

    def features(t):
        return [d[t], d[t - 1], d[t - 2]]

    x_now = np.array([features(n - 1)])
    out, last_change = [], 0.0
    for h in range(1, horizon + 1):
        rows = list(range(3, n - h))
        if len(rows) >= MIN_DIRECT_ROWS:
            X = np.array([features(t) for t in rows])
            target = np.array([y[t + h] - y[t] for t in rows])
            model = _chronological_ridge(X, target)
            last_change = float(model.predict(x_now)[0])
        out.append(y[-1] + last_change)
    return np.asarray(out)


def ardl_gdp(project: Callable[[pd.DataFrame, int], pd.DataFrame]) -> Forecaster:
    """Ridge on dlog(y)_t ~ dlog(gdp)_t + dlog(y)_{t-1}; future GDP is projected."""

    def forecast(raw: pd.DataFrame, horizon: int) -> np.ndarray:
        hist = prepare_history(raw[BASE_COLUMNS])
        ly = np.log(hist["y"].to_numpy(dtype=float))
        lg = np.log(hist["gdp"].to_numpy(dtype=float))
        dly, dlg = np.diff(ly), np.diff(lg)
        X = np.column_stack([dlg[1:], dly[:-1]])
        model = _chronological_ridge(X, dly[1:])
        future = project(raw[BASE_COLUMNS], horizon)
        gdp_path = np.concatenate([[lg[-1]], np.log(future["gdp"].to_numpy())])
        level, prev = ly[-1], dly[-1]
        out = []
        for step in range(horizon):
            change = float(
                model.predict(np.array([[gdp_path[step + 1] - gdp_path[step], prev]]))[
                    0
                ]
            )
            level += change
            prev = change
            out.append(np.exp(level))
        return np.asarray(out)

    return forecast


# --------------------------------------------------------------------------
# Tree models on annual changes (can extrapolate the level)
# --------------------------------------------------------------------------


def _diff_feature_row(y_hist: np.ndarray, exog_now: dict, exog_prev: dict) -> list:
    d = np.diff(y_hist)
    return [
        d[-1],
        d[-2],
        d[-3],
        d[-3:].mean(),
        d[-5:].mean(),
        y_hist[-1] - y_hist[-5:].mean(),
        exog_now["gdp"] / exog_prev["gdp"] - 1.0,
        exog_now["population"] / exog_prev["population"] - 1.0,
        exog_now["renewables_share_energy"] - exog_prev["renewables_share_energy"],
    ]


MIN_DIFF_HISTORY = 6


def _diff_training(hist: pd.DataFrame):
    y = hist["y"].to_numpy(dtype=float)
    exog = hist[EXOG_COLUMNS].to_dict("records")
    X, target = [], []
    for t in range(MIN_DIFF_HISTORY, len(y)):
        X.append(_diff_feature_row(y[:t], exog[t], exog[t - 1]))
        target.append(y[t] - y[t - 1])
    return np.asarray(X), np.asarray(target)


def _recursive_diff_forecast(model, hist, future):
    y_hist = list(hist["y"].to_numpy(dtype=float))
    prev = hist[EXOG_COLUMNS].iloc[-1].to_dict()
    out = []
    for row in future[EXOG_COLUMNS].to_dict("records"):
        x = np.array([_diff_feature_row(np.asarray(y_hist), row, prev)])
        level = y_hist[-1] + float(model.predict(x)[0])
        y_hist.append(level)
        out.append(level)
        prev = row
    return np.asarray(out)


def tree_diff(
    make_model: Callable[[], object],
    project: Callable[[pd.DataFrame, int], pd.DataFrame],
) -> Forecaster:
    """Tree model of the annual change with recursive level reconstruction."""

    def forecast(raw: pd.DataFrame, horizon: int) -> np.ndarray:
        hist = prepare_history(raw[BASE_COLUMNS])
        X, target = _diff_training(hist)
        model = make_model()
        model.fit(X, target)
        return _recursive_diff_forecast(
            model, hist, project(raw[BASE_COLUMNS], horizon)
        )

    return forecast


def make_xgb_diff(max_depth: int = 2, n_estimators: int = 200) -> XGBRegressor:
    """Shallow, regularised XGBoost for annual changes."""
    return XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=max_depth,
        subsample=0.8,
        min_child_weight=2,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=1,
    )


def make_rf_diff() -> RandomForestRegressor:
    """Shallow random forest for annual changes."""
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=3,
        min_samples_leaf=3,
        max_features=0.5,
        random_state=SEED,
        n_jobs=1,
    )


XGB_GRID = [(d, n) for d in (1, 2, 3) for n in (50, 200)]


def xgb_diff_tuned(project: Callable[[pd.DataFrame, int], pd.DataFrame]) -> Forecaster:
    """XGBoost on changes; depth/trees chosen by inner one-step chronological CV."""

    def forecast(raw: pd.DataFrame, horizon: int) -> np.ndarray:
        hist = prepare_history(raw[BASE_COLUMNS])
        X, target = _diff_training(hist)
        n = len(target)
        folds = list(range(max(6, n - 5), n))
        best, best_err = XGB_GRID[3], np.inf
        if len(folds) >= 2:
            for depth, trees in XGB_GRID:
                errs = []
                for t in folds:
                    model = make_xgb_diff(depth, trees).fit(X[:t], target[:t])
                    errs.append(abs(float(model.predict(X[t : t + 1])[0]) - target[t]))
                if np.mean(errs) < best_err:
                    best, best_err = (depth, trees), float(np.mean(errs))
        model = make_xgb_diff(*best).fit(X, target)
        return _recursive_diff_forecast(
            model, hist, project(raw[BASE_COLUMNS], horizon)
        )

    return forecast


# --------------------------------------------------------------------------
# Decomposition: generation = demand - net imports
# --------------------------------------------------------------------------

MIN_COMPONENT_YEARS = 8


def decomp_damped(raw: pd.DataFrame, horizon: int) -> np.ndarray:
    """Damped-trend demand minus persistent net imports (1990+ data only)."""
    comp = raw.dropna(subset=["demand", "net_imports"])
    if len(comp) < MIN_COMPONENT_YEARS:
        return persistence(raw, horizon)
    demand = ets_forecast(fit_ets(comp["demand"].to_numpy(), True, True), horizon)
    return demand - float(comp["net_imports"].iloc[-1])
