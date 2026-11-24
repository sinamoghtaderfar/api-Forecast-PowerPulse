"""
Prediction-interval methods evaluated sequentially (no look-ahead).

For a forecast made at ``origin`` for horizon ``h`` an interval may only use
backtest residuals whose target year is <= origin.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app.services.uncertainty import conformal_half_width, volatility_scale

LEVEL = 0.95
RECENCY_HALF_LIFE = 8.0
# Volatility-scaled variants tried after the pre-registered methods failed
# (window of annual changes for sigma, largest pooled horizon). "scaled_w10_p10"
# is the one adopted in production (app.services.uncertainty).
SCALED_VARIANTS = {
    "scaled_w5_pall": (5, 99),
    "scaled_w10_pall": (10, 99),
    "scaled_w10_p5": (10, 5),
    "scaled_w10_p10": (10, 10),
}


def _monotone_width_per_horizon(known: pd.DataFrame, horizon: int, level: float):
    """Current production rule: per-horizon conformal quantile, non-decreasing in h."""
    running, width = 0.0, None
    for h in range(1, horizon + 1):
        res = known.loc[known["horizon"] == h, "residual"]
        w = conformal_half_width(res, level)
        if w is None:
            return None
        running = max(running, w)
        width = running
    return width


def _pooled_sqrt_width(known: pd.DataFrame, horizon: int, level: float):
    scaled = np.abs(known["residual"]) / np.sqrt(known["horizon"])
    q = conformal_half_width(scaled, level)
    return None if q is None else q * math.sqrt(horizon)


def weighted_conformal_quantile(
    scores: np.ndarray, weights: np.ndarray, level: float, test_weight: float = 1.0
):
    """
    Non-exchangeable split-conformal quantile (Barber et al., 2023): smallest
    score q with sum(w_i * [s_i <= q]) / (sum(w) + w_test) >= level; None when
    the test-point mass alone prevents reaching ``level``.
    """
    order = np.argsort(scores)
    s, w = scores[order], weights[order]
    total = w.sum() + test_weight
    cumulative = np.cumsum(w) / total
    idx = np.searchsorted(cumulative, level - 1e-12)
    return None if idx >= len(s) else float(s[idx])


def _pooled_sqrt_recent_width(
    known: pd.DataFrame, horizon: int, level: float, origin: int
):
    if known.empty:
        return None
    scaled = (np.abs(known["residual"]) / np.sqrt(known["horizon"])).to_numpy()
    age = (origin - known["year"]).to_numpy(dtype=float)
    weights = 0.5 ** (age / RECENCY_HALF_LIFE)
    q = weighted_conformal_quantile(scaled, weights, level)
    return None if q is None else q * math.sqrt(horizon)


METHODS = (
    "conformal_h",
    "conformal_pooled_sqrt",
    "conformal_pooled_sqrt_recent",
    *SCALED_VARIANTS,
)


def _scaled_width(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    known, rec, sigma, window, pool, level
):
    """Volatility-scaled pooled conformal half-width (see app.services.uncertainty)."""
    pooled = known[known["horizon"] <= pool]
    if pooled.empty:
        return None
    scores = np.abs(pooled["residual"]) / (
        pooled["origin"].map(lambda o: sigma[(window, o)]) * np.sqrt(pooled["horizon"])
    )
    q = conformal_half_width(scores, level)
    return (
        None if q is None else q * sigma[(window, rec.origin)] * math.sqrt(rec.horizon)
    )


def sequential_intervals(
    records: pd.DataFrame,
    target: pd.Series,
    level: float = LEVEL,
    max_pool_horizon: int = 25,
) -> pd.DataFrame:
    """
    For every record (one model) compute the half-width each method would have
    produced at its origin and whether the actual value was covered.
    ``target`` (indexed by year) is only read up to each origin for sigma.
    """
    records = records.copy()
    records["residual"] = records["actual"] - records["predicted"]
    sigma = {
        (w, o): volatility_scale(target.loc[:o].to_numpy(), w)
        for w in {v[0] for v in SCALED_VARIANTS.values()}
        for o in records["origin"].unique()
    }
    rows = []
    for rec in records.itertuples(index=False):
        all_known = records[records["year"] <= rec.origin]
        known = all_known[all_known["horizon"] <= max_pool_horizon]
        widths = {
            "conformal_h": _monotone_width_per_horizon(known, rec.horizon, level),
            "conformal_pooled_sqrt": _pooled_sqrt_width(known, rec.horizon, level),
            "conformal_pooled_sqrt_recent": _pooled_sqrt_recent_width(
                known, rec.horizon, level, rec.origin
            ),
        }
        for name, (window, pool) in SCALED_VARIANTS.items():
            widths[name] = _scaled_width(all_known, rec, sigma, window, pool, level)
        for method, width in widths.items():
            if width is None:
                continue
            rows.append(
                {
                    "method": method,
                    "origin": rec.origin,
                    "horizon": rec.horizon,
                    "year": rec.year,
                    "half_width": width,
                    "covered": abs(rec.actual - rec.predicted) <= width,
                }
            )
    return pd.DataFrame(rows)


def coverage_summary(
    intervals: pd.DataFrame, first_origin: int = 2000, max_h: int = 10
) -> dict:
    """Coverage/width per method and horizon on all and on common (origin, h) pairs."""
    scored = intervals[
        (intervals["origin"] >= first_origin) & (intervals["horizon"] <= max_h)
    ]
    out: dict = {"all": {}, "common": {}}
    for method, sub in scored.groupby("method"):
        out["all"][method] = {
            int(h): {
                "n": int(len(g)),
                "coverage": float(g["covered"].mean()),
                "mean_width": float(2 * g["half_width"].mean()),
            }
            for h, g in sub.groupby("horizon")
        }
    keys = scored.groupby(["origin", "horizon"])["method"].nunique()
    common_keys = keys[keys == scored["method"].nunique()].index
    common = scored.set_index(["origin", "horizon"]).loc[common_keys].reset_index()
    for method, sub in common.groupby("method"):
        out["common"][method] = {
            int(h): {
                "n": int(len(g)),
                "coverage": float(g["covered"].mean()),
                "mean_width": float(2 * g["half_width"].mean()),
            }
            for h, g in sub.groupby("horizon")
        }
    return out
