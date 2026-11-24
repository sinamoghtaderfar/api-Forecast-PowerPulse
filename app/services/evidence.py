"""
Forecast evidence and assumptions reported in the API metadata.

Everything here is derived from the rolling-origin backtests that the
services already run for interval calibration (origins >= 2000, horizons
1..10), so the numbers describe genuine out-of-sample forecasts of the
current data and code.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from app.services.features import (
    EXOG_COLUMNS,
    GROWTH_WINDOW_YEARS,
    carried_forward_years,
)
from app.services.uncertainty import DEFAULT_MAX_HORIZON

BENCHMARK_MODEL = "persistence"
# A model must beat the benchmark's mean MAE over horizons 1-10 by this
# relative margin to be recommended over it (simpler model wins ties).
RECOMMENDATION_MARGIN = 0.02

HORIZON_GUIDANCE: dict[str, dict[str, str]] = {
    "near_term": {
        "years": "1-3",
        "evidence": "backtested on 23-25 forecast origins",
    },
    "medium_term": {
        "years": "4-5",
        "evidence": "backtested on 21-22 origins; overlapping windows",
    },
    "long_term": {
        "years": "6-10",
        "evidence": (
            "weak: 16-20 heavily overlapping origins; long-term accuracy is "
            "not established"
        ),
    },
    "exploratory": {
        "years": f">{DEFAULT_MAX_HORIZON}",
        "evidence": (
            "not validated; no prediction interval; values are mechanical "
            "extrapolations, not forecasts with known accuracy"
        ),
    },
}


def model_evidence(services: Mapping[str, Any]) -> dict[str, dict[str, Any] | None]:
    """
    Backtest MAE per horizon for each calibrated service and its MAE relative
    to the benchmark on the same forecasts (``relative_mae < 1`` = better).
    """
    maes = {
        key: (service.calibration.backtest_mae() if service.calibration else None)
        for key, service in services.items()
    }
    bench = maes.get(BENCHMARK_MODEL)
    out: dict[str, dict[str, Any] | None] = {}
    for key, by_h in maes.items():
        if not by_h:
            out[key] = None
            continue
        rows = {}
        for h, stats in by_h.items():
            row = dict(stats)
            if bench and h in bench and bench[h]["mae"] > 0:
                row["relative_mae"] = stats["mae"] / bench[h]["mae"]
            rows[str(h)] = row
        rel = [r["relative_mae"] for r in rows.values() if "relative_mae" in r]
        out[key] = {
            "first_origin": 2000,
            "by_horizon": rows,
            "mean_relative_mae_h1_10": float(np.mean(rel)) if rel else None,
        }
    return out


def recommended_model(evidence: Mapping[str, dict[str, Any] | None]) -> str | None:
    """
    Model with the lowest mean relative MAE over horizons 1-10; the benchmark
    unless another model beats it by more than ``RECOMMENDATION_MARGIN``.
    """
    scores = {
        key: ev["mean_relative_mae_h1_10"]
        for key, ev in evidence.items()
        if ev and ev.get("mean_relative_mae_h1_10") is not None
    }
    if not scores:
        return None
    best = min(scores, key=scores.get)
    if BENCHMARK_MODEL in scores and scores[best] > 1 - RECOMMENDATION_MARGIN:
        return BENCHMARK_MODEL
    return best


def assumptions(services: Mapping[str, Any]) -> dict[str, Any]:
    """Exogenous projection assumptions and data gaps behind the forecasts."""
    reference = next(
        (s for s in services.values() if s.raw is not None and s.uses_exogenous), None
    )
    if reference is None:
        return {}
    return {
        "last_observed_year": int(reference.last_year),
        "exogenous_projection": (
            f"compound annual growth over the trailing {GROWTH_WINDOW_YEARS} observed "
            "years, renewables share capped at 100 %; an assumption, not data"
        ),
        "exogenous_growth_rates": {
            col: float(reference.growth_rates.get(col, 0.0)) for col in EXOG_COLUMNS
        },
        "carried_forward_years": carried_forward_years(reference.raw),
        "models_using_exogenous": [
            key for key, s in services.items() if getattr(s, "uses_exogenous", True)
        ],
    }
