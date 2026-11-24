"""Point metrics, evaluation sets and paired forecast comparisons."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

FIRST_SCORED_ORIGIN = 2000
DEV_LAST_TARGET = 2014
HOLDOUT_FIRST_ORIGIN = 2014
MAIN_HORIZONS = (1, 3, 5, 10)
HORIZON_GROUPS = {"1-3": (1, 3), "4-5": (4, 5), "6-10": (6, 10)}


def select_set(records: pd.DataFrame, name: str) -> pd.DataFrame:
    """Records of the FULL, DEV or HOLDOUT evaluation set (see PROTOCOL.md)."""
    scored = records[records["origin"] >= FIRST_SCORED_ORIGIN]
    if name == "FULL":
        return scored
    if name == "DEV":
        return scored[scored["year"] <= DEV_LAST_TARGET]
    if name == "HOLDOUT":
        return scored[scored["origin"] >= HOLDOUT_FIRST_ORIGIN]
    raise ValueError(f"Unknown evaluation set {name!r}")


def point_metrics(actual: Iterable[float], predicted: Iterable[float]) -> dict:
    """MAE, RMSE, sMAPE, WAPE and bias (forecast - actual) of paired values."""
    actual = np.asarray(list(actual), dtype=float)
    predicted = np.asarray(list(predicted), dtype=float)
    if actual.size == 0:
        return {"n": 0}
    error = predicted - actual
    return {
        "n": int(actual.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "smape": float(
            np.mean(2 * np.abs(error) / (np.abs(actual) + np.abs(predicted))) * 100
        ),
        "wape": float(np.sum(np.abs(error)) / np.sum(np.abs(actual)) * 100),
        "bias": float(np.mean(error)),
    }


def metrics_by_horizon(
    records: pd.DataFrame, horizons: Iterable[int] | None = None
) -> dict[int, dict]:
    """Point metrics per horizon for one model's records."""
    if horizons is None:
        horizons = sorted(records["horizon"].unique())
    out = {}
    for h in horizons:
        sub = records[records["horizon"] == h]
        if len(sub):
            out[int(h)] = point_metrics(sub["actual"], sub["predicted"])
    return out


def newey_west_variance(d: np.ndarray, lag: int) -> float:
    """Long-run variance of the mean of ``d`` (Bartlett kernel)."""
    n = d.size
    d = d - d.mean()
    gamma0 = float(np.dot(d, d) / n)
    var = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        weight = 1.0 - k / (lag + 1.0)
        var += 2.0 * weight * float(np.dot(d[k:], d[:-k]) / n)
    return max(var, 1e-12) / n


def diebold_mariano(abs_err_a: np.ndarray, abs_err_b: np.ndarray, horizon: int):
    """
    DM test of equal mean absolute error (A vs B) with HLN correction.

    Returns ``(mean_diff, statistic, p_value)``; ``mean_diff < 0`` favours A.
    """
    d = np.asarray(abs_err_a, dtype=float) - np.asarray(abs_err_b, dtype=float)
    n = d.size
    if n < 3:
        return float(d.mean()) if n else float("nan"), float("nan"), float("nan")
    dm = d.mean() / np.sqrt(newey_west_variance(d, horizon - 1))
    correction = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
    if correction <= 0:
        return float(d.mean()), float("nan"), float("nan")
    stat = float(dm * np.sqrt(correction))
    p_value = float(2 * stats.t.sf(abs(stat), df=n - 1))
    return float(d.mean()), stat, p_value


def block_bootstrap_ratio(
    abs_err_a: np.ndarray,
    abs_err_b: np.ndarray,
    block: int,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """95 % moving-block bootstrap interval of MAE(A) / MAE(B) over origins."""
    a = np.asarray(abs_err_a, dtype=float)
    b = np.asarray(abs_err_b, dtype=float)
    n = a.size
    if n < 3:
        return float("nan"), float("nan")
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    starts_max = n - block + 1
    n_blocks = int(np.ceil(n / block))
    ratios = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, starts_max, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        denom = b[idx].mean()
        ratios[i] = a[idx].mean() / denom if denom > 0 else np.nan
    lo, hi = np.nanpercentile(ratios, [2.5, 97.5])
    return float(lo), float(hi)


def paired_comparison(
    records: pd.DataFrame, model: str, reference: str, horizon: int
) -> dict:
    """
    Paired (by origin) comparison of ``model`` against ``reference`` at ``horizon``.

    The DM test and the bootstrap are only computed when there are at least
    ``2h + 2`` paired forecasts; otherwise they are NaN (``testable = False``).
    """
    a = records[(records["model"] == model) & (records["horizon"] == horizon)]
    b = records[(records["model"] == reference) & (records["horizon"] == horizon)]
    merged = a.merge(b, on=["origin", "horizon", "year"], suffixes=("_a", "_b"))
    merged = merged.sort_values("origin")
    err_a = np.abs(merged["predicted_a"] - merged["actual_a"]).to_numpy()
    err_b = np.abs(merged["predicted_b"] - merged["actual_b"]).to_numpy()
    if merged.empty:
        return {"n": 0}
    testable = len(merged) >= 2 * horizon + 2
    if testable:
        mean_diff, stat, p_value = diebold_mariano(err_a, err_b, horizon)
        lo, hi = block_bootstrap_ratio(err_a, err_b, block=max(horizon, 2))
    else:
        # Too few non-overlapping windows for a meaningful test.
        mean_diff = float(np.mean(err_a - err_b))
        stat = p_value = lo = hi = float("nan")
    mae_b = err_b.mean()
    return {
        "n": int(len(merged)),
        "mae_model": float(err_a.mean()),
        "mae_reference": float(mae_b),
        "relative_mae": float(err_a.mean() / mae_b) if mae_b > 0 else float("nan"),
        "improvement_pct": (
            float((mae_b - err_a.mean()) / mae_b * 100) if mae_b > 0 else float("nan")
        ),
        "mean_abs_err_diff": mean_diff,
        "dm_stat": stat,
        "dm_p_value": p_value,
        "boot_rel_mae_95": [lo, hi],
        "wins": int(np.sum(err_a < err_b)),
        "testable": bool(testable),
    }


def relative_mae_table(
    records: pd.DataFrame, reference: str = "persistence", max_h: int = 10
) -> pd.DataFrame:
    """MAE of each model relative to ``reference`` on common records, per horizon."""
    ref = records[records["model"] == reference][
        ["origin", "horizon", "actual", "predicted"]
    ].rename(columns={"predicted": "ref_pred"})
    rows = []
    for model, sub in records.groupby("model"):
        merged = sub.merge(ref, on=["origin", "horizon", "actual"])
        for h in range(1, max_h + 1):
            m = merged[merged["horizon"] == h]
            if m.empty:
                continue
            mae = np.mean(np.abs(m["predicted"] - m["actual"]))
            ref_mae = np.mean(np.abs(m["ref_pred"] - m["actual"]))
            rows.append(
                {
                    "model": model,
                    "horizon": h,
                    "n": len(m),
                    "rel_mae": mae / ref_mae if ref_mae > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)
