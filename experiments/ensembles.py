"""
Online (nested) model selection and combination.

For a forecast made at ``origin`` for horizon ``h`` the weights only use
backtest records of the pool whose target year is <= origin, i.e. errors that
were observable when the forecast was made.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_PAST_RECORDS = 5


def _past_errors(pool: pd.DataFrame, origin: int, horizon: int) -> pd.DataFrame | None:
    """Past records at the largest horizon <= h with enough observed errors."""
    observed = pool[pool["year"] <= origin]
    for h in range(horizon, 0, -1):
        sub = observed[observed["horizon"] == h]
        counts = sub.groupby("model").size()
        if len(counts) == pool["model"].nunique() and counts.min() >= MIN_PAST_RECORDS:
            return sub
    return None


def online_combinations(
    pool: pd.DataFrame, fallback: str = "persistence"
) -> pd.DataFrame:
    """
    Records of ``online_best`` (lowest past MAE) and ``online_invmse``
    (inverse past-MSE weights) built from the pool's records.
    """
    wide = pool.pivot_table(
        index=["origin", "horizon", "year", "actual"],
        columns="model",
        values="predicted",
    ).reset_index()
    models = sorted(pool["model"].unique())
    rows = []
    for rec in wide.itertuples(index=False):
        rec = rec._asdict()
        past = _past_errors(pool, rec["origin"], rec["horizon"])
        preds = np.array([rec[m] for m in models])
        if past is None:
            best = rec[fallback]
            combo = float(np.mean(preds))
        else:
            err = past["predicted"] - past["actual"]
            mae = err.abs().groupby(past["model"]).mean().reindex(models)
            mse = (err**2).groupby(past["model"]).mean().reindex(models)
            best = rec[mae.idxmin()]
            weights = 1.0 / np.maximum(mse.to_numpy(), 1e-9)
            combo = float(np.dot(weights / weights.sum(), preds))
        for name, value in (("online_best", best), ("online_invmse", combo)):
            rows.append(
                {
                    "model": name,
                    "origin": rec["origin"],
                    "horizon": rec["horizon"],
                    "year": rec["year"],
                    "actual": rec["actual"],
                    "predicted": float(value),
                }
            )
    return pd.DataFrame(rows)
