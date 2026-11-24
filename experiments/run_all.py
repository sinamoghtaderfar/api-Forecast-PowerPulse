"""
Run all accuracy experiments and write machine-readable results.

Usage::

    python -m experiments.run_all            # uses cached backtest records
    python -m experiments.run_all --fresh    # recompute every backtest

Outputs (``experiments/results/``): ``records/*.csv`` (every forecast),
``baseline_frozen.json``, ``metrics.json``, ``comparisons.csv`` (paired DM /
bootstrap tests), ``comparison_vs_persistence.csv``, ``comparison_vs_original.csv``,
``relative_mae_*.csv`` and ``exog_projection_error.csv``. Interval methods:
``python -m experiments.run_intervals``; audit diagnostics:
``python -m experiments.diagnostics``.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings

import numpy as np
import pandas as pd

from app.services.features import EXOG_COLUMNS
from app.services.prophet_service import ProphetForecastService
from app.services.random_forest_service import RandomForestForecastService
from app.services.xgboost_service import XGBoostForecastService
from experiments import candidates as C
from experiments import exog_projections as X
from experiments.ensembles import online_combinations
from experiments.harness import RESULTS_DIR, load_extended_raw, load_raw, run_backtest
from experiments.metrics import (
    HORIZON_GROUPS,
    MAIN_HORIZONS,
    metrics_by_horizon,
    paired_comparison,
    relative_mae_table,
    select_set,
)

ORIGINAL = ["persistence", "drift", "prophet", "random_forest", "xgboost"]
POOL = [
    "persistence",
    "drift",
    "ses",
    "holt",
    "damped",
    "theta",
    "arima_auto",
    "prophet",
    "random_forest",
    "xgboost",
]


def _prophet_noreg():
    from prophet import Prophet  # pylint: disable=import-outside-toplevel

    from app.services.prophet_service import (  # pylint: disable=import-outside-toplevel
        _to_prophet_frame,
    )

    def forecast(raw, horizon):
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            growth="linear",
            uncertainty_samples=0,
        )
        model.fit(_to_prophet_frame(raw)[["ds", "y"]])
        years = pd.DataFrame(
            {
                "year": np.arange(
                    raw["year"].iloc[-1] + 1, raw["year"].iloc[-1] + 1 + horizon
                )
            }
        )
        return model.predict(_to_prophet_frame(years)[["ds"]])["yhat"].to_numpy()

    return forecast


def forecasters(raw: pd.DataFrame) -> dict:
    """Name -> (forecaster, dataset) for every experiment."""
    base = raw[C.BASE_COLUMNS]
    services = {
        "prophet": ProphetForecastService(calibrate=False),
        "random_forest": RandomForestForecastService(calibrate=False),
        "xgboost": XGBoostForecastService(calibrate=False),
    }
    cagr10 = X.PROJECTIONS["cagr10"]
    out = {
        # original system (frozen baseline)
        "persistence": (C.persistence, base),
        "drift": (C.drift, base),
        **{k: (s.forecast_from_raw, base) for k, s in services.items()},
        # univariate statistical
        "drift_recent5": (C.drift_recent(5), base),
        "drift_recent10": (C.drift_recent(10), base),
        "ses": (C.ses, base),
        "holt": (C.holt, base),
        "damped": (C.damped, base),
        "theta": (C.theta, base),
        "arima_auto": (C.arima_auto, base),
        "comb_shd": (C.comb_shd, base),
        # linear ML
        "ridge_ar_diff": (C.ridge_ar_diff, base),
        "direct_ridge": (C.direct_ridge, base),
        "ardl_gdp": (C.ardl_gdp(cagr10), base),
        # tree ML on changes
        "xgb_diff": (C.tree_diff(C.make_xgb_diff, cagr10), base),
        "rf_diff": (C.tree_diff(C.make_rf_diff, cagr10), base),
        "xgb_diff_tuned": (C.xgb_diff_tuned(cagr10), base),
        # prophet without regressors
        "prophet_noreg": (_prophet_noreg(), base),
        # decomposition
        "decomp_damped": (C.decomp_damped, raw),
    }
    # exogenous projection variants of the existing exog models
    for key, service in services.items():
        for proj_name, proj in X.PROJECTIONS.items():
            if proj_name == "cagr10":
                continue
            out[f"{key}__{proj_name}"] = (X.with_projection(service, proj), base)
        out[f"{key}__oracle"] = (X.with_projection(service, X.oracle(base)), base)
    return out


def metrics_report(records: pd.DataFrame) -> dict:
    """Metrics per set, model and horizon (all horizons 1..25) plus horizon groups."""
    report = {}
    for set_name in ("FULL", "DEV", "HOLDOUT"):
        subset = select_set(records, set_name)
        report[set_name] = {}
        for model, sub in subset.groupby("model"):
            by_h = metrics_by_horizon(sub)
            groups = {}
            for label, (lo, hi) in HORIZON_GROUPS.items():
                g = sub[(sub["horizon"] >= lo) & (sub["horizon"] <= hi)]
                if len(g):
                    err = g["predicted"] - g["actual"]
                    groups[label] = {
                        "n": int(len(g)),
                        "mae": float(err.abs().mean()),
                        "bias": float(err.mean()),
                    }
            report[set_name][model] = {
                "by_horizon": {str(h): m for h, m in by_h.items()},
                "groups": groups,
            }
    return report


def comparisons(records: pd.DataFrame, references=("persistence", "drift", "xgboost")):
    """Paired DM / bootstrap comparisons for every model vs the references."""
    rows = []
    for set_name in ("FULL", "DEV", "HOLDOUT"):
        subset = select_set(records, set_name)
        for model in sorted(subset["model"].unique()):
            for ref in references:
                if model == ref:
                    continue
                for h in MAIN_HORIZONS:
                    res = paired_comparison(subset, model, ref, h)
                    if res.get("n", 0) == 0:
                        continue
                    rows.append(
                        {"set": set_name, "model": model, "reference": ref, "h": h}
                        | {k: v for k, v in res.items() if k != "boot_rel_mae_95"}
                        | {
                            "boot_lo": res["boot_rel_mae_95"][0],
                            "boot_hi": res["boot_rel_mae_95"][1],
                        }
                    )
    return pd.DataFrame(rows)


def baseline_table(
    records: pd.DataFrame, baseline_model: dict[str, str]
) -> pd.DataFrame:
    """Required comparison table: model vs its original counterpart (FULL set)."""
    full = select_set(records, "FULL")
    rows = []
    for model, base_name in baseline_model.items():
        for h in MAIN_HORIZONS:
            new = full[(full["model"] == model) & (full["horizon"] == h)]
            old = full[(full["model"] == base_name) & (full["horizon"] == h)]
            merged = new.merge(old, on=["origin", "horizon", "year", "actual"])
            if merged.empty:
                continue
            new_err = merged["predicted_x"] - merged["actual"]
            old_err = merged["predicted_y"] - merged["actual"]
            base_mae = old_err.abs().mean()
            new_mae = new_err.abs().mean()
            actual = merged["actual"]
            rows.append(
                {
                    "model": model,
                    "baseline": base_name,
                    "horizon": h,
                    "baseline_mae": base_mae,
                    "new_mae": new_mae,
                    "improvement_pct": (base_mae - new_mae) / base_mae * 100,
                    "rmse": float(np.sqrt((new_err**2).mean())),
                    "smape": float(
                        (
                            2
                            * new_err.abs()
                            / (actual.abs() + merged["predicted_x"].abs())
                        ).mean()
                        * 100
                    ),
                    "bias": float(new_err.mean()),
                    "n": len(merged),
                }
            )
    return pd.DataFrame(rows)


def exog_projection_error(raw: pd.DataFrame) -> pd.DataFrame:
    """Accuracy of each exogenous projection against later observed values."""
    base = raw[C.BASE_COLUMNS]
    rows = []
    for name, proj in X.PROJECTIONS.items():
        for origin in range(2000, 2025):
            train = base[base["year"] <= origin].reset_index(drop=True)
            horizon = 2025 - origin
            path = proj(train, horizon)
            for _, row in path.iterrows():
                actual = base.loc[base["year"] == row["year"]].iloc[0]
                for col in EXOG_COLUMNS:
                    if pd.isna(actual[col]):
                        continue
                    rows.append(
                        {
                            "projection": name,
                            "variable": col,
                            "origin": origin,
                            "horizon": int(row["year"] - origin),
                            "ape": abs(row[col] - actual[col]) / abs(actual[col]) * 100,
                            "error": row[col] - actual[col],
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="ignore cached records")
    parser.add_argument("--only", nargs="*", help="run only these forecasters")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")

    raw = load_extended_raw()
    registry = forecasters(raw)
    names = args.only or list(registry)
    all_records = []
    for name in names:
        forecaster, data = registry[name]
        all_records.append(
            run_backtest(name, forecaster, data, use_cache=not args.fresh)
        )
    records = pd.concat(all_records, ignore_index=True)

    if set(POOL) <= set(records["model"]):
        online = online_combinations(records[records["model"].isin(POOL)])
        records = pd.concat([records, online], ignore_index=True)
        online.to_csv(RESULTS_DIR / "records" / "online_combinations.csv", index=False)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = metrics_report(records)
    baseline = {
        name: {m: v for m, v in by_model.items() if m in ORIGINAL}
        for name, by_model in report.items()
    }
    (RESULTS_DIR / "baseline_frozen.json").write_text(json.dumps(baseline, indent=2))
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    comparisons(records).to_csv(RESULTS_DIR / "comparisons.csv", index=False)
    models = sorted(records["model"].unique())
    baseline_table(
        records, {m: "persistence" for m in models if m != "persistence"}
    ).to_csv(RESULTS_DIR / "comparison_vs_persistence.csv", index=False)
    originals = {"rf_diff": "random_forest", "xgb_diff": "xgboost"}
    originals |= {"xgb_diff_tuned": "xgboost", "prophet_noreg": "prophet"}
    originals |= {m: m.split("__")[0] for m in models if "__" in m}
    baseline_table(records, originals).to_csv(
        RESULTS_DIR / "comparison_vs_original.csv", index=False
    )
    relative_mae_table(select_set(records, "FULL")).to_csv(
        RESULTS_DIR / "relative_mae_full.csv", index=False
    )
    relative_mae_table(select_set(records, "DEV")).to_csv(
        RESULTS_DIR / "relative_mae_dev.csv", index=False
    )
    relative_mae_table(select_set(records, "HOLDOUT")).to_csv(
        RESULTS_DIR / "relative_mae_holdout.csv", index=False
    )
    exog_projection_error(load_raw()).to_csv(
        RESULTS_DIR / "exog_projection_error.csv", index=False
    )
    print("done:", len(records), "records")


if __name__ == "__main__":
    main()
