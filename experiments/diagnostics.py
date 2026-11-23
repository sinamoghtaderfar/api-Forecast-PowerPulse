"""
Diagnostics behind the forecasting audit (writes ``results/diagnostics.json``).

Usage: ``python -m experiments.diagnostics`` (after ``experiments.run_all``).
"""

from __future__ import annotations

import json
import logging
import warnings

import pandas as pd
from scipy import stats

from app.services.features import build_training_frame, prepare_history
from app.services.random_forest_service import RandomForestForecastService
from app.services.xgboost_service import XGBoostForecastService
from app.services.prophet_service import ProphetForecastService
from experiments import exog_projections as X
from experiments.harness import RECORDS_DIR, RESULTS_DIR, load_extended_raw, load_raw
from experiments.metrics import select_set

PERIODS = {
    "1985-2007": (1986, 2007),
    "2008-2017": (2008, 2017),
    "2018-2025": (2018, 2025),
}


def structural_summary(raw: pd.DataFrame) -> dict:
    """Mean/std of annual changes per period and a Welch test for a 2018 shift."""
    y = raw.set_index("year")["y"]
    d = y.diff().dropna()
    out = {
        label: {
            "mean_change": float(d.loc[a:b].mean()),
            "std_change": float(d.loc[a:b].std(ddof=1)),
            "n": int(d.loc[a:b].size),
        }
        for label, (a, b) in PERIODS.items()
    }
    before, after = d.loc[:2017], d.loc[2018:]
    test = stats.ttest_ind(before, after, equal_var=False)
    out["welch_shift_2018"] = {"t": float(test.statistic), "p": float(test.pvalue)}
    out["largest_changes"] = {
        int(k): float(v)
        for k, v in d.abs().sort_values(ascending=False).head(6).items()
    }
    return out


def decomposition_summary(ext: pd.DataFrame) -> dict:
    """Change in generation explained by demand and by net imports."""
    e = ext.set_index("year")
    out = {}
    for a, b in ((2007, 2025), (2017, 2025), (2022, 2025)):
        out[f"{a}-{b}"] = {
            "generation": float(e.loc[b, "y"] - e.loc[a, "y"]),
            "demand": float(e.loc[b, "demand"] - e.loc[a, "demand"]),
            "minus_net_imports": float(
                -(e.loc[b, "net_imports"] - e.loc[a, "net_imports"])
            ),
        }
    return out


def tree_range_diagnostic(raw: pd.DataFrame) -> dict:
    """How often actual values fell below the training range of the level-target trees."""
    out = {}
    y = raw.set_index("year")["y"]
    for model in ("random_forest", "xgboost", "rf_diff", "xgb_diff"):
        rec = select_set(pd.read_csv(RECORDS_DIR / f"{model}.csv"), "FULL")
        train_min = rec["origin"].map(lambda o: y.loc[:o].min())
        below = rec["actual"] < train_min
        out[model] = {
            "records": int(len(rec)),
            "actual_below_training_min": int(below.sum()),
            "share_forecasts_below_training_min": float(
                (rec["predicted"] < train_min).mean()
            ),
            "mae_when_actual_below_min": (
                float(
                    (rec.loc[below, "predicted"] - rec.loc[below, "actual"])
                    .abs()
                    .mean()
                )
                if below.any()
                else None
            ),
            "bias_when_actual_below_min": (
                float((rec.loc[below, "predicted"] - rec.loc[below, "actual"]).mean())
                if below.any()
                else None
            ),
        }
    return out


def feature_importances(raw: pd.DataFrame) -> dict:
    """Top features of the production tree models fitted on all data."""
    history = prepare_history(raw)
    out = {}
    for key, cls in (
        ("random_forest", RandomForestForecastService),
        ("xgboost", XGBoostForecastService),
    ):
        service = cls(calibrate=False)
        model = service._fit_model(history)  # pylint: disable=protected-access
        X_train, _ = build_training_frame(history, service.feature_names)
        imp = pd.Series(model.feature_importances_, index=service.feature_names)
        exog = [
            f
            for f in service.feature_names
            if f
            in (
                "population",
                "gdp",
                "renewables_share_energy",
                "gdp_per_capita",
                "energy_intensity",
                "renewables_impact",
            )
        ]
        out[key] = {
            "training_rows": int(len(X_train)),
            "features": len(service.feature_names),
            "top": {
                k: float(v) for k, v in imp.sort_values(ascending=False).head(8).items()
            },
            "exogenous_importance_share": float(imp[exog].sum()),
        }
    return out


def exog_sensitivity(raw: pd.DataFrame, horizon: int = 10) -> dict:
    """Production-origin forecasts of the exog models under different projections."""
    history = prepare_history(raw)
    out = {}
    for key, cls in (
        ("prophet", ProphetForecastService),
        ("random_forest", RandomForestForecastService),
        ("xgboost", XGBoostForecastService),
    ):
        service = cls(calibrate=False)
        model = service._fit_model(history)  # pylint: disable=protected-access
        out[key] = {}
        for name in ("cagr10", "flat", "linear10", "cagr10_damped"):
            path = X.PROJECTIONS[name](raw, horizon)
            pred = service._predict_point(
                model, history, path
            )  # pylint: disable=protected-access
            out[key][name] = {
                "2026": float(pred[0]),
                "2030": float(pred[4]),
                "2035": float(pred[9]),
            }
    return out


def bias_by_period() -> dict:
    """Mean signed error (forecast - actual) by origin period, h = 1..5."""
    out = {}
    for model in (
        "persistence",
        "drift",
        "prophet",
        "random_forest",
        "xgboost",
        "damped",
    ):
        rec = select_set(pd.read_csv(RECORDS_DIR / f"{model}.csv"), "FULL")
        rec = rec[rec["horizon"] <= 5]
        err = rec["predicted"] - rec["actual"]
        groups = pd.cut(
            rec["origin"],
            [1999, 2007, 2014, 2024],
            labels=["2000-07", "2008-14", "2015-24"],
        )
        out[model] = {
            str(k): float(v)
            for k, v in err.groupby(groups, observed=True).mean().items()
        }
    return out


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    raw = load_raw()
    report = {
        "structural": structural_summary(raw),
        "decomposition": decomposition_summary(load_extended_raw()),
        "tree_range": tree_range_diagnostic(raw),
        "feature_importances": feature_importances(raw),
        "exog_sensitivity_2026_2035": exog_sensitivity(raw),
        "bias_by_origin_period_h1_5": bias_by_period(),
    }
    (RESULTS_DIR / "diagnostics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
