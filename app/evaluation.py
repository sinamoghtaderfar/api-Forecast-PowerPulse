"""
Reproducible rolling-origin evaluation of the forecasting models.

Usage: ``python -m app.evaluation [--first-origin 2000] [--json report.json]``

* Point accuracy: genuine recursive forecasts from every origin year
  ``>= first_origin`` (expanding window), compared with persistence and drift
  baselines; MAE, RMSE (TWh), sMAPE (%), WAPE (%) and bias (forecast - actual,
  TWh) per horizon.
* Interval coverage: for each origin/horizon the interval is built only from
  residuals whose target year was already observed at that origin (no
  look-ahead). ``intervals`` evaluates the production method
  (volatility-scaled pooled conformal, horizons 1-10);
  ``intervals_legacy_per_horizon`` the pre-v4 per-horizon method.

The research experiments (more candidates, DEV/HOLDOUT split, paired tests)
live in the separate ``experiments`` package.

Models are fitted with the same code and hyperparameters as in production
(``forecast_from_raw``). Fixed random seeds make results reproducible.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from typing import Any, Callable

import numpy as np
import pandas as pd

from app.services.backtesting import BacktestRecord, rolling_origin_backtest
from app.services.base import DEFAULT_DATA_PATH
from app.services.features import read_energy_csv
from app.services.persistence_service import PersistenceForecastService
from app.services.prophet_service import ProphetForecastService
from app.services.random_forest_service import RandomForestForecastService
from app.services.uncertainty import (
    DEFAULT_LEVEL,
    DEFAULT_MAX_HORIZON,
    ResidualCalibration,
    ScaledResidualCalibration,
)
from app.services.xgboost_service import XGBoostForecastService

DEFAULT_FIRST_ORIGIN = 2000
DEFAULT_HORIZONS = (1, 3, 5, 10)


def persistence_forecast(raw: pd.DataFrame, horizon: int) -> np.ndarray:
    """Last observed value for every future year."""
    return np.full(horizon, float(raw["y"].iloc[-1]))


def drift_forecast(raw: pd.DataFrame, horizon: int) -> np.ndarray:
    """Last value plus the average historical annual change."""
    y = raw["y"].to_numpy(dtype=float)
    slope = (y[-1] - y[0]) / (len(y) - 1)
    return y[-1] + slope * np.arange(1, horizon + 1)


def point_metrics(records: list[BacktestRecord]) -> dict[str, float]:
    """
    MAE, RMSE, symmetric MAPE (%), WAPE (%) and bias of backtest records.

    ``bias`` is the mean of forecast - actual (positive = over-forecast);
    ``wape`` is sum|error| / sum|actual| * 100.
    """
    actual = np.array([r.actual for r in records])
    predicted = np.array([r.predicted for r in records])
    error = actual - predicted
    smape = np.mean(2 * np.abs(error) / (np.abs(actual) + np.abs(predicted))) * 100
    return {
        "n": len(records),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "smape": float(smape),
        "wape": float(np.sum(np.abs(error)) / np.sum(np.abs(actual)) * 100),
        "bias": float(np.mean(-error)),
    }


def sequential_interval_coverage(
    records: list[BacktestRecord], level: float = DEFAULT_LEVEL
) -> dict[int, dict[str, float]]:
    """
    Legacy (pre-v4) per-horizon method: coverage of intervals that only use
    residuals observable at each origin.

    Returns per-horizon ``{"n": evaluated, "coverage": share, "mean_width": TWh}``;
    horizons where no interval could be formed are omitted.
    """
    results: dict[int, list[tuple[bool, float]]] = defaultdict(list)
    for target in records:
        known = [r for r in records if r.origin + r.horizon <= target.origin]
        calibration = ResidualCalibration.from_records(known, level)
        width = calibration.half_widths(target.horizon)[-1]
        if width is None:
            continue
        covered = target.predicted - width <= target.actual <= target.predicted + width
        results[target.horizon].append((covered, 2 * width))
    return {
        h: {
            "n": len(values),
            "coverage": float(np.mean([c for c, _ in values])),
            "mean_width": float(np.mean([w for _, w in values])),
        }
        for h, values in sorted(results.items())
    }


def scaled_interval_coverage(
    records: list[BacktestRecord],
    raw: pd.DataFrame,
    first_origin: int = DEFAULT_FIRST_ORIGIN,
    level: float = DEFAULT_LEVEL,
) -> dict[str, dict[str, float]]:
    """
    Production method (volatility-scaled pooled conformal): sequential coverage
    per horizon 1..10 and per horizon range, without look-ahead.
    """
    target = dict(zip(raw["year"].astype(int), raw["y"].astype(float)))
    calibration = ScaledResidualCalibration.from_backtest(records, target, level)
    groups = {str(h): (h, h) for h in range(1, DEFAULT_MAX_HORIZON + 1)}
    groups.update({"1-3": (1, 3), "4-5": (4, 5), "6-10": (6, 10), "all": (1, 10)})
    return calibration.sequential_coverage(first_origin=first_origin, groups=groups)


def evaluate(
    raw: pd.DataFrame,
    first_origin: int = DEFAULT_FIRST_ORIGIN,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> dict[str, Any]:
    """Run all backtests and return metrics per model and horizon."""
    forecasters: dict[str, Callable[[pd.DataFrame, int], np.ndarray]] = {
        "persistence": persistence_forecast,
        "drift": drift_forecast,
    }
    services = {
        "prophet": ProphetForecastService(calibrate=False),
        "random_forest": RandomForestForecastService(calibrate=False),
        "xgboost": XGBoostForecastService(calibrate=False),
    }
    forecasters.update({key: s.forecast_from_raw for key, s in services.items()})
    served = {"persistence", *services}
    # The served persistence model must equal the persistence baseline.
    check = rolling_origin_backtest(
        PersistenceForecastService(calibrate=False).forecast_from_raw, raw
    )
    if check != rolling_origin_backtest(persistence_forecast, raw):
        raise RuntimeError("Served persistence model differs from the baseline")

    report: dict[str, Any] = {
        "data_years": [int(raw["year"].iloc[0]), int(raw["year"].iloc[-1])],
        "first_origin": first_origin,
        "point": {},
        "intervals": {},
        "intervals_legacy_per_horizon": {},
    }
    for name, forecaster in forecasters.items():
        records = rolling_origin_backtest(forecaster, raw)
        scored = [r for r in records if r.origin >= first_origin]
        report["point"][name] = {
            str(h): point_metrics([r for r in scored if r.horizon == h])
            for h in horizons
            if any(r.horizon == h for r in scored)
        }
        if name in served:
            report["intervals"][name] = scaled_interval_coverage(
                records, raw, first_origin
            )
            report["intervals_legacy_per_horizon"][name] = {
                str(h): v for h, v in sequential_interval_coverage(records).items()
            }
    return report


def format_report(report: dict[str, Any]) -> str:
    """Markdown tables of the evaluation report."""
    lines = [
        f"Data {report['data_years'][0]}-{report['data_years'][1]}, "
        f"origins >= {report['first_origin']} (expanding window, recursive).",
        "",
        "| Model | h | n | MAE (TWh) | RMSE (TWh) | sMAPE (%) | WAPE (%) | bias (TWh) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, by_h in report["point"].items():
        for h, m in by_h.items():
            lines.append(
                f"| {name} | {h} | {m['n']} | {m['mae']:.1f} | {m['rmse']:.1f} "
                f"| {m['smape']:.1f} | {m['wape']:.1f} | {m['bias']:+.1f} |"
            )
    lines += [
        "",
        "Production intervals (volatility-scaled conformal), sequential coverage:",
        "",
        "| Model | horizons | n | 95% interval coverage | mean width (TWh) |",
        "|---|---|---|---|---|",
    ]
    for name, by_h in report["intervals"].items():
        for h in ("1", "3", "5", "10", "1-3", "4-5", "6-10", "all"):
            if h in by_h:
                m = by_h[h]
                lines.append(
                    f"| {name} | {h} | {m['n']} | {m['coverage']:.2f} "
                    f"| {m['mean_width']:.1f} |"
                )
    lines += [
        "",
        "Legacy per-horizon intervals (pre-v4), sequential coverage:",
        "",
        "| Model | h | n | 95% interval coverage | mean width (TWh) |",
        "|---|---|---|---|---|",
    ]
    for name, by_h in report["intervals_legacy_per_horizon"].items():
        for h, m in by_h.items():
            lines.append(
                f"| {name} | {h} | {m['n']} | {m['coverage']:.2f} | {m['mean_width']:.1f} |"
            )
    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--first-origin", type=int, default=DEFAULT_FIRST_ORIGIN)
    parser.add_argument("--json", help="Write the full report to this JSON file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    report = evaluate(
        read_energy_csv(DEFAULT_DATA_PATH), first_origin=args.first_origin
    )
    print(format_report(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
