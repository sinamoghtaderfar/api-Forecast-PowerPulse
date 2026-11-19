import json

import numpy as np

from app.services.backtesting import (
    BacktestRecord,
    backtest_origins,
    rolling_origin_backtest,
)
from app.services.uncertainty import (
    ResidualCalibration,
    conformal_half_width,
    min_residuals_for_level,
)
from app.services.xgboost_service import XGBoostForecastService


def test_conformal_half_width_requires_enough_residuals():
    assert min_residuals_for_level(0.95) == 19
    assert conformal_half_width(range(1, 19), 0.95) is None
    assert conformal_half_width(range(1, 20), 0.95) == 19
    assert conformal_half_width([-5, 1, 2, 3], 0.5) == 3
    assert conformal_half_width([], 0.95) is None


def test_degenerate_zero_residuals_give_zero_width():
    calibration = ResidualCalibration(0.95, {1: [0.0] * 20})
    assert calibration.half_widths(1) == [0.0]


def test_half_widths_monotone_and_none_beyond_evidence():
    calibration = ResidualCalibration(
        0.95, {1: [5.0] * 20, 2: [3.0] * 20, 3: [9.0] * 20, 4: [1.0] * 5, 5: [1.0] * 30}
    )
    assert calibration.half_widths(5) == [5.0, 5.0, 9.0, None, None]
    assert calibration.max_calibrated_horizon == 3


def test_calibration_serialisation_roundtrip():
    records = [
        BacktestRecord(2000, 1, 2001, 10.0, 8.0),
        BacktestRecord(2000, 2, 2002, 10.0, 12.0),
    ]
    calibration = ResidualCalibration.from_records(records)
    assert calibration.residuals == {1: [2.0], 2: [-2.0]}
    restored = ResidualCalibration.from_dict(
        json.loads(json.dumps(calibration.to_dict()))
    )
    assert restored == calibration


def test_backtest_only_passes_data_up_to_origin(raw):
    seen = []

    def forecaster(train, horizon):
        seen.append((int(train["year"].iloc[-1]), horizon))
        return np.zeros(horizon)

    records = rolling_origin_backtest(forecaster, raw)
    origins = backtest_origins(raw)
    assert origins[0] == 1996 and origins[-1] == 2024
    assert seen == [(o, 2025 - o) for o in origins]
    assert all(r.year == r.origin + r.horizon and r.year <= 2025 for r in records)


def test_backtest_predictions_ignore_data_after_origin(raw):
    """Altering targets and exogenous values after 2019 must not change forecasts
    made at origins up to 2019 (no temporal leakage in the real pipeline)."""
    service = XGBoostForecastService(calibrate=False)
    altered = raw.copy()
    later = altered["year"] >= 2020
    altered.loc[later, "y"] *= 2
    altered.loc[later, ["population", "gdp"]] *= 3
    kwargs = dict(first_origin=2016, max_horizon=3)
    base = rolling_origin_backtest(service.forecast_from_raw, raw, **kwargs)
    changed = rolling_origin_backtest(service.forecast_from_raw, altered, **kwargs)
    for a, b in zip(base, changed):
        if a.origin < 2020:
            assert a.predicted == b.predicted
