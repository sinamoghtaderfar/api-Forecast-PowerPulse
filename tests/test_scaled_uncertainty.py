"""Tests of the volatility-scaled, horizon-pooled conformal calibration (v4)."""

import json
import math

import numpy as np
import pytest

from app.services.backtesting import BacktestRecord
from app.services.uncertainty import (
    ScaledResidualCalibration,
    conformal_half_width,
    volatility_scale,
)

TARGET = {1990 + i: 100.0 + 2.0 * i + (5.0 if i % 3 == 0 else 0.0) for i in range(30)}


def _records(residual_fn, origins=range(1995, 2019), max_year=2019):
    out = []
    for origin in origins:
        for h in range(1, max_year - origin + 1):
            actual = TARGET[origin + h]
            out.append(
                BacktestRecord(
                    origin, h, origin + h, actual, actual - residual_fn(origin, h)
                )
            )
    return out


def test_volatility_scale_uses_last_window_changes():
    y = [1.0, 2.0, 4.0, 7.0, 11.0, 16.0]
    assert volatility_scale(y, window=3) == pytest.approx(
        np.std([3.0, 4.0, 5.0], ddof=1)
    )
    assert volatility_scale([5.0, 6.0], window=10) == pytest.approx(1e-9)
    assert volatility_scale([3.0] * 10) == pytest.approx(1e-9)


def test_scale_at_origin_ignores_later_targets():
    """sigma of an origin must not change when later targets change (no leakage)."""
    records = _records(lambda o, h: 1.0)
    base = ScaledResidualCalibration.from_backtest(records, TARGET)
    altered_target = {y: (v * 3 if y > 2005 else v) for y, v in TARGET.items()}
    altered = ScaledResidualCalibration.from_backtest(records, altered_target)
    for origin, s_base, s_alt in zip(base.origins, base.scales, altered.scales):
        if origin <= 2005:
            assert s_base == s_alt
    assert base.current_scale != altered.current_scale


def test_half_widths_scale_with_sqrt_h_and_stop_at_max_horizon():
    records = _records(lambda o, h: 2.0 * math.sqrt(h))
    cal = ScaledResidualCalibration.from_backtest(records, TARGET, max_horizon=5)
    widths = cal.half_widths(8)
    q = cal.quantile
    assert q is not None
    for h, width in enumerate(widths[:5], start=1):
        assert width == pytest.approx(q * cal.current_scale * math.sqrt(h))
    assert widths[5:] == [None, None, None]
    assert all(np.diff(widths[:5]) > 0)
    assert cal.max_calibrated_horizon == 5


def test_quantile_pools_only_horizons_up_to_max():
    records = _records(lambda o, h: 1.0 if h <= 3 else 1000.0)
    cal = ScaledResidualCalibration.from_backtest(records, TARGET, max_horizon=3)
    scores = [
        abs(r) / (s * math.sqrt(h))
        for r, s, h in zip(cal.residuals, cal.scales, cal.horizons)
        if h <= 3
    ]
    assert cal.quantile == pytest.approx(conformal_half_width(scores, 0.95))


def test_too_few_scores_give_no_interval():
    records = _records(lambda o, h: 1.0, origins=range(2015, 2018), max_year=2019)
    cal = ScaledResidualCalibration.from_backtest(records, TARGET, max_horizon=2)
    assert cal.quantile is None
    assert cal.half_widths(3) == [None, None, None]
    assert cal.max_calibrated_horizon == 0


def test_sequential_coverage_has_no_look_ahead():
    """Large misses for target year 2011 must not change intervals of origins <= 2010."""
    small = _records(lambda o, h: 1.0)
    with_miss = [
        (
            BacktestRecord(r.origin, r.horizon, r.year, r.actual, r.actual - 500.0)
            if r.year == 2011
            else r
        )
        for r in small
    ]
    a = ScaledResidualCalibration.from_backtest(small, TARGET)
    b = ScaledResidualCalibration.from_backtest(with_miss, TARGET)
    for origin in range(1995, 2011):
        assert a.quantile_at(origin) == b.quantile_at(origin)
    assert b.quantile_at(2011) > a.quantile_at(2011)
    cov_a = a.sequential_coverage(first_origin=2000)["all"]
    cov_b = b.sequential_coverage(first_origin=2000)["all"]
    assert cov_a["n"] == cov_b["n"]
    assert cov_b["coverage"] <= cov_a["coverage"]


def test_backtest_mae_sign_convention():
    records = [BacktestRecord(2001, 1, 2002, 10.0, 12.0)] * 3
    cal = ScaledResidualCalibration.from_backtest(records, TARGET)
    stats = cal.backtest_mae(first_origin=2000)[1]
    assert stats == {"n": 3, "mae": 2.0, "bias": 2.0}


def test_serialisation_roundtrip_and_legacy_rejection():
    cal = ScaledResidualCalibration.from_backtest(_records(lambda o, h: 1.5), TARGET)
    restored = ScaledResidualCalibration.from_dict(
        json.loads(json.dumps(cal.to_dict()))
    )
    assert restored == cal
    with pytest.raises(ValueError):
        ScaledResidualCalibration.from_dict({"level": 0.95, "residuals": {"1": [1.0]}})


def test_summary_is_a_superset_of_the_legacy_fields():
    summary = ScaledResidualCalibration.from_backtest(
        _records(lambda o, h: 1.0), TARGET
    ).summary()
    for key in (
        "method",
        "nominal_coverage",
        "max_calibrated_horizon",
        "residuals_per_horizon",
    ):
        assert key in summary
    assert summary["validated_coverage"] is False
    assert set(summary["backtest_coverage"]) == {"1-3", "4-5", "6-10", "all"}
