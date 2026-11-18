import math

import pytest

from app.controllers.energy_controller import calculate_model_summary


def rows(values, start_year=2026, width=None):
    return [
        {
            "year": start_year + i,
            "forecast": v,
            "lower": None if width is None else v - width / 2,
            "upper": None if width is None else v + width / 2,
        }
        for i, v in enumerate(values)
    ]


def test_one_year_horizon_has_no_growth_period():
    summary = calculate_model_summary(rows([500.0], width=10), "M")
    assert summary["period"] == "2026-2026"
    assert summary["growth_periods"] == 0
    assert summary["total_growth_percent"] is None
    assert summary["avg_annual_growth"] is None
    assert summary["avg_uncertainty_width"] == 10


def test_two_year_horizon():
    summary = calculate_model_summary(rows([100.0, 110.0]), "M")
    assert summary["growth_periods"] == 1
    assert summary["total_growth_percent"] == 10.0
    assert summary["avg_annual_growth"] == 10.0
    assert summary["avg_uncertainty_width"] is None


def test_three_year_horizon_compounds():
    summary = calculate_model_summary(rows([100.0, 110.0, 121.0]), "M")
    assert summary["growth_periods"] == 2
    assert summary["total_growth_percent"] == 21.0
    assert summary["avg_annual_growth"] == 10.0


def test_ten_year_horizon_cagr():
    values = [500.0 * 0.98**i for i in range(10)]
    summary = calculate_model_summary(rows(values), "M")
    assert summary["period"] == "2026-2035"
    assert summary["growth_periods"] == 9
    assert summary["total_growth_percent"] == pytest.approx(
        (0.98**9 - 1) * 100, abs=0.01
    )
    assert summary["avg_annual_growth"] == pytest.approx(-2.0, abs=0.01)


@pytest.mark.parametrize(
    "values", [[0.0, 10.0], [10.0, 0.0], [-5.0, 10.0], [math.inf, 1.0]]
)
def test_undefined_growth_for_non_positive_or_non_finite(values):
    summary = calculate_model_summary(rows(values), "M")
    assert summary["total_growth_percent"] is None
    assert summary["avg_annual_growth"] is None


def test_empty_forecast():
    assert calculate_model_summary([], "M") is None


def test_partial_bounds_average_only_available_widths():
    data = rows([100.0, 100.0], width=4)
    data[1]["lower"] = data[1]["upper"] = None
    assert calculate_model_summary(data, "M")["avg_uncertainty_width"] == 4
