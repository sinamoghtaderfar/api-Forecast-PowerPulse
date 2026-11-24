"""Fast checks of the research harness (no full backtests are run here)."""

import numpy as np
import pandas as pd
import pytest

from experiments import candidates as C
from experiments import exog_projections as X
from experiments.ensembles import online_combinations
from experiments.intervals import weighted_conformal_quantile
from experiments.metrics import (
    diebold_mariano,
    paired_comparison,
    point_metrics,
    select_set,
)


def _records(model, origins=range(1996, 2025), last=2025, fn=lambda o, h: 0.0):
    rows = []
    for o in origins:
        for h in range(1, last - o + 1):
            rows.append(
                {
                    "model": model,
                    "origin": o,
                    "horizon": h,
                    "year": o + h,
                    "actual": 100.0,
                    "predicted": 100.0 + fn(o, h),
                }
            )
    return pd.DataFrame(rows)


def test_dev_and_holdout_are_disjoint_and_information_separated():
    rec = _records("m")
    dev, hold, full = (select_set(rec, s) for s in ("DEV", "HOLDOUT", "FULL"))
    assert dev["year"].max() <= 2014 and hold["origin"].min() >= 2014
    assert full["origin"].min() == 2000
    keys = lambda f: set(zip(f["origin"], f["horizon"]))  # noqa: E731
    assert not keys(dev) & keys(hold)


def test_point_metrics_formulas():
    m = point_metrics([10.0, 20.0], [12.0, 17.0])
    assert m["mae"] == pytest.approx(2.5)
    assert m["bias"] == pytest.approx(-0.5)
    assert m["wape"] == pytest.approx(5 / 30 * 100)


def test_diebold_mariano_sign_and_small_samples():
    a, b = np.array([1.0, 1.0, 2.0, 1.0, 1.0, 2.0]), np.full(6, 3.0)
    diff, stat, p = diebold_mariano(a, b, horizon=1)
    assert diff < 0 and stat < 0 and 0 <= p <= 1
    assert np.isnan(diebold_mariano(a[:2], b[:2], 1)[2])


def test_paired_comparison_counts_only_common_origins():
    rec = pd.concat(
        [
            _records("a", fn=lambda o, h: 1.0),
            _records("b", origins=range(2005, 2025), fn=lambda o, h: 2.0),
        ]
    )
    res = paired_comparison(select_set(rec, "FULL"), "a", "b", 1)
    assert res["n"] == 20 and res["relative_mae"] == pytest.approx(0.5)


def test_online_combinations_only_use_past_errors():
    """Changing actuals observed after 2015 must not alter combinations made <= 2015."""
    pool = pd.concat(
        [
            _records("good", fn=lambda o, h: 1.0),
            _records("bad", fn=lambda o, h: 0.5 * h),
        ]
    )
    altered = pool.copy()
    altered.loc[altered["year"] > 2015, "actual"] = 300.0
    key = ["model", "origin", "horizon"]
    base = online_combinations(pool, fallback="good").sort_values(key)
    alt = online_combinations(altered, fallback="good").sort_values(key)
    early = base["origin"] <= 2015
    np.testing.assert_array_equal(
        base.loc[early, "predicted"].to_numpy(),
        alt.loc[alt["origin"] <= 2015, "predicted"].to_numpy(),
    )
    assert not np.array_equal(
        base.loc[~early, "predicted"].to_numpy(),
        alt.loc[alt["origin"] > 2015, "predicted"].to_numpy(),
    )


def test_ets_and_theta_behave_on_simple_series():
    trend = pd.DataFrame({"year": range(2000, 2015), "y": 100.0 + 2.0 * np.arange(15)})
    assert C.holt(trend, 3) == pytest.approx([130.0, 132.0, 134.0], abs=1e-3)
    flat = pd.DataFrame({"year": range(2000, 2015), "y": np.full(15, 50.0)})
    assert C.ses(flat, 2) == pytest.approx([50.0, 50.0])
    assert C.theta(flat, 2) == pytest.approx([50.0, 50.0])
    assert C.persistence(trend, 2).tolist() == [128.0, 128.0]


def test_exog_projections_anchor_on_last_observation_and_respect_limits(raw):
    train = raw[raw["year"] <= 2024].reset_index(drop=True)
    for name, project in X.PROJECTIONS.items():
        path = project(train, 5)
        assert path["year"].tolist() == list(range(2025, 2030)), name
        assert path["renewables_share_energy"].between(0, 100).all(), name
        assert (path[["population", "gdp"]] > 0).all().all(), name
    flat = X.flat(train, 3)
    assert (flat["gdp"] == train["gdp"].dropna().iloc[-1]).all()
    # GDP is missing after 2022: the gap-aware CAGR keeps growing through the gap.
    rate = X.exogenous_growth_rates(train, window=5)["gdp"]
    expected = train["gdp"].dropna().iloc[-1] * (1 + rate) ** 3
    assert X.cagr(5)(train, 1)["gdp"].iloc[0] == pytest.approx(expected)


def test_weighted_conformal_quantile():
    scores = np.arange(1.0, 21.0)
    q = weighted_conformal_quantile(scores, np.ones(20), 0.9)
    assert q == pytest.approx(19.0)
    assert weighted_conformal_quantile(scores[:3], np.ones(3), 0.95) is None
