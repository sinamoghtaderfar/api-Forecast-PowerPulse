import numpy as np
import pandas as pd
import pytest

from app.evaluation import (
    drift_forecast,
    persistence_forecast,
    point_metrics,
    sequential_interval_coverage,
)
from app.services.backtesting import BacktestRecord


def test_baselines_and_metrics():
    train = pd.DataFrame({"year": [2000, 2001, 2002], "y": [10.0, 12.0, 14.0]})
    assert persistence_forecast(train, 2).tolist() == [14.0, 14.0]
    assert drift_forecast(train, 2).tolist() == [16.0, 18.0]
    records = [BacktestRecord(0, 1, 1, 10.0, 8.0), BacktestRecord(1, 1, 2, 10.0, 14.0)]
    metrics = point_metrics(records)
    assert metrics["mae"] == pytest.approx(3.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(10.0))
    assert metrics["smape"] == pytest.approx((2 * 2 / 18 + 2 * 4 / 24) / 2 * 100)


def test_sequential_coverage_uses_only_observed_residuals():
    # 20 small residuals observed by origin 21, then a large miss at origin 21.
    records = [BacktestRecord(o, 1, o + 1, 10.0, 9.0) for o in range(1, 21)]
    records.append(BacktestRecord(21, 1, 22, 10.0, 0.0))
    result = sequential_interval_coverage(records, level=0.95)
    # Only origins with >= 19 observed residuals are evaluated (20 and 21).
    assert result[1]["n"] == 2
    assert result[1]["coverage"] == pytest.approx(0.5)
    assert result[1]["mean_width"] == pytest.approx(2.0)
