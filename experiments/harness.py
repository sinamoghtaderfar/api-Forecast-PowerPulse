"""Backtest runner shared by all experiments (records cached as CSV)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from app.services.backtesting import rolling_origin_backtest
from app.services.base import DEFAULT_DATA_PATH
from app.services.features import read_energy_csv

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RECORDS_DIR = RESULTS_DIR / "records"
EXTENDED_DATA_PATH = DEFAULT_DATA_PATH.parent / "germany-energy-data.csv"


def load_raw() -> pd.DataFrame:
    """Production dataset (no imputation)."""
    return read_energy_csv(DEFAULT_DATA_PATH)


def load_extended_raw() -> pd.DataFrame:
    """Production dataset plus OWID demand and net imports (decomposition only)."""
    raw = load_raw()
    extra = pd.read_csv(EXTENDED_DATA_PATH)[
        ["year", "electricity_demand", "net_elec_imports", "electricity_generation"]
    ]
    merged = raw.merge(extra, on="year", how="left")
    both = merged.dropna(subset=["electricity_demand", "net_elec_imports"])
    identity_gap = np.abs(
        both["electricity_demand"] - both["net_elec_imports"] - both["y"]
    ).max()
    if identity_gap > 0.05:
        raise ValueError(f"generation != demand - net imports (gap {identity_gap})")
    return merged.drop(columns="electricity_generation").rename(
        columns={"electricity_demand": "demand", "net_elec_imports": "net_imports"}
    )


def run_backtest(
    name: str,
    forecaster: Callable[[pd.DataFrame, int], np.ndarray],
    raw: pd.DataFrame,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Rolling-origin backtest records of one forecaster (origins 1996-2024)."""
    path = RECORDS_DIR / f"{name}.csv"
    if use_cache and path.exists():
        return pd.read_csv(path)
    start = time.perf_counter()
    records = rolling_origin_backtest(forecaster, raw)
    frame = pd.DataFrame(
        [
            {
                "model": name,
                "origin": r.origin,
                "horizon": r.horizon,
                "year": r.year,
                "actual": r.actual,
                "predicted": r.predicted,
            }
            for r in records
        ]
    )
    frame.attrs["seconds"] = time.perf_counter() - start
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"  {name}: {len(frame)} records in {frame.attrs['seconds']:.1f}s")
    return frame
