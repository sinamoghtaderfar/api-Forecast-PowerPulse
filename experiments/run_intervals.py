"""
Sequential (no look-ahead) evaluation of interval methods.

Usage: ``python -m experiments.run_intervals`` (after ``experiments.run_all``).
Writes ``results/intervals.json`` and ``results/interval_records.csv``.
"""

from __future__ import annotations

import json

import pandas as pd

from experiments.harness import RECORDS_DIR, RESULTS_DIR, load_raw
from experiments.intervals import coverage_summary, sequential_intervals

MODELS = (
    "persistence",
    "prophet",
    "random_forest",
    "xgboost",
    "drift",
    "ses",
    "theta",
    "damped",
    "decomp_damped",
)


def main() -> None:
    """CLI entry point."""
    target = load_raw().set_index("year")["y"]
    summaries, frames = {}, []
    for model in MODELS:
        records = pd.read_csv(RECORDS_DIR / f"{model}.csv")
        intervals = sequential_intervals(records, target)
        intervals["model"] = model
        frames.append(intervals)
        summaries[model] = coverage_summary(intervals)
        overall = (
            intervals[(intervals["origin"] >= 2000) & (intervals["horizon"] <= 10)]
            .groupby("method")["covered"]
            .agg(["mean", "size"])
        )
        summaries[model]["overall_h1_10"] = {
            m: {"coverage": float(r["mean"]), "n": int(r["size"])}
            for m, r in overall.iterrows()
        }
        print(model, overall.round(3).to_dict("index"))
    pd.concat(frames).to_csv(RESULTS_DIR / "interval_records.csv", index=False)
    (RESULTS_DIR / "intervals.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
