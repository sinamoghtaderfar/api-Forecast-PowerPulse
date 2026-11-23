"""
Markdown tables for docs/FORECAST_ACCURACY_REPORT.md, generated from the
machine-readable results (``python -m experiments.report_tables``).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments.harness import RESULTS_DIR

MAIN = ("1", "3", "5", "10")
CANDIDATES = [
    "drift",
    "drift_recent5",
    "drift_recent10",
    "ses",
    "holt",
    "damped",
    "theta",
    "arima_auto",
    "comb_shd",
    "ridge_ar_diff",
    "direct_ridge",
    "ardl_gdp",
    "xgb_diff",
    "rf_diff",
    "xgb_diff_tuned",
    "prophet_noreg",
    "decomp_damped",
    "online_best",
    "online_invmse",
    "prophet",
    "random_forest",
    "xgboost",
]


def _fmt(x, digits=1, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"


def baseline_table(metrics: dict) -> str:
    """Frozen baseline of the original system on FULL."""
    lines = [
        "| Model | h | n | MAE | RMSE | sMAPE % | WAPE % | Bias |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model in ("persistence", "drift", "prophet", "random_forest", "xgboost"):
        for h in MAIN:
            m = metrics["FULL"][model]["by_horizon"][h]
            lines.append(
                f"| {model} | {h} | {m['n']} | {_fmt(m['mae'])} | {_fmt(m['rmse'])} | "
                f"{_fmt(m['smape'])} | {_fmt(m['wape'])} | {_fmt(m['bias'], sign=True)} |"
            )
    return "\n".join(lines)


def set_table(metrics: dict, models: list[str]) -> str:
    """MAE by set and horizon plus relative MAE vs persistence."""
    lines = [
        "| Model | FULL h1 | h3 | h5 | h10 | DEV h1 | h3 | h5 | h10 | HOLDOUT h1 | h3 | h5 | h10 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model in ["persistence", *models]:
        cells = []
        for s in ("FULL", "DEV", "HOLDOUT"):
            for h in MAIN:
                m = metrics[s][model]["by_horizon"].get(h)
                ref = metrics[s]["persistence"]["by_horizon"].get(h)
                if not m:
                    cells.append("–")
                    continue
                rel = m["mae"] / ref["mae"]
                mark = "**" if rel < 1 and model != "persistence" else ""
                cells.append(f"{mark}{m['mae']:.1f}{mark} ({rel:.2f})")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "n (FULL / DEV / HOLDOUT): h1 25/14/11, h3 23/12/9, h5 21/10/7, h10 16/5/2. "
        "Bold = lower MAE than persistence on the same records; "
        "parentheses = MAE relative to persistence."
    )
    return "\n".join(lines)


def promotion_table(metrics: dict, models: list[str]) -> str:
    """Pre-registered promotion rule check."""
    lines = [
        "| Candidate | DEV: MAE <= persistence at h1/3/5 | DEV mean rel. MAE h1-10 | "
        "HOLDOUT: MAE <= persistence at h1/3/5 | Promoted |",
        "|---|---|---|---|---|",
    ]
    for model in models:
        rows = {}
        for s in ("DEV", "HOLDOUT"):
            by_h = metrics[s][model]["by_horizon"]
            ref = metrics[s]["persistence"]["by_horizon"]
            rows[s] = [by_h[h]["mae"] <= ref[h]["mae"] for h in ("1", "3", "5")]
            if s == "DEV":
                rel = [
                    by_h[str(h)]["mae"] / ref[str(h)]["mae"]
                    for h in range(1, 11)
                    if str(h) in by_h
                ]
                dev_rel = float(np.mean(rel))
        dev_ok = all(rows["DEV"]) and dev_rel < 1
        hold_ok = all(rows["HOLDOUT"])
        flags = {s: "/".join("yes" if v else "no" for v in rows[s]) for s in rows}
        lines.append(
            f"| {model} | {flags['DEV']} | {dev_rel:.2f} | {flags['HOLDOUT']} | "
            f"{'yes' if dev_ok and hold_ok else 'no'} |"
        )
    return "\n".join(lines)


def dm_table(comp: pd.DataFrame, models: list[str]) -> str:
    """DM p-values vs persistence (FULL)."""
    sub = comp[(comp["set"] == "FULL") & (comp["reference"] == "persistence")]
    lines = [
        "| Model | h1 rel. MAE (p) [95% boot] | h3 | h5 | h10 |",
        "|---|---|---|---|---|",
    ]
    for model in models:
        cells = []
        for h in (1, 3, 5, 10):
            r = sub[(sub["model"] == model) & (sub["h"] == h)]
            if r.empty:
                cells.append("–")
                continue
            r = r.iloc[0]
            if not r["testable"]:
                cells.append(f"{r['relative_mae']:.2f} (not testable)")
            else:
                cells.append(
                    f"{r['relative_mae']:.2f} (p={r['dm_p_value']:.2f}) "
                    f"[{r['boot_lo']:.2f}, {r['boot_hi']:.2f}]"
                )
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def required_table(path: str, models: list[str] | None = None) -> str:
    """The comparison table requested in the brief."""
    df = pd.read_csv(RESULTS_DIR / path)
    if models is not None:
        df = df[df["model"].isin(models)]
    lines = [
        "| Model | Baseline | Horizon | Baseline MAE | New MAE | Improvement % | RMSE | "
        "sMAPE % | Bias | Sample Size |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in df.itertuples():
        lines.append(
            f"| {r.model} | {r.baseline} | {r.horizon} | {r.baseline_mae:.1f} | "
            f"{r.new_mae:.1f} | {r.improvement_pct:+.1f} | {r.rmse:.1f} | {r.smape:.1f} | "
            f"{r.bias:+.1f} | {r.n} |"
        )
    return "\n".join(lines)


def exog_accuracy_table() -> str:
    """Projection error of the exogenous variables (MAPE %)."""
    e = pd.read_csv(RESULTS_DIR / "exog_projection_error.csv")
    e = e[e["horizon"].isin([1, 5, 10])]
    t = e.groupby(["variable", "projection", "horizon"])["ape"].mean().unstack()
    lines = ["| Variable | Projection | h1 | h5 | h10 |", "|---|---|---|---|---|"]
    for (var, proj), row in t.iterrows():
        lines.append(
            f"| {var} | {proj} | {row[1]:.1f} | {row[5]:.1f} | {row[10]:.1f} |"
        )
    return "\n".join(lines)


def interval_table(intervals: dict) -> str:
    """Overall sequential coverage (h1-10) per model and method."""
    methods = [
        "conformal_h",
        "conformal_pooled_sqrt",
        "conformal_pooled_sqrt_recent",
        "scaled_w5_pall",
        "scaled_w10_pall",
        "scaled_w10_p5",
        "scaled_w10_p10",
    ]
    lines = [
        "| Model | " + " | ".join(methods) + " |",
        "|---|" + "---|" * len(methods),
    ]
    for model, data in intervals.items():
        cells = []
        for m in methods:
            v = data["overall_h1_10"].get(m)
            cells.append(f"{v['coverage']:.2f} (n={v['n']})" if v else "–")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def interval_by_group(intervals: dict, method: str) -> str:
    """Coverage / mean width by horizon for one method."""
    lines = [
        "| Model | h1 cov / width | h3 | h5 | h10 |",
        "|---|---|---|---|---|",
    ]
    for model, data in intervals.items():
        by_h = data["all"].get(method, {})
        cells = []
        for h in ("1", "3", "5", "10"):
            v = by_h.get(h)
            cells.append(
                f"{v['coverage']:.2f} / {v['mean_width']:.0f} (n={v['n']})"
                if v
                else "–"
            )
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    """Print all tables."""
    metrics = json.loads((RESULTS_DIR / "metrics.json").read_text())
    intervals = json.loads((RESULTS_DIR / "intervals.json").read_text())
    comp = pd.read_csv(RESULTS_DIR / "comparisons.csv")
    new_models = [
        m for m in CANDIDATES if m not in ("prophet", "random_forest", "xgboost")
    ]
    sections = {
        "BASELINE": baseline_table(metrics),
        "SETS": set_table(metrics, CANDIDATES),
        "PROMOTION": promotion_table(metrics, new_models),
        "DM": dm_table(comp, CANDIDATES),
        "REQUIRED_VS_PERSISTENCE": required_table(
            "comparison_vs_persistence.csv", new_models
        ),
        "REQUIRED_VS_ORIGINAL": required_table("comparison_vs_original.csv"),
        "EXOG_ACCURACY": exog_accuracy_table(),
        "INTERVALS": interval_table(intervals),
        "INTERVALS_LEGACY": interval_by_group(intervals, "conformal_h"),
        "INTERVALS_NEW": interval_by_group(intervals, "scaled_w10_p10"),
    }
    for name, table in sections.items():
        print(f"<!-- {name} -->\n{table}\n")


if __name__ == "__main__":
    main()
