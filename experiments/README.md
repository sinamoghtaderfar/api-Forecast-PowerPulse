# Forecast-accuracy experiments

Research code behind `docs/FORECAST_ACCURACY_REPORT.md`. It is kept separate from
the production package `app/`: it imports the production pipeline (data loading,
backtesting, services) but nothing in `app/` imports it.

* `PROTOCOL.md` – validation design, candidates and promotion rule fixed before the
  runs, plus the recorded deviations.
* `candidates.py` – candidate forecasters (naive, ETS, Theta, ARIMA, ridge, tree models
  on changes, decomposition). Model selection inside a candidate only uses the rows of
  its forecast origin.
* `exog_projections.py` – alternative exogenous projections (+ the leaky `oracle`
  diagnostic, never a candidate).
* `ensembles.py` – online (nested) model selection / inverse-MSE combination.
* `intervals.py` – interval methods evaluated sequentially (no look-ahead).
* `metrics.py` – metrics, DEV/HOLDOUT/FULL sets, Diebold–Mariano and block bootstrap.

```bash
pip install -r requirements-experiments.txt     # adds statsmodels (ARIMA only)
python -m experiments.run_all --fresh           # ~3 min, all backtests -> results/
python -m experiments.run_intervals             # interval methods -> results/intervals.json
python -m experiments.diagnostics               # audit diagnostics -> results/diagnostics.json
python -m experiments.report_tables             # markdown tables used in the report
```

All results are deterministic (fixed seeds, single-threaded models). `results/records/`
holds every individual backtest forecast (`model, origin, horizon, year, actual,
predicted`).
