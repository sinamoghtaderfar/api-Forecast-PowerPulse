# Forecast PowerPulse API

FastAPI backend that forecasts **annual electricity generation in Germany (TWh)** with
Prophet, Random Forest and XGBoost. It serves the
[frontend-PowerPulse](https://github.com/sinamoghtaderfar/frontend-PowerPulse) web app.

Demo video: https://youtu.be/oULub0Q4xnE

## Data

`app/data/germany-energy-clean.csv` – Germany, 1985–2025 (41 consecutive years), extracted from the
[Our World in Data energy dataset](https://github.com/owid/energy-data) (bundled as `owid-energy-data.csv`).

| Column | Meaning | Unit |
|---|---|---|
| `electricity_generation` | **Target**: total electricity generation (not consumption/demand) | TWh |
| `population` | Population | people |
| `gdp` | OWID GDP (constant international-$) | $ |
| `renewables_share_energy` | Renewables share of primary energy | % |

Known gaps (verified against the OWID source): GDP is missing for 2023–2025 and the renewables
share for 2025. These cells are left empty; the loader carries the last observed value forward
(never backward). Values are not interpolated or invented.

## Models and assumptions

* All models use the same pipeline in production and evaluation: fit on the history up to the
  forecast origin, project the exogenous variables, forecast recursively year by year.
* **Future exogenous values** are projected with the compound annual growth rate of the observed
  values in the trailing 10 years (renewables capped at 100 %). They are assumptions, not data.
* **Random Forest / XGBoost** use lagged targets, rolling statistics, time and economic features.
  Features of year *t* only use targets before *t*. Tree models cannot predict outside the range
  of historical targets, which limits them during sustained declines.
* **Prophet** uses a linear trend with the three exogenous regressors.
* Model artifacts are cached in `app/models/` (or `POWERPULSE_MODELS_DIR`), keyed by a hash of the
  data, model version, calibration settings and library version, and retrained when any of these
  change. Artifacts are pickles: only load files written by this service.

## Uncertainty intervals

`lower`/`upper` are **empirical prediction intervals targeting 95 % coverage**, the same method for
all models: for horizon *h* the half-width is the split-conformal quantile of absolute errors of
recursive backtest forecasts made at origins 1996–2024 at that horizon (widths are non-decreasing
in *h*; lower bounds are clipped at 0). With 95 % at least 19 residuals are required, so intervals
exist up to horizon **11**; beyond that `lower`/`upper` are `null`.

They are **not validated 95 % intervals**: when re-evaluated without look-ahead (table below),
observed coverage was below nominal, largely because of the structural decline since 2018. Treat
them as indicative ranges. `metadata.intervals` in the API describes the evidence per model.

## Evaluation

`python -m app.evaluation` (≈15 s, deterministic). Expanding-window, recursive forecasts from
origins 2000–2024 (training starts 1985); *n* = number of origins with an observed target.

| Model | MAE h=1 | MAE h=3 | MAE h=5 | MAE h=10 | sMAPE h=1 | sMAPE h=10 |
|---|---|---|---|---|---|---|
| Persistence (last value) | 16.2 | 33.8 | 42.4 | 46.7 | 2.7 % | 8.0 % |
| Drift | 15.9 | 33.3 | 46.7 | 62.1 | 2.7 % | 10.3 % |
| Prophet | 20.3 | 36.5 | 49.6 | 65.4 | 3.4 % | 10.9 % |
| Random Forest | 26.3 | 41.9 | 46.6 | 53.9 | 4.5 % | 9.3 % |
| XGBoost | 18.5 | 34.4 | 42.5 | 45.8 | 3.1 % | 7.9 % |

MAE in TWh; *n* = 25 / 23 / 21 / 16 for h = 1 / 3 / 5 / 10. RMSE is in the CLI output.

Sequential 95 % interval coverage (intervals built only from residuals known at each origin):
h=1: Prophet 0.90, RF 0.80, XGBoost 0.90 (*n* = 10); h=3: 0.83 / 0.17 / 0.33 (*n* = 6).

**Conclusion:** none of the models clearly beats the naive persistence baseline; XGBoost is roughly
on par with it, Prophet and Random Forest are worse. With 41 annual observations and overlapping
backtest windows, the differences are not statistically conclusive. Reduced feature sets for the
tree models gave no material improvement, so the feature sets were left unchanged.

## API

| Endpoint | Description |
|---|---|
| `GET /forecast/all?years_ahead=10` | All models: `forecasts`, `summaries`, `metadata` |
| `GET /forecast/prophet` · `/forecast/random-forest` · `/forecast/xgboost` | Single model |
| `GET /health` | Status and per-model availability (always 200) |
| `GET /health/live` · `GET /health/ready` | Liveness; readiness (503 until all models are loaded) |

`years_ahead`: integer 1–50 (422 otherwise). Rows: `year`, `forecast` (TWh), `lower`, `upper`
(nullable); Random Forest adds `uncertainty_width` and `growth_rate`, XGBoost `growth_from_previous`
(% change vs. the previous year, the first year vs. the last observed year).

Summary fields: `growth_periods` = number of annual steps (years − 1);
`total_growth_percent` = (end/start − 1)·100; `avg_annual_growth` = compound annual growth rate
((end/start)^(1/growth_periods) − 1)·100. Both are `null` for a one-year horizon or non-positive
values. (Before v1.1 `avg_annual_growth` was total growth divided by the number of years.)

If a model fails to load, its endpoint returns 503 and `/forecast/all` returns an empty list for it,
listed in `metadata.unavailable_models`.

## Running

Python 3.12 or 3.13.

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run.py                                      # http://127.0.0.1:8000
```

The first start trains and calibrates all models (~20 s); later starts load the cached artifacts.

| Variable | Default | Purpose |
|---|---|---|
| `POWERPULSE_HOST` | `127.0.0.1` | Bind address (`0.0.0.0` to expose) |
| `POWERPULSE_PORT` | `8000` | Port |
| `POWERPULSE_RELOAD` | `0` | Auto-reload (development only) |
| `POWERPULSE_LOG_LEVEL` | `info` | Log level |
| `POWERPULSE_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Allowed origins (comma-separated) |
| `POWERPULSE_MODELS_DIR` | `app/models` | Model artifact directory |
| `POWERPULSE_SKIP_STARTUP_TRAINING` | `0` | Skip model loading at startup |

The frontend dev server proxies `/forecast/*` to `http://localhost:8000`.

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest
black --check app tests run.py && pylint app run.py
python -m app.evaluation --json report.json
```
