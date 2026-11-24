# Forecast PowerPulse API

FastAPI backend that forecasts **annual electricity generation in Germany (TWh)** with
Prophet, Random Forest, XGBoost and a persistence benchmark. It serves the
[frontend-PowerPulse](https://github.com/sinamoghtaderfar/frontend-PowerPulse) web app.

> **Accuracy in one sentence:** in leakage-free backtests none of the models beats the
> naive persistence forecast (last observed value) reliably; persistence is therefore
> served as the benchmark and currently reported as `recommended_model`. Details,
> experiments and limitations: [docs/FORECAST_ACCURACY_REPORT.md](docs/FORECAST_ACCURACY_REPORT.md).

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
* **Prophet** uses a linear trend with the three exogenous regressors. Its long-run path is
  driven by the exogenous assumptions (2035: 322 TWh with the production projection,
  435 TWh with frozen exogenous values) and turns negative from 2045 without the floor below.
* **Persistence** (benchmark): every future year equals the last observed value; no
  parameters, ignores the exogenous projections.
* Point forecasts are floored at 0 TWh (physical bound); when the floor binds (Prophet beyond
  ~20 years) `metadata.warnings` says so. The floor never binds in the backtests.
* Model artifacts are cached in `app/models/` (or `POWERPULSE_MODELS_DIR`), keyed by a hash of the
  data, model version (currently 4), calibration settings and library version, and retrained when
  any of these change. Artifacts are pickles: only load files written by this service.

## Uncertainty intervals

`lower`/`upper` are **empirical intervals targeting 95 % coverage**, the same method for all
models (since v4): volatility-scaled split-conformal on rolling-origin backtest residuals.
Each backtest error at origin *o* and horizon *h* is scaled by σ_o·√h (σ_o = standard deviation
of the last 10 annual changes known at *o*); the 95 % conformal quantile *q* of the pooled scores
for h = 1–10 gives half-widths q·σ_2025·√h. Lower bounds are clipped at 0. Intervals exist for
**h ≤ 10**; beyond that `lower`/`upper` are `null` (no evidence).

They are **not guaranteed 95 % intervals** (`validated_coverage: false`). Backtested without
look-ahead (origins 2000–2024, h = 1–10, 185 overlapping forecasts) they covered 0.95
(persistence), 0.94 (Random Forest), 0.98 (XGBoost) and only 0.80 (Prophet, 0.64 at h = 10).
The previous per-horizon method covered 0.50–0.77 on the 30 forecasts where it could be
evaluated. `metadata.intervals` reports the coverage per model and horizon range.

## Evaluation

`python -m app.evaluation` (≈15 s, deterministic). Expanding-window, recursive forecasts from
origins 2000–2024 (training starts 1985); *n* = number of origins with an observed target.

| Model | MAE h=1 | MAE h=3 | MAE h=5 | MAE h=10 | sMAPE h=1 | sMAPE h=10 | Bias h=10 |
|---|---|---|---|---|---|---|---|
| Persistence (last value) | 16.2 | 33.8 | 42.4 | 46.7 | 2.7 % | 8.0 % | +19.6 |
| Drift | 15.9 | 33.3 | 46.7 | 62.1 | 2.7 % | 10.3 % | +59.1 |
| Prophet | 20.3 | 36.5 | 49.6 | 65.4 | 3.4 % | 10.9 % | +19.4 |
| Random Forest | 26.3 | 41.9 | 46.6 | 53.9 | 4.5 % | 9.3 % | +6.5 |
| XGBoost | 18.5 | 34.4 | 42.5 | 45.8 | 3.1 % | 7.9 % | +16.6 |

MAE and bias (forecast − actual) in TWh; *n* = 25 / 23 / 21 / 16 for h = 1 / 3 / 5 / 10. RMSE and
WAPE are in the CLI output.

**Conclusion:** none of the models clearly beats persistence; XGBoost is roughly on par, Prophet
and Random Forest are worse (Random Forest significantly at h = 1). 19 further candidates
(ETS, damped trend, Theta, ARIMA, ridge, direct models, tree models on changes, a demand/net-import
decomposition, online ensembles) were tested under a pre-registered protocol with a 2015–2025
holdout; none passed. Main reasons: the structural decline since 2018 (about half of it a swing
from net exports to net imports, which the target includes), tree models that cannot forecast
below their training range, and exogenous variables without stable predictive value. Horizons
6–10 rest on few overlapping windows; long-term accuracy is not established. See the
[accuracy report](docs/FORECAST_ACCURACY_REPORT.md).

## API

| Endpoint | Description |
|---|---|
| `GET /forecast/all?years_ahead=10` | All models: `forecasts`, `summaries`, `metadata` |
| `GET /forecast/prophet` · `/forecast/random-forest` · `/forecast/xgboost` · `/forecast/persistence` | Single model |
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

Metadata (additive in v1.2, all optional for clients):

| Field | Meaning |
|---|---|
| `benchmark_model` | Always `persistence` |
| `recommended_model` | Lowest backtest MAE over h = 1–10 (origins ≥ 2000); the benchmark unless another model is > 2 % better |
| `evidence.{model}` | Out-of-sample MAE, bias and MAE relative to persistence per horizon 1–10 |
| `horizon_guidance` | Evidence per horizon range: 1–3, 4–5, 6–10 (weak), > 10 (exploratory, no interval) |
| `assumptions` | Exogenous projection method and growth rates, carried-forward data gaps |
| `warnings` | e.g. forecasts that reached the 0 TWh floor |
| `intervals.{model}.backtest_coverage` | Sequential interval coverage per horizon range |

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
black --check app tests run.py experiments && pylint app run.py
python -m app.evaluation --json report.json
```

Research experiments (separate from the API code, see `experiments/README.md`):

```bash
pip install -r requirements-experiments.txt
python -m experiments.run_all --fresh && python -m experiments.run_intervals
```
