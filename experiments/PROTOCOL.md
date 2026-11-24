# Experiment protocol (written before running the candidate models)

This file fixes the validation design, the candidate set and the promotion rule
*before* any candidate was evaluated, so that model choice is not tuned to the
realised 2015–2025 path. Deviations made later are listed at the end.

## Data

* Target: `electricity_generation` (TWh, annual, Germany, OWID), 1985–2025, 41 rows,
  read with `app.services.features.read_energy_csv` (no imputation at load time).
* Exogenous variables (only for models that use them): `population`, `gdp`,
  `renewables_share_energy`, imputed by carry-forward **inside each origin**.
* Decomposition experiment only: `electricity_demand` and `net_elec_imports`
  from `app/data/germany-energy-data.csv` (OWID, 1990–2025). Identity checked:
  `generation = demand - net_imports`.

## Backtest

* Expanding window, rolling origin, genuine multi-step forecasts
  (`app.services.backtesting.rolling_origin_backtest`).
* Origins 1996–2024 are run (12 training years minimum); every forecast is made
  for all years up to 2025. Each forecaster only receives rows `year <= origin`.
* Fixed seeds; no random splits; no shuffling.

## Evaluation sets

| Set | Records | Purpose |
|---|---|---|
| FULL | origin 2000–2024 | Comparable with the README baseline |
| DEV | origin >= 2000 **and** target year <= 2014 | Information available at end-2014 |
| HOLDOUT | origin 2014–2024 (targets 2015–2025) | Used once, to confirm the DEV decision |

DEV and HOLDOUT are disjoint and information-separated (no DEV record uses a target
observed after 2014). HOLDOUT is short (n = 11 at h = 1, n = 2 at h = 10); it
cannot confirm long-horizon accuracy and this is reported, not hidden.

## Metrics (per horizon h)

MAE, RMSE, sMAPE (%), WAPE (%) = sum|e| / sum|y| x 100, bias = mean(forecast - actual)
(positive = over-forecast), relative MAE = MAE / MAE(persistence) on the same records,
n. Horizon groups: 1–3, 4–5, 6–10.

Paired comparisons by origin: Diebold–Mariano test on absolute-error differences
with Newey–West variance (lag h-1) and the Harvey–Leybourne–Newbold small-sample
correction (t with n-1 df), plus a moving-block bootstrap (block = max(h, 2),
2000 resamples, seed 0) for the relative MAE. Overlapping h-step errors are
autocorrelated; the tests account for this only approximately and have low
power with n <= 25.

## Interval evaluation

Sequential (no look-ahead): an interval for (origin, h) may only use residuals
whose target year is <= origin. Coverage and mean width per horizon; methods
are compared on the common (origin, h) pairs where all of them produce an
interval, and separately on all pairs each method can cover.

## Pre-registered candidates

Univariate, target only:
`drift_recent5`, `drift_recent10`, `ses`, `holt`, `damped` (ETS A,Ad,N),
`theta` (standard Theta, Hyndman–Billah form), `arima_auto` (AICc over
p in {0,1,2}, d = 1, q in {0,1}, with/without drift, selected inside each origin),
`comb_shd` (mean of SES, Holt, damped; M3 "Comb S-H-D").

Linear ML: `ridge_ar_diff` (recursive ridge on differences, 3 lags, alpha chosen
by inner chronological CV), `direct_ridge` (one ridge model per horizon on
h-step changes), `ardl_gdp` (ridge on log-differences with GDP growth; needs
projected GDP).

Tree ML: `xgb_diff`, `rf_diff` (predict the annual change instead of the level,
reduced features, shallow fixed hyperparameters), `xgb_diff_tuned` (small grid,
nested inner one-step chronological CV).

Prophet: `prophet_noreg` (trend only).

Decomposition: `decomp_damped` (damped(demand) - persistence(net imports)).

Exogenous projections for Prophet / RF / XGBoost: `cagr10` (current), `cagr5`,
`cagr20`, `cagr10_damped` (growth multiplied by 0.8^step), `flat` (last value),
`linear10` (OLS trend over 10 years). `oracle` (actual future exogenous values)
is a **diagnostic upper bound only** - it leaks future information by design and
is never a candidate.

Online (nested) meta-forecasters: `online_best` (per horizon, the pool member
with the lowest MAE on records already observed at the origin) and
`online_invmse` (inverse-MSE weights from the same records). Pool: persistence,
drift, ses, holt, damped, theta, arima_auto, prophet, random_forest, xgboost.

Interval methods: `conformal_h` (current, per horizon), `conformal_pooled_sqrt`
(residuals scaled by sqrt(h) and pooled over horizons), `conformal_pooled_sqrt_recent`
(same, residuals weighted by recency, half-life 8 years).

## Promotion rule (fixed in advance)

A candidate becomes the recommended production model only if

1. on DEV its MAE is not larger than persistence at h = 1, 3 and 5, and its mean
   relative MAE over h = 1..10 is below 1; **and**
2. on HOLDOUT (evaluated once) its MAE is not larger than persistence at h = 1, 3, 5.

Among passing candidates the simplest (fewest fitted parameters) wins when the
FULL mean relative MAE over h = 1..10 is within 2 % of the best. If nothing passes,
no new model is promoted as default and the result is reported as negative.
Changes to the existing models (Prophet / RF / XGBoost) are kept only if they
lower FULL MAE at h = 1, 3, 5 and 10 relative to the original configuration,
without increasing it at any of these horizons by more than 2 %.

## Deviations

Recorded after the experiments; the data split, the candidate list and the
promotion rule above were **not** changed.

1. **Volatility-scaled intervals (post hoc).** All three pre-registered interval
   methods under-covered (sequential coverage 0.50–0.81 at 95 % nominal). Four
   volatility-scaled variants were then tried (`scaled_w5_pall`, `scaled_w10_pall`,
   `scaled_w10_p5`, `scaled_w10_p10`; window of annual changes for sigma / largest
   pooled horizon). `scaled_w10_p10` was adopted: the 10-year window matches the
   existing `GROWTH_WINDOW_YEARS`, and pooling horizons 1–10 matches the horizon
   range that the backtest can evaluate. Because the choice among four variants was
   made after seeing all periods, the reported coverage is mildly optimistic.
2. **Testability guard.** The DM test and block bootstrap are reported only when
   there are at least 2h + 2 paired forecasts (added after the h = 10 results with
   n = 16 and block length 10 were seen to be degenerate). No conclusion depended on
   the dropped values.
3. **Physical floor (post hoc).** A new test found that the existing Prophet
   service forecasts negative generation from 2045 (h = 20). A log-target Prophet
   was tried once and rejected (FULL MAE h = 10: 65.4 → 100.6 TWh). Point forecasts
   are instead floored at 0 TWh in backtests and production; no backtest forecast
   was negative, so no backtest metric changed.
4. **Persistence served as benchmark.** Not a candidate: since no candidate passed
   the rule, the benchmark itself is exposed in the API as the reference/recommended
   forecast.
5. All candidates were run in one batch, so FULL, DEV and HOLDOUT results were
   visible together. The promotion decision is mechanical under the fixed rule;
   any later idea (items 1 and 3) can no longer be confirmed on an untouched holdout.
