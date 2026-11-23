# PowerPulse forecast accuracy report

Scope: annual German **electricity generation** (TWh, OWID), 1985–2025, as served by
the PowerPulse API. Code version: API 1.2.0, model version 4. All numbers below are
produced by code in this repository and are reproducible (section 13).

## 1. Summary

* **The baseline in the README was reproduced exactly.** On the rolling-origin
  backtest (origins 2000–2024) none of Prophet, Random Forest or XGBoost beats the
  naive persistence forecast reliably. Random Forest is significantly worse at
  h = 1 (MAE 26.3 vs 16.2 TWh, DM p < 0.01).
* **19 further candidates were tested under a protocol fixed in advance**
  (`experiments/PROTOCOL.md`): ETS, damped trend, Theta, auto-ARIMA, ridge AR,
  direct multi-horizon models, tree models on annual changes, an ARDL GDP model,
  Prophet without regressors, a demand/net-imports decomposition and two online
  (nested) ensembles. **None passed the pre-registered promotion rule.** Methods
  that beat persistence on information up to 2014 (damped trend, Theta, Comb S-H-D)
  were worse than persistence on all of h = 1, 3, 5 in 2015–2025. No difference from
  persistence is statistically significant in favour of a candidate.
* **Main causes of error** (section 5): a structural break (mean annual change
  +4.9 TWh in 1985–2007, −18.4 TWh in 2018–2025); half of the 2017–2025 decline is a
  swing in net electricity trade, which the target includes but no feature
  describes; level-target tree models cannot forecast below their training minimum
  (71 of 325 backtest targets were below it; mean error +105 to +107 TWh); the exogenous
  regressors carry little stable information (even the *actual* future values do
  not improve RF/XGBoost, and they make Prophet worse at h = 1–5).
* **Implemented (evidence-backed):**
  1. **Prediction intervals** replaced by volatility-scaled, horizon-pooled
     split-conformal intervals. Sequential (no look-ahead) coverage over
     h = 1–10 rose from 0.50–0.77 (legacy, only 30 evaluable forecasts) to
     0.94–0.98 for persistence, RF and XGBoost (185 forecasts); Prophet reaches
     only 0.80 and is flagged. Intervals are reported for h ≤ 10 only.
  2. **Persistence benchmark served** (`/forecast/persistence`, also in
     `/forecast/all`) and reported as `recommended_model` by an explicit rule;
     relative to the previously best served model (XGBoost) it lowers MAE at h = 1
     by 12 % (16.2 vs 18.5 TWh) and is within ±2 % at h = 3, 5, 10.
  3. **Physical validity:** a new test showed the existing Prophet model forecasts
     *negative* generation from 2045. Forecasts are now floored at 0 TWh and
     flagged in `metadata.warnings`.
  4. **Honest metadata:** per-model out-of-sample evidence (MAE, bias and MAE
     relative to persistence per horizon), horizon guidance (1–3 / 4–5 / 6–10 /
     exploratory), interval coverage evidence, exogenous assumptions and data gaps.
  5. A reproducible experiment harness (`experiments/`) with all individual
     backtest forecasts and results in CSV/JSON.
* **Not changed:** the point forecasts of Prophet, Random Forest and XGBoost (no
  variant passed the rule for changing existing models), the exogenous projection
  (more accurate projections did not translate into better generation forecasts),
  and the target definition.

## 2. System audit

### 2.1 Data flow (verified in code)

1. `app/data/germany-energy-clean.csv` → `features.read_energy_csv`: validates
   consecutive years, positive target; out-of-range exogenous values → NaN. No
   imputation at load time.
2. `features.prepare_history`: exogenous gaps carried forward (never backward);
   leading rows without exogenous data dropped.
3. `features.exogenous_growth_rates`: CAGR of *observed* values in the trailing
   10 years; `exogenous_path` compounds them from the last (carried-forward) row,
   renewables share capped at 100 %.
4. Model fit (`_fit_model`) and forecast (`_predict_point`): Prophet predicts all
   years at once from the projected regressors; RF/XGBoost build lag/rolling/time/
   economic features (`compute_features`) and forecast **recursively**
   (`recursive_forecast`).
5. Calibration: `backtesting.rolling_origin_backtest` refits the full pipeline at
   every origin 1996–2024 on rows `year <= origin`, forecasts to 2025; residuals
   feed the interval calibration.
6. Persistence: `joblib`/JSON artifacts in `app/models/`, keyed by a SHA-256 of
   the data file, model version, calibration settings and library versions.
7. API: `/forecast/all`, `/forecast/{model}`; a per-horizon in-memory cache.

### 2.2 Documented assumptions checked

| Assumption | Result |
|---|---|
| Target is generation, not consumption | Verified (`TARGET_COLUMN = "electricity_generation"`; equals OWID demand − net imports to 0.05 TWh) |
| 41 annual observations 1985–2025 | Verified |
| Prophet, RF, XGBoost | Verified; Prophet: linear trend, default changepoints, 3 regressors |
| Lags, rolling statistics, population, GDP, renewables share | Verified; renewables share is of **primary energy**, not of electricity |
| GDP missing 2023–2025, renewables share 2025 | Verified; carried forward (production) — backtests at origins ≤ 2022 never see this gap |
| Recursive forecasting; projected exogenous values | Verified for RF/XGBoost; Prophet is not recursive (direct in time) |
| Expanding-window backtesting without leakage | Verified by reading and by the existing leakage test; origins 1996–2024 for calibration, 2000–2024 for evaluation |
| Residual-based 95 % intervals | Verified (per-horizon split conformal); legacy sequential coverage 0.0–0.90 |

No leakage was found in the existing pipeline. Two caveats remain that cannot be
fixed with the bundled data: the OWID file is a 2026 vintage, so historical values
(especially GDP) are revised values a forecaster at the time would not have had;
and the 2025 values come from the latest Ember update included in OWID and may be
revised.

### 2.3 Version check

The attached folder (`E:\api-Forecast-PowerPulse`, branch
`fix/forecast-correctness-validation` at `5fb8718`) is content-identical to GitHub
`main` (`f334e98`, the merge of that branch); only line endings differ (CRLF
locally). The work is on a new branch `feat/forecast-accuracy-research` from `main`.

## 3. Validation protocol

* Expanding-window rolling origin, genuine multi-step forecasts, every forecaster
  receives only rows `year <= origin` (imputation, growth rates, sigma, model
  selection all inside the origin). Fixed seeds, single-threaded models.
* Sets: **FULL** = origins 2000–2024 (comparable with the README); **DEV** = origin
  ≥ 2000 and target year ≤ 2014 (what was knowable at end-2014); **HOLDOUT** =
  origins 2014–2024 (targets 2015–2025), disjoint from DEV. n (FULL/DEV/HOLDOUT):
  h1 25/14/11, h3 23/12/9, h5 21/10/7, h10 16/5/2.
* Metrics: MAE, RMSE, sMAPE, WAPE, bias (= mean forecast − actual), MAE relative to
  persistence on the same forecasts.
* Paired comparisons by origin: Diebold–Mariano on absolute errors with Newey–West
  variance (lag h − 1) and the Harvey–Leybourne–Newbold correction, plus a
  moving-block bootstrap of the MAE ratio (block = max(h, 2)); only reported when
  n ≥ 2h + 2 ("not testable" otherwise).
* Intervals: sequential coverage — an interval for (origin, h) may only use
  residuals whose target year is ≤ origin.
* Promotion rule (fixed in advance): DEV MAE ≤ persistence at h = 1, 3, 5 and DEV mean
  relative MAE over h = 1–10 < 1, **and** HOLDOUT MAE ≤ persistence at h = 1, 3, 5.
  Existing models may only be changed if FULL MAE improves at h = 1, 3, 5, 10 with
  no horizon worse by more than 2 %.

**Statistical limitations.** 41 observations, 25 origins, and h-step errors from
neighbouring origins overlap (errors at h = 10 from origins 2000–2015 share most
of their target years). Effective sample sizes are far below the nominal n; the
tests are approximate and have low power. HOLDOUT cannot say anything about
h ≥ 6 (n ≤ 5). Long-term (6–10 year) accuracy is **not established** for any model.

## 4. Frozen baseline (original system)

`experiments/results/baseline_frozen.json`; FULL set, MAE/RMSE/bias in TWh.

| Model | h | n | MAE | RMSE | sMAPE % | WAPE % | Bias |
|---|---|---|---|---|---|---|---|
| persistence | 1 | 25 | 16.2 | 21.9 | 2.7 | 2.7 | +2.9 |
| persistence | 3 | 23 | 33.8 | 40.3 | 5.8 | 5.6 | +10.3 |
| persistence | 5 | 21 | 42.4 | 53.7 | 7.2 | 7.1 | +14.5 |
| persistence | 10 | 16 | 46.7 | 64.0 | 8.0 | 7.9 | +19.6 |
| drift | 1 | 25 | 15.9 | 22.4 | 2.7 | 2.7 | +6.0 |
| drift | 3 | 23 | 33.3 | 43.2 | 5.7 | 5.6 | +20.8 |
| drift | 5 | 21 | 46.7 | 60.6 | 7.8 | 7.8 | +33.0 |
| drift | 10 | 16 | 62.1 | 84.4 | 10.3 | 10.5 | +59.1 |
| prophet | 1 | 25 | 20.3 | 27.0 | 3.4 | 3.4 | +7.6 |
| prophet | 3 | 23 | 36.5 | 43.7 | 6.1 | 6.1 | +15.8 |
| prophet | 5 | 21 | 49.6 | 60.6 | 8.2 | 8.3 | +21.1 |
| prophet | 10 | 16 | 65.4 | 81.2 | 10.9 | 11.1 | +19.4 |
| random_forest | 1 | 25 | 26.3 | 31.5 | 4.5 | 4.4 | +1.7 |
| random_forest | 3 | 23 | 41.9 | 51.8 | 7.2 | 7.0 | +5.5 |
| random_forest | 5 | 21 | 46.6 | 61.2 | 7.9 | 7.8 | +6.7 |
| random_forest | 10 | 16 | 53.9 | 64.1 | 9.3 | 9.1 | +6.5 |
| xgboost | 1 | 25 | 18.5 | 24.1 | 3.1 | 3.1 | +1.8 |
| xgboost | 3 | 23 | 34.4 | 41.4 | 5.9 | 5.8 | +6.8 |
| xgboost | 5 | 21 | 42.5 | 53.2 | 7.2 | 7.1 | +12.0 |
| xgboost | 10 | 16 | 45.8 | 60.8 | 7.9 | 7.8 | +16.6 |

By horizon group (FULL, MAE / bias): persistence 24.0 / +6.5 (h1–3), 41.0 / +13.4
(h4–5), 46.5 / +19.9 (h6–10); XGBoost 26.3 / +4.5, 40.4 / +11.1, 45.0 / +16.7;
Prophet 27.8 / +11.5, 47.3 / +19.8, 62.5 / +25.4; RF 33.9 / +3.6, 45.9 / +6.6,
50.4 / +7.4.

Legacy interval coverage (95 % nominal, sequential; only 2–10 evaluable forecasts
per horizon, none beyond h = 5):

| Model | h1 cov / width (n) | h3 | h5 |
|---|---|---|---|
| persistence | 0.90 / 96 (10) | 0.50 / 122 (6) | 0.50 / 138 (2) |
| prophet | 0.90 / 125 (10) | 0.83 / 158 (6) | 0.50 / 181 (2) |
| random_forest | 0.80 / 98 (10) | 0.17 / 121 (6) | 0.00 / 146 (2) |
| xgboost | 0.90 / 96 (10) | 0.33 / 125 (6) | 0.50 / 151 (2) |

README claim verified: no complex model clearly beats persistence; XGBoost is on
par (relative MAE 1.14 / 1.02 / 1.00 / 0.98 at h = 1 / 3 / 5 / 10, none significant).

## 5. Why accuracy is limited (ranked)

1. **Structural break / regime change (dominant).** Mean annual change: +4.9 TWh
   (1986–2007, sd 10.0), +1.3 TWh (2008–2017, sd 22.4), −18.4 TWh (2018–2025, sd 23.9);
   Welch test for a shift in 2018: p = 0.035. Every model over-forecasts in the last
   decade: mean error for h = 1–5 from origins 2015–2024 is +42 to +56 TWh for all
   models (persistence +47, XGBoost +44, RF +56). Models that extrapolate the
   pre-2018 trend (drift, Holt, damped, Theta) suffer most; methods selected on DEV
   fail on HOLDOUT (section 6.2).
2. **Target definition: generation includes the trade balance.** Generation =
   demand − net imports. 2017–2025: generation −147 TWh = demand −75 and net-import
   swing −72 (Germany moved from −52 TWh net exports to +20 TWh net imports);
   2022–2025: −69 = −22 (demand) −47 (trade). Nothing in the feature set describes
   trade, nuclear phase-out or prices, so roughly half of the recent decline is
   unexplainable by the models.
3. **Tree models cannot extrapolate.** Random Forest and XGBoost never forecast
   below the minimum of their training targets (0 % of forecasts); 71 of 325 FULL
   targets were below it, with mean error +105 TWh (RF) / +107 TWh (XGBoost) on those. At the 2024 origin RF
   forecasts 527 TWh against a last value of 493 TWh.
4. **Weak and unstable exogenous relationships.** RF uses the exogenous features for
   8 % of its importance and its forecasts are identical under every projection
   tested; XGBoost 14 %. Feeding RF/XGBoost the *actual* future GDP, population and
   renewables share changes MAE by −2.4 % to +1.3 %. For Prophet the actual values
   make h = 1–5 worse (−4 % to −8 %); its regressor coefficients are not stable over
   time.
5. **Exogenous projections are compounding and drive Prophet's long run.** The
   renewables share (of primary energy) is projected with a 10-year CAGR (6.9 %/yr
   now): its projection error is 55 % at h = 10 (linear trend: 29 %). Prophet's 2035
   forecast ranges from 322 TWh (current CAGR) to 435 TWh (flat exogenous values),
   and it reaches negative values from 2045.
6. **Overfitting relative to data size.** RF: 28 features on 36 training rows, XGBoost
   16 on 38; RF is significantly worse than persistence at h = 1 (p < 0.01).
7. **Small sample / few independent periods.** Even a correct model cannot be shown
   to be better: the bootstrap 95 % intervals of relative MAE span roughly 0.8–1.2 at
   h = 1–5.
8. **Recursive error accumulation is not the main problem.** Direct multi-horizon
   ridge was worse than recursive ridge (FULL MAE h = 10: 76.4 vs 57.1).
9. **Data imputation is minor.** Carry-forward of GDP 2023–2025 only affects the
   exogenous inputs, which barely matter (item 4).

## 6. Experiments

### 6.1 Candidates and results

Hypothesis common to all: the candidate has lower out-of-sample MAE than
persistence at the important horizons, in both regimes. Runtime is for all 29
origins (single thread). FULL relative MAE vs persistence at h = 1 / 3 / 5 / 10.

| Candidate | Why it could work / assumptions | Data | Cost | FULL rel. MAE | Decision |
|---|---|---|---|---|---|
| drift_recent5 / 10 | Local trend more relevant than 40-year trend | target | <0.1 s | 1.04/0.86/1.19/1.67; 0.98/0.91/1.08/1.62 | rejected |
| ses | Level-only smoothing, robust to noise | target | 0.3 s | 1.01/1.01/1.01/0.99 | rejected (≈ persistence) |
| holt | Linear local trend | target | 1.4 s | 1.01/0.99/1.11/1.32 | rejected |
| damped (ETS A,Ad,N) | Strong in M3/M4 annual data; trend fades | target | 5.6 s | 1.01/0.98/1.05/1.09 | rejected |
| theta | M3 winner; SES + half linear trend | target | 0.3 s | 0.99/1.01/1.05/1.10 | rejected |
| arima_auto | AICc-selected ARIMA(p,1,q)±drift inside each origin | target | 4.6 s | 1.03/1.06/1.17/1.19 | rejected |
| comb_shd | Combination reduces model risk | target | 7.9 s | 1.00/0.99/1.05/1.05 | rejected |
| ridge_ar_diff | Regularised AR(3) on changes, nested penalty | target | 2.0 s | 1.05/1.02/1.10/1.22 | rejected |
| direct_ridge | One model per horizon avoids recursion | target | 13.7 s | 1.05/1.01/1.14/1.64 | rejected |
| ardl_gdp | Elasticity of generation to GDP growth | target, GDP (projected) | 2.3 s | 1.05/1.00/1.17/1.55 | rejected |
| xgb_diff / rf_diff | Trees on annual changes can extrapolate the level | target, exog | 0.7 s / 12 s | 1.22/1.09/1.11/1.41; 1.03/0.95/1.12/1.56 | rejected |
| xgb_diff_tuned | Same, depth/trees by nested CV | target, exog | 7.2 s | 1.10/1.04/1.13/1.73 | rejected |
| prophet_noreg | Removes spurious regressor effects | target | 2.2 s | 1.68/1.33/1.40/1.89 | rejected |
| decomp_damped | Model demand (smooth) and net imports (persistent) separately | demand, net imports 1990+ | 5.6 s | 1.01/0.92/0.97/0.94 | rejected by rule (closest) |
| online_best | Choose the model with the best past MAE (nested) | pool records | <1 s | 1.04/1.01/1.13/1.37 | rejected |
| online_invmse | Inverse past-MSE weights (nested) | pool records | <1 s | 1.03/1.01/1.06/1.01 | rejected |

All ETS/Theta implementations were cross-checked against statsmodels (identical
SES, Holt and Theta forecasts; damped trend differs only where statsmodels lets φ
exceed the 0.98 bound used here). No LSTM/Transformer/large ensemble was tried: with
36 training rows they cannot be estimated reliably and nothing simpler worked.

### 6.2 MAE by evaluation set

MAE (relative to persistence on the same forecasts). Bold = better than persistence.

| Model | FULL h1 | h3 | h5 | h10 | DEV h1 | h3 | h5 | h10 | HOLDOUT h1 | h3 | h5 | h10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| persistence | 16.2 (1.00) | 33.8 (1.00) | 42.4 (1.00) | 46.7 (1.00) | 14.2 (1.00) | 24.1 (1.00) | 27.6 (1.00) | 30.8 (1.00) | 18.7 (1.00) | 50.8 (1.00) | 76.6 (1.00) | 133.8 (1.00) |
| drift | **15.9** (0.98) | **33.3** (0.98) | 46.7 (1.10) | 62.1 (1.33) | **13.3** (0.93) | **20.5** (0.85) | 28.7 (1.04) | **17.3** (0.56) | 19.2 (1.02) | 56.7 (1.12) | 92.1 (1.20) | 169.8 (1.27) |
| ses | 16.3 (1.01) | 34.1 (1.01) | 42.9 (1.01) | **46.2** (0.99) | **13.8** (0.97) | 24.2 (1.00) | **27.6** (1.00) | 30.8 (1.00) | 19.5 (1.04) | 50.9 (1.00) | 77.2 (1.01) | **132.8** (0.99) |
| damped | 16.4 (1.01) | **33.0** (0.98) | 44.7 (1.05) | 51.0 (1.09) | **12.6** (0.89) | **20.6** (0.85) | **27.4** (1.00) | **13.7** (0.44) | 21.2 (1.13) | 54.7 (1.08) | 87.9 (1.15) | 156.1 (1.17) |
| theta | **16.1** (0.99) | 34.2 (1.01) | 44.7 (1.05) | 51.1 (1.10) | **13.2** (0.93) | **22.2** (0.92) | **27.5** (1.00) | **21.8** (0.71) | 19.7 (1.05) | 55.3 (1.09) | 87.4 (1.14) | 153.6 (1.15) |
| comb_shd | 16.3 (1.00) | **33.4** (0.99) | 44.3 (1.05) | 48.8 (1.05) | **12.9** (0.91) | **21.4** (0.89) | **27.2** (0.99) | **15.6** (0.51) | 20.6 (1.10) | 54.3 (1.07) | 86.4 (1.13) | 152.8 (1.14) |
| arima_auto | 16.7 (1.03) | 35.9 (1.06) | 49.6 (1.17) | 55.6 (1.19) | **14.1** (0.99) | 24.4 (1.01) | 35.2 (1.28) | 34.1 (1.11) | 19.9 (1.06) | 55.8 (1.10) | 87.3 (1.14) | 150.5 (1.13) |
| rf_diff | 16.7 (1.03) | **32.2** (0.95) | 47.6 (1.12) | 72.8 (1.56) | **13.9** (0.98) | **20.8** (0.86) | 35.0 (1.27) | 39.5 (1.28) | 20.3 (1.08) | 52.7 (1.04) | 83.2 (1.09) | 144.7 (1.08) |
| decomp_damped | 16.3 (1.01) | **31.0** (0.92) | **41.2** (0.97) | **43.8** (0.94) | **13.7** (0.97) | 24.2 (1.00) | 28.1 (1.02) | **11.0** (0.36) | 19.6 (1.04) | **43.8** (0.86) | **68.6** (0.90) | **124.2** (0.93) |
| online_invmse | 16.7 (1.03) | 34.2 (1.01) | 44.8 (1.06) | 47.3 (1.01) | **13.4** (0.94) | **22.5** (0.93) | 29.7 (1.08) | **20.0** (0.65) | 21.0 (1.12) | 54.1 (1.07) | 83.5 (1.09) | 144.5 (1.08) |
| prophet | 20.3 (1.25) | 36.5 (1.08) | 49.6 (1.17) | 65.4 (1.40) | 14.5 (1.02) | 25.4 (1.05) | 42.9 (1.56) | 46.6 (1.51) | 27.6 (1.47) | 51.1 (1.01) | **63.3** (0.83) | **36.3** (0.27) |
| random_forest | 26.3 (1.63) | 41.9 (1.24) | 46.6 (1.10) | 53.9 (1.15) | 20.8 (1.47) | 29.3 (1.21) | 31.3 (1.14) | 50.6 (1.64) | 33.2 (1.78) | 62.1 (1.22) | 79.9 (1.04) | **122.5** (0.92) |
| xgboost | 18.5 (1.14) | 34.4 (1.02) | 42.5 (1.00) | **45.8** (0.98) | **13.8** (0.97) | 24.8 (1.03) | 27.8 (1.01) | 33.5 (1.08) | 24.4 (1.31) | **49.8** (0.98) | **74.1** (0.97) | **126.0** (0.94) |

All candidates: `experiments/results/metrics.json`, `relative_mae_*.csv`.

Promotion rule outcome (all 19 candidates in `experiments/results`; selected rows):

| Candidate | DEV ≤ persistence h1/3/5 | DEV mean rel. MAE h1–10 | HOLDOUT ≤ persistence h1/3/5 | Promoted |
|---|---|---|---|---|
| damped | yes/yes/yes | 0.90 | no/no/no | no |
| theta | yes/yes/yes | 0.93 | no/no/no | no |
| comb_shd | yes/yes/yes | 0.89 | no/no/no | no |
| decomp_damped | yes/no/no | 0.80 | no/yes/yes | no |
| ses | yes/no/yes | 0.99 | no/no/no | no |
| drift_recent10 | yes/yes/no | 1.45 | yes/yes/no | no |
| online_invmse | yes/yes/no | 1.01 | no/no/no | no |

The three methods that would have been chosen with the information available at
the end of 2014 (damped, Theta, Comb S-H-D) lost to persistence at every short
horizon afterwards — the central finding of this study.

### 6.3 Paired tests vs persistence (FULL)

Relative MAE (DM p-value, HLN-corrected) [95 % block-bootstrap interval]:

| Model | h1 | h3 | h5 | h10 |
|---|---|---|---|---|
| decomp_damped | 1.01 (p=0.94) [0.85, 1.14] | 0.92 (p=0.62) [0.71, 1.37] | 0.97 (p=0.90) [0.78, 1.83] | 0.94 (not testable) |
| theta | 0.99 (p=0.82) [0.91, 1.05] | 1.01 (p=0.85) [0.86, 1.09] | 1.05 (p=0.53) [0.86, 1.16] | 1.10 (not testable) |
| damped | 1.01 (p=0.81) [0.88, 1.11] | 0.98 (p=0.78) [0.79, 1.11] | 1.05 (p=0.64) [0.86, 1.28] | 1.09 (not testable) |
| drift | 0.98 (p=0.61) [0.88, 1.05] | 0.98 (p=0.88) [0.74, 1.15] | 1.10 (p=0.52) [0.87, 1.46] | 1.33 (not testable) |
| online_invmse | 1.03 (p=0.43) [0.94, 1.12] | 1.01 (p=0.85) [0.87, 1.11] | 1.06 (p=0.49) [0.92, 1.28] | 1.01 (not testable) |
| xgb_diff | 1.22 (p=0.01) [1.06, 1.40] | 1.09 (p=0.65) [0.76, 1.50] | 1.11 (p=0.66) [0.95, 2.12] | 1.41 (not testable) |
| prophet | 1.25 (p=0.09) [1.00, 1.57] | 1.08 (p=0.56) [0.86, 1.48] | 1.17 (p=0.50) [0.95, 2.40] | 1.40 (not testable) |
| random_forest | 1.63 (p=0.00) [1.09, 2.30] | 1.24 (p=0.17) [0.91, 1.48] | 1.10 (p=0.50) [0.82, 1.18] | 1.15 (not testable) |
| xgboost | 1.14 (p=0.22) [0.94, 1.49] | 1.02 (p=0.81) [0.86, 1.13] | 1.00 (p=0.97) [0.93, 1.05] | 0.98 (not testable) |

Full table (also vs drift and vs XGBoost, all sets): `experiments/results/comparisons.csv`.

### 6.4 Required comparison table (selected rows)

Improvement (%) = (baseline MAE − new MAE) / baseline MAE × 100; negative = worse.
Baseline = persistence for new candidates; FULL set.

| Model | Baseline | Horizon | Baseline MAE | New MAE | Improvement % | RMSE | sMAPE % | Bias | Sample Size |
|---|---|---|---|---|---|---|---|---|---|
| decomp_damped | persistence | 1 | 16.2 | 16.3 | -0.7 | 23.8 | 2.8 | +1.9 | 25 |
| decomp_damped | persistence | 3 | 33.8 | 31.0 | +8.4 | 39.9 | 5.3 | +9.0 | 23 |
| decomp_damped | persistence | 5 | 42.4 | 41.2 | +2.8 | 55.1 | 7.1 | +14.3 | 21 |
| decomp_damped | persistence | 10 | 46.7 | 43.8 | +6.3 | 66.6 | 7.7 | +23.2 | 16 |
| damped | persistence | 1 | 16.2 | 16.4 | -1.5 | 23.0 | 2.8 | +4.7 | 25 |
| damped | persistence | 3 | 33.8 | 33.0 | +2.3 | 42.0 | 5.6 | +17.8 | 23 |
| damped | persistence | 5 | 42.4 | 44.7 | -5.4 | 58.3 | 7.5 | +27.4 | 21 |
| damped | persistence | 10 | 46.7 | 51.0 | -9.2 | 74.5 | 8.6 | +45.6 | 16 |
| comb_shd | persistence | 1 | 16.2 | 16.3 | -0.4 | 22.8 | 2.8 | +4.2 | 25 |
| comb_shd | persistence | 3 | 33.8 | 33.4 | +1.1 | 41.9 | 5.7 | +16.4 | 23 |
| comb_shd | persistence | 5 | 42.4 | 44.3 | -4.5 | 57.6 | 7.5 | +25.1 | 21 |
| comb_shd | persistence | 10 | 46.7 | 48.8 | -4.6 | 72.3 | 8.3 | +41.1 | 16 |
| arima_auto | persistence | 1 | 16.2 | 16.7 | -2.9 | 22.9 | 2.8 | +4.6 | 25 |
| arima_auto | persistence | 3 | 33.8 | 35.9 | -6.3 | 43.8 | 6.1 | +16.0 | 23 |
| arima_auto | persistence | 5 | 42.4 | 49.6 | -16.9 | 61.5 | 8.3 | +24.8 | 21 |
| arima_auto | persistence | 10 | 46.7 | 55.6 | -19.1 | 71.4 | 9.4 | +39.4 | 16 |
| direct_ridge | persistence | 10 | 46.7 | 76.4 | -63.7 | 95.0 | 12.6 | +49.7 | 16 |
| rf_diff | random_forest | 1 | 26.3 | 16.7 | +36.4 | 23.0 | 2.8 | +5.7 | 25 |
| rf_diff | random_forest | 3 | 41.9 | 32.2 | +23.1 | 42.2 | 5.4 | +22.2 | 23 |
| rf_diff | random_forest | 5 | 46.6 | 47.6 | -2.2 | 59.7 | 7.9 | +36.5 | 21 |
| rf_diff | random_forest | 10 | 53.9 | 72.8 | -35.1 | 86.7 | 11.8 | +71.1 | 16 |

Complete tables: `experiments/results/comparison_vs_persistence.csv` and
`comparison_vs_original.csv` (variants of the existing models vs the original).

### 6.5 Rejected change to the Random Forest

Modelling annual changes (`rf_diff`) removes the RF's floor problem at short
horizons (h1 −36 %, h3 −23 % MAE) but extrapolates the historical average growth at
long horizons (h10 bias +71 TWh, MAE +35 %). Under the rule for existing models it
was not adopted; the RF is documented as the weakest served model.

## 7. Exogenous projections

Projection accuracy against later observed values (MAPE %, origins 2000–2024):

| Variable | Projection | h1 | h5 | h10 |
|---|---|---|---|---|
| gdp | cagr10 (current) | 1.8 | 3.4 | 7.0 |
| gdp | linear10 | 1.7 | 2.5 | 2.7 |
| gdp | cagr10_damped | 1.7 | 3.2 | 8.8 |
| gdp | flat | 2.4 | 8.3 | 16.3 |
| population | cagr10 (current) | 0.3 | 1.4 | 3.2 |
| population | flat | 0.3 | 1.3 | 2.3 |
| renewables_share_energy | cagr10 (current) | 6.8 | 20.7 | 55.2 |
| renewables_share_energy | linear10 | 6.5 | 15.4 | 28.7 |
| renewables_share_energy | cagr10_damped | 6.8 | 15.3 | 35.5 |
| renewables_share_energy | flat | 9.4 | 36.3 | 58.2 |

The compounding CAGR is clearly the worst projection for the renewables share and a
linear trend is the most accurate for GDP and the renewables share. **But better
projections do not give better generation forecasts** (FULL MAE change vs the
original model, h = 1 / 3 / 5 / 10):

| Model | linear10 | cagr10_damped | cagr20 | flat | oracle (actual future values, diagnostic only) |
|---|---|---|---|---|---|
| Prophet | −2.4 / +4.5 / +5.9 / −25.0 % | −4.0 / +1.9 / +4.8 / −5.9 % | −9.0 / +3.5 / +13.2 / +48.5 % | −2.2 / +2.2 / +2.6 / −22.9 % | −4.0 / −5.0 / −8.4 / +21.8 % |
| Random Forest | ±0.0 everywhere | ±0.1 | ≤0.4 | ≤0.4 | −0.4 / 0.0 / −0.1 / −0.8 % |
| XGBoost | −0.7 / −0.4 / +0.2 / −0.6 % | ±0.4 | +0.1 / +2.0 / +1.3 / −0.5 % | ≤0.9 | −2.4 / −0.1 / +1.3 / −0.3 % |

(Positive = better.) Prophet + cagr20 looks attractive at h = 10 (+48.5 %, n = 16
overlapping windows) but is 9 % worse at h = 1; with the oracle variant also not
consistently better, this is treated as noise, not as an improvement. No projection
change passed the rule, so the production projection is unchanged and is reported
in `metadata.assumptions` as an assumption, not data.

**Scenario sensitivity** (production models fitted on 1985–2025, not validated
forecasts; `experiments/results/diagnostics.json`):

| Model | 2035 with cagr10 (production) | flat exogenous | linear10 | cagr10_damped |
|---|---|---|---|---|
| Prophet | 322 TWh | 435 TWh | 373 TWh | 407 TWh |
| Random Forest | 537 TWh | 537 TWh | 537 TWh | 537 TWh |
| XGBoost | 511 TWh | 517 TWh | 511 TWh | 497 TWh |
| Persistence | 497 TWh | 497 TWh | 497 TWh | 497 TWh |

The spread of Prophet's long-run forecast across assumptions (113 TWh in 2035)
shows that its long-run path is an assumption-driven scenario. An API scenario
endpoint was not added because only Prophet reacts to the assumptions and none of
the assumption sets is better supported by the backtest; the sensitivity is
documented here instead.

## 8. Horizon behaviour

* **1–3 years:** at h = 1 persistence, drift, SES, Theta and the decomposition are
  within ±2 % of each other (FULL); at h = 3 the decomposition is 8 % better and
  drift_recent5 14 % better, neither significant (p = 0.62, 0.53). XGBoost +14 %,
  Prophet +25 %, RF +63 % at h = 1.
* **4–5 years:** XGBoost and persistence equal (relative MAE 1.00 at h = 5);
  trend models drift away (+5 % to +17 %).
* **6–10 years:** persistence, SES, XGBoost and the decomposition are best (relative
  MAE 0.94–1.01); the trend-extrapolating methods are 5–89 % worse. These results rest
  on 16 heavily overlapping windows and **do not establish long-term accuracy**.
* **> 10 years:** no validation possible. The API keeps allowing up to 50 years but
  `metadata.horizon_guidance` marks them as exploratory, intervals are `null`
  beyond h = 10, and the 0 TWh floor is flagged in `metadata.warnings`.
* **Recursive vs direct:** direct ridge was worse than recursive ridge at h = 5 and
  10 (48.5 vs 46.8; 76.4 vs 57.1 TWh): too few rows per horizon (at origin 2000
  only horizons up to 5 have the 8 rows required; longer horizons reuse that model).

## 9. Uncertainty

### 9.1 Method (implemented, `app/services/uncertainty.py`)

For a backtest error *e* made at origin *o* for horizon *h*:
score = |e| / (σ_o · √h), with σ_o = standard deviation of the last 10 annual
changes observed at *o*. Scores of h = 1–10 from origins 1996–2024 are pooled;
*q* = split-conformal 95 % quantile (≥ 19 scores required). Half-width at horizon
*h* = q · σ_T · √h, with σ_T at the production origin (2025: 22.9 TWh, inflated by
the −63 TWh change of 2023). Lower bounds clipped at 0; no interval beyond h = 10.

Why: per-horizon quantiles (legacy) needed ≥ 19 residuals per horizon, so they
could only be evaluated on 30 forecasts and covered 50–60 % (persistence, RF, XGBoost) and 77 % (Prophet);
the residuals before 2018 were too small for the volatile period that followed.
Scaling by recent volatility makes the width react to regime changes, and pooling
gives evidence at every horizon up to 10.

### 9.2 Sequential coverage (95 % nominal, origins ≥ 2000, h = 1–10)

| Model | legacy per-h | pooled √h | pooled √h + recency | scaled w5 all-h | scaled w10 all-h | scaled w10 h≤5 | **scaled w10 h≤10 (adopted)** |
|---|---|---|---|---|---|---|---|
| persistence | 0.60 (n=30) | 0.75 (185) | 0.75 (175) | 0.91 | 0.93 | 0.95 | **0.95** |
| prophet | 0.77 (n=30) | 0.74 | 0.75 | 0.82 | 0.80 | 0.79 | **0.80** |
| random_forest | 0.50 (n=30) | 0.80 | 0.79 | 0.94 | 0.94 | 0.96 | **0.94** |
| xgboost | 0.60 (n=30) | 0.81 | 0.80 | 0.94 | 0.97 | 0.98 | **0.98** |

By horizon (adopted method, coverage / mean width TWh, n):

| Model | h1 | h3 | h5 | h10 |
|---|---|---|---|---|
| persistence | 0.91 / 93 (23) | 0.90 / 156 (21) | 0.95 / 199 (19) | 1.00 / 256 (14) |
| prophet | 0.91 / 152 (23) | 0.86 / 252 (21) | 0.79 / 318 (19) | 0.64 / 376 (14) |
| random_forest | 0.83 / 117 (23) | 0.86 / 194 (21) | 1.00 / 249 (19) | 1.00 / 320 (14) |
| xgboost | 0.91 / 112 (23) | 1.00 / 188 (21) | 1.00 / 243 (19) | 1.00 / 315 (14) |

Production half-widths at the 2025 origin (TWh, h = 1 / 3 / 5 / 10):

| Model | legacy | new |
|---|---|---|
| persistence | 63 / 86 / 129 / 152 | 60 / 104 / 134 / 189 |
| prophet | 71 / 88 / 143 / 169 | 103 / 178 / 230 / 326 |
| random_forest | 81 / 113 / 134 / 140 | 76 / 132 / 170 / 241 |
| xgboost | 69 / 91 / 131 / 150 | 67 / 117 / 151 / 213 |

Interpretation and caveats:

* The intervals are **not validated 95 % intervals** (`validated_coverage: false`
  in the API). Coverage estimates come from 185 overlapping forecasts (≈ 25
  origins), the method was chosen among four variants after seeing all periods
  (`experiments/PROTOCOL.md`, deviation 1), and a future break larger than 2023
  would again be under-covered.
* They are wider at long horizons than before; that width is what the data
  support: persistence errors at h = 10 reached 134 TWh (HOLDOUT MAE, n = 2).
* Prophet's intervals are wide *and* under-cover (0.64 at h = 10) because its
  errors are trend errors not explained by volatility — another reason it is not
  recommended.
* Beyond h = 10 there is no evidence; `lower`/`upper` are `null` (previously up to
  h = 11).

## 10. Implemented changes

| File | Change | Why |
|---|---|---|
| `app/services/uncertainty.py` | `ScaledResidualCalibration`, `volatility_scale`; legacy `ResidualCalibration` kept | Section 9 |
| `app/services/base.py` | Uses the scaled calibration; `model_version` 3 → 4; interval settings in the cache hash; `physical_floor` (0 TWh) in backtests and production; `uses_exogenous` flag | Calibration, cache invalidation, physical validity |
| `app/services/persistence_service.py` | New persistence benchmark service (JSON artifact) | Best-supported model (section 6) |
| `app/services/evidence.py` | Per-model backtest evidence, `recommended_model` rule, horizon guidance, assumptions | Honest metadata |
| `app/dependencies.py`, `app/main.py` | Registers `persistence`; API version 1.2.0 | Additive |
| `app/controllers/energy_controller.py` | `/forecast/persistence`; metadata `evidence`, `recommended_model`, `horizon_guidance`, `assumptions`, `warnings` | Additive |
| `app/schemas.py` | New optional fields (`IntervalInfo.backtest_coverage`, `validated_coverage`, …), `ForecastsByModel.persistence` | Additive; existing fields unchanged |
| `app/evaluation.py` | WAPE and bias; production-method sequential coverage; legacy coverage kept; check that the served persistence model equals the baseline | Reporting |
| `experiments/` | Protocol, candidates, harness, interval study, diagnostics, results | Reproducible research (not imported by `app/`) |
| `tests/` | 33 new tests (105 in total); `test_api.py` expects the additive `persistence` key | Verification |
| `README.md`, `docs/`, CI | Documentation; `black` also checks `experiments/` | |

`recommended_model` rule: the model with the lowest mean relative MAE over h = 1–10
(origins ≥ 2000); the benchmark unless another model beats it by more than 2 %.
Currently: persistence 1.000, XGBoost 1.014, Random Forest 1.203, Prophet 1.258.

Compatibility: endpoint names, row fields and existing metadata fields are
unchanged; the frontend (`frontend-PowerPulse`, `normalizeForecastResponse`) keeps
only the keys `prophet`, `random_forest`, `xgboost` and ignores unknown keys and
metadata fields, so it is unaffected. Two observable changes: interval widths, and
`max_calibrated_horizon` = 10 (was 11). Old model artifacts are retrained
automatically (hash includes model version and interval settings; a legacy
calibration payload is rejected).

## 11. Before and after

Point accuracy of the existing models: **unchanged** (identical backtest forecasts;
the 0 TWh floor never binds in backtests). Best served forecast, FULL:

| Horizon | Before: best served (XGBoost) MAE | After: recommended (persistence) MAE | Improvement % | Bias before → after | n |
|---|---|---|---|---|---|
| 1 | 18.5 | 16.2 | +12.4 | +1.8 → +2.9 | 25 |
| 3 | 34.4 | 33.8 | +1.7 | +6.8 → +10.3 | 23 |
| 5 | 42.5 | 42.4 | +0.2 | +12.0 → +14.5 | 21 |
| 10 | 45.8 | 46.7 | −2.0 | +16.6 → +19.6 | 16 |

HOLDOUT (2015–2025 targets): h1 24.4 → 18.7 (+23 %), h3 49.8 → 50.8 (−2 %),
h5 74.1 → 76.6 (−3 %). None of these differences is statistically significant
(DM p = 0.22 at h = 1, FULL); the honest statement is "persistence is at least as good
as the best previous model and simpler", not "persistence is better".

Interval coverage (95 % nominal, sequential): legacy 0.50–0.77 on 30 evaluable
forecasts, none beyond h = 5 → adopted method 0.80 (Prophet) / 0.94–0.98 (others)
on 185 forecasts up to h = 10 (section 9.2). RMSE and sMAPE of the existing models
are unchanged; for persistence see section 4.

## 12. Data and target notes

* **Target definition.** Generation includes net exports. Migration recommended
  as a separate change (not done here): (1) add OWID `electricity_demand` (1990+)
  as a second target with the same pipeline; (2) serve generation as
  demand − projected net imports, with an explicit net-import scenario; (3) keep
  `electricity_generation` responses until the frontend switches; (4) re-run this
  protocol on demand. The pre-registered `decomp_damped` candidate (closest to
  passing) is a first step.
* **Renewables share** is of primary energy; the share of electricity
  (`renewables_share_elec`) is closer to the power sector but is a ratio involving
  the target itself and would need careful lagging.
* **Variables considered** (bundled OWID file): generation by source (sums to the
  target → leakage unless lagged, and each source needs its own projection —
  rejected); demand and net imports (tested via decomposition); nuclear phase-out
  (policy known in advance, but the schedule changed in 2002, 2010, 2011 and 2022,
  so a leakage-free version needs vintage-correct policy data — not done); installed
  capacity and industrial production are not in the bundled data. Web search was
  not available in this session, so no external dataset was added and the
  OWID values could not be cross-checked against AGEB/Destatis.
* **Outliers** (2009 −44 TWh, 2010 +36, 2019 −32, 2020 −33, 2023 −63) are real events
  and were kept.

## 13. Reproduction

```bash
pip install -r requirements-dev.txt
python -m pytest                                  # 105 tests
python -m app.evaluation --json report.json       # ~12 s
pip install -r requirements-experiments.txt       # statsmodels for ARIMA
python -m experiments.run_all --fresh             # ~3 min
python -m experiments.run_intervals
python -m experiments.diagnostics
python -m experiments.report_tables               # tables of this report
```

## 14. Remaining limitations

* 25 forecast origins with overlapping errors: small effective sample, low power.
* Only one structural break is in the data; the regime after 2023 has three
  observations.
* Revised (not real-time) data vintages; 2025 values may be revised.
* Interval method chosen post hoc among four variants; coverage not guaranteed.
* Long-term (6–10 years) accuracy is not established; > 10 years is exploratory.
* The exogenous variables have no demonstrated predictive value for this target.

## 15. Future research (prioritised)

1. **Target migration to demand with an explicit trade component** (section 12);
   re-test `decomp_damped` with more data and vintage-correct policy information.
2. **Higher-frequency data** (monthly Ember/ENTSO-E/Destatis generation and demand)
   to multiply the number of observations for model validation (annual totals can
   still be forecast).
3. **Drivers with known or plannable future paths:** installed capacity by source,
   legislated nuclear/coal phase-out schedules (as of each origin), industrial
   production, electricity prices; each must be assessed for availability at
   forecast time and projection uncertainty.
4. **Real-time data vintages** for GDP to remove revision look-ahead.
5. **Scenario layer** for long horizons (official grid-development or ministry
   scenarios, clearly separated from validated forecasts).
6. **Interval research** on a longer or monthly sample: adaptive conformal
   inference and break-aware scaling, validated on an untouched holdout.
