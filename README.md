# Air Quality Forecasting — India (PM2.5)

Time series forecasting project predicting daily PM2.5 levels across five major Indian cities, built to demonstrate a rigorous forecasting methodology rather than a single model — validated not just via cross-validation, but against genuinely unseen real-world data collected after model training.

## Cities & Stations

Stations were selected by checking non-null PM2.5 coverage within the 2019–2023 window specifically, not just over full station history — an initial pass picked stations with the best full-history coverage, which produced misleadingly poor results for the actual modeling window (see EDA notebook for details).

| City      | Station | Location                 | Missing % (2019–2023) |
|-----------|---------|---------------------------|------------------------|
| Delhi     | DL030   | Dwarka Sector 8           | 0.1%                   |
| Mumbai    | MH015   | Sion                      | 2.4%                   |
| Chennai   | TN003   | Velachery Res. Area       | 1.5%                   |
| Hyderabad | TG004   | Bollaram Industrial Area  | 1.3%                   |
| Bengaluru | KA008   | Hebbal                    | 2.5%                   |

## Dataset

[Time Series Air Quality Data of India (2010–2023)](https://www.kaggle.com/datasets/abhisheksjha/time-series-air-quality-data-of-india-2010-2023) — CPCB-sourced hourly station data, resampled to daily averages.

**Window**: 2019–2023 (not full 2010–2023 history). EDA showed CPCB monitoring quality improved substantially post-2019 — pre-2019 data had 22–47% missingness across cities, while 2019–2023 sits under 2.5% for all five cities. This trades fewer seasonal cycles (4 years vs 13) for far cleaner data, judged the better tradeoff for this project.

> Raw data is not committed to this repo due to size. See setup instructions below to download it yourself.

## Methodology

1. **EDA** — seasonality, missingness, distribution analysis per city
2. **Walk-forward validation harness** — built before any model, plus a naive baseline
3. **Classical models** — SARIMA (Delhi proof of concept) / Prophet (full walk-forward CV)
4. **Feature engineering** — lags, rolling stats, calendar effects, holiday flags, outlier flags (leakage-aware)
5. **ML model** — LightGBM, walk-forward CV
6. **Comparison & error analysis** across all approaches
7. **Real-world holdout validation** — LightGBM tested against live OpenAQ station data collected after the training period ended, confirming generalization beyond cross-validated folds

## Setup

```bash
pip install -r requirements.txt
```

Configure Kaggle credentials, then download the dataset:

```bash
kaggle datasets download -d abhisheksjha/time-series-air-quality-data-of-india-2010-2023 --unzip -p data/raw
```

## Project Structure

```
notebooks/   - sequential analysis notebooks (01 through 07)
models/      - saved trained model artifacts
reports/     - figures and writeups
```

## Key Methodological Decisions

- **Log-transform (`log1p`) applied before all modeling and all engineered features** — chosen after EDA found right-skewed distributions and multiplicative seasonality in Delhi. Rolling stats in particular are only meaningful computed post-log, since `log(mean of x)` ≠ `mean of log(x))` on raw, spike-heavy PM2.5 series.
- **Linear interpolation** used to fill missing values before splitting into folds, and again when incorporating the real-world holdout data.
- **Expanding-window walk-forward CV** (365-day minimum train, 30-day horizon, 30-day step, 39 folds/city) used as the evaluation harness for classical and ML models.
- **SARIMA run as a single 80/20 split on Delhi only**, not full walk-forward CV — `m=365` on daily data made full CV computationally impractical (~10–30 hrs estimated, did not complete within a Colab runtime session). Prophet is the primary classical benchmark.
- **Outlier days kept in the data, not removed** — instead flagged as a binary feature for the ML model to use as it sees fit.
- **Real-world holdout validation performed as a final capstone check** — the trained LightGBM model was never exposed to this data, and predictions were compared against real station readings collected well after the original training window ended.

## EDA Findings (Notebook 01)

- **Missingness**: scattered/random across all 5 cities post-2019, no structural sensor outages. Pre-2019 data had 22–47% missingness, confirming the window choice.
- **Seasonality**: strong yearly cycle confirmed across all cities — winter spikes (Oct–Feb), monsoon dips (Jun–Sep). Delhi most extreme (peaks 400–600 µg/m³), Bengaluru flattest (10–50 µg/m³ typical).
- **Decomposition**: multiplicative model fits better than additive for Delhi (seasonal swings scale with trend level). Other cities ambiguous. Decision: log-transform series across all cities.
- **Distribution**: all cities heavily right-skewed — further confirms log-transform is appropriate.
- **Outliers**: two flagged — Bengaluru Jan 26 2023 (556 µg/m³, plausibly Republic Day fireworks) and Chennai Jul 5 2019 (291 µg/m³, no clear cause identified). Both kept as-is; formally captured as a z-score-based outlier flag in feature engineering (notebook 04), which independently confirmed both days plus several more per city.
- **Cross-city correlation**: no negative correlations — all cities share the same monsoon-driven seasonal cycle. Mumbai–Hyderabad most correlated (0.71), Chennai least correlated with others (0.16–0.22) due to its unique northeast monsoon pattern.

## Baseline Results (Notebook 02)

Walk-forward validation: expanding window, 365-day minimum training, 30-day forecast horizon, 30-day step size → 39 folds per city.

Persistence forecast ("repeat last value") beats seasonal naive ("same day last year") for every city. These are the minimum bars real models must beat:

| City      | Persistence MAE | Persistence RMSE |
|-----------|----------------|------------------|
| Delhi     | 45.07          | 55.01            |
| Mumbai    | 17.56          | 21.02            |
| Chennai   | 12.81          | 15.86            |
| Hyderabad | 15.54          | 18.22            |
| Bengaluru | 11.52          | 15.62            |

## SARIMA / Prophet Results (Notebook 03)

**ADF test**: all 5 cities stationary on raw series → `d=0` for all cities.

**SARIMA**: `SARIMA(1,0,1)(1,1,1,365)`, Delhi-only, single 80/20 train/test split attempted — did not complete within a single Colab runtime session due to computational cost at `m=365`. Excluded from final comparison; Prophet serves as the primary classical benchmark.

**Prophet**: `yearly_seasonality=True`, weekly/daily seasonality off, `seasonality_mode='multiplicative'`, log-transformed input, holiday regressors (Diwali, Republic Day, Holi, New Year), full walk-forward CV using the same harness as notebook 02.

| City      | MAE   | RMSE  | Beats persistence? |
|-----------|-------|-------|---------------------|
| Delhi     | 37.38 | 46.51 | ✅ −17%              |
| Mumbai    | 17.54 | 21.29 | ✅ barely             |
| Chennai   | 15.40 | 20.19 | ❌ worse              |
| Hyderabad | 12.71 | 15.11 | ✅ −18%              |
| Bengaluru | 11.08 | 14.80 | ✅ −4%               |

## Feature Engineering (Notebook 04)

Built entirely on `pm25_log` for consistency with the log-transform decision above. Leakage-checked throughout — all rolling stats computed on a series shifted by one day before rolling, so no feature ever has access to the current day's own value.

- **Lags**: 1, 2, 3, 7, 14, 30, 365 days
- **Rolling stats**: mean & std (7, 14, 30-day windows), min & max (7, 30-day windows)
- **Calendar**: day of week, month, day of year, is-weekend, plus sin/cos cyclical encodings of day-of-year and day-of-week (so year-end/year-start and Sunday/Monday are recognized as adjacent rather than numerically far apart)
- **Holiday flags**: Diwali, Republic Day, Holi, New Year, ±1 day buffer around each date
- **Outlier flags**: z-score (|z| > 3) computed per city on `pm25_log`, confirmed to catch both known EDA outliers (Bengaluru Jan 26 2023, Chennai Jul 5 2019) plus additional flagged days.

> Known limitation: outlier z-scores are computed on each city's full series mean/std rather than per-fold expanding-window statistics — a minor leakage accepted for a binary flag feature, consistent with the SARIMA single-split simplification above.

## LightGBM Results (Notebook 05)

Walk-forward CV, same harness as prior models. Feature importance analysis confirmed `lag_1` as the dominant predictor by a wide margin, followed by seasonal (`doy_cos`) and trend (`roll_mean_7/14/30`) features — no evidence the accepted outlier-flag leakage drove results.

| City      | MAE   | RMSE  | vs Persistence | vs Prophet |
|-----------|-------|-------|-----------------|-------------|
| Delhi     | 25.74 | 34.27 | −43%            | −31%        |
| Mumbai    | 10.13 | 13.14 | −42%            | −42%        |
| Chennai   | 6.60  | 9.33  | −48%            | −57%        |
| Hyderabad | 7.29  | 8.99  | −53%            | −43%        |
| Bengaluru | 7.45  | 11.37 | −35%            | −33%        |

LightGBM outperforms both baselines in every city. Per-fold error analysis (notebook 06) showed elevated MAE during Oct–Feb in Delhi and Hyderabad, tracking the same winter pollution season identified as most extreme and erratic in EDA — the model captures the general seasonal shift but individual peak-day magnitudes remain harder to predict precisely than the calmer rest of the year.

## Real-World Holdout Validation (Notebook 07)

The trained LightGBM models (retrained on the full 2019–Mar 2023 dataset) were evaluated against live station data pulled from the OpenAQ API, covering February 2025 through June/July 2026 — genuinely unseen data collected well after the training period ended, sourced independently of the original Kaggle dataset.

**Station matching**: 4 of 5 cities matched to their exact original CPCB station. Delhi could not be matched exactly — the nearest available OpenAQ station (NSIT Dwarka) was used instead of the original Dwarka Sector 8 station, introducing some location-based uncertainty into Delhi's holdout numbers.

**Data quality**: Mumbai's feed contained a cluster of physically implausible readings (609–985 µg/m³ during monsoon season, Sept 25 – Oct 9 2025) consistent with a sensor malfunction rather than a real pollution event — no comparably extreme value was ever observed across 4 years of training data, even in Delhi's worst winter smog. This window was excluded from scoring.

| City      | Original CV MAE | Holdout MAE | Holdout Window |
|-----------|------------------|-------------|-----------------|
| Delhi     | 25.74            | 22.41       | Feb 2025 – Jun 2026 (station mismatch) |
| Mumbai    | 10.13            | 23.44       | Feb 2025 – Jun 2026 (malfunction window removed) |
| Chennai   | 6.60             | 5.25        | Feb 2025 – Jun 2026 |
| Hyderabad | 7.29             | 6.95        | Feb 2025 – Mar 2026 |
| Bengaluru | 7.45             | 7.60        | Feb 2025 – Jun 2026 |

Chennai, Hyderabad, and Bengaluru matched or beat their cross-validated performance on this real, unseen data — strong evidence the model generalizes beyond its validation folds. Delhi held up well despite the station mismatch. Mumbai remains an open limitation: even after removing the confirmed malfunction window, holdout MAE is ~2.3x its CV value, driven by a small number of isolated large misses (including a plausible New Year's Eve pollution spike) the model could not anticipate — reported honestly rather than further adjusted away.

## Status

✅ Notebook 01 — EDA complete

✅ Notebook 02 — Walk-forward validation harness + naive baselines complete

🚧 Notebook 03 — Prophet complete, SARIMA (Delhi) attempted but did not complete

✅ Notebook 04 — Feature engineering complete

✅ Notebook 05 — LightGBM complete

✅ Notebook 06 — Model comparison complete

✅ Notebook 07 — Real-world holdout validation complete
