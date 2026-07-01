# Air Quality Forecasting — India (PM2.5)

Time series forecasting project predicting daily PM2.5 levels across five major Indian cities, built to demonstrate a rigorous forecasting methodology rather than a single model.

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
3. **Classical model** — SARIMA / Prophet
4. **Feature engineering** — lags, rolling stats, calendar effects (leakage-aware)
5. **ML model** — LightGBM
6. **Comparison & error analysis** across all three approaches

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
notebooks/   - sequential analysis notebooks (01 through 06)
src/         - reusable pipeline code (data loading, validation, features, evaluation)
models/      - saved trained model artifacts
reports/     - figures and writeups
```

## EDA Findings (Notebook 01)

- **Missingness**: scattered/random across all 5 cities post-2019, no structural sensor outages. Pre-2019 data had 22–47% missingness, confirming the window choice.
- **Seasonality**: strong yearly cycle confirmed across all cities — winter spikes (Oct–Feb), monsoon dips (Jun–Sep). Delhi most extreme (peaks 400–600 µg/m³), Bengaluru flattest (10–50 µg/m³ typical).
- **Decomposition**: multiplicative model fits better than additive for Delhi (seasonal swings scale with trend level). Other cities ambiguous. Decision: log-transform series before SARIMA across all cities.
- **Distribution**: all cities heavily right-skewed — further confirms log-transform is appropriate.
- **Outliers**: two flagged — Bengaluru Jan 26 2023 (556 µg/m³, plausibly Republic Day fireworks) and Chennai Jul 5 2019 (291 µg/m³, no clear cause identified). Both kept as-is, will be revisited during feature engineering.
- **Cross-city correlation**: no negative correlations — all cities share the same monsoon-driven seasonal cycle. Mumbai–Hyderabad most correlated (0.71), Chennai least correlated with others (0.16–0.22) due to its unique northeast monsoon pattern.

## Status

✅ Notebook 01 — EDA complete

🚧 In progress — Notebook 02 (walk-forward validation harness + naive baseline)
