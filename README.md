# Air Quality Forecasting — India (PM2.5)

Time series forecasting project predicting daily PM2.5 levels across five major Indian cities, built to demonstrate a rigorous forecasting methodology rather than a single model.

## Cities & Stations

| City      | Station |
|-----------|---------|
| Delhi     | DL005   |
| Mumbai    | MH004   |
| Chennai   | TN002   |
| Hyderabad | TG001   |
| Bengaluru | KA003   |

## Dataset

[Time Series Air Quality Data of India (2010–2023)](https://www.kaggle.com/datasets/abhisheksjha/time-series-air-quality-data-of-india-2010-2023) — CPCB-sourced hourly station data, resampled to daily averages for this project.

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

## Status

🚧 In progress — currently on EDA (notebook 01).
