import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Air Quality Forecasting — India", layout="wide")

CITIES = ['Delhi', 'Mumbai', 'Chennai', 'Hyderabad', 'Bengaluru']

st.title("🌫️ Air Quality Forecasting — India")
st.markdown(
    "LightGBM model predicting daily PM2.5 across 5 Indian cities, "
    "validated against real-world data collected after training. "
    "[View the full project on GitHub](https://github.com/shaheeeeeeeeem/air-quality-forecasting)"
)

city = st.selectbox("Select a city", CITIES)

# --- Load predictions for selected city ---
df = pd.read_csv(f"data/predictions/{city}_predictions.csv", parse_dates=['date'])

# --- Metrics ---
mae = mean_absolute_error(df['actual'], df['predicted'])
rmse = np.sqrt(mean_squared_error(df['actual'], df['predicted']))

col1, col2, col3 = st.columns(3)
col1.metric("MAE (µg/m³)", f"{mae:.2f}")
col2.metric("RMSE (µg/m³)", f"{rmse:.2f}")
col3.metric("Days evaluated", f"{len(df)}")

st.caption(
    f"Holdout window: {df['date'].min().date()} to {df['date'].max().date()} — "
    "real station readings from OpenAQ, collected after model training ended."
)

# --- Actual vs Predicted chart ---
st.subheader(f"{city}: Actual vs Predicted PM2.5")
chart_df = df.set_index('date')[['actual', 'predicted']]
st.line_chart(chart_df)

# --- Feature importance ---
st.subheader("What drives the model's predictions?")
feat_imp = pd.read_csv("data/feature_importance.csv", index_col=0)
feat_imp.columns = ['importance']
st.bar_chart(feat_imp.head(10))

st.caption(
    "Feature importance shown is from the Delhi model. `lag_1` (yesterday's PM2.5) "
    "dominates, followed by seasonal and rolling-average features."
)

# --- Project context ---
with st.expander("About this project"):
    st.markdown(
        """
        This model was trained on 4 years of CPCB station data (2019–2023) using
        LightGBM with engineered lag, rolling-average, calendar, and holiday features.
        It was benchmarked against a persistence baseline and Facebook Prophet, then
        validated against real-world data from OpenAQ collected well after the
        training period ended — not just cross-validation folds.

        **Known limitations**, documented transparently in the project README:
        - Delhi's holdout station is a nearby but non-identical match to the original training station
        - Mumbai's holdout data required removing a ~2-week window of sensor-malfunction readings
        - SARIMA was attempted as a classical benchmark but did not complete in time; Prophet
          serves as the primary classical comparison instead
        """
    )