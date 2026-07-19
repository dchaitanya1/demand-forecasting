import os
import sys

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Reuse the same recursive-forecast logic the CLI uses (single source of truth).
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "..", "scripts"))
from forecast_service import recursive_forecast  # noqa: E402

DATA_DIR = os.path.join(BASE, "..", "data")

# --------------------------
# Page Configuration
# --------------------------
st.set_page_config(page_title="🧭 Demand Forecasting Dashboard", layout="wide")
st.title("📊 Demand Forecasting - Rossmann Stores")
st.markdown("Predict future sales using **LightGBM** and explore store-level trends.")


# --------------------------
# Load Data and Model
# --------------------------
@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(DATA_DIR, "train_processed.csv"), parse_dates=["Date"])


@st.cache_resource
def load_model():
    return lgb.Booster(model_file=os.path.join(DATA_DIR, "lightgbm_model.txt"))


try:
    df = load_data()
    model = load_model()
except FileNotFoundError:
    st.error(
        "Data or model not found. From the project root, run:\n\n"
        "```\n"
        "python scripts/make_sample_data.py   # (or use the real Kaggle data)\n"
        "python scripts/data_ingestion.py\n"
        "python scripts/train_model.py\n"
        "```"
    )
    st.stop()

# --------------------------
# Sidebar Inputs
# --------------------------
st.sidebar.header("⚙️ Forecast Settings")
store_id = st.sidebar.selectbox("Select Store ID", sorted(df["Store"].unique()))
forecast_days = st.sidebar.slider("Forecast Horizon (days)", 7, 90, 30)

store_df = df[df["Store"] == store_id].copy().sort_values("Date")

# --------------------------
# Historical Sales
# --------------------------
st.subheader(f"🧾 Historical Sales - Store {store_id}")
st.line_chart(store_df.set_index("Date")["Sales"])

# --------------------------
# Forecast Next N Days (recursive: lag features update each day)
# --------------------------
st.subheader(f"🔮 Forecasting next {forecast_days} days")
future_df = recursive_forecast(df, model, store_id, forecast_days)

combined_df = pd.concat([
    store_df[["Date", "Sales"]].rename(columns={"Sales": "Actual"}),
    future_df.rename(columns={"PredictedSales": "Predicted"}),
])

fig, ax = plt.subplots(figsize=(12, 5))
recent = store_df.tail(120)
ax.plot(recent["Date"], recent["Sales"], label="Actual", color="blue")
ax.plot(future_df["Date"], future_df["PredictedSales"], label="Predicted", color="orange")
ax.set_title(f"Store {store_id} - Actual & Forecasted Sales")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

with st.expander("Forecast values (per day)"):
    st.dataframe(
        future_df.assign(PredictedSales=future_df["PredictedSales"].round(0).astype(int)),
        use_container_width=True,
    )

# --------------------------
# Feature Importance
# --------------------------
st.subheader("🎯 Top Feature Importances")
importance_df = (
    pd.DataFrame({
        "Feature": model.feature_name(),
        "Importance": model.feature_importance(importance_type="gain"),
    })
    .sort_values(by="Importance", ascending=False)
    .head(15)
)
st.bar_chart(importance_df.set_index("Feature"))

# --------------------------
# Notes
# --------------------------
st.info(
    "This dashboard uses a LightGBM model trained on engineered features (lags, rolling "
    "means, promotions, and store metadata). The forecast is recursive: each predicted "
    "day feeds into the next day's lag features, so weekly patterns are preserved."
)
