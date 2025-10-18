import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from datetime import timedelta

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
    df = pd.read_csv("../data/train_processed.csv", parse_dates=["Date"])
    return df

@st.cache_resource
def load_model():
    model = lgb.Booster(model_file="../data/lightgbm_model.txt")
    return model

df = load_data()
model = load_model()

# --------------------------
# Sidebar Inputs
# --------------------------
st.sidebar.header("⚙️ Forecast Settings")
store_id = st.sidebar.selectbox("Select Store ID", sorted(df["Store"].unique()))
forecast_days = st.sidebar.slider("Forecast Horizon (days)", 7, 90, 30)

# Filter data for selected store
store_df = df[df["Store"] == store_id].copy().sort_values("Date")

# --------------------------
# Show last few data points
# --------------------------
st.subheader(f"🧾 Historical Sales - Store {store_id}")
st.line_chart(store_df.set_index("Date")["Sales"])

# --------------------------
# Forecast Next N Days
# --------------------------
st.subheader(f"🔮 Forecasting next {forecast_days} days")

# Prepare input features for future dates
last_date = store_df["Date"].max()
future_dates = pd.date_range(last_date + timedelta(days=1), periods=forecast_days, freq="D")

future_df = pd.DataFrame({"Date": future_dates})
future_df["Store"] = store_id
future_df["DayOfWeek"] = future_df["Date"].dt.dayofweek + 1
future_df["Month"] = future_df["Date"].dt.month
future_df["Year"] = future_df["Date"].dt.year
future_df["Day"] = future_df["Date"].dt.day
future_df["WeekOfYear"] = future_df["Date"].dt.isocalendar().week
future_df["DayOfYear"] = future_df["Date"].dt.dayofyear
future_df["IsWeekend"] = future_df["DayOfWeek"].isin([6,7]).astype(int)

# Fill missing categorical and lag features with latest known values
latest_vals = store_df.iloc[-1]
cols_to_copy = [
    'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
    'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
    'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval',
    'Sales_lag_7', 'Sales_lag_30', 'Sales_roll_mean_7', 'Sales_roll_mean_30'
]
for col in cols_to_copy:
    future_df[col] = latest_vals.get(col, 0)

# Predict
preds = model.predict(future_df[model.feature_name()])
future_df["PredictedSales"] = preds

# Combine actual + predicted
combined_df = pd.concat([
    store_df[["Date", "Sales"]].rename(columns={"Sales": "Actual"}),
    future_df[["Date", "PredictedSales"]].rename(columns={"PredictedSales": "Predicted"})
])

# --------------------------
# Plot Forecast
# --------------------------
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(combined_df["Date"], combined_df["Actual"], label="Actual", color="blue")
ax.plot(combined_df["Date"], combined_df["Predicted"], label="Predicted", color="orange")
ax.set_title(f"Store {store_id} - Actual & Forecasted Sales")
ax.legend()
st.pyplot(fig)

# --------------------------
# Feature Importance
# --------------------------
st.subheader("🎯 Top Feature Importances")
importance_df = (
    pd.DataFrame({
        "Feature": model.feature_name(),
        "Importance": model.feature_importance(importance_type="gain")
    })
    .sort_values(by="Importance", ascending=False)
    .head(15)
)
st.bar_chart(importance_df.set_index("Feature"))

# --------------------------
# Notes
# --------------------------
st.info(
    "This dashboard uses LightGBM trained on engineered features (lags, rolling means, promotions, "
    "and store metadata) for store-level demand forecasting. You can extend this by integrating Prophet "
    "for trend explainability or deploying with AWS Lambda / S3 for production."
)
