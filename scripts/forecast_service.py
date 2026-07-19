"""
Forecasting service.

Provides `recursive_forecast()`, which produces a genuine multi-step forecast:
each predicted day is fed back so the lag / rolling features update as we move
forward (instead of being frozen at the last known values). The Streamlit app
imports this same function, so the app and the CLI stay consistent.

Usage (from the project root):
    python scripts/forecast_service.py --store 1 --horizon 30
"""

import argparse
import os
from datetime import timedelta

import lightgbm as lgb
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Store-level features carried forward from the last known day.
STATIC_COLS = [
    "Store", "Promo", "StateHoliday", "SchoolHoliday", "StoreType", "Assortment",
    "CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
    "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval",
]


def recursive_forecast(df, model, store_id, horizon):
    """Return a DataFrame [Date, PredictedSales] for `horizon` future days."""
    store_df = df[df["Store"] == store_id].copy().sort_values("Date")
    sales_by_date = {pd.Timestamp(d): float(s)
                     for d, s in zip(store_df["Date"], store_df["Sales"])}
    last_row = store_df.iloc[-1]
    last_date = store_df["Date"].max()
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=horizon, freq="D")

    def sales_on(day):
        return sales_by_date.get(pd.Timestamp(day), 0.0)

    records = []
    for d in future_dates:
        row = {c: last_row[c] for c in STATIC_COLS}
        row["DayOfWeek"] = d.dayofweek + 1
        row["Year"], row["Month"], row["Day"] = d.year, d.month, d.day
        row["WeekOfYear"] = int(d.isocalendar()[1])
        row["DayOfYear"] = d.dayofyear
        row["IsWeekend"] = 1 if d.weekday() in (5, 6) else 0
        # Lags / rolling means from the running (actual + predicted) series.
        row["Sales_lag_7"] = sales_on(d - timedelta(days=7))
        row["Sales_lag_30"] = sales_on(d - timedelta(days=30))
        row["Sales_roll_mean_7"] = np.mean([sales_on(d - timedelta(days=k)) for k in range(1, 8)])
        row["Sales_roll_mean_30"] = np.mean([sales_on(d - timedelta(days=k)) for k in range(1, 31)])

        X = pd.DataFrame([row])[model.feature_name()]
        yhat = max(float(model.predict(X)[0]), 0.0)
        records.append({"Date": d, "PredictedSales": yhat})
        sales_by_date[pd.Timestamp(d)] = yhat  # feed prediction back

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Forecast future sales for a store.")
    parser.add_argument("--store", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=30)
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(DATA_DIR, "train_processed.csv"), parse_dates=["Date"])
    model = lgb.Booster(model_file=os.path.join(DATA_DIR, "lightgbm_model.txt"))

    fc = recursive_forecast(df, model, args.store, args.horizon)
    print(f"\nForecast for store {args.store} ({args.horizon} days):\n")
    print(fc.assign(PredictedSales=fc["PredictedSales"].round(0)).to_string(index=False))


if __name__ == "__main__":
    main()
