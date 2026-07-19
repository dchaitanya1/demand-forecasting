"""
Train the LightGBM demand-forecasting model.

Reads data/train_processed.csv, trains a global LightGBM model with a time-based
split, prints MAE / RMSE on the validation period, and saves the trained model to
data/lightgbm_model.txt (this is the file the Streamlit app loads).

Usage (from the project root):
    python scripts/train_model.py
"""

import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
VALIDATION_START = "2015-05-01"

FEATURES = [
    "Store", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday",
    "StoreType", "Assortment", "CompetitionDistance",
    "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
    "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval",
    "Year", "Month", "Day", "WeekOfYear", "DayOfYear", "IsWeekend",
    "Sales_lag_7", "Sales_lag_30", "Sales_roll_mean_7", "Sales_roll_mean_30",
]

PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 64,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "train_processed.csv"), parse_dates=["Date"])
    print(f"Loaded processed data {df.shape}")

    train_df = df[df["Date"] < VALIDATION_START]
    val_df = df[df["Date"] >= VALIDATION_START]
    print(f"Train rows: {len(train_df)}, Validation rows: {len(val_df)}")

    dtrain = lgb.Dataset(train_df[FEATURES], label=train_df["Sales"])
    dvalid = lgb.Dataset(val_df[FEATURES], label=val_df["Sales"], reference=dtrain)

    model = lgb.train(
        PARAMS, dtrain, num_boost_round=500, valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    preds = model.predict(val_df[FEATURES], num_iteration=model.best_iteration)
    mae = mean_absolute_error(val_df["Sales"], preds)
    rmse = np.sqrt(mean_squared_error(val_df["Sales"], preds))
    print(f"\nValidation MAE:  {mae:.2f}")
    print(f"Validation RMSE: {rmse:.2f}")

    out = os.path.join(DATA_DIR, "lightgbm_model.txt")
    model.save_model(out, num_iteration=model.best_iteration)
    print(f"Model saved -> {out}")


if __name__ == "__main__":
    main()
