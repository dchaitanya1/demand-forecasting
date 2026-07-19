"""
Data ingestion and feature engineering.

Loads the raw Rossmann files (train.csv, store.csv), merges them, cleans missing
values, builds calendar / lag / rolling features, encodes categoricals, and
writes data/train_processed.csv. This is the same pipeline the notebooks perform,
packaged as a runnable script.

Usage (from the project root):
    python scripts/data_ingestion.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CATEGORICAL = ["StoreType", "Assortment", "StateHoliday", "PromoInterval"]


def load_raw():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                        parse_dates=["Date"], dtype={"StateHoliday": "str"}, low_memory=False)
    store = pd.read_csv(os.path.join(DATA_DIR, "store.csv"))
    print(f"Loaded train {train.shape}, store {store.shape}")
    return train, store


def add_features(df):
    # --- impute missing store metadata ---
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())
    for col in ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
                "Promo2SinceWeek", "Promo2SinceYear"]:
        df[col] = df[col].fillna(0)
    df["PromoInterval"] = df["PromoInterval"].fillna("None")
    df["StateHoliday"] = df["StateHoliday"].fillna("0").replace({"nan": "0"})

    # --- calendar features ---
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["IsWeekend"] = df["Date"].dt.weekday.isin([5, 6]).astype(int)

    # --- encode categoricals ---
    le = LabelEncoder()
    for col in CATEGORICAL:
        df[col] = le.fit_transform(df[col].astype(str))

    # --- lag / rolling features (computed within each store; no cross-store leak) ---
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    grp = df.groupby("Store")["Sales"]
    df["Sales_lag_7"] = grp.shift(7)
    df["Sales_lag_30"] = grp.shift(30)
    df["Sales_roll_mean_7"] = grp.transform(lambda s: s.shift(1).rolling(7).mean())
    df["Sales_roll_mean_30"] = grp.transform(lambda s: s.shift(1).rolling(30).mean())

    lag_cols = ["Sales_lag_7", "Sales_lag_30", "Sales_roll_mean_7", "Sales_roll_mean_30"]
    df[lag_cols] = df[lag_cols].fillna(0)
    return df


def main():
    train, store = load_raw()
    df = pd.merge(train, store, on="Store", how="left")
    df = add_features(df)
    out = os.path.join(DATA_DIR, "train_processed.csv")
    df.to_csv(out, index=False)
    print(f"Feature engineering complete -> {out}  ({df.shape[0]} rows, {df.shape[1]} cols)")


if __name__ == "__main__":
    main()
