"""
Generate small SAMPLE data in the Rossmann format (for quick testing / demo).

This writes data/train.csv and data/store.csv with the same columns as the real
Kaggle files, so the pipeline runs end-to-end without the Kaggle download. For
the numbers you report, use the real Rossmann data from Kaggle instead - it has
the identical schema, so nothing else changes.

Usage (from the project root):
    python scripts/make_sample_data.py --stores 30
"""

import argparse
import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
STORE_TYPES = ["a", "b", "c", "d"]
ASSORTMENTS = ["a", "b", "c"]
WEEKDAY = np.array([1.20, 1.02, 0.98, 0.97, 1.00, 0.90, 0.15])  # Mon..Sun


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stores", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    dates = pd.date_range("2013-01-01", "2015-07-31", freq="D")
    dow = dates.weekday.to_numpy()
    doy = dates.dayofyear.to_numpy()
    trend = 1.0 + 0.10 * (np.arange(len(dates)) / len(dates))
    yearly = 1.0 + 0.18 * np.sin(2 * np.pi * (doy - 80) / 365.25)

    store_rows, sales_rows = [], []
    for sid in range(1, args.stores + 1):
        stype = rng.choice(STORE_TYPES, p=[0.55, 0.05, 0.25, 0.15])
        assort = rng.choice(ASSORTMENTS, p=[0.50, 0.10, 0.40])
        comp_dist = float(np.clip(round(rng.lognormal(7.8, 1.0)), 20, 75000))
        store_rows.append({
            "Store": sid, "StoreType": stype, "Assortment": assort,
            "CompetitionDistance": comp_dist,
            "CompetitionOpenSinceMonth": rng.integers(1, 13),
            "CompetitionOpenSinceYear": rng.integers(2000, 2015),
            "Promo2": 0, "Promo2SinceWeek": np.nan, "Promo2SinceYear": np.nan,
            "PromoInterval": np.nan,
        })

        base = rng.lognormal(8.4, 0.3) * (2.4 if stype == "b" else 1.0)
        promo = (rng.random(len(dates)) < 0.4).astype(int)
        promo[dow >= 5] = 0
        open_flag = np.ones(len(dates), dtype=int)
        if stype != "b":
            open_flag[dow == 6] = 0
        school = (rng.random(len(dates)) < 0.18).astype(int)
        noise = rng.normal(1.0, 0.10, len(dates)).clip(0.5, 1.6)
        sales = base * trend * yearly * WEEKDAY[dow] * (1 + 0.22 * promo) * noise
        sales = np.where(open_flag == 1, np.round(sales), 0).astype(int)
        customers = np.where(open_flag == 1, np.round(sales / rng.normal(10, 1.5, len(dates)).clip(6, 16)), 0).astype(int)

        for i, d in enumerate(dates):
            sales_rows.append({
                "Store": sid, "DayOfWeek": int(dow[i]) + 1, "Date": d,
                "Sales": int(sales[i]), "Customers": int(customers[i]),
                "Open": int(open_flag[i]), "Promo": int(promo[i]),
                "StateHoliday": "0", "SchoolHoliday": int(school[i]),
            })

    os.makedirs(DATA_DIR, exist_ok=True)
    pd.DataFrame(sales_rows).to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    pd.DataFrame(store_rows).to_csv(os.path.join(DATA_DIR, "store.csv"), index=False)
    print(f"Wrote sample train.csv ({len(sales_rows)} rows) and store.csv ({args.stores} stores)")


if __name__ == "__main__":
    main()
