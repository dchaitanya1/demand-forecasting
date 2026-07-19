#  Demand Forecasting for Retail Stores (Rossmann Dataset)

**End-to-end demand forecasting project** built to demonstrate data-engineering and applied-ML principles.
The project predicts future store-level sales using **LightGBM**, explores **Prophet** as a per-store
baseline, and serves predictions through an interactive **Streamlit dashboard**.

---

##  Project Overview

Retailers need accurate demand forecasts to plan inventory, staffing, and promotions.
This project builds a forecasting pipeline that ingests raw sales data, performs feature
engineering, trains a model, and serves interactive forecasts through a web interface.

---

##  Methodology

1. **Data Ingestion**
   - Load `train.csv` and `store.csv`, merge, and clean (handle missing competition/promo fields).

2. **Exploratory Data Analysis (EDA)**
   - Understand seasonality, weekly patterns, and promo impact.
   - Visualize sales trends by day, store type, and holidays.

3. **Feature Engineering**
   - Temporal features: `Year`, `Month`, `WeekOfYear`, `DayOfWeek`.
   - Lag & rolling statistics: `Sales_lag_7`, `Sales_roll_mean_7`, etc.
   - Encoded categorical variables: `StoreType`, `Assortment`.

4. **Modeling**
   - **Prophet** — tried as an interpretable, single-store trend/seasonality baseline.
   - **LightGBM** — the deployed model: one global model across all stores, using the engineered features.

5. **Evaluation**
   - Metrics: **MAE**, **RMSE**.

6. **Serving**
   - Streamlit app shows historical sales and a recursive multi-step forecast per store.
   - User selects a store ID and forecast horizon (7–90 days).

## Key Results

| Model | Scope of evaluation | MAE | RMSE |
|-------|---------------------|-----|------|
| Prophet | Single store (per-store baseline) | 636 | 734 |
| LightGBM | All stores (global model) | 586 | 905 |

> **Note:** the two rows are measured on *different scopes* — Prophet on one store, LightGBM
> across all stores — so the numbers are **not directly comparable**. LightGBM is the model used
> in the app because a single global model serves every store and leverages the engineered features.
> (Numbers above are from the full Rossmann dataset; run the scripts on your data to reproduce.)

---

## How to Run

**1. Get the data.** Either use the real Rossmann data (place `train.csv` and `store.csv` in `data/`
— see [`data/README.md`](data/README.md)), or generate small sample data for a quick test:

```bash
python scripts/make_sample_data.py
```

**2. Build features and train the model** (from the project root):

```bash
python scripts/data_ingestion.py     # -> data/train_processed.csv
python scripts/train_model.py        # -> data/lightgbm_model.txt (+ prints MAE/RMSE)
```

**3. Launch the dashboard:**

```bash
cd app
streamlit run streamlit_app.py
```

A single forecast can also be printed from the command line:

```bash
python scripts/forecast_service.py --store 1 --horizon 30
```

---

##  Streamlit Dashboard

**Features**
- Select any store from the sidebar
- Choose the forecast horizon (7–90 days)
- View combined *historical + predicted* trends (recursive forecast)
- Explore top feature importances

##  Dashboard Preview

### 🔹 Overall View
![Dashboard Overview](images/screenshot2.png)

### 🔹 Forecasting
![Forecasting](images/screenshot1.png)

---

## Project Structure

```
scripts/
  make_sample_data.py    # optional: generate Rossmann-format sample data
  data_ingestion.py      # merge + clean + feature engineering -> train_processed.csv
  train_model.py         # train LightGBM, save data/lightgbm_model.txt
  forecast_service.py    # recursive multi-step forecast (used by the app + CLI)
airflow/
  dag_demand_pipeline.py # Airflow DAG: data_ingestion -> train_model
app/
  streamlit_app.py       # interactive dashboard
notebooks/               # EDA, feature engineering, and modelling notebooks
```
