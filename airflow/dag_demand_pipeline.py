"""
Airflow DAG for the demand-forecasting pipeline.

Runs the two offline stages in order: data ingestion + feature engineering,
then model training. Copy this file into your Airflow `dags/` folder and set
PROJECT_DIR to the project root.

    data_ingestion  ->  train_model
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/demand-forecasting"

default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="demand_forecasting_pipeline",
    description="Ingest + feature engineer, then train the LightGBM model",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["forecasting", "lightgbm"],
) as dag:

    data_ingestion = BashOperator(
        task_id="data_ingestion",
        bash_command=f"cd {PROJECT_DIR} && python scripts/data_ingestion.py",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"cd {PROJECT_DIR} && python scripts/train_model.py",
    )

    data_ingestion >> train_model
