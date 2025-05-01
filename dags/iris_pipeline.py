from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from scripts.data import load_data, prepare_data
from scripts.train import train_model
from scripts.test import test_model

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2023, 1, 1),
    "retries": 1,
}

with DAG(
    "iris_classification_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:

    load_task = PythonOperator(
        task_id="load_data",
        python_callable=load_data,
    )

    prepare_task = PythonOperator(
        task_id="prepare_data",
        python_callable=prepare_data,
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    test_task = PythonOperator(
        task_id="test_model",
        python_callable=test_model,
    )

    load_task >> prepare_task >> train_task >> test_task
