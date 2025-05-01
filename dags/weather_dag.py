from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from scripts.weather import save_weather_to_csv
from dotenv import load_dotenv
import os

load_dotenv("/app/.env")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2023, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

dag = DAG(
    "weather_data_collection",
    default_args=default_args,
    description="DAG for collecting weather data every minute",
    schedule_interval="* * * * *",  # Каждую минуту
    catchup=False,
)


def collect_weather_data():
    save_weather_to_csv(api_key=OPENWEATHER_KEY)


collect_weather_task = PythonOperator(
    task_id="collect_weather_data",
    python_callable=collect_weather_data,
    dag=dag,
)

collect_weather_task
