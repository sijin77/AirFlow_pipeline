# wine_quality_dag.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from typing import Dict, Any
import sys
import os

import joblib

# Добавляем пути для импорта
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.clearml_init import init_clearml_from_env
from scripts.data_proccessor.wine_data_processor import WineQualityDataProcessor
from models.config import config

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2023, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def verify_clearml_connection():
    try:
        task = init_clearml_from_env(project_name="WineQuality")
        task.close()
        print("ClearML подключен успешно!")
        return True
    except Exception as e:
        raise RuntimeError(f"Ошибка подключения к ClearML: {str(e)}")


def train_model(model_class, model_config_key: str) -> Dict[str, Any]:
    """Обучает модель и возвращает её метрики"""

    try:
        # Инициализируем ClearML задачу
        task = init_clearml_from_env(model_class)
        task.set_parameter("model_type", model_config_key)
        task.set_parameter("dataset", "wine_quality")

        # Подготовка данных
        data_processor = WineQualityDataProcessor(
            test_size=config["data"]["test_size"], random_state=config["random_state"]
        )

        # Динамический импорт класса модели
        module = __import__(f"models.{model_config_key}_model", fromlist=[model_class])
        model_class = getattr(module, model_class)

        # Создаем и обучаем модель
        model = model_class(
            data_processor=data_processor,
            **config[model_config_key],
            enable_logging=True,
        )
        model.train()
        metrics = model.evaluate()

        # Регистрируем модель
        model_id = model.register_best_model()
        task.close()
        return {
            "model_id": model_id,
            "metrics": metrics,
            "model_name": model_config_key,
        }

    except Exception as e:
        task.get_logger().report_text(f"Ошибка обучения: {str(e)}")
        raise


def deploy_best_model(**context) -> str:
    """Деплоит лучшую модель через ClearML и сохраняет её в S3 через DVC"""
    from clearml import Model
    import joblib
    import os
    import subprocess

    task = init_clearml_from_env()
    task.set_parameter("model_type", "serving")
    task.set_parameter("dataset", "wine_quality")

    try:
        # Получаем результаты всех задач
        models_data = context["ti"].xcom_pull(
            task_ids=["train_lr", "train_rf", "train_svm", "train_xgb"]
        )

        if not models_data or not all(models_data):
            raise ValueError("Нет данных о моделях для деплоя")

        # Выбираем лучшую модель по MSE
        best_model = min(
            models_data, key=lambda x: x["metrics"].get("mse", float("inf"))
        )

        # Получаем модель из ClearML
        model = Model(model_id=best_model["model_id"])
        model_path = model.get_local_copy()

        # Подготовка пути для сохранения
        dvc_root = "/opt/airflow/dvc"
        model_dir = os.path.join(dvc_root, "models")
        os.makedirs(model_dir, exist_ok=True)

        # Формируем имя файла модели
        model_filename = f"wine_quality_model.pkl"
        local_model_path = os.path.join(model_dir, model_filename)

        # Копируем модель в DVC директорию
        import shutil

        shutil.copy(model_path, local_model_path)

        # Добавляем модель в DVC и пушим в S3
        def run_dvc_command(cmd, cwd):
            result = subprocess.run(
                cmd.split(),
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"DVC command failed: {cmd}\nError: {result.stderr}")

        original_dir = os.getcwd()
        os.chdir(dvc_root)

        try:
            # Добавляем модель в DVC
            run_dvc_command(f"dvc add {local_model_path}", dvc_root)

            # Пушим в S3
            run_dvc_command("dvc push", dvc_root)

            # Коммитим изменения .dvc файла (если используется git)
            if os.path.exists(".git"):
                run_dvc_command(f"git add {local_model_path}.dvc", dvc_root)
                run_dvc_command(
                    f"git commit -m 'Add best model: {best_model['model_name']}'",
                    dvc_root,
                )
                run_dvc_command("git push", dvc_root)
        finally:
            os.chdir(original_dir)

        # Деплой модели ClearML
        serving_instance = model.deploy(
            engine="python",
            serving_service_name=f"wine-quality-{best_model['model_name']}",
            endpoint="/predict",
        )

        task.get_logger().report_text(
            f"Деплой успешен! Модель: {best_model['model_name']}\n"
            f"Service ID: {serving_instance.id}\n"
            f"Endpoint: {serving_instance.endpoint}\n"
            f"Метрики: {best_model['metrics']}\n"
            f"Модель сохранена в S3: {local_model_path}"
        )

        return serving_instance.id

    except Exception as e:
        task.get_logger().report_text(f"Ошибка деплоя: {str(e)}")
        raise


with DAG(
    "wine_quality_regression_pipeline",
    default_args=default_args,
    description="Обучение и деплой моделей для предсказания качества вина",
    schedule_interval=None,
    catchup=False,
) as dag:

    verify_connection = PythonOperator(
        task_id="verify_clearml_connection", python_callable=verify_clearml_connection
    )

    train_lr = PythonOperator(
        task_id="train_lr",
        python_callable=train_model,
        op_kwargs={
            "model_class": "LinearRegressionModel",
            "model_config_key": "linear_regression",
        },
    )

    train_rf = PythonOperator(
        task_id="train_rf",
        python_callable=train_model,
        op_kwargs={
            "model_class": "RandomForestModel",
            "model_config_key": "random_forest",
        },
    )

    train_svm = PythonOperator(
        task_id="train_svm",
        python_callable=train_model,
        op_kwargs={
            "model_class": "SVMModel",
            "model_config_key": "svm",
        },
    )

    train_xgb = PythonOperator(
        task_id="train_xgb",
        python_callable=train_model,
        op_kwargs={
            "model_class": "XGBoostModel",
            "model_config_key": "xgboost",
        },
    )

    deploy = PythonOperator(
        task_id="deploy_best_model",
        python_callable=deploy_best_model,
        provide_context=True,
    )

    verify_connection >> [train_lr, train_rf, train_svm, train_xgb] >> deploy
