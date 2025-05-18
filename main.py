# main.py

from scripts.clearml_init import init_clearml_from_env
from models.linear_regression_model import LinearRegressionModel
from typing import Dict, Any
from models.config import config
from scripts.data_proccessor.wine_data_processor import WineQualityDataProcessor


def train_model(model_class, model_config_key: str) -> Dict[str, Any]:
    """Обучает модель и возвращает её метрики"""

    try:
        # Инициализируем ClearML задачу
        task = init_clearml_from_env(project_name="WineQuality")
        task.set_parameter("model_type", model_config_key)

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
    """Деплоит лучшую модель через ClearML"""
    from clearml import Model

    task = init_clearml_from_env(project_name="WineQuality")
    task.set_parameter("model_type", "serving")

    try:
        # Получаем результаты всех задач
        models_data = context["ti"].xcom_pull(
            task_ids=["train_lr", "train_dt", "train_svm", "train_xgb"]
        )

        if not models_data or not all(models_data):
            raise ValueError("Нет данных о моделях для деплоя")

        # Выбираем лучшую модель по F1-score
        best_model = max(models_data, key=lambda x: x["metrics"].get("f1", 0))

        # Деплой модели
        model = Model(model_id=best_model["model_id"])
        serving_instance = model.deploy(
            engine="python",
            serving_service_name=f"digits-{best_model['model_name']}",
            endpoint="/predict",
        )

        task.get_logger().report_text(
            f"Деплой успешен! Модель: {best_model['model_name']}\n"
            f"Service ID: {serving_instance.id}\n"
            f"Endpoint: {serving_instance.endpoint}\n"
            f"Метрики: {best_model['metrics']}"
        )

        return serving_instance.id

    except Exception as e:
        task.get_logger().report_text(f"Ошибка деплоя: {str(e)}")
        raise


if __name__ == "__main__":

    train_lr = train_model(
        model_class="LinearRegressionModel", model_config_key="linear_regression"
    )

    train_lr = train_model(
        model_class="RandomForestModel", model_config_key="random_forest"
    )
