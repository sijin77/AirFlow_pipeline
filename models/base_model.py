from abc import abstractmethod
import os
import joblib
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    silhouette_score,
    calinski_harabasz_score,
)
from typing import Dict, Optional, Any
from clearml import Task, OutputModel
import optuna


class BaseModel:
    """Базовый класс модели с интеграцией ClearML"""

    def __init__(
        self,
        task_type: str,
        data_processor: Optional[Any] = None,
        model_name: str = "untitled_model",
        enable_logging: bool = True,
        enable_optimization: bool = False,
        best_metric_key: str = "f1",
        **kwargs,
    ):
        self.task_type = task_type
        self.model_name = model_name
        self.params = kwargs
        self.model = None
        self.data_processor = data_processor
        self.data = None
        self.enable_logging = enable_logging
        self.enable_optimization = enable_optimization
        self.best_metric_key = best_metric_key
        self.task = Task.current_task() if enable_logging else None
        self.logger = self.task.get_logger() if self.task else None

        self.log_params()

    def log_params(self) -> None:
        """Логирует параметры в ClearML с группировкой"""
        if not self.enable_logging or not self.task:
            return

        params_to_log = {
            f"model/{k}": v
            for k, v in {
                "type": self.task_type,
                "name": self.model_name,
                **self.params,
            }.items()
        }
        self.task.connect(params_to_log)

    def _get_metrics(self, X, y_true, y_pred, y_proba=None) -> Dict[str, float]:
        """Вычисляет метрики в зависимости от типа задачи"""
        metrics = {}
        if self.task_type == "classification":
            metrics.update(
                {
                    "accuracy": accuracy_score(y_true, y_pred),
                    "precision": precision_score(y_true, y_pred, average="weighted"),
                    "recall": recall_score(y_true, y_pred, average="weighted"),
                    "f1": f1_score(y_true, y_pred, average="weighted"),
                }
            )
            if y_proba is not None:
                try:
                    metrics["roc_auc"] = roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="weighted"
                    )
                except Exception as e:
                    print(f"[WARNING] ROC AUC calculation failed: {e}")

        elif self.task_type == "regression":
            metrics.update(
                {
                    "mse": mean_squared_error(y_true, y_pred),
                    "mae": mean_absolute_error(y_true, y_pred),
                    "r2": r2_score(y_true, y_pred),
                }
            )
        else:  # clustering
            metrics.update(
                {
                    "silhouette": silhouette_score(X, y_pred),
                    "calinski_harabasz": calinski_harabasz_score(X, y_pred),
                }
            )

        if self.logger:
            for k, v in metrics.items():
                self.logger.report_single_value(k, v)

        return metrics

    def log_artifacts(self, X, y_true, y_pred, y_proba=None) -> None:
        """Логирует графики и отчеты"""
        if not self.logger:
            return

        # Confusion Matrix
        if self.task_type == "classification":
            plt.figure()
            sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d")
            self.logger.report_matplotlib_figure(
                title="Confusion Matrix",
                series="metrics",
                figure=plt.gcf(),
                iteration=0,
            )
            plt.close()

    def _get_previous_best_metric(self) -> Optional[float]:
        """Возвращает значение лучшей метрики из ранее зарегистрированных моделей"""
        if not self.task:
            return None

        # Ищем ранее зарегистрированные модели с тегом 'best'
        from clearml import Model

        models = Model.query_models(
            project_name=self.task.get_project_name(),
            model_name=f"best_{self.model_name}",
            tags=["best"],
            only_published=False,
        )

        if not models:
            return None

        # Берем последнюю зарегистрированную модель
        latest_model = models[0]

        # Пытаемся извлечь метрику из тегов
        for tag in latest_model.tags:
            if tag.startswith(f"{self.best_metric_key}:"):
                return float(tag.split(":")[1])

        return None

    def register_best_model(
        self, metric_name: Optional[str] = None, force: bool = False
    ) -> str:
        """
        Регистрирует модель как артефакт только если она лучше предыдущих

        Args:
            metric_name: имя метрики для сравнения (по умолчанию self.best_metric_key)
            force: если True, регистрирует модель даже если она не лучше

        Returns:
            ID зарегистрированной модели или пустую строку
        """
        if not self.task:
            return ""

        # Получаем метрики
        metrics = self.evaluate()
        metric_name = metric_name or self.best_metric_key
        current_metric = metrics.get(metric_name, 0)

        # Получаем предыдущий лучший результат
        previous_best = self._get_previous_best_metric()

        # Проверяем, нужно ли регистрировать новую модель
        if not force and previous_best is not None:
            is_better = (
                (current_metric > previous_best)
                if self.task_type != "regression"
                else (current_metric < previous_best)
            )
            if not is_better:
                return ""

        # Сохраняем модель и параметры
        model_path = os.path.join(os.getcwd(), f"best_{self.model_name}.pkl")
        params_path = os.path.join(os.getcwd(), f"best_{self.model_name}_params.json")

        joblib.dump(self.model, model_path)

        with open(params_path, "w") as f:
            json.dump(self.params, f)

        # Регистрируем в ClearML
        output_model = OutputModel(
            task=self.task,
            name=f"best_{self.model_name}",
            tags=["production", "best", f"{metric_name}:{current_metric}"],
        )

        # Добавляем файлы
        output_model.update_weights(weights_filename=model_path)
        output_model.update_weights(weights_filename=params_path)  # Добавляем параметры

        # Удаляем временные файлы
        # os.remove(model_path)
        # os.remove(params_path)

        # Логируем параметры отдельно для удобства
        self.task.connect(self.params)

        return output_model.id

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        pass

    def optimize(self, n_trials=100) -> optuna.Study:
        """Оптимизация гиперпараметров через Optuna"""
        if not self.enable_optimization:
            raise RuntimeError("Optimization is disabled")

        def objective(trial):
            self.model.set_params(**self._suggest_hyperparams(trial))
            self.train()
            return self.evaluate()[self.best_metric_key]

        study = optuna.create_study(
            direction="maximize" if self.task_type != "regression" else "minimize"
        )
        study.optimize(objective, n_trials=n_trials)
        self.params.update(study.best_params)
        return study

    @abstractmethod
    def _suggest_hyperparams(self, trial: optuna.Trial) -> Dict[str, Any]:
        pass
