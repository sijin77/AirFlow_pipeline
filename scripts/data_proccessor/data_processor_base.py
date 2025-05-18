# data_processor.py

from abc import ABC, abstractmethod
import numpy as np
from sklearn.model_selection import train_test_split
from clearml import Task


class DataProcessor(ABC):
    """Абстрактный класс для обработки данных"""

    def __init__(
        self,
        task_type: str = "classification",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Args:
            task_type: 'classification', 'regression' или 'clustering'
            test_size: доля тестовой выборки
            random_state: seed для воспроизводимости
        """
        self.task_type = task_type
        self.test_size = test_size
        self.random_state = random_state
        task = Task.current_task()
        self.logger = task.get_logger() if task else None

    @abstractmethod
    def load_data(self) -> tuple:
        """Загрузка сырых данных"""
        pass

    @abstractmethod
    def preprocess(self, X, y=None) -> tuple:
        """Предобработка данных"""
        pass

    def split_data(self, X, y) -> dict:
        """Разделение данных на train/test с учётом типа задачи"""
        stratify = y if self.task_type == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    def get_processed_data(self) -> dict:
        """Полный пайплайн обработки: загрузка → предобработка → разделение"""
        X, y = self.load_data()
        X, y = self.preprocess(X, y)
        return self.split_data(X, y)

    def log_data_stats(self, data: dict) -> None:
        """Логирование статистики данных в ClearML"""
        stats = {
            "dataset.train_samples": len(data["y_train"]),
            "dataset.test_samples": len(data["y_test"]),
            "dataset.num_features": data["X_train"].shape[1],
        }
        if self.task_type == "classification":
            stats["dataset.num_classes"] = len(np.unique(data["y_train"]))

        print("[DataProcessor] Логируем статистику данных...")
        for key, value in stats.items():
            self.logger.report_single_value(key, value)
