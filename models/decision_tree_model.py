from typing import Any, Dict, Optional
from matplotlib import pyplot as plt
import optuna
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from models.base_model import BaseModel


class DecisionTreeModel(BaseModel):
    def __init__(
        self,
        data_processor: Optional[Any] = None,
        max_depth: int = 10,
        criterion: str = "gini",
        min_samples_split: int = 2,
        **kwargs
    ):
        super().__init__(
            task_type="classification",
            data_processor=data_processor,
            model_name="DecisionTree",
            best_metric_key="f1",
            **kwargs
        )

        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            criterion=criterion,
            min_samples_split=min_samples_split,
            random_state=42,
        )

    def train(self) -> None:
        if self.data is None:
            self.data = self.data_processor.get_processed_data()
        self.model.fit(self.data["X_train"], self.data["y_train"])

    def evaluate(self) -> Dict[str, float]:
        if not hasattr(self, "_cached_metrics"):
            # Вычисляем метрики впервые
            X_test = self.data["X_test"]
            y_test = self.data["y_test"]
            y_pred = self.model.predict(X_test)
            y_proba = self.model.predict_proba(X_test)
            self._cached_metrics = self._get_metrics(X_test, y_test, y_pred, y_proba)
            self.log_artifacts(X_test, y_test, y_pred)
        return self._cached_metrics

    def log_artifacts(self, X, y_true, y_pred, y_proba=None) -> None:
        super().log_artifacts(X, y_true, y_pred)

        # Логируем важность фичей
        if hasattr(self.model, "feature_importances_"):
            plt.figure(figsize=(10, 6))
            sns.barplot(
                x=self.model.feature_importances_,
                y=self.data_processor.feature_names,
                palette="viridis",
            )
            plt.title("Feature Importance")
            self.task.get_logger().report_matplotlib_figure(
                title="Feature Importance",
                series="metrics",
                figure=plt.gcf(),
                iteration=0,
            )
            plt.close()

    def _suggest_hyperparams(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        }
