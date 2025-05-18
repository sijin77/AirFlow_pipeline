from typing import Any, Dict, Optional
import optuna
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from models.base_model import BaseModel


class RandomForestModel(BaseModel):
    def __init__(
        self,
        data_processor: Optional[Any] = None,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = "auto",
        **kwargs
    ):
        super().__init__(
            task_type="regression",
            data_processor=data_processor,
            model_name="RandomForest",
            best_metric_key="mse",
            **kwargs
        )

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1,
        )

    def train(self) -> None:
        if self.data is None:
            self.data = self.data_processor.get_processed_data()

        self.model.fit(self.data["X_train"], self.data["y_train"])

    def evaluate(self) -> Dict[str, float]:
        if not hasattr(self, "_cached_metrics"):
            X_test = self.data["X_test"]
            y_test = self.data["y_test"]

            y_pred = self.model.predict(X_test)

            self._cached_metrics = self._get_metrics(X_test, y_test, y_pred)
            self.log_artifacts(X_test, y_test, y_pred)
        return self._cached_metrics

    def log_artifacts(self, X, y_true, y_pred, y_proba=None) -> None:
        super().log_artifacts(X, y_true, y_pred)

        # Log feature importance
        if hasattr(self.model, "feature_importances_"):
            plt.figure(figsize=(10, 6))
            sns.barplot(
                x=self.model.feature_importances_,
                y=self.data_processor.feature_names,
                palette="viridis",
            )
            plt.title("Feature Importances")
            self.task.get_logger().report_matplotlib_figure(
                title="Feature Importances",
                series="importances",
                figure=plt.gcf(),
                iteration=0,
            )
            plt.close()

    def _suggest_hyperparams(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical(
                "max_features", ["auto", "sqrt", "log2"]
            ),
        }
