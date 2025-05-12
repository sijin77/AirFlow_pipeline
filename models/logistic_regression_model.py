from typing import Any, Dict, Optional
import optuna
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from models.base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    def __init__(
        self,
        data_processor: Optional[Any] = None,
        max_iter: int = 100,
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        **kwargs
    ):
        super().__init__(
            task_type="classification",
            data_processor=data_processor,
            model_name="LogisticRegression",
            best_metric_key="f1",
            **kwargs
        )

        self.model = LogisticRegression(
            max_iter=max_iter,
            C=C,
            penalty=penalty,
            solver=solver,
            multi_class="multinomial",
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
            y_proba = self.model.predict_proba(X_test)

            self._cached_metrics = self._get_metrics(X_test, y_test, y_pred, y_proba)
            self.log_artifacts(X_test, y_test, y_pred, y_proba)
        return self._cached_metrics

    def log_artifacts(self, X, y_true, y_pred, y_proba=None) -> None:
        super().log_artifacts(X, y_true, y_pred, y_proba)

        # Log coefficients heatmap
        if hasattr(self.model, "coef_"):
            plt.figure(figsize=(10, 6))
            sns.heatmap(self.model.coef_, annot=True, fmt=".2f", cmap="coolwarm")
            self.task.get_logger().report_matplotlib_figure(
                title="Model Coefficients",
                series="coefficients",
                figure=plt.gcf(),
                iteration=0,
            )
            plt.close()

    def _suggest_hyperparams(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "C": trial.suggest_float("C", 0.01, 10, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l2", "none"]),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "sag"]),
            "max_iter": trial.suggest_int("max_iter", 50, 300),
        }
