from typing import Any, Dict, Optional
import optuna
from sklearn.svm import SVC

from models.base_model import BaseModel


class SVMModel(BaseModel):
    def __init__(
        self,
        data_processor: Optional[Any] = None,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: str = "scale",
        **kwargs
    ):
        super().__init__(
            task_type="classification",
            data_processor=data_processor,
            model_name="SVM",
            best_metric_key="f1",
            **kwargs
        )

        self.model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=True,
            random_state=42,
        )

    def train(self) -> None:
        if self.data is None:
            self.data = self.data_processor.get_processed_data()
        self.model.fit(self.data["X_train"], self.data["y_train"])

    def evaluate(self) -> Dict[str, float]:
        X_test = self.data["X_test"]
        y_test = self.data["y_test"]
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        metrics = self._get_metrics(X_test, y_test, y_pred, y_proba)
        self.log_artifacts(X_test, y_test, y_pred)
        return metrics

    def _suggest_hyperparams(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "C": trial.suggest_float("C", 0.01, 10, log=True),
            "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
