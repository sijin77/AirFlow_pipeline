from typing import Any, Dict, Optional

from matplotlib import pyplot as plt
import optuna
from models.base_model import BaseModel
from xgboost import XGBClassifier
import xgboost as xgb


class XGBoostModel(BaseModel):
    def __init__(
        self,
        data_processor: Optional[Any] = None,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(
            task_type="classification",
            data_processor=data_processor,
            model_name="XGBoost",
            best_metric_key="f1",
            **kwargs
        )

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )

    def train(self) -> None:
        if self.data is None:
            self.data = self.data_processor.get_processed_data()
        self.model.fit(
            self.data["X_train"],
            self.data["y_train"],
            eval_set=[(self.data["X_test"], self.data["y_test"])],
            verbose=False,
        )

    def evaluate(self) -> Dict[str, float]:
        X_test = self.data["X_test"]
        y_test = self.data["y_test"]
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        metrics = self._get_metrics(X_test, y_test, y_pred, y_proba)
        self.log_artifacts(X_test, y_test, y_pred)
        return metrics

    def log_artifacts(self, X, y_true, y_pred, y_proba=None) -> None:
        super().log_artifacts(X, y_true, y_pred)

        # Логируем важность фичей
        plt.figure(figsize=(10, 6))
        xgb.plot_importance(self.model)
        self.task.get_logger().report_matplotlib_figure(
            title="XGBoost Feature Importance",
            series="metrics",
            figure=plt.gcf(),
            iteration=0,
        )
        plt.close()

    def _suggest_hyperparams(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
