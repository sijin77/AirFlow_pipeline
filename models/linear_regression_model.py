from typing import Any, Dict, Optional
import optuna
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from models.base_model import BaseModel


class LinearRegressionModel(BaseModel):
    def __init__(
        self,
        data_processor: Optional[Any] = None,
        fit_intercept: bool = True,
        normalize: bool = False,
        **kwargs
    ):
        super().__init__(
            task_type="regression",
            data_processor=data_processor,
            model_name="LinearRegression",
            best_metric_key="mse",  # Можно использовать 'r2' для максимизации
            **kwargs
        )

        self.model = LinearRegression(
            fit_intercept=fit_intercept, copy_X=True, n_jobs=-1
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

        # Log feature importance (coefficients)
        if hasattr(self.model, "coef_"):
            plt.figure(figsize=(10, 6))
            sns.barplot(
                x=self.model.coef_,
                y=self.data_processor.feature_names,
                palette="viridis",
            )
            plt.title("Feature Coefficients")
            self.task.get_logger().report_matplotlib_figure(
                title="Feature Coefficients",
                series="coefficients",
                figure=plt.gcf(),
                iteration=0,
            )
            plt.close()

    def _suggest_hyperparams(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "normalize": trial.suggest_categorical("normalize", [True, False]),
        }
