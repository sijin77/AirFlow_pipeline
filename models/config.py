config = {
    # Общие настройки
    "random_state": 42,
    "n_jobs": -1,  # Использовать все ядра CPU
    # Настройки данных
    "data": {
        "test_size": 0.2,
        "shuffle": True,
        "stratify": False,  # Для регрессии стратификация обычно не применяется
    },
    # Logistic Regression
    "logistic_regression": {
        "max_iter": 100,
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "tol": 1e-4,
    },
    # Linear Regression
    "linear_regression": {
        "fit_intercept": True,
        "normalize": False,
        "copy_X": True,
    },
    # Random Forest
    "random_forest": {
        "n_estimators": 100,
        "max_depth": None,  # None означает неограниченную глубину
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "auto",  # auto = sqrt(n_features)
        "bootstrap": True,
        "oob_score": False,
    },
    # Decision Tree
    "decision_tree": {
        "max_depth": 10,
        "criterion": "gini",
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },
    # SVM
    "svm": {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale",
        "degree": 3,
        "epsilon": 0.1,
    },
    # XGBoost (может работать и для регрессии)
    "xgboost": {
        "n_estimators": 20,  # В ПРОД лучше побольше 50+
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",  # Для регрессии
        "eval_metric": "rmse",  # Метрика для регрессии
    },
    # Настройки логирования
    "logging": {
        "experiment_name": "Wine_Quality_Regression",
        "save_models": True,
        "metrics": ["mse", "mae", "r2"],  # Основные метрики для регрессии
    },
}
