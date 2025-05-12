config = {
    # Общие настройки
    "random_state": 42,
    "n_jobs": -1,  # Использовать все ядра CPU
    # Настройки данных
    "data": {
        "test_size": 0.2,
        "shuffle": True,
        "stratify": True,  # Стратифицированное разбиение
    },
    # Logistic Regression
    "logistic_regression": {
        "max_iter": 10,
        "penalty": "l2",  # l1, l2, elasticnet, none
        "C": 1.0,  # Сила регуляризации
        "solver": "lbfgs",  # Для multiclass
        "tol": 1e-4,  # Критерий остановки
    },
    # Decision Tree
    "decision_tree": {
        "max_depth": 10,
        "criterion": "gini",  # или "entropy"
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",  # Количество фичей для разделения
    },
    "svm": {
        "C": 1.0,
        "kernel": "rbf",  # linear, poly, rbf, sigmoid
        "gamma": "scale",
        "degree": 3,  # Для poly
    },
    "xgboost": {
        "n_estimators": 50,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    # Настройки логирования
    "logging": {
        "experiment_name": "Digits_Classification",
        "save_models": True,
    },
}
