from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataset.data_processor_base import DataProcessor


class DigitsDataProcessor(DataProcessor):
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        super().__init__(
            task_type="classification", test_size=test_size, random_state=random_state
        )

    def load_data(self) -> tuple:
        """Загрузка датасета load_digits"""
        digits = load_digits()
        self.feature_names = digits.feature_names
        X, y = digits.data, digits.target
        return X, y

    def preprocess(self, X, y=None) -> tuple:
        """Предобработка данных: нормализация признаков"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y
