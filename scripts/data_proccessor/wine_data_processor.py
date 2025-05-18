import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scripts.data_proccessor.data_processor_base import DataProcessor


class WineQualityDataProcessor(DataProcessor):
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        super().__init__(
            task_type="regression",
            test_size=test_size,
            random_state=random_state,
        )
        self.file_path = "dataset/winequality-red.csv"

    def load_data(self) -> tuple:
        """Загрузка датасета из CSV-файла"""
        wine_data = pd.read_csv(self.file_path, delimiter=",")

        # Разделяем на признаки и целевую переменную
        X = wine_data.drop("quality", axis=1).values
        y = wine_data["quality"].values

        # Сохраняем названия признаков
        self.feature_names = wine_data.drop("quality", axis=1).columns.tolist()

        return X, y

    def preprocess(self, X, y=None) -> tuple:
        """Предобработка данных: нормализация признаков"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y
