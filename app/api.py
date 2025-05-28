from datetime import datetime
import os
import dvc
import joblib
import numpy as np
from pydantic import BaseModel, Field, field_validator
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict

# Импортируем модель данных
from pydantic import BaseModel
import yaml

app = FastAPI()


class WineFeatures(BaseModel):
    fixed_acidity: float = Field(
        default=7.4,
        ge=4.0,
        le=16.0,
        description="Fixed acidity in g(tartaric acid)/dm³",
    )
    volatile_acidity: float = Field(
        default=0.7,
        ge=0.1,
        le=2.0,
        description="Volatile acidity in g(acetic acid)/dm³",
    )
    citric_acid: float = Field(
        default=0.0, ge=0.0, le=1.5, description="Citric acid in g/dm³"
    )
    residual_sugar: float = Field(
        default=1.9, ge=0.5, le=30.0, description="Residual sugar in g/dm³"
    )
    chlorides: float = Field(
        default=0.076,
        ge=0.01,
        le=0.5,
        description="Chlorides in g(sodium chloride)/dm³",
    )
    free_sulfur_dioxide: float = Field(
        default=11.0, ge=1.0, le=200.0, description="Free sulfur dioxide in mg/dm³"
    )
    total_sulfur_dioxide: float = Field(
        default=34.0, ge=5.0, le=400.0, description="Total sulfur dioxide in mg/dm³"
    )
    density: float = Field(
        default=0.9978, ge=0.98, le=1.1, description="Density in g/cm³"
    )
    pH: float = Field(default=3.51, ge=2.5, le=4.5, description="pH value")
    sulphates: float = Field(
        default=0.56,
        ge=0.3,
        le=2.0,
        description="Sulphates in g(potassium sulphate)/dm³",
    )
    alcohol: float = Field(default=9.4, ge=8.0, le=15.0, description="Alcohol in % vol")

    @field_validator("*", mode="before")
    def check_nan_values(cls, v):
        """Проверка на NaN/None значения"""
        if v is None or (isinstance(v, float) and np.isnan(v)):
            raise ValueError("Значение не может быть NaN или None")
        return v


# --- Пример простой "модели" ---
class WineQualityModel:
    def __init__(self):
        self.model = None
        self.metrics = {}
        self.model_info = {}
        self.load_model()

    def load_model(self):
        """Загрузка модели из DVC"""
        try:
            # Путь к модели в DVC
            model_path = dvc.api.get_url(
                "models/wine_quality_model.pkl",
            )

            # Скачиваем и загружаем модель
            os.system(f"dvc pull {model_path}")
            self.model = joblib.load("models/wine_quality_model.pkl")

            # Загрузка метрик
            with open("models/model_metrics.yaml", "r") as f:
                self.metrics = yaml.safe_load(f)

            # Информация о модели
            self.model_info = {
                "model_type": type(self.model).__name__,
                "training_date": datetime.fromtimestamp(
                    os.path.getmtime("models/wine_quality_model.pkl")
                ).isoformat(),
                "features": list(WineFeatures.__fields__.keys()),
            }

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def predict(self, features: WineFeatures) -> float:
        if not self.model:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            features_list = [
                features.fixed_acidity,
                features.volatile_acidity,
                features.citric_acid,
                features.residual_sugar,
                features.chlorides,
                features.free_sulfur_dioxide,
                features.total_sulfur_dioxide,
                features.density,
                features.pH,
                features.sulphates,
                features.alcohol,
            ]
            input_array = np.array(
                [[getattr(features, name) for name in features_list]]
            )

            # Проверка на совместимость с моделью
            if input_array.shape[1] != len(features_list):
                raise HTTPException(
                    status_code=422,
                    detail=f"Ожидается {len(features_list)} признаков, получено {input_array.shape[1]}",
                )

            return float(self.model.predict([features_list])[0])
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))


model = WineQualityModel()


# --- Роуты ---
@app.post("/predict")
async def predict(features: WineFeatures) -> Dict[str, float]:
    """Предсказание качества вина"""
    predicted_quality = model.predict(features)
    return {"predicted_quality": predicted_quality}


@app.get("/healthcheck")
async def healthcheck() -> Dict[str, str]:
    """Проверка работоспособности сервиса"""
    return {
        "status": "OK",
        "model_loaded": str(model.model is not None),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/model-info")
async def model_info() -> Dict[str, any]:
    """Информация о модели и её метриках"""
    if not model.model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {"model_info": model.model_info, "metrics": model.metrics}


# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
