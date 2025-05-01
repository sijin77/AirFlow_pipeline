import os
import requests
from dotenv import load_dotenv
import csv
from datetime import datetime


def fetch_weather(api_key: str, city="Moscow") -> dict:
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    weather_data = response.json()
    return weather_data


def save_weather_to_csv(api_key: str, city="Moscow"):
    # Получаем данные о погоде
    weather_data = fetch_weather(api_key, city)

    # Извлекаем нужные данные
    data_to_save = {
        "datetime": datetime.fromtimestamp(weather_data["dt"]).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "city": city,
        "weather_main": weather_data["weather"][0]["main"],
        "weather_description": weather_data["weather"][0]["description"],
        "temp": weather_data["main"]["temp"],
        "feels_like": weather_data["main"]["feels_like"],
        "pressure": weather_data["main"]["pressure"],
        "wind_speed": weather_data["wind"]["speed"],
    }

    # Определяем путь к файлу
    file_path = "/opt/airflow/dataset/weather.csv"

    # Проверяем, существует ли файл, чтобы определить, нужно ли писать заголовки
    file_exists = os.path.isfile(file_path)

    # Записываем данные в CSV
    with open(file_path, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=data_to_save.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(data_to_save)


if __name__ == "__main__":
    # test weather API
    load_dotenv(".env")
    api_key = os.getenv("OPENWEATHER_KEY")
    save_weather_to_csv(api_key=api_key)
