# Используем официальный образ Airflow в качестве базового
FROM apache/airflow:2.8.3

# Переключаемся на root для установки дополнительных пакетов
USER root

# Устанавливаем дополнительные системные пакеты, если нужно
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    libgomp1 && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Возвращаемся к пользователю airflow
USER airflow

COPY requirements.txt /tmp/requirements.txt

RUN pip install --user --no-cache-dir -r /tmp/requirements.txt

