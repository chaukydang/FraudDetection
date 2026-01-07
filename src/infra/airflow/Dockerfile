# src/infra/airflow/Dockerfile
FROM apache/airflow:2.9.3-python3.11

USER root

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      docker.io \
      curl \
      ca-certificates \
    ; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt

USER airflow
ENV PATH="/home/airflow/.local/bin:${PATH}"

RUN python -m pip install --no-cache-dir -U pip setuptools wheel && \
    python -m pip install --no-cache-dir -r /requirements.txt
