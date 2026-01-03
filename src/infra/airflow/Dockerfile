# infra/airflow/Dockerfile
FROM apache/airflow:2.9.3-python3.11

USER root

# Cần docker CLI vì bạn dùng BashOperator chạy: docker exec spark-master ...
# (mount /var/run/docker.sock chỉ đưa socket, không đưa docker client binary)
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      docker.io \
      curl \
      ca-certificates \
    ; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

# Requirements (nếu bạn cần thêm libs cho DAG / operators)
COPY requirements.txt /requirements.txt

USER airflow
ENV PATH="/home/airflow/.local/bin:${PATH}"

RUN python -m pip install --no-cache-dir -U pip setuptools wheel && \
    python -m pip install --no-cache-dir -r /requirements.txt
