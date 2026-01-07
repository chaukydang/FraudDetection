# src/infra/mlflow/Dockerfile
FROM python:3.10.16-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir \
    'urllib3>=1.26.18,<2.0.0' \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && pip cache purge

COPY . .

RUN useradd -m -u 1000 mlflow && chown -R mlflow:mlflow /app
USER mlflow

EXPOSE 5500

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5500/health || exit 1

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5500", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "./mlruns"]
