FROM python:3.10.16-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies in one layer (minimize layers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /app/requirements.txt

# Install Python dependencies with fixes
# 1. urllib3<2.0.0 - Fix MinIO HeaderParsingError
# 2. Specific versions for stability
# 3. Clean pip cache to reduce image size
RUN pip install --no-cache-dir \
    'urllib3>=1.26.18,<2.0.0' \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && pip cache purge

# Copy the rest of the application
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 mlflow && chown -R mlflow:mlflow /app
USER mlflow

# Expose MLflow port
EXPOSE 5500

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5500/health || exit 1

# Default command (can be overridden in docker-compose)
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5500", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "./mlruns"]