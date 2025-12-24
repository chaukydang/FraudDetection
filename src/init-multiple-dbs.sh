#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE USER mlflow WITH PASSWORD 'mlflow';
    CREATE DATABASE mlflow;
    GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;
EOSQL

# Connect to the mlflow database and grant schema privileges
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "mlflow" <<-EOSQL
    GRANT ALL ON SCHEMA public TO mlflow;
    GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mlflow;
    GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mlflow;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO mlflow;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO mlflow;
EOSQL