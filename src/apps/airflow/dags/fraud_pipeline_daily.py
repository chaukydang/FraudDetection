# apps/airflow/dags/fraud_pipeline_daily.py
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

DEFAULT_ARGS = {
    "owner": "fraud-platform",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
    "execution_timeout": timedelta(hours=2),
}

SPARK_WRAPPER = r"""
set -euo pipefail

# Load shared env
set -a
source /opt/.env
set +a

SPARK_MASTER_CONTAINER="${SPARK_MASTER_CONTAINER:-spark-master}"

# Pull minio endpoint from config
MINIO_ENDPOINT="$(docker exec "${SPARK_MASTER_CONTAINER}" bash -lc 'python - <<PY
import yaml
cfg=yaml.safe_load(open("/opt/config.yaml"))
print(cfg["minio"]["endpoint"])
PY')"

# MLflow/MinIO env
export MLFLOW_S3_ENDPOINT_URL="${MINIO_ENDPOINT}"
export AWS_EC2_METADATA_DISABLED="true"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export AWS_REGION="${AWS_REGION:-us-east-1}"

spark_submit () {
  local app="$1"; shift

  docker exec \
    -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
    -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
    -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}" \
    -e AWS_REGION="${AWS_REGION:-us-east-1}" \
    -e AWS_EC2_METADATA_DISABLED="true" \
    -e MLFLOW_S3_ENDPOINT_URL="${MINIO_ENDPOINT}" \
    "${SPARK_MASTER_CONTAINER}" \
    /opt/spark/bin/spark-submit \
      --master spark://spark-master:7077 \
      --deploy-mode client \
      --conf spark.jars.ivy=/opt/ivy-cache \
      --conf spark.network.timeout=600s \
      --conf spark.executor.heartbeatInterval=30s \
      --conf spark.executor.instances=1 \
      --conf spark.executor.cores=1 \
      --conf spark.executor.memory=2g \
      --conf spark.driver.memory=1g \
      --conf spark.sql.shuffle.partitions=4 \
      --conf spark.sql.sources.partitionOverwriteMode=dynamic \
      \
      --conf spark.executorEnv.AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
      --conf spark.executorEnv.AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
      --conf spark.executorEnv.AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}" \
      --conf spark.executorEnv.AWS_REGION="${AWS_REGION:-us-east-1}" \
      --conf spark.executorEnv.AWS_EC2_METADATA_DISABLED="true" \
      --conf spark.executorEnv.MLFLOW_S3_ENDPOINT_URL="${MINIO_ENDPOINT}" \
      \
      --conf spark.hadoop.fs.s3a.endpoint="${MINIO_ENDPOINT}" \
      --conf spark.hadoop.fs.s3a.path.style.access=true \
      --conf spark.hadoop.fs.s3a.connection.ssl.enabled=false \
      --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem \
      --conf spark.hadoop.fs.s3a.aws.credentials.provider=org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider \
      --conf spark.hadoop.fs.s3a.access.key="${AWS_ACCESS_KEY_ID}" \
      --conf spark.hadoop.fs.s3a.secret.key="${AWS_SECRET_ACCESS_KEY}" \
      \
      --conf spark.hadoop.fs.file.checksum.enabled=false \
      --conf spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version=2 \
      --conf spark.hadoop.mapreduce.fileoutputcommitter.cleanup-failures.ignored=true \
      --conf spark.hadoop.fs.s3a.committer.name=directory \
      --conf spark.hadoop.fs.s3a.committer.staging.conflict-mode=append \
      --conf spark.hadoop.fs.s3a.fast.upload=true \
      --conf spark.hadoop.fs.s3a.fast.upload.buffer=disk \
      \
      "$app" "$@"
}
"""

with DAG(
    dag_id="fraud_pipeline_daily",
    default_args=DEFAULT_ARGS,
    description="Fraud pipeline: bronze->silver -> build features -> train model (MLflow)",
    schedule="0 2 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["fraud", "spark", "mlflow"],
) as dag:

    validate_env = BashOperator(
        task_id="validate_environment",
        bash_command=SPARK_WRAPPER + r"""
        test -f /opt/config.yaml || (echo "❌ /opt/config.yaml not found (airflow)" && exit 1)
        test -f /opt/.env || (echo "❌ /opt/.env not found (airflow)" && exit 1)

        docker exec spark-master test -f /opt/config.yaml || (echo "❌ /opt/config.yaml not found (spark-master)" && exit 1)
        docker exec spark-master test -f /opt/jobs/bronze_to_silver.py || (echo "❌ /opt/jobs/bronze_to_silver.py not found" && exit 1)
        docker exec spark-master test -f /opt/jobs/build_features.py || (echo "❌ /opt/jobs/build_features.py not found" && exit 1)
        docker exec spark-master test -f /opt/trainner/train_entry.py || (echo "❌ /opt/trainner/train_entry.py not found" && exit 1)
        docker exec spark-master test -f /opt/trainner/fraud_detection_training.py || (echo "❌ /opt/trainner/fraud_detection_training.py not found" && exit 1)

        test -n "${AWS_ACCESS_KEY_ID:-}" || (echo "❌ AWS_ACCESS_KEY_ID is empty" && exit 1)
        test -n "${AWS_SECRET_ACCESS_KEY:-}" || (echo "❌ AWS_SECRET_ACCESS_KEY is empty" && exit 1)

        echo "✅ Environment validated"
        """,
    )

    bronze_to_silver = BashOperator(
        task_id="bronze_to_silver",
        bash_command=SPARK_WRAPPER + r"""
        spark_submit /opt/jobs/bronze_to_silver.py \
          --config /opt/config.yaml \
          --date {{ ds }} \
          --window_days 3
        """,
    )

    build_features = BashOperator(
        task_id="build_features",
        bash_command=SPARK_WRAPPER + r"""
        spark_submit /opt/jobs/build_features.py \
          --config /opt/config.yaml \
          --date {{ ds }} \
          --window_days 7
        """,
    )

    validate_gold_schema = BashOperator(
        task_id="validate_gold_schema",
        bash_command=SPARK_WRAPPER + r"""
DS="{{ ds }}"

GOLD_BASE="$(docker exec spark-master bash -lc 'python - <<PY
import yaml
cfg=yaml.safe_load(open("/opt/config.yaml"))
bucket=cfg["minio"]["bucket"]
gold=cfg["paths"]["gold_features"].strip("/")
print(f"s3a://{bucket}/{gold}")
PY')"

INPUT_PATH="${GOLD_BASE}/event_date=${DS}"
echo "[validate_gold_schema] checking ${INPUT_PATH}"

docker exec spark-master bash -lc "cat >/tmp/_check_schema.py <<'PY'
from pyspark.sql import SparkSession
import sys

spark = SparkSession.builder.appName('check_schema').getOrCreate()
p = sys.argv[1]
df = spark.read.parquet(p)
print('COLUMNS=', df.columns)
df.printSchema()

if 'features' not in df.columns:
    raise RuntimeError('Missing features column: ' + p)

spark.stop()
PY"

spark_submit /tmp/_check_schema.py "${INPUT_PATH}"
""",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=SPARK_WRAPPER + r"""
DS="{{ ds }}"

GOLD_BASE="$(docker exec spark-master bash -lc 'python - <<PY
import yaml
cfg=yaml.safe_load(open("/opt/config.yaml"))
bucket=cfg["minio"]["bucket"]
gold=cfg["paths"]["gold_features"].strip("/")
print(f"s3a://{bucket}/{gold}")
PY')"

INPUT_PATH="${GOLD_BASE}/event_date=${DS}"
echo "[train_model] input_path=${INPUT_PATH}"

# Run training with explicit error handling
set +e
spark_submit /opt/trainner/train_entry.py \
  --config /opt/config.yaml \
  --input "${INPUT_PATH}" \
  --model xgb \
  --run_name "xgb_daily_${DS}"

exit_code=$?
set -e

# Cleanup any hung processes
echo "[cleanup] Checking for hung Spark processes..."
docker exec spark-master pkill -9 -f "train_entry.py" 2>/dev/null || true

# Check exit code
if [ $exit_code -eq 0 ]; then
    echo "✅ Training completed successfully"
    exit 0
else
    echo "❌ Training failed with exit code $exit_code"
    exit $exit_code
fi
""",
        execution_timeout=timedelta(minutes=15),
    )

    cleanup_spark = BashOperator(
        task_id="cleanup_spark_processes",
        bash_command=r"""
echo "[cleanup] Final cleanup check..."

# Kill any remaining train_entry processes
docker exec spark-master pkill -9 -f "train_entry.py" 2>/dev/null && \
    echo "Killed remaining training processes" || \
    echo "No remaining processes found"

# Verify cleanup
remaining=$(docker exec spark-master ps aux | grep train_entry | grep -v grep | wc -l)
if [ $remaining -gt 0 ]; then
    echo "⚠️  Warning: $remaining train_entry processes still running"
    exit 1
else
    echo "✅ All Spark processes cleaned up"
    exit 0
fi
""",
        trigger_rule='all_done',  # Run even if train_model fails
    )

    # Task dependencies
    validate_env >> bronze_to_silver >> build_features >> validate_gold_schema >> train_model >> cleanup_spark