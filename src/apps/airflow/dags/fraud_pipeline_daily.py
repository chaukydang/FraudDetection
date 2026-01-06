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

# One unified wrapper used by ALL tasks to avoid drift.
SPARK_WRAPPER = r"""
export GIT_PYTHON_REFRESH=quiet
export MLFLOW_DISABLE_GIT_METADATA=true

set -euo pipefail
shopt -s inherit_errexit 2>/dev/null || true

# Load shared env (AWS keys, etc.)
set -a
source /opt/.env
set +a

SPARK_MASTER_CONTAINER="${SPARK_MASTER_CONTAINER:-spark-master}"

cleanup_on_term() {
  echo "[wrapper] received termination signal, killing process group..."
  kill -TERM 0 2>/dev/null || true
  sleep 2 || true
  kill -KILL 0 2>/dev/null || true
}
trap cleanup_on_term TERM INT HUP

# Get minio endpoint from config inside spark container
MINIO_ENDPOINT="$(docker exec "${SPARK_MASTER_CONTAINER}" bash -lc 'python3 - <<PY
import yaml
cfg=yaml.safe_load(open("/opt/config.yaml"))
print(cfg["minio"]["endpoint"])
PY')"

export MLFLOW_S3_ENDPOINT_URL="${MINIO_ENDPOINT}"
export AWS_EC2_METADATA_DISABLED="true"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export AWS_REGION="${AWS_REGION:-us-east-1}"

docker_exec_maybe_timeout() {
  local t="${1:-}"; shift
  local c="${1}"; shift

  if [[ -n "${t}" && "${t}" != "0" ]]; then
    timeout "${t}" docker exec \
      -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
      -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
      -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}" \
      -e AWS_REGION="${AWS_REGION:-us-east-1}" \
      -e AWS_EC2_METADATA_DISABLED="true" \
      -e MLFLOW_S3_ENDPOINT_URL="${MINIO_ENDPOINT}" \
      "${c}" "$@"
  else
    docker exec \
      -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
      -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
      -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}" \
      -e AWS_REGION="${AWS_REGION:-us-east-1}" \
      -e AWS_EC2_METADATA_DISABLED="true" \
      -e MLFLOW_S3_ENDPOINT_URL="${MINIO_ENDPOINT}" \
      "${c}" "$@"
  fi
}

spark_submit() {
  local app="$1"; shift
  local t="${SPARK_SUBMIT_TIMEOUT:-}"  # seconds; empty = no timeout

  docker_exec_maybe_timeout "${t}" "${SPARK_MASTER_CONTAINER}" \
    /opt/spark/bin/spark-submit \
      --master spark://spark-master:7077 \
      --deploy-mode client \
      --conf spark.jars.ivy=/opt/ivy-cache \
      --conf spark.network.timeout=600s \
      --conf spark.executor.heartbeatInterval=30s \
      --conf spark.dynamicAllocation.enabled=false \
      --conf spark.executor.instances=1 \
      --conf spark.executor.cores=1 \
      --conf spark.executor.memory=2g \
      --conf spark.driver.memory=1g \
      --conf spark.sql.shuffle.partitions=4 \
      --conf spark.sql.sources.partitionOverwriteMode=dynamic \
      --conf spark.ui.enabled=false \
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
export SPARK_SUBMIT_TIMEOUT=900
spark_submit /opt/jobs/bronze_to_silver.py \
  --config /opt/config.yaml \
  --date {{ ds }} \
  --window_days 3
""",
        execution_timeout=timedelta(minutes=30),
    )

    build_features = BashOperator(
        task_id="build_features",
        bash_command=SPARK_WRAPPER + r"""
export SPARK_SUBMIT_TIMEOUT=900
spark_submit /opt/jobs/build_features.py \
  --config /opt/config.yaml \
  --date {{ ds }} \
  --window_days 7
""",
        execution_timeout=timedelta(minutes=45),
    )

    validate_gold_schema = BashOperator(
        task_id="validate_gold_schema",
        bash_command=SPARK_WRAPPER + r"""
export SPARK_SUBMIT_TIMEOUT=600
DS="{{ ds }}"

GOLD_BASE="$(docker exec spark-master bash -lc 'python3 - <<PY
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
        execution_timeout=timedelta(minutes=20),
    )

    # ✅ FIX DỨT ĐIỂM: train_model cũng dùng SPARK_WRAPPER, và chạy bằng marker watchdog
    train_model = BashOperator(
        task_id="train_model",
        bash_command=SPARK_WRAPPER + r"""
export SPARK_SUBMIT_TIMEOUT=900
DS="{{ ds }}"

# Clean marker from previous runs
docker exec "${SPARK_MASTER_CONTAINER}" rm -f /tmp/train_done.ok || true

# Build input path (no heredoc, no indent bugs)
GOLD_BASE="$(docker exec "${SPARK_MASTER_CONTAINER}" python3 -c "import yaml; cfg=yaml.safe_load(open('/opt/config.yaml')); print(f\"s3a://{cfg['minio']['bucket']}/{cfg['paths']['gold_features'].strip('/')}\" )")"
INPUT_PATH="${GOLD_BASE}/event_date=${DS}"
echo "[train_model] input_path=${INPUT_PATH}"

# Run inside spark-master with watchdog: if python leaves stray threads, we still exit clean
docker_exec_maybe_timeout "${SPARK_SUBMIT_TIMEOUT}" "${SPARK_MASTER_CONTAINER}" bash -lc '
  set -euo pipefail

  /opt/spark/bin/spark-submit \
    --master "local[1]" \
    --deploy-mode client \
    --conf spark.ui.enabled=false \
    --conf spark.dynamicAllocation.enabled=false \
    \
    /opt/trainner/train_entry.py \
      --config /opt/config.yaml \
      --input "'"${INPUT_PATH}"'" \
      --model gbt \
      --run_name "gbt_bootstrap_'"${DS}"'" \
    &

  SPARK_PID=$!
  echo "[container] spark-submit pid=${SPARK_PID}"

  # Wait up to 10 minutes for marker
  for i in $(seq 1 600); do
    if test -f /tmp/train_done.ok; then
      echo "[container] DONE marker detected"
      break
    fi
    if ! ps -p "${SPARK_PID}" >/dev/null 2>&1; then
      wait "${SPARK_PID}" || true
      exit 0
    fi
    sleep 1
  done

  # If still running, kill it
  if ps -p "${SPARK_PID}" >/dev/null 2>&1; then
    echo "[container] spark-submit still alive => killing"
    kill -TERM "${SPARK_PID}" 2>/dev/null || true
    sleep 2 || true
    kill -KILL "${SPARK_PID}" 2>/dev/null || true
  else
    echo "[container] spark-submit already exited"
  fi

  # Return exit code from marker
  EC=0
  if test -f /tmp/train_done.ok; then
    EC=$(sed -n "s/^exit_code=//p" /tmp/train_done.ok | tail -n 1)
  fi

  rm -f /tmp/train_done.ok || true
  echo "[container] exit ${EC}"
  exit "${EC}"
'

echo "[train_model] ✅ Done"
""",
        execution_timeout=timedelta(minutes=20),
    )

    # Keep cleanup as "warn-only" safety net (optional)
    cleanup_spark = BashOperator(
        task_id="cleanup_spark_processes",
        bash_command=SPARK_WRAPPER + r"""
set -euo pipefail
echo "[cleanup] best-effort cleanup (warn-only)"

# clean marker to prevent confusing next runs
docker exec "${SPARK_MASTER_CONTAINER}" rm -f /tmp/train_done.ok 2>/dev/null || true

echo "[cleanup] ✅ done"
exit 0
""",
        trigger_rule="all_done",
    )

    validate_env >> bronze_to_silver >> build_features >> validate_gold_schema >> train_model >> cleanup_spark
