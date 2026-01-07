# src/apps/airflow/dags/fraud_pipeline_daily.py
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule

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

# In your docker-compose, containers are named exactly spark-master / spark-worker
SPARK_MASTER_CONTAINER="${SPARK_MASTER_CONTAINER:-spark-master}"
SPARK_MASTER_HOST="${SPARK_MASTER_HOST:-spark-master}"

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
export S3_ENDPOINT="${S3_ENDPOINT:-${MINIO_ENDPOINT}}"
export AWS_EC2_METADATA_DISABLED="true"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export AWS_REGION="${AWS_REGION:-us-east-1}"

# Helper: read key from /opt/config.yaml safely (inside spark-master)
# usage: cfg_get "minio.bucket"  or cfg_get "paths.bronze"
cfg_get() {
  local key="$1"
  docker exec -e CFG_KEY="${key}" "${SPARK_MASTER_CONTAINER}" bash -lc 'python3 - <<PY
import os,yaml
cfg=yaml.safe_load(open("/opt/config.yaml"))
key=os.environ["CFG_KEY"]
cur=cfg
for part in key.split("."):
    cur=cur[part]
print(cur)
PY'
}

# Ensure log4j2.properties exists on spark-master AND spark-worker
ensure_spark_log4j() {
  local targets=("${SPARK_MASTER_CONTAINER}" "spark-worker")

  for c in "${targets[@]}"; do
    if ! docker exec "${c}" bash -lc '
      set -euo pipefail
      cat > /tmp/log4j2.properties << "EOF"
status = error
name = SparkLogConfig

appender.console.type = Console
appender.console.name = console
appender.console.target = SYSTEM_ERR
appender.console.layout.type = PatternLayout
appender.console.layout.pattern = %d{yy/MM/dd HH:mm:ss} %p %c{1}: %m%n

rootLogger.level = ERROR
rootLogger.appenderRefs = console
rootLogger.appenderRef.console.ref = console

logger.spark.name = org.apache.spark
logger.spark.level = ERROR

logger.hadoop.name = org.apache.hadoop
logger.hadoop.level = ERROR

logger.aws.name = com.amazonaws
logger.aws.level = ERROR

logger.jetty.name = org.eclipse.jetty
logger.jetty.level = ERROR

logger.parquet.name = org.apache.parquet
logger.parquet.level = ERROR
EOF
      chmod 644 /tmp/log4j2.properties || true
      test -s /tmp/log4j2.properties
    ' >/dev/null 2>&1; then
      echo "[wrapper] WARN could not install /tmp/log4j2.properties in ${c} (container missing or no permission)"
    fi
  done
}
ensure_spark_log4j

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
      -e S3_ENDPOINT="${S3_ENDPOINT}" \
      "${c}" "$@"
  else
    docker exec \
      -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
      -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
      -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}" \
      -e AWS_REGION="${AWS_REGION:-us-east-1}" \
      -e AWS_EC2_METADATA_DISABLED="true" \
      -e MLFLOW_S3_ENDPOINT_URL="${MINIO_ENDPOINT}" \
      -e S3_ENDPOINT="${S3_ENDPOINT}" \
      "${c}" "$@"
  fi
}

spark_submit() {
  local app="$1"; shift
  local t="${SPARK_SUBMIT_TIMEOUT:-}"  # seconds; empty = no timeout

  docker_exec_maybe_timeout "${t}" "${SPARK_MASTER_CONTAINER}" \
    /opt/spark/bin/spark-submit \
      --master "spark://${SPARK_MASTER_HOST}:7077" \
      --deploy-mode client \
      --conf spark.jars.ivy=/opt/ivy-cache \
      --conf spark.network.timeout=600s \
      --conf spark.executor.heartbeatInterval=30s \
      --conf spark.dynamicAllocation.enabled=false \
      --conf spark.ui.enabled=false \
      --conf spark.ui.showConsoleProgress=false \
      --conf spark.eventLog.enabled=false \
      --conf spark.cores.max=1 \
      --conf spark.executor.instances=1 \
      --conf spark.executor.cores=1 \
      --conf spark.executor.memory=2g \
      --conf spark.driver.memory=1g \
      --conf spark.sql.shuffle.partitions=4 \
      --conf spark.sql.sources.partitionOverwriteMode=dynamic \
      --conf "spark.driver.extraJavaOptions=-Dlog4j2.configurationFile=file:/tmp/log4j2.properties -Dlog4j2.disable.jmx=true" \
      --conf "spark.executor.extraJavaOptions=-Dlog4j2.configurationFile=file:/tmp/log4j2.properties -Dlog4j2.disable.jmx=true" \
      --conf spark.executorEnv.AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
      --conf spark.executorEnv.AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
      --conf spark.executorEnv.AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}" \
      --conf spark.executorEnv.AWS_REGION="${AWS_REGION:-us-east-1}" \
      --conf spark.executorEnv.AWS_EC2_METADATA_DISABLED="true" \
      --conf spark.executorEnv.MLFLOW_S3_ENDPOINT_URL="${MINIO_ENDPOINT}" \
      --conf spark.executorEnv.S3_ENDPOINT="${S3_ENDPOINT}" \
      --conf spark.hadoop.fs.s3a.endpoint="${MINIO_ENDPOINT}" \
      --conf spark.hadoop.fs.s3a.path.style.access=true \
      --conf spark.hadoop.fs.s3a.connection.ssl.enabled=false \
      --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem \
      --conf spark.hadoop.fs.s3a.aws.credentials.provider=org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider \
      --conf spark.hadoop.fs.s3a.access.key="${AWS_ACCESS_KEY_ID}" \
      --conf spark.hadoop.fs.s3a.secret.key="${AWS_SECRET_ACCESS_KEY}" \
      --conf spark.hadoop.fs.file.checksum.enabled=false \
      --conf spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version=2 \
      --conf spark.hadoop.mapreduce.fileoutputcommitter.cleanup-failures.ignored=true \
      --conf spark.hadoop.fs.s3a.committer.name=directory \
      --conf spark.hadoop.fs.s3a.committer.staging.conflict-mode=replace \
      --conf spark.hadoop.fs.s3a.fast.upload=true \
      --conf spark.hadoop.fs.s3a.fast.upload.buffer=disk \
      "$app" "$@"
}
"""


with DAG(
    dag_id="fraud_pipeline_daily",
    default_args=DEFAULT_ARGS,
    description="Fraud daily batch: bronze->silver -> silver->gold(features) -> train (MLflow)",
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

docker exec "${SPARK_MASTER_CONTAINER}" test -f /opt/config.yaml || (echo "❌ /opt/config.yaml not found (spark-master)" && exit 1)
docker exec "${SPARK_MASTER_CONTAINER}" test -f /opt/jobs/bronze_to_silver.py || (echo "❌ /opt/jobs/bronze_to_silver.py not found" && exit 1)
docker exec "${SPARK_MASTER_CONTAINER}" test -f /opt/jobs/silver_to_gold_features.py || (echo "❌ /opt/jobs/silver_to_gold_features.py not found" && exit 1)

docker exec "${SPARK_MASTER_CONTAINER}" test -f /opt/trainner/train_entry.py || (echo "❌ /opt/trainner/train_entry.py not found" && exit 1)
docker exec "${SPARK_MASTER_CONTAINER}" test -f /opt/trainner/fraud_detection_training.py || (echo "❌ /opt/trainner/fraud_detection_training.py not found" && exit 1)

test -n "${AWS_ACCESS_KEY_ID:-}" || (echo "❌ AWS_ACCESS_KEY_ID is empty" && exit 1)
test -n "${AWS_SECRET_ACCESS_KEY:-}" || (echo "❌ AWS_SECRET_ACCESS_KEY is empty" && exit 1)

docker exec "${SPARK_MASTER_CONTAINER}" test -f /tmp/log4j2.properties || (echo "❌ /tmp/log4j2.properties missing in spark-master" && exit 1)
docker exec "spark-worker" test -f /tmp/log4j2.properties || (echo "❌ /tmp/log4j2.properties missing in spark-worker" && exit 1)

echo "✅ Environment validated"
""",
    )

    check_bronze_partition = BashOperator(
        task_id="check_bronze_partition",
        bash_command=SPARK_WRAPPER + r"""
export SPARK_SUBMIT_TIMEOUT=300
DS="{{ ds }}"

BUCKET="$(cfg_get minio.bucket)"
BRONZE_PREFIX="$(cfg_get paths.bronze | sed 's#^/##; s#/$##')"
BRONZE_PART_COL="$(cfg_get paths.bronze_partition_col)"

P="s3a://${BUCKET}/${BRONZE_PREFIX}/${BRONZE_PART_COL}=${DS}"
echo "[check_bronze_partition] checking ${P}"

docker exec "${SPARK_MASTER_CONTAINER}" bash -lc "cat >/tmp/_check_bronze_parquet.py <<'PY'
from pyspark.sql import SparkSession
import sys

spark = SparkSession.builder.appName('check_bronze_parquet').getOrCreate()
path = sys.argv[1]

jvm = spark._jvm
hconf = spark._jsc.hadoopConfiguration()

p = jvm.org.apache.hadoop.fs.Path(path)
fs = p.getFileSystem(hconf)

if not fs.exists(p):
    print('❌ NOT_EXISTS', path)
    spark.stop()
    raise SystemExit(2)

it = fs.listFiles(p, True)
has_parquet = False
while it.hasNext():
    st = it.next()
    name = st.getPath().getName()
    if name.endswith('.parquet'):
        has_parquet = True
        break

print('HAS_PARQUET=', has_parquet, 'PATH=', path)
spark.stop()

if not has_parquet:
    raise SystemExit(2)
PY"

spark_submit /tmp/_check_bronze_parquet.py "${P}"
echo "[check_bronze_partition] ✅ OK"
""",
        execution_timeout=timedelta(minutes=10),
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

    silver_to_gold_features = BashOperator(
        task_id="silver_to_gold_features",
        bash_command=SPARK_WRAPPER + r"""
export SPARK_SUBMIT_TIMEOUT=900
spark_submit /opt/jobs/silver_to_gold_features.py \
  --config /opt/config.yaml \
  --date {{ ds }} \
  --window_days 7
""",
        execution_timeout=timedelta(minutes=45),
    )

    # ✅ validate theo đúng train window (7 ngày), không chỉ mỗi ds
    validate_gold_schema = BashOperator(
        task_id="validate_gold_schema",
        bash_command=SPARK_WRAPPER + r"""
export SPARK_SUBMIT_TIMEOUT=600
DS="{{ ds }}"
WINDOW_DAYS="{{ params.window_days }}"

BUCKET="$(cfg_get minio.bucket)"
GOLD_PREFIX="$(cfg_get paths.gold_features | sed 's#^/##; s#/$##')"
PART_COL="$(cfg_get paths.gold_partition_col)"
GOLD_ROOT="s3a://${BUCKET}/${GOLD_PREFIX}"

echo "[validate_gold_schema] root=${GOLD_ROOT} part_col=${PART_COL} ds=${DS} window_days=${WINDOW_DAYS}"

docker exec "${SPARK_MASTER_CONTAINER}" bash -lc "cat >/tmp/_validate_gold_window.py <<'PY'
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import sys

spark = SparkSession.builder.appName('validate_gold_window').getOrCreate()

gold_root = sys.argv[1]
part_col  = sys.argv[2]
ds        = sys.argv[3]
window_days = int(sys.argv[4])

end = datetime.strptime(ds, '%Y-%m-%d').date()
start = end - timedelta(days=window_days - 1)
dates = [(start + timedelta(days=i)).isoformat() for i in range(window_days)]

jvm = spark._jvm
hconf = spark._jsc.hadoopConfiguration()

def path_exists(p):
    P = jvm.org.apache.hadoop.fs.Path(p)
    fs = P.getFileSystem(hconf)
    return bool(fs.exists(P))

parts = [f'{gold_root}/{part_col}={d}' for d in dates]
existing = [p for p in parts if path_exists(p)]
missing  = [p for p in parts if p not in existing]

print('[validate_gold_schema] requested_dates=', dates)
print('[validate_gold_schema] existing_parts=', len(existing))
if missing:
    print('[validate_gold_schema] WARN missing_parts=', missing)

if not existing:
    raise RuntimeError('No gold partitions exist in window; cannot train.')

# đọc basePath để giữ partition col
df = spark.read.option('basePath', gold_root).parquet(*existing)

req = set(['user_id', 'label', 'features', 'event_date'])
cols = set(df.columns)
miss = sorted(list(req - cols))
print('[validate_gold_schema] columns=', df.columns)
df.printSchema()

if miss:
    raise RuntimeError('Missing required columns in gold: ' + str(miss))

if df.limit(1).count() == 0:
    raise RuntimeError('Gold window is empty')

# check quick types (user_id nên là int hoặc cast được)
bad = df.select(F.col('user_id').cast('string').alias('s')) \
        .where(F.regexp_extract(F.col('s'), r'(\\d+)', 1) == '') \
        .limit(1).count()
if bad > 0:
    print('[validate_gold_schema] WARN user_id has non-digit values (will be sanitized in training)')

spark.stop()
print('[validate_gold_schema] ✅ OK')
PY"

spark_submit /tmp/_validate_gold_window.py "${GOLD_ROOT}" "${PART_COL}" "${DS}" "${WINDOW_DAYS}"
""",
        params={"window_days": 7},
        execution_timeout=timedelta(minutes=20),
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=SPARK_WRAPPER + r"""
    export SPARK_SUBMIT_TIMEOUT=1800
    DS="{{ ds }}"
    WINDOW_DAYS="{{ params.window_days }}"

    docker_exec_maybe_timeout "${SPARK_SUBMIT_TIMEOUT}" "${SPARK_MASTER_CONTAINER}" bash -lc "
      set -euo pipefail

      /opt/spark/bin/spark-submit \
        --master local[1] \
        --deploy-mode client \
        --conf spark.ui.enabled=false \
        --conf spark.ui.showConsoleProgress=false \
        --conf spark.dynamicAllocation.enabled=false \
        --conf spark.sql.session.timeZone=UTC \
        --conf spark.sql.files.ignoreMissingFiles=true \
        --conf spark.sql.files.ignoreCorruptFiles=true \
        --conf spark.hadoop.fs.s3a.endpoint='${MINIO_ENDPOINT}' \
        --conf spark.hadoop.fs.s3a.path.style.access=true \
        --conf spark.hadoop.fs.s3a.connection.ssl.enabled=false \
        --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem \
        --conf spark.hadoop.fs.s3.impl=org.apache.hadoop.fs.s3a.S3AFileSystem \
        --conf spark.hadoop.fs.AbstractFileSystem.s3.impl=org.apache.hadoop.fs.s3a.S3A \
        --conf spark.hadoop.fs.s3a.aws.credentials.provider=org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider \
        --conf spark.hadoop.fs.s3a.access.key='${AWS_ACCESS_KEY_ID}' \
        --conf spark.hadoop.fs.s3a.secret.key='${AWS_SECRET_ACCESS_KEY}' \
        --conf 'spark.driver.extraJavaOptions=-Dlog4j2.configurationFile=file:/tmp/log4j2.properties -Dlog4j2.disable.jmx=true' \
        /opt/trainner/train_entry.py \
          --config /opt/config.yaml \
          --ds '${DS}' \
          --window_days '${WINDOW_DAYS}' \
          --model gbt \
          --run_name 'gbt_windowed_${DS}' \
          --register 1
    "
    echo "[train_model] ✅ Done"
    """,
        params={"window_days": 7},
        execution_timeout=timedelta(minutes=35),
    )

    cleanup = BashOperator(
        task_id="cleanup",
        bash_command=SPARK_WRAPPER + r"""
set -euo pipefail
echo "[cleanup] best-effort cleanup (warn-only)"
echo "[cleanup] ✅ done"
exit 0
""",
        trigger_rule=TriggerRule.ALL_DONE,
    )

    validate_env >> check_bronze_partition >> bronze_to_silver >> silver_to_gold_features >> validate_gold_schema >> train_model >> cleanup
