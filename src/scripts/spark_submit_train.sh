#!/bin/bash
set -euo pipefail

echo "🚀 Submitting Spark XGBoost training job..."

MASTER_CONTAINER="${MASTER_CONTAINER:-spark-master}"
JAR="${JAR:-/opt/jobs/target/scala-2.12/spark_jobs_2.12-0.1.0.jar}"
MAIN_CLASS="${MAIN_CLASS:-TrainFraudXGB}"

# (optional) build jar trước khi submit (nếu bạn mount source vào container build)
# Nếu bạn build jar ở ngoài rồi copy vào /opt/jobs thì comment block này
if [[ "${BUILD_JAR:-0}" == "1" ]]; then
  echo "🔧 Building JAR..."
  docker exec -it "$MASTER_CONTAINER" bash -lc '
    set -e
    cd /opt/jobs
    sbt clean package
  '
  echo "✅ Build done"
fi

docker exec -it "$MASTER_CONTAINER" bash -lc "
set -e

# Ivy cache nên đã set trong spark-defaults.conf: spark.jars.ivy=/opt/ivy-cache
# S3A conf + creds provider cũng nên nằm trong spark-defaults.conf (đọc env từ compose)

if [ ! -f '$JAR' ]; then
  echo '❌ JAR not found: $JAR'
  exit 1
fi

/opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --deploy-mode client \
  --class $MAIN_CLASS \
  --packages ml.dmlc:xgboost4j-spark_2.12:1.7.6,org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262 \
  '$JAR'
"

echo "✅ Done"
