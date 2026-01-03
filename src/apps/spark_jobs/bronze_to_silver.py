# apps/spark_jobs/bronze_to_silver.py
import argparse
import os
from datetime import datetime, timedelta
import yaml

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_spark(cfg: dict) -> SparkSession:
    spark = SparkSession.builder.appName("bronze_to_silver").getOrCreate()
    spark.sparkContext.setLogLevel(cfg["spark"].get("log_level", "WARN"))

    hconf = spark._jsc.hadoopConfiguration()
    minio = cfg["minio"]
    hconf.set("fs.s3a.endpoint", minio["endpoint"])
    hconf.set("fs.s3a.path.style.access", "true")
    hconf.set("fs.s3a.connection.ssl.enabled", "false")
    hconf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    hconf.set("fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
    hconf.set("fs.s3a.access.key", os.environ.get("AWS_ACCESS_KEY_ID", str(minio["access_key"])))
    hconf.set("fs.s3a.secret.key", os.environ.get("AWS_SECRET_ACCESS_KEY", str(minio["secret_key"])))
    return spark


def daterange(ds: str, window_days: int):
    end = datetime.strptime(ds, "%Y-%m-%d").date()
    start = end - timedelta(days=window_days - 1)
    cur = start
    while cur <= end:
        yield cur.isoformat()
        cur += timedelta(days=1)


def main(config_path: str, ds: str, window_days: int):
    cfg = load_cfg(config_path)
    spark = build_spark(cfg)

    bronze_path = f"s3a://{cfg['minio']['bucket']}/{cfg['paths']['bronze']}"
    silver_path = f"s3a://{cfg['minio']['bucket']}/{cfg['paths']['silver']}"

    # Read only partitions in window (cheap)
    parts = [f"{bronze_path}/event_date={d}" for d in daterange(ds, window_days)]

    df = None
    for p in parts:
        try:
            tmp = spark.read.parquet(p)
            df = tmp if df is None else df.unionByName(tmp, allowMissingColumns=True)
        except Exception:
            continue

    if df is None or df.rdd.isEmpty():
        print(f"[bronze_to_silver] No bronze data in window_days={window_days} for ds={ds}. Exit 0.")
        return

    cleaned = (
        df.dropna(subset=["transaction_id", "user_id", "amount", "event_ts"])
        .withColumn("event_date", to_date(col("event_ts")))
    )

    # Optional dedup
    if bool(cfg["spark"].get("enable_stateful_dedup", False)):
        cleaned = cleaned.dropDuplicates(["transaction_id"])

    # Write silver (dynamic overwrite per partition)
    (
        cleaned.repartition(int(cfg["spark"].get("write_partitions", 2)))
        .write.mode("overwrite")
        .partitionBy("event_date")
        .parquet(silver_path)
    )
    print(f"[bronze_to_silver] Wrote silver to {silver_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--window_days", type=int, default=3)
    args = ap.parse_args()
    main(args.config, args.date, args.window_days)
