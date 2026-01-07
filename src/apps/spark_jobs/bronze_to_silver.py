# /opt/jobs/bronze_to_silver.py
import argparse
import os
from datetime import datetime, timedelta
import re
import yaml
from typing import List, Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(obj):
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    if isinstance(obj, str):
        def repl(m):
            name = m.group(1)
            return os.getenv(name, m.group(0))
        return _ENV_RE.sub(repl, obj)
    return obj


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return _expand_env(cfg)


def daterange(ds: str, window_days: int) -> List[str]:
    end = datetime.strptime(ds, "%Y-%m-%d").date()
    start = end - timedelta(days=window_days - 1)
    out: List[str] = []
    cur = start
    while cur <= end:
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def build_spark(cfg: dict) -> SparkSession:
    minio = cfg["minio"]
    spark_cfg = cfg.get("spark", {}) or {}
    log_level = spark_cfg.get("log_level", "WARN")
    session_tz = spark_cfg.get("session_timezone", "UTC")

    spark = (
        SparkSession.builder
        .appName("bronze_to_silver")
        .config("spark.sql.session.timeZone", session_tz)
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("spark.sql.files.ignoreMissingFiles", "true")
        .config("spark.sql.files.ignoreCorruptFiles", "true")
        # S3A / MinIO
        .config("spark.hadoop.fs.s3a.endpoint", minio["endpoint"])
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.hadoop.fs.s3a.access.key", os.environ.get("AWS_ACCESS_KEY_ID", str(minio.get("access_key", ""))))
        .config("spark.hadoop.fs.s3a.secret.key", os.environ.get("AWS_SECRET_ACCESS_KEY", str(minio.get("secret_key", ""))))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel(log_level)
    return spark


def _path_exists(spark: SparkSession, path_str: str) -> bool:
    jvm = spark._jvm
    hconf = spark._jsc.hadoopConfiguration()
    p = jvm.org.apache.hadoop.fs.Path(path_str)
    fs = p.getFileSystem(hconf)
    return fs.exists(p)


def _safe_col(df: DataFrame, name: str, dtype: str):
    if name in df.columns:
        return F.col(name).cast(dtype)
    return F.lit(None).cast(dtype)


def _canonicalize(df: DataFrame, ds: str) -> DataFrame:
    cols = set(df.columns)

    # event_ts: ưu tiên event_ts; else event_time; else kafka_timestamp; else null
    if "event_ts" in cols:
        event_ts = F.col("event_ts").cast("timestamp")
    elif "event_time" in cols:
        event_ts = F.to_timestamp(F.col("event_time").cast("string"))
    elif "kafka_timestamp" in cols:
        event_ts = F.col("kafka_timestamp").cast("timestamp")
    else:
        event_ts = F.lit(None).cast("timestamp")

    # ingest_date: nếu bronze thiếu (không nên) thì fallback ds
    if "ingest_date" in cols:
        ingest_date = F.to_date(F.col("ingest_date"))
    else:
        ingest_date = F.to_date(F.lit(ds))

    user_id_raw = _safe_col(df, "user_id", "string")
    user_id_digits = F.regexp_extract(F.trim(user_id_raw), r"(\d+)", 1)
    user_id = F.when(F.length(user_id_digits) == 0, F.lit(None)).otherwise(user_id_digits.cast("int"))

    # label/is_fraud
    fraud_src = (
        F.col("is_fraud") if "is_fraud" in cols
        else (F.col("label") if "label" in cols else F.lit(None))
    )
    is_fraud = fraud_src.cast("double")
    is_fraud = F.when(is_fraud.isNull(), F.lit(0.0)).otherwise(is_fraud)

    out = df.select(
        _safe_col(df, "transaction_id", "string").alias("transaction_id"),
        user_id_raw.alias("user_id_raw"),
        user_id.alias("user_id"),
        _safe_col(df, "amount", "double").alias("amount"),
        event_ts.alias("event_ts"),
        F.to_date(event_ts).alias("event_date"),
        _safe_col(df, "currency", "string").alias("currency"),
        _safe_col(df, "merchant", "string").alias("merchant"),
        _safe_col(df, "country", "string").alias("country"),
        ingest_date.alias("ingest_date"),
        is_fraud.alias("is_fraud"),
        # ✅ Kafka lineage (cast để tránh VOID)
        _safe_col(df, "kafka_partition", "int").alias("kafka_partition"),
        _safe_col(df, "kafka_offset", "long").alias("kafka_offset"),
    )
    return out


def main(config_path: str, ds: str, window_days: int):
    cfg = load_cfg(config_path)
    spark = build_spark(cfg)

    bucket = cfg["minio"]["bucket"]
    bronze_root = cfg["paths"]["bronze"].strip("/")
    silver_root = cfg["paths"]["silver"].strip("/")

    bronze_path = f"s3a://{bucket}/{bronze_root}"
    silver_path = f"s3a://{bucket}/{silver_root}"

    quarantine_root = (cfg.get("paths", {}) or {}).get("quarantine", "").strip("/")
    quarantine_path = f"s3a://{bucket}/{quarantine_root}" if quarantine_root else None

    spark_cfg = cfg.get("spark", {}) or {}
    write_parts = int(spark_cfg.get("write_partitions", 2))
    enable_dedup = bool(spark_cfg.get("enable_stateful_dedup", False))

    # ✅ theo chuẩn: input window theo ingest_date
    dates = daterange(ds, window_days)
    candidate_parts = [f"{bronze_path}/ingest_date={d}" for d in dates]
    existing_parts = [p for p in candidate_parts if _path_exists(spark, p)]

    print(f"[bronze_to_silver] ds={ds} window_days={window_days} ingest_parts_found={len(existing_parts)}", flush=True)

    if not existing_parts:
        raise RuntimeError(
            f"[bronze_to_silver] No bronze partitions found for ingest_date in window={dates}. "
            f"Expected {bronze_path}/ingest_date=YYYY-MM-DD"
        )

    df_raw = spark.read.parquet(*existing_parts)
    df = _canonicalize(df_raw, ds)

    good_cond = (
        F.col("transaction_id").isNotNull()
        & F.col("user_id").isNotNull()
        & F.col("amount").isNotNull()
        & F.col("event_ts").isNotNull()
    )

    good = df.filter(good_cond)
    bad = df.filter(~good_cond)

    if enable_dedup:
        good = good.dropDuplicates(["transaction_id"])

    # ✅ silver partition theo event_date (chuẩn analytics/features)
    partition_col = (cfg.get("paths", {}) or {}).get("silver_partition_col", "event_date").strip()
    if partition_col not in ("event_date", "ingest_date"):
        partition_col = "event_date"

    (
        good
        .repartition(write_parts, F.col(partition_col))
        .write
        .mode("overwrite")
        .partitionBy(partition_col)
        .parquet(silver_path)
    )
    print(f"[bronze_to_silver] ✅ silver={silver_path} partition_col={partition_col}", flush=True)

    if quarantine_path:
        (
            bad
            .withColumn("quarantine_ds", F.lit(ds))
            .coalesce(1)
            .write
            .mode("append")
            .partitionBy("quarantine_ds")
            .parquet(f"{quarantine_path}/bronze_to_silver")
        )
        print(f"[bronze_to_silver] 🧯 quarantine={quarantine_path}/bronze_to_silver ds={ds}", flush=True)

    spark.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--window_days", type=int, default=3)
    args = ap.parse_args()
    main(args.config, args.date, args.window_days)
