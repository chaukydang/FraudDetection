#!/usr/bin/env python3
# /opt/jobs/silver_to_gold_features.py

import argparse
import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List

import yaml
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(obj: Any) -> Any:
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


def load_cfg(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
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
        .appName("silver_to_gold_features")
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
    return bool(fs.exists(p))


def _read_existing_partitions_with_basepath(
    spark: SparkSession,
    base_path: str,
    partition_paths: List[str],
) -> DataFrame:
    existing = [p for p in partition_paths if _path_exists(spark, p)]
    if not existing:
        raise RuntimeError(f"No existing partitions found under basePath={base_path} for given paths.")
    return spark.read.option("basePath", base_path).parquet(*existing)


def _collect_event_dates_from_bronze(
    spark: SparkSession,
    bronze_base: str,
    ingest_dates: List[str],
) -> List[str]:
    bronze_parts = [f"{bronze_base}/ingest_date={d}" for d in ingest_dates]
    df_bronze = _read_existing_partitions_with_basepath(spark, bronze_base, bronze_parts)

    if "event_date" not in df_bronze.columns:
        raise RuntimeError("Bronze data missing required column: event_date")

    ev = (
        df_bronze
        .select(F.col("event_date").cast("date").alias("event_date"))
        .where(F.col("event_date").isNotNull())
        .distinct()
        .collect()
    )
    return sorted({r["event_date"].isoformat() for r in ev})


def main(config_path: str, ds: str, window_days: int):
    cfg = load_cfg(config_path)
    spark = build_spark(cfg)

    bucket = cfg["minio"]["bucket"]
    paths = cfg.get("paths", {}) or {}

    bronze_root = paths["bronze"].strip("/")
    silver_root = paths["silver"].strip("/")
    gold_root = paths["gold_features"].strip("/")

    bronze_base = f"s3a://{bucket}/{bronze_root}"
    silver_base = f"s3a://{bucket}/{silver_root}"
    gold_base = f"s3a://{bucket}/{gold_root}"

    spark_cfg = cfg.get("spark", {}) or {}
    write_parts = int(spark_cfg.get("write_partitions", 2))

    ingest_dates = daterange(ds, window_days)
    event_dates = _collect_event_dates_from_bronze(spark, bronze_base, ingest_dates)

    if not event_dates:
        raise RuntimeError(f"[silver_to_gold_features] No event_date found in bronze for ingest window={ingest_dates}")

    print(
        f"[silver_to_gold_features] ds={ds} window_days={window_days} "
        f"ingest_dates={ingest_dates} -> event_dates_found={event_dates}",
        flush=True,
    )

    silver_parts = [f"{silver_base}/event_date={d}" for d in event_dates]
    existing_silver = [p for p in silver_parts if _path_exists(spark, p)]
    missing_silver = [p for p in silver_parts if p not in existing_silver]
    if missing_silver:
        print(f"[silver_to_gold_features] WARN missing_silver_parts={missing_silver}", flush=True)
    if not existing_silver:
        raise RuntimeError("[silver_to_gold_features] No silver partitions exist for derived event_dates")

    df_silver = spark.read.option("basePath", silver_base).parquet(*existing_silver)

    # ensure event_date exists (partition column should exist, but keep safe)
    if "event_date" not in df_silver.columns and "event_ts" in df_silver.columns:
        df_silver = df_silver.withColumn("event_date", F.to_date(F.col("event_ts")))

    # ---- FIX DỨT ĐIỂM: label_src luôn tồn tại, không phụ thuộc is_fraud có/không
    label_src = F.coalesce(
        F.col("is_fraud") if "is_fraud" in df_silver.columns else F.lit(None),
        F.col("label") if "label" in df_silver.columns else F.lit(None),
        F.lit(0.0),
    ).cast("double")

    # user_id: cast an toàn để tránh schema drift (string/binary/int)
    user_id_str = F.col("user_id").cast("string") if "user_id" in df_silver.columns else F.lit(None).cast("string")
    user_id_digits = F.regexp_extract(F.trim(user_id_str), r"(\d+)", 1)
    user_id_int = F.when(F.length(user_id_digits) == 0, F.lit(None)).otherwise(user_id_digits.cast("int"))

    base = (
        df_silver.select(
            F.col("event_date").cast("date").alias("event_date"),
            user_id_int.alias("user_id"),
            F.col("amount").cast("double").alias("amount"),
            label_src.alias("label_src"),
        )
        .where(F.col("event_date").isNotNull() & F.col("user_id").isNotNull() & F.col("amount").isNotNull())
    )

    agg = (
        base.groupBy("event_date", "user_id")
        .agg(
            F.count("*").alias("txn_cnt"),
            F.sum("amount").alias("amt_sum"),
            F.avg("amount").alias("amt_avg"),
            F.stddev_pop("amount").alias("amt_std"),
            F.max("label_src").alias("label"),
        )
        .fillna({"amt_std": 0.0, "label": 0.0})
    )

    assembler = VectorAssembler(
        inputCols=["txn_cnt", "amt_sum", "amt_avg", "amt_std"],
        outputCol="features",
        handleInvalid="keep",
    )

    out = assembler.transform(agg).select(
        "user_id", "txn_cnt", "amt_sum", "amt_avg", "amt_std", "label", "features", "event_date"
    )

    # write gold partitionBy event_date
    (
        out.repartition(write_parts, F.col("event_date"))
        .write.mode("overwrite")
        .partitionBy("event_date")
        .parquet(gold_base)
    )

    print(
        f"[silver_to_gold_features] ✅ gold={gold_base} updated_event_dates={event_dates}",
        flush=True,
    )

    spark.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--date", required=True, help="Airflow ds (YYYY-MM-DD)")
    ap.add_argument("--window_days", type=int, default=7)
    args = ap.parse_args()
    main(args.config, args.date, args.window_days)
