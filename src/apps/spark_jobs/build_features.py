# apps/spark_jobs/build_features.py
import argparse
import os
from datetime import datetime, timedelta
from typing import List, Tuple

import yaml
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col,
    lit,
    sum as _sum,
    avg as _avg,
    count as _count,
    stddev as _stddev,
    max as _max,
)
from pyspark.ml.feature import VectorAssembler


# ----------------------------
# Utils
# ----------------------------
def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_spark(cfg: dict) -> SparkSession:
    spark = SparkSession.builder.appName("build_fraud_features").getOrCreate()
    spark.sparkContext.setLogLevel(cfg.get("spark", {}).get("log_level", "WARN"))

    hconf = spark._jsc.hadoopConfiguration()
    minio = cfg["minio"]

    hconf.set("fs.s3a.endpoint", str(minio["endpoint"]))
    hconf.set("fs.s3a.path.style.access", "true")
    hconf.set("fs.s3a.connection.ssl.enabled", "false")
    hconf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    hconf.set("fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
    hconf.set("fs.s3a.access.key", os.environ.get("AWS_ACCESS_KEY_ID", str(minio.get("access_key", ""))))
    hconf.set("fs.s3a.secret.key", os.environ.get("AWS_SECRET_ACCESS_KEY", str(minio.get("secret_key", ""))))

    hconf.set("fs.file.impl.disable.cache", "true")
    hconf.set("fs.s3a.impl.disable.cache", "true")

    return spark


def window_dates(ds: str, window_days: int) -> List[str]:
    end = datetime.strptime(ds, "%Y-%m-%d").date()
    start = end - timedelta(days=window_days - 1)
    cur = start
    out = []
    while cur <= end:
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def path_exists(spark: SparkSession, path: str) -> bool:
    try:
        jvm = spark._jvm
        hconf = spark._jsc.hadoopConfiguration()
        p = jvm.org.apache.hadoop.fs.Path(path)
        fs = p.getFileSystem(hconf)
        return bool(fs.exists(p))
    except Exception:
        return False


# ----------------------------
# Main
# ----------------------------
def main(
    config_path: str,
    ds: str,
    window_days: int,
    min_days_required: int,
    write_partitions: int,
):
    cfg = load_cfg(config_path)
    spark = build_spark(cfg)

    bucket = cfg["minio"]["bucket"]
    silver_root = f"s3a://{bucket}/{cfg['paths']['silver'].strip('/')}"
    gold_root = f"s3a://{bucket}/{cfg['paths']['gold_features'].strip('/')}"

    # ---- collect window paths ----
    expected = [f"{silver_root}/event_date={d}" for d in window_dates(ds, window_days)]
    found = [p for p in expected if path_exists(spark, p)]
    missing = [p for p in expected if p not in found]

    print(f"[build_features] window_days={window_days} found={len(found)} missing={len(missing)}")
    if missing:
        print(f"[build_features] missing (up to 20): {missing[:20]}")

    if not found:
        raise RuntimeError(
            f"[build_features] ❌ No silver data found for ds={ds}. "
            f"Expected {silver_root}/event_date=YYYY-MM-DD"
        )

    if len(found) < min_days_required:
        print(
            f"[build_features][WARN] Not enough history days "
            f"(found={len(found)} < required={min_days_required}). "
            f"Proceeding with available data."
        )

    # ---- read + union ----
    dfs = [spark.read.parquet(p) for p in found]
    df = dfs[0]
    for d in dfs[1:]:
        df = df.unionByName(d, allowMissingColumns=True)

    # ---- validate ----
    for c in ["user_id", "amount"]:
        if c not in df.columns:
            raise RuntimeError(f"[build_features] Missing required column '{c}'")

    has_is_fraud = "is_fraud" in df.columns

    # ---- aggregate features (as-of DS) ----
    agg_exprs = [
        _count("*").alias("txn_cnt"),
        _sum(col("amount").cast("double")).alias("amt_sum"),
        _avg(col("amount").cast("double")).alias("amt_avg"),
        _stddev(col("amount").cast("double")).alias("amt_std"),
    ]

    if has_is_fraud:
        agg_exprs.append(_max(col("is_fraud").cast("double")).alias("label"))
    else:
        agg_exprs.append(lit(0.0).alias("label"))

    feats_num = (
        df.groupBy("user_id")
        .agg(*agg_exprs)
        .fillna(0)
        .withColumn("event_date", lit(ds))
    )

    assembler = VectorAssembler(
        inputCols=["txn_cnt", "amt_sum", "amt_avg", "amt_std"],
        outputCol="features",
        handleInvalid="keep",
    )
    feats = assembler.transform(feats_num)

    # ---- stats ----
    total = feats.count()
    pos = feats.filter(col("label") == 1.0).count()
    neg = feats.filter(col("label") == 0.0).count()
    print(f"[build_features] users={total} pos={pos} neg={neg}")

    # ---- write gold ----
    (
        feats.repartition(int(write_partitions))
        .write.mode("overwrite")
        .partitionBy("event_date")
        .parquet(gold_root)
    )

    print(f"[build_features] ✅ Wrote gold features to {gold_root}/event_date={ds}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--window_days", type=int, default=7)
    ap.add_argument("--min_days_required", type=int, default=3)
    ap.add_argument("--write_partitions", type=int, default=2)
    args = ap.parse_args()

    main(
        config_path=args.config,
        ds=args.date,
        window_days=args.window_days,
        min_days_required=args.min_days_required,
        write_partitions=args.write_partitions,
    )
