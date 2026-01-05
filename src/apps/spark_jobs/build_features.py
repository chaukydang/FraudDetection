# apps/spark_jobs/build_features.py
import argparse
import os
from datetime import datetime, timedelta

import yaml
from pyspark.sql import SparkSession
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


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_spark(cfg: dict) -> SparkSession:
    spark = SparkSession.builder.appName("build_fraud_features").getOrCreate()
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

    silver_root = f"s3a://{cfg['minio']['bucket']}/{cfg['paths']['silver'].strip('/')}"
    gold_root = f"s3a://{cfg['minio']['bucket']}/{cfg['paths']['gold_features'].strip('/')}"

    # Read window partitions from silver
    dfs = []
    for d in daterange(ds, window_days):
        p = f"{silver_root}/event_date={d}"
        try:
            tmp = spark.read.parquet(p)
            # keep original event_date of the transaction day as txn_date
            tmp = tmp.withColumn("txn_date", lit(d))
            dfs.append(tmp)
        except Exception:
            continue

    if not dfs:
        raise RuntimeError(
            f"[build_features] No silver data found in window (days={window_days}, ds={ds}). "
            f"Expected partitions like: {silver_root}/event_date={{date}}. "
            f"Verify that bronze_to_silver task completed successfully."
        )

    df = dfs[0]
    for t in dfs[1:]:
        df = df.unionByName(t, allowMissingColumns=True)

    # ---- Build per-user features AS OF ds ----
    # Features computed over the whole window; output partition is ALWAYS event_date=ds
    has_is_fraud = "is_fraud" in df.columns

    agg_exprs = [
        _count("*").alias("txn_cnt"),
        _sum("amount").alias("amt_sum"),
        _avg("amount").alias("amt_avg"),
        _stddev("amount").alias("amt_std"),
    ]

    # label: any fraud txn in window => label=1 (demo-friendly)
    if has_is_fraud:
        agg_exprs.append(_max(col("is_fraud").cast("double")).alias("label"))
    else:
        agg_exprs.append(lit(0.0).alias("label"))

    feats_num = (
        df.groupBy("user_id")
        .agg(*agg_exprs)
        .fillna(0)
        .withColumn("event_date", lit(ds))  # IMPORTANT: always write ds partition
    )

    feature_cols = ["txn_cnt", "amt_sum", "amt_avg", "amt_std"]
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="keep",
    )
    feats = assembler.transform(feats_num)

    (
        feats.repartition(int(cfg["spark"].get("write_partitions", 2)))
        .write.mode("overwrite")
        .partitionBy("event_date")
        .parquet(gold_root)
    )

    print(f"[build_features] Wrote gold features (label + features) to {gold_root}/event_date={ds}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--window_days", type=int, default=7)
    args = ap.parse_args()
    main(args.config, args.date, args.window_days)
