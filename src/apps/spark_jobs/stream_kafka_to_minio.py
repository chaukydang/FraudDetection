#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

import yaml
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def expand_env_vars(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env_vars(v) for v in obj]
    if isinstance(obj, str):
        def repl(m):
            name = m.group(1)
            val = os.getenv(name)
            if val is None:
                raise RuntimeError(f"Missing env var: {name}")
            return val
        return _ENV_PATTERN.sub(repl, obj)
    return obj


def as_bool(x: Any, default: bool = False) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ("1", "true", "yes", "y", "on")


def norm_path(p: str) -> str:
    return p[:-1] if p.endswith("/") else p


def build_s3a_path(bucket: str, prefix: str) -> str:
    prefix = prefix.lstrip("/")
    return f"s3a://{bucket}/{prefix}"


@dataclass
class BronzeJobConfig:
    # Kafka
    bootstrap: str
    username: str
    password: str
    security_protocol: str
    sasl_mechanism: str
    topic_transactions: str

    # MinIO/S3A
    s3_endpoint: str
    s3_access_key: str
    s3_secret_key: str
    s3_region: str = "us-east-1"
    s3_path_style: bool = True
    s3_ssl_enabled: bool = False

    # Paths
    out_path: str = "s3a://datalake/bronze/transactions"
    checkpoint: str = "/opt/spark-checkpoints/kafka_to_bronze"

    # Stream options
    starting_offsets: str = "latest"
    max_offsets_per_trigger: int = 5000
    trigger: str = "30 seconds"
    fail_on_data_loss: bool = False

    # Reset
    reset: bool = False
    reset_full_output: bool = False

    # Spark
    log_level: str = "WARN"
    session_tz: str = "UTC"

    # Output tuning
    max_records_per_file: int = 200000
    write_partitions: int = 2


def load_cfg(config_path: str) -> BronzeJobConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw = expand_env_vars(raw)

    kafka = raw["kafka"]
    minio = raw["minio"]
    paths = raw["paths"]
    spark = raw.get("spark", {}) or {}

    bucket = minio["bucket"]
    bronze_prefix = paths["bronze"]  # bronze/transactions
    out_path = build_s3a_path(bucket, bronze_prefix)

    checkpoint_root = spark.get("checkpoint_root", "/opt/spark-checkpoints")
    checkpoint = norm_path(checkpoint_root) + "/kafka_to_bronze"

    reset = as_bool(spark.get("reset_bronze"), default=False) or as_bool(os.getenv("RESET_BRONZE"), default=False)
    reset_full_output = as_bool(spark.get("reset_full_output"), default=False) or as_bool(os.getenv("RESET_FULL_OUTPUT"), default=False)

    return BronzeJobConfig(
        bootstrap=kafka["bootstrap_servers"],
        username=kafka["username"],
        password=kafka["password"],
        security_protocol=kafka.get("security", {}).get("protocol", "SASL_SSL"),
        sasl_mechanism=kafka.get("security", {}).get("mechanism", "PLAIN"),
        topic_transactions=kafka.get("topic", {}).get("transactions", "transactions"),

        s3_endpoint=minio["endpoint"],
        s3_access_key=minio["access_key"],
        s3_secret_key=minio["secret_key"],

        out_path=out_path,
        checkpoint=checkpoint,

        starting_offsets=str(spark.get("starting_offsets", "latest")),
        max_offsets_per_trigger=int(spark.get("max_offsets_per_trigger", 5000)),
        trigger=str(spark.get("trigger_bronze_write", "30 seconds")),
        fail_on_data_loss=as_bool(spark.get("fail_on_data_loss", False), default=False),

        reset=reset,
        reset_full_output=reset_full_output,

        log_level=str(spark.get("log_level", "WARN")),
        session_tz=str(spark.get("session_timezone", "UTC")),

        max_records_per_file=int(spark.get("max_records_per_file", 200000)),
        write_partitions=int(spark.get("write_partitions", 2)),
    )


def build_spark(app_name: str, cfg: BronzeJobConfig) -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.adaptive.enabled", "false")  # streaming: tắt AQE
        .config("spark.sql.files.maxRecordsPerFile", str(cfg.max_records_per_file))
        .config("spark.sql.session.timeZone", cfg.session_tz)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel(cfg.log_level)

    hconf = spark._jsc.hadoopConfiguration()
    hconf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    hconf.set("fs.s3a.endpoint", cfg.s3_endpoint)
    hconf.set("fs.s3a.path.style.access", "true" if cfg.s3_path_style else "false")
    hconf.set("fs.s3a.connection.ssl.enabled", "true" if cfg.s3_ssl_enabled else "false")
    hconf.set("fs.s3a.access.key", cfg.s3_access_key)
    hconf.set("fs.s3a.secret.key", cfg.s3_secret_key)
    hconf.set("fs.s3a.endpoint.region", cfg.s3_region)
    hconf.set("fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
    return spark


def hadoop_delete(spark: SparkSession, path_str: str, recursive: bool = True) -> bool:
    jvm = spark._jvm
    hconf = spark._jsc.hadoopConfiguration()
    p = jvm.org.apache.hadoop.fs.Path(path_str)
    fs = p.getFileSystem(hconf)
    if fs.exists(p):
        return bool(fs.delete(p, recursive))
    return False


def reset_state_if_needed(spark: SparkSession, cfg: BronzeJobConfig) -> None:
    if not cfg.reset:
        return

    out_path = norm_path(cfg.out_path)

    print(f"[reset] RESET_BRONZE=true -> delete checkpoint={cfg.checkpoint}", flush=True)
    hadoop_delete(spark, cfg.checkpoint, recursive=True)

    if cfg.reset_full_output:
        print(f"[reset] RESET_FULL_OUTPUT=true -> delete out_path={out_path}", flush=True)
        hadoop_delete(spark, out_path, recursive=True)
    else:
        # ✅ KHÔNG xoá _spark_metadata riêng lẻ (dễ gây lỗi sink)
        print("[reset] keep output data; checkpoint deleted so stream can restart safely", flush=True)


def build_schema() -> StructType:
    return StructType([
        StructField("transaction_id", StringType(), True),
        StructField("user_id", StringType(), True),
        StructField("amount", DoubleType(), True),
        StructField("currency", StringType(), True),
        StructField("event_time", StringType(), True),
        StructField("merchant", StringType(), True),
        StructField("country", StringType(), True),
    ])


def read_kafka(spark: SparkSession, cfg: BronzeJobConfig) -> DataFrame:
    jaas = (
        'org.apache.kafka.common.security.plain.PlainLoginModule required '
        f'username="{cfg.username}" password="{cfg.password}";'
    )

    return (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", cfg.bootstrap)
        .option("subscribe", cfg.topic_transactions)
        .option("startingOffsets", cfg.starting_offsets)
        .option("maxOffsetsPerTrigger", int(cfg.max_offsets_per_trigger))
        .option("failOnDataLoss", "true" if cfg.fail_on_data_loss else "false")
        .option("kafka.security.protocol", cfg.security_protocol)
        .option("kafka.sasl.mechanism", cfg.sasl_mechanism)
        .option("kafka.sasl.jaas.config", jaas)
        .load()
    )


def transform(df_kafka: DataFrame, schema: StructType) -> DataFrame:
    # kafka source có: timestamp, partition, offset
    base = df_kafka.selectExpr(
        "CAST(value AS STRING) AS value_str",
        "timestamp AS kafka_timestamp",
        "partition AS kafka_partition",
        "offset AS kafka_offset"
    )

    df = (
        base
        .select(F.from_json(F.col("value_str"), schema).alias("j"),
                F.col("kafka_timestamp"),
                F.col("kafka_partition"),
                F.col("kafka_offset"))
        .select("j.*", "kafka_timestamp", "kafka_partition", "kafka_offset")
    )

    # ingest
    df = df.withColumn("ingest_ts", F.current_timestamp())
    df = df.withColumn("ingest_date", F.current_date())

    # event_ts: ưu tiên event_time nếu parse được; fallback kafka_timestamp
    # (event_time của bạn đang NULL nên thực tế sẽ fallback)
    df = df.withColumn(
        "event_ts",
        F.coalesce(F.to_timestamp(F.col("event_time").cast("string")), F.col("kafka_timestamp"))
    )
    df = df.withColumn("event_date", F.to_date(F.col("event_ts")))

    # đảm bảo type kafka columns (tránh NullType nếu lỡ missing)
    df = df.withColumn("kafka_partition", F.col("kafka_partition").cast("int"))
    df = df.withColumn("kafka_offset", F.col("kafka_offset").cast("long"))
    return df


def write_bronze(df: DataFrame, cfg: BronzeJobConfig):
    out_path = norm_path(cfg.out_path)

    print(f"[kafka_to_bronze] out_path={out_path}", flush=True)
    print(f"[kafka_to_bronze] checkpoint={cfg.checkpoint}", flush=True)
    print(f"[kafka_to_bronze] trigger={cfg.trigger}", flush=True)
    print("[kafka_to_bronze] partitionBy=ingest_date", flush=True)

    df_out = df.repartition(cfg.write_partitions)

    q = (
        df_out.writeStream
        .outputMode("append")
        .format("parquet")
        .option("checkpointLocation", cfg.checkpoint)
        .option("maxRecordsPerFile", str(cfg.max_records_per_file))
        .partitionBy("ingest_date")  # ✅ chuẩn
        .trigger(processingTime=cfg.trigger)
        .start(out_path)
    )
    return q


def main(config_path: str) -> int:
    cfg = load_cfg(config_path)
    spark = build_spark("fraud_stream_kafka_to_minio", cfg)

    try:
        reset_state_if_needed(spark, cfg)

        schema = build_schema()
        df_kafka = read_kafka(spark, cfg)
        df_out = transform(df_kafka, schema)

        q = write_bronze(df_out, cfg)
        q.awaitTermination()
        return 0
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return 2
    finally:
        try:
            spark.stop()
        except Exception:
            pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    raise SystemExit(main(args.config))
