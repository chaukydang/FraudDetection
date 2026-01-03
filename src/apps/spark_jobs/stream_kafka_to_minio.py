# apps/spark_jobs/stream_kafka_to_minio.py
import argparse
import json
import os
from datetime import datetime
import yaml

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_timestamp, to_date
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, TimestampType


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_spark(cfg: dict) -> SparkSession:
    spark = (
        SparkSession.builder.appName("fraud_stream_kafka_to_minio")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel(cfg["spark"].get("log_level", "WARN"))

    # S3A config
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


def main(config_path: str):
    cfg = load_cfg(config_path)
    spark = build_spark(cfg)

    kafka_cfg = cfg["kafka"]
    topic = kafka_cfg["topic"]["transactions"]

    # Minimal schema (bạn có thể mở rộng theo dữ liệu thực)
    schema = StructType([
        StructField("transaction_id", StringType(), True),
        StructField("user_id", StringType(), True),
        StructField("merchant_id", StringType(), True),
        StructField("amount", DoubleType(), True),
        StructField("currency", StringType(), True),
        StructField("event_time", StringType(), True),  # ISO string
        StructField("label", LongType(), True),         # 0/1 nếu có
    ])

    jaas = (
        'org.apache.kafka.common.security.plain.PlainLoginModule required '
        f'username="{os.environ.get("KAFKA_USERNAME")}" '
        f'password="{os.environ.get("KAFKA_PASSWORD")}";'
    )

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", os.environ.get("KAFKA_BOOTSTRAP_SERVERS", kafka_cfg["bootstrap_servers"]))
        .option("subscribe", topic)
        .option("startingOffsets", cfg["spark"].get("starting_offsets", "latest"))
        .option("maxOffsetsPerTrigger", int(cfg["spark"].get("max_offsets_per_trigger", 200)))
        .option("kafka.security.protocol", kafka_cfg["security"]["protocol"])
        .option("kafka.sasl.mechanism", kafka_cfg["security"]["mechanism"])
        .option("kafka.sasl.jaas.config", jaas)
        .load()
    )

    parsed = (
        raw.selectExpr("CAST(value AS STRING) AS json_str")
        .select(from_json(col("json_str"), schema).alias("d"))
        .select("d.*")
        .withColumn("event_ts", to_timestamp(col("event_time")))
        .withColumn("event_date", to_date(col("event_ts")))
        .drop("event_time")
    )

    out_path = f"s3a://{cfg['minio']['bucket']}/{cfg['paths']['bronze']}"
    ckpt = os.path.join(cfg["spark"]["checkpoint_root"], "kafka_to_bronze")

    (
        parsed.writeStream
        .format("parquet")
        .option("checkpointLocation", ckpt)
        .partitionBy("event_date")
        .outputMode("append")
        .trigger(processingTime=cfg["spark"].get("trigger_bronze_write", "30 seconds"))
        .start(out_path)
        .awaitTermination()
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
