# src/spark_jobs/stream_kafka_to_minio.py

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_json, to_timestamp, to_date, current_timestamp
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType
)

from config_loader import load_config


TRANSACTION_SCHEMA = StructType([
    StructField("transaction_id", StringType()),
    StructField("user_id", IntegerType()),
    StructField("amount", DoubleType()),
    StructField("currency", StringType()),
    StructField("merchant", StringType()),
    StructField("timestamp", StringType()),  # event time string from producer
    StructField("location", StringType()),
    StructField("is_fraud", IntegerType()),
])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    spark = (
        SparkSession.builder
        .appName("fraud_streaming")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel(cfg.get("spark", {}).get("log_level", "WARN"))

    # ================= Kafka config =================
    kafka_cfg = cfg["kafka"]
    jaas = (
        "org.apache.kafka.common.security.plain.PlainLoginModule required "
        f'username="{kafka_cfg["username"]}" '
        f'password="{kafka_cfg["password"]}";'
    )

    kafka_df = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", kafka_cfg["bootstrap_servers"])
        .option("subscribe", kafka_cfg["topic"]["transactions"])
        .option("startingOffsets", cfg["spark"]["starting_offsets"])
        .option("kafka.security.protocol", kafka_cfg["security"]["protocol"])
        .option("kafka.sasl.mechanism", kafka_cfg["security"]["mechanism"])
        .option("kafka.sasl.jaas.config", jaas)
        # Optional tuning (safe defaults):
        # .option("failOnDataLoss", "false")
        .load()
    )

    # ================= Bronze (Kafka -> Parquet on MinIO) =================
    bronze_path = f"s3a://{cfg['minio']['bucket']}/{cfg['paths']['bronze']}"
    bronze_ckpt = f"{cfg['spark']['checkpoint_root']}/bronze"

    raw = (
        kafka_df.select(
            col("value").cast("string").alias("raw_json"),
            col("timestamp").alias("kafka_ts")
        )
        .withColumn("ingest_ts", current_timestamp())
    )

    bronze_query = (
        raw.writeStream
        .format("parquet")
        .option("path", bronze_path)
        .option("checkpointLocation", bronze_ckpt)
        .outputMode("append")
        # strongly recommended: avoid “only batch 0 then quiet” confusion
        .trigger(processingTime=cfg["spark"].get("trigger_bronze", "10 seconds"))
        .queryName("bronze_kafka_to_minio")
        .start()
    )

    # ================= Silver (Bronze -> Silver) =================
    # IMPORTANT FIX:
    #   Do NOT read Kafka again for Silver.
    #   Read the bronze parquet stream you just wrote.
    # bronze_stream = spark.readStream.format("parquet").load(bronze_path)

    parsed = raw.select(
        from_json(col("raw_json"), TRANSACTION_SCHEMA).alias("t"),
        col("kafka_ts"),
        col("ingest_ts")
    ).select("t.*", "kafka_ts", "ingest_ts")
    
    watermark_minutes = int(cfg.get("spark", {}).get("watermark_minutes", 10))

    silver = (
        parsed
        .withColumn("event_time", to_timestamp("timestamp"))
        .withColumn("event_date", to_date(col("event_time")))
        .withWatermark("event_time", f"{watermark_minutes} minutes")
        .dropDuplicates(["transaction_id"])
    )

    silver_path = f"s3a://{cfg['minio']['bucket']}/{cfg['paths']['silver']}"
    silver_ckpt = f"{cfg['spark']['checkpoint_root']}/silver"

    silver_query = (
        silver.writeStream
        .format("parquet")
        .option("path", silver_path)
        .option("checkpointLocation", silver_ckpt)
        .partitionBy("event_date")
        .outputMode("append")
        .trigger(processingTime=cfg["spark"].get("trigger_silver", "30 seconds"))
        .queryName("silver_bronze_to_minio")
        .start()
    )

    # Keep driver alive
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
