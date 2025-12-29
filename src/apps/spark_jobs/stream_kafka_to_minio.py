# src/spark_jobs/stream_kafka_to_minio.py

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_timestamp, to_date, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

from config_loader import load_config

TRANSACTION_SCHEMA = StructType([
    StructField("transaction_id", StringType()),
    StructField("user_id", IntegerType()),
    StructField("amount", DoubleType()),
    StructField("currency", StringType()),
    StructField("merchant", StringType()),
    StructField("timestamp", StringType()),
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

    # ====== IMPORTANT: giảm shuffle partitions để giảm CPU/mem ======
    shuffle_parts = int(cfg.get("spark", {}).get("shuffle_partitions", 8))
    spark.conf.set("spark.sql.shuffle.partitions", str(shuffle_parts))

    # Giảm small-files: giới hạn records mỗi file
    # (tùy data, bạn chỉnh 50k-300k)
    max_records_per_file = int(cfg.get("spark", {}).get("max_records_per_file", 200000))
    spark.conf.set("spark.sql.files.maxRecordsPerFile", str(max_records_per_file))

    # ================= Kafka config =================
    kafka_cfg = cfg["kafka"]
    jaas = (
        "org.apache.kafka.common.security.plain.PlainLoginModule required "
        f'username="{kafka_cfg["username"]}" '
        f'password="{kafka_cfg["password"]}";'
    )

    max_offsets = int(cfg.get("spark", {}).get("max_offsets_per_trigger", 500))  # giảm để đỡ spike
    kafka_df = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", kafka_cfg["bootstrap_servers"])
        .option("subscribe", kafka_cfg["topic"]["transactions"])
        .option("startingOffsets", cfg["spark"].get("starting_offsets", "latest"))
        .option("maxOffsetsPerTrigger", max_offsets)
        .option("kafka.security.protocol", kafka_cfg["security"]["protocol"])
        .option("kafka.sasl.mechanism", kafka_cfg["security"]["mechanism"])
        .option("kafka.sasl.jaas.config", jaas)
        .load()
    )

    # raw stream (1 lần đọc kafka)
    raw = (
        kafka_df.select(
            col("value").cast("string").alias("raw_json"),
            col("timestamp").alias("kafka_ts")
        )
        .withColumn("ingest_ts", current_timestamp())
    )

    bronze_path = f"s3a://{cfg['minio']['bucket']}/{cfg['paths']['bronze']}"
    silver_path = f"s3a://{cfg['minio']['bucket']}/{cfg['paths']['silver']}"
    ckpt = f"{cfg['spark']['checkpoint_root']}/kafka_to_bronze_silver"

    watermark_minutes = int(cfg.get("spark", {}).get("watermark_minutes", 10))
    enable_stateful_dedup = bool(cfg.get("spark", {}).get("enable_stateful_dedup", False))

    # số partitions khi ghi để tránh small files (tuỳ cluster nhỏ, để 1-4)
    write_partitions = int(cfg.get("spark", {}).get("write_partitions", 2))

    def write_bronze_silver(batch_df, batch_id: int):
        """
        Chạy theo micro-batch.
        - batch_df là dataframe của raw (raw_json, kafka_ts, ingest_ts) trong batch.
        - Viết Bronze + Silver trong cùng 1 query => đỡ tốn tài nguyên.
        """
        if batch_df.rdd.isEmpty():
            return

        # ---- BRONZE: raw_json 그대로 ----
        (batch_df
         .coalesce(write_partitions)
         .write
         .mode("append")
         .parquet(bronze_path)
        )

        # ---- SILVER: parse + transform ----
        parsed = (
            batch_df.select(
                from_json(col("raw_json"), TRANSACTION_SCHEMA).alias("t"),
                col("kafka_ts"),
                col("ingest_ts")
            )
            .select("t.*", "kafka_ts", "ingest_ts")
            .withColumn("event_time", to_timestamp("timestamp"))
            .withColumn("event_date", to_date(col("event_time")))
        )

        # Stateful dedup rất tốn RAM -> mặc định tắt
        # Nếu muốn giữ dedup xuyên batches, bật enable_stateful_dedup=true,
        # nhưng nhớ giảm watermark + giảm maxOffsetsPerTrigger + tăng RAM container.
        if enable_stateful_dedup:
            parsed = (
                parsed
                .withWatermark("event_time", f"{watermark_minutes} minutes")
                .dropDuplicates(["transaction_id"])
            )
        else:
            # Dedup nhẹ trong batch (không stateful)
            parsed = parsed.dropDuplicates(["transaction_id"])

        (parsed
         .coalesce(write_partitions)
         .write
         .mode("append")
         .partitionBy("event_date")
         .parquet(silver_path)
        )

    trigger = cfg["spark"].get("trigger_bronze_silver", "10 seconds")

    query = (
        raw.writeStream
        .foreachBatch(write_bronze_silver)
        .option("checkpointLocation", ckpt)
        .trigger(processingTime=trigger)
        .queryName("kafka_to_bronze_and_silver")
        .start()
    )

    spark.streams.awaitAnyTermination()

if __name__ == "__main__":
    main()
