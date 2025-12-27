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

    spark = SparkSession.builder.appName("fraud_streaming").getOrCreate()

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
        .load()
    )

    raw = kafka_df.select(
        col("value").cast("string").alias("raw_json"),
        col("timestamp").alias("kafka_ts")
    ).withColumn("ingest_ts", current_timestamp())

    # ================= Bronze =================
    bronze_path = f"s3a://{cfg['minio']['bucket']}/{cfg['paths']['bronze']}"
    bronze_ckpt = f"{cfg['spark']['checkpoint_root']}/bronze"

    raw.writeStream \
        .format("parquet") \
        .option("path", bronze_path) \
        .option("checkpointLocation", bronze_ckpt) \
        .outputMode("append") \
        .start()

    # ================= Silver =================
    parsed = raw.select(
        from_json(col("raw_json"), TRANSACTION_SCHEMA).alias("t"),
        col("kafka_ts"),
        col("ingest_ts")
    ).select("t.*", "kafka_ts", "ingest_ts")

    silver = (
        parsed
        .withColumn("event_time", to_timestamp("timestamp"))
        .withColumn("event_date", to_date("event_time"))
        .withWatermark("event_time", f"{cfg['spark']['watermark_minutes']} minutes")
        .dropDuplicates(["transaction_id"])
    )

    silver_path = f"s3a://{cfg['minio']['bucket']}/{cfg['paths']['silver']}"
    silver_ckpt = f"{cfg['spark']['checkpoint_root']}/silver"

    silver.writeStream \
        .format("parquet") \
        .option("path", silver_path) \
        .option("checkpointLocation", silver_ckpt) \
        .partitionBy("event_date") \
        .outputMode("append") \
        .start()

    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
