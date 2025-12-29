# /opt/jobs/build_features.py
import argparse
import yaml
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import FeatureHasher, VectorAssembler


def main(cfg_path: str):
    # --------------------------------------------------
    # Load config
    # --------------------------------------------------
    cfg = yaml.safe_load(open(cfg_path, "r"))

    spark = (
        __import__("pyspark")
        .sql.SparkSession
        .builder
        .appName("build_fraud_features")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel(cfg["spark"].get("log_level", "WARN"))

    silver_path = f"s3a://{cfg['minio']['bucket']}/{cfg['paths']['silver']}"
    gold_path   = f"s3a://{cfg['minio']['bucket']}/gold/fraud_features"

    # --------------------------------------------------
    # Read silver
    # --------------------------------------------------
    df = spark.read.parquet(silver_path)

    if df.rdd.isEmpty():
        raise RuntimeError("Silver dataset is EMPTY – cannot build features")

    # --------------------------------------------------
    # 1. Basic cleaning & schema normalization
    # --------------------------------------------------
    df = (
        df
        .withColumn("label", F.col("is_fraud").cast("int"))
        .withColumn("amount", F.col("amount").cast("double"))
        .withColumn("event_time", F.col("event_time").cast("timestamp"))
        .filter(F.col("event_time").isNotNull())
        .filter(F.col("user_id").isNotNull())
        .filter(F.col("amount").isNotNull())
    )

    # 👉 FIX QUAN TRỌNG: đảm bảo event_date luôn tồn tại
    df = df.withColumn("event_date", F.to_date("event_time"))

    # --------------------------------------------------
    # 2. Time features
    # --------------------------------------------------
    df = (
        df
        .withColumn("hour", F.hour("event_time").cast("int"))
        .withColumn("dow",  F.dayofweek("event_time").cast("int"))  # 1=Sun..7=Sat
    )

    # --------------------------------------------------
    # 3. User rolling features (BATCH SAFE)
    # --------------------------------------------------
    # Dùng rowsBetween thay vì rangeBetween (ổn định, rẻ tài nguyên)
    w = (
        Window
        .partitionBy("user_id")
        .orderBy("event_time")
        .rowsBetween(-100, 0)   # last 100 transactions
    )

    df = (
        df
        .withColumn("u_cnt_100", F.count("*").over(w).cast("int"))
        .withColumn("u_avg_100", F.avg("amount").over(w))
        .withColumn("u_std_100", F.stddev_pop("amount").over(w))
        .fillna({"u_std_100": 0.0})
    )

    # --------------------------------------------------
    # 4. Categorical hashing
    # --------------------------------------------------
    hasher = FeatureHasher(
        inputCols=["currency", "merchant", "location"],
        outputCol="cat_features",
        numFeatures=2 ** 18
    )
    df = hasher.transform(df)

    # --------------------------------------------------
    # 5. Assemble final feature vector
    # --------------------------------------------------
    assembler = VectorAssembler(
        inputCols=[
            "amount",
            "hour",
            "dow",
            "u_cnt_100",
            "u_avg_100",
            "u_std_100",
            "cat_features",
        ],
        outputCol="features",
    )

    out = (
        assembler
        .transform(df)
        .select(
            "transaction_id",
            "user_id",
            "event_time",
            "event_date",
            "label",
            "features",
        )
    )

    # --------------------------------------------------
    # 6. Safety checks
    # --------------------------------------------------
    out = out.cache()
    total = out.count()

    if total == 0:
        raise RuntimeError("Gold features is EMPTY after transformation")

    print(f"[DEBUG] Gold feature rows = {total}")
    out.groupBy("label").count().show()

    # --------------------------------------------------
    # 7. Write GOLD (partitioned, optimized)
    # --------------------------------------------------
    (
        out
        .repartition(2, "event_date")
        .write
        .mode("overwrite")
        .partitionBy("event_date")
        .parquet(gold_path)
    )

    print(f"[OK] Gold features written to: {gold_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
