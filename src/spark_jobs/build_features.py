import argparse
from datetime import datetime

from pyspark.sql import functions as F
from pyspark.sql import Window


def main(input_path: str, output_base: str, run_date: str):
    spark = (
        __import__("pyspark").sql.SparkSession.builder
        .appName("build_features")
        .getOrCreate()
    )

    # Read silver transactions (parquet)
    df = spark.read.parquet(f"{input_path}/event_date=*")

    # Ensure timestamp is timestamp type
    # (your producer emits ISO string; silver may already cast; we handle both)
    df = df.withColumn(
        "event_ts",
        F.coalesce(
            F.col("timestamp").cast("timestamp"),
            F.to_timestamp("timestamp")
        )
    )

    # Basic time features
    df = (
        df.withColumn("hour", F.hour("event_ts"))
          .withColumn("dayofweek", F.dayofweek("event_ts"))  # 1=Sun..7=Sat
    )

    # Rolling window by user (ordered by event time)
    w_1h = (
        Window.partitionBy("user_id")
        .orderBy(F.col("event_ts").cast("long"))
        .rangeBetween(-3600, 0)
    )
    w_24h = (
        Window.partitionBy("user_id")
        .orderBy(F.col("event_ts").cast("long"))
        .rangeBetween(-86400, 0)
    )

    df_feat = (
        df
        # 1h rolling
        .withColumn("u_txn_cnt_1h", F.count(F.lit(1)).over(w_1h))
        .withColumn("u_amt_sum_1h", F.sum("amount").over(w_1h))
        .withColumn("u_amt_avg_1h", F.avg("amount").over(w_1h))
        # 24h rolling
        .withColumn("u_txn_cnt_24h", F.count(F.lit(1)).over(w_24h))
        .withColumn("u_amt_sum_24h", F.sum("amount").over(w_24h))
        .withColumn("u_amt_avg_24h", F.avg("amount").over(w_24h))
    )

    # Optional: keep only columns needed for training
    cols = [
        "transaction_id",
        "user_id",
        "amount",
        "currency",
        "merchant",
        "location",
        "event_ts",
        "hour",
        "dayofweek",
        "u_txn_cnt_1h",
        "u_amt_sum_1h",
        "u_amt_avg_1h",
        "u_txn_cnt_24h",
        "u_amt_sum_24h",
        "u_amt_avg_24h",
        "is_fraud",
    ]
    df_feat = df_feat.select(*[c for c in cols if c in df_feat.columns])

    out_path = f"{output_base}/date={run_date}"
    (
        df_feat
        .repartition(1)   # demo/local cho dễ nhìn file; prod thì bỏ
        .write
        .mode("overwrite")
        .parquet(out_path)
    )

    print(f"[OK] Wrote features to: {out_path}")
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--date", default=datetime.utcnow().strftime("%Y-%m-%d"))
    args = parser.parse_args()

    main(args.input, args.output, args.date)
