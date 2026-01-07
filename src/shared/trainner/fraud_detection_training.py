# /opt/trainner/fraud_detection_training.py
import json
from datetime import datetime, timedelta
from typing import Dict, List

import mlflow
from mlflow.tracking import MlflowClient

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array


def daterange(ds: str, window_days: int) -> List[str]:
    end = datetime.strptime(ds, "%Y-%m-%d").date()
    start = end - timedelta(days=window_days - 1)
    out: List[str] = []
    cur = start
    while cur <= end:
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def _s3a_path(cfg: Dict, key: str) -> str:
    bucket = cfg["minio"]["bucket"]
    prefix = (cfg["paths"][key] or "").strip("/")
    return f"s3a://{bucket}/{prefix}"


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


def _collect_event_dates_from_bronze_ingest(
    spark: SparkSession,
    cfg: Dict,
    ds: str,
    window_days: int,
) -> List[str]:
    """
    Drive training by ingest window in bronze (same idea as silver_to_gold_features):
    - Read bronze ingest partitions in [ds-window+1 .. ds]
    - Collect distinct event_date appearing (late events included)
    """
    bronze_base = _s3a_path(cfg, "bronze")
    ingest_part_col = (cfg.get("paths", {}) or {}).get("bronze_partition_col", "ingest_date") or "ingest_date"
    ingest_part_col = ingest_part_col.strip() or "ingest_date"

    ingest_dates = daterange(ds, window_days)
    bronze_parts = [f"{bronze_base}/{ingest_part_col}={d}" for d in ingest_dates]
    df_bronze = _read_existing_partitions_with_basepath(spark, bronze_base, bronze_parts)

    if "event_date" not in df_bronze.columns:
        if "event_ts" in df_bronze.columns:
            df_bronze = df_bronze.withColumn("event_date", F.to_date(F.col("event_ts")))
        else:
            raise RuntimeError("Bronze missing both event_date and event_ts; cannot derive event_dates.")

    ev = (
        df_bronze.select(F.col("event_date").cast("date").alias("event_date"))
        .where(F.col("event_date").isNotNull())
        .distinct()
        .collect()
    )
    return sorted({r["event_date"].isoformat() for r in ev})


def load_gold_by_event_dates(spark: SparkSession, cfg: Dict, event_dates: List[str]) -> DataFrame:
    gold_root = _s3a_path(cfg, "gold_features")
    part_col = (cfg.get("paths", {}) or {}).get("gold_partition_col", "event_date").strip() or "event_date"
    if part_col != "event_date":
        part_col = "event_date"

    gold_parts = [f"{gold_root}/{part_col}={d}" for d in event_dates]
    return _read_existing_partitions_with_basepath(spark, gold_root, gold_parts)


def validate_gold(df: DataFrame) -> None:
    required = {"user_id", "event_date", "label", "features"}
    cols = set(df.columns)
    missing = sorted(list(required - cols))
    if missing:
        raise RuntimeError(f"Missing required columns in gold: {missing}")

    if df.limit(1).count() == 0:
        raise RuntimeError("Gold window is empty")

    bad_label = df.where(~F.col("label").isin([0, 1, 0.0, 1.0])).limit(1).count()
    if bad_label > 0:
        raise RuntimeError("Label contains values outside {0,1}")

    null_feat = df.where(F.col("features").isNull()).limit(1).count()
    if null_feat > 0:
        raise RuntimeError("Found NULL features")


def prepare_training_frame(df_gold: DataFrame) -> DataFrame:
    # sanitize user_id (handles schema drift int/string)
    user_id_str = F.col("user_id").cast("string")
    user_id_digits = F.regexp_extract(F.trim(user_id_str), r"(\d+)", 1)
    user_id_int = F.when(F.length(user_id_digits) == 0, F.lit(None)).otherwise(user_id_digits.cast("int"))

    return df_gold.select(
        user_id_int.alias("user_id"),
        F.col("event_date").cast("string").alias("event_date"),
        F.col("features"),
        F.col("label").cast(IntegerType()).alias("label"),
    )


def build_model(model_name: str):
    if model_name == "gbt":
        return GBTClassifier(
            labelCol="label",
            featuresCol="features",
            maxIter=50,
            maxDepth=5,
            stepSize=0.1,
            subsamplingRate=0.8,
            seed=42,
        )
    if model_name == "lr":
        return LogisticRegression(
            labelCol="label",
            featuresCol="features",
            maxIter=50,
            regParam=0.01,
            elasticNetParam=0.0,
        )
    if model_name == "rf":
        return RandomForestClassifier(
            labelCol="label",
            featuresCol="features",
            numTrees=200,
            maxDepth=10,
            featureSubsetStrategy="auto",
            seed=42,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def evaluate(pred: DataFrame) -> Dict:
    auc = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    ).evaluate(pred)

    aupr = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR",
    ).evaluate(pred)

    with_p = pred.withColumn("p1", vector_to_array("probability")[1])
    cm = (
        with_p.select(
            F.col("label").alias("y"),
            (F.col("p1") >= F.lit(0.5)).cast(IntegerType()).alias("yhat"),
        )
        .groupBy("y", "yhat")
        .count()
        .collect()
    )
    cm_map = {f"y={r['y']},yhat={r['yhat']}": int(r["count"]) for r in cm}
    return {"auc": float(auc), "aupr": float(aupr), "confusion": cm_map}


def log_dataset_stats(df: DataFrame) -> Dict:
    dist = df.groupBy("label").count().collect()
    dist_map = {str(r["label"]): int(r["count"]) for r in dist}
    n = int(df.count())
    return {"n_rows": n, "label_dist": dist_map}


def maybe_register_model(cfg: Dict, run_id: str, artifact_path: str) -> None:
    ml_cfg = cfg.get("mlflow", {}) or {}
    registered_name = ml_cfg.get("registered_model_name")
    if not registered_name:
        return

    client = MlflowClient()
    model_uri = f"runs:/{run_id}/{artifact_path}"
    mv = mlflow.register_model(model_uri=model_uri, name=registered_name)

    try:
        client.set_model_version_tag(registered_name, mv.version, "source", "airflow_daily")
    except Exception:
        pass


def train_and_log(
    spark: SparkSession,
    cfg: Dict,
    ds: str,
    window_days: int,
    model_name: str,
    run_name: str,
    register: bool,
) -> int:
    # ✅ align window to bronze ingest -> event_dates (late events included)
    event_dates = _collect_event_dates_from_bronze_ingest(spark, cfg, ds, window_days)
    if not event_dates:
        raise RuntimeError(f"No event_date found in bronze ingest window for ds={ds} window_days={window_days}")

    print(f"[train] ds={ds} window_days={window_days} -> event_dates_used={event_dates}", flush=True)

    df_gold = load_gold_by_event_dates(spark, cfg, event_dates)
    df = prepare_training_frame(df_gold).cache()
    validate_gold(df)

    stats = log_dataset_stats(df)

    train_df, val_df = df.randomSplit([0.8, 0.2], seed=42)

    clf = build_model(model_name)
    pipeline = Pipeline(stages=[clf])

    ml_cfg = cfg.get("mlflow", {}) or {}
    exp_name = ml_cfg.get("experiment_name", "fraud_detection")

    tags = {
        "ds": ds,
        "window_days": str(window_days),
        "model": model_name,
        "pipeline": "gold->train",
        "experiment": exp_name,
        "event_dates_used": ",".join(event_dates),
    }

    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        run_id = run.info.run_id

        mlflow.log_params({
            "ds": ds,
            "window_days": window_days,
            "model": model_name,
            "train_ratio": 0.8,
            "spark_app": "fraud_training",
        })

        mlflow.log_metrics({
            "n_rows": stats["n_rows"],
            "n_label_0": stats["label_dist"].get("0", 0),
            "n_label_1": stats["label_dist"].get("1", 0),
        })

        model = pipeline.fit(train_df)
        pred = model.transform(val_df)

        metrics = evaluate(pred)
        mlflow.log_metrics({"auc": metrics["auc"], "aupr": metrics["aupr"]})

        summary = {
            "ds": ds,
            "window_days": window_days,
            "model": model_name,
            "metrics": metrics,
            "dataset": stats,
            "event_dates_used": event_dates,
        }
        tmp = "/tmp/train_summary.json"
        with open(tmp, "w") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(tmp, artifact_path="reports")

        # ✅ IMPORTANT FIX: do NOT "import mlflow.spark" inside function (causes UnboundLocalError)
        artifact_path = "model"
        try:
            from mlflow import spark as mlflow_spark
            mlflow_spark.log_model(model, artifact_path=artifact_path)
        except Exception:
            local_path = "/tmp/spark_model"
            model.write().overwrite().save(local_path)
            mlflow.log_artifacts(local_path, artifact_path=artifact_path)

        if register:
            maybe_register_model(cfg, run_id=run_id, artifact_path=artifact_path)

    df.unpersist()
    return 0
