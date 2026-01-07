# /opt/trainner/train_entry.py
import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List

import yaml
import mlflow
from pyspark.sql import SparkSession


_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(obj):
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    if isinstance(obj, str):
        def repl(m):
            name = m.group(1)
            return os.getenv(name, m.group(0))
        return _ENV_RE.sub(repl, obj)
    return obj


def load_cfg(path: str) -> Dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return _expand_env(cfg)


def daterange(ds: str, window_days: int) -> List[str]:
    end = datetime.strptime(ds, "%Y-%m-%d").date()
    start = end - timedelta(days=window_days - 1)
    out = []
    cur = start
    while cur <= end:
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def build_spark(cfg: Dict) -> SparkSession:
    minio = cfg["minio"]
    spark_cfg = cfg.get("spark", {}) or {}

    log_level = spark_cfg.get("log_level", "WARN")
    session_tz = spark_cfg.get("session_timezone", "UTC")
    shuffle_partitions = int(spark_cfg.get("shuffle_partitions", 4))

    spark = (
        SparkSession.builder
        .appName("fraud_train_entry")
        .config("spark.sql.session.timeZone", session_tz)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.files.ignoreMissingFiles", "true")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")

        # ✅ Map s3:// to S3A to avoid "No FileSystem for scheme s3"
        .config("spark.hadoop.fs.s3.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.AbstractFileSystem.s3.impl", "org.apache.hadoop.fs.s3a.S3A")
        .config("spark.hadoop.fs.s3n.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.AbstractFileSystem.s3n.impl", "org.apache.hadoop.fs.s3a.S3A")

        # S3A / MinIO
        .config("spark.hadoop.fs.s3a.endpoint", minio["endpoint"])
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.hadoop.fs.s3a.access.key", os.environ.get("AWS_ACCESS_KEY_ID", str(minio.get("access_key", ""))))
        .config("spark.hadoop.fs.s3a.secret.key", os.environ.get("AWS_SECRET_ACCESS_KEY", str(minio.get("secret_key", ""))))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel(log_level)
    return spark


@dataclass
class TrainArgs:
    config: str
    ds: str
    window_days: int
    model: str
    run_name: str
    register: bool


def configure_mlflow(cfg: Dict) -> None:
    ml = cfg.get("mlflow", {}) or {}
    tracking_uri = ml.get("tracking_uri") or os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    exp_name = ml.get("experiment_name", "fraud_detection")
    mlflow.set_experiment(exp_name)

    # ✅ Ensure endpoint for artifact store in MinIO
    s3_endpoint_url = ml.get("s3_endpoint_url") or os.getenv("MLFLOW_S3_ENDPOINT_URL")
    if s3_endpoint_url:
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_endpoint_url

    # ✅ Make sure boto3 has creds (so list_artifacts / log_artifacts works even outside wrapper)
    # Prefer env, fallback to config minio creds
    minio = cfg.get("minio", {}) or {}
    os.environ.setdefault("AWS_ACCESS_KEY_ID", str(minio.get("access_key", "")))
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", str(minio.get("secret_key", "")))
    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    os.environ.setdefault("AWS_REGION", "us-east-1")


def main(args: TrainArgs) -> int:
    cfg = load_cfg(args.config)
    configure_mlflow(cfg)

    spark = build_spark(cfg)

    try:
        from fraud_detection_training import train_and_log

        rc = train_and_log(
            spark=spark,
            cfg=cfg,
            ds=args.ds,
            window_days=args.window_days,
            model_name=args.model,
            run_name=args.run_name,
            register=args.register,
        )
        return rc
    finally:
        try:
            spark.stop()
        except Exception:
            pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ds", required=True)               # yyyy-mm-dd
    ap.add_argument("--window_days", type=int, default=7)
    ap.add_argument("--model", default="gbt", choices=["gbt", "lr", "rf"])
    ap.add_argument("--run_name", default="")
    ap.add_argument("--register", type=int, default=1)  # 1/0
    ns = ap.parse_args()

    run_name = ns.run_name or f"{ns.model}_windowed_{ns.ds}"

    code = main(
        TrainArgs(
            config=ns.config,
            ds=ns.ds,
            window_days=ns.window_days,
            model=ns.model,
            run_name=run_name,
            register=bool(ns.register),
        )
    )
    raise SystemExit(code)
