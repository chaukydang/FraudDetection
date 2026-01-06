# shared/trainner/fraud_detection_training.py
import logging
import os
import shutil
import tempfile
from typing import Dict, Any, Optional

import boto3
import mlflow
import yaml
from dotenv import load_dotenv

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, sum as Fsum

from pyspark.ml.classification import GBTClassifier

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class FraudDetectionTraining:
    """
    BOOTSTRAP TRAINING ONLY

    - Train on ONE partition: event_date=DS
    - NO evaluation
    - NO rolling window
    - NO AUC
    - MLflow run marked as: training_phase=bootstrap
    """

    def __init__(self, config_path: str = "/opt/config.yaml"):
        load_dotenv("/opt/.env")
        self.config = self._load_config(config_path)

        # ---- clean noisy logs (urllib3/botocore/mlflow) ----
        # This prevents MinIO header-parse noise from polluting Airflow logs.
        logging.getLogger("urllib3").setLevel(logging.ERROR)
        logging.getLogger("botocore").setLevel(logging.WARNING)
        logging.getLogger("boto3").setLevel(logging.WARNING)

        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        self._setup_mlflow_s3()

        # ✅ IMPORTANT: ensure the experiment artifact_location is S3 (not local like /app)
        self._ensure_experiment_artifact_location()

        self._spark_for_shutdown: Optional[SparkSession] = None

    # ------------------ setup ------------------

    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _setup_mlflow_s3(self) -> None:
        """
        Ensure MLflow bucket exists in MinIO.
        """
        s3 = boto3.client(
            "s3",
            endpoint_url=self.config["mlflow"]["s3_endpoint_url"],
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        )
        bucket = self.config["mlflow"].get("bucket", "mlflow")
        existing = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
        if bucket not in existing:
            s3.create_bucket(Bucket=bucket)
            logger.info("Created MLflow bucket: %s", bucket)

    def _ensure_experiment_artifact_location(self) -> None:
        """
        ✅ DỨT ĐIỂM lỗi Permission denied '/app'

        Root cause: experiment artifact_location trong MLflow DB có thể đang là local path
        (vd: /app/mlruns/1). Khi đó mlflow client dùng LocalArtifactRepository và sẽ cố
        mkdir /app -> fail trong spark container.

        Fix: tạo experiment với artifact_location = s3://... nếu chưa tồn tại.
        Nếu đã tồn tại mà artifact_location không phải s3:// thì fail rõ ràng để bạn clean DB.
        """
        exp_name = self.config["mlflow"]["experiment_name"]
        artifact_root = self.config["mlflow"]["artifact_location"]  # e.g. s3://mlflow/fraud_detection

        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(exp_name)

        if exp is None:
            logger.info(
                "MLflow experiment '%s' not found. Creating with artifact_location=%s",
                exp_name, artifact_root
            )
            client.create_experiment(name=exp_name, artifact_location=artifact_root)
        else:
            # exp.artifact_location can be like /app/mlruns/1 (BAD) or s3://mlflow/... (GOOD)
            if not (exp.artifact_location or "").startswith("s3://"):
                raise RuntimeError(
                    "❌ MLflow experiment artifact_location is LOCAL, must be S3.\n"
                    f"experiment_name={exp_name}\n"
                    f"artifact_location_in_db={exp.artifact_location}\n"
                    f"expected_s3={artifact_root}\n\n"
                    "➡️ Please CLEAN MLflow DB (delete or update the experiment artifact_location)."
                )

        # Finally set experiment
        mlflow.set_experiment(exp_name)

    def _create_spark(self) -> SparkSession:
        spark = (
            SparkSession.builder
            .appName("fraud_train_bootstrap")
            .config("spark.python.worker.reuse", "false")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel(
            self.config.get("spark", {}).get("log_level", "WARN")
        )

        hconf = spark._jsc.hadoopConfiguration()
        minio = self.config["minio"]

        hconf.set("fs.s3a.endpoint", minio["endpoint"])
        hconf.set("fs.s3a.path.style.access", "true")
        hconf.set("fs.s3a.connection.ssl.enabled", "false")
        hconf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        hconf.set(
            "fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
        hconf.set("fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID"))
        hconf.set("fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY"))

        # optional: reduce noisy checksum behaviours on local FS
        hconf.set("fs.file.checksum.enabled", "false")

        return spark

    # ------------------ helpers ------------------

    @staticmethod
    def _label_stats(df: DataFrame) -> Dict[str, int]:
        agg = df.select(
            Fsum(col("label")).alias("pos"),
            Fsum(1 - col("label")).alias("neg"),
        ).collect()[0]
        pos = int(agg["pos"] or 0)
        neg = int(agg["neg"] or 0)
        return {"pos": pos, "neg": neg, "total": pos + neg}

    @staticmethod
    def _parse_ds(path: str) -> str:
        return path.rsplit("event_date=", 1)[-1].strip("/")

    @staticmethod
    def _remove_crc_files(root_dir: str) -> int:
        """
        Spark/Hadoop local writer often produces *.crc (0 bytes) files.
        Uploading these via MLflow->boto3->MinIO is where you saw urllib3 header-parse spam.
        Removing them before mlflow.log_artifacts() makes logs clean & faster.
        """
        removed = 0
        for r, _, files in os.walk(root_dir):
            for fn in files:
                if fn.endswith(".crc") or fn.endswith("._COPYING_"):
                    p = os.path.join(r, fn)
                    try:
                        os.remove(p)
                        removed += 1
                    except OSError:
                        pass
        return removed

    # ------------------ main ------------------

    def train_model(
        self,
        input_path: str,
        model_type: str = "gbt",
        run_name: Optional[str] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:

        spark = self._create_spark()
        self._spark_for_shutdown = spark

        try:
            df = spark.read.parquet(input_path)

            if "label" not in df.columns and "is_fraud" in df.columns:
                df = df.withColumn("label", col("is_fraud").cast("double"))

            if "features" not in df.columns:
                raise RuntimeError("Missing features column. Run build_features first.")

            df = df.dropna(subset=["label", "features"])

            stats = self._label_stats(df)
            logger.info(
                "Train distribution: pos=%d neg=%d total=%d",
                stats["pos"], stats["neg"], stats["total"]
            )

            ds = self._parse_ds(input_path)

            with mlflow.start_run(run_name=run_name):
                # -------- params --------
                mlflow.log_param("training_phase", "bootstrap")
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("train_date", ds)
                mlflow.log_param("eval_enabled", False)
                mlflow.log_param("seed", seed)
                mlflow.log_param("has_positive_label", stats["pos"] > 0)

                # -------- metrics --------
                mlflow.log_metric("train_total", stats["total"])
                mlflow.log_metric("train_pos", stats["pos"])
                mlflow.log_metric("train_neg", stats["neg"])

                # -------- model --------
                if model_type != "gbt":
                    raise ValueError("Bootstrap only supports GBT for now")

                model = GBTClassifier(
                    labelCol="label",
                    featuresCol="features",
                    maxIter=100,
                    maxDepth=6,
                    subsamplingRate=0.8,
                    seed=seed,
                ).fit(df)

                # -------- artifacts --------
                tmp = tempfile.mkdtemp(prefix="bootstrap_model_")
                try:
                    model_dir = os.path.join(tmp, "model")
                    model.write().overwrite().save(model_dir)

                    # ✅ Clean crc noise BEFORE upload
                    removed = self._remove_crc_files(model_dir)
                    if removed > 0:
                        logger.info("Removed %d *.crc files before MLflow upload", removed)

                    # This MUST go to S3 now, not /app
                    mlflow.log_artifacts(model_dir, artifact_path="model")

                    info = os.path.join(tmp, "MODEL_INFO.txt")
                    with open(info, "w") as f:
                        f.write("training_phase=bootstrap\n")
                        f.write(f"train_date={ds}\n")
                        f.write("evaluation=disabled\n")
                    mlflow.log_artifact(info, artifact_path="model")
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)

                logger.info("✅ Bootstrap training done")
                return {"train_stats": stats}

        finally:
            spark.stop()
