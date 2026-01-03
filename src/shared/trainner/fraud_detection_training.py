# shared/trainner/fraud_detection_training.py
import logging
import os
import shutil
import tempfile
from typing import Optional, Dict, Any

import boto3
import mlflow
import yaml
from dotenv import load_dotenv

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as Fsum
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class FraudDetectionTraining:
    def __init__(self, config_path: str = "/opt/config.yaml"):
        # Avoid git warnings from MLflow in container
        os.environ["GIT_PYTHON_REFRESH"] = "quiet"
        os.environ["GIT_PYTHON_GIT_EXECUTABLE"] = "/usr/bin/git"

        load_dotenv(dotenv_path="/opt/.env")
        self.config = self._load_config(config_path)

        # MLflow tracking
        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])

        # Force MLflow to use MinIO S3 endpoint
        s3_endpoint = self.config["mlflow"].get("s3_endpoint_url")
        if s3_endpoint:
            os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", s3_endpoint)
            os.environ.setdefault("AWS_DEFAULT_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
            os.environ.setdefault("AWS_REGION", os.getenv("AWS_REGION", "us-east-1"))
            os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")

        self._check_minio_connection()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _check_minio_connection(self) -> None:
        """Ensure the MLflow artifact bucket exists in MinIO."""
        try:
            s3 = boto3.client(
                "s3",
                endpoint_url=self.config["mlflow"]["s3_endpoint_url"],
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            )
            buckets = s3.list_buckets()
            bucket_names = [b["Name"] for b in buckets.get("Buckets", [])]
            mlflow_bucket = self.config["mlflow"].get("bucket", "mlflow")
            if mlflow_bucket not in bucket_names:
                s3.create_bucket(Bucket=mlflow_bucket)
                logger.info("Created missing MLflow bucket: %s", mlflow_bucket)
        except Exception as e:
            logger.error("MinIO connection failed: %s", str(e))
            raise

    def _apply_s3a_conf(self, spark: SparkSession) -> None:
        """Configure S3A for MinIO access."""
        minio = self.config.get("minio", {})
        if not minio:
            return

        hconf = spark._jsc.hadoopConfiguration()
        endpoint = minio.get("endpoint")
        if endpoint:
            hconf.set("fs.s3a.endpoint", str(endpoint))
        hconf.set("fs.s3a.path.style.access", "true")
        hconf.set("fs.s3a.connection.ssl.enabled", "false")
        hconf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        hconf.set("fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")

        access_key = os.getenv("AWS_ACCESS_KEY_ID", str(minio.get("access_key", "")))
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", str(minio.get("secret_key", "")))
        if access_key:
            hconf.set("fs.s3a.access.key", access_key)
        if secret_key:
            hconf.set("fs.s3a.secret.key", secret_key)

        # Reduce cache issues
        hconf.set("fs.file.impl.disable.cache", "true")
        hconf.set("fs.s3a.impl.disable.cache", "true")

    def _create_spark(self, app_name: str = "fraud_train") -> SparkSession:
        spark = SparkSession.builder.appName(app_name).getOrCreate()
        try:
            lvl = self.config.get("spark", {}).get("log_level", "WARN")
            spark.sparkContext.setLogLevel(lvl)
        except Exception:
            pass
        self._apply_s3a_conf(spark)
        return spark

    def _compute_scale_pos_weight(self, df) -> float:
        agg = df.select(
            Fsum(col("label")).alias("pos"),
            Fsum((1 - col("label"))).alias("neg"),
        ).collect()[0]
        pos = float(agg["pos"]) if agg["pos"] is not None else 0.0
        neg = float(agg["neg"]) if agg["neg"] is not None else 0.0
        if pos <= 0:
            return 1.0
        return max(1.0, neg / pos)

    @staticmethod
    def _normalize_model_type(model_type: str) -> str:
        m = (model_type or "gbt").lower().strip()
        if m in ("xgboost", "xgb"):
            return "xgb"
        if m in ("gbt", "gbdt"):
            return "gbt"
        return m

    @staticmethod
    def _save_spark_model_local(model, dst_dir: str) -> None:
        """Save Spark model to local filesystem."""
        model.write().overwrite().save(dst_dir)

    @staticmethod
    def _should_drop_artifact(filename: str) -> bool:
        """Check if artifact should be dropped (markers, crc files, etc)."""
        base = os.path.basename(filename)
        if base.startswith("._"):
            return True
        if base.endswith(".crc"):
            return True
        if base in ("_SUCCESS", "_SUCCESS.crc"):
            return True
        if base.startswith(".spark-staging") or base.startswith("_temporary"):
            return True
        return False

    def _cleanup_artifact_tree(self, root_dir: str) -> None:
        """Remove marker/CRC files to avoid MinIO header parsing errors."""
        removed = 0
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Remove staging dirs
            for d in list(dirnames):
                if d.startswith(".spark-staging") or d.startswith("_temporary"):
                    full = os.path.join(dirpath, d)
                    shutil.rmtree(full, ignore_errors=True)
                    dirnames.remove(d)

            for fn in filenames:
                full = os.path.join(dirpath, fn)

                # Drop known-bad artifacts
                if self._should_drop_artifact(fn):
                    try:
                        os.remove(full)
                        removed += 1
                    except Exception:
                        pass
                    continue

                # Drop empty files (Content-Length: 0)
                try:
                    if os.path.getsize(full) == 0:
                        os.remove(full)
                        removed += 1
                except Exception:
                    pass

        if removed:
            logger.info("[artifact_cleanup] Removed %d marker/crc/empty files", removed)

    def train_model(
        self,
        input_path: str,
        model_type: str = "gbt",
        run_name: Optional[str] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        model_type = self._normalize_model_type(model_type)
        spark = None

        try:
            spark = self._create_spark(app_name=f"train_{model_type}")
            df = spark.read.parquet(input_path)

            if "label" not in df.columns and "is_fraud" in df.columns:
                df = df.withColumn("label", col("is_fraud").cast("double"))

            if "features" not in df.columns:
                raise RuntimeError(
                    "Input dataset must contain 'features' column (vector). "
                    "Use build_features.py first."
                )

            df = df.dropna(subset=["label", "features"])
            train_df, test_df = df.randomSplit([0.8, 0.2], seed=seed)

            evaluator_roc = BinaryClassificationEvaluator(
                labelCol="label",
                rawPredictionCol="rawPrediction",
                metricName="areaUnderROC",
            )
            evaluator_pr = BinaryClassificationEvaluator(
                labelCol="label",
                rawPredictionCol="rawPrediction",
                metricName="areaUnderPR",
            )

            if run_name is None:
                run_name = f"{model_type.upper()}_daily"

            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("input_path", input_path)
                mlflow.log_param("seed", seed)

                if model_type == "gbt":
                    clf = GBTClassifier(
                        labelCol="label",
                        featuresCol="features",
                        maxIter=120,
                        maxDepth=6,
                        stepSize=0.1,
                        subsamplingRate=0.8,
                        seed=seed,
                    )
                    model = clf.fit(train_df)

                elif model_type == "xgb":
                    from xgboost.spark import SparkXGBClassifier

                    spw = self._compute_scale_pos_weight(train_df)
                    mlflow.log_param("scale_pos_weight", spw)

                    xgb = SparkXGBClassifier(
                        features_col="features",
                        label_col="label",
                        prediction_col="prediction",
                        probability_col="probability",
                        raw_prediction_col="rawPrediction",
                        eval_metric="aucpr",
                        num_round=300,
                        max_depth=6,
                        eta=0.08,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        scale_pos_weight=spw,
                        seed=seed,
                    )
                    model = xgb.fit(train_df)
                else:
                    raise ValueError("model_type chỉ nhận 'gbt' hoặc 'xgb/xgboost'")

                pred = model.transform(test_df)
                auc_roc = float(evaluator_roc.evaluate(pred))
                auc_pr = float(evaluator_pr.evaluate(pred))

                mlflow.log_metric("auc_roc", auc_roc)
                mlflow.log_metric("auc_pr", auc_pr)

                # Save model locally, cleanup, then upload
                tmp_root = tempfile.mkdtemp(prefix="mlflow_spark_model_")
                try:
                    local_model_dir = os.path.join(tmp_root, "model")
                    self._save_spark_model_local(model, local_model_dir)

                    # Remove marker/crc/empty files BEFORE upload
                    self._cleanup_artifact_tree(local_model_dir)

                    # Upload artifacts
                    mlflow.log_artifacts(local_model_dir, artifact_path="model")

                    # Log model info
                    info_path = os.path.join(tmp_root, "MODEL_INFO.txt")
                    with open(info_path, "w") as f:
                        f.write("Spark model saved and logged with mlflow.log_artifacts().\n")
                        f.write("To load: download artifacts and use Spark ML .load(path)\n")
                        f.write(f"run_id={run.info.run_id}\n")
                    mlflow.log_artifact(info_path, artifact_path="model")

                finally:
                    shutil.rmtree(tmp_root, ignore_errors=True)

                logger.info(
                    "[OK] Train done. model=%s auc_roc=%.6f auc_pr=%.6f run_id=%s",
                    model_type,
                    auc_roc,
                    auc_pr,
                    run.info.run_id,
                )

                return {
                    "auc_roc": auc_roc,
                    "auc_pr": auc_pr,
                    "run_name": run_name,
                    "run_id": run.info.run_id,
                }

        except Exception as e:
            logger.error("Training failed: %s", e, exc_info=True)
            raise

        finally:
            if spark:
                try:
                    logger.info("Starting Spark cleanup...")

                    # Stop SparkContext first
                    if spark.sparkContext:
                        spark.sparkContext.stop()
                        logger.info("SparkContext stopped")

                    # Stop SparkSession
                    spark.stop()
                    logger.info("SparkSession stopped")

                    # Force garbage collection
                    import gc
                    gc.collect()

                    # Give OS time to close file descriptors
                    import time
                    time.sleep(2)

                    logger.info("Spark cleanup completed successfully")

                except Exception as e:
                    logger.warning("Cleanup error (non-fatal): %s", e)