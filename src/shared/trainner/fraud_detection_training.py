import logging
import os
from typing import Optional, Dict, Any

import boto3
import mlflow
import mlflow.spark
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
        os.environ["GIT_PYTHON_REFRESH"] = "quiet"
        os.environ["GIT_PYTHON_GIT_EXECUTABLE"] = "/usr/bin/git"

        load_dotenv(dotenv_path="/opt/.env")
        self.config = self._load_config(config_path)

        # MLflow
        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])

        # Ensure MLflow S3 endpoint env exists (some MLflow setups rely on env var)
        if self.config["mlflow"].get("s3_endpoint_url") and not os.getenv("MLFLOW_S3_ENDPOINT_URL"):
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = self.config["mlflow"]["s3_endpoint_url"]

        self._check_minio_connection()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _check_minio_connection(self) -> None:
        """
        Ensure the MLflow artifact bucket exists in MinIO.
        """
        try:
            s3 = boto3.client(
                "s3",
                endpoint_url=self.config["mlflow"]["s3_endpoint_url"],
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
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
        """
        Make trainer runnable even outside your Airflow spark_submit wrapper.
        If wrapper already sets these, this is harmless.
        """
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

        # Prefer env creds (Airflow wrapper / .env)
        access_key = os.getenv("AWS_ACCESS_KEY_ID", str(minio.get("access_key", "")))
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", str(minio.get("secret_key", "")))
        if access_key:
            hconf.set("fs.s3a.access.key", access_key)
        if secret_key:
            hconf.set("fs.s3a.secret.key", secret_key)

    def _create_spark(self, app_name: str = "fraud_train") -> SparkSession:
        """
        Key fixes:
        - Use RawLocalFileSystem to avoid generating .crc files when Spark/MLflow writes local tmp
          (this helps reduce MinIO/urllib3 header parsing warnings seen on ._SUCCESS.crc uploads).
        - Optionally disable _SUCCESS marker to reduce extra tiny files in model export dirs.
        """
        builder = (
            SparkSession.builder.appName(app_name)
            # Avoid local .crc files
            .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
            .config("spark.hadoop.fs.file.impl.disable.cache", "true")
            # Optional: avoid _SUCCESS marker files in some outputs
            .config("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
        )

        spark = builder.getOrCreate()

        # optional log level from config
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

    def _resolve_xgb_num_workers(self, spark: SparkSession) -> int:
        """
        Decide how many distributed workers XGBoost should use.

        Priority:
        1) ENV: XGB_NUM_WORKERS
        2) config.yaml: xgb.num_workers
        3) fallback: spark.defaultParallelism (capped >= 1)
        """
        env_val = os.getenv("XGB_NUM_WORKERS")
        if env_val:
            try:
                v = int(env_val)
                return max(1, v)
            except Exception:
                pass

        cfg_val = (self.config.get("xgb") or {}).get("num_workers")
        if cfg_val is not None:
            try:
                v = int(cfg_val)
                return max(1, v)
            except Exception:
                pass

        try:
            v = int(spark.sparkContext.defaultParallelism)
            return max(1, v)
        except Exception:
            return 1

    def train_model(
        self,
        input_path: str,
        model_type: str = "gbt",
        run_name: Optional[str] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        model_type = self._normalize_model_type(model_type)
        spark = self._create_spark(app_name=f"train_{model_type}")

        try:
            df = spark.read.parquet(input_path)

            # Expect GOLD schema: label + features
            if "label" not in df.columns and "is_fraud" in df.columns:
                df = df.withColumn("label", col("is_fraud").cast("double"))

            if "features" not in df.columns:
                raise RuntimeError(
                    "Input dataset must contain 'features' column (vector). Use build_features.py first."
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

            with mlflow.start_run(run_name=run_name):
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

                    num_workers = self._resolve_xgb_num_workers(spark)
                    mlflow.log_param("num_workers", num_workers)

                    # NOTE: SparkXGBClassifier does NOT allow setting custom 'objective'
                    xgb = SparkXGBClassifier(
                        features_col="features",
                        label_col="label",
                        prediction_col="prediction",
                        probability_col="probability",
                        raw_prediction_col="rawPrediction",
                        eval_metric="aucpr",
                        # distributed workers
                        num_workers=num_workers,
                        # training params
                        num_round=300,
                        max_depth=6,
                        eta=0.08,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        scale_pos_weight=spw,
                        seed=seed,
                        # optionally control per-worker threads if needed
                        nthread=1,
                    )
                    model = xgb.fit(train_df)

                else:
                    raise ValueError("model_type chỉ nhận 'gbt' hoặc 'xgb/xgboost'")

                pred = model.transform(test_df)
                auc_roc = float(evaluator_roc.evaluate(pred))
                auc_pr = float(evaluator_pr.evaluate(pred))

                mlflow.log_metric("auc_roc", auc_roc)
                mlflow.log_metric("auc_pr", auc_pr)

                # Logging model to MLflow
                mlflow.spark.log_model(model, artifact_path="model")

                logger.info(
                    "[OK] Train done. model=%s auc_roc=%.6f auc_pr=%.6f",
                    model_type,
                    auc_roc,
                    auc_pr,
                )

                return {"auc_roc": auc_roc, "auc_pr": auc_pr, "run_name": run_name}

        finally:
            spark.stop()
