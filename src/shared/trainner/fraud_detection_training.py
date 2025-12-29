import logging
import os
import mlflow
import mlflow.spark
import yaml
import boto3

from dotenv import load_dotenv
from typing import Optional

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as Fsum
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.sql.functions import unix_timestamp
from pyspark.ml.feature import StringIndexer, OneHotEncoder

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler('./fraud_detection_model.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class FraudDetectionTraining:
    def __init__(self, config_path='/opt/config.yaml'):
        os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
        os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = '/usr/bin/git'

        load_dotenv(dotenv_path='/opt/.env')

        self.config = self._load_config(config_path)

        os.environ.update({
            'AWS_ACCESS_KEY_ID': os.getenv('AWS_ACCESS_KEY_ID'),
            'AWS_SECRET_ACCESS_KEY': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'AWS_S3_ENDPOINT_URL': self.config['mlflow']['s3_endpoint_url']
        })

        self._validate_environment()

        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])

    def _load_config(self, config_path: str) -> dict:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info('Configuration loaded successfully')
            return config
        except Exception as e:
            logger.error('Failed to load configuration: %s', str(e))
            raise
    
    def _validate_environment(self):
        required_vars = [
            'KAFKA_BOOTSTRAP_SERVERS',
            'KAFKA_USERNAME',
            'KAFKA_PASSWORD'
        ]
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(
                f'Missing required environment variables: {missing}'
            )

        self._check_minio_connection()

    def _check_minio_connection(self):
        try:
            s3 = boto3.client(
                's3',
                endpoint_url=self.config['mlflow']['s3_endpoint_url'],
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )

            buckets = s3.list_buckets()
            bucket_names = [b['Name'] for b in buckets.get('Buckets', [])]
            logger.info('Minio connection verified. Buckets: %s', bucket_names)
            
            mlflow_bucket = self.config['mlflow'].get('bucket', 'mlflow')

            if mlflow_bucket not in bucket_names:
                s3.create_bucket(Bucket=mlflow_bucket)
                logger.info('Created missing MLFlow bucket: %s', mlflow_bucket)

        except Exception as e:
            logger.error('Minio connection failed: %s', str(e))
            raise
    
    def _create_spark(self, app_name: str = "fraud_train") -> SparkSession:
        """
        Tạo SparkSession cho batch training.
        Nếu bạn chạy bằng spark-submit (recommended) thì các conf s3a endpoint/access/secret
        nên set ở spark-submit (giống step 6). Ở đây chỉ tạo session.
        """
        spark = (SparkSession.builder
                 .appName(app_name)
                 .getOrCreate())
        return spark

    def _infer_feature_cols(self, df):
        drop_cols = {"is_fraud", "label", "transaction_id", "event_ts", "currency", "merchant", "location"}
        numeric_types = {"int", "bigint", "double", "float", "smallint", "tinyint", "decimal", "long"}

        feature_cols = []
        for c, t in df.dtypes:
            if c in drop_cols:
                continue
            if any(t.startswith(nt) for nt in numeric_types):
                feature_cols.append(c)

        # nếu có cột event_ts_unix thì add vào feature
        if "event_ts_unix" in df.columns:
            feature_cols.append("event_ts_unix")

        return feature_cols


    def _compute_scale_pos_weight(self, df) -> float:
        """
        scale_pos_weight = (#negative / #positive)
        giúp XGBoost tốt hơn khi fraud bị lệch lớp.
        """
        agg = df.select(
            Fsum(col("label")).alias("pos"),
            Fsum((1 - col("label"))).alias("neg")
        ).collect()[0]
        pos = float(agg["pos"]) if agg["pos"] is not None else 0.0
        neg = float(agg["neg"]) if agg["neg"] is not None else 0.0
        if pos <= 0:
            return 1.0
        return max(1.0, neg / pos)

    def train_model(
        self,
        input_path: str,
        model_type: str = "gbt",   # "gbt" | "xgb"
        run_name: Optional[str] = None,
        seed: int = 42
    ):
        """
        Train model từ parquet features (step 6) + log lên MLflow.
        """
        spark = self._create_spark(app_name=f"train_{model_type}")

        try:
            df = spark.read.parquet(input_path)
            df = df.dropna(subset=["is_fraud"])
            df = df.withColumn("label", col("is_fraud").cast("double"))

            # timestamp -> numeric
            if "event_ts" in df.columns:
                df = df.withColumn("event_ts_unix", unix_timestamp(col("event_ts")).cast("double"))

            # categorical encoding
            cat_cols = [c for c in ["currency", "merchant", "location"] if c in df.columns]

            indexers = [
                StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
                for c in cat_cols
            ]
            encoders = [
                OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe", handleInvalid="keep")
                for c in cat_cols
            ]

            # numeric feature cols
            numeric_cols = self._infer_feature_cols(df)

            # final features = numeric + ohe vectors
            final_feature_cols = numeric_cols + [f"{c}_ohe" for c in cat_cols]

            assembler = VectorAssembler(
                inputCols=final_feature_cols,
                outputCol="features",
                handleInvalid="keep"
            )

            # split đơn giản (nếu bạn có event_time thì nên split theo time để tránh leakage)
            train_df, test_df = df.randomSplit([0.8, 0.2], seed=seed)

            evaluator_roc = BinaryClassificationEvaluator(
                labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
            )
            evaluator_pr = BinaryClassificationEvaluator(
                labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
            )

            if run_name is None:
                run_name = f"{model_type.upper()}_{os.path.basename(input_path).replace('=', '-')}"  # nhẹ nhàng thôi

            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("input_path", input_path)
                mlflow.log_param("num_features", len(feature_cols))

                if model_type.lower() == "gbt":
                    clf = GBTClassifier(
                        labelCol="label",
                        featuresCol="features",
                        maxIter=120,
                        maxDepth=6,
                        stepSize=0.1,
                        subsamplingRate=0.8,
                        seed=seed
                    )
                    mlflow.log_param("maxIter", 120)
                    mlflow.log_param("maxDepth", 6)
                    mlflow.log_param("stepSize", 0.1)
                    mlflow.log_param("subsamplingRate", 0.8)

                    pipe = Pipeline(stages=indexers + encoders + [assembler, clf])
                    model = pipe.fit(train_df)

                elif model_type.lower() == "xgb":
                    from xgboost.spark import SparkXGBClassifier

                    # chạy stages encode + assembler trước
                    prep = Pipeline(stages=indexers + encoders + [assembler]).fit(df)
                    df2 = prep.transform(df).select("features", "label")

                    train2, test2 = df2.randomSplit([0.8, 0.2], seed=seed)

                    spw = self._compute_scale_pos_weight(train2)
                    mlflow.log_param("scale_pos_weight", spw)

                    model = SparkXGBClassifier(
                        features_col="features",
                        label_col="label",
                        objective="binary:logistic",
                        eval_metric="aucpr",
                        num_round=400,
                        max_depth=6,
                        eta=0.08,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        scale_pos_weight=spw,
                        seed=seed
                    ).fit(train2)

                    test_df = test2


                else:
                    raise ValueError("model_type chỉ nhận 'gbt' hoặc 'xgb'")

                pred = model.transform(test_df)

                auc_roc = float(evaluator_roc.evaluate(pred))
                auc_pr = float(evaluator_pr.evaluate(pred))

                mlflow.log_metric("auc_roc", auc_roc)
                mlflow.log_metric("auc_pr", auc_pr)

                # log Spark model artifact
                mlflow.spark.log_model(model, artifact_path="model")

                logger.info("[OK] Train done. model=%s auc_roc=%.6f auc_pr=%.6f",
                            model_type, auc_roc, auc_pr)

                return {"auc_roc": auc_roc, "auc_pr": auc_pr, "run_name": run_name}

        finally:
            spark.stop()
