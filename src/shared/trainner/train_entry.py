import argparse
import logging
import os
import sys

from fraud_detection_training import FraudDetectionTraining

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/opt/config.yaml")
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", default="gbt")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    model = (args.model or "gbt").lower().strip()
    if model == "xgboost":
        model = "xgb"

    exit_code = 0
    try:
        trainer = FraudDetectionTraining(config_path=args.config)
        res = trainer.train_model(
            input_path=args.input,
            model_type=model,
            run_name=args.run_name,
            seed=args.seed,
        )

        tracking_uri = None
        try:
            tracking_uri = trainer.config.get("mlflow", {}).get("tracking_uri")
        except Exception:
            tracking_uri = None

        run_id = res.get("run_id")
        if tracking_uri and run_id and tracking_uri.startswith("http"):
            logger.info(
                "🏃 MLflow run: %s/#/experiments/%s/runs/%s",
                tracking_uri.rstrip("/"),
                "1",
                run_id,
            )

        logger.info("Training result: %s", res)
        logger.info("✅ Training completed successfully")

    except Exception as e:
        logger.error("❌ Training failed: %s", e, exc_info=True)
        exit_code = 1

    finally:
        logger.info("Exiting with code %d", exit_code)

        # Flush before hard-exit
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        # HARD EXIT: tránh bị treo do non-daemon threads (PySpark/MLflow/boto3/xgboost...)
        os._exit(exit_code)


if __name__ == "__main__":
    main()
