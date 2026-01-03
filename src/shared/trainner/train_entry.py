import argparse
import logging

from fraud_detection_training import FraudDetectionTraining

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/opt/config.yaml")
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", default="gbt")  # gbt | xgb | xgboost
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model = (args.model or "gbt").lower().strip()
    if model == "xgboost":
        model = "xgb"

    trainer = FraudDetectionTraining(config_path=args.config)
    res = trainer.train_model(
        input_path=args.input,
        model_type=model,
        run_name=args.run_name,
        seed=args.seed,
    )
    logger.info("Training result: %s", res)


if __name__ == "__main__":
    main()
