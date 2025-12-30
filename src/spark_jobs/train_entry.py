import argparse

import sys
sys.path.append("/opt/trainner")

from fraud_detection_training import FraudDetectionTraining


# from dags.fraud_detection_training import FraudDetectionTraining

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/opt/config.yaml")
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", default="gbt")
    ap.add_argument("--run_name", default=None)
    args = ap.parse_args()

    trainer = FraudDetectionTraining(config_path=args.config)
    trainer.train_model(input_path=args.input, model_type=args.model, run_name=args.run_name)
