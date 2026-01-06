# shared/trainner/train_entry.py
import argparse
import logging
import os
import signal
import sys
import time
import threading

from fraud_detection_training import FraudDetectionTraining

MARKER_PATH = "/tmp/train_done.ok"
_SHOULD_STOP = False


def _setup_logging():
    """
    Clean logs for Airflow:
    - Reduce urllib3/botocore spam (MinIO header parsing warnings)
    - Keep our app logs at INFO
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    # Silence noisy libs
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.INFO)  # keep run links etc.


_setup_logging()
logger = logging.getLogger("train_entry")


def _handle_signal(signum, frame):
    global _SHOULD_STOP
    logger.warning("Received signal %s, attempting graceful shutdown...", signum)
    _SHOULD_STOP = True


def _write_marker(exit_code: int) -> None:
    try:
        with open(MARKER_PATH, "w") as f:
            f.write(f"exit_code={int(exit_code)}\n")
        logger.info("Wrote marker %s", MARKER_PATH)
    except Exception as e:
        logger.warning("Failed to write marker: %s", e)


def _shutdown_py4j_callback_server(trainer: FraudDetectionTraining) -> None:
    """
    Best-effort shutdown of Py4J callback server to avoid hanging threads.
    """
    try:
        spark = getattr(trainer, "_spark_for_shutdown", None)
        if not spark:
            return
        sc = spark.sparkContext
        gw = getattr(sc, "_gateway", None)
        if gw:
            try:
                gw.shutdown_callback_server()
                logger.info("Py4J callback server shutdown")
            except Exception:
                pass
    except Exception:
        pass


def _wait_threads(wait_sec: float = 5.0):
    main_thread = threading.current_thread()
    others = [t for t in threading.enumerate() if t != main_thread]
    if not others:
        return []

    deadline = time.time() + wait_sec
    for t in others:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        if t.is_alive():
            t.join(timeout=remaining)

    return [t for t in others if t.is_alive()]


def main():
    # keep logs clean even if wrapper forgot
    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
    os.environ.setdefault("MLFLOW_DISABLE_GIT_METADATA", "true")

    # optional: prevent MLflow from trying to talk to EC2 metadata
    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/opt/config.yaml")
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", default="gbt")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    exit_code = 0
    trainer = None

    try:
        if _SHOULD_STOP:
            raise RuntimeError("Stopped before init (received termination signal)")

        trainer = FraudDetectionTraining(config_path=args.config)

        if _SHOULD_STOP:
            raise RuntimeError("Stopped before training (received termination signal)")

        res = trainer.train_model(
            input_path=args.input,
            model_type=args.model,
            run_name=args.run_name,
            seed=args.seed,
        )
        logger.info("Training result: %s", res)
        logger.info("✅ Training completed successfully")

    except Exception as e:
        # keep stacktrace for debugging (Airflow logs still manageable now that urllib3 spam is gone)
        logger.error("❌ Training failed: %s", e, exc_info=True)
        exit_code = 1

    finally:
        _write_marker(exit_code)

        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        if trainer:
            _shutdown_py4j_callback_server(trainer)

        alive = _wait_threads(wait_sec=5.0)
        if alive:
            logger.warning("Threads still alive: %s", [t.name for t in alive])
            logger.warning("Force hard-exit to prevent hanging...")
            os._exit(exit_code)

        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
