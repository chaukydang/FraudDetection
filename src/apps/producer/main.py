# src/apps/producer/main.py
import json
import logging
import os
import random
import signal
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from confluent_kafka import Producer
from dotenv import load_dotenv
from faker import Faker
from jsonschema import validate as js_validate, ValidationError, FormatChecker

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
)
logger = logging.getLogger("transaction-producer")

# -----------------------------------------------------------------------------
# Env
# -----------------------------------------------------------------------------
load_dotenv(dotenv_path=os.getenv("DOTENV_PATH", "/app/.env"))

fake = Faker()

TRANSACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "schema_version": {"type": "integer"},
        "transaction_id": {"type": "string"},
        "user_id": {"type": "integer", "minimum": 1000, "maximum": 9999},
        "amount": {"type": "number", "minimum": 0.01, "maximum": 10000},
        "currency": {"type": "string", "pattern": "^[A-Z]{3}$"},
        "merchant": {"type": "string"},
        "timestamp": {"type": "string", "format": "date-time"},   # event_time
        "producer_ts": {"type": "string", "format": "date-time"}, # emit time
        "source_id": {"type": "string"},
        "location": {"type": "string", "pattern": "^[A-Z]{2}$"},
        "is_fraud": {"type": "integer", "minimum": 0, "maximum": 1},
    },
    "required": [
        "schema_version",
        "transaction_id",
        "user_id",
        "amount",
        "currency",
        "timestamp",
        "producer_ts",
        "source_id",
        "is_fraud",
    ],
}


class TransactionProducer:
    def __init__(self) -> None:
        self.topic = os.getenv("KAFKA_TOPIC", "transactions")
        self.running = False

        self.enable_validate = os.getenv("ENABLE_VALIDATE", "0") == "1"
        self.validate_every_n = int(os.getenv("VALIDATE_EVERY_N", "200"))

        self.batch_messages = int(os.getenv("BATCH_MESSAGES", "400"))
        self.batch_sleep_sec = float(os.getenv("BATCH_SLEEP_SEC", "0.6"))
        self.poll_timeout_sec = float(os.getenv("POLL_TIMEOUT_SEC", "0.2"))

        self.log_every_n = int(os.getenv("LOG_EVERY_N", "2000"))
        self.max_buffer_retries = int(os.getenv("PRODUCER_BUFFER_RETRIES", "5"))

        self.use_fast_merchants = os.getenv("FAST_MERCHANTS", "1") == "1"
        self.merchants = [
            "Amazon", "Walmart", "Target", "Apple", "Netflix",
            "Starbucks", "Costco", "Uber", "Airbnb", "Booking",
            "QuickCash", "GlobalDigital", "FastMoneyX"
        ]

        # Kafka (Confluent Cloud)
        bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "")
        username = os.getenv("KAFKA_USERNAME", "")
        password = os.getenv("KAFKA_PASSWORD", "")

        if not bootstrap:
            raise ValueError("Missing KAFKA_BOOTSTRAP_SERVERS")

        cfg: Dict[str, Any] = {
            "bootstrap.servers": bootstrap,
            "client.id": os.getenv("KAFKA_CLIENT_ID", "transaction-producer"),

            # batching / perf
            "compression.type": os.getenv("KAFKA_COMPRESSION", "snappy"),
            "linger.ms": int(os.getenv("KAFKA_LINGER_MS", "20")),
            "batch.size": int(os.getenv("KAFKA_BATCH_SIZE", "65536")),
            "queue.buffering.max.kbytes": int(os.getenv("KAFKA_QUEUE_KB", "10240")),

            # delivery semantics
            "enable.idempotence": os.getenv("KAFKA_ENABLE_IDEMPOTENCE", "true").lower() == "true",
            "acks": os.getenv("KAFKA_ACKS", "all"),
            "retries": int(os.getenv("KAFKA_RETRIES", "2147483647")),
            "retry.backoff.ms": int(os.getenv("KAFKA_RETRY_BACKOFF_MS", "200")),
            "max.in.flight.requests.per.connection": int(os.getenv("KAFKA_MAX_INFLIGHT", "5")),

            # timeouts
            "request.timeout.ms": int(os.getenv("KAFKA_REQUEST_TIMEOUT_MS", "30000")),
            "delivery.timeout.ms": int(os.getenv("KAFKA_DELIVERY_TIMEOUT_MS", "120000")),

            # keepalive
            "connections.max.idle.ms": int(os.getenv("KAFKA_CONNECTIONS_MAX_IDLE_MS", "540000")),
            "socket.keepalive.enable": os.getenv("KAFKA_SOCKET_KEEPALIVE", "True") == "True",
        }

        # Fix (minimal): idempotence requires max.in.flight <= 5
        if cfg["enable.idempotence"] and cfg["max.in.flight.requests.per.connection"] > 5:
            logger.warning("max.in.flight > 5 with idempotence; forcing to 5 for safety")
            cfg["max.in.flight.requests.per.connection"] = 5

        # Confluent Cloud requires SASL_SSL
        cfg.update({
            "security.protocol": os.getenv("KAFKA_SECURITY_PROTOCOL", "SASL_SSL"),
            "sasl.mechanism": os.getenv("KAFKA_SASL_MECHANISM", "PLAIN"),
            "sasl.username": username,
            "sasl.password": password,
        })

        if not username or not password:
            raise ValueError("Missing KAFKA_USERNAME / KAFKA_PASSWORD for Confluent Cloud")

        self.producer = Producer(cfg)
        logger.info("Kafka Producer initialized. topic=%s bootstrap=%s", self.topic, bootstrap)

        # Fraud simulation params
        self.compromised_users = set(random.sample(range(1000, 9999), k=50))
        self.high_risk_merchants = ["QuickCash", "GlobalDigital", "FastMoneyX"]

        # Counters
        self._produced_count = 0
        self._delivered_count = 0

        # Graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def _make_transaction(self) -> Dict[str, Any]:
        merchant = random.choice(self.merchants) if self.use_fast_merchants else fake.company()

        now = datetime.now(timezone.utc)
        event_time = (now + timedelta(seconds=random.randint(-300, 30))).isoformat()

        tx: Dict[str, Any] = {
            "schema_version": 1,
            "transaction_id": fake.uuid4(),
            "user_id": random.randint(1000, 9999),
            "amount": round(fake.pyfloat(min_value=0.01, max_value=10000), 2),
            "currency": "USD",
            "merchant": merchant,
            "timestamp": event_time,
            "producer_ts": now.isoformat(),
            "source_id": os.getenv("HOSTNAME", "producer"),
            "location": fake.country_code(),
            "is_fraud": 0,
        }

        is_fraud = 0
        amount = tx["amount"]
        user_id = tx["user_id"]

        if user_id in self.compromised_users and amount > 500 and random.random() < 0.3:
            is_fraud = 1
            tx["amount"] = round(random.uniform(500, 5000), 2)
            tx["merchant"] = random.choice(self.high_risk_merchants)

        if not is_fraud and amount < 2.0 and user_id % 1000 == 0 and random.random() < 0.25:
            is_fraud = 1
            tx["amount"] = round(random.uniform(0.01, 2), 2)
            tx["location"] = "US"

        if not is_fraud and tx["merchant"] in self.high_risk_merchants and amount > 3000 and random.random() < 0.15:
            is_fraud = 1
            tx["amount"] = round(random.uniform(300, 1500), 2)

        if not is_fraud and user_id % 500 == 0 and random.random() < 0.1:
            is_fraud = 1
            tx["location"] = random.choice(["CN", "RU", "GB"])

        if not is_fraud and random.random() < 0.002:
            is_fraud = 1
            tx["amount"] = round(random.uniform(100, 2000), 2)

        tx["is_fraud"] = 1 if (is_fraud and random.random() < 0.985) else 0
        return tx

    def _validate(self, tx: Dict[str, Any]) -> bool:
        if not self.enable_validate:
            return True
        if self.validate_every_n > 1 and (self._produced_count % self.validate_every_n != 0):
            return True
        try:
            js_validate(instance=tx, schema=TRANSACTION_SCHEMA, format_checker=FormatChecker())
            return True
        except ValidationError as e:
            logger.error("Invalid transaction: %s", e.message)
            return False

    def generate_transaction(self) -> Optional[Dict[str, Any]]:
        tx = self._make_transaction()
        return tx if self._validate(tx) else None

    def delivery_report(self, err, msg) -> None:
        if err is not None:
            logger.error("Delivery failed: %s topic=%s key=%s", err, msg.topic(), msg.key())
            return

        self._delivered_count += 1
        if self._delivered_count % self.log_every_n == 0:
            logger.info(
                "Delivered=%d produced=%d last=%s[%d] key=%s",
                self._delivered_count,
                self._produced_count,
                msg.topic(),
                msg.partition(),
                msg.key(),
            )

    def _produce_with_backpressure(self, key: bytes, value: bytes) -> bool:
        for attempt in range(self.max_buffer_retries + 1):
            try:
                self.producer.produce(self.topic, key=key, value=value, callback=self.delivery_report)
                return True
            except BufferError:
                self.producer.poll(self.poll_timeout_sec)
                time.sleep(0.05 * (attempt + 1))
                continue
            except Exception as e:
                logger.exception("Produce exception: %s", e)
                return False

        logger.warning("Queue still full after retries; backing off")
        time.sleep(0.2)
        return False

    def send_transaction(self) -> bool:
        tx = self.generate_transaction()
        if not tx:
            return False

        key = str(tx["user_id"]).encode("utf-8")
        value = json.dumps(tx, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

        ok = self._produce_with_backpressure(key, value)
        if ok:
            self._produced_count += 1
        return ok

    def run(self) -> None:
        self.running = True
        logger.info("Starting production loop. topic=%s", self.topic)

        try:
            while self.running:
                for _ in range(self.batch_messages):
                    if not self.running:
                        break
                    self.send_transaction()

                self.producer.poll(self.poll_timeout_sec)

                if self.batch_sleep_sec > 0:
                    time.sleep(self.batch_sleep_sec)

        finally:
            self.shutdown()

    def shutdown(self, signum=None, frame=None) -> None:
        if not self.running:
            return
        logger.info("Shutting down...")
        self.running = False
        try:
            remaining = self.producer.flush(timeout=30)
            if remaining:
                logger.warning("Flush timeout; remaining messages=%s", remaining)
        except Exception:
            pass
        logger.info("Producer stopped.")


if __name__ == "__main__":
    TransactionProducer().run()
