from datetime import datetime, timedelta, timezone
import json
import os
import random
import signal
import time
from typing import Optional, Dict, Any

from confluent_kafka import Producer
from dotenv import load_dotenv
import logging
from faker import Faker
from jsonschema import validate as js_validate, ValidationError, FormatChecker

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path="/app/.env")

fake = Faker()

TRANSACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "transaction_id": {"type": "string"},
        "user_id": {"type": "number", "minimum": 1000, "maximum": 9999},
        "amount": {"type": "number", "minimum": 0.01, "maximum": 10000},
        "currency": {"type": "string", "pattern": "^[A-Z]{3}$"},
        "merchant": {"type": "string"},
        "timestamp": {"type": "string", "format": "date-time"},
        "location": {"type": "string", "pattern": "^[A-Z]{2}$"},
        "is_fraud": {"type": "integer", "minimum": 0, "maximum": 1},
    },
    "required": ["transaction_id", "user_id", "amount", "currency", "timestamp", "is_fraud"],
}


class TransactionProducer:
    def __init__(self):
        self.bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self.kafka_username = os.getenv("KAFKA_USERNAME")
        self.kafka_password = os.getenv("KAFKA_PASSWORD")
        self.topic = os.getenv("KAFKA_TOPIC", "transactions")
        self.running = False

        # ---------- runtime knobs (reduce CPU / control throughput) ----------
        self.enable_validate = os.getenv("ENABLE_VALIDATE", "0") == "1"  # default OFF
        self.validate_every_n = int(os.getenv("VALIDATE_EVERY_N", "200"))  # validate 1/N msgs
        self.log_every_n = int(os.getenv("LOG_EVERY_N", "500"))  # log 1/N delivered msgs

        # Batch loop controls (giảm CPU + giảm tốn pin)
        self.batch_messages = int(os.getenv("BATCH_MESSAGES", "500"))
        self.batch_sleep_sec = float(os.getenv("BATCH_SLEEP_SEC", "0.5"))
        self.poll_timeout_sec = float(os.getenv("POLL_TIMEOUT_SEC", "0.2"))

        # Optional: dùng merchant list để giảm CPU (Faker company() khá nặng)
        self.use_fast_merchants = os.getenv("FAST_MERCHANTS", "1") == "1"
        self.merchants = [
            "Amazon", "Walmart", "Target", "Apple", "Netflix",
            "Starbucks", "Costco", "Uber", "Airbnb", "Booking",
            "QuickCash", "GlobalDigital", "FastMoneyX"
        ]

        # ---------- Kafka producer config ----------
        self.producer_config = {
            "bootstrap.servers": self.bootstrap_servers,
            "client.id": "transaction-producer",

            # Snappy/LZ4 nhẹ CPU hơn gzip
            "compression.type": os.getenv("KAFKA_COMPRESSION", "snappy"),

            # Batch tuning (giảm network/CPU)
            "linger.ms": int(os.getenv("KAFKA_LINGER_MS", "20")),
            "batch.size": int(os.getenv("KAFKA_BATCH_SIZE", "65536")),  # bytes

            # Tránh queue phình vô hạn khi downstream chậm
            "queue.buffering.max.kbytes": int(os.getenv("KAFKA_QUEUE_KB", "10240")),  # 10MB
            "message.send.max.retries": int(os.getenv("KAFKA_RETRIES", "3")),
            "retry.backoff.ms": int(os.getenv("KAFKA_RETRY_BACKOFF_MS", "200")),
            
            # Keepalive & Timeout (FIX SSL issues)
            "connections.max.idle.ms": int(os.getenv("KAFKA_CONNECTIONS_MAX_IDLE_MS", "540000")),
            "socket.keepalive.enable": os.getenv("KAFKA_SOCKET_KEEPALIVE", "True") == "True",
            "heartbeat.interval.ms": int(os.getenv("KAFKA_HEARTBEAT_INTERVAL_MS", "3000")),
            "session.timeout.ms": int(os.getenv("KAFKA_SESSION_TIMEOUT_MS", "45000")),
            "max.poll.interval.ms": int(os.getenv("KAFKA_MAX_POLL_INTERVAL_MS", "300000")),
            "retries": int(os.getenv("KAFKA_MAX_RETRIES", "2147483647")),
            "reconnect.backoff.ms": int(os.getenv("KAFKA_RECONNECT_BACKOFF_MS", "50")),
            "reconnect.backoff.max.ms": int(os.getenv("KAFKA_RECONNECT_BACKOFF_MAX_MS", "1000")),
            "request.timeout.ms": int(os.getenv("KAFKA_REQUEST_TIMEOUT_MS", "30000")),
            "delivery.timeout.ms": int(os.getenv("KAFKA_DELIVERY_TIMEOUT_MS", "120000")),
        }

        if self.kafka_username and self.kafka_password:
            # NOTE: nếu broker confluent cloud / SASL_SSL
            self.producer_config.update({
                "security.protocol": os.getenv("KAFKA_SECURITY_PROTOCOL", "SASL_SSL"),
                "sasl.mechanism": os.getenv("KAFKA_SASL_MECHANISM", "PLAIN"),
                "sasl.username": self.kafka_username,
                "sasl.password": self.kafka_password,
            })
        else:
            self.producer_config["security.protocol"] = os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")

        try:
            self.producer = Producer(self.producer_config)
            logger.info("Kafka Producer initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka Producer: {e}")
            raise

        # Fraud simulation params
        self.compromised_users = set(random.sample(range(1000, 9999), k=50))
        self.high_risk_merchants = ["QuickCash", "GlobalDigital", "FastMoneyX"]

        # Counters
        self._delivered_count = 0
        self._produced_count = 0

        # Graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def generate_transaction(self) -> Optional[Dict[str, Any]]:
        merchant = random.choice(self.merchants) if self.use_fast_merchants else fake.company()

        transaction = {
            "transaction_id": fake.uuid4(),
            "user_id": random.randint(1000, 9999),
            "amount": round(fake.pyfloat(min_value=0.01, max_value=10000), 2),
            "currency": "USD",
            "merchant": merchant,
            "timestamp": (
                datetime.now(timezone.utc) + timedelta(seconds=random.randint(-300, 3000))
            ).isoformat(),
            "location": fake.country_code(),
            "is_fraud": 0,
        }

        is_fraud = 0
        amount = transaction["amount"]
        user_id = transaction["user_id"]
        merchant = transaction["merchant"]

        # Account takeover
        if user_id in self.compromised_users and amount > 500:
            if random.random() < 0.3:
                is_fraud = 1
                transaction["amount"] = round(random.uniform(500, 5000), 2)
                transaction["merchant"] = random.choice(self.high_risk_merchants)

        # Card testing
        if not is_fraud and amount < 2.0:
            if user_id % 1000 == 0 and random.random() < 0.25:
                is_fraud = 1
                transaction["amount"] = round(random.uniform(0.01, 2), 2)
                transaction["location"] = "US"

        # Merchant collusion
        if not is_fraud and merchant in self.high_risk_merchants:
            if amount > 3000 and random.random() < 0.15:
                is_fraud = 1
                transaction["amount"] = round(random.uniform(300, 1500), 2)

        # Geographic anomalies
        if not is_fraud:
            if user_id % 500 == 0 and random.random() < 0.1:
                is_fraud = 1
                transaction["location"] = random.choice(["CN", "RU", "GB"])

        # Baseline random fraud
        if not is_fraud and random.random() < 0.002:
            is_fraud = 1
            transaction["amount"] = round(random.uniform(100, 2000), 2)

        # final fraud rate clamp-ish
        transaction["is_fraud"] = is_fraud if random.random() < 0.985 else 0

        if self.validate_transaction(transaction):
            return transaction
        return None

    def validate_transaction(self, transaction: Dict[str, Any]) -> bool:
        # Fast path: tắt validate hoặc validate theo tỉ lệ 1/N để giảm CPU
        if not self.enable_validate:
            return True

        if self.validate_every_n > 1 and (self._produced_count % self.validate_every_n != 0):
            return True

        try:
            js_validate(instance=transaction, schema=TRANSACTION_SCHEMA, format_checker=FormatChecker())
            return True
        except ValidationError as e:
            logger.error(f"Invalid transaction: {e.message}")
            return False

    def delivery_report(self, err, msg):
        if err is not None:
            # Log chi tiết hơn để debug
            logger.error(f"Message delivery failed: {err} | Key: {msg.key()}")
            
            # Nếu là network error, có thể cần reconnect
            if "Disconnected" in str(err) or "SSL" in str(err):
                logger.warning("Network/SSL error detected. Producer will auto-reconnect.")
            return

        self._delivered_count += 1
        if self._delivered_count % self.log_every_n == 0:
            logger.info(f"Delivered {self._delivered_count} msgs. Last: {msg.topic()}[{msg.partition()}]")

    def send_transaction(self) -> bool:
        """
        IMPORTANT:
          - KHÔNG poll() mỗi message (giảm CPU)
          - Nếu queue full -> poll nhẹ rồi skip
        """
        transaction = self.generate_transaction()
        if not transaction:
            return False

        try:
            self.producer.produce(
                self.topic,
                key=transaction["transaction_id"],
                value=json.dumps(transaction),
                callback=self.delivery_report,
            )
            self._produced_count += 1
            return True

        except BufferError:
            # Producer queue đầy: poll để giải phóng callback/queue, rồi thử lại ở batch sau
            self.producer.poll(self.poll_timeout_sec)
            return False

        except Exception as e:
            logger.error(f"Failed to send transaction: {e}")
            return False

    def run_continuous_production(self):
        """
        Batch produce -> poll theo batch -> sleep
        => Giảm CPU/battery rõ rệt
        """
        self.running = True
        logger.info("Starting producer for topic %s...", self.topic)

        try:
            while self.running:
                for _ in range(self.batch_messages):
                    if not self.running:
                        break
                    self.send_transaction()

                # Poll có timeout để xử lý delivery callbacks (đỡ busy loop)
                self.producer.poll(self.poll_timeout_sec)

                # Sleep để giới hạn throughput và giảm tải
                if self.batch_sleep_sec > 0:
                    time.sleep(self.batch_sleep_sec)

        finally:
            self.shutdown()

    def shutdown(self, signum=None, frame=None):
        if self.running:
            logger.info("Initiating shutdown...")
            self.running = False
            try:
                self.producer.flush(timeout=30)
            except Exception:
                pass
            logger.info("Producer stopped")


if __name__ == "__main__":
    producer = TransactionProducer()
    producer.run_continuous_production()
