import os
import re
import yaml
from typing import Any, Dict

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")

def _resolve_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(m):
            k = m.group(1)
            return os.getenv(k, m.group(0))
        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = _resolve_env(cfg)

    # Optional: overlay common env vars if present
    if os.getenv("S3_ENDPOINT"):
        cfg.setdefault("minio", {})["endpoint"] = os.getenv("S3_ENDPOINT")
    if os.getenv("AWS_ACCESS_KEY_ID"):
        cfg.setdefault("minio", {})["access_key"] = os.getenv("AWS_ACCESS_KEY_ID")
    if os.getenv("AWS_SECRET_ACCESS_KEY"):
        cfg.setdefault("minio", {})["secret_key"] = os.getenv("AWS_SECRET_ACCESS_KEY")

    if os.getenv("KAFKA_BOOTSTRAP_SERVERS"):
        cfg.setdefault("kafka", {})["bootstrap_servers"] = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
    if os.getenv("KAFKA_USERNAME"):
        cfg.setdefault("kafka", {})["username"] = os.getenv("KAFKA_USERNAME")
    if os.getenv("KAFKA_PASSWORD"):
        cfg.setdefault("kafka", {})["password"] = os.getenv("KAFKA_PASSWORD")

    return cfg
