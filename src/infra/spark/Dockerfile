FROM apache/spark:3.5.1-scala2.12-java17-ubuntu

USER root

RUN set -eux; \
  apt-get update; \
  apt-get install -y --no-install-recommends \
    python3 python3-pip python-is-python3 \
    ca-certificates \
    libgomp1 \
  ; \
  rm -rf /var/lib/apt/lists/*

USER spark
