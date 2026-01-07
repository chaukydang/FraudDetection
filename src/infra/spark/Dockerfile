# src/infra/spark/Dockerfile
FROM apache/spark:3.5.1-scala2.12-java17-ubuntu

USER root

RUN set -eux; \
  apt-get update; \
  apt-get install -y --no-install-recommends \
    python3 python3-pip python-is-python3 \
    ca-certificates curl \
    libgomp1 \
  ; \
  rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir -U pip setuptools wheel

RUN python3 -m pip install --no-cache-dir 'urllib3>=1.26.18,<2.0.0'

RUN python3 -m pip install --no-cache-dir \
  mlflow==2.16.2 \
  'pyarrow>=14.0.0,<16' \
  pyyaml==6.0.2 \
  python-dotenv==1.0.1 \
  kafka-python==2.0.2 \
  boto3==1.35.36 \
  botocore==1.35.36 \
  pandas==2.2.3 \
  scikit-learn==1.5.2 \
  'numpy>=1.24.0,<2.0.0' \
  'protobuf>=3.20.3,<5' \
  tqdm==4.66.5 \
  xgboost==2.1.1 \
  lightgbm==4.5.0 \
  catboost==1.2.7 \
  imbalanced-learn==0.12.4 \
  psycopg2-binary==2.9.9 \
  && python3 -m pip cache purge

ARG HADOOP_AWS_VERSION=3.3.4
ARG AWS_BUNDLE_VERSION=1.12.262

RUN set -eux; \
  curl -fL -o /opt/spark/jars/hadoop-aws-${HADOOP_AWS_VERSION}.jar \
    https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/${HADOOP_AWS_VERSION}/hadoop-aws-${HADOOP_AWS_VERSION}.jar; \
  curl -fL -o /opt/spark/jars/aws-java-sdk-bundle-${AWS_BUNDLE_VERSION}.jar \
    https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/${AWS_BUNDLE_VERSION}/aws-java-sdk-bundle-${AWS_BUNDLE_VERSION}.jar

ENV PYTHONUNBUFFERED=1

USER spark
