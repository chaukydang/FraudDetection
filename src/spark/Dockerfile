FROM apache/spark:3.5.1

USER root
COPY jars/*.jar /opt/spark/jars/
RUN pip3 install --no-cache-dir mlflow boto3 pyyaml python-dotenv scikit-learn
USER spark
