FROM apache/spark:3.5.1

USER root

# Ensure Spark conf dir exists (some images may not ship it pre-created)
RUN mkdir -p /opt/spark/conf /opt/ivy-cache /opt/spark/work /opt/spark-checkpoints \
 && chmod -R 777 /opt/ivy-cache /opt/spark/work /opt/spark-checkpoints || true

# Optional: copy custom jars if you put them into infra/spark/jars/
# (kept safe even if folder is empty)
COPY jars/ /opt/spark/jars/

USER spark