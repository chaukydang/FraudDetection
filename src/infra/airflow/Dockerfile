FROM apache/airflow:2.10.5

# Chuyển sang root để tạo thư mục và cài package
USER root

# Tạo thư mục models và set quyền cho airflow user
RUN mkdir -p /opt/airflow/models \
    && chown -R airflow: /opt/airflow/models

# Quay lại user airflow
USER airflow

# Copy file requirements
COPY requirements.txt /tmp/requirements.txt

# Cài các Python packages
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Đảm bảo container chạy với user airflow
USER airflow
