FROM python:3.10.16-slim-bookworm

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application
COPY . .

# Expose MLflow port
EXPOSE 5500