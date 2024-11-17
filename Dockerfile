# Use a base image with Python
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install -r requirements.txt \
    && pip install mlflow pytest

# Expose ports for MLflow tracking (if necessary)
EXPOSE 5000

# Default environment variables for MLflow
ENV MLFLOW_TRACKING_URI "http://127.0.0.1:5001"

# Set the entrypoint for training/retraining scripts
CMD ["python", "train.py"]
