# Use a slim Python image as base
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal OS dependencies required for Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application source
COPY . .

# Provide default config inside the image based on the example
RUN cp config_example.yaml config.yaml

# Default command runs the Prefect flow orchestrator
CMD ["python", "workflow_main.py"]
