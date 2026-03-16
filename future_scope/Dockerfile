# TMLR FutureScope - Complete Reproducibility Container
FROM python:3.10-slim

LABEL maintainer="FutureScope TMLR Team"
LABEL description="One-command reproducible benchmarking for TMLR submission"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code
COPY future_scope_fixed.py .
COPY benchmark_tmlr_final.py .
COPY download_real_data.py .
COPY fix_electricity.py .

# Create data directories
RUN mkdir -p data/real figures_tmlr_real

# Download datasets at build time (for faster runtime)
RUN python download_real_data.py && python fix_electricity.py

# Default command: run TMLR benchmark
CMD ["python", "-u", "benchmark_tmlr_final.py"]

# Alternative: Interactive shell
# docker run -it tmlr-futurescope /bin/bash
