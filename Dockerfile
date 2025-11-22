# Dockerfile for GPU-Accelerated Matrix Addition Service
# Uses NVIDIA CUDA base image with GPU runtime support

# Use NVIDIA CUDA base image with Python support
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    numpy \
    prometheus_client \
    numba \
    cuda-python \
    python-multipart

# Copy application files
COPY main.py ./
COPY cuda_test.py ./
COPY create_test_matrices.py ./

# Create directory for test matrices if needed
RUN mkdir -p /data

# Expose ports
# 8001: FastAPI service (change STUDENT_PORT in main.py if needed)
# 8000: Prometheus metrics (for Task 5)
EXPOSE 8001 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8001/health')" || exit 1

# Run the application
CMD ["python", "main.py"]
