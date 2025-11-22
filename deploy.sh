#!/bin/bash
# deploy.sh - Manual deployment script for testing before Jenkins
# This script simulates what Jenkins will do

set -e

echo "=========================================="
echo "GPU Service Deployment Script"
echo "=========================================="

# Configuration
DOCKER_IMAGE="gpu-service"
DOCKER_TAG="manual"
CONTAINER_NAME="gpu-service-manual"
STUDENT_PORT=${STUDENT_PORT:-8001}
METRICS_PORT=${METRICS_PORT:-8000}

# Step 1: Run CUDA test
echo ""
echo "Step 1: Running CUDA sanity check..."
python3 cuda_test.py || echo "Warning: CUDA test failed"

# Step 2: Build Docker image
echo ""
echo "Step 2: Building Docker image..."
docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} .
docker tag ${DOCKER_IMAGE}:${DOCKER_TAG} ${DOCKER_IMAGE}:latest

# Step 3: Test image
echo ""
echo "Step 3: Testing Docker image..."
docker run --rm ${DOCKER_IMAGE}:${DOCKER_TAG} python --version
docker run --rm ${DOCKER_IMAGE}:${DOCKER_TAG} python -c "import numpy; import fastapi; print('Dependencies OK')"

# Step 4: Stop old container
echo ""
echo "Step 4: Stopping old containers..."
docker stop ${CONTAINER_NAME} 2>/dev/null || true
docker rm ${CONTAINER_NAME} 2>/dev/null || true

# Step 5: Deploy new container
echo ""
echo "Step 5: Deploying container with GPU support..."
docker run -d \
    --name ${CONTAINER_NAME} \
    --gpus all \
    -p ${STUDENT_PORT}:${STUDENT_PORT} \
    -p ${METRICS_PORT}:${METRICS_PORT} \
    --restart unless-stopped \
    ${DOCKER_IMAGE}:${DOCKER_TAG}

# Step 6: Health check
echo ""
echo "Step 6: Waiting for service to start..."
sleep 10

echo "Step 7: Health check..."
curl -f http://localhost:${STUDENT_PORT}/health

echo ""
echo "Step 8: Verify GPU access..."
docker exec ${CONTAINER_NAME} nvidia-smi || echo "Warning: nvidia-smi not available"

echo ""
echo "=========================================="
echo "✅ Deployment completed successfully!"
echo "=========================================="
echo ""
echo "Service URLs:"
echo "  - Health:    http://localhost:${STUDENT_PORT}/health"
echo "  - GPU Info:  http://localhost:${STUDENT_PORT}/gpu-info"
echo "  - API Docs:  http://localhost:${STUDENT_PORT}/docs"
echo ""
echo "View logs:"
echo "  docker logs -f ${CONTAINER_NAME}"
echo ""
