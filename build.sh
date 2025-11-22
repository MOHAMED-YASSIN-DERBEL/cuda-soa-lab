#!/bin/bash
# build.sh - Build and test Docker image

set -e

echo "========================================"
echo "Building GPU Matrix Service Docker Image"
echo "========================================"

# Build the image
docker build -t gpu-service:latest .

echo ""
echo "✓ Image built successfully!"
echo ""
echo "To run the container:"
echo "  docker run --gpus all -p 8001:8001 gpu-service:latest"
echo ""
echo "Or use docker-compose:"
echo "  docker compose up"
echo ""
