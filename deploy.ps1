# deploy.ps1 - Manual deployment script for Windows
# This script simulates what Jenkins will do

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "GPU Service Deployment Script" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Configuration
$DOCKER_IMAGE = "gpu-service"
$DOCKER_TAG = "manual"
$CONTAINER_NAME = "gpu-service-manual"
$STUDENT_PORT = if ($env:STUDENT_PORT) { $env:STUDENT_PORT } else { "8001" }
$METRICS_PORT = if ($env:METRICS_PORT) { $env:METRICS_PORT } else { "8000" }

# Step 1: Run CUDA test
Write-Host ""
Write-Host "Step 1: Running CUDA sanity check..." -ForegroundColor Yellow
try {
    python cuda_test.py
} catch {
    Write-Host "Warning: CUDA test failed" -ForegroundColor Red
}

# Step 2: Build Docker image
Write-Host ""
Write-Host "Step 2: Building Docker image..." -ForegroundColor Yellow
docker build -t "${DOCKER_IMAGE}:${DOCKER_TAG}" .
docker tag "${DOCKER_IMAGE}:${DOCKER_TAG}" "${DOCKER_IMAGE}:latest"

# Step 3: Test image
Write-Host ""
Write-Host "Step 3: Testing Docker image..." -ForegroundColor Yellow
docker run --rm "${DOCKER_IMAGE}:${DOCKER_TAG}" python --version
docker run --rm "${DOCKER_IMAGE}:${DOCKER_TAG}" python -c "import numpy; import fastapi; print('Dependencies OK')"

# Step 4: Stop old container
Write-Host ""
Write-Host "Step 4: Stopping old containers..." -ForegroundColor Yellow
try {
    docker stop $CONTAINER_NAME 2>$null
    docker rm $CONTAINER_NAME 2>$null
} catch {
    # Ignore errors if container doesn't exist
}

# Step 5: Deploy new container
Write-Host ""
Write-Host "Step 5: Deploying container with GPU support..." -ForegroundColor Yellow
docker run -d `
    --name $CONTAINER_NAME `
    --gpus all `
    -p "${STUDENT_PORT}:${STUDENT_PORT}" `
    -p "${METRICS_PORT}:${METRICS_PORT}" `
    --restart unless-stopped `
    "${DOCKER_IMAGE}:${DOCKER_TAG}"

# Step 6: Health check
Write-Host ""
Write-Host "Step 6: Waiting for service to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host "Step 7: Health check..." -ForegroundColor Yellow
curl.exe -f "http://localhost:${STUDENT_PORT}/health"

Write-Host ""
Write-Host "Step 8: Verify GPU access..." -ForegroundColor Yellow
try {
    docker exec $CONTAINER_NAME nvidia-smi
} catch {
    Write-Host "Warning: nvidia-smi not available" -ForegroundColor Red
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "✅ Deployment completed successfully!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Service URLs:"
Write-Host "  - Health:    http://localhost:${STUDENT_PORT}/health"
Write-Host "  - GPU Info:  http://localhost:${STUDENT_PORT}/gpu-info"
Write-Host "  - API Docs:  http://localhost:${STUDENT_PORT}/docs"
Write-Host ""
Write-Host "View logs:"
Write-Host "  docker logs -f $CONTAINER_NAME"
Write-Host ""
