# build.ps1 - Build and test Docker image (Windows)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Building GPU Matrix Service Docker Image" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Build the image
docker build -t gpu-service:latest .

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✓ Image built successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "To run the container:"
    Write-Host "  docker run --gpus all -p 8001:8001 gpu-service:latest"
    Write-Host ""
    Write-Host "Or use docker-compose:"
    Write-Host "  docker compose up"
    Write-Host ""
} else {
    Write-Host "✗ Build failed!" -ForegroundColor Red
    exit 1
}
