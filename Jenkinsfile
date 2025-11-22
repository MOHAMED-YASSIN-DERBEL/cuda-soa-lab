pipeline {
    agent any

    environment {
        // Configuration variables
        DOCKER_IMAGE = "gpu-service"
        DOCKER_TAG = "${env.BUILD_NUMBER}"
        CONTAINER_NAME = "gpu-service-${env.BUILD_NUMBER}"
        STUDENT_PORT = "8115"  // Change this to your assigned port
        METRICS_PORT = "8000"
        PYTHON_VERSION = "python3"
    }

    stages {
        stage('Checkout') {
            steps {
                echo '📦 Checking out code from repository...'
                checkout scm
            }
        }

        stage('GPU Sanity Test') {
            steps {
                echo '🔍 Installing required dependencies for CUDA test...'
                script {
                    sh '''
                        ${PYTHON_VERSION} -m pip install --user numpy numba cuda-python || true
                    '''
                }
                
                echo '✅ Running CUDA sanity check...'
                script {
                    sh '''
                        ${PYTHON_VERSION} cuda_test.py || echo "Warning: CUDA test failed, but continuing..."
                    '''
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                echo "🐳 Building Docker image with GPU support..."
                script {
                    sh """
                        docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} .
                        docker tag ${DOCKER_IMAGE}:${DOCKER_TAG} ${DOCKER_IMAGE}:latest
                    """
                }
                echo "✅ Docker image built: ${DOCKER_IMAGE}:${DOCKER_TAG}"
            }
        }

        stage('Test Docker Image') {
            steps {
                echo "🧪 Testing Docker image..."
                script {
                    sh """
                        docker run --rm ${DOCKER_IMAGE}:${DOCKER_TAG} python --version
                        docker run --rm ${DOCKER_IMAGE}:${DOCKER_TAG} python -c "import numpy; import fastapi; print('Dependencies OK')"
                    """
                }
            }
        }

        stage('Stop Old Container') {
            steps {
                echo "🛑 Stopping and removing old containers..."
                script {
                    sh """
                        docker ps -a | grep ${DOCKER_IMAGE} | awk '{print \$1}' | xargs -r docker stop || true
                        docker ps -a | grep ${DOCKER_IMAGE} | awk '{print \$1}' | xargs -r docker rm || true
                    """
                }
            }
        }

        stage('Deploy Container') {
            steps {
                echo "🚀 Deploying Docker container with GPU support..."
                script {
                    sh """
                        docker run -d \
                            --name ${CONTAINER_NAME} \
                            --gpus all \
                            -p ${STUDENT_PORT}:${STUDENT_PORT} \
                            -p ${METRICS_PORT}:${METRICS_PORT} \
                            --restart unless-stopped \
                            ${DOCKER_IMAGE}:${DOCKER_TAG}
                    """
                }
                echo "✅ Container deployed: ${CONTAINER_NAME}"
            }
        }

        stage('Health Check') {
            steps {
                echo "🏥 Performing health check..."
                script {
                    sh """
                        sleep 10
                        curl -f http://localhost:${STUDENT_PORT}/health || exit 1
                    """
                }
                echo "✅ Service is healthy!"
            }
        }

        stage('Verify GPU Access') {
            steps {
                echo "🎮 Verifying GPU access in container..."
                script {
                    sh """
                        docker exec ${CONTAINER_NAME} nvidia-smi || echo "Warning: nvidia-smi not available"
                        curl -f http://localhost:${STUDENT_PORT}/gpu-info || echo "Warning: GPU info not available"
                    """
                }
            }
        }
    }

    post {
        success {
            echo "🎉 Deployment completed successfully!"
            echo "Service URL: http://localhost:${STUDENT_PORT}"
            echo "Health: http://localhost:${STUDENT_PORT}/health"
            echo "GPU Info: http://localhost:${STUDENT_PORT}/gpu-info"
        }
        failure {
            echo "💥 Deployment failed. Check logs for errors."
            script {
                sh """
                    docker logs ${CONTAINER_NAME} || true
                    docker ps -a | grep ${DOCKER_IMAGE} || true
                """
            }
        }
        always {
            echo "🧾 Pipeline finished."
            echo "Cleaning up old Docker images..."
            script {
                sh """
                    docker image prune -f || true
                """
            }
        }
    }
}
