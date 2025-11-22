"""
GPU-Accelerated Matrix Addition Service using FastAPI and Numba CUDA
"""
import io
import time
import re
import subprocess
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from numba import cuda
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="GPU Matrix Addition Service")

# Student port - change this to your assigned port
STUDENT_PORT = 8001


@cuda.jit
def matrix_add_kernel(A, B, C):
    """
    CUDA kernel for matrix addition.
    Each thread computes one element of the result matrix.
    """
    # Calculate global thread position
    i, j = cuda.grid(2)
    
    # Check bounds
    if i < C.shape[0] and j < C.shape[1]:
        C[i, j] = A[i, j] + B[i, j]


def gpu_matrix_add(matrix_a: np.ndarray, matrix_b: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Perform matrix addition on GPU using CUDA.
    
    Args:
        matrix_a: First input matrix
        matrix_b: Second input matrix
        
    Returns:
        Tuple of (result_matrix, elapsed_time_seconds)
    """
    # Ensure matrices are float32 for GPU compatibility
    matrix_a = matrix_a.astype(np.float32)
    matrix_b = matrix_b.astype(np.float32)
    
    # Start timing
    start_time = time.perf_counter()
    
    # Transfer data to GPU
    d_a = cuda.to_device(matrix_a)
    d_b = cuda.to_device(matrix_b)
    d_c = cuda.device_array_like(d_a)
    
    # Configure grid and block dimensions
    # Using 16x16 threads per block (common choice)
    threads_per_block = (16, 16)
    blocks_per_grid_x = (matrix_a.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (matrix_a.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    # Launch kernel
    matrix_add_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)
    
    # Copy result back to host
    result = d_c.copy_to_host()
    
    # End timing
    elapsed_time = time.perf_counter() - start_time
    
    return result, elapsed_time


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}


@app.get("/gpu-info")
async def gpu_info():
    """
    Get GPU information using nvidia-smi.
    
    Returns information about all available GPUs including memory usage.
    """
    try:
        # Run nvidia-smi command to get GPU info
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"nvidia-smi command failed: {result.stderr}"
            )
        
        # Parse the output
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    gpus.append({
                        "gpu": parts[0],
                        "name": parts[1],
                        "memory_used_MB": int(parts[2]),
                        "memory_total_MB": int(parts[3]),
                        "utilization_percent": int(parts[4]) if len(parts) > 4 else 0
                    })
        
        return {"gpus": gpus}
        
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=500,
            detail="nvidia-smi command timed out"
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="nvidia-smi not found. Ensure NVIDIA drivers are installed."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting GPU info: {str(e)}"
        )


@app.get("/gpu-load")
async def gpu_load():
    """
    Get GPU utilization percentage (Bonus endpoint).
    
    Returns the current GPU load for all available GPUs.
    """
    try:
        # Run nvidia-smi to get GPU utilization
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=index,name,utilization.gpu,utilization.memory',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"nvidia-smi command failed: {result.stderr}"
            )
        
        # Parse the output
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    gpus.append({
                        "gpu": parts[0],
                        "name": parts[1],
                        "gpu_utilization_percent": int(parts[2]),
                        "memory_utilization_percent": int(parts[3]) if len(parts) > 3 else 0
                    })
        
        return {"gpus": gpus}
        
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=500,
            detail="nvidia-smi command timed out"
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="nvidia-smi not found. Ensure NVIDIA drivers are installed."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting GPU load: {str(e)}"
        )


@app.post("/add")
async def add_matrices(
    file_a: UploadFile = File(..., description="First matrix (.npz file)"),
    file_b: UploadFile = File(..., description="Second matrix (.npz file)")
):
    """
    Add two matrices on GPU.
    
    Accepts two .npz files containing NumPy arrays and returns the computation time.
    """
    try:
        # Read uploaded files
        content_a = await file_a.read()
        content_b = await file_b.read()
        
        # Load matrices from .npz files
        with np.load(io.BytesIO(content_a)) as data:
            # Get the first array from the .npz file
            matrix_a = data[data.files[0]]
        
        with np.load(io.BytesIO(content_b)) as data:
            matrix_b = data[data.files[0]]
        
        # Validate matrix shapes
        if matrix_a.shape != matrix_b.shape:
            raise HTTPException(
                status_code=400,
                detail=f"Matrix shapes must match. Got {matrix_a.shape} and {matrix_b.shape}"
            )
        
        # Check if matrices are 2D
        if len(matrix_a.shape) != 2:
            raise HTTPException(
                status_code=400,
                detail=f"Matrices must be 2D. Got shape {matrix_a.shape}"
            )
        
        # Perform GPU matrix addition
        result, elapsed_time = gpu_matrix_add(matrix_a, matrix_b)
        
        # Return response (not returning the result matrix, only metadata)
        return {
            "matrix_shape": list(result.shape),
            "elapsed_time": round(elapsed_time, 6),
            "device": "GPU"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing matrices: {str(e)}"
        )


if __name__ == "__main__":
    # Check if CUDA is available
    try:
        cuda.detect()
        print(f"CUDA detected: {cuda.gpus}")
    except Exception as e:
        print(f"Warning: CUDA not detected - {e}")
    
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=STUDENT_PORT)
