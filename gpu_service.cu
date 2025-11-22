/*
 * GPU Matrix Addition Service - CUDA C Implementation
 * This file contains the CUDA kernel and host code for matrix addition
 */

#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel for matrix addition
__global__ void matrixAddKernel(const float* A, const float* B, float* C, int rows, int cols) {
    // Calculate global thread position
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // row
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // column
    
    // Check bounds and perform addition
    if (i < rows && j < cols) {
        int idx = i * cols + j;
        C[idx] = A[idx] + B[idx];
    }
}

// Host function callable from Python via ctypes
extern "C" {
    /**
     * Perform matrix addition on GPU
     * 
     * @param h_A Host pointer to matrix A (flattened)
     * @param h_B Host pointer to matrix B (flattened)
     * @param h_C Host pointer to result matrix C (flattened)
     * @param rows Number of rows
     * @param cols Number of columns
     * @param elapsed_time_ms Output parameter for execution time in milliseconds
     * @return 0 on success, non-zero on error
     */
    int gpu_add(const float* h_A, const float* h_B, float* h_C, 
                int rows, int cols, double* elapsed_time_ms) {
        
        // Calculate total size
        int size = rows * cols;
        size_t bytes = size * sizeof(float);
        
        // Device pointers
        float *d_A = NULL, *d_B = NULL, *d_C = NULL;
        
        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // Allocate device memory
        cudaError_t err;
        err = cudaMalloc((void**)&d_A, bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to allocate device memory for A: %s\n", 
                    cudaGetErrorString(err));
            return -1;
        }
        
        err = cudaMalloc((void**)&d_B, bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to allocate device memory for B: %s\n", 
                    cudaGetErrorString(err));
            cudaFree(d_A);
            return -1;
        }
        
        err = cudaMalloc((void**)&d_C, bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to allocate device memory for C: %s\n", 
                    cudaGetErrorString(err));
            cudaFree(d_A);
            cudaFree(d_B);
            return -1;
        }
        
        // Copy data from host to device
        err = cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to copy A to device: %s\n", 
                    cudaGetErrorString(err));
            cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
            return -1;
        }
        
        err = cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to copy B to device: %s\n", 
                    cudaGetErrorString(err));
            cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
            return -1;
        }
        
        // Configure kernel launch parameters
        dim3 threadsPerBlock(16, 16);  // 16x16 = 256 threads per block
        dim3 blocksPerGrid(
            (cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (rows + threadsPerBlock.y - 1) / threadsPerBlock.y
        );
        
        // Start timing
        cudaEventRecord(start);
        
        // Launch kernel
        matrixAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);
        
        // Stop timing
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        // Calculate elapsed time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        *elapsed_time_ms = (double)milliseconds;
        
        // Check for kernel errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return -1;
        }
        
        // Copy result back to host
        err = cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to copy result to host: %s\n", 
                    cudaGetErrorString(err));
            cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return -1;
        }
        
        // Clean up
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        return 0;
    }
    
    /**
     * Get GPU device information
     * @param device_count Output parameter for number of devices
     * @param device_name Output buffer for device name (should be at least 256 bytes)
     * @param memory_total_mb Output parameter for total memory in MB
     * @param memory_free_mb Output parameter for free memory in MB
     * @return 0 on success, non-zero on error
     */
    int get_gpu_info(int* device_count, char* device_name, 
                     float* memory_total_mb, float* memory_free_mb) {
        cudaError_t err = cudaGetDeviceCount(device_count);
        if (err != cudaSuccess) {
            return -1;
        }
        
        if (*device_count > 0) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            snprintf(device_name, 256, "%s", prop.name);
            
            size_t free_mem, total_mem;
            cudaMemGetInfo(&free_mem, &total_mem);
            
            *memory_total_mb = total_mem / (1024.0f * 1024.0f);
            *memory_free_mb = free_mem / (1024.0f * 1024.0f);
        }
        
        return 0;
    }
}
