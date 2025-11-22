"""
Script to create test matrices for the GPU matrix addition service.
"""
import numpy as np

def create_test_matrices():
    matrix_a = np.random.rand(512, 512).astype(np.float32)
    matrix_b = np.random.rand(512, 512).astype(np.float32)
    
    np.savez('matrix1.npz', matrix_a)
    np.savez('matrix2.npz', matrix_b)
    
    print(f"Created matrix1.npz with shape {matrix_a.shape}")
    print(f"Created matrix2.npz with shape {matrix_b.shape}")
    
    matrix_c = np.random.rand(256, 256).astype(np.float32)
    np.savez('matrix_wrong_shape.npz', matrix_c)
    print(f"Created matrix_wrong_shape.npz with shape {matrix_c.shape}")
    
    matrix_large_a = np.random.rand(2048, 2048).astype(np.float32)
    matrix_large_b = np.random.rand(2048, 2048).astype(np.float32)
    
    np.savez('matrix_large_a.npz', matrix_large_a)
    np.savez('matrix_large_b.npz', matrix_large_b)
    print(f"Created matrix_large_a.npz and matrix_large_b.npz with shape {matrix_large_a.shape}")

if __name__ == "__main__":
    create_test_matrices()
