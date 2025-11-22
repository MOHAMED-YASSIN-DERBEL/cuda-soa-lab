"""
Simple CUDA test to verify GPU availability and basic functionality.
"""
from numba import cuda
import numpy as np


@cuda.jit
def add_constant_kernel(x, out, constant):
    """Simple kernel that adds a constant to each element."""
    idx = cuda.grid(1)
    if idx < x.size:
        out[idx] = x[idx] + constant


def test_cuda_basic():
    """Test basic CUDA functionality."""
    print("=" * 60)
    print("CUDA Environment Test")
    print("=" * 60)
    
    # Check CUDA availability
    try:
        cuda.detect()
        print("\n✓ CUDA is available!")
        print(f"  Available GPUs: {cuda.gpus}")
        
        for gpu in cuda.gpus:
            with gpu:
                meminfo = cuda.current_context().get_memory_info()
                print(f"\n  GPU: {gpu.name}")
                print(f"  Compute Capability: {gpu.compute_capability}")
                print(f"  Memory Free: {meminfo[0] / 1024**2:.2f} MB")
                print(f"  Memory Total: {meminfo[1] / 1024**2:.2f} MB")
    except Exception as e:
        print(f"\n✗ CUDA not available: {e}")
        return False
    
    # Test simple kernel
    print("\n" + "-" * 60)
    print("Testing simple CUDA kernel...")
    print("-" * 60)
    
    try:
        # Create test data
        n = 1000
        x = np.arange(n, dtype=np.float32)
        constant = 5.0
        
        # Allocate device memory
        d_x = cuda.to_device(x)
        d_out = cuda.device_array_like(d_x)
        
        # Configure and launch kernel
        threads_per_block = 256
        blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
        
        add_constant_kernel[blocks_per_grid, threads_per_block](d_x, d_out, constant)
        
        # Copy result back
        result = d_out.copy_to_host()
        
        # Verify result
        expected = x + constant
        if np.allclose(result, expected):
            print(f"✓ Kernel execution successful!")
            print(f"  Input: [{x[0]}, {x[1]}, ..., {x[-1]}]")
            print(f"  Output: [{result[0]}, {result[1]}, ..., {result[-1]}]")
            print(f"  Expected: [{expected[0]}, {expected[1]}, ..., {expected[-1]}]")
            return True
        else:
            print("✗ Kernel execution failed - results don't match!")
            return False
            
    except Exception as e:
        print(f"✗ Error during kernel test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_cuda_basic()
    print("\n" + "=" * 60)
    if success:
        print("✓ All CUDA tests passed!")
    else:
        print("✗ CUDA tests failed!")
    print("=" * 60)
