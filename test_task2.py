"""
Test script for Task 2 - GPU Info Endpoint
"""
import requests
import json

def test_gpu_info(base_url="http://localhost:8001"):
    """Test the /gpu-info endpoint"""
    
    print("="*60)
    print("Task 2 - Testing /gpu-info Endpoint")
    print("="*60)
    
    try:
        # Test /gpu-info endpoint
        print(f"\nTesting {base_url}/gpu-info...")
        response = requests.get(f"{base_url}/gpu-info", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("\n✓ GPU Info Retrieved Successfully:")
            print(json.dumps(data, indent=2))
            
            # Display in a nice format
            if "gpus" in data and data["gpus"]:
                print("\n" + "-"*60)
                print("GPU Details:")
                print("-"*60)
                for gpu in data["gpus"]:
                    print(f"\nGPU {gpu['gpu']}:")
                    print(f"  Name: {gpu.get('name', 'N/A')}")
                    print(f"  Memory Used: {gpu['memory_used_MB']} MB")
                    print(f"  Memory Total: {gpu['memory_total_MB']} MB")
                    memory_percent = (gpu['memory_used_MB'] / gpu['memory_total_MB'] * 100) if gpu['memory_total_MB'] > 0 else 0
                    print(f"  Memory Usage: {memory_percent:.1f}%")
                    if 'utilization_percent' in gpu:
                        print(f"  GPU Utilization: {gpu['utilization_percent']}%")
                print("-"*60)
            else:
                print("⚠ No GPU information returned")
                
        else:
            print(f"\n✗ Request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"\n✗ Cannot connect to {base_url}")
        print("Make sure the service is running:")
        print("  python main.py")
    except requests.exceptions.Timeout:
        print("\n✗ Request timed out")
    except Exception as e:
        print(f"\n✗ Error: {e}")
    
    print("\n" + "="*60)


def test_all_endpoints(base_url="http://localhost:8001"):
    """Test all endpoints including Task 2"""
    
    print("\n" + "="*60)
    print("Complete Endpoint Testing (Tasks 1 & 2)")
    print("="*60)
    
    # Test 1: Health check
    print("\n1. Testing /health...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print(f"   ✓ {response.json()}")
        else:
            print(f"   ✗ Status: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 2: GPU Info
    print("\n2. Testing /gpu-info...")
    try:
        response = requests.get(f"{base_url}/gpu-info", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Found {len(data.get('gpus', []))} GPU(s)")
            for gpu in data.get('gpus', []):
                print(f"      GPU {gpu['gpu']}: {gpu['memory_used_MB']}/{gpu['memory_total_MB']} MB")
        else:
            print(f"   ✗ Status: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 3: Matrix addition (if test files exist)
    print("\n3. Testing /add...")
    try:
        import os
        if os.path.exists('matrix1.npz') and os.path.exists('matrix2.npz'):
            with open('matrix1.npz', 'rb') as f1, open('matrix2.npz', 'rb') as f2:
                files = {
                    'file_a': ('matrix1.npz', f1, 'application/octet-stream'),
                    'file_b': ('matrix2.npz', f2, 'application/octet-stream')
                }
                response = requests.post(f"{base_url}/add", files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✓ Matrix addition: {result['matrix_shape']}, {result['elapsed_time']}s")
            else:
                print(f"   ✗ Status: {response.status_code}")
        else:
            print("   ⚠ Test files not found. Run: python create_test_matrices.py")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    import sys
    
    # Allow custom URL as argument
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001"
    
    # Run detailed GPU info test
    test_gpu_info(base_url)
    
    # Run all endpoint tests
    test_all_endpoints(base_url)
