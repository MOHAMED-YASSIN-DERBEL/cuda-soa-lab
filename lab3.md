# **Lab Session – CI/CD & Monitoring of GPU-Accelerated Microservices**

**Course:** Service Oriented Architecture (SOA)
**Instructor:** Hamza Gbada - Walid Chainbi
**Server:** `http://10.90.90.100`
**Duration:** 3 hours

---

## **Learning Objectives**

By the end of this lab, you will:

1. Design and implement a **GPU-based microservice** using **FastAPI** and **CUDA/Numba**.
2. Expose REST endpoints for computation and GPU monitoring.
3. Containerize your service using **Docker (GPU runtime)**.
4. Deploy automatically to the instructor’s server using **Jenkins**.
5. Collect and visualize real-time GPU and request metrics using **Prometheus and Grafana**.

---

## **Context**

Service-oriented systems often integrate high-performance GPU-accelerated components — for example, matrix computations or neural inference services.
This lab simulates a real DevOps pipeline for such services.

You will:

* Implement a **matrix addition service on GPU**.
* Expose metrics about GPU memory and request latency.
* Deploy it via **Jenkins pipeline** to a shared GPU server.
* Monitor it in **Grafana dashboards**.

Add a part to explain how CUDA kernel work
---

## **Tools Provided**

| Tool                    | Role                    | Access                      |
| ----------------------- | ----------------------- |-----------------------------|
| Jenkins                 | Continuous Deployment   | `http://10.90.90.100:8090`  |
| Docker + NVIDIA Toolkit | Container runtime       | Installed on instructor server |
| Prometheus              | Metrics collector       | `http://10.90.90.100:9090`  |
| Grafana                 | Visualization dashboard | `http://10.90.90.100:3000`  |
| GitHub                  | Version control         | Use your personal account   |

---

##  **Lab Setup**

### 1. Fork the Instructor Repository**

You must start by **forking** the template repository:

```
https://github.com/HamzaGbada/cuda-soa-lab
```

* Click **Fork** (top-right of GitHub).
* Clone it to your local machine:

  ```bash
  git clone https://github.com/<your-username>/cuda-soa-lab.git
  cd cuda-soa-lab
  ```

---

## **Task Details**

### **Task 1 – GPU Matrix Addition Service (Python+Numba *or* C/CUDA)**

#### **Objective**

You will implement a GPU-accelerated matrix addition microservice with **FastAPI** that accepts two uploaded matrices, adds them on the GPU, and returns computation time.

Students may choose **one of two implementation paths**:

| Option    | Language                 | GPU API           | Difficulty |
| --------- | ------------------------ | ----------------- | ---------- |
| 🐍 Path A | Python                   | Numba (CUDA JIT)  | Easier     |
| ⚙️ Path B | C/CUDA + FastAPI wrapper | CUDA C via ctypes | Advanced   |

---

#### Service Specification**

| Feature            | Description                                                                          |
|--------------------|--------------------------------------------------------------------------------------|
| **Endpoint**       | `POST /add`                                                                          |
| **port**           | Each student must use a different port `<student_port>`                              |
| **Input**          | Two uploaded `.npz` files containing NumPy matrices                                  |
| **Output**         | JSON: `{ "matrix_shape": [rows, cols], "elapsed_time": <seconds>, "device": "GPU" }` |
| **Validation**     | Reject matrices with different shapes                                                |
| **Extra Endpoint** | `/health` → returns `{ "status": "ok" }`                                             |

---

####  **General API Behavior**

Your API must:

1. Accept **two uploaded `.npz` files** via `POST /add`.

   * Each file contains one NumPy array (e.g., `arr_0`).
   * Use FastAPI’s `UploadFile` type.
   * Example request:

     ```bash
     curl -X POST "http://localhost:<student_port>/add" \
       -F "file_a=@matrix_a.npz" \
       -F "file_b=@matrix_b.npz"
     ```

2. Load both matrices into GPU memory.

3. Perform **matrix addition** on GPU.

4. Return only:

   ```json
   {
     "matrix_shape": [512, 512],
     "elapsed_time": 0.0213,
     "device": "GPU"
   }
   ```

---

#### **Path A – Python + Numba Implementation**

Use **Numba** and its `cuda.jit` decorator to compile GPU kernels.

Steps to guide your implementation:

1. Write a CUDA kernel function with `@cuda.jit`.
2. Allocate device memory (`cuda.to_device`).
3. Launch the kernel with appropriate grid/block dimensions.
4. Copy results back to host.
5. Measure execution time using `time.time()` or `perf_counter()`.

---

####  **Path B – C/CUDA Implementation + FastAPI Wrapper**

In this path, you will:

1. Write a CUDA kernel in **C/CUDA** (e.g., `gpu_service.cu`).
2. Compile it to a shared library:

   ```bash
   nvcc -Xcompiler -fPIC -shared -o libgpuadd.so gpu_service.cu
   ```
3. Write a **FastAPI wrapper** in Python that:

   * Loads `libgpuadd.so` with `ctypes`.
   * Passes NumPy array pointers to the GPU function.
   * Receives results back and returns elapsed time.

Your `.cu` file should:

* Implement a function `void gpu_add(float* A, float* B, float* C, int N)`.
* Allocate device memory, copy data, launch kernel, copy results back.

Example of calling C code from Python:

```python
from ctypes import cdll, c_int, c_void_p
lib = cdll.LoadLibrary("./libgpuadd.so")
lib.gpu_add.argtypes = [c_void_p, c_void_p, c_void_p, c_int]
```

The student must handle type conversion between NumPy and ctypes manually.

---

#### **Validation Checklist**

| Test             | Command                                                                                        | Expected Output                |
|------------------|------------------------------------------------------------------------------------------------|--------------------------------|
| Health check     | `curl http://localhost:<student_port>/health`                                                  | `{"status":"ok"}`              |
| GPU computation  | `curl -F "file_a=@matrix_a.npz" -F "file_b=@matrix_b.npz" http://localhost:<student_port>/add` | JSON with matrix size + timing |
| Wrong shape      | Upload mismatched files                                                                        | Returns HTTP 400               |
| GPU load visible | Run `watch nvidia-smi` during test                                                             | GPU activity spikes            |


---

### **Task 2 – Add a /gpu-info Endpoint**

Add an endpoint `/gpu-info` that runs the `nvidia-smi` command and returns:

```json
{
  "gpus": [
    {"gpu": "0", "memory_used_MB": 312, "memory_total_MB": 4096}
  ]
}
```

* Use Python’s `subprocess` to call `nvidia-smi`.
* Also prepare to export these metrics later to Prometheus (Task 5).

---

### **Task 3 – Containerize the Application**

Create a **Dockerfile** that:

* Uses a CUDA base image (`nvidia/cuda:<version>-runtime-ubuntu22.04`)
* Installs dependencies (Python, FastAPI, Numba, etc.)
* Exposes ports `<student_port>` (FastAPI) and `8000` (Prometheus metrics)
* Starts your service with:

  ```bash
  python3 main.py
  ```

Then test locally (if possible):

```bash
docker build -t gpu-service .
docker run --gpus all -p <student_port>:<student_port> gpu-service
```

---

### **Task 4 – Automate Deployment with Jenkins**

Jenkins should do:

1. Pull your repository from GitHub.
2. write a script to test minimal cuda kernel
3. Build your Docker image.
4. Deploy it to the instructor’s GPU server.


**Note:**

1. Go to Jenkins: `http://10.90.90.100:8081`
2. Create pipeline job with the following name `gpu-lab-yourname`
3. Click **“Build Now”**

---

### **Task 5 – Monitoring and Visualization**

In the grafana Dashboard check the created dashboard for metric visualization on http://10.90.90.100:3000

---

### **Bonus Challenges (Optional)**

* Add `/gpu-load` to report GPU utilization percentage.
* Compare matrix addition performance CPU vs GPU.
* Use a CUDA C implementation instead of Numba.

---

## Advanced Section (Optional)

---

## **Annexe — How GPU Kernels Work (via CUDA / Numba in Python)**

### 1. What is a kernel?

A **kernel** is a function that runs on the GPU and is executed *in parallel* by many threads. Each thread executes the same code but on different data.
In the Python + Numba + CUDA context, a kernel is decorated with `@cuda.jit` and is launched from the host. 

### 2. Minimal example – vector addition / constant add

```python
from numba import cuda
import numpy as np

@cuda.jit
def add_gpu(x, out):
    idx = cuda.grid(1)
    out[idx] = x[idx] + 2

a = np.arange(10, dtype=np.float32)
d_a = cuda.to_device(a)
d_out = cuda.device_array_like(d_a)

nbr_block_per_grid = 2
nbr_thread_per_block = 5
add_gpu[nbr_block_per_grid, nbr_thread_per_block](d_a, d_out)

out = d_out.copy_to_host()
print(out)  # [2., 3., 4., … , 11.]
```

Here:

* `cuda.grid(1)` returns the **global thread index** (in a 1D grid) so that each thread knows which element to process. 
* We allocate device memory, launch the kernel, then copy results back to host.
* When the problem is too small, the overhead of copying to/from GPU may make the GPU version *slower* than a CPU loop. 

### 3. Memory management

Key points:

* Data must be explicitly transferred from host (CPU) to device (GPU), e.g., via `cuda.to_device(...)`. 
* After kernel execution, results must often be copied back to host via something like `copy_to_host()`.
* Overhead of data transfer can dominate when the computation is light.
* It is advisable to minimize data transfers by keeping data on the device when possible. 

### 4. Threads, blocks, grids & indexing

* Threads are grouped into blocks; blocks form a grid. 
* You can have 1-D, 2-D, or 3-D threading and block/grid structures. For example, `cuda.grid(2)` returns a 2-tuple `(i, j)` for 2D indexing. 
* In a 1-D scenario:

  ```text
  index = threadIdx.x + blockIdx.x * blockDim.x
  ```

  In Numba, this is abstracted by `cuda.grid(1)`. 
* Limits: e.g., max threads per block (commonly 1024 in many devices) etc. 
* Example: For a 2D image processing kernel you might choose `blockdim = (32, 32)` and compute `griddim = (image_shape[0]//32+1, image_shape[1]//32+1)`. Then use `i,j = cuda.grid(2)` to compute per‐pixel operations. 

### 5. Device (helper) functions

* In CUDA programming you often want helper functions that run on the device (GPU) but are *not* kernels. These are “device functions”. 
* In Numba you can declare them with `@cuda.jit(device=True)`. They can be called from inside a kernel.
* Example:

  ```python
  @cuda.jit(device=True)
  def maximum_device(arr):
      max_val = arr[0,0]
      for i in range(arr.shape[0]):
          for j in range(arr.shape[1]):
              if arr[i,j] > max_val:
                  max_val = arr[i,j]
      return max_val

  @cuda.jit
  def gamma_correction_device(image, out, gamma):
      i, j = cuda.grid(2)
      if i < image.shape[0] and j < image.shape[1]:
          m = maximum_device(image)
          out[i,j] = (image[i,j] ** gamma) / m
  ```

  Using device functions helps avoid code duplication and supports more complex algorithms. 

### 6. When does GPU help (and when not)?

* For *very small* problems or very light computation, the GPU version may be *slower* than a simple CPU loop — because of data transfer overhead and kernel launch overhead. 
* As the problem size grows (more data, heavier computation), the GPU gives large speed-ups. For example in the article: a convolution kernel ran ~70× faster than a comparable CPU library function (`scipy.ndimage.correlate`). 
* Thus: Use GPU when (1) you have enough data, (2) the computation per element is non-trivial, and (3) you can amortize the transfer/kernel overhead.

### 7. Example: Image convolution via GPU

Pseudo-steps:

1. Prepare input image (e.g., `image = np.random.randint(256, size=(n,n))`).
2. Transfer to device: `d_image = cuda.to_device(image)`.
3. Allocate device result: `d_result = cuda.device_array_like(image)`.
4. Choose block and grid dimensions, e.g., `blockdim = (32,32)`, `griddim = (n//32 + 1, n//32 + 1)`.
5. Launch convolution kernel: `convolve[griddim, blockdim](d_result, d_mask, d_image)`.
6. Copy back result: `result = d_result.copy_to_host()`.
7. Compare to CPU version for performance. 
