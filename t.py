import torch
import time

# Test with different sizes
sizes = [10, 100, 1000, 10000]

for n in sizes:
    X_cpu = torch.randn(n, 100)
    X_gpu = X_cpu.cuda()
    
    # CPU timing
    start = time.time()
    result_cpu = X_cpu @ X_cpu.T
    cpu_time = time.time() - start
    
    # GPU timing (with sync)
    start = time.time()
    result_gpu = X_gpu @ X_gpu.T
    torch.cuda.synchronize()  # Wait for GPU to finish
    gpu_time = time.time() - start
    
    print(f"Size {n}: CPU={cpu_time*1000:.2f}ms, GPU={gpu_time*1000:.2f}ms, Speedup={cpu_time/gpu_time:.2f}x")