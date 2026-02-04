import torch
import torch.nn.functional as F
import bitwise_ops
import time
import numpy as np

def benchmark():
    device = 'cpu' # Kernels are optimized for CPU
    batch_size = 1
    in_channels = 256
    out_channels = 256
    height = 16
    width = 16
    kernel_size = 3
    padding = 1
    stride = 1
    
    input_float = torch.randn(batch_size, in_channels, height, width).to(device).sign()
    weight_float = torch.randn(out_channels, in_channels, kernel_size, kernel_size).to(device).sign()
    
    # Warmup
    for _ in range(5):
        _ = F.conv2d(input_float, weight_float, padding=padding, stride=stride)
    
    # 1. Standard PyTorch Conv2d Benchmark
    iters = 100
    start = time.time()
    for _ in range(iters):
        res_pt = F.conv2d(input_float, weight_float, padding=padding, stride=stride)
    end = time.time()
    pt_time = (end - start) / iters
    print(f"Standard PyTorch Conv2d Avg Time: {pt_time*1000:.4f} ms")
    
    # 2. Bitwise Kernel Benchmark
    input_packed = bitwise_ops.pack_tensor(input_float)
    weight_packed = bitwise_ops.pack_tensor(weight_float)
    
    # Warmup
    for _ in range(5):
        _ = bitwise_ops.bitwise_conv2d(input_packed, weight_packed, in_channels, padding, stride)
        
    start = time.time()
    for _ in range(iters):
        res_bw = bitwise_ops.bitwise_conv2d(input_packed, weight_packed, in_channels, padding, stride)
    end = time.time()
    bw_time = (end - start) / iters
    print(f"Bitwise Kernel Conv2d Avg Time: {bw_time*1000:.4f} ms")
    
    print(f"Speedup: {pt_time / bw_time:.2f}x")

if __name__ == "__main__":
    benchmark()
