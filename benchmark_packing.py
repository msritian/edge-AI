
import torch
import time
import bitwise_ops

def benchmark_packing():
    print("Benchmarking Pack Tensor...")
    # Batch 128, 64 Channels, 32x32 Image
    x = torch.randn(128, 64, 32, 32).float()
    # Binarize
    x_bin = torch.where(x >= 0, 1.0, -1.0)
    
    # Warmup
    for _ in range(5):
        _ = bitwise_ops.pack_tensor(x_bin)
        
    start = time.time()
    for _ in range(100):
        packed = bitwise_ops.pack_tensor(x_bin)
    end = time.time()
    
    avg_time = (end - start) / 100
    print(f"Pack Tensor Avg Time (Batch 128): {avg_time*1000:.2f} ms")

if __name__ == "__main__":
    benchmark_packing()
