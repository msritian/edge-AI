import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import bitwise_ops
from models import get_model, OptimizedXNORNet, ResidualBinaryLayer

def load_models():
    print("Loading models...")
    device = 'cpu' # Benchmarking on CPU for bitwise kernels
    
    # Load Baseline FP32 (SimpleNet) - mapping to correct class if saved as dict
    # Note: verify what class 'baseline_cifar10.pth' contains. 
    # Based on inspect_checkpoints, it has features.0.weight etc, matching SimpleNet structure
    baseline_model = get_model('simplenet', num_classes=10)
    try:
        ckpt = torch.load('checkpoints/baseline_cifar10.pth', map_location=device, weights_only=True)
        if 'state_dict' in ckpt:
            baseline_model.load_state_dict(ckpt['state_dict'])
        else:
            baseline_model.load_state_dict(ckpt)
    except Exception as e:
        print(f"Failed to load baseline model: {e}")
        baseline_model = None

    if baseline_model:
        baseline_model.to(device)
        baseline_model.eval()

    # Load Optimized BNN
    optimized_model = get_model('xnor', num_classes=10)
    try:
        ckpt = torch.load('checkpoints/xnor_kd_cifar10.pth', map_location=device, weights_only=True)
        if 'state_dict' in ckpt:
            optimized_model.load_state_dict(ckpt['state_dict'])
        else:
            optimized_model.load_state_dict(ckpt)
    except Exception as e:
        print(f"Failed to load optimized model: {e}")
        optimized_model = None

    if optimized_model:
        optimized_model.to(device)
        optimized_model.eval()
        
    return baseline_model, optimized_model

def benchmark_standard(model, input_tensor, name="Standard"):
    if model is None:
        print(f"Skipping {name} benchmark (model not loaded)")
        return
        
    print(f"\n--- Benchmarking {name} ---")
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_tensor)
            
    start_time = time.time()
    iters = 100
    with torch.no_grad():
        for _ in range(iters):
            _ = model(input_tensor)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / iters
    print(f"Average Inference Time: {avg_time*1000:.4f} ms")
    print(f"Throughput: {1.0/avg_time:.2f} images/sec")
    return avg_time

def benchmark_bitwise_manual(model, input_tensor):
    if model is None:
        print("Skipping Bitwise benchmark (model not loaded)")
        return

    print(f"\n--- Benchmarking Optimized BNN (Bitwise Backend) ---")
    
    # Pre-pack weights to simulate inference-time optimization
    # In a real deployment, weights are packed once and stored
    packed_weights = []
    
    # Iterate through features to find Binary layers and pack their weights
    for layer in model.features:
        if isinstance(layer, ResidualBinaryLayer):
            # BinaryConv2d is at layer.conv
            # Binarize weight: sign(W) * alpha (alpha is handled after conv)
            # We only need sign(W) for packing
            w = layer.conv.weight
            w_bin = torch.where(w >= 0, 1.0, -1.0)
            w_packed = bitwise_ops.pack_tensor(w_bin.float())
            packed_weights.append(w_packed)
            
    # Define the custom forward pass
    def bitwise_forward(x):
        # 1. FP32 Conv1 + BN1 + ReLU
        out = model.conv1(x)
        out = model.bn1(out)
        out = F.relu(out)
        
        # 2. Residual Binary Layers
        packed_idx = 0
        for layer in model.features:
            if isinstance(layer, ResidualBinaryLayer):
                # BN
                out_bn = layer.bn(out)
                
                # Binarize Input
                out_bin = torch.where(out_bn >= 0, 1.0, -1.0)
                
                # Pack Input
                # Note: This is an overhead we must pay at inference time unless we cherish 
                # strictly keeping everything binary, but we have residuals so we switch back and forth
                out_packed = bitwise_ops.pack_tensor(out_bin)
                
                # Bitwise Conv
                # stride=1 unless specified (layer.conv.stride)
                # padding=1 (layer.conv.padding)
                w_packed = packed_weights[packed_idx]
                conv_out = bitwise_ops.bitwise_conv2d(
                    out_packed, 
                    w_packed, 
                    layer.conv.weight.size(1), # Real in_channels
                    layer.conv.padding, 
                    layer.conv.stride
                )
                
                # Apply Scaling (Alpha)
                # alpha shape is [out_channels, 1, 1, 1], need [1, out_channels, 1, 1] for broadcasting
                # reshape safely
                alpha_reshaped = layer.conv.alpha.view(1, -1, 1, 1)
                conv_out = conv_out * alpha_reshaped
                
                # Add Residual
                shortcut_out = layer.shortcut(out)
                
                # Pool if needed for residual matching (Bi-Real block specific logic in models.py)
                # In models.py: 
                # shortcut_out = self.shortcut(x)
                # if self.use_pool:
                #     shortcut_out = F.max_pool2d(shortcut_out, ...)
                #     out = self.pool(out) <= wait, this pools the convolution output?
                
                # Let's check models.py logic again carefully:
                # out = self.conv(out)
                # ...
                # if self.use_pool:
                #    shortcut_out = F.max_pool2d(...)
                #    out = self.pool(out)
                # return out + shortcut_out
                
                if layer.use_pool:
                    shortcut_out = F.max_pool2d(shortcut_out, kernel_size=2, stride=2)
                    conv_out = layer.pool(conv_out)
                
                out = conv_out + shortcut_out
                packed_idx += 1
            else:
                # Fallback for non-binary layers if any (none in current architecture)
                out = layer(out)
        
        # 3. Classifier
        out = out.view(out.size(0), -1)
        out = model.classifier(out)
        return out

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = bitwise_forward(input_tensor)
            
    start_time = time.time()
    iters = 100
    with torch.no_grad():
        for _ in range(iters):
            _ = bitwise_forward(input_tensor)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / iters
    print(f"Average Inference Time: {avg_time*1000:.4f} ms")
    print(f"Throughput: {1.0/avg_time:.2f} images/sec")
    return avg_time

def run():
    baseline_model, optimized_model = load_models()
    
    # Input: Single image batch for latency measurement
    # CIFAR-10: 1x3x32x32
    input_tensor = torch.randn(1, 3, 32, 32)
    
    t_baseline = benchmark_standard(baseline_model, input_tensor, "Baseline FP32 Model")
    t_simulated = benchmark_standard(optimized_model, input_tensor, "Optimized BNN (Simulated)")
    t_bitwise = benchmark_bitwise_manual(optimized_model, input_tensor)
    
    print("\n--- Summary ---")
    if t_baseline and t_bitwise:
        print(f"Speedup BNN (Bitwise) vs Baseline FP32: {t_baseline / t_bitwise:.2f}x")
    if t_simulated and t_bitwise:
        print(f"Speedup BNN (Bitwise) vs BNN (Simulated): {t_simulated / t_bitwise:.2f}x")

if __name__ == "__main__":
    run()
