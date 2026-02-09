import torch
import torch.nn as nn
import onnxruntime as ort
import time
import numpy as np
import os
import resource
import bitwise_ops
from models import get_model, OptimizedXNORNet, ResidualBinaryLayer

def get_peak_ram():
    # Returns peak RSS in MB
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # On macOS, ru_maxrss is in bytes. On Linux, it's in kilobytes.
    # Since USER is on mac:
    return usage.ru_maxrss / (1024 * 1024)

def load_optimized_model():
    model = get_model('xnor', num_classes=10)
    ckpt = torch.load('checkpoints/xnor_kd_cifar10.pth', map_location='cpu', weights_only=True)
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model

def pack_model_weights(model):
    packed_weights = []
    for layer in model.features:
        if isinstance(layer, ResidualBinaryLayer):
            w = layer.conv.weight
            w_bin = torch.where(w >= 0, 1.0, -1.0)
            w_packed = bitwise_ops.pack_tensor(w_bin.float())
            packed_weights.append(w_packed)
    return packed_weights

def bitwise_forward(model, x, packed_weights):
    with torch.no_grad():
        out = F.relu(model.bn1(model.conv1(x)))
        packed_idx = 0
        for layer in model.features:
            if isinstance(layer, ResidualBinaryLayer):
                out_bn = layer.bn(out)
                out_bin = torch.where(out_bn >= 0, 1.0, -1.0)
                out_packed = bitwise_ops.pack_tensor(out_bin)
                w_packed = packed_weights[packed_idx]
                in_channels = layer.conv_in_channels # Use cached metadata
                conv_out = bitwise_ops.bitwise_conv2d(
                    out_packed, w_packed, in_channels,
                    layer.conv.padding, layer.conv.stride
                )
                conv_out = conv_out * layer.conv.alpha.view(1, -1, 1, 1)
                shortcut_out = layer.shortcut(out)
                if layer.use_pool:
                    shortcut_out = F.max_pool2d(shortcut_out, kernel_size=2, stride=2)
                    conv_out = layer.pool(conv_out)
                out = conv_out + shortcut_out
                packed_idx += 1
            else:
                out = layer(out)
        out = out.view(out.size(0), -1)
        out = model.classifier(out)
        return out

import torch.nn.functional as F

def benchmark_memory():
    print("--- Memory Benchmarking (Peak RAM) ---")
    batch_size = 128
    inputs = torch.randn(batch_size, 3, 32, 32)

    # 1. PyTorch FP32 Simulation
    print("\nMeasuring PyTorch Simulated BNN...")
    model_sim = load_optimized_model()
    start_mem = get_peak_ram()
    with torch.no_grad():
        for _ in range(10):
            _ = model_sim(inputs)
    peak_sim = get_peak_ram()
    print(f"Simulated Peak RAM: {peak_sim:.2f} MB (Delta: {peak_sim - start_mem:.2f} MB)")

    # 2. ONNX Runtime
    print("\nMeasuring ONNX Runtime...")
    onnx_path = 'checkpoints/xnor_network.onnx'
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    session = ort.InferenceSession(onnx_path, sess_options)
    input_name = session.get_inputs()[0].name
    x_np = inputs.numpy()
    
    start_mem = get_peak_ram()
    for _ in range(10):
        _ = session.run(None, {input_name: x_np})
    peak_onnx = get_peak_ram()
    print(f"ONNX Peak RAM: {peak_onnx:.2f} MB (Delta: {peak_onnx - start_mem:.2f} MB)")

    # 3. Bitwise Kernel
    print("\nMeasuring Bitwise Kernel...")
    model_bw = load_optimized_model()
    packed_weights = pack_model_weights(model_bw)
    
    # We "delete" the huge original weights to simulate true bitwise efficiency
    # In a real C++ deployment, we wouldn't even load the FP32 weights.
    for layer in model_bw.features:
        if isinstance(layer, ResidualBinaryLayer):
            layer.conv_in_channels = layer.conv.weight.size(1) # Cache metadata
            layer.conv.weight = nn.Parameter(torch.empty(0)) 
            
    start_mem = get_peak_ram()
    for _ in range(10):
        _ = bitwise_forward(model_bw, inputs, packed_weights)
    peak_bw = get_peak_ram()
    print(f"Bitwise Peak RAM: {peak_bw:.2f} MB (Delta: {peak_bw - start_mem:.2f} MB)")

if __name__ == "__main__":
    benchmark_memory()
