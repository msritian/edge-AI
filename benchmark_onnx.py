import torch
import torch.nn as nn
import torch.onnx
import onnxruntime as ort
import time
import numpy as np
from models import get_model, OptimizedXNORNet

def export_to_onnx(model_path, onnx_path):
    print(f"Loading model from {model_path}...")
    device = 'cpu'
    model = get_model('xnor', num_classes=10)
    
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        if 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return False
        
    model.eval()
    
    # Dummy Input for ONNX Export
    dummy_input = torch.randn(1, 3, 32, 32, device=device)
    
    print(f"Exporting to ONNX: {onnx_path}...")
    torch.onnx.export(model, 
                      dummy_input, 
                      onnx_path, 
                      export_params=True, 
                      opset_version=11, 
                      do_constant_folding=True, 
                      input_names=['input'], 
                      output_names=['output'], 
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print("Export successful.")
    return True

def benchmark_onnx(onnx_path, batch_size=128):
    print(f"\nBenchmarking ONNX Runtime (Batch {batch_size}, Single Threaded)...")
    
    # Configure ONNX Runtime for Single Thread (Fair Fight)
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    session = ort.InferenceSession(onnx_path, sess_options)
    
    input_name = session.get_inputs()[0].name
    
    # Random Inputs for Speed Test
    x = np.random.randn(batch_size, 3, 32, 32).astype(np.float32)
    
    # Warmup
    print("Warmup...")
    for _ in range(10):
        _ = session.run(None, {input_name: x})
        
    # Benchmark
    print("Running Benchmark Loop (100 iters)...")
    start_time = time.time()
    for _ in range(100):
        _ = session.run(None, {input_name: x})
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    print(f"ONNX Avg Batch Time: {avg_time*1000:.2f} ms")
    return avg_time

if __name__ == "__main__":
    model_path = 'checkpoints/xnor_kd_cifar10.pth'
    onnx_path = 'checkpoints/xnor_network.onnx'
    
    if export_to_onnx(model_path, onnx_path):
        benchmark_onnx(onnx_path, batch_size=128)
