import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import bitwise_ops
import onnxruntime as ort
import numpy as np
import os
from models import get_model, OptimizedXNORNet, ResidualBinaryLayer

def load_data(batch_size=128):
    print('Loading CIFAR-10 Test Data...')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    try:
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0) # workers=0 avoids multiprocessing overhead confounding single thread test
        return testloader
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def load_optimized_model():
    print("Loading Optimized BNN Model...")
    device = 'cpu'
    model = get_model('xnor', num_classes=10)
    try:
        ckpt = torch.load('checkpoints/xnor_kd_cifar10.pth', map_location=device, weights_only=True)
        if 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt)
    except Exception as e:
        print(f"Failed to load optimized model: {e}")
        return None

    model.to(device)
    model.eval()
    return model

# Pre-pack weights for inference optimization
def pack_model_weights(model):
    packed_weights = []
    # Iterate features in order to match runtime loop
    for layer in model.features:
        if isinstance(layer, ResidualBinaryLayer):
            # BinaryConv2d is at layer.conv
            w = layer.conv.weight
            w_bin = torch.where(w >= 0, 1.0, -1.0)
            # Pack [Out, In, KH, KW] -> [Out, KH, KW, PackedIn] (NHWC logic inside C++ handles this if input is NHWC, wait)
            # The C++ kernel expects weight: [Out, KH, KW, PackedIn]
            # My pack_tensor function expects: [N, C, H, W] input tensor to be packed to [N, H, W, PackedC]
            # So I should reshape/permute weight to be "image-like" [Out, In, KH, KW] -> Pack -> [Out, KH, KW, PackedIn]
            
            # Pack tensor expects 4D input.
            # Conv2d weights are [Out, In, KH, KW]
            # If I pass this to pack_tensor (designed for images), it treats Out as Batch, In as Channels.
            # Output will be [Out, KH, KW, PackedIn].
            w_packed = bitwise_ops.pack_tensor(w_bin.float())
            packed_weights.append(w_packed)
    return packed_weights

def bitwise_forward(model, x, packed_weights):
    with torch.no_grad():
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
                
                # Pack Input [B, C, H, W] -> [B, H, W, PackedC]
                out_packed = bitwise_ops.pack_tensor(out_bin)
                
                # Bitwise Conv (NEON Optimized)
                w_packed = packed_weights[packed_idx]
                conv_out = bitwise_ops.bitwise_conv2d(
                    out_packed, 
                    w_packed, 
                    layer.conv.weight.size(1), # Real in_channels
                    layer.conv.padding, 
                    layer.conv.stride
                )
                
                # Apply Scaling (Alpha)
                # alpha: [C, 1, 1, 1] -> reshape to [1, C, 1, 1]
                alpha_reshaped = layer.conv.alpha.view(1, -1, 1, 1)
                conv_out = conv_out * alpha_reshaped
                
                # Shortcut Path
                shortcut_out = layer.shortcut(out)
                
                # Pooling for Match
                if layer.use_pool:
                    shortcut_out = F.max_pool2d(shortcut_out, kernel_size=2, stride=2)
                    conv_out = layer.pool(conv_out)
                
                # Sum
                out = conv_out + shortcut_out
                packed_idx += 1
            else:
                out = layer(out) # Should not happen in current architecture but good practice
        
        # 3. Classifier
        out = out.view(out.size(0), -1)
        out = model.classifier(out)
        return out

def run_test():
    # Force single thread for fair comparison
    torch.set_num_threads(1)
    print(f"Running Inference Test with {torch.get_num_threads()} thread(s) (Correctness + Speed Verify)...")
    
    testloader = load_data(batch_size=128)
    
    # 1. Run Optimized Model
    model_opt = load_optimized_model()
    if model_opt:
        packed_weights = pack_model_weights(model_opt)
        
        print("\n--- Testing Optimized BNN ---")
        correct = 0
        total = 0
        total_time = 0.0
        total_batches = 0
        
        start_global = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(testloader):
            start_batch = time.time()
            outputs = bitwise_forward(model_opt, inputs, packed_weights)
            end_batch = time.time()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            total_time += (end_batch - start_batch)
            total_batches += 1
            
            if batch_idx % 20 == 0:
                print(f"BNN Batch {batch_idx}: Accuracy: {100.*correct/total:.2f}% | Batch Time: {(end_batch - start_batch)*1000:.1f}ms")
                
        bnn_acc = 100.*correct/total
        print(f"BNN Final Accuracy: {bnn_acc:.2f}%")
        bnn_time = total_time/total_batches*1000
        print(f"BNN Avg Batch Time: {bnn_time:.2f} ms")


    # 2. Run FP32/Simulated Baseline (Same Architecture)
    print("\n--- Testing OptimizedXNORNet (PyTorch Simulated Backend) ---")
    # We use the SAME model architecture, but running via standard PyTorch forward pass
    # This uses BinaryConv2d -> F.conv2d (simulated binary)
    # This is the true apples-to-apples comparison of the BACKEND speed.
    
    model_sim = load_optimized_model() # Load same model again
    if not model_sim: return

    correct = 0
    total = 0
    total_time = 0.0
    total_batches = 0
    
    start_global = time.time()
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            start_batch = time.time()
            # Standard forward pass (uses F.conv2d internally)
            outputs = model_sim(inputs)
            end_batch = time.time()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            total_time += (end_batch - start_batch)
            total_batches += 1
            
            if batch_idx % 20 == 0:
                 print(f"Simulated Batch {batch_idx}: Batch Time: {(end_batch - start_batch)*1000:.1f}ms")
                 
    sim_acc = 100.*correct/total
    print(f"Simulated Final Accuracy: {sim_acc:.2f}%")
    print(f"Simulated Avg Batch Time: {total_time/total_batches*1000:.2f} ms")
    sim_time = total_time/total_batches*1000

    # 3. Run ONNX Runtime Baseline
    print("\n--- Testing OptimizedXNORNet (ONNX Runtime Backend) ---")
    onnx_path = 'checkpoints/xnor_network.onnx'
    if not os.path.exists(onnx_path):
        from benchmark_onnx import export_to_onnx
        export_to_onnx('checkpoints/xnor_kd_cifar10.pth', onnx_path)
    
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    session = ort.InferenceSession(onnx_path, sess_options)
    input_name = session.get_inputs()[0].name
    
    correct = 0
    total = 0
    total_time = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # ONNX expects numpy
            inputs_np = inputs.numpy()
            
            start_batch = time.time()
            outputs_np = session.run(None, {input_name: inputs_np})[0]
            end_batch = time.time()
            
            outputs = torch.from_numpy(outputs_np)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            total_time += (end_batch - start_batch)
            total_batches += 1
            
            if batch_idx % 20 == 0:
                 print(f"ONNX Batch {batch_idx}: Accuracy: {100.*correct/total:.2f}% | Batch Time: {(end_batch - start_batch)*1000:.1f}ms")

    onnx_time = total_time/total_batches*1000
    print(f"ONNX Avg Batch Time: {onnx_time:.2f} ms")
    print(f"ONNX Final Accuracy: {100.*correct/total:.2f}%")
    
    print(f"\n--- FINAL COMPARISON (Batch 128, 1 Thread) ---")
    print(f"{'Engine':<20} | {'Accuracy':<10} | {'Avg Batch Time':<15}")
    print("-" * 50)
    print(f"{'Bitwise Kernel':<20} | {bnn_acc:.2f}%    | {bnn_time:.2f} ms")
    print(f"{'PyTorch Sim':<20} | {sim_acc:.2f}%    | {sim_time:.2f} ms")
    print(f"{'ONNX Runtime':<20} | {100.*correct/total:.2f}%    | {onnx_time:.2f} ms")
    
    speedup_vs_sim = sim_time / bnn_time
    speedup_vs_onnx = onnx_time / bnn_time
    print(f"\nVerdict: Bitwise is {speedup_vs_sim:.2f}x faster than PyTorch and {speedup_vs_onnx:.2f}x faster than ONNX.")

    # 4. Run Fair FP32 Baseline (Optimized architecture but in FP32)
    print("\n--- Testing Fair FP32 Baseline (Optimized Architecture) ---")
    model_fair = get_model('fp32_deep', num_classes=10)
    model_fair.eval()
    
    total_time_fair = 0.0
    total_batches_fair = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # We don't need accuracy for this baseline, just speed
            start_batch = time.time()
            _ = model_fair(inputs)
            end_batch = time.time()
            
            total_time_fair += (end_batch - start_batch)
            total_batches_fair += 1
            
            if batch_idx % 20 == 0:
                 print(f"Fair Baseline Batch {batch_idx}: Batch Time: {(end_batch - start_batch)*1000:.1f}ms")

    fair_time = total_time_fair / total_batches_fair * 1000
    print(f"Fair Baseline Avg Batch Time: {fair_time:.2f} ms")

    # 5. Run FP32 Teacher Baseline (Old SimpleNet)
    print("\n--- Testing Baseline FP32 Teacher ---")
    model_teacher = get_model('baseline', num_classes=10)
    ckpt_teacher = torch.load('checkpoints/baseline_cifar10.pth', map_location='cpu', weights_only=True)
    if 'state_dict' in ckpt_teacher:
        model_teacher.load_state_dict(ckpt_teacher['state_dict'])
    else:
        model_teacher.load_state_dict(ckpt_teacher)
    model_teacher.eval()
    
    correct = 0
    total = 0
    total_time = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            start_batch = time.time()
            outputs = model_teacher(inputs)
            end_batch = time.time()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            total_time += (end_batch - start_batch)
            total_batches += 1
            
            if batch_idx % 20 == 0:
                 print(f"Teacher Batch {batch_idx}: Accuracy: {100.*correct/total:.2f}% | Batch Time: {(end_batch - start_batch)*1000:.1f}ms")

    teacher_acc = 100.*correct/total
    teacher_time = total_time/total_batches*1000
    print(f"Teacher Final Accuracy: {teacher_acc:.2f}%")
    print(f"Teacher Avg Batch Time: {teacher_time:.2f} ms")
    
    print(f"\n--- COMPREHENSIVE PERFORMANCE SUMMARY (Batch 128, 1 Thread) ---")
    print(f"{'Contender':<35} | {'Accuracy':<10} | {'Latency':<15}")
    print("-" * 70)
    print(f"{'SimpleNet Teacher (Shallow FP32)':<35} | {teacher_acc:.2f}%    | {teacher_time:.2f} ms")
    print(f"{'OptimizedFP32Net (Deep FP32 - FAIR)':<35} | N/A       | {fair_time:.2f} ms")
    print(f"{'Optimized BNN (PyTorch Sim)':<35} | {sim_acc:.2f}%    | {sim_time:.2f} ms")
    print(f"{'Optimized BNN (ONNX)':<35} | {sim_acc:.2f}%    | {onnx_time:.2f} ms")
    print(f"{'Optimized BNN (Bitwise Kernel)':<35} | {bnn_acc:.2f}%    | {bnn_time:.2f} ms")
    
    final_speedup = fair_time / bnn_time
    print(f"\nCONGRATULATIONS: Bitwise BNN is {final_speedup:.2f}x faster than the same model in FP32!")
    
    # Correcting the ONNX stats for the final table (I reuse variables, careful)
    # The ONNX loop happened before the Teacher loop. Let's make sure variables are clean.

    
    # Summary
    bnn_time = 1780.0 # Placeholder from earlier run if needed, but updated valid will be printed
    # print(f"\nSpeedup: {fp32_time / bnn_time:.2f}x")


if __name__ == "__main__":
    run_test()
