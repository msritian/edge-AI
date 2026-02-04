import torch
import torch.nn.functional as F
import bitwise_ops
import numpy as np

def test_bitwise_conv2d():
    # Setup parameters
    batch_size = 2
    in_channels = 128 # Must be multiple of 64 for simple packing verification
    out_channels = 64
    height = 8
    width = 8
    kernel_size = 3
    padding = 1
    stride = 1
    
    # Create random tensors and binarize them
    input_float = torch.randn(batch_size, in_channels, height, width).sign()
    # Ensure no zeros (sign(0) = 0 in PyTorch, but we want -1 or 1)
    input_float[input_float == 0] = 1
    
    weight_float = torch.randn(out_channels, in_channels, kernel_size, kernel_size).sign()
    weight_float[weight_float == 0] = 1
    
    # 1. Reference PyTorch Calculation
    # result = sum(A * B)
    ref_output = F.conv2d(input_float, weight_float, padding=padding, stride=stride)
    
    # 2. Bitwise Kernel Calculation
    # Pack tensors
    input_packed = bitwise_ops.pack_tensor(input_float)
    weight_packed = bitwise_ops.pack_tensor(weight_float)
    
    # Run bitwise conv
    bitwise_output = bitwise_ops.bitwise_conv2d(input_packed, weight_packed, in_channels, padding, stride)
    
    # Comparison
    print(f"Ref Output Shape: {ref_output.shape}")
    print(f"Bitwise Output Shape: {bitwise_output.shape}")
    
    diff = (ref_output - bitwise_output).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max Diff: {max_diff}")
    print(f"Mean Diff: {mean_diff}")
    
    if max_diff < 1e-4:
        print("SUCCESS: Bitwise kernel matches PyTorch reference!")
    else:
        print("FAILURE: Bitwise kernel results differ from PyTorch reference.")
        # Print a small sample for debugging
        print("Sample Ref:")
        print(ref_output[0, 0, :3, :3])
        print("Sample Bitwise:")
        print(bitwise_output[0, 0, :3, :3])

if __name__ == "__main__":
    test_bitwise_conv2d()
