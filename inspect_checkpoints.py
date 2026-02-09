import torch
import sys

def check_checkpoint(path):
    try:
        print(f"--- Checking {path} ---")
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        print(f"Keys: {list(state_dict.keys())[:5]}")
        
        # Check for specific keys to identify model architecture
        is_binary = any('alpha' in k for k in state_dict.keys())
        print(f"Contains 'alpha' (BinaryNet indicator): {is_binary}")
        
        # Check for residuals
        has_shortcut = any('shortcut' in k for k in state_dict.keys())
        print(f"Contains 'shortcut' (Residual/Bi-Real indicator): {has_shortcut}")

    except Exception as e:
        print(f"Error loading {path}: {e}")

if __name__ == "__main__":
    check_checkpoint('checkpoints/baseline_cifar10.pth')
    check_checkpoint('checkpoints/xnor_kd_cifar10.pth')
