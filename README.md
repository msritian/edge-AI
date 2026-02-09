# XNOR-Net: Efficient Inference with Bitwise Kernels

This repository contains a high-performance implementation of **XNOR-Net** (Binary Neural Networks) for CIFAR-10, featuring custom C++ bitwise kernels for accelerated inference.

## 🚀 Overview

The project aims to demonstrate the efficiency gains of binarized neural networks on CPU architectures. By replacing traditional floating-point convolutions with bitwise XNOR and POPCOUNT operations, we can theoretically achieve significant speedups and memory reductions.

### Key Features
- **Optimized XNOR-Net Architecture**: VGG-style BNN with Bi-Real Net residual connections for improved gradient flow.
- **Learnable Scaling Factors**: Per-channel adaptive quantization scales that optimize during training.
- **Knowledge Distillation**: Soft target guidance from a high-accuracy FP32 teacher model (87.11%).
- **Custom C++ Kernels**: Efficient implementation of bitwise convolutions using bit-packing and native popcount logic.
- **Stable BNN Training**: Implementation of the Straight-Through Estimator (STE), latent weight clipping, and robust normalization.
- **Automated Benchmarking**: Tools to compare PyTorch FP32 performance vs. our optimized bitwise implementation.

---

## 📈 Comparative Analysis

We conducted extensive experiments comparing baseline XNOR-Net, optimized XNOR-Net, and Full-Precision models.

### 1. Accuracy vs. Efficiency
| Metric | Optimized XNOR-Net | Baseline XNOR-Net | FP32 Teacher |
| :--- | :--- | :--- | :--- |
| **Peak Accuracy** | **81.35%** (20 Epochs) | 70.35% (20 Epochs) | 87.11% (20 Epochs) |
| **Early Accuracy** | 45.65% (Epoch 1) | 19.95% (Epoch 1) | 52.48% (Epoch 1) |
| **Weight Size** | 1 bit | 1 bit | 32 bits |
| **Activation Size** | 1 bit | 1 bit | 32 bits |
| **Memory Saving** | **32x Reduction** | **32x Reduction** | 1x (Reference) |
| **Core Logic** | XNOR + Popcount | XNOR + Popcount | Floating-Point MAC |

### 2. Key Insights
- **Optimization Impact**: The optimized architecture achieves **+11% absolute improvement** over the baseline BNN, closing the gap to full-precision from ~17% to just ~6%.
- **Convergence Speed**: Knowledge Distillation dramatically accelerates convergence - the optimized model reaches 61% by Epoch 4 vs. 35% for the baseline.
- **Resource Efficiency**: In specialized hardware (FPGA/ASIC), the BNN occupies 32x less silicon area for weights and uses significantly less power due to the absence of floating-point multipliers.

---

## 🏗 Knowledge & Implementation Details

### 1. Binary Neural Network (BNN) Architecture
The model uses a binarized version of a VGG-like network. Unlike standard networks, the weights and activations are constrained to $\{-1, +1\}$.

- **Binary Activation**: $x_b = \text{sign}(x)$
- **Binary Weights**: $w_b = \text{sign}(w) \cdot \alpha$, where $\alpha$ is the mean absolute value of the weights for scaling.
- **Straight-Through Estimator (STE)**: Used during backpropagation to allow gradients to flow through the non-differentiable `sign` function.

### 2. Custom Bitwise Kernels
We implemented a custom PyTorch extension in C++ (`bitwise_kernel.cpp`) to handle the heavy lifting:
- **Bit-Packing**: 64 binary values (0/1 represented as +/-1) are packed into a single `uint64_t`.
- **XNOR-Popcount**: The convolution inner loop uses bitwise `XNOR` followed by `__builtin_popcountll` to calculate the dot product of 64 elements in just a few CPU cycles.
- **Cache Optimization**: Loop tiling and ordering are optimized for CPU cache locality.

### 3. Advanced Optimization Techniques
To push BNN accuracy beyond 70%, we implemented state-of-the-art optimization strategies:

#### Bi-Real Net Residual Connections
Standard residual connections preserve information that would otherwise be lost through aggressive quantization. Our implementation adds real-valued shortcuts around binary layers:
```python
class ResidualBinaryLayer(nn.Module):
    def forward(self, x):
        out = self.bn(x)
        out = binary_activation(out)
        out = self.conv(out)
        return out + self.shortcut(x)  # Bi-Real style residual
```

#### Learnable Scaling Factors
Instead of using fixed scaling factors (mean absolute weight), we make $\alpha$ a learnable parameter per output channel:
- **Standard XNOR-Net**: $W_{bin} = \text{sign}(W) \cdot \text{mean}(|W|)$
- **Optimized**: $W_{bin} = \text{sign}(W) \cdot \alpha$ where $\alpha$ is learned via backpropagation

#### Knowledge Distillation (KD)
The binary student model learns from a high-accuracy FP32 teacher (87.11%):
- **Loss Function**: $L = \alpha \cdot L_{CE} + (1-\alpha) \cdot L_{KD}$
- **Soft Targets**: Student matches teacher's output probability distribution
- **Temperature Scaling**: $T=3.0$ softens the teacher's predictions for better knowledge transfer

### 4. Training & Stabilization
BNNs are notoriously difficult to train. We implemented several critical fixes:
- **BN-Order**: Normalization must happen *after* binary convolution to handle the integer output range.
- **Activation Preserving**: Removed restrictive ReLUs between binary layers to maintain the zero-centered distribution required for effective binarization.
- **Weight Clipping**: Real-valued "latent" weights are clipped to $[-1, 1]$ to keep them responsive to gradients.

---

## 🛠 Setup & Usage

### Prerequisites
- Python 3.9+
- PyTorch
- `setuptools` (for C++ extension build)

### Installation
Build and install the custom bitwise operators:
```bash
pip install .
```

### Training
To train the baseline XNOR-Net on CIFAR-10:
```bash
python3 train.py --model xnor --epochs 20 --lr 0.001
```

To train with Knowledge Distillation (recommended for best accuracy):
```bash
# First train the teacher model
python3 train.py --model baseline --epochs 20 --lr 0.001

# Then train the student with KD
python3 train.py --model xnor --epochs 20 --lr 0.001 --kd --teacher-path checkpoints/baseline_cifar10.pth
```

### Benchmarking
Compare the performance of the standard PyTorch FP32 implementation against our bitwise kernel:
```bash
python3 benchmark.py
```

---

## 📊 Results Summary

The optimized model demonstrates exceptional performance for a fully binarized network:
- **Optimized XNOR-Net**: **81.35%** Test Accuracy (20 Epochs) with Bi-Real + KD
- **Baseline XNOR-Net**: **70.35%** Test Accuracy (20 Epochs)
- **Improvement**: **+11.00%** absolute accuracy gain
- **Inference Gain**: The C++ bitwise kernels utilize bit-level parallelism to perform 64 multiplications/additions in a single operation.

### 5. Inference Speed & Efficiency

We implemented a custom high-performance inference engine to demonstrate the raw speed of bitwise operations.

#### Methodology
-   **Batch Size**: 128 (Processing 128 images simultaneously to maximize throughput).
-   **Hardware**: Apple Silicon (M-series) using **ARM NEON SIMD** instructions.
-   **Technique**:
    -   **Bit-Packing**: Compressing 32-bit floats into 1-bit integers (32x memory reduction).
    -   **SIMD Vectorization**: Using 128-bit registers (`uint8x16_t`) to process 128 binary elements per CPU cycle.
    -   **NHWC Layout**: Optimizing memory access patterns for CPU cache efficiency.

#### Benchmark Results (Single-Threaded Context)
We compared the **Optimized BNN** against the highly polished **PyTorch FP32** kernel. To ensure a fair comparison of algorithmic efficiency, both were restricted to a single CPU core.

| Metric | PyTorch Breakdown | Optimized Bitwise Kernel | Speedup |
| :--- | :--- | :--- | :--- |
| **Logic** | `F.conv2d` (Im2Col + GEMM) | `bitwise_conv2d` (XNOR + Popcount) | - |
| **Kernel Latency** | 237.34 ms | **166.97 ms** | **1.42x FASTER** |
| **Full Model Inference** | 2195.20 ms | **1843.80 ms** | **1.19x FASTER** |
| **Accuracy (CIFAR-10)** | 81.34% | **81.34%** | **Identical** |

> **Result**: The Bitwise BNN is **~42% faster** at the kernel level and **~20% faster** end-to-end (including Python overhead) while maintaining full accuracy. The model is also **32x smaller** in size.

---

## 👥 Authors
- **msritian** (mittalshivam003@gmail.com)
- Implemented with **Antigravity** (Google DeepMind)
