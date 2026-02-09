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
Compare the performance across three engines (PyTorch, ONNX, and Custom Bitwise):
```bash
# Optimized Kernel Micro-benchmark
python3 benchmark.py

# Full 3-Way End-to-End Comparison (Accuracy + Speed)
python3 test_cifar_inference.py
```

---

## 📊 Results Summary

The optimized model demonstrates exceptional performance for a fully binarized network:
- **Optimized XNOR-Net**: **81.35%** Test Accuracy (20 Epochs) with Bi-Real + KD
- **Baseline XNOR-Net**: **70.35%** Test Accuracy (20 Epochs)
- **Improvement**: **+11.00%** absolute accuracy gain
- **Inference Gain**: The C++ bitwise kernels utilize bit-level parallelism to perform 64 multiplications/additions in a single operation.

### 5. Multi-Engine Benchmark Results
We conducted a rigorous comparison between the **Custom Bitwise Kernel**, **PyTorch (Simulated)**, and **ONNX Runtime** on the full CIFAR-10 test set (10,000 images).

#### Hardware Context
- **CPU**: Apple Silicon (ARM64)
- **Threading**: Locked to **1 CPU Thread** for a fair algorithmic "Fair Fight".
- **Batch Size**: 128 (Chunking 128 images per forward pass for throughput).

#### Performance Matrix
| Model / Engine | Version | Accuracy | Avg Latency | Peak RAM | Weight Size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SimpleNet (Teacher)** | Shallow FP32 | **87.11%** | **616.89 ms** | **575.77 MB** | 8.00 MB |
| **OptimizedFP32Net** | **Fair FP32** | -- | **1503.13 ms** | **586.17 MB** | 9.20 MB |
| Optimized BNN (PyTorch) | Simulated | 81.35% | 2045.37 ms | 779.69 MB | 9.20 MB |
| Optimized BNN (ONNX) | Optimized KD | 81.35% | 2201.82 ms | 627.03 MB | 9.20 MB |
| **Optimized BNN (Bitwise)**| **Custom NEON** | 81.34% | **1735.30 ms*** | 790.02 MB | **0.28 MB (32x)!!** |

#### Key Insights
1.  **Engine Supremacy**: For the *exact same model* (`OptimizedXNORNet`), our handcrafted **Bitwise Kernel** is the fastest, beating PyTorch (Simulated) by **1.18x** and ONNX Runtime by **1.27x**.
2.  **Architecture Depth**: The **FP32 Teacher** is fastest primarily because it is a **shallower 3-layer network**. The true comparison for the BNN's 7-layer graph is the **OptimizedFP32Net**, which is only ~15% faster than the Bitwise BNN despite decades of optimization in native FP32 libraries.
3.  **The RAM Trade-off**: Interestingly, the BNN uses more runtime RAM than the FP32 models. This is due to the **Python-level buffer management** and explicit bit-packing required by the custom inference loop.
4.  **Storage Dominance**: The BNN achieves an absolute **32x reduction** in storage (0.28 MB vs 9+ MB), enabling deep models to run on devices where FP32 weights simply will not fit.

> [!IMPORTANT]
> \* The Bitwise speedup is **1.42x** at the core kernel level. The total end-to-end BNN inference is currently bottlenecked by Python packing logic. Moving the entire graph and residual flow into C++ would likely allow the BNN to match or exceed FP32 speed even on high-end CPUs.

---

## 🏗 Knowledge & Implementation Details
*(Remaining sections...)*

---

## 👥 Authors
- **msritian** (mittalshivam003@gmail.com)
- Implemented with **Antigravity** (Google DeepMind)
