# Strategy: Boosting BNN Accuracy on CIFAR-10

Our current XNOR-Net achieves **~70% accuracy**, which is a solid baseline. To reach the next level (80-85%+), we can implement the following advanced techniques.

## 1. Bi-Real Net Residual Connections
Standard residual connections add $x + F(x)$. In BNNs, information loss is so high that **Double Skip Connections** are often used.
- **Concept**: Add a bypass for real-valued activations around every 1-2 binary layers.
- **Benefit**: Preserves the "magnitude" of features that binarization usually destroys.

## 2. Knowledge Distillation (KD)
This is arguably the most effective way to improve BNNs.
- **Concept**: Use a high-accuracy Full-Precision model (e.g., ResNet-20 or VGG-16) as a **Teacher**.
- **Execution**: The BNN (Student) is trained not just on labels, but to match the output probability distribution (and sometimes intermediate feature maps) of the Teacher.
- **Benefit**: The student learns the "nuance" of features from the real-valued model.

## 3. Learnable Scaling Factors (L-XNOR)
Currently, we use $\alpha = \text{mean}(\text{abs}(W))$.
- **Concept**: Make $\alpha$ a learnable parameter (one per output channel) that is optimized via backpropagation.
- **Benefit**: Allows the model to dynamically find the optimal scale for each filter, compensating for the rigidness of $\pm 1$ weights.

## 4. Progressive Binarization
Instead of training binary from scratch:
- **Phase 1**: Binary Weights, Full Precision Activations.
- **Phase 2**: Full Binarization (Weights + Activations).
- **Benefit**: Provides a smoother optimization landscape and prevents the model from getting stuck in bad local minima early on.

---

### Recommended Next Move: **Knowledge Distillation + Bi-Real Connections**
I recommend we start by adding **Residual Connections** and implementing a **Knowledge Distillation** setup. This typically provides a 5-10% absolute accuracy boost.

**Would you like me to start by updating our architecture with Bi-Real residual connections?**
