`````markdown
# GPTQ from Scratch (nanoGPT Quantization Artifact)

## 1. What this is

This project is a from-scratch implementation of GPTQ applied to a nanoGPT-style transformer model. It demonstrates how second-order information from a calibration set can be used to quantize linear layers while preserving model accuracy.

---

## 2. Background

Neural network quantization reduces model size and inference cost by representing weights with lower precision (e.g., INT8 instead of FP32).

Two main approaches:

- **PTQ (Post-Training Quantization)**: naive rounding of weights to lower precision. Fast but often causes significant accuracy loss.
- **GPTQ**: uses second-order information (Hessian approximation) to model how weight perturbations affect loss, allowing error-aware quantization.

GPTQ improves over PTQ by propagating quantization error across remaining weights using curvature information.

---

## 3. Implementation Details

This implementation includes:

### PTQ (`ptq.py`)
- Symmetric min-max quantization
- Per-column quantization and dequantization
- INT8-style scaling simulation

### GPTQ (`gptq.py`)
- Collects layer input activations via forward hooks
- Builds Hessian approximation:  
  \( H = 2 X^T X \)
- Applies damping for numerical stability
- Uses Cholesky decomposition to compute inverse Hessian
- Performs column-wise greedy quantization
- Propagates quantization error using:
  \[
  W[:, j+1:] -= err_j \cdot \frac{H^{-1}[j, j+1:]}{H^{-1}[j, j]}
  \]

### Pipeline (`run_gptq.py`)
- Loads pretrained nanoGPT model
- Builds calibration dataset (10 batches)
- Applies GPTQ layer-by-layer

### Benchmark (`benchmark.py`)
- Evaluates model perplexity before and after quantization
- Uses identical calibration/evaluation setup for fair comparison

---

## 4. How to Run

### Run GPTQ quantization:
```bash
python quantization/run_gptq.py
````

### Run benchmark:

```bash
python quantization/benchmark.py
```

---

## 5. Results

Example output from this implementation:

| Model    | Loss   | Perplexity |
| -------- | ------ | ---------- |
| Baseline | 1.8521 | 6.37       |
| GPTQ     | 1.8623 | 6.44       |


| Metric Change   |
| ----------------|
| Δ Loss: +0.0102 |
| Δ PPL: +0.07    |

---

## 6. Key Insight

The core difference between PTQ and GPTQ is that PTQ treats each weight independently, while GPTQ treats quantization as a **global optimization problem over each layer**.

Instead of greedily rounding weights, GPTQ asks:

> “If I quantize this weight, how should the rest of the layer adjust to compensate for the induced error?”

This is captured through the inverse Hessian, which encodes sensitivity between parameters. As a result, GPTQ distributes quantization error intelligently across unquantized weights rather than letting it accumulate locally.

This is why GPTQ achieves near-PTQ speed but significantly better accuracy retention, especially in transformer models where weights are highly correlated.
