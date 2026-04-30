# Week Notes — SFT + DPO on nanoGPT (Tiny Shakespeare)

## 1. Overview

This week focused on implementing and analyzing a full pipeline consisting of:

- A baseline autoregressive Transformer (nanoGPT-style)
- Supervised Fine-Tuning (SFT) on Tiny Shakespeare
- Direct Preference Optimization (DPO) using synthetic preference pairs
- Investigation of training dynamics, stability, and generation quality

The main goal was to understand how sequence modeling, log-probabilities, and preference optimization interact in practice.

---

## 2. Model Architecture (SFT Base)

The base model is a decoder-only Transformer:

### Key components:
- Token embedding table
- Positional embedding table
- Multiple Transformer blocks:
  - LayerNorm → Multi-Head Self Attention → Residual connection
  - LayerNorm → Feedforward network → Residual connection
- Final linear projection to vocabulary logits

### Hyperparameters:
- block_size = 64
- embedding size = 128
- number of heads = 4
- number of layers = 3
- dropout = 0.2

---

## 3. Training Objective (SFT)

The supervised training objective is standard next-token prediction:

\[
\mathcal{L}_{SFT} = - \sum_t \log P(x_{t+1} | x_{\leq t})
\]

### Observations:
- Loss decreases smoothly from ~4.3 to ~1.9
- Model learns basic Shakespeare-like structure
- Output remains noisy and partially corrupted at character level
- Long-range coherence is weak due to small model capacity and dataset size

---

## 4. Sequence Log-Probability Computation

To support DPO, sequence-level log-probabilities were implemented:

### Key steps:
1. Forward pass to obtain logits
2. Shift logits and targets:
   - logits[:, :-1, :]
   - targets = idx[:, 1:]
3. Apply log-softmax over vocabulary dimension
4. Gather log-probability of correct tokens
5. Aggregate across sequence (mean or sum)

### Important detail:
- Using `dim=-1` ensures softmax is applied over vocabulary distribution
- `gather()` selects probability of correct token at each timestep

---

## 5. Preference Dataset (Synthetic DPO Setup)

A synthetic preference dataset was constructed:

### Structure:
- Prompt: `"KING:"`
- Chosen responses: grammatically structured Shakespeare-like sentences
- Rejected responses: simplified, informal, or nonsensical variations

### Example:
- Chosen: "My lords, attend and hear my royal will."
- Rejected: "My lord, I guess things are kinda okay maybe."

### Key insight:
- Early experiments used overly easy negatives (random text)
- Later improved to more realistic stylistic differences

---

## 6. DPO Objective

DPO optimizes preference ranking using both policy and reference model:

\[
\mathcal{L}_{DPO} = - \log \sigma \left( \beta \left[ (\log \pi(y_c) - \log \pi(y_r)) - (\log \pi_{ref}(y_c) - \log \pi_{ref}(y_r)) \right] \right)
\]

### Components:
- π: trainable policy model
- π_ref: frozen SFT model
- y_c: chosen response
- y_r: rejected response
- β: scaling factor controlling strength of optimization

---

## 7. Training Dynamics Observed

### Initial issues:
- Loss rapidly collapsed to near zero
- Margin grew excessively large
- Output degraded into repetition or collapsed tokens

### Causes:
- Preference pairs too easy (trivial separation)
- Lack of regularization
- Excessive optimization pressure

### Improved setup:
- Harder preference pairs
- Reduced learning rate (5e-6)
- Lower beta (0.005)
- Removed conflicting auxiliary KL term

### Result:
- Stable decrease in loss (~0.69 → ~0.56)
- Gradual increase in margin (~0 → ~12)
- More stable optimization behavior

---

## 8. Generation Behavior Analysis

### SFT model output:
- Partially coherent Shakespeare-like structure
- High character-level noise
- Weak spelling consistency
- Moderate syntactic structure

### DPO model output:
- Initial collapse (repeated characters, e.g., "eeeeeee")
- Later instability with corrupted tokens
- Occasional structured phrases mixed with noise

---

## 9. Key Failure Modes Identified

### 1. Token-level collapse
- Repetition of high-probability tokens
- Caused by over-optimization without strong language constraints

### 2. Distribution drift
- Model deviates from learned language manifold
- Preference optimization overwhelms language modeling prior

### 3. Over-simplified preference signal
- Easy rejection examples lead to trivial optimization
- Model learns shortcuts instead of general preference structure

---

## 10. Key Fixes Applied

- Normalized sequence log-probability (mean instead of sum)
- Reduced learning rate (1e-4 → 5e-6)
- Lowered beta (0.1 → 0.005)
- Improved rejected samples (more realistic negatives)
- Removed conflicting auxiliary KL term
- Ensured reference model remains frozen and identical to SFT checkpoint

---

## 11. Core Insights

### Insight 1: SFT quality is a bottleneck
DPO cannot fix a weak language model. It only reshapes an already meaningful distribution.

### Insight 2: Preference difficulty matters
If chosen/rejected pairs are too easy, DPO saturates and loses learning signal.

### Insight 3: Regularization is essential
Without explicit or implicit KL constraints, DPO causes distribution drift and collapse.

### Insight 4: Mean vs sum matters
Sequence normalization significantly affects gradient scale and training stability.

---

## 12. Current Status

### SFT model:
- Functionally correct
- Weak generation quality
- Suitable as a baseline, not production-quality

### DPO implementation:
- Correct mathematically
- Sensitive to hyperparameters and data quality
- Requires stronger SFT baseline for stable performance

---

## 13. Next Steps

### Model improvements:
- Increase model size (embedding 256–384, more layers)
- Train longer for better SFT convergence
- Improve decoding (temperature, top-k sampling)

### DPO improvements:
- Use harder, more realistic preference pairs
- Add KL regularization or SFT mixing loss
- Scale dataset diversity

### System-level goal:
- Establish stable SFT baseline before further alignment experiments
- Transition from toy-level behavior to controlled generation quality

---

## 14. Summary

This week demonstrated a full end-to-end understanding of:

- Transformer language modeling
- Sequence log-probability computation
- Supervised fine-tuning dynamics
- Direct Preference Optimization mechanics
- Stability issues in alignment training

The main takeaway is that alignment methods like DPO depend critically on the strength and stability of the underlying language model. Without a strong SFT baseline and properly constructed preference data, optimization quickly leads to collapse or degenerate token distributions.