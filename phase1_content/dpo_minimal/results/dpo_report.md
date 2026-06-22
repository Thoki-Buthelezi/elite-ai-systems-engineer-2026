# Week 7–8 Report — SFT + DPO on nanoGPT (Tiny Shakespeare)

## 1. Overview

This week implemented a full alignment pipeline on top of a pretrained nanoGPT character-level language model trained on Tiny Shakespeare. The pipeline consisted of three stages: a pretrained base model, supervised fine-tuning (SFT) on synthetic preference pairs, and Direct Preference Optimization (DPO) using a frozen SFT reference model. The goal was to implement DPO from scratch, understand its training dynamics, and produce a benchmark comparison between SFT and DPO.

---

## 2. Model Architecture

The base model is a decoder-only Transformer (nanoGPT) with the following configuration:

| Hyperparameter   | Value |
|------------------|-------|
| block_size       | 64    |
| embedding size   | 128   |
| attention heads  | 4     |
| layers           | 3     |
| dropout          | 0.2   |

The model was pretrained on Tiny Shakespeare using standard next-token prediction before any alignment training.

---

## 3. Preference Dataset

A synthetic preference dataset was constructed with the prompt `"KING:"` and three chosen/rejected pairs each. Chosen responses are grammatically structured Shakespeare-like sentences. Rejected responses are informal or incoherent variations.

**Example pair:**
- Chosen: `"My lords, attend and hear my royal will."`
- Rejected: `"My lord, I guess things are kinda okay maybe."`

Batch size was 32, with sequences padded or truncated to `block_size`. Padding used token ID 0, masked out during log-probability computation to avoid polluting the sequence score.

---

## 4. Sequence Log-Probability

Sequence-level log probabilities were computed as follows:

1. Forward pass to obtain logits of shape `(B, T, C)`
2. Shift: `logits[:, :-1, :]` and `targets = idx[:, 1:]`
3. Apply `log_softmax` over vocabulary dimension `(dim=-1)`
4. Use `gather` to select the log probability of the actual next token at each timestep
5. Apply padding mask and take the mean across the sequence dimension

Mean normalisation was used instead of sum to remove length bias — without this, longer sequences accumulate more negative log probability and DPO would be driven by length differences rather than preference quality.

---

## 5. DPO Objective

The DPO loss is:

$$\mathcal{L}_{DPO} = -\log \sigma \left( \beta \left[ (\log \pi_\theta(y_c) - \log \pi_\theta(y_r)) - (\log \pi_{ref}(y_c) - \log \pi_{ref}(y_r)) \right] \right)$$

Where:
- $\pi_\theta$ is the trainable policy model
- $\pi_{ref}$ is the frozen SFT reference model
- $y_c$ is the chosen response, $y_r$ is the rejected response
- $\beta = 0.01$ controls the strength of preference optimisation

The reference model acts as an anchor — it measures how much the policy's preference exceeds the reference model's preference, preventing over-optimisation and distribution collapse.

---

## 6. Training Configuration

| Parameter       | SFT     | DPO     |
|-----------------|---------|---------|
| iterations      | 2000    | 2000    |
| learning rate   | 5e-6    | 5e-6    |
| beta            | —       | 0.01    |
| batch size      | 32      | 32      |
| optimiser       | AdamW   | AdamW   |

Both models were initialised from the same pretrained nanoGPT checkpoint. The reference model for DPO was the same checkpoint, kept frozen throughout training.

---

## 7. Training Dynamics

### SFT

```
step 0    | loss 1.9327 | margin 0.1110
step 200  | loss 1.1820 | margin 0.6851
step 400  | loss 0.8245 | margin 1.0115
step 600  | loss 0.5602 | margin 1.4095
step 800  | loss 0.3465 | margin 1.6893
step 1000 | loss 0.2255 | margin 1.9620
step 1200 | loss 0.1610 | margin 2.0030
step 1400 | loss 0.1282 | margin 1.9388
step 1600 | loss 0.1018 | margin 2.1192
step 1800 | loss 0.0866 | margin 2.0761
```

SFT loss decreased smoothly from 1.93 to 0.09. Margin grew steadily to ~2.1 and stabilised, indicating the model learned to assign higher probability to chosen responses without over-optimising.

### DPO

```
step 0    | loss 0.6932 | margin 0.1769
step 200  | loss 0.6849 | margin 1.7765
step 400  | loss 0.6605 | margin 6.7491
step 600  | loss 0.6432 | margin 10.3229
step 800  | loss 0.6377 | margin 11.5685
step 1000 | loss 0.6351 | margin 12.0860
step 1200 | loss 0.6315 | margin 12.8937
step 1400 | loss 0.6292 | margin 13.3847
step 1600 | loss 0.6263 | margin 13.9525
step 1800 | loss 0.6242 | margin 14.4462
```

DPO loss converged to ~0.69 (near `log(2)`, the theoretical minimum for a random policy) and margin grew continuously to 14.4. The large margin indicates the policy has learned a strong preference separation between chosen and rejected responses.

---

## 8. Benchmark Results

| Model | logP(chosen) | logP(rejected) | Margin  |
|-------|--------------|----------------|---------|
| SFT   | -0.0463      | -2.3749        | 2.3286  |
| DPO   | -1.5009      | -16.8353       | 15.3344 |

**Key observations:**

- DPO margin (15.33) is 6.6x larger than SFT margin (2.33), confirming the preference training is working as intended.
- SFT logP(chosen) of -0.05 is very high — the model assigns near-certain probability to chosen responses on this small synthetic dataset, suggesting possible overfitting to the 3 chosen templates.
- DPO logP(rejected) of -16.84 is extremely low, meaning the policy has learned to strongly suppress rejected responses. This is the correct direction but the magnitude suggests the model may be over-optimised on the toy dataset.

---

## 9. Generation Samples

### Base nanoGPT (no fine-tuning)
```
Ayighd VusoWhere dayer, hour of of madores:
The hae seter is chere of buth you me not poting!
LUCEON OF INGMIS:
O, to my seeveld loper, their he
To is thou pry's echund in the have gave of Rombe.
```

### SFT model
```
Fromstrem, plainly, for I would know the truth.
MELING:
Stheed thI my forfull. Nurstr, Comaill.
Whou wrutield, the frimbleak fill.
Wouldou my lovest the fielfeaing the byour from thuthe plain?
Whear
```

**Observation:** SFT generation shows partial memorisation of training phrases (e.g. "plainly, for I would know the truth" directly echoes a chosen response). Structure is partially preserved but character-level noise remains high. The base model arguably produces more diverse output, suggesting the SFT fine-tuning is narrowing the distribution toward the 3 training templates rather than generalising.

DPO generation was not included as the model exhibited token collapse in earlier runs — this is documented in the failure modes section below.

---

## 10. Failure Modes

### Attempt 1 — DPO collapse
In the first run, DPO produced:
```
eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
```
The policy collapsed to a single high-frequency token. Diagnosis: the SFT baseline was too weak (loss had not converged sufficiently) and the preference pairs were too easy, giving the model a trivial optimisation shortcut.

### Attempt 1 — Oscillating margin
DPO margin oscillated around zero and went negative in the first run, indicating the policy was assigning higher probability to rejected responses than chosen ones on some steps. Root cause: the underlying distribution was too noisy to provide a stable preference signal.

### Auxiliary loss sign error
An auxiliary KL-like term `0.01 * (logp_c - logp_ref_c).mean()` was added to regularise the policy. This term had the wrong sign — it penalised the policy for assigning higher probability to chosen responses than the reference model, partially fighting the DPO objective. This was identified as a bug during implementation review.

---

## 11. Implementation Bugs Fixed

1. **`load_state_dict` reassignment** — `model = model.load_state_dict(...)` returns `None`. The assignment was removed so the in-place load is used correctly.
2. **Variable name mismatch** — `log_rec_r` was defined but `logp_ref_r` was passed to `dpo_loss`, silently feeding wrong values into the reference model terms. Fixed by consistent naming.
3. **Wrong import** — `from torch import functional as F` does not exist. Fixed to `import torch.nn.functional as F`.
4. **`F.logsigmoid` with `dim` argument** — `logsigmoid` is elementwise and does not accept a `dim` parameter. The argument was removed.
5. **`optimizer` wrapping model directly** — `AdamW(model, lr)` should be `AdamW(model.parameters(), lr=lr)`.

---

## 12. Key Insights

**Insight 1 — SFT quality is a hard bottleneck.** DPO reshapes an existing distribution; it cannot create one. A weak SFT baseline produces oscillating margins and eventual collapse because the policy has no stable probability mass to redistribute.

**Insight 2 — Preference difficulty determines signal quality.** Trivially easy rejected samples (random text) allow the model to saturate the loss without learning general preference structure. Harder, stylistically similar negatives produce more informative gradients.

**Insight 3 — Mean vs sum normalisation matters.** Summing log probabilities biases the objective toward longer sequences. Mean normalisation ensures the margin reflects per-token preference quality, not sequence length.

**Insight 4 — The reference model is an anchor, not a regulariser.** It prevents the policy from over-optimising by measuring how much the policy's preference exceeds the reference's. Without it, `pi_diff` can grow unboundedly, driving loss to zero and gradients to vanish.

---

## 13. Next Steps

- Verify SFT generation quality matches or exceeds base nanoGPT before running DPO — SFT should not degrade the base distribution.
- Expand preference dataset beyond 3 pairs per class to reduce memorisation and improve generalisation.
- Fix auxiliary loss sign or replace with proper SFT mixing loss: `loss = dpo_loss(...) + sft_weight * (-logp_c.mean())`.
- Scale model (embedding 256, 4–6 layers) and extend SFT pretraining to confirm findings hold beyond toy scale.
- Add temperature and top-k sampling to generation to better evaluate output quality.

---

## 14. Artefacts

- `dpo_minimal.py` — full implementation in monorepo
- `nano_gpt_model.pt` — pretrained base checkpoint
- `sft_final_model.pt` — SFT fine-tuned checkpoint