# Week 3 Report — nanoGPT Reproduction

## Artifact
Character-level GPT trained on tinyShakespeare. Implemented from scratch
without reference during coding.

## Training results
| Step | Train loss | Val loss |
|------|-----------|---------|
| 0    | 4.3300    | 4.3314  |
| 1000 | 2.4087    | 2.4133  |
| 2500 | 2.1116    | 2.1399  |
| 5000 | 1.9014    | 1.9756  |

Train/val gap at convergence: ~0.08. No overfitting observed.

## Key findings

**Pre-norm vs post-norm residual connections**
My first submission had the LayerNorm applied incorrectly inside the Block:

```python
# Wrong
x = self.ln1(x)
x = x + self.sa_head(x)
```

The correct pre-norm pattern applies the norm *inside* the residual branch:

```python
# Correct
x = x + self.sa_head(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```

Pre-norm (used here, deviating from the original "Attention Is All You Need" 
paper) places the LayerNorm before each sublayer rather than after. This 
stabilizes training by keeping the residual stream clean — gradients flow 
through the skip connection without passing through a norm operation, which 
reduces the risk of exploding/vanishing gradients at larger scales.

**Attention scaling**
Initial implementation scaled by `n_embd**-0.5` instead of `head_size**-0.5`.
Since dot products are computed in head_size space (32), not full embedding 
space (128), the correct scale factor is `head_size**-0.5`. Using the wrong 
dimension under-scales attention weights by 2x.

## Bugs fixed from supervisor review
- Residual connection order in Block.forward
- Attention scaling factor (C -> head_size)
- Removed unused sa_head and ffwd attributes from BiLanguageModel
- Renamed var_data -> val_data for clarity

## Next week
Week 4 — LoRA implementation. Fine-tune a pretrained model using low-rank 
adaptation. Target artifact: before/after generation comparison + parameter
count reduction analysis.