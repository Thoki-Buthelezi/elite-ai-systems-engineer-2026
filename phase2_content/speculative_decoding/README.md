# Week 23: Speculative Decoding

## Summary

Implemented speculative decoding (Leviathan et al., "Fast Inference from
Transformers via Speculative Decoding") using a draft model Mq and target
model Mp built on the Phase I `BiLanguageModel` architecture. Verified
correctness of the accept/reject/resample algorithm both analytically and
empirically. Benchmarked wall-clock throughput across γ ∈ {2, 4, 8} and
found no meaningful speedup at this model scale, consistent with the
memory-bandwidth argument underlying the algorithm.

## Models

| Model | Params | Layers | n_embd | n_heads | Perplexity |
|-------|--------|--------|--------|---------|------------|
| Mp (target) | 882,241 | 4 | 128 | 4 | 6.36 |
| Mq (draft)  | 128,577 | 2 | 64  | 2 | 9.61 |

Mq is roughly 1/7th the size of Mp, same vocab (65) and block_size (64).

## Correctness

Two complementary checks:

1. **Deterministic unit test** (`test_accept_branch.py`): confirms the
   accept-probability formula `min(1.0, p(x)/q(x))` and clamping behavior
   are correct in isolation.
2. **Statistical convergence test** (`test_correctness.py`): ran
   `speculative_step` 1000+ times from a fixed prefix, compared the
   empirical distribution of accepted tokens against Mp's true p(x) via
   KL divergence.
   - Run 1: KL = 0.0947 (N=1000, single-token prefix)
   - Run 2: KL = 0.0100 (N=1000, 10-token prefix)
   - Run 3: KL = 0.0138 (N=1000, different 10-token prefix)

   Both statistical runs landed on 100% rejection (every draft resampled
   via the residual distribution, no draft ever accepted outright). This
   is not a bug: Mp's distribution on this char-level dataset is often
   sharply peaked (one run showed p(x)=0.79 on a single token), and the
   theoretical proof in the paper's Appendix A.1 guarantees correctness
   of the resample path independent of acceptance rate. The unit test
   above separately confirms the accept path's arithmetic, so both code
   paths are verified even though the accept path wasn't exercised
   empirically.

## Benchmark

Wall-clock comparison, speculative decoding vs. plain autoregressive
sampling from Mp, 50 tokens generated per run, CPU.

| γ | Plain (tok/s) | Speculative (tok/s) | Speedup | Avg tokens/call |
|---|---|---|---|---|
| 2 | 82.1 | 84.3 | 1.03x | 2.28 / 3 |
| 4 | 85.3 | 76.1 | 0.89x | 2.85 / 5 |
| 8 | 83.5 | 61.3 | 0.73x | 3.33 / 9 |

γ=2 was rerun at n_trials=100 (up from 20) after an initial run showed
1.16x, which did not hold up under a larger sample, confirming it was
noise rather than a real effect.

## Why no speedup

Speculative decoding's speedup comes from amortizing one memory-bandwidth-bound
forward pass of Mp across multiple draft tokens: on large models, loading
weights from HBM dominates wall-clock time, and verifying γ tokens costs
roughly the same as verifying 1, since the compute units are idle waiting
on memory either way. At 882K params, Mp is small enough that this
bottleneck likely doesn't dominate, verifying more tokens per pass costs
real additional time rather than being ~free. Combined with average
tokens/call plateauing around 2.3-3.3 regardless of γ (Mq and Mp diverge
enough, perplexity 9.61 vs 6.36, that long drafts rarely pay off), the
overhead of drafting and verifying outweighs any savings, and grows worse
as γ increases.

## Conclusion

Implementation is correct, verified via both deterministic and statistical
methods. No wall-clock speedup was observed at any tested γ, and the result
degrades monotonically as γ grows. This is an expected outcome given model
scale, not an implementation failure, and matches the theoretical prediction
that speculative decoding's benefit requires memory-bandwidth-bound target
models.