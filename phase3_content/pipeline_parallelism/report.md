# Pipeline Parallelism (GPipe) Benchmark

**Week 27-28 — Phase III, Elite AI Systems Engineer program**

## Scope

Original plan was to compose pipeline parallelism with the existing FSDP harness on
the GPT-2 XL model. On a 2x T4 setup, that composition doesn't actually work:
splitting into K=2 pipeline stages puts one stage per GPU, leaving no GPU left within
a stage for FSDP to shard across. FSDP needs multiple ranks per shard group to shard
parameters at all; at group size 1 it's a no-op. 

Given the 2-GPU ceiling, this benchmark is scoped as **pure pipeline parallelism**,
validating the mechanics from the GPipe paper (micro-batch scheduling, bubble
overhead, activation re-materialization) rather than claiming a 3D-parallelism-style
composition. That's a deliberate scope decision, not a shortfall.

## Implementation

- **API:** `torch.distributed.pipelining`, using `ScheduleGPipe` (the built-in
  fill-drain schedule matching the paper) rather than a from-scratch implementation.
- **Model splitting:** manual, via `PipelineStage`. The custom GPT model doesn't fit
  the tracer-based `pipeline()` path cleanly, so stages are built by deleting the
  layers/embeddings/head that don't belong to each half:

  ```python
  if stage_index == 0:
      del model.blocks[n_layer // 2:]
      model.ln_f = None
      model.head = None
  elif stage_index == 1:
      del model.blocks[:n_layer // 2]
      model.tok_emb = None
      model.pos_emb = None
  ```

- **Loss:** moved out of the model entirely (`forward` returns hidden states / logits
  only) and passed to `ScheduleGPipe` as a separate `loss_fn`, since a pipeline
  stage's forward must return a single tensor, and only the last stage has access to
  targets.
- **Microbatch count:** M ≥ 4K is negligible-bubble territory per GPipe's own
  empirical finding; with K=2, `num_microbatches=8`.

## Model configuration

The original plan was to run this at GPT-2 XL scale (~1.3B params, same config as the
FSDP baseline) to keep the two benchmarks directly comparable. That hit a wall:
without FSDP sharding the optimizer state, each pipeline stage carries its full,
unsharded AdamW state. At fp32, that's ~16 bytes/param (4 each for weight, gradient,
`exp_avg`, `exp_avg_sq`) before a single activation tensor exists. At GPT-2 XL scale
split across 2 stages, that alone consumes essentially the full 14.56GB T4 budget,
confirmed empirically: the run OOM'd on step 1 (AdamW allocates its state lazily on
the first `optimizer.step()`, so step 0 completes before the wall is hit).

**Finding worth stating plainly:** plain pipeline parallelism with no parameter
sharding hits a memory wall on optimizer state well before FSDP does at the same
model size, on this hardware. FSDP's sharding is solving a problem pipeline
parallelism alone doesn't address.

The model was scaled down to fit within budget:

| | FSDP baseline | Pipeline benchmark |
|---|---|---|
| n_layer | 24 | 16 |
| n_head | 16 | 12 |
| n_embd | 2048 | 1536 |
| block_size | 512 | 512 |
| batch_size | 2 | 8 |
| Total params | ~1.3B | ~0.305B |

**Caveat:** because of this, throughput and memory numbers below are *not* a clean
apples-to-apples comparison with the FSDP baseline, they compare different model
sizes on the same hardware, not the same model under two parallelism strategies. The
useful comparison is qualitative (what each strategy makes possible, what it costs)
rather than a direct number-vs-number one.

## Results

### Correctness

Initial loss was 10.99, close to the theoretical random-init cross-entropy of
`ln(50257) ≈ 10.82`. Loss then decreased monotonically over 5 steps (10.99 → 9.29 →
7.42 → 6.78 → 6.02).

**Important nuance:** the training batch is fixed (generated once, reused every
step), not resampled. This loss curve confirms the gradient path through both
pipeline stages is wired correctly, it is *not* evidence of real learning dynamics,
since the model is simply overfitting a single memorized batch.

### Memory and throughput

```json
{
  "num_stages": 2,
  "n_params_billions": 0.305,
  "peak_mem_gb_by_stage": [8.572, 10.366],
  "avg_step_time_s": 2.3475,
  "tokens_per_sec": 1744.9,
  "batch_size": 8,
  "block_size": 512
}
```

### Stage imbalance

Stage 1 is both memory-heavier (10.366GB vs 8.572GB, ~1.8GB more) and
compute-slower (0.249s vs 0.195s per isolated microbatch, ~28% more) than stage 0.

This is **not** explained by parameter count — `tok_emb + pos_emb` (stage 0, ~78.0M
params) and `head` (stage 1, ~77.2M params) are nearly identical, a difference of
under 1M params. The actual driver is where the loss computation lives: stage 1's
`loss_fn` operates on a `(4096, 50257)`-shaped logits tensor (~823MB in fp32), plus
the `log_softmax` intermediate of the same size, both of which autograd retains for
backward. Stage 0 never touches a vocab-dimensional tensor.

**Takeaway:** balancing a pipeline by layer count or parameter count alone is
insufficient — where the loss lives matters, and in this case it made the last stage
the effective memory and compute ceiling for the whole pipeline.

### Bubble overhead

| | Value |
|---|---|
| Predicted bubble fraction, (K-1)/(M+K-1), K=2, M=8 | 11.1% |
| Observed bubble fraction | 15.3% |

The isolated per-stage compute times (0.195s / 0.249s) were used with `max()` as the
ideal-schedule baseline, so stage imbalance is already accounted for in the
prediction — it isn't double-counted into the gap. The remaining ~4 percentage
points is attributable to costs the isolated timing doesn't capture: the
`optimizer.step()` call and NCCL send/recv between the two GPUs, both real parts of
`avg_step_time` but absent from the isolated compute measurement. GPipe's structural
bubble prediction held reasonably well; the gap is explained, not a discrepancy with
the paper's formula.

## Summary

- Pipeline parallelism was implemented and validated correctly end to end using
  `torch.distributed.pipelining` / `ScheduleGPipe`, scoped deliberately to pure PP
  on 2 GPUs rather than a PP+FSDP composition the hardware can't meaningfully support.
- Plain pipeline parallelism hits an optimizer-state memory wall well before FSDP
  does at the same model size — a concrete demonstration of what FSDP's sharding is
  actually buying.
- Pipeline stages are not symmetric even at equal layer count: the stage holding the
  loss computation carries a real memory and compute cost the other stage doesn't.
- The observed bubble overhead (15.3%) was reasonably close to the GPipe paper's
  predicted value (11.1%), with the gap attributable to optimizer and communication
  overhead outside the pipeline schedule itself.

## Open for future work

- Re-materialization (activation checkpointing) on/off comparison, to empirically
  measure the memory/compute tradeoff `ScheduleGPipe` should expose.
