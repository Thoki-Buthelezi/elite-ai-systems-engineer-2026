# FSDP vs DDP: Memory, Speed, and Cost Comparison

**Week 27-28 — Phase III: FSDP & Tensor Parallelism**
Model: 1.42B parameter GPT-2 XL-scale decoder-only transformer (24 layers, 2048 embd, 16 heads, 512 seq len)
Hardware: 2x NVIDIA T4 (16GB VRAM each), Kaggle notebook environment

## Summary

DDP replicates the full model, gradients, and optimizer state on every GPU.
At this model size, that means each T4 would need to hold roughly 22.7GB
(1.42B params x 4 bytes x 4, for params + gradients + Adam's two moment
buffers)  well past the T4's 16GB ceiling. FSDP instead shards parameters,
gradients, and optimizer state across the two GPUs at rest, only
reconstructing the full weights for one transformer block at a time via
`all_gather`, then discarding them immediately after use.

The experiment confirms this directly: **DDP could not complete a single
training step and hit CUDA OOM. FSDP trained successfully.**

## Results

| Metric | DDP | FSDP |
|---|---|---|
| Params held per GPU | 1.416B (full replica) | 0.708B (sharded ~half) |
| Peak memory (rank 0) | 15.32GB (OOM before step completed) | 14.66GB |
| Completed training steps | 0 / 5 | 5 / 5 |
| Avg step time | N/A | 3.22s |
| Throughput | N/A | 635.1 tokens/sec |
| Cost per successful run | N/A — model unusable on this hardware under DDP | ~$0.0006 (16s wall-clock x 2 GPUs @ ~$0.35/hr per T4 on-demand) |

Cost is a rough on-demand estimate using a commonly cited T4 hourly rate
(~$0.35/hr per GPU); actual Kaggle GPU time is free within quota, this
figure is included to make the comparison portable to a paid cloud
setting.

## What the memory numbers actually mean

- **DDP's `n_params_billions: 1.416`** confirms every rank held the full,
  unsharded model. This is expected for DDP by design — it trades memory
  for simplicity and minimal communication (one gradient `all_reduce` per
  backward pass, no repeated gather/discard cycle).
- **FSDP's `n_params_billions: 0.708`** confirms parameter sharding is
  working exactly as expected , each of the 2 GPUs holds roughly half the
  1.42B total at rest. Full block weights are only reconstructed
  transiently during forward/backward via `all_gather`, then freed.
- FSDP's peak memory (14.66GB) still leaves under 1.5GB of headroom on a
  16GB T4 at this model size and batch size , sharding made the model
  *trainable*, not comfortable. A larger model or bigger batch would
  likely need `CPU offload` or `SHARD_GRAD_OP` (which trades some memory
  back for less communication) to stay within budget.

## Why DDP fails here and FSDP doesn't

DDP requires every rank to hold: full parameters + full gradients + full
optimizer state (Adam keeps two additional moment buffers per parameter).
For a 1.42B parameter model in fp32, that's roughly:

- Parameters: 1.42B x 4 bytes ≈ 5.7GB
- Gradients: ≈ 5.7GB
- Adam optimizer state (2x params): ≈ 11.3GB
- **Total per GPU: ≈ 22.7GB** —> exceeds the T4's 16GB before any
  activation memory is even accounted for.

FSDP shards all three of those (params, gradients, optimizer state)
across the 2 GPUs, so each GPU's baseline footprint is roughly half:
≈ 11.3GB before activations, comms buffers, and forward/backward
`all_gather` overhead , which is what pushed the observed peak up to
14.66GB.

## Sharding strategy note

This experiment used `ShardingStrategy.FULL_SHARD` , the most aggressive
option, resharding parameters immediately after both forward and
backward. `SHARD_GRAD_OP` keeps full parameters resident through backward
(skipping the second `all_gather`), trading higher memory for less
communication. At this model size, `SHARD_GRAD_OP` would likely push peak
memory close to or past the 16GB ceiling — worth testing as a follow-up
if communication overhead becomes the bottleneck at larger scale.

## Caveats

- Only 5 training steps were run (4 timed, after a warmup step) , enough
  to confirm mechanics and get a stable throughput reading, not enough
  for a statistically rigorous benchmark.
- Dummy random token data was used; the loss curve is not meaningful and
  wasn't the point of this experiment.
- The DDP failure is model-size-specific to this hardware. A smaller
  model (e.g. halving `n_layer` and `n_embd`) would let both strategies
  complete, enabling a genuine throughput comparison rather than
  DDP-fails-outright , a useful follow-up experiment if a like-for-like
  speed comparison is needed.

## Artifacts

- `model.py` — GPT model definition
- `benchmark.py` — DDP/FSDP benchmark harness (`torchrun --nproc_per_node=2 benchmark.py --mode {ddp|fsdp}`)
