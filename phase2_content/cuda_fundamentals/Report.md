# Week 15 — CUDA Fundamentals
**Date:** May 2026
**Phase:** II — GPU Systems
**Status:** ✅ Complete — 1 week ahead of schedule

---

## What I Built

10 progressively harder CUDA programs written and run
on Kaggle (T4 GPU), covering the full core of CUDA programming:

| # | Problem | Key Concept |
|---|---------|-------------|
| 1 | Vector Multiply | Explicit memory management |
| 2 | Array Reversal | Thread index mapping |
| 3 | Find Maximum | Shared memory reduction (thread 0) |
| 4 | Matrix Transpose | 2D thread indexing, shared memory |
| 5 | Dot Product | atomicAdd |
| 6 | Parallel Prefix Sum | Stride up pattern |
| 7 | Matrix Multiplication | Tiled shared memory matmul |
| 8 | Parallel Reduction Sum | Stride down pattern |
| 9 | Stencil / Halo Average | Halo elements, neighbour access |
| 10 | Stream Overlap | cudaStream, cudaMemcpyAsync |

---

## Key Concepts Learned

### Memory Hierarchy
CUDA has multiple physically distinct memory spaces.
Choosing the right one is the biggest performance decision:

| Memory | Scope | Speed | Size |
|--------|-------|-------|------|
| Registers | Thread | Fastest | Tiny |
| Shared | Block | Very fast | ~48KB |
| Global | All threads | Slow | GBs |
| Constant | All (read only) | Fast | 64KB |
| Unified | CPU + GPU | Variable | GBs |

### Thread Hierarchy
Threads → Blocks → Grid. Each thread finds its data
using its global ID:
```c
int gid = blockIdx.x * blockDim.x + threadIdx.x;
```

### The Three Questions Framework
Before writing any kernel, ask:
1. What is one unit of work? (what does one thread do?)
2. Are threads independent or do they depend on each other?
3. What is the final output — one value per element or one total?

This determines which of the three patterns to use:
- Pattern 1: embarrassingly parallel (vecAdd, matMul)
- Pattern 2: reduction to one value (sum, max, dot product)
- Pattern 3: scan / prefix (all elements need a result)

### Stride Patterns
Two reduction strategies depending on output type:

```c
// Stride UP — prefix sum, every element needs a result
for (int stride = 1; stride < blockDim.x; stride *= 2)
    if (tid >= stride) tile[tid] += tile[tid - stride];

// Stride DOWN — reduction, one value total needed
for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    if (tid < stride) tile[tid] += tile[tid + stride];
```

### Streams
Operations in different streams run concurrently,
overlapping memory transfers with kernel execution:

```c
// stream 1 pipeline
cudaMemcpyAsync(d_A, A, size, cudaMemcpyHostToDevice, stream1);
kernel<<<grid, block, 0, stream1>>>(d_A, d_C, N);
cudaMemcpyAsync(C, d_C, size, cudaMemcpyDeviceToHost, stream1);

// stream 2 pipeline — overlaps with stream 1
cudaMemcpyAsync(d_B, B, size, cudaMemcpyHostToDevice, stream2);
kernel<<<grid, block, 0, stream2>>>(d_B, d_D, N);
cudaMemcpyAsync(D, d_D, size, cudaMemcpyDeviceToHost, stream2);
```

---

## What Was Hard

**Thread and block configuration** — understanding that
blockDim.x * blockIdx.x + threadIdx.x gives every thread
a unique global index took repetition to fully internalise.

**Stride patterns** — distinguishing when to stride up vs
down, and why halving avoids data corruption while doubling
spreads information forward.

**Halo elements** — the stencil problem required loading
neighbour elements from adjacent blocks into a padded
shared memory tile (+2 elements), shifting all loads right
by 1 to make room for the left halo.

**cudaMemcpy argument order** — (destination, source) is
easy to swap. Rule: treat it like assignment, destination
always comes first.

---

## Common Bugs Encountered and Fixed

| Bug | Fix |
|-----|-----|
| Passing CPU pointer to kernel | Always cudaMalloc + cudaMemcpy first |
| __syncthreads() inside if block | Must be reached by ALL threads unconditionally |
| INT_MIN as neutral value for sum | Use 0 for sum, INT_MIN only for max |
| cudaMemcpy args swapped | (destination, source) — dst always first |
| Output array wrong size for reductions | blocks * sizeof(int) not N * sizeof(int) |
| cudaMallocHost freed with free() | Always match: cudaMallocHost → cudaFreeHost |

---

## Memory Management Rules

malloc          → free
cudaMalloc      → cudaFree
cudaMallocHost  → cudaFreeHost
cudaMallocManaged → cudaFree
cudaMallocAsync → cudaFreeAsync

---

## What is Next

Tomorrow: FlashAttention and Triton (Weeks 17-18).
Will profile CUDA kernels with Nsight Compute
when implementing attention mechanisms.

---

## Graduation Checklist Progress

- [x] Weekly report filed
- [ ] Blog post (planned: custom CUDA kernel deep dive)
- [ ] OSS PR
- [ ] Capstone
