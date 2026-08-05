# Week 22 (redo): vLLM and PagedAttention

## Why this got redone

I originally attempted this in June, wrote a `block_manager.py`, hit a copy-on-write
bug, and then paused. I never came back to it,
deleted the file, and never committed anything vLLM-related to the repo, so it went
on my CV as skipped. After finishing Pipeline Parallelism (Weeks 29-30), I re-read the
paper and it was noticeably more tractable the second time, so I'm going back to
actually ship it before moving on to Week 31 (KV cache tracing through real vLLM
source).

## The problem PagedAttention solves

Existing LLM serving systems store each request's KV cache (the cached attention key
and value vectors for every previously processed token, per transformer layer) in a
single contiguous block of GPU memory, sized upfront for the worst case. This causes
three problems:

- **External fragmentation**: requests of different actual lengths finish and free
  memory at different times, leaving gaps too small and too scattered to satisfy new
  requests, even when total free memory would be enough.
- **Internal fragmentation**: memory is reserved for a max possible sequence length,
  but the actual output is usually much shorter, wasting the difference.
- **No memory sharing**: since each request's cache is an independent contiguous
  block, there's no way for requests to share memory, e.g. multiple beam search
  branches or parallel samples that share a common prompt.

PagedAttention borrows virtual memory and paging from OS design to fix this. It
partitions the KV cache into fixed-size blocks, stored non-contiguously, and gives
each sequence a *block table*, a lightweight mapping from logical block index to
physical block ID. The attention computation itself only ever addresses logical
blocks; the block table handles translation to physical memory, the same way a CPU
issuing a virtual address never has to know which physical frame it lands in.

## What I built

Threee files, under `phase2_content/vllm_paged_attention/`:

- **`block_manager.py`**: the core paging data structures. `PhysicalBlock` (a single
  fixed-size slot with a reference count), a free list of block IDs, and a per-sequence
  block table. `BlockManager` supports `allocate` (first admission), `append_slot`
  (grow by one token per decode step), `fork` (share a prefix across sequences, e.g.
  for beam search, by incrementing ref counts rather than copying), and `free`
  (release blocks back to the pool, only once ref count hits zero). `block_size=16`,
  matching the paper's default. The KV cache pool itself is a stub, `copy_kv_data` is
  unimplemented, this artifact tests the *memory accounting*, not real attention
  correctness (see Limitations).
- **`benchmark.py`**: compares the paged approach against a naive baseline
  (`NaiveAllocator`) that reserves `prompt_len + max_output_len` contiguously per
  request, the way existing serving systems do. Both allocators get the same total
  token-equivalent budget so the comparison is apples to apples.
- **`memory_profile.py`**: compare the toy version I built, **`block_manager.py`** against a real production system
  (vLLM) and understand how it manages GPU memory for the KV cache.

## Bugs I hit building this (worth keeping, not just fixing silently)

1. `copy_kv_data` was originally defined inside `BlockManager` without `self`, then
   called as a bare function, guaranteed `NameError` the first time a copy-on-write
   branch fired. Pulled it out as a standalone function.
2. `free()` had a guard clause checking `if not self.free_blocks: raise`, but
   `free()` only ever *appends* to the free list, so an empty free list going in is
   exactly the normal, expected case under memory pressure, not an error. Removed.
3. `can_allocate` / `allocate` called `len(seq.prompt_tokens)`, but `Sequence` only
   has `prompt_len`, an int. Would have crashed on the first real workload.
4. The big one: `BlockManager.can_allocate` only checks blocks needed for the
   *prompt*, it has no way to reserve for a sequence's future decode growth, unlike
   `NaiveAllocator` which commits to the full `prompt_len + max_output_len` upfront.
   Under load, this let the paged path admit more concurrent sequences early (since
   it wasn't over-reserving) but then run out of blocks mid-decode with no fallback,
   an uncaught crash. This isn't really a bug so much as the actual problem vLLM's
   scheduler exists to solve. I added a lightweight preemption path: when a decode
   step needs a block and none are free, evict the most recently admitted other
   active sequence, discard its progress, and requeue it to be regenerated from
   scratch (vLLM's default "recompute" policy; the paper's alternative, "swap", copies
   evicted blocks to CPU memory instead, more expensive to implement, out of scope
   here).

## Benchmark setup

- Workload: 300 synthetic sequences, heavy-tailed lengths (80% short prompts
  20-200 tokens, 15% medium 200-800, 5% long-tail 2000-4000; output lengths drawn
  from a Pareto distribution, capped at 512). Uniform lengths would have hidden the
  effect I was trying to measure, real serving traffic is skewed and that skew is
  what makes naive's internal fragmentation visible.
- Capacity: both allocators sized to the same token-equivalent budget
  (`TOTAL_SLOTS=15000` for naive, `NUM_GPU_BLOCKS=937` blocks at `block_size=16`
  for paged, ≈14992 tokens), undersized relative to total workload demand so both
  systems are under real contention.
- Admission: FCFS, up to 8 sequences admitted per step, capacity permitting.

## Results

| | Naive (contiguous) | Paged (BlockManager) |
|---|---|---|
| Total steps to clear workload | 966 | 897 |
| Completed | 300 | 300 |
| Rejected admission events | 608 | 518 |
| Preemptions | 0 | 69 |

Paged clears the identical workload in **7.14% fewer steps**, despite paying a real
cost of 69 preemption-and-recompute events that naive never incurs. It gets there by
not reserving `max_output_len` upfront, so it admits sequences sooner (518 vs 608
blocked-admission events), and that speed advantage outweighs the recompute tax.

**Dollar translation** (illustrative, see caveat below): at a placeholder $1.50/hr
GPU rate and 0.05s/step, naive costs $0.020125 to clear this batch, paged costs
$0.0186875, a savings of $0.0014375 per 300-request batch (7.14%, same ratio as the
step reduction since cost scales linearly with steps at fixed rate).

## Limitations

- `copy_kv_data` is a stub. This benchmark validates the memory *accounting* logic
  (fragmentation, admission, sharing via ref counts), not real attention output
  correctness. A real KV tensor pool (shape `[num_gpu_blocks, block_size, num_heads,
  head_dim]` per K and V, indexed by block ID) would be the next step if I wanted to
  verify actual attention correctness under paging.
- `seconds_per_step=0.05` is a placeholder, not a measured value. The dollar figure
  above is directionally correct (same 7.14% ratio as the step reduction) but not a
  real cost estimate. Week 31 (tracing real vLLM source and profiling actual KV
  memory) is where I plan to replace this with a measured number.
- The preemption policy (evict most recently admitted, full recompute) is a
  simplification of vLLM's actual scheduler, which considers priority and supports
  swap-to-CPU as an alternative to recompute.
- 300 synthetic sequences at toy scale, not a production trace.

## Connecting back to the OS analogy

- `free_blocks` (the free list) is a free physical frame list.
- `ref_count` on each `PhysicalBlock` is what makes copy-on-write possible, the same
  mechanism `fork()` uses in a real OS: share pages until one side writes, then copy.
- The per-sequence block table (logical block index -> physical block ID) is a page
  table.
- The admission-and-preemption problem I hit while building `append_slot` is the
  same problem an OS scheduler faces under memory pressure, decide who gets evicted
  when there isn't enough physical memory for everyone who wants to run.

## Next steps

- Clean up and commit `block_manager.py`, `benchmark.py`, and this report to
  `phase2_content/vllm_paged_attention/`.
- Write the accompanying blog post (5th required blog post).
- Carry the placeholder `seconds_per_step` and the block table / COW concepts
  forward into Week 31, tracing the real vLLM source and profiling actual KV cache
  memory at varying batch sizes.
