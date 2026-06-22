"""
benchmark.py
------------
Benchmarks FlashAttention-Triton vs naive PyTorch attention.

Measurese
--------
- Latency (ms)   : median of 100 timed kernel launches (after 10-run warm-up)
- Peak HBM (MB)  : torch.cuda.max_memory_allocated() delta

Outputs
-------
- Console table
- benchmark_results.json  (ingested by the dashboard)

Usage
-----
    python benchmark.py
    python benchmark.py --seq_lens 256 512 1024 2048 4096
    python benchmark.py --batch 2 --heads 8 --d_head 64
"""

import argparse
import json
import time
import torch

from flash_attn_triton import flash_attention_triton, naive_attention_pytorch


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def time_fn(fn, *args, warmup: int = 10, reps: int = 100) -> float:
    """Return median latency in milliseconds."""
    # Warm-up
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[len(times) // 2]   # median


def peak_memory_mb(fn, *args) -> float:
    """Return peak HBM allocated (MB) during a single fn call."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn(*args)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmarks(
    seq_lens,
    batch: int,
    heads: int,
    d_head: int,
) -> list[dict]:
    results = []

    header = (
        f"{'Seq':>6}  "
        f"{'Naive lat (ms)':>16}  {'Triton lat (ms)':>16}  "
        f"{'Speedup':>8}  "
        f"{'Naive mem (MB)':>16}  {'Triton mem (MB)':>16}  "
        f"{'Mem saved':>10}"
    )
    print(header)
    print("-" * len(header))

    for N in seq_lens:
        q = torch.randn(batch, heads, N, d_head, dtype=torch.float16, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Latency
        lat_naive  = time_fn(naive_attention_pytorch, q, k, v)
        lat_triton = time_fn(flash_attention_triton,  q, k, v)
        speedup    = lat_naive / lat_triton

        # Peak memory
        mem_naive  = peak_memory_mb(naive_attention_pytorch, q, k, v)
        mem_triton = peak_memory_mb(flash_attention_triton,  q, k, v)
        mem_saved  = (1 - mem_triton / mem_naive) * 100   # %

        results.append({
            "seq_len":    N,
            "batch":      batch,
            "heads":      heads,
            "d_head":     d_head,
            "lat_naive":  round(lat_naive,  3),
            "lat_triton": round(lat_triton, 3),
            "speedup":    round(speedup,    3),
            "mem_naive":  round(mem_naive,  2),
            "mem_triton": round(mem_triton, 2),
            "mem_saved_pct": round(mem_saved, 1),
        })

        print(
            f"{N:>6}  "
            f"{lat_naive:>16.3f}  {lat_triton:>16.3f}  "
            f"{speedup:>7.2f}×  "
            f"{mem_naive:>16.1f}  {mem_triton:>16.1f}  "
            f"{mem_saved:>9.1f}%"
        )

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashAttention-Triton benchmark")
    parser.add_argument(
        "--seq_lens", nargs="+", type=int,
        default=[256, 512, 1024, 2048, 4096],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument("--batch",  type=int, default=1)
    parser.add_argument("--heads",  type=int, default=8)
    parser.add_argument("--d_head", type=int, default=64)
    args = parser.parse_args()

    print(f"\nFlashAttention-Triton Benchmark")
    print(f"Config: batch={args.batch}, heads={args.heads}, d_head={args.d_head}")
    print(f"Device: {torch.cuda.get_device_name(0)}\n")

    results = run_benchmarks(
        seq_lens=args.seq_lens,
        batch=args.batch,
        heads=args.heads,
        d_head=args.d_head,
    )

    # Save JSON for the dashboard
    out_path = "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")