import torch

from benchmark import (
    NaiveAllocator,
    generate_workload,
    run_simulation,
)
from block_manager import BlockManager

NUM_LAYERS = 16
N_EMBD = 1536
N_HEAD = 24
HEAD_DIM = N_EMBD // N_HEAD
DTYPE = torch.float16
DTYPE_BYTES = torch.finfo(DTYPE).bits // 8

BLOCK_SIZE = 16
TOTAL_SLOTS = 15000
NUM_GPU_BLOCKS = TOTAL_SLOTS // BLOCK_SIZE


def bytes_per_token(num_layers=NUM_LAYERS, num_kv_heads=N_HEAD, head_dim=HEAD_DIM,
                     dtype_bytes=DTYPE_BYTES):
    per_layer_per_token = 2 * num_kv_heads * head_dim * dtype_bytes
    return per_layer_per_token * num_layers


def run_comparison():
    workload_naive = generate_workload(300, rng_seed=42)
    workload_paged = generate_workload(300, rng_seed=42)

    naive_allocator = NaiveAllocator(TOTAL_SLOTS)
    naive_stats = run_simulation(naive_allocator, workload_naive)

    block_manager = BlockManager(NUM_GPU_BLOCKS, BLOCK_SIZE)
    paged_stats = run_simulation(block_manager, workload_paged, block_manager=block_manager)

    bpt = bytes_per_token()
    naive_peak_bytes = naive_stats["peak_used"] * bpt
    paged_peak_bytes = paged_stats["peak_used"] * bpt
    wasted_bytes = naive_peak_bytes - paged_peak_bytes

    print(f"Naive peak capacity reserved: {naive_stats['peak_used']} token-slots "
          f"= {naive_peak_bytes / (1024**2):.1f} MB")
    print(f"Paged peak capacity used:     {paged_stats['peak_used']} token-slots "
          f"= {paged_peak_bytes / (1024**2):.1f} MB")
    print(f"Memory paging avoids reserving unnecessarily: "
          f"{wasted_bytes / (1024**2):.1f} MB "
          f"({100 * wasted_bytes / naive_peak_bytes:.1f}% of naive's peak)")

    return naive_peak_bytes, paged_peak_bytes


def verify_on_gpu(naive_peak_bytes, paged_peak_bytes):
    if not torch.cuda.is_available():
        print("\nNo CUDA device, skipping GPU verification ")
        return

    num_elements = paged_peak_bytes // DTYPE_BYTES
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    tensor = torch.zeros(num_elements, dtype=DTYPE, device="cuda")
    after = torch.cuda.memory_allocated()
    actual_bytes = after - before
    del tensor
    torch.cuda.empty_cache()

    print(f"\nGPU verification: requested {paged_peak_bytes / (1024**2):.1f} MB, "
          f"torch reports {actual_bytes / (1024**2):.1f} MB allocated.")


if __name__ == "__main__":
    naive_peak_bytes, paged_peak_bytes = run_comparison()
    verify_on_gpu(naive_peak_bytes, paged_peak_bytes)