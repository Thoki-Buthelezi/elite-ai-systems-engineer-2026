"""
DDP vs FSDP benchmark on a ~1.42B param GPT model.

Run on Kaggle (2x T4) with:
    !torchrun --nproc_per_node=2 benchmark.py --mode fsdp
    !torchrun --nproc_per_node=2 benchmark.py --mode ddp

Expect --mode ddp to OOM on T4s at this model size, that's expected
and IS the result
"""
import argparse
import functools
import json
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig, Block


def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def cleanup():
    dist.destroy_process_group()


def build_model(cfg, mode, local_rank):
    model = GPT(cfg).to(local_rank)

    if mode == "ddp":
        model = DDP(model, device_ids=[local_rank])
    elif mode == "fsdp":
        # Wrap at the granularity of individual transformer blocks,
        # not the whole model as one flat unit. This means each
        # block's params get gathered/sharded independently, which
        # is what keeps peak memory down, we only ever hold ONE
        # block's full params at a time, not the whole model's.
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Block},
        )
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=local_rank,
        )
    else:
        raise ValueError(f"unknown mode {mode}")

    return model


def run_benchmark(args):
    local_rank, rank, world_size = setup()
    cfg = GPTConfig(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
    )

    if rank == 0:
        print(f"[rank0] mode={args.mode} world_size={world_size}")

    model = build_model(cfg, args.mode, local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # dummy data, we're benchmarking system behaviour not model quality
    torch.manual_seed(0)
    data = torch.randint(
        0, cfg.vocab_size, (args.batch_size, args.block_size), device=local_rank
    )
    targets = torch.randint(
        0, cfg.vocab_size, (args.batch_size, args.block_size), device=local_rank
    )

    torch.cuda.reset_peak_memory_stats(local_rank)
    dist.barrier()

    step_times = []
    oom = False
    try:
        for step in range(args.steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)
            _, loss = model(data, targets)
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            # first step includes CUDA warmup/alloc overhead, skip it
            if step > 0:
                step_times.append(t1 - t0)

            if rank == 0:
                print(f"step {step} loss {loss.item():.4f} time {t1 - t0:.3f}s")
    except torch.cuda.OutOfMemoryError:
        # this IS a result for --mode ddp at this model size, not a bug.
        # record it instead of letting torchrun print a raw stack trace.
        oom = True
        if rank == 0:
            print(f"[rank0] CUDA OOM under mode={args.mode}, this is expected ")

    peak_mem_gb = torch.cuda.max_memory_allocated(local_rank) / 1e9
    avg_step_time = sum(step_times) / len(step_times) if step_times else float("nan")
    tokens_per_step = args.batch_size * args.block_size * world_size
    throughput = tokens_per_step / avg_step_time if step_times else float("nan")

    if rank == 0:
        n_params = None
        try:
            n_params = round(
                (model.module.num_params() if hasattr(model, "module") else None)
                / 1e9,
                3,
            )
        except Exception:
            pass

        result = {
            "mode": args.mode,
            "world_size": world_size,
            "oom": oom,
            "n_params_billions": n_params,
            "peak_mem_gb_rank0": round(peak_mem_gb, 3),
            "avg_step_time_s": round(avg_step_time, 4) if step_times else None,
            "tokens_per_sec": round(throughput, 1) if step_times else None,
            "batch_size": args.batch_size,
            "block_size": args.block_size,
        }
        print(json.dumps(result, indent=2))
        os.makedirs("results", exist_ok=True)
        with open(f"results/{args.mode}_result.json", "w") as f:
            json.dump(result, f, indent=2)

    cleanup()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ddp", "fsdp"], required=True)
    p.add_argument("--n_layer", type=int, default=24)
    p.add_argument("--n_head", type=int, default=16)
    p.add_argument("--n_embd", type=int, default=2048)
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--steps", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
