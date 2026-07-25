import argparse
import json

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.pipelining import pipeline, SplitPoint, PipelineStage, ScheduleGPipe
import os
import time

# Setup
def setup():
    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    return rank, dist.get_rank(), dist.get_world_size(), device

# Cleanup

def cleanup():
    dist.destroy_process_group()

from gpipe_model import GPT, GPTConfig

def build_model(cfg, local_rank):
    model = GPT(cfg).to(local_rank)
    return model, cfg.n_layer

def partition_model(model, n_layer, stage_index, num_stages, device) -> PipelineStage:
    if stage_index == 0:
        del model.blocks[n_layer // 2:]
        model.ln_f = None
        model.head = None
    elif stage_index == 1:
        del model.blocks[:n_layer // 2]
        model.tok_emb = None
        model.pos_emb = None
    
    stage = PipelineStage(
        model,
        stage_index,
        num_stages,
        device,
    )
    return stage

def loss_fn(outputs, targets):
    return F.cross_entropy(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))

def measure_isolated_compute_time(model, stage_index, cfg, microbatch_size, device, n_reps=10):
    """Times forward+backward for one microbatch on this stage alone, no
    pipeline scheduling or communication. Ground truth for the bubble formula."""
    model.train()
    if stage_index == 0:
        x = torch.randint(0, cfg.vocab_size, (microbatch_size, cfg.block_size), device=device)
    else:
        x = torch.randn(microbatch_size, cfg.block_size, cfg.n_embd, device=device)

    def one_pass():
        out = model(x)
        if stage_index == 1:
            target = torch.randint(0, cfg.vocab_size, (microbatch_size, cfg.block_size), device=device)
            loss_fn(out, target).backward()
        else:
            out.sum().backward()
        model.zero_grad(set_to_none=True)

    for _ in range(3):
        one_pass()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        one_pass()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def run_benchmark(args):
    local_rank, stage_index, num_stages, device = setup()

    cfg = GPTConfig(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
    )

    if stage_index == 0:
        print(f"num of stages:{num_stages} ")
    model, n_layers = build_model(cfg, local_rank=local_rank)

    torch.manual_seed(0)
    data = torch.randint(0, cfg.vocab_size, (args.batch_size, args.block_size), device=local_rank)
    targets = torch.randint(0, cfg.vocab_size, (args.batch_size, args.block_size), device=local_rank)

    stage = partition_model(model, n_layers, stage_index, num_stages, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    torch.cuda.reset_peak_memory_stats(local_rank)
    dist.barrier()

    step_times = []
    num_microbatches = 4 * num_stages
    schedule = ScheduleGPipe(stage, num_microbatches, loss_fn=loss_fn)

    for step in range(args.steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if stage_index == 0:
            schedule.step(data)
        else:
            losses = []
            schedule.step(target=targets, losses=losses)
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        if step > 0:
            step_times.append(t1 - t0)

        if stage_index == 1:
            avg_loss = sum(l.item() for l in losses) / len(losses)
            print(f"step {step} loss {avg_loss:.4f} time {t1 - t0:.3f}s")

    peak_mem_gb = torch.cuda.max_memory_allocated(local_rank) / 1e9
    avg_step_time = sum(step_times) / len(step_times) if step_times else float("nan")
    tokens_per_step = args.batch_size * args.block_size
    throughput = tokens_per_step / avg_step_time if step_times else float("nan")

    microbatch_size = args.batch_size // num_microbatches
    t_compute = measure_isolated_compute_time(model, stage_index, cfg, microbatch_size, device)

    compute_times = [None] * num_stages
    dist.all_gather_object(compute_times, t_compute)
    t_compute_max = max(compute_times)

    predicted_bubble_fraction = (num_stages - 1) / (num_microbatches + num_stages - 1)
    ideal_no_bubble_time = num_microbatches * t_compute_max
    observed_bubble_fraction = max(0.0, (avg_step_time - ideal_no_bubble_time)) / avg_step_time if step_times else None

    # gather every stage's peak memory onto all ranks so rank 0 can log the full picture
    mem_by_stage = [None] * num_stages
    dist.all_gather_object(mem_by_stage, round(peak_mem_gb, 3))

    if stage_index == 0:
        try:
            n_params = round(model.num_params() / 1e9, 3)
        except Exception:
            n_params = None
            
        result = {
        "num_stages": num_stages,
        "n_params_billions": n_params,
        "peak_mem_gb_by_stage": mem_by_stage,   # replaces peak_mem_gb_rank0
        "avg_step_time_s": round(avg_step_time, 4) if step_times else None,
        "tokens_per_sec": round(throughput, 1) if step_times else None,
        "compute_times" : compute_times,
        "predicted_bubble_fraction" : predicted_bubble_fraction,
        "observed_bubble_fraction" : observed_bubble_fraction,
        "batch_size": args.batch_size,
        "block_size": args.block_size,
    }
        print(json.dumps(result, indent=2))
        os.makedirs("results", exist_ok=True)
        with open(f"results/gpipe_result.json", "w") as f:
            json.dump(result, f, indent=2)

    cleanup()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_layer", type=int, default=16)
    p.add_argument("--n_head", type=int, default=12)
    p.add_argument("--n_embd", type=int, default=1536)
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--steps", type=int, default=5)
    return p.parse_args()




if __name__ == "__main__":
    run_benchmark(parse_args())