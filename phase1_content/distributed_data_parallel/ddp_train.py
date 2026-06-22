"""
ddp_train.py — weeks 13-14: distributed data parallel experiment
wraps BiLanguageModel in DDP across 2 GPUs, logs per-step time,
communication overhead, and scaling efficiency vs single-GPU baseline
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn import functional as F
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size: int = 65
    block_size: int = 64
    n_embd: int     = 128
    n_layers: int   = 4
    n_heads: int    = 4
    dropout: float  = 0.2


BATCH_SIZE    = 32
MAX_ITERS     = 300
LOG_INTERVAL  = 50
LEARNING_RATE = 1e-4
OUTPUT_DIR    = "/kaggle/working"


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

class Head(nn.Module):
    def __init__(self, config: ModelConfig, head_size):
        super().__init__()
        self.head_size = head_size
        self.key   = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        head_size    = config.n_embd // config.n_heads
        self.heads   = nn.ModuleList([Head(config, head_size) for _ in range(config.n_heads)])
        self.proj    = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class Feedforward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.sa_head = MultiHeadAttention(config)
        self.ffwd    = Feedforward(config)
        self.ln1     = nn.LayerNorm(config.n_embd)
        self.ln2     = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BiLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table    = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layers)],
            nn.LayerNorm(config.n_embd)
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)
        pos_emb   = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

class ShakespeareDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data       = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx     : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def load_data(block_size):
    # auto-find the txt file under /kaggle/input
    path = None
    for root, dirs, files in os.walk("/kaggle/input"):
        for f in files:
            if f.endswith(".txt"):
                path = os.path.join(root, f)
                break
        if path:
            break

    if path is None:
        raise FileNotFoundError("no .txt file found under /kaggle/input")

    with open(path, "r") as f:
        text = f.read()

    chars  = sorted(set(text))
    stoi   = {ch: i for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]

    data = torch.tensor(encode(text), dtype=torch.long)
    n    = int(0.9 * len(data))
    return data[:n], data[n:]


# ---------------------------------------------------------------------------
# single GPU baseline
# ---------------------------------------------------------------------------

def run_single_gpu(config: ModelConfig):
    print("\n--- single GPU baseline ---")
    device = torch.device("cuda:0")

    train_data, _ = load_data(config.block_size)
    dataset       = ShakespeareDataset(train_data, config.block_size)
    loader        = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    model     = BiLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    step_times  = []
    loader_iter = iter(loader)

    for step in range(MAX_ITERS):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)

        x, y = x.to(device), y.to(device)

        t0 = time.perf_counter()
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)

        if step % LOG_INTERVAL == 0:
            print(f"  step {step:4d}  loss {loss.item():.4f}  step_time {step_times[-1]*1000:.1f}ms")

    avg_ms = sum(step_times) / len(step_times) * 1000
    print(f"\nsingle GPU avg step time: {avg_ms:.2f}ms")
    return avg_ms


# ---------------------------------------------------------------------------
# DDP worker
# ---------------------------------------------------------------------------

def ddp_worker(rank: int, world_size: int, config: ModelConfig):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    train_data, _ = load_data(config.block_size)
    dataset  = ShakespeareDataset(train_data, config.block_size)
    sampler  = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader   = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=True)

    model     = BiLanguageModel(config).to(device)
    model     = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    step_times    = []
    compute_times = []
    comm_times    = []
    loader_iter   = iter(loader)

    for step in range(MAX_ITERS):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            sampler.set_epoch(step)
            loader_iter = iter(loader)
            x, y = next(loader_iter)

        x, y = x.to(device), y.to(device)

        t0 = time.perf_counter()
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)

        t_compute_start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        t_compute_end = time.perf_counter()

        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        step_times.append(t1 - t0)
        compute_times.append(t_compute_end - t_compute_start)
        comm_times.append((t1 - t0) - (t_compute_end - t_compute_start))

        if rank == 0 and step % LOG_INTERVAL == 0:
            print(f"  step {step:4d}  loss {loss.item():.4f}  "
                  f"step {step_times[-1]*1000:.1f}ms  "
                  f"compute {compute_times[-1]*1000:.1f}ms  "
                  f"comm {comm_times[-1]*1000:.1f}ms")

    if rank == 0:
        avg_step    = sum(step_times) / len(step_times) * 1000
        avg_compute = sum(compute_times) / len(compute_times) * 1000
        avg_comm    = sum(comm_times) / len(comm_times) * 1000

        ddp_result = {
            "ddp_avg_step_ms":    round(avg_step, 2),
            "ddp_avg_compute_ms": round(avg_compute, 2),
            "ddp_avg_comm_ms":    round(avg_comm, 2),
            "comm_overhead_pct":  round(avg_comm / avg_step * 100, 1),
        }

        with open(f"{OUTPUT_DIR}/ddp_partial.json", "w") as f:
            json.dump(ddp_result, f)

        print(f"\nDDP ({world_size} GPUs)  step: {avg_step:.2f}ms  "
              f"compute: {avg_compute:.2f}ms  comm: {avg_comm:.2f}ms")

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config     = ModelConfig()
    world_size = torch.cuda.device_count()
    print(f"GPUs available: {world_size}")

    single_gpu_ms = run_single_gpu(config)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    print(f"\n--- DDP run ({world_size} GPUs) ---")
    mp.spawn(
        ddp_worker,
        args=(world_size, config),
        nprocs=world_size,
        join=True,
    )

    with open(f"{OUTPUT_DIR}/ddp_partial.json") as f:
        ddp_result = json.load(f)

    ddp_ms             = ddp_result["ddp_avg_step_ms"]
    scaling_efficiency = round((single_gpu_ms / world_size / ddp_ms) * 100, 1)

    summary = {
        "world_size":             world_size,
        "single_gpu_ms":          round(single_gpu_ms, 2),
        **ddp_result,
        "scaling_efficiency_pct": scaling_efficiency,
    }

    print("\n--- final summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(f"{OUTPUT_DIR}/ddp_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nresults saved to {OUTPUT_DIR}/ddp_results.json")
