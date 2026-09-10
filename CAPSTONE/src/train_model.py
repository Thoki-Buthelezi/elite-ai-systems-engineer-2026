from datasets import load_dataset
from transformers import AutoTokenizer


dataset = load_dataset("wikitext", "wikitext-103-v1")
tokenizer = AutoTokenizer.from_pretrained("gpt2")


import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


#hyperparameters
learning_rate = 1e-4
max_iters = 500
eval_iters = 50
eval_interval = 200
block_size = 1024
batch_size = 2
amp_dtype = torch.float16


if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required for this benchmark.")

device = "cuda" 

# tokenize each row
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding=False,
        truncation=False,
    )

# variable-length token lists
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# Flatten the tokenized dataset into a continuous stream
def flatten_tokens(dataset_split):
    tokens = []

    for ids in dataset_split["input_ids"]:
        tokens.extend(ids)

    return torch.tensor(tokens, dtype=torch.long)


train_data = flatten_tokens(tokenized_dataset["train"])
val_data = flatten_tokens(tokenized_dataset["validation"])


def get_batch(split):
    """
    randomly sample a starting position, 
    take 1024 tokens as the input, and take the same tokens + 1 as target. 
    The target is therefore the input shifted by one token, 
    allowing the model to learn next-token prediction.
    """
    data = train_data if split == "train" else val_data

    ix = torch.randint(
        len(data) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size]
        for i in ix
    ])

    y = torch.stack([
        data[i+1:i + block_size + 1]
        for i in ix
    ])

    return x, y


#function to estimate the loss on train and val data
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    #tell the model not to store intermediate value because we are not going to use backprop
    for split in ["train", "validation"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            X, Y = X.to(device), Y.to(device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                _ , loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


from models.model import GPT, GPTConfig

def setup_training(mode):
    """
    Set up the model and optimizer according to the selected
    distributed training mode.
    """

    cfg = GPTConfig()
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if mode == "baseline":

        model = GPT(cfg)
        model = model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            learning_rate
        )

        scaler = torch.amp.GradScaler("cuda")

    elif mode == "ddp":

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size = world_size,
            rank = rank
        )

        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

        model = GPT(cfg).to(device)
        model = DDP(model, device_ids=[local_rank])
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            learning_rate
        )
        scaler = torch.amp.GradScaler("cuda")

    elif mode == "fsdp":

        # FSDP setup code here
        pass

    else:
        raise ValueError(
            f"Unknown training mode: {mode}. "
            f"Expected 'baseline', 'ddp', or 'fsdp'."
        )

    return model, optimizer, scaler, world_size, rank, device


def write_to_log(mode, losses, throughput, step_times):
    """
    Write benchmark results to the appropriate log file.
    """

    if mode == "baseline":

        try:
            with open(
                "CAPSTONE/logs/week37_baseline.txt",
                "w",
                encoding="utf-8"
            ) as file:

                file.write(f"Model name: GPT2-406M\n")
                file.write(f"CUDA Version: {torch.version.cuda}\n")
                file.write(f"GPU Model: {torch.cuda.get_device_name()}\n")
                file.write(
                    f"Tokens/sec: "
                    f"{round(throughput, 1) if step_times else float('nan')}\n"
                )
                file.write(f"Batch size: {batch_size}\n")
                file.write(f"Autocast dtype: {amp_dtype}\n")
                file.write(
                    f"train loss {losses['train']:.4f}, "
                    f"val loss {losses['validation']:.4f}\n"
                )

                print("Logs written successfuly")

        except OSError as e:
            print("could not log results")
            return

    elif mode == "ddp":
        try:
            with open(
                "CAPSTONE/logs/week38_ddp.txt",
                "w",
                encoding="utf-8"
            ) as file:

                file.write(f"Model name: GPT2-406M\n")
                file.write(f"CUDA Version: {torch.version.cuda}\n")
                file.write(f"GPU Model: {torch.cuda.get_device_name()}\n")
                file.write(
                    f"Tokens/sec: "
                    f"{round(throughput, 1) if step_times else float('nan')}\n"
                )
                file.write(f"Batch size: {batch_size}\n")
                file.write(f"Autocast dtype: {amp_dtype}\n")
                file.write(
                    f"train loss {losses['train']:.4f}, "
                    f"val loss {losses['validation']:.4f}\n"
                )

                print("Logs written successfuly")
        except OSError as e:
            print("could not log results")

    elif mode == "fsdp":

        # FSDP logging code here
        pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "ddp", "fsdp"], required=True)
    return p.parse_args()

def train(args):

    model, optimizer, scaler, world_size, rank, device = setup_training(args.mode)

    step_times = []

    for iter in range(max_iters):

        if iter % eval_interval == 0:
            losses = estimate_loss(model=model)
            print(
                f"step {iter}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['validation']:.4f}"
            )

        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        xb, yb = get_batch("train")
        xb, yb = xb.to(device), yb.to(device)

        # Autocast context: runs some ops in float16 for speed
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            _ , loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)

        # Scale loss to prevent underflow
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if device.type == "cuda":
            torch.cuda.synchronize()

        t1 = time.perf_counter()

        if iter > 0:
            step_times.append(t1 - t0)

        avg_step_time = (
            sum(step_times) / len(step_times)
            if step_times
            else float("nan")
        )

        tokens_per_step = batch_size * block_size * world_size
        throughput = (
            tokens_per_step / avg_step_time
            if step_times
            else float("nan")
        )

    losses = estimate_loss(model=model)

    print(
        f"step {max_iters}: "
        f"train loss {losses['train']:.4f}, "
        f"val loss {losses['validation']:.4f}"
    )

    if rank == 0:
        write_to_log(
            mode=args.mode,
            losses=losses,
            throughput=throughput,
            step_times=step_times
       )

    return model, rank


if __name__ == "__main__":

    p = parse_args()
    model, rank = train(p)
    if rank == 0:
        if p.mode == "baseline":
            torch.save(model.state_dict(), "CAPSTONE/models/gpt2_406m_week37_baseline.pt")
        elif p.mode == "ddp":
            torch.save(model.module.state_dict(), "CAPSTONE/models/gpt2_406m_week38_ddp.pt")
        else:
            pass

