from datasets import load_dataset
from transformers import AutoTokenizer


dataset = load_dataset("wikitext", "wikitext-103-v1")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

#hyperparameters
batch_size = 8
learning_rate = 1e-4
max_iters = 500
eval_iters = 50
eval_interval = 200
device = "cuda" if torch.cuda.is_available() else "cpu"

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


block_size = 1024
batch_size = 8


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
    with torch.no_grad():
        for split in ["train", "validation"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                X, Y = X.to(device), Y.to(device)
                _ , loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out


from models.model import GPT, GPTConfig

cfg = GPTConfig()
model = GPT(cfg)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), learning_rate)

def train(mode):
    step_times = []
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(model=model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['validation']:.4f}")
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        xb, yb = get_batch("train")
        xb, yb = xb.to(device), yb.to(device)

        _ , loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)

        loss.backward()
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        if iter > 0:
            step_times.append(t1 - t0)
        avg_step_time = sum(step_times) / len(step_times) if step_times else float("nan")
        tokens_per_step = batch_size * block_size
        throughput = tokens_per_step / avg_step_time if step_times else float("nan")

    losses = estimate_loss(model=model)
    print(f"step {max_iters}: train loss {losses['train']:.4f}, val loss {losses['validation']:.4f}")
    
    try:
        if mode == "baseline":
            with open("logs/week37_baseline.txt", "w", encoding="utf-8") as file:
                file.write(f"Model name: GPT2-406M\n")
                file.write(f"CUDA Version: {torch.version.cuda}\n")
                file.write(f"GPU Model: {torch.cuda.get_device_name()}\n")
                file.write(f"Tokens/sec: {round(throughput, 1) if step_times else float('nan')}\n")
                file.write(f"train loss {losses['train']:.4f}, val loss {losses['validation']:.4f}\n")
                print("Logs written successfuly")
        else:
            """ 
                will write code here later for ddp and fsdp
            """
            pass
    except OSError as e:
        print("could not log results")
        return 

def write_to_log():
    pass

    


        
if __name__ == "__main__":
    train("baseline")
    torch.save(model.state_dict(), "CAPSTONE/models/gpt2_406m_week37_baseline.pt")

