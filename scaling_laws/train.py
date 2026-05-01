import torch
import json

from datasets import load_dataset
import tiktoken

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs("scaling_laws/results", exist_ok=True)

from config import SMALL, MEDIUM, LARGE
from nanoGPT_annotated.nano_gpt import BiLanguageModel, device

max_iter = 300
eval_iter = 20
learning_rate = 3e-4

model_size = [SMALL, MEDIUM, LARGE]

output = {
    "SMALL" : [],
    "MEDIUM" : [],
    "LARGE" : [],
}

# load dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
enc = tiktoken.get_encoding("gpt2")

# tokenize
def tokenize(examples):
    return enc.encode_ordinary(examples["text"])

# tokenize in batches 
train_tokens = enc.encode_ordinary(
    " ".join([ex["text"] for ex in dataset["train"] if ex["text"].strip()])
)
val_tokens = enc.encode_ordinary(
    " ".join([ex["text"] for ex in dataset["validation"] if ex["text"].strip()])
)

train_data = torch.tensor(train_tokens, dtype=torch.long)
val_data = torch.tensor(val_tokens, dtype=torch.long)



#define a function to sample a batch
def get_batch(split, config):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - config.block_size, (32,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    return x.to(device), y.to(device)

#define a function to return the average estimate of training vs val
@torch.no_grad()
def estimate_loss(model, config):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iter)
        for k in range(eval_iter):
            X, Y = get_batch(split, config)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

for size_name, config in zip(["SMALL", "MEDIUM", "LARGE"], model_size):
    #instantiate a new model for each model size
    model = BiLanguageModel(config).to(device=device)
    #set the mode to train
    model.train()
    #for each model instantiate an optimiser for weight update
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    #total number of parameters in the model
    param_count = sum(p.numel() for p in model.parameters())

    print(f"model_size:{size_name}")
    for iter in range(max_iter):
        #get training batch
        xb, yb = get_batch("train", config)
        xb = xb.to(device)
        yb = yb.to(device)

        #make predictions and compute loss
        _, loss = model(xb, yb)
        #set gradients to zero for each iteration
        optimizer.zero_grad()
        #backpropagation
        loss.backward()
        #update each model's weights 
        optimizer.step()

        #compute tokens seen so far
        seen_token = iter * xb.size(dim=0) * config.block_size
        #compute flops consumed so far using approximation C = 6 N * D
        flops_consumed = 6 * param_count * seen_token
        
        #print out logs
        if iter % eval_iter == 0:
            losses = estimate_loss(model, config)
            print(f"iter:{iter}, train loss:{losses["train"].item():.4f}, val loss:{losses["val"].item():.4f}",
                  f"FLOPS:{flops_consumed}")
            
                    #save logs to file
            out = {
                "step": iter,
                "flops": flops_consumed,  # 6 * N * D
                "val_loss": losses["val"].item()
            }
            output[size_name].append(out)
            with open(f"scaling_laws/results/{size_name}_output.json", "w") as f:
                json.dump({size_name: output[size_name]}, f, indent=2)
    with open(f"scaling_laws/results/{size_name}_output.json", "w") as f:
        json.dump({size_name: output[size_name]}, f, indent=2)






