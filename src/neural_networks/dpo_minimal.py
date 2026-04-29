import torch
import torch.nn.functional as F
import random

from nano_gpt import BiLanguageModel, encode, decode, device, block_size

torch.manual_seed(1997)
random.seed(1997)



# -----------------------------
# sequence logprob (core)
# a function to get the log probabilities of the entire sequence in the model
# -----------------------------
def sequence_logprob(model, idx):
    logits, _ = model(idx)

    logits = logits[:, :-1, :]
    targets = idx[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    selected = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

    # mask padding (assume pad token = 0 for now)
    mask = (targets != 0).float()

    return (selected * mask).sum(dim=1) / mask.sum(dim=1)


# -----------------------------
# DPO loss
# -----------------------------
def dpo_loss(logp_c, logp_r, logp_ref_c, logp_ref_r, beta=0.1):
    pi_diff = logp_c - logp_r
    ref_diff = logp_ref_c - logp_ref_r

    logits = beta * (pi_diff - ref_diff)
    loss = -F.logsigmoid(logits)

    return loss.mean()


# -----------------------------
# simple synthetic preference dataset
# -----------------------------
def build_pair():
    prompt = "KING:"

    good = [
        " My lords, attend and hear my royal will.",
        " What news dost thou bring from the field?",
        " Speak plainly, for I would know the truth.",
    ]

    bad = [
    " My lord, I guess things are kinda okay maybe.",
    " The king is like not sure what to do.",
    " I think stuff happens and maybe we go.",
]

    y_pos = random.choice(good)
    y_neg = random.choice(bad)

    chosen = encode(prompt + y_pos)
    rejected = encode(prompt + y_neg)

    return torch.tensor(chosen), torch.tensor(rejected)


def get_batch(batch_size=32):
    chosen_batch = []
    rejected_batch = []

    for _ in range(batch_size):
        c, r = build_pair()

        # pad/truncate to block_size
        c = c[:block_size]
        r = r[:block_size]

        c = F.pad(c, (0, block_size - len(c)))
        r = F.pad(r, (0, block_size - len(r)))

        chosen_batch.append(c)
        rejected_batch.append(r)

    return torch.stack(chosen_batch), torch.stack(rejected_batch)


# -----------------------------
# training
# -----------------------------
def train(mode="dpo", iters=2000, lr=5e-6, beta=0.01):
    model = BiLanguageModel().to(device)
    model.load_state_dict(torch.load("nano_gpt_model.pt"))
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # reference model for DPO
    if mode == "dpo":
        ref_model = BiLanguageModel().to(device)
        ref_model.load_state_dict(torch.load("nano_gpt_model.pt"))
        ref_model.eval()

        for p in ref_model.parameters():
            p.requires_grad = False

    for step in range(iters):
        chosen, rejected = get_batch()
        chosen = chosen.to(device)
        rejected = rejected.to(device)

        logp_c = sequence_logprob(model, chosen)
        logp_r = sequence_logprob(model, rejected)

        if mode == "sft":
            loss = -logp_c.mean()

        elif mode == "dpo":
            with torch.no_grad():
                logp_ref_c = sequence_logprob(ref_model, chosen)
                logp_ref_r = sequence_logprob(ref_model, rejected)

            loss = dpo_loss(logp_c, logp_r, logp_ref_c, logp_ref_r, beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            margin = (logp_c - logp_r).mean().item()
            print(f"step {step} | loss {loss.item():.4f} | margin {margin:.4f}")

    return model


def benchmark(sft_model, dpo_model, n_batches=10):
    sft_model.eval()
    dpo_model.eval()
    
    results = {
        "sft_logp_chosen": [],
        "sft_logp_rejected": [],
        "dpo_logp_chosen": [],
        "dpo_logp_rejected": [],
    }
    
    with torch.no_grad():
        for _ in range(n_batches):
            chosen, rejected = get_batch()
            chosen, rejected = chosen.to(device), rejected.to(device)
            
            results["sft_logp_chosen"].append(sequence_logprob(sft_model, chosen).mean().item())
            results["sft_logp_rejected"].append(sequence_logprob(sft_model, rejected).mean().item())
            results["dpo_logp_chosen"].append(sequence_logprob(dpo_model, chosen).mean().item())
            results["dpo_logp_rejected"].append(sequence_logprob(dpo_model, rejected).mean().item())
    
    # average across batches
    avg = {k: sum(v) / len(v) for k, v in results.items()}
    
    sft_margin = avg["sft_logp_chosen"] - avg["sft_logp_rejected"]
    dpo_margin = avg["dpo_logp_chosen"] - avg["dpo_logp_rejected"]
    
    print("\n=== Benchmark Table ===")
    print(f"{'Model':<8} {'logP(chosen)':>14} {'logP(rejected)':>16} {'Margin':>10}")
    print("-" * 52)
    print(f"{'SFT':<8} {avg['sft_logp_chosen']:>14.4f} {avg['sft_logp_rejected']:>16.4f} {sft_margin:>10.4f}")
    print(f"{'DPO':<8} {avg['dpo_logp_chosen']:>14.4f} {avg['dpo_logp_rejected']:>16.4f} {dpo_margin:>10.4f}")
    
    return avg


# -----------------------------
# run
# -----------------------------
if __name__ == "__main__":
    # train SFT baseline
    sft_model = train(mode="sft")
    torch.save(sft_model.state_dict(), "sft_final_model.pt")
    
    # train DPO on top
    dpo_model = train(mode="dpo")
    
    # benchmark both
    avg = benchmark(sft_model, dpo_model, n_batches=10)
    
    # generation samples
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    print("\n=== SFT Generation ===")
    sft_model.eval()
    print(decode(sft_model.generate(context, 200)[0].tolist()))
    
    # print("\n=== DPO Generation ===")
    # dpo_model.eval()
    # print(decode(dpo_model.generate(context, 200)[0].tolist()))