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
    #get the logits from the model
    logits, _ = model(idx)  # (B,T,C)

    #remove the last column on the predictions, we dont have next token
    logits = logits[:, :-1, :]
    #remove the 1st token and shift the targets to the right
    targets = idx[:, 1:]

    #get the log probabilities of each token in the sequence
    log_probs = F.log_softmax(logits, dim=-1)
    #pick the probabilities of each token
    selected = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
     #average over the sequence
    return selected.mean(dim=1)  # (B,)


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


def get_batch(batch_size=8):
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
def train(mode="dpo", iters=2000, lr=5e-6, beta=0.005):
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

            loss = dpo_loss(logp_c, logp_r, logp_ref_c, logp_ref_r, beta) + 0.01 * (logp_c - logp_ref_c).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            margin = (logp_c - logp_r).mean().item()
            print(f"step {step} | loss {loss.item():.4f} | margin {margin:.4f}")

    return model


# -----------------------------
# run
# -----------------------------
if __name__ == "__main__":
    model = train(mode="dpo")

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(context, max_new_tokens=500)[0].tolist()

    print(decode(out))
