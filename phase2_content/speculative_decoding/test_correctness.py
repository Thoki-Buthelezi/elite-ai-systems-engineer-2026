import torch
import time
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from phase1_content.nanoGPT_annotated.nano_gpt import BiLanguageModel, ModelConfig, get_batch
from speculative import speculative_step

mp_config = ModelConfig(vocab_size=65, block_size=64, n_embd=128, n_layers=4, n_heads=4, dropout=0.2)
mq_config = ModelConfig(vocab_size=65, block_size=64, n_embd=64, n_layers=2, n_heads=2, dropout=0.2)

model_p = BiLanguageModel(mp_config)
model_p.load_state_dict(torch.load("phase1_content/nanoGPT_annotated/model.pt", map_location="cpu"))
model_p.eval()

model_q = BiLanguageModel(mq_config)
model_q.load_state_dict(torch.load("phase1_content/nanoGPT_annotated/model_mq.pt", map_location="cpu"))
model_q.eval()


@torch.no_grad()
def plain_autoregressive(model, prefix, n_tokens):
    """Baseline: sample n_tokens one at a time from model_p alone."""
    current = prefix
    for _ in range(n_tokens):
        logits, _ = model(current)
        dist = torch.softmax(logits[:, -1, :], dim=-1)
        x = torch.multinomial(dist, num_samples=1)
        current = torch.cat([current, x], dim=1)
    return current


@torch.no_grad()
def speculative_generate(model_p, model_q, prefix, n_tokens, gamma):
    """Generate n_tokens total using repeated speculative_step calls, tracking acceptance."""
    current = prefix
    tokens_generated = 0
    total_accepted = 0
    total_calls = 0

    while tokens_generated < n_tokens:
        remaining = n_tokens - tokens_generated
        this_gamma = min(gamma, remaining)  # don't overdraft near the end
        accepted = speculative_step(model_p, model_q, current, this_gamma)
        total_calls += 1
        total_accepted += len(accepted)  # rough proxy
        for tok in accepted:
            current = torch.cat([current, tok], dim=1)
            tokens_generated += 1
            if tokens_generated >= n_tokens:
                break

    return current, total_accepted, total_calls


def benchmark_gamma(gamma, n_tokens=50, n_trials=20):
    x, y = get_batch("train", mp_config)
    prefix = x[0:1, :10]  # short, fixed-length starting prefix

    # plain autoregressive timing
    start = time.time()
    for _ in range(n_trials):
        plain_autoregressive(model_p, prefix, n_tokens)
    plain_time = (time.time() - start) / n_trials

    # speculative timing
    start = time.time()
    accept_counts = []
    for _ in range(n_trials):
        _, total_accepted, total_calls = speculative_generate(model_p, model_q, prefix, n_tokens, gamma)
        accept_counts.append(total_accepted / total_calls if total_calls else 0)
    spec_time = (time.time() - start) / n_trials

    avg_accept_rate = sum(accept_counts) / len(accept_counts)
    speedup = plain_time / spec_time

    print(f"=== gamma={gamma} ===")
    print(f"Plain autoregressive: {plain_time*1000:.2f} ms  ({n_tokens/plain_time:.1f} tok/s)")
    print(f"Speculative:          {spec_time*1000:.2f} ms  ({n_tokens/spec_time:.1f} tok/s)")
    print(f"Speedup:              {speedup:.2f}x")
    print(f"Avg tokens/call:      {avg_accept_rate:.2f} (out of gamma={gamma}+1 possible)")
    print()


if __name__ == "__main__":
    for gamma in [2, 4, 8]:
        benchmark_gamma(gamma)