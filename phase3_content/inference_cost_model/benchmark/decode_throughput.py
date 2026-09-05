"""
benchmark/decode_throughput.py

Measures REAL inference decode throughput (tokens/sec) on your own
hardware and model checkpoint. This is what feeds
cost_model.self_host_cost_per_request() - training throughput

This does naive autoregressive decode: no batching across concurrent
users, no KV-cache-reuse tricks beyond whatever your model.forward()
already does internally. It's a conservative single-user baseline.
Wiring in real batching (multiple concurrent sequences at once) will
raise throughput and is a natural next step once this baseline works.

Usage:
    python benchmark/decode_throughput.py --checkpoint path/to/model.pt

With no --checkpoint, runs against a tiny random-weight demo model so
you can confirm the benchmark harness itself works (including on CPU,
no GPU needed) before pointing it at your real checkpoint.
"""

import argparse
import time


import torch
import torch.nn as nn

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


class DemoGPT(nn.Module):
    """Minimal stand-in model with the same forward() shape as a real GPT:
    takes token ids, returns logits over the vocab for the next token.
    Swap load_model() below to load your actual checkpoint instead."""

    def __init__(self, vocab_size=1000, n_embd=128, n_layer=2, block_size=256):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=4, batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        return self.head(x)


def load_model(checkpoint_path, device):
    if checkpoint_path is None:
        print(
            "No --checkpoint given, using a tiny random-weight demo model "
            "to sanity-check the benchmark harness on this machine.\n"
        )
        model = DemoGPT()
    else:
        from phase1_content.nanoGPT_annotated.nano_gpt import BiLanguageModel, config

        model = BiLanguageModel(config=config)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict=state_dict)
    return model.to(device).eval()


def get_vocab_size(model, fallback=1000):
    """Best-effort: read the real vocab size off the model's token
    embedding table, instead of guessing a round number. Feeding random
    ids past the real vocab size causes an IndexError in
    the embedding lookup - this is the bug that just crashed."""
    for attr in ("token_embedding_table", "tok_emb", "wte"):
        emb = getattr(model, attr, None)
        if emb is not None:
            return emb.num_embeddings
    return fallback


def get_block_size(model, fallback=None):
    """Best-effort: read the real max context length off the model's
    position embedding table - same failure mode as vocab size above,
    just for position ids instead of token ids."""
    for attr in ("position_embedding_table", "pos_emb", "wpe"):
        emb = getattr(model, attr, None)
        if emb is not None:
            return emb.num_embeddings
    return getattr(model, "block_size", fallback)


@torch.no_grad()
def benchmark_decode(
    model, device, prompt_len=32, gen_tokens=200, batch_size=1, vocab_size=1000
):
    """Runs a naive autoregressive decode loop and times it end to end."""
    block_size = get_block_size(model)
    idx = torch.randint(0, vocab_size, (batch_size, prompt_len), device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(gen_tokens):
        context = idx[:, -block_size:] if block_size else idx
        output = model(context)
        # nanoGPT-tutorial style forward() returns (logits, loss) so it can
        # compute a loss during training when targets are passed - loss is
        # just None at inference time, but the tuple shape doesn't change.
        logits = output[0] if isinstance(output, tuple) else output
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_token], dim=1)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_tokens_generated = gen_tokens * batch_size
    tokens_per_sec = total_tokens_generated / elapsed

    return {
        "elapsed_seconds": elapsed,
        "tokens_generated": total_tokens_generated,
        "tokens_per_sec": tokens_per_sec,
        "batch_size": batch_size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--gen-tokens", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.checkpoint, device)
    vocab_size = get_vocab_size(model)
    print(f"Detected vocab size: {vocab_size}")
    result = benchmark_decode(
        model,
        device,
        gen_tokens=args.gen_tokens,
        batch_size=args.batch_size,
        vocab_size=vocab_size,
    )

    print(
        f"\nGenerated {result['tokens_generated']} tokens "
        f"(batch_size={result['batch_size']}) in {result['elapsed_seconds']:.2f}s"
    )
    print(f"Decode throughput: {result['tokens_per_sec']:.1f} tokens/sec")


if __name__ == "__main__":
    main()
