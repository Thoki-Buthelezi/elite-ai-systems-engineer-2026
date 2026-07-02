import json
import math
import time

import torch

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from phase1_content.scaling_laws.config import ModelConfig
from phase1_content.nanoGPT_annotated.nano_gpt import (
    BiLanguageModel,
    get_batch,
)

from quantize import (
    quantize_q8_0,
    quantize_q4_k_m,
    dequantize_q8_0,
    dequantize_q4_k_m,
)

# ============================================================
# Configuration
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = ModelConfig(
    vocab_size=65,
    block_size=64,
    n_embd=128,
    n_heads=4,
    n_layers=4,
    dropout=0.0,
)

CHECKPOINT = "phase1_content/nanoGPT_annotated/model.pt"

# ============================================================
# Model Loading
# ============================================================

def load_quantized_model(quant_type: str, device: torch.device):

    model = BiLanguageModel(config)

    state_dict = torch.load(
        CHECKPOINT,
        map_location="cpu",
    )

    reconstructed = {}

    for name, tensor in state_dict.items():

        # Skip integer tensors (there normally aren't any,
        # but this keeps the function generic)
        if not tensor.is_floating_point():
            reconstructed[name] = tensor
            continue

        if quant_type == "f32":

            reconstructed[name] = tensor.float()

        elif quant_type == "q8_0":

            q = quantize_q8_0(tensor)

            reconstructed[name] = dequantize_q8_0(
                q,
                tensor.shape,
            )

        elif quant_type == "q4_k_m":

            q = quantize_q4_k_m(tensor)

            reconstructed[name] = dequantize_q4_k_m(
                q,
                tensor.shape,
            )

        else:
            raise ValueError(f"Unknown quantization type: {quant_type}")

    model.load_state_dict(reconstructed)
    model.to(device)
    model.eval()

    return model


# ============================================================
# Perplexity Benchmark
# ============================================================

@torch.no_grad()
def benchmark_perplexity(
    model,
    n_batches: int = 20,
):

    model.eval()

    total_loss = 0.0

    for _ in range(n_batches):

        x, y = get_batch(
            "val",
            config,
        )

        x = x.to(device)
        y = y.to(device)

        _, loss = model(x, y)

        total_loss += loss.item()

    mean_loss = total_loss / n_batches

    return math.exp(mean_loss)


# ============================================================
# Speed Benchmark
# ============================================================

@torch.no_grad()
def benchmark_speed(
    model,
    n_forward_passes: int = 512,
):

    model.eval()

    x = torch.randint(
        0,
        config.vocab_size,
        (1, config.block_size),
        device=device,
    )

    # Warmup
    for _ in range(5):
        model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(n_forward_passes):
        model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    total_tokens = n_forward_passes * config.block_size

    return total_tokens / elapsed


# ============================================================
# Single Benchmark
# ============================================================

def benchmark(quant_type):

    print("=" * 60)
    print(f"Benchmarking {quant_type.upper()}")
    print("=" * 60)

    model = load_quantized_model(
        quant_type,
        device,
    )

    perplexity = benchmark_perplexity(model)

    tokens_per_sec = benchmark_speed(model)

    print(f"Perplexity : {perplexity:.4f}")
    print(f"Tokens/sec : {tokens_per_sec:.2f}")
    print()

    return {
        "perplexity": perplexity,
        "tokens_per_sec": tokens_per_sec,
    }


# ============================================================
# Main
# ============================================================

def main():

    results = {}

    for quant_type in [
        "f32",
        "q8_0",
        "q4_k_m",
    ]:

        results[quant_type] = benchmark(
            quant_type
        )

    output_path = "phase2_content/quant_4_bit_and_gguf_export/results/benchmark.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print("Benchmark complete.")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()