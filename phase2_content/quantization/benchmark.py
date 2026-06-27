import torch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


from phase1_content.nanoGPT_annotated.nano_gpt import (
    BiLanguageModel,
    config,
    get_batch
)

from ptq import PTQuantizer
from gptq import apply_gptq


@torch.no_grad()
def estimate_loss(model, eval_iters=20):
    model.eval()
    losses = []

    for _ in range(eval_iters):
        x, y = get_batch("train", config)
        logits, loss = model(x, y)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def main():

    # ---------------------------
    # 1. BASE MODEL (no quantization)
    # ---------------------------
    model_base = BiLanguageModel(config)
    model_base.load_state_dict(
        torch.load(
            "phase1_content/nanoGPT_annotated/model.pt",
            map_location="cpu"
        )
    )

    base_loss = estimate_loss(model_base)
    base_ppl = torch.exp(torch.tensor(base_loss)).item()

    print("=== BASE MODEL ===")
    print(f"Loss: {base_loss:.4f}")
    print(f"Perplexity: {base_ppl:.2f}\n")

    # ---------------------------
    # 2. GPTQ MODEL (fresh copy)
    # ---------------------------
    model_gptq = BiLanguageModel(config)
    model_gptq.load_state_dict(
        torch.load(
            "phase1_content/nanoGPT_annotated/model.pt",
            map_location="cpu"
        )
    )

    # calibration data
    calibration_data = []
    for _ in range(10):
        x, y = get_batch("train", config)
        calibration_data.append((x, y))

    quantizer = PTQuantizer()

    # apply GPTQ
    apply_gptq(
        model=model_gptq,
        calibration_data=calibration_data,
        quantizer=quantizer,
        damping=0.01
    )

    gptq_loss = estimate_loss(model_gptq)
    gptq_ppl = torch.exp(torch.tensor(gptq_loss)).item()

    print("=== GPTQ MODEL ===")
    print(f"Loss: {gptq_loss:.4f}")
    print(f"Perplexity: {gptq_ppl:.2f}\n")

    # ---------------------------
    # 3. SUMMARY TABLE
    # ---------------------------
    print("=== SUMMARY ===")
    print(f"Δ Loss: {gptq_loss - base_loss:+.4f}")
    print(f"Δ Perplexity: {gptq_ppl - base_ppl:+.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()