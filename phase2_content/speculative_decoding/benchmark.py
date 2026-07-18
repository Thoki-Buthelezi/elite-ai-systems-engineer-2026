import torch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


from phase1_content.nanoGPT_annotated.nano_gpt import (
    BiLanguageModel,

    get_batch
)

from phase1_content.scaling_laws.config import ModelConfig

mp_config = ModelConfig(vocab_size=65, block_size=64, n_embd=128, n_layers=4, n_heads=4, dropout=0.2)
mq_config = ModelConfig(vocab_size=65, block_size=64, n_embd=64, n_layers=2, n_heads=2, dropout=0.2)

from speculative import speculative_step

@torch.no_grad()
def estimate_loss(model, config:ModelConfig, eval_iters=20):
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
    # 1. Target MODEL 
    # ---------------------------
    target_model = BiLanguageModel(mp_config)
    target_model.load_state_dict(torch.load("phase1_content/nanoGPT_annotated/model.pt", map_location="cpu"))


    base_loss = estimate_loss(target_model, mp_config)
    base_ppl = torch.exp(torch.tensor(base_loss)).item()

    print("=== Target MODEL ===")
    print(f"Loss: {base_loss:.4f}")
    print(f"Perplexity: {base_ppl:.2f}\n")

    # ---------------------------
    # 2. Efficient Approximation model
    # ---------------------------
    draft_model = BiLanguageModel(mq_config)
    draft_model.load_state_dict(torch.load("phase1_content/nanoGPT_annotated/model_mq.pt", map_location="cpu"))


    # calibration data
    calibration_data = []
    for _ in range(10):
        x, y = get_batch("train", mq_config)
        calibration_data.append((x, y))

  
    draft_loss = estimate_loss(draft_model, mq_config)
    draft_ppl = torch.exp(torch.tensor(draft_loss)).item()

    print("=== Draft MODEL ===")
    print(f"Loss: {draft_loss:.4f}")
    print(f"Perplexity: {draft_ppl:.2f}\n")

    # ---------------------------
    # 3. SUMMARY TABLE
    # ---------------------------
    print("=== SUMMARY ===")
    print(f"Δ Loss: {draft_loss - base_loss:+.4f}")
    print(f"Δ Perplexity: {draft_ppl - base_ppl:+.2f}")

    print("\nDone.")

    #---------------------------
    # 4. Speculative Decoding 
    #---------------------------
    print("== Speculative Decoding ==")
    prefix = torch.zeros((1,1),  dtype=torch.long)
    test_result = speculative_step(target_model, draft_model, prefix, 2)

if __name__ == "__main__":
    main()