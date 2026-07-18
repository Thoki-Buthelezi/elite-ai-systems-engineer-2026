import torch
ckpt = torch.load("phase1_content/nanoGPT_annotated/model_mq.pt", map_location="cpu")
sd = ckpt.get("model", ckpt)  # depends how you saved it
n_params = sum(p.numel() for p in sd.values())
print(f"{n_params:,} params ({n_params/1e6:.1f}M)")