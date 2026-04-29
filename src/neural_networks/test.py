import torch

from nano_gpt import BiLanguageModel, device, decode

m = BiLanguageModel().to(device)
m.load_state_dict(torch.load("nano_gpt_model.pt", map_location=device))
m.eval()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))