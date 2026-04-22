import torch
import torch.nn as nn

#set seed
torch.manual_seed(1992)


class LoRALinear(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha 
        self.scale = alpha / rank

        # #frozen pretrained layer
        # self.linear.requires_grad_(False)

        #get input and output dimensions
        in_dim = self.linear.in_features
        out_dim = self.linear.out_features

        #rank decomposition matrices A and B
        self.A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)  # Gaussian init
        self.B = nn.Parameter(torch.zeros(out_dim, rank))          # zero init
    
    def forward(self, x):
        # x: (batch, seq, in_dim) OR (batch, in_dim)
        base = self.linear(x)
        # efficient computation
        lora = (x @ self.A.T) @ self.B.T  # x @ A.T @ B.T
        return base +  self.scale * lora

def is_target_layer(name, module):
    return (
        isinstance(module, nn.Linear)
        and ("key" in name or "query" in name or "value" in name)
    )


def get_parent(model, module_name):
    parts = module_name.split(".")
    parent = model

    for p in parts[:-1]:
        parent = getattr(parent, p)

    return parent, parts[-1]

def replace(parent, child_name, new_module):
    setattr(parent, child_name, new_module)

def inject_lora(model, rank=8, alpha=16):
    for name, module in list(model.named_modules()):

        if is_target_layer(name, module):

            parent, child_name = get_parent(model, name)

            # replace with LoRA wrapper
            replace(parent, child_name, LoRALinear(module, rank, alpha))

    return model

#full usage of LoRA
from nano_gpt import BiLanguageModel, get_batch, estimate_loss

device = "cuda" if torch.cuda.is_available() else "cpu"

#load the pretrained model
model = BiLanguageModel()
model.load_state_dict(torch.load("nano_gpt_model.pt", map_location=device))
#inject lora to the  base model
model = inject_lora(model, rank=8, alpha=16)

#freeze everything except lora
for param in model.parameters():
    param.requires_grad = False

#selectively unfreeze lora parameters
for module in model.modules():
    if isinstance(module, LoRALinear):
        module.A.requires_grad = True
        module.B.requires_grad = True

total = 0
trainable = 0

#get the total number of parameters vs trainable parameters
for p in model.parameters():
    total += p.numel()
    if p.requires_grad:
        trainable += p.numel()

print(f"Total Parameters: {total}\n",
      f"Frozen Parameters: {total - trainable}\n",
      f"Trainable %: {trainable / total * 100:.2}\n",
      f"Trainable (LoRA): {trainable}\n",
      f"Alpha: {16}\n"
      f"Rank: {8}"
    )


optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)


def train():
    for iter in range(5000):
        xb, yb = get_batch("train")
        xb, yb = xb.to(device), yb.to(device)
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % 200 == 0:
            losses = estimate_loss(model=model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

if __name__ == "__main__":
    train()