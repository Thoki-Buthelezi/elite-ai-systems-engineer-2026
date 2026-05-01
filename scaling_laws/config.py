from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int
    n_embd: int
    n_heads: int
    n_layers: int
    dropout: float = 0.2

SMALL = ModelConfig(
    vocab_size=50257,     
    block_size=256,
    n_embd=64,
    n_heads=4,
    n_layers=4,
    dropout=0.1
)

MEDIUM = ModelConfig(
    vocab_size=50257,
    block_size=256,
    n_embd=128,
    n_heads=4,
    n_layers=4,
    dropout=0.1
)

LARGE = ModelConfig(
    vocab_size=50257,
    block_size=256,
    n_embd=256,
    n_heads=8,
    n_layers=6,
    dropout=0.1
)
