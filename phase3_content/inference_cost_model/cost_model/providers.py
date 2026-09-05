"""
cost_model/providers.py

Pricing data for API providers (top-down, $/token) and raw GPU rental
rates (bottom-up, $/GPU-second). These are placeholder numbers, pull
current rates from the pricing pages before treating any report output
as real:

  OpenAI:   https://platform.openai.com/pricing
  Together: https://docs.together.ai/docs/inference/pricing
  Modal:    https://modal.com/pricing
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ApiRate:
    """Per-million-token pricing for a hosted API model."""
    name: str
    input_per_million: float   # USD per 1M input tokens
    output_per_million: float  # USD per 1M output tokens


@dataclass(frozen=True)
class GpuRate:
    """Per-second rental rate for a GPU type."""
    name: str
    dollars_per_second: float
    vram_gb: float


# Verified against provider pricing pages on 2026-09-02 (short-context,
# standard tier). Re-check before treating these as current - pricing on
# all three of these providers has moved multiple times in 2026.

API_RATES = {
    # https://developers.openai.com/api/docs/pricing
    # gpt-5.6-sol is running promotional pricing through 2026-11-21;
    # confirm it hasn't reverted before using it in a real report.
    "openai-gpt5.6-luna": ApiRate(
        "openai-gpt5.6-luna", input_per_million=0.20, output_per_million=1.20
    ),
    "openai-gpt5.6-terra": ApiRate(
        "openai-gpt5.6-terra", input_per_million=2.00, output_per_million=12.00
    ),
    "openai-gpt5.6-sol": ApiRate(
        "openai-gpt5.6-sol", input_per_million=4.00, output_per_million=20.00
    ),
    # https://together.ai/pricing - flat rate (input == output) for this model
    "together-llama-3.3-70b": ApiRate(
        "together-llama-3.3-70b", input_per_million=0.88, output_per_million=0.88
    ),
}

# https://modal.com/pricing (per-second, on-demand, us region, no
# non-preemptible or region multiplier applied)
GPU_RATES = {
    "t4": GpuRate("t4", dollars_per_second=0.000164, vram_gb=16),
    "l4": GpuRate("l4", dollars_per_second=0.000222, vram_gb=24),
    "a10": GpuRate("a10", dollars_per_second=0.000306, vram_gb=24),
    "a100-40gb": GpuRate("a100-40gb", dollars_per_second=0.000583, vram_gb=40),
    "a100-80gb": GpuRate("a100-80gb", dollars_per_second=0.000694, vram_gb=80),
    "h100-sxm": GpuRate("h100-sxm", dollars_per_second=0.001097, vram_gb=80),
    "h200-sxm": GpuRate("h200-sxm", dollars_per_second=0.001261, vram_gb=141),
}
