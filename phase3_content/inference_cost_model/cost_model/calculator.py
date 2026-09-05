"""
cost_model/calculator.py

Two ways to price an inference workload:

1. Top-down (API):        cost = tokens * $/token, from a provider's rate card.
2. Bottom-up (self-host):  cost = GPU-seconds * $/GPU-second, derived from
   your own measured throughput (tokens/sec) and a utilization assumption.

Both converge on the same output: dollars per request, and dollars at
10K / 1M requests.
"""

from dataclasses import dataclass

from .providers import ApiRate, GpuRate


@dataclass
class Workload:
    """Describes one inference request shape, used for both cost paths."""
    avg_input_tokens: int
    avg_output_tokens: int


def api_cost_per_request(workload: Workload, rate: ApiRate) -> float:
    """Top-down: $ per request from a provider's per-token rate card."""
    input_cost = (workload.avg_input_tokens / 1_000_000) * rate.input_per_million
    output_cost = (workload.avg_output_tokens / 1_000_000) * rate.output_per_million
    return input_cost + output_cost


def self_host_cost_per_request(
    workload: Workload,
    gpu: GpuRate,
    tokens_per_second: float,
    utilization: float = 1.0,
) -> float:
    """
    Bottom-up: $ per request from measured throughput on your own hardware.

    tokens_per_second: your benchmarked decode throughput.
    utilization: fraction of GPU time actually doing useful work.
    """
    if tokens_per_second <= 0:
        raise ValueError("tokens_per_second must be positive")
    if not (0 < utilization <= 1.0):
        raise ValueError("utilization must be in (0, 1.0]")

    total_tokens = workload.avg_output_tokens  # decode-bound cost driver
    seconds_of_compute = total_tokens / tokens_per_second
    raw_cost = seconds_of_compute * gpu.dollars_per_second
    return raw_cost / utilization


def cost_at_scale(cost_per_request: float, request_counts=(10_000, 1_000_000)) -> dict:
    """Dollar figures at each request count, e.g. {10000: 12.3, 1000000: 1230.0}."""
    return {n: round(cost_per_request * n, 4) for n in request_counts}
