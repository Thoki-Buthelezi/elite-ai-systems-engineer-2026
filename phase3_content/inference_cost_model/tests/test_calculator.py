"""Sanity tests for the cost calculator. Run with: pytest tests/"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cost_model.calculator import (
    Workload,
    api_cost_per_request,
    self_host_cost_per_request,
    cost_at_scale,
)
from cost_model.providers import ApiRate, GpuRate


def test_api_cost_per_request():
    workload = Workload(avg_input_tokens=1_000_000, avg_output_tokens=1_000_000)
    rate = ApiRate("test", input_per_million=1.0, output_per_million=2.0)
    assert api_cost_per_request(workload, rate) == 3.0


def test_self_host_cost_scales_with_utilization():
    workload = Workload(avg_input_tokens=0, avg_output_tokens=1000)
    gpu = GpuRate("test-gpu", dollars_per_second=1.0, vram_gb=80)
    full_util = self_host_cost_per_request(
        workload, gpu, tokens_per_second=1000, utilization=1.0
    )
    half_util = self_host_cost_per_request(
        workload, gpu, tokens_per_second=1000, utilization=0.5
    )
    assert half_util == full_util * 2


def test_cost_at_scale():
    scaled = cost_at_scale(0.001, request_counts=(10_000, 1_000_000))
    assert scaled[10_000] == 10.0
    assert scaled[1_000_000] == 1000.0
