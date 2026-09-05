"""
cost_model/sla.py

Latency SLA definitions and a checker that scores a batch of observed
latencies (e.g. pulled from a Locust run) against them.
"""

from dataclasses import dataclass
from statistics import quantiles
from typing import Sequence


@dataclass
class LatencySLA:
    """Target latencies in seconds. Set these deliberately - they're the
    other half of 'cost'. A cheap endpoint that blows its SLA isn't cheap."""
    p50_seconds: float
    p95_seconds: float
    p99_seconds: float


def percentiles(latencies: Sequence[float]) -> dict:
    """p50/p95/p99 from a list of observed request latencies, in seconds."""
    if len(latencies) < 2:
        raise ValueError("Need at least 2 samples to compute percentiles")
    sorted_lat = sorted(latencies)
    q = quantiles(sorted_lat, n=100, method="inclusive")
    return {
        "p50": q[49],
        "p95": q[94],
        "p99": q[98],
    }


def check_sla(latencies: Sequence[float], sla: LatencySLA) -> dict:
    """Returns observed percentiles plus pass/fail against each target."""
    observed = percentiles(latencies)
    return {
        "observed": observed,
        "pass": {
            "p50": observed["p50"] <= sla.p50_seconds,
            "p95": observed["p95"] <= sla.p95_seconds,
            "p99": observed["p99"] <= sla.p99_seconds,
        },
    }


def check_sla_from_percentiles(observed: dict, sla: LatencySLA) -> dict:
    """Like check_sla, but takes percentiles that were already computed
    for you (e.g. straight from a Locust --csv export's 50%/95%/99%
    columns) instead of raw latency samples. Prefer this when you have
    it - Locust's own percentile computation is more trustworthy than
    re-deriving percentiles from a re-sampled or reconstructed dataset."""
    return {
        "observed": observed,
        "pass": {
            "p50": observed["p50"] <= sla.p50_seconds,
            "p95": observed["p95"] <= sla.p95_seconds,
            "p99": observed["p99"] <= sla.p99_seconds,
        },
    }
