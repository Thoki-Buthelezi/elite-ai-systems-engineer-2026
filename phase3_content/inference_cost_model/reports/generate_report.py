"""
reports/generate_report.py

Ties the calculator and SLA checker together into the Week 35-36
deliverable: cost at 10K / 1M requests, both API and self-hosted,
checked against latency SLAs from a real load test.

Usage:
    python reports/generate_report.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cost_model import (
    Workload,
    api_cost_per_request,
    self_host_cost_per_request,
    cost_at_scale,
    API_RATES,
    GPU_RATES,
    LatencySLA,
    check_sla,
    check_sla_from_percentiles,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_locust_percentiles(stats_csv_path: Path) -> dict:
    """Reads a Locust --csv=<name> export's <name>_stats.csv and returns
    the Aggregated row's p50/p95/p99, converted from Locust's milliseconds
    to seconds (to match LatencySLA's units)."""
    with open(stats_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["Name"] == "Aggregated":
                return {
                    "p50": float(row["50%"]) / 1000,
                    "p95": float(row["95%"]) / 1000,
                    "p99": float(row["99%"]) / 1000,
                }
    raise ValueError(f"No Aggregated row found in {stats_csv_path}")


def main():
    workload = Workload(avg_input_tokens=300, avg_output_tokens=200)

    print("=== Top-down: hosted API (cost per PROVIDER token) ===")
    for name, rate in API_RATES.items():
        per_req = api_cost_per_request(workload, rate)
        scaled = cost_at_scale(per_req)
        print(f"{name}: ${per_req:.6f}/request -> {scaled}")

    print("\n=== Bottom-up: self-hosted BiLanguageModel (cost per CHARACTER) ===")
    print(
        "NOTE: BiLanguageModel is character-level (vocab_size=65), so its "
        "'tokens' are single characters, not the ~4-char provider tokens "
        "above. These two sections are deliberately NOT converted to a "
        "shared unit and are not meant to be netted against each other in "
        "the report - they represent different use cases (self-hosting "
        "your own trained model vs. calling a hosted API), not two prices "
        "for the same thing."
    )
    # Real measured decode throughput, from benchmark/decode_throughput.py
    # on Kaggle T4, batch_size=1, naive (unbatched) autoregressive decode:
    #   124.7 chars/sec
    # This is ~1.7x the CPU result (71.6 chars/sec), not the 10-50x a rough
    # GPU-vs-CPU ballpark would suggest, because a single-sequence decode
    # step is too small to fill a T4's parallelism - fixed per-step
    # overhead (Python loop, kernel launch, CPU/GPU sync) dominates instead
    # of raw compute. Real batching (many concurrent sequences) would very
    # likely close that gap - a natural follow-up once this baseline exists.
    measured_tokens_per_sec = 124.7

    # This number is ONLY valid for T4 - it was measured on a T4. It is NOT
    # extrapolated to A100/H100/etc below, because the overhead-dominated
    # behavior above means throughput does NOT scale predictably across GPU
    # classes for a workload this small - that would need its own real
    # measurement, not a guess.
    gpu = GPU_RATES["t4"]
    utilization = 0.45  # realistic production duty cycle, not 1.0
    per_req = self_host_cost_per_request(workload, gpu, measured_tokens_per_sec, utilization)
    scaled = cost_at_scale(per_req)
    print(f"t4 (measured): ${per_req:.6f}/request -> {scaled}")
    print(
        "(Other GPU_RATES entries in providers.py are left unused here - "
        "extrapolating this T4 number to A100/H100/etc would be a guess, "
        "not a measurement. Benchmark on those GPUs directly if you need "
        "real figures for them.)"
    )

    print("\n=== SLA check ===")
    # Defaults for a synchronous chat-style response of ~200 output tokens.
    # p50 1.0s: the typical response should feel snappy, not read as
    #   "thinking".
    # p95 3.0s: the worst 1-in-20 requests still land inside a tolerable
    #   wait, doesn't need to be as tight as p50.
    # p99 5.0s: hard ceiling before a meaningful chunk of users would
    #   consider the request to have hung.
    # Tighten or loosen these once you know your actual traffic pattern
    # and what your users will tolerate - these are a reasoned starting
    # point, not a measured requirement.
    sla = LatencySLA(p50_seconds=1.0, p95_seconds=3.0, p99_seconds=5.0)

    stats_csv = REPO_ROOT / "loadtest" / "results_stats.csv"
    if stats_csv.exists():
        observed = load_locust_percentiles(stats_csv)
        result = check_sla_from_percentiles(observed, sla)
        print(f"(loaded real Locust results from {stats_csv})")
    else:
        print(f"No Locust results at {stats_csv}, run the load test first:")
        print(
            "  locust -f loadtest/locustfile.py --host <your-endpoint> "
            "--headless -u 20 -r 5 --run-time 30s --csv=loadtest/results"
        )
        print("Falling back to placeholder samples for now.")
        sample_latencies = [0.3, 0.4, 0.5, 0.6, 0.8, 1.1, 1.9, 2.4, 3.8]
        result = check_sla(sample_latencies, sla)
    print(result)


if __name__ == "__main__":
    main()
