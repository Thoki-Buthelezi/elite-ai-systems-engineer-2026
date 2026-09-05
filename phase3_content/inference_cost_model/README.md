# inference_cost_model

Week 35-36 deliverable: cost calculator + SLA-checked load test for an
inference stack, with dollar figures at 10K and 1M requests.

## Structure

- `cost_model/providers.py` - pricing data. API rates verified against the
  OpenAI and Together pricing pages on 2026-09-02; GPU rates from Modal's
  pricing page (same date). Re-check before trusting these later.
- `cost_model/calculator.py` - two cost paths:
  - **top-down**: `api_cost_per_request` (hosted API, $/token)
  - **bottom-up**: `self_host_cost_per_request`
- `cost_model/sla.py` - latency SLA definitions (p50/p95/p99), plus two
  checkers: `check_sla` (from raw latency samples) and
  `check_sla_from_percentiles` (from percentiles Locust already computed).
- `benchmark/decode_throughput.py` - measures REAL inference decode
  throughput on your own hardware. Runs standalone against a tiny demo
  model (no GPU needed) to prove the harness works; point `--checkpoint`
  at your real model to get the number that actually belongs in the
  report.
- `loadtest/locustfile.py` - Locust load test for an inference endpoint.
- `loadtest/stub_server.py` - local FastAPI stand-in endpoint (simulated
  decode latency, no real model) 
- `loadtest/results_stats.csv` - a real captured Locust run against the
  stub server (20 users, 20s), included as a worked example. 
- `reports/generate_report.py` - ties it together: cost at 10K/1M requests
  for both cost paths, plus an SLA check against `loadtest/results_stats.csv`
  if present.
- `tests/test_calculator.py` - sanity tests for the calculator math.
- `REPORT.md` - the actual writeup: real measured numbers, cost at 10K/1M
  requests, SLA results, and the caveats that matter when reading them.

## Setup

    pip install -r requirements.txt

## Measure your real decode throughput

    python benchmark/decode_throughput.py --checkpoint path/to/your/model.pt

Take the printed tokens/sec and use it in `reports/generate_report.py`.

## Run the load test

Against the local stub (to see how it works, no GPU or deployed model
needed):

    uvicorn loadtest.stub_server:app --port 8000 &
    locust -f loadtest/locustfile.py --host http://127.0.0.1:8000 \
        --headless -u 20 -r 5 --run-time 30s --csv=loadtest/results

Against your real endpoint, same command with `--host` pointed at it
instead. Either way this overwrites `loadtest/results_stats.csv`, which
`generate_report.py` reads automatically.

## Generate the report

    python reports/generate_report.py
