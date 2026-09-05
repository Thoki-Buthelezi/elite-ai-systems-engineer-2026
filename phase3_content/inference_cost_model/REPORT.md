# Inference Cost & SLA Report — Week 35-36

## 1. What this covers

Two different questions, kept deliberately separate:

1. **If I call a hosted API**, what does it cost per request, and at 10K / 1M
   requests?
2. **If I self-host my own trained model** (`BiLanguageModel`, a
   character-level nanoGPT-style model) on rented GPU hardware, what does
   that cost, and does it meet a latency SLA under load?

These are not two prices for the same product. The API path buys access to
a large, general-purpose model billed in ~4-character "tokens." The
self-host path runs a small model I trained myself, whose vocabulary is 65
single characters, not subword tokens. The numbers below are intentionally
never converted to a shared unit or subtracted from each other, see
Section 5 for why that would be misleading.

## 2. Method

### 2a. Top-down: hosted API

`cost = (input_chars_worth_of_tokens × $/input_token) + (output_tokens ×
$/output_token)`, using each provider's published per-token rate card
(`cost_model/providers.py`, verified against the OpenAI and Together
pricing pages, 2026-09-02).

Workload assumption: 300 input tokens, 200 output tokens per request (a
short chat-style exchange).

### 2b. Bottom-up: self-hosted

`cost = (output_chars / measured_chars_per_second) × $/GPU-second ÷
utilization`

`measured_chars_per_second` comes from `benchmark/decode_throughput.py`,
run against the real `model.pt` checkpoint on a Kaggle T4 GPU:

```
Using device: cuda
Detected vocab size: 65, block size: 64
Generated 200 chars in 1.60s
Decode throughput: 124.7 chars/sec on T4
```

This is naive, unbatched (batch_size=1) autoregressive decode - a
single-user latency baseline, not a throughput-optimized serving setup.
`utilization = 0.45` accounts for the fact that a real production GPU
isn't decoding 100% of the time (traffic isn't constant, there's idle time
between requests).

**Note on the CPU comparison**: the same benchmark on CPU produced 71.6
chars/sec, only ~1.7x slower than the T4 result. That's much smaller than
a typical GPU-vs-CPU gap, because a single unbatched decode step is too
small to fill a T4's parallelism - fixed per-step overhead (Python loop,
kernel launch, CPU/GPU sync) dominates over raw compute at this scale.
Batching many concurrent requests together would very likely widen that
gap substantially; that's a natural next step, not done here.

## 3. Cost results

### Hosted API (cost per *provider token*, 300 in / 200 out)

| Provider / model | $/request | $ at 10K requests | $ at 1M requests |
|---|---|---|---|
| OpenAI gpt-5.6-luna | $0.000300 | $3.00 | $300.00 |
| OpenAI gpt-5.6-terra | $0.003000 | $30.00 | $3,000.00 |
| OpenAI gpt-5.6-sol | $0.005200 | $52.00 | $5,200.00 |
| Together Llama 3.3 70B | $0.000440 | $4.40 | $440.00 |

Notes: gpt-5.6-sol's rate is promotional through 2026-11-21 and will rise
after that. gpt-5.6-luna is the cheapest option here by a wide margin -
worth checking whether its quality is sufficient for the actual use case
before defaulting to it purely on price.

### Self-hosted BiLanguageModel (cost per *character*, T4, 200 output chars)

| Hardware | $/request | $ at 10K requests | $ at 1M requests |
|---|---|---|---|
| T4 (measured) | $0.000585 | $5.85 | $584.51 |

Only T4 is reported here because it's the only hardware this was actually
benchmarked on. `providers.py` has rate cards for L4/A10/A100/H100/H200,
but given how much the overhead-dominated behavior above suggests
throughput doesn't scale predictably at this workload size, extrapolating
to those GPUs without benchmarking them directly would be a guess
presented as a measurement.

## 4. Latency SLA

**Targets** (chat-style response, ~200 output tokens):

| Percentile | Target | Reasoning |
|---|---|---|
| p50 | ≤ 1.0s | Typical response should feel snappy |
| p95 | ≤ 3.0s | Worst 1-in-20 requests still land in a tolerable wait |
| p99 | ≤ 5.0s | Hard ceiling before requests start reading as "hung" |

**Load test setup**: `loadtest/locustfile.py` against
`loadtest/stub_server.py` (a local FastAPI stand-in that simulates decode
latency rather than running the real model - no live deployed endpoint
existed to test against). 20 simulated users, ramped at 5/sec, 20 second
run.

**Result**:

| Percentile | Observed | Target | Pass? |
|---|---|---|---|
| p50 | 2.8s | 1.0s |  Fail |
| p95 | 3.3s | 3.0s |  Fail |
| p99 | 3.3s | 5.0s |  Pass |

The stub's simulated latency was tuned arbitrarily (`SECONDS_PER_TOKEN =
0.02` in `stub_server.py`), so this specific fail/pass pattern is an
artifact of that tuning, not a real finding about the model. What it does
demonstrate correctly: a cost number alone says nothing about whether the
service is usable - this system is "cheap" per Section 3, but by the
p50/p95 targets above, it would currently fail its own SLA. Cost and SLA
have to be reported together, not separately, or the cost number is
misleading on its own.

## 5. Limitations - what these numbers don't show

- **Character vs. token units**: self-host figures price *characters*,
  API figures price *provider tokens* (~4 characters each). They are not
  converted to a shared unit and should not be subtracted or ratioed
  against each other as if they were.
- **Different output lengths in practice**: 200 self-host "tokens" is 200
  characters (~35-40 words); 200 API tokens is roughly 800 characters
  (~140 words). The two workloads aren't actually equivalent in how much
  text gets generated, even before the pricing-unit issue above.
- **Simulated load, not real load**: the SLA numbers come from a stub
  server with made-up latency, not the real model under real traffic.
- **Single-request decode, not batched serving**: the T4 throughput
  reflects one request at a time. Real production serving (continuous
  batching, as in vLLM) would very likely lower true per-request cost by
  using the GPU's idle capacity , this baseline is a conservative,
  probably-pessimistic starting point.
- **One GPU type measured**: T4 only. No real data on whether a bigger
  GPU would help, hurt, or be irrelevant for a model this small.

## 6. Takeaway

At this workload size and this model's scale, self-hosting on a T4
($0.000585/request) lands in the same order of magnitude as the cheapest
API option (OpenAI gpt-5.6-luna, $0.000300/request) , not a slam-dunk
either way, and not really a fair fight given the two aren't pricing the
same unit of output. The more actionable finding is the SLA miss: before
this system could be called "production-ready" at this cost, it needs
either a real load test against the actual model (not the stub) or
changes to hit the latency targets, since a cheap system that fails its
own SLA isn't actually a usable one.
