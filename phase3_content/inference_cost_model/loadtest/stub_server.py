"""
loadtest/stub_server.py

Minimal local inference-endpoint stand-in, so you can run and understand
loadtest/locustfile.py before pointing it at a real deployed model.
Simulates decode latency (scales with max_tokens, plus jitter to mimic
contention under concurrent load) instead of running a real model, so
it starts instantly and needs no GPU.

Run:
    uvicorn loadtest.stub_server:app --reload

Then in another terminal:
    locust -f loadtest/locustfile.py --host http://127.0.0.1:8000
    # open http://localhost:8089 to drive it
"""

import random
import time

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 128


# Simulated per-token decode time. Tune this to roughly match your real
# measured_tokens_per_sec from benchmark/decode_throughput.py - e.g. if
# your model does 40 tok/s, that's about 0.025s/token.
SECONDS_PER_TOKEN = 0.02


@app.post("/generate")
def generate(req: GenerateRequest):
    # Jitter simulates real hardware contention under concurrent load -
    # without it every request would take identically long, which is
    # unrealistic and makes the p95/p99 gap meaningless to look at.
    simulated_latency = req.max_tokens * SECONDS_PER_TOKEN * random.uniform(0.9, 1.3)
    time.sleep(simulated_latency)
    return {
        "text": f"[stub response, {req.max_tokens} tokens simulated]",
        "tokens_generated": req.max_tokens,
    }
