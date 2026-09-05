"""
loadtest/locustfile.py

Load test for your inference endpoint. Run with:

    locust -f loadtest/locustfile.py --host https://your-endpoint.example.com

Then open http://localhost:8089, set concurrent users and ramp rate, and
watch the p95/p99 columns. Export with --csv=results to get per-endpoint
stats you can feed into cost_model.sla.check_sla.
"""

import random

from locust import HttpUser, task, between


PROMPTS = [
    "Explain the difference between FSDP and DDP in two sentences.",
    "Summarize what PagedAttention does for KV cache memory.",
    "Write a haiku about GPU memory fragmentation.",
]


class InferenceUser(HttpUser):
    # Wait 0.5-2s between requests per simulated user - tune to match
    # your expected real traffic pattern, not just a default.
    wait_time = between(0.5, 2.0)

    @task
    def generate(self):
        payload = {
            "prompt": random.choice(PROMPTS),
            "max_tokens": 128,
        }
        with self.client.post("/generate", json=payload, catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"status {resp.status_code}")
