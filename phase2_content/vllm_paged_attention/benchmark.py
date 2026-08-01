import random
from collections import deque
from dataclasses import dataclass

from block_manager import BlockManager


@dataclass
class Sequence:
    id: int
    prompt_len: int
    max_output_len: int
    num_tokens: int

    @property
    def total_reserved(self):
        return self.prompt_len + self.max_output_len

    @property
    def generated_tokens(self):
        return self.num_tokens - self.prompt_len

    def is_done(self):
        return self.generated_tokens >= self.max_output_len


def generate_workload(num_sequences, rng_seed):
    random.seed(rng_seed)
    workload = []
    for seq_id in range(num_sequences):
        r = random.random()
        if r < 0.80:
            prompt = random.randint(20, 200)
        elif r < 0.95:
            prompt = random.randint(200, 800)
        else:
            prompt = random.randint(2000, 4000)
        output = min(int(random.paretovariate(2.0) * 40), 512)
        workload.append(Sequence(id=seq_id, prompt_len=prompt, max_output_len=output, num_tokens=prompt))
    return workload


class NaiveAllocator:
    def __init__(self, total_slots):
        self.total_slots = total_slots
        self.used_slots = 0
        self.reservations = {}

    def can_allocate(self, seq) -> bool:
        return self.used_slots + seq.total_reserved <= self.total_slots

    def allocate(self, seq):
        self.reservations[seq.id] = seq.total_reserved
        self.used_slots += seq.total_reserved

    def free(self, seq):
        self.used_slots -= self.reservations.pop(seq.id)

    def current_used(self):
        return self.used_slots


def preempt_one(active, block_manager, requesting_seq_id):
    """
    Evict a victim sequence to free blocks for `requesting_seq_id`.
    Prefers the most recently admitted OTHER active sequence (so
    longer-running sequences aren't punished for arriving early);
    falls back to self-preemption if the requester is the only one active.

    Uses vLLM's default 'recompute' preemption policy: the victim's
    progress is discarded (num_tokens reset to prompt_len) and it goes
    back into the waiting queue to be re-admitted and regenerated from
    scratch later. (The paper's other option, 'swap', would copy the
    victim's blocks to CPU memory instead of discarding them, more
    expensive to implement, out of scope here.)
    """
    victim = None
    for s in reversed(active):
        if s.id != requesting_seq_id:
            victim = s
            break
    if victim is None:
        victim = next(s for s in active if s.id == requesting_seq_id)

    active[:] = [s for s in active if s.id != victim.id]
    block_manager.free(victim)
    victim.num_tokens = victim.prompt_len
    return victim


def run_simulation(allocator, workload, admit_rate_per_step=8, max_steps=10000, block_manager=None):
    active = []
    waiting = deque(workload)
    stats = {"peak_used": 0, "step_used": [], "completed": 0, "rejected": 0, "preempted": 0}

    while max_steps > 0:
        max_steps -= 1
        for _ in range(admit_rate_per_step):
            if not waiting:
                break
            seq = waiting[0]
            if allocator.can_allocate(seq):
                waiting.popleft()
                allocator.allocate(seq)
                active.append(seq)
            else:
                stats["rejected"] += 1
                break

        for seq in list(active):
            if not any(s.id == seq.id for s in active):
                continue  # preempted earlier this same step, by another sequence's growth

            if block_manager is None:
                seq.num_tokens += 1
                continue

            while not block_manager.can_append_slot(seq):
                victim = preempt_one(active, block_manager, seq.id)
                waiting.appendleft(victim)
                stats["preempted"] += 1
                if not any(s.id == seq.id for s in active):
                    break  # seq preempted itself, nothing left to append to

            if any(s.id == seq.id for s in active):
                block_manager.append_slot(seq)
                seq.num_tokens += 1

        finished = []
        for seq in active:
            if seq.is_done():
                allocator.free(seq)
                finished.append(seq)
                stats["completed"] += 1
        for seq in finished:
            active.remove(seq)

        if block_manager is None:
            used = allocator.current_used()
        else:
            used = block_manager.num_used_blocks() * block_manager.block_size
        stats["step_used"].append(used)
        stats["peak_used"] = max(stats["peak_used"], used)

        if not active and not waiting:
            break

    return stats


def dollars_per_hour(naive_total_steps, paged_total_steps, gpu_hourly_rate, seconds_per_step=0.05):
    naive_hours = (naive_total_steps * seconds_per_step) / 3600
    paged_hours = (paged_total_steps * seconds_per_step) / 3600
    naive_cost = naive_hours * gpu_hourly_rate
    paged_cost = paged_hours * gpu_hourly_rate
    return {
        "step_reduction_pct": 100 * (1 - paged_total_steps / naive_total_steps),
        "naive_cost_per_batch": naive_cost,
        "paged_cost_per_batch": paged_cost,
        "savings_per_batch": naive_cost - paged_cost,
    }


if __name__ == "__main__":
    TOTAL_SLOTS = 15000
    BLOCK_SIZE = 16
    NUM_GPU_BLOCKS = TOTAL_SLOTS // BLOCK_SIZE

    workload_naive = generate_workload(300, rng_seed=42)
    workload_paged = generate_workload(300, rng_seed=42)  # identical workload, separate Sequence objects

    mean_reserved = sum(s.total_reserved for s in workload_naive) / len(workload_naive)
    print(f"Workload: {len(workload_naive)} sequences, mean total_reserved={mean_reserved:.1f} tokens")
    print(f"Capacity: TOTAL_SLOTS={TOTAL_SLOTS}, NUM_GPU_BLOCKS={NUM_GPU_BLOCKS} (block_size={BLOCK_SIZE})")
    print()

    naive_allocator = NaiveAllocator(TOTAL_SLOTS)
    naive_stats = run_simulation(naive_allocator, workload_naive)

    block_manager = BlockManager(NUM_GPU_BLOCKS, BLOCK_SIZE)
    paged_stats = run_simulation(block_manager, workload_paged, block_manager=block_manager)

    econ = dollars_per_hour(len(naive_stats["step_used"]), len(paged_stats["step_used"]), gpu_hourly_rate=1.50)
    print("=== $ translation (A10-ish, $1.50/hr) ===")
    print(econ)