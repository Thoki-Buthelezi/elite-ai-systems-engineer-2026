# Elite AI Systems Engineer — 52-week plan

This is my personal repo documenting a structured 52-week self-study program
aimed at landing an AI systems engineering role. The focus is on
actually understanding how modern AI systems work, not just using libraries.

Every week has a shippable artifact — working code, a trained model, a
benchmark, or a writeup. No week is complete without one.

---

## What this is

I am building up from first principles across four phases: transformers and
alignment, CUDA and inference, distributed systems, and a final capstone.
Each phase has concrete deliverables tied to real engineering skills.

The approach throughout is to implement things from scratch, understand the
math behind them, run experiments, and document what I find.

---

## Phases

### Phase I — Transformers, LoRA, Alignment (weeks 3-14)

The focus here is on transformer architecture, fine-tuning with LoRA, and
alignment fundamentals. Starting with a nanoGPT reproduction and working up
to a fine-tuned model with a full evaluation pipeline.

### Phase II — CUDA, Quantisation, Inference (weeks 15-26)

Getting into the systems layer. Custom CUDA kernels, model quantisation,
and inference optimisation. The goal is to understand what happens after
training — how models actually run fast in production.

### Phase III — Distributed Systems, OSS contributions (weeks 27-36)

Distributed training mechanics, parallelism strategies, and contributing to
real open-source projects. At least two merged PRs by the end of this phase.

### Phase IV — Capstone and public presence (weeks 37-52)

Pulling everything together into a full AI system with a training pipeline,
evaluation infrastructure, cost-performance analysis, and a public writeup.
The capstone needs to score at least 85% on the graduation rubric.

---

## Graduation requirements

To close out the program I need to have shipped:

- 1 capstone project
- 52 weekly reports
- 3 or more polished repos
- 2 or more merged OSS pull requests
- 5 or more blog posts
- all benchmarks documented with real dollar figures
- capstone rubric score of 85% or above

---

## Current status

Week 3 complete.

Artifact: nanoGPT reproduction trained on tinyShakespeare. Character-level
tokenizer, 3-block transformer, 4 attention heads, n_embd=128. Trained for
5000 steps. Final val loss: 1.98. Train/val gap: 0.07.

Next up: LoRA implementation and fine-tuning.

---

## Repo structure

```
elite-ai-systems-engineer-2026/
|-- assets/          plots, results, and visualisations
|-- datasets/        raw data used in experiments
|-- docs/            longer writeups and notes
|-- experiments/     scripts for running experiments
|-- reports/         weekly reports, one per week
|-- src/             model implementations and core code
|-- requirements.txt
|-- README.md
```

---

## References

- Attention Is All You Need — Vaswani et al.
- nanoGPT — Andrej Karpathy
- LoRA: Low-Rank Adaptation of Large Language Models — Hu et al.
- Deep Learning — Goodfellow, Bengio, Courville
- Understanding Deep Learning — Simon Prince
- Mathematics for Machine Learning — Deisenroth, Faisal, Ong