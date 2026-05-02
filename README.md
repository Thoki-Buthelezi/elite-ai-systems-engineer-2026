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

Week 9 complete.

Artifact: 'Reproducing Chinchilla scaling on a budget' Log-log scaling charts
Next up: Evaluation Science.

---

## Repo structure

```
elite-ai-systems-engineer-2026/
|-- scaling_laws                 implementation of the Chinchilla scaling law
|-- lora_from_scarch             implementation of lora fine-tuning on nanoGPT from scracth
|-- dpo_minial                   implementation of the DPO(Directi Preference Optimization) on the nanoGPT model
|-- nanoGPT_annotated            nanoGPT model implementation with results
|-- phase0_audit                 full directory of content covered in phase 0
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
