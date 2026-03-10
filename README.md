# Elite AI Systems Engineer

This repository documents my structured journey toward becoming an **AI Systems Engineer**.  
The goal of this project is to deeply understand how modern AI systems work by **studying theory, implementing algorithms from first principles, and running controlled experiments**.

The study plan spans **52 weeks (~1040 hours)** and focuses on the mathematical, algorithmic, and systems aspects of machine learning.

Instead of relying only on libraries, the approach emphasizes:

- implementing algorithms from scratch
- understanding the mathematics behind them
- benchmarking and documenting results
- building increasingly complex AI systems

---

# Learning Plan

The learning process is divided into four main phases.

## Phase I — Mathematical & Optimization Core

Focus:

- Linear algebra
- Gradient-based optimization
- Backpropagation
- Neural network fundamentals

Implementations:

- Linear regression
- SGD and adaptive optimizers
- computational graphs
- mini autograd engine
- CNN training

Experiments include benchmarking optimizers and understanding convergence behaviour.

---

## Phase II — Architecture & Training Systems

Focus:

- transformer architectures
- GPT-style models
- distributed training mechanics
- LoRA fine-tuning
- reinforcement learning from human feedback (RLHF)

Deliverables include:

- training a small transformer model
- implementing LoRA injection
- building an evaluation pipeline

---

## Phase III — Systems & Inference Optimization

Focus:

- model quantization
- inference acceleration
- latency profiling
- deployment pipelines

Implementations include:

- symmetric and asymmetric quantization
- speculative decoding
- inference API deployment

---

## Phase IV — Capstone AI System

The final phase integrates all components into a full AI system.

Tasks include:

- training pipeline design
- evaluation infrastructure
- performance optimization
- cost-performance analysis
- deployment of a complete AI system

---

# Current Work

### Optimization Algorithms

The current focus is implementing and benchmarking several optimization algorithms from scratch.

Implemented optimizers:

- Adam
- AdaGrad
- RMSProp
- Momentum

These implementations are tested on a simple regression task to study their convergence behaviour.

Example experiment:

- recover parameter θ from synthetic data generated using  
  **y = 3x**

Results and analysis are documented in:

---

# Repository Structure

### Directory Overview

**src/**  
Contains implementations of algorithms and models.

**experiments/**  
Scripts used to run experiments and generate results.

**docs/**  
Detailed documentation of experiments and findings.

**visualizations/**  
Graphs and animations showing optimization behaviour.

---

# Example Experiment

A simple regression model was used to evaluate optimizer behaviour.

Model: ŷ = θx
Objective: minimize MSE(ŷ, y)


Dataset:

- 100 synthetic samples
- generated using y = 3x

The goal is for the optimizer to recover: θ ≈ 3


Detailed findings are available in the experiment report.

---

# Motivation

Modern AI systems are built from several layers of abstraction.  
This project focuses on understanding those layers by rebuilding them step by step.

Key goals include:

- understanding optimization dynamics
- implementing training algorithms
- studying neural architectures
- building scalable AI systems

By the end of the program, the repository should contain a full implementation pipeline covering:

- training
- evaluation
- optimization
- deployment

---

# Future Experiments

Planned experiments include:

- learning rate sensitivity analysis
- optimizer convergence comparison
- visualization of gradient descent trajectories
- animated optimization paths
- scaling law experiments
- transformer training

---

# References

Primary resources used throughout this study include:

- *Deep Learning* — Goodfellow, Bengio, Courville
- *Understanding Deep Learning* — Simon Prince
- *Mathematics for Machine Learning*
- research papers on transformers, optimization, and scaling laws

---

# Progress

This repository will evolve over time as each phase of the learning plan is completed.  
All experiments, implementations, and reports will be documented here.