# LoRA Fine-Tuning (Low-Rank Adaptation)

This repository contains an implementation and experiments for **LoRA (Low-Rank Adaptation)** applied to transformer-based language models.

LoRA is a parameter-efficient fine-tuning method that freezes the base model weights and injects trainable low-rank matrices into attention layers.

---

## What is LoRA

Instead of updating all parameters of a large model, LoRA learns two small matrices:

\[
W' = W + \Delta W = W + A B
\]

where:
- W is the frozen pretrained weight matrix  
- A and B are low-rank trainable matrices  
- rank r << d, so training is much cheaper

This reduces:
- memory usage  
- compute cost  
- training time  

while maintaining strong performance.

---

## Why this repo

This repo is designed to:
- implement LoRA from scratch (or with minimal libraries)
- demonstrate fine-tuning on a downstream task
- compare full fine-tuning vs LoRA
- experiment with rank size, scaling, and stability

---

## Features

- LoRA injected into attention layers (Q, V projections)
- Configurable rank `r`
- Scalable alpha parameter
- Frozen base model weights
- Lightweight training loop
- Logging for loss and evaluation metrics
- Easy swap between full fine-tuning and LoRA mode

---

## My Numbers
- 664k total params, 46k trainable
- 6.9% of the model doing all the learning
- loss: 1.88 down to 1.83 over 2200 steps