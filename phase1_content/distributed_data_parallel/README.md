#Distributed Data Parallel DDP #


ddp_train.py — weeks 13-14: distributed data parallel experiment
wraps BiLanguageModel in DDP across 2 GPUs, logs per-step time,
communication overhead, and scaling efficiency vs single-GPU baseline

The experiment was done on Kaggle to make use of the T4 * 2 GPUs

# Measurement Error #

- there is a bug on measuring compute time
- what i did, i put ```python time.perf_counter() ``` only to measure the compute of the backward pass
- the correct way to do it, is to move ``` t_compute_start ``` before the forward pass and place the and place ``` t_compute_end ``` after the backward pass.

# What this means

- this implies that the compute time is higher than the one reported and the real communication overhead is lower.
- however, the conclusion still holds, the model is too small for DDP to help, but the measurement are slightly off

