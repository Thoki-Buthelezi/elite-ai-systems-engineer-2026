#Distributed Data Parallel DDP #


ddp_train.py — weeks 13-14: distributed data parallel experiment
wraps BiLanguageModel in DDP across 2 GPUs, logs per-step time,
communication overhead, and scaling efficiency vs single-GPU baseline

The experiment was done on Kaggle to make use of the T4 * 2 GPUs