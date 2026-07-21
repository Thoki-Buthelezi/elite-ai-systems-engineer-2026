# FSDP vs DDP Benchmark (Week 27-28)

Model: ~1.42B param GPT-2 XL-scale decoder (24 layers, 2048 embd, 16 heads).
Sized so DDP genuinely runs out of memory on T4s

## Kaggle setup
1. New Notebook > Settings > Accelerator > GPU T4 x2
2. Upload `model.py` and `benchmark.py` to `/kaggle/working/`
3. Run in a cell:

```
%cd /kaggle/working
!torchrun --nproc_per_node=2 benchmark.py --mode fsdp
```

Then separately:

```
!torchrun --nproc_per_node=2 benchmark.py --mode ddp
```

Expect the DDP run to hit `torch.cuda.OutOfMemoryError` and get caught,
printing `"oom": true` in the result instead of crashing. 

## Output
Each run writes `results/{mode}_result.json` with peak memory (rank 0),
average step time, tokens/sec, and OOM status. 
