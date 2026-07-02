# nanoGPT GGUF Quantization Artifact

## Overview

This project implements a complete GGUF export and quantization pipeline for a nanoGPT language model trained on Tiny Shakespeare.

The artifact includes:

* Exporting a trained PyTorch checkpoint to GGUF
* Q8_0 quantization
* Q4_K_M quantization (float16 scale approximation)
* Dequantization back to FP32
* Benchmarking model quality (perplexity) and inference throughput (tokens/sec)

The implementation is written from scratch for educational purposes and demonstrates the complete workflow from a trained model to compressed GGUF artifacts.

---

## Project Structure

```
gguf_utils.py
    GGUF binary serialization primitives

quantize.py
    Q8_0 quantization
    Q4_K_M quantization
    Q8_0 dequantization
    Q4_K_M dequantization

export_gguf.py
    Exports the trained nanoGPT checkpoint to GGUF

benchmark.py
    Benchmarks perplexity and throughput

models/
    nanogpt_f32.gguf
    nanogpt_q8_0.gguf
    nanogpt_q4_k_m.gguf

results/
    benchmark.json
```

---

## GGUF Format

GGUF is a binary model format designed for efficient storage and inference of large language models.

Each model consists of:

* file header
* metadata
* tensor descriptors
* raw tensor data

The tensor data is stored using different quantization schemes.

For this artifact:

* **FP32** stores every weight as a 32-bit floating-point value.
* **Q8_0** stores one float32 scale per 32 weights together with 32 signed int8 values.
* **Q4_K_M** stores one float16 scale per 32-weight sub-block and packs two 4-bit values into each byte.

---

## Quantization Methods

### FP32

* 32-bit floating-point weights
* Highest accuracy
* Largest model size

### Q8_0

For every block of 32 weights:

* compute one scale
* quantize each weight to int8
* store one float32 scale and 32 int8 values

Block size:

* 36 bytes per 32 weights

This provides a large reduction in model size while maintaining almost identical model quality.

### Q4_K_M

For every 256-weight super-block:

* split into eight 32-weight sub-blocks
* compute one scale per sub-block
* quantize each weight into a 4-bit value
* store scales as float16
* pack two weights into every byte

The original GGUF implementation uses 6-bit quantized scales. This artifact uses float16 scales as a faithful approximation while keeping the implementation manageable.

---

## Exported Models

| Model  |    Size | Relative Size |
| ------ | ------: | ------------: |
| FP32   | 3.12 MB |          100% |
| Q8_0   | 0.93 MB |           30% |
| Q4_K_M | 0.51 MB |           16% |

---

## Benchmark Results

| Quantization | Perplexity | Tokens/sec |
| ------------ | ---------: | ---------: |
| FP32         |     7.0223 |       6397 |
| Q8_0         |     6.9631 |       3916 |
| Q4_K_M       |     7.1614 |       5063 |

### Observations

* Q8_0 preserves model quality almost perfectly and slightly improves perplexity, likely due to mild regularization from quantization noise.
* Q4_K_M increases perplexity by approximately 2.7%, remaining within an acceptable range while providing significantly better compression.
* Q4_K_M achieves the smallest model size at roughly 16% of the original FP32 checkpoint.

---

## Speed Caveat

The benchmark **does not measure native quantized inference**.

For benchmarking, each quantized tensor is first dequantized back to FP32 before being loaded into the PyTorch model. Consequently, the reported throughput includes Python-side dequantization overhead.

Therefore:

* the throughput numbers are **not representative of real GGUF inference performance**
* they measure the cost of dequantization plus standard FP32 inference

In production inference engines such as GGML or llama.cpp, matrix multiplication is performed directly on quantized weights. Under those implementations, lower-bit formats such as Q4_K_M generally achieve higher throughput than FP32.

---

## Running

Train the model:

```bash
python phase1_content/nanoGPT_annotated/nano_gpt.py
```

Export GGUF models:

```bash
cd phase2_content/quant_4_bit_and_gguf_export
mkdir models
python export_gguf.py ../../phase1_content/nanoGPT_annotated/model.pt f32
python export_gguf.py ../../phase1_content/nanoGPT_annotated/model.pt q8_0
python export_gguf.py ../../phase1_content/nanoGPT_annotated/model.pt q4_k_m
```

Run benchmarks:

```bash
python phase2_content/quant_4_bit_and_gguf_export/benchmark.py
```

Benchmark results will be written to:

```
phase2_content/quant_4_bit_and_gguf_export/results/benchmark.json
```

The exported GGUF models are stored in:

```
phase2_content/quant_4_bit_and_gguf_export/models/
```
