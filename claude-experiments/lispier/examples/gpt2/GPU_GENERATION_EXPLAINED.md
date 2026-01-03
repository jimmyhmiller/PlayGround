# GPT-2 GPU Text Generation in Lispier

A complete GPT-2 124M implementation written in Lispier (a Lisp dialect) that compiles to MLIR and executes on AMD GPUs via ROCm.

## Overview

This file implements autoregressive text generation using the GPT-2 Small architecture:

| Parameter | Value | Description |
|-----------|-------|-------------|
| T | 64 | Maximum sequence length |
| C | 768 | Embedding/channel dimension |
| V | 50257 | Vocabulary size |
| L | 12 | Number of transformer layers |
| NH | 12 | Number of attention heads |
| hs | 64 | Head size (768 ÷ 12) |

The model has ~124 million parameters and generates text one token at a time.

## Compilation Pipeline

Lispier code goes through MLIR passes to target the GPU:

```
Lispier Source (.lisp)
        │
        ▼
   MLIR Linalg Dialect (high-level tensor ops)
        │
        ▼
   Parallel Loop Nests (scf.parallel)
        │
        ▼
   Tiled Loops (16×16 = 256 threads/block)
        │
        ▼
   GPU Dialect (gpu.launch, gpu.shuffle)
        │
        ▼
   ROCDL (AMD GPU intrinsics)
        │
        ▼
   GPU Binary (executed via ROCm runtime)
```

---

## CPU vs GPU Execution

### What Runs on CPU

| Operation | Why CPU? |
|-----------|----------|
| File I/O (checkpoint, tokenizer) | Sequential file reads |
| Weight loading & preprocessing | One-time setup cost |
| Memory allocation | Host memory management |
| Token storage & retrieval | Small, sequential access |
| Argmax over logits | Sequential scan, 50K elements |
| Softmax over logits | Sequential, runs once per token |
| Token printing | I/O bound |
| Generation loop control | Orchestration |

### What Runs on GPU

| Operation | Parallelism | Threads |
|-----------|-------------|---------|
| Embedding lookup | Per (position, channel) | 64 × 768 = 49,152 |
| LayerNorm | Per row, warp reduction | 64 × 32 = 2,048 |
| QKV projection | Per output element | 64 × 2304 = 147,456 |
| Attention projection | Per output element | 64 × 768 = 49,152 |
| Q·K^T matmul | Batched, per (head, query, key) | 12 × 64 × 64 = 49,152 |
| Causal softmax | Per (head, query), warp | 768 × 32 = 24,576 |
| Attention·V matmul | Batched, per output | 12 × 64 × 64 = 49,152 |
| MLP FC (768→3072) | Per output element | 64 × 3072 = 196,608 |
| GELU activation | Per element | 64 × 3072 = 196,608 |
| MLP projection (3072→768) | Per output element | 64 × 768 = 49,152 |
| Residual additions | Per element | 64 × 768 = 49,152 |

---

## Execution Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INITIALIZATION (CPU)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. Load checkpoint header (256 ints) → extract model config                │
│  2. Allocate ~500MB for parameters, read from disk                          │
│  3. Load tokenizer (vocab strings)                                          │
│  4. Allocate activation buffers (memref)                                    │
│  5. Register all buffers with GPU runtime (gpu.host_register)               │
│  6. Pre-load all 12 layers of weights into GPU-visible memory               │
│     - Convert weight matrices from F32 → BF16 (50% memory reduction)        │
│     - Transpose weights from checkpoint format to computation format        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GENERATION LOOP (20 iterations)                      │
├──────────────────────────┬──────────────────────────────────────────────────┤
│         CPU              │                      GPU                          │
├──────────────────────────┼──────────────────────────────────────────────────┤
│                          │                                                   │
│  for step in 1..21:      │                                                   │
│    │                     │                                                   │
│    ├─── Launch ──────────┼──► Embedding Lookup                              │
│    │                     │    ┌─────────────────────────────────────────┐   │
│    │                     │    │ 64×768 threads: each loads wte + wpe    │   │
│    │                     │    │ out[t,c] = wte[token[t], c] + wpe[t, c] │   │
│    │                     │    └─────────────────────────────────────────┘   │
│    │                     │                       │                           │
│    │                     │                       ▼                           │
│    │                     │    ┌─────────────────────────────────────────┐   │
│    │  for layer in 0..12:│    │         TRANSFORMER BLOCK ×12           │   │
│    │    │                │    ├─────────────────────────────────────────┤   │
│    │    ├── Launch ──────┼──► │ LayerNorm1 (64 warps, shuffle reduce)   │   │
│    │    ├── Launch ──────┼──► │ QKV Matmul (linalg.generic, BF16→F32)   │   │
│    │    ├── Launch ──────┼──► │ Reshape QKV → Q, K, V batched           │   │
│    │    ├── Launch ──────┼──► │ Transpose K → K^T                       │   │
│    │    ├── Launch ──────┼──► │ Q @ K^T (linalg.batch_matmul)           │   │
│    │    ├── Launch ──────┼──► │ Causal Softmax (768 warps, fused)       │   │
│    │    ├── Launch ──────┼──► │ Attn @ V (linalg.batch_matmul)          │   │
│    │    ├── Launch ──────┼──► │ Reshape attention output                │   │
│    │    ├── Launch ──────┼──► │ Attention Projection Matmul             │   │
│    │    ├── Launch ──────┼──► │ Residual Add (linalg.add)               │   │
│    │    ├── Launch ──────┼──► │ LayerNorm2                              │   │
│    │    ├── Launch ──────┼──► │ MLP FC Matmul (768→3072)                │   │
│    │    ├── Launch ──────┼──► │ GELU Activation                         │   │
│    │    ├── Launch ──────┼──► │ MLP Projection Matmul (3072→768)        │   │
│    │    └── Launch ──────┼──► │ Residual Add                            │   │
│    │                     │    └─────────────────────────────────────────┘   │
│    │                     │                       │                           │
│    ├─── Launch ──────────┼──► Final LayerNorm                               │
│    │                     │                       │                           │
│    │  ◄── Sync ──────────┼───────────────────────┘                           │
│    │                     │                                                   │
│    ├─ Logits (CPU) ──────┤  (50257 dot products, sequential)                │
│    ├─ Softmax (CPU) ─────┤  (numerically stable, sequential)                │
│    ├─ Argmax (CPU) ──────┤  (find max index)                                │
│    ├─ Print token ───────┤                                                   │
│    └─ Store token ───────┤                                                   │
│                          │                                                   │
└──────────────────────────┴──────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLEANUP (CPU)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Print timing statistics, deallocate all buffers, free parameter memory     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Parallelism Analysis

### Matrix Multiplications

The largest computational cost. Each uses `linalg.generic` with 3 iterator dimensions:

```
Dimensions: (M, N, K)
- M, N: parallel (can run independently)
- K: reduction (must accumulate)

Example: QKV projection (64, 2304, 768)
- 64 × 2304 = 147,456 independent output elements
- Each performs 768 multiply-accumulates
- Tiled into 16×16 thread blocks = 576 blocks
```

### LayerNorm (Warp-Based Reduction)

Each of the 64 rows processed by one warp (32 threads):

```
┌──────────────────────────────────────────────────────────────┐
│  Row 0:  Thread 0-31 each handle 24 elements (768/32)        │
│          ├── Local sum (24 adds per thread)                  │
│          ├── Warp shuffle XOR reduce (5 rounds: 16,8,4,2,1)  │
│          ├── Broadcast mean to all 32 threads                │
│          ├── Local variance (24 ops per thread)              │
│          ├── Warp shuffle reduce for variance                │
│          └── Apply normalization (24 writes per thread)      │
├──────────────────────────────────────────────────────────────┤
│  Row 1-63: Same pattern, all 64 warps run in parallel        │
└──────────────────────────────────────────────────────────────┘
```

Total: 64 blocks × 32 threads = 2,048 threads, but effectively 64-way parallelism at block level.

### Causal Softmax (Fused Kernel)

768 blocks (12 heads × 64 query positions), each with 32 threads:

```
Block (head=h, query=q):
  ├── Find max score (2 elements/thread, warp reduce)
  ├── Compute exp sum (2 elements/thread, warp reduce)
  └── Normalize & write (2 elements/thread)

Causal mask applied inline: score = (key ≤ query) ? score : -∞
```

### Batched Attention Matmuls

Uses `linalg.batch_matmul` for Q@K^T and Attn@V:

```
Shape: (12, 64, 64) @ (12, 64, 64) → (12, 64, 64)
- 12 independent batch dimensions (heads)
- 64×64 = 4096 output elements per head
- Total: 49,152 parallel output computations
```

---

## Memory Layout & Bandwidth Optimization

### BF16 Weight Compression

Weight matrices stored as BF16 (2 bytes) instead of F32 (4 bytes):

```
┌─────────────────────────────────────────────────────┐
│  Weight Type          │  F32 Size  │  BF16 Size    │
├───────────────────────┼────────────┼───────────────┤
│  QKV weights (×12)    │  ~81 MB    │  ~40 MB       │
│  Attn proj (×12)      │  ~27 MB    │  ~13 MB       │
│  MLP FC (×12)         │  ~108 MB   │  ~54 MB       │
│  MLP proj (×12)       │  ~108 MB   │  ~54 MB       │
├───────────────────────┼────────────┼───────────────┤
│  Total                │  ~324 MB   │  ~161 MB      │
└─────────────────────────────────────────────────────┘

Computation: Load BF16 → Extend to F32 → Accumulate in F32
```

### Zero-Copy Weight Access

Weights pre-loaded once, accessed via `memref.subview`:

```lisp
;; No data movement - just pointer arithmetic
(def qkv_w_view (memref.subview {...} all_qkv_w layer))
```

### GPU Memory Registration

All buffers registered with `gpu.host_register` for unified memory access:

```lisp
(gpu.host_register (memref.cast {:result "memref<*xf32>"} x))
```

This enables the GPU to directly access host memory without explicit copies.

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Time per token | ~450 ms |
| Tokens generated | 20 |
| Total generation time | ~9 seconds |
| GPU | AMD gfx1151 (ROCm) |
| Memory footprint | ~500 MB parameters + ~50 MB activations |

### Sample Output
```
[GPT-2 Config]
  max_seq_len: 1024
  vocab_size: 50257
  num_layers: 12
  num_heads: 12
  channels: 768
  total_params: 124475904

Prompt: <|endoftext|>
Generated: The first time I saw the new version of the game, I was so excited. I was

Time for 20 tokens: 8979 ms
Per token: 448 ms
```

### Bottleneck Analysis

1. **Matrix multiplications** dominate compute time (~80%)
2. **Memory bandwidth** is the limiting factor for large matmuls
3. **Kernel launch overhead** from many small kernels per layer
4. **CPU logits computation** is sequential (could be GPU-accelerated)

---

## Code Organization

```
gpt2_generate_gpu.lisp
├── Dialect requires (gpu, func, linalg, scf, etc.)
├── Compilation pipeline (MLIR passes for ROCm)
├── External function declarations (malloc, fopen, printf, etc.)
├── GPU kernel functions
│   ├── matmul_qkv, matmul_attn_proj, matmul_fc, matmul_fc_proj
│   ├── layernorm_forward (warp-based)
│   ├── attention_forward (orchestrator)
│   │   ├── reshape_qkv_to_batched
│   │   ├── transpose_k_for_attention
│   │   ├── batched_qk_matmul
│   │   ├── causal_softmax (fused warp kernel)
│   │   ├── batched_attn_v_matmul
│   │   └── reshape_attn_output
│   ├── gelu_forward
│   ├── residual_forward, copy_buffer
│   └── embedding_lookup
├── CPU functions
│   ├── logits_forward, softmax_logits, argmax
│   ├── tokenizer_init, print_token
│   └── main (orchestration)
└── main function
    ├── Load checkpoint & tokenizer
    ├── Allocate & register buffers
    ├── Pre-load all layer weights (F32→BF16, transpose)
    ├── Generation loop (20 tokens)
    └── Cleanup
```
