# MLIR GPT-2 Performance Plan: Matching llm.c

## Goal

Match llm.c inference performance using pure MLIR.

| Metric | Current | Target | llm.c Reference |
|--------|---------|--------|-----------------|
| Time per token | 460 ms | 6 ms | 5.84 ms |
| Tokens/sec | 2.2 | 170 | 171 |
| Speedup needed | - | **79×** | - |

---

## Performance Gap Analysis

### Why llm.c is 79× faster

| Factor | llm.c | Our Current MLIR | Impact |
|--------|-------|------------------|--------|
| Precision | BF16 | FP32 | 2× memory bandwidth |
| MatMul | hipBLASLt | linalg.matmul→ROCDL | ~1.5× |
| LayerNorm | Fused warp kernel | linalg.generic→parallel | 3-5× |
| Softmax | Fused warp kernel | linalg.generic→parallel | 3-5× |
| GELU | Fused kernel | linalg.generic→parallel | 2× |
| Attention | Tiled + shared memory | Naive full recompute | 5-10× |
| Memory access | Vectorized (128-bit) | Scalar | 2-4× |
| Kernel launches | Fused operations | Many small kernels | 1.5× |

### Compound Effect

These factors multiply: 2 × 1.5 × 4 × 4 × 2 × 7 × 3 × 1.5 ≈ **1500×** theoretical gap. Actual gap is 79× because some factors overlap and the GPU is memory-bound.

---

## Implementation Plan

### Phase 1: BF16 Support (Target: 2× speedup → 230 ms/token)

**Why:** Half the memory = half the transfer time. GPU is memory-bound.

#### 1.1 BF16 Type - Already Works!

`bf16` is a built-in MLIR type, no changes needed:

```lisp
(: x bf16)
(: weights memref<768x768xbf16>)
(arith.constant {:value 1.0 :result bf16})
```

#### 1.2 BF16 Conversion Ops - Already Work!

```lisp
;; Convert f32 ↔ bf16
(arith.truncf {:result bf16} f32_val)
(arith.extf {:result f32} bf16_val)
```

#### 1.3 Update Weight Loading

- Load weights as BF16 (or convert FP32→BF16 at load time)
- Keep accumulation in FP32 for numerical stability

---

### Phase 2: Warp-Level Reductions (Target: 4× speedup → 58 ms/token)

**Why:** LayerNorm and Softmax are 35% of runtime, currently using slow parallel reduction.

#### 2.1 Implement Warp Reduce Sum

Already proven working in `test_warp_reduce.lisp`:

```lisp
(defn warp-reduce-sum [val]
  ;; Tree reduction: 32 → 16 → 8 → 4 → 2 → 1
  (def v16 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} val (: 16 i32) (: 32 i32)))
  (def s16 (arith.addf val v16))
  (def v8 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} s16 (: 8 i32) (: 32 i32)))
  (def s8 (arith.addf s16 v8))
  (def v4 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} s8 (: 4 i32) (: 32 i32)))
  (def s4 (arith.addf s8 v4))
  (def v2 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} s4 (: 2 i32) (: 32 i32)))
  (def s2 (arith.addf s4 v2))
  (def v1 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} s2 (: 1 i32) (: 32 i32)))
  (arith.addf s2 v1))
```

#### 2.2 Implement Warp Reduce Max

```lisp
(defn warp-reduce-max [val]
  (def v16 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} val (: 16 i32) (: 32 i32)))
  (def s16 (arith.maximumf val v16))
  ;; ... same pattern with maximumf
  )
```

#### 2.3 Fused LayerNorm Kernel

**Key insight:** One warp (32 threads) per row, each thread handles 24 elements (768/32).

```lisp
(gpu.launch {:gridSizeX 64 :blockSizeX 32}  ;; 64 rows, 32 threads each
  (region
    (block [...]
      (def row (gpu.block_id x))
      (def lane (gpu.thread_id x))

      ;; Each thread computes partial sum of 24 elements
      (def partial_sum
        (scf.for {:result f32} (: 0 index) (: 24 index) (: 1 index) (: 0.0 f32)
          (region
            (block [(: i index) (: acc f32)]
              (def col (arith.addi lane (arith.muli i (: 32 index))))
              (def x (memref.load inp row col))
              (scf.yield (arith.addf acc x))))))

      ;; Warp reduce to get row sum
      (def row_sum (warp-reduce-sum partial_sum))
      (def mean (arith.divf row_sum (: 768.0 f32)))

      ;; Compute variance (similar pattern)
      ;; ...

      ;; Normalize each element
      (scf.for (: 0 index) (: 24 index) (: 1 index)
        (region
          (block [(: i index)]
            (def col (arith.addi lane (arith.muli i (: 32 index))))
            (def x (memref.load inp row col))
            (def w (memref.load weight col))
            (def b (memref.load bias col))
            (def norm (arith.addf (arith.mulf (arith.mulf (arith.subf x mean) rstd) w) b))
            (memref.store norm out row col)
            (scf.yield))))

      (gpu.terminator))))
```

#### 2.4 Fused Softmax Kernel (Online Algorithm)

**Key insight:** Compute max and sum in single pass, no intermediate storage.

```lisp
(gpu.launch {:gridSizeX num_rows :blockSizeX 32}
  (region
    (block [...]
      (def row (gpu.block_id x))
      (def lane (gpu.thread_id x))

      ;; Each thread finds local max
      (def local_max
        (scf.for {:result f32} ... (: -inf f32)
          (region
            (block [(: i index) (: m f32)]
              (def x (memref.load inp row col))
              (scf.yield (arith.maximumf m x))))))

      ;; Warp reduce max
      (def global_max (warp-reduce-max local_max))

      ;; Each thread computes local exp sum
      (def local_sum
        (scf.for {:result f32} ... (: 0.0 f32)
          (region
            (block [(: i index) (: s f32)]
              (def x (memref.load inp row col))
              (def exp_x (math.exp (arith.subf x global_max)))
              (scf.yield (arith.addf s exp_x))))))

      ;; Warp reduce sum
      (def total_sum (warp-reduce-sum local_sum))
      (def scale (arith.divf (: 1.0 f32) total_sum))

      ;; Write normalized values
      (scf.for ...
        (region
          (block [(: i index)]
            (def x (memref.load inp row col))
            (def exp_x (math.exp (arith.subf x global_max)))
            (def softmax_x (arith.mulf exp_x scale))
            (memref.store softmax_x out row col)
            (scf.yield))))

      (gpu.terminator))))
```

---

### Phase 3: Unified Compilation Pipeline (Target: 2× speedup → 29 ms/token)

**Why:** Current pipeline can't mix explicit `gpu.launch` with linalg→GPU path.

#### 3.1 New Pass Pipeline

```lisp
(compilation
  (target rocm
    ;; All operations use explicit gpu.launch
    (pass lower-affine)
    (pass convert-scf-to-cf)
    (pass gpu-kernel-outlining)
    (pass rocdl-attach-target {:chip "gfx1151"})
    (pass convert-gpu-to-rocdl {:use-bare-ptr-memref-call-conv true})
    (pass gpu-module-to-binary)
    (pass gpu-to-llvm {:use-bare-pointers-for-kernels true})
    (pass convert-cf-to-llvm)
    (pass convert-arith-to-llvm)
    (pass convert-math-to-llvm)
    (pass convert-index-to-llvm)
    (pass finalize-memref-to-llvm)
    (pass convert-func-to-llvm)
    (pass reconcile-unrealized-casts)))
```

---

### Phase 4: Fused Attention Kernel (Target: 3× speedup → 10 ms/token)

**Why:** Attention is the bottleneck. Current implementation recomputes full QK^T for each token.

#### 4.1 Tiled Attention with Shared Memory

```lisp
(gpu.launch {:gridSizeX num_heads :blockSizeX 128}
  (region
    (block [...]
      (def head (gpu.block_id x))
      (def tid (gpu.thread_id x))

      ;; Shared memory for Q, K, V tiles (64×64 each)
      (def q_tile (memref.alloc {:addressSpace workgroup} : memref<64x64xbf16>))
      (def k_tile (memref.alloc {:addressSpace workgroup} : memref<64x64xbf16>))
      (def v_tile (memref.alloc {:addressSpace workgroup} : memref<64x64xbf16>))

      ;; Cooperative load Q tile
      (cooperative-load Q q_tile ...)
      (gpu.barrier)

      ;; Initialize accumulator in registers
      (def acc (vector.splat (: 0.0 f32) : vector<64xf32>))
      (def m_i (vector.splat (: -inf f32) : vector<64xf32>))
      (def l_i (vector.splat (: 0.0 f32) : vector<64xf32>))

      ;; Loop over K,V tiles
      (scf.for {:iter_args [acc m_i l_i]} (: 0 index) num_kv_tiles (: 1 index)
        (region
          (block [(: tile index) (: acc ...) (: m_i ...) (: l_i ...)]
            ;; Load K,V tiles
            (cooperative-load K k_tile tile)
            (cooperative-load V v_tile tile)
            (gpu.barrier)

            ;; Compute S = Q @ K^T (using tensor cores if available)
            (def S (tile-matmul q_tile k_tile))

            ;; Online softmax update
            (def m_ij (row-max S))
            (def m_new (arith.maximumf m_i m_ij))
            (def alpha (math.exp (arith.subf m_i m_new)))
            (def beta (math.exp (arith.subf m_ij m_new)))
            (def P (scale-exp S m_new))

            ;; Update accumulator: acc = alpha * acc + P @ V
            (def acc_new (arith.addf
                           (arith.mulf alpha acc)
                           (tile-matmul P v_tile)))
            (def l_new (arith.addf (arith.mulf alpha l_i) (row-sum P)))

            (gpu.barrier)
            (scf.yield acc_new m_new l_new))))

      ;; Final normalization
      (def output (arith.divf acc l_i))
      (cooperative-store output O head)

      (gpu.terminator))))
```

#### 4.2 KV Cache Support

**Critical for inference:** Don't recompute attention for previous tokens.

```lisp
;; Structure to hold cached K,V for each layer
(def kv_cache
  {:k (memref.alloc : memref<12x1024x768xbf16>)  ;; [layers, max_seq, channels]
   :v (memref.alloc : memref<12x1024x768xbf16>)})

;; During generation, only compute new K,V and append to cache
(defn attention-with-cache [Q k_cache v_cache seq_pos]
  ;; Compute K,V for current position only
  (def K_new (matmul x Wk))
  (def V_new (matmul x Wv))

  ;; Store in cache at seq_pos
  (memref.store K_new k_cache seq_pos)
  (memref.store V_new v_cache seq_pos)

  ;; Attention uses cached K,V up to seq_pos
  (attention Q (slice k_cache 0 seq_pos) (slice v_cache 0 seq_pos)))
```

---

### Phase 5: Vectorized Memory Access (Target: 1.5× speedup → 6.7 ms/token)

**Why:** Loading 4 floats at once is 4× more efficient than scalar loads.

#### 5.1 Add Vector Type Support

```lisp
(: v vector<4xbf16>)
(def v (vector.load mem idx : vector<4xbf16>))
(vector.store v mem idx)
```

#### 5.2 Update Kernels for Coalesced Access

```lisp
;; Each thread loads 4 elements at a time
(def tid (gpu.thread_id x))
(def base (arith.muli tid (: 4 index)))
(def v (vector.load mem base : vector<4xbf16>))
```

---

### Phase 6: Kernel Fusion (Target: 1.2× speedup → 5.6 ms/token)

**Why:** Reduce kernel launch overhead and intermediate memory traffic.

#### 6.1 Fused Residual + LayerNorm

```lisp
;; Instead of:
;;   residual = x + attn_out
;;   norm = layernorm(residual)
;; Do both in one kernel:

(gpu.launch ...
  (region
    (block [...]
      ;; Load x and attn_out
      (def x_val (memref.load x ...))
      (def attn_val (memref.load attn_out ...))

      ;; Residual add
      (def residual (arith.addf x_val attn_val))

      ;; Immediately do layernorm (same kernel)
      ;; ... warp reduce for mean/var ...
      (def norm (layernorm-inline residual mean rstd weight bias))

      (memref.store norm out ...)
      (gpu.terminator))))
```

#### 6.2 Fused MLP Block

```lisp
;; Fuse: Linear1 → GELU → Linear2
;; Share intermediate results in registers/shared memory
```

---

## Implementation Order

| Phase | Effort | Impact | Dependencies | Target |
|-------|--------|--------|--------------|--------|
| 1. BF16 | Medium | 2× | None | 230 ms |
| 2. Warp Reductions | Medium | 4× | New pipeline | 58 ms |
| 3. Pipeline Unification | High | 2× | Phase 2 | 29 ms |
| 4. Fused Attention | High | 3× | Phase 3, Shared mem | 10 ms |
| 5. Vectorization | Low | 1.5× | BF16 | 6.7 ms |
| 6. Kernel Fusion | Medium | 1.2× | All above | 5.6 ms |

**Recommended order:** 3 → 2 → 1 → 4 → 5 → 6

Start with pipeline unification because it unblocks everything else.

---

## Required MLIR Dialect Support

### Currently Working
- `gpu.launch` with block/thread dimensions
- `gpu.shuffle` for warp communication
- `gpu.barrier` for synchronization
- `gpu.block_id`, `gpu.thread_id`
- `memref` operations
- `arith`, `math` operations
- `scf.for`, `scf.if`

### Need to Add/Verify
- `memref.alloc` with `#gpu.address_space<workgroup>` (shared memory)
- `vector.load`/`vector.store` with bf16
- `amdgpu.mfma` or `gpu.subgroup_mma` for tensor cores (optional, for extra speed)

---

## Validation Strategy

### Unit Tests
1. Warp reduce sum: Input [0,1,...,31] → Output 496 (all lanes)
2. Warp reduce max: Input [0,1,...,31] → Output 31 (all lanes)
3. LayerNorm: Compare against CPU reference
4. Softmax: Compare against CPU reference
5. Attention: Compare against CPU reference

### Integration Tests
1. Single transformer block: Compare output against PyTorch
2. Full model: Compare generated tokens against llm.c

### Performance Tests
1. LayerNorm kernel: Target < 0.1 ms for 64×768
2. Softmax kernel: Target < 0.1 ms for 64×64
3. Attention kernel: Target < 1 ms for seq_len=64
4. Full forward pass: Target < 6 ms

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Pipeline incompatibility | Start fresh with explicit gpu.launch only |
| Shared memory not working | Fallback to register tiling |
| BF16 numerical issues | Keep accumulation in FP32 |
| Tensor cores unavailable on gfx1151 | Use standard FMA (still fast) |
| KV cache complexity | Implement incrementally, validate each step |

---

## Success Criteria

**Minimum success:** 50 ms/token (9× speedup, 10× behind llm.c)
**Good success:** 15 ms/token (30× speedup, 3× behind llm.c)
**Full success:** 6 ms/token (77× speedup, matching llm.c)
