# GPU Parallelization Plan for GPT-2

## Problem
GPT-2 GPU inference is 1.2s/token when it should be much faster. Only 8 linalg operations are GPU-parallelized while 62 `scf.for` loops run sequentially on CPU with GPU sync overhead between each kernel.

## Goal
Convert ALL CPU-bound operations to GPU-parallelized linalg ops using `linalg.generic` with full `indexing_maps` support.

---

## Phase 0: Infrastructure - Add linalg.generic with indexing_maps

### 0.1 Update ir_gen.rs to parse indexing_maps
**File:** `src/ir_gen.rs`

**Current state:** linalg.generic has basic support but indexing_maps marked as TODO

**Required changes:**
1. Parse `affine_map<(d0,d1) -> (d0,d1)>` syntax from attributes
2. Generate MLIR AffineMapAttr
3. Parse `iterator_types` array attribute (e.g., `["parallel", "parallel"]` or `["parallel", "reduction"]`)
4. Handle reduction iterator type with proper combiner

**Code location:** Lines 1177-1220 in ir_gen.rs (linalg operation handling)

**New syntax to support:**
```lisp
(linalg.generic
  {:indexing_maps [affine_map<(d0,d1) -> (d0,d1)>
                   affine_map<(d0,d1) -> (d1)>
                   affine_map<(d0,d1) -> (d0,d1)>]
   :iterator_types ["parallel" "parallel"]}
  out bias result
  (region
    (block [(: a f32) (: b f32) (: c f32)]
      (def sum (arith.addf a b))
      (linalg.yield sum))))
```

### 0.2 Add linalg.transpose with permutation
**File:** `src/ir_gen.rs`

**Required:** Parse `{:permutation [0, 2, 1]}` attribute for transpose operations

---

## Phase 1: Element-wise Operations via linalg.generic

### 1.1 Bias Additions (4 functions)
**Functions:** `matmul_qkv`, `matmul_attn_proj`, `matmul_fc`, `matmul_fc_proj`

**Convert from:**
```lisp
(scf.for c0 T c1
  (scf.for c0 K c1
    out[t,k] += bias[k]))
```

**Convert to:**
```lisp
(linalg.generic
  {:indexing_maps [affine_map<(d0,d1) -> (d0,d1)>   ;; out
                   affine_map<(d0,d1) -> (d1)>      ;; bias (broadcast)
                   affine_map<(d0,d1) -> (d0,d1)>]  ;; result
   :iterator_types ["parallel" "parallel"]}
  out bias out
  (region
    (block [(: val f32) (: b f32) (: _acc f32)]
      (def result (arith.addf val b))
      (linalg.yield result))))
```

### 1.2 Scaling in batched_qk_matmul
**Convert to:**
```lisp
(linalg.generic
  {:indexing_maps [affine_map<(d0,d1,d2) -> (d0,d1,d2)>
                   affine_map<(d0,d1,d2) -> ()>]  ;; scalar broadcast
   :iterator_types ["parallel" "parallel" "parallel"]}
  scores scale_scalar scores
  (region
    (block [(: val f32) (: s f32) (: _acc f32)]
      (def result (arith.mulf val s))
      (linalg.yield result))))
```

### 1.3 GELU Activation
**Convert to:**
```lisp
(linalg.generic
  {:indexing_maps [affine_map<(d0,d1) -> (d0,d1)>
                   affine_map<(d0,d1) -> (d0,d1)>]
   :iterator_types ["parallel" "parallel"]}
  inp out
  (region
    (block [(: x f32) (: _out f32)]
      ;; GELU formula inline
      (def x2 (arith.mulf x x))
      (def x3 (arith.mulf x2 x))
      ;; ... rest of GELU computation
      (linalg.yield result))))
```

---

## Phase 2: Transpose and Reshape

### 2.1 transpose_k_for_attention
```lisp
(linalg.transpose K K_t {:permutation [0, 2, 1]})
```

### 2.2 reshape_qkv_to_batched
**Pattern:** (64, 2304) -> 3x (12, 64, 64)
```lisp
;; Use linalg.generic for the reshape/split operation
(linalg.generic
  {:indexing_maps [affine_map<(d0,d1,d2) -> (d1, d0*64+d2)>        ;; Q source
                   affine_map<(d0,d1,d2) -> (d0,d1,d2)>]            ;; Q dest
   :iterator_types ["parallel" "parallel" "parallel"]}
  qkv Q_out ...)
```

### 2.3 reshape_attn_output
**Pattern:** (12, 64, 64) -> (64, 768)
```lisp
(linalg.generic
  {:indexing_maps [affine_map<(d0,d1,d2) -> (d1, d0*64+d2)>   ;; source indexing
                   affine_map<(d0,d1,d2) -> (d0,d1,d2)>]       ;; dest
   :iterator_types ["parallel" "parallel" "parallel"]}
  attn_values out ...)
```

---

## Phase 3: Reductions with scf.parallel + scf.reduce

### 3.1 layernorm_forward
**Strategy:** Parallel over T (64 rows), reduce over C (768 elements)

```lisp
;; Pass 1: Compute mean per row
(scf.parallel (t) = (c0) to (T) step (c1) init (zero)
  (region
    (block [(: t index)]
      ;; Reduction over C dimension
      (def row_sum (scf.parallel (c) = (c0) to (C) step (c1) init (zero)
        (region
          (block [(: c index)]
            (def x (memref.load inp t c))
            (scf.reduce (x)
              (region
                (block [(: lhs f32) (: rhs f32)]
                  (scf.reduce.return (arith.addf lhs rhs)))))))))
      (def mean (arith.divf row_sum C_f32))
      (memref.store mean mean_buf t)
      (scf.reduce))))

;; Pass 2: Compute variance (similar pattern)
;; Pass 3: Normalize (element-wise linalg.generic)
```

### 3.2 softmax_logits
```lisp
;; Pass 1: Max reduction
(def max_val (scf.parallel (v) = (c0) to (V) step (c1) init (neg_inf)
  (region
    (block [(: v index)]
      (def val (memref.load logits v))
      (scf.reduce (val)
        (region
          (block [(: lhs f32) (: rhs f32)]
            (scf.reduce.return (arith.maximumf lhs rhs)))))))))

;; Pass 2: Exp and sum (parallel exp, reduce sum)
;; Pass 3: Normalize (linalg.generic element-wise divide)
```

### 3.3 causal_softmax
**Strategy:** Parallel over (head, query_pos), reduction over key positions with dynamic bounds

```lisp
(scf.parallel (h t) = (c0 c0) to (c12 c64) step (c1 c1)
  (region
    (block [(: h index) (: t index)]
      (def valid_len (arith.addi t c1))
      ;; Max reduction over valid_len positions
      ;; Exp-sum reduction
      ;; Normalize
      (scf.reduce))))
```

---

## Phase 4: Gather Operations (Full GPU)

### 4.1 Pre-load weights to memref
**At startup:** Load wte (50257 x 768) and wpe (1024 x 768) into memrefs
```lisp
;; Allocate GPU-accessible weight buffers
(def wte_memref (memref.alloc {:result memref<50257x768xf32>}))
(def wpe_memref (memref.alloc {:result memref<1024x768xf32>}))
(gpu.host_register (memref.cast wte_memref))
(gpu.host_register (memref.cast wpe_memref))

;; Copy from raw pointer to memref (once at startup)
;; ... loading loop ...
```

### 4.2 embedding_lookup via linalg.generic
```lisp
(linalg.generic
  {:indexing_maps [affine_map<(d0,d1) -> (d0)>         ;; token_ids[t]
                   affine_map<(d0,d1) -> (d0,d1)>      ;; wte[token_ids[t], c]
                   affine_map<(d0,d1) -> (d0,d1)>      ;; wpe[t, c]
                   affine_map<(d0,d1) -> (d0,d1)>]     ;; output
   :iterator_types ["parallel" "parallel"]}
  token_ids wte_memref wpe_memref out
  (region
    (block [(: tok i32) (: wte_val f32) (: wpe_val f32) (: _out f32)]
      (def sum (arith.addf wte_val wpe_val))
      (linalg.yield sum))))
```

**Note:** This requires indexed access via the token value, which may need special handling in linalg.generic or an explicit gather op.

### 4.3 logits_forward via linalg.matvec
```lisp
;; Extract single row from activations at position
(def x_row (memref.subview x [pos, 0] [1, 768] [1, 1]))

;; Matrix-vector: wte.T @ x_row -> logits
;; wte is (50257, 768), x_row is (768,), output is (50257,)
(linalg.matvec wte_transposed x_row logits)
```

**Alternative:** Use linalg.matmul with reshaped tensors

---

## Implementation Order

| Step | Task | Effort | Impact | Status |
|------|------|--------|--------|--------|
| **0.1** | Add indexing_maps parsing to ir_gen.rs | High | Critical | ✅ DONE |
| **0.2** | Add iterator_types parsing (parallel/reduction) | Med | Critical | ✅ DONE |
| **0.3** | Add linalg.transpose permutation support | Med | Med | ✅ DONE |
| **1.1** | Convert 4 bias additions to linalg.generic | Med | High | ✅ DONE |
| **1.2** | Convert scaling to linalg.generic | Low | Med | ✅ DONE |
| **1.3** | Convert GELU to linalg.generic | Med | High | ✅ DONE |
| **2.1** | Convert transpose_k_for_attention | Low | Med | ✅ DONE |
| **2.2-2.3** | Convert reshape operations | Med | Med | SKIPPED (needs memref.reshape) |
| **3.1** | Convert layernorm with scf.parallel + reduce | High | High | ❌ CPU FASTER (471ms vs 450ms) |
| **3.2** | Convert softmax_logits | Med | Med | TODO (likely same issue) |
| **3.3** | Convert causal_softmax | High | High | TODO (likely same issue) |
| **4.1** | Pre-load weights to memref | Med | Prereq | ✅ DONE |
| **4.2** | Convert embedding_lookup to scf.parallel | High | Med | ✅ DONE (450ms) |
| **4.3** | Convert logits_forward | Med | Med | TODO |

---

## Key Files to Modify

1. **`src/ir_gen.rs`** (lines 1177-1480)
   - Add indexing_maps attribute parsing
   - Add iterator_types attribute parsing
   - Add reduction combiner support
   - Add transpose permutation attribute

2. **`examples/gpt2/gpt2_generate_gpu.lisp`**
   - Convert all 62 scf.for loops
   - Add weight pre-loading at startup
   - Convert all functions to linalg.generic or scf.parallel

3. **`src/parser.rs`** (if needed)
   - May need updates for affine_map parsing

---

## Current Progress

- **Before:** 62 scf.for loops on CPU, ~100 GPU sync points, 1.2s/token
- **After Phase 1+2:** 44 scf.for loops (reduced by 18)
  - Converted: 4 bias additions, GELU, scaling, K transpose
  - Remaining: layernorm, causal_softmax, embedding_lookup, reshapes, logits
- **After Phase 4.1+4.2:** ~450ms/token
  - Added: wte/wpe weight pre-loading to memref, embedding_lookup GPU parallel
  - Embedding uses scf.parallel with operandSegmentSizes attribute

## Key Findings

### GPU Parallelization Limits

**Reductions don't benefit from simple GPU parallelization:**
- layernorm: 64 rows × 768 column reduction → GPU was 21ms SLOWER (471 vs 450ms)
- Reason: GPU kernel launch overhead dominates for small parallelism (64 threads)
- Each thread still does 768*2 sequential memory accesses
- CPU cache-friendly loops outperform GPU for this pattern

**Operations that DO benefit:**
- Element-wise ops via linalg.generic (high parallelism: 64×768 = 49,152 threads)
- Matrix multiplications (already GPU via linalg.matmul → GPU kernels)
- Embedding lookup (64×768 parallel writes)

**Operations that DON'T benefit:**
- Reductions with small outer dimension (layernorm, softmax per row)
- Would need proper GPU reduction primitives (shared memory, warp shuffles)

## Expected Outcome

- **Current:** ~450ms/token (down from 1.2s, 2.7x improvement)
- **Realistic target:** 300-400ms/token (limited by CPU-bound reductions)
- GPU-only reductions would require custom kernel support not available in current MLIR passes

**Note:** For true 60-120ms/token, would need:
1. Custom GPU reduction kernels with shared memory
2. Fused attention kernels (FlashAttention-style)
3. Batch processing across multiple tokens
