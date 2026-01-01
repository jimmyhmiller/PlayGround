# GPU Optimization Roadmap: Pure MLIR Approach

> **See also**: [MLIR_GPU_BEST_PRACTICES.md](./MLIR_GPU_BEST_PRACTICES.md) - Comprehensive guide to optimal MLIR-only GPU patterns

## Current State vs Target

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| **Per-token latency** | 450ms | 100-150ms | 3-4× |
| **Layernorm** | CPU sequential | GPU warp reductions | Major |
| **Softmax** | CPU sequential | GPU warp reductions | Major |
| **MatMul** | GPU (linalg→ROCDL) | Already GPU | Minor tuning |
| **Memory access** | Scalar loads | Vectorized | Moderate |

## Performance Breakdown (Estimated)

Based on 450ms/token:
- **Matrix multiplications**: ~40% (180ms) - Already GPU
- **Reductions (layernorm, softmax)**: ~35% (160ms) - CPU-bound ← **Main target**
- **Memory transfers/sync**: ~15% (65ms) - CPU↔GPU overhead
- **Other (embeddings, residuals)**: ~10% (45ms)

---

## Phase 1: gpu.shuffle for Warp Reductions (Highest Impact)

**Goal:** GPU-accelerate layernorm and softmax using MLIR's gpu.shuffle
**Expected improvement:** 160ms → ~30-50ms (3-5× faster for reductions)

### 1.1 Add gpu.shuffle Operation Support

**File:** `src/ir_gen.rs`

MLIR's GPU dialect has `gpu.shuffle` for warp-level communication:
```mlir
%result, %valid = gpu.shuffle xor %val, %offset, %width : f32
```

This lowers to:
- NVIDIA: `__shfl_xor_sync`
- AMD/ROCm: `__shfl_xor` (wavefront operation)

**Syntax to add:**
```lisp
;; Returns (value, valid) tuple
(gpu.shuffle {:mode "xor" :width 32 :result [f32 i1]} val offset)
```

**Modes available:**
- `xor` - XOR shuffle (for reductions)
- `up` - Shift up
- `down` - Shift down
- `idx` - Direct index

### 1.2 Implement Warp Reduce Sum

Using shuffle XOR pattern for parallel reduction:

```lisp
;; Reduce 32 values to 1 using tree reduction
;; Each step halves the active lanes
(defn warp-reduce-sum [val]
  (let [v16 (arith.addf val (gpu.shuffle {:mode "xor"} val (: 16 i32)))
        v8  (arith.addf v16 (gpu.shuffle {:mode "xor"} v16 (: 8 i32)))
        v4  (arith.addf v8  (gpu.shuffle {:mode "xor"} v8  (: 4 i32)))
        v2  (arith.addf v4  (gpu.shuffle {:mode "xor"} v4  (: 2 i32)))
        v1  (arith.addf v2  (gpu.shuffle {:mode "xor"} v2  (: 1 i32)))]
    v1))
```

### 1.3 Rewrite LayerNorm with gpu.launch + Warp Shuffles

**Strategy:**
- Launch 64 blocks × 32 threads (1 warp per row)
- Each thread loads C/32 = 24 elements, computes partial sum
- Warp shuffle reduces to single sum per row
- Broadcast result back via shuffle

```lisp
(func.func {:sym_name "layernorm_forward_gpu"
            :function_type (-> [memref<64x768xf32> ...] [])}
  (region
    (block [(: out ...) (: inp ...) (: weight ...) (: bias ...) ...]

      ;; Launch kernel: 64 blocks, 32 threads each
      (gpu.launch {:gridSizeX 64 :blockSizeX 32}
        (region
          (block [(: bx index) (: tx index) ...]
            (def row bx)  ;; Each block handles one row
            (def lane tx) ;; Thread lane within warp (0-31)

            ;; Each thread sums 24 elements (stride by 32)
            (def partial_sum
              (scf.for {:result f32} (: 0 index) (: 24 index) (: 1 index) (: 0.0 f32)
                (region
                  (block [(: i index) (: acc f32)]
                    (def col (arith.addi (arith.muli i (: 32 index)) lane))
                    (def x (memref.load inp row col))
                    (scf.yield (arith.addf acc x))))))

            ;; Warp reduce to get row sum (all lanes get result)
            (def row_sum (warp-reduce-sum partial_sum))
            (def mean (arith.divf row_sum (: 768.0 f32)))

            ;; Similar for variance...
            ;; Then normalize each element

            (gpu.terminator)))))))
```

### 1.4 Rewrite Softmax with Online Algorithm

Online softmax computes max and sum in single pass:

```lisp
;; Each thread tracks local max and adjusted sum
(def local_max -inf)
(def local_sum 0.0)

(scf.for ...
  ;; For each element this thread handles:
  (when (arith.cmpf "ogt" x local_max)
    ;; New max found - adjust running sum
    (set! local_sum (arith.mulf local_sum (math.exp (arith.subf old_max x))))
    (set! local_max x))
  (set! local_sum (arith.addf local_sum (math.exp (arith.subf x local_max)))))

;; Warp reduce max (need warp-reduce-max)
(def global_max (warp-reduce-max local_max))

;; Adjust local sums to global max, then reduce
(def adjusted_sum (arith.mulf local_sum (math.exp (arith.subf local_max global_max))))
(def total_sum (warp-reduce-sum adjusted_sum))

;; Normalize
(def scale (arith.divf (: 1.0 f32) total_sum))
```

---

## Phase 2: Vectorized Memory Access (Medium Impact)

**Goal:** 4× memory bandwidth via MLIR vector dialect
**Expected improvement:** ~30-50ms saved

### 2.1 Add vector Type Support

**File:** `src/ir_gen.rs`

```lisp
;; Vector type syntax
(: val vector<4xf32>)
```

### 2.2 Add vector.load/store Operations

```lisp
;; Load 4 floats at once (128 bits)
(def v (vector.load {:result vector<4xf32>} mem idx))

;; Store 4 floats
(vector.store v mem idx)
```

### 2.3 Update Kernels to Use Vectors

Instead of 4 scalar iterations:
```lisp
;; Old: 4 scalar loads
(def x0 (memref.load mem i))
(def x1 (memref.load mem (+ i 1)))
...

;; New: 1 vector load
(def v (vector.load mem i))  ;; Gets 4 floats
(def sum (vector.reduction "add" v))  ;; Sum them
```

---

## Phase 3: Kernel Fusion via Custom gpu.launch (Medium Impact)

**Goal:** Reduce kernel launches and memory traffic
**Expected improvement:** ~30-50ms saved

### 3.1 Fused Residual + LayerNorm

Current: 2 separate operations
```lisp
(def residual_out (arith.addf x residual))
(layernorm_forward out residual_out weight bias ...)
```

Fused: Single kernel
```lisp
(fused_residual_layernorm out x residual weight bias ...)
```

### 3.2 Fused Attention Components

Fuse where possible:
- Score computation + scaling
- Softmax + value multiplication

---

## Phase 4: Reduce CPU↔GPU Sync (Lower Impact)

### 4.1 Batch Kernel Launches

Group independent operations into single launch where possible.

### 4.2 Async Memory Operations

Use `gpu.memcpy async` where supported.

### 4.3 Stream/Queue Optimization

Pipeline independent operations.

---

## Implementation Order

| Step | Task | Effort | Impact | Status |
|------|------|--------|--------|--------|
| **1.1** | Add gpu.shuffle to ir_gen.rs | Medium | Critical | ✅ DONE (works generically) |
| **1.2** | Test gpu.shuffle → ROCm lowering | Low | Critical | ✅ DONE (verified XOR shuffle) |
| **1.3** | Implement warp-reduce-sum | Low | High | ✅ DONE (496 = sum 0-31) |
| **1.4** | Rewrite layernorm with warps | Medium | High | ⚠️ DONE but no speedup (459ms vs 450ms) |
| **1.5** | Rewrite softmax with warps | Medium | High | ❌ BLOCKED (pipeline incompatibility) |
| **2.1** | Add vector type support | Medium | Medium | TODO |
| **2.2** | Add vector.load/store | Medium | Medium | TODO |
| **3.1** | Fused residual+layernorm | Medium | Medium | TODO |
| **3.2** | MLIR fusion passes (linalg-fuse-elementwise-ops) | Low | Low | ❌ NO BENEFIT (455ms vs 453ms baseline) |

## Key Findings (2024-01-01)

### gpu.launch and gpu.shuffle work with ROCm!

The explicit `gpu.launch` approach works when using the correct pass pipeline:
```lisp
(compilation
  (target rocm
    (pass lower-affine)
    (pass convert-scf-to-cf)
    (pass gpu-kernel-outlining)
    (pass rocdl-attach-target)
    (pass convert-gpu-to-rocdl {:use-bare-ptr-memref-call-conv true})
    (pass gpu-module-to-binary)
    (pass gpu-to-llvm {:use-bare-pointers-for-kernels true})
    ...))
```

Key requirements:
- Need `lower-affine` pass before SCF conversion
- Need `{:use-bare-ptr-memref-call-conv true}` on convert-gpu-to-rocdl
- Need `{:use-bare-pointers-for-kernels true}` on gpu-to-llvm

### Warp reduction pattern verified

Tested full warp reduction using XOR shuffles:
```lisp
;; Tree reduction: 32 → 16 → 8 → 4 → 2 → 1
(def other16 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} val c16_i32 c32_i32))
(def sum16 (arith.addf val other16))
(def other8 (gpu.shuffle {:mode "#gpu<shuffle_mode xor>"} sum16 c8_i32 c32_i32))
;; ... continues for 4, 2, 1
```

Result: Successfully computes sum of 0+1+...+31 = 496 using 5 shuffle operations.

## Key Findings (2026-01-01) - Integration Challenges

### Explicit gpu.launch is incompatible with linalg→GPU pipeline

The main GPT-2 code uses the `linalg→parallel loops→GPU` pipeline:
```lisp
(compilation
  (target rocm
    (pass convert-linalg-to-parallel-loops)
    (pass scf-parallel-loop-tiling)
    (pass gpu-map-parallel-loops)
    (pass convert-parallel-loops-to-gpu)
    ;; ... rest of pipeline without bare-pointer options
    (pass convert-gpu-to-rocdl)  ;; No bare-ptr for strided memref support
    (pass gpu-to-llvm)))
```

This is **incompatible** with explicit `gpu.launch` kernels that need:
```lisp
(pass convert-gpu-to-rocdl {:use-bare-ptr-memref-call-conv true})
(pass gpu-to-llvm {:use-bare-pointers-for-kernels true})
```

### Integration results

| Configuration | Performance | Notes |
|--------------|-------------|-------|
| linalg.generic softmax (baseline) | 450ms/token | Works correctly |
| Warp-based layernorm only | 459ms/token | Essentially same, no speedup |
| Warp-based layernorm + softmax | 836ms/token | Regression to CPU baseline! |

### Why warp reductions don't help

1. **Pipeline conflict**: Mixing explicit `gpu.launch` with linalg-generated kernels causes the linalg kernels to fail silently, falling back to CPU execution.

2. **linalg path is already efficient**: The `convert-linalg-to-parallel-loops` path already converts reductions to GPU reasonably well. The overhead of explicit kernel launches offsets any theoretical shuffle benefit.

3. **Strided memref incompatibility**: The GPT-2 code uses strided memrefs for weight/bias slices (e.g., `memref<768xf32, strided<[1], offset: ?>>`), which require the non-bare-pointer passes.

### Recommendations

1. **Keep the linalg→GPU path** for all operations - it's already providing ~450ms/token
2. **Focus on other optimizations**: vectorized memory access, kernel fusion, reducing sync overhead
3. **If warp reductions are needed**: Would require rewriting the entire compilation pipeline to use only explicit gpu.launch, losing the convenience of linalg's automatic parallelization

## Key Findings (2026-01-01) - MLIR Fusion Passes

### Tested Passes

Tested adding MLIR fusion passes before `convert-linalg-to-parallel-loops`:
- `linalg-fuse-elementwise-ops` - Fuses element-wise operations
- `linalg-fold-unit-extent-dims` - Simplifies dimensions
- `linalg-generalize-named-ops` - Uniform representation
- `scf-parallel-loop-fusion` - Fuses parallel loops

### Results

| Configuration | Performance | Notes |
|--------------|-------------|-------|
| Baseline (no fusion passes) | 453ms/token | Current working state |
| linalg-fuse-elementwise-ops only | 455ms/token | Slight regression |
| All fusion passes | 459ms/token | No improvement |

### Why Fusion Passes Don't Help

1. **Already efficient**: The linalg→parallel loops→GPU pipeline already handles operations efficiently
2. **Memory-bound**: At ~450ms/token, the bottleneck is likely memory bandwidth, not kernel launch overhead
3. **Incompatible patterns**: Many operations have reduction dimensions that prevent fusion
4. **Interference**: Some passes (like `linalg-generalize-named-ops`) may pessimize specialized linalg ops

### Conclusion

MLIR's automatic fusion passes do not provide benefit for this workload. Manual fusion would be needed, but is blocked by the pipeline incompatibility with explicit gpu.launch kernels.

---

## Technical Investigation Needed

### Question 1: Does gpu.shuffle lower correctly to ROCm?

Need to test:
```lisp
(gpu.launch {:gridSizeX 1 :blockSizeX 32}
  (region
    (block [(: tx index)]
      (def val (arith.sitofp tx))
      (def shuffled (gpu.shuffle {:mode "xor"} val (: 1 i32) (: 32 i32)))
      ;; Print or store result
      (gpu.terminator))))
```

### Question 2: What's the wavefront size on target AMD GPU?

- gfx1151 (RDNA3): Likely 32
- Need to verify and possibly make configurable

### Question 3: Does MLIR's gpu.launch support enough control?

Need: grid size, block size, shared memory allocation
Check if all are available in current MLIR version.

---

## Expected Performance Trajectory

| Milestone | Latency | Improvement |
|-----------|---------|-------------|
| Current | 450ms | baseline |
| After warp reductions (Phase 1) | 280-320ms | 1.4-1.6× |
| After vectorization (Phase 2) | 230-280ms | 1.6-2× |
| After fusion (Phase 3) | 180-230ms | 2-2.5× |
| After sync optimization (Phase 4) | 150-200ms | 2.2-3× |

---

## First Step: Validate gpu.shuffle

Before committing to this approach, need to verify gpu.shuffle works end-to-end:

1. Add minimal gpu.shuffle support to ir_gen.rs
2. Create test file with simple shuffle operation
3. Run through ROCm pipeline
4. Verify correct output

If gpu.shuffle doesn't lower correctly to ROCm, we'll need to either:
- Fix the MLIR→ROCDL lowering
- Use inline assembly via `llvm.inline_asm`
- Accept CPU-bound reductions as a limitation

---

## Stretch Goal: Shared Memory

If warp shuffles aren't enough, MLIR also supports shared memory:

```mlir
%shm = gpu.alloc () : memref<32xf32, #gpu.address_space<workgroup>>
```

This could enable block-level reductions for larger problems.
