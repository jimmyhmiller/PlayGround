# Optimal GPU Inference with MLIR

The single best approach for high-performance GPU inference using MLIR.

---

## Core Principles

1. **Explicit `gpu.launch` kernels** - Full control over grid/block dimensions
2. **Shared memory tiling** - Keep working data in fast on-chip memory
3. **Tensor core intrinsics** - Use hardware matrix units directly
4. **Warp-level primitives** - Shuffle operations for fast reductions
5. **Vectorized memory access** - Coalesced loads/stores
6. **Kernel fusion** - Minimize memory traffic between operations

---

## The Optimal Pipeline

```mlir
// Use bare pointers for maximum performance
gpu-to-llvm{use-bare-pointers-for-kernels=true}
convert-gpu-to-rocdl{use-bare-ptr-memref-call-conv=true}  // AMD
convert-gpu-to-nvvm{use-bare-ptr-memref-call-conv=true}   // NVIDIA
```

---

## 1. Explicit GPU Kernels

Always use `gpu.launch` for performance-critical operations:

```mlir
gpu.launch blocks(%bx, %by, %bz) in (%grid_x, %grid_y, %grid_z)
           threads(%tx, %ty, %tz) in (%block_x, %block_y, %block_z) {
  // Full control over thread behavior
  gpu.terminator
}
```

**Why**: Automatic lowering (linalg→parallel loops→GPU) cannot express:
- Shared memory usage
- Warp-level operations
- Complex tiling strategies
- Persistent kernels

---

## 2. Shared Memory (On-Chip SRAM)

Use workgroup-scoped memory for data reuse:

```mlir
// Allocate in shared memory (fast, ~100 KB per block)
%shm = memref.alloc() : memref<64x64xf32, #gpu.address_space<workgroup>>

gpu.launch blocks(...) threads(...) {
  // Load from global to shared (coalesced)
  %global_val = memref.load %global[%gidx] : memref<...xf32>
  memref.store %global_val, %shm[%tidx] : memref<64x64xf32, #gpu.address_space<workgroup>>

  // Synchronize threads
  gpu.barrier

  // Compute using shared memory (fast random access)
  %local_val = memref.load %shm[%idx] : memref<64x64xf32, #gpu.address_space<workgroup>>
  // ... compute ...

  gpu.terminator
}
```

**Memory Hierarchy**:
| Level | Size | Latency | Use For |
|-------|------|---------|---------|
| Registers | ~256 KB/SM | 1 cycle | Thread-local accumulators |
| Shared Memory | ~100 KB/block | ~20 cycles | Tile reuse within block |
| L2 Cache | ~6 MB | ~200 cycles | Cross-block reuse |
| Global Memory | 8-80 GB | ~400 cycles | Input/output only |

---

## 3. Tensor Core Operations

Use hardware matrix multiply units directly:

### AMD (RDNA3/CDNA)
```mlir
// Matrix Fused Multiply-Add: D = A @ B + C
%result = amdgpu.mfma %a, %b, %c {
  m = 16, n = 16, k = 16,   // Tile dimensions
  blocks = 1,
  cbsz = 0, abid = 0, blgp = 0
} : vector<4xf16>, vector<4xf16>, vector<4xf32>
```

### NVIDIA (Ampere/Hopper)
```mlir
// Warp-level matrix multiply
%d = nvvm.mma.sync %a, %b, %c {
  shape = #nvvm.mma_shape<m = 16, n = 8, k = 16>,
  a_layout = #nvvm.mma_layout<row>,
  b_layout = #nvvm.mma_layout<col>
} : vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>
```

**Performance**: Tensor cores provide 4-16× throughput vs standard FMA.

---

## 4. Warp-Level Reductions

Use `gpu.shuffle` for fast parallel reductions:

```mlir
// Tree reduction across 32 threads (one warp)
// Sum: 0+1+2+...+31 = 496 in 5 operations

%v16 = gpu.shuffle xor %val, %c16, %c32 : f32
%s16 = arith.addf %val, %v16 : f32

%v8 = gpu.shuffle xor %s16, %c8, %c32 : f32
%s8 = arith.addf %s16, %v8 : f32

%v4 = gpu.shuffle xor %s8, %c4, %c32 : f32
%s4 = arith.addf %s8, %v4 : f32

%v2 = gpu.shuffle xor %s4, %c2, %c32 : f32
%s2 = arith.addf %s4, %v2 : f32

%v1 = gpu.shuffle xor %s2, %c1, %c32 : f32
%result = arith.addf %s2, %v1 : f32  // All lanes have sum
```

**Why**:
- No shared memory needed
- No synchronization barriers
- 5 operations to reduce 32 values

---

## 5. Vectorized Memory Access

Load/store multiple elements per instruction:

```mlir
// Load 4 floats (128 bits) in one instruction
%vec = vector.load %mem[%idx] : memref<1024xf32>, vector<4xf32>

// Coalesced access pattern: adjacent threads access adjacent memory
// Thread 0: [0,1,2,3], Thread 1: [4,5,6,7], ...
%tid = gpu.thread_id x
%base = arith.muli %tid, %c4 : index
%vec = vector.load %mem[%base] : memref<1024xf32>, vector<4xf32>
```

**Memory Bandwidth**: Vector loads achieve 4× bandwidth vs scalar loads.

---

## 6. Flash Attention Pattern

The optimal attention implementation:

```mlir
// Flash Attention: O(N) memory instead of O(N²)
// Tile Q, K, V to fit in shared memory

gpu.launch blocks(%batch, %head, %q_tile) threads(%tx, %ty, %c1) {
  // Allocate shared memory for tiles
  %q_shm = memref.alloc() : memref<64x64xf32, #gpu.address_space<workgroup>>
  %k_shm = memref.alloc() : memref<64x64xf32, #gpu.address_space<workgroup>>
  %v_shm = memref.alloc() : memref<64x64xf32, #gpu.address_space<workgroup>>

  // Load Q tile (stays in shared memory)
  // cooperative load: each thread loads part of the tile
  load_tile(%Q, %q_shm, %q_tile)
  gpu.barrier

  // Initialize accumulators in registers
  %acc = arith.constant dense<0.0> : vector<64xf32>
  %max = arith.constant dense<-inf> : vector<64xf32>
  %sum = arith.constant dense<0.0> : vector<64xf32>

  // Stream through K,V tiles
  scf.for %kv_tile = 0 to %num_tiles {
    // Load K,V tiles to shared memory
    load_tile(%K, %k_shm, %kv_tile)
    load_tile(%V, %v_shm, %kv_tile)
    gpu.barrier

    // Compute attention scores: S = Q @ K^T (in shared memory)
    %scores = matmul_tile(%q_shm, %k_shm)  // Uses tensor cores

    // Online softmax update
    %new_max = max(%max, row_max(%scores))
    %scale = exp(%max - %new_max)
    %acc = %acc * %scale + softmax(%scores, %new_max) @ %v_shm
    %sum = %sum * %scale + row_sum(exp(%scores - %new_max))
    %max = %new_max

    gpu.barrier
  }

  // Final normalization
  %output = %acc / %sum
  store_tile(%output, %O, %q_tile)

  gpu.terminator
}
```

**Benefits**:
- Memory: O(N) vs O(N²) for standard attention
- Speed: 2-4× faster due to reduced memory traffic
- Fused: softmax + matmul in single kernel

---

## 7. Fused LayerNorm

Single-kernel layernorm with warp reductions:

```mlir
// One block per row, one warp per block
gpu.launch blocks(%rows, %c1, %c1) threads(%c32, %c1, %c1) {
  %row = gpu.block_id x
  %lane = gpu.thread_id x

  // Each thread loads multiple elements
  %partial_sum = arith.constant 0.0 : f32
  scf.for %i = 0 to %elements_per_thread {
    %col = %lane + %i * 32
    %x = memref.load %input[%row, %col] : f32
    %partial_sum = arith.addf %partial_sum, %x
  }

  // Warp reduction for mean
  %row_sum = warp_reduce_sum(%partial_sum)  // Using gpu.shuffle
  %mean = arith.divf %row_sum, %num_elements

  // Compute variance (same pattern)
  %partial_var = ...
  %row_var = warp_reduce_sum(%partial_var)
  %rstd = rsqrt(%row_var / %num_elements + %eps)

  // Normalize and write output (fused with weight/bias)
  scf.for %i = 0 to %elements_per_thread {
    %col = %lane + %i * 32
    %x = memref.load %input[%row, %col]
    %w = memref.load %weight[%col]
    %b = memref.load %bias[%col]
    %norm = (%x - %mean) * %rstd * %w + %b
    memref.store %norm, %output[%row, %col]
  }

  gpu.terminator
}
```

---

## 8. Optimal MatMul Tiling

Hierarchical tiling for matrix multiplication:

```mlir
// Tile dimensions (tune for specific GPU)
// Block tile: 128x128, Warp tile: 64x64, Thread tile: 8x8

gpu.launch blocks(%M/128, %N/128, %c1) threads(%c256, %c1, %c1) {
  // Shared memory for A and B tiles
  %a_shm = memref.alloc() : memref<128x32xf32, #gpu.address_space<workgroup>>
  %b_shm = memref.alloc() : memref<32x128xf32, #gpu.address_space<workgroup>>

  // Register file for accumulator
  %acc = alloca() : memref<8x8xf32, #gpu.address_space<private>>

  // Main loop over K dimension
  scf.for %k_tile = 0 to %K step 32 {
    // Cooperative load A,B tiles to shared memory
    load_tile_async(%A, %a_shm, %block_row, %k_tile)
    load_tile_async(%B, %b_shm, %k_tile, %block_col)
    gpu.barrier

    // Compute 128x128 output tile using tensor cores
    scf.for %k = 0 to 32 step 16 {
      %a_frag = load_fragment(%a_shm, %warp_row, %k)
      %b_frag = load_fragment(%b_shm, %k, %warp_col)
      %acc = amdgpu.mfma %a_frag, %b_frag, %acc  // or nvvm.mma.sync
    }

    gpu.barrier
  }

  // Write accumulated result to global memory
  store_tile(%acc, %C, %block_row, %block_col)

  gpu.terminator
}
```

---

## 9. Persistent Kernels

Keep threads alive across multiple operations:

```mlir
// Process entire transformer layer in one kernel launch
gpu.launch blocks(%num_sms, %c1, %c1) threads(%c256, %c1, %c1) {

  // Grid-stride loop: each block processes multiple work items
  scf.for %work_id = %block_id to %total_work step %num_blocks {

    // Decode work item
    %layer = %work_id / %ops_per_layer
    %op = %work_id % %ops_per_layer

    // Execute operation based on type
    scf.switch %op {
      case 0: layernorm_kernel(...)
      case 1: qkv_matmul_kernel(...)
      case 2: attention_kernel(...)
      case 3: projection_kernel(...)
      case 4: mlp_kernel(...)
    }

    // Implicit synchronization via grid-stride pattern
  }

  gpu.terminator
}
```

**Benefits**:
- Eliminates kernel launch overhead
- Better GPU occupancy
- Enables cross-operation optimizations

---

## 10. Memory Prefetching

Overlap computation with memory access:

```mlir
// Double buffering with async copies
%buf_a = memref.alloc() : memref<2x64x64xf32, #gpu.address_space<workgroup>>
%ping = 0, %pong = 1

// Prefetch first tile
async_copy(%global, %buf_a[%ping])

scf.for %tile = 0 to %num_tiles {
  // Wait for current tile
  gpu.barrier

  // Start loading next tile (overlapped with compute)
  %next = (%tile + 1) % %num_tiles
  async_copy(%global, %buf_a[%pong], %next)

  // Compute on current tile
  compute(%buf_a[%ping])

  // Swap buffers
  %ping, %pong = %pong, %ping
}
```

---

## Complete Transformer Block (Optimal)

```mlir
// Single fused kernel for entire attention block
func.func @fused_attention(%Q, %K, %V, %O, %weight, %bias) {
  gpu.launch blocks(%batch, %heads, %q_tiles) threads(%c128, %c1, %c1) {

    // Shared memory
    %q_shm = memref.alloc() : memref<64x64xf16, #gpu.address_space<workgroup>>
    %k_shm = memref.alloc() : memref<64x64xf16, #gpu.address_space<workgroup>>
    %v_shm = memref.alloc() : memref<64x64xf16, #gpu.address_space<workgroup>>

    // Registers
    %acc = vector.splat %c0 : vector<8x8xf32>
    %m_i = vector.splat %-inf : vector<8xf32>
    %l_i = vector.splat %c0 : vector<8xf32>

    // Load Q tile once
    cooperative_load(%Q, %q_shm)
    gpu.barrier

    // Flash attention loop
    scf.for %j = 0 to %kv_tiles {
      cooperative_load(%K, %k_shm, %j)
      cooperative_load(%V, %v_shm, %j)
      gpu.barrier

      // S = Q @ K^T using tensor cores
      %S = mfma_matmul(%q_shm, transpose(%k_shm))

      // Online softmax
      %m_ij = row_max(%S)
      %m_new = max(%m_i, %m_ij)
      %alpha = exp(%m_i - %m_new)
      %beta = exp(%m_ij - %m_new)
      %l_i = %alpha * %l_i + %beta * row_sum(exp(%S - %m_ij))

      // Update accumulator
      %P = exp(%S - %m_new)
      %acc = %alpha * %acc + mfma_matmul(%P, %v_shm)
      %m_i = %m_new

      gpu.barrier
    }

    // Final output
    %O_tile = %acc / broadcast(%l_i)
    cooperative_store(%O_tile, %O)

    gpu.terminator
  }
  return
}
```

---

## Expected Performance

With all optimizations applied:

| Model | Baseline | Optimized | Speedup |
|-------|----------|-----------|---------|
| GPT-2 124M | 830ms/token (CPU) | ~50-80ms/token | 10-16× |
| GPT-2 124M | 450ms/token (naive GPU) | ~50-80ms/token | 5-9× |

**Performance breakdown**:
- Tensor cores: 4-8× over standard FMA
- Flash attention: 2-4× over standard attention
- Kernel fusion: 1.5-2× from reduced memory traffic
- Warp reductions: 1.2-1.5× for LayerNorm/Softmax

---

## Summary

The optimal MLIR GPU approach:

1. **Always use explicit `gpu.launch`** - Never rely on automatic lowering for performance code
2. **Tile everything to shared memory** - Global memory is 20× slower
3. **Use tensor cores** - `amdgpu.mfma` / `nvvm.mma.sync` for all matmuls
4. **Warp shuffles for reductions** - `gpu.shuffle` instead of shared memory atomics
5. **Fuse aggressively** - One kernel per transformer layer, not per operation
6. **Vectorize all memory access** - `vector.load/store` with 128-bit+ transactions
7. **Double buffer** - Overlap memory access with computation
8. **Persistent kernels** - Eliminate launch overhead for inference
