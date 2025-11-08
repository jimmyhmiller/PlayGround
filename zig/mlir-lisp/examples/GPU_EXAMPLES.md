# GPU Programming Examples

This directory contains examples of GPU programming using MLIR's `gpu` dialect in your lisp.

## Fixed Issues

### Tokenizer Fix for `::` in Symbol References

The tokenizer now properly handles `::` in nested symbol references like `@module::@function`. This is essential for GPU kernel references in `gpu.launch_func`.

**Example:**
```lisp
(attributes {:kernel @kernel_module::@square_kernel})
```

This is parsed as a single symbol `@kernel_module::@square_kernel` instead of being split at the `::`.

## Available Examples

### 1. `gpu_square_converted.lisp` - Element-wise Square

A simple GPU kernel that squares each element of a 10x10 matrix.

**Key concepts:**
- `gpu.module` - Container for GPU kernels
- `gpu.func` with `kernel` attribute - GPU device function
- `gpu.block_id` - Get current block coordinates
- `gpu.thread_id` - Get current thread coordinates within block
- `gpu.launch_func` - Launch kernel from host code

**Grid configuration:**
- 10x10 blocks, 10x10 threads per block
- Each thread computes one matrix element

**Run:**
```bash
./zig-out/bin/mlir_lisp examples/gpu_square_converted.lisp
```

### 2. `gpu_matmul.lisp` - Matrix Multiplication

Parallel matrix multiplication for 16x16 matrices.

**Key concepts:**
- 2D block indexing with `gpu.block_id x` and `gpu.block_id y`
- `scf.for` loop within GPU kernel for accumulation
- Shared memory access patterns

**Algorithm:**
- Each block computes one element of the output matrix
- Uses a loop to compute dot product: C[i,j] = Σ(A[i,k] * B[k,j])

**Grid configuration:**
- 16x16 blocks, 1x1 threads per block
- Total 256 blocks, one per output element

**Run:**
```bash
./zig-out/bin/mlir_lisp examples/gpu_matmul.lisp
```

## GPU Dialect Operations

### Kernel Definition

```lisp
(operation
  (name gpu.module)
  (attributes {:sym_name @kernel_module})
  (regions
    (region
      (block
        (operation
          (name gpu.func)
          (attributes
            {:gpu.kernel true
             :sym_name @kernel_name
             :function_type (!function (inputs ...) (results))})
          (regions ...))))))
```

### Thread/Block Indexing

```lisp
;; Get block ID (which block we're in)
(operation
  (name gpu.block_id)
  (result-bindings [%block_x])
  (result-types index)
  (attributes {:dimension x}))  ;; can be x, y, or z

;; Get thread ID (which thread within block)
(operation
  (name gpu.thread_id)
  (result-bindings [%thread_x])
  (result-types index)
  (attributes {:dimension x}))
```

### Launching Kernels

```lisp
(operation
  (name gpu.launch_func)
  (operand-uses %grid_x %grid_y %grid_z %block_x %block_y %block_z %arg1 %arg2 ...)
  (attributes
    {:kernel @module::@kernel_name
     :operandSegmentSizes array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, N, 0>}))
```

Where:
- First 6 operands: grid dimensions (x,y,z) and block dimensions (x,y,z)
- Remaining operands: kernel arguments
- `N` in operandSegmentSizes: number of kernel arguments

### Container Module

All GPU code must be in a module with the `gpu.container_module` attribute:

```lisp
(operation
  (name builtin.module)
  (attributes {:gpu.container_module true})
  (regions ...))
```

## Creating Your Own GPU Kernels

1. Start with MLIR GPU code
2. Use `mlir-to-lisp` to convert to lisp format:
   ```bash
   cd /Users/jimmyhmiller/Documents/Code/PlayGround/zig/mlir-parser
   zig build
   ./zig-out/bin/mlir-to-lisp your_gpu_code.mlir > output.lisp
   ```

3. Fix the output:
   - Remove `(mlir ...)` wrapper
   - Merge duplicate `(attributes ...)` sections
   - Balance parentheses: `paredit-like balance output.lisp --in-place`

4. Test:
   ```bash
   ./zig-out/bin/mlir_lisp output.lisp
   ```

## GPU Programming Patterns

### Pattern 1: 1D Parallel Map

Each thread processes one element:

```lisp
;; global_idx = block_id * block_dim + thread_id
;; output[global_idx] = f(input[global_idx])
```

### Pattern 2: 2D Parallel Map (Images/Matrices)

Each thread processes one pixel/element:

```lisp
;; row = block_id_x
;; col = block_id_y (or thread_id_y)
;; output[row][col] = f(input[row][col])
```

### Pattern 3: Reduction/Accumulation

Each thread computes a partial result:

```lisp
;; Use scf.for within kernel for local accumulation
;; result = 0
;; for k in range:
;;   result += a[idx, k] * b[k, idx]
;; output[idx] = result
```

## Resources

- [MLIR GPU Dialect Docs](https://mlir.llvm.org/docs/Dialects/GPU/)
- [Stephen Diehl's MLIR GPU Tutorial](https://www.stephendiehl.com/posts/mlir_gpu/)
- Your project's grammar: `docs/grammar.md`
