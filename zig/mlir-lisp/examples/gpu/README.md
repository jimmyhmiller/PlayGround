# MLIR AMD GPU Vector Addition Example

## Overview

This example demonstrates a complete MLIR pipeline for AMD GPU computation using ROCm. It performs vector addition on the GPU: adding two arrays of 5 elements (each containing 1.23) to produce an output array of 5 elements (each containing 2.46).

## What This Example Does

### High-Level Flow

1. **Host Setup**: Allocates three 5-element float arrays on the host
2. **Data Initialization**: Fills two input arrays with the value 1.23
3. **GPU Registration**: Registers host memory with the GPU runtime
4. **Host-to-Device Transfer**: Copies data to GPU memory
5. **GPU Kernel Execution**: Runs vector addition kernel on GPU (5 threads, each adding one pair of elements)
6. **Result Access**: Result is accessible on host and printed

### Memory Model

The example uses MLIR's `memref` abstraction:
- `memref.alloc()` - Allocates host memory
- `gpu.host_register` - Makes host memory accessible to GPU (registers with ROCm)
- `mgpuMemGetDeviceMemRef1dFloat()` - Transfers data to GPU and returns device pointer
- GPU kernel operates on device memory
- Result remains accessible from host due to registration

### GPU Kernel Details

The `@vecadd` function launches a GPU kernel with:
- **Grid dimensions**: 1x1x1 blocks
- **Block dimensions**: 5x1x1 threads (one thread per element)
- Each thread:
  - Loads its element from input arrays using thread ID (`%tx`)
  - Performs floating-point addition
  - Stores result to output array

## Running the Example

### Prerequisites

- AMD GPU with ROCm support
- LLVM/MLIR 20+ built with ROCm support (`-DMLIR_ENABLE_ROCM_RUNNER=ON`)
- ROCm runtime libraries installed

### Command

```bash
mlir-opt vecadd.mlir \
  | mlir-opt -convert-scf-to-cf \
  | mlir-opt -gpu-kernel-outlining \
  | mlir-opt -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-rocdl{use-bare-ptr-memref-call-conv=true}),rocdl-attach-target{chip=gfx1151})' \
  | mlir-opt -gpu-to-llvm=use-bare-pointers-for-kernels=true -reconcile-unrealized-casts -gpu-module-to-binary \
  | mlir-runner \
      --shared-libs=/usr/local/lib/libmlir_rocm_runtime.so \
      --shared-libs=/usr/local/lib/libmlir_runner_utils.so \
      --entry-point-result=void
```

**Note**: Replace `gfx1151` with your GPU's architecture (check with `rocminfo | grep gfx`).

### Pipeline Stages

1. **`mlir-opt vecadd.mlir`** - Parse and validate input
2. **`-convert-scf-to-cf`** - Convert structured control flow (SCF) to control flow (CF) dialect
3. **`-gpu-kernel-outlining`** - Extract GPU kernel into separate module
4. **GPU-specific passes**:
   - `strip-debuginfo` - Remove debug information
   - `convert-gpu-to-rocdl` - Convert GPU dialect to ROCm LLVM dialect (ROCDL)
   - `rocdl-attach-target` - Attach target chip information
5. **`-gpu-to-llvm`** - Convert GPU host code to LLVM
6. **`-reconcile-unrealized-casts`** - Clean up type conversions
7. **`-gpu-module-to-binary`** - Compile GPU module to binary (HSA code object)
8. **`mlir-runner`** - JIT compile and execute

## Runtime System Interaction

### MLIR ROCm Runtime API

The example uses the following runtime functions (provided by `libmlir_rocm_runtime.so`):

```c
// Memory management
void* mgpuMemGetDeviceMemRef1dFloat(void* hostPtr);  // Transfer to device
void mgpuMemHostRegisterMemRef(void* ptr, size_t size); // Register host memory

// Module/kernel management
void* mgpuModuleLoad(void* data);                    // Load compiled kernel
void* mgpuModuleGetFunction(void* module, const char* name);
void mgpuModuleUnload(void* module);

// Execution
void* mgpuStreamCreate();
void mgpuLaunchKernel(void* function, size_t gridX, size_t gridY, size_t gridZ,
                      size_t blockX, size_t blockY, size_t blockZ,
                      size_t sharedMem, void* stream, void** args);
void mgpuStreamSynchronize(void* stream);
void mgpuStreamDestroy(void* stream);
```

### Under the Hood

When you run the pipeline:

1. **Compilation**: The GPU kernel is compiled to an HSA code object (AMD's GPU binary format)
2. **JIT Execution**: `mlir-runner` uses LLVM's JIT to execute the `@main` function
3. **Runtime Calls**: Host code calls the runtime API to:
   - Register memory with ROCm
   - Transfer data to GPU
   - Load the kernel binary
   - Launch the kernel
   - Synchronize execution

## JIT Compilation via C API (for Zig or other languages)

To JIT compile and run this MLIR code programmatically, you would:

### 1. Set Up MLIR Context and Modules

```c
// C API (similar pattern in Zig)
#include "mlir-c/IR.h"
#include "mlir-c/ExecutionEngine.h"
#include "mlir-c/Pass.h"

// Create context and load MLIR
MlirContext ctx = mlirContextCreate();
MlirModule module = mlirModuleCreateParse(ctx, mlirStringRefCreate(source));
```

### 2. Apply Transformation Passes

```c
// Create pass manager and add passes
MlirPassManager pm = mlirPassManagerCreate(ctx);

// Add passes in sequence
mlirPassManagerAddOwnedPass(pm, mlirCreateConversionConvertSCFToCF());
mlirPassManagerAddOwnedPass(pm, mlirCreateGPUGpuKernelOutlining());
// ... add remaining passes

// Run passes
mlirPassManagerRun(pm, module);
```

### 3. Create Execution Engine

```c
// Set up execution engine options
MlirExecutionEngineOptions options = mlirExecutionEngineOptionsCreate();

// Add shared libraries
mlirExecutionEngineOptionsAddSharedLib(options,
    mlirStringRefCreate("/usr/local/lib/libmlir_rocm_runtime.so"));
mlirExecutionEngineOptionsAddSharedLib(options,
    mlirStringRefCreate("/usr/local/lib/libmlir_runner_utils.so"));

// Create engine
MlirExecutionEngine engine = mlirExecutionEngineCreate(module, options);
```

### 4. Lookup and Invoke Functions

```c
// Look up the main function
void* mainFunc = mlirExecutionEngineLookup(engine, "main");

// Invoke (no arguments for this example)
void (*entryPoint)(void) = (void (*)(void))mainFunc;
entryPoint();
```

### 5. Cleanup

```c
mlirExecutionEngineDestroy(engine);
mlirPassManagerDestroy(pm);
mlirModuleDestroy(module);
mlirContextDestroy(ctx);
```

### Key Considerations for Zig

1. **Library Linking**: Link against:
   - `libMLIR.so` (or individual MLIR dialect libraries)
   - `libMLIRExecutionEngine.so`
   - `libmlir_rocm_runtime.so`
   - `libmlir_runner_utils.so`

2. **Pass Registration**: You may need to manually register passes before using them:
   ```c
   mlirRegisterAllPasses();
   mlirRegisterAllDialects(ctx);
   ```

3. **Memory Management**: The GPU runtime expects specific memory layouts. The `memref` descriptor contains:
   ```c
   struct MemRefDescriptor {
       float* allocated;  // Base pointer
       float* aligned;    // Aligned pointer
       intptr_t offset;   // Offset
       intptr_t sizes[rank];    // Dimensions
       intptr_t strides[rank];  // Strides
   };
   ```

4. **Zig FFI Example**:
   ```zig
   const mlir = @cImport({
       @cInclude("mlir-c/IR.h");
       @cInclude("mlir-c/ExecutionEngine.h");
   });

   pub fn jitCompileAndRun(source: []const u8) !void {
       const ctx = mlir.mlirContextCreate();
       defer mlir.mlirContextDestroy(ctx);

       const module_ref = mlir.mlirStringRefCreate(source.ptr, source.len);
       const module = mlir.mlirModuleCreateParse(ctx, module_ref);
       defer mlir.mlirModuleDestroy(module);

       // Apply passes...
       // Create execution engine...
       // Invoke...
   }
   ```

## Advanced: Programmatic GPU Memory Management

If you want more control (e.g., for a custom runtime), you can:

1. **Allocate GPU memory directly**:
   ```c
   hipMalloc(&devicePtr, size);
   hipMemcpy(devicePtr, hostPtr, size, hipMemcpyHostToDevice);
   ```

2. **Pass pointers to kernel**: Instead of using `mgpuMemGetDeviceMemRef1dFloat`, construct `MemRefDescriptor` structures with device pointers

3. **Launch kernels manually**: Use HIP API directly:
   ```c
   hipModuleLoadData(&module, hsaCodeObject);
   hipModuleGetFunction(&kernel, module, "vecadd");
   hipLaunchKernel(kernel, gridDim, blockDim, args, sharedMem, stream);
   ```

This gives you full control but bypasses MLIR's convenience runtime.

## Expected Output

```
Unranked Memref base@ = 0x... rank = 1 offset = 0 sizes = [5] strides = [1] data =
[2.46,  2.46,  2.46,  2.46,  2.46]
```

This confirms that 1.23 + 1.23 = 2.46 was computed correctly on the GPU for all 5 elements.
