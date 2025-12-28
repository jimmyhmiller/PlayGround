# Server Build Notes for Lispier GPU Support

## Server Access
```
ssh jimmyhmiller@192.168.0.55
```

## Goal
Build lispier on the server to run GPU code via JIT on the AMD GPU.

## Current State

### LLVM 21 Built from Source
- Source: `~/llvm-project-21` (checked out at tag `llvmorg-21.1.8`)
- Build: `~/llvm-21-build`
- MLIR runtime libraries are built and available

### Lispier Project Location
```
~/Documents/Code/Playground/claude-experiments/lispier
```

### Melior (Rust MLIR bindings)
Currently at `~/Documents/Code/open-source/melior` but it's **not a git repo** (was rsync'd without `.git`).

**First, clone the fork properly:**
```bash
rm -rf ~/Documents/Code/open-source/melior
git clone git@github.com:jimmyhmiller/melior.git ~/Documents/Code/open-source/melior
```

## The Problem

The `mlir-sys` crate generates Rust bindings from C headers using `bindgen`. The bindings are being generated with **incomplete include paths**, causing function name mismatches.

### Root Cause
`llvm-config --includedir` returns:
```
/home/jimmyhmiller/llvm-project-21/llvm/include
```

But MLIR headers are in **three** locations:
1. `/home/jimmyhmiller/llvm-project-21/llvm/include` - LLVM headers
2. `/home/jimmyhmiller/llvm-project-21/mlir/include` - MLIR source headers
3. `/home/jimmyhmiller/llvm-21-build/tools/mlir/include` - MLIR **generated** headers (pass declarations, etc.)

The generated headers (like `Transforms.capi.h.inc`, `Passes.capi.h.inc`) contain the actual pass function declarations. Without them, bindgen generates wrong/incomplete bindings.

## Solution

Add the missing include paths when building. Set this environment variable before `cargo build`:

```bash
export BINDGEN_EXTRA_CLANG_ARGS="-I$HOME/llvm-project-21/mlir/include -I$HOME/llvm-21-build/tools/mlir/include"
```

## Full Build Command

```bash
source ~/.cargo/env
cd ~/Documents/Code/Playground/claude-experiments/lispier

# Clean stale mlir-sys build artifacts
rm -rf target/release/build/mlir-sys-*

# Set all required environment variables
export TABLEGEN_210_PREFIX=~/llvm-21-build
export LLVM_SYS_210_PREFIX=~/llvm-21-build
export MLIR_SYS_210_PREFIX=~/llvm-21-build
export BINDGEN_EXTRA_CLANG_ARGS="-I$HOME/llvm-project-21/mlir/include -I$HOME/llvm-21-build/tools/mlir/include"

# Build
cargo build --release
```

## Verification

After building, check that the bindings have correct function names:
```bash
grep "mlirCreateGPUGpuKernelOutliningPass" target/release/build/mlir-sys-*/out/bindings.rs
```

Should show the function WITH the "Pass" suffix. If it shows `mlirCreateGPUGpuKernelOutlining` (without "Pass"), the include paths are still wrong.

## Testing

Once built, test with:
```bash
./target/release/lispier run examples/gpu_vecadd.lisp
```

## Additional Files

### GPU C Interface Shim
There's a shim library at `~/libgpu_ciface_shim.so` that provides `_mlir_ciface_` wrappers for GPU runtime functions. This was compiled with:
```bash
gcc -shared -fPIC -o ~/libgpu_ciface_shim.so src/gpu_ciface_shim.c \
    -L ~/llvm-21-build/lib -lmlir_rocm_runtime \
    -Wl,-rpath,/home/jimmyhmiller/llvm-21-build/lib
```

The lispier code in `src/main.rs` already references this shim.
