# TensorOps Proper Dialect Build Guide

## Overview
This project implements a proper MLIR TensorOps dialect using C++ with TableGen definitions, exposed to Rust through FFI bindings.

## Prerequisites

### Required Dependencies
1. **LLVM/MLIR 19.x** - The melior crate requires LLVM version 19.x.x
2. **CMake 3.20+** - For building the C++ dialect
3. **Rust toolchain** - Latest stable Rust
4. **C++ compiler** - Supporting C++17 standard

### Installing LLVM/MLIR on macOS (Homebrew)
```bash
# Install LLVM 19 specifically
brew install llvm@19

# Set environment variables
export PATH="/opt/homebrew/Cellar/llvm@19/19.1.7/bin:$PATH"
export MLIR_DIR="/opt/homebrew/Cellar/llvm@19/19.1.7/lib/cmake/mlir"
```

### Installing LLVM/MLIR on Linux
```bash
# Ubuntu/Debian
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 19

# Set environment variables
export MLIR_DIR="/usr/lib/llvm-19/lib/cmake/mlir"
```

## Project Structure
```
melior-test/
├── CMakeLists.txt              # Root CMake configuration
├── build.rs                    # Rust build script for C++ integration
├── cpp/                        # C++ dialect implementation
│   ├── include/TensorOps/      # C++ headers
│   ├── lib/                    # C++ implementation
│   └── td/                     # TableGen definitions
├── capi/                       # C API bindings
├── rust/                       # Rust wrapper code
│   ├── src/
│   │   ├── tensor_ops_ffi.rs   # FFI bindings
│   │   └── tensor_ops_proper.rs # Rust wrapper
│   └── Cargo.toml
└── TENSOROPS_IMPLEMENTATION_LOG.md
```

## Building

### Step 1: Build C++ Dialect (Manual)
```bash
# Create build directory
mkdir build
cd build

# Configure with CMake
cmake .. -DMLIR_DIR=$MLIR_DIR -DCMAKE_BUILD_TYPE=Release

# Build the dialect library
cmake --build . --config Release

# The shared library will be in build/libtensor_ops.so (Linux) or .dylib (macOS)
```

### Step 2: Build Rust Code
```bash
# Set environment variables
export MLIR_DIR="/path/to/mlir/cmake"
export PATH="/path/to/llvm19/bin:$PATH"

# Build Rust code (will automatically build C++ via build.rs)
cd rust/
cargo build

# Or run directly
cargo run --bin melior-test
```

### Step 3: Run Tests
```bash
# Run dialect tests
cargo test --test tensor_dialect_tests

# Run all tests
cargo test
```

## Build Process Details

### Automatic C++ Build via build.rs
The `build.rs` script automatically:
1. Detects MLIR installation via environment variables or common paths
2. Configures CMake with detected MLIR path
3. Builds the C++ dialect library
4. Links the library to Rust

### Manual C++ Build
For development or debugging, you can build the C++ component manually:
```bash
# Generate TableGen files
cmake --build build --target TensorOpsDialectIncGen
cmake --build build --target TensorOpsOpsIncGen

# Build specific targets
cmake --build build --target TensorOpsDialect
cmake --build build --target TensorOpsAPI
```

## Environment Variables

### Required
- `MLIR_DIR` - Path to MLIR CMake configuration directory
- `PATH` - Must include LLVM 19 tools (llvm-config, mlir-tblgen)

### Optional
- `CMAKE_BUILD_TYPE` - Build type (Debug/Release, default: Release)
- `LLVM_DIR` - LLVM CMake directory (usually auto-detected)

## Troubleshooting

### MLIR Not Found
```
Error: failed to find correct version (19.x.x) of llvm-config
```
**Solution**: Ensure LLVM 19 is installed and PATH includes the correct llvm-config:
```bash
which llvm-config
llvm-config --version  # Should show 19.x.x
```

### CMake Configuration Failed
```
Error: Could not find a package configuration file provided by "MLIR"
```
**Solution**: Set MLIR_DIR environment variable:
```bash
export MLIR_DIR="/path/to/llvm19/lib/cmake/mlir"
```

### Link Errors
```
Error: cannot find -ltensor_ops
```
**Solution**: Ensure C++ build completed successfully and check library output directory.

## Usage Example

### Using the Proper Dialect
```rust
use melior::Context;
use melior_test::ProperTensorOpsDialect;

let context = Context::new();
// Register the proper dialect
ProperTensorOpsDialect::register(&registry);

// Create operations using the C++ implementation
let add_op = ProperTensorOpsDialect::create_add_op(
    &context, location, lhs, rhs, result_type
)?;
```

### Benefits Over Unregistered Operations
- ✅ Proper operation verification
- ✅ Type inference and checking  
- ✅ Integration with MLIR passes
- ✅ Round-trip parsing/printing
- ✅ No crashes with module validation
- ✅ Standard MLIR dialect behavior

## Development Notes

### TableGen Regeneration
When modifying .td files, regenerate with:
```bash
cmake --build build --target TensorOpsDialectIncGen TensorOpsOpsIncGen
```

### Adding New Operations
1. Add operation definition to `cpp/td/TensorOpsOps.td`
2. Implement operation class in `cpp/lib/TensorOpsOps.cpp`
3. Add C API function to `capi/TensorOpsAPI.h` and `.cpp`
4. Add FFI binding to `rust/src/tensor_ops_ffi.rs`
5. Add Rust wrapper to `rust/src/tensor_ops_proper.rs`

### Testing Changes
```bash
# Quick compilation check
cargo check --manifest-path rust/Cargo.toml

# Full build and test
cargo test --manifest-path rust/Cargo.toml
```