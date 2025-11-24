# Detailed Build Instructions

## Prerequisites

### 1. LLVM/MLIR Installation

You need LLVM and MLIR libraries installed. You have several options:

#### Option A: Build from source (Recommended for development)

```bash
# Clone LLVM project
git clone https://github.com/llvm/llvm-project.git
cd llvm-project

# Create build directory
mkdir build
cd build

# Configure - enable MLIR and required dialects
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++

# Build (this will take a while)
ninja

# Optionally install
ninja install
```

#### Option B: Use package manager

On macOS with Homebrew:
```bash
brew install llvm
```

On Ubuntu/Debian:
```bash
sudo apt-get install llvm-dev mlir-dev
```

Note: Package manager versions may be older. MLIR is rapidly evolving.

### 2. CMake

```bash
# macOS
brew install cmake

# Ubuntu/Debian
sudo apt-get install cmake
```

Minimum version: 3.20

### 3. Compiler

You need a C++17-compatible compiler:
- Clang 10+
- GCC 9+
- MSVC 2019+

## Building This Project

### Step 1: Configure and Build

The build system will automatically find MLIR in common locations.

```bash
# Navigate to project directory
cd mlir-introspection-c-api

# Create build directory
mkdir build
cd build

# Configure - MLIR is found automatically
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . -j$(nproc)
```

The CMake configuration automatically searches these locations:
- `/usr/local/lib/cmake/mlir` (system install)
- `/opt/homebrew/opt/llvm/lib/cmake/mlir` (Homebrew Apple Silicon)
- `/usr/local/opt/llvm/lib/cmake/mlir` (Homebrew Intel)
- `~/llvm-project/build/lib/cmake/mlir` (local source build)

If your MLIR is elsewhere, you can still specify:
```bash
cmake .. -DMLIR_DIR=/custom/path/to/mlir/cmake
```

### Step 2: Run Example

```bash
./introspection-example
```

Expected output:
```
=== MLIR Dialect Introspection Example ===

Loading dialects...
Dialects loaded.

=== Enumerating all loaded dialects ===

Dialect: arith
Operations:
  - arith.addi
  - arith.subi
  - arith.muli
  ...

=== Testing operation lookup ===
Does 'arith.addi' belong to arith dialect? YES
...
```

## Common Issues

### Issue: "Could not find MLIR"

**Solution**: Make sure MLIR_DIR points to the CMake config:
```bash
# Check if the path exists
ls $MLIR_DIR/MLIRConfig.cmake
```

### Issue: "undefined reference to mlir::..."

**Solution**: Make sure you're linking against all required MLIR libraries. Check CMakeLists.txt.

### Issue: ABI compatibility errors

**Solution**: Rebuild with the same compiler used to build LLVM/MLIR:
```bash
cmake .. -DCMAKE_CXX_COMPILER=$(which clang++)
```

### Issue: "No such file or directory: mlir/IR/Dialect.h"

**Solution**: Include directories may not be set correctly. Try:
```bash
cmake .. -DMLIR_DIR=$MLIR_DIR -DLLVM_DIR=$LLVM_DIR
```

## Debugging Build Issues

Enable verbose output:
```bash
cmake --build . --verbose
```

Check what CMake found:
```bash
cmake .. -DMLIR_DIR=$MLIR_DIR --debug-output
```

## Cross-Compilation

To cross-compile, set appropriate toolchain:
```bash
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/toolchain.cmake \
  -DMLIR_DIR=$MLIR_DIR
```

## Installation

To install the library system-wide:
```bash
cmake --build . --target install
```

Default install location is `/usr/local`. To change:
```bash
cmake .. -DCMAKE_INSTALL_PREFIX=/custom/path
```

## Development Builds

For development with debug info and address sanitizer:
```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer" \
  -DMLIR_DIR=$MLIR_DIR
```

## Integration into Your Project

### Option 1: Install and find_package

```cmake
find_package(MLIRIntrospection REQUIRED)
target_link_libraries(your_target PRIVATE mlir-introspection)
```

### Option 2: Add as subdirectory

```cmake
add_subdirectory(mlir-introspection-c-api)
target_link_libraries(your_target PRIVATE mlir-introspection)
```

### Option 3: Use directly

```cmake
target_include_directories(your_target PRIVATE
    /path/to/mlir-introspection-c-api/include
)
target_link_libraries(your_target PRIVATE
    /path/to/build/libmlir-introspection.so
)
```
