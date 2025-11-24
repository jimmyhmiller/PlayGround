# Easy Integration Guide

## The Easiest Way: Single Command Install

```bash
# From this directory
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make
sudo make install
```

Then in your project:

```c
#include "mlir-introspection.h"
```

Compile with:
```bash
gcc your_file.c -lmlir-introspection -lMLIRCAPIIR
```

That's it! 🎉

## For Zig Projects

### Step 1: Install the library (above)

### Step 2: Update your build.zig

```zig
exe.addIncludePath(.{ .path = "/usr/local/include" });
exe.addLibraryPath(.{ .path = "/usr/local/lib" });
exe.linkSystemLibrary("mlir-introspection");
exe.linkSystemLibrary("MLIRCAPIIR");
exe.linkLibC();
exe.linkLibCpp();
```

### Step 3: Use in your Zig code

```zig
const c = @cImport({
    @cInclude("mlir-introspection.h");
    @cInclude("mlir-c/IR.h");
});

// Use c.mlirEnumerateDialectOperations(...) etc
```

## For CMake Projects

### Option 1: After installing system-wide

```cmake
find_library(MLIR_INTROSPECTION mlir-introspection REQUIRED)
target_link_libraries(your_target PRIVATE ${MLIR_INTROSPECTION})
```

### Option 2: As a subdirectory

```cmake
add_subdirectory(vendor/mlir-introspection-c-api)
target_link_libraries(your_target PRIVATE mlir-introspection)
```

## For Make Projects

```makefile
CFLAGS += -I/usr/local/include
LDFLAGS += -L/usr/local/lib -lmlir-introspection -lMLIRCAPIIR

your_program: your_program.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)
```

## For Meson Projects

```meson
mlir_introspection = dependency('mlir-introspection', required: true)
executable('your_program',
  'your_program.c',
  dependencies: [mlir_introspection]
)
```

## Without Installing (Vendored)

If you want to vendor this library in your project:

```bash
# Copy to your project
cp -r mlir-introspection-c-api vendor/

# In your CMakeLists.txt
add_subdirectory(vendor/mlir-introspection-c-api)
target_link_libraries(your_target PRIVATE mlir-introspection)
```

## Distribution

If you're distributing your application:

### Static Linking (Recommended for portability)

Build with static libs:
```bash
cmake .. -DBUILD_SHARED_LIBS=OFF
```

Then your binary is self-contained (except for system MLIR).

### Dynamic Linking

Include `libmlir-introspection.so` with your app:
```bash
# Copy the library
cp /usr/local/lib/libmlir-introspection.so dist/lib/

# Set rpath in your binary
gcc ... -Wl,-rpath,'$ORIGIN/lib'
```

## Minimal Example Project Structure

```
my-mlir-project/
├── vendor/
│   └── mlir-introspection-c-api/  (this library)
├── src/
│   └── main.c
├── CMakeLists.txt
└── README.md
```

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.20)
project(my-mlir-project)

find_package(MLIR REQUIRED)

add_subdirectory(vendor/mlir-introspection-c-api)

add_executable(my_app src/main.c)
target_link_libraries(my_app PRIVATE
    mlir-introspection
    MLIRCAPIIR
)
```

**src/main.c:**
```c
#include "mlir-introspection.h"
#include "mlir-c/IR.h"
#include <stdio.h>

int main() {
    MlirContext ctx = mlirContextCreate();
    // ... your code ...
    mlirContextDestroy(ctx);
}
```

Build:
```bash
mkdir build && cd build
cmake ..
make
./my_app
```

## Summary

**Easiest for quick testing:**
```bash
sudo make install
gcc your.c -lmlir-introspection -lMLIRCAPIIR
```

**Best for distribution:**
```cmake
add_subdirectory(vendor/mlir-introspection-c-api)
target_link_libraries(your_target PRIVATE mlir-introspection)
```

**Works with:**
- ✅ C (obviously)
- ✅ Zig
- ✅ Rust (via FFI)
- ✅ Python (via ctypes/cffi)
- ✅ Any language with C FFI
