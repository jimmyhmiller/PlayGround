# C Hot Reload

Hot reloadable C code using dynamic bundles and atomic indirection.

## Concept

- **Namespace**: Container for definitions
- **Definition**: Atomic pointer wrapper (var-like closure pattern)
- **Bundle**: Dynamically loaded .so with `bundle_init()` function
- **Indirection**: All calls go through atomic var, always getting latest definition

This establishes patterns for hot-reloading functions, structs, and enums that will be useful for a compiled language.

## Architecture

### Core Components

**namespace.{h,c}** - Namespace and definition management
- `Namespace`: Named container holding definitions
- `Definition`: Atomic pointer wrapper for any type (function, struct, enum, var)
- Thread-safe atomic updates via C11 atomics
- Lookup by name for cross-bundle references

**bundle.{h,c}** - Dynamic bundle loading
- Load/reload .so files using `dlopen`
- Each bundle exports `bundle_init(Namespace *ns)` function
- On reload: closes old handle, loads new version, calls init
- Init function registers/updates definitions atomically

**Cross-bundle references** - Bundles can reference each other
- Store namespace reference in bundle
- Look up symbols: `namespace_lookup(ns, "symbol_name")`
- Get pointer: `definition_get(def)`
- Call through indirection - always gets latest version

## Build & Test

```bash
# Build everything
make

# Run comprehensive test suite
make run-tests

# Build just tests
make tests

# Clean build
make clean
```

## Tests

All tests pass with zero warnings:

- **test_namespace** (5 tests) - Namespace operations
  - Create/destroy namespaces
  - Define and lookup definitions
  - Atomic updates
  - Multiple definitions

- **test_bundle** (4 tests) - Bundle loading
  - Load bundles
  - Call functions through indirection
  - Hot reload with new implementations
  - Multiple functions in one bundle

- **test_cross_bundle** (2 tests) - Cross-bundle calls
  - Functions calling functions from other bundles
  - Bundle load order independence

## Interactive Demos

### counter_demo

Stateful hot reload with a counter:

```bash
./counter_demo
```

- Maintains count state
- Applies `change()` function every second
- Press 1-5 to hot reload different change functions:
  - 1: +1 (increment)
  - 2: +10 (add 10)
  - 3: ×2 (double)
  - 4: -5 (decrement by 5)
  - 5: Reset to 0
- State persists across reloads - only the behavior changes!

### greeting_demo

Interactive hot reload demonstration:

```bash
./greeting_demo
```

- Displays greeting every 3 seconds
- Press 1, 2, or 3 to switch greeting
- Automatically rewrites bundle source and hot reloads
- Clear console prevents spam

### hot_reload

Basic polling demo:

```bash
./hot_reload
```

- Calls functions from example_bundle.so
- Checks for reload every 2 seconds
- Modify `example_bundle.c` and `make rebuild-bundle` to see updates

## Hot Reload Pattern

1. **Define functions in bundle:**
```c
int my_function(int a, int b) {
    return a + b;
}
```

2. **Register in bundle_init:**
```c
void bundle_init(Namespace *ns) {
    Definition *def = namespace_lookup(ns, "my_function");
    if (def) {
        definition_update(def, my_function);  // Update existing
    } else {
        namespace_define(ns, "my_function", DEF_FUNCTION, my_function);  // New
    }
}
```

3. **Call through indirection:**
```c
Definition *def = namespace_lookup(ns, "my_function");
MyFnType fn = (MyFnType)definition_get(def);
int result = fn(5, 3);  // Always calls latest version
```

4. **Reload:**
```c
bundle_reload(bundle);  // Atomically swaps in new version
```

## Key Implementation Details

- **-rdynamic flag**: Required for bundles to access namespace symbols from main executable
- **dlclose then dlopen**: Close old handle before loading new to avoid stale function pointers
- **Atomic pointers**: C11 `_Atomic(void*)` for thread-safe updates
- **RTLD_NOW**: Resolve all symbols immediately on load
- **Type safety**: Cast function pointers with typedef for type checking

## Future Directions

- Meta system for A/B testing different implementations
- Struct and enum hot reloading with versioning
- Namespace hierarchies and imports
- Compilation target for custom language
