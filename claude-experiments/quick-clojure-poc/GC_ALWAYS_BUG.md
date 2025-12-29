# Root Cause Analysis: gc_always Test Failures

## The Problem

The `gc_always` tests segfault with `EXC_BAD_ACCESS` when trying to scan stack memory.

## Architecture Background

1. **JIT Stack**: The `Trampoline` allocates a dedicated stack for JIT code execution via `std::alloc::alloc()`. This stack is typically at a heap address like `0xC40000000`. The `stack_base` (top of this region) is stored in `GCRuntime`.

2. **Native Stack**: Rust code runs on the native macOS stack at addresses like `0x16F...`.

3. **Trampoline Wrappers**: When JIT code calls allocating functions (like `make-array`), it goes through wrappers that capture the current SP and pass it to the runtime for GC root scanning.

## The Bug

When **macros are invoked during compilation** (via `invoke_macro` → `apply_fn`), JIT code executes but **from within Rust code**, not from the JIT trampoline. In this context:

1. The JIT code runs on the **native stack** (SP ≈ `0x16F...`)
2. But `stack_base` in GCRuntime points to the **JIT-allocated stack** (`0xC40...`)
3. When JIT code calls an allocating wrapper, it captures SP (native stack)
4. `maybe_gc_before_alloc` compares: `stack_base - stack_pointer`
5. This subtraction **underflows** because native SP can be numerically different from the heap-allocated JIT stack range
6. `StackWalker::get_live_stack()` creates a slice with garbage length
7. Iterating this slice causes the segfault

## Code Path

```
analyze_call_tagged (Rust, native stack)
  → invoke_macro (Rust)
    → apply_fn (Rust)
      → call_jit_variadic_4 (JIT code starts, but SP is still native)
        → trampoline_invoke_multi_arity
          → [JIT function code]
            → make-array wrapper
              → captures SP (native: 0x16F...)
              → trampoline_make_array(stack_pointer=0x16F...)
                → maybe_gc_before_alloc(sp=0x16F...)
                  → gc() with stack_base=0xC40..., stack_pointer=0x16F...
                    → StackWalker::get_live_stack()
                      → stack_base - stack_pointer UNDERFLOWS
                      → SEGFAULT
```

## Key Insight

The fundamental issue is that JIT code can be invoked from two contexts:

1. **Via Trampoline.execute()** - SP is on the JIT stack, GC scanning works correctly
2. **Via invoke_macro/apply_fn** - SP is on the native stack, GC scanning is invalid

Any path that calls JIT code and expects GC to work must ensure the stack context is properly set up for GC scanning.
