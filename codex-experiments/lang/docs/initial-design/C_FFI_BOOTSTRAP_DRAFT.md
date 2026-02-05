# Minimal C-FFI Draft (Bootstrap)

This draft defines a minimal C-FFI surface for the bootstrap compiler and runtime. The goal is to enable:

- Linking against libc and a small runtime shim
- Calling C functions from language code
- Defining C-compatible structs for interop

This does not cover calling language code from C, nor GC-managed values across FFI boundaries.

## 1. Surface Syntax

### Extern function declarations

```lang
extern fn printf(fmt: RawPointer<I8>, ...) -> I32;
extern fn malloc(size: I64) -> RawPointer<U8>;
extern fn free(ptr: RawPointer<U8>) -> Unit;
```

Rules:

- `extern fn` declares a function with C ABI and no name mangling.
- `...` enables C varargs (only allowed in `extern fn`).
- No bodies for externs.

### Opaque pointers

```lang
extern fn get_buffer() -> RawPointer<U8>;
extern fn consume(ptr: RawPointer<U8>, len: I64) -> Unit;
```

- `RawPointer<T>` is a raw, untraced pointer.
- `RawPointer<T>` is not a GC root and is never traced by the collector.

### C layout structs

```lang
repr(C)
struct Bytes {
  ptr: RawPointer<U8>,
  len: I64,
}
```

Rules:

- `repr(C)` requests C layout and alignment rules.
- Only `repr(C)` structs may be used directly in extern signatures.

## 2. Type Rules (Stage 0/1)

Allowed extern parameter and return types:

- Primitives: `I8/I16/I32/I64`, `U8/U16/U32/U64`, `F32/F64`, `Bool`
- Raw pointers: `RawPointer<T>` where `T` is any type
- `repr(C)` structs composed of allowed field types
- Varargs (only in externs, only C ABI)

Not allowed in extern signatures (for bootstrap):

- GC-managed types (`struct`/`enum` without `repr(C)`)
- `String` or other managed buffers
- Traits, generics, higher-order functions

## 3. ABI Mapping (LLVM / Inkwell)

- `extern fn` uses C calling convention.
- No name mangling, symbol name equals declared name.
- `Bool` is lowered to `i1` (LLVM) and widened to `i8` for C ABI return if needed.
- `RawPointer<T>` lowered as `ptr` (opaque pointer), element type only for IR typing.
- `repr(C)` struct layout uses target ABI layout.

Varargs:

- Only allowed for `extern fn`.
- Lower using LLVM varargs function types.

## 4. GC and FFI Safety (Bootstrap)

- Raw pointers are not GC roots.
- Passing GC-managed values across FFI is forbidden in stage 0/1.
- If required later, introduce `Gc<T>` or `Pinned<T>` with explicit root registration.

## 5. Runtime Shims

Provide a small C ABI runtime library (Rust `extern "C"` or C):

Required functions (examples):

```c
// allocation / gc hooks
void* gc_allocate(Thread* thread, void* meta, long size);
void* gc_allocate_array(Thread* thread, void* meta, long length);
void  gc_pollcheck_slow(Thread* thread, void* frame_origin);

// stdlib
void  print_int(long v);
```

These are linked into AOT output and resolved by the system linker.

## 6. Compiler Checks

The type checker should enforce:

- Extern signatures only use allowed types.
- `repr(C)` required for structs used in externs.
- Varargs only in `extern fn`.

## 7. Future Extensions (Post-bootstrap)

- `extern "C"` block syntax for grouping externs
- `extern "system"` or other ABIs
- Explicit `Gc<T>` / pinning for safe managed interop
- Strings as `{ ptr, len }` with lifecycle annotations
- Callback interop (function pointers)

## 8. Minimal Examples

```lang
repr(C)
struct Bytes { ptr: RawPointer<U8>, len: I64 }

extern fn malloc(size: I64) -> RawPointer<U8>;
extern fn free(ptr: RawPointer<U8>) -> Unit;
extern fn memcpy(dst: RawPointer<U8>, src: RawPointer<U8>, len: I64) -> RawPointer<U8>;

fn make_bytes(n: I64) -> Bytes {
  let p = malloc(n);
  Bytes { ptr: p, len: n }
}
```
