# Runtime ABI v0 (Draft)

This defines the interface between compiled code and the runtime/GC.

## 1. Thread parameter

- All language functions take an implicit `Thread*` as arg0.
- The runtime provides the initial `Thread*` when calling `main`.

## 2. Object header

All GC-managed heap objects begin with a 16-byte header:

```
struct ObjectHeader {
  void* meta;     // TypeInfo
  u32   gc_flags; // mark/forwarding/size bits
  u32   aux;      // array length or extra info
}
```

Payload immediately follows the header at offset 16.

## 3. Type metadata (TypeInfo)

Each type emits a `TypeInfo` structure with:

- kind (struct or array)
- name string
- size and alignment
- number of pointer fields
- array of pointer field offsets

This is sufficient for GC scanning and reflection.

## 4. Stack roots (frame chain)

Each function creates a stack frame:

```
struct Frame {
  Frame* parent;
  FrameOrigin* origin;
  void* roots[N];
}
```

- `roots` contains addresses of GC-managed locals/params.
- `FrameOrigin` contains `num_roots` and function name.

## 5. Runtime entrypoints

The compiled program links against the following runtime symbols:

```c
void* gc_allocate(Thread* thread, void* meta, long size);
void* gc_allocate_array(Thread* thread, void* meta, long length);
void  gc_pollcheck_slow(Thread* thread, void* frame_origin);
void  gc_write_barrier(Thread* thread, void* obj, void* value);
```

Other runtime helpers (v0):

```c
void  print_int(long v);
```

## 6. Safepoints

- Poll checks are inserted after allocations, calls, and loop backedges.
- `gc_pollcheck_slow` walks the frame chain to find roots.

## 7. Write barrier (v0)

- Write barriers are not required for initial object construction.
- Mutating field stores will require barriers in the future.

