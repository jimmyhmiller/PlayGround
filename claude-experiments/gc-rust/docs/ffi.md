# FFI — calling C from gc-rust

gc-rust runs on a **precise, relocating, generational GC**. That single fact
dictates the entire FFI design: the collector can only run at *safepoints*
(points where every live pointer's location is known via the frame walker), and
any pointer it cannot find-and-update must not be allowed to move. Native code
has neither stack maps nor any obligation to the collector, so the boundary
between gc-rust and C is where those invariants are most easily broken.

This document records how comparable safe, GC'd languages solve the problem,
the design we chose for gc-rust, and the phased plan for building it. Phases A–D
are implemented today: scalar `extern "C"` calls; the safe-by-default native
transition and `#[repr(C)]` value-struct passing by pointer; passing
`String`/`Array` bytes to C by copy-to-stack (with copy-out for `mut` buffers);
and **callbacks from C into gc-rust** via synthesized C-ABI trampolines.

## What other GC'd languages do

The dominating decision in every design below is *how native code is prevented
from holding a pointer the collector will move or free.*

| Language | GC moves? | Boundary mechanism | What may cross | Safe vs unsafe |
|---|---|---|---|---|
| **OCaml** | minor heap copies | `caml_c_call` saves runtime state; `[@@noalloc]` skips it | `value` (tagged); C must register roots via `CAMLparam`/`CAMLlocal` so the GC rewrites C's variables after a move; field writes go through `Store_field`/`caml_modify` (the write barrier) | `[@@noalloc]` = promise not to allocate/GC/call back — **unchecked** |
| **Go** | heap non-moving, **stacks move** | `entersyscall` + switch to the fixed g0 stack (~100× a Go call) | args implicitly pinned for the call; C may **not** retain a Go pointer past the call unless pinned with `runtime.Pinner` | `GODEBUG=cgocheck` dynamic checks |
| **.NET** | compacting | runtime/source-generated marshalling stub | **blittable** POD types cross by copy (pinned, not converted); everything else marshals; real addresses require **pinning** (`GCHandle`, `fixed`) | blittable vs non-blittable |
| **JNI** | moving | thread-state transitions | **opaque handles** (`jobject`) resolved through an indirection table — the GC moves the object freely and updates the slot | local vs global refs; "critical" regions suspend GC |
| **GHC** | moving generational | `safe` re-enters the managed world; `unsafe` forbids it | never pass a raw heap ref; use pinned `ByteArray#` or copy | `safe` (may GC/callback) vs `unsafe` (GC cannot fire during the call) |
| **Rust** | no GC | `extern "C"` + `#[repr(C)]` | addresses are stable for an object's lifetime, so a pointer just works | every foreign call is `unsafe` |

### Cross-cutting lessons

1. **Model the call as a thread-state transition.** Managed→native on call (so
   a stop-the-world can skip the thread — "in native" counts as a safepoint),
   native→managed on return (re-acquire the safepoint, *block* if a collection
   is in progress). This is Go's `entersyscall`, JNI's transitions, GHC's
   `safe`. Offer a fast tier (OCaml `[@@noalloc]`, GHC `unsafe`) that skips the
   transition but is forbidden from allocating, triggering GC, or calling back.

2. **Never hand opaque native code a raw, relocatable pointer.** The four
   standard tools, each with a clear trade-off:
   - **Pinning** (.NET `fixed`, GHC pinned arrays): real address, but fights a
     compactor → fragmentation. Keep pins few and short; segregate them.
   - **Handle table / indirection** (JNI): GC moves freely, updates the slot;
     cost is an indirection + scope discipline. Best default for object refs.
   - **Copy at the boundary** (JNI `GetArrayRegion`, .NET non-blittable): safest,
     GC stays free, but O(n) and needs copy-back for mutation.
   - **Scalars / POD only** (Rust `#[repr(C)]`): cheapest, eliminates the whole
     bug class — at the cost of pushing structured data onto marshalling.

3. **Callbacks from C back into managed code are the hard direction.** They
   re-enter the allocator, so they must re-establish thread state + a fresh root
   scope, reentrantly. Only the safe tier may host callbacks.

4. **Generational correctness needs a write barrier at the boundary.** If native
   code stores a managed pointer into a managed object, it must go through a
   barrier-aware setter (OCaml's `Store_field`) or the next minor GC will
   move/reclaim the young object out from under the field. Simplest safe rule:
   forbid native code from writing managed fields at all.

5. **Don't make safety unchecked hand-discipline.** OCaml's central failure mode
   is that `CAMLparam`/`Store_field`/`[@@noalloc]` are correct-by-convention and
   unverified — violations are intermittent crashes that pass tests for months.
   gc-rust *owns its compiler*, so it should **generate** boundary glue from
   typed declarations and **reject** unsafe declarations at compile time.

## gc-rust's design

### Surface syntax

```rust
// Tier 1 (Phase A): scalars only — safe by construction, zero GC interaction.
extern "C" fn cos(x: f64) -> f64;
extern "C" fn abs(x: i32) -> i32;

// Tier 1+ (Phase B): #[repr(C)] value structs of scalars cross by copy.
#[repr(C)]
value struct Timespec { sec: i64, nsec: i64 }
extern "C" fn clock_gettime(clk: i32, ts: mut Timespec) -> i32;

// Fast tier (Phase B): promises not to allocate / GC / call back.
extern "C" #[noalloc] fn strlen(s: RawPtr) -> i64;
```

An `extern "C"` item is a **declaration only** — no body. It names a C symbol
resolved at link time (AOT) or via the host process / `dlsym` (JIT).

### The rules (compiler-enforced, not documented-and-hoped)

1. **Blittable-only across the boundary.** Phase A permits only scalar types
   (`i8`..`i64`, `u8`..`u64`, `f32`, `f64`, `bool`, `char`) as parameters and
   return. Phase B adds `#[repr(C)]` value structs whose fields are transitively
   blittable. Passing a `String`, `Vec<T>`, or any heap (`Ref`) type to an
   `extern` function is a **compile error** — a managed pointer never crosses
   raw.

2. **No leading `Thread*`.** Ordinary gc-rust functions take a hidden leading
   `Thread*` (the GC ABI). Extern functions are plain C — codegen declares them
   with exactly their written signature and the call site omits the `Thread*`.

3. **Safe-by-default boundary transition (Phase B).** Every `extern` call is
   wrapped in a managed→native transition: `ai_ffi_enter` publishes this thread's
   frame chain (so a concurrent GC can find its roots and treat it as parked at a
   safepoint), the C call runs, then `ai_ffi_leave` clears it. Correctness does
   **not** depend on the C function promising not to allocate — a GC may safely
   run during the call. This deliberately avoids OCaml/GHC's central failure mode
   (`[@@noalloc]`/`unsafe` are *unchecked* promises whose violation corrupts
   memory). A future `#[noalloc]`-style annotation may *skip* the transition as a
   pure performance hint — a wrong hint costs a missed transition, never memory
   safety. (In a future multi-threaded runtime, `ai_ffi_leave` is also where a
   thread returning from native code blocks on an in-progress collection; today,
   with a single mutator, the clear suffices.)

4. **Value structs cross by pointer, and only from a stack local.** A `#[repr(C)]`
   value struct of transitively-blittable fields may cross — passed as a pointer
   to its storage. The compiler only allows this for a value-struct *local* (a
   native-stack alloca that the collector never moves) or a boundary copy of one;
   a pointer *into* a GC heap object would dangle on a move and is rejected. A
   `mut` value-struct parameter is an out-pointer C may write through (the write
   lands in the caller's local); a non-`mut` one is read-only. This is exactly
   .NET's "value types on the stack need no pinning" rule.

5. **No leading `Thread*`.** Ordinary gc-rust functions take a hidden leading
   `Thread*` (the GC ABI). Extern functions are plain C — codegen declares them
   with exactly their written signature and the call site omits the `Thread*`.

6. **No managed field writes from C, ever** (keeps the generational write
   barrier sound without trusting native code).

### Why Phase A is safe with no machinery

Scalars are not GC pointers. Nothing the collector tracks crosses the boundary,
nothing native code holds can move or be freed, and a libm/syscall-style call
that only takes and returns scalars cannot trigger a gc-rust allocation or call
back. So Phase A needs no pinning, no roots, no barrier, and no transition — it
is correct by the blittable-only rule alone. This is deliberately the same
"easy mode" Rust gets for free, carved out as a safe subset of a moving-GC FFI.

### Implementation (Phase A)

- **Lexer/parser**: `extern` keyword; `extern "C" fn name(params) -> ret;` parses
  to an `FnDef` with `is_extern: true` and an empty body. The ABI string must be
  `"C"`.
- **Resolve/lower**: extern fns enter the symbol table like any function. Lowering
  enforces blittable-only on params + return, lowers each to a `CoreFn` with
  `is_extern: true` and an unmangled `name` (the C symbol).
- **Codegen**: `declare_fn` emits an extern with no `Thread*` param and the C
  name; `define_fn` skips extern fns (no body). `gen_call` omits the `Thread*`
  for extern callees.
- **JIT symbol resolution**: unmapped externs resolve against the host process
  symbol table — libm/libc are linked into the compiler binary, so `cos`,
  `sqrt`, `strlen`, etc. resolve automatically. (A future `dlopen` path would
  let user code bind arbitrary shared libraries.)
- **AOT**: the system linker resolves the symbols against libc/libm as usual.

### Implementation (Phase B)

- **Runtime**: `ai_ffi_enter`/`ai_ffi_leave` (in `gcrust-rt/src/runtime.rs`)
  publish/clear the parked frame pointer around the call — the same mechanism the
  allocator uses to expose roots to a collection.
- **Codegen**: `gen_call` wraps every extern call with `ai_ffi_enter` … call …
  `ai_ffi_leave`. A value-struct argument is passed by pointer: if it is a local,
  the address of the local's own alloca (so C's writes land in the caller's
  variable); otherwise a fresh spilled alloca. `declare_fn` emits `ptr` for a
  value-struct extern parameter.
- **Lowering**: `repr_is_blittable` recurses through value-struct fields; a value
  struct of transitively-blittable fields is accepted, a value *enum* or any heap
  `Ref` is rejected with a clear error.
- **Verified**: `examples/ffi_struct.gcr` calls `gettimeofday` to fill a
  `TimeVal { tv_sec, tv_usec }` through a pointer (JIT + AOT + `--gc-stress`).

### Implementation (Phase C — heap bytes by copy-to-stack)

Passing a `String`/array's *contents* is the hard case: those bytes live on the
moving heap. We chose **copy-at-boundary** over pinning (a copying semi-space
collector has no "don't move this" support, so a pinned region would be a
substantial GC change; copying is safe by construction and needs none).

- **`RawPtr` type** (`Prim::RawPtr` / `ScalarRepr::Ptr`, lowers to LLVM `ptr`):
  an opaque, non-GC, pointer-sized scalar. Only meaningful at the boundary — an
  `extern` may name it; the GC never traces it.
- **`as_c_bytes(s)` intrinsic**: copies `s`'s UTF-8 bytes + a NUL into a
  dynamically-sized **stack** alloca and yields a `RawPtr` to it. Codegen emits
  `len = ai_str_len` → `alloca i8, len+1` → `ai_str_copy_to_buf`. The stack never
  moves, so the pointer is stable for the call with no pinning.
- **Scope enforcement**: `as_c_bytes` is legal *only* as a direct argument to an
  extern call (the pointer is valid only for that one call). Used anywhere else,
  or against a non-`RawPtr` parameter, it's a compile error. There is **no other
  way** to obtain a `RawPtr` to heap memory — passing a raw `String` to a
  `RawPtr` parameter is a type error — so raw heap pointers can never reach C.
- **Verified**: `examples/ffi_bytes.gcr` calls `strlen`/`atoi` on copied String
  bytes (JIT + AOT + `--gc-stress`, the last proving correctness under
  relocation of the source String).

### Implementation (Phase C — arrays + copy-out)

`as_c_bytes` also accepts a scalar `Array<T>` (same `[header][count][data…]`
layout as `String`; byte length = `count` for a `Bytes` array, `count * stride`
for a `Values` array). An array of non-scalars (`Array<String>` etc.) is rejected
— a GC pointer must never cross.

- **Copy-out**: when the extern parameter is `mut`, the stack buffer is also
  written **back** into the heap array after the call, so writes the C function
  made are visible to the caller (the `read(fd, buf, n)` / `memset` pattern). The
  `AsCBytes` node carries `copy_out`; codegen queues `(heap obj, stack buf, byte
  len)` into `FnCtx::pending_copy_outs` and `gen_call` drains them via
  `ai_buf_copy_out` right after the call. Runtime: `ai_buf_copy_in` /
  `ai_buf_copy_out` are the generic (non-NUL) heap↔stack copies.
- **Verified**: `examples/ffi_buffer.gcr` uses `memset` (copy-out, C fills the
  array in place) and `memcmp` (read-only) — JIT + AOT + `--gc-stress`.

### Implementation (Phase D — callbacks from C)

A C function can call **back** into gc-rust. The hard part: a gc-rust function
takes a hidden leading `Thread*` (the GC ABI), but C invokes a callback with no
such argument, and the calling thread is currently "in native". The solution is
a synthesized **C-ABI trampoline** per callback function:

- **Surface**: an extern parameter may have type `extern fn(A, B) -> R` (a C
  callback type, distinct from a gc-rust closure `fn`). Passing a named, non-
  generic gc-rust function there checks the signature and synthesizes a
  trampoline; its address crosses as a `RawPtr`.
- **Trampoline** (`__cb_<fn>`, codegen `callback_trampoline`): C-ABI signature
  (callback params, no `Thread*`). It recovers the ambient thread via
  `ai_current_thread()` (a process-global set at entry), calls `ai_ffi_reenter`
  (re-acquire managed state — clear the parked fp), invokes the real function
  with `(thread, args…)`, calls `ai_ffi_exit` (re-publish the frame, we return to
  C), and returns the value. This is the inverse of the outer call's
  enter/leave; both run because callbacks nest inside an `extern` call.
- **`ptr_read_i64(p)`**: reads an `i64` through a `RawPtr`, so a comparator can
  dereference the pointers C hands it.
- **Verified**: `examples/ffi_callback.gcr` passes a gc-rust comparator to libc
  `qsort` (with `mut base` for copy-out of the sorted buffer) — JIT + AOT +
  `--gc-stress`, the last proving callbacks are correct under relocation.

## Phased rollout

- **Phase A — DONE.** Scalar `extern "C" fn`. Unlocks libm + scalar syscalls. No
  GC-safety hazard because nothing managed crosses.
- **Phase B — DONE.** Safe-by-default managed→native transition wrapped around
  every extern call + `#[repr(C)]` value-struct passing by pointer (stack-local
  only; `mut` = out-pointer). Unlocks struct-filling C APIs (`gettimeofday`, etc.).
- **Phase B follow-ups (not yet built).** A `#[noalloc]` annotation that *skips*
  the transition as a perf hint (never a correctness requirement); by-value small
  struct passing (System V register decomposition); pointer-typed FFI args so a
  null/real pointer can be named directly rather than passed as `i64`.
- **Phase C — DONE.** Passing a `String`'s bytes to C via `as_c_bytes` (copy to a
  stack buffer; `RawPtr` type). Unlocks C string APIs (`strlen`, `atoi`, …).
  Read-only and scoped to one call. (Pinning + a segregated non-moving region —
  for zero-copy or C-writes-in-place over large buffers — remains a possible
  future refinement, but copy-to-stack covers the common case without touching
  the collector.)
- **Phase C arrays + copy-out — DONE.** `as_c_bytes` over scalar `Array<T>`, and
  copy-*out* for a `mut` array parameter (C fills it; written back after the
  call). Unlocks `read`/`recv`/`memset`-style buffer-filling APIs.
- **Phase C follow-ups (not yet built).** Passing `Vec<T>` contents (currently
  only the built-in `Array`); a multi-call scoped form (`with_c_bytes`) when one
  pointer must serve several calls; sub-range / offset+length views.
- **Phase D — DONE.** Callbacks from C into gc-rust via synthesized C-ABI
  trampolines (`ai_current_thread` + `ai_ffi_reenter`/`ai_ffi_exit`). Supports
  named, non-generic functions with scalar/`RawPtr` signatures. Unlocks `qsort`,
  `bsearch`, signal handlers, and GLib-style callback APIs.
- **Phase D follow-ups (not yet built).** Passing closures (not just named
  functions) as callbacks — needs the env pointer threaded via a userdata arg;
  when the runtime gains multiple mutator threads, `ai_ffi_leave` must block on an
  in-progress collection (the structure is already in place).

### A note on multi-threading

gc-rust is currently single-threaded at the language level (no way to spawn a
mutator thread). The transition machinery is built for the multi-threaded future
— `ai_ffi_enter` publishes roots exactly so a *concurrent* collector could run
while this thread is in native code — but the only piece still owed when threads
land is the block-on-collection in `ai_ffi_leave`. The stack-local-only rule for
struct pointers already holds under concurrent GC: a stack local is never on the
moving heap, so no thread's collector can relocate it.
