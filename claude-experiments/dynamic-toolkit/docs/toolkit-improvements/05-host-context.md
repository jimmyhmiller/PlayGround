# 05 — Host Context for JIT Extern Thunks

## Implementation status: ✅ Implemented (TLS-slot variant)

- **Code**: [`crates/dynlang/src/host.rs`](../../crates/dynlang/src/host.rs) — `install_thread<H>(host)` returning a `HostGuard`, `host<H>()` accessor.
- **Tests**: 4 tests in `host::tests` (install/read, guard restores previous, slot is null after drop, panic-when-uninstalled).
- **Migration**: beagle's three thread_locals (`STRINGS`, `ARRAY_SEQ`, `TIME_ORIGIN`) collapsed into a single `BeagleHost` struct. Each extern thunk that needed host data calls `host::<BeagleHost>()`. The "install before JIT" invariant is now enforced by `HostGuard`'s lifetime.
- **Differs from plan**: went with **option 1 (TLS slot)**, not the hidden first-arg ABI change. Per the doc, the latter is opt-in for embedders that need cross-thread JIT calls — no current need. The `compile_jit_with_host` API was not added; instead, embedders manage the guard themselves alongside `gc.run_jit`.

## Problem

JIT-bound externs are bare `extern "C" fn` — they have no `&self` or
implicit context, so any state they need (interned strings, IC tables,
type metadata, monotonic clock origin) has to live somewhere reachable
from a free function. In beagle this means four `thread_local!`s
([`main.rs:22-39`](../../crates/beagle/src/main.rs#L22)):

```rust
thread_local! {
    static STRINGS:     RefCell<Option<StringPool>>   = const { ... };
    static IC:          RefCell<Option<IcContext>>    = const { ... };
    static ARRAY_INFO:  Cell<Option<ArrayInfo>>       = const { ... };
    static TIME_ORIGIN: Cell<Option<Instant>>         = const { ... };
}
```

Plus the install dance in `real_main` ([`main.rs:263-273`](../../crates/beagle/src/main.rs#L263))
to populate them after lowering, and `RefCell::borrow_mut` plus
`expect("not installed")` in every thunk that touches them.

This is structurally awkward in three ways:

1. **Multi-threaded JITs can't share state.** Beagle's main thread is
   a single dedicated worker, so this doesn't bite — but if the JIT
   ever runs callbacks on another thread (e.g. a future async runtime,
   a finalizer thread, or just a multi-language host on another
   thread), the thread-locals are empty and every extern panics.
2. **No type safety on context shape.** `ARRAY_INFO` is a `Cell<Option<…>>`
   for `Copy` types; `IC` is a `RefCell<Option<…>>` for non-`Copy` ones.
   Embedders have to pick the wrapper that matches their data and
   reinvent the install ceremony each time.
3. **The "install before JIT" ordering is invisible.** `IC` *must* be
   installed before `compile_jit` because the IR embeds a raw pointer
   into `ic.array`. The contract is documented in a comment
   ([`main.rs:266-267`](../../crates/beagle/src/main.rs#L266)) but the
   compiler doesn't help.

## Proposed API

A typed host-context handle threaded through `compile_jit` and reachable
from every extern:

```rust
struct BeagleHost {
    strings: StringPool,
    ic: IcContext,
    array: ArrayInfo,
    time_origin: OnceCell<Instant>,
}

// compile_jit takes ownership of the host. The runtime stores a stable
// pointer that extern thunks can fetch.
let jit = gc.compile_jit_with_host::<NanBoxConfig, _, _, BeagleHost>(
    &module, host, externs);

// Extern thunks take the host as an extra (hidden) first argument.
// The macro from doc 04 wires this up; thunks just declare the type.
extern "C" fn ext_print(host: &BeagleHost, val: u64) {
    print_value(&host.strings, val, false);
}
```

### Implementation choices

The hidden first arg is the cleanest from a thunk-author's perspective but
imposes ABI work in the codegen layer (every extern call site shifts user
args by one). Two acceptable implementations:

1. **TLS slot owned by the runtime.** `dynexec` reserves one thread-local
   `*const c_void` and writes it before transferring control to JIT code.
   `host()` returns it cast to the embedder's type. Cheap, no ABI work,
   but still TLS — same multi-thread pitfall as today, just abstracted.
   Win is *only* the unification (one TLS instead of N).
2. **Hidden first arg.** Extends the calling convention. `compile_jit_with_host`
   declares each user extern with an extra `Type::Ptr` prefix; call sites
   prepend the host pointer. Multi-thread safe (each thread can have its
   own host pointer in a register or on the call stack via the runtime
   shim). More invasive — touches `dynlower`'s call lowering.

Recommend (1) for first cut, (2) as an opt-in once a use case justifies
the codegen change.

## Implementation plan

1. **`dynlang::HostHandle<T>`.** Wraps a `*const T` with a `T: 'static + Sync`
   bound. Construction privately stores the pointer in a runtime-owned TLS
   slot (or per-call shim, depending on choice above).

2. **`compile_jit_with_host<Cfg, Backend, RA, H>(module, host, externs)`.**
   Takes ownership of `host: H`, leaks it to a stable address (or stores
   in `Pin<Box<H>>` inside the returned `Jit` so it lives as long as the
   compiled code), installs the TLS slot, returns the `Jit`.

3. **`dynlang::host<H>() -> &'static H`.** Inside an extern thunk, fetches
   the typed host. Panics with a clear message if no host is installed
   (i.e. the thunk was called outside the JIT).

4. **Beagle migration.** Replace four `thread_local!` blocks + the
   `real_main` install ceremony with one `BeagleHost` struct + one
   `compile_jit_with_host` call. Each extern thunk gets a single
   `let host = dynlang::host::<BeagleHost>();` line. Estimated delta:
   −40 LOC and the "install before compile" invariant becomes
   structural (pass-by-move).

## Open questions / risks

- **`OnceCell` for lazy state.** `time_origin` is set on first use. The
  host struct can use `OnceCell<Instant>` directly — the host is
  immutable from the embedder's perspective once installed, so interior
  mutability is the right pattern.
- **Host vs. module lifetime.** If the JIT outlives the host (because
  `compile_jit_with_host` borrowed instead of owned), the TLS pointer
  dangles. Owning the host inside `Jit` is the safe default; offer a
  `compile_jit_with_host_borrowed(&'static H, ...)` for embedders that
  manage lifetime explicitly.
- **Multiple Jits in one process.** If two `Jit`s are alive concurrently
  (e.g. running tests in parallel), the TLS slot races. The runtime
  should refuse to install over an existing host — return a
  `HostInstalledError` from `compile_jit_with_host`. Or, scope by
  thread + a counter to allow nesting.
