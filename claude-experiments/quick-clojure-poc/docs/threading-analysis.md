# Threading Analysis

This document catalogs all multi-threading concerns in the runtime. The runtime is currently **single-threaded by design** — none of these are bugs today, but they are the full list of things that need to change before threads can be introduced.

## Critical: Global Mutable State

### 1. Two `static mut` Runtime Pointers

**builtins.rs:19**
```rust
static mut RUNTIME_PTR: Option<Arc<UnsafeCell<GCRuntime>>> = None;
```

**trampoline.rs:152**
```rust
static mut RUNTIME: Option<Arc<UnsafeCell<GCRuntime>>> = None;
```

Every builtin and trampoline function calls into these to get `&mut GCRuntime`. Two threads calling any builtin simultaneously = data race / UB.

### ~~2. Trampoline Saved Stack Pointer~~ FIXED

Now uses `thread_local!` with `Cell<usize>`. Each thread gets its own saved SP.

## ~~High: Atom Operations Are Not Atomic~~ FIXED

Atom operations now use `AtomicUsize` via `HeapObject::field_as_atomic()`:
- `atom_deref` — `Acquire` load
- `atom_reset` — `Release` store
- `atom_compare_and_set` — real `compare_exchange` with `AcqRel`/`Acquire` ordering

`swap!`, `swap-vals!`, and `reset-vals!` in `core.clj` now use CAS retry loops matching real Clojure semantics.

## High: Var Operations Are Unprotected

**gc_runtime.rs:1274-1310**

- `var_set_value` — plain heap write
- `var_set_meta` — plain heap write
- `var_alter_meta` — read-modify-write sequence, not atomic

## ~~High: Dynamic Bindings Are Not Per-Thread~~ FIXED

Dynamic binding stacks moved from `GCRuntime` field to `thread_local!` storage. Each thread now gets its own `HashMap<usize, Vec<usize>>` of binding stacks, matching real Clojure's per-thread `binding` semantics.

`dynamic_vars` (the set tracking which vars are marked dynamic) stays on `GCRuntime` — it's a global property, not per-thread.

GC relocation code updated to access the thread-local when updating binding keys after compaction.

## High: GC Has No Stop-the-World

- Stack walker (`gc/stack_walker.rs`) does raw pointer traversal with no synchronization
- No mechanism to pause other threads during collection
- The optional `MutexAllocator` (`#[cfg(feature = "thread-safe")]`) is incomplete:
  - `registered_threads: usize` is not atomic
  - Feature is not enabled by default
  - Even when enabled, the outer `static mut` is still unprotected

## Medium: Namespace State

- `namespace_roots: HashMap<String, usize>` — no synchronization
- `next_namespace_id: usize` — plain counter, not atomic
- Concurrent namespace creation would race

## Safe

- `ANON_FN_COUNTER` in `reader.rs:8` — uses `AtomicUsize`
- `GENSYM_COUNTER` in `builtins.rs:760` — uses `AtomicUsize`

## Remaining Priority Order

1. ~~Atoms~~ — DONE
2. ~~Dynamic bindings~~ — DONE
3. ~~Trampoline SP~~ — DONE
4. **Global runtime access** — replace `static mut` with proper synchronization
5. **GC** — add stop-the-world with thread registration
6. **Vars/Namespaces** — add synchronization for mutation operations
