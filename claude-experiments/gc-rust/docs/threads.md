# Threads — OS-thread concurrency

gc-rust supports real OS threads that share one garbage-collected heap. The
collector stops the world across **all** registered mutator threads when it
collects, so allocation and GC are safe from any thread. The surface is
Java-shaped (`Thread::spawn`, `t.join()`, `Thread::sleep`) but functions and
lambdas are interchangeable, and there is no subclassing — you spawn a function.

## Surface

```rust
fn main() -> i64 {
    let t = Thread::spawn(|| fib(30));   // runs on a new OS thread
    let local = fib(28);                 // runs on this thread, concurrently
    local + t.join()                     // join blocks and returns the result
}
```

- `Thread::spawn(f)` — `f` is a `fn() -> T` (a lambda or a named function) for
  **any** result type `T` (a scalar, a `Vec`, a struct, …); runs it on a fresh OS
  thread sharing this heap. Returns a `Thread<T>` handle.
- `t.join()` — block until the thread finishes; return its `T` result. The result
  travels back via the handle's cell (the Future model, like Java's
  `FutureTask.outcome`): the worker stores it on finish, and `join`'s OS thread
  join provides the cross-thread happens-before. Because the parent holds the
  GC-managed handle, a `Vec`/struct result stays traced and relocated until read.
- `Thread::sleep(ms)`, `Thread::yield_now()`, `Thread::current_id()` — scheduler
  operations (Java's static `Thread` methods).

`Thread::spawn` / `Thread::sleep` are **associated functions** (`impl` functions
with no `self`); `t.join()` is a method (`self`). Both forms coexist in an `impl`.

> **M2 scope:** the spawned closure must return `i64` (a scalar or pointer-width
> value). A generic `Thread<T>` over any `Send` result is a planned follow-up.

## How it works (and why it's GC-safe)

The runtime was built multi-thread-ready: the heap holds a `Mutex<Vec<Arc<
ThreadState>>>` registry, stop-the-world coordination via per-thread poll flags +
condvars, an atomic bump allocator, and a collector that scans **every**
registered thread's roots. Allocation is never inlined — every allocation calls
the runtime's atomic `ai_gc_alloc_*`, so two threads can't race a bump pointer.
Threading just wires the language to that.

- **Spawn** (`ai_thread_spawn`): clones the shared heap `Arc`, starts an OS
  thread. The child registers its own mutator `Thread` (so STW can pause it and
  the collector scans its roots), publishes its thread-local current thread,
  runs the closure, captures the result, and deregisters.
- **Blocking calls transition to a parked state.** `join` and `sleep` block
  this thread in libc, where it can't poll its safepoint flag. If they blocked
  naively, a collection triggered by another thread would spin forever waiting
  for this thread to reach a safepoint — a deadlock. So both wrap the blocking
  call in a *blocking region*: publish the thread's frame (so the collector
  scans its live roots — e.g. a `Vec` held across `join`), transition to BLOCKED
  (which counts as already-parked, letting stop-the-world proceed without this
  thread), block, then re-acquire RUNNING (waiting out any in-progress GC) and
  clear the frame — all panic-safe via a drop guard.
- **GC-safe env handoff** — the subtle part. The closure's env is a GC pointer;
  if the parent triggered a collection between spawning and the child reading it,
  it would move. So the parent (synchronously, while the env is still rooted in
  its own frame) registers the env as a **global root** and hands the child its
  index; the child reads the (relocated) env through that root, moves it into its
  own scratch root, and clears the global slot. No raw cross-thread pointer is
  ever read after a possible move.
- **Thread-local current thread**: `ai_current_thread` (used by FFI callback
  trampolines) is thread-local, so each mutator recovers its own `Thread*`.

## Correctness

`examples/threads.gcr` runs three concurrently-allocating threads (each builds
and sums a 5000-element `Vec`) and checks the total. It passes under normal mode
and under `--gc-stress` (collect on **every** allocation), which forces
stop-the-world coordination at every allocation point across all three threads —
the strongest available proof that cross-thread pausing, root scanning, and
relocation are correct. `tests/threads.rs` covers spawn/join, the multi-thread
allocation workload, the same under stress, and the AOT-linked binary.

## Shared-memory concurrency

Threads share the heap, so they can share *mutable* state too — safely, via
synchronized primitives.

### `Atom<T>` — a Clojure-style atom

```rust
let a: Atom<i64> = Atom::new(0);
a.swap(|x| x + 1);          // apply a pure fn, atomically; retries on contention
a.compare_and_set(old, new);// explicit CAS, returns bool
a.reset(42);                // unconditional set
let v = a.deref();          // current value (atomic load)
```

An atom holds a single immutable value; you never mutate in place — you
atomically *swap which value* the atom points at, which fits the
immutable-by-default model exactly. It's an ordinary GC heap object with one
field; `deref` is an atomic load of that field, `compare_and_set` an LLVM
`cmpxchg` + write barrier, and `swap`/`reset` are CAS-retry loops written in the
prelude over those (so the candidate values are ordinary frame roots — GC-safe
without special handling). This works on the relocating collector because
collection is stop-the-world: a thread performing the CAS is running, so no
collection (and no relocation of the atom) is in progress during the operation.
`examples/atom.gcr` runs three threads swap-incrementing a shared atom and checks
the exact total (no lost updates), including under `--gc-stress`.

### `AtomicI64` — a lock-free shared integer

```rust
let c = AtomicI64::new(0);
c.fetch_add(1);             // returns the previous value
c.compare_and_set(old, new);
c.load(); c.store(v);
```

Backed by an off-heap atomic cell (stable address; scalars need no GC tracing).
The headline correctness test: many threads `fetch_add` into one counter and the
exact total proves no updates are lost under contention.

### `Send` / `Sync` — making shared mutation sound

Shared *mutable* state is allowed — but only when the type vouches for its own
thread-safety. This is enforced with two marker properties:

- **`Send`** — safe to move to another thread.
- **`Sync`** — safe to *share* (reference from multiple threads at once).

A spawned closure shares its captures with the parent, so each capture must be
`Sync`. Sync is **auto-derived**:

- scalars, `String`, and **value** (`#[value]`) structs/enums — immutable, so
  `Sync` for free (you can't race what can't change);
- `Atom<T>` and `AtomicI64` — synchronized internally;
- a **reference** struct/enum is `Sync` iff every field is a synchronization
  primitive (`Atom`/`AtomicI64`, or a container of `Sync`) or an immutable value
  type — so a concurrent map built from `Atom`-held buckets is `Sync`
  automatically, with no annotation;
- escape hatch: `impl Sync for T {}` lets an author assert thread-safety the
  deriver can't prove (e.g. a hand-rolled lock-free structure).

A *plain* reference struct (e.g. `Counter { n: i64 }`) is **not** `Sync` — its
field slots are mutable, so sharing it across threads would race. Capturing it
into a `Thread::spawn` is a compile error directing you to `Atom<Counter>`. The
upshot, given immutable-by-default: the vast majority of types are `Sync`
automatically, and you only hit the bound when genuinely sharing mutable state —
at which point the compiler makes you reach for a synchronized primitive.

```rust
struct Shared { a: Atom<i64>, b: Atom<i64> }   // auto-Sync (Atom fields)
let s = Shared { a: Atom::new(0), b: Atom::new(0) };
Thread::spawn(|| worker(s));                    // OK — shared, but synchronized
```

> Note: the rule is type-based and conservative — sharing a plain reference
> struct *read-only* is actually safe but is still rejected (the checker can't see
> it's read-only). Share read-only data as a `#[value]` (immutable) type.

### Channels — bounded, blocking message passing

```rust
let ch: Channel<Vec<i64>> = Channel::new(4);   // capacity-4 bounded channel
Thread::spawn(|| { ch.send(make_msg()); });    // blocks while full
let msg = ch.recv();                            // blocks while empty
ch.close();
```

A channel is the hybrid every mainstream runtime uses (Go's `hchan`, Java's
`ArrayBlockingQueue`): the element **buffer is an ordinary on-heap object**, so
the GC traces and relocates the queued values for free — nothing lives off-heap
except the indices and synchronization. A runtime control block holds a mutex
plus two condition variables (`not_empty`/`not_full`); `send` blocks while full
and `recv` blocks while empty, **parking** on a condvar (never busy-spinning) and
transitioning to the blocked state so a stop-the-world GC can proceed. The buffer
pointer is re-read from the (rooted) channel on each access — never held across a
park — so relocation during a wait can't dangle it. `Channel` is declared `Sync`
(its control block synchronizes access), so it's freely shareable across threads.

## Not yet built (M3+)

- **`Mutex<T>`** (scoped access) — for shared mutable state that isn't a single
  swappable cell or a channel.
- **Channels**: `Sender<T>` / `Receiver<T>` (`send` / `recv`).
- **`Send` / `Sync` marker traits** + `spawn`/channel bounds. Immutable values
  (the default) are `Send + Sync` automatically — you can't race what can't
  change — so these mainly constrain the genuinely mutable/non-shareable handles.
- **Generic `Thread<T>`** (any result type, not just `i64`).
- **FFI**: `ai_ffi_leave` should block on an in-progress collection now that
  multiple mutators exist (the structure is in place).
