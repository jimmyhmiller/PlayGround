# Hop (working name)

A small statically typed language for writing programs that ship code over the network. The same source runs in a single process or across a cluster of nodes without code changes. Syntax is borrowed from Rust; semantics are not.

## Goals

- **Code is content-addressed.** Every definition is identified by the hash of its expression, not its name. Two nodes that have computed the same hash for `foo` provably have the same `foo`.
- **Closures are shippable by default.** A function value is a `(code-hash, captured-environment)` pair. Sending one to another node means sending the captured values and letting the receiver resolve the code-hash against its local codebase, pulling what it doesn't have.
- **Single-node and multi-node use the same code.** A program written against a cluster of one node and a cluster of one thousand looks identical at the source level.
- **Failure is a value, not an exception.** Anything that crosses the network returns a `Result`. The boundary is the type.
- **The language stays small.** No effect system, no lifetimes, no ownership, no macros, no async coloring. Garbage collected. The type system has exactly one job beyond conventional checking: prove that values flowing across `at` are `Wire`.

## Non-goals

- A complete distributed-systems runtime. No supervision trees, no transactions, no consensus. Those are libraries on top.
- Replacing general-purpose languages. Hop is meant for programs whose central concern is distribution.
- Hiding distribution. Latency, partial failure, and partitions are visible in types and APIs.

## Surface tour

Rust-flavored syntax, dropped down to the essentials.

```rust
struct Point { x: Int, y: Int }

enum Shape {
    Circle(Float),
    Rect(Point, Point),
}

fn area(s: Shape) -> Float = match s {
    Circle(r)      => 3.14 * r * r,
    Rect(a, b)     => abs((b.x - a.x) * (b.y - a.y)).to_float(),
}

fn main() = {
    let p = Point { x: 1, y: 2 };
    let s = Rect(p, Point { x: 4, y: 6 });
    println(area(s));
}
```

What's there: `struct`, `enum`, `match`, `fn`, `let`, generics (`List<T>`), `|x| x + 1` closures, block expressions `{ … }`, `;`-separated statements with a final expression, the `?` operator on `Result`.

What's not there: lifetimes, borrowing, `&` references, `&mut`, `mut`, `unsafe`, `impl` blocks with methods (use plain functions), macros, traits with implementations (only auto-derived marker traits), async/await coloring, modules-as-files-with-mod (use namespaces).

## Top-level forms: `def` and `def local`

There are exactly two ways to bind a top-level name.

### `def` — content-addressed

```rust
def double(x: Int) -> Int = x * 2

def quadruple(x: Int) -> Int = double(double(x))
```

A `def` is an expression. Its right-hand side is hashed (after name resolution to other hashes and de-Bruijn indexing of locals). The hash *is* the identity. Two `def`s in different files with the same body have the same hash. Renaming `double` doesn't change anything any other definition observes; what `quadruple` refers to is `#h:abc…`, not the name.

Restrictions on `def`:

- The body must be a `Wire` expression (almost everything is — see below).
- May not capture node-local handles (atoms, sockets, files).
- May reference other `def`s (by hash) and built-in primitives (`#"core/+"`, `#"net/at"`, …).

### `def local` — per-node singleton

```rust
def local store: Atom<Map<Bytes, Bytes>> = atom(Map::new())
```

A `def local` is also expression-shaped, but its semantics are different:

- Each node evaluates the right-hand side independently. Every node ends up with its own `store` atom.
- The right-hand side *may* contain non-`Wire` values like `Atom`, `Socket`, `File`.
- The *value* of a `def local` is not portable across nodes. Code that references a `def local` resolves it to *whatever the executing node has under that hash*.

This is what makes the distributed hashmap work: every node has its own `store`, all keyed by the same hash. There is no globally-shared mutable cell, and there can't be — but every node has an equivalent local one, and the code that operates on them is portable.

If a `def` body tries to capture a non-`Wire` value, that's a compile error. If a `def local` body is fully `Wire`, that's allowed (but the `local` keyword is suggesting node-locality you didn't actually use; the compiler may warn).

## The `Wire` trait

`Wire` is the static answer to "can this value travel across `at`?" It is auto-derived. You almost never write it; it appears in compile errors and in the signatures of `at`, `spawn`, and similar.

### Auto-derivation rules

A type `T` is `Wire` unless it is excluded. By default:

- All primitives (`Int`, `Float`, `Bool`, `String`, `Bytes`, `Hash`, `Duration`, `NodeId`) are `Wire`.
- `struct`s are `Wire` iff every field is `Wire`.
- `enum`s are `Wire` iff every variant's payload is `Wire`.
- Generic types are `Wire` iff their type arguments are `Wire`. So `List<Int>` is `Wire`, `List<Atom<Int>>` is not.
- Function values `fn(A) -> B` are `Wire` iff `A`, `B`, and *all of the closure's captured variables* are `Wire`.
- `Node`, `Cluster`, `Result<T, Failure>` (when `T` is `Wire`), `Promise<T>` — all `Wire`.

### Opting out: `#[local]`

A type opts out of `Wire` by tagging its declaration:

```rust
#[local]
struct Atom<T> { /* runtime-managed */ }

#[local]
struct Socket { /* runtime-managed */ }

#[local]
struct File { /* runtime-managed */ }
```

Any struct or enum that *contains* a `#[local]` type is also not `Wire` by transitivity — you don't need to mark it. This is exactly Rust's auto-trait propagation, just inverted in spirit (Rust opts in to `Send`; we opt out of `Wire`).

User code rarely needs `#[local]`. It's for things genuinely tied to one host: connection handles, file descriptors, GPU buffers, etc.

### What the compiler enforces

The only place `Wire` is actually checked is at calls to `at`, `spawn`, and a small handful of related primitives. The check is on the closure passed in: every captured variable must be `Wire`. Example error:

```
error: closure captures non-Wire value
   --> main.hop:42
    |
42  |     at(node, || my_socket.send(data))
    |              ^^^^^^^^^^^^^^^^^^^^^^^^
    |              this closure captures `my_socket: Socket` (#[local])
    |
    | help: #[local] values cannot cross node boundaries.
    |       Consider:
    |        - opening the socket on the receiver via a `def local`, or
    |        - sending only the data and doing the I/O on the remote.
```

Notice that an `Atom` *referenced through a `def local`* doesn't trigger this error — the closure captures the *hash* of the def, not the atom itself. Each node resolves the hash to its own atom on arrival.

## Built-in types

```rust
struct Node { id: NodeId, addr: Address }

enum Result<T, E>   { Ok(T), Err(E) }
enum Option<T>      { Some(T), None }

enum Failure {
    Unreachable(Node),
    Timeout(Node, Duration),
    Crashed(Node),
    Raised(Node, Bytes),                // bytes = serialized thrown value
    CodeMissing(Node, List<Hash>),
    Cancelled(Node),
}

#[local] struct Atom<T>
#[local] struct Socket
#[local] struct File

struct Promise<T>                       // Wire — handles are portable
struct Cluster                          // Wire — see below
struct Duration                         // Wire

// helpers
struct List<T>
struct Map<K, V>
struct Set<T>
```

`Promise<T>` and `Cluster` are intentionally `Wire`. A `Promise` is just an opaque ticket (a (node, id) pair); awaiting it from a different node sends the wait to the home node. A `Cluster` is a marker value whose live state lives in per-node `def local` atoms.

## Primitives

```rust
fn here()   -> Node
fn caller() -> Option<Node>

fn at<T: Wire>(
    node:    Node,
    thunk:   fn() -> T,
    timeout: Duration = 30s,
) -> Result<T, Failure>

fn spawn<T: Wire>(node: Node, thunk: fn() -> T)
    -> Promise<Result<T, Failure>>

fn await<T>(p: Promise<T>) -> T
fn cancel<T>(p: Promise<T>) -> ()

fn atom<T>(init: T) -> Atom<T>          // returns Atom; not Wire
```

Semantics worth pinning down:

- **`at` is at-most-once.** On any error short of a clean reply, it returns an `Err(Failure)`. It does not retry. If the message reached the remote and the work executed but the reply was lost, you get `Err(Unreachable | Timeout | Crashed)` even though the side effect did happen. The language refuses to lie about this.
- **`at(here(), …)` short-circuits.** Same-node calls don't serialize or hit the network. This is what makes single-node and multi-node code identical at the source level.
- **Timeouts default to 30 seconds.** Override per call.
- **`spawn` is `at` without blocking.** The returned `Promise` is itself `Wire` — you can hand it to another node and that node can `await` it.
- **`cancel` is local-only and cooperative.** Your local `await` immediately resolves to `Err(Cancelled)`; the remote *may* honor the cancel signal, but treat it as best-effort. Real cooperative interruption is out of scope for v1.

## Retries and combinators

Standard library, written in the language on top of the primitives:

```rust
fn retry_at<T: Wire>(
    node:     Node,
    thunk:    fn() -> T,
    attempts: Int,
    backoff:  List<Duration>,
) -> Result<T, Failure>

fn all_of <T: Wire>(ps: List<Promise<Result<T, Failure>>>)
    -> Result<List<T>, Failure>

fn settle <T: Wire>(ps: List<Promise<Result<T, Failure>>>)
    -> List<Result<T, Failure>>

fn any_of <T: Wire>(ps: List<Promise<Result<T, Failure>>>)
    -> Result<T, Failure>

fn quorum <T: Wire>(n: Int, ps: List<Promise<Result<T, Failure>>>)
    -> Result<List<T>, List<Failure>>
```

- `all_of` — all must succeed; on first failure, cancel the rest and return it.
- `settle` — wait for all, never short-circuits. Returns a vector of results parallel to the input.
- `any_of` — first success wins; cancel the rest.
- `quorum n` — succeed once `n` succeed; otherwise return collected failures.

These are the four shapes that cover most real partial-failure code. None of them are built into the language.

## Cluster membership

The cluster is a value with no state of its own. All state lives in per-node `def local` atoms; the cluster value is a marker that names the convention.

```rust
def local membership: Atom<Set<Node>> = atom(set![here()])

def cluster: Cluster = Cluster {}        // marker, content-addressed

def members(_: Cluster) -> List<Node>
    = membership.deref().to_list()

def member_for(c: Cluster, key: Bytes) -> Node = {
    let ms = members(c).sort_by(|n| n.id);
    ms[hash(key) % ms.len()]
}

def join(seed: Node) -> Result<(), Failure> = {
    let theirs = at(seed, || members(cluster))?;
    membership.swap(|s| s.union(theirs).insert(here()));
    at(seed, || membership.swap(|s| s.insert(caller().unwrap())))?;
    spawn_thread(gossip_loop);
    Ok(())
}

fn main() = {
    start_listener();
    match env("SEED") {
        Some(s) => { join(parse_node(s)).unwrap(); },
        None    => {},                   // single-node cluster, just us
    };
    run_app();
}
```

- `members(cluster)` reads the local `membership` atom — the cluster value is just there to give the operation a name.
- Routing is decided by whoever calls `member_for`, based on *that node's* current view. Membership converges via gossip but isn't guaranteed identical at any instant.
- Same binary runs single-node (`SEED` unset) or as a cluster member (`SEED=host:port`). Application code is unchanged.

## Worked example: distributed hashmap

```rust
def local store: Atom<Map<Bytes, Bytes>> = atom(Map::new())

def local_put(k: Bytes, v: Bytes) -> Bytes = {
    store.swap(|m| m.insert(k, v));
    v
}

def local_get(k: Bytes) -> Option<Bytes> = store.deref().get(k)

def dput(k: Bytes, v: Bytes) -> Result<Bytes, Failure> =
    at(member_for(cluster, k), || local_put(k, v))

def dget(k: Bytes) -> Result<Option<Bytes>, Failure> =
    at(member_for(cluster, k), || local_get(k))
```

What's happening at the type level:

- `local_put` is a regular `def` because its *body* is `Wire` (`store` is referenced by hash, not captured by value).
- The closure `|| local_put(k, v)` captures `k: Bytes` and `v: Bytes`. Both `Wire`. The closure is `Wire`. `at` accepts it.
- `dget` returns `Result<Option<Bytes>, Failure>`. The caller must handle both layers: the remote failure case and the local "key not present" case. This is the correct shape — they are different conditions.

Single-node:

```rust
dput(b"hello", b"world").unwrap();
assert_eq!(dget(b"hello").unwrap(), Some(b"world"));
```

Multi-node: identical source. `member_for` returns a different node depending on the key, `at` serializes and ships, the receiver routes to its own `store`. The application code can't tell the difference (and ideally, doesn't need to).

Replication is a tiny library on top:

```rust
def replicas_for(c: Cluster, k: Bytes, n: Int) -> List<Node> = {
    let ms = members(c).sort_by(|n| n.id);
    let start = hash(k) % ms.len();
    range(0, n).map(|i| ms[(start + i) % ms.len()])
}

def dget_replicated(k: Bytes) -> Result<Option<Bytes>, Failure> =
    any_of(replicas_for(cluster, k, 3).map(|n|
        spawn(n, || local_get(k))))
```

## Worked example: map-reduce

```rust
def preduce<T: Wire, U: Wire>(
    f:    fn(U, T) -> U,
    init: U,
    coll: List<T>,
) -> Result<U, Failure> = {
    let ms     = members(cluster);
    let chunks = partition(ms.len(), coll);
    let ps     = zip(ms, chunks).map(|(n, c)|
        spawn(n, || c.fold(init, f)));
    let parts  = all_of(ps)?;
    Ok(parts.fold(init, f))
}
```

- The `<T: Wire, U: Wire>` bounds are usually inferable from use; written here for clarity.
- `f` is a `fn(U, T) -> U` whose `Wire` requirement is induced by `spawn`. If a caller passes a closure capturing a `Socket`, the call doesn't compile.
- `?` propagates `Err(Failure)` to the caller. Best-effort variants use `settle` instead.

One-node mode: `members(cluster)` returns `[here()]`, `partition(1, coll)` returns `[coll]`, the single `spawn` short-circuits via `at(here(), …)`, the fold is local. Twelve-node mode: twelve parallel chunks. Same function.

## What the language deliberately does not have

Not in v1, not because they're bad, but because they're library-shaped:

- **Supervision trees / restart policies.** Build on top of `spawn` + `Promise` + `Failure`.
- **Distributed transactions, 2PC, consensus.** Build on top.
- **Causal ordering / vector clocks / CRDTs.** Build on top.
- **Production-quality distributed key-value store.** The `dput`/`dget` example is a demonstration, not a product.
- **Hot code reload as a language feature.** Content addressing makes it cheap to write as a library: ship the new code, swap a `def local` pointer.

And not at all:

- An effect system. Functions don't track IO/network/mutation in their types. The one cross-machine concern (`Wire`) is enough.
- Lifetimes, borrowing, ownership. Garbage collected.
- Macros, including derive macros. Auto-derivation of `Wire` is built-in; nothing else needs derive.
- Async/await coloring. `spawn` returns a `Promise`; `await` blocks. That's it.

## Compile model

1. **Parse** source into AST.
2. **Resolve** names: lexical variables become de-Bruijn indices, top-level references become placeholder hashes, builtins become stable strings.
3. **Hash** each `def`: serialize the resolved tree to canonical bytes (Blake3), the digest is the def's identity. Mutually recursive defs form a component, hashed together; each member is `(component-hash, index)`.
4. **Typecheck** with `Wire` auto-derivation.
5. **Emit** a portable ANF-ish bytecode keyed by hash. Store as `(hash, bytecode)` in the codebase.
6. **Run.** The interpreter resolves hashes from the local codebase; on miss during a remote call, the receiver requests the missing hashes from the caller and caches them.

The codebase is just a `Map<Hash, Code>`. Disk format: a directory of files named by hash. Caching is forever-valid because code is immutable.

## Wire format (sketch)

A serialized value is:

```
version: u32
kind: u8           // primitive | tuple | struct | enum | closure | hash-ref
payload: ...       // kind-dependent
```

Closures serialize as:

```
kind = closure
code_hash: 32 bytes
n_captures: u32
captures: [Value; n_captures]
```

References to other defs serialize as their hash. Primitives serialize compactly. Persistent collections walk their structure (structure-sharing is a local optimization, denormalized on the wire).

The `at` protocol on the wire:

1. Caller sends `Call { thunk: SerializedValue, reply_addr: ... }`.
2. Receiver deserializes the thunk. On `CodeMissing(hs)`, it sends `NeedCode(hs)` back and waits.
3. Caller responds with `Code([(hash, bytecode)])`. Receiver caches and retries deserialization. Repeat until resolved (or budget exhausted).
4. Receiver evaluates the thunk.
5. Receiver sends `Reply(Result<SerializedValue, Failure>)`.

Failure cases: connection refused → `Unreachable`. No reply within timeout → `Timeout`. Connection drops mid-stream → `Crashed`. Receiver raised → `Raised`. Couldn't resolve code → `CodeMissing`.

## Open design questions

- **`Wire` inference through generics.** When a generic function takes a closure, does the `Wire` bound need to be written? Probably not for most cases — propagate from the use site. Worth nailing down.
- **Builtin compatibility across runtime versions.** Two nodes with different `core` builtin sets need to detect this on handshake and refuse cleanly, not silently corrupt.
- **Codebase garbage collection.** When can we evict cached code? Probably "never, this is cheap"; revisit if it isn't.
- **Promise location.** A `Promise<T>` is `(node, id)`. If that node dies before you `await`, you get `Err(Unreachable)`. Document.
- **`def local` initialization order across files.** Probably lazy on first use, with a memoization atom under the hood. Avoids order-dependence; matches the model.
- **Tagged-union representation on the wire.** Variants by index (compact, fragile across edits) vs by name (verbose, robust). Pick name + length-prefixed payload; pay the bytes for forward compat.

## Minimum viable milestone

To prove the model end-to-end:

1. Reader + resolver + hasher. Prove `def`s hash identically regardless of variable names.
2. Tree-walking interpreter that resolves hashes from a local `Map<Hash, Code>`.
3. Value serializer / deserializer including closures.
4. Typechecker with `Wire` auto-derivation.
5. TCP listener + `at` protocol + negotiation loop.
6. The DHM example from this doc, running on two processes on one machine.
7. `spawn` + `await` + the map-reduce example, running on four processes.

If those work, the language is real. Everything past this point — supervision, replication, persistence, cluster autoscaling — is libraries written in the language itself.
