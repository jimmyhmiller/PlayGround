# Node-resident `state` + remote handler invocation

## Goal

Let an app author write a stateful service as a top-level `state` binding plus
ordinary handler `def`s that close over it, and let a remote participant run
those handlers on the owning node with `at(node, || handler(msg))`. The atom
never travels. The handler is resolved by hash on the node, and the `state` it
references resolves to that node's single live cell.

## Core invariant

A `state` binding is a **node singleton keyed by its content hash**. Its
initializer runs exactly once per node per hash; installing a hash that is
already live is a no-op. Every reference, local or via a shipped handler,
resolves to the one live cell. This is the opposite of `at()`'s by-value
capture (which forks).

## Example: counter service

Node app (author writes this):

```
enum Cmd { Bump(Int), Get }

state counter: Atom<Int> = atom(0)     // node singleton, evaluated once at boot

def handle(c: Cmd) -> Int =
    match c {
        Cmd::Bump(d) => swap(counter, |n: Int| n + d),
        Cmd::Get     => deref(counter),
    }

def boot() -> Int =
    match tcp_listen(9000) { Ok(fd) => serve(fd), Err(_e) => 0 - 1 }
```

Remote participant (shares the declarations so hashes match):

```
def bump_remote(node: Node, d: Int) -> Int =
    match at(node, || handle(Cmd::Bump(d))) { Result::Ok(v) => v, Result::Err(_) => 0 - 1 }

def read_remote(node: Node) -> Int =
    match at(node, || handle(Cmd::Get)) { Result::Ok(v) => v, Result::Err(_) => 0 - 1 }
```

Two participants bumping +5 then +10, then reading, observe 15. The counter is
shared on the node.

## The wrong way (rejected)

```
let c = atom(0);
at(node, || swap(c, |n: Int| n + 1))   // captures a LOCAL atom -> typecheck error
```

A bare `Atom<T>` may not cross `at()` as a capture or thunk return: it would
fork. Reference a `state` binding to share node state, or `deref` for a
snapshot.

## Semantic rules

1. `state NAME: T = INIT` is a top-level item; `INIT` resolves like a def body.
2. Content hash = hash of canonical `(T, INIT)`; same definition -> same identity.
3. Install is once-per-node-per-hash and idempotent (later installs of a live
   hash are no-ops).
4. A state reference compiles to a load of the live cell (`StateRef`), never to
   re-running `INIT`.
5. State cells are GC roots (reachable only from the node table).
6. A bare `Atom<T>` may not cross `at()`; node state crosses by reference (a
   hash), not by value.

## Implementation phases

1. surface + parse `state`.
2. resolve + content-address (TopKind::State, Expr::StateRef, SCC hashing with
   states as nodes); typecheck.
3. node-resident state table (hash -> AtomicPtr<u8>) registered as a GC root;
   ai_state_get/present/set runtime fns.
4. codegen StateRef + idempotent installer thunks + state-install pass at boot.
5. wire/knowledge base: ItemKind::State, collect_transitive_deps, idempotent
   server install.
6. reject bare Atom crossing at() (extend type_contains_ptr machinery).
7. generic serve(fd) lib fn + end-to-end tests.

## Status (built)

All seven phases are implemented and tested (lib suite green):

- `state NAME: T = INIT` parses, resolves (content-hashed in the unified
  fn+state SCC), typechecks, and codegens an idempotent installer.
- Runtime node table: `Heap.state_slots` (hash -> `globals` GC-root slot) +
  `ai_state_present`/`set`/`get`. State cells are GC roots, relocated
  correctly.
- `StateRef` lowers to `ai_state_get(thread, &hash)`; installers run at
  `Jit::new` startup and on the `IncrementalJit` (at-server) install path,
  idempotent by hash.
- Bare `Atom` crossing `at()` (captured or returned) is a typecheck error.
- Tests: `node_state_shared_singleton` (local shared counter + PMap store),
  `node_state_remote_handler_shared` (in-process `at()`: two participants
  bump, a third reads the shared node cell — the atom never travels),
  `at_thunk_{capturing,returning}_atom_is_rejected`.

## at() memoization vs stateful handlers (FIXED)

`serve_one` memoizes Call replies by `blake3(encoded closure + captures)`,
which is sound ONLY for pure thunks. A handler that touches node `state` must
NOT be memoized (else a repeated identical call skips its mutation). Fixed:
`knowledge::stateful_hashes` computes (fixpoint over the module) every def +
lambda hash whose body transitively reaches a `StateRef`; these are copied into
`Runtime.stateful_hashes` at JIT install (`Jit::new` + `IncrementalJit`).
`serve_one` reads the shipped closure's lambda hash from the payload prefix
(`payload[1..33]`, no decode) and BYPASSES the cache when it's stateful - it
never consults or stores. Pure thunks still cache. Test:
`node_state_stateful_calls_bypass_cache` (two identical `Bump(5)` -> 5 then 10,
0 cache consults). The ail `serve` path uses `wire_invoke`, which has no cache
at all, so it is always correct for stateful handlers.

## Generic `serve` (built)

The stdlib provides a handler-agnostic node loop over the ail transport:
`serve_turn(fd)` (one request: accept -> recv -> `wire_invoke` -> send),
`serve_turns(fd, n)` (bounded, testable), `serve(fd)` (forever, TCO loop).
`wire_invoke` decodes the shipped closure, invokes it on this node, and encodes
the result - no memoization. The node author just `serve`s; remotes ship
`|| handler(msg)`. Proven over REAL loopback TCP by
`ail_tcp_node_state_remote_handlers` (two participants bump, a third reads, the
state cell never leaves the node).

## Open decisions / not built

- Identity granularity: pure content hash (default) vs per-app namespace salt.
- Concurrency: install currently checks-then-adds under the `state_slots`
  mutex; a compare_exchange-to-claim would harden against a future
  multi-threaded boot.
- No persistence/replication/auth/eviction in v1.
