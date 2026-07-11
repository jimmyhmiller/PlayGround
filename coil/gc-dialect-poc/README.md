# GC dialect PoC — a garbage collector as a metaprogram

Proof that a memory model can be a **dialect** in coil: a stack of macros +
runtime library you import, not a feature baked into the core. The core language
knows nothing about GC. Two dialects expose the *same surface* (`new` / `fset` /
`fget` / `gc-let` / `gctype`), so one identical program runs under either — you
swap **one import line**.

## Run it

```sh
cd gc-dialect-poc
../coil run demo.coil       # implicit allocation (arena), no Allocator threaded
../coil run gcdemo.coil     # mark-sweep GC, flat objects, bounded memory
../coil run gctree.coil     # mark-sweep GC, a linked list traced through 1 root
# same source, both dialects:
cp arena.coil DIALECT.coil && ../coil run unified.coil   # never frees
cp gc.coil    DIALECT.coil && ../coil run unified.coil   # bounded; rm DIALECT.coil
```

## Result

`unified.coil` builds 100 fifty-node linked lists (5000 nodes total, sums each):

| dialect | total_sum | objects ever | **peak live** | collections |
|---------|-----------|--------------|---------------|-------------|
| arena   | 122500 ✓  | 5000         | **5000**      | 0           |
| gc      | 122500 ✓  | 5000         | **50**        | 1899        |

Same answer; 100× less peak memory under the GC dialect. Nothing changed but the
import.

## What it proves

- **Implicit allocation.** `build-list` takes no `Allocator` — coil's "allocation
  is a visible value" is *inverted by the dialect*. `new` injects an ambient heap
  from a global cell; the metaprogram threads it for you.
- **Implicit reclamation.** No `free` anywhere. The GC dialect reclaims; memory is
  bounded even though the program allocates far more than fits live at once.
- **Precise, no conservative scanning.** Roots are a shadow stack maintained by the
  `gc-let` binding form. Collection is triggered synchronously inside `gc-alloc`,
  so roots are always consistent at a collection point — no safepoints needed.
- **Reflection-generated tracers.** `(gctype T)` reflects `T`'s fields
  (`code-field-count` / `code-field-kind` — kind 5 = pointer) and emits `T-trace`,
  which follows pointer fields. `gctree.coil` keeps a 50-node list alive through a
  single root: the sum is only correct because tracing follows every `next`.
- **Core untouched.** Everything is `.coil` — macros + a runtime library + libc
  malloc/free. No compiler change was made for any of this.

## Honest limitations (the upgrade path)

These are exactly the boundaries predicted in the design discussion, now confirmed:

- **`gc-let` roots at binding granularity.** A *mutating* GC local would need its
  root slot updated on assignment — that's the transformer's job. The demos use
  recursion so each frame's `gc-let` roots the growing tail; a transformer hook
  that rewrites plain `let`/`defn` would make rooting fully transparent (no
  `gc-let`, no `gctype` — just write normal coil). That is the **Axis-2 hook**.
- **Return-value rooting.** A function returning a GC pointer pops its root before
  returning, so the caller must root it before the next allocation (the demos do:
  `build-list`'s result flows straight into a `gc-let` init). Standard shadow-stack
  caveat; the transformer removes it.
- **Non-moving, stop-the-world, non-generational.** Generational or incremental
  collection needs write barriers on `store!` → that needs `store!` demoted from a
  core special form to an interceptable macro (**Axis-2, core-form demotion**).
  Concurrent collection needs real safepoints (genuine core work).
- **Per-object malloc/free.** A real heap would be bump-in-blocks; irrelevant to
  the thesis.

## Files

- `arena.coil` — dialect: bump heap, never frees (implicit alloc only).
- `gc.coil` — dialect: precise mark-sweep GC (implicit alloc + reclamation).
- `demo.coil` — implicit-allocation demo (arena).
- `gcdemo.coil` — GC, flat objects, bounded memory.
- `gctree.coil` — GC, pointer-bearing graph, tracing.
- `unified.coil` — one program; import `arena.coil` or `gc.coil` (as `DIALECT.coil`).
