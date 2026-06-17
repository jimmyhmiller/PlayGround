# λ-Tally POC — making the design runnable

A proof-of-concept that the language design (the L3 address/permission split
behind safe intrusive data structures) **really works**: it accepts safe code,
rejects the unsafe variants with precise errors, and *runs* accepted programs on
a modelled heap under a runtime safety monitor that never fires.

This is the surface-language counterpart of the proved kernel in `../agda/`: the
linear discipline the checker enforces is the one `CombinedSound.agda` proves
sound (well-typed ⟹ no double-free / use-after-free / leak), now exercised on
real programs you can read and run.

## Run it

```
python3 tally_poc.py          # (Python 3.11+, no dependencies)
```

You'll see three things:

1. **Accepted** programs (alloc/write/free, aliased back-pointers, the
   doubly-linked essence) — checker silent, and each runs leak-free.
2. **Rejected** programs (double-free, use-after-free, use-after-move, leak, and
   dereferencing a bare aliased pointer) — each with the error explaining the
   bug *before* it can happen.
3. **Differential proof-of-work** — bypass the checker on the rejected programs,
   run them, and the runtime monitor fires on the *exact* predicted bug.
4. A **fuzz** of 20 000 random programs: every program the checker accepts runs
   with the monitor silent and no leak (0 violations).

## The mechanism it demonstrates

- `Addr` — a raw machine address, **unrestricted** (freely copyable), carries
  **no** permission. `next`/`prev` are `Addr`s, so aliasing them is trivially
  sound — a copied address can't *do* anything.
- A capability (`Perm`) — **linear** (used exactly once) and **zero-sized**
  (erased at runtime). Holding it is what authorises read/write/free and proves
  the cell is live. There is exactly one, so mutation is exclusive *even though
  pointers alias*.
- The single invariant that makes aliased back-pointers safe: **you can never
  fabricate a capability by reading memory** — `Perm`s come only from `alloc`,
  so a field read always yields a bare `Addr`. That's why the checker rejects
  `a.next.next = …` (you hold no `Perm` for `a.next`).

## Run it (v1 — the intrusive doubly-linked list)

```
python3 tally_dll.py
```

The operation safe Rust cannot express: **O(1) removal of a node by handle**, no
GC, no `Rc`/`RefCell`, no `unsafe` — fully checked. You'll see:

1. **Accepted** list programs (build, traverse by cursor, O(1) `remove` of the
   middle, drain, drop) — checker silent, each runs **leak-free**.
2. **Rejected** — double-remove, use-after-remove, **cross-list remove**, and
   leaks (drop a non-empty list / forget to drop) — each with a precise error.
3. **Differential** — bypass the checker, run the rejected programs, and the
   monitor fires on the exact heap bug (use-after-free / leak).
4. **Fuzz** — 20 000 random list op-sequences: every accepted program runs with
   the monitor silent and no leak (0 violations).

The mechanism: a `List` is a **linear** handle to a ghost **region** (the bundle
of one `Perm` per node + the link order); it can't be dropped implicitly, so no
leaks. A `Cursor` is a **copyable** address + an **erased** membership proof —
holding one does *not* borrow the list, which is exactly why `remove(cursor)`
can be O(1). The checker is a **symbolic executor** over the region: for a closed
program it tracks the exact live-node set and order, so every
membership/liveness/disjointness obligation is decided by lookup.

## Status / roadmap

- **v0 (`tally_poc.py`) — the L3 core, running.** Owned cells, the Addr/Perm
  split, linear checking + heap monitor + differential demo + fuzz. ✅
- **v1 (`tally_dll.py`) — the intrusive doubly-linked list.** Linear list owning
  a symbolic region, cursors (copyable address + erased membership proof), and
  **O(1) `remove`**. Checks and runs leak-free; double-remove / use-after-remove
  / cross-list remove / leaks rejected; differential + fuzz. ✅
- **v2 — ergonomics.** Rust-flavoured surface syntax (functions, `impl`,
  borrows), inferred ghost arguments, better error messages.

The deliberate scope line: v0/v1 check **closed/bounded** programs via a
symbolic heap. Verifying an abstract `fn remove<T>` *once and for all* (for every
list) is the inductive separation-logic proof sketched in
`../docs/07-implementation-guide.md` §6 — deferred to the kernel, not the POC.
