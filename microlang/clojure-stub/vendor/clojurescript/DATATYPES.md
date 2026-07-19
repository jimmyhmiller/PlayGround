# ClojureScript core datatypes — inventory & verbatim-port feasibility

`core.cljs` defines **86 `deftype`s** and **59 `defprotocol`s**. Not all are data
structures; below they're grouped by what they are and whether a verbatim(-ish)
port into microlang is in scope.

## A. Persistent collections — PORT TARGETS
- **Seqs:** `List`, `EmptyList`, `Cons`, `LazySeq`, `IndexedSeq`, `RSeq`,
  `ChunkedCons`, `ChunkBuffer`, `ArrayChunk`, `ChunkedSeq`, `KeySeq`, `ValSeq`,
  `NodeSeq`, `ArrayNodeSeq`, `PersistentArrayMapSeq`, `PersistentTreeMapSeq`,
  `PersistentQueueSeq`
- **Vector:** `PersistentVector`, `VectorNode`, `Subvec`
- **Array map (small):** `PersistentArrayMap`
- **Hash map (HAMT):** `PersistentHashMap`, `BitmapIndexedNode`, `ArrayNode`,
  `HashCollisionNode`, `Box`
- **Tree map (RB):** `PersistentTreeMap`, `BlackNode`, `RedNode`
- **Sets:** `PersistentHashSet`, `PersistentTreeSet`
- **Queue:** `PersistentQueue`
- **Entry / misc:** `MapEntry`, `Reduced`, `NeverEquiv`

## B. Transients (mutable-transient variants) — later
`TransientVector`, `TransientArrayMap`, `TransientHashMap`, `TransientHashSet`

## C. Lazy generators
`Range`, `IntegerRange` (+ `IntegerRangeChunk`), `Cycle`, `Repeat`, `Iterate`,
`Eduction`

## D. Scalar / value types
`Symbol`, `Keyword`, `Var`, `UUID`, `TaggedLiteral`, `Namespace`

## E. Reference types
`Atom`, `Volatile`, `Delay`

## F. Host iterators — SKIP (JS/ES6-specific)
`ES6Iterator`, `ES6IteratorSeq`, `IndexedSeqIterator`, `StringIter`, `ArrayIter`,
`SeqIter`, `MultiIterator`, `TransformerIterator`, `RangedIterator`,
`RangeIterator`, `PersistentQueueIter`, `RecordIter`, `ES6EntriesIterator`,
`ES6SetEntriesIterator`, `ArrayNodeIterator`, `HashMapIter`, `HashSetIter`,
`PersistentArrayMapIterator`

## G. Misc host — SKIP or shim
`StringBufferWriter`, `MetaFn`, `ArrayList`, `MultiFn` (we already have one)

---

## What "verbatim" actually requires

The cljs datatypes are written against the **JavaScript host**. Pasting one in
today does NOT compile on microlang. Concretely, `PersistentVector` alone uses:

- **`deftype` features we don't have yet:** inline protocol method bodies,
  `^:mutable` fields (`__hash`), inline `Object` methods, multi-arity method
  impls, marker protocols (`ISequential`), and static type fields
  (`(.-EMPTY PersistentVector)` set via `set!`).
- **Host intrinsics** (usage counts across core.cljs): `aget` ×206, `alength`
  ×118, `aset` ×96, `make-array` ×35, `aclone` ×22, `^:mutable` ×94,
  `unchecked-*` ×110, `caching-hash` ×28, `bit-shift-right-zero-fill` ×16,
  `goog.*` ×20, `js-arguments` ×7, `js-delete` ×4, `js*` ×2, plus `.slice`,
  `array`, `js/Error.`, `instance?`, `identical?`, ES6 iterators, and `m3-hash`
  based hashing.
- **59 protocols**, ~20 implemented per major type.

So a literal copy is not feasible without first building the cljs host contract.
The realistic "verbatim" path (a genuine cljs-to-new-host port) is:

1. **Extend `deftype`/`defprotocol`** to the full cljs surface: inline protocol
   methods, `^:mutable` fields, `Object` methods, multi-arity methods, marker
   protocols, static fields. (Biggest single enabler.)
2. **Host-primitive shim** mapping the intrinsics to microlang prims: array ops
   (have), unsigned shift / `bit-shift-right-zero-fill`, a `hash` prim + `m3-hash`
   / `caching-hash`, `identical?`/`instance?` (have), `array`/`.slice`.
3. **Port the collection bodies** with only the shim swapped — keeping the actual
   cljs source as the reference, under EPL with attribution. Iterators (group F)
   are replaced by microlang's seq-based equivalents; transients (B) come later.

This is a multi-stage effort (roughly: PersistentVector+nodes, then
PersistentArrayMap → PersistentHashMap+HAMT nodes, then sets, then tree/queue).
The license is already vendored (see `NOTICE.md`).
