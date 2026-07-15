# Stage F — the collection cluster (post-Stage-E gap #1)

Target: the >10× band in EXEC_MODEL_V2's post-Stage-D table — group-by
(~44×), into-xform (~40×), interleave (~26×), apply (~24×), transduce
(~24×), vecbuild (~14×), assoc-build (~12×), comp (~12×).

## Profile attribution (sample, release, 2026-07-15)

Top-of-stack samples for the three core workloads (pv/hamt internals are
Stage-D-shaped; Stage E does not change them):

- assoc-build (100k assoc): view_gc 841, decode 552, equal 486, tag_of 440,
  memmove 246, drop_in_place<ObjView> 222, hamt_assoc 123, prim 94.
- vecbuild (100k conj): memmove 196, decode 167, prim (generic dispatch)
  138, tag_of 122, view_gc 117, alloc 84, prim_from_tag 49, mk_pv 37.
- group-by (100k): decode 314, prim 284, tag_of 272, view_gc 267,
  memmove 198, resolve_call 171, run_trampoline 144, alloc 132.

READ THE NUMBERS: the dominant cost is NOT node copying (memmove is a
minority) and not the algorithms (hamt_assoc itself is 123 of ~3000). It is
PER-NODE GLUE — every trie/PV node touched goes through `view_gc`
(constructs + drops a full ObjView enum), `decode`, `tag_of`, and the
generic `prim` dispatch with `prim_from_tag`. The old Obj-enum heap forced
that shape; the raw heap does not.

## The plan (leverage order)

### F1. View-free internals for the PV/HAMT/collection prims
Rewrite the INTERNALS of the native collection paths (`pv_*`, `hamt_*`,
`mk_pv`, `node_arr`/`arr_handle` helpers, `equal`'s record/vector arms,
`hash_value`) on RAW `Gc` accessors: one `type_id()` load + direct
`field/values/raw_word` reads, no ObjView construction, no `decode` for
category tests (use `tag_of` once, or better `try_decode_ptr` + type_id).
ObjView remains the general seam for everything else — these few prims are
the measured hot core and warrant the raw layer (same judgment the JIT
already applies). Expect this alone to roughly halve assoc/vecbuild/group-by.

### F2. Monomorphic register-arg shims for the hot collection prims
PvConj/PvNth/PvAssoc/HamtAssoc/HamtLookup/ArrPush currently go through the
generic `shim_prim` (arg spill + `prim_from_tag` + the giant match). Give
them the same monomorphic `extern "C"` treatment the arithmetic shims got
in Stage A3 (register args, direct call, no tag round-trip).

### F3. Tail/transient building
- PV: real 32-wide TAIL (conj into the tail array; amortize node pushes) if
  the in-language/native PV lacks it; batched `into`/`group-by` builders
  can use `%pv-from-array` chunks (already exists) more aggressively.
- HAMT: transient-style batched building for group-by/into: mutate nodes
  OWNED by the builder (edit-session id in the node header's spare u16 —
  the raw heap makes ownership stamping trivial), freeze on exit. Surface
  as the standard transient!/persistent! protocol in the clj library so
  `into`, `group-by`, `frequencies` get it for free.

### F4. Producer/glue follow-ups (separate, after re-measuring F1-F3)
- interleave/into-xform: chunked producers.
- apply: register-arity fast path for small realized lists.
- comp/transduce: param-value specialization (EXEC_MODEL_V2 gap #4).

Gates per phase: all four suites + gc-stress battery (collection prims
mutate/allocate mid-walk — the stress battery is the soundness hammer for
F3's ownership stamping in particular) + the %nanos micro suite re-run with
before/after in this doc.

## F1+F2 results (measured 2026-07-15, %nanos micro suite, microclj --jit,
## best-of-4, ns/op; before = Stage-E-final binary, same harness/machine)

| workload | before | after | Δ |
|---|---|---|---|
| assoc-build (10k) | 1943 | 779 | **2.5×** |
| vecbuild (10k) | 163 | 151 | 1.08× |
| into-xform (10k) | 289 | 272 | 1.06× |
| group-by (10k) | 1454 | 1421 | ~1× |
| reduce+map | 56 | 55 | ~1× |
| raw-loop / calls / captures | 3 / 4 / 5 | 3 / 4 / 5 | unchanged |

Profile (sample, assoc-build, after): view_gc 841→219, drop_in_place<ObjView>
222→73, prim(generic) 284→153; the stacks now bottom out in
shim_hamt_assoc→hamt_assoc as they should. Remaining top frames: `equal`
(the comparator itself — now the bit/tag fast path, high call volume),
`arr_clone`+malloc/free (the per-node Vec round trip that F3's transients
eliminate), and residual view_gc/decode from the INTERPRETED record glue
AROUND the prims (map-record field reads / MakeRecord — the F4 producer/glue
cluster, also why group-by barely moved: its time is call glue + lazy-seq
stepping, not the trie ops).

F1 = raw `Gc` internals for pv_*/hamt_*/equal/hash_value/seq_step/as_chunked
(one type_id check, loud poison panic, no ObjView/decode per node; ObjView
stays the seam everywhere else). F2 = monomorphic register-arg shims for
PvConj/PvNth/PvAssoc/HamtAssoc/HamtLookup/ArrPush (+ Cons/First/Rest for the
non-inline models), PARKING-classified and excluded from `body_pure_loop` so
loops around them use precise demotion. Gates: all four suites + gc-stress
battery green.

## F3 results (measured 2026-07-15, same harness; before = F1+F2 numbers)

| workload | F1+F2 | F3 | Δ |
|---|---|---|---|
| vecbuild via transient (10k conj!) | 151 (persistent conj) | **24** | **6.3×** |
| assoc-build via transient (10k assoc!) | 779 (persistent assoc) | **64** | **12×** (30× vs pre-F1) |
| into [1] (range 10k) | — | 43 | (transient tail-pushes) |
| into-xform | 272 | **155** | 1.8× |
| group-by (8 groups) | 1421 | 1733 | 0.82× (see below) |
| persistent conj / assoc / lookups | 151 / 779 | 153 / 788 | unchanged |
| core band (raw/calls/captures) | 3 / 4 / 5 | 3 / 4 / 5 | unchanged |

Implementation: native `%tv-*` / `%tam-*` / `%thm-*` prims over three transient
record types; the edit-session stamp lives in the trie nodes' EXISTING `edit`
field (field 0 — the slot the cljs port reserved), NOT the header spare u16: a
u16 wraps after 65k transients (group-by makes one per call — a real-program
killer), the layouts already had the slot, and a traced value field is
GC-neutral for free (mid-transient collections rewrite the session value and
every stamp coherently — pinned by `transients_survive_moving_collections`).
Sessions come from a per-runtime u64 counter (loud panic at 2^59 — never).
persistent! is an O(1) session invalidation; stale stamps are inert (ids never
repeat). TransientVector owns a 32-capacity tail from birth (31/32 conj!s are
one in-place array push — the "real tail"); TransientArrayMap edits its kv
array in place and PROMOTES to TransientHashMap past 8 pairs under the same
session (so `persistent!` of small maps returns an insertion-ordered
PersistentArrayMap — the conformance-visible property); TransientHashMap
edits session-owned HAMT nodes in place, copy-on-first-touch otherwise.
Surface: transient/persistent!/conj!/assoc!/dissoc!/disj!/pop! with real
Clojure semantics (use-after-persistent! throws catchably; conj!/assoc!
return values must be used), ITransientCollection/ITransientAssociative
protocols for IC-dispatched hot paths, count/get/nth on transients, and
into/group-by/frequencies build through transients (exactly clojure.core's
own definitions).

group-by honesty: the 8-group microbench got 22% SLOWER — its per-element
cost is dominated by the inner `(conj (get m k []) x)` persistent-vector
build (which real Clojure's group-by shares, and which transients cannot help
— the vectors are persistent VALUES inside the map), so the outer map's
transient win is noise here while the transient/protocol plumbing adds a
little. Workloads with many distinct keys (frequencies-shaped) get the full
assoc! win. The remaining group-by gap is the inner-collection build + call
glue (F4).

## Post-F3 matched-suite scoreboard (2026-07-15, same harness as the
## EXEC_MODEL_V2 post-Stage-D table; pressure GC LIVE during the runs)

| workload | post-D | post-F3 | JVM | note |
|---|---|---|---|---|
| loop-arith | 2 | 3 | <1 | poll residual |
| defn-call | 4 | 4 | 4 | parity |
| closure-call | 5 | 6 | 2 | |
| reduce-map | 51 | 53 | 19 | |
| vecbuild (persistent conj) | 150 | 147 | 11 | conj! path = 24 |
| into-xform | 279 | 151 | 7 | transient into |
| comp-chain | 186 | 194 | 15 | F4 |
| transduce | 121 | 121 | 5 | F4 |
| apply | 885 | 877 | 37 | F4 |
| group-by | 1150 | 1039 | 26 | inner per-key conj |
| assoc-build (persistent) | 1761 | 1036 | 149 | assoc! path = 64 |
| interleave | 1875 | 3000* | 71 | *suite-order effect |

*interleave isolated = 1865 (unchanged; pressure on/off within noise). The
suite number includes collections copying the ACCUMULATED live data of
earlier workloads — the non-generational semi-space tax, now visible
because the runtime actually collects. A nursery/generational split is the
structural answer (future stage); not an F1-F3 regression.

Explicit transient use (conj!/assoc!) is at 24/64 ns/op — within 2.2x/0.4x
of the JVM PERSISTENT numbers — so the remaining persistent-path gap is
per-op glue the F4 cluster (call/lazy-seq/apply) shares.
