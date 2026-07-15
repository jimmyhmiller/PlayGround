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
