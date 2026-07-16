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

## F4 results (measured 2026-07-15, same harness; before = post-F3 scoreboard)

| workload | post-F3 | F4 | JVM | Δ |
|---|---|---|---|---|
| apply | 877 | **25** | 37 | **35× — beats the JVM** |
| interleave (isolated) | 1865 | **101** | 71 | **18×** |
| transduce | 121 | 106 | 5 | 1.14× |
| into-xform | 151 | 134 | 7 | 1.13× |
| reduce-map | 53 | 52 | 19 | ~ |
| comp-chain | 194 | 191 | 15 | unchanged (see below) |
| core band / transients / persistents | | unchanged (2-3 / 4 / 5; 23 / 63; 147 / 798) | | |

What landed:
1. APPLY: profiled first — the cost was CLJ-SIDE GLUE (the old
   `(defn apply [f & args] …)` paid a variadic rest-list allocation, an
   `-apply-flatten` walk, and a `seq` protocol dispatch PER CALL) plus the
   shim's Vec round trip. Fixes: clojure.core-style fixed arities
   (`[f args]` pays one multifn select + the prim — the native %apply forces
   lazy nodes itself, so the `seq` wrapper was redundant), and a shim fast
   path — a ≤8-element fully REALIZED cons-list tail flattens into a stack
   buffer and register-arity fast-invokes (chunked/lazy/vector/long tails
   keep the general seq_flatten path; the realized-list rest-arg contract is
   untouched).
2. INTERLEAVE: `-interleave2` now zips one output CHUNK (16 pairs) per lazy
   step when BOTH inputs are chunked — one buffer + one ChunkedCons + one
   thunk instead of 32 cons + 16 thunks — reusing the existing chunk
   machinery (`chunked?`/ChunkedCons/`-chunk-rest-n`). map/filter were
   already chunk-aware (checked; nothing to do).
3. Call-glue bonus found by the transduce profile: `run_trampoline`
   allocated `first_args.to_vec()` on EVERY invoke (a malloc/free pair per
   call — a top profile frame). The bounce buffer now starts EMPTY and is
   only filled by actual tail bounces. This shaved transduce/into-xform/
   reduce-map across the board.

SCOPE PUNT (pre-authorized): comp/transduce PARAM-VALUE SPECIALIZATION
(EXEC_MODEL_V2 gap #4) is a deep cut — per-body specialization variants
keyed on parameter-held closure bits, entry guards, and profiling/blacklist
plumbing through the Decision policy. comp-chain's remaining ~190ns/el is
exactly that shape (5 chained closure invokes ≈ 35ns of per-invoke glue
each). Not attempted here; it is the headline item for the next stage.

## Stage G results (measured 2026-07-15, same harness; before = post-F4)

The punt above is closed. G1 (fast-invoke coverage) turned out to be the
whole story; G2 (param-value specialization) landed on top and is real but
modest, for a structural reason recorded below.

| workload | post-F4 | G1 | G1+G2 (suite) | isolated | JVM |
|---|---|---|---|---|---|
| transduce | 106 | 39 | 36 | **24** | 5 |
| comp-chain | 191 | 60 | 53 | 55 | 15 |
| group-by | 1039 | 632 | ~630 | 647 | 26 |
| into-xform | 134 | 59 | 57 | 45 | 7 |
| reduce-map | 52 | 35 | 33 | | 19 |
| interleave (isolated) | 101 | 67 | 66 | | 71 |
| apply | 25 | 21 | 21 | | 37 |
| core band / transients / persistents | | unchanged (2-3 / 4 / 4-5; 23 / 63; ~148 / ~790) | | | |

### G1 — why per-element calls missed the fast path (four distinct holes)

Profiling the isolated transduce/group-by drivers bottomed out in
`run_trampoline`/`resolve_call`/`invoke` PER ELEMENT. Four causes, fixed in
order, each re-measured:

1. **The selected multifn CLAUSE was never stamped.** `invoke` published the
   fast entry under the value call sites see (the MultiFn), but
   `shim_fast_invoke` re-selects per call and reads the CLAUSE's code word —
   which stayed null forever, so every multifn-routed call took the
   resolve/trampoline path. One-line fix: publish under BOTH.
2. **The operator fns were pure-variadic.** `(defn + [& xs] …)` (likewise
   `- * / < > <= >= = not=`, `max`/`min`) meant every VALUE call `(rf a x)`
   with `rf = +` selected a variadic clause — which the register fast path
   can, by design, never take (rest-list + frame per call). clojure.core
   answer applied: real fixed arities up front, variadic tail preserved
   byte-for-byte (`core.clj`). This alone halved group-by.
3. **Non-self TAIL BOUNCES went through the trampoline.** The per-element
   shape of every step fn is `(rf a (inc x))` — a tail call to a callee
   that is NOT self, which `shim_fast_invoke`/`shim_finish_tail` finished
   via `top.invoke` (full resolve + frame) per element. Now
   `finish_tail_fast` bounces the whole chain in a register-path LOOP
   (bounded stack, same guards as the call sites; anything ineligible still
   finishes through `top`).
4. **The emitted call site rejected MULTIFN callees outright**, shimming per
   element. `emit_call` now selects the fixed clause INLINE — one bounds
   check + one load off the MultiFn object (`MULTIFN_FIXED_OFF`) — and runs
   the same closure checks on the clause (block-param re-dispatch).

Two supporting inline emissions the same profiles demanded (`reduced?` =
`(%num-eq (type-of x) 'Reduced)` ran two GENERIC prim shims per element in
every reduce loop):

- `%num-eq` fast path widened from both-fixnums to ANY two non-ref
  immediates — LowBit has no immediate float, so bit-equality is exact
  (`emit_eq_immediates`, per-model; default stays both-int).
- `type-of` inlined for the two shapes dispatch predicates actually hit:
  immediate fixnum → the interned `'Long` sym as a compile-time constant;
  RECORD ref → one load of the type sym off the object. Everything else
  keeps the runtime's `type_tag`.

Plus `multifn_select_raw` (F1-style raw accessor: one type_id check, direct
clause-table reads) replacing the decode/ObjView select on all hot paths.

### G2 — param-value specialization (landed, honest scope)

`resolve_call` now passes the triggering invoke's ARGUMENT VALUES and the
invoked closure's CAPTURE VALUES into a first compile (`SpecEnv`). A
non-tail `(f …)` whose callee is a parameter or capture read looks up the
observed value and, if it is a closure (or a multifn with a fixed clause at
this arity) within the existing inline budget, splices the clause body
inline behind a guard — `try_value_inline_plan` / `emit_value_specialized`.

Two deliberate deviations from the sketched design, both for soundness:

- **The guard is the clause's META WORD (template id + arity + nslots +
  non-variadic in one u64), not the value's bits.** A bits-equality guard
  dies at the first GC move and — worse — is ABA-unsound under semi-space
  address reuse (a DIFFERENT closure landing at the guarded address would
  pass). The meta compare survives moves, applies to EVERY closure of the
  template, and a different object at the same address fails it.
- **Captures are never baked as constants.** The inlined body reads them
  through the live guarded value (a declared, stack-mapped SSA value — so
  reads after an inner safepoint see the post-move object). The plan-time
  capture snapshot only drives NESTED plans; each nested guard re-validates
  its own live value.

The stack-map interplay stayed clean: inlined slots are `declare_root_var`d
like the existing global inliner (which now shares `emit_inlined_body`),
the guarded callee/clause values are explicitly declared, and
`inline_outer_vars` keeps enclosing frames spilled — no new walker work.

**Why the win is modest (~3-7 ns/el on the suite): first-caller bias.**
Bodies compile ONCE per template, so the specialization is frozen to
whatever values the FIRST invoke of that template observed. Library
reducers (`-reduce-chunk`, `reduce-seq`) first run during core.clj load
under some unrelated `f`, so the hot caller's step fn often isn't the
observed one. The isolated transduce driver — where the first observation
IS the hot one — shows the real ceiling: **24 ns/el**. The structural
answer is per-observation specialized COPIES (a compiled-variant cache
keyed on observed templates + recompilation triggers), i.e. real tiering —
out of scope here.

Gates: default + jit suites, gc-stress battery, scheme (incl. conformance),
clojure-stub jit + default (incl. the oracle suite) — all green.

## Stage H results (measured 2026-07-15, same harness; before = Stage-G binary
## re-run on this machine, so the "before" column is the honest baseline, not
## the Stage-G table above — suite-order effects move apply/interleave a lot)

Target: vecbuild (147 vs JVM 11) and group-by (~630 vs JVM 26) down to the rest
of the suite's band — vecbuild ≤45, group-by ≤160.

| workload | before | after | JVM | Δ |
|---|---|---|---|---|
| vecbuild (persistent conj) | 144 | **57** | 12 | **2.5×** |
| group-by | 522 | **197** | 27 | **2.6×** |
| reduce-map | 30 | 26 | 20 | 1.15× |
| into-xform | 41 | 39 | 7 | ~ |
| transduce | 22 | 21 | 5 | ~ |
| comp-chain | 55 | 57 | 16 | ~ |
| assoc-build | 1052 | 1017 | 146 | ~ |
| apply | 193 | 207 | 39 | ~ (suite-order noise) |
| interleave | 725 | 622 | 83 | ~ |
| core band (loop/defn/closure) | 3 / 4 / 6 | 3 / 4 / 6 | 0 / 4 / 2 | unchanged |

Isolated: conj 148→58 (the prim alone is 56); group-by 443→190.

Both targets are MISSED (57 vs 45, 197 vs 160). The attribution is below and it
is the same wall in both cases; read it before proposing a next step.

### The profile overturned the hypothesis list again (the useful part)

H1 said PV nodes pay a 2-object ARRAY (handle + data blob) where the JVM pays
one, and that this is structural. MEASURED: it is not the lever. `Heap::alloc`
is the single top frame (1111/3900 samples, 28%), but its cost tracks BYTES
(cold-page first touch + `__bzero`), not the call count — the handle is 16B of
the ~336B a conj allocates. Cutting one alloc in three would buy ~5ns and cost
the mutable-array contract (`%aget`/`%aset`/`%apush`, the D5 JIT inline arms,
and F3's in-place transient tail). NOT DONE, deliberately.

H2 said the conj CALL PATH still had glue. It did — but not the protocol
dispatch itself (the 2-way IC hits; `shim_dispatch` was 28/3238 samples). See
H-1 below.

H3 said group-by's inner PERSISTENT conj was its cost and transient inner
vectors were the fix. MEASURED: wrong twice over. Decomposing group-by
(reduce-noop 14 / +keyfn 33 / +get 319 / +assoc 348) put the cost in `get` and
`assoc`, not `conj`. The `[]` DEFAULT ARGUMENT was ~370ns/op on its own.

### What landed, in profile order

1. **`(field r i)` was a generic prim shim** — the top frame of BOTH profiles.
   Every deftype method body opens with `(let [f0 (field this 0) …] …)` for
   EVERY field, so a protocol call ran ~6 generic `shim_prim`→`prim`→
   `prim_from_tag`→`decode`/`view_gc`/`tag_of` round trips before its body
   started. `(-count v)` on a 3-element vector cost **75 ns/op**. Now inlined
   under `INLINE_OBJECTS` exactly like `%aget`: RECORD guard, immediate-int
   guard, unsigned bounds check against the header aux, one indexed load off
   `RECORD_FIELDS_OFF` (pinned to RECORD's TypeInfo by
   `record_fields_off_matches_type_info`; a failed guard takes the shim, which
   keeps the loud panic). `-count` 75→13, `-nth` 84→21, `(field v 1)` 24→11.
2. **`pv_conj` round-tripped a `Vec` per conj** — `Vec::with_capacity` +
   `extend_from_slice` + `alloc_vector` = a malloc/free pair and a SECOND
   memmove. Fused into `alloc_vector_append1` / `alloc_vector_set1` (one
   heap→heap copy; allocation never collects, so the source borrow is live
   across the alloc), used by the tail conj and every trie path-copy
   (`pv_push_tail`, `pv_do_assoc`, `pv_assoc`). conj 148→77→58.
3. **Collection literals of constants are now CONSTANTS**, as in Clojure
   (`Compiler.java` parses a constant vector/map/set to a `ConstantExpr`).
   `(get m k [])` was rebuilding an empty vector through a real `vector` call
   per element. `[]` 393→24 ns/op; `(get m2 k [])` 501→79.
   The gate is `data::is_final_rep` — the DATUM's representation, not the
   ambient phase. Two wrong gates were tried and caught by a smoke test: no
   gate, and `user_phase`. `clojure.core` is READ while only core's `PVec`
   exists but its bodies RUN after `cljs_types` installs `PersistentVector`,
   so freezing those datums hands a `PVec` out of `group-by`; and the ambient
   phase does not decide it either, because core datums baked into MACRO
   TEMPLATES are expanded in user phase and are still phase-1 objects. Quoting
   is sound only when the datum already IS what its constructor would build.
4. **`vector` was pure-variadic** — `(defn vector [& args] (vec args))`, so a
   literal cost a rest-list + `-to-array` seq walk + `%pv-from-array` trie
   assembly. Real fixed arities 0-4 up front, variadic tail preserved (the
   Stage-G1 medicine, and clojure.core's own shape); `[]` is now the shared
   `-EMPTY-PV`, as `[]` is `PersistentVector/EMPTY` on the JVM. This is what
   fixed core.clj's OWN literals, which the representation gate can never
   quote: core group-by 443→204, i.e. exactly to the user-phase copy's 222.
5. **group-by/frequencies/zipmap build through a transient** and `persistent!`
   on the way out — clojure.core's own definitions. Justified by measurement,
   not by symmetry: `assoc!` on a 2-key map is 34 ns/op vs `assoc`'s 309.
   (This corrects the F3 note above, which said the outer transient was "noise
   here": it was not — `assoc`, not the inner conj, was group-by's cost. H3's
   suggestion of transient INNER vectors was NOT taken: it deviates from
   clojure.core, and the profile does not justify it.)
6. **TransientArrayMap's `-lookup` hand-rolled its scan** out of the generic
   `=`/`>=`/`inc` fns instead of `array-index-of`, the prim-style helper its
   own persistent counterpart uses (same `-eq2` comparator, same bound). Now
   shared. `get` on a transient 69→61.

### The wall (why 45/160 were not reached)

Both remainders are ALLOCATION-BOUND, and the evidence is a heap-size sweep,
not an argument. `Heap::alloc` is 28% of vecbuild; a conj allocates ~336B (a
~144B tail blob + a 16B handle + a 64B PV record) — the same shape the JVM
allocates, and the JVM does it in 12ns. The difference is not the algorithm: it
is a 4 GiB non-generational semi-space bumping through virgin pages (kernel
faults + `__bzero`, both visible in the profile) versus a TLAB in a nursery
that never leaves L2.

Sweeping `MICROLANG_HEAP_MB` (trigger = 50% of a space) isolates it:

| MICROLANG_HEAP_MB | vecbuild | group-by | apply | interleave |
|---|---|---|---|---|
| 4096 (default) | 57 | 192 | 202 | 617 |
| 512 | 62 | 233 | 239 | 771 |
| 256 | **48** | 252 | 243 | 751 |
| 128 | 53 | 201 | 245 | 741 |

vecbuild reaches the ≤45 band on page locality ALONE (57→48) — but group-by,
apply and interleave pay for it in collection frequency. That is the whole
finding: it is a real trade, not a tuning win, so the default is UNCHANGED. A
nursery/generational split is what buys both ends at once, and it is already
named as the structural answer in the post-F3 scoreboard above. It is the
honest next stage; nothing at the library or JIT level closes this gap.

group-by's residual 197 is diffuse — no frame over ~4%. Per element it is
still ~3 protocol dispatches (`get`, `assoc!`, `conj`), a 2-entry `-eq2` scan,
and 3 allocations. The two structural items behind it are the same two the
scoreboard already names: per-element dispatch (2-way IC, `DISPATCH_IC_WAYS`)
and allocation throughput.

### Conformance

25 new oracle lines in `tests/core_suite/`, every expectation generated by REAL
Clojure and matching on the default tier: group-by/frequencies/zipmap ORDERING
(key order and vector element order, including the >8-key HAMT-promotion path
via `(into (sorted-map) (group-by #(mod % 9) (range 30)))`), `vector` at every
new arity plus `apply vector` over 100 elements, and the literal-as-constant
value/type lines.

Real Clojure confirms the constant rule exactly, including its edges:
`(identical? [] [])` and `(let [f (fn [] [1 2])] (identical? (f) (f)))` are
TRUE on the JVM, and `(let [f (fn [] [(inc 0)])] (identical? (f) (f)))` is
FALSE — constant elements only. Those four lines are NOT in the corpus: this
dialect's `identical?` is `%num-eq` ("pointer identity approximated by
structural =", cljs_types.clj:42), a pre-existing documented deviation that
answers true regardless, so they cannot discriminate and one of them fails for
an unrelated reason.

Gates: default + jit suites, gc-stress battery (mandatory — H-1 changes what
the GC-visible RECORD load reads), scheme, clojure-stub jit + default (78
tests incl. the oracle suite) — all green.
