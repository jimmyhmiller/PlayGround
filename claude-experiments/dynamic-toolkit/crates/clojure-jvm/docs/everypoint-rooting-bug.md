# EveryPoint rooting bug — stale closure/seq cell in JIT'd macro body

## Status: OPEN. This is a REAL GC-rooting bug, currently MASKED by GC policy.

## Symptom

The upstream `core.clj` loader crashes at **form 430** — the `defn` of
`elide-top-frames` (`[^Throwable ex class-name] (let [tr (.getStackTrace ex)] …
(drop-while #(= class-name (.getClassName %1)) tr))`), whose `defn`
macroexpansion runs JIT'd higher-order core fns over closures — with:

```
panicked at crates/clojure-jvm/src/runtime.rs: clojure-jvm: RT.first on unsupported heap type_id 136
```

Because `macroexpand_once` runs the macro body under `GcPolicy::EveryPoint`
(hardcoded, the precise-rooting oracle), this crash reproduces **regardless** of
the `CLJVM_GC` env (both the default and `CLJVM_GC=every` crash at form 430). The
`import` macro referenced in earlier notes was removed from our `core.clj`; the
real form 430 is `elide-top-frames`.

## Why it is a rooting bug (not a runtime-type gap)

`type_id 136` is garbage read from a relocated/reclaimed heap cell, not a real
unregistered type. The `import` macro body is JIT-compiled and uses higher-order
`map`/`reduce1` over closures. A value that is **live across an allocating
safepoint** inside that JIT body — a closure parameter or an intermediate
seq/rest-list cell — is **not recorded in that safepoint's stackmap**, so when GC
fires (every safepoint under EveryPoint) it relocates the cell but does not
update this slot. `RT.first` then reads the stale pointer → `type_id 136`.

Evidence: running the WHOLE loader under `CLJVM_GC=every` (so `eval_form` is also
EveryPoint) crashes at the SAME `import` macroexpand, proving the bug is in the
JIT'd macro BODY's rooting, not in macro-arg passing (already disproven) or the
analyzer.

## Likely location (verify before editing)

SHARED crate `crates/dynlower` — the regalloc / stackmap machinery
(`regalloc.rs`, `linear_scan.rs`, `batch_lower.rs` stackmap emission): the
liveness of a closure-captured / parameter pointer or an intermediate cell across
a call safepoint is not being emitted into the stackmap that GC scans. Possibly
related to the cross-block-liveness regalloc issues already noted in project
memory. There may be a reproducer at `crates/clojure-jvm/tests/stale_pointer_repro.rs`.

## Instrumented findings (2026-05-31 investigation — narrows the cause)

The crash value was confirmed to be a *forwarded* object whose holding slot was
not updated: at the panic, `RT.first`'s ptr had header bit 63 set
(`header=0x8000000ab1000088`, `fwd_bit=true`), and `type_id 136` (=0x88) is the
low byte of the forwarding address. So a live `Cons` pointer in some slot was
left pointing at the relocated-but-not-updated from-space copy.

What was RULED OUT by instrumentation (diagnostics added then removed):
  - **Incomplete safepoint records.** `record_call_return_safepoint` and
    `collect_live_root_slots` were instrumented to flag any live, root-eligible
    (`I64`/`GcPtr`) value with a *defined* location (`loc != Unassigned`) but no
    spill slot. ZERO such drops. (The many `loc=Unassigned, remaining_uses>0`
    values are simply not-yet-defined values — correctly excluded.)
  - **Incomplete ancestor walk.** `walk_jit_ancestor_roots`' `Err` (no record for
    a return PC) never fired.
  - **Missed object fields.** A post-collection to-space scan (every object field
    checked for a forwarded-but-un-updated pointee) found ZERO missed fields.
  - **Missing root (heap-reachability).** A `DIAG_CONSERVATIVE` experiment that
    scanned EVERY from-space object as a root (over-rooting) did **not** fix the
    crash — so it is NOT a value reachable only via an un-scanned heap root.
  - **Alloc-path GC.** With the 16MB no-nursery heap, the alloc-triggered GC
    (`PublishJitFp` / `walk_parked_thread_jit_roots`) NEVER fires before form 430;
    all collections go through the safepoint-handler path
    (`active_jit_safepoint_handler` → `mutator_triggered_gc_with_extras`).

What remains: the stale slot is a **JIT-frame slot / register that holds a live
pointer at a collection which forwards the pointee but does not update that
slot** — even though the safepoint-handler path's stackmap + FP-chain walk both
verify complete on the path that runs. The most likely remaining mechanism is a
**register-vs-spill-slot staleness**: a value whose live copy is in a register
(or whose slot the regalloc believes holds it) at the moment GC updates the
recorded slot, but the runtime register / a later re-spill of a stale register
clobbers the GC-updated slot. Next investigator: instrument the exact safepoint
where the form-430 `defn` macro's intermediate `Cons` is built, and compare the
recorded slot offset against where the runtime value physically lives across the
collection.

## Status of the masking (LAW #3)

`macroexpand_once` keeps `GcPolicy::EveryPoint` HARDCODED (not env-driven). The
canary fires identically under the default policy AND `CLJVM_GC=every` (both
crash at form 430). The bug is NOT masked. A prior attempt downgraded this to
env-driven OnPressure (advancing the OnPressure frontier to ~611) — that was
reverted because it hid the crash from normal runs (LAW #3). The frontier under
the un-masked oracle is form 430 until the rooting bug above is actually fixed.

## DEFINITIVE localization (2026-05-31, hands-on, two decisive experiments)

Corrects all earlier theories. Facts established by experiment, not reasoning:

1. **It is a plain SEMI-SPACE heap, NOT generational.** `GcConfig::generational(16MB)`
   sets `nursery_size: None` (dynlang/src/lib.rs:81) → `Heap::new` (no nursery). So
   `has_nursery()==false`, the card table / write barrier are inert, and EVERY
   collection is a whole-heap Cheney copy (`collect_inner`). The from→to delta at the
   crash is exactly 16MB (the semi-space size). **Iteration 6's "missing generational
   write barrier" diagnosis is WRONG** — there is no minor GC / remembered set in play.

2. **The collection is the MAJOR/whole-heap path** via the safepoint poll:
   `active_jit_safepoint_handler` (dynruntime/jit.rs:1023) → `mutator_triggered_gc_with_extras`.

3. **EXPERIMENT A — the missed root is STACK-RESIDENT.** Conservatively scanning the JIT
   stack extent `[jit_fp-8192 .. fence]` in `walk_jit_ancestor_roots` and forwarding every
   from-space pointer makes form 430 PASS — the loader then advances to **form 611** (a
   different, unrelated panic). So the live pointer is on the JIT stack, just not in any
   enumerated precise root.

4. **EXPERIMENT B — it is NOT a regalloc-tracked vreg.** Forcing `safepoint_action` to
   `SpillAndRecord` for EVERY type (not just `GcPtr`/`I64`) does NOT fix it. So the missed
   pointer is not owned by the register allocator's stackmap at all — ruling out the
   F64-NaN-box-classification gap and the liveness/spill-slot theories (the earlier
   liveness-stub "fix" also had no effect).

**Conclusion:** the live `Cons` pointer is held in a **non-JIT (Rust runtime-helper) stack
frame** that is on the stack between JIT frames during JIT→Rust→JIT interleaving (the
form-430 `defn`/macroexpand path calls a Rust runtime helper — e.g. a seq/list builder or
`cljvm_rt_invoke` — which holds the rest-arg `Cons` in a Rust local across a re-entry into
JIT that hits the safepoint poll and triggers GC). Rust frames have no stackmaps, so such
temporaries must be EXPLICITLY rooted. `dynruntime` already has `ScopedJitRoots`/`RootSet`
for exactly this. This is the same JIT↔Rust interleaving family as the
`parked-walker-interleaved-frame` and `alloc-GC-walks-JIT-frame` fixes in project memory.

**Next step:** localize WHICH Rust runtime helper on the form-430 path holds the unrooted
`Cons` across a GC-triggering callback (instrument the conservative scan to report whether
the dangling slot lies in a JIT frame, the handler frame, or a Rust helper frame between
FPs), then root it via `ScopedJitRoots`. The conservative scan is a confirmed correctness
oracle for the fix but is NOT itself an acceptable fix (it scans non-precisely).

## Update 2 (2026-05-31) — pinned to the EXTERNAL register-alloc crate + guardrail plan

- **Backtrace at the crash shows `cljvm_rt_first` is called DIRECTLY from JIT** (no Rust
  frames above it). So the stale `Cons` arrives as a **JIT vreg argument**, NOT from a Rust
  helper local. Combined with Experiment B (forcing `SpillAndRecord` for every type didn't
  help), the vreg is **not considered live at the earlier safepoint where GC ran** → omitted
  from that safepoint's stackmap → its spill slot/register is never updated.
- **The allocator on the hot path is the EXTERNAL `register-alloc` crate**
  (`claude-experiments/register-alloc/src/{liveness.rs,linear_scan.rs}`), reached via
  `dynlower/src/regalloc_bridge.rs`. **NOT `dynlower/src/regalloc.rs`** — that is the
  *fallback* allocator and is never exercised here. (An earlier liveness "fix" was applied to
  the fallback by mistake and had no effect.) `liveness.rs` does correct backward dataflow but
  projects to a single `[min,max]` interval; `linear_scan.rs::build_stackmaps` decides
  safepoint liveness via `pos < start || pos > end`. The miss is in this liveness→stackmap
  path (a genuinely-live GcPtr/I64 vreg not present in a safepoint's stackmap). Pin the exact
  vreg/safepoint by instrumenting `build_stackmaps`.

### Two deliverables
1. **Fix** the missed root at its source (register-alloc liveness/stackmap) so a GcPtr/I64
   vreg live across a safepoint is always recorded.
2. **Permanent guardrail (stop-it-forever):** a PRECISE missed-root *verification* at GC time,
   gated to `CLJVM_GC=every` / debug builds (NOT in production, NOT part of the collector's
   real rooting — the GC stays precise). After a collection, scan the triggering thread's JIT
   stack extent; if any aligned word points at an object carrying the FORWARDING_BIT (i.e. a
   relocated-but-un-updated slot), PANIC with the slot address + containing frame (JIT fn via
   return-addr lookup, or "Rust frame"). This converts the whole missed-root class from silent
   far-away corruption (`type_id 136`) into an immediate, localized failure at the offending
   GC — caught in CI under the EveryPoint oracle every time.

Validation oracle (already proven): conservatively forwarding every from-space pointer in
`[jit_fp-8192 .. fence]` makes form 430 pass and advances to form 611. The real fix must make
`CLJVM_GC=every` reach ≥611 WITHOUT any conservative forwarding in the live collector.

## Success criterion (how the fix is judged)

- `CLJVM_GC=every cargo test -p clojure-jvm --test load_upstream_core -- --ignored --nocapture`
  gets **past form 430** (ideally to the same form 632 OnPressure reaches). This is
  the real gate — EveryPoint must stop masking.
- No regression: lib `cargo test -p clojure-jvm --lib -- --test-threads=1` stays 353/353;
  `load_core_subset` stays 27/27.
- This touches a SHARED crate → MANDATORY dependent-crate gating: `cargo test -p dynlower`
  and `cargo build -p beagle -p lox` (and ideally their tests) must stay green — a wrong
  stackmap/regalloc change breaks beagle/lox.
- The macroexpand OnPressure default stays (it is correct); the point is to make EveryPoint
  ALSO pass, i.e. fix the rooting — NOT to lower GC frequency anywhere else.

## Repro

```
cd crates/clojure-jvm
CLJVM_GC=every cargo test -p clojure-jvm --test load_upstream_core -- --ignored --nocapture
# crashes at form 430 import macroexpand: RT.first on unsupported heap type_id 136
```
