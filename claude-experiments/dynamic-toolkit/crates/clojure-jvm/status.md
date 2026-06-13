# clojure-jvm — Real Status

After ripping out the silent special-cases I'd accumulated. Numbers below
re-verified 2026-06-09.

## What works

- **Lib unit tests: 356/356 pass**
- **`load_core_subset` integration test: 27/27 pass** (fixture is `tests/fixtures/core_subset.clj`, ~238 lines of hand-curated forms)
- **`load_upstream_core` loads ALL 719 forms** of upstream `clojure/core.clj`.
  Verified 2026-06-09: zero forms skipped, zero eval panics, in ALL FOUR of
  debug/release × default-GC/`CLJVM_GC=every`. The `CLJVM_LOADER_SKIP_PANICS`
  escape hatch never triggers, and the byte-position skip-list in
  `tests/load_upstream_core.rs` (`SKIP_BYTE_POS`) is empty — the forms it used
  to carry (`*clojure-version*`, `map-indexed`, `prep-hashes`, `bounded-count`,
  `default-data-readers`) all load now.

## Fixed 2026-06-09: release-build GC crash at form 609 (uninitialized recorded root slots)

An earlier revision of this doc claimed the loader was clean in release/
EveryPoint without having actually run those configs. It wasn't: **release
builds crashed deterministically at form 609** (`defmacro case`, whose
analysis runs `condp`'s recursive local `emit` macro helper through the JIT)
with `to-space exhausted during collection: copying type_id=0
varlen_len=0x7ffc...` — the GC chasing a garbage "pointer".

Root cause (in shared `dynlower`, not clojure-jvm): safepoint records
over-approximate the live set — emission-order liveness plus
`record_materialized_gc_spill_slots` keeps every ever-materialized GC-capable
spill slot in every later record. A recorded slot whose defining store sits on
a branch the execution didn't take holds **leftover stack junk**. Debug builds
passed by luck (different Rust frame layouts leave different junk); in release
the junk at `emit__908` frame offset 480 bit-patterned like a NaN-boxed heap
pointer, and the FP-chain ancestor walker (`walk_jit_ancestor_roots`) handed
it to the moving collector.

Fix: the JIT prologue now **zeroes the frame's locals region**
`[FP+16, FP+frame_size)` via a fixed-shape loop whose count is backpatched by
`emit_frame_size_patch` (`crates/dynlower/src/backend.rs`, both arm64 and
x86_64). Zeroed never-written slots decode as non-pointers and are skipped by
the root walk; written slots are real values the GC updates. This is the
standard sound completion for over-approximate stackmaps. Diagnostics added en
route: `CLJVM_WALK=1` now also prints per-slot bits in the ancestor walk;
the to-space-exhaustion assert reports the object/space stats; the loader
test prints per-form heap stats under `CLJVM_LOADER_HEAPSTATS=1`.

## case* — resolved, no longer a frontier

An earlier revision of this doc said the loader died at form 215 and implied
`case*` was a blocker. Both halves of that are stale:

- The form-215 wall (`(defn agent ...)` → `Agent$Action/pooledExecutor`) is
  past — unresolved host references now compile to a deferred runtime throw
  (see "Deferral mechanisms" below) instead of either silently nil-ing (old
  hack) or aborting the whole load.
- `case*` is **fully implemented** across all three layers, verified working
  in the full-load run:
  - parser: `parse_case_form`, `src/lang/compiler.rs` ~5790–5919
  - IR emission: `CaseExpr::emit`, `src/lang/compiler.rs` ~5631–5747
  - runtime dispatch: `cljvm_case_dispatch`, `src/runtime.rs` ~4827–4855
  (Stale claims about `case*`/form-215 SIGABRTs were an artifact of *other*
  forms' panics aborting the loader before it could progress; those panics
  are themselves fixed — see the empty skip-list above.)

## What broke when I removed the hacks (history — do NOT reintroduce these shapes)

The old "loader passes 719/719" claim was held up by silent fallbacks that made unimplemented Java references resolve to nil. Removed:

1. `analyze_seq` host-side rewrites for `apply` / `reduce` / `reverse` / `map` / `filter` / `take` / `range` / `interpose` / `distinct` / `coll?` / `comp` / `partial` / `merge` / `set` / `constantly` — these special-cased clojure.core fns by name, not real implementations
2. `analyze_symbol` "primops-as-values" rewrite that turned `+`/`-`/`*`/`/`/`<`/`>` etc. into wrapper closures when used outside head position
3. `macroexpand_once` silent-nil intercepts for `defmethod`, `extend-protocol`, `extend-type`, `extend`, `reify`, `proxy`, `gen-class`, `gen-interface`, `import`, `use`, `require`, `refer`, `refer-clojure`, `load`, `load-file`, `load-string`, `assert`, `add-doc-and-meta`, `alter-meta!`, `alter-var-root`, `reset-meta!`, `vary-meta`, `with-meta`, `when-class`, `when-let*`, `when-not-empty`, `assert-args`, `case`, `volatile!`, `vswap!`, `vreset!`
4. `analyze_symbol` silent fallback for any namespace-qualified or uppercase-leading symbol → nil. This was the load-bearing one — it made `Math/ceil`, `clojure.lang.Numbers/add`, `Long/valueOf`, `java.util.X` etc. all resolve to nil if not registered.
5. `parse_dot_form` / `parse_instance_method_form` / `parse_new_form`: unregistered method/constructor → `NIL_EXPR`. These are now `panic!("unregistered ...")` or deferred throws.
6. `parse_var_form`: unresolvable var → `NIL_EXPR`. Now `panic!`.

Today's 719/719 is NOT those hacks back in disguise: unresolved references
become **loud runtime throws**, not silent nils, and unimplemented runtime
stubs **log** every hit (see below).

## What's still in (and why)

**Macroexpansions that match upstream** — these are real reproductions of upstream macro expansions, not fakes:
`when-let`, `if-let`, `dotimes`, `declare`, `lazy-seq`, `doseq`, `for`, `defprotocol`, `definterface`, `deftype`, `defrecord`, `def-aset`, `comment` (real upstream returns nil), `or`, `and`, `cond`, `doto`, `->`, `->>`, `with-out-str`, `definline`, `defonce`, `..`, `dosync`, `sync`, `locking`, `while`, `with-open`, `letfn`, `future`, `future-call`, `doall`, `dorun`, `time`, `binding`.

`defmulti` / `defmethod` are NOT host-intercepted (the old `(defmulti name f)` → `(def name nil)` hack is deleted): they go through upstream core.clj's real `defmacro defmulti`/`defmacro defmethod` via the normal macro-Var path, building real `clojure.lang.MultiFn` instances. Without core.clj loaded they fail loudly as unresolved symbols.

**Skip-list of `defn`/`defmacro` redefinitions of special forms** — `let`/`loop`/`fn`/`if`/`do`/`quote`/`def`/`var`/`throw`/`try`/`recur`/`new`/`.`/`set!`/`monitor-enter`/`monitor-exit`/`case`/`case*`. Without skipping, upstream's redefinition stack-overflows because the new macro fn body references `let` which now expands to itself. This is a structural workaround, not a per-fn special case.

**Deferral mechanisms (analyze-time)** — `analyze_symbol`
(`src/lang/compiler.rs` ~6465): an unresolvable dotted symbol ("host class we
don't model") or unresolvable plain symbol compiles to a `ThrowExpr` carrying
a clear message (`"unresolved class reference: ..."` / `"Unable to resolve
symbol: ..."`). The enclosing `defn` loads; the error fires loudly only if
that code path actually runs. `Session::eval_form` catches a top-level throw,
prints `[cljvm] top-level form threw: ...`, leaves the Var unbound, and lets
the loader continue. This is a deferral, not a nil-fallback.

## Honest numbers

| Test | Result |
|---|---|
| `cargo test -p clojure-jvm --lib` | 356/356 pass |
| `tests/load_core_subset.rs` | 27/27 pass |
| `tests/load_upstream_core.rs` (debug, default GC) | 719/719, 0 skipped, 0 panics |
| `tests/load_upstream_core.rs` (debug, `CLJVM_GC=every`) | 719/719 |
| `tests/load_upstream_core.rs` (release, default GC) | 719/719 (crashed at form 609 before the frame-zeroing fix) |
| `tests/load_upstream_core.rs` (release, `CLJVM_GC=every`) | 719/719 |
| `tests/conformance.rs` vs `/opt/homebrew/bin/clojure` | 292/295 exact match (99.0%), 3 mismatches: Ratio `(/ 1 2)`, defrecord field access, defprotocol/extend-protocol |
| `tests/try_catch.rs` (run individually, `-- --ignored`) | try/catch/finally semantics vs oracle: catch+finally, ordering, class dispatch, fn-boundary throw, re-throw, multi-arm — all pass, also under `CLJVM_GC=every` |

## try/catch — done for real (2026-06-10)

`throw` lowers to `abort_to_prompt`; `TryExpr.emit` installs the handler block
and the catch dispatch is now CLASS-BASED (Java first-assignable-catch):

- Each `(catch Class e …)` arm analyzes its class symbol like any expression
  (host-class registry; `prelude` mock vars cannot shadow registered classes)
  and emits a `cljvm_inst_isInstance(class, thrown)` test — the same extern
  `instance?` uses. Mismatch falls through to the next arm; no arm matching
  re-raises outward (running `finally` on the way).
- `(catch Throwable e …)` elides the test (hierarchy root — always true) and
  ends the chain.
- Multiple catch arms work (the old `unimplemented_port` is gone).
- The thrown value rides the dispatch chain as block params, so EveryPoint GC
  rooting is sound (verified).
- `ExceptionInfo` is a real GC heap type (message/data/cause/stack-trace traced
  slots, real 2/3-arity ctors, real accessors) — no nil-returning ctor stub.
- java.lang exception classes (`ArithmeticException`, `NullPointerException`,
  `ClassCastException`, …) now carry their short-name aliases, matching Java's
  `java.lang.*` auto-import; `(catch ArithmeticException e …)` resolves.

Known divergence (pre-existing): we don't reject `(throw <non-Throwable>)` at
compile time like Java does. A thrown Long is only caught by `catch Throwable`
(class tests are honest: `isInstance(Exception, 42)` is false).

## The real frontier now: making "loads" mean "works"

All 719 forms *load and analyze*; a handful lean on deferrals/stubs that fire
at load time or would fire when the affected fns run. From the 2026-06-09 full
load, every logged gap:

| Gap | Hits at load | Mechanism | What's needed |
|---|---|---|---|
| `.alterMeta` | 22 | `[cljvm-stub]` logged no-op | real Var/IRef metadata alteration |
| `clojure.core.import*` | 3 forms threw | deferred throw → Var unbound | real `import*` (host class registry) |
| `clojure.lang.Compiler.LOADER` | 1 form threw | deferred throw → Var unbound | model or honestly bypass Compiler.LOADER dynamic var |
| multi-binding `for` (2+ bindings) | 1 | `[cljvm-stub]` returns nil | full `for` expansion with nested bindings |
| `MultiFn.addMethod` on a NIL multifn var | 1 | `[cljvm-defer]` logged, method not registered | embed the sub-file (e.g. core_print) whose `defmulti` defines the var. Multimethod dispatch itself is REAL (`defmulti`/`defmethod` via upstream macros + `clojure.lang.MultiFn`); only the nil-receiver load-time defer remains |

The stub rows are still hacks by this project's own standard (logged, but they
return a value instead of erroring) — they predate this doc revision and are
listed so they get replaced with real implementations or hard errors, not
forgotten. Beyond these, running each core fn correctly still requires filling
out the runtime types upstream assumes (Pattern/Matcher, agents, refs,
java.util/java.io surface we choose to support via C-FFI, etc.) — that work is
tracked by the conformance harness (`tests/conformance.rs`) against real
Clojure, not by this loader test.
