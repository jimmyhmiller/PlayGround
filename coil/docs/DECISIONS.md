# Coil — Design Decisions (Jimmy, from the questionnaire)

These are the chosen directions for the rough-edge findings that were design calls, not
bugs. Ordered here by a sensible execution sequence, not by decision number. Each is real
work; do them one at a time, gated by the usual rebootstrap (fixpoint + gate-full +
arm64 gate-run + gate-cli + gate-diag) and the finding's own repro.

## Execution order (dependency-aware)

1. **store! returns unit (std-12).** ✅ DONE. Do this FIRST and standalone — it touches every file
   (each effect-only `(store! x v)` in a value position), so it must land when nothing else
   is in flight or it conflicts with everything. Jimmy: "We have lint and fix stuff. Use it!"
   → drive the ripple sweep with the `hivemind` test-gated swarm (gate = the real rebootstrap
   / gate suite) and/or `verify`, not by hand. Make `store!` return unit (or canonical i64 0);
   sweep lib/, selfhost/, examples; keep gates green.
   OUTCOME: `store!` now yields unit (canonical `i64` 0) — changed in `check.coil` (`synth-store`
   type), both codegen backends (`codegen.coil` LLVM + `codegen_a64.coil` arm64), and the
   `comptime.coil` evaluator; guide/LANGUAGE_GUIDE updated. The actual tree-wide ripple was 3
   sites (a full emit-obj survey of lib/, examples/, apps/ found zero others — the codebase
   already wrapped non-`i64` stores), so the swarm wasn't warranted; the 3 were fixed by hand
   with the `(do (store! …) 0)` idiom. Full IR snapshot regenerated (82 store→0 phi/ret lines);
   fixpoint + all 7 gates green; teeth-tested regression added to `gate-cli.sh`.

2. **Type ascription `(: value type)` (gen-9)** ✅ DONE + fix the let-inference hole. Small, unblocks
   ergonomics, and useful while doing the bigger type work below. General ascription
   expression, works anywhere; also fix inference across a `let`.
   OUTCOME (two independently-green sub-steps):
   (a) `(: value type)` is a new general expression (`EAscribe`, parser-produced, same field
       shape as `ECast` so the sum size is unchanged). It CHECKS `value` against `type`
       (flowing the expected type in, any legal coercion; a mismatch is a located "has type X
       but expected Y") — NOT a numeric cast — and the checker LOWERS it to the coerced inner
       value, so no node reaches mono/codegen. Ripple: parser, resolve (qualify type),
       astdump, check (consume/strip), the comptime interpreter (eval/fold/collect), and a
       clear compiler-bug guard in mono/codegen/codegen_a64 for the node that can never reach
       them. A bare `:` reads as an empty keyword, so the parser intercepts it before head-sym.
   (b) Inference now flows across a `let`: when the body tail is a bare `(EVar nm)` and the let
       has an expected type, that returned binding's value is checked against it — so
       `(let [r (Okk 5)] r)` returned as `(Res i64 bool)` type-checks with NO annotation. The
       flow is failure-gated (`synth-bind-value` tries no-expectation first, only retrying with
       the expected type if inference strands), so the success path is untouched — gate-full
       stayed byte-identical and the fixpoint held; zero snapshot regen.
   Both halves are teeth-tested in `gate-cli.sh` (fail on the seed, pass here). NOTE: the
   self-host source can't yet USE either feature (rebootstrap stage0 is the committed pre-gen-9
   seed, which can't compile them) — the `parser.coil::parse-expr` workaround stays until the
   seed is refreshed past this commit; its comment now says so.

3. **Qualified trait-method calls `(A::go x)` (gen-6).** ✅ DONE. Add `::`-qualified method calls
   that resolve `Trait::method` through the receiver's impls table; a same-name collision becomes
   recoverable. Fix the collision error to stop suggesting `:use` when both traits are local.
   OUTCOME: a `Trait::method` call head now pins dispatch to the named trait. The reader already
   tokenizes `A::go` as one symbol, so no lexer change; `resolve.coil` gains `find-coloncolon` and
   exempts a `::`-head from the strict undefined-reference check (the method is validated in the
   checker, not resolve). `check.coil::synth-call` splits the head into `qual`/`mname` and routes
   through a new `resolve-method-qual`, which selects the MethodEntry whose trait matches `qual`
   by full name OR last-dotted component — so `A::go`, `app.A::go`, and a `:use`'d trait's
   `Show::show` all resolve, bypassing in-scope ambiguity/visibility (an explicit trait name IS
   the disambiguation, like Rust's fully-qualified syntax). A plain call has no `::`, so `qual`
   is "" and `mname` is `func`: byte-identical to the old path (gate-full/mono/checked stayed
   green with zero snapshot regen). Both collision errors were reworked: they now name the
   candidate traits and lead with `call it qualified: \`A::go\` or \`B::go\`` instead of the
   misleading `:use` advice (which never helps when the colliding traits are your own — the exact
   case that fired the `:use` message, since a bare-named caller like `main` defeats the module
   visibility check). An unknown qualifier/method is a located trait-method error, never a bare
   "undefined function". Teeth in `gate-cli.sh` (A::go→111, B::go→222, `A::nope` located,
   collision message advertises the qualified form) — all FAIL on the pre-gen-6 seed. NOTE: a
   cross-module trait reached through an `:as` alias (`(impl t.Show …)` / `t.Show::method`) is
   still unsupported (cross-module impls are a separate gap) — `:use` the trait to reach it.

4. **Supertrait `(deftrait Derived [Self] :requires [Base] …)` (gen-8).** ✅ DONE. A `:requires`
   clause, separate from the type-param vector, so associated-type params and supertraits stop
   sharing one syntax. Warn (or error) on the old ambiguous form.
   OUTCOME: `deftrait` now takes an optional `:requires [Base …]` clause between the `[Self …]`
   vector and the methods. `TraitDef` gains a `requires` field (`ast.coil`), the parser reads the
   clause (`parser.coil::parse-deftrait`, keyword-detected like `extern`'s `:cc`; a non-vector
   argument is a located error), the resolver qualifies each supertrait name with the SAME
   `qualify-trait-names` used for bounds (`resolve.coil::qualify-one-trait`, now wrapped in a
   `try`) — so an unknown supertrait is a located "unknown trait" error and the stored names are
   canonical/qualified. ENFORCEMENT is a new `check.coil::check-supertraits` pass run right after
   `setup-impls` (order-independent — the base impl may appear anywhere): for every registered
   `(impl Derived T)` whose trait declares supertraits, each `Base` must have a matching impl for
   the same `T`, checked with the existing `impl-match` (works for concrete AND generic
   `(impl [T] … (Box T))` self-types); otherwise a located "impl Derived for T: requires an impl
   of supertrait 'Base' …". The OLD AMBIGUOUS FORM — a supertrait smuggled into the `[Self …]`
   vector — is caught in `qualify-one-trait`: an extra type-param that RESOLVES to a known trait
   (`resolve … 2` returns a dotted name) is a located error pointing at `:requires`; a genuine
   associated-type name like `K`/`E` (à la `Get [Self K E]`) resolves to nothing and is untouched.
   The change is purely additive: fixpoint byte-identical and ALL gates (ast/resolved/checked/
   mono/full/diag/cli + arm64 gate-run) green with ZERO snapshot regen — no corpus trait uses
   `:requires`, and `d-trait` omits `requires` exactly as it already omits `type_params`. Six
   teeth in `gate-cli.sh` (a `:requires` supertrait builds+runs → 11; a missing base impl is
   rejected and NAMES the supertrait; the `[Self Trait]` form is rejected pointing at `:requires`;
   `[Self K E]` still parses) — all FAIL on the pre-gen-8 seed (which parses `:requires` as a bad
   trait method and silently accepts `[Self Animal]`). NOTE: this is the ENFORCEMENT + syntax
   half. Using a supertrait through a bound — calling a `Base` method on a `T: Derived`-bounded
   generic — is a natural follow-on (a transitive `bounds-has` over `requires`) not yet wired;
   the guarantee exists, generic code that leans on it is the next step.

5. **Iteration & generics over collections — BOTH (gen-1 · std-11 · std-4).** ✅ DONE. The biggest item.
   (a) Associated-type bounds: make parameterized traits (Get/Set/Push/Pop/Iter) usable as
   bounds by treating non-Self params as associated types determined by the impl. (b) An
   Iterator/Iterable protocol collapsing hm-for/al-for/slice-for/for-in into one `(for x (iter
   coll))`. Sequence: bounds FIRST (unblocks generic collection code), iterator SECOND. std-4
   (`(for-in (in map))` iterates garbage) is folded in — it goes away with the real protocol.
   OUTCOME (two independently-green sub-steps):
   (a) **Associated-type bounds.** The old hard guard ("trait 'Get' takes type parameters —
       bounds over parameterized traits aren't supported yet") is gone. The deferred
       (type-param `Self`) trait-call path in `check.coil` now substitutes the trait method
       signature `Self -> the bounded param` AND `each extra param -> a synthetic associated
       type` (`build-trait-subst` + the existing `subst-apply`), so the bounded body
       type-checks against the associated types; those names (`C$assoc$Get$K`, a spelling no
       user identifier can produce) are added to the body's tps (`assoc-body-tps`), while the
       stored func type_params are unchanged (associated types are impl-determined, never named
       by a caller). Mono already resolves an `ETraitCall` by selecting the receiver's impl and
       reading its type params off the receiver, so a deferred parameterized-trait call lowers
       to the concrete impl method with NO new mono code — the associated types never reach a
       stored position, and codegen re-derives concrete types (gate-full stayed byte-identical,
       zero regen). `resolve.coil::ty-str` renders a synthetic associated type as `<C as Get>::K`.
       Return-position associated types (Iterator's Item, Pop's E) are fully generic; an
       associated type in argument position (a Get key) needs a value of that type — a literal
       is a located "expected <C as Get>::K" (the honest limit: no literal-indexing a generic).
   (b) **Iterator/Iterable protocol.** `(deftrait Iterator [Self Item] (next … (-> (Option
       Item))))` + `(deftrait Iterable [Self It] (iter … (-> It)))` in the prelude (Item/It are
       associated types, so `(I Iterator)` is a usable bound). `(SliceIter T)` in slice.coil
       (covers strings) + Iterable(slice)/Iterator(SliceIter); ArrayList's Iterable reuses it
       over `al-slice`; `(MapIter K V)` in hashmap.coil iterates the map's KEYS (so `(in map)`
       is correct — std-4 fixed). `for-in`'s `(iter COLL)`/`(in COLL)` drive the protocol
       uniformly, and `(for x COLL …)` (bare-symbol binding) is the `(for x (iter coll))`
       surface. `slice-for`/`al-for` collapse to thin aliases over it; `hm-for` stays the
       direct key+value walk. gate-full IR byte-identical (dead-stripped when unused); the only
       churn is gate-expand (pre-codegen loaded forms) + `lib/fs.coil` joining that corpus
       (stale-missing). Teeth for BOTH halves in `gate-cli.sh` (Pop/custom bounds; iter-slice,
       in-map, generic `(I Iterator)`, al-for alias) — all FAIL on the pre-gen-1 seed.

6. **File-relative imports + migrate the self-host source (tool-1).** ✅ DONE. Resolve a relative
   import against `dirname(importing file)` (the documented, universal rule).
   OUTCOME: a relative `(import "x.coil")` now resolves against the IMPORTING FILE's directory,
   not the process CWD — so the `src/main.coil` layout `coil new` scaffolds can finally import a
   sibling `src/util.coil`, and multi-file apps (chip8, invaders — bare-name sibling imports) build
   from any directory (they were broken from repo root). The base for the entry file is
   `(dirname path)` — changed at the one shared front-end seam (`run-pipeline`) plus the dump/expand
   wrappers and the two `cheader` call sites in `driver.coil`; nested imports already used each
   imported file's `incdir`, so no other site needed touching. `cwd` still flows for DWARF comp_dir.
   The self-host ENTRY files (`main.coil`, `main_a64.coil`) were the only source written root-relative
   (`"selfhost/src/driver.coil"`); every sub-module already imported siblings by bare name, so the
   migration was just those two files' import blocks → bare names.
   THE TWO THINGS THE PRIOR ATTEMPT HIT, run down:
   (a) *"broke the bootstrap's own imports"* — a naive base switch made the OLD seed unable to build
       the migrated `main.coil` (its `(import "driver.coil")` no longer resolves under the old
       CWD rule). Inherent: the seed must understand the new rule. Both seeds were REFRESHED (built
       via an intermediate compiler that has the new resolution) — see `selfhost/seed/SEED_VERSION*`.
   (b) *"emit-ir change I couldn't explain"* — under a file-relative base, a BUNDLED lib's imports
       (and the prelude's `control.coil`→`print`→`io` chain) were resolving against the entry file's
       directory, where demos like `examples/io.coil` / `examples/control.coil` SHADOW the real
       library — so the io/fmt half of the prelude silently vanished from every `examples/*` build.
       Fixed: the prelude and any bundled lib resolve their own imports against a `<bundled>` sentinel
       (`loader.coil::bundled-base`), never a real directory — the bundled stdlib is self-contained.
       With that, emit-ir is byte-IDENTICAL to the reference (disk lib == bundled lib, so which copy
       loads doesn't matter). Fixpoint + all 7 gates (ast/resolved/checked/mono/full/diag/cli) +
       arm64 gate-run + the nollvm bootstrap all green; the expander gate GAINED coverage (chip8 /
       invaders now resolve their siblings, so their expansions include the sibling modules). Three
       teeth in `gate-cli.sh` (sibling import in project mode; from an arbitrary CWD; and the prelude
       reaching bundled io despite a same-named decoy in the entry dir) — all FAIL on the pre-tool-1
       seed. The three gated fixtures that imported non-bundled/root-relative paths were migrated too
       (`examples/dyn_write.coil` → `../lib/dyn.coil`; a diag + a resolve fixture).

7. **Delete the comptime interpreter — 3 steps (mac-8 · mac-12 · diag-4).**
   1) ✅ DONE. Route `(comptime E)`/`(const …)` through the compiled engine (closes mac-8).
   2) ✅ DONE. `export-c` on the arm64 backend → `main_a64` registers a builder → the compiled
      engine becomes available on the LLVM-free compiler too (the last dependency blocking the
      deletion; the LLVM-free compiler stops being secretly weaker).
   3) Delete `comptime.coil`'s evaluator, the `COIL_META` flag, parity.sh, and guide.coil:426.
   The deletion CANNOT come first — steps 1–2 remove what still depends on it.
   STEP 2 OUTCOME: the arm64 backend is AAPCS64-native, so `(export-c …)` needs no C-ABI thunk —
   an exported function is emitted directly under its C symbol with external linkage
   (`codegen_a64.coil::g-register-sigs!` + `g-export-c-sym`); the sole exception, a by-value
   STRUCT param (this backend passes aggregates by pointer), is a clear located error, never a
   silently-wrong symbol (`g-export-needs-thunk`). `cga64-build-object` gained an `emit-dwarf`
   flag (true for a normal build, byte-identical; false for a metaprogram dylib). `main_a64`
   registers `meta-build-obj-a64`, so the LLVM-free compiler now builds metaprogram dylibs with
   the arm64 backend and the compiled engine is its DEFAULT (verified: a `(meta …)` program runs
   to 42 with NO libLLVM linked; the seed still fails loudly on `COIL_META=compiled`). Both
   rebootstraps (LLVM fixpoint + all 7 gates; nollvm fixpoint + gate-run) green with ZERO snapshot
   regen — the compiler has no exports, so its own object is unchanged. Teeth in `gate-cli.sh` (an
   arm64 export-c object built and called from C; a by-value struct param rejected by name) FAIL
   on the seed, PASS here.
   STEP 1 OUTCOME: a `(comptime E)` / non-literal `(const …)` site is elaborated to ONE `EComptime`
   node (a const REFERENCE elaborates to `(comptime value)`, see `check.coil::2610`), so the whole
   fold has a single seam: `comptime.coil::fold-expr`'s `EComptime` arm. That arm now tries the
   tree-walk interpreter FIRST (unchanged — monomorphic calls and AGGREGATES still fold on it, no
   regression), and only when the interpreter reports a CAPABILITY gap (its "…isn't supported yet"
   family: sizeof/alignof/offsetof, a generic call, a string, a bitfield op, a fn pointer, an
   aggregate-const ref, a trait call) does it route THAT site through the compiled engine via a
   registered hook (`ct-fold-hook`, wired by `main`/`main_a64`'s `register-comptime-fold!`, exactly
   like `set-meta-build-obj!`). A genuine SEMANTIC error (division by zero, an arithmetic type
   mismatch) is left to stand — the discriminator is the interpreter's own "supported yet" wording,
   so the compiled engine never masks a real bug with a target-specific value. The engine
   (`comptime_eval.coil`) recovers the site's checked type (type map by nid), builds a MINIMAL closure
   sub-program of what E calls plus a synthetic `(defn coil.ct.thunk [] (-> T) E)` exported as C
   symbol `coil_ct_thunk`, monomorphizes + builds + dlopens it (its OWN raw build, no metashim
   handshake — comptime has no code ops), runs the entry, and reads the SCALAR result (int/bool/f64)
   back out of the return register, rebuilding a literal via `build-value`. It DECLINES (→ interp
   error stands) for an aggregate/string/f32 result, so aggregate comptime keeps folding on the
   interpreter (no regression). Both compilers do this (the LLVM-free one via its arm64 engine — the
   step-2 export-c is exactly what unblocks it, verified: sizeof/generic fold with NO libLLVM linked).
   Fixpoint held on BOTH rebootstraps + all 7 gates green: the compiler's own source has no comptime
   sites, so the hook never fires during the self-build and gate-full is byte-identical; only the two
   diag build-fixtures 06/07 flipped from "error" to a folded literal (regenerated, exit 1→0). Five
   teeth in `gate-cli.sh` (sizeof→8, generic→7, `(const … (sizeof))`→8, c-string-in-comptime→5, and an
   aggregate-comptime no-regression check) — the four capability tests FAIL on the pre-mac-8 seed.
   STEP 3 STILL OPEN (the actual deletion): needs (a) aggregate comptime ON the compiled engine — the
   scalar readback here is register-only; an aggregate needs a native-memory readback (a write-through
   `(ptr T)` entry + the C layout to walk field offsets) so deleting the interpreter doesn't regress
   it; (b) re-routing `run-metas` (the `(meta …)` path still uses `eval`) and `finish-macro`'s
   interpreter fallback off `eval`; (c) relocating `CtVal`/`CtCtx` (metaengine/metalower/metahost reuse
   them) out of `comptime.coil` before its evaluator, the `COIL_META` flag, `parity.sh`, and
   guide.coil:426 can be removed. mac-8 is closed; mac-12 + diag-4 wait on the deletion.

8. **String HashMap keys own by default (std-3).** ✅ DONE. String-keyed maps copy keys into the
   map's allocator on insert and free on remove; borrowing becomes the opt-in/unsafe path.
   OUTCOME: ownership is a capability on the `KeyOps` vtable, not a compiler concept. `KeyOps` gained
   three fields — `copy`/`free` (both `(ptr Allocator) (ptr i8) -> i64`, acting IN PLACE on the key
   slot) and an `owns` flag. When `owns=1` the map deep-copies each genuinely-inserted key into its
   own allocator and frees it on remove/clear/free, so a `(slice u8)` key never aliases caller
   storage (the footgun: two keys built over one reused buffer both read its latest bytes). The
   copy is gated at the ONE place a key enters a slot (`hm-insert-raw`, new `own`/`alloc` params) by
   `own AND ko-owns?`: a rehash MOVE (`hm-grow!`, `own=0`) relocates the owned fat pointer without
   re-copying or freeing, and `hm-remove!`/`hm-drop-keys!` (walked by `hm-free!`/`hm-clear!`, only
   when `ko-owns?`) free exactly the state-1 keys — no double-free of a tombstone, no leak.
   `str-keyops` (lib/str.coil) is now the owning default (`str-key-copy` = raw-alloc+mem-copy+
   slice-new; `str-key-free` = raw-free; empty keys own nothing and are skipped); `str-keyops-
   borrowed` is the deliberate opt-in that leaves the hooks zero. Scalar/derive/struct keyops leave
   `owns` at the `alloc-static` zero-init default (borrow — correct, since their value carries no
   external storage) and pay nothing: the guards are dead branches. AUDIT of existing string-key
   users (self-host `driver.cheader`/`cimport`, examples/json·lisp·calc, lib/sexp): all key off
   freshly-allocated bytes or static literals and only put/get, so owning is a pure superset — every
   corpus program's stdout+exit is byte-identical (owning changes memory ownership, not results),
   and none of them run during the self-build (cimport/cheader are CLI subcommands), so the fixpoint
   is untouched. Only the `full` IR stage regenerated (7 hashmap/str/KeyOps-touching refs; the KeyOps
   struct type grew `{ptr,ptr,i64}` → `{ptr,ptr,i64,ptr,ptr,i64}` and the owning hooks thread
   through the monomorphized maps); ast/resolved/checked/mono/diag stayed byte-identical. Fixpoint +
   all 7 mandated gates + arm64 gate-run green. Teeth in gate-cli.sh (owning→122 as "alpha" survives
   a reused-buffer overwrite; borrowed→22 reproducing the aliasing on purpose) — both FAIL on the
   pre-std-3 seed (borrow → 22; `str-keyops-borrowed` undefined). guide/LANGUAGE_GUIDE updated.

9. **Debug-checks build mode — BOTH (mem-2 · mem-6 · mem-7 · mem-8).** ✅ DONE.
   (a) `--debug-checks`: a comptime predicate exposed to library code so slice-get/subslice
       bounds-check (and reject negative-length subslice), a poison-on-free debug-allocator in
       lib/alloc.coil, and a stack-return lint, all zero-cost when off.
   (b) `--sanitize=address`: wire LLVM's ASan pass + link the runtime; and stop
       `--link-flag -fsanitize=address` aborting the compiler.
   OUTCOME: `(debug-checks?)` is a new comptime code op (39), answered in BOTH engines (the
   interpreter's code-op + the compiled engine's metahost callback) from a host cell the driver
   flips before expansion — like `(target-arch)`, so it needs NO seed refresh: the seed builds
   stage1 against its OLD bundled libs and include-str bakes the new ones as opaque strings; from
   stage1 on the op is known. Every check lives inside a MACRO branched on `(debug-checks?)` at
   expansion time, so with the flag off it expands to the exact unchecked form — every off-path IR
   is byte-identical (fixpoint held on BOTH rebootstraps, all 7 gates green, ZERO snapshot regen).
   • **mem-6**: `slice-get`/`slice-set!`/`subslice` bounds-check (and `subslice` rejects `lo>hi`,
     the length-(-2) invariant break). The panic helper is GENERIC (T inferred from the slice) so
     it is emitted only when instantiated — off, nothing calls it and no byte reaches the output;
     slice.coil gained an `(export …)` list so its private `write` extern can't collide with io's.
   • **mem-2**: `(debug-allocator inner)` (a NEW bundled module lib/dbgalloc.coil — kept out of the
     always-loaded alloc.coil, whose concrete hooks mono can't dead-strip) — a macro that is `inner`
     when off, and when on wraps with a `{magic,size}`-header allocator that detects double-free
     (located abort) and poisons freed payloads to 0xDE (quarantining blocks so detection is
     reliable). Zero effect on existing dumps.
   • **mem-8**: a bundled `(checker …)` (lib/stacklint.coil) the driver auto-`--use`s under
     --debug-checks; it WARNS (like clang, non-fatal) when a user function returns a pointer to a
     stack local (direct or let-bound alloc-stack). No false positives on heap/param returns or the
     corpus. LESSON: the checker-dylib closure walk skips a user call nested in an expanded and/or.
   • **mem-7**: `--sanitize=address` runs LLVM's AddressSanitizer pass (`default<O3>,asan`) on the
     PROGRAM object only (`build-object` sanitize flag; false for the metaprogram dylib). Sanitizer
     flags are filtered off the metaprogram-dylib link (`filter-meta-link-flags`), so a bare
     `--link-flag -fsanitize=address` no longer aborts the compiler; `--sanitize=address` on a
     non-LLVM backend is a clear located error. (The instrumented object is provably ASan-marked; the
     final exe-link's runtime must match the instrumenting LLVM — a toolchain concern, delegated to
     `cc -fsanitize=address`.)
   Teeth for all four in `gate-cli.sh` (all FAIL on the seed).

10. **Testing & debugging story (tool-12 · tool-11).** ✅ DONE. `lib/assert.coil` (assert/assert-eq with
    file:line via the span machinery), a `coil test` runner that discovers + runs test mains,
    a `deftest` macro as a pure library (per the macro-power thesis), AND fix `-g`: run
    dsymutil, keep the .o, and fix the arm64 line-program so lldb maps source (currently 0 line
    rows / "No source available").
    OUTCOME (two independently-green sub-steps):
    (a) **Testing (tool-12).** `lib/assert.coil` is a bundled library, entirely macros + one
        `(transform …)` — no compiler builtin. `(assert COND)` / `(assert-eq A B)` / `(assert-ne A B)`
        check at runtime and, on failure, print the OFFENDING EXPRESSION and its `file:line` then
        abort (SIGABRT) like C's assert(). To recover the expression + location at expansion time,
        two new comptime code ops were added (the trivial `code-file` shape, in the parser +
        `codeop-rty` + the interpreter's `code-op`, which the compiled engine reuses via metahost;
        metalower is generic over ops): `code-line` (op 40 → i64) and `code-src` (op 41 → the source
        substring of a node's span). `(deftest NAME body…)` expands to a zero-arg `coil-test$NAME` fn;
        the `(transform …)` discovers every such fn and synthesizes a `main` that runs each in a
        FORKED child (io is unbuffered, so an aborting test's message survives; a crash in one test
        never stops the suite). `coil test FILE` (driver) auto-`--use`s assert.coil, so a test file
        needs NO import; exit 0 iff all pass. Purely additive to the compiler (no corpus file uses the
        ops) — fixpoint byte-identical, all 7 gates green, zero snapshot regen.
    (b) **-g / DWARF (tool-11).** The arm64 backend emitted DWARF addresses with NO relocations, so
        `dsymutil` rejected the object ("No valid relocations found. Skipping.") → an EMPTY `.dSYM`,
        0 line rows, "No source available" in lldb; and the driver never ran `dsymutil` at all. Fixed
        to clang's exact Mach-O scheme: every DWARF address (CU + each subprogram `low_pc`, the line
        program's `set_address`) now carries an 8-byte UNSIGNED **section** relocation to `__text`
        with the offset as the addend (`dwarf.coil::dw-reloc!` populating the reloc list; codegen_a64
        emitting them; a new SECTION-reloc path in `macho.coil`'s writer, gated on a negative-sym
        sentinel so every existing extern reloc is byte-identical). The driver runs `dsymutil` after a
        `-g` link and keeps the `.o`. lldb now maps source from the `.dSYM` alone — line breakpoints
        resolve and hit at `file:line`. The arm64 self-build (which emits DWARF) reproduced
        byte-identical (fixpoint held) since the compiler is built without `-g`… the DWARF relocs it
        does emit are deterministic. Teeth for both halves in `gate-cli.sh`; all FAIL on the seed.
        FOLLOW-ON: a function-NAME breakpoint still needs the module-qualified name (`dbg.add`); line
        breakpoints (the primary workflow) are unaffected.

## Notes carried from the session
- The mechanical rough-edge fixes are DONE (~60 findings across many commits + two workflows).
- diag-9 is half done (body/return spanned; param/field/return needs spans on AST type nodes).
- gen-10 (mono O(n^1.7)→O(n)) is a deferred workflow item worth reviving as focused work.
- The debug-checks mode (#9) is the landing spot lib/slice.coil's own "Phase-2" comment promises.
