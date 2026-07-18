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
   1) Route `(comptime E)`/`(const …)` through the compiled engine (closes mac-8; also fixes
      the "const can't call a monomorphic fn" bug found while proving this).
   2) `export-c` on the arm64 backend → `main_a64` registers a builder → the compiled engine
      becomes the only engine (closes mac-12; the LLVM-free compiler stops being secretly weaker).
   3) Delete `comptime.coil`'s evaluator, the `COIL_META` flag, parity.sh, and guide.coil:426.
   The deletion CANNOT come first — steps 1–2 remove what still depends on it.

8. **String HashMap keys own by default (std-3).** String-keyed maps copy keys into the map's
   allocator on insert and free on remove; borrowing becomes the opt-in/unsafe path. Behavior
   change — audit existing string-key users.

9. **Debug-checks build mode — BOTH (mem-2 · mem-6 · mem-7 · mem-8).**
   (a) `--debug-checks`: a comptime predicate exposed to library code so slice-get/subslice
       bounds-check (and reject negative-length subslice), a poison-on-free debug-allocator in
       lib/alloc.coil, and a stack-return lint, all zero-cost when off.
   (b) `--sanitize=address`: wire LLVM's ASan pass + link the runtime; and stop
       `--link-flag -fsanitize=address` aborting the compiler.

10. **Testing & debugging story (tool-12 · tool-11).** `lib/assert.coil` (assert/assert-eq with
    file:line via the span machinery), a `coil test` runner that discovers + runs test mains,
    a `deftest` macro as a pure library (per the macro-power thesis), AND fix `-g`: run
    dsymutil, keep the .o, and fix the arm64 line-program so lldb maps source (currently 0 line
    rows / "No source available").

## Notes carried from the session
- The mechanical rough-edge fixes are DONE (~60 findings across many commits + two workflows).
- diag-9 is half done (body/return spanned; param/field/return needs spans on AST type nodes).
- gen-10 (mono O(n^1.7)→O(n)) is a deferred workflow item worth reviving as focused work.
- The debug-checks mode (#9) is the landing spot lib/slice.coil's own "Phase-2" comment promises.
