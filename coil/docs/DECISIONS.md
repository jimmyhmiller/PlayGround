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

3. **Qualified trait-method calls `(A::go x)` (gen-6).** Add `::`-qualified method calls that
   resolve `Trait::method` through the receiver's impls table; a same-name collision becomes
   recoverable. Fix the collision error to stop suggesting `:use` when both traits are local.

4. **Supertrait `(deftrait Derived [Self] :requires [Base] …)` (gen-8).** A `:requires` clause,
   separate from the type-param vector, so associated-type params and supertraits stop sharing
   one syntax. Warn (or error) on the old ambiguous form.

5. **Iteration & generics over collections — BOTH (gen-1 · std-11 · std-4).** The biggest item.
   (a) Associated-type bounds: make parameterized traits (Get/Set/Push/Pop/Iter) usable as
   bounds by treating non-Self params as associated types determined by the impl. (b) An
   Iterator/Iterable protocol collapsing hm-for/al-for/slice-for/for-in into one `(for x (iter
   coll))`. Sequence: bounds FIRST (unblocks generic collection code), iterator SECOND. std-4
   (`(for-in (in map))` iterates garbage) is folded in — it goes away with the real protocol.

6. **File-relative imports + migrate the self-host source (tool-1).** Resolve a relative import
   against `dirname(importing file)` (the documented, universal rule). The self-host source is
   written CWD/root-relative, so rewrite its imports too. NOTE from the prior attempt: a
   file-relative switch changed `emit-ir` output in a way I couldn't explain and broke the
   bootstrap's own imports — run that down as part of this (it's why a naive switch failed).

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
