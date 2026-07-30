# Coil — Rough Edges: the fix list

79 findings from six agents that each spent an hour writing real Coil against the
self-hosted compiler at HEAD. Every one is backed by a repro that was actually run;
verbatim output lives in the triage export. 78 are accepted for fixing.

**The through-line, found independently in all six domains:** the typechecker refuses
to let you be wrong, and everything wrapped around it refuses to tell you that you are.
The single largest cluster (19 findings) is *compiles clean, exits 0, silently wrong*.

This list is ordered by **leverage, not severity** — several findings collapse into one
root cause, and one small fix at the bottom of the stack makes the rest observable.

---

## Decisions taken (Jimmy, on triage)

- **gen-6** (trait methods are global; collision unfixable) — ✅ DONE. A `Trait::method` call
  head (`(A::go x)`) now pins dispatch to the named trait, so a same-name collision is
  recoverable. The two collision errors name the candidate traits and suggest the qualified
  form instead of a misleading `:use`. Teeth-tested in `gate-cli.sh`; see DECISIONS.md #3.
- **gen-8** (supertrait syntax silently means something else) — ✅ DONE. `deftrait` now takes a
  separate `:requires [Base …]` clause; every impl of the trait must also impl each `Base` (a new
  `check-supertraits` pass enforces it, order-independent). A trait name smuggled into the
  `[Self …]` vector is now a located error pointing at `:requires`, instead of being silently
  swallowed as an associated-type parameter. Teeth-tested in `gate-cli.sh`; see DECISIONS.md #4.
- **gen-9** — ✅ DONE. Added type ascription `(: value type)` (a general checked-annotation
  expression that flows the expected type in, not a cast) **and** fixed the let-inference hole
  (a returned `let` binding now takes the return type, so `(let [r (Okk 5)] r)` infers without
  ceremony). Both teeth-tested in `gate-cli.sh`.
- **std-12** — `store!` should return **unit**.
- **gen-1 / std-11 / std-4** — ✅ DONE. Two green sub-steps: (a) parameterized traits are now
  usable as bounds — their non-Self params are associated types determined by the impl (the
  deferred trait-call path substitutes synthetic associated types the body checks against; mono
  resolves them off the receiver's impl). (b) One `Iterator`/`Iterable` protocol (prelude
  traits + SliceIter/MapIter impls) drives `(for x (iter coll))` over slice/list/map;
  `slice-for`/`al-for` collapse to aliases over it, and `(in map)` now iterates keys correctly
  (std-4). Teeth in `gate-cli.sh`; see DECISIONS.md #5.
- **The interpreter — DELETE IT. Decided. Scheduled AFTER everything else on this list.**
  Verdict: it is a second implementation of the language semantics that is silently weaker
  than the first — the "ONE definition per concept" violation, sitting in the metaprogram layer.
  It is *not* a backend; it is a tree-walking evaluator (`comptime.coil`). Backend and engine
  are orthogonal: `--backend arm64` + the **compiled** engine works fine (verified: the
  ArrayList/malloc macro returns 15), while either backend + interp fails.

  Two things still depend on it, so it cannot simply be removed:
  1. **The LLVM-free bootstrap.** `codegen_a64.coil:2896` — `export-c is not supported by the
     arm64 backend yet` → `main_a64` registers no object builder → cannot build a metaprogram
     dylib → interp is its only engine. Delete interp today and `rebootstrap-nollvm.sh` dies at
     stage 2 unable to expand `cond`. (`seed/coil-seed-nollvm` is committed and gated — real path.)
  2. **`(comptime E)` and `(const …)` folding — in EVERY build, including the default
     compiled-engine one.** Verified: with `COIL_META=compiled` *forced*,
     `(comptime (id [i64] 7))` still fails `comptime: generic call … isn't supported yet`, a
     `comptime.coil` message. The compiled engine only ever took over **macro call sites**.
     This is the true cause of **mac-8**: `comptime` and a macro body are not one phase run two
     ways, they are two evaluators, and the weaker one owns `comptime`/`const`.

  Order (each step green on its own; the deletion cannot come first):
  1. Route `(comptime E)`/`(const …)` through the compiled engine → closes **mac-8**. ✅ DONE
     (interp-first at `fold-expr`'s `EComptime` arm; a "…isn't supported yet" CAPABILITY gap —
     sizeof/generics/strings — routes to a registered compiled-engine hook that builds a minimal
     closure sub-program + a `() -> T` entry, runs it, and reads the scalar back; aggregates stay
     on the interpreter — no regression. See DECISIONS.md #7 step 1.)
  2. ✅ DONE. `export-c` in the arm64 backend → `main_a64` registers a builder → the LLVM-free
     compiler builds metaprogram dylibs with the arm64 backend and defaults to the compiled
     engine (it stops being secretly weaker). AAPCS64-native, so no C-ABI thunk is needed
     except for a by-value struct param, which is a clear error. See DECISIONS.md #7 step 2.
  3. Delete `comptime.coil`'s evaluator, the `COIL_META` flag, `parity.sh`, and guide.coil:426.
     mac-8, mac-12 and the diag-4 dual-engine problem all vanish rather than get documented.
     [OPEN — needs step 1 first; note metaengine/metalower/metahost REUSE comptime's CtVal/CtCtx
     types, so the deletion must relocate those, not just drop the file.]

- **NEW (found while proving the above, worse than gen-7 reported):** `(const FIVE (fact 5))`
  where `fact` is an ordinary **monomorphic** recursive function fails with
  `call to undefined function 'ct.fact'`. Not a generics limitation — `const` appears unable to
  call *any* user-defined function, which guts the compile-time-lookup-table idiom the docs sell.
  Folded into step 1 above (the compiled engine can already do this).

---

## Batch 0 — Observability first (unblocks verifying everything else)

- [x] **mem-1** `coil run` reports EXIT=0 for a program that dies by signal. The documented
      dev loop discards the only signal an unsafe-by-design language ever produces, and every
      example's `echo $?` convention depends on it. Inspect the child's wait status; if
      `WIFSIGNALED`, print `program terminated by signal N (SIGABRT)` and exit `128+WTERMSIG`.
      *Do this first — mem-2, mem-4, std-1 and the whole memory domain are invisible until it lands.*

## Batch 1 — One file-read fix closes four findings

`loader.coil:21 read-file` turns ENOENT into `""`. **`read-file-opt` (loader.coil:36) already
exists and its comment already states this exact rationale** — these are call sites not using it.

- [x] **tool-2** A nonexistent source file compiles as empty → ld error about `_main`.
- [x] **diag-2** Same, via `coil build` — the first wall a user can hit, pointing 100% wrong.
- [x] **tool-3** `coil fmt --write ghost.coil` **fabricates** the file; `--check` claims a
      missing file is "not formatted". A formatter must never invent a file.
- [x] Coil.toml `entry = "src/nope.coil"` → same linker error instead of "entry not found".
- [x] Make `--check` exit 2 on error (vs 1 unformatted, 0 clean) — the gofmt/black convention.

## Batch 2 — One argv fix closes four more

`driver.coil:542` does `file (load (index argv 2))`, and `manifest-mode?` (955) keys off
argv[2] starting with `-`. Scan argv for the first non-flag positional; consume known flags
as pairs; **reject unknown flags** instead of skipping them.

- [x] **tool-5** `coil build -o out in.coil` (universal Unix order) → "no Coil.toml", a message
      that never mentions the file the user passed and advises what they just did.
- [x] **tool-4** Project mode silently discards **every** flag: `--target wasm32-unknown-unknown`
      emits a native Mach-O and exits 0; `-o` is ignored; a bogus triple builds fine.
      Route project and single-file mode through **one** options struct and one entry point.
- [x] **tool-13** Unknown flags silently accepted; no per-subcommand `--help`; missing `-o`
      aborts with SIGABRT (134) instead of exit 1.
- [x] **tool-14** `coil fmt a.coil b.coil` silently formats only the first; a directory no-ops.
- [x] Add a `coil build --target wasm32` **project** test to the gate corpus.

- [x] **xcompile (NEW, found while fixing tool-4):** cross-compiling to a non-host *native*
      triple (`--target x86_64-apple-macosx11.0.0`) emitted a correct x86_64 object and then
      linked it with the **host arm64** `cc`, failing with "found architecture 'x86_64',
      required arm64". Fixed by passing `-arch <arch>` to the link step (`build-cmd`), so macOS
      `cc` cross-links every slice its SDK carries — the build now succeeds and produces a real
      x86_64 executable (the seed's earlier stopgap merely rejected it). wasm/host builds are
      untouched. Teeth in gate-cli ("cross-target build links an x86_64 Mach-O executable").

## Batch 3 — `alloc-static` for per-instance state (one bug, three sites)

Found independently by two agents from opposite directions. `alloc-static` is one cell **per
call site**, not per call, so constructors that use it return the same object every time.
`malloc-allocator` uses it *legitimately* (a true singleton) — copying that pattern is what broke these.

- [x] **std-1 / mem-4** `arena-allocator`: every call returns the same arena; the second arena
      silently destroys the first and leaks its buffer. Carve `Arena`/`Allocator` out of the
      caller's buffer, or take a caller-owned `(ptr Arena)`.
- [x] **std-2** `fixed-buffer-writer` shares one static `Writer` — two live writers merge.
      Same fix; also `null-writer`/`fd-writer`.
- [x] Audit every `alloc-static` in a constructor position across `lib/`.

- [x] **NEW (found while fixing Batch 3): `:lower shim` could not build on the LLVM backend
      AT ALL.** No `AsmParser` was ever initialized, so the inline asm a naked trampoline
      lowers to made LLVM hard-error and abort — killing the language's headline
      calling-convention-as-a-type feature on the default backend. `examples/shim.coil`, a
      committed example, did not build. Ungated because `gate-full` stops at emit-ir and
      `arm64/gate-run.sh` only exercises `--backend arm64` (which bypasses LLVM's asm
      printer and worked fine). Fixed + now covered by gate-cli.
- [x] **NEW: `stdin` aliased any other `fd-reader`** — the same static-sharing io.coil had
      already patched for stdout/stderr, still live for stdin. Fixed by the same change.

## Batch 4 — Dies with zero output (no depth/fuel guards)

docs/COMPTIME.md promises "a fuel budget bounds runaway loops/recursion". The **loop** half is
real; the **recursion** half does not exist, and neither does any depth guard on generated code.

- [x] **mac-5** No stack-depth guard anywhere: an 800-clause stdlib `cond` and a 900-field
      `derive-eq` **segfault the compiler** with zero output. Cliff bisected to ~400-420 nodes.
      This caps the practical size of every generated construct — in a compiler whose purpose
      is generating code. Add a depth check in the recursive walks; long term move hot passes
      to explicit worklists so size is bounded by heap, not the C stack.
- [~] **diag-4** Non-terminating `comptime` → SIGBUS, zero bytes out. **MOVED to the
      interpreter project — do NOT patch here.** A call-depth counter beside the existing
      `fuel` field would be work that exists to be deleted: `fuel` only exists *because*
      `comptime` runs on the interpreter, and step 1 (route `comptime`/`const` through the
      compiled engine) makes the whole mechanism dead code. Patching it also entrenches the
      evaluator we intend to remove.
      The fix is also *different* than it looks: once `comptime` is native code, a divergent
      `comptime` is a hang or a native stack overflow — precisely what a runaway **macro**
      already is today. That is ONE problem ("a compile-time program diverges") needing ONE
      answer (watchdog / timeout / stack guard) that covers macros and `comptime` uniformly,
      not a special case inside a dying interpreter.
- [x] **gen-3** Infinitely-recursive monomorphization hangs forever, zero output, no limit.
- [x] **diag-3** Imported file missing `(module …)` → SIGABRT without naming the file.
      The loader has the path in hand.

- [x] **NEW (found while fixing mac-5): the span renderer prints the ENTIRE source line.**
      An error on a macro-generated line dumped 92 KB of output — the whole 40 KB line, twice
      (source + caret rule). Truncate to a window around the span with ellipses. This makes
      every diagnostic on generated or merely long lines unreadable, and it is why the new
      depth-limit error, while correct, is unusable as printed.
- [x] **NEW: `expand` and the dump-* commands run on the MAIN thread (8 MiB)**, not the
      512 MiB pipeline thread, so `coil expand` still segfaults on deeply nested input that
      `coil build` now handles. Route them through run-on-big-stack too.
      FIXED (b4d497018): `driver-main` routes `expand` and every `dump-*` (plus `check`)
      through `run-dump-on-big-stack` -> `run-on-big-stack`, the identical 512 MiB pthread
      `build` uses (driver.coil). Regression guard in gate-cli.sh runs `dump-read`/`dump-ast`
      on 40000-deep nesting that overflows an 8 MiB stack but fits 512 MiB.

## Batch 5 — The good renderer exists; these call sites route around it

Coil's span renderer is genuinely excellent. Every finding here is a site bypassing it.

- [x] **diag-1** Adding `(module …)` — which every multi-file project must do — **strips the span**
      off `call to undefined function`, the most common error in the language. Proven by deleting
      one line from a byte-identical file.
- [~] **diag-9** PARTIAL. The **body/return mismatch** now points at the offending tail
      expression (the case that mattered: "which of a 60-line body's expressions is the tail?").
      The **unknown type in a param/field/return** half is NOT fixed and needs an AST change:
      `Param`, `Field` and `Type` carry **no span at all** (ast.coil:33/35), so there is nothing
      to point at. The parser must stamp spans on type nodes first — real work, not a call-site fix.
- [x] **mac-2** `error` in a macro is unlocated — while `report` is perfect and undocumented for
      macros. The whole stdlib uses `error`. Give `error` the call-site span; document `report`.
- [x] **mac-7** Macro arity errors unlocated and countless, while the *identical* ordinary-function
      error is perfect — directly contradicting "a macro is just a function".
- [x] **mac-9** Resolve errors in generated code get no expansion note; type errors get a great one.
- [x] **gen-11** `derive` over a generic type fails with a leaked mangled internal name, no span.
- [x] **diag-8** One resolve error suppresses every typecheck error; the survivor is the unspanned one.

## Batch 6 — Bounds are parsed and thrown away

- [x] **gen-4** `defstruct` bounds silently ignored — **including bounds naming traits that don't
      exist**. `defsum` rejects the same syntax; `defn` honours it. One spelling, three meanings.
- [x] **gen-12** A bound naming a nonexistent trait is diagnosed at the *call site* as
      "i64 does not implement NoSuchTrait" — no type could ever satisfy it.
- [x] Resolve trait names in **every** bound position (defn/defstruct/defsum/impl) via one shared
      helper. Per the repo's own "ONE definition per concept" rule: delegate, don't copy.
      ✅ DONE. Extracted `qualify-trait-name` (singular) in resolve.coil as THE resolver for a
      bound-position trait: `resolve … 2` then reject a dotless result as `unknown trait '…' in
      bound`. `qualify-trait-names` (plural, used by defn/defstruct/defsum bounds and `:requires`)
      now loops over it, and `qualify-one-impl` routes the impl's `trait_name` through it too —
      previously the impl trait resolved leniently and an unknown one was caught late in the
      checker (`setup-one-impl`) with a divergent, unspanned `impl: unknown trait '…'`. All four
      positions (plus impling a declared non-trait) now emit the identical located error. Teeth in
      `gate-cli.sh` (impl over an unknown trait / a non-trait name — both FAIL on the seed's old
      message). Behaviour-only: gate-full stays byte-identical.

## Batch 7 — Silently wrong answers

- [x] **mac-1** `(target-arch)` is a hardcoded `"aarch64"` **string literal** (comptime.coil:681).
      The only reason it exists is to branch when target ≠ host, and that is exactly and only when
      it lies. Thread the resolved triple into CtCtx. Until then it must **hard-error**, not fabricate.
- [x] **tool-6** REPL truncates f64 to i64: `(/ 1.0 2.0)` prints `0`. The REPL is the thing you reach
      for when you're unsure; it is the one tool in the toolchain that gives wrong answers.
- [x] **tool-10** Colliding `:use *` names resolve first-import-wins, silently — reordering two
      import lines changes which function runs. NAMESPACING.md already specifies the strict rule
      for trait methods; apply it to function refers and reuse the error text.
- [x] **mem-3** A bare `(mut x)` passes its **address** into a variadic C call while auto-dereffing
      everywhere else. Reject it (the metal-op error is the right precedent).
- [x] **std-3** String-keyed HashMap never copies keys — produced a map with two keys both reading
      "gamma". ✅ DONE: string keys now OWN by default. `KeyOps` gained `copy`/`free`/`owns`; a map
      with `owns=1` deep-copies each inserted key into its own allocator (`str-key-copy`) and frees
      it on remove/clear/free (`str-key-free`, `hm-drop-keys!`) — a key never aliases caller memory.
      Rehash MOVES keys (no re-copy/free). `str-keyops` is owning; `str-keyops-borrowed` is the
      opt-in unsafe borrow. Scalar/derive/struct ops leave the hooks zero (borrow) — no cost. See
      DECISIONS.md #8; teeth in gate-cli.sh (owning→122, borrowed→22, both FAIL on the seed).
- [x] **std-4** `(for-in [x (in map)])` compiled and iterated garbage (`get`-by-slot-index on a
      map, whose `get` takes a KEY). ✅ DONE via the real iterator protocol: `(in map)` / `(iter
      map)` now iterate the map's KEYS through `MapIter` (recover the value with `hm-get`). See
      DECISIONS.md #5.
- [x] **std-10** fmt `{x}` routes to signed `print-hex` and emits a garbage byte for negatives;
      `print-uhex` is unreachable from fmt. Point `x` at `print-uhex`.
- [x] **gen-5** Blanket `(impl [T] Show T)` accepted, silently does nothing. Reject it, or support it
      (a bare `T` is just the bottom of the specialization lattice that already exists).
- [x] **diag-10** `:use [name]` with a name the module doesn't export is silently accepted.
      ✅ DONE: `resolve-program` now runs `check-use-exports` eagerly at the import site (right
      after the tool-10 dup check), consulting the SAME `exports-check` export table the resolver
      itself uses. A `:use [name]` whose target module has an explicit `(export …)` that omits the
      name is a located error — `in module 'app': :use names 'secret', which module 'util' does
      not export` — instead of the old silent rc=0. Open modules (no export list) and exported
      consts still resolve. Regression in gate-cli.sh (FAILS on the seed, PASSES on the build).
- [x] **mac-4** Binder hygiene: macros capture silently (200 instead of 105) while free identifiers
      *are* hygienic. Half-hygiene is worse than none for the user's model. Audit `lib/derive.coil`
      and `lib/match.coil` templates for latent capture (`h`, `a`, `b`, `x`).

## Batch 8 — `store!` returns unit *(Jimmy: "probably should be unit")*

- [x] **std-12** `store!` yields the stored value's type, so effect-only `if` mismatches constantly.
      A documented gotcha is a gotcha that fires. Making it unit may ripple — expect churn in
      `lib/` and examples. DONE: `store!` now yields unit (canonical `i64` 0) in check + both
      codegen backends + the comptime evaluator. The tree-wide ripple was tiny (the codebase
      already wrapped non-`i64` stores): only 3 self-host sites (`reader.coil` ×2, `parser.coil`
      ×1) used the `(store! f true)`/`false` bool-flag shape and needed the `(do (store! …) 0)`
      wrap; `lib/`, `examples/`, and `apps/` had zero ripple. Regression in `gate-cli.sh`.

## Batch 9 — Docs that promise what doesn't exist

Not staleness: specific guarantees the docs state and the implementation never had.

- [x] **mem-5** LAYOUT.md's "Verification (checked total control)" — `:size` is padded-to but never
      asserted (`:size 8` silently yields 40), and overlap at unequal `:at` is accepted and clobbers.
      Its header says "tiers 1-3 done and tested". Implement both checks (`static-assert` already
      renders the error) or correct the doc.
- [x] **tool-1** — ✅ DONE. Relative imports now resolve against `dirname(importing file)` (the
      documented rule) instead of the CWD, so the `src/main.coil` layout `coil new` scaffolds can
      import a sibling and multi-file apps (chip8/invaders) build from any directory. The prelude and
      bundled libs resolve their OWN imports against a `<bundled>` sentinel so a demo like
      `examples/io.coil` can't shadow the library (the cause of the prior attempt's "unexplained
      emit-ir change"); the self-host entry files were migrated to bare-name imports and both seeds
      refreshed so the bootstrap understands the new rule. Teeth-tested in `gate-cli.sh`; see
      DECISIONS.md #6.
- [x] **mem-11** The guide's loudest memory warning (never `alloc-stack` in a loop → "eventually
      segfaults") does not reproduce on either backend; static-size allocas get hoisted. Hoist in
      Coil's own lowering and delete the warning, or show the shape that actually breaks.
- [x] **std-14** fmt.coil's header says the format-string front end isn't implemented — it's 190
      lines below, working.
- [x] **std-9** control.coil's `case` doc names `(derive Eq …)`, which doesn't exist; the macro is
      `derive-eq` and it emits a free function, not an impl.
- [x] **tool-12** — ✅ DONE. The Coil.toml reader is now STRICT: it knows exactly five sections
      (`package`/`build`/`link`/`cc`/`run`) and a fixed key set per section. An unknown section
      (e.g. `[dependencies]`, which used to be silently accepted, inviting the false inference that
      deps resolve) or a typo'd key (e.g. `entrypoint` for `entry`, previously swallowed) is a
      LOCATED hard error naming the offending Coil.toml line, section, and key — not a silent no-op.
      See `parse-manifest` in `driver.coil` (`m-known-section?` / `m-bad-key` / `m-err`); teeth in
      `gate-cli.sh` (all FAIL on the seed, PASS on the build).
- [x] **mem-9** `(ref T)` is a working, undocumented third reference spelling — and the only way to
      write a read-only reference to a scalar. Document it or reject it in source.
- [x] **mem-10** "Immutable reference" is C's `const T*`, not Rust's `&T` — say "read-only".
- [x] **mac-11** `bytes->str`/`str-bytes` are undocumented but are the only int→name path a generator
      has. Document; add `int->str`.
- [x] **mac-8** — ✅ DONE. `(comptime E)` / non-literal `(const …)` sites the tree-walk interpreter
      rejects for a CAPABILITY gap (generics, strings, sizeof/alignof/offsetof, bitfields, fn pointers,
      aggregate-const refs, trait calls) now route through the COMPILED metaprogram engine (real
      codegen), so they fold to a literal exactly like a macro body would evaluate them. Interp-first
      keeps monomorphic + aggregate comptime unchanged (no regression); a semantic error (div by zero)
      still stands. Both compilers (the LLVM-free one via its arm64 export-c engine). See DECISIONS.md
      #7 step 1. (Deleting the interpreter — mac-12, diag-4 — is step 3, still open.)

## Batch 10 — No debug mode for a check to live in

Coil is unsafe by design — legitimate. The finding is that the design's own escape hatch is missing.
`lib/slice.coil`'s own comment promises "debug-mode bounds checking is Phase-2 safety work", but
**there is no debug flag for Phase-2 to land in.**

- [x] **mem-6** — ✅ DONE. `slice-get`/`slice-set!`/`subslice` bounds-check under `--debug-checks`,
      and `subslice` rejects `lo > hi` (the length-(-2) invariant break). The check is in a macro
      gated on `(debug-checks?)`, so it is byte-identical zero-cost when off. See DECISIONS.md #9.
- [x] **mem-7** — ✅ DONE. `--sanitize=address` runs LLVM's AddressSanitizer pass on the program
      object; sanitizer flags are filtered off the metaprogram dylib, so `--link-flag
      -fsanitize=address` no longer aborts the compiler. A non-LLVM backend is a clear error. See
      DECISIONS.md #9.
- [x] **mem-2** — ✅ DONE. `(debug-allocator inner)` (bundled lib/dbgalloc.coil) — a macro that is
      `inner` when off; on, it detects double-free (located abort) and poisons freed memory to 0xDE
      (quarantine so detection is reliable). See DECISIONS.md #9.
- [x] **mem-8 / diag-11** — ✅ DONE. A bundled `(checker …)` (lib/stacklint.coil) the driver
      auto-applies under `--debug-checks` warns (like clang, non-fatal) when a user function returns
      a pointer to a stack local. No false positives on heap/param returns. See DECISIONS.md #9.
- [x] Introduce the debug-checks build mode these all depend on. — ✅ DONE: `(debug-checks?)`, a
      comptime code op set from the driver, gating each library check inside a macro (zero-cost off).

## Batch 11 — Stdlib gaps (a 15-line CSV program needed 36 lines of stdlib first)

- [x] **std-5** No file IO whatsoever — reading a file means hand-declaring libc `open` and hardcoding
      `O_RDONLY=0`. Add `lib/fs.coil` (~60 lines; the difference between a demo and a usable language).
- [x] **std-7** `lib/result.coil` is a **one-line empty module** — zero Option/Result combinators,
      while every collection and allocation call returns Option. Fill it in.
- [x] **std-13** No sort anywhere; no `str-split`/`trim`/`parse-int`; no `al-clear!`/`hm-keys`;
      `hm-put!` throws away the insert/update signal it already computes. Priority: sort first.
- [x] **std-8** Strings have no `Eq`/`Ord`/`Hash` impl, so `=`, `case`, `<` and sorting don't work on
      them — though `str-eq`/`str-hash` are sitting right there. Three impls.
- [x] **std-6** Reader has `read-some` but no `read-all` (Writer has both); no cstr↔slice bridge.
- [x] **std-15** `al-slice`/`sb-str` views become use-after-free on the next push, returning plausible
      wrong data. Add an owning `sb-finish!`.
- [x] **std-9** `derive-eq` should also emit `(impl Eq T)` — lights up case/=/HashMap keys at once.

## Batch 12 — Diagnostic quality (the typechecker is already the best part)

- [x] **diag-7** Non-exhaustive match counts missing variants but never names them — the compiler
      computed the set to produce the count. Cheapest large win available.
- [x] **diag-6** No "did you mean", and no "you forgot to import" hint — *even though the stdlib is
      bundled inside the compiler and its export tables are a static lookup*.
- [x] **diag-5** Shadowing a builtin (`call`, `block`) errors about a form the user never wrote.
      The guide ships a gotcha section for exactly this; check names at definition instead.
- [x] **diag-13** Argument type mismatches underline the whole call, not the offending argument.
- [x] **diag-12** `-o /dev/null` aborts with an opaque LLVM message; add a real `--check` mode.
- [x] **mac-3** `coil expand` — the only macro debugger — prints an internal `(error@D:D:D:0 …)`
      node to **stdout** and **exits 0** on failure. The tool you reach for when a macro breaks is
      the tool that lies.
- [x] **mac-6** `code-symbol` rejects the generic instantiations `code-field-count` accepts, so
      derive-over-a-generic is impossible — half the advertised generic-reflection feature.
- [x] **mac-10** Macro output can't feed a literal-consuming macro; the error names an undocumented
      internal (`str-bytes`). Add `code-expand`.
- [x] **mac-12** Engine parity violation: the same helper errors differently from a macro call site
      vs `(meta …)` — `meta` runs on the interpreter, macros on the compiled engine. Undocumented.
- [x] **gen-2** `dyn` + generic impl → `UNIMPLEMENTED` + Abort trap 6. A compiler abort is never OK.
- [x] **gen-7** A generic call in `(const …)` reports "undefined function" for a function defined
      three lines up.
- [x] **gen-10** — ✅ DONE. The mono worklist's three per-instantiation lookups were LINEAR SCANS of
      lists that grow with every instantiation: the dedupe sets (`queued-structs/sums/funcs` via
      `set-contains`), the output insert-or-replace-by-name (`upsert-*` scanning `out-*`), and the
      instantiation-origin map (`io-record!`/`io-find` scanning `io-mangled`) — so mono was ~O(n^2)
      while its IR output is exactly linear. All three are now hash-indexed (`HashMap (slice u8) i64`
      side-indexes: `queued-*` as sets, `out-*-idx` for upsert, `io-index` for origins) → O(1) each.
      BEHAVIOUR IS BYTE-IDENTICAL: the out-* ArrayLists stay the source of truth for order (append on
      first insert, in-place replace after), first-encounter pending order is preserved, and `io-find`
      returns the same index — gate-full stays byte-exact, fixpoint held. Measured on an isolated
      LLVM-backend build: 600→2400 instantiations went from 3.2×/doubling (≈O(n^1.9)) to ≈linear, a
      33× speedup at N=2400 (7.26s → 0.22s). Teeth in `gate-cli.sh` (relative, load-invariant timing
      probe: FAILS on the seed's O(n^2) mono, PASSES here).
- [x] **gen-13** No const generics: an impl can't be generic over array length. Guide advertises
      `(array T N)` as implementable without noting N must be literal.
- [x] **tool-7** REPL is i64-only, can't `println`, and leaks `/tmp/coil-repl-eval.coil` in every error.
- [x] **tool-8** `coil fmt` explodes `(defn f :cc c …)` into one token per line — and the mangled form
      is a **stable fixpoint**, so fmt will never recover it. Mangles `examples/fib.coil`, the repo's
      own hello-world, and calling conventions are the headline feature.
- [x] **tool-9** `coil fmt` disagrees with 43 of 44 repo examples and would rewrite 91% of their lines.
      Fix tool-8 **first**, or a mechanical reformat bakes in the mangled `:cc` signatures.
- [x] **tool-11** — ✅ DONE (line info + .dSYM; the mangled-name half noted below). The arm64 `-g`
      path now stamps `__text`-relative RELOCATIONS on every DWARF address (the CU + each subprogram
      `low_pc`, and the line program's `set_address`), so `dsymutil` stops rejecting the object ("No
      valid relocations found. Skipping." → an EMPTY `.dSYM`, 0 line rows, "No source available") and
      produces a real line table; the driver runs `dsymutil` after a `-g` link (the `.o` is kept for
      it). lldb now maps source from the `.dSYM` alone — line breakpoints resolve and hit at
      `file:line`. Teeth in `gate-cli.sh` (`.dSYM` produced, line rows present, lldb maps source from
      the `.dSYM` with the `.o` removed) — all FAIL on the seed. See DECISIONS.md #10. NOTE: a
      **function-name** breakpoint still needs the module-qualified name (`dbg.add`) — the subprogram
      `DW_AT_name` mirrors the mangled symbol; **line** breakpoints (the primary workflow) need no name.
- [x] **tool-12** (test story) — ✅ DONE. `lib/assert.coil` is a bundled library: `(assert COND)` /
      `(assert-eq A B)` bake the offending expression AND its `file:line` into the emitted code (the
      span machinery, via new `code-src`/`code-line` comptime ops) then abort like C's assert();
      `(deftest NAME body…)` is a macro, and a `(transform …)` discovers every test and synthesizes a
      `main` that runs each in a FORKED child (one failure never stops the suite). `coil test FILE`
      auto-loads it (no import needed). Pure library + macros + reflection, no compiler builtin. See
      DECISIONS.md #10; teeth in `gate-cli.sh` (all FAIL on the seed). NOTE: the *other* tool-12 (strict
      `Coil.toml` parsing / `[dependencies]`, Batch 9) is a separate finding, still open.

---

## Deferred — design required, do not patch piecemeal

- **gen-1 + std-11 + std-4** — ✅ DONE (was *the biggest hole in the language*). Parameterized traits
  are now usable as bounds — their non-Self params are associated types determined by the impl — so
  generic code over any collection works, and there is a real `Iterator`/`Iterable` protocol: one
  `(for x (iter coll))` over slice/list/map, with `slice-for`/`al-for` collapsed to aliases and
  `(in map)` iterating keys correctly (std-4). `lib/arraylist.coil`'s trait impls are no longer dead
  weight — `(C Pop)`, `(I Iterator)`, etc. are real bounds. See DECISIONS.md #5.
- **gen-6** — ✅ DONE. Qualified-call syntax (`A::go`) added — pins dispatch to the named trait;
  same-name collisions are now recoverable. See DECISIONS.md #3.
- **gen-8** — ✅ DONE. Supertrait syntax `(deftrait D [Self] :requires [Base] …)` — separate from
  the `[Self …]` associated-type axis, enforced at impl time. See DECISIONS.md #4.
