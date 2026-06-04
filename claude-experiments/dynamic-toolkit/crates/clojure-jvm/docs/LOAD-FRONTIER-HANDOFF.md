# Handoff: the upstream-core load frontier

**Status (2026-06-03):** the ENTIRE upstream `clojure/core.clj` now loads —
`load_upstream_core` reports **`processed 719 forms successfully`** (the test
passes, no `#[ignore]`-skip wall left). This started as the form-632 `(load …)`
no-op (SOLVED first) and continued through every subsequent frontier to the end
of the file. The lib suite is 356/0, `load_core_subset` 27/0,
`char_literals_are_characters_not_longs` passes, and the **`CLJVM_GC=every`
stress run completes the full 719-form load with no rooting crash**.

NOTE: "loads" means every top-level form analyzes + evals without a fatal
error. Many forms that exercise host features we don't model (multimethods,
`reify` over JVM interfaces, vars/fns from non-embedded sub-files) compile to
**deferred runtime throw-stubs** or skip registration with a `[cljvm-stub]` /
`[cljvm-deftype]` log — they load but throw a clear error if actually called.
The next work is making those deferrals real (multimethods, the print
subsystem, etc.), not getting core to load.

---

## 1. What the previous handoff got WRONG

The prior theory ("the static `RT/load` call is analyzed in `ctx=Statement` and
its side-effecting call is elided") was a **red herring**. `StaticMethodExpr::emit`
emits the `.call` *before* the `context` match, so the side effect is never
elided. The real chain of bugs was:

1. **`(load "core/protocols")` never reached `RT/load`** because the clojure
   `load` fn computes the sub-file path via `root-directory` → `root-resource`,
   which calls **`.replace` (String, char, char)** — an **unregistered instance
   method**. That threw an analyze-time host-method error which `eval_form`
   caught as a top-level throw, returned nil, and the whole `(load …)` silently
   no-op'd. Every load form completed "Ok" while doing nothing.
2. After registering `.replace`, the path was built but **mangled**:
   `(str (root-directory …) \/ path)` produced `"…47…"` instead of `"…/…"`
   because **characters were modeled as boxed `Long`s**, so `str`/`pr-str`
   rendered `\/` as its codepoint `47`, not `"/"`.

The fix was therefore (a) the missing String interop methods and (b) a **real
`Character` type**, not any analyze/codegen-context change.

## 2. The fixes that landed

- **New `clojure.lang.Character` heap type** (`runtime.rs`): a Raw64
  codepoint cell mirroring `Long`. `Object::Char(u32)`; the reader emits it for
  `\c`/`\newline`/`\uHHHH`; `box_char`/`unbox_char`/`is_boxed_char`;
  `CharExpr` + `PendingLiteral::Char` analyze/emit path; `str` renders the char,
  `pr-str` renders `\c`/`\space`/…; `value_eq` treats `Character` as its own
  type (`(= \a 97)` is false, `(= \a \a)` true); `arg_to_i64` unboxes a char
  (`(int \a)` → 97); `charCast` now returns a `Character` (`(char 47)` → `\/`).
  Registered **last** in `Compiler::new` so no existing type id shifts; added to
  the `gc_alloc_lockin` allowlist.
- **String instance methods** (`runtime.rs` + `host_method` registration):
  `.replace` (both `(char,char)` and `(CharSequence,CharSequence)` overloads),
  `.lastIndexOf` (String), `.substring` arity-3 `(start,end)` (backs
  `clojure.core/subs`).
- **`resolve_var` falls back to `clojure.core`** for unqualified symbols when
  the current ns isn't core (compiler.rs ~6074). This is "every ns auto-refers
  clojure.core" modeled at resolution time — it unblocked `(ns
  clojure.core.protocols)` using `seq`/`first`/`reduced`.
- **`.`-form method name uses the symbol's NAME, ignoring any namespace**
  (compiler.rs ~8197). Syntax-quote qualifies the method symbol in macro
  templates (e.g. `binding`'s `(. clojure.lang.Var (pushThreadBindings …))`
  arrives as `clojure.core/pushThreadBindings`); Clojure's HostExpr reads
  `((Symbol)…).name`, so we now do too.

## 2b. Deferral mechanisms that carried 632 → end-of-file (719)

After 632, every remaining frontier was a host feature we don't model used by a
form that only needs to LOAD. Each was deferred (not aborted) so analysis
continues:

- **`reify` / `reify*`** → special-form handler emitting a runtime ThrowExpr
  (body NOT analyzed). `core_deftype.clj` (where the `reify` macro lives) isn't
  embedded and the uses are JVM-interop. (compiler.rs, in `analyze_seq`.)
- **`deftype` over unmodeled host interfaces** (`Iterable`, `IReduceInit`,
  `Sequential` on `Eduction`) → `expand_extend_type` SKIPS sections whose
  protocol name isn't registered, logging `[cljvm-deftype] … skipping …`. The
  type is still created with the protocols we do model.
- **Unresolved non-dotted symbol** → runtime ThrowExpr instead of an
  analyze-time panic (generalizes the existing unresolved-class-ref path).
  Lets `defn`/`defmethod` bodies referencing fns from non-embedded sub-files
  (`print-sequential`, …) compile. (compiler.rs `analyze_symbol` end.)
- **Unresolved `(var x)` / `#'x`** → runtime ThrowExpr (same idea) for vars
  from non-embedded sub-files (`#'default-uuid-reader`). (compiler.rs
  `parse_var_form`.)
- **Multimethod *registration* externs** (`MultiFn.reset/addMethod/
  removeMethod/preferMethod`) → no longer hard-`panic!` (which aborts the
  loader across `extern "C"`); now `[cljvm-stub]` log + return nil so
  `defmethod`/`remove-all-methods` top-level forms run. Dispatch-side
  read externs still panic. (runtime.rs.)
- **Composite-expr divergence propagation**: `DynamicMapExpr::emit` returns
  `None` when a key/value diverges (a throw-stub terminates the block) instead
  of `expect`-panicking — needed for `default-data-readers`'s `#'`-stub values.
- **Macroexpand-time throw** → if a macro fn throws during expansion (its body
  hit a deferred throw-stub, e.g. `case` constructing `IllegalArgumentException.`),
  `macroexpand_once` stashes the message in the `MACRO_EXPAND_THREW`
  thread-local and returns nil; `analyze_seq` emits a runtime ThrowExpr. (Was a
  hard panic.)

These are deferrals, not implementations. The "no silent stub" rule is honored:
each emits a clear runtime throw or a visible `[cljvm-*]` log.

## 3. Reproduce

```bash
cd dynamic-toolkit
cargo test -p clojure-jvm --test load_upstream_core -- --ignored --nocapture 2>&1 | tail -5
# => [upstream] processed 719 forms successfully  /  test result: ok.
```

## 4. Gates (all green)

- `cargo test -p clojure-jvm --test load_upstream_core -- --ignored` → **719/719,
  passes** (was the open frontier)
- `cargo test -p clojure-jvm --lib` → 356/0
- `cargo test -p clojure-jvm --test load_core_subset` → 27/0
- `char_literals_are_characters_not_longs` → passes
- `CLJVM_TRAP=1 CLJVM_GC=every … load_upstream_core` → full 719-form load, no GC
  crash (chars + deferrals introduce no rooting bug)

## 4b. Next work (no longer a load wall — these are real-impl tasks)

Make the deferrals real, in rough dependency order: embed/port the print
subsystem (`core_print.clj` + multimethods so `print-method` works), then
`reify*`/`deftype` over host interfaces, then the remaining sub-files
(`gvec`, `uuid`, `genclass`). Each removes a class of throw-stubs.

## 5. Code anchors

| What | File:line (approx) |
| --- | --- |
| `cljvm_rt_load` extern | `runtime.rs` `fn cljvm_rt_load` |
| `with_active_session_load_resource` + `resource_source` | `compiler.rs` |
| `Object::Char` | `object.rs` enum |
| char literal in reader | `lisp_reader.rs` `\\` branch |
| `box_char`/`unbox_char`/`is_boxed_char` | `runtime.rs` (after `is_boxed_long`) |
| `CharExpr` | `compiler.rs` (before `ConstantExpr`) |
| `PendingLiteral::Char` + pool-fill arms | `compiler.rs` |
| `Character` type registration | `compiler.rs` `Compiler::new` (registered last) |
| `.replace`/`.lastIndexOf`/`.substring2` externs | `runtime.rs` |
| their `host_method`/`instance_methods` registration + `resolve_clojure_extern` | `compiler.rs` |
| `resolve_var` clojure.core fallback | `compiler.rs` `fn resolve_var` |
| `.`-form method-name (name-only) | `compiler.rs` `parse_member` |
| `reify`/`reify*` defer | `compiler.rs` `analyze_seq` (before primop) |
| deftype host-interface skip | `compiler.rs` `expand_extend_type` |
| unresolved-symbol / `#'var` → ThrowExpr | `compiler.rs` `analyze_symbol`, `parse_var_form` |
| multifn registration no-op stubs | `runtime.rs` `cljvm_inst_multifn_*` |
| macroexpand-throw defer | `compiler.rs` `MACRO_EXPAND_THREW` + `analyze_seq` |
| `DynamicMapExpr` divergence propagation | `compiler.rs` `DynamicMapExpr::emit` |
