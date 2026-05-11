# TODO

Outstanding cleanup items from the review. Tracked items live in the task
list; this file holds everything else.

## Live correctness bugs

### Keyword/Map equality bug
`(get {:a 1} :a)` may silently return nil. `map_get`
(`collections.rs:195-218`) and `namespace.rs:118-169` use bitwise pointer
equality, but `alloc_keyword` (`collections.rs:90-108`) does NOT intern —
two `:foo` occurrences allocate distinct Keyword objects. Doc-comment
claims "= compares by inner sym-id"; the lookup code does not.

Fix one of:
- intern keywords at allocation (matches doc), or
- dispatch lookup on `IEquiv`/sym-id instead of bitwise eq.

## Stale / lying doc comments

- `externs.rs:836-848` — `clj_satisfies_p` doc says "always returns false";
  code checks `protocol_membership`. Delete the obsolete paragraph.
- `value.rs:1-30` — doc claims tag 3 reserved for char; code uses tag 3
  as `TAG_SYMID`. Reconcile.
- `value.rs:160` — "For checkpoint 2 the only heap objects are Lists" —
  no longer true.
- `types.rs:113-119` — `Vector` ObjType declares HAMT-shaped fields
  (`count, shift, root, tail`); implementation in `collections.rs`
  ignores `tail`/`shift` and treats `root` as a side-array pointer.
  Either drop the unused fields or actually implement HAMT.
- `compile.rs` header — still describes "Checkpoint 3" while the file
  implements protocols/deftype/macros/multi-arity.
- `collections.rs:9, 191`, `namespace.rs:386` — "v1 layout" /
  "no captured env" stale in light of current code.

## Structural duplication / refactors

### Centralize special-form symbol IDs
Three independent string-matched lists drift today:
- `expand.rs:25-29` `SPECIAL_FORMS`
- `freevars.rs:21-24` `SPECIAL_HEADS` (already missing `deftype*`,
  `defprotocol`, `extend-type`)
- `compile.rs` `compile_top` head dispatch (1626-1636 etc.)

Pre-intern once into a `SpecialForms { def: u32, fn_: u32, ... }` struct
on the host; dispatch on `u32` equality. Kills the per-form
`.to_string()` allocs too.

### Extract repeated extern lookup
`self.func_refs.get("__name").copied().expect("__name not registered").fref()`
appears ~15+ times in `compile.rs` (`__check_args_list`,
`__reader_list_first/rest/count`, `__alloc_fn`, `__alloc_closure`,
`__cons`, `__method_lookup`, `__record_get_field`, `__record_set_field`,
`__alloc_record`, `__register_deftype`, `__register_method`,
`__def_value`, `__register_protocol_member`, `__arity_check`).

Add `Compiler::extern_fref(&self, name) -> FuncRef`, or pre-resolve into
a struct of named `FuncRef`s in the constructor.

### Multi-arity dispatch duplication
`compile_clauses_body` (`compile.rs:1217-1366`) and
`compile_closure_body_clauses` (`compile.rs:2063-2193`) are ~95%
identical. Differences: closure passes `self_fn` through clause blocks
and uses `lower_closure_clause_body` instead of
`lower_single_clause_body`. Extract a shared `emit_arity_dispatch`
helper parameterized by per-clause-block prepended `Value`s.

### Brittle clause identity via `std::ptr::eq`
`compile.rs:1293, 1297, 2114, 2118` use `std::ptr::eq` on borrowed
`&Clause`. Iterate with `.enumerate()` when building `fixed`/`order` and
carry the index.

### Dead `_self_fn_tagged_opt` parameter
`lower_single_clause_body` (`compile.rs:1376-1455`) takes
`self_fn_tagged_opt: Option<Value>`, immediately
`debug_assert!(_.is_none())`, never reads it. Sole caller passes `None`.
Either delete or actually use it.

### Dead `expect_fn=false` branch
`compile_def_like` (`compile.rs:1108`) has
`if !expect_fn { unreachable!("non-fn def shape not yet supported") }`.
Only ever called with `expect_fn=true`. Delete the parameter and the
dead branch.

### `compile_deftype_combined` silently returns None
`compile.rs:522-528` — when both halves yield non-Expr, returns
`TopResult::None`; a malformed deftype just disappears. Should at least
assert.

### Reader-collection bridge externs
~18 near-identical `clj_reader_{list,vector,set,map}_{count,first,rest,nth,...}`
externs in `externs.rs:1012-1247`. Bootstrap shim because
`extend-type __ReaderXxx` blocks in core.clj need to bottom out
somewhere. Once protocol dispatch works (it does), built-ins should
register methods through the same path. Sweep these out.

### Primitive registry redundancy
`externs.rs:1385-1532` registers many fns under 3-4 names
(`+`/`prim-add`, `cons`/`__cons`, `keyword?`/`__is_keyword`,
`atom`/`__atom_create`). Comments justify each ("POC-era alias",
"core.clj redefines and would shadow"). Real but accumulating —
follow-up pass to delete POC-era aliases once core.clj no longer
references them. Consider `Prim::with_aliases(&[...])` builder.

### Two copies of `value_type_name_sym`
`externs.rs:882` and `protocol.rs:138`. Delete the externs.rs copy and
call the public one.

### `alloc_fn` vs `alloc_fn_with_captures`
`namespace.rs:372-429` — `alloc_fn` is `alloc_fn_with_captures(.., &[])`.
Same with `alloc_array` and `alloc_array_nil`
(`collections.rs:381-419`). Fold.

## Reader correctness

- `reader.rs:217-222` — odd map entries `panic!` instead of returning
  the existing `ReadError::OddMapEntries` variant. Dead variant + panic
  on user input.
- `reader.rs:339` — `BadEscape` reused for "multi-char unknown char
  literal". Add a dedicated variant.
- `reader.rs:381` — `if self.src[begin] == b'-' { -n }` ignores
  negative-hex overflow (`-0x8000000000000000` will panic on negation).

## Macroexpansion correctness

- `expand.rs:218-226` — quasiquote arm calls `self.expand_all(rewritten)`
  which restarts the iteration budget. Pathological macros that re-emit
  unquote bypass `max_iters`. Thread a global remaining-iters counter or
  recursion depth.
- `expand.rs:293-305` — `walk_def` drops everything after the value
  form. `(def NAME docstring value)` and `(def NAME value meta...)` are
  valid Clojure shapes. Reject extra forms with a clear error or thread
  them through.
- `expand.rs:176-229` — `walk_subforms` has no `loop`/`recur`/`deftype*`/
  `defprotocol`/`extend-type` arm despite all being in `SPECIAL_FORMS`.
  `(loop [x (some-macro)] ...)` won't expand the bound value because
  bindings live inside a vector. Add a `loop` arm mirroring
  `walk_let_like`.
- `expand.rs:343-346` — orphaned doc comment for `list_from_vec` is
  stale from a prior version.

## Free-var analysis gaps

In `freevars.rs`:
- `:147-170` — `fn` arm doesn't handle multi-arity. A multi-arity fn
  treats the first arity-list's params as if `(arity-list)` were the
  param vector — wrong shape.
- `:115-142` — let bindings don't handle `:as`, destructuring, `&`
  rest. Any destructured local is reported as free.
- `:171-178` — `def`/`defmacro` walk skips the name (good) but doesn't
  treat `defmacro`'s param vector as a binder. Macro body referencing
  its own param is reported as free.

## Migrate hot-path list/map accessors to `dynobj::TypedPtr`

`dynobj::TypedPtr<T>` exists as a phantom-typed wrapper around a raw
heap ptr (with checked construction via `try_cast`). The clojure crate
hasn't migrated existing call sites — `value::first`/`rest`/`list_iter`
now panic on misuse (the immediate safety fix), but the type-level
guarantee is still future work.

To opt in: define markers in `value.rs` (e.g. `pub struct ListMarker;
pub type ListPtr = dynobj::TypedPtr<ListMarker>;`), add a NanBox-aware
helper that returns a `Result<ListPtr, _>` from a tagged `u64`, and
sweep `expand.rs` / `compile.rs` / `quasiquote.rs` / `printer.rs` /
`externs.rs` callers (40+ sites). Big refactor — only do it if the
runtime panics from the entry assertions reveal real bugs that the
type system would have caught earlier.

## Exception system — current state

### Working end-to-end

- **In-function and cross-function `(try)`/`(throw)`/`(catch)`.**
  All 14 tests in `eval_throw.rs` pass: catching throws from
  callees up to 3 frames deep, nested try, inner re-throw caught
  by outer, vector-literal results in catch handlers, uncaught
  throws panicking at eval.

### How it's wired

- `(throw v)` calls the `__raise_exception` asm stub (per-arch;
  aarch64 implemented in `externs.rs`) through the JIT call table
  (indirect call → control-aware). The stub sets
  `JitOutcomeKind::Exception` and `payload0 = v` then returns.
- Plain Call sites in control-aware mode auto-propagate non-Return
  outcomes — exception bubbles up frame-by-frame.
- `(try body... (catch _T name handler))` synthesizes the body as
  an anonymous internal IR fn taking captured locals as args, then
  the outer fn `fb.invoke`s it with normal=merge_bb,
  exception=handler_bb. The Invoke's exception path routes the
  thrown value to handler_bb's first param (dynlower convention
  established by this work).
- `Engine::eval` / `call_compiled` match on
  `JitOutcome::Exception(v)` to surface uncaught throws as panics.

### Toolkit changes that landed for this

- `fb.try_scope(handler) -> PromptId`, `fb.throw_to(prompt, v)` in
  `dynir` (prompt-based API). Kept for a future effects/
  continuations layer; current throw/catch uses Invoke instead.
- `fb.import_module_func(fref, name, sig)` in `dynir`: lets a
  `FunctionBuilder` learn about a callee declared after
  `define_func` time (required for synthesizing helpers
  mid-compilation, like try-body fns or closures).
- `dynlower::lower_invoke_common` now routes the runtime exception
  value (preserved in x0 across the suspended-frame save/restore)
  into the exception block's first param. Mirrors the existing
  normal-block return-value-as-first-param convention.
- **Fixed `LinearScanAllocator + Invoke` bug** in `dynlower`:
  Invoke's normal/exception path was writing the return value to
  the canonical frame slot, but LinearScan's `enter_block` honors
  pre-assigned registers/spill slots and never loads from canonical
  slots. Added `RegisterAllocator::place_call_return_in_block_param`
  trait method; LinearScan's impl emits a move/store to the param's
  assigned home; Greedy default falls back to canonical slot.
  `write_invoke_return_to_target` now goes through this method.
  Repro test: `invoke_internal_declared_late_with_safepoint_inst_and_linscan`
  in dynlower. Once fixed, switched clojure back to LinearScan
  permanently.
- **Made `GreedyRegState` non-public** (`pub(crate)`). Only used
  internally by `LinearScanAllocator` for dynamic (non-prepared)
  lowering paths. `LinearScanAllocator` is now the only allocator
  frontends can pass to `JitModule::compile_with_regalloc`.

### Deferred follow-ups

1. **Type-filtered catches.** Today `(catch _T name body)` parses
   `_T` but matches every exception (catch-all). Real Clojure
   semantics: instance check on the thrown record's type-name +
   protocol like `IException`. Multiple catch arms.
2. **`(finally)` clauses.** Always-run cleanup. Integrates with
   both body's normal exit and handler block.
3. **Convert panicking externs to throws.** `clj_arity_check`,
   `clj_method_lookup`, `clj_no_matching_arity_panic`, type-check
   panics in `aget`/`aset!`/`set!`/etc — should construct an
   exception value and throw it instead of `panic!`. Once #1 lands
   (type filter), user code can catch specific exception classes.
4. **Map-literal evaluation.** Vector literals with non-literal
   elements now evaluate each element at runtime
   (`__vector_from_list` extern + `lower_vector_literal`). Maps
   with non-literal values still pin as-is — `{:k (foo)}` would
   bake in the symbol `foo` instead of its value. Same pattern as
   vectors: add `__map_from_list`, detect non-literal keys/values,
   route through.
5. **`LinearScanAllocator` + `Invoke` bug** (toolkit). Switch
   clojure back once fixed.

## Hide `call_table_base` from compile-side state (toolkit)

`Compiler::call_table_base: u64` (in `compile.rs`) is the JIT call
table's runtime address, baked in as `iconst` whenever we emit a
`call_via_func_ref`. This couples compile-time codegen to JIT memory
layout. The high-level indirect-call op (toolkit task #3) folded the
arithmetic into dynir, but the address itself still leaks through.

Cleaner shape: dynir/dynlower exposes the call-table base as a
late-bound symbol (resolved at extend/link time), so `call_via_func_ref`
emits a relocation rather than a constant. Then `Compiler` doesn't
need the address at all.

## Minor

- `quasiquote.rs:135-148` — `match_splice` re-walks `name` lookup on
  every element; cache splice/unquote sym-ids on the ctx.
- `externs.rs:980` — `clj_register_deftype` panics inside
  `from_raw_parts(...).iter().map(...)` on caller-owned memory; verify
  caller's IR safepoint, add a comment.
- `compile.rs` `Compiler::call_table_base: u64` couples compile.rs to
  JIT memory layout. Goes away once toolkit gives us a high-level
  indirect-call op (tracked task).

## What's good — preserve when refactoring

- `lib.rs` bootstrap and concurrency (Mutex/RwLock split, per-call
  FrameChain, lock-drop-then-reacquire to avoid macro-expand deadlock).
- `SeqCursor` (`collections.rs:485-559`) and `printer::write_record` —
  protocol dispatch instead of hardcoded type names.
- GC root discipline in allocating paths: `build_args_list`,
  closure-capture spill, reader's `read_collection`.
- `Host::intern_keyword` (`host.rs:154`) drop-lock-realloc-recheck
  pattern.
- `LoopTarget` (`compile.rs:33-47`) unifies `loop`/fn-body/closure-body
  recur via `prepend`+`pack_args` flags.
- `compile.rs:1604` undefined-symbol panic comment explaining why
  silent runtime lookup would corrupt forward-referenced macros.
