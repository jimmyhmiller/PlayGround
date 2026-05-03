# 04 — Extern + Slow-Path Registration

## Implementation status: ✅ Implemented (slow-path defaults only)

- **Code**:
  - 9 `panic_*` thunks in [`crates/dynlang/src/slow_paths.rs`](../../crates/dynlang/src/slow_paths.rs).
  - `DynModule::auto_externs`, `register_slow_paths_with_defaults`, and `override_extern` in [`crates/dynlang/src/lib.rs`](../../crates/dynlang/src/lib.rs).
  - `DynGcRuntime::set_auto_externs` and the auto-bind branch in [`crates/dynlang/src/gc.rs`](../../crates/dynlang/src/gc.rs)'s `build_extern_table`.
- **Tests**: 3 tests in `slow_paths::tests` (defaults populate, override replaces, plain `register_slow_paths` leaves auto_externs empty). Panic-stub bodies aren't `should_panic`-tested because `extern "C"` aborts on panic; visual review of the macro is sufficient.
- **Migration**: beagle deleted the `slow_stub!` macro, 7 stub fns + 2 unary ones, and 9 arms in `jit_extern_for`. One call to `register_slow_paths_with_defaults` replaces all of it.
- **Not implemented**: typed extern-fn declaration (`Signature::from_extern_fn` + `extern_thunk!` macro / `dm.declare_extern_fn`). The 8 user-extern declarations + matching `jit_extern_for` arms in beagle still exist as a parallel registry. Worth doing if a second frontend ever needs to declare its own externs.

## Problem

Every JIT-bound extern lives in *two* parallel registries that must agree by
string name. Drift between them is silent — a typo gives a runtime
"unresolved extern" with no compile-time signal.

In beagle:

- **lower.rs side.** Eight `dm.declare_extern("beagle_…", Signature{…})` calls
  for `print`, `println`, `length`, `get`, `to_number`, `cos`, `sin`,
  `time_now`, `prop_slow`
  ([`lower.rs:135-184`](../../crates/beagle/src/lower.rs#L135)). The returned
  `FuncRef`s are stored in `Lowerer` fields used at every call site.
- **main.rs side.** Eight matching arms in `jit_extern_for`
  ([`main.rs:210-232`](../../crates/beagle/src/main.rs#L210)) mapping the
  same string name to a function pointer. The signature is *implicit* —
  Rust's type system never checks that `ext_print: extern "C" fn(u64)`
  matches the `Signature { params: vec![Type::I64], ret: None }` declared
  in lower.rs.

Layered on top, the slow-path registration is half-automated and confusing:

- `dm.register_slow_paths("beagle")`
  ([`lower.rs:133`](../../crates/beagle/src/lower.rs#L133)) declares the
  arithmetic slow paths in the module — but the embedder still has to write
  the seven `slow_stub!` thunks
  ([`main.rs:111-138`](../../crates/beagle/src/main.rs#L111)) and bind each
  in `jit_extern_for`. So "register" doesn't actually register; it declares.
- The thunks themselves are panic stubs because the binary_trees subset
  doesn't actually trigger them. A future embedder that *does* hit
  `ext_add` has to discover all seven exist, write real implementations,
  and remember the calling convention.

## Proposed API

### One-call user externs

```rust
use dynlang::extern_thunk;

// Macro derives the Signature from the function's Rust type. Returns a
// FuncRef ready to use at call sites.
let print_ref = extern_thunk!(dm, "beagle_print", ext_print);
let length_ref = extern_thunk!(dm, "beagle_length", ext_length);

// Bind the whole batch to a JIT extern resolver in one shot.
let externs = dynlang::ExternMap::from_thunks(&[
    ("beagle_print", ext_print as *const u8),
    ("beagle_length", ext_length as *const u8),
    // ...
]);
let jit = gc.compile_jit::<Cfg, Backend, RA>(&module, externs.resolver());
```

The macro-derived `Signature` removes the silent-drift class of bug. If
`ext_print` is `extern "C" fn(u64)`, `extern_thunk!` writes
`Signature { params: vec![Type::I64], ret: None }` itself.

### Default slow paths

```rust
// Single call: declares all slow-path externs in the module AND binds
// default panic-stub thunks in the extern map. Embedders override the
// ones they want via the returned handles.
let slow = dynlang::SlowPaths::register(&mut dm, &mut externs);

// Override only the ones you've implemented:
slow.add.bind(&mut externs, my_real_add as *const u8);
slow.lt.bind(&mut externs, my_real_lt as *const u8);
// The rest stay as panic stubs.
```

## Implementation plan

1. **`dynlang::Signature::from_extern_fn<F>()` derived via a sealed trait.**
   Implement for `extern "C" fn(...)` shapes up to N=6 args (matches the
   widest current slow path). Map Rust `u64`/`i64`/`f64`/`*const u8` to
   `Type::I64`/`Type::F64`/`Type::Ptr`. Compile-time error on shapes the
   IR doesn't model.

2. **`extern_thunk!` macro.** Three-line wrapper: derive signature, call
   `dm.declare_extern(name, sig)`, return `FuncRef`. Optional second form
   that also pushes into an `ExternMap`.

3. **`dynlang::ExternMap` and `Resolver`.** A typed key-value store wrapping
   the `&str -> *const u8` lookup that `compile_jit` already takes. Lets
   embedders accumulate entries and pass `externs.resolver()` once.
   Replaces the 23-line `match`/`return None` arm pattern in beagle.

4. **`dynlang::SlowPaths`.** Owns the canonical list of slow-path externs
   (currently `add/sub/mul/div/eq/lt/gt/neg/not`). `register` calls
   `dm.register_slow_paths` *and* binds default panic stubs. Returns a
   struct of typed handles (`SlowPaths { add: SlowPathHandle, ... }`)
   so overrides are checked by name at compile time, not string lookup.

5. **Beagle migration.** Replace `register_slow_paths` + 7 `slow_stub!` +
   the matching arms in `jit_extern_for` with one `SlowPaths::register`
   call. Replace the 8 user-extern declarations + `jit_extern_for` with
   `extern_thunk!` calls. Estimated delta: −100 LOC, plus the typo class
   becomes a compile error.

## Open questions / risks

- **Macro vs. trait.** A pure trait approach (no macro) would let
  `dm.declare_extern_fn(name, ext_print)` do the same job. Macros help
  only if we want to interpolate the function name into a `static`
  (e.g. for hot reload). Default to the trait — fewer macros to maintain.
- **`compile_jit`'s closure resolver.** Today's `jit_extern_for(&str) ->
  Option<*const u8>` is fine for simple cases but composes poorly when
  multiple sources contribute (user code + default slow paths +
  toolkit-provided IC slow path). `ExternMap` should be the canonical form;
  `compile_jit` should accept either a closure or `&ExternMap`.
- **Ownership of `__gc_alloc__` and `__dynlang_prop_slow__`.** Already
  auto-bound by the runtime. The `ExternMap` API needs to surface this
  (e.g. `ExternMap::with_runtime_defaults(&gc)`) so embedders aren't
  surprised by name collisions.
