# Dual-register Result ABI

Semantics unchanged — representation only. A def declared `-> Result<T, E>`
(structurally: Apply/TypeRef of an enum with payload-carrying `Ok` and `Err`
variants, detected at `declare_enum` → `enum_okerr`) returns LLVM
`{i64, i64}` = (tag = variant index, payload) instead of a heap enum pointer.
This is the same move Swift's error ABI and Rust's ScalarPair make: the
compiler stops materializing the Result at function boundaries where both
sides can see through it.

## Representation

- Payload repr follows the SIGNATURE's arm type (`ResultAbi.{ok,err}_payload`,
  a `PayloadKind`): concrete `Int`/`Float`/`Bool`/`Ptr` ⇒ raw bits in the
  register (`RawScalar`); everything else, including TypeVar arms ⇒ pointer
  (`Pointer`; generic instantiations stay uniform-boxed, exactly like the
  materialized enum slot). Caller and callee both read the signature, so the
  interpretation is symmetric.
- `Ok(x)` / `Err(e)` in tail position ⇒ `ret {variant_idx, x}` — ZERO
  allocation (`emit_result_leaf_return`).
- `f(...)?` on a direct call to a dual def ⇒ branch on the tag REGISTER;
  Ok ⇒ use the payload register raw (unboxing a generic-arm scalar); Err ⇒
  materialize the Err variant (cold) and early-return through the shared
  defer-aware path, which itself re-returns a pair when the enclosing def is
  dual (`compile_try_fused_dual_call`, `emit_try_err_return`).
- Any other consumption of a dual call (stored, matched, captured, wired) ⇒
  `materialize_result_pair`: branch on tag, allocate the matching variant,
  boxing RawScalar payloads back into boxed slots. Conversely a dual def
  whose body tail is already a materialized Result pointer lowers it with
  `lower_materialized_result_to_pair`.
- Closures/lambdas keep the uniform boxed ABI (a lambda returning Result
  materializes; eta-adapters materialize via the normal call path).

## External callers (raw fn-ptr invokers)

`CompiledModule.def_result_abi: HashMap<Hash, ResultAbi>` is the registry,
persisted through the bitcode cache (`.shapes`, cache version 17) and
accumulated across `IncrementalJit::install` batches
(`IncrementalJit::def_result_abi(h)`).

- `evalrun::eval` calls EVERY def through a `-> ResultPair {tag, payload}`
  signature (ABI-safe widening: a 16-byte pair returns in two integer
  registers, no sret, argument registers unaffected; non-dual defs leave
  `payload` undefined and we read only `tag`). Dual results render via
  `render_result_pair` — no materialized object exists to walk.
- `cmd_run` checks `def_result_abi` for the root and prints the Ok payload
  register / `Err(bits)`.
- deploy `invoke` is Int-only-gated (unreachable for dual defs); a state
  MIGRATION fn returning Result is rejected loudly at apply time (no
  Rust-side materialization path exists).
- net/serve invokes lambdas (uniform closure ABI) — unaffected.

## GC pitfalls found (the hard-won bits)

1. **Diverged bodies must still finalize frame zeroing.** A dual def whose
   every path early-returns (zero-alloc leaf / self-tail-call / `?`) leaves
   the body block terminated, and `compile_def`/`compile_lifted_lambda` used
   to skip `finalize_frame_zeroing` entirely — leaving the placeholder
   `[0 x ptr]` frame alloca and an origin scanning ZERO root slots while the
   body wrote live roots past it. Symptom: "array op on a non-array value"
   after a collection. Both sites now finalize before the early return.
2. **`emit_self_tail_call`'s placeholder type** must follow the VALUE-level
   result (a materialized Result pointer ⇒ Closure placeholder), not the
   fn's LLVM return type (`{i64,i64}` would have picked Int and broken
   sibling `if`/`match` phis).
3. **All-arms-diverge merges in tail position** (`compile_if`,
   `compile_match`) terminate the merge block with `unreachable` so the
   enclosing dual def doesn't mistake the dummy phi value for a result.

## Measured (M-series, 100M iters)

- `loop_call_result` (call → `Ok(x)` → `?` per iteration): 2088ms substrate
  → **60ms fused**, parity with `loop_call_plain` (61ms). The 18× / 11.3ns
  function-boundary Result tax is gone (~0.6ns/iter).
- `loop_ctrl` (tail-`Ok` loop): 355ms → 91ms (zero-alloc leaf).
- nbody: 179ms → **~123ms** (checksum unchanged). binary_trees 49ms, fib
  9ms, mandelbrot 67ms — no regressions; full suite + AI_LANG_GC_STRESS
  green.
