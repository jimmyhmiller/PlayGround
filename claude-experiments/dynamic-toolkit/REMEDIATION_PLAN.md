# Dynamic Toolkit Remediation Plan

## Problems To Fix

### 1. Lua register file is modeled as function parameters

`lua2dynir` currently translates Lua functions to signatures like:

`fn(closure, r0, r1, ..., rN) -> value`

where `r0..rN` are the Lua VM registers. This leaks a frontend implementation detail into the IR and JIT ABI. It is the direct cause of the current `<= 16 register params => JIT, else interpreter` split.

### 2. JIT calling convention is incomplete

`call_jit` only passes the first 16 GP arguments on AArch64 and ignores the rest. Entry lowering then zero-initializes parameters beyond that register window instead of receiving them properly. This is not a valid general calling convention.

### 3. JIT is generic over value layout but not root strategy

The interpreter is parameterized over both `TagScheme` and `InterpRootManager`. The JIT only carries `TagScheme` and exposes GC integration via a raw safepoint callback. This bypasses the abstraction boundary the toolkit should preserve.

### 4. GC rooting requirements are expressed as ad hoc runtime flags

The need to conservatively/root-all-values is currently modeled as a boolean capability (`roots_all_values`) and a builder-style override (`with_roots_all`). This should be derived from the runtime layout / pointer encoding contract, not toggled manually.

### 5. Toolkit/backend boundaries are blurred

`dynlower` is effectively an ARM64 backend with generic-looking APIs. The calling convention, frame layout assumptions, and register assignment are mixed directly into the generic lowering path.

## Production Architecture

The reusable execution boundary should be expressed explicitly, not inferred from whatever the current interpreter or ARM64 JIT happens to do.

### Shared execution contracts

Add a shared execution-contract layer that both interpreter and JIT depend on:

- `ValueLayout`
- `RootStrategy`
- `FrameStrategy`
- `CallingConvention`
- `SafepointStrategy`
- `ExecutionConfig`

Those contracts should be reusable across engines and backends so frontends can choose different implementations without rewriting lowering logic.

### Reusable axes

The intended pluggable axes are:

- value layout
- root strategy
- frame strategy
- calling convention
- safepoint strategy
- lowering backend

### Default implementations to ship first

Ship concrete defaults for the current stack, but behind the shared contracts:

- `LowBit` + precise stack roots
- `NanBox` + conservative word roots
- stack-slot frames
- AArch64 internal calling convention
- callback safepoints

### Validation matrix

The stack should validate supported combinations explicitly:

- interpreter + low-bit + precise roots
- interpreter + nanbox + conservative roots
- JIT + low-bit + precise roots
- JIT + nanbox + conservative roots
- >16-arg calls
- direct and indirect calls
- safepoints during nested calls
- invalid config combinations rejected deterministically

## Current Status

Phase 1 has started:

- added a shared `dynexec` contract crate
- switched interpreter/JIT layout bounds to `ValueLayout`
- added `ExecutionConfig`-driven JIT entrypoints in `dynlower`
- preserved compatibility wrappers for existing `LowBit`/`NanBox` callers

Phase 2 has started:

- interpreter root management now uses explicit `RootPrecision` instead of `roots_all_values`
- `dynruntime` exports named root strategies (`MutatorPreciseRoots`, `MutatorConservativeRoots`, `FrameChainPreciseRoots`)
- `dynlower` now has config-driven tests proving external config selection and invalid config rejection
- stack accounting in `dynlower` has been moved into a dedicated frame-layout object that now owns:
  - local slot allocation
  - canonical block-param slots
  - outgoing arg reservation
  - safepoint-visible root slots
- `dynexec::FrameStrategy` now constructs the lowerer frame layout, and `dynlower` is parameterized by full `ExecutionConfig`, not just layout
- `dynexec::CallingConvention` now actively drives register-window size and stack-arg placement in `dynlower`, with host-side test coverage for a non-default 8-register convention
- direct and indirect JIT call lowering now share one CC-driven argument assignment / call cleanup path instead of duplicating call setup logic
- destination writes in `dynlower` now go through a shared assignment target model (`call arg` vs `frame slot`) instead of separate handwritten paths for calls and CFG edge transfers
- `dynlower` now has matrix-style tests covering:
  - low-bit + precise roots
  - nanbox + conservative roots
  - internal CC with more than 16 args
  - configured block-param/frame-slot assignment across CFG edges
- `dynlower` now has the first backend extraction point in [`crates/dynlower/src/backend.rs`](crates/dynlower/src/backend.rs): machine locations plus an `Arm64Backend` that owns prologue/epilogue patching, GP/FP moves, and frame-slot stack traffic
- `Arm64Backend` now also owns backend-side instruction selection for:
  - integer and floating-point binops
  - compare/set operations
  - basic branch/call/return glue (`cbz`, unconditional branch, `blr`, return-value moves)
- `Arm64Backend` now also owns:
  - `select` lowering for GP and FP values
  - base+offset load/store lowering for GP and FP values, including large-offset address materialization
- relocation-aware branch plumbing for the currently extracted CFG paths now goes through `Arm64Backend` instead of `dynlower` manipulating `Arm64RelocKind` directly
- the backend contract is now architecture-typed and machine-word-size-neutral:
  - `LoweringBackend` carries an associated `Arch`
  - extracted backend operations use `MachineWordSize` instead of `RegSize`
- `dynlower` orchestration is now backend-typed as well:
  - `Lowerer` is parameterized as `Lowerer<Cfg, B>`
  - JIT entrypoints can compile through an explicit backend type (`compile_with_backend_and_config`)
  - extracted lowering paths now dispatch through `B::...` rather than always naming `Arm64Backend`
- `dynlower` now has direct regression coverage for backend-routed:
  - `select`
  - load-after-payload
  - store-then-load round trip
- `dynlower` now has an explicit `X64Backend` stub so a second backend target exists in code, not just in comments

## Target Architecture

### A. Frontends should use real call signatures, not exploded VM state

Lua should compile to callable signatures that reflect real inputs:

- main chunk: no user arguments, optionally just `closure`
- child function: `closure` + declared Lua parameters
- vararg function: `closure` + declared parameters + an explicit vararg representation only where Lua semantics require it

The full Lua register set is not part of the callable ABI. It is compiler-internal state and should be represented inside DynIR using values/block params across control flow.

### B. The JIT needs a real calling convention layer

Calling convention concerns should be explicit and backend-owned:

- register-passed arguments
- stack-passed arguments
- result passing
- caller/callee-saved conventions
- frame layout
- safepoint-visible live state

`dynlower` should lower against an ABI description instead of hard-coding ARM64 entry assumptions into generic code paths.

### C. Runtime layout and root strategy must be typed together

The toolkit should have a shared contract that describes:

- pointer encoding/decoding
- whether non-`GcPtr` machine words may contain heap pointers
- how object fields are encoded
- what the GC must scan at safepoints

Valid combinations should be constructible; invalid combinations should fail at compile time or constructor time.

## Execution Plan

### Phase 1. Eliminate the Lua register-parameter model

1. Change Lua function signatures to real call inputs derived from `num_params` / `is_vararg`.
2. Seed the full Lua register state inside the entry block.
3. Keep threading live Lua registers through DynIR block params within the function.
4. Remove the `main chunk <= 16 params => JIT` special case.

### Phase 2. Fix the JIT ABI properly

1. Replace the current `call_jit` entry shortcut with a real entry ABI.
2. Support stack-passed overflow args where a backend ABI needs them.
3. Keep frontend entrypoints small and fixed-width where possible.
4. Make JIT-to-JIT and host-to-JIT calls go through the same ABI rules.

### Phase 3. Align interpreter and JIT GC integration

1. Introduce a typed JIT root/safepoint abstraction.
2. Stop exposing raw safepoint callbacks as the primary public API.
3. Remove `roots_all_values` as a caller-managed boolean.
4. Encode conservative vs precise rooting requirements in the runtime layout contract.

### Phase 4. Split generic lowering from backend implementation

1. Isolate ARM64 lowering into a backend module.
2. Move calling convention and frame layout logic behind backend interfaces.
3. Preserve the path for an x86_64 backend.

### Phase 5. Strengthen tests

1. Add tests that prove large-register Lua functions compile and run under the JIT.
2. Add a compatibility matrix for supported value-layout/root-strategy combinations.
3. Reject unsupported combinations explicitly.
4. Remove tests that encode current shortcuts as acceptable behavior.

## Immediate Work Order

### First

Refactor Lua away from `fn(closure, r0..rN) -> value`.
Use real function signatures and keep Lua register state purely as intra-function DynIR state.

### Second

Replace the JIT entry ABI so it no longer truncates arguments.

### Third

Lift JIT GC/root handling to the same abstraction level as the interpreter.

## Notes

- Lua is allowed to remain `NanBox`-specific.
- The toolkit is not allowed to grow frontend-specific calling convention hacks.
- The current register allocator is not the main problem. The bigger issue is that callable ABI and intra-function state were conflated.
