# Dynlower Backend Architecture

This document describes how `dynlower` works now, what is generic, what a backend must provide, and what is still incomplete.

## Overview

`dynlower` is no longer just "the ARM64 JIT with some generic parameters".

It is now split into two layers:

1. Generic lowering/orchestration in [`crates/dynlower/src/lib.rs`](crates/dynlower/src/lib.rs)
2. Backend implementations in [`crates/dynlower/src/backend.rs`](crates/dynlower/src/backend.rs)

The generic lowerer is parameterized by:

- `Cfg: ExecutionConfig`
- `B: LoweringBackend`

So the current shape is:

`Lowerer<Cfg, B>`

Where:

- `Cfg` describes runtime/layout/root/frame/calling-convention choices
- `B` describes how machine code is emitted for a concrete architecture

## What The Generic Lowerer Owns

The generic layer still owns program lowering logic. It is responsible for:

- walking DynIR functions and blocks
- liveness/use counting
- register-state tracking
- spill decisions
- block labels and CFG structure
- frame-slot ownership through `StackFrameLayout`
- block-param canonical slot layout
- call argument assignment planning
- result assignment
- deciding when safepoints occur
- choosing which backend operation to ask for

The important point is that the generic layer decides *what must happen*, but not *how a given ISA encodes it*.

Examples:

- it decides "emit a GP add"
- it decides "emit a compare-and-set"
- it decides "store this value into a frame slot"
- it decides "branch to this label if zero"

But it does not directly choose ARM64 or x86-64 instructions for those operations anymore.

## What `ExecutionConfig` Supplies

The generic lowerer is not just parameterized by a backend. It also depends on the reusable execution contracts from `dynexec`.

An `ExecutionConfig` supplies these axes:

- `Layout`
- `Roots`
- `RootTransport`
- `Frames`
- `CallingConvention`
- `Safepoints`

In practice, `dynlower` currently uses these parts most directly:

### `Layout`

Used for:

- tagged-value lowering
- payload/tag extraction
- `MakeTagged`
- `IsTag`
- deciding layout-specific constants such as `PAYLOAD_BITS`

### `Roots`

Used indirectly through config validation and safepoint/runtime compatibility.

This is now the root *policy* axis. It answers:

- are roots precise?
- are roots conservative?

It does **not** answer how the JIT exposes those roots to the collector.

### `RootTransport`

This is now a separate axis from root precision.

It answers:

- are roots exposed by scanning frame slots?
- are roots maintained in a shadow stack?
- are roots described by stack-map metadata?

The current transport kinds are:

- `FrameScanRoots`
- `ShadowStackRoots`
- `StackMapRoots`

That split matters because "precise roots" and "shadow stack" are not the same kind of choice.

Operationally:

- `FrameScanRoots` still uses frame-memory scanning
- `StackMapRoots` records explicit root-slot metadata per safepoint
- `ShadowStackRoots` now also records explicit root-slot metadata, but those slots are the shadow-root slots tracked by lowering rather than the general frame scan range

### `Frames`

Used to construct the lowerer frame layout.

`dynlower` now works against the `FrameLayout` trait rather than requiring `Layout = StackFrameLayout`.

That means a frame strategy can supply a wrapper or alternate layout type as long as it implements the required frame operations:

- local/root/shadow slot allocation
- outgoing arg reservation
- total frame size computation
- block-param slot storage/query
- root scan size query

### `CallingConvention`

Used for:

- argument register window
- incoming stack-arg offsets
- outgoing stack-arg placement
- outgoing stack byte count

The lowerer asks the selected calling convention where each logical argument goes. It no longer hardcodes the old `16`-register ARM64 window in generic code.

### `Safepoints`

This axis exists in the config model, but the implementation is not fully lifted yet.

Today the backend can emit the machine-side safepoint call sequence, but the public JIT/host contract is still callback-shaped rather than fully strategy-shaped.

The model now distinguishes:

- callback safepoints
- stack-map safepoints

and validates them against the selected root transport.

The runtime-facing handler contract is now also explicit:

- `FrameScanRoots` uses handler payload kind `FrameSize`
- `StackMapRoots` and `ShadowStackRoots` use handler payload kind `SafepointIndex`

That payload kind is exposed on compiled artifacts through:

- `JitFunction::handler_payload_kind()`
- `JitModule::handler_payload_kind()`

So a runtime can tell whether the second callback argument is a raw frame size or an index into the compiled safepoint table.

## What A Backend Must Provide

The backend contract is the `LoweringBackend` trait in [`crates/dynlower/src/backend.rs`](crates/dynlower/src/backend.rs).

A backend must provide an associated architecture:

```rust
type Arch: Arch;
```

That architecture is the dynasm architecture used by `CodeBuffer<B::Arch>`.

### Machine model types

The generic lowerer uses these backend-neutral types:

- `MachineRegClass`
- `MachineReg`
- `MachineLocation`
- `MachineGpBinOp`
- `MachineFpBinOp`
- `MachineWordSize`

These are the generic interface between lowering and the backend.

### Required backend responsibilities

A backend must implement machine emission for:

- prologue/epilogue/frame-size patching
- GP/FP moves
- frame-slot loads/stores
- outgoing stack-arg stores
- integer and FP binops
- integer and FP compare/set
- integer and FP `select`
- memory loads/stores
- unary ops and casts
- immediates and float constants
- incoming stack-arg loads
- tagged-value helpers
- safepoint handler call setup
- branch/call/return/trap emission

In other words: the generic lowerer no longer emits raw machine instructions for these categories.

## Root Model Split

`dynexec` now separates:

1. root precision/policy
2. root transport

That means configs can now express combinations like:

- precise roots + frame scan
- precise roots + shadow stack
- precise roots + stack maps
- conservative roots + frame scan

Validation now rejects incompatible combinations such as:

- precise roots with a layout that requires conservative scanning
- shadow-stack transport with frames that do not support shadow roots
- stack-map safepoints without stack-map transport

What is still missing is full JIT operational support for all of those transports. The type/config model is now ahead of the lowering/runtime implementation.

That gap is smaller than it was:

- `dynlower` now tracks shadow-root slots during value movement
- `dynlower` records stack-map and shadow-stack safepoint metadata
- `dynruntime` now has transport-specific runtime adapters in [`crates/dynruntime/src/jit.rs`](crates/dynruntime/src/jit.rs)

What is still not complete is a fully integrated production runtime protocol for:

- true shadow-stack push/pop ownership outside frame-local metadata
- PC-to-safepoint lookup independent of the direct callback payload
- non-AArch64 host trampoline/runtime glue

## Current Backends

## `Arm64Backend`

`Arm64Backend` is the complete backend today.

It implements the full trait and is the only backend used for real execution in the current test/runtime flow.

It owns:

- AArch64 prologue/epilogue generation
- branch relocation setup using `Arm64RelocKind`
- all currently supported GP/FP instruction emission
- tagged-value machine lowering
- safepoint call emission
- trap emission

The host-side `call_jit` trampoline is also still AArch64-only.

That means:

- generic lowering is backend-parameterized
- emitted machine code can now go through `B`
- but end-to-end "execute the code from the host" is still only implemented for AArch64
- and JIT root exposure is still operationally the frame-scan/callback path even though the config model can express more

## `X64Backend`

`X64Backend` is now a real backend, but only for a minimal subset.

It currently supports enough to compile simple integer functions through the shared lowerer:

- prologue/epilogue/frame patching
- integer constants
- GP moves
- add/sub/mul
- neg/not
- integer compare/set
- branch/call/return/trap plumbing

It does **not** yet support the full lowering surface. Many operations are still explicit `todo!()` in the backend.

That is intentional. Unsupported x64 behavior is explicit rather than silently wrong.

## What "Properly Generic" Means Here

There are two different senses of "generic":

### 1. Structurally generic

This is mostly true now:

- `Lowerer` is generic over `B: LoweringBackend`
- compile entrypoints accept an explicit backend type
- backend-owned operations route through `B`
- the lowerer itself is no longer constrained to `Arch = Arm64`

### 2. Fully implemented for multiple backends

This is not true yet:

- only ARM64 is production-capable
- x64 only supports a small compile-time subset
- host execution trampoline is still ARM64-only
- x64 frame traffic, stack args, safepoints, FP, and many data-movement cases are not complete

So the architecture is now real enough to support multiple backends, but only one backend is near-complete.

## How Lowering Flows Now

For a given function, the rough flow is:

1. Validate `ExecutionConfig`
2. Construct `Lowerer<Cfg, B>`
3. Create backend-architecture `CodeBuffer<B::Arch>`
4. Create labels for blocks
5. Count uses / initialize value metadata
6. Construct frame layout via `Cfg::Frames`
7. Emit backend prologue
8. Lower instructions and terminators
9. Emit backend branches/calls/returns
10. Patch final frame size through backend API
11. Finalize code memory

Within instruction lowering, the pattern is:

1. Generic layer decides logical operation
2. Generic layer ensures values are in suitable `MachineReg`s
3. Generic layer asks the backend to emit the machine operation
4. Generic layer updates value locations and liveness state

## What Parameters A Backend Needs

If you are implementing a new backend, the important inputs are:

### Architecture type

You need a dynasm architecture implementation:

- ARM64 uses `dynasm::arm64::Arm64`
- x64 uses `dynasm::x86_64::X64`

### Register mapping

You must define how `MachineReg { class, index }` maps to actual hardware registers.

That includes:

- GP registers
- FP/XMM registers
- any scratch registers your backend assumes
- any reserved frame/base/safepoint registers your backend assumes

### Prologue/frame convention

You must define:

- how frame/base pointers are established
- how frame size patching works
- how frame slots are addressed
- what register holds incoming-stack base if needed

### Control-flow relocation model

You must define:

- label binding
- unconditional branch relocation
- conditional branch relocation
- call relocation or register-call strategy

### Calling/trap conventions

You must define:

- how indirect calls are emitted
- how returns are emitted
- how traps are emitted
- how safepoint callbacks are called

### Data-movement rules

You must define:

- GP to GP moves
- GP to FP and FP to GP bit moves
- memory addressing forms for frame and base+offset accesses
- immediate materialization

## What Is Still Not Abstract Enough

The major remaining gaps are:

### Host trampoline

`call_jit` and `call_jit_with_reg_limit` are still AArch64-specific.

That means the host->JIT execution path is not yet abstracted the same way as backend emission.

### Backend capability model

The trait assumes one large backend surface. It does not yet expose capabilities like:

- "supports this op"
- "requires this frame convention"
- "supports FP"
- "supports stack maps"

Right now unsupported operations are just `todo!()` in partial backends.

### Frame layout type

The lowerer still expects:

- `Layout = StackFrameLayout`

So there is still one concrete frame layout shape in practice, even though the frame strategy is selected through `ExecutionConfig`.

### Calling convention vs backend register files

Calling convention selection is external now, but some concrete register-file assumptions still live in backend implementations rather than in a more explicit ABI descriptor.

### Regalloc is still shared and simplistic

The register allocator is still generic/shared and not yet backend-specialized in a serious way.

It assumes enough uniformity across backends to work for the current subset, but it is not yet a production-quality multi-backend allocator.

## What Tests Prove Right Now

The new backend architecture is currently proven by:

- ARM64 still compiling/running existing JIT tests
- Lua JIT tests still passing on the ARM64 path
- x64 compile-only tests for:
  - constant return
  - integer add
  - integer compare
  - integer negation

Those x64 tests matter because they show the generic lowering path can now drive a second backend for real code generation, not just type-check a stub.

## Recommended Next Steps

If the goal is to keep making backend support "properly generic", the next steps should be:

1. Introduce a backend-specific host trampoline layer so execution is not AArch64-only.
2. Implement x64 frame-slot loads/stores and incoming/outgoing stack args.
3. Implement x64 return/argument ABI compatibility instead of compile-only code generation.
4. Add x64 direct execution tests on x86-64 hosts.
5. Push safepoint and root exposure further into typed runtime/backend contracts.
6. Eventually split the very large `LoweringBackend` trait into smaller backend roles if it starts to ossify.

## Short Summary

`dynlower` now works like this:

- generic lowering decides what code needs to do
- `ExecutionConfig` decides runtime/layout/frame/calling-convention policy
- `LoweringBackend` decides how that becomes machine code

ARM64 is the complete backend today.

x64 is now a real but partial second backend that proves the abstraction is no longer fake.
