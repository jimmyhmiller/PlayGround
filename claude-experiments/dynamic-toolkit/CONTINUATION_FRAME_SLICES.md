# Continuation Frame Slice Design

This document defines the reusable substrate for multi-shot delimited continuations in the toolkit.

## Goal

The core abstraction is not `shift/reset` directly. It is:

- prompt boundaries
- captured frame slices
- slice cloning
- slice resume

Delimited continuations are then a lowering pattern on top of that substrate.

## Current state

The current implementation has two pieces in place.

### 1. IR substrate in `dynir`

- `Type::FrameSlice`
- `PromptId`
- `Inst::PushPrompt`
- `Inst::PopPrompt`
- `Inst::CaptureSlice`
- `Inst::CloneSlice`
- `Terminator::ResumeSlice`
- `Terminator::AbortToPrompt`

### 2. Shared frame-slice payload model

`dynexec` now defines the reusable payload shape and storage contract:

- `FrameSliceMode`
- `FrameResumePoint`
- `CapturedFrame`
- `FrameSliceSnapshot`
- `FrameSliceStore`

`dynruntime` now has a concrete in-memory implementation:

- `OwnedFrameSliceStore`
- `FrameSliceHandle`
- `FrameSliceRootSource`

This means the system now has a common representation for captured frame slices and a common way to enumerate their roots.

### 3. Interpreter execution support

The interpreter now supports:

- `PushPrompt`
- `PopPrompt`
- `CaptureSlice`
- `CloneSlice`
- `ResumeSlice`
- `AbortToPrompt`

through the shared `FrameSliceStore` contract.

Current behavior:

- `CaptureSlice` captures the active delimited suffix up to the nearest matching prompt.
- `CloneSlice` duplicates the captured slice for multi-shot use.
- `ResumeSlice` restores the captured slice and injects resume arguments into the saved capture result position.
- `AbortToPrompt` unwinds to the nearest matching prompt owner and returns the abort payload from that prompt-owning frame.

Still not implemented:

- fully general native frame-slice capture for arbitrary native state that is not part of the current control-aware `Call` / `Invoke` / `InvokeIndirect` continuation protocol

### 4. JIT control-outcome support

`dynlower` now lowers:

- `CaptureSlice`
- `CloneSlice`
- `ResumeSlice`
- `AbortToPrompt`

into explicit JIT control outcomes plus `FrameReifyRecord` metadata, instead of panicking on those IR operations.

Current JIT behavior:

- `CaptureSlice` exits JIT execution with a `JitOutcome::CaptureSlice` carrying the selected value payloads and a metadata record index.
- `CloneSlice` exits JIT execution with a `JitOutcome::CloneSlice` carrying the source slice handle bits and a metadata record index.
- `ResumeSlice` exits JIT execution with a `JitOutcome::ResumeSlice` carrying the source slice handle bits plus resume arguments and a metadata record index.
- `AbortToPrompt` exits JIT execution with a `JitOutcome::AbortToPrompt` carrying the abort payloads and a metadata record index.
- `PushPrompt` / `PopPrompt` are currently lowering no-ops in the JIT substrate.

These operations still cross the runtime boundary as explicit control exits, but native JIT slice materialization and native resume now build on that boundary instead of stopping at it.

### 5. Runtime decode/materialization support

`dynruntime` now understands those JIT control exits:

- `decode_frame_control_outcome(...)` converts `JitOutcome::{CaptureSlice,CloneSlice,ResumeSlice,AbortToPrompt}` plus `FrameReifyRecord` metadata into runtime control requests.
- `materialize_capture_slice(...)` can turn a JIT capture outcome into a stored `FrameSliceSnapshot`.
- `execute_jit_function(...)` and `execute_jit_module_function(...)` run JIT code and surface either a normal result or a frame-control result through one runtime-facing API.
- `execute_jit_function_to_terminal(...)` and `execute_jit_module_function_to_terminal(...)` keep driving native clone/resume control exits until they reach a real terminal result.
- clone requests can now be satisfied through the configured `FrameSliceStore`
- resume requests can now be surfaced as typed runtime requests carrying the slice handle and resume arguments
- one-shot resume requests now mark the stored slice consumed in the runtime bridge, matching interpreter semantics
- `resume_stored_slice_with_interpreter(...)` still exists as a shared-substrate/debug path
- `resume_stored_slice_with_jit(...)` and `resume_stored_slice_with_jit_module(...)` can resume the current JIT capture format natively by entering a generated resume stub for the original capture site

The JIT reification records now also carry:

- total frame value count for the captured function
- active prompt stack at the control exit
- payload SSA value indices
- payload value types
- explicit root payload positions

so runtime materialization does not have to guess which returned payload values are GC roots.

This is still an intermediate step:

- capture materialization currently uses the explicit payload values returned by JIT
- those payloads are now expanded into value-indexed frame snapshots using the recorded IR value indices and frame size
- captured snapshots also preserve the active prompt stack recorded by JIT lowering
- internal JIT `Call`, `Invoke`, and `InvokeIndirect` chains now preserve suspended caller frames during capture, so module capture materialization can produce multi-frame snapshots instead of collapsing everything to the innermost frame
- suspended invoke frames now preserve real resume arguments instead of placeholder slot identifiers
- the native JIT resume path now handles the existing single-frame capture model plus multi-frame internal `Call`, `Invoke`, and `InvokeIndirect` chains, including normal and exception propagation through the terminal execution helpers
- `CaptureSlice`, `CloneSlice`, and `ResumeSlice` still cross the runtime bridge as explicit control exits, but clone and resume can now be driven back into native execution instead of stopping at the bridge

## Semantics

- `PushPrompt(prompt)` installs a dynamic prompt boundary.
- `PopPrompt(prompt)` removes a matching prompt boundary.
- `CaptureSlice(prompt, live...)` captures the delimited dynamic suffix up to `prompt` and returns an opaque `frameslice` handle.
- `CloneSlice(slice)` deep-copies a captured slice so the original can still be resumed later.
- `ResumeSlice(slice, args...)` transfers control into the captured slice.
- `AbortToPrompt(prompt, args...)` aborts non-locally to the nearest matching prompt.

The intended multi-shot model is:

- captured slices are one-shot by default
- multi-shot behavior is expressed by cloning before each additional resume

## Why this shape

This substrate can support:

- delimited continuations
- effect handlers
- coroutines/generators
- later fiber-like control abstractions

without baking one frontend control construct directly into the IR.

## Next steps

1. Extend JIT capture beyond the current explicit control-aware continuation protocol to more general native frame reification when needed.
2. Keep clone/resume metadata aligned with that fuller native frame representation.
3. Lower `shift/reset`-style constructs on top of this substrate in frontends.
