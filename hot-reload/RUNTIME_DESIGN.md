# Resumable native execution

The production execution unit is not a conventional native stack frame. It is a
heap-resident `Frame` containing a function-version ID, program counter, typed
register slots, and return destination. This makes the exact paused computation
GC-visible and durable across recompilation.

The compiler lowers source to typed, register-based continuation IR. Every
potentially trapping operation ends a basic block. Initially the interpreter
executes one instruction at a time. LLVM later compiles groups of non-trapping
instructions into functions with this ABI:

```text
step(frame*, runtime*) -> { continue, call, return, condition, yield }
```

LLVM therefore accelerates execution without owning continuation semantics.
Native temporaries never survive a step boundary; all live references are in
typed frame slots, giving the precise GC a complete root map.

## Updating code

Published function versions are immutable. Existing frames pin their version.
New calls resolve the current entry. A schema edit re-verifies affected current
functions and publishes broken entries for invalid ones while retaining old code
for pinned frames. Entering a broken function raises a condition before any of
its instructions execute.

Old code can be reclaimed when no frame identifies its version. Later optimized
direct calls must remain within one immutable code-version group; every call
that may cross a live boundary goes through an entry slot.

## Updating data

References name stable object handles. Each handle owns a body tagged with a
schema version. Field access is a migration barrier. A migration builds and
validates a replacement body before swapping it into the handle, preserving
aliases and preventing partial layouts from becoming observable. Missing plans
raise conditions without advancing the frame.

The prototype uses a map as the stable-handle table and implements precise
mark/sweep collection from frame registers. A native runtime can use moving
bodies behind non-moving handles, or add a handle-resolution read barrier. The
nominal type ID and schema version must not be conflated with a compact GC layout
table index.

## Effects

A frame advances past `Emit` only after the effect is committed. A later pause
resumes at its exact PC, so earlier effects are not replayed. For compound
external operations, actors will use a transactional outbox: stage messages and
state changes, commit them together, and assign stable effect IDs for downstream
deduplication.

## What “resume” means

Repair resumes the suspended continuation, not an arbitrary machine instruction
and not the beginning of a tick. Values already computed remain in registers;
completed effects remain completed. The repair may satisfy the condition by
installing a valid function version or a validated migration. The trapping
instruction is then retried.

This model supports loops, branches, and suspension once they are added to the
IR: their continuation is simply a PC plus registers. It does not require stack
copying, native deoptimization, or replay of an entire callback.

