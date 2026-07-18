# Scry in WebAssembly

The Scry VM — bytecode interpreter, generational GC, typechecker, compiler, and the live
eval channel — compiled to `wasm32-unknown-unknown` and running in a JS host. This is the
foundation for the in-browser prototype (viewer + agent TUI, no server, no native process).

Design + rationale: [`../docs/08-wasm-port.md`](../docs/08-wasm-port.md).
Coil-side asks (all landed): [`../docs/09-coil-asks.md`](../docs/09-coil-asks.md).

## Build

```sh
./build.sh          # -> wasm/scry.wasm (gitignored), then wasm-tools validate
```

Requires the self-hosted `coil` with the wasm finalizer's **C0** (shadow stack /
`__stack_pointer`) and **C1** (function table / `call_indirect`) support.

## Run the headless tests

```sh
node test-boot.mjs   # boot a program, eval expressions, reflect over the live heap
node test-live.mjs   # live instance enumeration, allocation via eval, method invoke
node test-swap.mjs   # hot body-swap: greet() 1 -> redefine -> 99
```

## What works today

- `scry_boot(path)` — loads, typechecks, compiles and runs a program's `main()`. The whole
  native multi-file loader is reused; file reads are served from an in-memory VFS.
- `scry_eval(src, len) -> ptr` — the **one** viewer wire op (DECISIONS #8), returning
  NUL-terminated `{"value":…}` / `{"error":{…}}` JSON. Verified against the live heap:
  `types()` reflection with real `liveCount`, `instances()`, allocating objects via eval,
  invoking methods on live instances, and hot body-swap.
- Typed diagnostics come back through the in-memory diag sink (no fd-2 / temp files).
- **Uncrashable eval** — a hard panic (stale ref, arena-OOM, bad opcode, internal compiler
  invariant) comes back as typed `{"error":{…}}` JSON and the instance stays live, exactly
  as on native. wasm32 has no `setjmp`/`longjmp`, so the host unwinds instead: `setjmp`
  returns 0, the `longjmp` import throws a `ScryLongjmp` whose JS exception unwinds the
  wasm frames, and `eval()` catches it, restores the shadow-stack pointer, and calls
  `scry_eval_recover()` to finish the response through the same `write-error-inner` path.
  Verified: 200 consecutive panics, 200 typed errors, 200/200 healthy evals in between,
  heap intact.

  The unwind leaves **no** trace: the host restores the shadow stack via the exported
  `__stack_pointer` global, so the pointer returns to exactly its post-boot value after
  thousands of panics (`drift=0`). `test-panic.mjs` guards against any regression.

## The bridge (`scry-wasm.js`)

The module defines and exports its own linear memory; the bridge supplies the `env.*`
imports over it:

- **libc in JS** — `malloc`/`realloc`/`free` (first-fit free-list above the exported
  `__heap_base`, growing the `Memory`), `mem*`/`str*`, and a `snprintf` that implements the
  specifiers scry uses **including star precision** (`%.*s`, used for non-NUL-terminated
  slices).
- **VFS** — `fopen`/`fread`/`fseek`/`ftell`/`rewind`/`fclose` over an in-memory
  `path -> bytes` map, seeded via the `vfs` option. This is what lets the unmodified loader
  work with no filesystem.
- **Host surface** — `write` → `onStdout`/`onStderr`, clock, `getenv` → null (so the agent's
  `chooseBrain` auto-selects the offline `ScriptedModel`).
- **Stubs** — curl is a benign no-op; sockets/pthreads trap loudly. Both disappear once the
  `when-wasm` guards (docs §5b) land.

⚠ **`isize` is 32-bit on wasm32.** Bridge functions backing `isize`-typed signatures
(`strlen`, `write`, `read`) must return plain **numbers**, not `BigInt` — see docs §5c.

## Run the demo

```sh
./build.sh && ./serve.sh      # then open the printed URL
```

- `demo.html` — the prototype: an xterm.js terminal beside the live viewer, both driven by
  the VM in the page. Try `hello`, `weather in Tokyo`, `research wasm`.
- `index.html` — just the viewer, over `demo.scry`.
- `node ui-smoke-wasm.mjs` — drives the whole thing in real headless Chrome.

## Green threads

`Thread.spawn` works, without pthreads. The VMThread is registered with its `run()` frame
pushed exactly as on native, then driven cooperatively: `scry_tick()` gives each unfinished
thread an instruction-budget slice, and `Clock.sleep` ends the slice so a paced worker
advances one step per tick. A page pumps this with `scry.startScheduler()`. This is sound
because a thread's whole execution state lives in its heap-allocated VMThread and `run-to`
never recurses for a Scry call — yielding is just returning from `run-to`.

**The native build is untouched**: the dispatch-loop hook is `(when-wasm (sched-should-yield) 0)`,
a compile-time macro that emits *nothing* off wasm32. Native keeps real OS threads
(DECISIONS #4b); cooperative scheduling is a wasm-target-only concession.

## Not yet

- **`readLine` is non-blocking** (returns `None` when no input is buffered) rather than
  suspending, so a program written as a blocking REPL loop exits at EOF instead of waiting.
  The browser demo therefore drives turns through the eval channel. Making a blocking
  `readLine` suspend and resume would need the same yield seam the scheduler already has.
- Packaging as a single self-contained file (P6) — today it is a directory of static assets.

## Gotcha worth knowing

Eval source buffers **must be NUL-terminated** — the lexer reads past `len`. Without the
terminator it reads whatever follows in memory, which looks fine while the allocator hands
out fresh zeroed pages and then produces bogus "unknown identifier" errors quoting
unrelated source once freed blocks start being recycled. `server.coil` documents the same
hazard on the native path.
