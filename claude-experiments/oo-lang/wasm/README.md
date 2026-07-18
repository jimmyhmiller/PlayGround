# Scry in WebAssembly

The Scry VM — bytecode interpreter, generational GC, typechecker, compiler, and the live
eval channel — compiled to `wasm32-unknown-unknown` and running in a JS host. This is the
foundation for the in-browser prototype (viewer + agent TUI, no server, no native process).

Design + rationale: [`../docs/08-wasm-port.md`](../docs/08-wasm-port.md).

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

## Not yet

- **Threads** — `Thread.spawn` still references pthreads; the wasm build needs the
  cooperative green-thread scheduler (docs §3.2, P3).
- **`readLine`** — returns EOF; it must become a host callback that pumps the scheduler
  (docs §4) or background workers freeze at the prompt.
- **Uncrashable eval** — `setjmp` returns 0 and `longjmp` throws. The eval landing pad's
  non-local exit needs a wasm story (exception handling) before eval is panic-proof here.
- Viewer transport swap (P4), xterm + VFS + fake model end-to-end (P5), packaging (P6).
