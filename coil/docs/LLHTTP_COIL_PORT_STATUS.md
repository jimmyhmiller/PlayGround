# llhttp 9.4.3 → Coil port status

Last updated: 2026-08-01

## Current handoff / stopped state

Work is intentionally stopped at a clean verification boundary so another task can use the repository without a background build running.

- `field`, `load`, and `store!` are now direct `coil.core` primitive aliases in `src/compiler/prelude.coil`; `/tmp/coil-rb2 run tests/compiler/features/core_load_store.coil` passes.
- `coil.http.server` has been migrated from the native shim to `coil.llhttp`. Its request/header API remains zero-copy, decoded bodies are allocator-owned, and `request-free` releases the pure-Coil allocations.
- `/tmp/coil-rb2 check src/stdlib/http_server.coil` passes.
- `/tmp/coil-rb2 test tests/stdlib_parsers_test.coil` passes all 8 tests.
- `/tmp/coil-rb2 test tests/http_server_pure_test.coil` passes.
- Self-host fixpoint, LLVM self-build, snapshot/IR/ARM64/runtime/CLI/metaprogram/Wasm gates all pass with the core aliases. `python3 scripts/dev.py build full` completed successfully and installed the verified artifact at `build/bin/coil`.
- The verified artifact was installed globally at `/Users/jimmyhmiller/.cargo/bin/coil`; its SHA-256 matches `build/bin/coil` (`657c61e3f7aa65e12724af31ab9ca0ee7f96a3177db497f7983b8e1a3e655fbe`). The previous global binary is backed up as `/Users/jimmyhmiller/.cargo/bin/coil.pre-llhttp-port-20260801`.
- No build or test process from this work remains running.

Resume in this order:

1. Remove the compiler driver's automatic native-llhttp symbol detection/linking and the native server shim/build tests, while retaining a clearly test-only upstream oracle path for differential checks.
2. Expand differential coverage to responses, adversarial chunk schedules, callbacks, and the upstream corpus.
3. Complete the llhttp 9.4.3 exported API audit.
4. Re-run every repository gate after those remaining port changes and update this document with final evidence.

## Objective

Port the pinned llhttp 9.4.3 parser completely to Coil by adding a Coil backend for llparse. The finished port must:

- contain no native llhttp parser dependency;
- preserve llhttp's public parser/settings interface and streaming behavior closely enough for existing callers;
- preserve `coil.http.server` behavior;
- match the upstream C implementation through a differential C-versus-Coil harness over the upstream llhttp suite; and
- pass the repository's complete verification gates.

This objective is **not complete yet**. The worktree currently contains a functional first version of the generator, generated state machine, Coil runtime, public wrapper, nine smoke/runtime tests, and an initial four-case C-versus-Coil differential gate. Full upstream corpus adaptation, API-completeness work, native removal, and full repository verification remain.

## Current implementation

### Generator

[`scripts/llhttp/generate-coil.ts`](../scripts/llhttp/generate-coil.ts) registers Coil implementations for the llparse frontend node/code/transform classes, loads the parser graph from an exact llhttp checkout, and emits structured Coil source.

The generator currently:

- requires llhttp version `9.4.3`;
- uses the exact `llparse` dependency installed by that llhttp checkout;
- traverses and assigns dense numeric IDs to the complete parser graph;
- emits a resumable state dispatcher;
- lowers consume, empty, error, pause, invoke, sequence, single-byte, span-start, span-end, and table-lookup nodes;
- lowers the llparse code operations used by llhttp;
- preserves parser state, sequence progress, span progress, and input position across calls; and
- emits callback dispatch for all llhttp external callbacks represented in the graph.

Generator support files are in `scripts/llhttp/`:

- `package.json` and `package-lock.json`
- `tsconfig.json`
- `README.md`

`npm run check` in this directory passes after the numeric sequence-emission edit.

### Generated parser

[`src/stdlib/llhttp_generated.coil`](../src/stdlib/llhttp_generated.coil) is the generated llhttp state machine. The inspected upstream graph contains:

- 573 nodes;
- 254 resumable targets;
- 2 consume nodes;
- 6 empty nodes;
- 118 error nodes;
- 161 invoke nodes;
- 33 pause nodes;
- 60 sequence nodes;
- 97 single nodes;
- 63 span-end nodes;
- 17 span-start nodes; and
- 16 table-lookup nodes.

The generated module passes after regeneration from the numeric sequence-byte emitter:

```sh
build/bin/coil check src/stdlib/llhttp_generated.coil
```

### Coil runtime representation

[`src/stdlib/llhttp_types.coil`](../src/stdlib/llhttp_types.coil) defines the current `Parser` and `Settings` representations and the handwritten operations required by generated code, including:

- parser reset/zeroing;
- generated property loads and stores;
- ASCII transforms;
- error recording;
- callback invocation and callback-specific helpers;
- before/after-headers behavior;
- message-completion behavior;
- keep-alive and EOF decisions; and
- span flushing across input boundaries.

Callback fields are currently stored as erased pointers and cast to the appropriate function-pointer type at invocation. This works around the lack of a direct nullable/function-pointer abstraction in the current Coil interface, but requires thorough ABI and callback-error differential coverage.

### Public API wrapper

[`src/stdlib/llhttp.coil`](../src/stdlib/llhttp.coil) provides an initial llhttp-shaped Coil API:

- parser initialization and reset;
- settings initialization;
- execute and finish;
- pause and resume;
- error/status/method/version getters;
- leniency setters; and
- simple/span callback setters.

It passes:

```sh
build/bin/coil check src/stdlib/llhttp.coil
```

This is not yet proof of API parity. Known API work includes reviewing every exported llhttp 9.4.3 symbol and behavior, deciding on a canonical public home for `Parser`/`Settings`, and checking exact integer width/overflow behavior. `finish` now checks the parser error field, handles safe/safe-with-callback/unsafe states like upstream, and has smoke coverage for EOF-delimited responses and incomplete messages.

The multiply/add helper now follows llparse's declared unsigned field widths: `content_length` accepts the complete `uint64_t` range and `status_code` is bounded to `uint16_t`. Exact-limit and first-overflow cases are covered directly. Null callback settings are accepted like upstream, and callback pause/resume has coverage including the resume offset.

Span callback error handling matches the upstream wrapper contract: `-1` becomes `HPE_USER` with a callback-specific reason, while explicit error codes preserve a reason set by the application. Generated span failure paths no longer overwrite callback reasons.

### Smoke tests

[`tests/llhttp_coil_test.coil`](../tests/llhttp_coil_test.coil) currently exercises:

- a request delivered in one buffer;
- the same streaming machinery with one-byte chunks;
- method, URL, header-field, header-value, and body span callbacks; and
- message completion and selected parser getters.
- EOF completion for a response body delimited by connection close; and
- invalid EOF detection for an incomplete request.
- unsigned multiply/add boundaries for content length and status code;
- parsing with a null settings pointer; and
- callback pause, error position, and resume.

Generated llparse sequence matching now emits numeric byte cases rather than using byte strings, and test requests use `StrBuf` with explicit byte values 13 and 10. This avoids depending on Coil source-string escape interpretation for CRLF. All nine current runtime tests pass.

[`tests/llhttp_differential_test.coil`](../tests/llhttp_differential_test.coil) compares structured native and Coil results in one test process. [`scripts/tests/llhttp-differential.sh`](../scripts/tests/llhttp-differential.sh) builds/locates the pinned native oracle and runs the gate. The initial cases cover content-length, chunked with trailers, incomplete input, and duplicate content-length; all four pass. This is a seed harness, not yet the complete upstream corpus adapter.

## Upstream source used during development

Development used an exact checkout at:

```text
/private/tmp/coil-llhttp-v9.4.3-source
```

The checkout was made from the llhttp `v9.4.3` tag, resolving to commit beginning `45c869`, and its npm dependencies were installed. This temporary checkout is not a durable repository input. The final generation/test workflow needs a reproducible pinned-source mechanism rather than relying on this path existing.

## Important generated graph details

The generator observed these llhttp parser properties in order:

| ID | Name | Type |
|---:|---|---|
| 0 | `content_length` | `i64` |
| 1 | `type` | `i8` |
| 2 | `method` | `i8` |
| 3 | `http_major` | `i8` |
| 4 | `http_minor` | `i8` |
| 5 | `header_state` | `i8` |
| 6 | `lenient_flags` | `i16` |
| 7 | `upgrade` | `i8` |
| 8 | `finish` | `i8` |
| 9 | `flags` | `i16` |
| 10 | `status_code` | `i16` |
| 11 | `initial_message_completed` | `i8` |
| 12 | `settings` | pointer |

The graph exposes 28 callback entry points, from `on_message_begin` through `on_reset`. The current runtime has numeric dispatch for these callbacks, but the full callback contract has not yet been proven against C.

## Known correctness gaps

The following are known incomplete or unverified areas:

- Remaining pause/resume and callback-return/error paths need differential validation.
- Response parsing, HTTP-both mode, upgrades, CONNECT, chunk extensions, trailers, pipelining, EOF-delimited messages, lenient modes, and invalid-message diagnostics are not comprehensively tested.
- Error codes, reasons, and exact error positions are not yet compared against upstream C.
- `Parser` and `Settings` currently live in `llhttp_types.coil`; imports do not automatically re-export those types through `llhttp.coil`, so the final public module organization is unresolved.
- The existing `coil.http.server` integration has not yet been switched to and verified against this parser.
- The native shim at [`scripts/native/llhttp_shim.c`](../scripts/native/llhttp_shim.c) still exists.
- Build scripts and dependency declarations have not yet been audited to remove all native llhttp linkage.
- The differential harness currently covers four curated request cases only; response mode, chunk schedules, exact reasons, and the complete upstream corpus remain.
- The upstream llhttp suite has not been run against Coil.
- The repository's runtime/CLI/compiler/full rebootstrap gates have not been run for this port.

## Verification evidence so far

Completed checks after the latest edits:

```text
scripts/llhttp: npm run check                                      PASS
build/bin/coil check src/stdlib/llhttp_generated.coil             PASS
build/bin/coil check src/stdlib/llhttp.coil                       PASS
build/bin/coil test tests/llhttp_coil_test.coil                   PASS (9 tests)
scripts/tests/llhttp-differential.sh                              PASS (4 cases)
```

`build/bin/coil` and the globally installed `/Users/jimmyhmiller/.cargo/bin/coil` now contain the same verified compiler artifact.

## Next work, in order

1. Audit and correct the handwritten runtime against llhttp 9.4.3 semantics, especially overflow, callbacks, pause/resume, errors, and error positions.
2. Expand the structured C-versus-Coil differential gate to response mode and adversarial chunk schedules.
3. Adapt the complete upstream llhttp corpus into the differential harness.
4. Close every API/export mismatch identified by the upstream header and tests.
5. Replace the existing HTTP server's native parser path with the Coil implementation and run its behavior tests.
6. Remove the native shim, native llhttp sources/libraries, linkage, and build dependencies; verify with repository-wide searches and clean builds.
7. Run the complete repository verification, including `python3 scripts/dev.py build full`, and resolve or correctly re-bless any required compiler snapshots.

## Completion standard

This port should only be called complete when all of the following have direct evidence:

- generated parser source comes reproducibly from pinned llhttp 9.4.3/llparse inputs;
- all relevant upstream parser states and callback paths are supported;
- the public Coil API covers the required llhttp 9.4.3 interface;
- C and Coil traces agree for the full upstream suite and required streaming chunk schedules;
- `coil.http.server` tests pass using only the Coil parser;
- repository searches and clean builds show no native llhttp dependency remains; and
- all full repository gates pass.
