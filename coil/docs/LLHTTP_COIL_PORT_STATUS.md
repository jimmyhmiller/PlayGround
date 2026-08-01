# llhttp 9.4.3 → Coil port status

Last updated: 2026-08-01

## Objective

Port the pinned llhttp 9.4.3 parser completely to Coil by adding a Coil backend for llparse. The finished port must:

- contain no native llhttp parser dependency;
- preserve llhttp's public parser/settings interface and streaming behavior closely enough for existing callers;
- preserve `coil.http.server` behavior;
- match the upstream C implementation through a differential C-versus-Coil harness over the upstream llhttp suite; and
- pass the repository's complete verification gates.

This objective is **not complete yet**. The worktree currently contains a functional first version of the generator, generated state machine, Coil runtime, public wrapper, and two smoke tests. Differential testing, API-completeness work, native removal, and full repository verification remain.

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

`npm run check` in this directory passed before the latest sequence-emission edit. It needs to be rerun.

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

The generated module previously passed:

```sh
build/bin/coil check src/stdlib/llhttp_generated.coil
```

The generator was subsequently changed so sequence byte matching no longer relies on Coil string escape interpretation. The generated file has **not yet been regenerated from that latest generator edit**, so the check must be repeated after regeneration.

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

It previously passed:

```sh
build/bin/coil check src/stdlib/llhttp.coil
```

This is not yet proof of API parity. Known API work includes reviewing every exported llhttp 9.4.3 symbol and behavior, correcting `finish`, deciding on a canonical public home for `Parser`/`Settings`, and checking exact integer width/overflow behavior.

### Smoke tests

[`tests/llhttp_coil_test.coil`](../tests/llhttp_coil_test.coil) currently exercises:

- a request delivered in one buffer;
- the same streaming machinery with one-byte chunks;
- method, URL, header-field, header-value, and body span callbacks; and
- message completion and selected parser getters.

The first run reached the generated parser and failed at the request-line CRLF with `HPE_INVALID_VERSION` because Coil string literals did not interpret `\r\n` as CR/LF bytes. The observed failure was:

```text
err=9 off=20 byte=114 reason=Expected CRLF after version
```

Two fixes are in progress:

1. Generated llparse sequence matching now emits numeric byte cases rather than using byte strings. This avoids depending on source-string escaping for binary parser constants.
2. Test requests are being built with `StrBuf` and explicit byte values 13 and 10.

The test fixture conversion is incomplete: the first request still embeds `\r\n` between its two header lines, so that separator must also be emitted with the explicit CRLF helper. Tests have not been rerun after these edits.

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

- The latest generator edit has not been regenerated or typechecked.
- The smoke-test request builder still has one escaped multi-header separator.
- The smoke tests are not green yet.
- `finish` currently consults the wrong state in at least one path and needs correction against upstream behavior.
- `content_length` multiply/add behavior is not yet guaranteed to reproduce exact unsigned 64-bit overflow and error behavior.
- Every pause/resume and callback-return/error path needs differential validation.
- Response parsing, HTTP-both mode, upgrades, CONNECT, chunk extensions, trailers, pipelining, EOF-delimited messages, lenient modes, and invalid-message diagnostics are not comprehensively tested.
- Error codes, reasons, and exact error positions are not yet compared against upstream C.
- `Parser` and `Settings` currently live in `llhttp_types.coil`; imports do not automatically re-export those types through `llhttp.coil`, so the final public module organization is unresolved.
- The existing `coil.http.server` integration has not yet been switched to and verified against this parser.
- The native shim at [`scripts/native/llhttp_shim.c`](../scripts/native/llhttp_shim.c) still exists.
- Build scripts and dependency declarations have not yet been audited to remove all native llhttp linkage.
- No upstream differential harness exists yet.
- The upstream llhttp suite has not been run against Coil.
- The repository's runtime/CLI/compiler/full rebootstrap gates have not been run for this port.

## Verification evidence so far

Completed checks before the most recent edits:

```text
scripts/llhttp: npm run check                                      PASS
build/bin/coil check src/stdlib/llhttp_generated.coil             PASS
build/bin/coil check src/stdlib/llhttp.coil                       PASS
build/bin/coil test tests/llhttp_coil_test.coil                   RUNS, 2 failing tests
```

The two test failures were caused by request fixtures containing literal `r`/`n` bytes instead of CR/LF at the first observed failure point. This diagnosis does not establish that the parser is otherwise correct; green reruns and broader tests are still required.

Use `build/bin/coil` for this work. The globally installed `coil` executable is older than the worktree compiler and rejects syntax/features used by the current source.

## Next work, in order

1. Finish the explicit-CRLF test builder so no request fixture depends on string escapes.
2. Regenerate `llhttp_generated.coil` from the numeric sequence-byte emitter.
3. Rerun TypeScript checking, Coil checking, and both streaming smoke tests; diagnose parser/runtime defects until green.
4. Audit and correct the handwritten runtime against llhttp 9.4.3 semantics, especially finish/EOF, overflow, callbacks, pause/resume, errors, and error positions.
5. Build a machine-readable trace format and two runners: pinned upstream C llhttp and Coil.
6. Adapt the complete upstream llhttp corpus into the differential harness and compare results under whole-buffer and adversarial chunk schedules.
7. Close every API/export mismatch identified by the upstream header and tests.
8. Replace the existing HTTP server's native parser path with the Coil implementation and run its behavior tests.
9. Remove the native shim, native llhttp sources/libraries, linkage, and build dependencies; verify with repository-wide searches and clean builds.
10. Run the complete repository verification, including `python3 scripts/dev.py build full`, and resolve or correctly re-bless any required compiler snapshots.

## Completion standard

This port should only be called complete when all of the following have direct evidence:

- generated parser source comes reproducibly from pinned llhttp 9.4.3/llparse inputs;
- all relevant upstream parser states and callback paths are supported;
- the public Coil API covers the required llhttp 9.4.3 interface;
- C and Coil traces agree for the full upstream suite and required streaming chunk schedules;
- `coil.http.server` tests pass using only the Coil parser;
- repository searches and clean builds show no native llhttp dependency remains; and
- all full repository gates pass.
