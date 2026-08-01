# COIL native-gap report for the agent harness

## Goal

The harness should contain no project-written C. All operating-system and network
facilities should be available as supported COIL libraries with typed errors,
explicit ownership, and portable behavior.

This report is based on the complete public surface of `native/harness_native.c`,
its COIL call sites, and the current COIL standard-library modules `io.coil`,
`fs.coil`, `thread.coil`, and `http.coil`.

## Executive finding

The blocker is not a general inability to express native operations in COIL. COIL
already supports C ABI externs, callbacks, variadic calls, opaque pointers, structs,
and direct POSIX calls. Its standard library demonstrates all of these. The problem
is missing hosted standard-library abstractions:

1. no outbound HTTP client;
2. no child-process API with redirected stdin/stdout, timeout-aware reads, and
   deterministic termination/reaping;
3. no standard environment, current-directory, clock, and sleep APIs;
4. incomplete reuse of the existing `io.coil` fd reader/writer APIs in this harness.

Only the first two are substantial. The small OS calls could be implemented in pure
COIL library modules immediately. Raw output can already be replaced with `io.coil`.
No new compiler intrinsic is required for the current native surface.

## Inventory and diagnosis

| Current C entry point | Why it exists | Actual COIL gap | Recommended home | Priority |
| --- | --- | --- | --- | --- |
| `harness_http_post` | Configure libcurl, submit a POST, collect the response, expose a chunk callback, and translate errors | `stdlib/http.coil` is currently only an HTTP/1.x **request parser**; there is no client transport, TLS, URL handling, timeout, headers, response ownership, or streaming contract | `stdlib/http_client.coil` or a separately versioned `coil-curl` package | P0 |
| `harness_codex_spawn` | Create two pipes, fork, redirect stdin/stdout, exec `codex app-server`, return a process handle | No typed process builder or child handle; callers would have to reproduce raw POSIX details and platform layouts | `stdlib/process.coil` | P0 |
| `harness_process_write_line` | Retry short/EINTR writes and add a newline | `io.coil` already has `fd-writer` and `write-all`; only a small `write-line` combinator is absent | `stdlib/io.coil` | P1; mostly already solved |
| `harness_process_read_line` | Poll with a timeout and return an allocated line | `io.coil` has `fd-reader` and fixed-buffer `read-line`, but no readiness/deadline API and no allocator-growing line reader | `stdlib/io.coil` plus `stdlib/poll.coil`, or process-owned read methods | P0 for timeout; P1 for allocation convenience |
| `harness_process_close` | Close both streams, terminate a live child, and reap it | No child lifecycle API (`try-wait`, `wait`, `terminate`, close semantics) | `stdlib/process.coil` | P0 |
| `harness_now_ms` | Read `CLOCK_REALTIME` and convert to milliseconds | No standard clock module | `stdlib/time.coil` | P1 |
| `harness_sleep_ms` | Sleep, retrying on `EINTR` | No standard sleep API | `stdlib/time.coil` | P1 |
| `harness_getenv` | Read provider credentials | No standard environment API or explicit borrowed/owned string contract | `stdlib/process_env.coil` or `stdlib/os.coil` | P1 |
| `harness_current_dir` | Obtain a dynamically sized cwd | `fs.coil` handles files but does not expose cwd | `stdlib/fs.coil` | P1 |
| `harness_write_fd` | Perform a complete raw write | Already covered by `io.coil`'s `fd-writer` + `write-all` | Harness migration only | P1 |
| `harness_buffer_free` | Free buffers returned by the shim | Artifact of the C ownership boundary, not an independent capability | Delete after the other migrations | — |

## Detailed requirements

### 1. Outbound HTTP client

The existing module named `http.coil` parses inbound HTTP/1.x request heads. It does
not make requests. The harness needs an HTTPS client with:

- method, URL, ordered/repeated headers, and a byte body;
- connect and total deadlines;
- status code and response body;
- transport errors distinct from HTTP status responses;
- optional incremental body delivery with backpressure/cancellation semantics;
- allocator-explicit owned output;
- TLS certificate verification enabled by default;
- cleanup on success, callback abort, allocation failure, and transport failure.

A libcurl-backed COIL package is the shortest route. It can be written in COIL using
extern declarations, but directly exposing `curl_easy_setopt` is a poor public API:
its option-dependent variadic argument types and C numeric constants are easy to use
incorrectly. Put those details behind typed COIL functions such as
`set-url`, `set-timeout`, `set-headers`, and `perform`.

The application should depend on that package through `Coil.toml`; it should not own
a C adapter. A future socket/TLS-native client can replace the backend without
changing the public request/result contract.

Suggested shape (illustrative, not mandated):

```coil
(defstruct ClientRequest
  [(method Method)
   (url (slice u8))
   (headers (slice Header))
   (body (slice u8))
   (connect-timeout-ms i64)
   (timeout-ms i64)])

(defstruct ClientResponse [(status i64) (headers (slice Header)) (body (slice u8))])
(defsum HttpError (InvalidRequest) (Transport [(code i64) (message (slice u8))]) (OutOfMemory))

(defn request [(a (ptr Allocator)) (request (ptr ClientRequest))]
  (-> (Result ClientResponse HttpError)) ...)
```

Acceptance tests should cover binary bodies, repeated headers, empty responses,
non-2xx responses as successful transport results, timeout, connection failure,
callback abort, and repeated calls under leak detection.

### 2. Child processes and pipes

The harness needs to launch one executable without a shell, pipe bytes to its stdin,
read bytes from stdout with a timeout, close stdin, and guarantee reaping. A COIL
process package should expose behavior rather than raw `fork` bookkeeping.

Required operations:

- executable plus an argv slice; never shell interpolation by default;
- configurable inherited/piped/null stdin, stdout, and stderr;
- `spawn`, `stdin-writer`, `stdout-reader`, `try-wait`, `wait`, `terminate`, and
  `kill`;
- typed `ExitStatus` distinguishing exit code and signal;
- idempotent stream closure and child cleanup;
- readiness or deadline-aware reads;
- correct close-on-exec behavior and cleanup after partial pipe/spawn failure;
- macOS and Linux implementations selected at compile time.

Prefer `posix_spawn` where supported. It avoids doing substantial work between
`fork` and `exec` in a multithreaded program. If a fallback uses `fork`, document and
test the restrictions. The public API should compose with existing `io.Reader` and
`io.Writer`, so process code does not need separate line-reading and write-all stacks.

The current shim merges `close`, terminate-if-running, and wait. The standard API
should make the policy explicit: closing a handle should not surprisingly terminate
a child unless the handle was configured with kill-on-drop/close semantics.

Acceptance tests should cover argv fidelity (spaces and empty args), stdin/stdout
round trips, stderr policy, EOF, read timeout, missing executable, normal exit,
signal termination, already-exited children, repeated close/wait, and concurrent
spawn from a multithreaded program.

### 3. Readiness, deadlines, and line I/O

`io.coil` already supplies caller-owned fd readers/writers, short-read handling,
`write-all`, and fixed-capacity `read-line`. What is missing for the Codex protocol is:

- wait for fd readability until a deadline;
- distinguish timeout, EOF, and syscall error;
- optionally grow a line buffer with an allocator;
- define what happens when a line exceeds a configured maximum.

Do not make line reading process-specific. Add a reusable readiness/deadline layer and
an allocator-backed line helper that accepts any `Reader` where possible. For an fd,
the readiness implementation may use `poll`/`ppoll` (or the platform equivalent).

### 4. Time

Add a hosted `time.coil` library. At minimum it should provide:

- wall time for event timestamps;
- monotonic time for elapsed time and deadlines;
- duration-based sleep that resumes after interruption;
- overflow-safe unit conversions;
- typed failure instead of returning `0` on error.

The present name/comment calls `harness_now_ms` a timestamp and its implementation
uses `CLOCK_REALTIME`. Keep wall and monotonic clocks distinct: wall time is suitable
for event records; monotonic time is required for timeout accounting.

### 5. Environment and cwd

Add small hosted APIs with explicit lifetimes:

- `env-get-borrowed(name)` if the result follows libc environment lifetime rules;
- preferably `env-get(a, name) -> Option owned-string` for ordinary application use;
- `current-dir(a) -> Result owned-string OsError` with no fixed path limit.

The API must document concurrent environment mutation. An owned result is safer for
provider configuration and makes the allocator boundary visible.

### 6. Raw output

No COIL feature is missing. Replace `harness_write_fd` with a caller-owned
`io.Writer` created by `fd-writer`, followed by `write-all`. A small standard
`write-line` helper would reduce repetition, but is not necessary to remove C.

One quality issue remains in `io.coil`: its fd backend currently places the negative
return value in `Errno` rather than capturing the platform `errno`. Correct error
capture should be addressed before relying on detailed I/O diagnostics.

## Compiler work versus library work

### Required compiler work

None has been demonstrated by this harness. COIL already expresses the required ABI
shapes, including callbacks and variadic externs. Do not add compiler intrinsics for
ordinary hosted OS operations.

### Required standard-library/package work

- outbound HTTP client package;
- process package;
- fd readiness/deadline support;
- time package;
- environment and cwd helpers;
- accurate `errno` capture and typed OS errors;
- optional `io.write-line` and growing `io.read-line-alloc` conveniences.

### Portability concern worth fixing centrally

Direct declarations of libc structs and constants are platform-sensitive. The COIL
library currently handles some constants with `target-os`, but types such as
`timespec`, `pollfd`, `pid_t`, and spawn file-action storage should not be recopied by
applications. Keep their layouts inside supported COIL platform modules or generated
bindings. This is an argument for a standard package boundary, not for project C.

## Recommended delivery order

1. Add/fix `io.write-line`, errno reporting, and allocator-backed line reading.
2. Add `time.coil`, environment access, and `fs.current-dir`; migrate those harness
   calls immediately.
3. Add readiness/deadline support.
4. Add `process.coil`; migrate the Codex App Server adapter to it.
5. Add the outbound HTTP client; migrate all direct-provider adapters.
6. Delete `src/infra/native.coil`, `native/harness_native.c`, and
   `native/harness_native.h`; remove `[cc]`, the native include path, and the
   application-owned C source from `Coil.toml`.

HTTP and process work can proceed independently. The deletion gate is a clean
`coil verify` with no project `.c`/`.h` files and no `[cc].sources` entry.

## Definition of done

- The harness repository contains no authored C or C header files.
- `Coil.toml` does not compile application-owned native sources.
- Providers use a supported outbound HTTP client API.
- Codex uses a supported process API composed with standard COIL I/O.
- Time, environment, cwd, and output use COIL standard-library APIs.
- Timeout tests use monotonic deadlines.
- Error results preserve actual OS/transport error codes and messages.
- Linux and macOS CI exercise the hosted APIs.
- `coil verify` and the harness test suite pass without the native shim.

## Evidence in this repository

- `Coil.toml` explicitly compiles `native/harness_native.c` and links libcurl.
- `src/infra/native.coil` declares all eleven C bridge operations in one module.
- `src/infra/http.coil` wraps the C response buffers and manually copies/frees them.
- `src/providers/codex_app_server.coil` depends on the shim for the entire child
  process lifecycle and timeout-aware JSONL transport.
- `src/main.coil` uses the shim for raw output and cwd.
- provider and runtime modules use the shim for credentials and timestamps.
- `tests/tool_executor_test.coil` uses the shim only to sleep.

The current architecture document accurately calls the C code a narrow bridge, but
“not yet ergonomic” should be read as “not packaged as safe standard COIL APIs,” not
as “impossible to implement in COIL.”
