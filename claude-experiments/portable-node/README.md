# portable-node

Node.js's own JavaScript standard library, lifted verbatim from `nodejs/node@v24.3.0`, running on QuickJS on top of a small host interface. The Rust binary in `src/` is one implementation of that interface. The point of the project is that it is only one: everything above the host interface is portable JavaScript, so a host written in C, Go, Zig, Java, or another JS engine's embedding API gets the same `http`, `net`, `fs`, `stream`, `buffer`, `crypto`, and real npm packages (Express 5 runs) without touching the JS.

This document specifies what a host has to provide.

## Running it

```bash
cargo build --release

# Run a script, node-style.
./target/release/portable-node app.js

# Run a script that requires npm packages: point at the project whose
# node_modules should be searched (npm install express in there first).
PORTABLE_NODE_PROJECT=/path/to/project ./target/release/portable-node /path/to/project/app.js

# Built-in self-test: smokes, event-loop round-trips, and 41 of Node's own tests.
./target/release/portable-node            # or --self-test
```

`app.js` is ordinary CommonJS. `require` implements the full Node resolution algorithm, including `node_modules` walking and package `exports`, so `require('express')` works.

`PORTABLE_NODE_PROJECT` exists because the resolver's search roots come from the script's own directory; setting it adds one more global root. `PORTABLE_NODE_SERVE=<seconds>` keeps the self-test's embedded HTTP server alive so you can curl it (0 means forever).

## The layering

```text
  user code, npm packages          app.js, express, body-parser, ...
  ────────────────────────────────────────────────────────────────────
  Node's real lib sources          js/node-src/*.js  (verbatim upstream)
  ────────────────────────────────────────────────────────────────────
  bootstrap + binding shims        js/bootstrap.js
    primordials, CJS resolver,       __binding/tcp_wrap, stream_wrap,
    event-loop hooks, polyfills      http_parser, fs, uv, ...  (portable JS)
  ────────────────────────────────────────────────────────────────────
  THE HOST INTERFACE               globalThis.__host.*   <-- port this
  ────────────────────────────────────────────────────────────────────
  host implementation              src/*.rs (Rust today)
```

Node's internals do not call the OS directly. They call `internalBinding('tcp_wrap')`, `internalBinding('fs')`, and so on. Those bindings are C++ in real Node; here they are JavaScript modules in `js/bootstrap.js` that call `__host.*`. That is the seam. A new host reimplements `__host.*` and nothing else.

Two things sit slightly outside that rule and are covered in "Native bindings" below.

## The host contract

Everything hangs off `globalThis.__host`, installed before `bootstrap.js` is evaluated. All of it is synchronous and single-threaded. No host function may call back into JS from another thread; async work is delivered only through `io.poll`.

### `__host.time`

| Call | Returns |
| --- | --- |
| `now_ms()` | milliseconds since the Unix epoch, as a double |
| `sleep_ms(ms)` | blocks; used only when there is no I/O to wait on |

### `__host.process`

Data properties read once at startup: `platform` (Node's spelling: `darwin`, `linux`, `win32`), `arch` (`x64`, `arm64`, ...), `pid`, `env` (a plain object snapshot), `argv` (array; `argv[0]` is the binary, `argv[1]` the script).

Functions: `cwd()`, `chdir(path) -> bool`, `exit(code)`, `hrtime_ns()` (monotonic nanoseconds from an arbitrary fixed base), `stdout_write(v) -> bool`, `stderr_write(v) -> bool`. The two write functions accept a string (written as UTF-8) or a `Uint8Array` (written verbatim) and must flush.

### `__host.file`

Synchronous POSIX-shaped file I/O. `node:fs`'s sync API is built directly on this; the async API is the sync one deferred onto the microtask queue.

```js
open(path, flags, mode) -> fd
close(fd)
read(fd, buf, offset, length, position) -> bytesRead      // position < 0 means "current"
write(fd, buf, offset, length, position) -> bytesWritten
stat(path) / lstat(path) / fstat(fd) -> stat object
readdir(path) -> [{ name, type }]                          // type: 1 file, 2 dir, 3 symlink, 4 other
realpath(path) -> string
unlink(path) / mkdir(path, mode) / rmdir(path) / rename(from, to) / access(path, mode)
read_to_string(path) -> string
exists(path) / is_file(path) / is_dir(path) -> bool         // never throw
flags: { O_RDONLY, O_WRONLY, O_RDWR, O_CREAT, O_EXCL, O_TRUNC, O_APPEND,
         O_NOFOLLOW, O_DIRECTORY, O_NONBLOCK, F_OK, R_OK, W_OK, X_OK }
```

The stat object carries `dev, ino, mode, nlink, uid, gid, rdev, size, blksize, blocks` plus `atime_ms, mtime_ms, ctime_ms, birthtime_ms`.

**Error convention, and it matters:** a failing call throws a JS `Error` with `.code` (`'ENOENT'`, `'EACCES'`, ...), `.syscall`, `.path`, and a negative `.errno`. Node's `fs` internals read those fields directly. A host that throws a bare `Error`, or returns `-1`, will produce failures that surface far from their cause.

The three non-throwing predicates (`exists`, `is_file`, `is_dir`) exist for the CommonJS resolver, which probes dozens of paths per `require` and must not pay for exceptions.

### `__host.tcp` and `__host.io`

This is the interesting one. The design is **completion-based, not readiness-based**, so a host can implement it over epoll/kqueue (as the Rust one does, via mio), and equally over IOCP, io_uring, or a language runtime's own async primitives.

```js
create_tcp() -> handle                              // reserves an opaque id
listen(handle, ip, port, backlog) -> 0 | -errno     // sync: bind and listen together
connect(handle, ip, port) -> op_id                  // async
accept(handle) -> op_id                             // async, one connection per call
read(handle, buf, off, len) -> op_id                // async
write(handle, buf, off, len) -> op_id               // async
shutdown(handle, how) -> op_id                      // async
close(handle)                                       // sync
set_no_delay(handle, on) / set_keep_alive(handle, on, delay)
local_addr(handle) / peer_addr(handle) -> { ip, port, family }   // family: 4 | 6 | 0

io.poll(timeout_ms) -> [completion, ...]            // THE ONLY BLOCKING CALL
io.has_pending() -> bool
io.cancel(op_id)                                    // best effort
```

A completion is `{ op_id, status, kind, ...payload }` where `status` is `0` or a negative errno, and `kind` is one of:

| kind | payload |
| --- | --- |
| `connect` | none |
| `accept` | `handle` of the new connection |
| `read` | `n` bytes read; `n === 0` with `status === 0` means EOF |
| `write` | `n` bytes written |
| `shutdown` | none |

Invariants a host must preserve:

- **Opaque integer handles.** No file descriptor ever reaches JS.
- **One blocking primitive.** Every call except `io.poll(timeout_ms)` returns immediately. The driver's whole idle time is inside `poll`.
- **Completions are delivered only from `poll`,** on the JS thread, never from a worker.
- **`listen` binds and listens in one step,** and returns a negative errno rather than throwing. `net.js` probes the unspecified IPv6 address (`::`) first when no host is given, so a host must handle `::` and `0.0.0.0` and must return an errno, not a crash, when the bind fails. (Getting this wrong is exactly how `app.listen(3000)` stayed broken here while `app.listen(3000, '127.0.0.1')` worked.)
- **One read op in flight per handle** is all Node's stream layer will ask for; writes may queue and must complete FIFO.

`__host.dns.lookup(name, family)` is declared but not implemented; see "Gaps".

### `__host.http.parser`

HTTP/1.x parsing is a host primitive rather than JS, because every ecosystem already has a hardened parser and a hand-rolled JS one is a liability. Rust uses `httparse` plus a body-framing state machine; the equivalents are `llhttp` in C, `net/http` in Go, `h11` in Python.

```js
create(kind) -> handle                    // kind: 'request' | 'response'
execute(handle, buf, off, len) -> { nread, events: [...], error? }
finish(handle) -> { events: [...], error? }   // signals EOF for identity bodies / HTTP/1.0
reset(handle, kind)
free(handle)
```

Events:

```js
{ kind: 'headers', method, url, status_code, status_message,
  http_major, http_minor, headers: [name, value, name, value, ...],
  upgrade, should_keep_alive }
{ kind: 'body', data: Uint8Array }
{ kind: 'message_complete' }
```

The parser must auto-reset to the header state after `message_complete` when the connection is keep-alive, so pipelined requests parse without JS intervention. It is also the security boundary: reject conflicting `Content-Length` values, reject `Transfer-Encoding` combined with `Content-Length`, reject unknown transfer codings, validate header characters per RFC 7230, and bound the header block.

The JS side (`__binding/http_parser`) is a thin translation from these events into the slots Node's `_http_common.js` expects (`kOnHeadersComplete`, `kOnBody`, `kOnMessageComplete`).

### `__host.crypto`

```js
random_bytes(n) -> Uint8Array           // CSPRNG, must not be seedable or predictable
hash(algorithm, data) -> Uint8Array
hmac(algorithm, key, data) -> Uint8Array
timing_safe_equal(a, b) -> bool         // constant time
supported_hashes() -> string[]
```

Algorithms currently: `sha1`, `sha224`, `sha256`, `sha384`, `sha512`, `md5`. Everything else in `node:crypto` (`createHash` streaming, `pbkdf2`, `randomUUID`, `randomInt`, `webcrypto.getRandomValues`) is built in JS on top of these four calls.

### `__host.zlib`

```js
deflate_raw(bytes, level) / inflate_raw(bytes)
deflate(bytes, level) / inflate(bytes)      // RFC 1950, zlib-wrapped
gzip(bytes, level) / gunzip(bytes)          // RFC 1952
crc32(bytes) -> u32
```

All one-shot. Streaming zlib is assembled in JS.

### `__host.os`

`hostname()`, `uptime()`, `totalmem()`, `freemem()`, `loadavg()`, `homedir()`, `tmpdir()`, `platform()`, `arch()`, `endianness()`, `osType()`, `osRelease()`, `osVersion()`, `availableParallelism()`, `cpus()`, `userInfo()`, `networkInterfaces()`, `getPriority()`, `setPriority()`. A host that cannot answer one of these should return a plausible constant rather than throw; `node:os` is queried casually by libraries that do not expect it to fail.

## The event loop is the host's job

`bootstrap.js` exposes the loop's moving parts but does not drive them. The host's driver owns the loop and must do exactly this, forever, until nothing is pending:

1. **Drain microtasks** (promise jobs plus `process.nextTick`). In Rust this is `rt.execute_pending_job()` in a loop. It must happen outside any borrow of the JS context.
2. **Ask what is pending**: `__eventLoopHasWork()` for timers, `__ioHasPending()` for I/O. If neither, the loop is done and the process can exit.
3. **Compute the wait**: `__eventLoopNextDueMs()` minus `__host.time.now_ms()`, or an arbitrary long wait if only I/O is pending.
4. **Block once**: `__ioDrain(timeoutMs)`, which calls `io.poll` and dispatches each completion to its registered JS callback.
5. **Fire due timers**: `__eventLoopFireDue()`.

One deviation to know about: `process.nextTick` is a promise microtask here, not a separate queue drained ahead of promises. Node runs all nextTicks before any promise callback. Nothing in the lifted sources has depended on that ordering so far, but code that does will behave differently.

Two rules that cost real debugging time here:

- **Do not re-enter the JS context while you hold it.** Driving the loop from inside a context borrow deadlocks. Read state, drop the borrow, drive, re-enter.
- **A callback that throws must not be swallowed silently.** `__ioDrain` reports the completion kind and the stack. Every masked exception in this codebase so far turned out to be a real bug wearing a `TypeError: not a function` costume.

## Native bindings that are not `__host`

Two `internalBinding` names are implemented natively rather than as portable JS over `__host`, and a new host has to supply them:

- **`buffer`** (`src/buffer_binding.rs`): the byte-level primitives `lib/buffer.js` calls, including the encode/decode slices (`utf8Slice`, `base64Slice`, `hexSlice`, ...), `fill`, `copy`, `compare`, `indexOf*`, `swap16/32/64`, `atob`/`btoa`, and `createUnsafeArrayBuffer`. These are portable in principle (some are already implemented in JS inside the binding) but they are the hot path for every stream and every HTTP response, so they belong in the host language.
- **`util` and `config`** (`src/util_binding.rs`): `getOwnNonIndexProperties`, `privateSymbols`, `isInsideNodeModules`, and `config.hasIntl = false`. Small, and mostly there to keep `util.inspect` and `buffer.js` off their ICU paths.

Anything else Node asks for via `internalBinding(name)` is resolved as the JS module `__binding/<name>` in `bootstrap.js`. If neither exists, the require throws a message naming the missing binding, which is the intended way to discover the next gap.

## What the JS engine has to provide

QuickJS is the reference target, so the bar is "ES2020 plus typed arrays", not "V8".

Required: `Proxy` and `Reflect`, typed arrays and `DataView`, `Symbol` and well-known symbols, generators and async generators, `WeakMap`/`WeakSet`, `Object.getOwnPropertyDescriptor` on built-in prototypes (primordials are built by uncurrying them), and a promise job queue the host can pump one job at a time.

Polyfilled by `bootstrap.js` when absent, so a host does not need them: `TextEncoder`/`TextDecoder`, `Error.captureStackTrace` with V8's `prepareStackTrace` CallSite API (depd, debug, and source-map-support all assume it), `Event`/`EventTarget`, `AbortController`/`AbortSignal`, `Promise.withResolvers`, `Array.fromAsync`.

Engine quirks worth knowing before porting: strings may contain lone surrogates that will not survive a round trip through a UTF-8 host string type, and module bodies are wrapped in a function whose parameters must be exactly Node's five (`exports, require, module, __filename, __dirname`), because npm packages contain top-level `var Buffer` and `const { internalBinding }` declarations that collide with any extra parameter name in strict mode. `primordials` and `internalBinding` are passed as globals for that reason.

## Rules that keep it portable

- Anything under `__binding/*` is portable JS calling `__host.*`. If you find yourself writing platform logic there, it belongs in the host.
- A JS facade over a portable library is fine (zlib, the HTTP parser) because every ecosystem has an equivalent. A JS reimplementation of something the platform should do is not.
- Stubs throw. A missing capability must fail loudly at the point of use, with the name of what is missing, never by returning a plausible zero.
- The host surface grows additively and stays small. Roughly 80 functions today across nine namespaces, a quarter of them `file.*`.

## Gaps

Honest list of what a host does not have to implement yet, because nothing above it works:

- **DNS.** `__host.dns.lookup` is a stub. The JS `dns` shim resolves IPv4 literals, `localhost`, and `::1`, and returns `ENOTFOUND` for everything else, so outbound requests to a hostname do not work. Servers are unaffected. Real `getaddrinfo` is the next host primitive.
- **UDP, TLS, child processes, worker threads, signals, file watching.** Not present at any layer.
- **Async file I/O** is sync I/O on the microtask queue, so a slow disk blocks the loop.
- **`crypto`** covers hashes, HMAC, and randomness. No ciphers, signing, key generation, X.509, or ECDH.
- **`os.networkInterfaces()`** returns loopback only, `os.cpus()` returns count-many placeholder entries, and `freemem()` on macOS is an approximation.
- **`process` is thin.** `env` is a snapshot and mutations are not written back to the OS. `version` is absent and `versions` is empty, which some packages sniff. `process.on` is a no-op, so there are no `uncaughtException`, `unhandledRejection`, or signal handlers.
- The `dns` shim still pushes debug entries into a global `_netLog` array on every lookup, a self-test artifact that grows without bound in a long-running server.
- Three of Node's own tests fail: `test-buffer-alloc` (engine-specific `RangeError` text), `test-buffer-fill`, and `test-string-decoder` (WHATWG UTF-8 error-resync edge cases).

## Layout

| Path | What |
| --- | --- |
| `src/host.rs` | installs `__host`; `os`, `process`, `file`, `zlib`, `time` |
| `src/io_loop.rs` | `__host.tcp` and `__host.io`, mio-backed, completion-based |
| `src/http_parser.rs` | `__host.http.parser`, httparse plus body framing |
| `src/host_crypto.rs` | `__host.crypto` |
| `src/buffer_binding.rs` | native `internalBinding('buffer')` |
| `src/util_binding.rs` | native `internalBinding('util')` and `('config')` |
| `src/main.rs` | script runner, self-test harness, the event-loop driver |
| `js/bootstrap.js` | primordials, all `__binding/*` and `internal/*` shims, CJS resolver, loop hooks, polyfills |
| `js/node-src/` | Node's lib sources, verbatim, plus the test files |
