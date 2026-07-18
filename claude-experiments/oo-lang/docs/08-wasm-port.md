# Scry — WASM Port: Architecture & Coil Change Spec

_Status: DESIGN. Written 2026-07-17. Goal: run the **entire** Scry system — VM, GC,
viewer, and the agent-TUI demo — as a self-contained WASM module in the browser, as an
embeddable prototype. No server, no native process, no real LLM. Backed by a full
source-level audit of the VM (`src/*.coil`), the viewer (`viewer/app.js`), and the Coil
self-hosted WASM finalizer (`coil/selfhost/src/wasm.coil`)._

This doc is the source of truth for the port. Where it names a `file:line`, that line was
read during the audit; treat it as a starting point, not a guarantee it hasn't moved.

---

## 1. The goal, precisely

An embeddable web page where:
- The Scry **bytecode VM + generational GC run in-browser** as one WASM module (compiled
  from the same `src/*.coil`, via `~/.cargo/bin/coil --target wasm32-unknown-unknown`).
- The **live viewer** (the existing React app) drives the VM directly — no HTTP.
- The **agent TUI** (`examples/assistant.scry`) runs against a **scripted/fake model**
  (the `ScriptedModel` that already exists in `agent/core.scry`), rendered into a
  **web terminal** (xterm.js).
- The demo moment survives: type into the terminal, watch `Message`/`Agent`/`Conversation`
  instances appear and climb in the viewer, invoke a method, hot-swap a body — all live.

**One pinned decision is knowingly relaxed for this target.** DECISIONS.md #4b says
concurrency is *real OS threads, explicitly not a cooperative turn-scheduler*. WASM in the
browser has no pthreads. So the **WASM build only** degrades language threads to a
cooperative green-thread scheduler. The **native build keeps real threads unchanged.**
This is a per-target concession, recorded here and not applied to native.

---

## 2. Architecture: two builds, one source

```
                          src/*.coil  (one codebase)
                                 │
             ┌───────────────────┴───────────────────┐
      native build                             wasm build (new)
   coil build (Coil.toml)          coil build --target wasm32-unknown-unknown
   - real pthreads                        - green-thread scheduler
   - libcurl HTTP                         - http = Coil stub (never called)
   - sockets + accept/conn threads        - NO sockets; eval-core exported directly
   - libc (malloc/write/poll/…)           - Coil-native libc shim + JS-bridge host imports
   - files/subprocess = real syscalls     - in-memory VFS + mock shell via JS bridge
```

The native path is untouched. The wasm path is selected by a build config (a
`--target`-gated module set, e.g. `main-wasm.coil` as an alternate entry, or `comptime`
target-detection that swaps `server.coil`/`http.coil` for wasm variants). The two differ
only in the **host boundary** modules; the VM, GC, typechecker, compiler, serializer, and
the whole language are shared byte-for-byte.

### The JS bridge (host boundary)

Reuses the proven model from `coil/web/` (`js-bridge.js` + `js.coil`): undefined
`extern … :cc c` symbols become `env.<name>` WASM imports; strings cross as
`(ptr u8)` + `i32` length; the module exports functions JS calls in. New app-specific
imports the page provides:

| Import (`env.*`)        | Backs                         | JS implementation |
|-------------------------|-------------------------------|-------------------|
| `host_write(fd,ptr,len)`| `Console.log/print`           | `term.write(...)` (xterm) |
| `host_read_line() -> …` | `Console.readLine`            | line buffer fed by xterm keystrokes; **must pump the scheduler** (see §4) |
| `host_now_ms() -> f64`  | `Clock.now`                   | `Date.now()` |
| `host_fmt_double(x,ptr,cap)` | `snprintf("%g")`          | JS `Number.toString` into memory (avoids a float-printf shim) |
| `host_vfs_read/write/exists` | `File.*`                 | in-memory `Map<path,bytes>` |
| `host_proc_run(ptr,len) -> …`| `Process.run`             | mock shell over the VFS (`ls`/`cat`/`grep`/`echo`) |

Exports JS calls in:

| Export                        | Purpose |
|-------------------------------|---------|
| `scry_eval(srcPtr, len) -> jsonPtr` | **The entire viewer wire.** Calls `eval-core` directly (§3). |
| `scry_boot()` / `scry_tick()` | start the program; run one scheduler quantum (drives green threads + the TUI) |
| `scry_feed_stdin(ptr,len)`    | deliver a typed line to the `readLine` buffer |
| `malloc`/`free` (optional export) | so JS can allocate arg buffers in linear memory |

---

## 3. Subsystem dispositions (from the audit)

### 3.1 Viewer transport → one exported function ✅ easy

The viewer already funnels **every** operation through one JS function, `evalSource`
(`viewer/app.js:58`), which does `POST /eval` with `{id, source}`. Every pane —
`types()`, `graph()`, `fields("X")`, `trace(...)`, `gc()`, live redefinition, the REPL
dock — is just a `source` string (DECISIONS #8: the only wire op is `eval`). On the VM
side, all of it lands in **one Coil entry point**, `eval-core(src, len, readonly)`
(`src/server.coil:823`), which writes a `{value|error}` JSON buffer with **zero socket or
thread dependency** — those live only in the HTTP plumbing around it.

**Change:** export a thin wrapper over `eval-core`; rewrite `evalSource` to call
`window.scry_eval(source)` and parse the returned JSON instead of `await fetch`. Delete
the 5 portal-only fetch sites (`app.js:2657,2698,2731,2735,2781`) and the `evalBase`
plumbing. **No pane logic changes** — they're all decoupled behind the source/value
contract.

**One real snag:** `eval-core`'s diagnostic capture redirects fd 2 to a `/tmp` file via
`open`/`dup2`/`read`/`unlink` (`src/evalrt.coil:82-105`) to grab typechecker text. No
browser equivalent. **Rework the typecheck error writer to emit into an in-memory buffer**
directly (a small, self-contained change; benefits native too).

### 3.2 Threads / STW / GC → cooperative, GC unchanged ✅ tractable

No pthreads are fundamental. Real threads live only in `coil/lib/thread.coil` (4 call
sites) + 3 server spawns + one `pthread_detach` (`server.coil:35`). **GC needs zero
changes**: the STW protocol (`src/safepoint.coil`) already degrades to a no-op with one
registered thread — `sp-park-wait` has nobody to wait for, so `gc-collect` just
stops-itself/collects/resumes. Atomics (`coil/lib/atomic.coil`) lower fine on wasm32.

Green threads are 90% pre-shaped: each language thread is a self-contained `VMThread`
running the same re-entrant `run-to` loop (pointer threaded explicitly, no TLS), already
yielding at `safepoint-poll` and in every blocking primitive. The work:
- `vm-spawn` (`vm.coil:576`): drop `thread-spawn`; **enqueue** the (already alloc'd,
  registered) `VMThread` onto a ready-queue.
- `vm-join` (`vm.coil:585`): drop `thread-join`; its cooperative `done`-spin already works.
- `run-to` (`vm.coil:593`): **yield to a round-robin scheduler** at the safepoint seam
  instead of only self-parking. This is the one structural piece. Confirm `run-to`'s
  self-calls (nested Scry calls) sit on the VM's own explicit frame stack — the audit
  indicates frames are heap VMThread state walked by stack maps, so a Scry call is a loop
  iteration, not host recursion, which makes "yield = return from run-to" clean. **Verify
  before building.**
- Server (`server.coil`): in the wasm build, no sockets/accept/conn threads exist. `Ask`
  actions and `TurnWorker` (`assistant.scry:394`) enqueue and run on the next tick.

### 3.3 Host / IO → JS bridge + fake model ✅ well-defined

Every host builtin dispatches from `do-builtin` (`vm.coil:511`) and bottoms out in externs
declared once in `src/ioutil.coil`. Dispositions:
- `Console.log/print` (`write`), `readLine` (`poll`+`read`), `Clock.now/sleep`
  (`gettimeofday`/`nanosleep`) → JS bridge imports (table in §2).
- `File.*`, `Process.run` → **intercept at the VM builtin boundary** (`BLT_FILE_*`,
  `BLT_PROC_RUN`), not by emulating `fopen`/`popen`. Route to `host_vfs_*` / `host_proc_run`.
- **The LLM is already free:** `agent/core.scry:367` has a pure, no-IO `ScriptedModel`, and
  `chooseBrain` (`assistant.scry:313`) auto-selects it when no API key is present. Make the
  wasm `Env.get` return null → the fake model is chosen; `Http.request`/libcurl is never
  reached. `http.coil` gets a Coil **stub** in the wasm build so no `curl_*` externs exist.
- `research`/`loop` background commands rely on threads — they keep working under the
  scheduler **iff `readLine` pumps the scheduler** rather than truly blocking the one OS
  thread. This is the single demo regression risk; the bridge's `host_read_line` must yield.

---

## 4. The `readLine` invariant (the one thing that can break the demo)

Today the main thread sits "blocked" in `vm-readline` (`vm.coil:250`) while subagent
threads run on real OS threads. Under cooperative scheduling, background workers only
advance when someone yields. `vm-readline` already yields every 20 ms poll-slice — so the
UX survives **only if** the browser's line input is delivered asynchronously and the
scheduler keeps ticking while the prompt waits. Concretely: `readLine` returns "no line
yet → yield to scheduler"; xterm keystrokes accumulate a line JS-side; on Enter,
`scry_feed_stdin` makes the next `readLine` poll succeed. Never implement `readLine` as a
blocking host call.

---

## 5. Coil compiler / finalizer change spec (hand-off)

**Validated by build experiment (2026-07-17).** I compiled minimal Coil programs to
`wasm32-unknown-unknown` and ran them in node. Findings that pin this section down:
- A program using only `malloc`/`memcpy`/`host_write` as `env` imports **finalizes and runs
  correctly** — JS provides the libc functions over the module's own exported memory (a JS
  bump allocator above the static high-water, growing the `Memory` as needed). Verified:
  `go(0x4142434445464748)` malloc'd, stored, and `host_write` emitted the exact LE bytes.
  ⇒ **The libc shim is JS-provided, not Coil-native** (see S1 below — simpler than the
  original plan, and zero `ioutil.coil` changes for those functions).
- The **first** thing that broke was **`env.__stack_pointer`**, not function pointers. Any
  function that uses `alloc-stack` or a variadic call (`snprintf`) references the shadow-stack
  pointer global, which the finalizer rejects exactly like `GOT.func.*`. This gates almost
  everything (20 `alloc-stack` sites + 28 `snprintf` calls across ast/builtins/vm/lexer/json,
  plus inevitable spills). ⇒ **C0 below is the primary, prerequisite compiler change.**

So: **two finalizer changes** (C0, C1), same shape (define an imported global/table the
finalizer currently rejects). No Coil-native libc needed. No `i128` in scry (audited), so
the compiler-rt `__multi3`/`__udivti3` risk does **not** apply. `memcpy`/`memset` *intrinsics*
already lower to WASM `memory.copy`/`memory.fill` (bulk-memory on) — explicit `memcpy`/`memset`
*calls* become `env` imports (JS-provided). Both C0 and C1 live in `selfhost/src/wasm.coil`.

### CHANGE C0 (compiler) — define `env.__stack_pointer` (shadow stack) ⭐ PREREQUISITE

**Why:** The pre-finalize object imports `(import "env" "__stack_pointer" (global (mut i32)))`
whenever a function spills, takes a local's address (`alloc-stack`), or makes a variadic call.
`wasm-global-value` (`wasm.coil:244`) only resolves `__memory_base`/`GOT.mem.*`, so
`wasm-globals-error` (`wasm.coil:268`) returns `-1` and `wasm-finalize` bails with
`"wasm: unresolved global import …"` (`wasm.coil:369`) — no `.wasm` written. Confirmed by
experiment: a program with a `snprintf`/`alloc-stack` fails to finalize; the identical program
without them finalizes and runs. This is standard wasm-ld behavior the finalizer must
reimplement.

**What to implement in `wasm.coil`:**
1. Reserve a shadow-stack region in linear memory (e.g. 1 MiB; wasm-ld default is 64 KiB but
   a VM recurses deep — size generously, make it a constant). Convention: place it just above
   the static-data high-water (the finalizer already computes data-segment end via
   `wasm-datasym-addr`/the data section) OR at a fixed low region with data shifted above —
   either works as long as it can't collide with data or the JS-managed heap.
2. **Define** `__stack_pointer` as a mutable i32 global initialized to the **top** of that
   region (stack grows down), by extending `wasm-global-value` (`wasm.coil:244`) and
   `wasm-globals-section` (`wasm.coil:277`) — same site C1 touches. Drop it from the import
   list in `wasm-imports` (`wasm.coil:318-330`).
3. Bump the **defined memory's** min-pages so the initial linear memory covers static data +
   shadow stack (so neither the stack nor JS's heap bump start inside live data).
4. **Export `__heap_base`** (= data_end + shadow-stack size, page-aligned) so JS starts its
   `malloc` bump allocator there instead of guessing from `memory.buffer.byteLength`. (Nice
   to have; without it JS can start at the initial byteLength, which the experiment did
   safely — but an explicit export is robust against future static-data growth.)

### CHANGE C1 (compiler) — WASM function-table / `call_indirect` finalization

**Why:** Coil's WASM target uses PIC reloc mode (`codegen.coil:1396`, 6th arg `2` =
`LLVMRelocPIC`). Any address-taken function lowers to `call_indirect` against an imported
`env.__indirect_function_table`, with each function's table index fetched via a
`GOT.func.<sym>` global import and `env.__table_base`. The finalizer
(`selfhost/src/wasm.coil`) resolves **only** `env.__memory_base` and `GOT.mem.*`; any other
global import makes `wasm-globals-error` return `-1` and `wasm-finalize` bails with
`"wasm: unresolved global import …"` (`wasm.coil:268-269, 369`). It also has **no** table
section, **no** elem segment, and preserves the table import unpopulated. So **any Coil
`fnptr`/`call-ptr`/`dyn` on wasm produces no module today.** (The two shipped demos
contain zero `call_indirect` — the feature was never exercised.) This also makes the
language guide dishonest: it advertises `fnptr-of`/`call-ptr`/`dyn` unconditionally
(`guide.coil:319-324`).

**What to implement in `wasm.coil`'s finalizer:**
1. Enumerate address-taken functions = every `GOT.func.<sym>` global import (mirror the
   import walk in `wasm-globals-error`, `wasm.coil:255`). Resolve each `<sym>` to its final
   function index via the linking symbol table you already parse in `wasm-read-sym!`
   (`wasm.coil:106`).
2. Assign table indices `1..N` (reserve slot 0 as a null/trap entry).
3. Emit a **defined table** (section id 4, `funcref`, min `N+1`) and **drop** the
   `env.__indirect_function_table` import in `wasm-imports` (`wasm.coil:318-330`).
4. Emit an **active elem segment** (section id 9) at offset 0 listing those function
   indices in table-index order.
5. Define `GOT.func.<sym>` globals = `i32.const <table-index>` and `env.__table_base` =
   `i32.const 0`, by extending `wasm-global-value` (`wasm.coil:244`) and
   `wasm-globals-section` (`wasm.coil:277`) next to the existing `GOT.mem.*` path.
6. Confirm no `R_WASM_TABLE_INDEX_*` relocations in `reloc.CODE` need patching **before**
   that section is discarded (`wasm.coil:397`) — under PIC the index goes through
   `global.get GOT.func`, so code bytes likely need no edit, but this would be the
   finalizer's first time *consuming* a reloc rather than ignoring it. Verify.

**Alternative to weigh:** switch the wasm build to non-PIC (`LLVMRelocStatic`) in
`codegen.coil:1396` for `arch==2`. LLVM then emits a defined table + elem segment directly,
possibly simpler — but it also moves data addressing off `GOT.mem`/`__memory_base`, so the
existing data-resolution path (`wasm.coil:161-238`) must be re-validated. Implementer's call.

**Scoping note — how much scry actually needs C1.** In the *wasm* build, almost every
Coil `fnptr` disappears: the curl callbacks (`http.coil:137`, `portalclient.coil:57`) are
in the stubbed-out HTTP path, and the `thread-spawn` fnptrs (`vm.coil:576`,
`server.coil:1133/1199/1237`) vanish with green threads + no sockets. **Exactly one**
internal `call-ptr` remains: `arena-for-each-live` (`arena.coil:250-255`), used by
`reflect`'s `instances`. So there are two ways forward:
- **Do C1 (recommended).** It's the principled fix, makes the guide honest, benefits every
  future Coil wasm program, and you offered to take Coil changes. The demo is not blocked
  waiting on it (see fallback).
- **Fallback if C1 slips:** refactor the single `arena-for-each-live` site to not use a
  fnptr (inline the live-slot walk as a macro, or fill a caller-provided buffer and loop in
  the caller). Then the wasm build needs **zero** compiler changes. Keep this as the
  unblock-the-demo path, not the destination.

### SHIM S1 (app + JS, no compiler change) — JS-provided libc

`wasm32-unknown-unknown` has no libc and the finalizer can't link one — but the build
experiment showed **JS provides it, over the module's own exported memory**, with **no
`ioutil.coil` changes**: the externs (`malloc`, `realloc`, `free`, `strlen` (47 calls),
`memcpy` (20), `strcmp` (19), `snprintf` (28), `strtod` (3), `memset` (2), `atoi` (1)) stay
declared and simply resolve to `env.*` imports on wasm. The JS bridge implements them:
- **Allocator:** a JS bump/free-list `malloc`/`realloc`/`free` starting at `__heap_base`
  (exported by C0; the experiment safely used the initial `byteLength`), calling
  `memory.grow()` when it runs past the end. `realloc` = malloc+`copyWithin`; `free` can be a
  no-op for the demo or a real free-list.
- **mem/str:** `memcpy`/`memset` = `Uint8Array.copyWithin`/`.fill`; `strlen`/`strcmp`/`atoi`
  trivial over the memory view; `strtod` via JS `parseFloat`.
- **`snprintf`:** implement in JS by reading the format C-string from memory and the arg — for
  the demo we only need the specifiers scry actually uses (`%lld`, `%d`, `%s`, `%g`, `%f`,
  `%c`; audit the 28 sites to confirm). `%g`/`%f` are *easiest* in JS (`Number.toString`),
  which is the main reason JS-provided beats a hand-rolled Coil float-printf. **Note:** because
  `snprintf` is variadic, its call sites are exactly what forces C0 (shadow stack); C0 must
  land for these to compile at all, but the *implementation* is pure JS.

This keeps `src/ioutil.coil` byte-identical between native and wasm for the libc functions;
only the OS-boundary externs that have **no** browser meaning (sockets, curl, pthreads, files,
subprocess) need the wasm build to avoid referencing them on a live path (§3.3: `http.coil`
stub, green threads, VFS/mock-shell at the builtin boundary, `Env.get`→null).

---

## 5b. Build configuration & module set (how the wasm build is selected)

**Validated 2026-07-17.** Two facts settle this:
1. **Undefined externs only become wasm imports if *referenced*.** An `extern` declaration in
   `ioutil.coil` that no reachable code calls produces nothing. So the wasm build does **not**
   need to fork `ioutil.coil` or delete the socket/curl/pthread externs — it only needs to keep
   those *code paths* unreferenced from its exports (dead code is dropped).
2. **Compile-time target branching works via a macro** (once Coil is current — see caveat).
   The canonical primitive is `(code-eq (target-arch) `wasm32)`. A one-liner macro gives
   `#ifdef`-style conditional compilation that *removes* a call on wasm rather than branching at
   runtime (a runtime `if` would keep both branches → the curl/socket refs would survive):
   ```coil
   ; expands to BODY on native, STUB on wasm — the wasm branch never references BODY's callees
   (defn when-native [(body Code) (stub Code)] (-> Code)
     (if (code-eq (target-arch) `wasm32) stub body))
   ```
   Verified end-to-end: a program guarding a fake `curl_thing` call with `when-native` builds
   natively (references it) and, on `--target wasm32`, drops the reference and finalizes with **no
   import** for it.

**So the module set stays shared — no file forking.** Apply `when-native`/`when-wasm` guards at
exactly the host-boundary sites so the wasm build never references the browser-hostile externs:
- **`http.coil` `http-request`** — wasm branch returns a transport-error `HttpResult` without
  touching any `curl_*` extern (the whole libcurl block goes unreferenced → zero curl imports).
- **`vm.coil` `vm-spawn` / OP_THREAD_SPAWN** — wasm branch enqueues a green-thread `VMThread`
  instead of calling `thread-spawn` (`pthread_create`); native keeps real threads. (This is the
  same edit as the P3 scheduler work.)
- **entry (`main.coil`)** — wasm branch skips the socket accept loop and instead wires the
  `scry_eval`/`scry_boot`/`scry_tick` exports; native keeps `server-accept-loop`. Because
  `eval-core` (in `server.coil`) has no socket dependency, the socket-calling functions are simply
  dead on wasm and dropped — `server.coil` need not be split.
- `Env.get` (`getenv`) — provided by the JS bridge as always-null (fake model auto-selected); no
  guard needed (getenv is harmless as an import).

**⚠ Build-tooling dependency — Coil must be rebuilt from current HEAD.** The installed
`~/.cargo/bin/coil` (dated Jul 17 00:36) is **stale**: it predates commit `d85202678`
(*"(target-arch) reflects the real --target"*, dated Jul 17 19:09), so on it `(target-arch)`
returns the **host** arch even under `--target wasm32` at macro-expansion time — the guards above
would silently bake the *native* branch into the wasm module. The fix is already in HEAD;
`coil/selfhost/rebootstrap.sh` produces a correct compiler. **Confirm `(code-eq (target-arch)
`wasm32)` actually folds true under `--target wasm32` (the `expand` gate in
`selfhost/oracle/gate-cli.sh`) before relying on the module-set guards.** This rebuild pairs
naturally with landing C0/C1, which also live in the self-hosted compiler.

---

## 5c. Target-width libc types (`isize`) — the thing that actually bit

**Learned the hard way 2026-07-18.** This was the single largest source of wasm breakage,
and it is invisible on arm64.

Coil's prelude declares the libc surface with **target-width** types:
```coil
(extern read   :cc c [i32 (ptr i8) isize] (-> isize))   ; ssize_t read(int, void*, size_t)
(extern write  :cc c [i32 (ptr i8) isize] (-> isize))
(extern malloc :cc c [isize] (-> (ptr i8)))
(extern strlen :cc c [(ptr i8)] (-> isize))
```
`isize` is **64-bit on arm64 but 32-bit on wasm32** (`size_t`/`ssize_t` are 32-bit there), and
fd is C `int` (i32). scry had redeclared these with fixed `i64`. Two consequences:

1. **One C symbol admits one signature.** A module holds ONE function per C symbol, so scry's
   disagreeing redeclaration was *silently dropped* and its calls built against the prelude's
   type. On arm64 (`isize` = i64) that is indistinguishable from correct; on wasm32 it emits an
   **invalid module** (`func N: type mismatch: expected i64, found i32`). Coil now rejects the
   mismatch outright at compile time instead of miscompiling.
2. **Fixing the declarations pushes the width outward.** Matching the prelude made 68 call
   sites fail on wasm32 where `isize` (i32) met i64 code.

**The pattern that worked** — don't cast at ~90 call sites. Keep the externs matching the
prelude exactly, then wrap them once in `ioutil` with i64-facing helpers, so the rest of the
codebase keeps speaking i64 and the width conversion lives at ONE boundary:
```coil
(defn s-len    [(s (ptr i8))] (-> i64) (cast i64 (strlen s)))
(defn xrealloc [(p (ptr i8)) (n i64)] (-> (ptr i8)) (realloc p (cast isize n)))
(defn xmemcpy  [(d (ptr i8)) (s (ptr i8)) (n i64)] (-> (ptr i8)) (memcpy d s (cast isize n)))
(defn xmemset  [(d (ptr i8)) (c i64) (n i64)] (-> (ptr i8)) (memset d c (cast isize n)))
```
plus `xalloc` narrowing for `malloc`. Then rewrite call sites mechanically
(`(strlen ` → `(s-len `, etc.), skipping the extern/wrapper lines themselves.

Two gotchas: a Coil `defn` wrapper is **stricter about pointer types** than the C extern was
(`(ptr u8)` vs `(ptr i8)` now needs an explicit cast), and an `if` whose branches straddle the
boundary fails with *"if branches on mixed widths"* — make both branches i64.

**JS-side mirror:** the bridge must return **plain numbers** (not `BigInt`) for `isize`-typed
returns on wasm32 (`strlen`, `write`, `read`) — returning a BigInt where wasm wants i32 throws
*"Cannot convert a BigInt value to a number"* on the return conversion, which is easy to
misattribute to the callee. `snprintf` must also implement **star precision** (`%.*s`): scry
uses it to print non-NUL-terminated slices, and each `*` consumes an int arg *before* the
value. Without it, diagnostics render a literal `%.*s`.

---

## 6. Phased plan

- **P0 — Compiler unblock (Jimmy). ✅ DONE 2026-07-18.** C0 (shadow stack) + C1 (function
  table) landed in `selfhost/src/wasm.coil`; coil rebuilt. All verified: `call_indirect`,
  `__stack_pointer`, variadic `snprintf`, and `(target-arch)` under `--target`. (C1's real
  shape differed from the spec below — LLVM emits an imported `__indirect_function_table` +
  elem segment + `reloc.CODE` table-index placeholders the finalizer must patch, not
  `GOT.func.*`.) The `arena-for-each-live` fallback was **not** needed.
- **P1 — JS libc (S1) + headless VM. ✅ DONE.** The whole VM compiles to a **valid** wasm32
  module and **boots headless in node**: `scry_boot` loads → typechecks → compiles → runs
  `main()`, with `Console.log` reaching the host via `write`. Required aligning scry's libc
  externs with the prelude's target-width (`isize`) signatures — see §5c.
- **P2 — Eval waist. ✅ DONE.** `scry_eval(src,len) → JSON` works against the live heap:
  `types()` reflection (with correct `liveCount`), `instances()`, allocating new objects via
  eval, method invoke on a live instance, and **hot body-swap** (`greet()` 1 → redefine →
  99). Diagnostic capture works through the in-memory sink (typed errors, no fd-2). The
  `evalrt.coil` fd-2 rework landed earlier (commit `e96a94099`).
- **P3 — Green-thread scheduler.** Convert `run-to` to yield; `scry_tick`; enqueue on
  spawn; cooperative `readLine`/`sleep`. _Gate: a 2-thread `.scry` interleaves under the
  scheduler; `assistant.scry`'s `research`/`loop` advance while the prompt waits._
- **P4 — Viewer transport swap.** Rewrite `evalSource`→`scry_eval`; delete portal fetch
  sites. Serve the existing React app + the wasm module as static files. _Gate: every pane
  works against the in-page VM._
- **P5 — Terminal + fake model.** Wire xterm.js ↔ `host_write`/`scry_feed_stdin`; VFS +
  mock shell; run `assistant.scry` end-to-end with `ScriptedModel`. _Gate: the full demo
  moment — type in the terminal, watch the viewer update live._
- **P6 — Package.** One embeddable page (inlined/self-contained), a scripted demo path,
  polish per DECISIONS #11 (must look genuinely good).

---

## 7. Open decisions for Jimmy

1. **C1 vs the one-site fallback** (§5). Recommend C1 (proper fnptr support), fallback
   available so the demo never blocks on it.
2. **Terminal library:** xterm.js is the default choice (mature, self-contained). Confirm,
   or name an alternative.
3. **VFS/shell fidelity:** how real should the mock shell be? Proposal: a handful of
   commands (`ls`/`cat`/`grep`/`echo`) over a seeded in-memory tree — enough to make
   `ShellTool`/`SearchTool`/`FileReadTool` demo convincingly, not a real POSIX shell.
4. **Threads-are-cooperative-on-wasm** is a knowing relaxation of DECISIONS #4b, wasm-build
   only. Recorded here; flag if you'd rather gate the demo differently.
</content>
</invoke>
