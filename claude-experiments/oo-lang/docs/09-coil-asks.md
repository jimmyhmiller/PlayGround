# Coil changes the Scry WASM port needs

_Last updated 2026-07-18. Companion to `08-wasm-port.md`. Everything here is a change in the
**Coil** repo (`~/Documents/Code/PlayGround/coil`), not in Scry — scry-side issues are fixed
in scry and are not listed. Each item says what breaks without it, exactly what to change,
and how to know it works._

**Status: DONE. A1 (required) landed in Coil; B1 and B2 landed as guide docs; B3 skipped
(not needed by Scry).** The port is complete and working: the VM, GC, viewer, terminal and
the unmodified `examples/assistant.scry` all run in the browser.

---

## A1 — Export `__stack_pointer` ✅ DONE (2026-07-18) — ✅ LANDED

**Landed.** `selfhost/src/wasm.coil`'s finalizer now exports the shadow-stack pointer global
as `__stack_pointer` (kind-3 global export) whenever the object uses the shadow stack, right
beside the existing `__heap_base` export. `wasm-tools print` shows
`(export "__stack_pointer" (global 0))` and the module validates; the JS bridge restores SP
on unwind, closing the ~111 B/panic leak. Regression in `selfhost/oracle/gate-cli.sh`.

**Verified fixed.** The module now exports `(export "__stack_pointer" (global 0))`, and the
host restores it after unwinding. Measured after the fix: the stack pointer returns to
**exactly** its post-boot value after 1 / 10 / 100 / 1000 / 5000 / 12000 panics — `drift=0`,
zero unrestorable cases, VM healthy and heap intact throughout (the old build died at ~9421).
`wasm/test-panic.mjs` now guards this permanently.

**What used to break.** The uncrashable-eval invariant leaked and the instance eventually
died: **~111 bytes per eval panic, hard failure at ~9421 panics** with
`RuntimeError: memory access out of bounds`.

**Why.** wasm32 has no `setjmp`/`longjmp`, so Scry's eval landing pad is implemented by the
host: `eval-panic` records the error into `alloc-static` globals and calls `longjmp`, whose
import throws a JS exception; that exception unwinds the wasm frames, and the host then calls
`scry_eval_recover()` to emit the same typed JSON the native landing pad would
(`src/server.coil`, `wasm/scry-wasm.js`). Native's real `longjmp` restores the stack pointer
as part of unwinding. The host cannot, because the shadow-stack pointer is a **defined but
unexported** mutable global — so every panic permanently strands the frames between
`eval-core` and the panic site.

Confirmed by inspection of a built module: `(global (;0;) (mut i32) i32.const 2137584)` is the
shadow-stack pointer (top of the 1 MiB region C0 reserves — the exact constant moves as the
code changes), and the export section contains only `main`, `memory`, `__heap_base`, and the
`scry_*` functions.

**Change.** In `selfhost/src/wasm.coil`, export the shadow-stack pointer global alongside the
existing `__heap_base` export — same mechanism, one more entry:

```
(export "__stack_pointer" (global <the mut i32 global C0 defines>))
```

Mutable-global export requires the `mutable-globals` feature, which is already on (it is in
the emitted `target_features` section).

**Gate.** Rebuild `wasm/scry.wasm`, then:

```sh
wasm-tools print wasm/scry.wasm | grep '(export "__stack_pointer"'   # present
node wasm/test-panic.mjs                                             # already passes
```
and the leak check — this is the real gate, it currently dies at ~9421:
```js
// 100k panics must survive; the bridge restores SP automatically once the export exists
for (let i = 0; i < 100000; i++) scry.eval("Point.instances().get(99)");
```

**No Scry-side work needed.** `wasm/scry-wasm.js` already does
`const sp = this.instance.exports.__stack_pointer; … if (sp) sp.value = savedSp;` — it
restores when present and counts leaks when absent. The export alone closes it.

---

## Optional / nice-to-have

### B1 — Document target-width extern types in the guide — ✅ LANDED (guide.coil "Declaring C externs portably")

This cost the most time of anything in the port, and nothing warns you. Coil's prelude
declares libc with **target-width** `isize` (`lib/io.coil`, `lib/slice.coil`, `lib/alloc.coil`):

```coil
(extern read   :cc c [i32 (ptr i8) isize] (-> isize))   ; ssize_t read(int, void*, size_t)
(extern malloc :cc c [isize] (-> (ptr i8)))
```

`isize` is 64-bit on arm64 but **32-bit on wasm32**, and fd is C `int` (i32). An application
that redeclares any of these with fixed `i64` is silently wrong: a module holds ONE function
per C symbol, so the app's declaration is dropped and its calls are built against the
prelude's type. On arm64 that is indistinguishable from correct; on wasm32 it emits an
**invalid module**. (Coil now rejects the mismatch outright — that fix already landed and is
what surfaced this — but the guide still doesn't tell you to use `isize` for `size_t`/
`ssize_t` and `i32` for `int` when declaring your own C externs.)

Suggested: a short "declaring C externs portably" note in the guide listing
`size_t`/`ssize_t` → `isize`, `int` → `i32`, `off_t` → `i64`, plus "prefer the prelude's
declaration over redeclaring".

### B2 — A standard target-conditional macro — ✅ LANDED (guide.coil "Target-conditional code", `when-wasm` snippet next to `(target-arch)`)

The port needed compile-time target branching to keep wasm-only code out of the native build
(the green-thread yield hook must cost the native dispatch loop *nothing*). Scry defines its
own one-liner in `src/ioutil.coil`:

```coil
(defn when-wasm [(a Code) (b Code)] (-> Code)
  (if (code-eq (target-arch) `wasm32) a b))
```

This works well and needs no compiler change — but every project targeting both natively and
wasm will write the same macro. A blessed `when-target` / `cfg`-style helper in the prelude
(or just this snippet in the guide next to `(target-arch)`) would save the rediscovery.

### B3 — `coil build --emit-js`

Already on Coil's own roadmap (mirror `--emit-header`). Not needed by Scry: the bridge
(`wasm/scry-wasm.js`) is app-specific — it carries a libc shim, an in-memory VFS, a terminal
surface, and the eval/scheduler API — so a generic emitted bridge wouldn't replace it. Listed
only so it is not assumed to be a blocker.

---

## Already landed (for the record)

These were needed by the port and are **done** — kept here so the history is in one place.

| | Change | Why it was needed |
|---|---|---|
| **C0** | Define `env.__stack_pointer` — 1 MiB shadow stack above the static-data high-water, memory min-pages bumped, `__heap_base` exported | Any `alloc-stack` or variadic call (20 + 28 sites in Scry) referenced the shadow-stack pointer; without it `wasm-finalize` rejected the module and nothing non-trivial compiled |
| **C1** | Function table / `call_indirect` finalization | Scry's one internal `call-ptr` (`arena-for-each-live`). **The hand-off spec's model was wrong**: LLVM PIC does not emit `GOT.func.*` for locally-defined address-taken fns — it emits an imported `__indirect_function_table` + an active elem segment + `env.__table_base`, and leaves each fnptr as a padded 5-byte `R_WASM_TABLE_INDEX_REL_SLEB` **placeholder in `reloc.CODE` that the finalizer must patch**. Doc §5 C1 point 6 ("code bytes likely need no edit") was false |
| — | `(target-arch)` reflects the real `--target` at macro-expansion time (`d85202678`) | Compile-time target branching (B2) silently baked the *native* branch into wasm builds on a stale compiler |
| — | Reject one C symbol declared twice with different signatures | Turned a silent miscompile (app declaration dropped, calls built against the prelude's type — invisible on arm64, invalid wasm) into a clean compile-time error |
| — | Import paths resolve relative to the importing file | Tightened to match the guide; Scry's `main.coil` was relying on the older cwd-relative behaviour |
