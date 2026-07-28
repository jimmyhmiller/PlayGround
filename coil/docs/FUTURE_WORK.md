# Coil — what's left

An honest map of the gap between "a sharp, self-hosted compiler" and "a language you'd
reach for instead of C or Zig". Ordered by what blocks adoption, not by how interesting
the work is.

## Where Coil stands

The compiler core is done and is not the bottleneck. Coil is self-hosted, self-verifying
(rebootstrap fixpoint + the `selfhost/oracle` gates over a 96-file corpus), self-hosts on
macOS arm64 and Linux x86-64, emits wasm, and even runs *inside* wasm where it
self-compiles to a byte-identical arm64 binary. Diagnostics carry `file:line:col` and a
caret. DWARF works through lldb. Traits, generics, sums, slices, strings, a module
system, comptime with the whole language available, whole-program checkers and
transforms, `coil test`, `coil fmt`, `coil repl`, `Coil.toml` projects, `cimport` — all
shipped.

What is missing is livability, reach, and the ability for someone who is not the author
to use it.

---

## 1. Nobody else can adopt it yet

### 1.1 No dependency story

`Coil.toml` understands `[package] name/entry`, `[build]`, `[link]`, `[cc]` and `[run]`.
There is no way to depend on someone else's code: no `[dependencies]`, no fetch, no
version pinning, no lockfile, no library search path. Everything must be vendored by
hand or reached with a relative import.

Needed: a dependency section, a fetch-and-pin mechanism with a lockfile, content
addressing, and a resolution rule that composes with the existing file-relative import
semantics and the bundled-stdlib `<bundled>` sentinel.

### 1.2 No release or versioning story

`coil --version` does not exist (it prints usage and exits 1). There is no release
artifact, no install path, no channel — the compiler is a binary committed to a repo.
Anyone adopting Coil needs to answer "which Coil?" and today there is no answer.

### 1.3 Diagnostics stop at the first error

The type checker reports one error and stops. Spans across `import`/`include` are
`DUMMY`, so a diagnostic about imported code cannot point at it — that needs multi-source
span ids, which is also the last gap in DWARF for imported functions.

This blocks more than daily use: `docs/SEMANTIC_METAPROGRAMS.md` needs a
collect-and-continue mode for metaprogram-authored diagnostics, so one fix pays twice.
`warn`/`report` already collect — the compiler's own checker is the part that doesn't.

### 1.4 No LSP

`emacs/coil-mode.el` is the entire editor story. Spans exist, the resolver already
computes definitions and references, and `coil check` is fast on a single file, so the
hard inputs are in place — this is mostly plumbing, and it is the highest-visibility
adoption item. It requires 1.3 first: an editor cannot show one error at a time.

---

## 2. Blocks writing real programs

### 2.1 Standard library breadth

25 modules, ~3.2k lines total. Missing outright: **time/clock**, **process/env**,
**sockets**, **random**, general **sort**, **path** manipulation, **buffered** reader/
writer, **UTF-8** handling beyond bytes, a growable **string builder**, **JSON** (it
lives in `examples/`).

**Concurrency is the biggest hole**: `lib/thread.coil` is 23 lines wrapping
`pthread_create`/`join`, with no mutex, condvar, channel or thread pool — while
`metaengine.coil` contains a working portable counting semaphore that should be lifted
into `lib/`.

### 2.2 Compile speed and scale

Measured: the 31k-line self-host builds in ~18s wall (`emit-obj`, ~14.4s user); a small
file is ~0.24s. There is no incremental compilation, no per-module object cache, no
parallel codegen, and monomorphization is whole-program. At 100k lines that is a minute
per edit.

Options, roughly in order of payoff per effort: cache per-module expansion and IR;
parallel codegen; use the arm64 backend as the fast debug path (it is already ~17× faster
than LLVM on the compiler itself); then attack incremental mono.

### 2.3 Windows

macOS arm64 and Linux x86-64 only. Windows needs PE/COFF and the MS x64 ABI. This is the
single biggest reach gap for desktop users.

---

## 3. Known defects

These are real, reproduced, and each has a clear repro:

- **Runaway comptime crashes the compiler.** A self-tail-recursive `(comptime …)` is not
  TCO'd on the comptime-thunk path, so around 10M frames it dies with a bus error
  instead of erroring. Core `loop` at comptime is fine, and the same function at runtime
  is fine. The tree-walking interpreter had a fuel budget; nothing replaced it.
- **A sum-typed `const` aborts the build** with `UNIMPLEMENTED: codegen: unknown static
  const <name>` rather than being supported or refused cleanly.
- **A comptime result cannot be a generic-instance aggregate** — `(Option i64)`,
  `(Pair i64 i64)` report "cannot be materialized". Plain structs, plain sums and arrays
  work.
- **`--use` requires the target file to declare `(module …)`**, which surprises through
  `coil test` and `--debug-checks`: a bare single-file program is refused with an error
  whose span points at `<cli-use>`, a file the user never wrote.
- **`--sanitize=address` cannot link** where the system `cc`'s ASan runtime does not match
  the instrumenting LLVM (macOS with Homebrew LLVM + Apple clang). The instrumentation is
  correct; the link is not, and the driver hardcodes `cc` with no override.

## 4. Robustness

Gating is snapshot- and corpus-based. There is no fuzzing in the gates and no
differential property testing of the arm64 backend against LLVM beyond the fixed corpus.
A 60-case mutation fuzz (truncations, byte flips, chunk deletions) over `coil check`
produced no crashes or hangs, which is a good sign but not a guarantee. Worth adding: a
fuzz target in the gates, and a random-program generator diffing the two backends.

---

## 5. The moat — what to lean into once the above is handled

Coil has capabilities that are unusual or unique, and they are the reason someone would
switch rather than use Zig:

1. **Calling-convention-as-type.** Hand-rolled ABIs, syscall conventions, interrupt
   handlers, naked functions, JIT trampolines, register-pinned hot paths. The remaining
   piece is `adapt` — general convention-to-convention trampolines synthesized from two
   `defcc` descriptions.
2. **Raw LLVM IR + C embedding.** Coil hosts arbitrary LLVM IR and therefore hosts C.
   That opens `coil cc`, mixing C and Coil in one module with cross-language inlining,
   and reaching every LLVM intrinsic without compiler changes — a capability `@cImport`
   does not have.
3. **Metaprograms.** Checkers and transforms make a *dialect* a single import. The open
   work is core-form demotion (an interceptable `store!`), which unlocks write barriers
   and bounds checks on unmodified Coil, and function-signature reflection.
4. **Layout-as-types.** Per-field endianness in `:explicit` layouts is the missing piece
   for wire formats by value.
5. **Freestanding and embedded.** `--target …-none`, linker scripts, interrupt
   conventions, and MMIO are a natural fit; an MCU blink/UART demo in pure Coil, with
   device registers as `:bits` layouts and the vector table as shim-convention handlers,
   would be a compelling proof.

## 6. Suggested order

1. Multi-error reporting + multi-source spans (1.3) — unblocks the LSP and metaprogram
   diagnostics.
2. LSP (1.4) — the highest-visibility adoption win.
3. Packages + versioning (1.1, 1.2) — largely independent, can run in parallel.
4. The defects in §3 — each is small and each is a trust problem.
5. Stdlib breadth (2.1), starting with concurrency primitives and time.
6. Compile speed (2.2) before anyone has a 100k-line program, not after.
