# Coil C bootstrap

A **self-contained, platform-agnostic C bootstrap** for the Coil compiler.
Given only a C compiler and the committed `coilc.wasm` seed, it builds a
working `coil` compiler — no Node, no Rust, no LLVM, no wasm engine. This is
the same model Zig uses to bootstrap itself from `zig1.wasm`.

```sh
./build.sh
./coil-bootstrap build ../examples/fib.coil -o /tmp/fib && /tmp/fib; echo $?   # -> 55
```

## How it works

`coilc.wasm` is the entire Coil compiler compiled to a single static
**wasm64** module (from `selfhost/src/main_wasm.coil`), in which comptime runs
via an in-process bytecode interpreter — so there is no runtime code
generation and the module's only imports are libc/filesystem. See the repo
memory notes `project_coil_wasm_running_compiler` / `project_coil_bytecode_interp`.

`build.sh` does three steps, all with plain `cc`:

1. **build the translator** — `cc wasm2c.c -o wasm2c`. This is Zig's stage1
   `wasm2c` (vendored, MIT — see `NOTICE`/`LICENSE-ZIG`), extended by us to
   handle the two non-MVP features this module uses: **memory64** (i64 linear
   memory addresses) and **bulk-memory** (`memory.copy`/`memory.fill`).
2. **translate** — `./wasm2c coilc.wasm coilc.c little`. Emits a large
   (~900k-line) but fully self-contained C file: every wasm function becomes a
   C function, the linear memory is a `uint8_t*` buffer, and the `env.*`
   imports are left as `extern` declarations.
3. **compile** — `cc -O1 coilc.c runtime.c -o coil-bootstrap -lm`. `runtime.c`
   provides those `env.*` imports (an allocator over the module's linear
   memory, filesystem I/O, libc/math shims) and a `main()` that lays out
   `argv` in linear memory and calls the module's exported `main`.

Only `coilc.wasm` (the ~2 MB seed) and the source files are committed;
`wasm2c`, `coilc.c` and `coil-bootstrap` are build artifacts (gitignored).

## Files

| file | origin | role |
|------|--------|------|
| `wasm2c.c`, `FuncGen.h`, `InputStream.h`, `wasm.h`, `panic.h` | Zig stage1 (MIT), **modified** | the wasm→C translator |
| `config.h.in`, `zig.h`, `wasi.c` | Zig stage1 (MIT), pristine | vendored for a faithful import; **unused** by this build |
| `LICENSE-ZIG`, `NOTICE` | — | attribution; lists exactly what we changed |
| `runtime.c` | ours | `env.*` imports + `main()` driver |
| `build.sh` | ours | the 3-step build |
| `coilc.wasm` | generated seed | the compiler, as a wasm64 module |

## Regenerating the seed

`coilc.wasm` is produced by the Coil compiler itself. To refresh it (e.g. after
changing the compiler), from the repo root:

```sh
# 1. build a native compiler
./coil build selfhost/src/main.coil -o /tmp/coil-i $(./selfhost/llvm-link-flags.sh dynamic)
# 2. build the single static wasm64 module
#    --wasm-stack-size=64: the compiler recurses deep, so it opts into a 64 MiB shadow
#    stack (the default is a modest 16 MiB, generous for ordinary wasm programs).
/tmp/coil-i build selfhost/src/main_wasm.coil --target wasm64-unknown-unknown --wasm-stack-size=64 -o coil/bootstrap/coilc.wasm
```

Then re-run `./build.sh`.

## What is verified

* `./coil-bootstrap build examples/fib.coil -o /tmp/fib && /tmp/fib` exits **55**.
* **Self-build reproduction**: the C-bootstrapped compiler builds a compiler
  from `selfhost/src/main_a64.coil --backend arm64`, and that binary's
  `__TEXT,__text` (`otool -X -s __TEXT __text`) is **byte-identical** to the
  natively-built compiler's — i.e. the C bootstrap faithfully reproduces the
  compiler. The reproduced compiler itself builds `fib` → 55.

## Portability notes

* `runtime.c` uses POSIX (`open`/`read`/`write`/`realpath`/`system`/…). The
  guest's `O_*` flag values match the host's because the seed and the bootstrap
  run on the same OS family.
* The translator infers host endianness; pass `big`/`little` as `wasm2c`'s 3rd
  argument to override.
* Dead imports (JIT/dylib/raw-mmap/real-threads: `dlopen`, `mmap`,
  `pthread_create`, …) are never reached under interpreted comptime and are
  stubbed to `abort()` with a message, so any regression is loud rather than
  silent.
