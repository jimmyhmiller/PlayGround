# Linux x86-64 bootstrap IR

Cross-emitted LLVM IR of the Coil self-hosted compiler (and two smoke-test
programs) targeting **x86_64-unknown-linux-gnu**, produced on macOS/arm64. The
Linux box links, runs, and chases the fixpoint from here — nothing in this
directory has been run, only emitted and cross-verified (object emission +
`llvm-as` parse).

## Artifacts (xz-compressed textual IR)

| file | program | raw | notes |
|------|---------|-----|-------|
| `coil-linux.ll.xz` | the compiler (`selfhost/src/main.coil`) | 14 MB | 281,284 lines |
| `fib-linux.ll.xz`  | `examples/fib.coil` (prints `55` via exit code) | 62 KB | pure compute smoke test |
| `io-linux.ll.xz`   | `examples/io.coil` (prints `answer=42`) | 75 KB | strings + buffer + `write(2)` smoke test |

Decompress: `xz -dk coil-linux.ll.xz` (etc.).

## Provenance

- **Coil commit:** emitted from the tree at the commit that adds this file
  (parent `5b60ba148`). The only compiler-source change in that commit is the
  x86 tail-call fix described below (`selfhost/src/codegen.coil`); macOS
  `selfhost/rebootstrap.sh` is green (fixpoint + all gates) with that change in.
- **LLVM used to emit/print:** Homebrew **LLVM 21.1.8** (`llvm-config --version`
  → 21.1.8). The textual IR therefore uses LLVM-21 printer spelling — see the
  one 21-ism below.
- **Verified on macOS, without running:**
  - `clang --target=x86_64-unknown-linux-gnu -c <f>.ll -o <f>.o` → clean ELF
    `x86-64 SYSV relocatable` for all three.
  - `llvm-as <f>.ll` → parses clean for all three.

## The one LLVM-21-only token (matters for your 20.1.8)

The compiler module contains **`captures(none)`** exactly twice, both on the
`llvm.memcpy` intrinsic declaration:

```
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none),
                                     ptr noalias readonly captures(none),
                                     i64, i1 immarg) #1
```

`captures(none)` is the LLVM-21 spelling; the pre-21 name is `nocapture`.
**LLVM 20.1.8 will not parse `captures(none)`.** Everything else in the module
(`memory(argmem: readwrite)`, `willreturn`, `nocallback`, `nofree`, `immarg`,
`noalias`, `writeonly`, `readonly`, `alwaysinline`, the datalayout) is stable and
parses on LLVM 20.

Two options, either works — I verified the downgraded form still compiles on 21
too (so it's version-agnostic):

1. **Recommended — install LLVM 21** (apt.llvm.org). Then the shipped `.ll`
   parses as-is, and (see below) the compiler's libLLVM C-API calls line up with
   a matching runtime. You offered to do this; it's the clean path.
2. **Stay on LLVM 20** — one sed before compiling:
   ```
   sed -i 's/captures(none)/nocapture/g' coil-linux.ll
   ```
   `nocapture` is a plain optimization hint on an intrinsic decl; the downgrade
   is semantically inert. (fib/io don't contain the token at all.)

## Target correctness (what was checked)

- **datalayout:** `e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128`
  and `target triple = "x86_64-unknown-linux-gnu"` — taken from clang's own
  x86_64-linux output. Correct ELF datalayout.
- **No Mach-O symbol mangling:** zero underscore-prefixed globals
  (`@memcpy`, not `@_memcpy`; `@main`, not `@_main`). ELF-clean.
- **No Mach-O-isms in module structure:** the strings `__TEXT`/`__DATA`/`macho`/
  `apple-macosx` appear only inside the compiler's own string *constants* (it is
  a compiler that emits Mach-O); none are in the module's own structure,
  sections, or symbol names.
- **Varargs / SysV AL protocol — clean.** The only genuinely-variadic call the
  compiler makes is `snprintf`, and it **is** declared variadic
  (`declare i32 @snprintf(ptr, i64, ptr, ...)`). The compiler opens existing
  files with 2-arg `open(path, O_RDONLY)` (glibc reads no varargs without
  `O_CREAT`, so the fixed-arity decl is safe) and creates files via non-variadic
  `creat`. No fixed-arity declaration of a varargs function is ever called in a
  way that reads varargs. No AL-register hazard.

## x86 tail-call fix (the one compiler-source change)

`x86_64-unknown-linux-gnu -c coil-linux.ll` originally aborted:

```
fatal error: failed to perform tail call elimination on a call site marked musttail
  ... on function '@"comptime.concat-into!"'
```

Cause: the codegen self-tail-call path emits `musttail` (LLVM tail-call kind 2)
for every self-recursion. AArch64 returns large aggregates via `x8` and
tail-calls them fine, so the macOS build is happy. But x86-64 SysV classifies a
large struct return (e.g. `%coil.core.Option__ast.Diag = { i32, [6 x i64] }`,
56 bytes → MEMORY) as an sret return, and LLVM's x86 backend *cannot* satisfy
`musttail` for it, so it aborts.

Fix (`selfhost/src/codegen.coil`, in `emit-tail`'s `ECall` branch): when
`arch == 0` (x86_64) **and** the callee's LLVM return type kind is Struct (10)
or Array (11), downgrade `musttail` → `tail` (kind 1). `tail` is a hint LLVM
applies where it can (scalar self-calls still get real TCO) and degrades to a
normal call otherwise. AArch64 and scalar-return x86 self-calls keep guaranteed
`musttail`, unchanged.

Effect: in the linux IR, 25 aggregate-returning self-tail-calls became `tail`
(84 `musttail` + 25 `tail` = the 109 the arm64 build still emits as `musttail`).
The macOS/arm64 emission is byte-for-byte unchanged (the IR gate still passes),
because the new branch is inert when `arch != 0`.

⚠ Trade-off to know: the 25 downgraded sites lose *guaranteed* TCO on x86.
They're bounded recursions (list walks over program-sized inputs), and LLVM's
x86 sibcall opt still tail-calls most of them; worst case is extra stack on
pathological input. If you ever see a stack overflow in `comptime.*` on a huge
input, that's the place — but it should not bite a bootstrap.

## External symbols the compiler module needs at link time

### LLVM C API — 149 `LLVMxxx` symbols → link **libLLVM**

Standard C-API surface: `LLVMContextCreate`, `LLVMModuleCreateWithNameInContext`,
`LLVMBuildCall2`, `LLVMBuildGEP2`, `LLVMArrayType2`, `LLVMConstArray2`,
`LLVMCreatePassBuilderOptions`, `LLVMRunPasses`, `LLVMTargetMachineEmitToFile`,
`LLVMGetTargetFromTriple`, `LLVMVerifyModule`, the `LLVMInitialize{X86,AArch64,
WebAssembly}*` target inits, etc. Every one of these exists in **both LLVM 20
and 21** (the newest, `LLVMArrayType2`/`LLVMConstArray2`, landed in LLVM 17), so
the compiler can *run* against either libLLVM — the C API is stable across the
20↔21 boundary. The only 20-vs-21 mismatch is the textual `captures(none)` above,
which is a *parse-time* concern for the shipped `.ll`, not a C-API/runtime one.

### libc / libm / pthread / dl (glibc all provide these)

```
abort access close closedir creat exit fclose fopen free fwrite getcwd getenv
getpid malloc memcmp memcpy memmove memset mmap mprotect munmap open opendir
read realloc realpath rename snprintf strlen strtod system unlink write
dlopen dlsym                       -> -ldl
pthread_attr_init pthread_attr_setstacksize pthread_create pthread_exit
pthread_join                       -> -lpthread
(strtod)                           -> -lm
```

### ⚠ Four Darwin-only externs — you must remap these (the "Darwin-isms")

These are **called** in the module and will not link on Linux as-is:

| symbol | source | what it is | Linux remap |
|--------|--------|-----------|-------------|
| `dispatch_semaphore_create` | `selfhost/src/metaengine.coil:47,97-98` | GCD (libdispatch) counting semaphore | POSIX `sem_t` (`sem_init`/`sem_post`/`sem_wait`) or a pthread mutex+condvar |
| `dispatch_semaphore_wait`   | `metaengine.coil:48,106,119` | GCD sem wait (`-1` = FOREVER) | `sem_wait` |
| `dispatch_semaphore_signal` | `metaengine.coil:49,109,114,118` | GCD sem post | `sem_post` |
| `sys_icache_invalidate`     | `selfhost/src/jit.coil:52,348` | Darwin i-cache flush after JIT codegen | on x86 a **no-op** (coherent i-cache); elsewhere `__builtin___clear_cache` |

`metaengine.coil` is the parallel metaprogram job runner (a small thread pool);
`jit.coil` is the JIT. Both are *referenced* by `main.coil`, so the linker needs
them resolved even for a batch `build` that never enters those paths. Fix them in
`.coil` source on your side (per the task split) — the smallest change is a
per-OS `extern` shim mapping these three sem ops to `sem_*` and stubbing
`sys_icache_invalidate` to return 0 on x86. Two related runtime Darwin-isms to
watch while you're in there (not link failures, but behavioural): the compiler
`dlopen`s metaprogram plugins as `.dylib` (→ `.so` on Linux), and any
`-install_name`/codesign assumptions in its own linker-driver invocation.

## Expected Linux link command

Object then link (dynamic libLLVM, the default the compiler expects):

```sh
# 1. IR -> ELF object (use LLVM-21 clang, or sed captures(none)->nocapture first)
clang -c coil-linux.ll -o coil.o          # native x86-64 clang; no --target needed on the box

# 2. link the stage-1 compiler
clang coil.o -o coil-stage1 \
    -L"$(llvm-config --libdir)" -lLLVM \
    -lm -lpthread -ldl -lstdc++ \
    $(llvm-config --system-libs)          # typically -lrt -ldl -lm -lz -lzstd -ltinfo
# (…after you've provided the four Darwin symbols above, or the link will
#  report them as undefined references.)

# 3. smoke test the toolchain BEFORE the big self-build:
clang -c fib-linux.ll -o fib.o && clang fib.o -o fib && ./fib; echo "exit=$? (expect 55)"
clang -c io-linux.ll  -o io.o  && clang io.o  -o io  && ./io        # expect: answer=42

# 4. then the real thing:
./coil-stage1 build selfhost/src/main.coil -o coil-stage2 \
    -L"$(llvm-config --libdir)" -lLLVM -lm -lpthread -ldl -lstdc++ $(llvm-config --system-libs)
#   -> chase the fixpoint: stage2.o == stage3.o
```

fib and io need only libc — link them with plain `clang <f>.o -o <f>`.

## Status

- ✅ Compiler IR emits, cross-compiles to ELF x86-64, and `llvm-as`-parses clean.
- ✅ fib + io IR likewise (libc-only, no libLLVM, no Darwin symbols).
- ✅ macOS `rebootstrap.sh` green with the codegen fix (fixpoint + gates).
- ⚠ Unresolved *by design* (your side, per the task split): the four Darwin
  externs above must be remapped in `.coil` source; on LLVM 20 apply the one-line
  `captures(none)`→`nocapture` sed (or install LLVM 21).
