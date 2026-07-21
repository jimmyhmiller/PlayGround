# Linux x86-64 bootstrap IR

LLVM IR of the Coil self-hosted compiler (and two smoke-test programs) for
**x86_64-unknown-linux-gnu**. This is the version-mismatch **escape hatch** for the
Linux toolchain: the normal path is the committed ELF seed
(`selfhost/seed/coil-seed-linux-x86_64`) via `selfhost/rebootstrap-linux.sh`; if that
seed's libLLVM (21) doesn't match your system, rebuild a stage0 from this IR against
whatever libLLVM you have (the C-API surface used — 149 `LLVMxxx` symbols, newest is
`LLVMArrayType2`, LLVM 17 — is stable across 20/21).

## Artifacts (xz-compressed textual IR)

| file | program | notes |
|------|---------|-------|
| `coil-linux.ll.xz` | the compiler (`selfhost/src/main.coil`) | links with NO extra shims |
| `fib-linux.ll.xz`  | `examples/fib.coil` (exit code 55) | libc-only smoke test |
| `io-linux.ll.xz`   | `examples/io.coil` (prints `answer=42`) | strings + write(2) smoke test |

## Provenance

Emitted **natively on Linux** by the self-hosted compiler at the commit that adds
this revision (`./coil-linux emit-ir selfhost/src/main.coil`), after the port fixes
landed in `selfhost/src` (portable pthread semaphores replacing Darwin GCD, dlsym'd
i-cache flush, host-aware dylib link lines, layout-aware SysV classification). The
relinked binary was verified to build and run programs before committing.

An earlier revision of these files was cross-emitted from macOS (LLVM 21.1.8,
Homebrew) as the original one-shot bootstrap and needed a 4-symbol Darwin shim; that
is no longer necessary — the compiler source no longer references any Darwin-only
extern. (History has the old NOTES if you need the cross-emission story, including
the x86 `musttail` aggregate-return downgrade in `codegen.coil::emit-tail`.)

## Rebuilding a stage0 from this IR

```sh
xz -dk coil-linux.ll.xz
# LLVM 20's parser: first  sed -i 's/captures(none)/nocapture/g' coil-linux.ll
clang -c coil-linux.ll -o coil.o
clang coil.o -o coil-stage0 \
    -L"$(llvm-config --libdir)" -Wl,-rpath,"$(llvm-config --libdir)" -lLLVM \
    -lstdc++ -lm -lpthread -ldl

# smoke-test the toolchain before the big one:
xz -dk fib-linux.ll.xz && clang -c fib-linux.ll -o fib.o && clang fib.o -o fib
./fib; echo "exit=$? (expect 55)"

# then: STAGE0=./coil-stage0 selfhost/rebootstrap-linux.sh
```

The `captures(none)` token (twice, on the `llvm.memcpy` declaration) is the LLVM-21
spelling; pre-21 parsers want `nocapture`. The sed is semantically inert.

## External link surface

libLLVM (C API), libc/libm/libpthread/libdl. **No Darwin symbols** — the historical
`dispatch_semaphore_*` (now pthread mutex+condvar in `metaengine.coil`) and
`sys_icache_invalidate` (now resolved via `dlsym` at runtime in `jit.coil`, null and
skipped on ELF hosts) are gone from the link surface.
