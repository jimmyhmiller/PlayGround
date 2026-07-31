# wasm-host — running the Coil compiler inside a wasm sandbox

`main_a64.coil` builds to a Wasm 3.0 **memory64** module and runs here:

    coil build src/compiler/main_a64.coil --target wasm64-unknown-unknown -o coilc.wasm
    node src/tooling/wasm-host/run-coil-wasm.mjs coilc.wasm check some.coil

`run-coil-wasm.mjs` provides the 44 `env.*` imports: a bump allocator over the module's own
linear memory (from the exported `__heap_base`), real-filesystem I/O, and LOUD TRAPS for the
Wall-1 comptime imports (mmap/mprotect/dlopen/dlsym/system/pthread_create). The compiler runs
until it needs comptime (macro expansion), which runs metaprograms as native code — impossible
in the sandbox.

## Crossing Wall 1 (option B, in progress)
Metaprograms will compile to wasm SIDE-MODULES that share the compiler's memory + table and are
instantiated by the host (`meta_run_wasm` primitive). `b1-*` proves the mechanism: a side-module
shares a running module's memory64/table64 and calls back both directly and via `call_indirect`.
Next: a native wasm backend (`codegen_wasm.coil`) to emit those modules. See the
`project_coil_wasm_running_compiler` memory for the full plan.
