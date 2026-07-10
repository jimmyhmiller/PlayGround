// In-process WebAssembly linker for the self-hosted compiler.
//
// LLVM's C API emits relocatable Wasm objects, but does not expose a Wasm
// linker. LLD does, as a C++ API. Keep that ABI boundary here so Coil only
// calls a plain C function and never needs to spawn wasm-ld.

#include "lld/Common/Driver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

LLD_HAS_DRIVER(wasm)

extern "C" int coil_wasm_link(const char *object_path, const char *output_path) {
  const char *args[] = {
      "wasm-ld",
      "--no-entry",
      "--export=main",
      "--allow-undefined",
      "-o",
      output_path,
      object_path,
  };

  return lld::wasm::link(llvm::ArrayRef(args), llvm::outs(), llvm::errs(),
                         false, false)
             ? 0
             : 1;
}
