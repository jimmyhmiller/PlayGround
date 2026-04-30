# === samples/one_var_unused.mlirbc ===
# MLIR bytecode: 2301 bytes, Modular trailer: 221 bytes
# decoded to 3322 chars of textual MLIR

#memory_handle = #interp.memory_handle<16, "Runtime\00" string>
module attributes {M.target_info = #M.target<triple = "arm64-apple-darwin25.3.0", arch = "apple-m2", features = "+aes,+bf16,+complxnum,+crc,+dotprod,+fp-armv8,+fp16fml,+fpac,+fullfp16,+i8mm,+jsconv,+lse,+neon,+pauth,+perfmon,+ras,+rcpc,+rdm,+sha2,+sha3,+ssbs", data_layout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32", relocation_model = "pic", simd_bit_width = 128, index_bit_width = 64, accelerator_arch = "metal:2-metal4">, kgen.env = #kgen.env<{__OPTIMIZATION_LEVEL = 3 : index, __SANITIZE_ADDRESS = 0 : index}>} {
  kgen.func @main_closure_0() -> !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)> {
    %0 = pop.external_call @KGEN_CompilerRT_AsyncRT_GetOrCreateRuntime() : () -> !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>
    kgen.return %0 : !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>
  }
  kgen.func @main_closure_1(%arg0: !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>) {
    pop.external_call @KGEN_CompilerRT_AsyncRT_ReleaseRuntime(%arg0) : (!kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>) -> ()
    kgen.return
  }
  kgen.func export C @main(%arg0: !pop.scalar<si32>, %arg1: !kgen.pointer<pointer<scalar<ui8>>>) -> !pop.scalar<si32> {
    %index-1 = kgen.param.constant = <-1>
    %index0 = kgen.param.constant = <0>
    %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle, const_global, [], [])], []}, 0, 0>, 7 }>
    %simd = kgen.param.constant: scalar<si32> = <0>
    %0 = pop.external_call @KGEN_CompilerRT_AsyncRT_GetCurrentRuntime() : () -> !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>
    %1 = kgen.struct.extract %0[0] : <(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>
    %2 = kgen.struct.extract %1[0] : <(struct<(struct<(pointer<none>) memoryOnly>)>)>
    %3 = kgen.struct.extract %2[0] : <(struct<(pointer<none>) memoryOnly>)>
    %4 = kgen.struct.extract %3[0] : <(pointer<none>) memoryOnly>
    %5 = pop.pointer_to_index %4 : <none>
    %6 = index.cmp eq(%5, %index0)
    %7 = pop.select %6, %index0, %index-1 : index
    %8 = index.cmp eq(%7, %index-1)
    hlcf.if %8 {
      hlcf.yield
    } else {
      %9 = kgen.create_closure[() -> !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>: @main_closure_0]() 
      %10 = kgen.create_closure[(!kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>) -> (): @main_closure_1]() 
      %11 = pop.external_call @KGEN_CompilerRT_GetOrCreateGlobal(%struct, %9, %10) : (!kgen.struct<(pointer<none>, index)>, !kgen.generator<() -> !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>>, !kgen.generator<(!kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>) -> ()>) -> !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>
      hlcf.yield
    }
    pop.external_call @KGEN_CompilerRT_SetArgV(%arg0, %arg1) : (!pop.scalar<si32>, !kgen.pointer<pointer<scalar<ui8>>>) -> ()
    pop.external_call @KGEN_CompilerRT_PrintStackTraceOnFault() : () -> ()
    pop.external_call @KGEN_CompilerRT_DestroyGlobals() : () -> ()
    kgen.return %simd : !pop.scalar<si32>
  }
}


