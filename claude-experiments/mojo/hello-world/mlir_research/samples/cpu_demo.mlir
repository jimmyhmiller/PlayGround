# === samples/cpu_demo.mlirbc ===
# MLIR bytecode: 123642 bytes, Modular trailer: 32 bytes
# decoded to 780166 chars of textual MLIR

#memory_handle = #interp.memory_handle<16, "\0A\00" string>
#memory_handle1 = #interp.memory_handle<16, " \00" string>
#memory_handle2 = #interp.memory_handle<16, "Runtime\00" string>
#memory_handle3 = #interp.memory_handle<16, "" string>
#memory_handle4 = #interp.memory_handle<16, "ABORT:\00" string>
#memory_handle5 = #interp.memory_handle<16, "0123456789abcdefghijklmnopqrstuvwxyz\00" string>
#memory_handle6 = #interp.memory_handle<16, "a\00" string>
module attributes {M.target_info = #M.target<triple = "arm64-apple-darwin25.3.0", arch = "apple-m2", features = "+aes,+bf16,+complxnum,+crc,+dotprod,+fp-armv8,+fp16fml,+fpac,+fullfp16,+i8mm,+jsconv,+lse,+neon,+pauth,+perfmon,+ras,+rcpc,+rdm,+sha2,+sha3,+ssbs", data_layout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32", relocation_model = "pic", simd_bit_width = 128, index_bit_width = 64, accelerator_arch = "metal:2-metal4">, kgen.env = #kgen.env<{__OPTIMIZATION_LEVEL = 3 : index, __SANITIZE_ADDRESS = 0 : index}>} {
  kgen.func @"cpu_demo::main()"() {
    %0 = kgen.param.constant: i1 = <0>
    %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle, const_global, [], [])], []}, 0, 0>, 1 }>
    %struct_0 = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle1, const_global, [], [])], []}, 0, 0>, 1 }>
    %index1 = kgen.param.constant = <1>
    %simd = kgen.param.constant: scalar<f64> = <"0">
    %index2305843009213693952 = kgen.param.constant = <2305843009213693952>
    %index5 = kgen.param.constant = <5>
    %string = kgen.param.constant: string = <"sum =">
    %simd_1 = kgen.param.constant: scalar<index> = <1>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %index0 = kgen.param.constant = <0>
    %idx10001 = index.constant 10001
    %idx-8 = index.constant -8
    %1 = hlcf.loop "_loop_0" (%arg0 = %index1 : index, %arg1 = %simd : !pop.scalar<f64>) -> !pop.scalar<f64> {
      %11 = index.add %arg0, %index1
      %12 = index.cmp eq(%arg0, %idx10001)
      %13 = pop.select %12, %arg0, %11 : index
      %14:2 = lit.try "try0" -> index, index {
        hlcf.if %12 {
          lit.try.raise "try0" %13, %arg0 : index, index
        } else {
          hlcf.yield
        }
        lit.try.yield %13, %arg0 : index, index
      } except (%arg2: index, %arg3: index) {
        hlcf.break "_loop_0" %arg1 : !pop.scalar<f64>
      } else (%arg2: index, %arg3: index) {
        lit.try.yield %arg2, %arg3 : index, index
      }
      %15 = pop.cast_from_builtin %14#1 : index to !pop.scalar<index>
      %16 = pop.cast %15 : !pop.scalar<index> to !pop.scalar<f64>
      %17 = kgen.struct.create(%16) : !kgen.struct<(scalar<f64>)>
      %18 = pop.call_llvm_intrinsic side_effecting<0> "llvm.sqrt", (%17) : (!kgen.struct<(scalar<f64>)>) -> !pop.scalar<f64>
      %19 = pop.add %arg1, %18 : !pop.scalar<f64>
      hlcf.continue "_loop_0" %14#0, %19 : index, !pop.scalar<f64>
    }
    %2 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
    pop.stack_alloc.lifetime.start(%2) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %3 = kgen.struct.gep %2[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index5, %3 : !kgen.pointer<index>
    %4 = kgen.struct.gep %2[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %5 = pop.string.address %string
    %6 = pop.pointer.bitcast %5 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    pop.store %6, %4 : !kgen.pointer<pointer<none>>
    %7 = kgen.struct.gep %2[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index2305843009213693952, %7 : !kgen.pointer<index>
    kgen.call @"std::io::io::print[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?],flush:::Bool,file:::FileDescriptor$),Ts.values`=[[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::builtin::simd::SIMD,dtype=f64,size=1\22>>, scalar<f64>]]"(%2, %1, %struct_0, %struct, %0, %index1) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !pop.scalar<f64>, !kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index)>, i1, index owned) -> ()
    %8 = pop.load %7 : !kgen.pointer<index>
    %9 = index.and %8, %index4611686018427387904
    %10 = index.cmp ne(%9, %index0)
    hlcf.if %10 {
      %11 = pop.load %4 : !kgen.pointer<pointer<none>>
      %12 = pop.pointer.bitcast %11 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %13 = pop.offset %12[%idx-8] : !kgen.pointer<scalar<ui8>>
      %14 = pop.pointer.bitcast %13 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
      %15 = kgen.struct.gep %14[0] : <struct<(scalar<index>) memoryOnly>>
      %16 = pop.atomic.rmw sub(%15, %simd_1) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
      %17 = pop.cmp eq(%16, %simd_1) : <1, index>
      %18 = pop.cast_to_builtin %17 : !pop.scalar<bool> to i1
      hlcf.if %18 {
        pop.fence syncscope("") acquire
        pop.aligned_free %13 : <scalar<ui8>>
        hlcf.yield
      } else {
        hlcf.yield
      }
      hlcf.yield
    } else {
      hlcf.yield
    }
    kgen.return
  }
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
    %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle2, const_global, [], [])], []}, 0, 0>, 7 }>
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
    kgen.call @"cpu_demo::main()"() : () -> ()
    pop.external_call @KGEN_CompilerRT_DestroyGlobals() : () -> ()
    kgen.return %simd : !pop.scalar<si32>
  }
  kgen.func @"std::builtin::builtin_slice::ContiguousSlice::indices(::ContiguousSlice,::Int)"(%arg0: !kgen.struct<(struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>, struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>) memoryOnly>, %arg1: index) -> !kgen.struct<(struct<(index, index)>)> {
    %0 = kgen.param.constant: i1 = <1>
    %simd = kgen.param.constant: scalar<ui8> = <1>
    %struct = kgen.param.constant: struct<()> = <{  }>
    %simd_0 = kgen.param.constant: scalar<ui8> = <0>
    %index0 = kgen.param.constant = <0>
    %1 = kgen.struct.extract %arg0[0] : <(struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>, struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>) memoryOnly>
    %2 = kgen.struct.extract %1[0] : <(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>
    %3 = kgen.struct.extract %2[0] : <(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>
    %4 = kgen.struct.extract %3[0] : <(struct<(union<struct<()>, index>, scalar<ui8>)>)>
    %5 = kgen.struct.extract %4[0] : <(union<struct<()>, index>, scalar<ui8>)>
    %6 = kgen.struct.extract %4[1] : <(union<struct<()>, index>, scalar<ui8>)>
    %7 = kgen.struct.extract %arg0[1] : <(struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>, struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>) memoryOnly>
    %8 = kgen.struct.extract %7[0] : <(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>
    %9 = kgen.struct.extract %8[0] : <(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>
    %10 = kgen.struct.extract %9[0] : <(struct<(union<struct<()>, index>, scalar<ui8>)>)>
    %11 = kgen.struct.extract %10[0] : <(union<struct<()>, index>, scalar<ui8>)>
    %12 = kgen.struct.extract %10[1] : <(union<struct<()>, index>, scalar<ui8>)>
    %13 = pop.stack_allocation 1 x union<struct<()>, index>
    %14 = pop.union.bitcast %13 : <union<struct<()>, index>> as <index>
    pop.store %5, %13 : !kgen.pointer<union<struct<()>, index>>
    %15 = pop.cmp eq(%6, %simd) : <1, ui8>
    %16 = pop.cast_to_builtin %15 : !pop.scalar<bool> to i1
    %17 = pop.stack_allocation 1 x union<struct<()>, index>
    %18 = pop.union.bitcast %17 : <union<struct<()>, index>> as <index>
    %19 = pop.union.bitcast %17 : <union<struct<()>, index>> as <struct<()>>
    %20 = pop.cmp eq(%6, %simd_0) : <1, ui8>
    %21 = pop.cast_to_builtin %20 : !pop.scalar<bool> to i1
    %22 = hlcf.if %21 -> !kgen.struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)> {
      pop.store %struct, %19 : !kgen.pointer<struct<()>>
      %61 = pop.load %17 : !kgen.pointer<union<struct<()>, index>>
      %62 = kgen.struct.create(%61, %6) : !kgen.struct<(union<struct<()>, index>, scalar<ui8>)>
      %63 = kgen.struct.create(%62) : !kgen.struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>
      hlcf.yield %63 : !kgen.struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>
    } else {
      %61 = hlcf.if %16 -> !kgen.struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)> {
        %62 = pop.load %14 : !kgen.pointer<index>
        pop.store %62, %18 : !kgen.pointer<index>
        %63 = pop.load %17 : !kgen.pointer<union<struct<()>, index>>
        %64 = kgen.struct.create(%63, %6) : !kgen.struct<(union<struct<()>, index>, scalar<ui8>)>
        %65 = kgen.struct.create(%64) : !kgen.struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>
        hlcf.yield %65 : !kgen.struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>
      } else {
        %62 = pop.load %17 : !kgen.pointer<union<struct<()>, index>>
        %63 = kgen.struct.create(%62, %6) : !kgen.struct<(union<struct<()>, index>, scalar<ui8>)>
        %64 = kgen.struct.create(%63) : !kgen.struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>
        hlcf.yield %64 : !kgen.struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>
      }
      hlcf.yield %61 : !kgen.struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>
    }
    %23 = kgen.struct.extract %22[0] : <(struct<(union<struct<()>, index>, scalar<ui8>)>)>
    %24 = kgen.struct.extract %23[0] : <(union<struct<()>, index>, scalar<ui8>)>
    %25 = kgen.struct.extract %23[1] : <(union<struct<()>, index>, scalar<ui8>)>
    %26 = pop.cmp eq(%25, %simd_0) : <1, ui8>
    %27 = pop.cast_to_builtin %26 : !pop.scalar<bool> to i1
    %28 = pop.xor %27, %0
    %29 = hlcf.if %28 -> index {
      %61 = pop.stack_allocation 1 x union<struct<()>, index>
      pop.store %24, %61 : !kgen.pointer<union<struct<()>, index>>
      %62 = pop.union.bitcast %61 : <union<struct<()>, index>> as <index>
      %63 = pop.load %62 : !kgen.pointer<index>
      hlcf.yield %63 : index
    } else {
      hlcf.yield %index0 : index
    }
    %30 = index.cmp sge(%29, %arg1)
    %31 = pop.select %30, %arg1, %29 : index
    %32 = index.add %29, %arg1
    %33 = index.maxs %32, %index0
    %34 = pop.stack_allocation 1 x union<struct<()>, index>
    %35 = pop.union.bitcast %34 : <union<struct<()>, index>> as <index>
    pop.store %11, %34 : !kgen.pointer<union<struct<()>, index>>
    %36 = pop.cmp eq(%12, %simd) : <1, ui8>
    %37 = pop.cast_to_builtin %36 : !pop.scalar<bool> to i1
    %38 = pop.stack_allocation 1 x union<struct<()>, index>
    %39 = pop.union.bitcast %38 : <union<struct<()>, index>> as <index>
    %40 = pop.union.bitcast %38 : <union<struct<()>, index>> as <struct<()>>
    %41 = pop.cmp eq(%12, %simd_0) : <1, ui8>
    %42 = pop.cast_to_builtin %41 : !pop.scalar<bool> to i1
    %43 = hlcf.if %42 -> !kgen.struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)> {
      pop.store %struct, %40 : !kgen.pointer<struct<()>>
      %61 = pop.load %38 : !kgen.pointer<union<struct<()>, index>>
      %62 = kgen.struct.create(%61, %12) : !kgen.struct<(union<struct<()>, index>, scalar<ui8>)>
      %63 = kgen.struct.create(%62) : !kgen.struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>
      hlcf.yield %63 : !kgen.struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>
    } else {
      %61 = hlcf.if %37 -> !kgen.struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)> {
        %62 = pop.load %35 : !kgen.pointer<index>
        pop.store %62, %39 : !kgen.pointer<index>
        %63 = pop.load %38 : !kgen.pointer<union<struct<()>, index>>
        %64 = kgen.struct.create(%63, %12) : !kgen.struct<(union<struct<()>, index>, scalar<ui8>)>
        %65 = kgen.struct.create(%64) : !kgen.struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>
        hlcf.yield %65 : !kgen.struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>
      } else {
        %62 = pop.load %38 : !kgen.pointer<union<struct<()>, index>>
        %63 = kgen.struct.create(%62, %12) : !kgen.struct<(union<struct<()>, index>, scalar<ui8>)>
        %64 = kgen.struct.create(%63) : !kgen.struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>
        hlcf.yield %64 : !kgen.struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>
      }
      hlcf.yield %61 : !kgen.struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>
    }
    %44 = kgen.struct.extract %43[0] : <(struct<(union<struct<()>, index>, scalar<ui8>)>)>
    %45 = kgen.struct.extract %44[0] : <(union<struct<()>, index>, scalar<ui8>)>
    %46 = kgen.struct.extract %44[1] : <(union<struct<()>, index>, scalar<ui8>)>
    %47 = pop.cmp eq(%46, %simd_0) : <1, ui8>
    %48 = pop.cast_to_builtin %47 : !pop.scalar<bool> to i1
    %49 = pop.xor %48, %0
    %50 = hlcf.if %49 -> index {
      %61 = pop.stack_allocation 1 x union<struct<()>, index>
      pop.store %45, %61 : !kgen.pointer<union<struct<()>, index>>
      %62 = pop.union.bitcast %61 : <union<struct<()>, index>> as <index>
      %63 = pop.load %62 : !kgen.pointer<index>
      hlcf.yield %63 : index
    } else {
      hlcf.yield %arg1 : index
    }
    %51 = index.cmp sge(%50, %arg1)
    %52 = pop.select %51, %arg1, %50 : index
    %53 = index.add %50, %arg1
    %54 = index.maxs %53, %index0
    %55 = index.cmp slt(%29, %index0)
    %56 = pop.select %55, %33, %31 : index
    %57 = index.cmp slt(%50, %index0)
    %58 = pop.select %57, %54, %52 : index
    %59 = kgen.struct.create(%56, %58) : !kgen.struct<(index, index)>
    %60 = kgen.struct.create(%59) : !kgen.struct<(struct<(index, index)>)>
    kgen.return %60 : !kgen.struct<(struct<(index, index)>)>
  }
  kgen.func @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%arg0: index, %arg1: !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly> {
    %0 = pop.cast_from_builtin %arg0 : index to !pop.scalar<index>
    %1 = pop.cast %0 : !pop.scalar<index> to !pop.scalar<si64>
    %2 = kgen.call @"std::format::__init__::Writer::write[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $3]]($0&,*$1),_Self`=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>],Ts.values`2x=[[typevalue<#kgen.instref<\1B\22std::builtin::simd::SIMD,dtype=si64,size=1\22>>, scalar<si64>]]"(%arg1, %1) : (!kgen.struct<(pointer<none>, index) memoryOnly>, !pop.scalar<si64>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
    kgen.return %2 : !kgen.struct<(pointer<none>, index) memoryOnly>
  }
  kgen.func @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>]"(%arg0: index, %arg1: !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut) {
    %0 = pop.cast_from_builtin %arg0 : index to !pop.scalar<index>
    %1 = pop.cast %0 : !pop.scalar<index> to !pop.scalar<si64>
    kgen.call @"std::format::__init__::Writer::write[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $3]]($0&,*$1),_Self`=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>],Ts.values`2x=[[typevalue<#kgen.instref<\1B\22std::builtin::simd::SIMD,dtype=si64,size=1\22>>, scalar<si64>]]"(%arg1, %1) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !pop.scalar<si64>) -> ()
    kgen.return
  }
  kgen.func @"std::builtin::simd::SIMD::write_to[::Writer](::SIMD[$0, $1],$2&),dtype=f64,size=1,writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>]"(%arg0: !pop.scalar<f64>, %arg1: !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut) no_inline {
    %string = kgen.param.constant: string = <", ">
    %simd = kgen.param.constant: scalar<index> = <1>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %index56 = kgen.param.constant = <56>
    %index2233785415175766016 = kgen.param.constant = <2233785415175766016>
    %idx-8 = index.constant -8
    %index2 = kgen.param.constant = <2>
    %index2305843009213693952 = kgen.param.constant = <2305843009213693952>
    %index1 = kgen.param.constant = <1>
    %index0 = kgen.param.constant = <0>
    %idx1 = index.constant 1
    %0 = pop.string.address %string
    %1 = pop.pointer.bitcast %0 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    hlcf.loop "_loop_0" (%arg2 = %idx1 : index) {
      %2 = index.sub %idx1, %arg2
      %3 = index.sub %arg2, %index1
      %4 = index.cmp eq(%arg2, %index0)
      %5:2 = lit.try "try0" -> index, index {
        %7 = pop.select %4, %arg2, %3 : index
        hlcf.if %4 {
          lit.try.raise "try0" %7, %2 : index, index
        } else {
          hlcf.yield
        }
        lit.try.yield %7, %2 : index, index
      } except (%arg3: index, %arg4: index) {
        hlcf.break "_loop_0"
      } else (%arg3: index, %arg4: index) {
        lit.try.yield %arg3, %arg4 : index, index
      }
      %6 = index.cmp ne(%5#1, %index0)
      hlcf.if %6 {
        %7 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
        pop.stack_alloc.lifetime.start(%7) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
        %8 = kgen.struct.gep %7[1] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %index2, %8 : !kgen.pointer<index>
        %9 = kgen.struct.gep %7[0] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %1, %9 : !kgen.pointer<pointer<none>>
        %10 = kgen.struct.gep %7[2] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %index2305843009213693952, %10 : !kgen.pointer<index>
        %11 = pop.pointer.bitcast %7 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
        %12 = pop.load %10 : !kgen.pointer<index>
        %13 = index.and %12, %index-9223372036854775808
        %14 = index.cmp ne(%13, %index0)
        %15 = hlcf.if %14 -> !kgen.pointer<none> {
          hlcf.yield %11 : !kgen.pointer<none>
        } else {
          %24 = pop.load %9 : !kgen.pointer<pointer<none>>
          hlcf.yield %24 : !kgen.pointer<none>
        }
        %16 = pop.load %10 : !kgen.pointer<index>
        %17 = index.and %16, %index-9223372036854775808
        %18 = index.cmp ne(%17, %index0)
        %19 = hlcf.if %18 -> index {
          %24 = pop.load %10 : !kgen.pointer<index>
          %25 = index.and %24, %index2233785415175766016
          %26 = index.shrs %25, %index56
          hlcf.yield %26 : index
        } else {
          %24 = pop.load %8 : !kgen.pointer<index>
          hlcf.yield %24 : index
        }
        %20 = kgen.struct.create(%15, %19) : !kgen.struct<(pointer<none>, index)>
        kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg1, %20) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
        %21 = pop.load %10 : !kgen.pointer<index>
        %22 = index.and %21, %index4611686018427387904
        %23 = index.cmp ne(%22, %index0)
        hlcf.if %23 {
          %24 = pop.load %9 : !kgen.pointer<pointer<none>>
          %25 = pop.pointer.bitcast %24 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
          %26 = pop.offset %25[%idx-8] : !kgen.pointer<scalar<ui8>>
          %27 = pop.pointer.bitcast %26 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
          %28 = kgen.struct.gep %27[0] : <struct<(scalar<index>) memoryOnly>>
          %29 = pop.atomic.rmw sub(%28, %simd) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
          %30 = pop.cmp eq(%29, %simd) : <1, index>
          %31 = pop.cast_to_builtin %30 : !pop.scalar<bool> to i1
          hlcf.if %31 {
            pop.fence syncscope("") acquire
            pop.aligned_free %26 : <scalar<ui8>>
            hlcf.yield
          } else {
            hlcf.yield
          }
          hlcf.yield
        } else {
          hlcf.yield
        }
        pop.stack_alloc.lifetime.end(%7) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
        hlcf.yield
      } else {
        hlcf.yield
      }
      kgen.call @"std::builtin::_format_float::_write_float[::Writer,::DType]($0&,::SIMD[$1, ::Int(1)]){#pop.cast_to_builtin<#pop.simd_xor<#pop.simd_cmp<eq, #pop.simd_and<#pop.cast_from_builtin<#pop.dtype_to_ui8<#lit.struct.extract<:!lit.struct<_std::_builtin::_dtype::_DType> *(0,1), \22_mlir_value\22>> : ui8> : !pop.scalar<ui8>, #pop<simd 64> : !pop.scalar<ui8>> : !pop.scalar<ui8>, #pop<simd 0> : !pop.scalar<ui8>> : !pop.scalar<bool>, #pop<simd true> : !pop.scalar<bool>> : !pop.scalar<bool>>},W=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>],dtype=f64"(%arg1, %arg0) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !pop.scalar<f64>) -> ()
      hlcf.continue "_loop_0" %5#0 : index
    }
    kgen.return
  }
  kgen.func @"std::builtin::simd::SIMD::write_to[::Writer](::SIMD[$0, $1],$2&),dtype=si64,size=1,writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%arg0: !pop.scalar<si64>, %arg1: !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly> no_inline {
    %idx1 = index.constant 1
    %index0 = kgen.param.constant = <0>
    %index1 = kgen.param.constant = <1>
    %index2305843009213693952 = kgen.param.constant = <2305843009213693952>
    %index2 = kgen.param.constant = <2>
    %idx-8 = index.constant -8
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %simd = kgen.param.constant: scalar<index> = <1>
    %string = kgen.param.constant: string = <", ">
    %0 = kgen.struct.extract %arg1[0] : <(pointer<none>, index) memoryOnly>
    %1 = kgen.struct.extract %arg1[1] : <(pointer<none>, index) memoryOnly>
    %2 = pop.string.address %string
    %3 = pop.pointer.bitcast %2 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %4:2 = hlcf.loop "_loop_0" (%arg2 = %idx1 : index, %arg3 = %0 : !kgen.pointer<none>, %arg4 = %1 : index) -> (!kgen.pointer<none>, index) {
      %6 = index.sub %idx1, %arg2
      %7 = index.sub %arg2, %index1
      %8 = index.cmp eq(%arg2, %index0)
      %9:2 = lit.try "try0" -> index, index {
        %16 = pop.select %8, %arg2, %7 : index
        hlcf.if %8 {
          lit.try.raise "try0" %16, %6 : index, index
        } else {
          hlcf.yield
        }
        lit.try.yield %16, %6 : index, index
      } except (%arg5: index, %arg6: index) {
        hlcf.break "_loop_0" %arg3, %arg4 : !kgen.pointer<none>, index
      } else (%arg5: index, %arg6: index) {
        lit.try.yield %arg5, %arg6 : index, index
      }
      %10 = index.cmp ne(%9#1, %index0)
      %11:2 = hlcf.if %10 -> !kgen.pointer<none>, index {
        %16 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
        pop.stack_alloc.lifetime.start(%16) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
        %17 = kgen.struct.gep %16[1] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %index2, %17 : !kgen.pointer<index>
        %18 = kgen.struct.gep %16[0] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %3, %18 : !kgen.pointer<pointer<none>>
        %19 = kgen.struct.gep %16[2] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %index2305843009213693952, %19 : !kgen.pointer<index>
        %20 = kgen.struct.create(%arg3, %arg4) : !kgen.struct<(pointer<none>, index) memoryOnly>
        %21 = kgen.call @"std::collections::string::string::String::write_to[::Writer](::String,$0&),writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%16, %20) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
        %22 = kgen.struct.extract %21[0] : <(pointer<none>, index) memoryOnly>
        %23 = kgen.struct.extract %21[1] : <(pointer<none>, index) memoryOnly>
        %24 = pop.load %19 : !kgen.pointer<index>
        %25 = index.and %24, %index4611686018427387904
        %26 = index.cmp ne(%25, %index0)
        hlcf.if %26 {
          %27 = pop.load %18 : !kgen.pointer<pointer<none>>
          %28 = pop.pointer.bitcast %27 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
          %29 = pop.offset %28[%idx-8] : !kgen.pointer<scalar<ui8>>
          %30 = pop.pointer.bitcast %29 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
          %31 = kgen.struct.gep %30[0] : <struct<(scalar<index>) memoryOnly>>
          %32 = pop.atomic.rmw sub(%31, %simd) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
          %33 = pop.cmp eq(%32, %simd) : <1, index>
          %34 = pop.cast_to_builtin %33 : !pop.scalar<bool> to i1
          hlcf.if %34 {
            pop.fence syncscope("") acquire
            pop.aligned_free %29 : <scalar<ui8>>
            hlcf.yield
          } else {
            hlcf.yield
          }
          hlcf.yield
        } else {
          hlcf.yield
        }
        pop.stack_alloc.lifetime.end(%16) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
        hlcf.yield %22, %23 : !kgen.pointer<none>, index
      } else {
        hlcf.yield %arg3, %arg4 : !kgen.pointer<none>, index
      }
      %12 = kgen.struct.create(%11#0, %11#1) : !kgen.struct<(pointer<none>, index) memoryOnly>
      %13 = kgen.call tail @"std::builtin::simd::_write_scalar[::DType,::Writer]($1&,::SIMD[$0, ::Int(1)]),dtype=si64,W=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%12, %arg0) : (!kgen.struct<(pointer<none>, index) memoryOnly>, !pop.scalar<si64>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
      %14 = kgen.struct.extract %13[0] : <(pointer<none>, index) memoryOnly>
      %15 = kgen.struct.extract %13[1] : <(pointer<none>, index) memoryOnly>
      hlcf.continue "_loop_0" %9#0, %14, %15 : index, !kgen.pointer<none>, index
    }
    %5 = kgen.struct.create(%4#0, %4#1) : !kgen.struct<(pointer<none>, index) memoryOnly>
    kgen.return %5 : !kgen.struct<(pointer<none>, index) memoryOnly>
  }
  kgen.func @"std::builtin::simd::SIMD::write_to[::Writer](::SIMD[$0, $1],$2&),dtype=si64,size=1,writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>]"(%arg0: !pop.scalar<si64>, %arg1: !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut) no_inline {
    %string = kgen.param.constant: string = <", ">
    %simd = kgen.param.constant: scalar<index> = <1>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %index56 = kgen.param.constant = <56>
    %index2233785415175766016 = kgen.param.constant = <2233785415175766016>
    %idx-8 = index.constant -8
    %index2 = kgen.param.constant = <2>
    %index2305843009213693952 = kgen.param.constant = <2305843009213693952>
    %index1 = kgen.param.constant = <1>
    %index0 = kgen.param.constant = <0>
    %idx1 = index.constant 1
    %0 = pop.string.address %string
    %1 = pop.pointer.bitcast %0 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    hlcf.loop "_loop_0" (%arg2 = %idx1 : index) {
      %2 = index.sub %idx1, %arg2
      %3 = index.sub %arg2, %index1
      %4 = index.cmp eq(%arg2, %index0)
      %5:2 = lit.try "try0" -> index, index {
        %7 = pop.select %4, %arg2, %3 : index
        hlcf.if %4 {
          lit.try.raise "try0" %7, %2 : index, index
        } else {
          hlcf.yield
        }
        lit.try.yield %7, %2 : index, index
      } except (%arg3: index, %arg4: index) {
        hlcf.break "_loop_0"
      } else (%arg3: index, %arg4: index) {
        lit.try.yield %arg3, %arg4 : index, index
      }
      %6 = index.cmp ne(%5#1, %index0)
      hlcf.if %6 {
        %7 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
        pop.stack_alloc.lifetime.start(%7) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
        %8 = kgen.struct.gep %7[1] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %index2, %8 : !kgen.pointer<index>
        %9 = kgen.struct.gep %7[0] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %1, %9 : !kgen.pointer<pointer<none>>
        %10 = kgen.struct.gep %7[2] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %index2305843009213693952, %10 : !kgen.pointer<index>
        %11 = pop.pointer.bitcast %7 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
        %12 = pop.load %10 : !kgen.pointer<index>
        %13 = index.and %12, %index-9223372036854775808
        %14 = index.cmp ne(%13, %index0)
        %15 = hlcf.if %14 -> !kgen.pointer<none> {
          hlcf.yield %11 : !kgen.pointer<none>
        } else {
          %24 = pop.load %9 : !kgen.pointer<pointer<none>>
          hlcf.yield %24 : !kgen.pointer<none>
        }
        %16 = pop.load %10 : !kgen.pointer<index>
        %17 = index.and %16, %index-9223372036854775808
        %18 = index.cmp ne(%17, %index0)
        %19 = hlcf.if %18 -> index {
          %24 = pop.load %10 : !kgen.pointer<index>
          %25 = index.and %24, %index2233785415175766016
          %26 = index.shrs %25, %index56
          hlcf.yield %26 : index
        } else {
          %24 = pop.load %8 : !kgen.pointer<index>
          hlcf.yield %24 : index
        }
        %20 = kgen.struct.create(%15, %19) : !kgen.struct<(pointer<none>, index)>
        kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg1, %20) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
        %21 = pop.load %10 : !kgen.pointer<index>
        %22 = index.and %21, %index4611686018427387904
        %23 = index.cmp ne(%22, %index0)
        hlcf.if %23 {
          %24 = pop.load %9 : !kgen.pointer<pointer<none>>
          %25 = pop.pointer.bitcast %24 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
          %26 = pop.offset %25[%idx-8] : !kgen.pointer<scalar<ui8>>
          %27 = pop.pointer.bitcast %26 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
          %28 = kgen.struct.gep %27[0] : <struct<(scalar<index>) memoryOnly>>
          %29 = pop.atomic.rmw sub(%28, %simd) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
          %30 = pop.cmp eq(%29, %simd) : <1, index>
          %31 = pop.cast_to_builtin %30 : !pop.scalar<bool> to i1
          hlcf.if %31 {
            pop.fence syncscope("") acquire
            pop.aligned_free %26 : <scalar<ui8>>
            hlcf.yield
          } else {
            hlcf.yield
          }
          hlcf.yield
        } else {
          hlcf.yield
        }
        pop.stack_alloc.lifetime.end(%7) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
        hlcf.yield
      } else {
        hlcf.yield
      }
      kgen.call tail @"std::builtin::simd::_write_scalar[::DType,::Writer]($1&,::SIMD[$0, ::Int(1)]),dtype=si64,W=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>]"(%arg1, %arg0) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !pop.scalar<si64>) -> ()
      hlcf.continue "_loop_0" %5#0 : index
    }
    kgen.return
  }
  kgen.func @"std::builtin::simd::SIMD::write_to[::Writer](::SIMD[$0, $1],$2&),dtype=ui64,size=1,writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>]"(%arg0: !pop.scalar<ui64>, %arg1: !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut) no_inline {
    %string = kgen.param.constant: string = <", ">
    %simd = kgen.param.constant: scalar<index> = <1>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %index56 = kgen.param.constant = <56>
    %index2233785415175766016 = kgen.param.constant = <2233785415175766016>
    %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle3, const_global, [], [])], []}, 0, 0>, 0 }>
    %idx-8 = index.constant -8
    %index2 = kgen.param.constant = <2>
    %index2305843009213693952 = kgen.param.constant = <2305843009213693952>
    %index1 = kgen.param.constant = <1>
    %index0 = kgen.param.constant = <0>
    %idx1 = index.constant 1
    %0 = pop.string.address %string
    %1 = pop.pointer.bitcast %0 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    hlcf.loop "_loop_0" (%arg2 = %idx1 : index) {
      %2 = index.sub %idx1, %arg2
      %3 = index.sub %arg2, %index1
      %4 = index.cmp eq(%arg2, %index0)
      %5:2 = lit.try "try0" -> index, index {
        %7 = pop.select %4, %arg2, %3 : index
        hlcf.if %4 {
          lit.try.raise "try0" %7, %2 : index, index
        } else {
          hlcf.yield
        }
        lit.try.yield %7, %2 : index, index
      } except (%arg3: index, %arg4: index) {
        hlcf.break "_loop_0"
      } else (%arg3: index, %arg4: index) {
        lit.try.yield %arg3, %arg4 : index, index
      }
      %6 = index.cmp ne(%5#1, %index0)
      hlcf.if %6 {
        %7 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
        pop.stack_alloc.lifetime.start(%7) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
        %8 = kgen.struct.gep %7[1] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %index2, %8 : !kgen.pointer<index>
        %9 = kgen.struct.gep %7[0] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %1, %9 : !kgen.pointer<pointer<none>>
        %10 = kgen.struct.gep %7[2] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %index2305843009213693952, %10 : !kgen.pointer<index>
        %11 = pop.pointer.bitcast %7 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
        %12 = pop.load %10 : !kgen.pointer<index>
        %13 = index.and %12, %index-9223372036854775808
        %14 = index.cmp ne(%13, %index0)
        %15 = hlcf.if %14 -> !kgen.pointer<none> {
          hlcf.yield %11 : !kgen.pointer<none>
        } else {
          %24 = pop.load %9 : !kgen.pointer<pointer<none>>
          hlcf.yield %24 : !kgen.pointer<none>
        }
        %16 = pop.load %10 : !kgen.pointer<index>
        %17 = index.and %16, %index-9223372036854775808
        %18 = index.cmp ne(%17, %index0)
        %19 = hlcf.if %18 -> index {
          %24 = pop.load %10 : !kgen.pointer<index>
          %25 = index.and %24, %index2233785415175766016
          %26 = index.shrs %25, %index56
          hlcf.yield %26 : index
        } else {
          %24 = pop.load %8 : !kgen.pointer<index>
          hlcf.yield %24 : index
        }
        %20 = kgen.struct.create(%15, %19) : !kgen.struct<(pointer<none>, index)>
        kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg1, %20) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
        %21 = pop.load %10 : !kgen.pointer<index>
        %22 = index.and %21, %index4611686018427387904
        %23 = index.cmp ne(%22, %index0)
        hlcf.if %23 {
          %24 = pop.load %9 : !kgen.pointer<pointer<none>>
          %25 = pop.pointer.bitcast %24 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
          %26 = pop.offset %25[%idx-8] : !kgen.pointer<scalar<ui8>>
          %27 = pop.pointer.bitcast %26 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
          %28 = kgen.struct.gep %27[0] : <struct<(scalar<index>) memoryOnly>>
          %29 = pop.atomic.rmw sub(%28, %simd) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
          %30 = pop.cmp eq(%29, %simd) : <1, index>
          %31 = pop.cast_to_builtin %30 : !pop.scalar<bool> to i1
          hlcf.if %31 {
            pop.fence syncscope("") acquire
            pop.aligned_free %26 : <scalar<ui8>>
            hlcf.yield
          } else {
            hlcf.yield
          }
          hlcf.yield
        } else {
          hlcf.yield
        }
        pop.stack_alloc.lifetime.end(%7) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
        hlcf.yield
      } else {
        hlcf.yield
      }
      kgen.call @"std::builtin::format_int::_write_int[::DType,::Writer,::Int,::StringSlice[::Bool(False), StaticConstantOrigin, *?]]($1&,::SIMD[$0, ::Int(1)],prefix:::StringSlice[::Bool(False), StaticConstantOrigin, *?]),dtype=ui64,W=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>],radix=10,digit_chars={ #interp.memref<{[(#interp.memory_handle<16, \220123456789abcdefghijklmnopqrstuvwxyz\\00\22 string>, const_global, [], [])], []}, 0, 0>, 36 }"(%arg1, %arg0, %struct) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !pop.scalar<ui64>, !kgen.struct<(pointer<none>, index)>) -> ()
      hlcf.continue "_loop_0" %5#0 : index
    }
    kgen.return
  }
  kgen.func @"std::builtin::simd::SIMD::write_to[::Writer](::SIMD[$0, $1],$2&),dtype=ui8,size=1,writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>]"(%arg0: !pop.scalar<ui8>, %arg1: !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut) no_inline {
    %string = kgen.param.constant: string = <", ">
    %simd = kgen.param.constant: scalar<index> = <1>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %index56 = kgen.param.constant = <56>
    %index2233785415175766016 = kgen.param.constant = <2233785415175766016>
    %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle3, const_global, [], [])], []}, 0, 0>, 0 }>
    %idx-8 = index.constant -8
    %index2 = kgen.param.constant = <2>
    %index2305843009213693952 = kgen.param.constant = <2305843009213693952>
    %index1 = kgen.param.constant = <1>
    %index0 = kgen.param.constant = <0>
    %idx1 = index.constant 1
    %0 = pop.string.address %string
    %1 = pop.pointer.bitcast %0 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    hlcf.loop "_loop_0" (%arg2 = %idx1 : index) {
      %2 = index.sub %idx1, %arg2
      %3 = index.sub %arg2, %index1
      %4 = index.cmp eq(%arg2, %index0)
      %5:2 = lit.try "try0" -> index, index {
        %7 = pop.select %4, %arg2, %3 : index
        hlcf.if %4 {
          lit.try.raise "try0" %7, %2 : index, index
        } else {
          hlcf.yield
        }
        lit.try.yield %7, %2 : index, index
      } except (%arg3: index, %arg4: index) {
        hlcf.break "_loop_0"
      } else (%arg3: index, %arg4: index) {
        lit.try.yield %arg3, %arg4 : index, index
      }
      %6 = index.cmp ne(%5#1, %index0)
      hlcf.if %6 {
        %7 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
        pop.stack_alloc.lifetime.start(%7) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
        %8 = kgen.struct.gep %7[1] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %index2, %8 : !kgen.pointer<index>
        %9 = kgen.struct.gep %7[0] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %1, %9 : !kgen.pointer<pointer<none>>
        %10 = kgen.struct.gep %7[2] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %index2305843009213693952, %10 : !kgen.pointer<index>
        %11 = pop.pointer.bitcast %7 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
        %12 = pop.load %10 : !kgen.pointer<index>
        %13 = index.and %12, %index-9223372036854775808
        %14 = index.cmp ne(%13, %index0)
        %15 = hlcf.if %14 -> !kgen.pointer<none> {
          hlcf.yield %11 : !kgen.pointer<none>
        } else {
          %24 = pop.load %9 : !kgen.pointer<pointer<none>>
          hlcf.yield %24 : !kgen.pointer<none>
        }
        %16 = pop.load %10 : !kgen.pointer<index>
        %17 = index.and %16, %index-9223372036854775808
        %18 = index.cmp ne(%17, %index0)
        %19 = hlcf.if %18 -> index {
          %24 = pop.load %10 : !kgen.pointer<index>
          %25 = index.and %24, %index2233785415175766016
          %26 = index.shrs %25, %index56
          hlcf.yield %26 : index
        } else {
          %24 = pop.load %8 : !kgen.pointer<index>
          hlcf.yield %24 : index
        }
        %20 = kgen.struct.create(%15, %19) : !kgen.struct<(pointer<none>, index)>
        kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg1, %20) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
        %21 = pop.load %10 : !kgen.pointer<index>
        %22 = index.and %21, %index4611686018427387904
        %23 = index.cmp ne(%22, %index0)
        hlcf.if %23 {
          %24 = pop.load %9 : !kgen.pointer<pointer<none>>
          %25 = pop.pointer.bitcast %24 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
          %26 = pop.offset %25[%idx-8] : !kgen.pointer<scalar<ui8>>
          %27 = pop.pointer.bitcast %26 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
          %28 = kgen.struct.gep %27[0] : <struct<(scalar<index>) memoryOnly>>
          %29 = pop.atomic.rmw sub(%28, %simd) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
          %30 = pop.cmp eq(%29, %simd) : <1, index>
          %31 = pop.cast_to_builtin %30 : !pop.scalar<bool> to i1
          hlcf.if %31 {
            pop.fence syncscope("") acquire
            pop.aligned_free %26 : <scalar<ui8>>
            hlcf.yield
          } else {
            hlcf.yield
          }
          hlcf.yield
        } else {
          hlcf.yield
        }
        pop.stack_alloc.lifetime.end(%7) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
        hlcf.yield
      } else {
        hlcf.yield
      }
      kgen.call @"std::builtin::format_int::_write_int[::DType,::Writer,::Int,::StringSlice[::Bool(False), StaticConstantOrigin, *?]]($1&,::SIMD[$0, ::Int(1)],prefix:::StringSlice[::Bool(False), StaticConstantOrigin, *?]),dtype=ui8,W=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>],radix=10,digit_chars={ #interp.memref<{[(#interp.memory_handle<16, \220123456789abcdefghijklmnopqrstuvwxyz\\00\22 string>, const_global, [], [])], []}, 0, 0>, 36 }"(%arg1, %arg0, %struct) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !pop.scalar<ui8>, !kgen.struct<(pointer<none>, index)>) -> ()
      hlcf.continue "_loop_0" %5#0 : index
    }
    kgen.return
  }
  kgen.func @"std::collections::string::string::String::write_to[::Writer](::String,$0&),writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%arg0: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, %arg1: !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly> {
    %index1 = kgen.param.constant = <1>
    %index2048 = kgen.param.constant = <2048>
    %index16 = kgen.param.constant = <16>
    %index2 = kgen.param.constant = <2>
    %index5 = kgen.param.constant = <5>
    %index8 = kgen.param.constant = <8>
    %index32 = kgen.param.constant = <32>
    %simd = kgen.param.constant: scalar<uindex> = <32>
    %idx-8 = index.constant -8
    %idx-4 = index.constant -4
    %none = kgen.param.constant: none = <#kgen.none>
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %index0 = kgen.param.constant = <0>
    %index56 = kgen.param.constant = <56>
    %index2233785415175766016 = kgen.param.constant = <2233785415175766016>
    %0 = kgen.struct.extract %arg1[0] : <(pointer<none>, index) memoryOnly>
    %1 = kgen.struct.extract %arg1[1] : <(pointer<none>, index) memoryOnly>
    %2 = kgen.struct.gep %arg0[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %3 = pop.pointer.bitcast %arg0 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
    %4 = kgen.struct.gep %arg0[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %5 = pop.load %4 : !kgen.pointer<index>
    %6 = index.and %5, %index-9223372036854775808
    %7 = index.cmp ne(%6, %index0)
    %8 = hlcf.if %7 -> !kgen.pointer<none> {
      hlcf.yield %3 : !kgen.pointer<none>
    } else {
      %61 = pop.load %2 : !kgen.pointer<pointer<none>>
      hlcf.yield %61 : !kgen.pointer<none>
    }
    %9 = kgen.struct.gep %arg0[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %10 = pop.load %4 : !kgen.pointer<index>
    %11 = index.and %10, %index-9223372036854775808
    %12 = index.cmp ne(%11, %index0)
    %13 = hlcf.if %12 -> index {
      %61 = pop.load %4 : !kgen.pointer<index>
      %62 = index.and %61, %index2233785415175766016
      %63 = index.shrs %62, %index56
      hlcf.yield %63 : index
    } else {
      %61 = pop.load %9 : !kgen.pointer<index>
      hlcf.yield %61 : index
    }
    %14 = index.add %13, %1
    %15 = index.cmp sgt(%14, %index2048)
    hlcf.if %15 {
      kgen.call @"std::io::io::_printf[KGENParamList[::AnyType],::StringSlice[::Bool(False), StaticConstantOrigin, *?],*::AnyType,LITImmutOrigin,::Origin[::Bool(False), $3]](*$0,file:::FileDescriptor),types.values`=[],fmt={ #interp.memref<{[(#interp.memory_handle<16, \22HEAP_BUFFER_BYTES exceeded, increase with: `mojo -D HEAP_BUFFER_BYTES=4096`\\0A\\00\22 string>, const_global, [], [])], []}, 0, 0>, 76 }"(%index1) : (index) -> ()
      llvm.intr.trap
      hlcf.loop "_loop_0" {
        hlcf.continue "_loop_0"
      }
      kgen.unreachable
    } else {
      hlcf.yield
    }
    %16 = pop.pointer.bitcast %0 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
    %17 = pop.offset %16[%1] : !kgen.pointer<scalar<ui8>>
    %18 = pop.pointer.bitcast %17 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<none>
    %19 = pop.stack_allocation 1 x pointer<none>
    %20 = pop.pointer.bitcast %19 : !kgen.pointer<pointer<none>> to !kgen.pointer<struct<(array<1, pointer<none>>)>>
    %21 = kgen.struct.gep %20[0] : <struct<(array<1, pointer<none>>)>>
    %22 = pop.pointer.bitcast %21 : !kgen.pointer<array<1, pointer<none>>> to !kgen.pointer<pointer<none>>
    pop.store %18, %22 : !kgen.pointer<pointer<none>>
    %23 = pop.load %19 : !kgen.pointer<pointer<none>>
    %24 = pop.pointer.bitcast %23 : !kgen.pointer<none> to !kgen.pointer<scalar<ui64>>
    %25 = pop.pointer.bitcast %23 : !kgen.pointer<none> to !kgen.pointer<scalar<ui32>>
    %26 = pop.pointer.bitcast %23 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
    %27 = pop.offset %26[%index1] : !kgen.pointer<scalar<ui8>>
    %28 = pop.offset %26[%13] : !kgen.pointer<scalar<ui8>>
    %29 = pop.offset %28[%idx-8] : !kgen.pointer<scalar<ui8>>
    %30 = pop.pointer.bitcast %29 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui64>>
    %31 = pop.offset %28[%idx-4] : !kgen.pointer<scalar<ui8>>
    %32 = pop.pointer.bitcast %31 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui32>>
    %33 = pop.stack_allocation 1 x pointer<none>
    %34 = pop.pointer.bitcast %33 : !kgen.pointer<pointer<none>> to !kgen.pointer<struct<(array<1, pointer<none>>)>>
    %35 = kgen.struct.gep %34[0] : <struct<(array<1, pointer<none>>)>>
    %36 = pop.pointer.bitcast %35 : !kgen.pointer<array<1, pointer<none>>> to !kgen.pointer<pointer<none>>
    pop.store %8, %36 : !kgen.pointer<pointer<none>>
    %37 = pop.load %33 : !kgen.pointer<pointer<none>>
    %38 = pop.cast_from_builtin %13 : index to !pop.scalar<index>
    %39 = pop.cast %38 : !pop.scalar<index> to !pop.scalar<uindex>
    %40 = index.cmp sge(%13, %index8)
    %41 = index.cmp slt(%13, %index5)
    %42 = index.cmp sle(%13, %index2)
    %43 = index.sub %13, %index2
    %44 = pop.offset %26[%43] : !kgen.pointer<scalar<ui8>>
    %45 = index.cmp sle(%13, %index16)
    %46 = index.sub %13, %index1
    %47 = pop.offset %26[%46] : !kgen.pointer<scalar<ui8>>
    %48 = pop.pointer.bitcast %37 : !kgen.pointer<none> to !kgen.pointer<scalar<ui64>>
    %49 = pop.pointer.bitcast %37 : !kgen.pointer<none> to !kgen.pointer<scalar<ui32>>
    %50 = pop.pointer.bitcast %37 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
    %51 = pop.offset %50[%43] : !kgen.pointer<scalar<ui8>>
    %52 = pop.offset %50[%46] : !kgen.pointer<scalar<ui8>>
    %53 = pop.offset %50[%index1] : !kgen.pointer<scalar<ui8>>
    %54 = pop.offset %50[%13] : !kgen.pointer<scalar<ui8>>
    %55 = pop.offset %54[%idx-8] : !kgen.pointer<scalar<ui8>>
    %56 = pop.pointer.bitcast %55 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui64>>
    %57 = pop.offset %54[%idx-4] : !kgen.pointer<scalar<ui8>>
    %58 = pop.pointer.bitcast %57 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui32>>
    %59 = index.cmp eq(%13, %index0)
    hlcf.if %59 {
      hlcf.yield
    } else {
      hlcf.if %41 {
        %61 = pop.load %50 : !kgen.pointer<scalar<ui8>>
        pop.store %61, %26 : !kgen.pointer<scalar<ui8>>
        %62 = pop.load %52 : !kgen.pointer<scalar<ui8>>
        pop.store %62, %47 : !kgen.pointer<scalar<ui8>>
        hlcf.if %42 {
          hlcf.yield
        } else {
          %63 = pop.load %53 : !kgen.pointer<scalar<ui8>>
          pop.store %63, %27 : !kgen.pointer<scalar<ui8>>
          %64 = pop.load %51 : !kgen.pointer<scalar<ui8>>
          pop.store %64, %44 : !kgen.pointer<scalar<ui8>>
          hlcf.yield
        }
        hlcf.yield
      } else {
        hlcf.if %45 {
          hlcf.if %40 {
            %61 = pop.load volatile<0> invariant<0> nontemporal<0> %48 align<1> : !kgen.pointer<scalar<ui64>>
            pop.store volatile<0> nontemporal<0> %61, %24 align<1> : !kgen.pointer<scalar<ui64>>
            %62 = pop.load volatile<0> invariant<0> nontemporal<0> %56 align<1> : !kgen.pointer<scalar<ui64>>
            pop.store volatile<0> nontemporal<0> %62, %30 align<1> : !kgen.pointer<scalar<ui64>>
            hlcf.yield
          } else {
            %61 = pop.load volatile<0> invariant<0> nontemporal<0> %49 align<1> : !kgen.pointer<scalar<ui32>>
            pop.store volatile<0> nontemporal<0> %61, %25 align<1> : !kgen.pointer<scalar<ui32>>
            %62 = pop.load volatile<0> invariant<0> nontemporal<0> %58 align<1> : !kgen.pointer<scalar<ui32>>
            pop.store volatile<0> nontemporal<0> %62, %32 align<1> : !kgen.pointer<scalar<ui32>>
            hlcf.yield
          }
          hlcf.yield
        } else {
          %61 = pop.floordiv %39, %simd : !pop.scalar<uindex>
          %62 = pop.mul %61, %simd : !pop.scalar<uindex>
          %63 = pop.cast fast %62 : !pop.scalar<uindex> to !pop.scalar<index>
          %64 = pop.cast_to_builtin %63 : !pop.scalar<index> to index
          hlcf.loop "_loop_0" (%arg2 = %index0 : index) {
            %66 = index.add %arg2, %index32
            %67 = index.sub %64, %arg2
            %68 = index.cmp slt(%arg2, %64)
            %69 = pop.select %68, %67, %index0 : index
            %70 = index.cmp sle(%69, %index0)
            %71 = pop.select %70, %arg2, %66 : index
            %72:2 = lit.try "try0" -> index, index {
              hlcf.if %70 {
                lit.try.raise "try0" %71, %arg2 : index, index
              } else {
                hlcf.yield
              }
              lit.try.yield %71, %arg2 : index, index
            } except (%arg3: index, %arg4: index) {
              hlcf.break "_loop_0"
            } else (%arg3: index, %arg4: index) {
              lit.try.yield %arg3, %arg4 : index, index
            }
            %73 = pop.offset %50[%72#1] : !kgen.pointer<scalar<ui8>>
            %74 = pop.pointer.bitcast %73 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<simd<32, ui8>>
            %75 = pop.load volatile<0> invariant<0> nontemporal<0> %74 align<1> : !kgen.pointer<simd<32, ui8>>
            %76 = pop.offset %26[%72#1] : !kgen.pointer<scalar<ui8>>
            %77 = pop.pointer.bitcast %76 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<simd<32, ui8>>
            pop.store volatile<0> nontemporal<0> %75, %77 align<1> : !kgen.pointer<simd<32, ui8>>
            hlcf.continue "_loop_0" %72#0 : index
          }
          %65 = index.maxs %64, %13
          hlcf.loop "_loop_2" (%arg2 = %64 : index) {
            %66 = index.add %arg2, %index1
            %67 = index.cmp eq(%arg2, %65)
            %68 = pop.select %67, %arg2, %66 : index
            %69:2 = lit.try "try2" -> index, index {
              hlcf.if %67 {
                lit.try.raise "try2" %68, %arg2 : index, index
              } else {
                hlcf.yield
              }
              lit.try.yield %68, %arg2 : index, index
            } except (%arg3: index, %arg4: index) {
              hlcf.break "_loop_2"
            } else (%arg3: index, %arg4: index) {
              lit.try.yield %arg3, %arg4 : index, index
            }
            %70 = pop.offset %50[%69#1] : !kgen.pointer<scalar<ui8>>
            %71 = pop.load volatile<0> invariant<0> nontemporal<0> %70 align<1> : !kgen.pointer<scalar<ui8>>
            %72 = pop.offset %26[%69#1] : !kgen.pointer<scalar<ui8>>
            pop.store volatile<0> nontemporal<0> %71, %72 align<1> : !kgen.pointer<scalar<ui8>>
            hlcf.continue "_loop_2" %69#0 : index
          }
          hlcf.yield
        }
        hlcf.yield
      }
      hlcf.yield
    }
    %60 = kgen.struct.create(%0, %14) : !kgen.struct<(pointer<none>, index) memoryOnly>
    kgen.return %60 : !kgen.struct<(pointer<none>, index) memoryOnly>
  }
  kgen.func @"std::collections::string::string_slice::StringSlice::write_to[::Writer](::StringSlice[$0, $1, $2],$3&),mut=0,writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%arg0: !kgen.struct<(pointer<none>, index)>, %arg1: !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly> {
    %none = kgen.param.constant: none = <#kgen.none>
    %idx-4 = index.constant -4
    %idx-8 = index.constant -8
    %simd = kgen.param.constant: scalar<uindex> = <32>
    %index32 = kgen.param.constant = <32>
    %index8 = kgen.param.constant = <8>
    %index5 = kgen.param.constant = <5>
    %index2 = kgen.param.constant = <2>
    %index16 = kgen.param.constant = <16>
    %index0 = kgen.param.constant = <0>
    %index2048 = kgen.param.constant = <2048>
    %index1 = kgen.param.constant = <1>
    %0 = kgen.struct.extract %arg1[0] : <(pointer<none>, index) memoryOnly>
    %1 = kgen.struct.extract %arg1[1] : <(pointer<none>, index) memoryOnly>
    %2 = kgen.struct.extract %arg0[1] : <(pointer<none>, index)>
    %3 = index.add %2, %1
    %4 = index.cmp sgt(%3, %index2048)
    hlcf.if %4 {
      kgen.call @"std::io::io::_printf[KGENParamList[::AnyType],::StringSlice[::Bool(False), StaticConstantOrigin, *?],*::AnyType,LITImmutOrigin,::Origin[::Bool(False), $3]](*$0,file:::FileDescriptor),types.values`=[],fmt={ #interp.memref<{[(#interp.memory_handle<16, \22HEAP_BUFFER_BYTES exceeded, increase with: `mojo -D HEAP_BUFFER_BYTES=4096`\\0A\\00\22 string>, const_global, [], [])], []}, 0, 0>, 76 }"(%index1) : (index) -> ()
      llvm.intr.trap
      hlcf.loop "_loop_0" {
        hlcf.continue "_loop_0"
      }
      kgen.unreachable
    } else {
      hlcf.yield
    }
    %5 = pop.pointer.bitcast %0 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
    %6 = pop.offset %5[%1] : !kgen.pointer<scalar<ui8>>
    %7 = pop.pointer.bitcast %6 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<none>
    %8 = kgen.struct.extract %arg0[0] : <(pointer<none>, index)>
    %9 = pop.stack_allocation 1 x pointer<none>
    %10 = pop.pointer.bitcast %9 : !kgen.pointer<pointer<none>> to !kgen.pointer<struct<(array<1, pointer<none>>)>>
    %11 = kgen.struct.gep %10[0] : <struct<(array<1, pointer<none>>)>>
    %12 = pop.pointer.bitcast %11 : !kgen.pointer<array<1, pointer<none>>> to !kgen.pointer<pointer<none>>
    pop.store %7, %12 : !kgen.pointer<pointer<none>>
    %13 = pop.load %9 : !kgen.pointer<pointer<none>>
    %14 = pop.pointer.bitcast %13 : !kgen.pointer<none> to !kgen.pointer<scalar<ui64>>
    %15 = pop.pointer.bitcast %13 : !kgen.pointer<none> to !kgen.pointer<scalar<ui32>>
    %16 = pop.pointer.bitcast %13 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
    %17 = pop.offset %16[%index1] : !kgen.pointer<scalar<ui8>>
    %18 = pop.offset %16[%2] : !kgen.pointer<scalar<ui8>>
    %19 = pop.offset %18[%idx-8] : !kgen.pointer<scalar<ui8>>
    %20 = pop.pointer.bitcast %19 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui64>>
    %21 = pop.offset %18[%idx-4] : !kgen.pointer<scalar<ui8>>
    %22 = pop.pointer.bitcast %21 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui32>>
    %23 = pop.stack_allocation 1 x pointer<none>
    %24 = pop.pointer.bitcast %23 : !kgen.pointer<pointer<none>> to !kgen.pointer<struct<(array<1, pointer<none>>)>>
    %25 = kgen.struct.gep %24[0] : <struct<(array<1, pointer<none>>)>>
    %26 = pop.pointer.bitcast %25 : !kgen.pointer<array<1, pointer<none>>> to !kgen.pointer<pointer<none>>
    pop.store %8, %26 : !kgen.pointer<pointer<none>>
    %27 = pop.load %23 : !kgen.pointer<pointer<none>>
    %28 = pop.cast_from_builtin %2 : index to !pop.scalar<index>
    %29 = pop.cast %28 : !pop.scalar<index> to !pop.scalar<uindex>
    %30 = index.cmp sge(%2, %index8)
    %31 = index.cmp slt(%2, %index5)
    %32 = index.cmp sle(%2, %index2)
    %33 = index.sub %2, %index2
    %34 = pop.offset %16[%33] : !kgen.pointer<scalar<ui8>>
    %35 = index.cmp sle(%2, %index16)
    %36 = index.sub %2, %index1
    %37 = pop.offset %16[%36] : !kgen.pointer<scalar<ui8>>
    %38 = pop.pointer.bitcast %27 : !kgen.pointer<none> to !kgen.pointer<scalar<ui64>>
    %39 = pop.pointer.bitcast %27 : !kgen.pointer<none> to !kgen.pointer<scalar<ui32>>
    %40 = pop.pointer.bitcast %27 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
    %41 = pop.offset %40[%33] : !kgen.pointer<scalar<ui8>>
    %42 = pop.offset %40[%36] : !kgen.pointer<scalar<ui8>>
    %43 = pop.offset %40[%index1] : !kgen.pointer<scalar<ui8>>
    %44 = pop.offset %40[%2] : !kgen.pointer<scalar<ui8>>
    %45 = pop.offset %44[%idx-8] : !kgen.pointer<scalar<ui8>>
    %46 = pop.pointer.bitcast %45 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui64>>
    %47 = pop.offset %44[%idx-4] : !kgen.pointer<scalar<ui8>>
    %48 = pop.pointer.bitcast %47 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui32>>
    %49 = index.cmp eq(%2, %index0)
    hlcf.if %49 {
      hlcf.yield
    } else {
      hlcf.if %31 {
        %51 = pop.load %40 : !kgen.pointer<scalar<ui8>>
        pop.store %51, %16 : !kgen.pointer<scalar<ui8>>
        %52 = pop.load %42 : !kgen.pointer<scalar<ui8>>
        pop.store %52, %37 : !kgen.pointer<scalar<ui8>>
        hlcf.if %32 {
          hlcf.yield
        } else {
          %53 = pop.load %43 : !kgen.pointer<scalar<ui8>>
          pop.store %53, %17 : !kgen.pointer<scalar<ui8>>
          %54 = pop.load %41 : !kgen.pointer<scalar<ui8>>
          pop.store %54, %34 : !kgen.pointer<scalar<ui8>>
          hlcf.yield
        }
        hlcf.yield
      } else {
        hlcf.if %35 {
          hlcf.if %30 {
            %51 = pop.load volatile<0> invariant<0> nontemporal<0> %38 align<1> : !kgen.pointer<scalar<ui64>>
            pop.store volatile<0> nontemporal<0> %51, %14 align<1> : !kgen.pointer<scalar<ui64>>
            %52 = pop.load volatile<0> invariant<0> nontemporal<0> %46 align<1> : !kgen.pointer<scalar<ui64>>
            pop.store volatile<0> nontemporal<0> %52, %20 align<1> : !kgen.pointer<scalar<ui64>>
            hlcf.yield
          } else {
            %51 = pop.load volatile<0> invariant<0> nontemporal<0> %39 align<1> : !kgen.pointer<scalar<ui32>>
            pop.store volatile<0> nontemporal<0> %51, %15 align<1> : !kgen.pointer<scalar<ui32>>
            %52 = pop.load volatile<0> invariant<0> nontemporal<0> %48 align<1> : !kgen.pointer<scalar<ui32>>
            pop.store volatile<0> nontemporal<0> %52, %22 align<1> : !kgen.pointer<scalar<ui32>>
            hlcf.yield
          }
          hlcf.yield
        } else {
          %51 = pop.floordiv %29, %simd : !pop.scalar<uindex>
          %52 = pop.mul %51, %simd : !pop.scalar<uindex>
          %53 = pop.cast fast %52 : !pop.scalar<uindex> to !pop.scalar<index>
          %54 = pop.cast_to_builtin %53 : !pop.scalar<index> to index
          hlcf.loop "_loop_0" (%arg2 = %index0 : index) {
            %56 = index.add %arg2, %index32
            %57 = index.sub %54, %arg2
            %58 = index.cmp slt(%arg2, %54)
            %59 = pop.select %58, %57, %index0 : index
            %60 = index.cmp sle(%59, %index0)
            %61 = pop.select %60, %arg2, %56 : index
            %62:2 = lit.try "try0" -> index, index {
              hlcf.if %60 {
                lit.try.raise "try0" %61, %arg2 : index, index
              } else {
                hlcf.yield
              }
              lit.try.yield %61, %arg2 : index, index
            } except (%arg3: index, %arg4: index) {
              hlcf.break "_loop_0"
            } else (%arg3: index, %arg4: index) {
              lit.try.yield %arg3, %arg4 : index, index
            }
            %63 = pop.offset %40[%62#1] : !kgen.pointer<scalar<ui8>>
            %64 = pop.pointer.bitcast %63 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<simd<32, ui8>>
            %65 = pop.load volatile<0> invariant<0> nontemporal<0> %64 align<1> : !kgen.pointer<simd<32, ui8>>
            %66 = pop.offset %16[%62#1] : !kgen.pointer<scalar<ui8>>
            %67 = pop.pointer.bitcast %66 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<simd<32, ui8>>
            pop.store volatile<0> nontemporal<0> %65, %67 align<1> : !kgen.pointer<simd<32, ui8>>
            hlcf.continue "_loop_0" %62#0 : index
          }
          %55 = index.maxs %54, %2
          hlcf.loop "_loop_2" (%arg2 = %54 : index) {
            %56 = index.add %arg2, %index1
            %57 = index.cmp eq(%arg2, %55)
            %58 = pop.select %57, %arg2, %56 : index
            %59:2 = lit.try "try2" -> index, index {
              hlcf.if %57 {
                lit.try.raise "try2" %58, %arg2 : index, index
              } else {
                hlcf.yield
              }
              lit.try.yield %58, %arg2 : index, index
            } except (%arg3: index, %arg4: index) {
              hlcf.break "_loop_2"
            } else (%arg3: index, %arg4: index) {
              lit.try.yield %arg3, %arg4 : index, index
            }
            %60 = pop.offset %40[%59#1] : !kgen.pointer<scalar<ui8>>
            %61 = pop.load volatile<0> invariant<0> nontemporal<0> %60 align<1> : !kgen.pointer<scalar<ui8>>
            %62 = pop.offset %16[%59#1] : !kgen.pointer<scalar<ui8>>
            pop.store volatile<0> nontemporal<0> %61, %62 align<1> : !kgen.pointer<scalar<ui8>>
            hlcf.continue "_loop_2" %59#0 : index
          }
          hlcf.yield
        }
        hlcf.yield
      }
      hlcf.yield
    }
    %50 = kgen.struct.create(%0, %3) : !kgen.struct<(pointer<none>, index) memoryOnly>
    kgen.return %50 : !kgen.struct<(pointer<none>, index) memoryOnly>
  }
  kgen.func @"std::collections::string::string_slice::StringSlice::write_to[::Writer](::StringSlice[$0, $1, $2],$3&),mut=1,writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%arg0: !kgen.struct<(pointer<none>, index)>, %arg1: !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly> {
    %none = kgen.param.constant: none = <#kgen.none>
    %idx-4 = index.constant -4
    %idx-8 = index.constant -8
    %simd = kgen.param.constant: scalar<uindex> = <32>
    %index32 = kgen.param.constant = <32>
    %index8 = kgen.param.constant = <8>
    %index5 = kgen.param.constant = <5>
    %index2 = kgen.param.constant = <2>
    %index16 = kgen.param.constant = <16>
    %index0 = kgen.param.constant = <0>
    %index2048 = kgen.param.constant = <2048>
    %index1 = kgen.param.constant = <1>
    %0 = kgen.struct.extract %arg1[0] : <(pointer<none>, index) memoryOnly>
    %1 = kgen.struct.extract %arg1[1] : <(pointer<none>, index) memoryOnly>
    %2 = kgen.struct.extract %arg0[1] : <(pointer<none>, index)>
    %3 = index.add %2, %1
    %4 = index.cmp sgt(%3, %index2048)
    hlcf.if %4 {
      kgen.call @"std::io::io::_printf[KGENParamList[::AnyType],::StringSlice[::Bool(False), StaticConstantOrigin, *?],*::AnyType,LITImmutOrigin,::Origin[::Bool(False), $3]](*$0,file:::FileDescriptor),types.values`=[],fmt={ #interp.memref<{[(#interp.memory_handle<16, \22HEAP_BUFFER_BYTES exceeded, increase with: `mojo -D HEAP_BUFFER_BYTES=4096`\\0A\\00\22 string>, const_global, [], [])], []}, 0, 0>, 76 }"(%index1) : (index) -> ()
      llvm.intr.trap
      hlcf.loop "_loop_0" {
        hlcf.continue "_loop_0"
      }
      kgen.unreachable
    } else {
      hlcf.yield
    }
    %5 = pop.pointer.bitcast %0 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
    %6 = pop.offset %5[%1] : !kgen.pointer<scalar<ui8>>
    %7 = pop.pointer.bitcast %6 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<none>
    %8 = kgen.struct.extract %arg0[0] : <(pointer<none>, index)>
    %9 = pop.stack_allocation 1 x pointer<none>
    %10 = pop.pointer.bitcast %9 : !kgen.pointer<pointer<none>> to !kgen.pointer<struct<(array<1, pointer<none>>)>>
    %11 = kgen.struct.gep %10[0] : <struct<(array<1, pointer<none>>)>>
    %12 = pop.pointer.bitcast %11 : !kgen.pointer<array<1, pointer<none>>> to !kgen.pointer<pointer<none>>
    pop.store %7, %12 : !kgen.pointer<pointer<none>>
    %13 = pop.load %9 : !kgen.pointer<pointer<none>>
    %14 = pop.pointer.bitcast %13 : !kgen.pointer<none> to !kgen.pointer<scalar<ui64>>
    %15 = pop.pointer.bitcast %13 : !kgen.pointer<none> to !kgen.pointer<scalar<ui32>>
    %16 = pop.pointer.bitcast %13 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
    %17 = pop.offset %16[%index1] : !kgen.pointer<scalar<ui8>>
    %18 = pop.offset %16[%2] : !kgen.pointer<scalar<ui8>>
    %19 = pop.offset %18[%idx-8] : !kgen.pointer<scalar<ui8>>
    %20 = pop.pointer.bitcast %19 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui64>>
    %21 = pop.offset %18[%idx-4] : !kgen.pointer<scalar<ui8>>
    %22 = pop.pointer.bitcast %21 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui32>>
    %23 = pop.stack_allocation 1 x pointer<none>
    %24 = pop.pointer.bitcast %23 : !kgen.pointer<pointer<none>> to !kgen.pointer<struct<(array<1, pointer<none>>)>>
    %25 = kgen.struct.gep %24[0] : <struct<(array<1, pointer<none>>)>>
    %26 = pop.pointer.bitcast %25 : !kgen.pointer<array<1, pointer<none>>> to !kgen.pointer<pointer<none>>
    pop.store %8, %26 : !kgen.pointer<pointer<none>>
    %27 = pop.load %23 : !kgen.pointer<pointer<none>>
    %28 = pop.cast_from_builtin %2 : index to !pop.scalar<index>
    %29 = pop.cast %28 : !pop.scalar<index> to !pop.scalar<uindex>
    %30 = index.cmp sge(%2, %index8)
    %31 = index.cmp slt(%2, %index5)
    %32 = index.cmp sle(%2, %index2)
    %33 = index.sub %2, %index2
    %34 = pop.offset %16[%33] : !kgen.pointer<scalar<ui8>>
    %35 = index.cmp sle(%2, %index16)
    %36 = index.sub %2, %index1
    %37 = pop.offset %16[%36] : !kgen.pointer<scalar<ui8>>
    %38 = pop.pointer.bitcast %27 : !kgen.pointer<none> to !kgen.pointer<scalar<ui64>>
    %39 = pop.pointer.bitcast %27 : !kgen.pointer<none> to !kgen.pointer<scalar<ui32>>
    %40 = pop.pointer.bitcast %27 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
    %41 = pop.offset %40[%33] : !kgen.pointer<scalar<ui8>>
    %42 = pop.offset %40[%36] : !kgen.pointer<scalar<ui8>>
    %43 = pop.offset %40[%index1] : !kgen.pointer<scalar<ui8>>
    %44 = pop.offset %40[%2] : !kgen.pointer<scalar<ui8>>
    %45 = pop.offset %44[%idx-8] : !kgen.pointer<scalar<ui8>>
    %46 = pop.pointer.bitcast %45 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui64>>
    %47 = pop.offset %44[%idx-4] : !kgen.pointer<scalar<ui8>>
    %48 = pop.pointer.bitcast %47 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui32>>
    %49 = index.cmp eq(%2, %index0)
    hlcf.if %49 {
      hlcf.yield
    } else {
      hlcf.if %31 {
        %51 = pop.load %40 : !kgen.pointer<scalar<ui8>>
        pop.store %51, %16 : !kgen.pointer<scalar<ui8>>
        %52 = pop.load %42 : !kgen.pointer<scalar<ui8>>
        pop.store %52, %37 : !kgen.pointer<scalar<ui8>>
        hlcf.if %32 {
          hlcf.yield
        } else {
          %53 = pop.load %43 : !kgen.pointer<scalar<ui8>>
          pop.store %53, %17 : !kgen.pointer<scalar<ui8>>
          %54 = pop.load %41 : !kgen.pointer<scalar<ui8>>
          pop.store %54, %34 : !kgen.pointer<scalar<ui8>>
          hlcf.yield
        }
        hlcf.yield
      } else {
        hlcf.if %35 {
          hlcf.if %30 {
            %51 = pop.load volatile<0> invariant<0> nontemporal<0> %38 align<1> : !kgen.pointer<scalar<ui64>>
            pop.store volatile<0> nontemporal<0> %51, %14 align<1> : !kgen.pointer<scalar<ui64>>
            %52 = pop.load volatile<0> invariant<0> nontemporal<0> %46 align<1> : !kgen.pointer<scalar<ui64>>
            pop.store volatile<0> nontemporal<0> %52, %20 align<1> : !kgen.pointer<scalar<ui64>>
            hlcf.yield
          } else {
            %51 = pop.load volatile<0> invariant<0> nontemporal<0> %39 align<1> : !kgen.pointer<scalar<ui32>>
            pop.store volatile<0> nontemporal<0> %51, %15 align<1> : !kgen.pointer<scalar<ui32>>
            %52 = pop.load volatile<0> invariant<0> nontemporal<0> %48 align<1> : !kgen.pointer<scalar<ui32>>
            pop.store volatile<0> nontemporal<0> %52, %22 align<1> : !kgen.pointer<scalar<ui32>>
            hlcf.yield
          }
          hlcf.yield
        } else {
          %51 = pop.floordiv %29, %simd : !pop.scalar<uindex>
          %52 = pop.mul %51, %simd : !pop.scalar<uindex>
          %53 = pop.cast fast %52 : !pop.scalar<uindex> to !pop.scalar<index>
          %54 = pop.cast_to_builtin %53 : !pop.scalar<index> to index
          hlcf.loop "_loop_0" (%arg2 = %index0 : index) {
            %56 = index.add %arg2, %index32
            %57 = index.sub %54, %arg2
            %58 = index.cmp slt(%arg2, %54)
            %59 = pop.select %58, %57, %index0 : index
            %60 = index.cmp sle(%59, %index0)
            %61 = pop.select %60, %arg2, %56 : index
            %62:2 = lit.try "try0" -> index, index {
              hlcf.if %60 {
                lit.try.raise "try0" %61, %arg2 : index, index
              } else {
                hlcf.yield
              }
              lit.try.yield %61, %arg2 : index, index
            } except (%arg3: index, %arg4: index) {
              hlcf.break "_loop_0"
            } else (%arg3: index, %arg4: index) {
              lit.try.yield %arg3, %arg4 : index, index
            }
            %63 = pop.offset %40[%62#1] : !kgen.pointer<scalar<ui8>>
            %64 = pop.pointer.bitcast %63 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<simd<32, ui8>>
            %65 = pop.load volatile<0> invariant<0> nontemporal<0> %64 align<1> : !kgen.pointer<simd<32, ui8>>
            %66 = pop.offset %16[%62#1] : !kgen.pointer<scalar<ui8>>
            %67 = pop.pointer.bitcast %66 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<simd<32, ui8>>
            pop.store volatile<0> nontemporal<0> %65, %67 align<1> : !kgen.pointer<simd<32, ui8>>
            hlcf.continue "_loop_0" %62#0 : index
          }
          %55 = index.maxs %54, %2
          hlcf.loop "_loop_2" (%arg2 = %54 : index) {
            %56 = index.add %arg2, %index1
            %57 = index.cmp eq(%arg2, %55)
            %58 = pop.select %57, %arg2, %56 : index
            %59:2 = lit.try "try2" -> index, index {
              hlcf.if %57 {
                lit.try.raise "try2" %58, %arg2 : index, index
              } else {
                hlcf.yield
              }
              lit.try.yield %58, %arg2 : index, index
            } except (%arg3: index, %arg4: index) {
              hlcf.break "_loop_2"
            } else (%arg3: index, %arg4: index) {
              lit.try.yield %arg3, %arg4 : index, index
            }
            %60 = pop.offset %40[%59#1] : !kgen.pointer<scalar<ui8>>
            %61 = pop.load volatile<0> invariant<0> nontemporal<0> %60 align<1> : !kgen.pointer<scalar<ui8>>
            %62 = pop.offset %16[%59#1] : !kgen.pointer<scalar<ui8>>
            pop.store volatile<0> nontemporal<0> %61, %62 align<1> : !kgen.pointer<scalar<ui8>>
            hlcf.continue "_loop_2" %59#0 : index
          }
          hlcf.yield
        }
        hlcf.yield
      }
      hlcf.yield
    }
    %50 = kgen.struct.create(%0, %3) : !kgen.struct<(pointer<none>, index) memoryOnly>
    kgen.return %50 : !kgen.struct<(pointer<none>, index) memoryOnly>
  }
  kgen.func @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg0: !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, %arg1: !kgen.struct<(pointer<none>, index)>) {
    %index4096 = kgen.param.constant = <4096>
    %idx-4 = index.constant -4
    %idx-8 = index.constant -8
    %simd = kgen.param.constant: scalar<uindex> = <32>
    %index32 = kgen.param.constant = <32>
    %index8 = kgen.param.constant = <8>
    %index5 = kgen.param.constant = <5>
    %index2 = kgen.param.constant = <2>
    %index16 = kgen.param.constant = <16>
    %index1 = kgen.param.constant = <1>
    %index0 = kgen.param.constant = <0>
    %0 = kgen.struct.gep %arg0[0] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>>
    %1 = kgen.struct.gep %0[0] : <struct<(array<4096, scalar<ui8>>) memoryOnly>>
    %2 = pop.pointer.bitcast %1 : !kgen.pointer<array<4096, scalar<ui8>>> to !kgen.pointer<none>
    %3 = kgen.struct.gep %arg0[2] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>>
    %4 = kgen.struct.extract %arg1[0] : <(pointer<none>, index)>
    %5 = pop.pointer.bitcast %1 : !kgen.pointer<array<4096, scalar<ui8>>> to !kgen.pointer<scalar<ui8>>
    %6 = kgen.struct.gep %arg0[1] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>>
    %7 = kgen.struct.extract %arg1[1] : <(pointer<none>, index)>
    %8 = index.cmp eq(%7, %index0)
    %9 = index.sub %7, %index1
    %10 = index.cmp sle(%7, %index16)
    %11 = index.sub %7, %index2
    %12 = index.cmp sle(%7, %index2)
    %13 = index.cmp slt(%7, %index5)
    %14 = index.cmp sge(%7, %index8)
    %15 = pop.cast_from_builtin %7 : index to !pop.scalar<index>
    %16 = pop.cast %15 : !pop.scalar<index> to !pop.scalar<uindex>
    %17 = index.cmp sgt(%7, %index4096)
    hlcf.if %17 {
      %18 = pop.load %3 : !kgen.pointer<pointer<index>>
      %19 = pop.load %6 : !kgen.pointer<index>
      %20 = pop.load %18 : !kgen.pointer<index>
      %21 = pop.external_call @write(%20, %2, %19) : (index, !kgen.pointer<none>, index) -> index
      pop.store %index0, %6 : !kgen.pointer<index>
      %22 = pop.load %3 : !kgen.pointer<pointer<index>>
      %23 = pop.load %22 : !kgen.pointer<index>
      %24 = pop.external_call @write(%23, %4, %7) : (index, !kgen.pointer<none>, index) -> index
      hlcf.yield
    } else {
      %18 = pop.load %6 : !kgen.pointer<index>
      %19 = index.add %18, %7
      %20 = index.cmp sgt(%19, %index4096)
      hlcf.if %20 {
        %58 = pop.load %3 : !kgen.pointer<pointer<index>>
        %59 = pop.load %6 : !kgen.pointer<index>
        %60 = pop.load %58 : !kgen.pointer<index>
        %61 = pop.external_call @write(%60, %2, %59) : (index, !kgen.pointer<none>, index) -> index
        pop.store %index0, %6 : !kgen.pointer<index>
        hlcf.yield
      } else {
        hlcf.yield
      }
      %21 = pop.load %6 : !kgen.pointer<index>
      %22 = pop.offset %5[%21] : !kgen.pointer<scalar<ui8>>
      %23 = pop.pointer.bitcast %22 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<none>
      %24 = pop.stack_allocation 1 x pointer<none>
      %25 = pop.pointer.bitcast %24 : !kgen.pointer<pointer<none>> to !kgen.pointer<struct<(array<1, pointer<none>>)>>
      %26 = kgen.struct.gep %25[0] : <struct<(array<1, pointer<none>>)>>
      %27 = pop.pointer.bitcast %26 : !kgen.pointer<array<1, pointer<none>>> to !kgen.pointer<pointer<none>>
      pop.store %23, %27 : !kgen.pointer<pointer<none>>
      %28 = pop.load %24 : !kgen.pointer<pointer<none>>
      %29 = pop.pointer.bitcast %28 : !kgen.pointer<none> to !kgen.pointer<scalar<ui64>>
      %30 = pop.pointer.bitcast %28 : !kgen.pointer<none> to !kgen.pointer<scalar<ui32>>
      %31 = pop.pointer.bitcast %28 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %32 = pop.offset %31[%11] : !kgen.pointer<scalar<ui8>>
      %33 = pop.offset %31[%9] : !kgen.pointer<scalar<ui8>>
      %34 = pop.offset %31[%index1] : !kgen.pointer<scalar<ui8>>
      %35 = pop.offset %31[%7] : !kgen.pointer<scalar<ui8>>
      %36 = pop.offset %35[%idx-8] : !kgen.pointer<scalar<ui8>>
      %37 = pop.pointer.bitcast %36 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui64>>
      %38 = pop.offset %35[%idx-4] : !kgen.pointer<scalar<ui8>>
      %39 = pop.pointer.bitcast %38 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui32>>
      %40 = pop.stack_allocation 1 x pointer<none>
      %41 = pop.pointer.bitcast %40 : !kgen.pointer<pointer<none>> to !kgen.pointer<struct<(array<1, pointer<none>>)>>
      %42 = kgen.struct.gep %41[0] : <struct<(array<1, pointer<none>>)>>
      %43 = pop.pointer.bitcast %42 : !kgen.pointer<array<1, pointer<none>>> to !kgen.pointer<pointer<none>>
      pop.store %4, %43 : !kgen.pointer<pointer<none>>
      %44 = pop.load %40 : !kgen.pointer<pointer<none>>
      %45 = pop.pointer.bitcast %44 : !kgen.pointer<none> to !kgen.pointer<scalar<ui64>>
      %46 = pop.pointer.bitcast %44 : !kgen.pointer<none> to !kgen.pointer<scalar<ui32>>
      %47 = pop.pointer.bitcast %44 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %48 = pop.offset %47[%11] : !kgen.pointer<scalar<ui8>>
      %49 = pop.offset %47[%9] : !kgen.pointer<scalar<ui8>>
      %50 = pop.offset %47[%index1] : !kgen.pointer<scalar<ui8>>
      %51 = pop.offset %47[%7] : !kgen.pointer<scalar<ui8>>
      %52 = pop.offset %51[%idx-8] : !kgen.pointer<scalar<ui8>>
      %53 = pop.pointer.bitcast %52 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui64>>
      %54 = pop.offset %51[%idx-4] : !kgen.pointer<scalar<ui8>>
      %55 = pop.pointer.bitcast %54 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui32>>
      hlcf.if %8 {
        hlcf.yield
      } else {
        hlcf.if %13 {
          %58 = pop.load %47 : !kgen.pointer<scalar<ui8>>
          pop.store %58, %31 : !kgen.pointer<scalar<ui8>>
          %59 = pop.load %49 : !kgen.pointer<scalar<ui8>>
          pop.store %59, %33 : !kgen.pointer<scalar<ui8>>
          hlcf.if %12 {
            hlcf.yield
          } else {
            %60 = pop.load %50 : !kgen.pointer<scalar<ui8>>
            pop.store %60, %34 : !kgen.pointer<scalar<ui8>>
            %61 = pop.load %48 : !kgen.pointer<scalar<ui8>>
            pop.store %61, %32 : !kgen.pointer<scalar<ui8>>
            hlcf.yield
          }
          hlcf.yield
        } else {
          hlcf.if %10 {
            hlcf.if %14 {
              %58 = pop.load volatile<0> invariant<0> nontemporal<0> %45 align<1> : !kgen.pointer<scalar<ui64>>
              pop.store volatile<0> nontemporal<0> %58, %29 align<1> : !kgen.pointer<scalar<ui64>>
              %59 = pop.load volatile<0> invariant<0> nontemporal<0> %53 align<1> : !kgen.pointer<scalar<ui64>>
              pop.store volatile<0> nontemporal<0> %59, %37 align<1> : !kgen.pointer<scalar<ui64>>
              hlcf.yield
            } else {
              %58 = pop.load volatile<0> invariant<0> nontemporal<0> %46 align<1> : !kgen.pointer<scalar<ui32>>
              pop.store volatile<0> nontemporal<0> %58, %30 align<1> : !kgen.pointer<scalar<ui32>>
              %59 = pop.load volatile<0> invariant<0> nontemporal<0> %55 align<1> : !kgen.pointer<scalar<ui32>>
              pop.store volatile<0> nontemporal<0> %59, %39 align<1> : !kgen.pointer<scalar<ui32>>
              hlcf.yield
            }
            hlcf.yield
          } else {
            %58 = pop.floordiv %16, %simd : !pop.scalar<uindex>
            %59 = pop.mul %58, %simd : !pop.scalar<uindex>
            %60 = pop.cast fast %59 : !pop.scalar<uindex> to !pop.scalar<index>
            %61 = pop.cast_to_builtin %60 : !pop.scalar<index> to index
            hlcf.loop "_loop_0" (%arg2 = %index0 : index) {
              %63 = index.add %arg2, %index32
              %64 = index.sub %61, %arg2
              %65 = index.cmp slt(%arg2, %61)
              %66 = pop.select %65, %64, %index0 : index
              %67 = index.cmp sle(%66, %index0)
              %68 = pop.select %67, %arg2, %63 : index
              %69:2 = lit.try "try0" -> index, index {
                hlcf.if %67 {
                  lit.try.raise "try0" %68, %arg2 : index, index
                } else {
                  hlcf.yield
                }
                lit.try.yield %68, %arg2 : index, index
              } except (%arg3: index, %arg4: index) {
                hlcf.break "_loop_0"
              } else (%arg3: index, %arg4: index) {
                lit.try.yield %arg3, %arg4 : index, index
              }
              %70 = pop.offset %47[%69#1] : !kgen.pointer<scalar<ui8>>
              %71 = pop.pointer.bitcast %70 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<simd<32, ui8>>
              %72 = pop.load volatile<0> invariant<0> nontemporal<0> %71 align<1> : !kgen.pointer<simd<32, ui8>>
              %73 = pop.offset %31[%69#1] : !kgen.pointer<scalar<ui8>>
              %74 = pop.pointer.bitcast %73 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<simd<32, ui8>>
              pop.store volatile<0> nontemporal<0> %72, %74 align<1> : !kgen.pointer<simd<32, ui8>>
              hlcf.continue "_loop_0" %69#0 : index
            }
            %62 = index.maxs %61, %7
            hlcf.loop "_loop_2" (%arg2 = %61 : index) {
              %63 = index.add %arg2, %index1
              %64 = index.cmp eq(%arg2, %62)
              %65 = pop.select %64, %arg2, %63 : index
              %66:2 = lit.try "try2" -> index, index {
                hlcf.if %64 {
                  lit.try.raise "try2" %65, %arg2 : index, index
                } else {
                  hlcf.yield
                }
                lit.try.yield %65, %arg2 : index, index
              } except (%arg3: index, %arg4: index) {
                hlcf.break "_loop_2"
              } else (%arg3: index, %arg4: index) {
                lit.try.yield %arg3, %arg4 : index, index
              }
              %67 = pop.offset %47[%66#1] : !kgen.pointer<scalar<ui8>>
              %68 = pop.load volatile<0> invariant<0> nontemporal<0> %67 align<1> : !kgen.pointer<scalar<ui8>>
              %69 = pop.offset %31[%66#1] : !kgen.pointer<scalar<ui8>>
              pop.store volatile<0> nontemporal<0> %68, %69 align<1> : !kgen.pointer<scalar<ui8>>
              hlcf.continue "_loop_2" %66#0 : index
            }
            hlcf.yield
          }
          hlcf.yield
        }
        hlcf.yield
      }
      %56 = pop.load %6 : !kgen.pointer<index>
      %57 = index.add %56, %7
      pop.store %57, %6 : !kgen.pointer<index>
      hlcf.yield
    }
    kgen.return
  }
  kgen.func @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=1"(%arg0: !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, %arg1: !kgen.struct<(pointer<none>, index)>) {
    %index4096 = kgen.param.constant = <4096>
    %idx-4 = index.constant -4
    %idx-8 = index.constant -8
    %simd = kgen.param.constant: scalar<uindex> = <32>
    %index32 = kgen.param.constant = <32>
    %index8 = kgen.param.constant = <8>
    %index5 = kgen.param.constant = <5>
    %index2 = kgen.param.constant = <2>
    %index16 = kgen.param.constant = <16>
    %index1 = kgen.param.constant = <1>
    %index0 = kgen.param.constant = <0>
    %0 = kgen.struct.gep %arg0[0] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>>
    %1 = kgen.struct.gep %0[0] : <struct<(array<4096, scalar<ui8>>) memoryOnly>>
    %2 = pop.pointer.bitcast %1 : !kgen.pointer<array<4096, scalar<ui8>>> to !kgen.pointer<none>
    %3 = kgen.struct.gep %arg0[2] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>>
    %4 = kgen.struct.extract %arg1[0] : <(pointer<none>, index)>
    %5 = pop.pointer.bitcast %1 : !kgen.pointer<array<4096, scalar<ui8>>> to !kgen.pointer<scalar<ui8>>
    %6 = kgen.struct.gep %arg0[1] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>>
    %7 = kgen.struct.extract %arg1[1] : <(pointer<none>, index)>
    %8 = index.cmp eq(%7, %index0)
    %9 = index.sub %7, %index1
    %10 = index.cmp sle(%7, %index16)
    %11 = index.sub %7, %index2
    %12 = index.cmp sle(%7, %index2)
    %13 = index.cmp slt(%7, %index5)
    %14 = index.cmp sge(%7, %index8)
    %15 = pop.cast_from_builtin %7 : index to !pop.scalar<index>
    %16 = pop.cast %15 : !pop.scalar<index> to !pop.scalar<uindex>
    %17 = index.cmp sgt(%7, %index4096)
    hlcf.if %17 {
      %18 = pop.load %3 : !kgen.pointer<pointer<index>>
      %19 = pop.load %6 : !kgen.pointer<index>
      %20 = pop.load %18 : !kgen.pointer<index>
      %21 = pop.external_call @write(%20, %2, %19) : (index, !kgen.pointer<none>, index) -> index
      pop.store %index0, %6 : !kgen.pointer<index>
      %22 = pop.load %3 : !kgen.pointer<pointer<index>>
      %23 = pop.load %22 : !kgen.pointer<index>
      %24 = pop.external_call @write(%23, %4, %7) : (index, !kgen.pointer<none>, index) -> index
      hlcf.yield
    } else {
      %18 = pop.load %6 : !kgen.pointer<index>
      %19 = index.add %18, %7
      %20 = index.cmp sgt(%19, %index4096)
      hlcf.if %20 {
        %58 = pop.load %3 : !kgen.pointer<pointer<index>>
        %59 = pop.load %6 : !kgen.pointer<index>
        %60 = pop.load %58 : !kgen.pointer<index>
        %61 = pop.external_call @write(%60, %2, %59) : (index, !kgen.pointer<none>, index) -> index
        pop.store %index0, %6 : !kgen.pointer<index>
        hlcf.yield
      } else {
        hlcf.yield
      }
      %21 = pop.load %6 : !kgen.pointer<index>
      %22 = pop.offset %5[%21] : !kgen.pointer<scalar<ui8>>
      %23 = pop.pointer.bitcast %22 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<none>
      %24 = pop.stack_allocation 1 x pointer<none>
      %25 = pop.pointer.bitcast %24 : !kgen.pointer<pointer<none>> to !kgen.pointer<struct<(array<1, pointer<none>>)>>
      %26 = kgen.struct.gep %25[0] : <struct<(array<1, pointer<none>>)>>
      %27 = pop.pointer.bitcast %26 : !kgen.pointer<array<1, pointer<none>>> to !kgen.pointer<pointer<none>>
      pop.store %23, %27 : !kgen.pointer<pointer<none>>
      %28 = pop.load %24 : !kgen.pointer<pointer<none>>
      %29 = pop.pointer.bitcast %28 : !kgen.pointer<none> to !kgen.pointer<scalar<ui64>>
      %30 = pop.pointer.bitcast %28 : !kgen.pointer<none> to !kgen.pointer<scalar<ui32>>
      %31 = pop.pointer.bitcast %28 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %32 = pop.offset %31[%11] : !kgen.pointer<scalar<ui8>>
      %33 = pop.offset %31[%9] : !kgen.pointer<scalar<ui8>>
      %34 = pop.offset %31[%index1] : !kgen.pointer<scalar<ui8>>
      %35 = pop.offset %31[%7] : !kgen.pointer<scalar<ui8>>
      %36 = pop.offset %35[%idx-8] : !kgen.pointer<scalar<ui8>>
      %37 = pop.pointer.bitcast %36 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui64>>
      %38 = pop.offset %35[%idx-4] : !kgen.pointer<scalar<ui8>>
      %39 = pop.pointer.bitcast %38 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui32>>
      %40 = pop.stack_allocation 1 x pointer<none>
      %41 = pop.pointer.bitcast %40 : !kgen.pointer<pointer<none>> to !kgen.pointer<struct<(array<1, pointer<none>>)>>
      %42 = kgen.struct.gep %41[0] : <struct<(array<1, pointer<none>>)>>
      %43 = pop.pointer.bitcast %42 : !kgen.pointer<array<1, pointer<none>>> to !kgen.pointer<pointer<none>>
      pop.store %4, %43 : !kgen.pointer<pointer<none>>
      %44 = pop.load %40 : !kgen.pointer<pointer<none>>
      %45 = pop.pointer.bitcast %44 : !kgen.pointer<none> to !kgen.pointer<scalar<ui64>>
      %46 = pop.pointer.bitcast %44 : !kgen.pointer<none> to !kgen.pointer<scalar<ui32>>
      %47 = pop.pointer.bitcast %44 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %48 = pop.offset %47[%11] : !kgen.pointer<scalar<ui8>>
      %49 = pop.offset %47[%9] : !kgen.pointer<scalar<ui8>>
      %50 = pop.offset %47[%index1] : !kgen.pointer<scalar<ui8>>
      %51 = pop.offset %47[%7] : !kgen.pointer<scalar<ui8>>
      %52 = pop.offset %51[%idx-8] : !kgen.pointer<scalar<ui8>>
      %53 = pop.pointer.bitcast %52 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui64>>
      %54 = pop.offset %51[%idx-4] : !kgen.pointer<scalar<ui8>>
      %55 = pop.pointer.bitcast %54 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui32>>
      hlcf.if %8 {
        hlcf.yield
      } else {
        hlcf.if %13 {
          %58 = pop.load %47 : !kgen.pointer<scalar<ui8>>
          pop.store %58, %31 : !kgen.pointer<scalar<ui8>>
          %59 = pop.load %49 : !kgen.pointer<scalar<ui8>>
          pop.store %59, %33 : !kgen.pointer<scalar<ui8>>
          hlcf.if %12 {
            hlcf.yield
          } else {
            %60 = pop.load %50 : !kgen.pointer<scalar<ui8>>
            pop.store %60, %34 : !kgen.pointer<scalar<ui8>>
            %61 = pop.load %48 : !kgen.pointer<scalar<ui8>>
            pop.store %61, %32 : !kgen.pointer<scalar<ui8>>
            hlcf.yield
          }
          hlcf.yield
        } else {
          hlcf.if %10 {
            hlcf.if %14 {
              %58 = pop.load volatile<0> invariant<0> nontemporal<0> %45 align<1> : !kgen.pointer<scalar<ui64>>
              pop.store volatile<0> nontemporal<0> %58, %29 align<1> : !kgen.pointer<scalar<ui64>>
              %59 = pop.load volatile<0> invariant<0> nontemporal<0> %53 align<1> : !kgen.pointer<scalar<ui64>>
              pop.store volatile<0> nontemporal<0> %59, %37 align<1> : !kgen.pointer<scalar<ui64>>
              hlcf.yield
            } else {
              %58 = pop.load volatile<0> invariant<0> nontemporal<0> %46 align<1> : !kgen.pointer<scalar<ui32>>
              pop.store volatile<0> nontemporal<0> %58, %30 align<1> : !kgen.pointer<scalar<ui32>>
              %59 = pop.load volatile<0> invariant<0> nontemporal<0> %55 align<1> : !kgen.pointer<scalar<ui32>>
              pop.store volatile<0> nontemporal<0> %59, %39 align<1> : !kgen.pointer<scalar<ui32>>
              hlcf.yield
            }
            hlcf.yield
          } else {
            %58 = pop.floordiv %16, %simd : !pop.scalar<uindex>
            %59 = pop.mul %58, %simd : !pop.scalar<uindex>
            %60 = pop.cast fast %59 : !pop.scalar<uindex> to !pop.scalar<index>
            %61 = pop.cast_to_builtin %60 : !pop.scalar<index> to index
            hlcf.loop "_loop_0" (%arg2 = %index0 : index) {
              %63 = index.add %arg2, %index32
              %64 = index.sub %61, %arg2
              %65 = index.cmp slt(%arg2, %61)
              %66 = pop.select %65, %64, %index0 : index
              %67 = index.cmp sle(%66, %index0)
              %68 = pop.select %67, %arg2, %63 : index
              %69:2 = lit.try "try0" -> index, index {
                hlcf.if %67 {
                  lit.try.raise "try0" %68, %arg2 : index, index
                } else {
                  hlcf.yield
                }
                lit.try.yield %68, %arg2 : index, index
              } except (%arg3: index, %arg4: index) {
                hlcf.break "_loop_0"
              } else (%arg3: index, %arg4: index) {
                lit.try.yield %arg3, %arg4 : index, index
              }
              %70 = pop.offset %47[%69#1] : !kgen.pointer<scalar<ui8>>
              %71 = pop.pointer.bitcast %70 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<simd<32, ui8>>
              %72 = pop.load volatile<0> invariant<0> nontemporal<0> %71 align<1> : !kgen.pointer<simd<32, ui8>>
              %73 = pop.offset %31[%69#1] : !kgen.pointer<scalar<ui8>>
              %74 = pop.pointer.bitcast %73 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<simd<32, ui8>>
              pop.store volatile<0> nontemporal<0> %72, %74 align<1> : !kgen.pointer<simd<32, ui8>>
              hlcf.continue "_loop_0" %69#0 : index
            }
            %62 = index.maxs %61, %7
            hlcf.loop "_loop_2" (%arg2 = %61 : index) {
              %63 = index.add %arg2, %index1
              %64 = index.cmp eq(%arg2, %62)
              %65 = pop.select %64, %arg2, %63 : index
              %66:2 = lit.try "try2" -> index, index {
                hlcf.if %64 {
                  lit.try.raise "try2" %65, %arg2 : index, index
                } else {
                  hlcf.yield
                }
                lit.try.yield %65, %arg2 : index, index
              } except (%arg3: index, %arg4: index) {
                hlcf.break "_loop_2"
              } else (%arg3: index, %arg4: index) {
                lit.try.yield %arg3, %arg4 : index, index
              }
              %67 = pop.offset %47[%66#1] : !kgen.pointer<scalar<ui8>>
              %68 = pop.load volatile<0> invariant<0> nontemporal<0> %67 align<1> : !kgen.pointer<scalar<ui8>>
              %69 = pop.offset %31[%66#1] : !kgen.pointer<scalar<ui8>>
              pop.store volatile<0> nontemporal<0> %68, %69 align<1> : !kgen.pointer<scalar<ui8>>
              hlcf.continue "_loop_2" %66#0 : index
            }
            hlcf.yield
          }
          hlcf.yield
        }
        hlcf.yield
      }
      %56 = pop.load %6 : !kgen.pointer<index>
      %57 = index.add %56, %7
      pop.store %57, %6 : !kgen.pointer<index>
      hlcf.yield
    }
    kgen.return
  }
  kgen.func @"std::reflection::location::SourceLocation::write_to[::Writer](::SourceLocation,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>]"(%arg0: !kgen.struct<(index, index, struct<(pointer<none>, index)>)>, %arg1: !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut) {
    %index2305843009213693952 = kgen.param.constant = <2305843009213693952>
    %index1 = kgen.param.constant = <1>
    %string = kgen.param.constant: string = <":">
    %simd = kgen.param.constant: scalar<index> = <1>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %index0 = kgen.param.constant = <0>
    %idx-8 = index.constant -8
    %0 = kgen.struct.extract %arg0[2] : <(index, index, struct<(pointer<none>, index)>)>
    %1 = kgen.struct.extract %arg0[0] : <(index, index, struct<(pointer<none>, index)>)>
    %2 = kgen.struct.extract %arg0[1] : <(index, index, struct<(pointer<none>, index)>)>
    %3 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
    pop.stack_alloc.lifetime.start(%3) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %4 = kgen.struct.gep %3[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index1, %4 : !kgen.pointer<index>
    %5 = kgen.struct.gep %3[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %6 = pop.string.address %string
    %7 = pop.pointer.bitcast %6 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    pop.store %7, %5 : !kgen.pointer<pointer<none>>
    %8 = kgen.struct.gep %3[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index2305843009213693952, %8 : !kgen.pointer<index>
    %9 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
    pop.stack_alloc.lifetime.start(%9) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %10 = kgen.struct.gep %9[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index1, %10 : !kgen.pointer<index>
    %11 = kgen.struct.gep %9[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %7, %11 : !kgen.pointer<pointer<none>>
    %12 = kgen.struct.gep %9[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index2305843009213693952, %12 : !kgen.pointer<index>
    kgen.call @"std::format::__init__::Writer::write[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $3]]($0&,*$1),_Self`=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>],Ts.values`2x=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::builtin::int::Int\22>>, index], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::builtin::int::Int\22>>, index]]"(%arg1, %0, %3, %1, %9, %2) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, index, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, index) -> ()
    %13 = pop.load %8 : !kgen.pointer<index>
    %14 = index.and %13, %index4611686018427387904
    %15 = index.cmp ne(%14, %index0)
    hlcf.if %15 {
      %19 = pop.load %5 : !kgen.pointer<pointer<none>>
      %20 = pop.pointer.bitcast %19 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %21 = pop.offset %20[%idx-8] : !kgen.pointer<scalar<ui8>>
      %22 = pop.pointer.bitcast %21 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
      %23 = kgen.struct.gep %22[0] : <struct<(scalar<index>) memoryOnly>>
      %24 = pop.atomic.rmw sub(%23, %simd) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
      %25 = pop.cmp eq(%24, %simd) : <1, index>
      %26 = pop.cast_to_builtin %25 : !pop.scalar<bool> to i1
      hlcf.if %26 {
        pop.fence syncscope("") acquire
        pop.aligned_free %21 : <scalar<ui8>>
        hlcf.yield
      } else {
        hlcf.yield
      }
      hlcf.yield
    } else {
      hlcf.yield
    }
    %16 = pop.load %12 : !kgen.pointer<index>
    %17 = index.and %16, %index4611686018427387904
    %18 = index.cmp ne(%17, %index0)
    hlcf.if %18 {
      %19 = pop.load %11 : !kgen.pointer<pointer<none>>
      %20 = pop.pointer.bitcast %19 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %21 = pop.offset %20[%idx-8] : !kgen.pointer<scalar<ui8>>
      %22 = pop.pointer.bitcast %21 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
      %23 = kgen.struct.gep %22[0] : <struct<(scalar<index>) memoryOnly>>
      %24 = pop.atomic.rmw sub(%23, %simd) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
      %25 = pop.cmp eq(%24, %simd) : <1, index>
      %26 = pop.cast_to_builtin %25 : !pop.scalar<bool> to i1
      hlcf.if %26 {
        pop.fence syncscope("") acquire
        pop.aligned_free %21 : <scalar<ui8>>
        hlcf.yield
      } else {
        hlcf.yield
      }
      hlcf.yield
    } else {
      hlcf.yield
    }
    kgen.return
  }
  kgen.func @"std::builtin::_format_float::_write_float[::Writer,::DType]($0&,::SIMD[$1, ::Int(1)]){#pop.cast_to_builtin<#pop.simd_xor<#pop.simd_cmp<eq, #pop.simd_and<#pop.cast_from_builtin<#pop.dtype_to_ui8<#lit.struct.extract<:!lit.struct<_std::_builtin::_dtype::_DType> *(0,1), \22_mlir_value\22>> : ui8> : !pop.scalar<ui8>, #pop<simd 64> : !pop.scalar<ui8>> : !pop.scalar<ui8>, #pop<simd 0> : !pop.scalar<ui8>> : !pop.scalar<bool>, #pop<simd true> : !pop.scalar<bool>> : !pop.scalar<bool>>},W=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>],dtype=f64"(%arg0: !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, %arg1: !pop.scalar<f64>) {
    %string = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/builtin/_format_float.mojo">
    %simd = kgen.param.constant: scalar<uindex> = <21>
    %idx20 = index.constant 20
    %index54 = kgen.param.constant = <54>
    %index277 = kgen.param.constant = <277>
    %index36 = kgen.param.constant = <36>
    %index281 = kgen.param.constant = <281>
    %index294 = kgen.param.constant = <294>
    %simd_0 = kgen.param.constant: scalar<uindex> = <10>
    %idx9 = index.constant 9
    %index298 = kgen.param.constant = <298>
    %index40 = kgen.param.constant = <40>
    %index305 = kgen.param.constant = <305>
    %index316 = kgen.param.constant = <316>
    %index19 = kgen.param.constant = <19>
    %index263 = kgen.param.constant = <263>
    %string_1 = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/collections/check_bounds.mojo">
    %index57 = kgen.param.constant = <57>
    %string_2 = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/builtin/debug_assert.mojo">
    %index27 = kgen.param.constant = <27>
    %index330 = kgen.param.constant = <330>
    %index53 = kgen.param.constant = <53>
    %string_3 = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/collections/optional.mojo">
    %index18 = kgen.param.constant = <18>
    %index610 = kgen.param.constant = <610>
    %idx-8 = index.constant -8
    %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle, const_global, [], [])], []}, 0, 0>, 1 }>
    %struct_4 = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle4, const_global, [], [])], []}, 0, 0>, 6 }>
    %string_5 = kgen.param.constant: string = <" ">
    %string_6 = kgen.param.constant: string = <": ">
    %string_7 = kgen.param.constant: string = <"">
    %string_8 = kgen.param.constant: string = <"`Optional.value()` called on empty `Optional`. Consider using `if optional:` to check whether the `Optional` is empty before calling `.value()`, or use `.or_else()` to provide a default value.">
    %index192 = kgen.param.constant = <192>
    %simd_9 = kgen.param.constant: scalar<ui8> = <0>
    %index2048 = kgen.param.constant = <2048>
    %string_10 = kgen.param.constant: string = <" is out of bounds, valid range is 0 to ">
    %index39 = kgen.param.constant = <39>
    %string_11 = kgen.param.constant: string = <"index ">
    %index6 = kgen.param.constant = <6>
    %simd_12 = kgen.param.constant: scalar<ui32> = <3>
    %index2233785415175766016 = kgen.param.constant = <2233785415175766016>
    %index56 = kgen.param.constant = <56>
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %simd_13 = kgen.param.constant: scalar<ui32> = <516>
    %idx63 = index.constant 63
    %simd_14 = kgen.param.constant: scalar<ui64> = <52>
    %simd_15 = kgen.param.constant: scalar<ui64> = <2047>
    %idx52 = index.constant 52
    %none = kgen.param.constant: none = <#kgen.none>
    %string_16 = kgen.param.constant: string = <".0">
    %string_17 = kgen.param.constant: string = <"0.">
    %string_18 = kgen.param.constant: string = <"0">
    %string_19 = kgen.param.constant: string = <"e+">
    %string_20 = kgen.param.constant: string = <"e-">
    %string_21 = kgen.param.constant: string = <".">
    %array = kgen.param.constant: array<1, struct<(index, index, struct<(pointer<none>, index)>)>> = <[{ -1, 0, { 0, 0 } }]>
    %string_22 = kgen.param.constant: string = <"0.0">
    %string_23 = kgen.param.constant: string = <"-">
    %string_24 = kgen.param.constant: string = <"nan">
    %string_25 = kgen.param.constant: string = <"inf">
    %index2305843009213693952 = kgen.param.constant = <2305843009213693952>
    %index4 = kgen.param.constant = <4>
    %string_26 = kgen.param.constant: string = <"-inf">
    %true = index.bool.constant true
    %false = index.bool.constant false
    %0 = kgen.param.constant: i1 = <1>
    %simd_27 = kgen.param.constant: scalar<ui64> = <0>
    %index-1 = kgen.param.constant = <-1>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %simd_28 = kgen.param.constant: scalar<index> = <1>
    %index0 = kgen.param.constant = <0>
    %simd_29 = kgen.param.constant: scalar<ui64> = <10>
    %index1 = kgen.param.constant = <1>
    %index3 = kgen.param.constant = <3>
    %index15 = kgen.param.constant = <15>
    %index10 = kgen.param.constant = <10>
    %1 = kgen.param.constant: i1 = <0>
    %index2 = kgen.param.constant = <2>
    %2 = pop.string.address %string
    %3 = pop.pointer.bitcast %2 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %4 = kgen.struct.create(%3, %index54) : !kgen.struct<(pointer<none>, index)>
    %5 = kgen.struct.create(%index316, %index36, %4) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %6 = kgen.struct.create(%index305, %index36, %4) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %7 = kgen.struct.create(%index298, %index40, %4) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %8 = kgen.struct.create(%index281, %index36, %4) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %9 = kgen.struct.create(%index277, %index36, %4) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %10 = kgen.struct.create(%index263, %index19, %4) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %11 = pop.string.address %string_1
    %12 = pop.pointer.bitcast %11 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %13 = kgen.struct.create(%12, %index57) : !kgen.struct<(pointer<none>, index)>
    %14 = pop.string.address %string_2
    %15 = pop.pointer.bitcast %14 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %16 = kgen.struct.create(%index294, %index27, %4) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %17 = kgen.struct.create(%15, %index53) : !kgen.struct<(pointer<none>, index)>
    %18 = kgen.struct.create(%index330, %index27, %17) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %19 = pop.string.address %string_3
    %20 = pop.pointer.bitcast %19 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %21 = kgen.struct.create(%20, %index53) : !kgen.struct<(pointer<none>, index)>
    %22 = kgen.struct.create(%index610, %index18, %21) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %23 = pop.string.address %string_5
    %24 = pop.pointer.bitcast %23 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %25 = pop.string.address %string_6
    %26 = pop.pointer.bitcast %25 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %27 = pop.string.address %string_7
    %28 = pop.pointer.bitcast %27 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %29 = pop.string.address %string_8
    %30 = pop.pointer.bitcast %29 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %31 = pop.string.address %string_10
    %32 = pop.pointer.bitcast %31 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %33 = pop.string.address %string_11
    %34 = pop.pointer.bitcast %33 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %35 = kgen.struct.create(%index57, %index6, %13) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %36 = kgen.struct.create(%arg1, %simd_12) : !kgen.struct<(scalar<f64>, scalar<ui32>)>
    %37 = pop.call_llvm_intrinsic side_effecting<0> "llvm.is.fpclass", (%36) : (!kgen.struct<(scalar<f64>, scalar<ui32>)>) -> !pop.scalar<bool>
    %38 = pop.cast_to_builtin %37 : !pop.scalar<bool> to i1
    %39 = pop.string.address %string_16
    %40 = pop.pointer.bitcast %39 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %41 = pop.string.address %string_17
    %42 = pop.pointer.bitcast %41 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %43 = pop.string.address %string_18
    %44 = pop.pointer.bitcast %43 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %45 = pop.string.address %string_19
    %46 = pop.pointer.bitcast %45 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %47 = pop.string.address %string_20
    %48 = pop.pointer.bitcast %47 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %49 = pop.string.address %string_21
    %50 = pop.pointer.bitcast %49 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %51 = pop.string.address %string_22
    %52 = pop.pointer.bitcast %51 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %53 = pop.string.address %string_23
    %54 = pop.pointer.bitcast %53 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %55 = pop.string.address %string_24
    %56 = pop.pointer.bitcast %55 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %57 = pop.string.address %string_25
    %58 = pop.pointer.bitcast %57 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %59 = pop.string.address %string_26
    %60 = pop.pointer.bitcast %59 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %61 = kgen.struct.create(%28, %index0) : !kgen.struct<(pointer<none>, index)>
    %62 = pop.bitcast %arg1 : !pop.scalar<f64> to !pop.scalar<ui64>
    %63 = index.shl %index1, %idx52
    %64 = index.sub %63, %index1
    %65 = pop.cast_from_builtin %64 : index to !pop.scalar<index>
    %66 = pop.cast %65 : !pop.scalar<index> to !pop.scalar<ui64>
    %67 = pop.simd.and %62, %66 : <1, ui64>
    %68 = pop.shr %62, %simd_14 : !pop.scalar<ui64>
    %69 = pop.simd.and %68, %simd_15 : <1, ui64>
    %70 = pop.cast fast %69 : !pop.scalar<ui64> to !pop.scalar<index>
    %71 = pop.cast_to_builtin %70 : !pop.scalar<index> to index
    %72 = pop.bitcast %arg1 : !pop.scalar<f64> to !pop.scalar<si64>
    %73 = pop.cast fast %72 : !pop.scalar<si64> to !pop.scalar<index>
    %74 = pop.cast_to_builtin %73 : !pop.scalar<index> to index
    %75 = index.shl %index1, %idx63
    %76 = index.and %74, %75
    %77 = index.cmp ne(%76, %index0)
    %78 = kgen.struct.create(%arg1, %simd_13) : !kgen.struct<(scalar<f64>, scalar<ui32>)>
    %79 = pop.call_llvm_intrinsic side_effecting<0> "llvm.is.fpclass", (%78) : (!kgen.struct<(scalar<f64>, scalar<ui32>)>) -> !pop.scalar<bool>
    %80 = pop.cast_to_builtin %79 : !pop.scalar<bool> to i1
    hlcf.if %80 {
      hlcf.if %77 {
        %81 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
        pop.stack_alloc.lifetime.start(%81) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
        %82 = kgen.struct.gep %81[1] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %index4, %82 : !kgen.pointer<index>
        %83 = kgen.struct.gep %81[0] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %60, %83 : !kgen.pointer<pointer<none>>
        %84 = kgen.struct.gep %81[2] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %index2305843009213693952, %84 : !kgen.pointer<index>
        %85 = pop.pointer.bitcast %81 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
        %86 = pop.load %84 : !kgen.pointer<index>
        %87 = index.and %86, %index-9223372036854775808
        %88 = index.cmp ne(%87, %index0)
        %89 = hlcf.if %88 -> !kgen.pointer<none> {
          hlcf.yield %85 : !kgen.pointer<none>
        } else {
          %98 = pop.load %83 : !kgen.pointer<pointer<none>>
          hlcf.yield %98 : !kgen.pointer<none>
        }
        %90 = pop.load %84 : !kgen.pointer<index>
        %91 = index.and %90, %index-9223372036854775808
        %92 = index.cmp ne(%91, %index0)
        %93 = hlcf.if %92 -> index {
          %98 = pop.load %84 : !kgen.pointer<index>
          %99 = index.and %98, %index2233785415175766016
          %100 = index.shrs %99, %index56
          hlcf.yield %100 : index
        } else {
          %98 = pop.load %82 : !kgen.pointer<index>
          hlcf.yield %98 : index
        }
        %94 = kgen.struct.create(%89, %93) : !kgen.struct<(pointer<none>, index)>
        kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg0, %94) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
        %95 = pop.load %84 : !kgen.pointer<index>
        %96 = index.and %95, %index4611686018427387904
        %97 = index.cmp ne(%96, %index0)
        hlcf.if %97 {
          %98 = pop.load %83 : !kgen.pointer<pointer<none>>
          %99 = pop.pointer.bitcast %98 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
          %100 = pop.offset %99[%idx-8] : !kgen.pointer<scalar<ui8>>
          %101 = pop.pointer.bitcast %100 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
          %102 = kgen.struct.gep %101[0] : <struct<(scalar<index>) memoryOnly>>
          %103 = pop.atomic.rmw sub(%102, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
          %104 = pop.cmp eq(%103, %simd_28) : <1, index>
          %105 = pop.cast_to_builtin %104 : !pop.scalar<bool> to i1
          hlcf.if %105 {
            pop.fence syncscope("") acquire
            pop.aligned_free %100 : <scalar<ui8>>
            hlcf.yield
          } else {
            hlcf.yield
          }
          hlcf.yield
        } else {
          hlcf.yield
        }
        pop.stack_alloc.lifetime.end(%81) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
        hlcf.yield
      } else {
        %81 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
        pop.stack_alloc.lifetime.start(%81) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
        %82 = kgen.struct.gep %81[1] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %index3, %82 : !kgen.pointer<index>
        %83 = kgen.struct.gep %81[0] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %58, %83 : !kgen.pointer<pointer<none>>
        %84 = kgen.struct.gep %81[2] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %index2305843009213693952, %84 : !kgen.pointer<index>
        %85 = pop.pointer.bitcast %81 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
        %86 = pop.load %84 : !kgen.pointer<index>
        %87 = index.and %86, %index-9223372036854775808
        %88 = index.cmp ne(%87, %index0)
        %89 = hlcf.if %88 -> !kgen.pointer<none> {
          hlcf.yield %85 : !kgen.pointer<none>
        } else {
          %98 = pop.load %83 : !kgen.pointer<pointer<none>>
          hlcf.yield %98 : !kgen.pointer<none>
        }
        %90 = pop.load %84 : !kgen.pointer<index>
        %91 = index.and %90, %index-9223372036854775808
        %92 = index.cmp ne(%91, %index0)
        %93 = hlcf.if %92 -> index {
          %98 = pop.load %84 : !kgen.pointer<index>
          %99 = index.and %98, %index2233785415175766016
          %100 = index.shrs %99, %index56
          hlcf.yield %100 : index
        } else {
          %98 = pop.load %82 : !kgen.pointer<index>
          hlcf.yield %98 : index
        }
        %94 = kgen.struct.create(%89, %93) : !kgen.struct<(pointer<none>, index)>
        kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg0, %94) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
        %95 = pop.load %84 : !kgen.pointer<index>
        %96 = index.and %95, %index4611686018427387904
        %97 = index.cmp ne(%96, %index0)
        hlcf.if %97 {
          %98 = pop.load %83 : !kgen.pointer<pointer<none>>
          %99 = pop.pointer.bitcast %98 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
          %100 = pop.offset %99[%idx-8] : !kgen.pointer<scalar<ui8>>
          %101 = pop.pointer.bitcast %100 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
          %102 = kgen.struct.gep %101[0] : <struct<(scalar<index>) memoryOnly>>
          %103 = pop.atomic.rmw sub(%102, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
          %104 = pop.cmp eq(%103, %simd_28) : <1, index>
          %105 = pop.cast_to_builtin %104 : !pop.scalar<bool> to i1
          hlcf.if %105 {
            pop.fence syncscope("") acquire
            pop.aligned_free %100 : <scalar<ui8>>
            hlcf.yield
          } else {
            hlcf.yield
          }
          hlcf.yield
        } else {
          hlcf.yield
        }
        pop.stack_alloc.lifetime.end(%81) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
        hlcf.yield
      }
      hlcf.yield
    } else {
      hlcf.if %38 {
        %81 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
        pop.stack_alloc.lifetime.start(%81) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
        %82 = kgen.struct.gep %81[1] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %index3, %82 : !kgen.pointer<index>
        %83 = kgen.struct.gep %81[0] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %56, %83 : !kgen.pointer<pointer<none>>
        %84 = kgen.struct.gep %81[2] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %index2305843009213693952, %84 : !kgen.pointer<index>
        %85 = pop.pointer.bitcast %81 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
        %86 = pop.load %84 : !kgen.pointer<index>
        %87 = index.and %86, %index-9223372036854775808
        %88 = index.cmp ne(%87, %index0)
        %89 = hlcf.if %88 -> !kgen.pointer<none> {
          hlcf.yield %85 : !kgen.pointer<none>
        } else {
          %98 = pop.load %83 : !kgen.pointer<pointer<none>>
          hlcf.yield %98 : !kgen.pointer<none>
        }
        %90 = pop.load %84 : !kgen.pointer<index>
        %91 = index.and %90, %index-9223372036854775808
        %92 = index.cmp ne(%91, %index0)
        %93 = hlcf.if %92 -> index {
          %98 = pop.load %84 : !kgen.pointer<index>
          %99 = index.and %98, %index2233785415175766016
          %100 = index.shrs %99, %index56
          hlcf.yield %100 : index
        } else {
          %98 = pop.load %82 : !kgen.pointer<index>
          hlcf.yield %98 : index
        }
        %94 = kgen.struct.create(%89, %93) : !kgen.struct<(pointer<none>, index)>
        kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg0, %94) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
        %95 = pop.load %84 : !kgen.pointer<index>
        %96 = index.and %95, %index4611686018427387904
        %97 = index.cmp ne(%96, %index0)
        hlcf.if %97 {
          %98 = pop.load %83 : !kgen.pointer<pointer<none>>
          %99 = pop.pointer.bitcast %98 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
          %100 = pop.offset %99[%idx-8] : !kgen.pointer<scalar<ui8>>
          %101 = pop.pointer.bitcast %100 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
          %102 = kgen.struct.gep %101[0] : <struct<(scalar<index>) memoryOnly>>
          %103 = pop.atomic.rmw sub(%102, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
          %104 = pop.cmp eq(%103, %simd_28) : <1, index>
          %105 = pop.cast_to_builtin %104 : !pop.scalar<bool> to i1
          hlcf.if %105 {
            pop.fence syncscope("") acquire
            pop.aligned_free %100 : <scalar<ui8>>
            hlcf.yield
          } else {
            hlcf.yield
          }
          hlcf.yield
        } else {
          hlcf.yield
        }
        pop.stack_alloc.lifetime.end(%81) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
        hlcf.yield
      } else {
        hlcf.if %77 {
          %85 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
          pop.stack_alloc.lifetime.start(%85) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
          %86 = kgen.struct.gep %85[1] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index1, %86 : !kgen.pointer<index>
          %87 = kgen.struct.gep %85[0] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %54, %87 : !kgen.pointer<pointer<none>>
          %88 = kgen.struct.gep %85[2] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index2305843009213693952, %88 : !kgen.pointer<index>
          %89 = pop.pointer.bitcast %85 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
          %90 = pop.load %88 : !kgen.pointer<index>
          %91 = index.and %90, %index-9223372036854775808
          %92 = index.cmp ne(%91, %index0)
          %93 = hlcf.if %92 -> !kgen.pointer<none> {
            hlcf.yield %89 : !kgen.pointer<none>
          } else {
            %102 = pop.load %87 : !kgen.pointer<pointer<none>>
            hlcf.yield %102 : !kgen.pointer<none>
          }
          %94 = pop.load %88 : !kgen.pointer<index>
          %95 = index.and %94, %index-9223372036854775808
          %96 = index.cmp ne(%95, %index0)
          %97 = hlcf.if %96 -> index {
            %102 = pop.load %88 : !kgen.pointer<index>
            %103 = index.and %102, %index2233785415175766016
            %104 = index.shrs %103, %index56
            hlcf.yield %104 : index
          } else {
            %102 = pop.load %86 : !kgen.pointer<index>
            hlcf.yield %102 : index
          }
          %98 = kgen.struct.create(%93, %97) : !kgen.struct<(pointer<none>, index)>
          kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg0, %98) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
          %99 = pop.load %88 : !kgen.pointer<index>
          %100 = index.and %99, %index4611686018427387904
          %101 = index.cmp ne(%100, %index0)
          hlcf.if %101 {
            %102 = pop.load %87 : !kgen.pointer<pointer<none>>
            %103 = pop.pointer.bitcast %102 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
            %104 = pop.offset %103[%idx-8] : !kgen.pointer<scalar<ui8>>
            %105 = pop.pointer.bitcast %104 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
            %106 = kgen.struct.gep %105[0] : <struct<(scalar<index>) memoryOnly>>
            %107 = pop.atomic.rmw sub(%106, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
            %108 = pop.cmp eq(%107, %simd_28) : <1, index>
            %109 = pop.cast_to_builtin %108 : !pop.scalar<bool> to i1
            hlcf.if %109 {
              pop.fence syncscope("") acquire
              pop.aligned_free %104 : <scalar<ui8>>
              hlcf.yield
            } else {
              hlcf.yield
            }
            hlcf.yield
          } else {
            hlcf.yield
          }
          pop.stack_alloc.lifetime.end(%85) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
          hlcf.yield
        } else {
          hlcf.yield
        }
        %81 = pop.cmp ne(%67, %simd_27) : <1, ui64>
        %82 = pop.cast_to_builtin %81 : !pop.scalar<bool> to i1
        %83 = pop.xor %82, %0
        %84 = hlcf.if %83 -> i1 {
          %85 = index.cmp ne(%71, %index0)
          %86 = pop.xor %85, %0
          hlcf.yield %86 : i1
        } else {
          hlcf.yield %false : i1
        }
        hlcf.if %84 {
          %85 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
          pop.stack_alloc.lifetime.start(%85) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
          %86 = kgen.struct.gep %85[1] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index3, %86 : !kgen.pointer<index>
          %87 = kgen.struct.gep %85[0] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %52, %87 : !kgen.pointer<pointer<none>>
          %88 = kgen.struct.gep %85[2] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index2305843009213693952, %88 : !kgen.pointer<index>
          %89 = pop.pointer.bitcast %85 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
          %90 = pop.load %88 : !kgen.pointer<index>
          %91 = index.and %90, %index-9223372036854775808
          %92 = index.cmp ne(%91, %index0)
          %93 = hlcf.if %92 -> !kgen.pointer<none> {
            hlcf.yield %89 : !kgen.pointer<none>
          } else {
            %102 = pop.load %87 : !kgen.pointer<pointer<none>>
            hlcf.yield %102 : !kgen.pointer<none>
          }
          %94 = pop.load %88 : !kgen.pointer<index>
          %95 = index.and %94, %index-9223372036854775808
          %96 = index.cmp ne(%95, %index0)
          %97 = hlcf.if %96 -> index {
            %102 = pop.load %88 : !kgen.pointer<index>
            %103 = index.and %102, %index2233785415175766016
            %104 = index.shrs %103, %index56
            hlcf.yield %104 : index
          } else {
            %102 = pop.load %86 : !kgen.pointer<index>
            hlcf.yield %102 : index
          }
          %98 = kgen.struct.create(%93, %97) : !kgen.struct<(pointer<none>, index)>
          kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg0, %98) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
          %99 = pop.load %88 : !kgen.pointer<index>
          %100 = index.and %99, %index4611686018427387904
          %101 = index.cmp ne(%100, %index0)
          hlcf.if %101 {
            %102 = pop.load %87 : !kgen.pointer<pointer<none>>
            %103 = pop.pointer.bitcast %102 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
            %104 = pop.offset %103[%idx-8] : !kgen.pointer<scalar<ui8>>
            %105 = pop.pointer.bitcast %104 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
            %106 = kgen.struct.gep %105[0] : <struct<(scalar<index>) memoryOnly>>
            %107 = pop.atomic.rmw sub(%106, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
            %108 = pop.cmp eq(%107, %simd_28) : <1, index>
            %109 = pop.cast_to_builtin %108 : !pop.scalar<bool> to i1
            hlcf.if %109 {
              pop.fence syncscope("") acquire
              pop.aligned_free %104 : <scalar<ui8>>
              hlcf.yield
            } else {
              hlcf.yield
            }
            hlcf.yield
          } else {
            hlcf.yield
          }
          pop.stack_alloc.lifetime.end(%85) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
          hlcf.yield
        } else {
          %85:2 = kgen.call @"std::builtin::_format_float::_to_decimal[::DType,::DType](::SIMD[$0, ::Int(1)]&,::Int&),CarrierDType=ui64,dtype=f64"(%67, %71) : (!pop.scalar<ui64>, index) -> (!pop.scalar<ui64>, index)
          %86 = index.cmp slt(%85#1, %index0)
          %87 = index.mul %85#1, %index-1
          %88 = pop.select %86, %87, %85#1 : index
          %89 = pop.stack_allocation 1 x array<21, scalar<ui8>> align 1 marked
          pop.stack_alloc.lifetime.start(%89) : !kgen.pointer<array<21, scalar<ui8>>>
          %90:2 = hlcf.loop "_loop_2" (%arg2 = %index0 : index, %arg3 = %85#0 : !pop.scalar<ui64>, %arg4 = %85#1 : index) -> (index, index) {
            %110 = pop.cmp gt(%arg3, %simd_27) : <1, ui64>
            %111 = pop.cast_to_builtin %110 : !pop.scalar<bool> to i1
            hlcf.if %111 {
              hlcf.yield
            } else {
              hlcf.break "_loop_2" %arg2, %arg4 : index, index
            }
            %112 = pop.cast_from_builtin %arg2 : index to !pop.scalar<index>
            %113 = pop.cast %112 : !pop.scalar<index> to !pop.scalar<uindex>
            %114 = pop.cmp lt(%113, %simd) : <1, uindex>
            %115 = pop.cast_to_builtin %114 : !pop.scalar<bool> to i1
            %116 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
            pop.stack_alloc.lifetime.start(%116) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
            pop.store %array, %116 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
            %117 = pop.pointer.bitcast %116 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<index>
            %118 = pop.load %117 : !kgen.pointer<index>
            %119 = index.cmp eq(%118, %index-1)
            %120 = pop.select %119, %index0, %index-1 : index
            pop.stack_alloc.lifetime.end(%116) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
            %121 = index.cmp eq(%120, %index-1)
            %122 = hlcf.if %121 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
              %154 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
              pop.stack_alloc.lifetime.start(%154) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
              pop.store %array, %154 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
              %155 = pop.pointer.bitcast %154 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
              %156 = pop.load %155 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
              pop.stack_alloc.lifetime.end(%154) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
              hlcf.yield %156 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
            } else {
              hlcf.yield %10 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
            }
            %123 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
            pop.stack_alloc.lifetime.start(%123) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
            %124 = kgen.struct.gep %123[1] : <struct<(pointer<none>, index, index) memoryOnly>>
            pop.store %index6, %124 : !kgen.pointer<index>
            %125 = kgen.struct.gep %123[0] : <struct<(pointer<none>, index, index) memoryOnly>>
            pop.store %34, %125 : !kgen.pointer<pointer<none>>
            %126 = kgen.struct.gep %123[2] : <struct<(pointer<none>, index, index) memoryOnly>>
            pop.store %index2305843009213693952, %126 : !kgen.pointer<index>
            %127 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
            pop.stack_alloc.lifetime.start(%127) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
            %128 = kgen.struct.gep %127[1] : <struct<(pointer<none>, index, index) memoryOnly>>
            pop.store %index39, %128 : !kgen.pointer<index>
            %129 = kgen.struct.gep %127[0] : <struct<(pointer<none>, index, index) memoryOnly>>
            pop.store %32, %129 : !kgen.pointer<pointer<none>>
            %130 = kgen.struct.gep %127[2] : <struct<(pointer<none>, index, index) memoryOnly>>
            pop.store %index2305843009213693952, %130 : !kgen.pointer<index>
            %131 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
            %132 = pop.pointer.bitcast %131 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
            pop.store %122, %132 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
            %133 = pop.load %131 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
            %134 = pop.array.get %133[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
            %135 = pop.array.create [%134] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
            %136 = kgen.struct.create(%135) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
            %137 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
            %138 = pop.pointer.bitcast %137 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
            pop.store %136, %137 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
            hlcf.if %115 {
              hlcf.yield
            } else {
              %154 = pop.stack_allocation 2048 x scalar<ui8> align 1
              %155 = pop.pointer.bitcast %154 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<none>
              %156 = kgen.struct.create(%155, %index0) : !kgen.struct<(pointer<none>, index) memoryOnly>
              %157 = kgen.call @"std::collections::string::string::String::write_to[::Writer](::String,$0&),writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%123, %156) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
              %158 = kgen.call @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%arg2, %157) : (index, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
              %159 = kgen.call @"std::collections::string::string::String::write_to[::Writer](::String,$0&),writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%127, %158) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
              %160 = kgen.call @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%idx20, %159) : (index, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
              %161 = kgen.struct.extract %160[0] : <(pointer<none>, index) memoryOnly>
              %162 = kgen.struct.extract %160[1] : <(pointer<none>, index) memoryOnly>
              %163 = index.add %162, %index1
              %164 = index.cmp sgt(%163, %index2048)
              hlcf.if %164 {
                kgen.call @"std::io::io::_printf[KGENParamList[::AnyType],::StringSlice[::Bool(False), StaticConstantOrigin, *?],*::AnyType,LITImmutOrigin,::Origin[::Bool(False), $3]](*$0,file:::FileDescriptor),types.values`=[],fmt={ #interp.memref<{[(#interp.memory_handle<16, \22HEAP_BUFFER_BYTES exceeded, increase with: `mojo -D HEAP_BUFFER_BYTES=4096`\\0A\\00\22 string>, const_global, [], [])], []}, 0, 0>, 76 }"(%index1) : (index) -> ()
                llvm.intr.trap
                hlcf.loop "_loop_0" {
                  hlcf.continue "_loop_0"
                }
                kgen.unreachable
              } else {
                hlcf.yield
              }
              %165 = pop.pointer.bitcast %161 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
              %166 = pop.offset %165[%162] : !kgen.pointer<scalar<ui8>>
              pop.store %simd_9, %166 : !kgen.pointer<scalar<ui8>>
              %167 = pop.load %137 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
              %168 = kgen.struct.extract %167[0] : <(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
              %169 = pop.array.get %168[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
              %170 = pop.array.create [%169] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
              %171 = kgen.struct.create(%170) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
              %172 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
              pop.store %171, %172 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
              %173 = pop.pointer.bitcast %172 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
              %174 = pop.load %173 : !kgen.pointer<index>
              %175 = index.cmp eq(%174, %index-1)
              %176 = pop.select %175, %index0, %index-1 : index
              %177 = index.cmp eq(%176, %index-1)
              %178 = hlcf.if %177 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
                %179 = pop.load %137 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                %180 = kgen.struct.extract %179[0] : <(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                %181 = pop.array.get %180[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                %182 = pop.array.create [%181] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                %183 = kgen.struct.create(%182) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                %184 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                pop.store %183, %184 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                %185 = pop.pointer.bitcast %184 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
                %186 = pop.load %185 : !kgen.pointer<index>
                %187 = index.cmp eq(%186, %index-1)
                %188 = pop.select %187, %index0, %index-1 : index
                %189 = index.cmp eq(%188, %index-1)
                %190 = pop.xor %189, %0
                hlcf.if %190 {
                  %192 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
                  pop.stack_alloc.lifetime.start(%192) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                  %193 = kgen.struct.gep %192[1] : <struct<(pointer<none>, index, index) memoryOnly>>
                  pop.store %index192, %193 : !kgen.pointer<index>
                  %194 = kgen.struct.gep %192[0] : <struct<(pointer<none>, index, index) memoryOnly>>
                  pop.store %30, %194 : !kgen.pointer<pointer<none>>
                  %195 = kgen.struct.gep %192[2] : <struct<(pointer<none>, index, index) memoryOnly>>
                  pop.store %index2305843009213693952, %195 : !kgen.pointer<index>
                  %196 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                  %197 = pop.pointer.bitcast %196 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                  pop.store %18, %197 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                  %198 = pop.load %196 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                  %199 = pop.array.get %198[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                  %200 = pop.array.create [%199] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                  %201 = kgen.struct.create(%200) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                  %202 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                  %203 = pop.pointer.bitcast %202 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                  pop.store %201, %202 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                  %204 = pop.pointer.bitcast %202 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
                  %205 = pop.load %204 : !kgen.pointer<index>
                  %206 = index.cmp eq(%205, %index-1)
                  %207 = pop.select %206, %index0, %index-1 : index
                  %208 = index.cmp eq(%207, %index-1)
                  %209 = hlcf.if %208 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
                    %227 = pop.load %203 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                    hlcf.yield %227 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
                  } else {
                    hlcf.yield %22 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
                  }
                  %210 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
                  pop.stack_alloc.lifetime.start(%210) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                  %211 = kgen.struct.gep %210[1] : <struct<(pointer<none>, index, index) memoryOnly>>
                  pop.store %index1, %211 : !kgen.pointer<index>
                  %212 = kgen.struct.gep %210[0] : <struct<(pointer<none>, index, index) memoryOnly>>
                  pop.store %24, %212 : !kgen.pointer<pointer<none>>
                  %213 = kgen.struct.gep %210[2] : <struct<(pointer<none>, index, index) memoryOnly>>
                  pop.store %index2305843009213693952, %213 : !kgen.pointer<index>
                  %214 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
                  pop.stack_alloc.lifetime.start(%214) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                  %215 = kgen.struct.gep %214[1] : <struct<(pointer<none>, index, index) memoryOnly>>
                  pop.store %index2, %215 : !kgen.pointer<index>
                  %216 = kgen.struct.gep %214[0] : <struct<(pointer<none>, index, index) memoryOnly>>
                  pop.store %26, %216 : !kgen.pointer<pointer<none>>
                  %217 = kgen.struct.gep %214[2] : <struct<(pointer<none>, index, index) memoryOnly>>
                  pop.store %index2305843009213693952, %217 : !kgen.pointer<index>
                  kgen.call @"std::io::io::print[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?],flush:::Bool,file:::FileDescriptor$),Ts.values`=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::reflection::location::SourceLocation\22>>, struct<(index, index, struct<(pointer<none>, index)>)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]]"(%struct_4, %210, %209, %214, %192, %61, %struct, %0, %index1) : (!kgen.struct<(pointer<none>, index)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index)>, i1, index owned) -> ()
                  %218 = pop.load %213 : !kgen.pointer<index>
                  %219 = index.and %218, %index4611686018427387904
                  %220 = index.cmp ne(%219, %index0)
                  hlcf.if %220 {
                    %227 = pop.load %212 : !kgen.pointer<pointer<none>>
                    %228 = pop.pointer.bitcast %227 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                    %229 = pop.offset %228[%idx-8] : !kgen.pointer<scalar<ui8>>
                    %230 = pop.pointer.bitcast %229 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                    %231 = kgen.struct.gep %230[0] : <struct<(scalar<index>) memoryOnly>>
                    %232 = pop.atomic.rmw sub(%231, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                    %233 = pop.cmp eq(%232, %simd_28) : <1, index>
                    %234 = pop.cast_to_builtin %233 : !pop.scalar<bool> to i1
                    hlcf.if %234 {
                      pop.fence syncscope("") acquire
                      pop.aligned_free %229 : <scalar<ui8>>
                      hlcf.yield
                    } else {
                      hlcf.yield
                    }
                    hlcf.yield
                  } else {
                    hlcf.yield
                  }
                  %221 = pop.load %217 : !kgen.pointer<index>
                  %222 = index.and %221, %index4611686018427387904
                  %223 = index.cmp ne(%222, %index0)
                  hlcf.if %223 {
                    %227 = pop.load %216 : !kgen.pointer<pointer<none>>
                    %228 = pop.pointer.bitcast %227 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                    %229 = pop.offset %228[%idx-8] : !kgen.pointer<scalar<ui8>>
                    %230 = pop.pointer.bitcast %229 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                    %231 = kgen.struct.gep %230[0] : <struct<(scalar<index>) memoryOnly>>
                    %232 = pop.atomic.rmw sub(%231, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                    %233 = pop.cmp eq(%232, %simd_28) : <1, index>
                    %234 = pop.cast_to_builtin %233 : !pop.scalar<bool> to i1
                    hlcf.if %234 {
                      pop.fence syncscope("") acquire
                      pop.aligned_free %229 : <scalar<ui8>>
                      hlcf.yield
                    } else {
                      hlcf.yield
                    }
                    hlcf.yield
                  } else {
                    hlcf.yield
                  }
                  %224 = pop.load %195 : !kgen.pointer<index>
                  %225 = index.and %224, %index4611686018427387904
                  %226 = index.cmp ne(%225, %index0)
                  hlcf.if %226 {
                    %227 = pop.load %194 : !kgen.pointer<pointer<none>>
                    %228 = pop.pointer.bitcast %227 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                    %229 = pop.offset %228[%idx-8] : !kgen.pointer<scalar<ui8>>
                    %230 = pop.pointer.bitcast %229 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                    %231 = kgen.struct.gep %230[0] : <struct<(scalar<index>) memoryOnly>>
                    %232 = pop.atomic.rmw sub(%231, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                    %233 = pop.cmp eq(%232, %simd_28) : <1, index>
                    %234 = pop.cast_to_builtin %233 : !pop.scalar<bool> to i1
                    hlcf.if %234 {
                      pop.fence syncscope("") acquire
                      pop.aligned_free %229 : <scalar<ui8>>
                      hlcf.yield
                    } else {
                      hlcf.yield
                    }
                    hlcf.yield
                  } else {
                    hlcf.yield
                  }
                  llvm.intr.trap
                  hlcf.loop "_loop_0" {
                    hlcf.continue "_loop_0"
                  }
                  kgen.unreachable
                } else {
                  hlcf.yield
                }
                %191 = pop.load %138 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                hlcf.yield %191 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
              } else {
                hlcf.yield %35 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
              }
              kgen.call @"std::builtin::debug_assert::_debug_assert_msg[LITImmutOrigin,::Origin[::Bool(False), $0]](::UnsafePointer[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1, ::AddressSpace(::Int(0))],::Int,::SourceLocation)"(%161, %178) : (!kgen.pointer<none>, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>) -> ()
              hlcf.yield
            }
            %139 = pop.load %126 : !kgen.pointer<index>
            %140 = index.and %139, %index4611686018427387904
            %141 = index.cmp ne(%140, %index0)
            hlcf.if %141 {
              %154 = pop.load %125 : !kgen.pointer<pointer<none>>
              %155 = pop.pointer.bitcast %154 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
              %156 = pop.offset %155[%idx-8] : !kgen.pointer<scalar<ui8>>
              %157 = pop.pointer.bitcast %156 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
              %158 = kgen.struct.gep %157[0] : <struct<(scalar<index>) memoryOnly>>
              %159 = pop.atomic.rmw sub(%158, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
              %160 = pop.cmp eq(%159, %simd_28) : <1, index>
              %161 = pop.cast_to_builtin %160 : !pop.scalar<bool> to i1
              hlcf.if %161 {
                pop.fence syncscope("") acquire
                pop.aligned_free %156 : <scalar<ui8>>
                hlcf.yield
              } else {
                hlcf.yield
              }
              hlcf.yield
            } else {
              hlcf.yield
            }
            %142 = pop.load %130 : !kgen.pointer<index>
            %143 = index.and %142, %index4611686018427387904
            %144 = index.cmp ne(%143, %index0)
            hlcf.if %144 {
              %154 = pop.load %129 : !kgen.pointer<pointer<none>>
              %155 = pop.pointer.bitcast %154 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
              %156 = pop.offset %155[%idx-8] : !kgen.pointer<scalar<ui8>>
              %157 = pop.pointer.bitcast %156 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
              %158 = kgen.struct.gep %157[0] : <struct<(scalar<index>) memoryOnly>>
              %159 = pop.atomic.rmw sub(%158, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
              %160 = pop.cmp eq(%159, %simd_28) : <1, index>
              %161 = pop.cast_to_builtin %160 : !pop.scalar<bool> to i1
              hlcf.if %161 {
                pop.fence syncscope("") acquire
                pop.aligned_free %156 : <scalar<ui8>>
                hlcf.yield
              } else {
                hlcf.yield
              }
              hlcf.yield
            } else {
              hlcf.yield
            }
            %145 = pop.array.gep %89[%arg2] : <array<21, scalar<ui8>>>
            %146 = pop.rem %arg3, %simd_29 : !pop.scalar<ui64>
            %147 = pop.cast fast %146 : !pop.scalar<ui64> to !pop.scalar<ui8>
            pop.store %147, %145 : !kgen.pointer<scalar<ui8>>
            %148 = pop.floordiv %arg3, %simd_29 : !pop.scalar<ui64>
            %149 = index.add %arg2, %index1
            %150 = pop.cmp gt(%148, %simd_27) : <1, ui64>
            %151 = pop.cast_to_builtin %150 : !pop.scalar<bool> to i1
            %152 = index.add %arg4, %index1
            %153 = pop.select %151, %152, %arg4 : index
            hlcf.continue "_loop_2" %149, %148, %153 : index, !pop.scalar<ui64>, index
          }
          %91 = index.sub %90#0, %index1
          %92 = index.maxs %90#0, %index0
          %93 = index.sub %92, %index1
          %94 = pop.array.gep %89[%91] : <array<21, scalar<ui8>>>
          %95 = pop.cast_from_builtin %91 : index to !pop.scalar<index>
          %96 = pop.cast %95 : !pop.scalar<index> to !pop.scalar<uindex>
          %97 = pop.cmp lt(%96, %simd) : <1, uindex>
          %98 = pop.cast_to_builtin %97 : !pop.scalar<bool> to i1
          %99 = index.cmp sgt(%88, %90#0)
          %100 = index.maxs %91, %index0
          %101 = index.sub %100, %index1
          %102 = index.sub %88, %90#0
          %103 = index.cmp slt(%102, %index1)
          %104 = index.maxs %102, %index0
          %105 = index.cmp slt(%90#1, %index0)
          %106 = index.cmp sgt(%102, %index3)
          %107 = pop.select %105, %106, %false : i1
          %108 = index.cmp sgt(%90#1, %index15)
          %109 = pop.select %107, %true, %108 : i1
          hlcf.if %109 {
            %110 = pop.cmp lt(%85#0, %simd_29) : <1, ui64>
            %111 = pop.cast_to_builtin %110 : !pop.scalar<bool> to i1
            hlcf.if %111 {
              kgen.call @"std::builtin::simd::SIMD::write_to[::Writer](::SIMD[$0, $1],$2&),dtype=ui64,size=1,writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>]"(%85#0, %arg0) : (!pop.scalar<ui64>, !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut) -> ()
              hlcf.yield
            } else {
              %118 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
              pop.stack_alloc.lifetime.start(%118) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
              pop.store %array, %118 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
              %119 = pop.pointer.bitcast %118 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<index>
              %120 = pop.load %119 : !kgen.pointer<index>
              %121 = index.cmp eq(%120, %index-1)
              %122 = pop.select %121, %index0, %index-1 : index
              pop.stack_alloc.lifetime.end(%118) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
              %123 = index.cmp eq(%122, %index-1)
              %124 = hlcf.if %123 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
                %165 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
                pop.stack_alloc.lifetime.start(%165) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                pop.store %array, %165 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                %166 = pop.pointer.bitcast %165 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                %167 = pop.load %166 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                pop.stack_alloc.lifetime.end(%165) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                hlcf.yield %167 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
              } else {
                hlcf.yield %9 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
              }
              %125 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
              pop.stack_alloc.lifetime.start(%125) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
              %126 = kgen.struct.gep %125[1] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %index6, %126 : !kgen.pointer<index>
              %127 = kgen.struct.gep %125[0] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %34, %127 : !kgen.pointer<pointer<none>>
              %128 = kgen.struct.gep %125[2] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %index2305843009213693952, %128 : !kgen.pointer<index>
              %129 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
              pop.stack_alloc.lifetime.start(%129) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
              %130 = kgen.struct.gep %129[1] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %index39, %130 : !kgen.pointer<index>
              %131 = kgen.struct.gep %129[0] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %32, %131 : !kgen.pointer<pointer<none>>
              %132 = kgen.struct.gep %129[2] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %index2305843009213693952, %132 : !kgen.pointer<index>
              %133 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
              %134 = pop.pointer.bitcast %133 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
              pop.store %124, %134 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
              %135 = pop.load %133 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
              %136 = pop.array.get %135[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
              %137 = pop.array.create [%136] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
              %138 = kgen.struct.create(%137) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
              %139 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
              %140 = pop.pointer.bitcast %139 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
              pop.store %138, %139 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
              hlcf.if %98 {
                hlcf.yield
              } else {
                %165 = pop.stack_allocation 2048 x scalar<ui8> align 1
                %166 = pop.pointer.bitcast %165 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<none>
                %167 = kgen.struct.create(%166, %index0) : !kgen.struct<(pointer<none>, index) memoryOnly>
                %168 = kgen.call @"std::collections::string::string::String::write_to[::Writer](::String,$0&),writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%125, %167) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
                %169 = kgen.call @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%91, %168) : (index, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
                %170 = kgen.call @"std::collections::string::string::String::write_to[::Writer](::String,$0&),writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%129, %169) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
                %171 = kgen.call @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%idx20, %170) : (index, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
                %172 = kgen.struct.extract %171[0] : <(pointer<none>, index) memoryOnly>
                %173 = kgen.struct.extract %171[1] : <(pointer<none>, index) memoryOnly>
                %174 = index.add %173, %index1
                %175 = index.cmp sgt(%174, %index2048)
                hlcf.if %175 {
                  kgen.call @"std::io::io::_printf[KGENParamList[::AnyType],::StringSlice[::Bool(False), StaticConstantOrigin, *?],*::AnyType,LITImmutOrigin,::Origin[::Bool(False), $3]](*$0,file:::FileDescriptor),types.values`=[],fmt={ #interp.memref<{[(#interp.memory_handle<16, \22HEAP_BUFFER_BYTES exceeded, increase with: `mojo -D HEAP_BUFFER_BYTES=4096`\\0A\\00\22 string>, const_global, [], [])], []}, 0, 0>, 76 }"(%index1) : (index) -> ()
                  llvm.intr.trap
                  hlcf.loop "_loop_0" {
                    hlcf.continue "_loop_0"
                  }
                  kgen.unreachable
                } else {
                  hlcf.yield
                }
                %176 = pop.pointer.bitcast %172 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                %177 = pop.offset %176[%173] : !kgen.pointer<scalar<ui8>>
                pop.store %simd_9, %177 : !kgen.pointer<scalar<ui8>>
                %178 = pop.load %139 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                %179 = kgen.struct.extract %178[0] : <(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                %180 = pop.array.get %179[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                %181 = pop.array.create [%180] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                %182 = kgen.struct.create(%181) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                %183 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                pop.store %182, %183 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                %184 = pop.pointer.bitcast %183 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
                %185 = pop.load %184 : !kgen.pointer<index>
                %186 = index.cmp eq(%185, %index-1)
                %187 = pop.select %186, %index0, %index-1 : index
                %188 = index.cmp eq(%187, %index-1)
                %189 = hlcf.if %188 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
                  %190 = pop.load %139 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                  %191 = kgen.struct.extract %190[0] : <(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                  %192 = pop.array.get %191[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                  %193 = pop.array.create [%192] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                  %194 = kgen.struct.create(%193) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                  %195 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                  pop.store %194, %195 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                  %196 = pop.pointer.bitcast %195 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
                  %197 = pop.load %196 : !kgen.pointer<index>
                  %198 = index.cmp eq(%197, %index-1)
                  %199 = pop.select %198, %index0, %index-1 : index
                  %200 = index.cmp eq(%199, %index-1)
                  %201 = pop.xor %200, %0
                  hlcf.if %201 {
                    %203 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
                    pop.stack_alloc.lifetime.start(%203) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                    %204 = kgen.struct.gep %203[1] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %index192, %204 : !kgen.pointer<index>
                    %205 = kgen.struct.gep %203[0] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %30, %205 : !kgen.pointer<pointer<none>>
                    %206 = kgen.struct.gep %203[2] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %index2305843009213693952, %206 : !kgen.pointer<index>
                    %207 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                    %208 = pop.pointer.bitcast %207 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                    pop.store %18, %208 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                    %209 = pop.load %207 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                    %210 = pop.array.get %209[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                    %211 = pop.array.create [%210] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                    %212 = kgen.struct.create(%211) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                    %213 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                    %214 = pop.pointer.bitcast %213 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                    pop.store %212, %213 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                    %215 = pop.pointer.bitcast %213 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
                    %216 = pop.load %215 : !kgen.pointer<index>
                    %217 = index.cmp eq(%216, %index-1)
                    %218 = pop.select %217, %index0, %index-1 : index
                    %219 = index.cmp eq(%218, %index-1)
                    %220 = hlcf.if %219 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
                      %238 = pop.load %214 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                      hlcf.yield %238 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
                    } else {
                      hlcf.yield %22 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
                    }
                    %221 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
                    pop.stack_alloc.lifetime.start(%221) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                    %222 = kgen.struct.gep %221[1] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %index1, %222 : !kgen.pointer<index>
                    %223 = kgen.struct.gep %221[0] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %24, %223 : !kgen.pointer<pointer<none>>
                    %224 = kgen.struct.gep %221[2] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %index2305843009213693952, %224 : !kgen.pointer<index>
                    %225 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
                    pop.stack_alloc.lifetime.start(%225) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                    %226 = kgen.struct.gep %225[1] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %index2, %226 : !kgen.pointer<index>
                    %227 = kgen.struct.gep %225[0] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %26, %227 : !kgen.pointer<pointer<none>>
                    %228 = kgen.struct.gep %225[2] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %index2305843009213693952, %228 : !kgen.pointer<index>
                    kgen.call @"std::io::io::print[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?],flush:::Bool,file:::FileDescriptor$),Ts.values`=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::reflection::location::SourceLocation\22>>, struct<(index, index, struct<(pointer<none>, index)>)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]]"(%struct_4, %221, %220, %225, %203, %61, %struct, %0, %index1) : (!kgen.struct<(pointer<none>, index)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index)>, i1, index owned) -> ()
                    %229 = pop.load %224 : !kgen.pointer<index>
                    %230 = index.and %229, %index4611686018427387904
                    %231 = index.cmp ne(%230, %index0)
                    hlcf.if %231 {
                      %238 = pop.load %223 : !kgen.pointer<pointer<none>>
                      %239 = pop.pointer.bitcast %238 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                      %240 = pop.offset %239[%idx-8] : !kgen.pointer<scalar<ui8>>
                      %241 = pop.pointer.bitcast %240 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                      %242 = kgen.struct.gep %241[0] : <struct<(scalar<index>) memoryOnly>>
                      %243 = pop.atomic.rmw sub(%242, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                      %244 = pop.cmp eq(%243, %simd_28) : <1, index>
                      %245 = pop.cast_to_builtin %244 : !pop.scalar<bool> to i1
                      hlcf.if %245 {
                        pop.fence syncscope("") acquire
                        pop.aligned_free %240 : <scalar<ui8>>
                        hlcf.yield
                      } else {
                        hlcf.yield
                      }
                      hlcf.yield
                    } else {
                      hlcf.yield
                    }
                    %232 = pop.load %228 : !kgen.pointer<index>
                    %233 = index.and %232, %index4611686018427387904
                    %234 = index.cmp ne(%233, %index0)
                    hlcf.if %234 {
                      %238 = pop.load %227 : !kgen.pointer<pointer<none>>
                      %239 = pop.pointer.bitcast %238 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                      %240 = pop.offset %239[%idx-8] : !kgen.pointer<scalar<ui8>>
                      %241 = pop.pointer.bitcast %240 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                      %242 = kgen.struct.gep %241[0] : <struct<(scalar<index>) memoryOnly>>
                      %243 = pop.atomic.rmw sub(%242, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                      %244 = pop.cmp eq(%243, %simd_28) : <1, index>
                      %245 = pop.cast_to_builtin %244 : !pop.scalar<bool> to i1
                      hlcf.if %245 {
                        pop.fence syncscope("") acquire
                        pop.aligned_free %240 : <scalar<ui8>>
                        hlcf.yield
                      } else {
                        hlcf.yield
                      }
                      hlcf.yield
                    } else {
                      hlcf.yield
                    }
                    %235 = pop.load %206 : !kgen.pointer<index>
                    %236 = index.and %235, %index4611686018427387904
                    %237 = index.cmp ne(%236, %index0)
                    hlcf.if %237 {
                      %238 = pop.load %205 : !kgen.pointer<pointer<none>>
                      %239 = pop.pointer.bitcast %238 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                      %240 = pop.offset %239[%idx-8] : !kgen.pointer<scalar<ui8>>
                      %241 = pop.pointer.bitcast %240 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                      %242 = kgen.struct.gep %241[0] : <struct<(scalar<index>) memoryOnly>>
                      %243 = pop.atomic.rmw sub(%242, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                      %244 = pop.cmp eq(%243, %simd_28) : <1, index>
                      %245 = pop.cast_to_builtin %244 : !pop.scalar<bool> to i1
                      hlcf.if %245 {
                        pop.fence syncscope("") acquire
                        pop.aligned_free %240 : <scalar<ui8>>
                        hlcf.yield
                      } else {
                        hlcf.yield
                      }
                      hlcf.yield
                    } else {
                      hlcf.yield
                    }
                    llvm.intr.trap
                    hlcf.loop "_loop_0" {
                      hlcf.continue "_loop_0"
                    }
                    kgen.unreachable
                  } else {
                    hlcf.yield
                  }
                  %202 = pop.load %140 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                  hlcf.yield %202 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
                } else {
                  hlcf.yield %35 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
                }
                kgen.call @"std::builtin::debug_assert::_debug_assert_msg[LITImmutOrigin,::Origin[::Bool(False), $0]](::UnsafePointer[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1, ::AddressSpace(::Int(0))],::Int,::SourceLocation)"(%172, %189) : (!kgen.pointer<none>, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>) -> ()
                hlcf.yield
              }
              %141 = pop.load %128 : !kgen.pointer<index>
              %142 = index.and %141, %index4611686018427387904
              %143 = index.cmp ne(%142, %index0)
              hlcf.if %143 {
                %165 = pop.load %127 : !kgen.pointer<pointer<none>>
                %166 = pop.pointer.bitcast %165 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                %167 = pop.offset %166[%idx-8] : !kgen.pointer<scalar<ui8>>
                %168 = pop.pointer.bitcast %167 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                %169 = kgen.struct.gep %168[0] : <struct<(scalar<index>) memoryOnly>>
                %170 = pop.atomic.rmw sub(%169, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                %171 = pop.cmp eq(%170, %simd_28) : <1, index>
                %172 = pop.cast_to_builtin %171 : !pop.scalar<bool> to i1
                hlcf.if %172 {
                  pop.fence syncscope("") acquire
                  pop.aligned_free %167 : <scalar<ui8>>
                  hlcf.yield
                } else {
                  hlcf.yield
                }
                hlcf.yield
              } else {
                hlcf.yield
              }
              %144 = pop.load %132 : !kgen.pointer<index>
              %145 = index.and %144, %index4611686018427387904
              %146 = index.cmp ne(%145, %index0)
              hlcf.if %146 {
                %165 = pop.load %131 : !kgen.pointer<pointer<none>>
                %166 = pop.pointer.bitcast %165 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                %167 = pop.offset %166[%idx-8] : !kgen.pointer<scalar<ui8>>
                %168 = pop.pointer.bitcast %167 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                %169 = kgen.struct.gep %168[0] : <struct<(scalar<index>) memoryOnly>>
                %170 = pop.atomic.rmw sub(%169, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                %171 = pop.cmp eq(%170, %simd_28) : <1, index>
                %172 = pop.cast_to_builtin %171 : !pop.scalar<bool> to i1
                hlcf.if %172 {
                  pop.fence syncscope("") acquire
                  pop.aligned_free %167 : <scalar<ui8>>
                  hlcf.yield
                } else {
                  hlcf.yield
                }
                hlcf.yield
              } else {
                hlcf.yield
              }
              %147 = pop.load %94 : !kgen.pointer<scalar<ui8>>
              kgen.call @"std::builtin::simd::SIMD::write_to[::Writer](::SIMD[$0, $1],$2&),dtype=ui8,size=1,writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>]"(%147, %arg0) : (!pop.scalar<ui8>, !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut) -> ()
              %148 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
              pop.stack_alloc.lifetime.start(%148) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
              %149 = kgen.struct.gep %148[1] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %index1, %149 : !kgen.pointer<index>
              %150 = kgen.struct.gep %148[0] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %50, %150 : !kgen.pointer<pointer<none>>
              %151 = kgen.struct.gep %148[2] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %index2305843009213693952, %151 : !kgen.pointer<index>
              %152 = pop.pointer.bitcast %148 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
              %153 = pop.load %151 : !kgen.pointer<index>
              %154 = index.and %153, %index-9223372036854775808
              %155 = index.cmp ne(%154, %index0)
              %156 = hlcf.if %155 -> !kgen.pointer<none> {
                hlcf.yield %152 : !kgen.pointer<none>
              } else {
                %165 = pop.load %150 : !kgen.pointer<pointer<none>>
                hlcf.yield %165 : !kgen.pointer<none>
              }
              %157 = pop.load %151 : !kgen.pointer<index>
              %158 = index.and %157, %index-9223372036854775808
              %159 = index.cmp ne(%158, %index0)
              %160 = hlcf.if %159 -> index {
                %165 = pop.load %151 : !kgen.pointer<index>
                %166 = index.and %165, %index2233785415175766016
                %167 = index.shrs %166, %index56
                hlcf.yield %167 : index
              } else {
                %165 = pop.load %149 : !kgen.pointer<index>
                hlcf.yield %165 : index
              }
              %161 = kgen.struct.create(%156, %160) : !kgen.struct<(pointer<none>, index)>
              kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg0, %161) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
              %162 = pop.load %151 : !kgen.pointer<index>
              %163 = index.and %162, %index4611686018427387904
              %164 = index.cmp ne(%163, %index0)
              hlcf.if %164 {
                %165 = pop.load %150 : !kgen.pointer<pointer<none>>
                %166 = pop.pointer.bitcast %165 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                %167 = pop.offset %166[%idx-8] : !kgen.pointer<scalar<ui8>>
                %168 = pop.pointer.bitcast %167 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                %169 = kgen.struct.gep %168[0] : <struct<(scalar<index>) memoryOnly>>
                %170 = pop.atomic.rmw sub(%169, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                %171 = pop.cmp eq(%170, %simd_28) : <1, index>
                %172 = pop.cast_to_builtin %171 : !pop.scalar<bool> to i1
                hlcf.if %172 {
                  pop.fence syncscope("") acquire
                  pop.aligned_free %167 : <scalar<ui8>>
                  hlcf.yield
                } else {
                  hlcf.yield
                }
                hlcf.yield
              } else {
                hlcf.yield
              }
              pop.stack_alloc.lifetime.end(%148) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
              hlcf.yield
            }
            hlcf.loop "_loop_5" (%arg2 = %101 : index) {
              %118 = index.add %arg2, %index-1
              %119 = index.sub %arg2, %index-1
              %120 = index.cmp sgt(%arg2, %index-1)
              %121 = pop.select %120, %119, %index0 : index
              %122 = index.cmp sle(%121, %index0)
              %123 = pop.select %122, %arg2, %118 : index
              %124:2 = lit.try "try3" -> index, index {
                hlcf.if %122 {
                  pop.stack_alloc.lifetime.end(%89) : !kgen.pointer<array<21, scalar<ui8>>>
                  lit.try.raise "try3" %123, %arg2 : index, index
                } else {
                  hlcf.yield
                }
                lit.try.yield %123, %arg2 : index, index
              } except (%arg3: index, %arg4: index) {
                hlcf.break "_loop_5"
              } else (%arg3: index, %arg4: index) {
                lit.try.yield %arg3, %arg4 : index, index
              }
              %125 = pop.cast_from_builtin %124#1 : index to !pop.scalar<index>
              %126 = pop.cast %125 : !pop.scalar<index> to !pop.scalar<uindex>
              %127 = pop.cmp lt(%126, %simd) : <1, uindex>
              %128 = pop.cast_to_builtin %127 : !pop.scalar<bool> to i1
              %129 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
              pop.stack_alloc.lifetime.start(%129) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
              pop.store %array, %129 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
              %130 = pop.pointer.bitcast %129 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<index>
              %131 = pop.load %130 : !kgen.pointer<index>
              %132 = index.cmp eq(%131, %index-1)
              %133 = pop.select %132, %index0, %index-1 : index
              pop.stack_alloc.lifetime.end(%129) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
              %134 = index.cmp eq(%133, %index-1)
              %135 = hlcf.if %134 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
                %160 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
                pop.stack_alloc.lifetime.start(%160) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                pop.store %array, %160 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                %161 = pop.pointer.bitcast %160 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                %162 = pop.load %161 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                pop.stack_alloc.lifetime.end(%160) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                hlcf.yield %162 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
              } else {
                hlcf.yield %8 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
              }
              %136 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
              pop.stack_alloc.lifetime.start(%136) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
              %137 = kgen.struct.gep %136[1] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %index6, %137 : !kgen.pointer<index>
              %138 = kgen.struct.gep %136[0] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %34, %138 : !kgen.pointer<pointer<none>>
              %139 = kgen.struct.gep %136[2] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %index2305843009213693952, %139 : !kgen.pointer<index>
              %140 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
              pop.stack_alloc.lifetime.start(%140) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
              %141 = kgen.struct.gep %140[1] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %index39, %141 : !kgen.pointer<index>
              %142 = kgen.struct.gep %140[0] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %32, %142 : !kgen.pointer<pointer<none>>
              %143 = kgen.struct.gep %140[2] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %index2305843009213693952, %143 : !kgen.pointer<index>
              %144 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
              %145 = pop.pointer.bitcast %144 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
              pop.store %135, %145 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
              %146 = pop.load %144 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
              %147 = pop.array.get %146[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
              %148 = pop.array.create [%147] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
              %149 = kgen.struct.create(%148) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
              %150 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
              %151 = pop.pointer.bitcast %150 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
              pop.store %149, %150 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
              hlcf.if %128 {
                hlcf.yield
              } else {
                %160 = pop.stack_allocation 2048 x scalar<ui8> align 1
                %161 = pop.pointer.bitcast %160 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<none>
                %162 = kgen.struct.create(%161, %index0) : !kgen.struct<(pointer<none>, index) memoryOnly>
                %163 = kgen.call @"std::collections::string::string::String::write_to[::Writer](::String,$0&),writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%136, %162) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
                %164 = kgen.call @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%124#1, %163) : (index, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
                %165 = kgen.call @"std::collections::string::string::String::write_to[::Writer](::String,$0&),writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%140, %164) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
                %166 = kgen.call @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%idx20, %165) : (index, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
                %167 = kgen.struct.extract %166[0] : <(pointer<none>, index) memoryOnly>
                %168 = kgen.struct.extract %166[1] : <(pointer<none>, index) memoryOnly>
                %169 = index.add %168, %index1
                %170 = index.cmp sgt(%169, %index2048)
                hlcf.if %170 {
                  kgen.call @"std::io::io::_printf[KGENParamList[::AnyType],::StringSlice[::Bool(False), StaticConstantOrigin, *?],*::AnyType,LITImmutOrigin,::Origin[::Bool(False), $3]](*$0,file:::FileDescriptor),types.values`=[],fmt={ #interp.memref<{[(#interp.memory_handle<16, \22HEAP_BUFFER_BYTES exceeded, increase with: `mojo -D HEAP_BUFFER_BYTES=4096`\\0A\\00\22 string>, const_global, [], [])], []}, 0, 0>, 76 }"(%index1) : (index) -> ()
                  llvm.intr.trap
                  hlcf.loop "_loop_0" {
                    hlcf.continue "_loop_0"
                  }
                  kgen.unreachable
                } else {
                  hlcf.yield
                }
                %171 = pop.pointer.bitcast %167 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                %172 = pop.offset %171[%168] : !kgen.pointer<scalar<ui8>>
                pop.store %simd_9, %172 : !kgen.pointer<scalar<ui8>>
                %173 = pop.load %150 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                %174 = kgen.struct.extract %173[0] : <(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                %175 = pop.array.get %174[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                %176 = pop.array.create [%175] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                %177 = kgen.struct.create(%176) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                %178 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                pop.store %177, %178 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                %179 = pop.pointer.bitcast %178 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
                %180 = pop.load %179 : !kgen.pointer<index>
                %181 = index.cmp eq(%180, %index-1)
                %182 = pop.select %181, %index0, %index-1 : index
                %183 = index.cmp eq(%182, %index-1)
                %184 = hlcf.if %183 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
                  %185 = pop.load %150 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                  %186 = kgen.struct.extract %185[0] : <(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                  %187 = pop.array.get %186[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                  %188 = pop.array.create [%187] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                  %189 = kgen.struct.create(%188) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                  %190 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                  pop.store %189, %190 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                  %191 = pop.pointer.bitcast %190 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
                  %192 = pop.load %191 : !kgen.pointer<index>
                  %193 = index.cmp eq(%192, %index-1)
                  %194 = pop.select %193, %index0, %index-1 : index
                  %195 = index.cmp eq(%194, %index-1)
                  %196 = pop.xor %195, %0
                  hlcf.if %196 {
                    %198 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
                    pop.stack_alloc.lifetime.start(%198) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                    %199 = kgen.struct.gep %198[1] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %index192, %199 : !kgen.pointer<index>
                    %200 = kgen.struct.gep %198[0] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %30, %200 : !kgen.pointer<pointer<none>>
                    %201 = kgen.struct.gep %198[2] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %index2305843009213693952, %201 : !kgen.pointer<index>
                    %202 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                    %203 = pop.pointer.bitcast %202 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                    pop.store %18, %203 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                    %204 = pop.load %202 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                    %205 = pop.array.get %204[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                    %206 = pop.array.create [%205] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                    %207 = kgen.struct.create(%206) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                    %208 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                    %209 = pop.pointer.bitcast %208 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                    pop.store %207, %208 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                    %210 = pop.pointer.bitcast %208 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
                    %211 = pop.load %210 : !kgen.pointer<index>
                    %212 = index.cmp eq(%211, %index-1)
                    %213 = pop.select %212, %index0, %index-1 : index
                    %214 = index.cmp eq(%213, %index-1)
                    %215 = hlcf.if %214 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
                      %233 = pop.load %209 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                      hlcf.yield %233 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
                    } else {
                      hlcf.yield %22 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
                    }
                    %216 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
                    pop.stack_alloc.lifetime.start(%216) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                    %217 = kgen.struct.gep %216[1] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %index1, %217 : !kgen.pointer<index>
                    %218 = kgen.struct.gep %216[0] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %24, %218 : !kgen.pointer<pointer<none>>
                    %219 = kgen.struct.gep %216[2] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %index2305843009213693952, %219 : !kgen.pointer<index>
                    %220 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
                    pop.stack_alloc.lifetime.start(%220) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                    %221 = kgen.struct.gep %220[1] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %index2, %221 : !kgen.pointer<index>
                    %222 = kgen.struct.gep %220[0] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %26, %222 : !kgen.pointer<pointer<none>>
                    %223 = kgen.struct.gep %220[2] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %index2305843009213693952, %223 : !kgen.pointer<index>
                    kgen.call @"std::io::io::print[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?],flush:::Bool,file:::FileDescriptor$),Ts.values`=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::reflection::location::SourceLocation\22>>, struct<(index, index, struct<(pointer<none>, index)>)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]]"(%struct_4, %216, %215, %220, %198, %61, %struct, %0, %index1) : (!kgen.struct<(pointer<none>, index)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index)>, i1, index owned) -> ()
                    %224 = pop.load %219 : !kgen.pointer<index>
                    %225 = index.and %224, %index4611686018427387904
                    %226 = index.cmp ne(%225, %index0)
                    hlcf.if %226 {
                      %233 = pop.load %218 : !kgen.pointer<pointer<none>>
                      %234 = pop.pointer.bitcast %233 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                      %235 = pop.offset %234[%idx-8] : !kgen.pointer<scalar<ui8>>
                      %236 = pop.pointer.bitcast %235 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                      %237 = kgen.struct.gep %236[0] : <struct<(scalar<index>) memoryOnly>>
                      %238 = pop.atomic.rmw sub(%237, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                      %239 = pop.cmp eq(%238, %simd_28) : <1, index>
                      %240 = pop.cast_to_builtin %239 : !pop.scalar<bool> to i1
                      hlcf.if %240 {
                        pop.fence syncscope("") acquire
                        pop.aligned_free %235 : <scalar<ui8>>
                        hlcf.yield
                      } else {
                        hlcf.yield
                      }
                      hlcf.yield
                    } else {
                      hlcf.yield
                    }
                    %227 = pop.load %223 : !kgen.pointer<index>
                    %228 = index.and %227, %index4611686018427387904
                    %229 = index.cmp ne(%228, %index0)
                    hlcf.if %229 {
                      %233 = pop.load %222 : !kgen.pointer<pointer<none>>
                      %234 = pop.pointer.bitcast %233 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                      %235 = pop.offset %234[%idx-8] : !kgen.pointer<scalar<ui8>>
                      %236 = pop.pointer.bitcast %235 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                      %237 = kgen.struct.gep %236[0] : <struct<(scalar<index>) memoryOnly>>
                      %238 = pop.atomic.rmw sub(%237, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                      %239 = pop.cmp eq(%238, %simd_28) : <1, index>
                      %240 = pop.cast_to_builtin %239 : !pop.scalar<bool> to i1
                      hlcf.if %240 {
                        pop.fence syncscope("") acquire
                        pop.aligned_free %235 : <scalar<ui8>>
                        hlcf.yield
                      } else {
                        hlcf.yield
                      }
                      hlcf.yield
                    } else {
                      hlcf.yield
                    }
                    %230 = pop.load %201 : !kgen.pointer<index>
                    %231 = index.and %230, %index4611686018427387904
                    %232 = index.cmp ne(%231, %index0)
                    hlcf.if %232 {
                      %233 = pop.load %200 : !kgen.pointer<pointer<none>>
                      %234 = pop.pointer.bitcast %233 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                      %235 = pop.offset %234[%idx-8] : !kgen.pointer<scalar<ui8>>
                      %236 = pop.pointer.bitcast %235 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                      %237 = kgen.struct.gep %236[0] : <struct<(scalar<index>) memoryOnly>>
                      %238 = pop.atomic.rmw sub(%237, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                      %239 = pop.cmp eq(%238, %simd_28) : <1, index>
                      %240 = pop.cast_to_builtin %239 : !pop.scalar<bool> to i1
                      hlcf.if %240 {
                        pop.fence syncscope("") acquire
                        pop.aligned_free %235 : <scalar<ui8>>
                        hlcf.yield
                      } else {
                        hlcf.yield
                      }
                      hlcf.yield
                    } else {
                      hlcf.yield
                    }
                    llvm.intr.trap
                    hlcf.loop "_loop_0" {
                      hlcf.continue "_loop_0"
                    }
                    kgen.unreachable
                  } else {
                    hlcf.yield
                  }
                  %197 = pop.load %151 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                  hlcf.yield %197 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
                } else {
                  hlcf.yield %35 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
                }
                kgen.call @"std::builtin::debug_assert::_debug_assert_msg[LITImmutOrigin,::Origin[::Bool(False), $0]](::UnsafePointer[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1, ::AddressSpace(::Int(0))],::Int,::SourceLocation)"(%167, %184) : (!kgen.pointer<none>, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>) -> ()
                hlcf.yield
              }
              %152 = pop.load %139 : !kgen.pointer<index>
              %153 = index.and %152, %index4611686018427387904
              %154 = index.cmp ne(%153, %index0)
              hlcf.if %154 {
                %160 = pop.load %138 : !kgen.pointer<pointer<none>>
                %161 = pop.pointer.bitcast %160 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                %162 = pop.offset %161[%idx-8] : !kgen.pointer<scalar<ui8>>
                %163 = pop.pointer.bitcast %162 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                %164 = kgen.struct.gep %163[0] : <struct<(scalar<index>) memoryOnly>>
                %165 = pop.atomic.rmw sub(%164, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                %166 = pop.cmp eq(%165, %simd_28) : <1, index>
                %167 = pop.cast_to_builtin %166 : !pop.scalar<bool> to i1
                hlcf.if %167 {
                  pop.fence syncscope("") acquire
                  pop.aligned_free %162 : <scalar<ui8>>
                  hlcf.yield
                } else {
                  hlcf.yield
                }
                hlcf.yield
              } else {
                hlcf.yield
              }
              %155 = pop.load %143 : !kgen.pointer<index>
              %156 = index.and %155, %index4611686018427387904
              %157 = index.cmp ne(%156, %index0)
              hlcf.if %157 {
                %160 = pop.load %142 : !kgen.pointer<pointer<none>>
                %161 = pop.pointer.bitcast %160 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                %162 = pop.offset %161[%idx-8] : !kgen.pointer<scalar<ui8>>
                %163 = pop.pointer.bitcast %162 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                %164 = kgen.struct.gep %163[0] : <struct<(scalar<index>) memoryOnly>>
                %165 = pop.atomic.rmw sub(%164, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                %166 = pop.cmp eq(%165, %simd_28) : <1, index>
                %167 = pop.cast_to_builtin %166 : !pop.scalar<bool> to i1
                hlcf.if %167 {
                  pop.fence syncscope("") acquire
                  pop.aligned_free %162 : <scalar<ui8>>
                  hlcf.yield
                } else {
                  hlcf.yield
                }
                hlcf.yield
              } else {
                hlcf.yield
              }
              %158 = pop.array.gep %89[%124#1] : <array<21, scalar<ui8>>>
              %159 = pop.load %158 : !kgen.pointer<scalar<ui8>>
              kgen.call @"std::builtin::simd::SIMD::write_to[::Writer](::SIMD[$0, $1],$2&),dtype=ui8,size=1,writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>]"(%159, %arg0) : (!pop.scalar<ui8>, !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut) -> ()
              hlcf.continue "_loop_5" %124#0 : index
            }
            %112 = hlcf.if %105 -> index {
              %118 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
              pop.stack_alloc.lifetime.start(%118) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
              %119 = kgen.struct.gep %118[1] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %index2, %119 : !kgen.pointer<index>
              %120 = kgen.struct.gep %118[0] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %48, %120 : !kgen.pointer<pointer<none>>
              %121 = kgen.struct.gep %118[2] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %index2305843009213693952, %121 : !kgen.pointer<index>
              %122 = pop.pointer.bitcast %118 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
              %123 = pop.load %121 : !kgen.pointer<index>
              %124 = index.and %123, %index-9223372036854775808
              %125 = index.cmp ne(%124, %index0)
              %126 = hlcf.if %125 -> !kgen.pointer<none> {
                hlcf.yield %122 : !kgen.pointer<none>
              } else {
                %136 = pop.load %120 : !kgen.pointer<pointer<none>>
                hlcf.yield %136 : !kgen.pointer<none>
              }
              %127 = pop.load %121 : !kgen.pointer<index>
              %128 = index.and %127, %index-9223372036854775808
              %129 = index.cmp ne(%128, %index0)
              %130 = hlcf.if %129 -> index {
                %136 = pop.load %121 : !kgen.pointer<index>
                %137 = index.and %136, %index2233785415175766016
                %138 = index.shrs %137, %index56
                hlcf.yield %138 : index
              } else {
                %136 = pop.load %119 : !kgen.pointer<index>
                hlcf.yield %136 : index
              }
              %131 = kgen.struct.create(%126, %130) : !kgen.struct<(pointer<none>, index)>
              kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg0, %131) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
              %132 = pop.load %121 : !kgen.pointer<index>
              %133 = index.and %132, %index4611686018427387904
              %134 = index.cmp ne(%133, %index0)
              hlcf.if %134 {
                %136 = pop.load %120 : !kgen.pointer<pointer<none>>
                %137 = pop.pointer.bitcast %136 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                %138 = pop.offset %137[%idx-8] : !kgen.pointer<scalar<ui8>>
                %139 = pop.pointer.bitcast %138 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                %140 = kgen.struct.gep %139[0] : <struct<(scalar<index>) memoryOnly>>
                %141 = pop.atomic.rmw sub(%140, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                %142 = pop.cmp eq(%141, %simd_28) : <1, index>
                %143 = pop.cast_to_builtin %142 : !pop.scalar<bool> to i1
                hlcf.if %143 {
                  pop.fence syncscope("") acquire
                  pop.aligned_free %138 : <scalar<ui8>>
                  hlcf.yield
                } else {
                  hlcf.yield
                }
                hlcf.yield
              } else {
                hlcf.yield
              }
              pop.stack_alloc.lifetime.end(%118) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
              %135 = index.mul %90#1, %index-1
              hlcf.yield %135 : index
            } else {
              %118 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
              pop.stack_alloc.lifetime.start(%118) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
              %119 = kgen.struct.gep %118[1] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %index2, %119 : !kgen.pointer<index>
              %120 = kgen.struct.gep %118[0] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %46, %120 : !kgen.pointer<pointer<none>>
              %121 = kgen.struct.gep %118[2] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %index2305843009213693952, %121 : !kgen.pointer<index>
              %122 = pop.pointer.bitcast %118 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
              %123 = pop.load %121 : !kgen.pointer<index>
              %124 = index.and %123, %index-9223372036854775808
              %125 = index.cmp ne(%124, %index0)
              %126 = hlcf.if %125 -> !kgen.pointer<none> {
                hlcf.yield %122 : !kgen.pointer<none>
              } else {
                %135 = pop.load %120 : !kgen.pointer<pointer<none>>
                hlcf.yield %135 : !kgen.pointer<none>
              }
              %127 = pop.load %121 : !kgen.pointer<index>
              %128 = index.and %127, %index-9223372036854775808
              %129 = index.cmp ne(%128, %index0)
              %130 = hlcf.if %129 -> index {
                %135 = pop.load %121 : !kgen.pointer<index>
                %136 = index.and %135, %index2233785415175766016
                %137 = index.shrs %136, %index56
                hlcf.yield %137 : index
              } else {
                %135 = pop.load %119 : !kgen.pointer<index>
                hlcf.yield %135 : index
              }
              %131 = kgen.struct.create(%126, %130) : !kgen.struct<(pointer<none>, index)>
              kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg0, %131) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
              %132 = pop.load %121 : !kgen.pointer<index>
              %133 = index.and %132, %index4611686018427387904
              %134 = index.cmp ne(%133, %index0)
              hlcf.if %134 {
                %135 = pop.load %120 : !kgen.pointer<pointer<none>>
                %136 = pop.pointer.bitcast %135 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                %137 = pop.offset %136[%idx-8] : !kgen.pointer<scalar<ui8>>
                %138 = pop.pointer.bitcast %137 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                %139 = kgen.struct.gep %138[0] : <struct<(scalar<index>) memoryOnly>>
                %140 = pop.atomic.rmw sub(%139, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                %141 = pop.cmp eq(%140, %simd_28) : <1, index>
                %142 = pop.cast_to_builtin %141 : !pop.scalar<bool> to i1
                hlcf.if %142 {
                  pop.fence syncscope("") acquire
                  pop.aligned_free %137 : <scalar<ui8>>
                  hlcf.yield
                } else {
                  hlcf.yield
                }
                hlcf.yield
              } else {
                hlcf.yield
              }
              pop.stack_alloc.lifetime.end(%118) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
              hlcf.yield %90#1 : index
            }
            %113 = index.cmp slt(%112, %index10)
            hlcf.if %113 {
              %118 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
              pop.stack_alloc.lifetime.start(%118) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
              %119 = kgen.struct.gep %118[1] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %index1, %119 : !kgen.pointer<index>
              %120 = kgen.struct.gep %118[0] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %44, %120 : !kgen.pointer<pointer<none>>
              %121 = kgen.struct.gep %118[2] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %index2305843009213693952, %121 : !kgen.pointer<index>
              %122 = pop.pointer.bitcast %118 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
              %123 = pop.load %121 : !kgen.pointer<index>
              %124 = index.and %123, %index-9223372036854775808
              %125 = index.cmp ne(%124, %index0)
              %126 = hlcf.if %125 -> !kgen.pointer<none> {
                hlcf.yield %122 : !kgen.pointer<none>
              } else {
                %135 = pop.load %120 : !kgen.pointer<pointer<none>>
                hlcf.yield %135 : !kgen.pointer<none>
              }
              %127 = pop.load %121 : !kgen.pointer<index>
              %128 = index.and %127, %index-9223372036854775808
              %129 = index.cmp ne(%128, %index0)
              %130 = hlcf.if %129 -> index {
                %135 = pop.load %121 : !kgen.pointer<index>
                %136 = index.and %135, %index2233785415175766016
                %137 = index.shrs %136, %index56
                hlcf.yield %137 : index
              } else {
                %135 = pop.load %119 : !kgen.pointer<index>
                hlcf.yield %135 : index
              }
              %131 = kgen.struct.create(%126, %130) : !kgen.struct<(pointer<none>, index)>
              kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg0, %131) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
              %132 = pop.load %121 : !kgen.pointer<index>
              %133 = index.and %132, %index4611686018427387904
              %134 = index.cmp ne(%133, %index0)
              hlcf.if %134 {
                %135 = pop.load %120 : !kgen.pointer<pointer<none>>
                %136 = pop.pointer.bitcast %135 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                %137 = pop.offset %136[%idx-8] : !kgen.pointer<scalar<ui8>>
                %138 = pop.pointer.bitcast %137 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                %139 = kgen.struct.gep %138[0] : <struct<(scalar<index>) memoryOnly>>
                %140 = pop.atomic.rmw sub(%139, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                %141 = pop.cmp eq(%140, %simd_28) : <1, index>
                %142 = pop.cast_to_builtin %141 : !pop.scalar<bool> to i1
                hlcf.if %142 {
                  pop.fence syncscope("") acquire
                  pop.aligned_free %137 : <scalar<ui8>>
                  hlcf.yield
                } else {
                  hlcf.yield
                }
                hlcf.yield
              } else {
                hlcf.yield
              }
              pop.stack_alloc.lifetime.end(%118) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
              hlcf.yield
            } else {
              hlcf.yield
            }
            %114 = pop.stack_allocation 1 x array<10, scalar<ui8>> align 1 marked
            pop.stack_alloc.lifetime.start(%114) : !kgen.pointer<array<10, scalar<ui8>>>
            %115 = hlcf.loop "_loop_6" (%arg2 = %index0 : index, %arg3 = %112 : index) -> index {
              %118 = index.cmp sgt(%arg3, %index0)
              hlcf.if %118 {
                hlcf.yield
              } else {
                hlcf.break "_loop_6" %arg2 : index
              }
              %119 = pop.cast_from_builtin %arg2 : index to !pop.scalar<index>
              %120 = pop.cast %119 : !pop.scalar<index> to !pop.scalar<uindex>
              %121 = pop.cmp lt(%120, %simd_0) : <1, uindex>
              %122 = pop.cast_to_builtin %121 : !pop.scalar<bool> to i1
              %123 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
              pop.stack_alloc.lifetime.start(%123) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
              pop.store %array, %123 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
              %124 = pop.pointer.bitcast %123 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<index>
              %125 = pop.load %124 : !kgen.pointer<index>
              %126 = index.cmp eq(%125, %index-1)
              %127 = pop.select %126, %index0, %index-1 : index
              pop.stack_alloc.lifetime.end(%123) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
              %128 = index.cmp eq(%127, %index-1)
              %129 = hlcf.if %128 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
                %165 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
                pop.stack_alloc.lifetime.start(%165) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                pop.store %array, %165 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                %166 = pop.pointer.bitcast %165 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                %167 = pop.load %166 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                pop.stack_alloc.lifetime.end(%165) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                hlcf.yield %167 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
              } else {
                hlcf.yield %16 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
              }
              %130 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
              pop.stack_alloc.lifetime.start(%130) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
              %131 = kgen.struct.gep %130[1] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %index6, %131 : !kgen.pointer<index>
              %132 = kgen.struct.gep %130[0] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %34, %132 : !kgen.pointer<pointer<none>>
              %133 = kgen.struct.gep %130[2] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %index2305843009213693952, %133 : !kgen.pointer<index>
              %134 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
              pop.stack_alloc.lifetime.start(%134) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
              %135 = kgen.struct.gep %134[1] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %index39, %135 : !kgen.pointer<index>
              %136 = kgen.struct.gep %134[0] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %32, %136 : !kgen.pointer<pointer<none>>
              %137 = kgen.struct.gep %134[2] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %index2305843009213693952, %137 : !kgen.pointer<index>
              %138 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
              %139 = pop.pointer.bitcast %138 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
              pop.store %129, %139 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
              %140 = pop.load %138 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
              %141 = pop.array.get %140[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
              %142 = pop.array.create [%141] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
              %143 = kgen.struct.create(%142) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
              %144 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
              %145 = pop.pointer.bitcast %144 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
              pop.store %143, %144 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
              hlcf.if %122 {
                hlcf.yield
              } else {
                %165 = pop.stack_allocation 2048 x scalar<ui8> align 1
                %166 = pop.pointer.bitcast %165 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<none>
                %167 = kgen.struct.create(%166, %index0) : !kgen.struct<(pointer<none>, index) memoryOnly>
                %168 = kgen.call @"std::collections::string::string::String::write_to[::Writer](::String,$0&),writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%130, %167) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
                %169 = kgen.call @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%arg2, %168) : (index, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
                %170 = kgen.call @"std::collections::string::string::String::write_to[::Writer](::String,$0&),writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%134, %169) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
                %171 = kgen.call @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%idx9, %170) : (index, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
                %172 = kgen.struct.extract %171[0] : <(pointer<none>, index) memoryOnly>
                %173 = kgen.struct.extract %171[1] : <(pointer<none>, index) memoryOnly>
                %174 = index.add %173, %index1
                %175 = index.cmp sgt(%174, %index2048)
                hlcf.if %175 {
                  kgen.call @"std::io::io::_printf[KGENParamList[::AnyType],::StringSlice[::Bool(False), StaticConstantOrigin, *?],*::AnyType,LITImmutOrigin,::Origin[::Bool(False), $3]](*$0,file:::FileDescriptor),types.values`=[],fmt={ #interp.memref<{[(#interp.memory_handle<16, \22HEAP_BUFFER_BYTES exceeded, increase with: `mojo -D HEAP_BUFFER_BYTES=4096`\\0A\\00\22 string>, const_global, [], [])], []}, 0, 0>, 76 }"(%index1) : (index) -> ()
                  llvm.intr.trap
                  hlcf.loop "_loop_0" {
                    hlcf.continue "_loop_0"
                  }
                  kgen.unreachable
                } else {
                  hlcf.yield
                }
                %176 = pop.pointer.bitcast %172 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                %177 = pop.offset %176[%173] : !kgen.pointer<scalar<ui8>>
                pop.store %simd_9, %177 : !kgen.pointer<scalar<ui8>>
                %178 = pop.load %144 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                %179 = kgen.struct.extract %178[0] : <(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                %180 = pop.array.get %179[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                %181 = pop.array.create [%180] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                %182 = kgen.struct.create(%181) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                %183 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                pop.store %182, %183 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                %184 = pop.pointer.bitcast %183 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
                %185 = pop.load %184 : !kgen.pointer<index>
                %186 = index.cmp eq(%185, %index-1)
                %187 = pop.select %186, %index0, %index-1 : index
                %188 = index.cmp eq(%187, %index-1)
                %189 = hlcf.if %188 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
                  %190 = pop.load %144 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                  %191 = kgen.struct.extract %190[0] : <(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                  %192 = pop.array.get %191[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                  %193 = pop.array.create [%192] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                  %194 = kgen.struct.create(%193) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                  %195 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                  pop.store %194, %195 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                  %196 = pop.pointer.bitcast %195 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
                  %197 = pop.load %196 : !kgen.pointer<index>
                  %198 = index.cmp eq(%197, %index-1)
                  %199 = pop.select %198, %index0, %index-1 : index
                  %200 = index.cmp eq(%199, %index-1)
                  %201 = pop.xor %200, %0
                  hlcf.if %201 {
                    %203 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
                    pop.stack_alloc.lifetime.start(%203) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                    %204 = kgen.struct.gep %203[1] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %index192, %204 : !kgen.pointer<index>
                    %205 = kgen.struct.gep %203[0] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %30, %205 : !kgen.pointer<pointer<none>>
                    %206 = kgen.struct.gep %203[2] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %index2305843009213693952, %206 : !kgen.pointer<index>
                    %207 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                    %208 = pop.pointer.bitcast %207 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                    pop.store %18, %208 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                    %209 = pop.load %207 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                    %210 = pop.array.get %209[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                    %211 = pop.array.create [%210] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                    %212 = kgen.struct.create(%211) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                    %213 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                    %214 = pop.pointer.bitcast %213 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                    pop.store %212, %213 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                    %215 = pop.pointer.bitcast %213 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
                    %216 = pop.load %215 : !kgen.pointer<index>
                    %217 = index.cmp eq(%216, %index-1)
                    %218 = pop.select %217, %index0, %index-1 : index
                    %219 = index.cmp eq(%218, %index-1)
                    %220 = hlcf.if %219 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
                      %238 = pop.load %214 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                      hlcf.yield %238 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
                    } else {
                      hlcf.yield %22 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
                    }
                    %221 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
                    pop.stack_alloc.lifetime.start(%221) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                    %222 = kgen.struct.gep %221[1] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %index1, %222 : !kgen.pointer<index>
                    %223 = kgen.struct.gep %221[0] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %24, %223 : !kgen.pointer<pointer<none>>
                    %224 = kgen.struct.gep %221[2] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %index2305843009213693952, %224 : !kgen.pointer<index>
                    %225 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
                    pop.stack_alloc.lifetime.start(%225) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                    %226 = kgen.struct.gep %225[1] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %index2, %226 : !kgen.pointer<index>
                    %227 = kgen.struct.gep %225[0] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %26, %227 : !kgen.pointer<pointer<none>>
                    %228 = kgen.struct.gep %225[2] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %index2305843009213693952, %228 : !kgen.pointer<index>
                    kgen.call @"std::io::io::print[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?],flush:::Bool,file:::FileDescriptor$),Ts.values`=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::reflection::location::SourceLocation\22>>, struct<(index, index, struct<(pointer<none>, index)>)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]]"(%struct_4, %221, %220, %225, %203, %61, %struct, %0, %index1) : (!kgen.struct<(pointer<none>, index)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index)>, i1, index owned) -> ()
                    %229 = pop.load %224 : !kgen.pointer<index>
                    %230 = index.and %229, %index4611686018427387904
                    %231 = index.cmp ne(%230, %index0)
                    hlcf.if %231 {
                      %238 = pop.load %223 : !kgen.pointer<pointer<none>>
                      %239 = pop.pointer.bitcast %238 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                      %240 = pop.offset %239[%idx-8] : !kgen.pointer<scalar<ui8>>
                      %241 = pop.pointer.bitcast %240 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                      %242 = kgen.struct.gep %241[0] : <struct<(scalar<index>) memoryOnly>>
                      %243 = pop.atomic.rmw sub(%242, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                      %244 = pop.cmp eq(%243, %simd_28) : <1, index>
                      %245 = pop.cast_to_builtin %244 : !pop.scalar<bool> to i1
                      hlcf.if %245 {
                        pop.fence syncscope("") acquire
                        pop.aligned_free %240 : <scalar<ui8>>
                        hlcf.yield
                      } else {
                        hlcf.yield
                      }
                      hlcf.yield
                    } else {
                      hlcf.yield
                    }
                    %232 = pop.load %228 : !kgen.pointer<index>
                    %233 = index.and %232, %index4611686018427387904
                    %234 = index.cmp ne(%233, %index0)
                    hlcf.if %234 {
                      %238 = pop.load %227 : !kgen.pointer<pointer<none>>
                      %239 = pop.pointer.bitcast %238 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                      %240 = pop.offset %239[%idx-8] : !kgen.pointer<scalar<ui8>>
                      %241 = pop.pointer.bitcast %240 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                      %242 = kgen.struct.gep %241[0] : <struct<(scalar<index>) memoryOnly>>
                      %243 = pop.atomic.rmw sub(%242, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                      %244 = pop.cmp eq(%243, %simd_28) : <1, index>
                      %245 = pop.cast_to_builtin %244 : !pop.scalar<bool> to i1
                      hlcf.if %245 {
                        pop.fence syncscope("") acquire
                        pop.aligned_free %240 : <scalar<ui8>>
                        hlcf.yield
                      } else {
                        hlcf.yield
                      }
                      hlcf.yield
                    } else {
                      hlcf.yield
                    }
                    %235 = pop.load %206 : !kgen.pointer<index>
                    %236 = index.and %235, %index4611686018427387904
                    %237 = index.cmp ne(%236, %index0)
                    hlcf.if %237 {
                      %238 = pop.load %205 : !kgen.pointer<pointer<none>>
                      %239 = pop.pointer.bitcast %238 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                      %240 = pop.offset %239[%idx-8] : !kgen.pointer<scalar<ui8>>
                      %241 = pop.pointer.bitcast %240 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                      %242 = kgen.struct.gep %241[0] : <struct<(scalar<index>) memoryOnly>>
                      %243 = pop.atomic.rmw sub(%242, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                      %244 = pop.cmp eq(%243, %simd_28) : <1, index>
                      %245 = pop.cast_to_builtin %244 : !pop.scalar<bool> to i1
                      hlcf.if %245 {
                        pop.fence syncscope("") acquire
                        pop.aligned_free %240 : <scalar<ui8>>
                        hlcf.yield
                      } else {
                        hlcf.yield
                      }
                      hlcf.yield
                    } else {
                      hlcf.yield
                    }
                    llvm.intr.trap
                    hlcf.loop "_loop_0" {
                      hlcf.continue "_loop_0"
                    }
                    kgen.unreachable
                  } else {
                    hlcf.yield
                  }
                  %202 = pop.load %145 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                  hlcf.yield %202 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
                } else {
                  hlcf.yield %35 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
                }
                kgen.call @"std::builtin::debug_assert::_debug_assert_msg[LITImmutOrigin,::Origin[::Bool(False), $0]](::UnsafePointer[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1, ::AddressSpace(::Int(0))],::Int,::SourceLocation)"(%172, %189) : (!kgen.pointer<none>, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>) -> ()
                hlcf.yield
              }
              %146 = pop.load %133 : !kgen.pointer<index>
              %147 = index.and %146, %index4611686018427387904
              %148 = index.cmp ne(%147, %index0)
              hlcf.if %148 {
                %165 = pop.load %132 : !kgen.pointer<pointer<none>>
                %166 = pop.pointer.bitcast %165 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                %167 = pop.offset %166[%idx-8] : !kgen.pointer<scalar<ui8>>
                %168 = pop.pointer.bitcast %167 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                %169 = kgen.struct.gep %168[0] : <struct<(scalar<index>) memoryOnly>>
                %170 = pop.atomic.rmw sub(%169, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                %171 = pop.cmp eq(%170, %simd_28) : <1, index>
                %172 = pop.cast_to_builtin %171 : !pop.scalar<bool> to i1
                hlcf.if %172 {
                  pop.fence syncscope("") acquire
                  pop.aligned_free %167 : <scalar<ui8>>
                  hlcf.yield
                } else {
                  hlcf.yield
                }
                hlcf.yield
              } else {
                hlcf.yield
              }
              %149 = pop.load %137 : !kgen.pointer<index>
              %150 = index.and %149, %index4611686018427387904
              %151 = index.cmp ne(%150, %index0)
              hlcf.if %151 {
                %165 = pop.load %136 : !kgen.pointer<pointer<none>>
                %166 = pop.pointer.bitcast %165 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                %167 = pop.offset %166[%idx-8] : !kgen.pointer<scalar<ui8>>
                %168 = pop.pointer.bitcast %167 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                %169 = kgen.struct.gep %168[0] : <struct<(scalar<index>) memoryOnly>>
                %170 = pop.atomic.rmw sub(%169, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                %171 = pop.cmp eq(%170, %simd_28) : <1, index>
                %172 = pop.cast_to_builtin %171 : !pop.scalar<bool> to i1
                hlcf.if %172 {
                  pop.fence syncscope("") acquire
                  pop.aligned_free %167 : <scalar<ui8>>
                  hlcf.yield
                } else {
                  hlcf.yield
                }
                hlcf.yield
              } else {
                hlcf.yield
              }
              %152 = pop.array.gep %114[%arg2] : <array<10, scalar<ui8>>>
              %153 = index.rems %arg3, %index10
              %154 = index.cmp slt(%arg3, %index0)
              %155 = index.cmp ne(%153, %index0)
              %156 = pop.and %154, %155
              %157 = index.add %153, %index10
              %158 = pop.select %156, %157, %153 : index
              %159 = pop.cast_from_builtin %158 : index to !pop.scalar<index>
              %160 = pop.cast %159 : !pop.scalar<index> to !pop.scalar<ui8>
              pop.store %160, %152 : !kgen.pointer<scalar<ui8>>
              %161 = index.divs %arg3, %index10
              %162 = index.sub %161, %index1
              %163 = pop.select %156, %162, %161 : index
              %164 = index.add %arg2, %index1
              hlcf.continue "_loop_6" %164, %163 : index, index
            }
            %116 = index.maxs %115, %index0
            %117 = index.sub %116, %index1
            hlcf.loop "_loop_7" (%arg2 = %117 : index) {
              %118 = index.add %arg2, %index-1
              %119 = index.sub %arg2, %index-1
              %120 = index.cmp sgt(%arg2, %index-1)
              %121 = pop.select %120, %119, %index0 : index
              %122 = index.cmp sle(%121, %index0)
              %123 = pop.select %122, %arg2, %118 : index
              %124:2 = lit.try "try4" -> index, index {
                hlcf.if %122 {
                  pop.stack_alloc.lifetime.end(%114) : !kgen.pointer<array<10, scalar<ui8>>>
                  lit.try.raise "try4" %123, %arg2 : index, index
                } else {
                  hlcf.yield
                }
                lit.try.yield %123, %arg2 : index, index
              } except (%arg3: index, %arg4: index) {
                hlcf.break "_loop_7"
              } else (%arg3: index, %arg4: index) {
                lit.try.yield %arg3, %arg4 : index, index
              }
              %125 = pop.cast_from_builtin %124#1 : index to !pop.scalar<index>
              %126 = pop.cast %125 : !pop.scalar<index> to !pop.scalar<uindex>
              %127 = pop.cmp lt(%126, %simd_0) : <1, uindex>
              %128 = pop.cast_to_builtin %127 : !pop.scalar<bool> to i1
              %129 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
              pop.stack_alloc.lifetime.start(%129) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
              pop.store %array, %129 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
              %130 = pop.pointer.bitcast %129 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<index>
              %131 = pop.load %130 : !kgen.pointer<index>
              %132 = index.cmp eq(%131, %index-1)
              %133 = pop.select %132, %index0, %index-1 : index
              pop.stack_alloc.lifetime.end(%129) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
              %134 = index.cmp eq(%133, %index-1)
              %135 = hlcf.if %134 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
                %160 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
                pop.stack_alloc.lifetime.start(%160) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                pop.store %array, %160 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                %161 = pop.pointer.bitcast %160 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                %162 = pop.load %161 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                pop.stack_alloc.lifetime.end(%160) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                hlcf.yield %162 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
              } else {
                hlcf.yield %7 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
              }
              %136 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
              pop.stack_alloc.lifetime.start(%136) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
              %137 = kgen.struct.gep %136[1] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %index6, %137 : !kgen.pointer<index>
              %138 = kgen.struct.gep %136[0] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %34, %138 : !kgen.pointer<pointer<none>>
              %139 = kgen.struct.gep %136[2] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %index2305843009213693952, %139 : !kgen.pointer<index>
              %140 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
              pop.stack_alloc.lifetime.start(%140) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
              %141 = kgen.struct.gep %140[1] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %index39, %141 : !kgen.pointer<index>
              %142 = kgen.struct.gep %140[0] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %32, %142 : !kgen.pointer<pointer<none>>
              %143 = kgen.struct.gep %140[2] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %index2305843009213693952, %143 : !kgen.pointer<index>
              %144 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
              %145 = pop.pointer.bitcast %144 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
              pop.store %135, %145 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
              %146 = pop.load %144 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
              %147 = pop.array.get %146[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
              %148 = pop.array.create [%147] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
              %149 = kgen.struct.create(%148) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
              %150 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
              %151 = pop.pointer.bitcast %150 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
              pop.store %149, %150 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
              hlcf.if %128 {
                hlcf.yield
              } else {
                %160 = pop.stack_allocation 2048 x scalar<ui8> align 1
                %161 = pop.pointer.bitcast %160 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<none>
                %162 = kgen.struct.create(%161, %index0) : !kgen.struct<(pointer<none>, index) memoryOnly>
                %163 = kgen.call @"std::collections::string::string::String::write_to[::Writer](::String,$0&),writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%136, %162) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
                %164 = kgen.call @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%124#1, %163) : (index, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
                %165 = kgen.call @"std::collections::string::string::String::write_to[::Writer](::String,$0&),writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%140, %164) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
                %166 = kgen.call @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%idx9, %165) : (index, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
                %167 = kgen.struct.extract %166[0] : <(pointer<none>, index) memoryOnly>
                %168 = kgen.struct.extract %166[1] : <(pointer<none>, index) memoryOnly>
                %169 = index.add %168, %index1
                %170 = index.cmp sgt(%169, %index2048)
                hlcf.if %170 {
                  kgen.call @"std::io::io::_printf[KGENParamList[::AnyType],::StringSlice[::Bool(False), StaticConstantOrigin, *?],*::AnyType,LITImmutOrigin,::Origin[::Bool(False), $3]](*$0,file:::FileDescriptor),types.values`=[],fmt={ #interp.memref<{[(#interp.memory_handle<16, \22HEAP_BUFFER_BYTES exceeded, increase with: `mojo -D HEAP_BUFFER_BYTES=4096`\\0A\\00\22 string>, const_global, [], [])], []}, 0, 0>, 76 }"(%index1) : (index) -> ()
                  llvm.intr.trap
                  hlcf.loop "_loop_0" {
                    hlcf.continue "_loop_0"
                  }
                  kgen.unreachable
                } else {
                  hlcf.yield
                }
                %171 = pop.pointer.bitcast %167 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                %172 = pop.offset %171[%168] : !kgen.pointer<scalar<ui8>>
                pop.store %simd_9, %172 : !kgen.pointer<scalar<ui8>>
                %173 = pop.load %150 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                %174 = kgen.struct.extract %173[0] : <(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                %175 = pop.array.get %174[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                %176 = pop.array.create [%175] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                %177 = kgen.struct.create(%176) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                %178 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                pop.store %177, %178 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                %179 = pop.pointer.bitcast %178 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
                %180 = pop.load %179 : !kgen.pointer<index>
                %181 = index.cmp eq(%180, %index-1)
                %182 = pop.select %181, %index0, %index-1 : index
                %183 = index.cmp eq(%182, %index-1)
                %184 = hlcf.if %183 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
                  %185 = pop.load %150 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                  %186 = kgen.struct.extract %185[0] : <(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                  %187 = pop.array.get %186[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                  %188 = pop.array.create [%187] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                  %189 = kgen.struct.create(%188) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                  %190 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                  pop.store %189, %190 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                  %191 = pop.pointer.bitcast %190 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
                  %192 = pop.load %191 : !kgen.pointer<index>
                  %193 = index.cmp eq(%192, %index-1)
                  %194 = pop.select %193, %index0, %index-1 : index
                  %195 = index.cmp eq(%194, %index-1)
                  %196 = pop.xor %195, %0
                  hlcf.if %196 {
                    %198 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
                    pop.stack_alloc.lifetime.start(%198) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                    %199 = kgen.struct.gep %198[1] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %index192, %199 : !kgen.pointer<index>
                    %200 = kgen.struct.gep %198[0] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %30, %200 : !kgen.pointer<pointer<none>>
                    %201 = kgen.struct.gep %198[2] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %index2305843009213693952, %201 : !kgen.pointer<index>
                    %202 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                    %203 = pop.pointer.bitcast %202 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                    pop.store %18, %203 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                    %204 = pop.load %202 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                    %205 = pop.array.get %204[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                    %206 = pop.array.create [%205] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                    %207 = kgen.struct.create(%206) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                    %208 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                    %209 = pop.pointer.bitcast %208 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                    pop.store %207, %208 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                    %210 = pop.pointer.bitcast %208 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
                    %211 = pop.load %210 : !kgen.pointer<index>
                    %212 = index.cmp eq(%211, %index-1)
                    %213 = pop.select %212, %index0, %index-1 : index
                    %214 = index.cmp eq(%213, %index-1)
                    %215 = hlcf.if %214 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
                      %233 = pop.load %209 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                      hlcf.yield %233 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
                    } else {
                      hlcf.yield %22 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
                    }
                    %216 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
                    pop.stack_alloc.lifetime.start(%216) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                    %217 = kgen.struct.gep %216[1] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %index1, %217 : !kgen.pointer<index>
                    %218 = kgen.struct.gep %216[0] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %24, %218 : !kgen.pointer<pointer<none>>
                    %219 = kgen.struct.gep %216[2] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %index2305843009213693952, %219 : !kgen.pointer<index>
                    %220 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
                    pop.stack_alloc.lifetime.start(%220) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                    %221 = kgen.struct.gep %220[1] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %index2, %221 : !kgen.pointer<index>
                    %222 = kgen.struct.gep %220[0] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %26, %222 : !kgen.pointer<pointer<none>>
                    %223 = kgen.struct.gep %220[2] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %index2305843009213693952, %223 : !kgen.pointer<index>
                    kgen.call @"std::io::io::print[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?],flush:::Bool,file:::FileDescriptor$),Ts.values`=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::reflection::location::SourceLocation\22>>, struct<(index, index, struct<(pointer<none>, index)>)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]]"(%struct_4, %216, %215, %220, %198, %61, %struct, %0, %index1) : (!kgen.struct<(pointer<none>, index)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index)>, i1, index owned) -> ()
                    %224 = pop.load %219 : !kgen.pointer<index>
                    %225 = index.and %224, %index4611686018427387904
                    %226 = index.cmp ne(%225, %index0)
                    hlcf.if %226 {
                      %233 = pop.load %218 : !kgen.pointer<pointer<none>>
                      %234 = pop.pointer.bitcast %233 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                      %235 = pop.offset %234[%idx-8] : !kgen.pointer<scalar<ui8>>
                      %236 = pop.pointer.bitcast %235 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                      %237 = kgen.struct.gep %236[0] : <struct<(scalar<index>) memoryOnly>>
                      %238 = pop.atomic.rmw sub(%237, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                      %239 = pop.cmp eq(%238, %simd_28) : <1, index>
                      %240 = pop.cast_to_builtin %239 : !pop.scalar<bool> to i1
                      hlcf.if %240 {
                        pop.fence syncscope("") acquire
                        pop.aligned_free %235 : <scalar<ui8>>
                        hlcf.yield
                      } else {
                        hlcf.yield
                      }
                      hlcf.yield
                    } else {
                      hlcf.yield
                    }
                    %227 = pop.load %223 : !kgen.pointer<index>
                    %228 = index.and %227, %index4611686018427387904
                    %229 = index.cmp ne(%228, %index0)
                    hlcf.if %229 {
                      %233 = pop.load %222 : !kgen.pointer<pointer<none>>
                      %234 = pop.pointer.bitcast %233 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                      %235 = pop.offset %234[%idx-8] : !kgen.pointer<scalar<ui8>>
                      %236 = pop.pointer.bitcast %235 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                      %237 = kgen.struct.gep %236[0] : <struct<(scalar<index>) memoryOnly>>
                      %238 = pop.atomic.rmw sub(%237, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                      %239 = pop.cmp eq(%238, %simd_28) : <1, index>
                      %240 = pop.cast_to_builtin %239 : !pop.scalar<bool> to i1
                      hlcf.if %240 {
                        pop.fence syncscope("") acquire
                        pop.aligned_free %235 : <scalar<ui8>>
                        hlcf.yield
                      } else {
                        hlcf.yield
                      }
                      hlcf.yield
                    } else {
                      hlcf.yield
                    }
                    %230 = pop.load %201 : !kgen.pointer<index>
                    %231 = index.and %230, %index4611686018427387904
                    %232 = index.cmp ne(%231, %index0)
                    hlcf.if %232 {
                      %233 = pop.load %200 : !kgen.pointer<pointer<none>>
                      %234 = pop.pointer.bitcast %233 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                      %235 = pop.offset %234[%idx-8] : !kgen.pointer<scalar<ui8>>
                      %236 = pop.pointer.bitcast %235 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                      %237 = kgen.struct.gep %236[0] : <struct<(scalar<index>) memoryOnly>>
                      %238 = pop.atomic.rmw sub(%237, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                      %239 = pop.cmp eq(%238, %simd_28) : <1, index>
                      %240 = pop.cast_to_builtin %239 : !pop.scalar<bool> to i1
                      hlcf.if %240 {
                        pop.fence syncscope("") acquire
                        pop.aligned_free %235 : <scalar<ui8>>
                        hlcf.yield
                      } else {
                        hlcf.yield
                      }
                      hlcf.yield
                    } else {
                      hlcf.yield
                    }
                    llvm.intr.trap
                    hlcf.loop "_loop_0" {
                      hlcf.continue "_loop_0"
                    }
                    kgen.unreachable
                  } else {
                    hlcf.yield
                  }
                  %197 = pop.load %151 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                  hlcf.yield %197 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
                } else {
                  hlcf.yield %35 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
                }
                kgen.call @"std::builtin::debug_assert::_debug_assert_msg[LITImmutOrigin,::Origin[::Bool(False), $0]](::UnsafePointer[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1, ::AddressSpace(::Int(0))],::Int,::SourceLocation)"(%167, %184) : (!kgen.pointer<none>, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>) -> ()
                hlcf.yield
              }
              %152 = pop.load %139 : !kgen.pointer<index>
              %153 = index.and %152, %index4611686018427387904
              %154 = index.cmp ne(%153, %index0)
              hlcf.if %154 {
                %160 = pop.load %138 : !kgen.pointer<pointer<none>>
                %161 = pop.pointer.bitcast %160 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                %162 = pop.offset %161[%idx-8] : !kgen.pointer<scalar<ui8>>
                %163 = pop.pointer.bitcast %162 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                %164 = kgen.struct.gep %163[0] : <struct<(scalar<index>) memoryOnly>>
                %165 = pop.atomic.rmw sub(%164, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                %166 = pop.cmp eq(%165, %simd_28) : <1, index>
                %167 = pop.cast_to_builtin %166 : !pop.scalar<bool> to i1
                hlcf.if %167 {
                  pop.fence syncscope("") acquire
                  pop.aligned_free %162 : <scalar<ui8>>
                  hlcf.yield
                } else {
                  hlcf.yield
                }
                hlcf.yield
              } else {
                hlcf.yield
              }
              %155 = pop.load %143 : !kgen.pointer<index>
              %156 = index.and %155, %index4611686018427387904
              %157 = index.cmp ne(%156, %index0)
              hlcf.if %157 {
                %160 = pop.load %142 : !kgen.pointer<pointer<none>>
                %161 = pop.pointer.bitcast %160 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                %162 = pop.offset %161[%idx-8] : !kgen.pointer<scalar<ui8>>
                %163 = pop.pointer.bitcast %162 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                %164 = kgen.struct.gep %163[0] : <struct<(scalar<index>) memoryOnly>>
                %165 = pop.atomic.rmw sub(%164, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                %166 = pop.cmp eq(%165, %simd_28) : <1, index>
                %167 = pop.cast_to_builtin %166 : !pop.scalar<bool> to i1
                hlcf.if %167 {
                  pop.fence syncscope("") acquire
                  pop.aligned_free %162 : <scalar<ui8>>
                  hlcf.yield
                } else {
                  hlcf.yield
                }
                hlcf.yield
              } else {
                hlcf.yield
              }
              %158 = pop.array.gep %114[%124#1] : <array<10, scalar<ui8>>>
              %159 = pop.load %158 : !kgen.pointer<scalar<ui8>>
              kgen.call @"std::builtin::simd::SIMD::write_to[::Writer](::SIMD[$0, $1],$2&),dtype=ui8,size=1,writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>]"(%159, %arg0) : (!pop.scalar<ui8>, !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut) -> ()
              hlcf.continue "_loop_7" %124#0 : index
            }
            hlcf.yield
          } else {
            %110 = pop.select %105, %99, %false : i1
            hlcf.if %110 {
              %111 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
              pop.stack_alloc.lifetime.start(%111) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
              %112 = kgen.struct.gep %111[1] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %index2, %112 : !kgen.pointer<index>
              %113 = kgen.struct.gep %111[0] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %42, %113 : !kgen.pointer<pointer<none>>
              %114 = kgen.struct.gep %111[2] : <struct<(pointer<none>, index, index) memoryOnly>>
              pop.store %index2305843009213693952, %114 : !kgen.pointer<index>
              %115 = pop.pointer.bitcast %111 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
              %116 = pop.load %114 : !kgen.pointer<index>
              %117 = index.and %116, %index-9223372036854775808
              %118 = index.cmp ne(%117, %index0)
              %119 = hlcf.if %118 -> !kgen.pointer<none> {
                hlcf.yield %115 : !kgen.pointer<none>
              } else {
                %128 = pop.load %113 : !kgen.pointer<pointer<none>>
                hlcf.yield %128 : !kgen.pointer<none>
              }
              %120 = pop.load %114 : !kgen.pointer<index>
              %121 = index.and %120, %index-9223372036854775808
              %122 = index.cmp ne(%121, %index0)
              %123 = hlcf.if %122 -> index {
                %128 = pop.load %114 : !kgen.pointer<index>
                %129 = index.and %128, %index2233785415175766016
                %130 = index.shrs %129, %index56
                hlcf.yield %130 : index
              } else {
                %128 = pop.load %112 : !kgen.pointer<index>
                hlcf.yield %128 : index
              }
              %124 = kgen.struct.create(%119, %123) : !kgen.struct<(pointer<none>, index)>
              kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg0, %124) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
              %125 = pop.load %114 : !kgen.pointer<index>
              %126 = index.and %125, %index4611686018427387904
              %127 = index.cmp ne(%126, %index0)
              hlcf.if %127 {
                %128 = pop.load %113 : !kgen.pointer<pointer<none>>
                %129 = pop.pointer.bitcast %128 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                %130 = pop.offset %129[%idx-8] : !kgen.pointer<scalar<ui8>>
                %131 = pop.pointer.bitcast %130 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                %132 = kgen.struct.gep %131[0] : <struct<(scalar<index>) memoryOnly>>
                %133 = pop.atomic.rmw sub(%132, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                %134 = pop.cmp eq(%133, %simd_28) : <1, index>
                %135 = pop.cast_to_builtin %134 : !pop.scalar<bool> to i1
                hlcf.if %135 {
                  pop.fence syncscope("") acquire
                  pop.aligned_free %130 : <scalar<ui8>>
                  hlcf.yield
                } else {
                  hlcf.yield
                }
                hlcf.yield
              } else {
                hlcf.yield
              }
              pop.stack_alloc.lifetime.end(%111) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
              hlcf.loop "_loop_8" (%arg2 = %104 : index) {
                %128 = index.sub %104, %arg2
                %129 = index.sub %arg2, %index1
                %130 = index.cmp eq(%arg2, %index0)
                %131:2 = lit.try "try5" -> index, index {
                  %149 = pop.select %130, %arg2, %129 : index
                  hlcf.if %130 {
                    lit.try.raise "try5" %149, %128 : index, index
                  } else {
                    hlcf.yield
                  }
                  lit.try.yield %149, %128 : index, index
                } except (%arg3: index, %arg4: index) {
                  hlcf.break "_loop_8"
                } else (%arg3: index, %arg4: index) {
                  lit.try.yield %arg3, %arg4 : index, index
                }
                %132 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
                pop.stack_alloc.lifetime.start(%132) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                %133 = kgen.struct.gep %132[1] : <struct<(pointer<none>, index, index) memoryOnly>>
                pop.store %index1, %133 : !kgen.pointer<index>
                %134 = kgen.struct.gep %132[0] : <struct<(pointer<none>, index, index) memoryOnly>>
                pop.store %44, %134 : !kgen.pointer<pointer<none>>
                %135 = kgen.struct.gep %132[2] : <struct<(pointer<none>, index, index) memoryOnly>>
                pop.store %index2305843009213693952, %135 : !kgen.pointer<index>
                %136 = pop.pointer.bitcast %132 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
                %137 = pop.load %135 : !kgen.pointer<index>
                %138 = index.and %137, %index-9223372036854775808
                %139 = index.cmp ne(%138, %index0)
                %140 = hlcf.if %139 -> !kgen.pointer<none> {
                  hlcf.yield %136 : !kgen.pointer<none>
                } else {
                  %149 = pop.load %134 : !kgen.pointer<pointer<none>>
                  hlcf.yield %149 : !kgen.pointer<none>
                }
                %141 = pop.load %135 : !kgen.pointer<index>
                %142 = index.and %141, %index-9223372036854775808
                %143 = index.cmp ne(%142, %index0)
                %144 = hlcf.if %143 -> index {
                  %149 = pop.load %135 : !kgen.pointer<index>
                  %150 = index.and %149, %index2233785415175766016
                  %151 = index.shrs %150, %index56
                  hlcf.yield %151 : index
                } else {
                  %149 = pop.load %133 : !kgen.pointer<index>
                  hlcf.yield %149 : index
                }
                %145 = kgen.struct.create(%140, %144) : !kgen.struct<(pointer<none>, index)>
                kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg0, %145) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
                %146 = pop.load %135 : !kgen.pointer<index>
                %147 = index.and %146, %index4611686018427387904
                %148 = index.cmp ne(%147, %index0)
                hlcf.if %148 {
                  %149 = pop.load %134 : !kgen.pointer<pointer<none>>
                  %150 = pop.pointer.bitcast %149 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                  %151 = pop.offset %150[%idx-8] : !kgen.pointer<scalar<ui8>>
                  %152 = pop.pointer.bitcast %151 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                  %153 = kgen.struct.gep %152[0] : <struct<(scalar<index>) memoryOnly>>
                  %154 = pop.atomic.rmw sub(%153, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                  %155 = pop.cmp eq(%154, %simd_28) : <1, index>
                  %156 = pop.cast_to_builtin %155 : !pop.scalar<bool> to i1
                  hlcf.if %156 {
                    pop.fence syncscope("") acquire
                    pop.aligned_free %151 : <scalar<ui8>>
                    hlcf.yield
                  } else {
                    hlcf.yield
                  }
                  hlcf.yield
                } else {
                  hlcf.yield
                }
                pop.stack_alloc.lifetime.end(%132) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                hlcf.continue "_loop_8" %131#0 : index
              }
              hlcf.loop "_loop_9" (%arg2 = %93 : index) {
                %128 = index.add %arg2, %index-1
                %129 = index.sub %arg2, %index-1
                %130 = index.cmp sgt(%arg2, %index-1)
                %131 = pop.select %130, %129, %index0 : index
                %132 = index.cmp sle(%131, %index0)
                %133 = pop.select %132, %arg2, %128 : index
                %134:2 = lit.try "try6" -> index, index {
                  hlcf.if %132 {
                    pop.stack_alloc.lifetime.end(%89) : !kgen.pointer<array<21, scalar<ui8>>>
                    lit.try.raise "try6" %133, %arg2 : index, index
                  } else {
                    hlcf.yield
                  }
                  lit.try.yield %133, %arg2 : index, index
                } except (%arg3: index, %arg4: index) {
                  hlcf.break "_loop_9"
                } else (%arg3: index, %arg4: index) {
                  lit.try.yield %arg3, %arg4 : index, index
                }
                %135 = pop.cast_from_builtin %134#1 : index to !pop.scalar<index>
                %136 = pop.cast %135 : !pop.scalar<index> to !pop.scalar<uindex>
                %137 = pop.cmp lt(%136, %simd) : <1, uindex>
                %138 = pop.cast_to_builtin %137 : !pop.scalar<bool> to i1
                %139 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
                pop.stack_alloc.lifetime.start(%139) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                pop.store %array, %139 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                %140 = pop.pointer.bitcast %139 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<index>
                %141 = pop.load %140 : !kgen.pointer<index>
                %142 = index.cmp eq(%141, %index-1)
                %143 = pop.select %142, %index0, %index-1 : index
                pop.stack_alloc.lifetime.end(%139) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                %144 = index.cmp eq(%143, %index-1)
                %145 = hlcf.if %144 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
                  %170 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
                  pop.stack_alloc.lifetime.start(%170) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                  pop.store %array, %170 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                  %171 = pop.pointer.bitcast %170 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                  %172 = pop.load %171 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                  pop.stack_alloc.lifetime.end(%170) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                  hlcf.yield %172 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
                } else {
                  hlcf.yield %6 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
                }
                %146 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
                pop.stack_alloc.lifetime.start(%146) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                %147 = kgen.struct.gep %146[1] : <struct<(pointer<none>, index, index) memoryOnly>>
                pop.store %index6, %147 : !kgen.pointer<index>
                %148 = kgen.struct.gep %146[0] : <struct<(pointer<none>, index, index) memoryOnly>>
                pop.store %34, %148 : !kgen.pointer<pointer<none>>
                %149 = kgen.struct.gep %146[2] : <struct<(pointer<none>, index, index) memoryOnly>>
                pop.store %index2305843009213693952, %149 : !kgen.pointer<index>
                %150 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
                pop.stack_alloc.lifetime.start(%150) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                %151 = kgen.struct.gep %150[1] : <struct<(pointer<none>, index, index) memoryOnly>>
                pop.store %index39, %151 : !kgen.pointer<index>
                %152 = kgen.struct.gep %150[0] : <struct<(pointer<none>, index, index) memoryOnly>>
                pop.store %32, %152 : !kgen.pointer<pointer<none>>
                %153 = kgen.struct.gep %150[2] : <struct<(pointer<none>, index, index) memoryOnly>>
                pop.store %index2305843009213693952, %153 : !kgen.pointer<index>
                %154 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                %155 = pop.pointer.bitcast %154 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                pop.store %145, %155 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                %156 = pop.load %154 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                %157 = pop.array.get %156[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                %158 = pop.array.create [%157] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                %159 = kgen.struct.create(%158) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                %160 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                %161 = pop.pointer.bitcast %160 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                pop.store %159, %160 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                hlcf.if %138 {
                  hlcf.yield
                } else {
                  %170 = pop.stack_allocation 2048 x scalar<ui8> align 1
                  %171 = pop.pointer.bitcast %170 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<none>
                  %172 = kgen.struct.create(%171, %index0) : !kgen.struct<(pointer<none>, index) memoryOnly>
                  %173 = kgen.call @"std::collections::string::string::String::write_to[::Writer](::String,$0&),writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%146, %172) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
                  %174 = kgen.call @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%134#1, %173) : (index, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
                  %175 = kgen.call @"std::collections::string::string::String::write_to[::Writer](::String,$0&),writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%150, %174) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
                  %176 = kgen.call @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%idx20, %175) : (index, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
                  %177 = kgen.struct.extract %176[0] : <(pointer<none>, index) memoryOnly>
                  %178 = kgen.struct.extract %176[1] : <(pointer<none>, index) memoryOnly>
                  %179 = index.add %178, %index1
                  %180 = index.cmp sgt(%179, %index2048)
                  hlcf.if %180 {
                    kgen.call @"std::io::io::_printf[KGENParamList[::AnyType],::StringSlice[::Bool(False), StaticConstantOrigin, *?],*::AnyType,LITImmutOrigin,::Origin[::Bool(False), $3]](*$0,file:::FileDescriptor),types.values`=[],fmt={ #interp.memref<{[(#interp.memory_handle<16, \22HEAP_BUFFER_BYTES exceeded, increase with: `mojo -D HEAP_BUFFER_BYTES=4096`\\0A\\00\22 string>, const_global, [], [])], []}, 0, 0>, 76 }"(%index1) : (index) -> ()
                    llvm.intr.trap
                    hlcf.loop "_loop_0" {
                      hlcf.continue "_loop_0"
                    }
                    kgen.unreachable
                  } else {
                    hlcf.yield
                  }
                  %181 = pop.pointer.bitcast %177 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                  %182 = pop.offset %181[%178] : !kgen.pointer<scalar<ui8>>
                  pop.store %simd_9, %182 : !kgen.pointer<scalar<ui8>>
                  %183 = pop.load %160 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                  %184 = kgen.struct.extract %183[0] : <(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                  %185 = pop.array.get %184[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                  %186 = pop.array.create [%185] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                  %187 = kgen.struct.create(%186) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                  %188 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                  pop.store %187, %188 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                  %189 = pop.pointer.bitcast %188 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
                  %190 = pop.load %189 : !kgen.pointer<index>
                  %191 = index.cmp eq(%190, %index-1)
                  %192 = pop.select %191, %index0, %index-1 : index
                  %193 = index.cmp eq(%192, %index-1)
                  %194 = hlcf.if %193 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
                    %195 = pop.load %160 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                    %196 = kgen.struct.extract %195[0] : <(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                    %197 = pop.array.get %196[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                    %198 = pop.array.create [%197] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                    %199 = kgen.struct.create(%198) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                    %200 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                    pop.store %199, %200 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                    %201 = pop.pointer.bitcast %200 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
                    %202 = pop.load %201 : !kgen.pointer<index>
                    %203 = index.cmp eq(%202, %index-1)
                    %204 = pop.select %203, %index0, %index-1 : index
                    %205 = index.cmp eq(%204, %index-1)
                    %206 = pop.xor %205, %0
                    hlcf.if %206 {
                      %208 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
                      pop.stack_alloc.lifetime.start(%208) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                      %209 = kgen.struct.gep %208[1] : <struct<(pointer<none>, index, index) memoryOnly>>
                      pop.store %index192, %209 : !kgen.pointer<index>
                      %210 = kgen.struct.gep %208[0] : <struct<(pointer<none>, index, index) memoryOnly>>
                      pop.store %30, %210 : !kgen.pointer<pointer<none>>
                      %211 = kgen.struct.gep %208[2] : <struct<(pointer<none>, index, index) memoryOnly>>
                      pop.store %index2305843009213693952, %211 : !kgen.pointer<index>
                      %212 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                      %213 = pop.pointer.bitcast %212 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                      pop.store %18, %213 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                      %214 = pop.load %212 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                      %215 = pop.array.get %214[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                      %216 = pop.array.create [%215] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                      %217 = kgen.struct.create(%216) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                      %218 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                      %219 = pop.pointer.bitcast %218 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                      pop.store %217, %218 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                      %220 = pop.pointer.bitcast %218 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
                      %221 = pop.load %220 : !kgen.pointer<index>
                      %222 = index.cmp eq(%221, %index-1)
                      %223 = pop.select %222, %index0, %index-1 : index
                      %224 = index.cmp eq(%223, %index-1)
                      %225 = hlcf.if %224 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
                        %243 = pop.load %219 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                        hlcf.yield %243 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
                      } else {
                        hlcf.yield %22 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
                      }
                      %226 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
                      pop.stack_alloc.lifetime.start(%226) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                      %227 = kgen.struct.gep %226[1] : <struct<(pointer<none>, index, index) memoryOnly>>
                      pop.store %index1, %227 : !kgen.pointer<index>
                      %228 = kgen.struct.gep %226[0] : <struct<(pointer<none>, index, index) memoryOnly>>
                      pop.store %24, %228 : !kgen.pointer<pointer<none>>
                      %229 = kgen.struct.gep %226[2] : <struct<(pointer<none>, index, index) memoryOnly>>
                      pop.store %index2305843009213693952, %229 : !kgen.pointer<index>
                      %230 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
                      pop.stack_alloc.lifetime.start(%230) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                      %231 = kgen.struct.gep %230[1] : <struct<(pointer<none>, index, index) memoryOnly>>
                      pop.store %index2, %231 : !kgen.pointer<index>
                      %232 = kgen.struct.gep %230[0] : <struct<(pointer<none>, index, index) memoryOnly>>
                      pop.store %26, %232 : !kgen.pointer<pointer<none>>
                      %233 = kgen.struct.gep %230[2] : <struct<(pointer<none>, index, index) memoryOnly>>
                      pop.store %index2305843009213693952, %233 : !kgen.pointer<index>
                      kgen.call @"std::io::io::print[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?],flush:::Bool,file:::FileDescriptor$),Ts.values`=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::reflection::location::SourceLocation\22>>, struct<(index, index, struct<(pointer<none>, index)>)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]]"(%struct_4, %226, %225, %230, %208, %61, %struct, %0, %index1) : (!kgen.struct<(pointer<none>, index)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index)>, i1, index owned) -> ()
                      %234 = pop.load %229 : !kgen.pointer<index>
                      %235 = index.and %234, %index4611686018427387904
                      %236 = index.cmp ne(%235, %index0)
                      hlcf.if %236 {
                        %243 = pop.load %228 : !kgen.pointer<pointer<none>>
                        %244 = pop.pointer.bitcast %243 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                        %245 = pop.offset %244[%idx-8] : !kgen.pointer<scalar<ui8>>
                        %246 = pop.pointer.bitcast %245 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                        %247 = kgen.struct.gep %246[0] : <struct<(scalar<index>) memoryOnly>>
                        %248 = pop.atomic.rmw sub(%247, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                        %249 = pop.cmp eq(%248, %simd_28) : <1, index>
                        %250 = pop.cast_to_builtin %249 : !pop.scalar<bool> to i1
                        hlcf.if %250 {
                          pop.fence syncscope("") acquire
                          pop.aligned_free %245 : <scalar<ui8>>
                          hlcf.yield
                        } else {
                          hlcf.yield
                        }
                        hlcf.yield
                      } else {
                        hlcf.yield
                      }
                      %237 = pop.load %233 : !kgen.pointer<index>
                      %238 = index.and %237, %index4611686018427387904
                      %239 = index.cmp ne(%238, %index0)
                      hlcf.if %239 {
                        %243 = pop.load %232 : !kgen.pointer<pointer<none>>
                        %244 = pop.pointer.bitcast %243 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                        %245 = pop.offset %244[%idx-8] : !kgen.pointer<scalar<ui8>>
                        %246 = pop.pointer.bitcast %245 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                        %247 = kgen.struct.gep %246[0] : <struct<(scalar<index>) memoryOnly>>
                        %248 = pop.atomic.rmw sub(%247, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                        %249 = pop.cmp eq(%248, %simd_28) : <1, index>
                        %250 = pop.cast_to_builtin %249 : !pop.scalar<bool> to i1
                        hlcf.if %250 {
                          pop.fence syncscope("") acquire
                          pop.aligned_free %245 : <scalar<ui8>>
                          hlcf.yield
                        } else {
                          hlcf.yield
                        }
                        hlcf.yield
                      } else {
                        hlcf.yield
                      }
                      %240 = pop.load %211 : !kgen.pointer<index>
                      %241 = index.and %240, %index4611686018427387904
                      %242 = index.cmp ne(%241, %index0)
                      hlcf.if %242 {
                        %243 = pop.load %210 : !kgen.pointer<pointer<none>>
                        %244 = pop.pointer.bitcast %243 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                        %245 = pop.offset %244[%idx-8] : !kgen.pointer<scalar<ui8>>
                        %246 = pop.pointer.bitcast %245 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                        %247 = kgen.struct.gep %246[0] : <struct<(scalar<index>) memoryOnly>>
                        %248 = pop.atomic.rmw sub(%247, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                        %249 = pop.cmp eq(%248, %simd_28) : <1, index>
                        %250 = pop.cast_to_builtin %249 : !pop.scalar<bool> to i1
                        hlcf.if %250 {
                          pop.fence syncscope("") acquire
                          pop.aligned_free %245 : <scalar<ui8>>
                          hlcf.yield
                        } else {
                          hlcf.yield
                        }
                        hlcf.yield
                      } else {
                        hlcf.yield
                      }
                      llvm.intr.trap
                      hlcf.loop "_loop_0" {
                        hlcf.continue "_loop_0"
                      }
                      kgen.unreachable
                    } else {
                      hlcf.yield
                    }
                    %207 = pop.load %161 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                    hlcf.yield %207 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
                  } else {
                    hlcf.yield %35 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
                  }
                  kgen.call @"std::builtin::debug_assert::_debug_assert_msg[LITImmutOrigin,::Origin[::Bool(False), $0]](::UnsafePointer[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1, ::AddressSpace(::Int(0))],::Int,::SourceLocation)"(%177, %194) : (!kgen.pointer<none>, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>) -> ()
                  hlcf.yield
                }
                %162 = pop.load %149 : !kgen.pointer<index>
                %163 = index.and %162, %index4611686018427387904
                %164 = index.cmp ne(%163, %index0)
                hlcf.if %164 {
                  %170 = pop.load %148 : !kgen.pointer<pointer<none>>
                  %171 = pop.pointer.bitcast %170 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                  %172 = pop.offset %171[%idx-8] : !kgen.pointer<scalar<ui8>>
                  %173 = pop.pointer.bitcast %172 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                  %174 = kgen.struct.gep %173[0] : <struct<(scalar<index>) memoryOnly>>
                  %175 = pop.atomic.rmw sub(%174, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                  %176 = pop.cmp eq(%175, %simd_28) : <1, index>
                  %177 = pop.cast_to_builtin %176 : !pop.scalar<bool> to i1
                  hlcf.if %177 {
                    pop.fence syncscope("") acquire
                    pop.aligned_free %172 : <scalar<ui8>>
                    hlcf.yield
                  } else {
                    hlcf.yield
                  }
                  hlcf.yield
                } else {
                  hlcf.yield
                }
                %165 = pop.load %153 : !kgen.pointer<index>
                %166 = index.and %165, %index4611686018427387904
                %167 = index.cmp ne(%166, %index0)
                hlcf.if %167 {
                  %170 = pop.load %152 : !kgen.pointer<pointer<none>>
                  %171 = pop.pointer.bitcast %170 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                  %172 = pop.offset %171[%idx-8] : !kgen.pointer<scalar<ui8>>
                  %173 = pop.pointer.bitcast %172 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                  %174 = kgen.struct.gep %173[0] : <struct<(scalar<index>) memoryOnly>>
                  %175 = pop.atomic.rmw sub(%174, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                  %176 = pop.cmp eq(%175, %simd_28) : <1, index>
                  %177 = pop.cast_to_builtin %176 : !pop.scalar<bool> to i1
                  hlcf.if %177 {
                    pop.fence syncscope("") acquire
                    pop.aligned_free %172 : <scalar<ui8>>
                    hlcf.yield
                  } else {
                    hlcf.yield
                  }
                  hlcf.yield
                } else {
                  hlcf.yield
                }
                %168 = pop.array.gep %89[%134#1] : <array<21, scalar<ui8>>>
                %169 = pop.load %168 : !kgen.pointer<scalar<ui8>>
                kgen.call @"std::builtin::simd::SIMD::write_to[::Writer](::SIMD[$0, $1],$2&),dtype=ui8,size=1,writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>]"(%169, %arg0) : (!pop.scalar<ui8>, !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut) -> ()
                hlcf.continue "_loop_9" %134#0 : index
              }
              hlcf.yield
            } else {
              %111 = hlcf.loop "_loop_3" (%arg2 = %1 : i1, %arg3 = %93 : index) -> i1 {
                %116 = index.add %arg3, %index-1
                %117 = index.sub %arg3, %index-1
                %118 = index.cmp sgt(%arg3, %index-1)
                %119 = pop.select %118, %117, %index0 : index
                %120 = index.cmp sle(%119, %index0)
                %121 = pop.select %120, %arg3, %116 : index
                %122:2 = lit.try "try1" -> index, index {
                  hlcf.if %120 {
                    pop.stack_alloc.lifetime.end(%89) : !kgen.pointer<array<21, scalar<ui8>>>
                    lit.try.raise "try1" %121, %arg3 : index, index
                  } else {
                    hlcf.yield
                  }
                  lit.try.yield %121, %arg3 : index, index
                } except (%arg4: index, %arg5: index) {
                  hlcf.break "_loop_3" %arg2 : i1
                } else (%arg4: index, %arg5: index) {
                  lit.try.yield %arg4, %arg5 : index, index
                }
                %123 = index.cmp eq(%122#1, %91)
                %124 = index.sub %90#0, %122#1
                %125 = index.sub %124, %index2
                %126 = index.cmp eq(%90#1, %125)
                %127 = pop.select %103, %126, %false : i1
                %128 = pop.select %127, %0, %arg2 : i1
                hlcf.if %127 {
                  hlcf.if %123 {
                    %181 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
                    pop.stack_alloc.lifetime.start(%181) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                    %182 = kgen.struct.gep %181[1] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %index1, %182 : !kgen.pointer<index>
                    %183 = kgen.struct.gep %181[0] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %44, %183 : !kgen.pointer<pointer<none>>
                    %184 = kgen.struct.gep %181[2] : <struct<(pointer<none>, index, index) memoryOnly>>
                    pop.store %index2305843009213693952, %184 : !kgen.pointer<index>
                    %185 = pop.pointer.bitcast %181 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
                    %186 = pop.load %184 : !kgen.pointer<index>
                    %187 = index.and %186, %index-9223372036854775808
                    %188 = index.cmp ne(%187, %index0)
                    %189 = hlcf.if %188 -> !kgen.pointer<none> {
                      hlcf.yield %185 : !kgen.pointer<none>
                    } else {
                      %198 = pop.load %183 : !kgen.pointer<pointer<none>>
                      hlcf.yield %198 : !kgen.pointer<none>
                    }
                    %190 = pop.load %184 : !kgen.pointer<index>
                    %191 = index.and %190, %index-9223372036854775808
                    %192 = index.cmp ne(%191, %index0)
                    %193 = hlcf.if %192 -> index {
                      %198 = pop.load %184 : !kgen.pointer<index>
                      %199 = index.and %198, %index2233785415175766016
                      %200 = index.shrs %199, %index56
                      hlcf.yield %200 : index
                    } else {
                      %198 = pop.load %182 : !kgen.pointer<index>
                      hlcf.yield %198 : index
                    }
                    %194 = kgen.struct.create(%189, %193) : !kgen.struct<(pointer<none>, index)>
                    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg0, %194) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
                    %195 = pop.load %184 : !kgen.pointer<index>
                    %196 = index.and %195, %index4611686018427387904
                    %197 = index.cmp ne(%196, %index0)
                    hlcf.if %197 {
                      %198 = pop.load %183 : !kgen.pointer<pointer<none>>
                      %199 = pop.pointer.bitcast %198 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                      %200 = pop.offset %199[%idx-8] : !kgen.pointer<scalar<ui8>>
                      %201 = pop.pointer.bitcast %200 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                      %202 = kgen.struct.gep %201[0] : <struct<(scalar<index>) memoryOnly>>
                      %203 = pop.atomic.rmw sub(%202, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                      %204 = pop.cmp eq(%203, %simd_28) : <1, index>
                      %205 = pop.cast_to_builtin %204 : !pop.scalar<bool> to i1
                      hlcf.if %205 {
                        pop.fence syncscope("") acquire
                        pop.aligned_free %200 : <scalar<ui8>>
                        hlcf.yield
                      } else {
                        hlcf.yield
                      }
                      hlcf.yield
                    } else {
                      hlcf.yield
                    }
                    pop.stack_alloc.lifetime.end(%181) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                    hlcf.yield
                  } else {
                    hlcf.yield
                  }
                  %164 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
                  pop.stack_alloc.lifetime.start(%164) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                  %165 = kgen.struct.gep %164[1] : <struct<(pointer<none>, index, index) memoryOnly>>
                  pop.store %index1, %165 : !kgen.pointer<index>
                  %166 = kgen.struct.gep %164[0] : <struct<(pointer<none>, index, index) memoryOnly>>
                  pop.store %50, %166 : !kgen.pointer<pointer<none>>
                  %167 = kgen.struct.gep %164[2] : <struct<(pointer<none>, index, index) memoryOnly>>
                  pop.store %index2305843009213693952, %167 : !kgen.pointer<index>
                  %168 = pop.pointer.bitcast %164 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
                  %169 = pop.load %167 : !kgen.pointer<index>
                  %170 = index.and %169, %index-9223372036854775808
                  %171 = index.cmp ne(%170, %index0)
                  %172 = hlcf.if %171 -> !kgen.pointer<none> {
                    hlcf.yield %168 : !kgen.pointer<none>
                  } else {
                    %181 = pop.load %166 : !kgen.pointer<pointer<none>>
                    hlcf.yield %181 : !kgen.pointer<none>
                  }
                  %173 = pop.load %167 : !kgen.pointer<index>
                  %174 = index.and %173, %index-9223372036854775808
                  %175 = index.cmp ne(%174, %index0)
                  %176 = hlcf.if %175 -> index {
                    %181 = pop.load %167 : !kgen.pointer<index>
                    %182 = index.and %181, %index2233785415175766016
                    %183 = index.shrs %182, %index56
                    hlcf.yield %183 : index
                  } else {
                    %181 = pop.load %165 : !kgen.pointer<index>
                    hlcf.yield %181 : index
                  }
                  %177 = kgen.struct.create(%172, %176) : !kgen.struct<(pointer<none>, index)>
                  kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg0, %177) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
                  %178 = pop.load %167 : !kgen.pointer<index>
                  %179 = index.and %178, %index4611686018427387904
                  %180 = index.cmp ne(%179, %index0)
                  hlcf.if %180 {
                    %181 = pop.load %166 : !kgen.pointer<pointer<none>>
                    %182 = pop.pointer.bitcast %181 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                    %183 = pop.offset %182[%idx-8] : !kgen.pointer<scalar<ui8>>
                    %184 = pop.pointer.bitcast %183 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                    %185 = kgen.struct.gep %184[0] : <struct<(scalar<index>) memoryOnly>>
                    %186 = pop.atomic.rmw sub(%185, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                    %187 = pop.cmp eq(%186, %simd_28) : <1, index>
                    %188 = pop.cast_to_builtin %187 : !pop.scalar<bool> to i1
                    hlcf.if %188 {
                      pop.fence syncscope("") acquire
                      pop.aligned_free %183 : <scalar<ui8>>
                      hlcf.yield
                    } else {
                      hlcf.yield
                    }
                    hlcf.yield
                  } else {
                    hlcf.yield
                  }
                  pop.stack_alloc.lifetime.end(%164) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                  hlcf.yield
                } else {
                  hlcf.yield
                }
                %129 = pop.cast_from_builtin %122#1 : index to !pop.scalar<index>
                %130 = pop.cast %129 : !pop.scalar<index> to !pop.scalar<uindex>
                %131 = pop.cmp lt(%130, %simd) : <1, uindex>
                %132 = pop.cast_to_builtin %131 : !pop.scalar<bool> to i1
                %133 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
                pop.stack_alloc.lifetime.start(%133) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                pop.store %array, %133 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                %134 = pop.pointer.bitcast %133 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<index>
                %135 = pop.load %134 : !kgen.pointer<index>
                %136 = index.cmp eq(%135, %index-1)
                %137 = pop.select %136, %index0, %index-1 : index
                pop.stack_alloc.lifetime.end(%133) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                %138 = index.cmp eq(%137, %index-1)
                %139 = hlcf.if %138 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
                  %164 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
                  pop.stack_alloc.lifetime.start(%164) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                  pop.store %array, %164 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                  %165 = pop.pointer.bitcast %164 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                  %166 = pop.load %165 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                  pop.stack_alloc.lifetime.end(%164) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                  hlcf.yield %166 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
                } else {
                  hlcf.yield %5 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
                }
                %140 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
                pop.stack_alloc.lifetime.start(%140) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                %141 = kgen.struct.gep %140[1] : <struct<(pointer<none>, index, index) memoryOnly>>
                pop.store %index6, %141 : !kgen.pointer<index>
                %142 = kgen.struct.gep %140[0] : <struct<(pointer<none>, index, index) memoryOnly>>
                pop.store %34, %142 : !kgen.pointer<pointer<none>>
                %143 = kgen.struct.gep %140[2] : <struct<(pointer<none>, index, index) memoryOnly>>
                pop.store %index2305843009213693952, %143 : !kgen.pointer<index>
                %144 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
                pop.stack_alloc.lifetime.start(%144) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                %145 = kgen.struct.gep %144[1] : <struct<(pointer<none>, index, index) memoryOnly>>
                pop.store %index39, %145 : !kgen.pointer<index>
                %146 = kgen.struct.gep %144[0] : <struct<(pointer<none>, index, index) memoryOnly>>
                pop.store %32, %146 : !kgen.pointer<pointer<none>>
                %147 = kgen.struct.gep %144[2] : <struct<(pointer<none>, index, index) memoryOnly>>
                pop.store %index2305843009213693952, %147 : !kgen.pointer<index>
                %148 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                %149 = pop.pointer.bitcast %148 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                pop.store %139, %149 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                %150 = pop.load %148 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                %151 = pop.array.get %150[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                %152 = pop.array.create [%151] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                %153 = kgen.struct.create(%152) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                %154 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                %155 = pop.pointer.bitcast %154 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                pop.store %153, %154 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                hlcf.if %132 {
                  hlcf.yield
                } else {
                  %164 = pop.stack_allocation 2048 x scalar<ui8> align 1
                  %165 = pop.pointer.bitcast %164 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<none>
                  %166 = kgen.struct.create(%165, %index0) : !kgen.struct<(pointer<none>, index) memoryOnly>
                  %167 = kgen.call @"std::collections::string::string::String::write_to[::Writer](::String,$0&),writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%140, %166) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
                  %168 = kgen.call @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%122#1, %167) : (index, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
                  %169 = kgen.call @"std::collections::string::string::String::write_to[::Writer](::String,$0&),writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%144, %168) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
                  %170 = kgen.call @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%idx20, %169) : (index, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
                  %171 = kgen.struct.extract %170[0] : <(pointer<none>, index) memoryOnly>
                  %172 = kgen.struct.extract %170[1] : <(pointer<none>, index) memoryOnly>
                  %173 = index.add %172, %index1
                  %174 = index.cmp sgt(%173, %index2048)
                  hlcf.if %174 {
                    kgen.call @"std::io::io::_printf[KGENParamList[::AnyType],::StringSlice[::Bool(False), StaticConstantOrigin, *?],*::AnyType,LITImmutOrigin,::Origin[::Bool(False), $3]](*$0,file:::FileDescriptor),types.values`=[],fmt={ #interp.memref<{[(#interp.memory_handle<16, \22HEAP_BUFFER_BYTES exceeded, increase with: `mojo -D HEAP_BUFFER_BYTES=4096`\\0A\\00\22 string>, const_global, [], [])], []}, 0, 0>, 76 }"(%index1) : (index) -> ()
                    llvm.intr.trap
                    hlcf.loop "_loop_0" {
                      hlcf.continue "_loop_0"
                    }
                    kgen.unreachable
                  } else {
                    hlcf.yield
                  }
                  %175 = pop.pointer.bitcast %171 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                  %176 = pop.offset %175[%172] : !kgen.pointer<scalar<ui8>>
                  pop.store %simd_9, %176 : !kgen.pointer<scalar<ui8>>
                  %177 = pop.load %154 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                  %178 = kgen.struct.extract %177[0] : <(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                  %179 = pop.array.get %178[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                  %180 = pop.array.create [%179] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                  %181 = kgen.struct.create(%180) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                  %182 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                  pop.store %181, %182 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                  %183 = pop.pointer.bitcast %182 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
                  %184 = pop.load %183 : !kgen.pointer<index>
                  %185 = index.cmp eq(%184, %index-1)
                  %186 = pop.select %185, %index0, %index-1 : index
                  %187 = index.cmp eq(%186, %index-1)
                  %188 = hlcf.if %187 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
                    %189 = pop.load %154 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                    %190 = kgen.struct.extract %189[0] : <(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                    %191 = pop.array.get %190[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                    %192 = pop.array.create [%191] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                    %193 = kgen.struct.create(%192) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                    %194 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                    pop.store %193, %194 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                    %195 = pop.pointer.bitcast %194 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
                    %196 = pop.load %195 : !kgen.pointer<index>
                    %197 = index.cmp eq(%196, %index-1)
                    %198 = pop.select %197, %index0, %index-1 : index
                    %199 = index.cmp eq(%198, %index-1)
                    %200 = pop.xor %199, %0
                    hlcf.if %200 {
                      %202 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
                      pop.stack_alloc.lifetime.start(%202) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                      %203 = kgen.struct.gep %202[1] : <struct<(pointer<none>, index, index) memoryOnly>>
                      pop.store %index192, %203 : !kgen.pointer<index>
                      %204 = kgen.struct.gep %202[0] : <struct<(pointer<none>, index, index) memoryOnly>>
                      pop.store %30, %204 : !kgen.pointer<pointer<none>>
                      %205 = kgen.struct.gep %202[2] : <struct<(pointer<none>, index, index) memoryOnly>>
                      pop.store %index2305843009213693952, %205 : !kgen.pointer<index>
                      %206 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                      %207 = pop.pointer.bitcast %206 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                      pop.store %18, %207 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                      %208 = pop.load %206 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
                      %209 = pop.array.get %208[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                      %210 = pop.array.create [%209] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
                      %211 = kgen.struct.create(%210) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                      %212 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
                      %213 = pop.pointer.bitcast %212 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                      pop.store %211, %212 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
                      %214 = pop.pointer.bitcast %212 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
                      %215 = pop.load %214 : !kgen.pointer<index>
                      %216 = index.cmp eq(%215, %index-1)
                      %217 = pop.select %216, %index0, %index-1 : index
                      %218 = index.cmp eq(%217, %index-1)
                      %219 = hlcf.if %218 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
                        %237 = pop.load %213 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                        hlcf.yield %237 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
                      } else {
                        hlcf.yield %22 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
                      }
                      %220 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
                      pop.stack_alloc.lifetime.start(%220) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                      %221 = kgen.struct.gep %220[1] : <struct<(pointer<none>, index, index) memoryOnly>>
                      pop.store %index1, %221 : !kgen.pointer<index>
                      %222 = kgen.struct.gep %220[0] : <struct<(pointer<none>, index, index) memoryOnly>>
                      pop.store %24, %222 : !kgen.pointer<pointer<none>>
                      %223 = kgen.struct.gep %220[2] : <struct<(pointer<none>, index, index) memoryOnly>>
                      pop.store %index2305843009213693952, %223 : !kgen.pointer<index>
                      %224 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
                      pop.stack_alloc.lifetime.start(%224) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                      %225 = kgen.struct.gep %224[1] : <struct<(pointer<none>, index, index) memoryOnly>>
                      pop.store %index2, %225 : !kgen.pointer<index>
                      %226 = kgen.struct.gep %224[0] : <struct<(pointer<none>, index, index) memoryOnly>>
                      pop.store %26, %226 : !kgen.pointer<pointer<none>>
                      %227 = kgen.struct.gep %224[2] : <struct<(pointer<none>, index, index) memoryOnly>>
                      pop.store %index2305843009213693952, %227 : !kgen.pointer<index>
                      kgen.call @"std::io::io::print[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?],flush:::Bool,file:::FileDescriptor$),Ts.values`=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::reflection::location::SourceLocation\22>>, struct<(index, index, struct<(pointer<none>, index)>)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]]"(%struct_4, %220, %219, %224, %202, %61, %struct, %0, %index1) : (!kgen.struct<(pointer<none>, index)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index)>, i1, index owned) -> ()
                      %228 = pop.load %223 : !kgen.pointer<index>
                      %229 = index.and %228, %index4611686018427387904
                      %230 = index.cmp ne(%229, %index0)
                      hlcf.if %230 {
                        %237 = pop.load %222 : !kgen.pointer<pointer<none>>
                        %238 = pop.pointer.bitcast %237 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                        %239 = pop.offset %238[%idx-8] : !kgen.pointer<scalar<ui8>>
                        %240 = pop.pointer.bitcast %239 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                        %241 = kgen.struct.gep %240[0] : <struct<(scalar<index>) memoryOnly>>
                        %242 = pop.atomic.rmw sub(%241, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                        %243 = pop.cmp eq(%242, %simd_28) : <1, index>
                        %244 = pop.cast_to_builtin %243 : !pop.scalar<bool> to i1
                        hlcf.if %244 {
                          pop.fence syncscope("") acquire
                          pop.aligned_free %239 : <scalar<ui8>>
                          hlcf.yield
                        } else {
                          hlcf.yield
                        }
                        hlcf.yield
                      } else {
                        hlcf.yield
                      }
                      %231 = pop.load %227 : !kgen.pointer<index>
                      %232 = index.and %231, %index4611686018427387904
                      %233 = index.cmp ne(%232, %index0)
                      hlcf.if %233 {
                        %237 = pop.load %226 : !kgen.pointer<pointer<none>>
                        %238 = pop.pointer.bitcast %237 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                        %239 = pop.offset %238[%idx-8] : !kgen.pointer<scalar<ui8>>
                        %240 = pop.pointer.bitcast %239 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                        %241 = kgen.struct.gep %240[0] : <struct<(scalar<index>) memoryOnly>>
                        %242 = pop.atomic.rmw sub(%241, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                        %243 = pop.cmp eq(%242, %simd_28) : <1, index>
                        %244 = pop.cast_to_builtin %243 : !pop.scalar<bool> to i1
                        hlcf.if %244 {
                          pop.fence syncscope("") acquire
                          pop.aligned_free %239 : <scalar<ui8>>
                          hlcf.yield
                        } else {
                          hlcf.yield
                        }
                        hlcf.yield
                      } else {
                        hlcf.yield
                      }
                      %234 = pop.load %205 : !kgen.pointer<index>
                      %235 = index.and %234, %index4611686018427387904
                      %236 = index.cmp ne(%235, %index0)
                      hlcf.if %236 {
                        %237 = pop.load %204 : !kgen.pointer<pointer<none>>
                        %238 = pop.pointer.bitcast %237 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                        %239 = pop.offset %238[%idx-8] : !kgen.pointer<scalar<ui8>>
                        %240 = pop.pointer.bitcast %239 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                        %241 = kgen.struct.gep %240[0] : <struct<(scalar<index>) memoryOnly>>
                        %242 = pop.atomic.rmw sub(%241, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                        %243 = pop.cmp eq(%242, %simd_28) : <1, index>
                        %244 = pop.cast_to_builtin %243 : !pop.scalar<bool> to i1
                        hlcf.if %244 {
                          pop.fence syncscope("") acquire
                          pop.aligned_free %239 : <scalar<ui8>>
                          hlcf.yield
                        } else {
                          hlcf.yield
                        }
                        hlcf.yield
                      } else {
                        hlcf.yield
                      }
                      llvm.intr.trap
                      hlcf.loop "_loop_0" {
                        hlcf.continue "_loop_0"
                      }
                      kgen.unreachable
                    } else {
                      hlcf.yield
                    }
                    %201 = pop.load %155 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
                    hlcf.yield %201 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
                  } else {
                    hlcf.yield %35 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
                  }
                  kgen.call @"std::builtin::debug_assert::_debug_assert_msg[LITImmutOrigin,::Origin[::Bool(False), $0]](::UnsafePointer[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1, ::AddressSpace(::Int(0))],::Int,::SourceLocation)"(%171, %188) : (!kgen.pointer<none>, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>) -> ()
                  hlcf.yield
                }
                %156 = pop.load %143 : !kgen.pointer<index>
                %157 = index.and %156, %index4611686018427387904
                %158 = index.cmp ne(%157, %index0)
                hlcf.if %158 {
                  %164 = pop.load %142 : !kgen.pointer<pointer<none>>
                  %165 = pop.pointer.bitcast %164 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                  %166 = pop.offset %165[%idx-8] : !kgen.pointer<scalar<ui8>>
                  %167 = pop.pointer.bitcast %166 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                  %168 = kgen.struct.gep %167[0] : <struct<(scalar<index>) memoryOnly>>
                  %169 = pop.atomic.rmw sub(%168, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                  %170 = pop.cmp eq(%169, %simd_28) : <1, index>
                  %171 = pop.cast_to_builtin %170 : !pop.scalar<bool> to i1
                  hlcf.if %171 {
                    pop.fence syncscope("") acquire
                    pop.aligned_free %166 : <scalar<ui8>>
                    hlcf.yield
                  } else {
                    hlcf.yield
                  }
                  hlcf.yield
                } else {
                  hlcf.yield
                }
                %159 = pop.load %147 : !kgen.pointer<index>
                %160 = index.and %159, %index4611686018427387904
                %161 = index.cmp ne(%160, %index0)
                hlcf.if %161 {
                  %164 = pop.load %146 : !kgen.pointer<pointer<none>>
                  %165 = pop.pointer.bitcast %164 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                  %166 = pop.offset %165[%idx-8] : !kgen.pointer<scalar<ui8>>
                  %167 = pop.pointer.bitcast %166 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                  %168 = kgen.struct.gep %167[0] : <struct<(scalar<index>) memoryOnly>>
                  %169 = pop.atomic.rmw sub(%168, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                  %170 = pop.cmp eq(%169, %simd_28) : <1, index>
                  %171 = pop.cast_to_builtin %170 : !pop.scalar<bool> to i1
                  hlcf.if %171 {
                    pop.fence syncscope("") acquire
                    pop.aligned_free %166 : <scalar<ui8>>
                    hlcf.yield
                  } else {
                    hlcf.yield
                  }
                  hlcf.yield
                } else {
                  hlcf.yield
                }
                %162 = pop.array.gep %89[%122#1] : <array<21, scalar<ui8>>>
                %163 = pop.load %162 : !kgen.pointer<scalar<ui8>>
                kgen.call @"std::builtin::simd::SIMD::write_to[::Writer](::SIMD[$0, $1],$2&),dtype=ui8,size=1,writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>]"(%163, %arg0) : (!pop.scalar<ui8>, !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut) -> ()
                hlcf.continue "_loop_3" %128, %122#0 : i1, index
              }
              %112 = index.sub %90#1, %90#0
              %113 = index.add %112, %index1
              %114 = index.maxs %113, %index0
              hlcf.loop "_loop_4" (%arg2 = %114 : index) {
                %116 = index.sub %114, %arg2
                %117 = index.sub %arg2, %index1
                %118 = index.cmp eq(%arg2, %index0)
                %119:2 = lit.try "try2" -> index, index {
                  %137 = pop.select %118, %arg2, %117 : index
                  hlcf.if %118 {
                    lit.try.raise "try2" %137, %116 : index, index
                  } else {
                    hlcf.yield
                  }
                  lit.try.yield %137, %116 : index, index
                } except (%arg3: index, %arg4: index) {
                  hlcf.break "_loop_4"
                } else (%arg3: index, %arg4: index) {
                  lit.try.yield %arg3, %arg4 : index, index
                }
                %120 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
                pop.stack_alloc.lifetime.start(%120) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                %121 = kgen.struct.gep %120[1] : <struct<(pointer<none>, index, index) memoryOnly>>
                pop.store %index1, %121 : !kgen.pointer<index>
                %122 = kgen.struct.gep %120[0] : <struct<(pointer<none>, index, index) memoryOnly>>
                pop.store %44, %122 : !kgen.pointer<pointer<none>>
                %123 = kgen.struct.gep %120[2] : <struct<(pointer<none>, index, index) memoryOnly>>
                pop.store %index2305843009213693952, %123 : !kgen.pointer<index>
                %124 = pop.pointer.bitcast %120 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
                %125 = pop.load %123 : !kgen.pointer<index>
                %126 = index.and %125, %index-9223372036854775808
                %127 = index.cmp ne(%126, %index0)
                %128 = hlcf.if %127 -> !kgen.pointer<none> {
                  hlcf.yield %124 : !kgen.pointer<none>
                } else {
                  %137 = pop.load %122 : !kgen.pointer<pointer<none>>
                  hlcf.yield %137 : !kgen.pointer<none>
                }
                %129 = pop.load %123 : !kgen.pointer<index>
                %130 = index.and %129, %index-9223372036854775808
                %131 = index.cmp ne(%130, %index0)
                %132 = hlcf.if %131 -> index {
                  %137 = pop.load %123 : !kgen.pointer<index>
                  %138 = index.and %137, %index2233785415175766016
                  %139 = index.shrs %138, %index56
                  hlcf.yield %139 : index
                } else {
                  %137 = pop.load %121 : !kgen.pointer<index>
                  hlcf.yield %137 : index
                }
                %133 = kgen.struct.create(%128, %132) : !kgen.struct<(pointer<none>, index)>
                kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg0, %133) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
                %134 = pop.load %123 : !kgen.pointer<index>
                %135 = index.and %134, %index4611686018427387904
                %136 = index.cmp ne(%135, %index0)
                hlcf.if %136 {
                  %137 = pop.load %122 : !kgen.pointer<pointer<none>>
                  %138 = pop.pointer.bitcast %137 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                  %139 = pop.offset %138[%idx-8] : !kgen.pointer<scalar<ui8>>
                  %140 = pop.pointer.bitcast %139 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                  %141 = kgen.struct.gep %140[0] : <struct<(scalar<index>) memoryOnly>>
                  %142 = pop.atomic.rmw sub(%141, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                  %143 = pop.cmp eq(%142, %simd_28) : <1, index>
                  %144 = pop.cast_to_builtin %143 : !pop.scalar<bool> to i1
                  hlcf.if %144 {
                    pop.fence syncscope("") acquire
                    pop.aligned_free %139 : <scalar<ui8>>
                    hlcf.yield
                  } else {
                    hlcf.yield
                  }
                  hlcf.yield
                } else {
                  hlcf.yield
                }
                pop.stack_alloc.lifetime.end(%120) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                hlcf.continue "_loop_4" %119#0 : index
              }
              %115 = pop.xor %111, %0
              hlcf.if %115 {
                %116 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
                pop.stack_alloc.lifetime.start(%116) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                %117 = kgen.struct.gep %116[1] : <struct<(pointer<none>, index, index) memoryOnly>>
                pop.store %index2, %117 : !kgen.pointer<index>
                %118 = kgen.struct.gep %116[0] : <struct<(pointer<none>, index, index) memoryOnly>>
                pop.store %40, %118 : !kgen.pointer<pointer<none>>
                %119 = kgen.struct.gep %116[2] : <struct<(pointer<none>, index, index) memoryOnly>>
                pop.store %index2305843009213693952, %119 : !kgen.pointer<index>
                %120 = pop.pointer.bitcast %116 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
                %121 = pop.load %119 : !kgen.pointer<index>
                %122 = index.and %121, %index-9223372036854775808
                %123 = index.cmp ne(%122, %index0)
                %124 = hlcf.if %123 -> !kgen.pointer<none> {
                  hlcf.yield %120 : !kgen.pointer<none>
                } else {
                  %133 = pop.load %118 : !kgen.pointer<pointer<none>>
                  hlcf.yield %133 : !kgen.pointer<none>
                }
                %125 = pop.load %119 : !kgen.pointer<index>
                %126 = index.and %125, %index-9223372036854775808
                %127 = index.cmp ne(%126, %index0)
                %128 = hlcf.if %127 -> index {
                  %133 = pop.load %119 : !kgen.pointer<index>
                  %134 = index.and %133, %index2233785415175766016
                  %135 = index.shrs %134, %index56
                  hlcf.yield %135 : index
                } else {
                  %133 = pop.load %117 : !kgen.pointer<index>
                  hlcf.yield %133 : index
                }
                %129 = kgen.struct.create(%124, %128) : !kgen.struct<(pointer<none>, index)>
                kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg0, %129) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
                %130 = pop.load %119 : !kgen.pointer<index>
                %131 = index.and %130, %index4611686018427387904
                %132 = index.cmp ne(%131, %index0)
                hlcf.if %132 {
                  %133 = pop.load %118 : !kgen.pointer<pointer<none>>
                  %134 = pop.pointer.bitcast %133 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
                  %135 = pop.offset %134[%idx-8] : !kgen.pointer<scalar<ui8>>
                  %136 = pop.pointer.bitcast %135 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
                  %137 = kgen.struct.gep %136[0] : <struct<(scalar<index>) memoryOnly>>
                  %138 = pop.atomic.rmw sub(%137, %simd_28) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
                  %139 = pop.cmp eq(%138, %simd_28) : <1, index>
                  %140 = pop.cast_to_builtin %139 : !pop.scalar<bool> to i1
                  hlcf.if %140 {
                    pop.fence syncscope("") acquire
                    pop.aligned_free %135 : <scalar<ui8>>
                    hlcf.yield
                  } else {
                    hlcf.yield
                  }
                  hlcf.yield
                } else {
                  hlcf.yield
                }
                pop.stack_alloc.lifetime.end(%116) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
                hlcf.yield
              } else {
                hlcf.yield
              }
              hlcf.yield
            }
            hlcf.yield
          }
          hlcf.yield
        }
        hlcf.yield
      }
      hlcf.yield
    }
    kgen.return
  }
  kgen.func @"std::builtin::_format_float::_to_decimal[::DType,::DType](::SIMD[$0, ::Int(1)]&,::Int&),CarrierDType=ui64,dtype=f64"(%arg0: !pop.scalar<ui64>, %arg1: index) -> (!pop.scalar<ui64>, index) {
    %simd = kgen.param.constant: scalar<ui64> = <1000>
    %index-1074 = kgen.param.constant = <-1074>
    %simd_0 = kgen.param.constant: scalar<ui64> = <9007199254740992>
    %index1 = kgen.param.constant = <1>
    %simd_1 = kgen.param.constant: scalar<ui64> = <10>
    %simd_2 = kgen.param.constant: scalar<ui64> = <1>
    %0 = kgen.param.constant: i1 = <0>
    %index-292 = kgen.param.constant = <-292>
    %index21 = kgen.param.constant = <21>
    %index261663 = kgen.param.constant = <261663>
    %index631305 = kgen.param.constant = <631305>
    %index-1075 = kgen.param.constant = <-1075>
    %index0 = kgen.param.constant = <0>
    %simd_3 = kgen.param.constant: scalar<ui64> = <2>
    %index-1 = kgen.param.constant = <-1>
    %1 = kgen.param.constant: i1 = <1>
    %simd_4 = kgen.param.constant: scalar<ui64> = <0>
    %index2 = kgen.param.constant = <2>
    %index19 = kgen.param.constant = <19>
    %false = index.bool.constant false
    %simd_5 = kgen.param.constant: scalar<ui64> = <2576980378>
    %simd_6 = kgen.param.constant: scalar<ui64> = <429496729>
    %simd_7 = kgen.param.constant: scalar<ui64> = <32>
    %index315653 = kgen.param.constant = <315653>
    %index20 = kgen.param.constant = <20>
    %simd_8 = kgen.param.constant: scalar<ui64> = <3332894622>
    %simd_9 = kgen.param.constant: scalar<ui64> = <1099511627>
    %simd_10 = kgen.param.constant: scalar<ui64> = <8>
    %idx-1741647 = index.constant -1741647
    %idx3 = index.constant 3
    %simd_11 = kgen.param.constant: scalar<ui64> = <50>
    %2 = pop.mul %arg0, %simd_3 : !pop.scalar<ui64>
    %3 = pop.simd.or %2, %simd_0 : <1, ui64>
    %4 = pop.cmp eq(%2, %simd_4) : <1, ui64>
    %5 = pop.cast_to_builtin %4 : !pop.scalar<bool> to i1
    %6 = index.add %arg1, %index-1075
    %7 = index.cmp sge(%6, %index2)
    %8 = index.mul %6, %index631305
    %9 = index.sub %8, %index261663
    %10 = index.shrs %9, %index21
    %11 = index.add %10, %index1
    %12 = index.mul %10, %idx-1741647
    %13 = index.shrs %12, %index19
    %14 = index.add %6, %13
    %15 = index.mul %10, %index-1
    %16 = index.sub %15, %index-292
    %17 = index.cmp ne(%arg1, %index0)
    %18 = pop.select %17, %3, %2 : !pop.scalar<ui64>
    %19 = pop.select %17, %6, %index-1074 : index
    hlcf.if %17 {
      hlcf.if %5 {
        %68 = kgen.call tail @"std::builtin::_format_float::_compute_endpoint[::DType,::Int,::Int,::Int](::Int,::Int,::Bool),CarrierDType=ui64,sig_bits=52,total_bits=64,cache_bits=128"(%16, %14, %1) : (index, index, i1) -> !pop.scalar<ui64>
        %69 = kgen.call tail @"std::builtin::_format_float::_compute_endpoint[::DType,::Int,::Int,::Int](::Int,::Int,::Bool),CarrierDType=ui64,sig_bits=52,total_bits=64,cache_bits=128"(%16, %14, %0) : (index, index, i1) -> !pop.scalar<ui64>
        %70 = hlcf.if %7 -> i1 {
          %100 = kgen.call @"std::builtin::_format_float::_case_shorter_interval_left_endpoint_upper_threshold[::DType,::Int](),CarrierDType=ui64,sig_bits=52"() : () -> index
          %101 = index.cmp sle(%6, %100)
          hlcf.yield %101 : i1
        } else {
          hlcf.yield %false : i1
        }
        %71 = pop.xor %70, %1
        %72 = pop.add %68, %simd_2 : !pop.scalar<ui64>
        %73 = pop.select %71, %72, %68 : !pop.scalar<ui64>
        %74 = pop.shr %69, %simd_7 : !pop.scalar<ui64>
        %75 = pop.cast fast %74 : !pop.scalar<ui64> to !pop.scalar<ui32>
        %76 = pop.cast fast %69 : !pop.scalar<ui64> to !pop.scalar<ui32>
        %77 = pop.cast fast %75 : !pop.scalar<ui32> to !pop.scalar<ui64>
        %78 = pop.mul %77, %simd_6 : !pop.scalar<ui64>
        %79 = pop.cast fast %76 : !pop.scalar<ui32> to !pop.scalar<ui64>
        %80 = pop.mul %79, %simd_6 : !pop.scalar<ui64>
        %81 = pop.mul %77, %simd_5 : !pop.scalar<ui64>
        %82 = pop.mul %79, %simd_5 : !pop.scalar<ui64>
        %83 = pop.shr %82, %simd_7 : !pop.scalar<ui64>
        %84 = pop.cast fast %81 : !pop.scalar<ui64> to !pop.scalar<ui32>
        %85 = pop.cast fast %84 : !pop.scalar<ui32> to !pop.scalar<ui64>
        %86 = pop.add %83, %85 : !pop.scalar<ui64>
        %87 = pop.cast fast %80 : !pop.scalar<ui64> to !pop.scalar<ui32>
        %88 = pop.cast fast %87 : !pop.scalar<ui32> to !pop.scalar<ui64>
        %89 = pop.add %86, %88 : !pop.scalar<ui64>
        %90 = pop.shr %89, %simd_7 : !pop.scalar<ui64>
        %91 = pop.add %78, %90 : !pop.scalar<ui64>
        %92 = pop.shr %81, %simd_7 : !pop.scalar<ui64>
        %93 = pop.add %91, %92 : !pop.scalar<ui64>
        %94 = pop.shr %80, %simd_7 : !pop.scalar<ui64>
        %95 = pop.add %93, %94 : !pop.scalar<ui64>
        %96 = pop.mul %95, %simd_1 : !pop.scalar<ui64>
        %97 = pop.cmp ge(%96, %73) : <1, ui64>
        %98 = pop.cast_to_builtin %97 : !pop.scalar<bool> to i1
        %99:2 = hlcf.if %98 -> !pop.scalar<ui64>, index {
          %100:2 = kgen.call tail @"std::builtin::_format_float::_remove_trailing_zeros[::DType](::SIMD[$0, ::Int(1)]&,::Int&),CarrierDType=ui64"(%95, %11) : (!pop.scalar<ui64>, index) -> (!pop.scalar<ui64>, index)
          hlcf.yield %100#0, %100#1 : !pop.scalar<ui64>, index
        } else {
          %100 = kgen.call tail @"std::builtin::_format_float::_compute_round_up_for_shorter_interval_case[::DType,::Int,::Int,::Int](::Int,::Int),CarrierDType=ui64,total_bits=64,sig_bits=52,cache_bits=128"(%16, %14) : (index, index) -> !pop.scalar<ui64>
          %101 = pop.cmp lt(%100, %73) : <1, ui64>
          %102 = pop.cast_to_builtin %101 : !pop.scalar<bool> to i1
          %103 = pop.add %100, %simd_2 : !pop.scalar<ui64>
          %104 = pop.select %102, %103, %100 : !pop.scalar<ui64>
          hlcf.yield %104, %10 : !pop.scalar<ui64>, index
        }
        kgen.return %99#0, %99#1 : !pop.scalar<ui64>, index
      } else {
        hlcf.yield
      }
      hlcf.yield
    } else {
      hlcf.yield
    }
    %20 = pop.sub %18, %simd_2 : !pop.scalar<ui64>
    %21 = index.mul %19, %index315653
    %22 = index.shrs %21, %index20
    %23 = index.sub %22, %index2
    %24 = index.add %23, %idx3
    %25 = index.add %23, %index2
    %26 = index.mul %23, %index-1
    %27 = index.mul %23, %idx-1741647
    %28 = index.shrs %27, %index19
    %29 = index.add %19, %28
    %30 = index.sub %26, %index-292
    %31 = kgen.call tail @"std::builtin::_format_float::_compute_delta[::DType,::Int,::Int](::Int,::Int),CarrierDType=ui64,total_bits=64,cache_bits=128"(%30, %29) : (index, index) -> !pop.scalar<ui64>
    %32 = pop.simd.or %18, %simd_2 : <1, ui64>
    %33 = pop.cast_from_builtin %29 : index to !pop.scalar<index>
    %34 = pop.cast %33 : !pop.scalar<index> to !pop.scalar<ui64>
    %35 = pop.shl %32, %34 : !pop.scalar<ui64>
    %36 = kgen.call tail @"std::builtin::_format_float::_compute_mul[::DType](::SIMD[$0, ::Int(1)],::Int),CarrierDType=ui64"(%35, %30) : (!pop.scalar<ui64>, index) -> !kgen.struct<(scalar<ui64>, i1)>
    %37 = kgen.struct.extract %36[0] : <(scalar<ui64>, i1)>
    %38 = pop.shr %37, %simd_7 : !pop.scalar<ui64>
    %39 = pop.cast fast %38 : !pop.scalar<ui64> to !pop.scalar<ui32>
    %40 = pop.cast fast %37 : !pop.scalar<ui64> to !pop.scalar<ui32>
    %41 = pop.cast fast %39 : !pop.scalar<ui32> to !pop.scalar<ui64>
    %42 = pop.mul %41, %simd_9 : !pop.scalar<ui64>
    %43 = pop.cast fast %40 : !pop.scalar<ui32> to !pop.scalar<ui64>
    %44 = pop.mul %43, %simd_9 : !pop.scalar<ui64>
    %45 = pop.mul %41, %simd_8 : !pop.scalar<ui64>
    %46 = pop.mul %43, %simd_8 : !pop.scalar<ui64>
    %47 = pop.shr %46, %simd_7 : !pop.scalar<ui64>
    %48 = pop.cast fast %45 : !pop.scalar<ui64> to !pop.scalar<ui32>
    %49 = pop.cast fast %48 : !pop.scalar<ui32> to !pop.scalar<ui64>
    %50 = pop.add %47, %49 : !pop.scalar<ui64>
    %51 = pop.cast fast %44 : !pop.scalar<ui64> to !pop.scalar<ui32>
    %52 = pop.cast fast %51 : !pop.scalar<ui32> to !pop.scalar<ui64>
    %53 = pop.add %50, %52 : !pop.scalar<ui64>
    %54 = pop.shr %53, %simd_7 : !pop.scalar<ui64>
    %55 = pop.add %42, %54 : !pop.scalar<ui64>
    %56 = pop.shr %45, %simd_7 : !pop.scalar<ui64>
    %57 = pop.add %55, %56 : !pop.scalar<ui64>
    %58 = pop.shr %44, %simd_7 : !pop.scalar<ui64>
    %59 = pop.add %57, %58 : !pop.scalar<ui64>
    %60 = pop.shr %59, %simd_10 : !pop.scalar<ui64>
    %61 = pop.mul %60, %simd : !pop.scalar<ui64>
    %62 = pop.sub %37, %61 : !pop.scalar<ui64>
    %63 = pop.cmp eq(%62, %31) : <1, ui64>
    %64 = pop.cast_to_builtin %63 : !pop.scalar<bool> to i1
    %65 = pop.cmp lt(%62, %31) : <1, ui64>
    %66 = pop.cast_to_builtin %65 : !pop.scalar<bool> to i1
    %67:2 = hlcf.if %66 -> !pop.scalar<ui64>, index {
      %68:2 = kgen.call tail @"std::builtin::_format_float::_remove_trailing_zeros[::DType](::SIMD[$0, ::Int(1)]&,::Int&),CarrierDType=ui64"(%60, %24) : (!pop.scalar<ui64>, index) -> (!pop.scalar<ui64>, index)
      hlcf.yield %68#0, %68#1 : !pop.scalar<ui64>, index
    } else {
      hlcf.if %64 {
        %80 = kgen.call tail @"std::builtin::_format_float::_compute_mul_parity[::DType](::SIMD[$0, ::Int(1)],::Int,::Int),CarrierDType=ui64"(%20, %30, %29) : (!pop.scalar<ui64>, index, index) -> !kgen.struct<(i1, i1)>
        %81 = kgen.struct.extract %80[0] : <(i1, i1)>
        %82 = kgen.struct.extract %80[1] : <(i1, i1)>
        %83 = pop.or %81, %82
        hlcf.if %83 {
          %84:2 = kgen.call tail @"std::builtin::_format_float::_remove_trailing_zeros[::DType](::SIMD[$0, ::Int(1)]&,::Int&),CarrierDType=ui64"(%60, %24) : (!pop.scalar<ui64>, index) -> (!pop.scalar<ui64>, index)
          kgen.return %84#0, %84#1 : !pop.scalar<ui64>, index
        } else {
          hlcf.yield
        }
        hlcf.yield
      } else {
        hlcf.yield
      }
      %68 = pop.mul %60, %simd_1 : !pop.scalar<ui64>
      %69 = pop.floordiv %31, %simd_3 : !pop.scalar<ui64>
      %70 = pop.sub %62, %69 : !pop.scalar<ui64>
      %71 = pop.add %70, %simd_11 : !pop.scalar<ui64>
      %72 = pop.simd.xor %71, %simd_11 : <1, ui64>
      %73 = pop.simd.and %72, %simd_2 : <1, ui64>
      %74 = pop.cmp eq(%73, %simd_4) : <1, ui64>
      %75 = pop.cast_to_builtin %74 : !pop.scalar<bool> to i1
      %76 = pop.xor %75, %1
      %77:2 = kgen.call @"std::builtin::_format_float::_check_divisibility_and_divide_by_pow10[::DType,::Int,::InlineArray[::SIMD[::DType(uint32), ::Int(1)], ::Int(2)]](::SIMD[$0, ::Int(1)]&,::Int),CarrierDType=ui64,carrier_bits=64,divide_magic_number={ [6554, 656] }"(%71, %index2) : (!pop.scalar<ui64>, index) -> (i1, !pop.scalar<ui64>)
      %78 = pop.add %68, %77#1 : !pop.scalar<ui64>
      %79 = hlcf.if %77#0 -> !pop.scalar<ui64> {
        %80 = kgen.call tail @"std::builtin::_format_float::_compute_mul_parity[::DType](::SIMD[$0, ::Int(1)],::Int,::Int),CarrierDType=ui64"(%18, %30, %29) : (!pop.scalar<ui64>, index, index) -> !kgen.struct<(i1, i1)>
        %81 = kgen.struct.extract %80[0] : <(i1, i1)>
        %82 = pop.xor %81, %76
        %83 = pop.sub %78, %simd_2 : !pop.scalar<ui64>
        %84 = pop.select %82, %83, %78 : !pop.scalar<ui64>
        hlcf.yield %84 : !pop.scalar<ui64>
      } else {
        hlcf.yield %78 : !pop.scalar<ui64>
      }
      hlcf.yield %79, %25 : !pop.scalar<ui64>, index
    }
    kgen.return %67#0, %67#1 : !pop.scalar<ui64>, index
  }
  kgen.func @"std::builtin::_format_float::_compute_endpoint[::DType,::Int,::Int,::Int](::Int,::Int,::Bool),CarrierDType=ui64,sig_bits=52,total_bits=64,cache_bits=128"(%arg0: index, %arg1: index, %arg2: i1) -> !pop.scalar<ui64> {
    %string = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/builtin/_format_float.mojo">
    %simd = kgen.param.constant: scalar<uindex> = <619>
    %idx618 = index.constant 618
    %index54 = kgen.param.constant = <54>
    %index49 = kgen.param.constant = <49>
    %index477 = kgen.param.constant = <477>
    %simd_0 = kgen.param.constant: scalar<ui128> = <64>
    %string_1 = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/collections/check_bounds.mojo">
    %index57 = kgen.param.constant = <57>
    %string_2 = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/builtin/debug_assert.mojo">
    %index27 = kgen.param.constant = <27>
    %index330 = kgen.param.constant = <330>
    %index53 = kgen.param.constant = <53>
    %string_3 = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/collections/optional.mojo">
    %index18 = kgen.param.constant = <18>
    %index610 = kgen.param.constant = <610>
    %idx-8 = index.constant -8
    %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle, const_global, [], [])], []}, 0, 0>, 1 }>
    %struct_4 = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle4, const_global, [], [])], []}, 0, 0>, 6 }>
    %string_5 = kgen.param.constant: string = <" ">
    %index2 = kgen.param.constant = <2>
    %string_6 = kgen.param.constant: string = <": ">
    %string_7 = kgen.param.constant: string = <"">
    %string_8 = kgen.param.constant: string = <"`Optional.value()` called on empty `Optional`. Consider using `if optional:` to check whether the `Optional` is empty before calling `.value()`, or use `.or_else()` to provide a default value.">
    %index192 = kgen.param.constant = <192>
    %simd_9 = kgen.param.constant: scalar<ui8> = <0>
    %index2048 = kgen.param.constant = <2048>
    %0 = kgen.param.constant: i1 = <1>
    %index-1 = kgen.param.constant = <-1>
    %index0 = kgen.param.constant = <0>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %simd_10 = kgen.param.constant: scalar<index> = <1>
    %string_11 = kgen.param.constant: string = <" is out of bounds, valid range is 0 to ">
    %index39 = kgen.param.constant = <39>
    %string_12 = kgen.param.constant: string = <"index ">
    %index6 = kgen.param.constant = <6>
    %index2305843009213693952 = kgen.param.constant = <2305843009213693952>
    %index1 = kgen.param.constant = <1>
    %none = kgen.param.constant: none = <#kgen.none>
    %array = kgen.param.constant: array<1, struct<(index, index, struct<(pointer<none>, index)>)>> = <[{ -1, 0, { 0, 0 } }]>
    %simd_13 = kgen.param.constant: scalar<ui64> = <54>
    %index11 = kgen.param.constant = <11>
    %simd_14 = kgen.param.constant: scalar<ui64> = <53>
    %1 = pop.string.address %string
    %2 = pop.pointer.bitcast %1 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %3 = kgen.struct.create(%2, %index54) : !kgen.struct<(pointer<none>, index)>
    %4 = kgen.struct.create(%index477, %index49, %3) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %5 = index.sub %index11, %arg1
    %6 = pop.cast_from_builtin %5 : index to !pop.scalar<index>
    %7 = pop.cast %6 : !pop.scalar<index> to !pop.scalar<ui64>
    %8 = pop.global_constant: struct<(array<619, scalar<ui128>>) memoryOnly> = <{ [339574632262346462319337857089816694651, 212234145163966538949586160681135434157, 265292681454958173686982700851419292696, 331615851818697717108728376064274115870, 207259907386686073192955235040171322419, 259074884233357591491194043800214153024, 323843605291696989363992554750267691280, 202402253307310618352495346718917307050, 253002816634138272940619183398646633812, 316253520792672841175773979248308292265, 197658450495420525734858737030192682666, 247073063119275657168573421287740853332, 308841328899094571460716776609676066665, 193025830561934107162947985381047541666, 241282288202417633953684981726309427083, 301602860253022042442106227157886783853, 188501787658138776526316391973679239908, 235627234572673470657895489967099049885, 294534043215841838322369362458873812356, 184083777009901148951480851536796132723, 230104721262376436189351064420995165904, 287630901577970545236688830526243957379, 179769313486231590772930519078902473362, 224711641857789488466163148848628091703, 280889552322236860582703936060785114628, 175555970201398037864189960037990696643, 219444962751747547330237450047488370803, 274306203439684434162796812559360463504, 171441377149802771351748007849600289690, 214301721437253464189685009812000362113, 267877151796566830237106262265000452641, 334846439745708537796382827831250565801, 209279024841067836122739267394531603626, 261598781051334795153424084243164504532, 326998476314168493941780105303955630665, 204374047696355308713612565814972269166, 255467559620444135892015707268715336457, 319334449525555169865019634085894170571, 199584030953471981165637271303683856607, 249480038691839976457046589129604820759, 311850048364799970571308236412006025949, 194906280227999981607067647757503766218, 243632850284999977008834559696879707772, 304541062856249971261043199621099634715, 190338164285156232038151999763187271697, 237922705356445290047689999703984089622, 297403381695556612559612499629980112027, 185877113559722882849757812268737570017, 232346391949653603562197265335921962521, 290432989937067004452746581669902453151, 181520618710666877782966613543689033220, 226900773388333597228708266929611291524, 283625966735416996535885333662014114405, 177266229209635622834928333538758821504, 221582786512044528543660416923448526879, 276978483140055660679575521154310658599, 173111551962534787924734700721444161625, 216389439953168484905918375901805202031, 270486799941460606132397969877256502538, 338108499926825757665497462346570628173, 211317812454266098540935913966606642608, 264147265567832623176169892458258303260, 330184081959790778970212365572822879075, 206365051224869236856382728483014299422, 257956314031086546070478410603767874277, 322445392538858182588098013254709842846, 201528370336786364117561258284193651779, 251910462920982955146951572855242064724, 314888078651228693933689466069052580905, 196805049157017933708555916293157863066, 246006311446272417135694895366447328832, 307507889307840521419618619208059161040, 192192430817400325887261637005036975650, 240240538521750407359077046256296219562, 300300673152188009198846307820370274453, 187687920720117505749278942387731421533, 234609900900146882186598677984664276916, 293262376125183602733248347480830346145, 183288985078239751708280217175518966341, 229111231347799689635350271469398707926, 286389039184749612044187839336748384908, 178993149490468507527617399585467740568, 223741436863085634409521749481834675709, 279676796078857043011902186852293344636, 174797997549285651882438866782683340398, 218497496936607064853048583478354175497, 273121871170758831066310729347942719372, 170701169481724269416444205842464199607, 213376461852155336770555257303080249509, 266720577315194170963194071628850311886, 333400721643992713703992589536062889858, 208375451027495446064995368460039306161, 260469313784369307581244210575049132701, 325586642230461634476555263218811415877, 203491651394038521547847039511757134923, 254364564242548151934808799389696418654, 317955705303185189918510999237120523317, 198722315814490743699069374523200327073, 248402894768113429623836718154000408842, 310503618460141787029795897692500511052, 194064761537588616893622436057812819408, 242580951921985771117028045072266024259, 303226189902482213896285056340332530324, 189516368689051383685178160212707831453, 236895460861314229606472700265884789316, 296119326076642787008090875332355986645, 185074578797901741880056797082722491653, 231343223497377177350070996353403114566, 289179029371721471687588745441753893208, 180736893357325919804742965901096183255, 225921116696657399755928707376370229069, 282401395870821749694910884220462786336, 176500872419263593559319302637789241460, 220626090524079491949149128297236551825, 275782613155099364936436410371545689781, 172364133221937103085272756482216056113, 215455166527421378856590945602770070141, 269318958159276723570738682003462587677, 336648697699095904463423352504328234596, 210405436061934940289639595315205146623, 263006795077418675362049494144006433278, 328758493846773344202561867680008041597, 205474058654233340126601167300005025999, 256842573317791675158251459125006282498, 321053216647239593947814323906257853122, 200658260404524746217383952441411158202, 250822825505655932771729940551763947752, 313528531882069915964662425689704934690, 195955332426293697477914016056065584181, 244944165532867121847392520070081980227, 306180206916083902309240650087602475283, 191362629322552438943275406304751547052, 239203286653190548679094257880939433815, 299004108316488185848867822351174292269, 186877567697805116155542388969483932668, 233596959622256395194427986211854915835, 291996199527820493993034982764818644794, 182497624704887808745646864228011652996, 228122030881109760932058580285014566245, 285152538601387201165073225356268207806, 178220336625867000728170765847667629879, 222775420782333750910213457309584537349, 278469275977917188637766821636980671686, 174043297486198242898604263523112919804, 217554121857747803623255329403891149755, 271942652322184754529069161754863937193, 339928315402730943161336452193579921491, 212455197126706839475835282620987450932, 265568996408383549344794103276234313665, 331961245510479436680992629095292892081, 207475778444049647925620393184558057551, 259344723055062059907025491480697571939, 324180903818827574883781864350871964923, 202613064886767234302363665219294978077, 253266331108459042877954581524118722596, 316582913885573803597443226905148403245, 197864321178483627248402016815717752029, 247330401473104534060502521019647190036, 309163001841380667575628151274558987544, 193226876150862917234767594546599367215, 241533595188578646543459493183249209019, 301916993985723308179324366479061511274, 188698121241077067612077729049413444546, 235872651551346334515097161311766805683, 294840814439182918143871451639708507103, 184275509024489323839919657274817816940, 230344386280611654799899571593522271175, 287930482850764568499874464491902838968, 179956551781727855312421540307439274355, 224945689727159819140526925384299092944, 281182112158949773925658656730373866180, 175738820099343608703536660456483666363, 219673525124179510879420825570604582953, 274591906405224388599276031963255728691, 171619941503265242874547519977034830432, 214524926879081553593184399971293538040, 268156158598851941991480499964116922550, 335195198248564927489350624955146153187, 209496998905353079680844140596966345742, 261871248631691349601055175746207932178, 327339060789614187001318969682759915222, 204586912993508866875824356051724947014, 255733641241886083594780445064656183767, 319667051552357604493475556330820229709, 199791907220223502808422222706762643568, 249739884025279378510527778383453304460, 312174855031599223138159722979316630575, 195109284394749514461349826862072894110, 243886605493436893076687283577591117637, 304858256866796116345859104471988897046, 190536410541747572716161940294993060654, 238170513177184465895202425368741325818, 297713141471480582369003031710926657272, 186070713419675363980626894819329160795, 232588391774594204975783618524161450994, 290735489718242756219729523155201813742, 181709681073901722637330951972001133589, 227137101342377153296663689965001416986, 283921376677971441620829612456251771232, 177450860423732151013018507785157357020, 221813575529665188766273134731446696275, 277266969412081485957841418414308370344, 173291855882550928723650886508942731465, 216614819853188660904563608136178414331, 270768524816485826130704510170223017914, 338460656020607282663380637712778772393, 211537910012879551664612898570486732746, 264422387516099439580766123213108415932, 330527984395124299475957654016385519915, 206579990246952687172473533760240949947, 258224987808690858965591917200301187433, 322781234760863573706989896500376484292, 201738271725539733566868685312735302683, 252172839656924666958585856640919128353, 315216049571155833698232320801148910441, 197010030981972396061395200500718069026, 246262538727465495076744000625897586282, 307828173409331868845930000782371982853, 192392608380832418028706250488982489283, 240490760476040522535882813111228111604, 300613450595050653169853516389035139505, 187883406621906658231158447743146962191, 234854258277383322788948059678933702738, 293567822846729153486185074598667128422, 183479889279205720928865671624166955264, 229349861599007151161082089530208694080, 286687326998758938951352611912760867600, 179179579374224336844595382445475542250, 223974474217780421055744228056844427813, 279968092772225526319680285071055534766, 174980057982640953949800178169409709229, 218725072478301192437250222711762136536, 273406340597876490546562778389702670670, 170878962873672806591601736493564169169, 213598703592091008239502170616955211461, 266998379490113760299377713271194014326, 333747974362642200374222141588992517907, 208592483976651375233888838493120323692, 260740604970814219042361048116400404615, 325925756213517773802951310145500505769, 203703597633448608626844568840937816106, 254629497041810760783555711051172270132, 318286871302263450979444638813965337665, 198929294563914656862152899258728336041, 248661618204893321077691124073410420051, 310827022756116651347113905091763025063, 194266889222572907091946190682351890665, 242833611528216133864932738352939863331, 303542014410270167331165922941174829163, 189713759006418854581978701838234268227, 237142198758023568227473377297792835284, 296427748447529460284341721622241044105, 185267342779705912677713576013900652566, 231584178474632390847141970017375815707, 289480223093290488558927462521719769634, 180925139433306555349329664076074856021, 226156424291633194186662080095093570026, 282695530364541492733327600118866962533, 176684706477838432958329750074291851583, 220855883097298041197912187592864814479, 276069853871622551497390234491081018099, 172543658669764094685868896556925636312, 215679573337205118357336120696157045390, 269599466671506397946670150870196306737, 336999333339382997433337688587745383421, 210624583337114373395836055367340864638, 263280729171392966744795069209176080798, 329100911464241208430993836511470100997, 205688069665150755269371147819668813123, 257110087081438444086713934774586016404, 321387608851798055108392418468232520505, 200867255532373784442745261542645325316, 251084069415467230553431576928306656645, 313855086769334038191789471160383320806, 196159429230833773869868419475239575504, 245199286538542217337335524344049469379, 306499108173177771671669405430061836724, 191561942608236107294793378393788647953, 239452428260295134118491722992235809941, 299315535325368917648114653740294762426, 187072209578355573530071658587684226516, 233840261972944466912589573234605283145, 292300327466180583640736966543256603932, 182687704666362864775460604089535377457, 228359630832953580969325755111919221822, 285449538541191976211657193889899027277, 178405961588244985132285746181186892048, 223007451985306231415357182726483615060, 278759314981632789269196478408104518825, 174224571863520493293247799005065324266, 217780714829400616616559748756331655332, 272225893536750770770699685945414569165, 170141183460469231731687303715884105728, 212676479325586539664609129644855132160, 265845599156983174580761412056068915200, 332306998946228968225951765070086144000, 207691874341393105141219853168803840000, 259614842926741381426524816461004800000, 324518553658426726783156020576256000000, 202824096036516704239472512860160000000, 253530120045645880299340641075200000000, 316912650057057350374175801344000000000, 198070406285660843983859875840000000000, 247588007857076054979824844800000000000, 309485009821345068724781056000000000000, 193428131138340667952988160000000000000, 241785163922925834941235200000000000000, 302231454903657293676544000000000000000, 188894659314785808547840000000000000000, 236118324143482260684800000000000000000, 295147905179352825856000000000000000000, 184467440737095516160000000000000000000, 230584300921369395200000000000000000000, 288230376151711744000000000000000000000, 180143985094819840000000000000000000000, 225179981368524800000000000000000000000, 281474976710656000000000000000000000000, 175921860444160000000000000000000000000, 219902325555200000000000000000000000000, 274877906944000000000000000000000000000, 171798691840000000000000000000000000000, 214748364800000000000000000000000000000, 268435456000000000000000000000000000000, 335544320000000000000000000000000000000, 209715200000000000000000000000000000000, 262144000000000000000000000000000000000, 327680000000000000000000000000000000000, 204800000000000000000000000000000000000, 256000000000000000000000000000000000000, 320000000000000000000000000000000000000, 200000000000000000000000000000000000000, 250000000000000000000000000000000000000, 312500000000000000000000000000000000000, 195312500000000000000000000000000000000, 244140625000000000000000000000000000000, 305175781250000000000000000000000000000, 190734863281250000000000000000000000000, 238418579101562500000000000000000000000, 298023223876953125000000000000000000000, 186264514923095703125000000000000000000, 232830643653869628906250000000000000000, 291038304567337036132812500000000000000, 181898940354585647583007812500000000000, 227373675443232059478759765625000000000, 284217094304040074348449707031250000000, 177635683940025046467781066894531250000, 222044604925031308084726333618164062500, 277555756156289135105907917022705078125, 173472347597680709441192448139190673829, 216840434497100886801490560173988342286, 271050543121376108501863200217485427857, 338813178901720135627329000271856784821, 211758236813575084767080625169910490513, 264697796016968855958850781462388113142, 330872245021211069948563476827985141427, 206795153138256918717852173017490713392, 258493941422821148397315216271863391740, 323117426778526435496644020339829239675, 201948391736579022185402512712393274797, 252435489670723777731753140890491593496, 315544362088404722164691426113114491870, 197215226305252951352932141320696557419, 246519032881566189191165176650870696773, 308148791101957736488956470813588370967, 192592994438723585305597794258492731854, 240741243048404481631997242823115914818, 300926553810505602039996553528894893522, 188079096131566001274997845955559308451, 235098870164457501593747307444449135564, 293873587705571876992184134305561419455, 183670992315982423120115083940975887160, 229588740394978028900143854926219858949, 286985925493722536125179818657774823687, 179366203433576585078237386661109264804, 224207754291970731347796733326386581005, 280259692864963414184745916657983226257, 175162308040602133865466197911239516411, 218952885050752667331832747389049395513, 273691106313440834164790934236311744391, 171056941445900521352994333897694840245, 213821176807375651691242917372118550306, 267276471009219564614053646715148187882, 334095588761524455767567058393935234852, 208809742975952784854729411496209521783, 261012178719940981068411764370261902229, 326265223399926226335514705462827377786, 203915764624953891459696690914267111116, 254894705781192364324620863642833888895, 318618382226490455405776079553542361119, 199136488891556534628610049720963975699, 248920611114445668285762562151204969624, 311150763893057085357203202689006212030, 194469227433160678348252001680628882519, 243086534291450847935315002100786103149, 303858167864313559919143752625982628936, 189911354915195974949464845391239143085, 237389193643994968686831056739048928856, 296736492054993710858538820923811161070, 185460307534371069286586763077381975669, 231825384417963836608233453846727469586, 289781730522454795760291817308409336982, 181113581576534247350182385817755835614, 226391976970667809187727982272194794518, 282989971213334761484659977840243493147, 176868732008334225927912486150152183217, 221085915010417782409890607687690229021, 276357393763022228012363259609612786276, 172723371101888892507727037256007991423, 215904213877361115634658796570009989278, 269880267346701394543323495712512486598, 337350334183376743179154369640640608247, 210843958864610464486971481025400380155, 263554948580763080608714351281750475193, 329443685725953850760892939102188093991, 205902303578721156725558086938867558745, 257377879473401445906947608673584448431, 321722349341751807383684510841980560539, 201076468338594879614802819276237850337, 251345585423243599518503524095297312921, 314181981779054499398129405119121641151, 196363738611909062123830878199451025720, 245454673264886327654788597749313782149, 306818341581107909568485747186642227686, 191761463488192443480303591991651392304, 239701829360240554350379489989564240380, 299627286700300692937974362486955300475, 187267054187687933086233976554347062797, 234083817734609916357792470692933828496, 292604772168262395447240588366167285620, 182877982605163997154525367728854553513, 228597478256454996443156709661068191891, 285746847820568745553945887076335239863, 178591779887855465971216179422709524915, 223239724859819332464020224278386906143, 279049656074774165580025280347983632679, 174406035046733853487515800217489770425, 218007543808417316859394750271862213031, 272509429760521646074243437839827766288, 170318393600326028796402148649892353930, 212897992000407535995502685812365442413, 266122490000509419994378357265456803016, 332653112500636774992972946581821003770, 207908195312897984370608091613638127356, 259885244141122480463260114517047659195, 324856555176403100579075143146309573994, 203035346985251937861921964466443483746, 253794183731564922327402455583054354683, 317242729664456152909253069478817943353, 198276706040285095568283168424261214596, 247845882550356369460353960530326518245, 309807353187945461825442450662908147806, 193629595742465913640901531664317592379, 242036994678082392051126914580396990474, 302546243347602990063908643225496238092, 189091402092251868789942902015935148808, 236364252615314835987428627519918936009, 295455315769143544984285784399898670012, 184659572355714715615178615249936668757, 230824465444643394518973269062420835947, 288530581805804243148716586328026044933, 180331613628627651967947866455016278083, 225414517035784564959934833068770347604, 281768146294730706199918541335962934505, 176105091434206691374949088334976834066, 220131364292758364218686360418721042582, 275164205365947955273357950523401303228, 171977628353717472045848719077125814518, 214972035442146840057310898846407268147, 268715044302683550071638623558009085183, 335893805378354437589548279447511356479, 209933628361471523493467674654694597800, 262417035451839404366834593318368247249, 328021294314799255458543241647960309062, 205013308946749534661589526029975193164, 256266636183436918326986907537468991454, 320333295229296147908733634421836239318, 200208309518310092442958521513647649574, 250260386897887615553698151892059561967, 312825483622359519442122689865074452459, 195515927263974699651326681165671532787, 244394909079968374564158351457089415984, 305493636349960468205197939321361769979, 190933522718725292628248712075851106237, 238666903398406615785310890094813882797, 298333629248008269731638612618517353496, 186458518280005168582274132886573345935, 233073147850006460727842666108216682419, 291341434812508075909803332635270853023, 182088396757817547443627082897044283140, 227610495947271934304533853621305353924, 284513119934089917880667317026631692405, 177820699958806198675417073141644807754, 222275874948507748344271341427056009692, 277844843685634685430339176783820012115, 173653027303521678393961985489887507572, 217066284129402097992452481862359384465, 271332855161752622490565602327949230581, 339166068952190778113207002909936538226, 211978793095119236320754376818710336391, 264973491368899045400942971023387920489, 331216864211123806751178713779234900611, 207010540131952379219486696112021812882, 258763175164940474024358370140027266102, 323453968956175592530447962675034082628, 202158730597609745331529976671896301643, 252698413247012181664412470839870377053, 315873016558765227080515588549837971316, 197420635349228266925322242843648732073, 246775794186535333656652803554560915091, 308469742733169167070816004443201143864, 192793589208230729419260002777000714915, 240991986510288411774075003471250893644, 301239983137860514717593754339063617054, 188274989461162821698496096461914760659, 235343736826453527123120120577393450824, 294179671033066908903900150721741813530, 183862294395666818064937594201088633456, 229827867994583522581171992751360791820, 287284834993229403226464990939200989775, 179553021870768377016540619337000618610, 224441277338460471270675774171250773262, 280551596673075589088344717714063466577, 175344747920672243180215448571289666611, 219180934900840303975269310714112083264, 273976168626050379969086638392640104079, 171235105391281487480679148995400065050, 214043881739101859350848936244250081312, 267554852173877324188561170305312601640, 334443565217346655235701462881640752050, 209027228260841659522313414301025470031, 261284035326052074402891767876281837539, 326605044157565093003614709845352296924, 204128152598478183127259193653345185578, 255160190748097728909073992066681481972, 318950238435122161136342490083351852465, 199343899021951350710214056302094907791, 249179873777439188387767570377618634738, 311474842221798985484709462972023293422, 194671776388624365927943414357514558389, 243339720485780457409929267946893197986, 304174650607225571762411584933616497483, 190109156629515982351507240583510310927, 237636445786894977939384050729387888659, 297045557233618722424230063411734860823, 185653473271011701515143789632334288015, 232066841588764626893929737040417860018, 290083551985955783617412171300522325023, 181302219991222364760882607062826453139, 226627774989027955951103258828533066424, 283284718736284944938879073535666333030, 177052949210178090586799420959791458144, 221316186512722613233499276199739322680, 276645233140903266541874095249674153350, 172903270713064541588671309531046345844, 216129088391330676985839136913807932304, 270161360489163346232298921142259915380, 337701700611454182790373651427824894225, 211063562882158864243983532142390558891, 263829453602698580304979415177988198614, 329786817003373225381224268972485248267, 206116760627108265863265168107803280167, 257645950783885332329081460134754100209, 322057438479856665411351825168442625261, 201285899049910415882094890730276640788, 251607373812388019852618613412845800985, 314509217265485024815773266766057251231, 196568260790928140509858291728785782020, 245710325988660175637322864660982227524, 307137907485825219546653580826227784405, 191961192178640762216658488016392365254, 239951490223300952770823110020490456567, 299939362779126190963528887525613070708, 187462101736953869352205554703508169193, 234327627171192336690256943379385211491, 292909533963990420862821179224231514364, 183068458727494013039263237015144696478, 228835573409367516299079046268930870597, 286044466761709395373848807836163588246, 178777791726068372108655504897602242654, 223472239657585465135819381122002803317, 279340299571981831419774226402503504146, 174587687232488644637358891501564690092, 218234609040610805796698614376955862614, 272793261300763507245873267971194828268, 170495788312977192028670792481996767668, 213119735391221490035838490602495959584, 266399669239026862544798113253119949480, 332999586548783578180997641566399936850, 208124741592989736363123525978999960532, 260155926991237170453904407473749950665, 325194908739046463067380509342187438331, 203246817961904039417112818338867148957, 254058522452380049271391022923583936196, 317573153065475061589238778654479920245, 198483220665921913493274236659049950153, 248104025832402391866592795823812437691, 310130032290502989833240994779765547114, 193831270181564368645775621737353466946, 242289087726955460807219527171691833683, 302861359658694326009024408964614792103, 189288349786683953755640255602884245065, 236610437233354942194550319503605306331, 295763046541693677743187899379506632914, 184851904088558548589492437112191645571, 231064880110698185736865546390239556964, 288831100138372732171081932987799446205, 180519437586482957606926208117374653878, 225649296983103697008657760146718317347, 282061621228879621260822200183397896684, 176288513268049763288013875114623685428, 220360641585062204110017343893279606785, 275450801981327755137521679866599508481, 172156751238329846960951049916624692801, 215195939047912308701188812395780866001, 268994923809890385876486015494726082501, 336243654762362982345607519368407603126, 210152284226476863966004699605254751954, 262690355283096079957505874506568439942, 328362944103870099946882343133210549928] }>
    %9 = pop.string.address %string_1
    %10 = pop.pointer.bitcast %9 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %11 = kgen.struct.create(%10, %index57) : !kgen.struct<(pointer<none>, index)>
    %12 = kgen.struct.create(%index57, %index6, %11) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %13 = pop.cast_from_builtin %arg0 : index to !pop.scalar<index>
    %14 = pop.cast %13 : !pop.scalar<index> to !pop.scalar<uindex>
    %15 = pop.cmp lt(%14, %simd) : <1, uindex>
    %16 = pop.cast_to_builtin %15 : !pop.scalar<bool> to i1
    %17 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
    pop.stack_alloc.lifetime.start(%17) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
    pop.store %array, %17 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
    %18 = pop.pointer.bitcast %17 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<index>
    %19 = pop.load %18 : !kgen.pointer<index>
    %20 = index.cmp eq(%19, %index-1)
    %21 = pop.select %20, %index0, %index-1 : index
    pop.stack_alloc.lifetime.end(%17) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
    %22 = index.cmp eq(%21, %index-1)
    %23 = hlcf.if %22 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
      %79 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
      pop.stack_alloc.lifetime.start(%79) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
      pop.store %array, %79 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
      %80 = pop.pointer.bitcast %79 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
      %81 = pop.load %80 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
      pop.stack_alloc.lifetime.end(%79) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
      hlcf.yield %81 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    } else {
      hlcf.yield %4 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    }
    %24 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
    pop.stack_alloc.lifetime.start(%24) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %25 = kgen.struct.gep %24[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index6, %25 : !kgen.pointer<index>
    %26 = kgen.struct.gep %24[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %27 = pop.string.address %string_12
    %28 = pop.pointer.bitcast %27 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    pop.store %28, %26 : !kgen.pointer<pointer<none>>
    %29 = kgen.struct.gep %24[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index2305843009213693952, %29 : !kgen.pointer<index>
    %30 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
    pop.stack_alloc.lifetime.start(%30) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %31 = kgen.struct.gep %30[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index39, %31 : !kgen.pointer<index>
    %32 = kgen.struct.gep %30[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %33 = pop.string.address %string_11
    %34 = pop.pointer.bitcast %33 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    pop.store %34, %32 : !kgen.pointer<pointer<none>>
    %35 = kgen.struct.gep %30[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index2305843009213693952, %35 : !kgen.pointer<index>
    %36 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
    %37 = pop.pointer.bitcast %36 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
    pop.store %23, %37 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
    %38 = pop.load %36 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
    %39 = pop.array.get %38[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
    %40 = pop.array.create [%39] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
    %41 = kgen.struct.create(%40) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
    %42 = pop.string.address %string_2
    %43 = pop.pointer.bitcast %42 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %44 = kgen.struct.create(%43, %index53) : !kgen.struct<(pointer<none>, index)>
    %45 = kgen.struct.create(%index330, %index27, %44) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %46 = pop.string.address %string_3
    %47 = pop.pointer.bitcast %46 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %48 = kgen.struct.create(%47, %index53) : !kgen.struct<(pointer<none>, index)>
    %49 = kgen.struct.create(%index610, %index18, %48) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %50 = pop.string.address %string_5
    %51 = pop.pointer.bitcast %50 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %52 = pop.string.address %string_6
    %53 = pop.pointer.bitcast %52 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %54 = pop.string.address %string_7
    %55 = pop.pointer.bitcast %54 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %56 = pop.string.address %string_8
    %57 = pop.pointer.bitcast %56 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %58 = kgen.struct.create(%55, %index0) : !kgen.struct<(pointer<none>, index)>
    %59 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
    %60 = pop.pointer.bitcast %59 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
    pop.store %41, %59 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
    hlcf.if %16 {
      hlcf.yield
    } else {
      %79 = pop.stack_allocation 2048 x scalar<ui8> align 1
      %80 = pop.pointer.bitcast %79 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<none>
      %81 = kgen.struct.create(%80, %index0) : !kgen.struct<(pointer<none>, index) memoryOnly>
      %82 = kgen.call @"std::collections::string::string::String::write_to[::Writer](::String,$0&),writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%24, %81) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
      %83 = kgen.call @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%arg0, %82) : (index, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
      %84 = kgen.call @"std::collections::string::string::String::write_to[::Writer](::String,$0&),writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%30, %83) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
      %85 = kgen.call @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%idx618, %84) : (index, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
      %86 = kgen.struct.extract %85[0] : <(pointer<none>, index) memoryOnly>
      %87 = kgen.struct.extract %85[1] : <(pointer<none>, index) memoryOnly>
      %88 = index.add %87, %index1
      %89 = index.cmp sgt(%88, %index2048)
      hlcf.if %89 {
        kgen.call @"std::io::io::_printf[KGENParamList[::AnyType],::StringSlice[::Bool(False), StaticConstantOrigin, *?],*::AnyType,LITImmutOrigin,::Origin[::Bool(False), $3]](*$0,file:::FileDescriptor),types.values`=[],fmt={ #interp.memref<{[(#interp.memory_handle<16, \22HEAP_BUFFER_BYTES exceeded, increase with: `mojo -D HEAP_BUFFER_BYTES=4096`\\0A\\00\22 string>, const_global, [], [])], []}, 0, 0>, 76 }"(%index1) : (index) -> ()
        llvm.intr.trap
        hlcf.loop "_loop_0" {
          hlcf.continue "_loop_0"
        }
        kgen.unreachable
      } else {
        hlcf.yield
      }
      %90 = pop.pointer.bitcast %86 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %91 = pop.offset %90[%87] : !kgen.pointer<scalar<ui8>>
      pop.store %simd_9, %91 : !kgen.pointer<scalar<ui8>>
      %92 = pop.load %59 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
      %93 = kgen.struct.extract %92[0] : <(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
      %94 = pop.array.get %93[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
      %95 = pop.array.create [%94] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
      %96 = kgen.struct.create(%95) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
      %97 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
      pop.store %96, %97 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
      %98 = pop.pointer.bitcast %97 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
      %99 = pop.load %98 : !kgen.pointer<index>
      %100 = index.cmp eq(%99, %index-1)
      %101 = pop.select %100, %index0, %index-1 : index
      %102 = index.cmp eq(%101, %index-1)
      %103 = hlcf.if %102 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
        %104 = pop.load %59 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
        %105 = kgen.struct.extract %104[0] : <(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
        %106 = pop.array.get %105[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
        %107 = pop.array.create [%106] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
        %108 = kgen.struct.create(%107) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
        %109 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
        pop.store %108, %109 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
        %110 = pop.pointer.bitcast %109 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
        %111 = pop.load %110 : !kgen.pointer<index>
        %112 = index.cmp eq(%111, %index-1)
        %113 = pop.select %112, %index0, %index-1 : index
        %114 = index.cmp eq(%113, %index-1)
        %115 = pop.xor %114, %0
        hlcf.if %115 {
          %117 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
          pop.stack_alloc.lifetime.start(%117) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
          %118 = kgen.struct.gep %117[1] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index192, %118 : !kgen.pointer<index>
          %119 = kgen.struct.gep %117[0] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %57, %119 : !kgen.pointer<pointer<none>>
          %120 = kgen.struct.gep %117[2] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index2305843009213693952, %120 : !kgen.pointer<index>
          %121 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
          %122 = pop.pointer.bitcast %121 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
          pop.store %45, %122 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
          %123 = pop.load %121 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
          %124 = pop.array.get %123[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
          %125 = pop.array.create [%124] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
          %126 = kgen.struct.create(%125) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
          %127 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
          %128 = pop.pointer.bitcast %127 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
          pop.store %126, %127 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
          %129 = pop.pointer.bitcast %127 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
          %130 = pop.load %129 : !kgen.pointer<index>
          %131 = index.cmp eq(%130, %index-1)
          %132 = pop.select %131, %index0, %index-1 : index
          %133 = index.cmp eq(%132, %index-1)
          %134 = hlcf.if %133 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
            %152 = pop.load %128 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
            hlcf.yield %152 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
          } else {
            hlcf.yield %49 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
          }
          %135 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
          pop.stack_alloc.lifetime.start(%135) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
          %136 = kgen.struct.gep %135[1] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index1, %136 : !kgen.pointer<index>
          %137 = kgen.struct.gep %135[0] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %51, %137 : !kgen.pointer<pointer<none>>
          %138 = kgen.struct.gep %135[2] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index2305843009213693952, %138 : !kgen.pointer<index>
          %139 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
          pop.stack_alloc.lifetime.start(%139) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
          %140 = kgen.struct.gep %139[1] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index2, %140 : !kgen.pointer<index>
          %141 = kgen.struct.gep %139[0] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %53, %141 : !kgen.pointer<pointer<none>>
          %142 = kgen.struct.gep %139[2] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index2305843009213693952, %142 : !kgen.pointer<index>
          kgen.call @"std::io::io::print[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?],flush:::Bool,file:::FileDescriptor$),Ts.values`=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::reflection::location::SourceLocation\22>>, struct<(index, index, struct<(pointer<none>, index)>)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]]"(%struct_4, %135, %134, %139, %117, %58, %struct, %0, %index1) : (!kgen.struct<(pointer<none>, index)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index)>, i1, index owned) -> ()
          %143 = pop.load %138 : !kgen.pointer<index>
          %144 = index.and %143, %index4611686018427387904
          %145 = index.cmp ne(%144, %index0)
          hlcf.if %145 {
            %152 = pop.load %137 : !kgen.pointer<pointer<none>>
            %153 = pop.pointer.bitcast %152 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
            %154 = pop.offset %153[%idx-8] : !kgen.pointer<scalar<ui8>>
            %155 = pop.pointer.bitcast %154 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
            %156 = kgen.struct.gep %155[0] : <struct<(scalar<index>) memoryOnly>>
            %157 = pop.atomic.rmw sub(%156, %simd_10) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
            %158 = pop.cmp eq(%157, %simd_10) : <1, index>
            %159 = pop.cast_to_builtin %158 : !pop.scalar<bool> to i1
            hlcf.if %159 {
              pop.fence syncscope("") acquire
              pop.aligned_free %154 : <scalar<ui8>>
              hlcf.yield
            } else {
              hlcf.yield
            }
            hlcf.yield
          } else {
            hlcf.yield
          }
          %146 = pop.load %142 : !kgen.pointer<index>
          %147 = index.and %146, %index4611686018427387904
          %148 = index.cmp ne(%147, %index0)
          hlcf.if %148 {
            %152 = pop.load %141 : !kgen.pointer<pointer<none>>
            %153 = pop.pointer.bitcast %152 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
            %154 = pop.offset %153[%idx-8] : !kgen.pointer<scalar<ui8>>
            %155 = pop.pointer.bitcast %154 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
            %156 = kgen.struct.gep %155[0] : <struct<(scalar<index>) memoryOnly>>
            %157 = pop.atomic.rmw sub(%156, %simd_10) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
            %158 = pop.cmp eq(%157, %simd_10) : <1, index>
            %159 = pop.cast_to_builtin %158 : !pop.scalar<bool> to i1
            hlcf.if %159 {
              pop.fence syncscope("") acquire
              pop.aligned_free %154 : <scalar<ui8>>
              hlcf.yield
            } else {
              hlcf.yield
            }
            hlcf.yield
          } else {
            hlcf.yield
          }
          %149 = pop.load %120 : !kgen.pointer<index>
          %150 = index.and %149, %index4611686018427387904
          %151 = index.cmp ne(%150, %index0)
          hlcf.if %151 {
            %152 = pop.load %119 : !kgen.pointer<pointer<none>>
            %153 = pop.pointer.bitcast %152 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
            %154 = pop.offset %153[%idx-8] : !kgen.pointer<scalar<ui8>>
            %155 = pop.pointer.bitcast %154 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
            %156 = kgen.struct.gep %155[0] : <struct<(scalar<index>) memoryOnly>>
            %157 = pop.atomic.rmw sub(%156, %simd_10) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
            %158 = pop.cmp eq(%157, %simd_10) : <1, index>
            %159 = pop.cast_to_builtin %158 : !pop.scalar<bool> to i1
            hlcf.if %159 {
              pop.fence syncscope("") acquire
              pop.aligned_free %154 : <scalar<ui8>>
              hlcf.yield
            } else {
              hlcf.yield
            }
            hlcf.yield
          } else {
            hlcf.yield
          }
          llvm.intr.trap
          hlcf.loop "_loop_0" {
            hlcf.continue "_loop_0"
          }
          kgen.unreachable
        } else {
          hlcf.yield
        }
        %116 = pop.load %60 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
        hlcf.yield %116 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
      } else {
        hlcf.yield %12 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
      }
      kgen.call @"std::builtin::debug_assert::_debug_assert_msg[LITImmutOrigin,::Origin[::Bool(False), $0]](::UnsafePointer[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1, ::AddressSpace(::Int(0))],::Int,::SourceLocation)"(%86, %103) : (!kgen.pointer<none>, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>) -> ()
      hlcf.yield
    }
    %61 = pop.load %29 : !kgen.pointer<index>
    %62 = index.and %61, %index4611686018427387904
    %63 = index.cmp ne(%62, %index0)
    hlcf.if %63 {
      %79 = pop.load %26 : !kgen.pointer<pointer<none>>
      %80 = pop.pointer.bitcast %79 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %81 = pop.offset %80[%idx-8] : !kgen.pointer<scalar<ui8>>
      %82 = pop.pointer.bitcast %81 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
      %83 = kgen.struct.gep %82[0] : <struct<(scalar<index>) memoryOnly>>
      %84 = pop.atomic.rmw sub(%83, %simd_10) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
      %85 = pop.cmp eq(%84, %simd_10) : <1, index>
      %86 = pop.cast_to_builtin %85 : !pop.scalar<bool> to i1
      hlcf.if %86 {
        pop.fence syncscope("") acquire
        pop.aligned_free %81 : <scalar<ui8>>
        hlcf.yield
      } else {
        hlcf.yield
      }
      hlcf.yield
    } else {
      hlcf.yield
    }
    %64 = pop.load %35 : !kgen.pointer<index>
    %65 = index.and %64, %index4611686018427387904
    %66 = index.cmp ne(%65, %index0)
    hlcf.if %66 {
      %79 = pop.load %32 : !kgen.pointer<pointer<none>>
      %80 = pop.pointer.bitcast %79 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %81 = pop.offset %80[%idx-8] : !kgen.pointer<scalar<ui8>>
      %82 = pop.pointer.bitcast %81 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
      %83 = kgen.struct.gep %82[0] : <struct<(scalar<index>) memoryOnly>>
      %84 = pop.atomic.rmw sub(%83, %simd_10) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
      %85 = pop.cmp eq(%84, %simd_10) : <1, index>
      %86 = pop.cast_to_builtin %85 : !pop.scalar<bool> to i1
      hlcf.if %86 {
        pop.fence syncscope("") acquire
        pop.aligned_free %81 : <scalar<ui8>>
        hlcf.yield
      } else {
        hlcf.yield
      }
      hlcf.yield
    } else {
      hlcf.yield
    }
    %67 = kgen.struct.gep %8[0] : <struct<(array<619, scalar<ui128>>) memoryOnly>>
    %68 = pop.array.gep %67[%arg0] : <array<619, scalar<ui128>>>
    %69 = pop.load %68 : !kgen.pointer<scalar<ui128>>
    %70 = pop.shr %69, %simd_0 : !pop.scalar<ui128>
    %71 = pop.cast fast %70 : !pop.scalar<ui128> to !pop.scalar<ui64>
    %72 = pop.shr %71, %simd_14 : !pop.scalar<ui64>
    %73 = pop.add %71, %72 : !pop.scalar<ui64>
    %74 = pop.shr %73, %7 : !pop.scalar<ui64>
    %75 = pop.shr %71, %simd_13 : !pop.scalar<ui64>
    %76 = pop.sub %71, %75 : !pop.scalar<ui64>
    %77 = pop.shr %76, %7 : !pop.scalar<ui64>
    %78 = pop.select %arg2, %77, %74 : !pop.scalar<ui64>
    kgen.return %78 : !pop.scalar<ui64>
  }
  kgen.func @"std::builtin::_format_float::_remove_trailing_zeros[::DType](::SIMD[$0, ::Int(1)]&,::Int&),CarrierDType=ui64"(%arg0: !pop.scalar<ui64>, %arg1: index) -> (!pop.scalar<ui64>, index) {
    %simd = kgen.param.constant: scalar<ui64> = <1844674407370955162>
    %simd_0 = kgen.param.constant: scalar<ui64> = <14757395258967641293>
    %simd_1 = kgen.param.constant: scalar<ui64> = <184467440737095517>
    %simd_2 = kgen.param.constant: scalar<ui64> = <10330176681277348905>
    %index2 = kgen.param.constant = <2>
    %simd_3 = kgen.param.constant: scalar<ui64> = <1844674407370956>
    %simd_4 = kgen.param.constant: scalar<ui64> = <182622766329724561>
    %simd_5 = kgen.param.constant: scalar<ui64> = <184467440738>
    %simd_6 = kgen.param.constant: scalar<ui64> = <28999941890838049>
    %index0 = kgen.param.constant = <0>
    %index1 = kgen.param.constant = <1>
    %simd_7 = kgen.param.constant: scalar<ui64> = <8>
    %simd_8 = kgen.param.constant: scalar<ui64> = <56>
    %simd_9 = kgen.param.constant: scalar<ui64> = <4>
    %simd_10 = kgen.param.constant: scalar<ui64> = <60>
    %simd_11 = kgen.param.constant: scalar<ui64> = <2>
    %simd_12 = kgen.param.constant: scalar<ui64> = <62>
    %simd_13 = kgen.param.constant: scalar<ui64> = <1>
    %simd_14 = kgen.param.constant: scalar<ui64> = <63>
    %0 = pop.mul %arg0, %simd_6 : !pop.scalar<ui64>
    %1 = pop.shr %0, %simd_7 : !pop.scalar<ui64>
    %2 = pop.shl %0, %simd_8 : !pop.scalar<ui64>
    %3 = pop.simd.or %1, %2 : <1, ui64>
    %4 = pop.cmp lt(%3, %simd_5) : <1, ui64>
    %5 = pop.cast_to_builtin %4 : !pop.scalar<bool> to i1
    %6 = pop.select %5, %index1, %index0 : index
    %7 = pop.select %5, %3, %arg0 : !pop.scalar<ui64>
    %8 = pop.mul %7, %simd_4 : !pop.scalar<ui64>
    %9 = pop.shr %8, %simd_9 : !pop.scalar<ui64>
    %10 = pop.shl %8, %simd_10 : !pop.scalar<ui64>
    %11 = pop.simd.or %9, %10 : <1, ui64>
    %12 = pop.cmp lt(%11, %simd_3) : <1, ui64>
    %13 = pop.cast_to_builtin %12 : !pop.scalar<bool> to i1
    %14 = index.mul %6, %index2
    %15 = pop.select %13, %index1, %index0 : index
    %16 = index.add %14, %15
    %17 = pop.select %13, %11, %7 : !pop.scalar<ui64>
    %18 = pop.mul %17, %simd_2 : !pop.scalar<ui64>
    %19 = pop.shr %18, %simd_11 : !pop.scalar<ui64>
    %20 = pop.shl %18, %simd_12 : !pop.scalar<ui64>
    %21 = pop.simd.or %19, %20 : <1, ui64>
    %22 = pop.cmp lt(%21, %simd_1) : <1, ui64>
    %23 = pop.cast_to_builtin %22 : !pop.scalar<bool> to i1
    %24 = index.mul %16, %index2
    %25 = pop.select %23, %index1, %index0 : index
    %26 = index.add %24, %25
    %27 = pop.select %23, %21, %17 : !pop.scalar<ui64>
    %28 = pop.mul %27, %simd_0 : !pop.scalar<ui64>
    %29 = pop.shr %28, %simd_13 : !pop.scalar<ui64>
    %30 = pop.shl %28, %simd_14 : !pop.scalar<ui64>
    %31 = pop.simd.or %29, %30 : <1, ui64>
    %32 = pop.cmp lt(%31, %simd) : <1, ui64>
    %33 = pop.cast_to_builtin %32 : !pop.scalar<bool> to i1
    %34 = index.mul %26, %index2
    %35 = pop.select %33, %index1, %index0 : index
    %36 = index.add %34, %35
    %37 = pop.select %33, %31, %27 : !pop.scalar<ui64>
    %38 = index.add %arg1, %36
    kgen.return %37, %38 : !pop.scalar<ui64>, index
  }
  kgen.func @"std::builtin::_format_float::_umul192_lower128(::SIMD[::DType(uint64), ::Int(1)],::SIMD[::DType(uint128), ::Int(1)])"(%arg0: !pop.scalar<ui64>, %arg1: !pop.scalar<ui128>) -> !pop.scalar<ui128> {
    %simd = kgen.param.constant: scalar<ui128> = <64>
    %simd_0 = kgen.param.constant: scalar<ui64> = <18446744073709551615>
    %simd_1 = kgen.param.constant: scalar<ui64> = <32>
    %0 = pop.shr %arg1, %simd : !pop.scalar<ui128>
    %1 = pop.cast fast %0 : !pop.scalar<ui128> to !pop.scalar<ui64>
    %2 = pop.mul %arg0, %1 : !pop.scalar<ui64>
    %3 = pop.cast fast %arg1 : !pop.scalar<ui128> to !pop.scalar<ui64>
    %4 = pop.shr %arg0, %simd_1 : !pop.scalar<ui64>
    %5 = pop.cast fast %4 : !pop.scalar<ui64> to !pop.scalar<ui32>
    %6 = pop.cast fast %arg0 : !pop.scalar<ui64> to !pop.scalar<ui32>
    %7 = pop.shr %3, %simd_1 : !pop.scalar<ui64>
    %8 = pop.cast fast %7 : !pop.scalar<ui64> to !pop.scalar<ui32>
    %9 = pop.cast fast %3 : !pop.scalar<ui64> to !pop.scalar<ui32>
    %10 = pop.cast fast %5 : !pop.scalar<ui32> to !pop.scalar<ui64>
    %11 = pop.cast fast %8 : !pop.scalar<ui32> to !pop.scalar<ui64>
    %12 = pop.mul %10, %11 : !pop.scalar<ui64>
    %13 = pop.cast fast %6 : !pop.scalar<ui32> to !pop.scalar<ui64>
    %14 = pop.mul %13, %11 : !pop.scalar<ui64>
    %15 = pop.cast fast %9 : !pop.scalar<ui32> to !pop.scalar<ui64>
    %16 = pop.mul %10, %15 : !pop.scalar<ui64>
    %17 = pop.mul %13, %15 : !pop.scalar<ui64>
    %18 = pop.shr %17, %simd_1 : !pop.scalar<ui64>
    %19 = pop.cast fast %16 : !pop.scalar<ui64> to !pop.scalar<ui32>
    %20 = pop.cast fast %19 : !pop.scalar<ui32> to !pop.scalar<ui64>
    %21 = pop.add %18, %20 : !pop.scalar<ui64>
    %22 = pop.cast fast %14 : !pop.scalar<ui64> to !pop.scalar<ui32>
    %23 = pop.cast fast %22 : !pop.scalar<ui32> to !pop.scalar<ui64>
    %24 = pop.add %21, %23 : !pop.scalar<ui64>
    %25 = pop.shr %24, %simd_1 : !pop.scalar<ui64>
    %26 = pop.add %12, %25 : !pop.scalar<ui64>
    %27 = pop.shr %16, %simd_1 : !pop.scalar<ui64>
    %28 = pop.add %26, %27 : !pop.scalar<ui64>
    %29 = pop.shr %14, %simd_1 : !pop.scalar<ui64>
    %30 = pop.add %28, %29 : !pop.scalar<ui64>
    %31 = pop.shl %24, %simd_1 : !pop.scalar<ui64>
    %32 = pop.cast fast %17 : !pop.scalar<ui64> to !pop.scalar<ui32>
    %33 = pop.cast fast %32 : !pop.scalar<ui32> to !pop.scalar<ui64>
    %34 = pop.add %31, %33 : !pop.scalar<ui64>
    %35 = pop.cast fast %30 : !pop.scalar<ui64> to !pop.scalar<ui128>
    %36 = pop.shl %35, %simd : !pop.scalar<ui128>
    %37 = pop.cast fast %34 : !pop.scalar<ui64> to !pop.scalar<ui128>
    %38 = pop.simd.or %36, %37 : <1, ui128>
    %39 = pop.shr %38, %simd : !pop.scalar<ui128>
    %40 = pop.cast fast %39 : !pop.scalar<ui128> to !pop.scalar<ui64>
    %41 = pop.add %2, %40 : !pop.scalar<ui64>
    %42 = pop.simd.and %41, %simd_0 : <1, ui64>
    %43 = pop.cast fast %38 : !pop.scalar<ui128> to !pop.scalar<ui64>
    %44 = pop.cast fast %42 : !pop.scalar<ui64> to !pop.scalar<ui128>
    %45 = pop.shl %44, %simd : !pop.scalar<ui128>
    %46 = pop.cast fast %43 : !pop.scalar<ui64> to !pop.scalar<ui128>
    %47 = pop.simd.or %45, %46 : <1, ui128>
    kgen.return %47 : !pop.scalar<ui128>
  }
  kgen.func @"std::builtin::_format_float::_compute_mul_parity[::DType](::SIMD[$0, ::Int(1)],::Int,::Int),CarrierDType=ui64"(%arg0: !pop.scalar<ui64>, %arg1: index, %arg2: index) -> !kgen.struct<(i1, i1)> {
    %string = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/builtin/_format_float.mojo">
    %simd = kgen.param.constant: scalar<uindex> = <619>
    %idx618 = index.constant 618
    %index54 = kgen.param.constant = <54>
    %index41 = kgen.param.constant = <41>
    %index689 = kgen.param.constant = <689>
    %simd_0 = kgen.param.constant: scalar<ui128> = <64>
    %string_1 = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/collections/check_bounds.mojo">
    %index57 = kgen.param.constant = <57>
    %string_2 = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/builtin/debug_assert.mojo">
    %index27 = kgen.param.constant = <27>
    %index330 = kgen.param.constant = <330>
    %index53 = kgen.param.constant = <53>
    %string_3 = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/collections/optional.mojo">
    %index18 = kgen.param.constant = <18>
    %index610 = kgen.param.constant = <610>
    %idx-8 = index.constant -8
    %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle, const_global, [], [])], []}, 0, 0>, 1 }>
    %struct_4 = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle4, const_global, [], [])], []}, 0, 0>, 6 }>
    %string_5 = kgen.param.constant: string = <" ">
    %index2 = kgen.param.constant = <2>
    %string_6 = kgen.param.constant: string = <": ">
    %string_7 = kgen.param.constant: string = <"">
    %string_8 = kgen.param.constant: string = <"`Optional.value()` called on empty `Optional`. Consider using `if optional:` to check whether the `Optional` is empty before calling `.value()`, or use `.or_else()` to provide a default value.">
    %index192 = kgen.param.constant = <192>
    %simd_9 = kgen.param.constant: scalar<ui8> = <0>
    %index2048 = kgen.param.constant = <2048>
    %index-1 = kgen.param.constant = <-1>
    %index0 = kgen.param.constant = <0>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %simd_10 = kgen.param.constant: scalar<index> = <1>
    %string_11 = kgen.param.constant: string = <" is out of bounds, valid range is 0 to ">
    %index39 = kgen.param.constant = <39>
    %string_12 = kgen.param.constant: string = <"index ">
    %index6 = kgen.param.constant = <6>
    %index2305843009213693952 = kgen.param.constant = <2305843009213693952>
    %none = kgen.param.constant: none = <#kgen.none>
    %0 = kgen.param.constant: i1 = <1>
    %array = kgen.param.constant: array<1, struct<(index, index, struct<(pointer<none>, index)>)>> = <[{ -1, 0, { 0, 0 } }]>
    %index1 = kgen.param.constant = <1>
    %index64 = kgen.param.constant = <64>
    %simd_13 = kgen.param.constant: scalar<ui64> = <1>
    %simd_14 = kgen.param.constant: scalar<ui64> = <0>
    %simd_15 = kgen.param.constant: scalar<ui64> = <18446744073709551615>
    %1 = pop.string.address %string
    %2 = pop.pointer.bitcast %1 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %3 = kgen.struct.create(%2, %index54) : !kgen.struct<(pointer<none>, index)>
    %4 = kgen.struct.create(%index689, %index41, %3) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %5 = pop.global_constant: struct<(array<619, scalar<ui128>>) memoryOnly> = <{ [339574632262346462319337857089816694651, 212234145163966538949586160681135434157, 265292681454958173686982700851419292696, 331615851818697717108728376064274115870, 207259907386686073192955235040171322419, 259074884233357591491194043800214153024, 323843605291696989363992554750267691280, 202402253307310618352495346718917307050, 253002816634138272940619183398646633812, 316253520792672841175773979248308292265, 197658450495420525734858737030192682666, 247073063119275657168573421287740853332, 308841328899094571460716776609676066665, 193025830561934107162947985381047541666, 241282288202417633953684981726309427083, 301602860253022042442106227157886783853, 188501787658138776526316391973679239908, 235627234572673470657895489967099049885, 294534043215841838322369362458873812356, 184083777009901148951480851536796132723, 230104721262376436189351064420995165904, 287630901577970545236688830526243957379, 179769313486231590772930519078902473362, 224711641857789488466163148848628091703, 280889552322236860582703936060785114628, 175555970201398037864189960037990696643, 219444962751747547330237450047488370803, 274306203439684434162796812559360463504, 171441377149802771351748007849600289690, 214301721437253464189685009812000362113, 267877151796566830237106262265000452641, 334846439745708537796382827831250565801, 209279024841067836122739267394531603626, 261598781051334795153424084243164504532, 326998476314168493941780105303955630665, 204374047696355308713612565814972269166, 255467559620444135892015707268715336457, 319334449525555169865019634085894170571, 199584030953471981165637271303683856607, 249480038691839976457046589129604820759, 311850048364799970571308236412006025949, 194906280227999981607067647757503766218, 243632850284999977008834559696879707772, 304541062856249971261043199621099634715, 190338164285156232038151999763187271697, 237922705356445290047689999703984089622, 297403381695556612559612499629980112027, 185877113559722882849757812268737570017, 232346391949653603562197265335921962521, 290432989937067004452746581669902453151, 181520618710666877782966613543689033220, 226900773388333597228708266929611291524, 283625966735416996535885333662014114405, 177266229209635622834928333538758821504, 221582786512044528543660416923448526879, 276978483140055660679575521154310658599, 173111551962534787924734700721444161625, 216389439953168484905918375901805202031, 270486799941460606132397969877256502538, 338108499926825757665497462346570628173, 211317812454266098540935913966606642608, 264147265567832623176169892458258303260, 330184081959790778970212365572822879075, 206365051224869236856382728483014299422, 257956314031086546070478410603767874277, 322445392538858182588098013254709842846, 201528370336786364117561258284193651779, 251910462920982955146951572855242064724, 314888078651228693933689466069052580905, 196805049157017933708555916293157863066, 246006311446272417135694895366447328832, 307507889307840521419618619208059161040, 192192430817400325887261637005036975650, 240240538521750407359077046256296219562, 300300673152188009198846307820370274453, 187687920720117505749278942387731421533, 234609900900146882186598677984664276916, 293262376125183602733248347480830346145, 183288985078239751708280217175518966341, 229111231347799689635350271469398707926, 286389039184749612044187839336748384908, 178993149490468507527617399585467740568, 223741436863085634409521749481834675709, 279676796078857043011902186852293344636, 174797997549285651882438866782683340398, 218497496936607064853048583478354175497, 273121871170758831066310729347942719372, 170701169481724269416444205842464199607, 213376461852155336770555257303080249509, 266720577315194170963194071628850311886, 333400721643992713703992589536062889858, 208375451027495446064995368460039306161, 260469313784369307581244210575049132701, 325586642230461634476555263218811415877, 203491651394038521547847039511757134923, 254364564242548151934808799389696418654, 317955705303185189918510999237120523317, 198722315814490743699069374523200327073, 248402894768113429623836718154000408842, 310503618460141787029795897692500511052, 194064761537588616893622436057812819408, 242580951921985771117028045072266024259, 303226189902482213896285056340332530324, 189516368689051383685178160212707831453, 236895460861314229606472700265884789316, 296119326076642787008090875332355986645, 185074578797901741880056797082722491653, 231343223497377177350070996353403114566, 289179029371721471687588745441753893208, 180736893357325919804742965901096183255, 225921116696657399755928707376370229069, 282401395870821749694910884220462786336, 176500872419263593559319302637789241460, 220626090524079491949149128297236551825, 275782613155099364936436410371545689781, 172364133221937103085272756482216056113, 215455166527421378856590945602770070141, 269318958159276723570738682003462587677, 336648697699095904463423352504328234596, 210405436061934940289639595315205146623, 263006795077418675362049494144006433278, 328758493846773344202561867680008041597, 205474058654233340126601167300005025999, 256842573317791675158251459125006282498, 321053216647239593947814323906257853122, 200658260404524746217383952441411158202, 250822825505655932771729940551763947752, 313528531882069915964662425689704934690, 195955332426293697477914016056065584181, 244944165532867121847392520070081980227, 306180206916083902309240650087602475283, 191362629322552438943275406304751547052, 239203286653190548679094257880939433815, 299004108316488185848867822351174292269, 186877567697805116155542388969483932668, 233596959622256395194427986211854915835, 291996199527820493993034982764818644794, 182497624704887808745646864228011652996, 228122030881109760932058580285014566245, 285152538601387201165073225356268207806, 178220336625867000728170765847667629879, 222775420782333750910213457309584537349, 278469275977917188637766821636980671686, 174043297486198242898604263523112919804, 217554121857747803623255329403891149755, 271942652322184754529069161754863937193, 339928315402730943161336452193579921491, 212455197126706839475835282620987450932, 265568996408383549344794103276234313665, 331961245510479436680992629095292892081, 207475778444049647925620393184558057551, 259344723055062059907025491480697571939, 324180903818827574883781864350871964923, 202613064886767234302363665219294978077, 253266331108459042877954581524118722596, 316582913885573803597443226905148403245, 197864321178483627248402016815717752029, 247330401473104534060502521019647190036, 309163001841380667575628151274558987544, 193226876150862917234767594546599367215, 241533595188578646543459493183249209019, 301916993985723308179324366479061511274, 188698121241077067612077729049413444546, 235872651551346334515097161311766805683, 294840814439182918143871451639708507103, 184275509024489323839919657274817816940, 230344386280611654799899571593522271175, 287930482850764568499874464491902838968, 179956551781727855312421540307439274355, 224945689727159819140526925384299092944, 281182112158949773925658656730373866180, 175738820099343608703536660456483666363, 219673525124179510879420825570604582953, 274591906405224388599276031963255728691, 171619941503265242874547519977034830432, 214524926879081553593184399971293538040, 268156158598851941991480499964116922550, 335195198248564927489350624955146153187, 209496998905353079680844140596966345742, 261871248631691349601055175746207932178, 327339060789614187001318969682759915222, 204586912993508866875824356051724947014, 255733641241886083594780445064656183767, 319667051552357604493475556330820229709, 199791907220223502808422222706762643568, 249739884025279378510527778383453304460, 312174855031599223138159722979316630575, 195109284394749514461349826862072894110, 243886605493436893076687283577591117637, 304858256866796116345859104471988897046, 190536410541747572716161940294993060654, 238170513177184465895202425368741325818, 297713141471480582369003031710926657272, 186070713419675363980626894819329160795, 232588391774594204975783618524161450994, 290735489718242756219729523155201813742, 181709681073901722637330951972001133589, 227137101342377153296663689965001416986, 283921376677971441620829612456251771232, 177450860423732151013018507785157357020, 221813575529665188766273134731446696275, 277266969412081485957841418414308370344, 173291855882550928723650886508942731465, 216614819853188660904563608136178414331, 270768524816485826130704510170223017914, 338460656020607282663380637712778772393, 211537910012879551664612898570486732746, 264422387516099439580766123213108415932, 330527984395124299475957654016385519915, 206579990246952687172473533760240949947, 258224987808690858965591917200301187433, 322781234760863573706989896500376484292, 201738271725539733566868685312735302683, 252172839656924666958585856640919128353, 315216049571155833698232320801148910441, 197010030981972396061395200500718069026, 246262538727465495076744000625897586282, 307828173409331868845930000782371982853, 192392608380832418028706250488982489283, 240490760476040522535882813111228111604, 300613450595050653169853516389035139505, 187883406621906658231158447743146962191, 234854258277383322788948059678933702738, 293567822846729153486185074598667128422, 183479889279205720928865671624166955264, 229349861599007151161082089530208694080, 286687326998758938951352611912760867600, 179179579374224336844595382445475542250, 223974474217780421055744228056844427813, 279968092772225526319680285071055534766, 174980057982640953949800178169409709229, 218725072478301192437250222711762136536, 273406340597876490546562778389702670670, 170878962873672806591601736493564169169, 213598703592091008239502170616955211461, 266998379490113760299377713271194014326, 333747974362642200374222141588992517907, 208592483976651375233888838493120323692, 260740604970814219042361048116400404615, 325925756213517773802951310145500505769, 203703597633448608626844568840937816106, 254629497041810760783555711051172270132, 318286871302263450979444638813965337665, 198929294563914656862152899258728336041, 248661618204893321077691124073410420051, 310827022756116651347113905091763025063, 194266889222572907091946190682351890665, 242833611528216133864932738352939863331, 303542014410270167331165922941174829163, 189713759006418854581978701838234268227, 237142198758023568227473377297792835284, 296427748447529460284341721622241044105, 185267342779705912677713576013900652566, 231584178474632390847141970017375815707, 289480223093290488558927462521719769634, 180925139433306555349329664076074856021, 226156424291633194186662080095093570026, 282695530364541492733327600118866962533, 176684706477838432958329750074291851583, 220855883097298041197912187592864814479, 276069853871622551497390234491081018099, 172543658669764094685868896556925636312, 215679573337205118357336120696157045390, 269599466671506397946670150870196306737, 336999333339382997433337688587745383421, 210624583337114373395836055367340864638, 263280729171392966744795069209176080798, 329100911464241208430993836511470100997, 205688069665150755269371147819668813123, 257110087081438444086713934774586016404, 321387608851798055108392418468232520505, 200867255532373784442745261542645325316, 251084069415467230553431576928306656645, 313855086769334038191789471160383320806, 196159429230833773869868419475239575504, 245199286538542217337335524344049469379, 306499108173177771671669405430061836724, 191561942608236107294793378393788647953, 239452428260295134118491722992235809941, 299315535325368917648114653740294762426, 187072209578355573530071658587684226516, 233840261972944466912589573234605283145, 292300327466180583640736966543256603932, 182687704666362864775460604089535377457, 228359630832953580969325755111919221822, 285449538541191976211657193889899027277, 178405961588244985132285746181186892048, 223007451985306231415357182726483615060, 278759314981632789269196478408104518825, 174224571863520493293247799005065324266, 217780714829400616616559748756331655332, 272225893536750770770699685945414569165, 170141183460469231731687303715884105728, 212676479325586539664609129644855132160, 265845599156983174580761412056068915200, 332306998946228968225951765070086144000, 207691874341393105141219853168803840000, 259614842926741381426524816461004800000, 324518553658426726783156020576256000000, 202824096036516704239472512860160000000, 253530120045645880299340641075200000000, 316912650057057350374175801344000000000, 198070406285660843983859875840000000000, 247588007857076054979824844800000000000, 309485009821345068724781056000000000000, 193428131138340667952988160000000000000, 241785163922925834941235200000000000000, 302231454903657293676544000000000000000, 188894659314785808547840000000000000000, 236118324143482260684800000000000000000, 295147905179352825856000000000000000000, 184467440737095516160000000000000000000, 230584300921369395200000000000000000000, 288230376151711744000000000000000000000, 180143985094819840000000000000000000000, 225179981368524800000000000000000000000, 281474976710656000000000000000000000000, 175921860444160000000000000000000000000, 219902325555200000000000000000000000000, 274877906944000000000000000000000000000, 171798691840000000000000000000000000000, 214748364800000000000000000000000000000, 268435456000000000000000000000000000000, 335544320000000000000000000000000000000, 209715200000000000000000000000000000000, 262144000000000000000000000000000000000, 327680000000000000000000000000000000000, 204800000000000000000000000000000000000, 256000000000000000000000000000000000000, 320000000000000000000000000000000000000, 200000000000000000000000000000000000000, 250000000000000000000000000000000000000, 312500000000000000000000000000000000000, 195312500000000000000000000000000000000, 244140625000000000000000000000000000000, 305175781250000000000000000000000000000, 190734863281250000000000000000000000000, 238418579101562500000000000000000000000, 298023223876953125000000000000000000000, 186264514923095703125000000000000000000, 232830643653869628906250000000000000000, 291038304567337036132812500000000000000, 181898940354585647583007812500000000000, 227373675443232059478759765625000000000, 284217094304040074348449707031250000000, 177635683940025046467781066894531250000, 222044604925031308084726333618164062500, 277555756156289135105907917022705078125, 173472347597680709441192448139190673829, 216840434497100886801490560173988342286, 271050543121376108501863200217485427857, 338813178901720135627329000271856784821, 211758236813575084767080625169910490513, 264697796016968855958850781462388113142, 330872245021211069948563476827985141427, 206795153138256918717852173017490713392, 258493941422821148397315216271863391740, 323117426778526435496644020339829239675, 201948391736579022185402512712393274797, 252435489670723777731753140890491593496, 315544362088404722164691426113114491870, 197215226305252951352932141320696557419, 246519032881566189191165176650870696773, 308148791101957736488956470813588370967, 192592994438723585305597794258492731854, 240741243048404481631997242823115914818, 300926553810505602039996553528894893522, 188079096131566001274997845955559308451, 235098870164457501593747307444449135564, 293873587705571876992184134305561419455, 183670992315982423120115083940975887160, 229588740394978028900143854926219858949, 286985925493722536125179818657774823687, 179366203433576585078237386661109264804, 224207754291970731347796733326386581005, 280259692864963414184745916657983226257, 175162308040602133865466197911239516411, 218952885050752667331832747389049395513, 273691106313440834164790934236311744391, 171056941445900521352994333897694840245, 213821176807375651691242917372118550306, 267276471009219564614053646715148187882, 334095588761524455767567058393935234852, 208809742975952784854729411496209521783, 261012178719940981068411764370261902229, 326265223399926226335514705462827377786, 203915764624953891459696690914267111116, 254894705781192364324620863642833888895, 318618382226490455405776079553542361119, 199136488891556534628610049720963975699, 248920611114445668285762562151204969624, 311150763893057085357203202689006212030, 194469227433160678348252001680628882519, 243086534291450847935315002100786103149, 303858167864313559919143752625982628936, 189911354915195974949464845391239143085, 237389193643994968686831056739048928856, 296736492054993710858538820923811161070, 185460307534371069286586763077381975669, 231825384417963836608233453846727469586, 289781730522454795760291817308409336982, 181113581576534247350182385817755835614, 226391976970667809187727982272194794518, 282989971213334761484659977840243493147, 176868732008334225927912486150152183217, 221085915010417782409890607687690229021, 276357393763022228012363259609612786276, 172723371101888892507727037256007991423, 215904213877361115634658796570009989278, 269880267346701394543323495712512486598, 337350334183376743179154369640640608247, 210843958864610464486971481025400380155, 263554948580763080608714351281750475193, 329443685725953850760892939102188093991, 205902303578721156725558086938867558745, 257377879473401445906947608673584448431, 321722349341751807383684510841980560539, 201076468338594879614802819276237850337, 251345585423243599518503524095297312921, 314181981779054499398129405119121641151, 196363738611909062123830878199451025720, 245454673264886327654788597749313782149, 306818341581107909568485747186642227686, 191761463488192443480303591991651392304, 239701829360240554350379489989564240380, 299627286700300692937974362486955300475, 187267054187687933086233976554347062797, 234083817734609916357792470692933828496, 292604772168262395447240588366167285620, 182877982605163997154525367728854553513, 228597478256454996443156709661068191891, 285746847820568745553945887076335239863, 178591779887855465971216179422709524915, 223239724859819332464020224278386906143, 279049656074774165580025280347983632679, 174406035046733853487515800217489770425, 218007543808417316859394750271862213031, 272509429760521646074243437839827766288, 170318393600326028796402148649892353930, 212897992000407535995502685812365442413, 266122490000509419994378357265456803016, 332653112500636774992972946581821003770, 207908195312897984370608091613638127356, 259885244141122480463260114517047659195, 324856555176403100579075143146309573994, 203035346985251937861921964466443483746, 253794183731564922327402455583054354683, 317242729664456152909253069478817943353, 198276706040285095568283168424261214596, 247845882550356369460353960530326518245, 309807353187945461825442450662908147806, 193629595742465913640901531664317592379, 242036994678082392051126914580396990474, 302546243347602990063908643225496238092, 189091402092251868789942902015935148808, 236364252615314835987428627519918936009, 295455315769143544984285784399898670012, 184659572355714715615178615249936668757, 230824465444643394518973269062420835947, 288530581805804243148716586328026044933, 180331613628627651967947866455016278083, 225414517035784564959934833068770347604, 281768146294730706199918541335962934505, 176105091434206691374949088334976834066, 220131364292758364218686360418721042582, 275164205365947955273357950523401303228, 171977628353717472045848719077125814518, 214972035442146840057310898846407268147, 268715044302683550071638623558009085183, 335893805378354437589548279447511356479, 209933628361471523493467674654694597800, 262417035451839404366834593318368247249, 328021294314799255458543241647960309062, 205013308946749534661589526029975193164, 256266636183436918326986907537468991454, 320333295229296147908733634421836239318, 200208309518310092442958521513647649574, 250260386897887615553698151892059561967, 312825483622359519442122689865074452459, 195515927263974699651326681165671532787, 244394909079968374564158351457089415984, 305493636349960468205197939321361769979, 190933522718725292628248712075851106237, 238666903398406615785310890094813882797, 298333629248008269731638612618517353496, 186458518280005168582274132886573345935, 233073147850006460727842666108216682419, 291341434812508075909803332635270853023, 182088396757817547443627082897044283140, 227610495947271934304533853621305353924, 284513119934089917880667317026631692405, 177820699958806198675417073141644807754, 222275874948507748344271341427056009692, 277844843685634685430339176783820012115, 173653027303521678393961985489887507572, 217066284129402097992452481862359384465, 271332855161752622490565602327949230581, 339166068952190778113207002909936538226, 211978793095119236320754376818710336391, 264973491368899045400942971023387920489, 331216864211123806751178713779234900611, 207010540131952379219486696112021812882, 258763175164940474024358370140027266102, 323453968956175592530447962675034082628, 202158730597609745331529976671896301643, 252698413247012181664412470839870377053, 315873016558765227080515588549837971316, 197420635349228266925322242843648732073, 246775794186535333656652803554560915091, 308469742733169167070816004443201143864, 192793589208230729419260002777000714915, 240991986510288411774075003471250893644, 301239983137860514717593754339063617054, 188274989461162821698496096461914760659, 235343736826453527123120120577393450824, 294179671033066908903900150721741813530, 183862294395666818064937594201088633456, 229827867994583522581171992751360791820, 287284834993229403226464990939200989775, 179553021870768377016540619337000618610, 224441277338460471270675774171250773262, 280551596673075589088344717714063466577, 175344747920672243180215448571289666611, 219180934900840303975269310714112083264, 273976168626050379969086638392640104079, 171235105391281487480679148995400065050, 214043881739101859350848936244250081312, 267554852173877324188561170305312601640, 334443565217346655235701462881640752050, 209027228260841659522313414301025470031, 261284035326052074402891767876281837539, 326605044157565093003614709845352296924, 204128152598478183127259193653345185578, 255160190748097728909073992066681481972, 318950238435122161136342490083351852465, 199343899021951350710214056302094907791, 249179873777439188387767570377618634738, 311474842221798985484709462972023293422, 194671776388624365927943414357514558389, 243339720485780457409929267946893197986, 304174650607225571762411584933616497483, 190109156629515982351507240583510310927, 237636445786894977939384050729387888659, 297045557233618722424230063411734860823, 185653473271011701515143789632334288015, 232066841588764626893929737040417860018, 290083551985955783617412171300522325023, 181302219991222364760882607062826453139, 226627774989027955951103258828533066424, 283284718736284944938879073535666333030, 177052949210178090586799420959791458144, 221316186512722613233499276199739322680, 276645233140903266541874095249674153350, 172903270713064541588671309531046345844, 216129088391330676985839136913807932304, 270161360489163346232298921142259915380, 337701700611454182790373651427824894225, 211063562882158864243983532142390558891, 263829453602698580304979415177988198614, 329786817003373225381224268972485248267, 206116760627108265863265168107803280167, 257645950783885332329081460134754100209, 322057438479856665411351825168442625261, 201285899049910415882094890730276640788, 251607373812388019852618613412845800985, 314509217265485024815773266766057251231, 196568260790928140509858291728785782020, 245710325988660175637322864660982227524, 307137907485825219546653580826227784405, 191961192178640762216658488016392365254, 239951490223300952770823110020490456567, 299939362779126190963528887525613070708, 187462101736953869352205554703508169193, 234327627171192336690256943379385211491, 292909533963990420862821179224231514364, 183068458727494013039263237015144696478, 228835573409367516299079046268930870597, 286044466761709395373848807836163588246, 178777791726068372108655504897602242654, 223472239657585465135819381122002803317, 279340299571981831419774226402503504146, 174587687232488644637358891501564690092, 218234609040610805796698614376955862614, 272793261300763507245873267971194828268, 170495788312977192028670792481996767668, 213119735391221490035838490602495959584, 266399669239026862544798113253119949480, 332999586548783578180997641566399936850, 208124741592989736363123525978999960532, 260155926991237170453904407473749950665, 325194908739046463067380509342187438331, 203246817961904039417112818338867148957, 254058522452380049271391022923583936196, 317573153065475061589238778654479920245, 198483220665921913493274236659049950153, 248104025832402391866592795823812437691, 310130032290502989833240994779765547114, 193831270181564368645775621737353466946, 242289087726955460807219527171691833683, 302861359658694326009024408964614792103, 189288349786683953755640255602884245065, 236610437233354942194550319503605306331, 295763046541693677743187899379506632914, 184851904088558548589492437112191645571, 231064880110698185736865546390239556964, 288831100138372732171081932987799446205, 180519437586482957606926208117374653878, 225649296983103697008657760146718317347, 282061621228879621260822200183397896684, 176288513268049763288013875114623685428, 220360641585062204110017343893279606785, 275450801981327755137521679866599508481, 172156751238329846960951049916624692801, 215195939047912308701188812395780866001, 268994923809890385876486015494726082501, 336243654762362982345607519368407603126, 210152284226476863966004699605254751954, 262690355283096079957505874506568439942, 328362944103870099946882343133210549928] }>
    %6 = pop.string.address %string_1
    %7 = pop.pointer.bitcast %6 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %8 = kgen.struct.create(%7, %index57) : !kgen.struct<(pointer<none>, index)>
    %9 = kgen.struct.create(%index57, %index6, %8) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %10 = pop.cast_from_builtin %arg1 : index to !pop.scalar<index>
    %11 = pop.cast %10 : !pop.scalar<index> to !pop.scalar<uindex>
    %12 = pop.cmp lt(%11, %simd) : <1, uindex>
    %13 = pop.cast_to_builtin %12 : !pop.scalar<bool> to i1
    %14 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
    pop.stack_alloc.lifetime.start(%14) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
    pop.store %array, %14 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
    %15 = pop.pointer.bitcast %14 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<index>
    %16 = pop.load %15 : !kgen.pointer<index>
    %17 = index.cmp eq(%16, %index-1)
    %18 = pop.select %17, %index0, %index-1 : index
    pop.stack_alloc.lifetime.end(%14) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
    %19 = index.cmp eq(%18, %index-1)
    %20 = hlcf.if %19 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
      %88 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
      pop.stack_alloc.lifetime.start(%88) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
      pop.store %array, %88 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
      %89 = pop.pointer.bitcast %88 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
      %90 = pop.load %89 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
      pop.stack_alloc.lifetime.end(%88) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
      hlcf.yield %90 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    } else {
      hlcf.yield %4 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    }
    %21 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
    pop.stack_alloc.lifetime.start(%21) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %22 = kgen.struct.gep %21[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index6, %22 : !kgen.pointer<index>
    %23 = kgen.struct.gep %21[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %24 = pop.string.address %string_12
    %25 = pop.pointer.bitcast %24 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    pop.store %25, %23 : !kgen.pointer<pointer<none>>
    %26 = kgen.struct.gep %21[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index2305843009213693952, %26 : !kgen.pointer<index>
    %27 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
    pop.stack_alloc.lifetime.start(%27) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %28 = kgen.struct.gep %27[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index39, %28 : !kgen.pointer<index>
    %29 = kgen.struct.gep %27[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %30 = pop.string.address %string_11
    %31 = pop.pointer.bitcast %30 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    pop.store %31, %29 : !kgen.pointer<pointer<none>>
    %32 = kgen.struct.gep %27[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index2305843009213693952, %32 : !kgen.pointer<index>
    %33 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
    %34 = pop.pointer.bitcast %33 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
    pop.store %20, %34 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
    %35 = pop.load %33 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
    %36 = pop.array.get %35[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
    %37 = pop.array.create [%36] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
    %38 = kgen.struct.create(%37) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
    %39 = pop.string.address %string_2
    %40 = pop.pointer.bitcast %39 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %41 = kgen.struct.create(%40, %index53) : !kgen.struct<(pointer<none>, index)>
    %42 = kgen.struct.create(%index330, %index27, %41) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %43 = pop.string.address %string_3
    %44 = pop.pointer.bitcast %43 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %45 = kgen.struct.create(%44, %index53) : !kgen.struct<(pointer<none>, index)>
    %46 = kgen.struct.create(%index610, %index18, %45) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %47 = pop.string.address %string_5
    %48 = pop.pointer.bitcast %47 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %49 = pop.string.address %string_6
    %50 = pop.pointer.bitcast %49 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %51 = pop.string.address %string_7
    %52 = pop.pointer.bitcast %51 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %53 = pop.string.address %string_8
    %54 = pop.pointer.bitcast %53 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %55 = kgen.struct.create(%52, %index0) : !kgen.struct<(pointer<none>, index)>
    %56 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
    %57 = pop.pointer.bitcast %56 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
    pop.store %38, %56 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
    hlcf.if %13 {
      hlcf.yield
    } else {
      %88 = pop.stack_allocation 2048 x scalar<ui8> align 1
      %89 = pop.pointer.bitcast %88 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<none>
      %90 = kgen.struct.create(%89, %index0) : !kgen.struct<(pointer<none>, index) memoryOnly>
      %91 = kgen.call @"std::collections::string::string::String::write_to[::Writer](::String,$0&),writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%21, %90) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
      %92 = kgen.call @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%arg1, %91) : (index, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
      %93 = kgen.call @"std::collections::string::string::String::write_to[::Writer](::String,$0&),writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%27, %92) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
      %94 = kgen.call @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%idx618, %93) : (index, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
      %95 = kgen.struct.extract %94[0] : <(pointer<none>, index) memoryOnly>
      %96 = kgen.struct.extract %94[1] : <(pointer<none>, index) memoryOnly>
      %97 = index.add %96, %index1
      %98 = index.cmp sgt(%97, %index2048)
      hlcf.if %98 {
        kgen.call @"std::io::io::_printf[KGENParamList[::AnyType],::StringSlice[::Bool(False), StaticConstantOrigin, *?],*::AnyType,LITImmutOrigin,::Origin[::Bool(False), $3]](*$0,file:::FileDescriptor),types.values`=[],fmt={ #interp.memref<{[(#interp.memory_handle<16, \22HEAP_BUFFER_BYTES exceeded, increase with: `mojo -D HEAP_BUFFER_BYTES=4096`\\0A\\00\22 string>, const_global, [], [])], []}, 0, 0>, 76 }"(%index1) : (index) -> ()
        llvm.intr.trap
        hlcf.loop "_loop_0" {
          hlcf.continue "_loop_0"
        }
        kgen.unreachable
      } else {
        hlcf.yield
      }
      %99 = pop.pointer.bitcast %95 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %100 = pop.offset %99[%96] : !kgen.pointer<scalar<ui8>>
      pop.store %simd_9, %100 : !kgen.pointer<scalar<ui8>>
      %101 = pop.load %56 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
      %102 = kgen.struct.extract %101[0] : <(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
      %103 = pop.array.get %102[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
      %104 = pop.array.create [%103] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
      %105 = kgen.struct.create(%104) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
      %106 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
      pop.store %105, %106 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
      %107 = pop.pointer.bitcast %106 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
      %108 = pop.load %107 : !kgen.pointer<index>
      %109 = index.cmp eq(%108, %index-1)
      %110 = pop.select %109, %index0, %index-1 : index
      %111 = index.cmp eq(%110, %index-1)
      %112 = hlcf.if %111 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
        %113 = pop.load %56 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
        %114 = kgen.struct.extract %113[0] : <(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
        %115 = pop.array.get %114[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
        %116 = pop.array.create [%115] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
        %117 = kgen.struct.create(%116) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
        %118 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
        pop.store %117, %118 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
        %119 = pop.pointer.bitcast %118 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
        %120 = pop.load %119 : !kgen.pointer<index>
        %121 = index.cmp eq(%120, %index-1)
        %122 = pop.select %121, %index0, %index-1 : index
        %123 = index.cmp eq(%122, %index-1)
        %124 = pop.xor %123, %0
        hlcf.if %124 {
          %126 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
          pop.stack_alloc.lifetime.start(%126) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
          %127 = kgen.struct.gep %126[1] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index192, %127 : !kgen.pointer<index>
          %128 = kgen.struct.gep %126[0] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %54, %128 : !kgen.pointer<pointer<none>>
          %129 = kgen.struct.gep %126[2] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index2305843009213693952, %129 : !kgen.pointer<index>
          %130 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
          %131 = pop.pointer.bitcast %130 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
          pop.store %42, %131 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
          %132 = pop.load %130 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
          %133 = pop.array.get %132[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
          %134 = pop.array.create [%133] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
          %135 = kgen.struct.create(%134) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
          %136 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
          %137 = pop.pointer.bitcast %136 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
          pop.store %135, %136 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
          %138 = pop.pointer.bitcast %136 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
          %139 = pop.load %138 : !kgen.pointer<index>
          %140 = index.cmp eq(%139, %index-1)
          %141 = pop.select %140, %index0, %index-1 : index
          %142 = index.cmp eq(%141, %index-1)
          %143 = hlcf.if %142 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
            %161 = pop.load %137 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
            hlcf.yield %161 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
          } else {
            hlcf.yield %46 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
          }
          %144 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
          pop.stack_alloc.lifetime.start(%144) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
          %145 = kgen.struct.gep %144[1] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index1, %145 : !kgen.pointer<index>
          %146 = kgen.struct.gep %144[0] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %48, %146 : !kgen.pointer<pointer<none>>
          %147 = kgen.struct.gep %144[2] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index2305843009213693952, %147 : !kgen.pointer<index>
          %148 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
          pop.stack_alloc.lifetime.start(%148) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
          %149 = kgen.struct.gep %148[1] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index2, %149 : !kgen.pointer<index>
          %150 = kgen.struct.gep %148[0] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %50, %150 : !kgen.pointer<pointer<none>>
          %151 = kgen.struct.gep %148[2] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index2305843009213693952, %151 : !kgen.pointer<index>
          kgen.call @"std::io::io::print[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?],flush:::Bool,file:::FileDescriptor$),Ts.values`=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::reflection::location::SourceLocation\22>>, struct<(index, index, struct<(pointer<none>, index)>)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]]"(%struct_4, %144, %143, %148, %126, %55, %struct, %0, %index1) : (!kgen.struct<(pointer<none>, index)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index)>, i1, index owned) -> ()
          %152 = pop.load %147 : !kgen.pointer<index>
          %153 = index.and %152, %index4611686018427387904
          %154 = index.cmp ne(%153, %index0)
          hlcf.if %154 {
            %161 = pop.load %146 : !kgen.pointer<pointer<none>>
            %162 = pop.pointer.bitcast %161 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
            %163 = pop.offset %162[%idx-8] : !kgen.pointer<scalar<ui8>>
            %164 = pop.pointer.bitcast %163 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
            %165 = kgen.struct.gep %164[0] : <struct<(scalar<index>) memoryOnly>>
            %166 = pop.atomic.rmw sub(%165, %simd_10) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
            %167 = pop.cmp eq(%166, %simd_10) : <1, index>
            %168 = pop.cast_to_builtin %167 : !pop.scalar<bool> to i1
            hlcf.if %168 {
              pop.fence syncscope("") acquire
              pop.aligned_free %163 : <scalar<ui8>>
              hlcf.yield
            } else {
              hlcf.yield
            }
            hlcf.yield
          } else {
            hlcf.yield
          }
          %155 = pop.load %151 : !kgen.pointer<index>
          %156 = index.and %155, %index4611686018427387904
          %157 = index.cmp ne(%156, %index0)
          hlcf.if %157 {
            %161 = pop.load %150 : !kgen.pointer<pointer<none>>
            %162 = pop.pointer.bitcast %161 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
            %163 = pop.offset %162[%idx-8] : !kgen.pointer<scalar<ui8>>
            %164 = pop.pointer.bitcast %163 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
            %165 = kgen.struct.gep %164[0] : <struct<(scalar<index>) memoryOnly>>
            %166 = pop.atomic.rmw sub(%165, %simd_10) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
            %167 = pop.cmp eq(%166, %simd_10) : <1, index>
            %168 = pop.cast_to_builtin %167 : !pop.scalar<bool> to i1
            hlcf.if %168 {
              pop.fence syncscope("") acquire
              pop.aligned_free %163 : <scalar<ui8>>
              hlcf.yield
            } else {
              hlcf.yield
            }
            hlcf.yield
          } else {
            hlcf.yield
          }
          %158 = pop.load %129 : !kgen.pointer<index>
          %159 = index.and %158, %index4611686018427387904
          %160 = index.cmp ne(%159, %index0)
          hlcf.if %160 {
            %161 = pop.load %128 : !kgen.pointer<pointer<none>>
            %162 = pop.pointer.bitcast %161 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
            %163 = pop.offset %162[%idx-8] : !kgen.pointer<scalar<ui8>>
            %164 = pop.pointer.bitcast %163 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
            %165 = kgen.struct.gep %164[0] : <struct<(scalar<index>) memoryOnly>>
            %166 = pop.atomic.rmw sub(%165, %simd_10) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
            %167 = pop.cmp eq(%166, %simd_10) : <1, index>
            %168 = pop.cast_to_builtin %167 : !pop.scalar<bool> to i1
            hlcf.if %168 {
              pop.fence syncscope("") acquire
              pop.aligned_free %163 : <scalar<ui8>>
              hlcf.yield
            } else {
              hlcf.yield
            }
            hlcf.yield
          } else {
            hlcf.yield
          }
          llvm.intr.trap
          hlcf.loop "_loop_0" {
            hlcf.continue "_loop_0"
          }
          kgen.unreachable
        } else {
          hlcf.yield
        }
        %125 = pop.load %57 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
        hlcf.yield %125 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
      } else {
        hlcf.yield %9 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
      }
      kgen.call @"std::builtin::debug_assert::_debug_assert_msg[LITImmutOrigin,::Origin[::Bool(False), $0]](::UnsafePointer[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1, ::AddressSpace(::Int(0))],::Int,::SourceLocation)"(%95, %112) : (!kgen.pointer<none>, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>) -> ()
      hlcf.yield
    }
    %58 = pop.load %26 : !kgen.pointer<index>
    %59 = index.and %58, %index4611686018427387904
    %60 = index.cmp ne(%59, %index0)
    hlcf.if %60 {
      %88 = pop.load %23 : !kgen.pointer<pointer<none>>
      %89 = pop.pointer.bitcast %88 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %90 = pop.offset %89[%idx-8] : !kgen.pointer<scalar<ui8>>
      %91 = pop.pointer.bitcast %90 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
      %92 = kgen.struct.gep %91[0] : <struct<(scalar<index>) memoryOnly>>
      %93 = pop.atomic.rmw sub(%92, %simd_10) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
      %94 = pop.cmp eq(%93, %simd_10) : <1, index>
      %95 = pop.cast_to_builtin %94 : !pop.scalar<bool> to i1
      hlcf.if %95 {
        pop.fence syncscope("") acquire
        pop.aligned_free %90 : <scalar<ui8>>
        hlcf.yield
      } else {
        hlcf.yield
      }
      hlcf.yield
    } else {
      hlcf.yield
    }
    %61 = pop.load %32 : !kgen.pointer<index>
    %62 = index.and %61, %index4611686018427387904
    %63 = index.cmp ne(%62, %index0)
    hlcf.if %63 {
      %88 = pop.load %29 : !kgen.pointer<pointer<none>>
      %89 = pop.pointer.bitcast %88 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %90 = pop.offset %89[%idx-8] : !kgen.pointer<scalar<ui8>>
      %91 = pop.pointer.bitcast %90 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
      %92 = kgen.struct.gep %91[0] : <struct<(scalar<index>) memoryOnly>>
      %93 = pop.atomic.rmw sub(%92, %simd_10) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
      %94 = pop.cmp eq(%93, %simd_10) : <1, index>
      %95 = pop.cast_to_builtin %94 : !pop.scalar<bool> to i1
      hlcf.if %95 {
        pop.fence syncscope("") acquire
        pop.aligned_free %90 : <scalar<ui8>>
        hlcf.yield
      } else {
        hlcf.yield
      }
      hlcf.yield
    } else {
      hlcf.yield
    }
    %64 = kgen.struct.gep %5[0] : <struct<(array<619, scalar<ui128>>) memoryOnly>>
    %65 = pop.array.gep %64[%arg1] : <array<619, scalar<ui128>>>
    %66 = pop.load %65 : !kgen.pointer<scalar<ui128>>
    %67 = kgen.call tail @"std::builtin::_format_float::_umul192_lower128(::SIMD[::DType(uint64), ::Int(1)],::SIMD[::DType(uint128), ::Int(1)])"(%arg0, %66) : (!pop.scalar<ui64>, !pop.scalar<ui128>) -> !pop.scalar<ui128>
    %68 = pop.shr %67, %simd_0 : !pop.scalar<ui128>
    %69 = pop.cast fast %68 : !pop.scalar<ui128> to !pop.scalar<ui64>
    %70 = pop.cast fast %67 : !pop.scalar<ui128> to !pop.scalar<ui64>
    %71 = index.sub %index64, %arg2
    %72 = pop.cast_from_builtin %71 : index to !pop.scalar<index>
    %73 = pop.cast %72 : !pop.scalar<index> to !pop.scalar<ui64>
    %74 = pop.shr %69, %73 : !pop.scalar<ui64>
    %75 = pop.simd.and %74, %simd_13 : <1, ui64>
    %76 = pop.cmp eq(%75, %simd_14) : <1, ui64>
    %77 = pop.cast_to_builtin %76 : !pop.scalar<bool> to i1
    %78 = pop.xor %77, %0
    %79 = pop.cast_from_builtin %arg2 : index to !pop.scalar<index>
    %80 = pop.cast %79 : !pop.scalar<index> to !pop.scalar<ui64>
    %81 = pop.shl %69, %80 : !pop.scalar<ui64>
    %82 = pop.simd.and %81, %simd_15 : <1, ui64>
    %83 = pop.shr %70, %73 : !pop.scalar<ui64>
    %84 = pop.simd.or %82, %83 : <1, ui64>
    %85 = pop.cmp eq(%84, %simd_14) : <1, ui64>
    %86 = pop.cast_to_builtin %85 : !pop.scalar<bool> to i1
    %87 = kgen.struct.create(%78, %86) : !kgen.struct<(i1, i1)>
    kgen.return %87 : !kgen.struct<(i1, i1)>
  }
  kgen.func @"std::builtin::_format_float::_check_divisibility_and_divide_by_pow10[::DType,::Int,::InlineArray[::SIMD[::DType(uint32), ::Int(1)], ::Int(2)]](::SIMD[$0, ::Int(1)]&,::Int),CarrierDType=ui64,carrier_bits=64,divide_magic_number={ [6554, 656] }"(%arg0: !pop.scalar<ui64>, %arg1: index) -> (i1, !pop.scalar<ui64>) {
    %string = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/builtin/_format_float.mojo">
    %simd = kgen.param.constant: scalar<uindex> = <2>
    %idx1 = index.constant 1
    %index54 = kgen.param.constant = <54>
    %index58 = kgen.param.constant = <58>
    %index729 = kgen.param.constant = <729>
    %array = kgen.param.constant: array<2, scalar<ui32>> = <[6554, 656]>
    %string_0 = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/collections/check_bounds.mojo">
    %index57 = kgen.param.constant = <57>
    %string_1 = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/builtin/debug_assert.mojo">
    %index27 = kgen.param.constant = <27>
    %index330 = kgen.param.constant = <330>
    %index53 = kgen.param.constant = <53>
    %string_2 = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/collections/optional.mojo">
    %index18 = kgen.param.constant = <18>
    %index610 = kgen.param.constant = <610>
    %idx-8 = index.constant -8
    %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle, const_global, [], [])], []}, 0, 0>, 1 }>
    %struct_3 = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle4, const_global, [], [])], []}, 0, 0>, 6 }>
    %string_4 = kgen.param.constant: string = <" ">
    %string_5 = kgen.param.constant: string = <": ">
    %string_6 = kgen.param.constant: string = <"">
    %string_7 = kgen.param.constant: string = <"`Optional.value()` called on empty `Optional`. Consider using `if optional:` to check whether the `Optional` is empty before calling `.value()`, or use `.or_else()` to provide a default value.">
    %index192 = kgen.param.constant = <192>
    %simd_8 = kgen.param.constant: scalar<ui8> = <0>
    %index2048 = kgen.param.constant = <2048>
    %0 = kgen.param.constant: i1 = <1>
    %index-1 = kgen.param.constant = <-1>
    %index0 = kgen.param.constant = <0>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %simd_9 = kgen.param.constant: scalar<index> = <1>
    %string_10 = kgen.param.constant: string = <" is out of bounds, valid range is 0 to ">
    %index39 = kgen.param.constant = <39>
    %string_11 = kgen.param.constant: string = <"index ">
    %index6 = kgen.param.constant = <6>
    %index2305843009213693952 = kgen.param.constant = <2305843009213693952>
    %none = kgen.param.constant: none = <#kgen.none>
    %index2 = kgen.param.constant = <2>
    %array_12 = kgen.param.constant: array<1, struct<(index, index, struct<(pointer<none>, index)>)>> = <[{ -1, 0, { 0, 0 } }]>
    %index1 = kgen.param.constant = <1>
    %simd_13 = kgen.param.constant: scalar<ui32> = <65535>
    %simd_14 = kgen.param.constant: scalar<ui32> = <16>
    %1 = pop.string.address %string
    %2 = pop.pointer.bitcast %1 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %3 = kgen.struct.create(%2, %index54) : !kgen.struct<(pointer<none>, index)>
    %4 = kgen.struct.create(%index729, %index58, %3) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %5 = pop.stack_allocation 1 x array<2, scalar<ui32>> align 1 marked
    pop.stack_alloc.lifetime.start(%5) : !kgen.pointer<array<2, scalar<ui32>>>
    pop.store %array, %5 : !kgen.pointer<array<2, scalar<ui32>>>
    %6 = index.sub %arg1, %index1
    %7 = pop.string.address %string_0
    %8 = pop.pointer.bitcast %7 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %9 = kgen.struct.create(%8, %index57) : !kgen.struct<(pointer<none>, index)>
    %10 = kgen.struct.create(%index57, %index6, %9) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %11 = pop.cast_from_builtin %6 : index to !pop.scalar<index>
    %12 = pop.cast %11 : !pop.scalar<index> to !pop.scalar<uindex>
    %13 = pop.cmp lt(%12, %simd) : <1, uindex>
    %14 = pop.cast_to_builtin %13 : !pop.scalar<bool> to i1
    %15 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
    pop.stack_alloc.lifetime.start(%15) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
    pop.store %array_12, %15 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
    %16 = pop.pointer.bitcast %15 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<index>
    %17 = pop.load %16 : !kgen.pointer<index>
    %18 = index.cmp eq(%17, %index-1)
    %19 = pop.select %18, %index0, %index-1 : index
    pop.stack_alloc.lifetime.end(%15) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
    %20 = index.cmp eq(%19, %index-1)
    %21 = hlcf.if %20 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
      %75 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
      pop.stack_alloc.lifetime.start(%75) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
      pop.store %array_12, %75 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
      %76 = pop.pointer.bitcast %75 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
      %77 = pop.load %76 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
      pop.stack_alloc.lifetime.end(%75) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
      hlcf.yield %77 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    } else {
      hlcf.yield %4 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    }
    %22 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
    pop.stack_alloc.lifetime.start(%22) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %23 = kgen.struct.gep %22[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index6, %23 : !kgen.pointer<index>
    %24 = kgen.struct.gep %22[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %25 = pop.string.address %string_11
    %26 = pop.pointer.bitcast %25 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    pop.store %26, %24 : !kgen.pointer<pointer<none>>
    %27 = kgen.struct.gep %22[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index2305843009213693952, %27 : !kgen.pointer<index>
    %28 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
    pop.stack_alloc.lifetime.start(%28) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %29 = kgen.struct.gep %28[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index39, %29 : !kgen.pointer<index>
    %30 = kgen.struct.gep %28[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %31 = pop.string.address %string_10
    %32 = pop.pointer.bitcast %31 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    pop.store %32, %30 : !kgen.pointer<pointer<none>>
    %33 = kgen.struct.gep %28[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index2305843009213693952, %33 : !kgen.pointer<index>
    %34 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
    %35 = pop.pointer.bitcast %34 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
    pop.store %21, %35 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
    %36 = pop.load %34 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
    %37 = pop.array.get %36[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
    %38 = pop.array.create [%37] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
    %39 = kgen.struct.create(%38) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
    %40 = pop.string.address %string_1
    %41 = pop.pointer.bitcast %40 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %42 = kgen.struct.create(%41, %index53) : !kgen.struct<(pointer<none>, index)>
    %43 = kgen.struct.create(%index330, %index27, %42) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %44 = pop.string.address %string_2
    %45 = pop.pointer.bitcast %44 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %46 = kgen.struct.create(%45, %index53) : !kgen.struct<(pointer<none>, index)>
    %47 = kgen.struct.create(%index610, %index18, %46) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %48 = pop.string.address %string_4
    %49 = pop.pointer.bitcast %48 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %50 = pop.string.address %string_5
    %51 = pop.pointer.bitcast %50 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %52 = pop.string.address %string_6
    %53 = pop.pointer.bitcast %52 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %54 = pop.string.address %string_7
    %55 = pop.pointer.bitcast %54 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %56 = kgen.struct.create(%53, %index0) : !kgen.struct<(pointer<none>, index)>
    %57 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
    %58 = pop.pointer.bitcast %57 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
    pop.store %39, %57 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
    hlcf.if %14 {
      hlcf.yield
    } else {
      %75 = pop.stack_allocation 2048 x scalar<ui8> align 1
      %76 = pop.pointer.bitcast %75 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<none>
      %77 = kgen.struct.create(%76, %index0) : !kgen.struct<(pointer<none>, index) memoryOnly>
      %78 = kgen.call @"std::collections::string::string::String::write_to[::Writer](::String,$0&),writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%22, %77) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
      %79 = kgen.call @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%6, %78) : (index, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
      %80 = kgen.call @"std::collections::string::string::String::write_to[::Writer](::String,$0&),writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%28, %79) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
      %81 = kgen.call @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%idx1, %80) : (index, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
      %82 = kgen.struct.extract %81[0] : <(pointer<none>, index) memoryOnly>
      %83 = kgen.struct.extract %81[1] : <(pointer<none>, index) memoryOnly>
      %84 = index.add %83, %index1
      %85 = index.cmp sgt(%84, %index2048)
      hlcf.if %85 {
        kgen.call @"std::io::io::_printf[KGENParamList[::AnyType],::StringSlice[::Bool(False), StaticConstantOrigin, *?],*::AnyType,LITImmutOrigin,::Origin[::Bool(False), $3]](*$0,file:::FileDescriptor),types.values`=[],fmt={ #interp.memref<{[(#interp.memory_handle<16, \22HEAP_BUFFER_BYTES exceeded, increase with: `mojo -D HEAP_BUFFER_BYTES=4096`\\0A\\00\22 string>, const_global, [], [])], []}, 0, 0>, 76 }"(%index1) : (index) -> ()
        llvm.intr.trap
        hlcf.loop "_loop_0" {
          hlcf.continue "_loop_0"
        }
        kgen.unreachable
      } else {
        hlcf.yield
      }
      %86 = pop.pointer.bitcast %82 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %87 = pop.offset %86[%83] : !kgen.pointer<scalar<ui8>>
      pop.store %simd_8, %87 : !kgen.pointer<scalar<ui8>>
      %88 = pop.load %57 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
      %89 = kgen.struct.extract %88[0] : <(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
      %90 = pop.array.get %89[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
      %91 = pop.array.create [%90] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
      %92 = kgen.struct.create(%91) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
      %93 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
      pop.store %92, %93 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
      %94 = pop.pointer.bitcast %93 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
      %95 = pop.load %94 : !kgen.pointer<index>
      %96 = index.cmp eq(%95, %index-1)
      %97 = pop.select %96, %index0, %index-1 : index
      %98 = index.cmp eq(%97, %index-1)
      %99 = hlcf.if %98 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
        %100 = pop.load %57 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
        %101 = kgen.struct.extract %100[0] : <(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
        %102 = pop.array.get %101[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
        %103 = pop.array.create [%102] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
        %104 = kgen.struct.create(%103) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
        %105 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
        pop.store %104, %105 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
        %106 = pop.pointer.bitcast %105 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
        %107 = pop.load %106 : !kgen.pointer<index>
        %108 = index.cmp eq(%107, %index-1)
        %109 = pop.select %108, %index0, %index-1 : index
        %110 = index.cmp eq(%109, %index-1)
        %111 = pop.xor %110, %0
        hlcf.if %111 {
          %113 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
          pop.stack_alloc.lifetime.start(%113) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
          %114 = kgen.struct.gep %113[1] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index192, %114 : !kgen.pointer<index>
          %115 = kgen.struct.gep %113[0] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %55, %115 : !kgen.pointer<pointer<none>>
          %116 = kgen.struct.gep %113[2] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index2305843009213693952, %116 : !kgen.pointer<index>
          %117 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
          %118 = pop.pointer.bitcast %117 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
          pop.store %43, %118 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
          %119 = pop.load %117 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
          %120 = pop.array.get %119[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
          %121 = pop.array.create [%120] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
          %122 = kgen.struct.create(%121) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
          %123 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
          %124 = pop.pointer.bitcast %123 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
          pop.store %122, %123 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
          %125 = pop.pointer.bitcast %123 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
          %126 = pop.load %125 : !kgen.pointer<index>
          %127 = index.cmp eq(%126, %index-1)
          %128 = pop.select %127, %index0, %index-1 : index
          %129 = index.cmp eq(%128, %index-1)
          %130 = hlcf.if %129 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
            %148 = pop.load %124 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
            hlcf.yield %148 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
          } else {
            hlcf.yield %47 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
          }
          %131 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
          pop.stack_alloc.lifetime.start(%131) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
          %132 = kgen.struct.gep %131[1] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index1, %132 : !kgen.pointer<index>
          %133 = kgen.struct.gep %131[0] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %49, %133 : !kgen.pointer<pointer<none>>
          %134 = kgen.struct.gep %131[2] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index2305843009213693952, %134 : !kgen.pointer<index>
          %135 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
          pop.stack_alloc.lifetime.start(%135) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
          %136 = kgen.struct.gep %135[1] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index2, %136 : !kgen.pointer<index>
          %137 = kgen.struct.gep %135[0] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %51, %137 : !kgen.pointer<pointer<none>>
          %138 = kgen.struct.gep %135[2] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index2305843009213693952, %138 : !kgen.pointer<index>
          kgen.call @"std::io::io::print[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?],flush:::Bool,file:::FileDescriptor$),Ts.values`=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::reflection::location::SourceLocation\22>>, struct<(index, index, struct<(pointer<none>, index)>)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]]"(%struct_3, %131, %130, %135, %113, %56, %struct, %0, %index1) : (!kgen.struct<(pointer<none>, index)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index)>, i1, index owned) -> ()
          %139 = pop.load %134 : !kgen.pointer<index>
          %140 = index.and %139, %index4611686018427387904
          %141 = index.cmp ne(%140, %index0)
          hlcf.if %141 {
            %148 = pop.load %133 : !kgen.pointer<pointer<none>>
            %149 = pop.pointer.bitcast %148 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
            %150 = pop.offset %149[%idx-8] : !kgen.pointer<scalar<ui8>>
            %151 = pop.pointer.bitcast %150 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
            %152 = kgen.struct.gep %151[0] : <struct<(scalar<index>) memoryOnly>>
            %153 = pop.atomic.rmw sub(%152, %simd_9) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
            %154 = pop.cmp eq(%153, %simd_9) : <1, index>
            %155 = pop.cast_to_builtin %154 : !pop.scalar<bool> to i1
            hlcf.if %155 {
              pop.fence syncscope("") acquire
              pop.aligned_free %150 : <scalar<ui8>>
              hlcf.yield
            } else {
              hlcf.yield
            }
            hlcf.yield
          } else {
            hlcf.yield
          }
          %142 = pop.load %138 : !kgen.pointer<index>
          %143 = index.and %142, %index4611686018427387904
          %144 = index.cmp ne(%143, %index0)
          hlcf.if %144 {
            %148 = pop.load %137 : !kgen.pointer<pointer<none>>
            %149 = pop.pointer.bitcast %148 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
            %150 = pop.offset %149[%idx-8] : !kgen.pointer<scalar<ui8>>
            %151 = pop.pointer.bitcast %150 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
            %152 = kgen.struct.gep %151[0] : <struct<(scalar<index>) memoryOnly>>
            %153 = pop.atomic.rmw sub(%152, %simd_9) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
            %154 = pop.cmp eq(%153, %simd_9) : <1, index>
            %155 = pop.cast_to_builtin %154 : !pop.scalar<bool> to i1
            hlcf.if %155 {
              pop.fence syncscope("") acquire
              pop.aligned_free %150 : <scalar<ui8>>
              hlcf.yield
            } else {
              hlcf.yield
            }
            hlcf.yield
          } else {
            hlcf.yield
          }
          %145 = pop.load %116 : !kgen.pointer<index>
          %146 = index.and %145, %index4611686018427387904
          %147 = index.cmp ne(%146, %index0)
          hlcf.if %147 {
            %148 = pop.load %115 : !kgen.pointer<pointer<none>>
            %149 = pop.pointer.bitcast %148 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
            %150 = pop.offset %149[%idx-8] : !kgen.pointer<scalar<ui8>>
            %151 = pop.pointer.bitcast %150 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
            %152 = kgen.struct.gep %151[0] : <struct<(scalar<index>) memoryOnly>>
            %153 = pop.atomic.rmw sub(%152, %simd_9) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
            %154 = pop.cmp eq(%153, %simd_9) : <1, index>
            %155 = pop.cast_to_builtin %154 : !pop.scalar<bool> to i1
            hlcf.if %155 {
              pop.fence syncscope("") acquire
              pop.aligned_free %150 : <scalar<ui8>>
              hlcf.yield
            } else {
              hlcf.yield
            }
            hlcf.yield
          } else {
            hlcf.yield
          }
          llvm.intr.trap
          hlcf.loop "_loop_0" {
            hlcf.continue "_loop_0"
          }
          kgen.unreachable
        } else {
          hlcf.yield
        }
        %112 = pop.load %58 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
        hlcf.yield %112 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
      } else {
        hlcf.yield %10 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
      }
      kgen.call @"std::builtin::debug_assert::_debug_assert_msg[LITImmutOrigin,::Origin[::Bool(False), $0]](::UnsafePointer[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1, ::AddressSpace(::Int(0))],::Int,::SourceLocation)"(%82, %99) : (!kgen.pointer<none>, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>) -> ()
      hlcf.yield
    }
    %59 = pop.load %27 : !kgen.pointer<index>
    %60 = index.and %59, %index4611686018427387904
    %61 = index.cmp ne(%60, %index0)
    hlcf.if %61 {
      %75 = pop.load %24 : !kgen.pointer<pointer<none>>
      %76 = pop.pointer.bitcast %75 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %77 = pop.offset %76[%idx-8] : !kgen.pointer<scalar<ui8>>
      %78 = pop.pointer.bitcast %77 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
      %79 = kgen.struct.gep %78[0] : <struct<(scalar<index>) memoryOnly>>
      %80 = pop.atomic.rmw sub(%79, %simd_9) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
      %81 = pop.cmp eq(%80, %simd_9) : <1, index>
      %82 = pop.cast_to_builtin %81 : !pop.scalar<bool> to i1
      hlcf.if %82 {
        pop.fence syncscope("") acquire
        pop.aligned_free %77 : <scalar<ui8>>
        hlcf.yield
      } else {
        hlcf.yield
      }
      hlcf.yield
    } else {
      hlcf.yield
    }
    %62 = pop.load %33 : !kgen.pointer<index>
    %63 = index.and %62, %index4611686018427387904
    %64 = index.cmp ne(%63, %index0)
    hlcf.if %64 {
      %75 = pop.load %30 : !kgen.pointer<pointer<none>>
      %76 = pop.pointer.bitcast %75 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %77 = pop.offset %76[%idx-8] : !kgen.pointer<scalar<ui8>>
      %78 = pop.pointer.bitcast %77 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
      %79 = kgen.struct.gep %78[0] : <struct<(scalar<index>) memoryOnly>>
      %80 = pop.atomic.rmw sub(%79, %simd_9) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
      %81 = pop.cmp eq(%80, %simd_9) : <1, index>
      %82 = pop.cast_to_builtin %81 : !pop.scalar<bool> to i1
      hlcf.if %82 {
        pop.fence syncscope("") acquire
        pop.aligned_free %77 : <scalar<ui8>>
        hlcf.yield
      } else {
        hlcf.yield
      }
      hlcf.yield
    } else {
      hlcf.yield
    }
    %65 = pop.array.gep %5[%6] : <array<2, scalar<ui32>>>
    %66 = pop.load %65 : !kgen.pointer<scalar<ui32>>
    pop.stack_alloc.lifetime.end(%5) : !kgen.pointer<array<2, scalar<ui32>>>
    %67 = pop.cast fast %66 : !pop.scalar<ui32> to !pop.scalar<ui64>
    %68 = pop.mul %arg0, %67 : !pop.scalar<ui64>
    %69 = pop.cast fast %68 : !pop.scalar<ui64> to !pop.scalar<ui32>
    %70 = pop.simd.and %69, %simd_13 : <1, ui32>
    %71 = pop.cmp lt(%70, %66) : <1, ui32>
    %72 = pop.cast_to_builtin %71 : !pop.scalar<bool> to i1
    %73 = pop.shr %69, %simd_14 : !pop.scalar<ui32>
    %74 = pop.cast fast %73 : !pop.scalar<ui32> to !pop.scalar<ui64>
    kgen.return %72, %74 : i1, !pop.scalar<ui64>
  }
  kgen.func @"std::builtin::_format_float::_compute_mul[::DType](::SIMD[$0, ::Int(1)],::Int),CarrierDType=ui64"(%arg0: !pop.scalar<ui64>, %arg1: index) -> !kgen.struct<(scalar<ui64>, i1)> {
    %string = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/builtin/_format_float.mojo">
    %simd = kgen.param.constant: scalar<uindex> = <619>
    %idx618 = index.constant 618
    %index54 = kgen.param.constant = <54>
    %index66 = kgen.param.constant = <66>
    %index764 = kgen.param.constant = <764>
    %simd_0 = kgen.param.constant: scalar<ui128> = <64>
    %string_1 = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/collections/check_bounds.mojo">
    %index57 = kgen.param.constant = <57>
    %string_2 = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/builtin/debug_assert.mojo">
    %index27 = kgen.param.constant = <27>
    %index330 = kgen.param.constant = <330>
    %index53 = kgen.param.constant = <53>
    %string_3 = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/collections/optional.mojo">
    %index18 = kgen.param.constant = <18>
    %index610 = kgen.param.constant = <610>
    %idx-8 = index.constant -8
    %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle, const_global, [], [])], []}, 0, 0>, 1 }>
    %struct_4 = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle4, const_global, [], [])], []}, 0, 0>, 6 }>
    %string_5 = kgen.param.constant: string = <" ">
    %index2 = kgen.param.constant = <2>
    %string_6 = kgen.param.constant: string = <": ">
    %string_7 = kgen.param.constant: string = <"">
    %string_8 = kgen.param.constant: string = <"`Optional.value()` called on empty `Optional`. Consider using `if optional:` to check whether the `Optional` is empty before calling `.value()`, or use `.or_else()` to provide a default value.">
    %index192 = kgen.param.constant = <192>
    %simd_9 = kgen.param.constant: scalar<ui8> = <0>
    %index2048 = kgen.param.constant = <2048>
    %index-1 = kgen.param.constant = <-1>
    %index0 = kgen.param.constant = <0>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %simd_10 = kgen.param.constant: scalar<index> = <1>
    %string_11 = kgen.param.constant: string = <" is out of bounds, valid range is 0 to ">
    %index39 = kgen.param.constant = <39>
    %string_12 = kgen.param.constant: string = <"index ">
    %index6 = kgen.param.constant = <6>
    %index2305843009213693952 = kgen.param.constant = <2305843009213693952>
    %index1 = kgen.param.constant = <1>
    %none = kgen.param.constant: none = <#kgen.none>
    %array = kgen.param.constant: array<1, struct<(index, index, struct<(pointer<none>, index)>)>> = <[{ -1, 0, { 0, 0 } }]>
    %0 = kgen.param.constant: i1 = <1>
    %simd_13 = kgen.param.constant: scalar<ui64> = <0>
    %1 = pop.string.address %string
    %2 = pop.pointer.bitcast %1 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %3 = kgen.struct.create(%2, %index54) : !kgen.struct<(pointer<none>, index)>
    %4 = kgen.struct.create(%index764, %index66, %3) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %5 = pop.global_constant: struct<(array<619, scalar<ui128>>) memoryOnly> = <{ [339574632262346462319337857089816694651, 212234145163966538949586160681135434157, 265292681454958173686982700851419292696, 331615851818697717108728376064274115870, 207259907386686073192955235040171322419, 259074884233357591491194043800214153024, 323843605291696989363992554750267691280, 202402253307310618352495346718917307050, 253002816634138272940619183398646633812, 316253520792672841175773979248308292265, 197658450495420525734858737030192682666, 247073063119275657168573421287740853332, 308841328899094571460716776609676066665, 193025830561934107162947985381047541666, 241282288202417633953684981726309427083, 301602860253022042442106227157886783853, 188501787658138776526316391973679239908, 235627234572673470657895489967099049885, 294534043215841838322369362458873812356, 184083777009901148951480851536796132723, 230104721262376436189351064420995165904, 287630901577970545236688830526243957379, 179769313486231590772930519078902473362, 224711641857789488466163148848628091703, 280889552322236860582703936060785114628, 175555970201398037864189960037990696643, 219444962751747547330237450047488370803, 274306203439684434162796812559360463504, 171441377149802771351748007849600289690, 214301721437253464189685009812000362113, 267877151796566830237106262265000452641, 334846439745708537796382827831250565801, 209279024841067836122739267394531603626, 261598781051334795153424084243164504532, 326998476314168493941780105303955630665, 204374047696355308713612565814972269166, 255467559620444135892015707268715336457, 319334449525555169865019634085894170571, 199584030953471981165637271303683856607, 249480038691839976457046589129604820759, 311850048364799970571308236412006025949, 194906280227999981607067647757503766218, 243632850284999977008834559696879707772, 304541062856249971261043199621099634715, 190338164285156232038151999763187271697, 237922705356445290047689999703984089622, 297403381695556612559612499629980112027, 185877113559722882849757812268737570017, 232346391949653603562197265335921962521, 290432989937067004452746581669902453151, 181520618710666877782966613543689033220, 226900773388333597228708266929611291524, 283625966735416996535885333662014114405, 177266229209635622834928333538758821504, 221582786512044528543660416923448526879, 276978483140055660679575521154310658599, 173111551962534787924734700721444161625, 216389439953168484905918375901805202031, 270486799941460606132397969877256502538, 338108499926825757665497462346570628173, 211317812454266098540935913966606642608, 264147265567832623176169892458258303260, 330184081959790778970212365572822879075, 206365051224869236856382728483014299422, 257956314031086546070478410603767874277, 322445392538858182588098013254709842846, 201528370336786364117561258284193651779, 251910462920982955146951572855242064724, 314888078651228693933689466069052580905, 196805049157017933708555916293157863066, 246006311446272417135694895366447328832, 307507889307840521419618619208059161040, 192192430817400325887261637005036975650, 240240538521750407359077046256296219562, 300300673152188009198846307820370274453, 187687920720117505749278942387731421533, 234609900900146882186598677984664276916, 293262376125183602733248347480830346145, 183288985078239751708280217175518966341, 229111231347799689635350271469398707926, 286389039184749612044187839336748384908, 178993149490468507527617399585467740568, 223741436863085634409521749481834675709, 279676796078857043011902186852293344636, 174797997549285651882438866782683340398, 218497496936607064853048583478354175497, 273121871170758831066310729347942719372, 170701169481724269416444205842464199607, 213376461852155336770555257303080249509, 266720577315194170963194071628850311886, 333400721643992713703992589536062889858, 208375451027495446064995368460039306161, 260469313784369307581244210575049132701, 325586642230461634476555263218811415877, 203491651394038521547847039511757134923, 254364564242548151934808799389696418654, 317955705303185189918510999237120523317, 198722315814490743699069374523200327073, 248402894768113429623836718154000408842, 310503618460141787029795897692500511052, 194064761537588616893622436057812819408, 242580951921985771117028045072266024259, 303226189902482213896285056340332530324, 189516368689051383685178160212707831453, 236895460861314229606472700265884789316, 296119326076642787008090875332355986645, 185074578797901741880056797082722491653, 231343223497377177350070996353403114566, 289179029371721471687588745441753893208, 180736893357325919804742965901096183255, 225921116696657399755928707376370229069, 282401395870821749694910884220462786336, 176500872419263593559319302637789241460, 220626090524079491949149128297236551825, 275782613155099364936436410371545689781, 172364133221937103085272756482216056113, 215455166527421378856590945602770070141, 269318958159276723570738682003462587677, 336648697699095904463423352504328234596, 210405436061934940289639595315205146623, 263006795077418675362049494144006433278, 328758493846773344202561867680008041597, 205474058654233340126601167300005025999, 256842573317791675158251459125006282498, 321053216647239593947814323906257853122, 200658260404524746217383952441411158202, 250822825505655932771729940551763947752, 313528531882069915964662425689704934690, 195955332426293697477914016056065584181, 244944165532867121847392520070081980227, 306180206916083902309240650087602475283, 191362629322552438943275406304751547052, 239203286653190548679094257880939433815, 299004108316488185848867822351174292269, 186877567697805116155542388969483932668, 233596959622256395194427986211854915835, 291996199527820493993034982764818644794, 182497624704887808745646864228011652996, 228122030881109760932058580285014566245, 285152538601387201165073225356268207806, 178220336625867000728170765847667629879, 222775420782333750910213457309584537349, 278469275977917188637766821636980671686, 174043297486198242898604263523112919804, 217554121857747803623255329403891149755, 271942652322184754529069161754863937193, 339928315402730943161336452193579921491, 212455197126706839475835282620987450932, 265568996408383549344794103276234313665, 331961245510479436680992629095292892081, 207475778444049647925620393184558057551, 259344723055062059907025491480697571939, 324180903818827574883781864350871964923, 202613064886767234302363665219294978077, 253266331108459042877954581524118722596, 316582913885573803597443226905148403245, 197864321178483627248402016815717752029, 247330401473104534060502521019647190036, 309163001841380667575628151274558987544, 193226876150862917234767594546599367215, 241533595188578646543459493183249209019, 301916993985723308179324366479061511274, 188698121241077067612077729049413444546, 235872651551346334515097161311766805683, 294840814439182918143871451639708507103, 184275509024489323839919657274817816940, 230344386280611654799899571593522271175, 287930482850764568499874464491902838968, 179956551781727855312421540307439274355, 224945689727159819140526925384299092944, 281182112158949773925658656730373866180, 175738820099343608703536660456483666363, 219673525124179510879420825570604582953, 274591906405224388599276031963255728691, 171619941503265242874547519977034830432, 214524926879081553593184399971293538040, 268156158598851941991480499964116922550, 335195198248564927489350624955146153187, 209496998905353079680844140596966345742, 261871248631691349601055175746207932178, 327339060789614187001318969682759915222, 204586912993508866875824356051724947014, 255733641241886083594780445064656183767, 319667051552357604493475556330820229709, 199791907220223502808422222706762643568, 249739884025279378510527778383453304460, 312174855031599223138159722979316630575, 195109284394749514461349826862072894110, 243886605493436893076687283577591117637, 304858256866796116345859104471988897046, 190536410541747572716161940294993060654, 238170513177184465895202425368741325818, 297713141471480582369003031710926657272, 186070713419675363980626894819329160795, 232588391774594204975783618524161450994, 290735489718242756219729523155201813742, 181709681073901722637330951972001133589, 227137101342377153296663689965001416986, 283921376677971441620829612456251771232, 177450860423732151013018507785157357020, 221813575529665188766273134731446696275, 277266969412081485957841418414308370344, 173291855882550928723650886508942731465, 216614819853188660904563608136178414331, 270768524816485826130704510170223017914, 338460656020607282663380637712778772393, 211537910012879551664612898570486732746, 264422387516099439580766123213108415932, 330527984395124299475957654016385519915, 206579990246952687172473533760240949947, 258224987808690858965591917200301187433, 322781234760863573706989896500376484292, 201738271725539733566868685312735302683, 252172839656924666958585856640919128353, 315216049571155833698232320801148910441, 197010030981972396061395200500718069026, 246262538727465495076744000625897586282, 307828173409331868845930000782371982853, 192392608380832418028706250488982489283, 240490760476040522535882813111228111604, 300613450595050653169853516389035139505, 187883406621906658231158447743146962191, 234854258277383322788948059678933702738, 293567822846729153486185074598667128422, 183479889279205720928865671624166955264, 229349861599007151161082089530208694080, 286687326998758938951352611912760867600, 179179579374224336844595382445475542250, 223974474217780421055744228056844427813, 279968092772225526319680285071055534766, 174980057982640953949800178169409709229, 218725072478301192437250222711762136536, 273406340597876490546562778389702670670, 170878962873672806591601736493564169169, 213598703592091008239502170616955211461, 266998379490113760299377713271194014326, 333747974362642200374222141588992517907, 208592483976651375233888838493120323692, 260740604970814219042361048116400404615, 325925756213517773802951310145500505769, 203703597633448608626844568840937816106, 254629497041810760783555711051172270132, 318286871302263450979444638813965337665, 198929294563914656862152899258728336041, 248661618204893321077691124073410420051, 310827022756116651347113905091763025063, 194266889222572907091946190682351890665, 242833611528216133864932738352939863331, 303542014410270167331165922941174829163, 189713759006418854581978701838234268227, 237142198758023568227473377297792835284, 296427748447529460284341721622241044105, 185267342779705912677713576013900652566, 231584178474632390847141970017375815707, 289480223093290488558927462521719769634, 180925139433306555349329664076074856021, 226156424291633194186662080095093570026, 282695530364541492733327600118866962533, 176684706477838432958329750074291851583, 220855883097298041197912187592864814479, 276069853871622551497390234491081018099, 172543658669764094685868896556925636312, 215679573337205118357336120696157045390, 269599466671506397946670150870196306737, 336999333339382997433337688587745383421, 210624583337114373395836055367340864638, 263280729171392966744795069209176080798, 329100911464241208430993836511470100997, 205688069665150755269371147819668813123, 257110087081438444086713934774586016404, 321387608851798055108392418468232520505, 200867255532373784442745261542645325316, 251084069415467230553431576928306656645, 313855086769334038191789471160383320806, 196159429230833773869868419475239575504, 245199286538542217337335524344049469379, 306499108173177771671669405430061836724, 191561942608236107294793378393788647953, 239452428260295134118491722992235809941, 299315535325368917648114653740294762426, 187072209578355573530071658587684226516, 233840261972944466912589573234605283145, 292300327466180583640736966543256603932, 182687704666362864775460604089535377457, 228359630832953580969325755111919221822, 285449538541191976211657193889899027277, 178405961588244985132285746181186892048, 223007451985306231415357182726483615060, 278759314981632789269196478408104518825, 174224571863520493293247799005065324266, 217780714829400616616559748756331655332, 272225893536750770770699685945414569165, 170141183460469231731687303715884105728, 212676479325586539664609129644855132160, 265845599156983174580761412056068915200, 332306998946228968225951765070086144000, 207691874341393105141219853168803840000, 259614842926741381426524816461004800000, 324518553658426726783156020576256000000, 202824096036516704239472512860160000000, 253530120045645880299340641075200000000, 316912650057057350374175801344000000000, 198070406285660843983859875840000000000, 247588007857076054979824844800000000000, 309485009821345068724781056000000000000, 193428131138340667952988160000000000000, 241785163922925834941235200000000000000, 302231454903657293676544000000000000000, 188894659314785808547840000000000000000, 236118324143482260684800000000000000000, 295147905179352825856000000000000000000, 184467440737095516160000000000000000000, 230584300921369395200000000000000000000, 288230376151711744000000000000000000000, 180143985094819840000000000000000000000, 225179981368524800000000000000000000000, 281474976710656000000000000000000000000, 175921860444160000000000000000000000000, 219902325555200000000000000000000000000, 274877906944000000000000000000000000000, 171798691840000000000000000000000000000, 214748364800000000000000000000000000000, 268435456000000000000000000000000000000, 335544320000000000000000000000000000000, 209715200000000000000000000000000000000, 262144000000000000000000000000000000000, 327680000000000000000000000000000000000, 204800000000000000000000000000000000000, 256000000000000000000000000000000000000, 320000000000000000000000000000000000000, 200000000000000000000000000000000000000, 250000000000000000000000000000000000000, 312500000000000000000000000000000000000, 195312500000000000000000000000000000000, 244140625000000000000000000000000000000, 305175781250000000000000000000000000000, 190734863281250000000000000000000000000, 238418579101562500000000000000000000000, 298023223876953125000000000000000000000, 186264514923095703125000000000000000000, 232830643653869628906250000000000000000, 291038304567337036132812500000000000000, 181898940354585647583007812500000000000, 227373675443232059478759765625000000000, 284217094304040074348449707031250000000, 177635683940025046467781066894531250000, 222044604925031308084726333618164062500, 277555756156289135105907917022705078125, 173472347597680709441192448139190673829, 216840434497100886801490560173988342286, 271050543121376108501863200217485427857, 338813178901720135627329000271856784821, 211758236813575084767080625169910490513, 264697796016968855958850781462388113142, 330872245021211069948563476827985141427, 206795153138256918717852173017490713392, 258493941422821148397315216271863391740, 323117426778526435496644020339829239675, 201948391736579022185402512712393274797, 252435489670723777731753140890491593496, 315544362088404722164691426113114491870, 197215226305252951352932141320696557419, 246519032881566189191165176650870696773, 308148791101957736488956470813588370967, 192592994438723585305597794258492731854, 240741243048404481631997242823115914818, 300926553810505602039996553528894893522, 188079096131566001274997845955559308451, 235098870164457501593747307444449135564, 293873587705571876992184134305561419455, 183670992315982423120115083940975887160, 229588740394978028900143854926219858949, 286985925493722536125179818657774823687, 179366203433576585078237386661109264804, 224207754291970731347796733326386581005, 280259692864963414184745916657983226257, 175162308040602133865466197911239516411, 218952885050752667331832747389049395513, 273691106313440834164790934236311744391, 171056941445900521352994333897694840245, 213821176807375651691242917372118550306, 267276471009219564614053646715148187882, 334095588761524455767567058393935234852, 208809742975952784854729411496209521783, 261012178719940981068411764370261902229, 326265223399926226335514705462827377786, 203915764624953891459696690914267111116, 254894705781192364324620863642833888895, 318618382226490455405776079553542361119, 199136488891556534628610049720963975699, 248920611114445668285762562151204969624, 311150763893057085357203202689006212030, 194469227433160678348252001680628882519, 243086534291450847935315002100786103149, 303858167864313559919143752625982628936, 189911354915195974949464845391239143085, 237389193643994968686831056739048928856, 296736492054993710858538820923811161070, 185460307534371069286586763077381975669, 231825384417963836608233453846727469586, 289781730522454795760291817308409336982, 181113581576534247350182385817755835614, 226391976970667809187727982272194794518, 282989971213334761484659977840243493147, 176868732008334225927912486150152183217, 221085915010417782409890607687690229021, 276357393763022228012363259609612786276, 172723371101888892507727037256007991423, 215904213877361115634658796570009989278, 269880267346701394543323495712512486598, 337350334183376743179154369640640608247, 210843958864610464486971481025400380155, 263554948580763080608714351281750475193, 329443685725953850760892939102188093991, 205902303578721156725558086938867558745, 257377879473401445906947608673584448431, 321722349341751807383684510841980560539, 201076468338594879614802819276237850337, 251345585423243599518503524095297312921, 314181981779054499398129405119121641151, 196363738611909062123830878199451025720, 245454673264886327654788597749313782149, 306818341581107909568485747186642227686, 191761463488192443480303591991651392304, 239701829360240554350379489989564240380, 299627286700300692937974362486955300475, 187267054187687933086233976554347062797, 234083817734609916357792470692933828496, 292604772168262395447240588366167285620, 182877982605163997154525367728854553513, 228597478256454996443156709661068191891, 285746847820568745553945887076335239863, 178591779887855465971216179422709524915, 223239724859819332464020224278386906143, 279049656074774165580025280347983632679, 174406035046733853487515800217489770425, 218007543808417316859394750271862213031, 272509429760521646074243437839827766288, 170318393600326028796402148649892353930, 212897992000407535995502685812365442413, 266122490000509419994378357265456803016, 332653112500636774992972946581821003770, 207908195312897984370608091613638127356, 259885244141122480463260114517047659195, 324856555176403100579075143146309573994, 203035346985251937861921964466443483746, 253794183731564922327402455583054354683, 317242729664456152909253069478817943353, 198276706040285095568283168424261214596, 247845882550356369460353960530326518245, 309807353187945461825442450662908147806, 193629595742465913640901531664317592379, 242036994678082392051126914580396990474, 302546243347602990063908643225496238092, 189091402092251868789942902015935148808, 236364252615314835987428627519918936009, 295455315769143544984285784399898670012, 184659572355714715615178615249936668757, 230824465444643394518973269062420835947, 288530581805804243148716586328026044933, 180331613628627651967947866455016278083, 225414517035784564959934833068770347604, 281768146294730706199918541335962934505, 176105091434206691374949088334976834066, 220131364292758364218686360418721042582, 275164205365947955273357950523401303228, 171977628353717472045848719077125814518, 214972035442146840057310898846407268147, 268715044302683550071638623558009085183, 335893805378354437589548279447511356479, 209933628361471523493467674654694597800, 262417035451839404366834593318368247249, 328021294314799255458543241647960309062, 205013308946749534661589526029975193164, 256266636183436918326986907537468991454, 320333295229296147908733634421836239318, 200208309518310092442958521513647649574, 250260386897887615553698151892059561967, 312825483622359519442122689865074452459, 195515927263974699651326681165671532787, 244394909079968374564158351457089415984, 305493636349960468205197939321361769979, 190933522718725292628248712075851106237, 238666903398406615785310890094813882797, 298333629248008269731638612618517353496, 186458518280005168582274132886573345935, 233073147850006460727842666108216682419, 291341434812508075909803332635270853023, 182088396757817547443627082897044283140, 227610495947271934304533853621305353924, 284513119934089917880667317026631692405, 177820699958806198675417073141644807754, 222275874948507748344271341427056009692, 277844843685634685430339176783820012115, 173653027303521678393961985489887507572, 217066284129402097992452481862359384465, 271332855161752622490565602327949230581, 339166068952190778113207002909936538226, 211978793095119236320754376818710336391, 264973491368899045400942971023387920489, 331216864211123806751178713779234900611, 207010540131952379219486696112021812882, 258763175164940474024358370140027266102, 323453968956175592530447962675034082628, 202158730597609745331529976671896301643, 252698413247012181664412470839870377053, 315873016558765227080515588549837971316, 197420635349228266925322242843648732073, 246775794186535333656652803554560915091, 308469742733169167070816004443201143864, 192793589208230729419260002777000714915, 240991986510288411774075003471250893644, 301239983137860514717593754339063617054, 188274989461162821698496096461914760659, 235343736826453527123120120577393450824, 294179671033066908903900150721741813530, 183862294395666818064937594201088633456, 229827867994583522581171992751360791820, 287284834993229403226464990939200989775, 179553021870768377016540619337000618610, 224441277338460471270675774171250773262, 280551596673075589088344717714063466577, 175344747920672243180215448571289666611, 219180934900840303975269310714112083264, 273976168626050379969086638392640104079, 171235105391281487480679148995400065050, 214043881739101859350848936244250081312, 267554852173877324188561170305312601640, 334443565217346655235701462881640752050, 209027228260841659522313414301025470031, 261284035326052074402891767876281837539, 326605044157565093003614709845352296924, 204128152598478183127259193653345185578, 255160190748097728909073992066681481972, 318950238435122161136342490083351852465, 199343899021951350710214056302094907791, 249179873777439188387767570377618634738, 311474842221798985484709462972023293422, 194671776388624365927943414357514558389, 243339720485780457409929267946893197986, 304174650607225571762411584933616497483, 190109156629515982351507240583510310927, 237636445786894977939384050729387888659, 297045557233618722424230063411734860823, 185653473271011701515143789632334288015, 232066841588764626893929737040417860018, 290083551985955783617412171300522325023, 181302219991222364760882607062826453139, 226627774989027955951103258828533066424, 283284718736284944938879073535666333030, 177052949210178090586799420959791458144, 221316186512722613233499276199739322680, 276645233140903266541874095249674153350, 172903270713064541588671309531046345844, 216129088391330676985839136913807932304, 270161360489163346232298921142259915380, 337701700611454182790373651427824894225, 211063562882158864243983532142390558891, 263829453602698580304979415177988198614, 329786817003373225381224268972485248267, 206116760627108265863265168107803280167, 257645950783885332329081460134754100209, 322057438479856665411351825168442625261, 201285899049910415882094890730276640788, 251607373812388019852618613412845800985, 314509217265485024815773266766057251231, 196568260790928140509858291728785782020, 245710325988660175637322864660982227524, 307137907485825219546653580826227784405, 191961192178640762216658488016392365254, 239951490223300952770823110020490456567, 299939362779126190963528887525613070708, 187462101736953869352205554703508169193, 234327627171192336690256943379385211491, 292909533963990420862821179224231514364, 183068458727494013039263237015144696478, 228835573409367516299079046268930870597, 286044466761709395373848807836163588246, 178777791726068372108655504897602242654, 223472239657585465135819381122002803317, 279340299571981831419774226402503504146, 174587687232488644637358891501564690092, 218234609040610805796698614376955862614, 272793261300763507245873267971194828268, 170495788312977192028670792481996767668, 213119735391221490035838490602495959584, 266399669239026862544798113253119949480, 332999586548783578180997641566399936850, 208124741592989736363123525978999960532, 260155926991237170453904407473749950665, 325194908739046463067380509342187438331, 203246817961904039417112818338867148957, 254058522452380049271391022923583936196, 317573153065475061589238778654479920245, 198483220665921913493274236659049950153, 248104025832402391866592795823812437691, 310130032290502989833240994779765547114, 193831270181564368645775621737353466946, 242289087726955460807219527171691833683, 302861359658694326009024408964614792103, 189288349786683953755640255602884245065, 236610437233354942194550319503605306331, 295763046541693677743187899379506632914, 184851904088558548589492437112191645571, 231064880110698185736865546390239556964, 288831100138372732171081932987799446205, 180519437586482957606926208117374653878, 225649296983103697008657760146718317347, 282061621228879621260822200183397896684, 176288513268049763288013875114623685428, 220360641585062204110017343893279606785, 275450801981327755137521679866599508481, 172156751238329846960951049916624692801, 215195939047912308701188812395780866001, 268994923809890385876486015494726082501, 336243654762362982345607519368407603126, 210152284226476863966004699605254751954, 262690355283096079957505874506568439942, 328362944103870099946882343133210549928] }>
    %6 = pop.string.address %string_1
    %7 = pop.pointer.bitcast %6 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %8 = kgen.struct.create(%7, %index57) : !kgen.struct<(pointer<none>, index)>
    %9 = kgen.struct.create(%index57, %index6, %8) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %10 = pop.cast_from_builtin %arg1 : index to !pop.scalar<index>
    %11 = pop.cast %10 : !pop.scalar<index> to !pop.scalar<uindex>
    %12 = pop.cmp lt(%11, %simd) : <1, uindex>
    %13 = pop.cast_to_builtin %12 : !pop.scalar<bool> to i1
    %14 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
    pop.stack_alloc.lifetime.start(%14) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
    pop.store %array, %14 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
    %15 = pop.pointer.bitcast %14 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<index>
    %16 = pop.load %15 : !kgen.pointer<index>
    %17 = index.cmp eq(%16, %index-1)
    %18 = pop.select %17, %index0, %index-1 : index
    pop.stack_alloc.lifetime.end(%14) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
    %19 = index.cmp eq(%18, %index-1)
    %20 = hlcf.if %19 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
      %74 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
      pop.stack_alloc.lifetime.start(%74) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
      pop.store %array, %74 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
      %75 = pop.pointer.bitcast %74 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
      %76 = pop.load %75 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
      pop.stack_alloc.lifetime.end(%74) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
      hlcf.yield %76 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    } else {
      hlcf.yield %4 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    }
    %21 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
    pop.stack_alloc.lifetime.start(%21) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %22 = kgen.struct.gep %21[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index6, %22 : !kgen.pointer<index>
    %23 = kgen.struct.gep %21[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %24 = pop.string.address %string_12
    %25 = pop.pointer.bitcast %24 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    pop.store %25, %23 : !kgen.pointer<pointer<none>>
    %26 = kgen.struct.gep %21[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index2305843009213693952, %26 : !kgen.pointer<index>
    %27 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
    pop.stack_alloc.lifetime.start(%27) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %28 = kgen.struct.gep %27[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index39, %28 : !kgen.pointer<index>
    %29 = kgen.struct.gep %27[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %30 = pop.string.address %string_11
    %31 = pop.pointer.bitcast %30 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    pop.store %31, %29 : !kgen.pointer<pointer<none>>
    %32 = kgen.struct.gep %27[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index2305843009213693952, %32 : !kgen.pointer<index>
    %33 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
    %34 = pop.pointer.bitcast %33 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
    pop.store %20, %34 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
    %35 = pop.load %33 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
    %36 = pop.array.get %35[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
    %37 = pop.array.create [%36] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
    %38 = kgen.struct.create(%37) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
    %39 = pop.string.address %string_2
    %40 = pop.pointer.bitcast %39 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %41 = kgen.struct.create(%40, %index53) : !kgen.struct<(pointer<none>, index)>
    %42 = kgen.struct.create(%index330, %index27, %41) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %43 = pop.string.address %string_3
    %44 = pop.pointer.bitcast %43 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %45 = kgen.struct.create(%44, %index53) : !kgen.struct<(pointer<none>, index)>
    %46 = kgen.struct.create(%index610, %index18, %45) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %47 = pop.string.address %string_5
    %48 = pop.pointer.bitcast %47 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %49 = pop.string.address %string_6
    %50 = pop.pointer.bitcast %49 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %51 = pop.string.address %string_7
    %52 = pop.pointer.bitcast %51 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %53 = pop.string.address %string_8
    %54 = pop.pointer.bitcast %53 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %55 = kgen.struct.create(%52, %index0) : !kgen.struct<(pointer<none>, index)>
    %56 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
    %57 = pop.pointer.bitcast %56 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
    pop.store %38, %56 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
    hlcf.if %13 {
      hlcf.yield
    } else {
      %74 = pop.stack_allocation 2048 x scalar<ui8> align 1
      %75 = pop.pointer.bitcast %74 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<none>
      %76 = kgen.struct.create(%75, %index0) : !kgen.struct<(pointer<none>, index) memoryOnly>
      %77 = kgen.call @"std::collections::string::string::String::write_to[::Writer](::String,$0&),writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%21, %76) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
      %78 = kgen.call @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%arg1, %77) : (index, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
      %79 = kgen.call @"std::collections::string::string::String::write_to[::Writer](::String,$0&),writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%27, %78) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
      %80 = kgen.call @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%idx618, %79) : (index, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
      %81 = kgen.struct.extract %80[0] : <(pointer<none>, index) memoryOnly>
      %82 = kgen.struct.extract %80[1] : <(pointer<none>, index) memoryOnly>
      %83 = index.add %82, %index1
      %84 = index.cmp sgt(%83, %index2048)
      hlcf.if %84 {
        kgen.call @"std::io::io::_printf[KGENParamList[::AnyType],::StringSlice[::Bool(False), StaticConstantOrigin, *?],*::AnyType,LITImmutOrigin,::Origin[::Bool(False), $3]](*$0,file:::FileDescriptor),types.values`=[],fmt={ #interp.memref<{[(#interp.memory_handle<16, \22HEAP_BUFFER_BYTES exceeded, increase with: `mojo -D HEAP_BUFFER_BYTES=4096`\\0A\\00\22 string>, const_global, [], [])], []}, 0, 0>, 76 }"(%index1) : (index) -> ()
        llvm.intr.trap
        hlcf.loop "_loop_0" {
          hlcf.continue "_loop_0"
        }
        kgen.unreachable
      } else {
        hlcf.yield
      }
      %85 = pop.pointer.bitcast %81 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %86 = pop.offset %85[%82] : !kgen.pointer<scalar<ui8>>
      pop.store %simd_9, %86 : !kgen.pointer<scalar<ui8>>
      %87 = pop.load %56 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
      %88 = kgen.struct.extract %87[0] : <(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
      %89 = pop.array.get %88[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
      %90 = pop.array.create [%89] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
      %91 = kgen.struct.create(%90) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
      %92 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
      pop.store %91, %92 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
      %93 = pop.pointer.bitcast %92 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
      %94 = pop.load %93 : !kgen.pointer<index>
      %95 = index.cmp eq(%94, %index-1)
      %96 = pop.select %95, %index0, %index-1 : index
      %97 = index.cmp eq(%96, %index-1)
      %98 = hlcf.if %97 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
        %99 = pop.load %56 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
        %100 = kgen.struct.extract %99[0] : <(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
        %101 = pop.array.get %100[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
        %102 = pop.array.create [%101] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
        %103 = kgen.struct.create(%102) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
        %104 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
        pop.store %103, %104 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
        %105 = pop.pointer.bitcast %104 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
        %106 = pop.load %105 : !kgen.pointer<index>
        %107 = index.cmp eq(%106, %index-1)
        %108 = pop.select %107, %index0, %index-1 : index
        %109 = index.cmp eq(%108, %index-1)
        %110 = pop.xor %109, %0
        hlcf.if %110 {
          %112 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
          pop.stack_alloc.lifetime.start(%112) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
          %113 = kgen.struct.gep %112[1] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index192, %113 : !kgen.pointer<index>
          %114 = kgen.struct.gep %112[0] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %54, %114 : !kgen.pointer<pointer<none>>
          %115 = kgen.struct.gep %112[2] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index2305843009213693952, %115 : !kgen.pointer<index>
          %116 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
          %117 = pop.pointer.bitcast %116 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
          pop.store %42, %117 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
          %118 = pop.load %116 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
          %119 = pop.array.get %118[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
          %120 = pop.array.create [%119] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
          %121 = kgen.struct.create(%120) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
          %122 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
          %123 = pop.pointer.bitcast %122 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
          pop.store %121, %122 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
          %124 = pop.pointer.bitcast %122 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
          %125 = pop.load %124 : !kgen.pointer<index>
          %126 = index.cmp eq(%125, %index-1)
          %127 = pop.select %126, %index0, %index-1 : index
          %128 = index.cmp eq(%127, %index-1)
          %129 = hlcf.if %128 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
            %147 = pop.load %123 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
            hlcf.yield %147 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
          } else {
            hlcf.yield %46 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
          }
          %130 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
          pop.stack_alloc.lifetime.start(%130) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
          %131 = kgen.struct.gep %130[1] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index1, %131 : !kgen.pointer<index>
          %132 = kgen.struct.gep %130[0] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %48, %132 : !kgen.pointer<pointer<none>>
          %133 = kgen.struct.gep %130[2] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index2305843009213693952, %133 : !kgen.pointer<index>
          %134 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
          pop.stack_alloc.lifetime.start(%134) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
          %135 = kgen.struct.gep %134[1] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index2, %135 : !kgen.pointer<index>
          %136 = kgen.struct.gep %134[0] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %50, %136 : !kgen.pointer<pointer<none>>
          %137 = kgen.struct.gep %134[2] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index2305843009213693952, %137 : !kgen.pointer<index>
          kgen.call @"std::io::io::print[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?],flush:::Bool,file:::FileDescriptor$),Ts.values`=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::reflection::location::SourceLocation\22>>, struct<(index, index, struct<(pointer<none>, index)>)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]]"(%struct_4, %130, %129, %134, %112, %55, %struct, %0, %index1) : (!kgen.struct<(pointer<none>, index)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index)>, i1, index owned) -> ()
          %138 = pop.load %133 : !kgen.pointer<index>
          %139 = index.and %138, %index4611686018427387904
          %140 = index.cmp ne(%139, %index0)
          hlcf.if %140 {
            %147 = pop.load %132 : !kgen.pointer<pointer<none>>
            %148 = pop.pointer.bitcast %147 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
            %149 = pop.offset %148[%idx-8] : !kgen.pointer<scalar<ui8>>
            %150 = pop.pointer.bitcast %149 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
            %151 = kgen.struct.gep %150[0] : <struct<(scalar<index>) memoryOnly>>
            %152 = pop.atomic.rmw sub(%151, %simd_10) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
            %153 = pop.cmp eq(%152, %simd_10) : <1, index>
            %154 = pop.cast_to_builtin %153 : !pop.scalar<bool> to i1
            hlcf.if %154 {
              pop.fence syncscope("") acquire
              pop.aligned_free %149 : <scalar<ui8>>
              hlcf.yield
            } else {
              hlcf.yield
            }
            hlcf.yield
          } else {
            hlcf.yield
          }
          %141 = pop.load %137 : !kgen.pointer<index>
          %142 = index.and %141, %index4611686018427387904
          %143 = index.cmp ne(%142, %index0)
          hlcf.if %143 {
            %147 = pop.load %136 : !kgen.pointer<pointer<none>>
            %148 = pop.pointer.bitcast %147 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
            %149 = pop.offset %148[%idx-8] : !kgen.pointer<scalar<ui8>>
            %150 = pop.pointer.bitcast %149 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
            %151 = kgen.struct.gep %150[0] : <struct<(scalar<index>) memoryOnly>>
            %152 = pop.atomic.rmw sub(%151, %simd_10) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
            %153 = pop.cmp eq(%152, %simd_10) : <1, index>
            %154 = pop.cast_to_builtin %153 : !pop.scalar<bool> to i1
            hlcf.if %154 {
              pop.fence syncscope("") acquire
              pop.aligned_free %149 : <scalar<ui8>>
              hlcf.yield
            } else {
              hlcf.yield
            }
            hlcf.yield
          } else {
            hlcf.yield
          }
          %144 = pop.load %115 : !kgen.pointer<index>
          %145 = index.and %144, %index4611686018427387904
          %146 = index.cmp ne(%145, %index0)
          hlcf.if %146 {
            %147 = pop.load %114 : !kgen.pointer<pointer<none>>
            %148 = pop.pointer.bitcast %147 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
            %149 = pop.offset %148[%idx-8] : !kgen.pointer<scalar<ui8>>
            %150 = pop.pointer.bitcast %149 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
            %151 = kgen.struct.gep %150[0] : <struct<(scalar<index>) memoryOnly>>
            %152 = pop.atomic.rmw sub(%151, %simd_10) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
            %153 = pop.cmp eq(%152, %simd_10) : <1, index>
            %154 = pop.cast_to_builtin %153 : !pop.scalar<bool> to i1
            hlcf.if %154 {
              pop.fence syncscope("") acquire
              pop.aligned_free %149 : <scalar<ui8>>
              hlcf.yield
            } else {
              hlcf.yield
            }
            hlcf.yield
          } else {
            hlcf.yield
          }
          llvm.intr.trap
          hlcf.loop "_loop_0" {
            hlcf.continue "_loop_0"
          }
          kgen.unreachable
        } else {
          hlcf.yield
        }
        %111 = pop.load %57 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
        hlcf.yield %111 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
      } else {
        hlcf.yield %9 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
      }
      kgen.call @"std::builtin::debug_assert::_debug_assert_msg[LITImmutOrigin,::Origin[::Bool(False), $0]](::UnsafePointer[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1, ::AddressSpace(::Int(0))],::Int,::SourceLocation)"(%81, %98) : (!kgen.pointer<none>, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>) -> ()
      hlcf.yield
    }
    %58 = pop.load %26 : !kgen.pointer<index>
    %59 = index.and %58, %index4611686018427387904
    %60 = index.cmp ne(%59, %index0)
    hlcf.if %60 {
      %74 = pop.load %23 : !kgen.pointer<pointer<none>>
      %75 = pop.pointer.bitcast %74 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %76 = pop.offset %75[%idx-8] : !kgen.pointer<scalar<ui8>>
      %77 = pop.pointer.bitcast %76 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
      %78 = kgen.struct.gep %77[0] : <struct<(scalar<index>) memoryOnly>>
      %79 = pop.atomic.rmw sub(%78, %simd_10) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
      %80 = pop.cmp eq(%79, %simd_10) : <1, index>
      %81 = pop.cast_to_builtin %80 : !pop.scalar<bool> to i1
      hlcf.if %81 {
        pop.fence syncscope("") acquire
        pop.aligned_free %76 : <scalar<ui8>>
        hlcf.yield
      } else {
        hlcf.yield
      }
      hlcf.yield
    } else {
      hlcf.yield
    }
    %61 = pop.load %32 : !kgen.pointer<index>
    %62 = index.and %61, %index4611686018427387904
    %63 = index.cmp ne(%62, %index0)
    hlcf.if %63 {
      %74 = pop.load %29 : !kgen.pointer<pointer<none>>
      %75 = pop.pointer.bitcast %74 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %76 = pop.offset %75[%idx-8] : !kgen.pointer<scalar<ui8>>
      %77 = pop.pointer.bitcast %76 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
      %78 = kgen.struct.gep %77[0] : <struct<(scalar<index>) memoryOnly>>
      %79 = pop.atomic.rmw sub(%78, %simd_10) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
      %80 = pop.cmp eq(%79, %simd_10) : <1, index>
      %81 = pop.cast_to_builtin %80 : !pop.scalar<bool> to i1
      hlcf.if %81 {
        pop.fence syncscope("") acquire
        pop.aligned_free %76 : <scalar<ui8>>
        hlcf.yield
      } else {
        hlcf.yield
      }
      hlcf.yield
    } else {
      hlcf.yield
    }
    %64 = kgen.struct.gep %5[0] : <struct<(array<619, scalar<ui128>>) memoryOnly>>
    %65 = pop.array.gep %64[%arg1] : <array<619, scalar<ui128>>>
    %66 = pop.load %65 : !kgen.pointer<scalar<ui128>>
    %67 = kgen.call tail @"std::builtin::_format_float::_umul192_upper128[::DType](::SIMD[$0, ::Int(1)],::SIMD[::DType(uint128), ::Int(1)]),CarrierDType=ui64"(%arg0, %66) : (!pop.scalar<ui64>, !pop.scalar<ui128>) -> !pop.scalar<ui128>
    %68 = pop.shr %67, %simd_0 : !pop.scalar<ui128>
    %69 = pop.cast fast %68 : !pop.scalar<ui128> to !pop.scalar<ui64>
    %70 = pop.cast fast %67 : !pop.scalar<ui128> to !pop.scalar<ui64>
    %71 = pop.cmp eq(%70, %simd_13) : <1, ui64>
    %72 = pop.cast_to_builtin %71 : !pop.scalar<bool> to i1
    %73 = kgen.struct.create(%69, %72) : !kgen.struct<(scalar<ui64>, i1)>
    kgen.return %73 : !kgen.struct<(scalar<ui64>, i1)>
  }
  kgen.func @"std::builtin::_format_float::_compute_delta[::DType,::Int,::Int](::Int,::Int),CarrierDType=ui64,total_bits=64,cache_bits=128"(%arg0: index, %arg1: index) -> !pop.scalar<ui64> {
    %string = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/builtin/_format_float.mojo">
    %simd = kgen.param.constant: scalar<uindex> = <619>
    %idx618 = index.constant 618
    %index54 = kgen.param.constant = <54>
    %index49 = kgen.param.constant = <49>
    %index780 = kgen.param.constant = <780>
    %simd_0 = kgen.param.constant: scalar<ui128> = <64>
    %string_1 = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/collections/check_bounds.mojo">
    %index57 = kgen.param.constant = <57>
    %string_2 = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/builtin/debug_assert.mojo">
    %index27 = kgen.param.constant = <27>
    %index330 = kgen.param.constant = <330>
    %index53 = kgen.param.constant = <53>
    %string_3 = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/collections/optional.mojo">
    %index18 = kgen.param.constant = <18>
    %index610 = kgen.param.constant = <610>
    %idx-8 = index.constant -8
    %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle, const_global, [], [])], []}, 0, 0>, 1 }>
    %struct_4 = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle4, const_global, [], [])], []}, 0, 0>, 6 }>
    %string_5 = kgen.param.constant: string = <" ">
    %index2 = kgen.param.constant = <2>
    %string_6 = kgen.param.constant: string = <": ">
    %string_7 = kgen.param.constant: string = <"">
    %string_8 = kgen.param.constant: string = <"`Optional.value()` called on empty `Optional`. Consider using `if optional:` to check whether the `Optional` is empty before calling `.value()`, or use `.or_else()` to provide a default value.">
    %index192 = kgen.param.constant = <192>
    %simd_9 = kgen.param.constant: scalar<ui8> = <0>
    %index2048 = kgen.param.constant = <2048>
    %index-1 = kgen.param.constant = <-1>
    %index0 = kgen.param.constant = <0>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %simd_10 = kgen.param.constant: scalar<index> = <1>
    %string_11 = kgen.param.constant: string = <" is out of bounds, valid range is 0 to ">
    %index39 = kgen.param.constant = <39>
    %string_12 = kgen.param.constant: string = <"index ">
    %index6 = kgen.param.constant = <6>
    %index2305843009213693952 = kgen.param.constant = <2305843009213693952>
    %index1 = kgen.param.constant = <1>
    %none = kgen.param.constant: none = <#kgen.none>
    %array = kgen.param.constant: array<1, struct<(index, index, struct<(pointer<none>, index)>)>> = <[{ -1, 0, { 0, 0 } }]>
    %0 = kgen.param.constant: i1 = <1>
    %index63 = kgen.param.constant = <63>
    %1 = pop.string.address %string
    %2 = pop.pointer.bitcast %1 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %3 = kgen.struct.create(%2, %index54) : !kgen.struct<(pointer<none>, index)>
    %4 = kgen.struct.create(%index780, %index49, %3) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %5 = pop.global_constant: struct<(array<619, scalar<ui128>>) memoryOnly> = <{ [339574632262346462319337857089816694651, 212234145163966538949586160681135434157, 265292681454958173686982700851419292696, 331615851818697717108728376064274115870, 207259907386686073192955235040171322419, 259074884233357591491194043800214153024, 323843605291696989363992554750267691280, 202402253307310618352495346718917307050, 253002816634138272940619183398646633812, 316253520792672841175773979248308292265, 197658450495420525734858737030192682666, 247073063119275657168573421287740853332, 308841328899094571460716776609676066665, 193025830561934107162947985381047541666, 241282288202417633953684981726309427083, 301602860253022042442106227157886783853, 188501787658138776526316391973679239908, 235627234572673470657895489967099049885, 294534043215841838322369362458873812356, 184083777009901148951480851536796132723, 230104721262376436189351064420995165904, 287630901577970545236688830526243957379, 179769313486231590772930519078902473362, 224711641857789488466163148848628091703, 280889552322236860582703936060785114628, 175555970201398037864189960037990696643, 219444962751747547330237450047488370803, 274306203439684434162796812559360463504, 171441377149802771351748007849600289690, 214301721437253464189685009812000362113, 267877151796566830237106262265000452641, 334846439745708537796382827831250565801, 209279024841067836122739267394531603626, 261598781051334795153424084243164504532, 326998476314168493941780105303955630665, 204374047696355308713612565814972269166, 255467559620444135892015707268715336457, 319334449525555169865019634085894170571, 199584030953471981165637271303683856607, 249480038691839976457046589129604820759, 311850048364799970571308236412006025949, 194906280227999981607067647757503766218, 243632850284999977008834559696879707772, 304541062856249971261043199621099634715, 190338164285156232038151999763187271697, 237922705356445290047689999703984089622, 297403381695556612559612499629980112027, 185877113559722882849757812268737570017, 232346391949653603562197265335921962521, 290432989937067004452746581669902453151, 181520618710666877782966613543689033220, 226900773388333597228708266929611291524, 283625966735416996535885333662014114405, 177266229209635622834928333538758821504, 221582786512044528543660416923448526879, 276978483140055660679575521154310658599, 173111551962534787924734700721444161625, 216389439953168484905918375901805202031, 270486799941460606132397969877256502538, 338108499926825757665497462346570628173, 211317812454266098540935913966606642608, 264147265567832623176169892458258303260, 330184081959790778970212365572822879075, 206365051224869236856382728483014299422, 257956314031086546070478410603767874277, 322445392538858182588098013254709842846, 201528370336786364117561258284193651779, 251910462920982955146951572855242064724, 314888078651228693933689466069052580905, 196805049157017933708555916293157863066, 246006311446272417135694895366447328832, 307507889307840521419618619208059161040, 192192430817400325887261637005036975650, 240240538521750407359077046256296219562, 300300673152188009198846307820370274453, 187687920720117505749278942387731421533, 234609900900146882186598677984664276916, 293262376125183602733248347480830346145, 183288985078239751708280217175518966341, 229111231347799689635350271469398707926, 286389039184749612044187839336748384908, 178993149490468507527617399585467740568, 223741436863085634409521749481834675709, 279676796078857043011902186852293344636, 174797997549285651882438866782683340398, 218497496936607064853048583478354175497, 273121871170758831066310729347942719372, 170701169481724269416444205842464199607, 213376461852155336770555257303080249509, 266720577315194170963194071628850311886, 333400721643992713703992589536062889858, 208375451027495446064995368460039306161, 260469313784369307581244210575049132701, 325586642230461634476555263218811415877, 203491651394038521547847039511757134923, 254364564242548151934808799389696418654, 317955705303185189918510999237120523317, 198722315814490743699069374523200327073, 248402894768113429623836718154000408842, 310503618460141787029795897692500511052, 194064761537588616893622436057812819408, 242580951921985771117028045072266024259, 303226189902482213896285056340332530324, 189516368689051383685178160212707831453, 236895460861314229606472700265884789316, 296119326076642787008090875332355986645, 185074578797901741880056797082722491653, 231343223497377177350070996353403114566, 289179029371721471687588745441753893208, 180736893357325919804742965901096183255, 225921116696657399755928707376370229069, 282401395870821749694910884220462786336, 176500872419263593559319302637789241460, 220626090524079491949149128297236551825, 275782613155099364936436410371545689781, 172364133221937103085272756482216056113, 215455166527421378856590945602770070141, 269318958159276723570738682003462587677, 336648697699095904463423352504328234596, 210405436061934940289639595315205146623, 263006795077418675362049494144006433278, 328758493846773344202561867680008041597, 205474058654233340126601167300005025999, 256842573317791675158251459125006282498, 321053216647239593947814323906257853122, 200658260404524746217383952441411158202, 250822825505655932771729940551763947752, 313528531882069915964662425689704934690, 195955332426293697477914016056065584181, 244944165532867121847392520070081980227, 306180206916083902309240650087602475283, 191362629322552438943275406304751547052, 239203286653190548679094257880939433815, 299004108316488185848867822351174292269, 186877567697805116155542388969483932668, 233596959622256395194427986211854915835, 291996199527820493993034982764818644794, 182497624704887808745646864228011652996, 228122030881109760932058580285014566245, 285152538601387201165073225356268207806, 178220336625867000728170765847667629879, 222775420782333750910213457309584537349, 278469275977917188637766821636980671686, 174043297486198242898604263523112919804, 217554121857747803623255329403891149755, 271942652322184754529069161754863937193, 339928315402730943161336452193579921491, 212455197126706839475835282620987450932, 265568996408383549344794103276234313665, 331961245510479436680992629095292892081, 207475778444049647925620393184558057551, 259344723055062059907025491480697571939, 324180903818827574883781864350871964923, 202613064886767234302363665219294978077, 253266331108459042877954581524118722596, 316582913885573803597443226905148403245, 197864321178483627248402016815717752029, 247330401473104534060502521019647190036, 309163001841380667575628151274558987544, 193226876150862917234767594546599367215, 241533595188578646543459493183249209019, 301916993985723308179324366479061511274, 188698121241077067612077729049413444546, 235872651551346334515097161311766805683, 294840814439182918143871451639708507103, 184275509024489323839919657274817816940, 230344386280611654799899571593522271175, 287930482850764568499874464491902838968, 179956551781727855312421540307439274355, 224945689727159819140526925384299092944, 281182112158949773925658656730373866180, 175738820099343608703536660456483666363, 219673525124179510879420825570604582953, 274591906405224388599276031963255728691, 171619941503265242874547519977034830432, 214524926879081553593184399971293538040, 268156158598851941991480499964116922550, 335195198248564927489350624955146153187, 209496998905353079680844140596966345742, 261871248631691349601055175746207932178, 327339060789614187001318969682759915222, 204586912993508866875824356051724947014, 255733641241886083594780445064656183767, 319667051552357604493475556330820229709, 199791907220223502808422222706762643568, 249739884025279378510527778383453304460, 312174855031599223138159722979316630575, 195109284394749514461349826862072894110, 243886605493436893076687283577591117637, 304858256866796116345859104471988897046, 190536410541747572716161940294993060654, 238170513177184465895202425368741325818, 297713141471480582369003031710926657272, 186070713419675363980626894819329160795, 232588391774594204975783618524161450994, 290735489718242756219729523155201813742, 181709681073901722637330951972001133589, 227137101342377153296663689965001416986, 283921376677971441620829612456251771232, 177450860423732151013018507785157357020, 221813575529665188766273134731446696275, 277266969412081485957841418414308370344, 173291855882550928723650886508942731465, 216614819853188660904563608136178414331, 270768524816485826130704510170223017914, 338460656020607282663380637712778772393, 211537910012879551664612898570486732746, 264422387516099439580766123213108415932, 330527984395124299475957654016385519915, 206579990246952687172473533760240949947, 258224987808690858965591917200301187433, 322781234760863573706989896500376484292, 201738271725539733566868685312735302683, 252172839656924666958585856640919128353, 315216049571155833698232320801148910441, 197010030981972396061395200500718069026, 246262538727465495076744000625897586282, 307828173409331868845930000782371982853, 192392608380832418028706250488982489283, 240490760476040522535882813111228111604, 300613450595050653169853516389035139505, 187883406621906658231158447743146962191, 234854258277383322788948059678933702738, 293567822846729153486185074598667128422, 183479889279205720928865671624166955264, 229349861599007151161082089530208694080, 286687326998758938951352611912760867600, 179179579374224336844595382445475542250, 223974474217780421055744228056844427813, 279968092772225526319680285071055534766, 174980057982640953949800178169409709229, 218725072478301192437250222711762136536, 273406340597876490546562778389702670670, 170878962873672806591601736493564169169, 213598703592091008239502170616955211461, 266998379490113760299377713271194014326, 333747974362642200374222141588992517907, 208592483976651375233888838493120323692, 260740604970814219042361048116400404615, 325925756213517773802951310145500505769, 203703597633448608626844568840937816106, 254629497041810760783555711051172270132, 318286871302263450979444638813965337665, 198929294563914656862152899258728336041, 248661618204893321077691124073410420051, 310827022756116651347113905091763025063, 194266889222572907091946190682351890665, 242833611528216133864932738352939863331, 303542014410270167331165922941174829163, 189713759006418854581978701838234268227, 237142198758023568227473377297792835284, 296427748447529460284341721622241044105, 185267342779705912677713576013900652566, 231584178474632390847141970017375815707, 289480223093290488558927462521719769634, 180925139433306555349329664076074856021, 226156424291633194186662080095093570026, 282695530364541492733327600118866962533, 176684706477838432958329750074291851583, 220855883097298041197912187592864814479, 276069853871622551497390234491081018099, 172543658669764094685868896556925636312, 215679573337205118357336120696157045390, 269599466671506397946670150870196306737, 336999333339382997433337688587745383421, 210624583337114373395836055367340864638, 263280729171392966744795069209176080798, 329100911464241208430993836511470100997, 205688069665150755269371147819668813123, 257110087081438444086713934774586016404, 321387608851798055108392418468232520505, 200867255532373784442745261542645325316, 251084069415467230553431576928306656645, 313855086769334038191789471160383320806, 196159429230833773869868419475239575504, 245199286538542217337335524344049469379, 306499108173177771671669405430061836724, 191561942608236107294793378393788647953, 239452428260295134118491722992235809941, 299315535325368917648114653740294762426, 187072209578355573530071658587684226516, 233840261972944466912589573234605283145, 292300327466180583640736966543256603932, 182687704666362864775460604089535377457, 228359630832953580969325755111919221822, 285449538541191976211657193889899027277, 178405961588244985132285746181186892048, 223007451985306231415357182726483615060, 278759314981632789269196478408104518825, 174224571863520493293247799005065324266, 217780714829400616616559748756331655332, 272225893536750770770699685945414569165, 170141183460469231731687303715884105728, 212676479325586539664609129644855132160, 265845599156983174580761412056068915200, 332306998946228968225951765070086144000, 207691874341393105141219853168803840000, 259614842926741381426524816461004800000, 324518553658426726783156020576256000000, 202824096036516704239472512860160000000, 253530120045645880299340641075200000000, 316912650057057350374175801344000000000, 198070406285660843983859875840000000000, 247588007857076054979824844800000000000, 309485009821345068724781056000000000000, 193428131138340667952988160000000000000, 241785163922925834941235200000000000000, 302231454903657293676544000000000000000, 188894659314785808547840000000000000000, 236118324143482260684800000000000000000, 295147905179352825856000000000000000000, 184467440737095516160000000000000000000, 230584300921369395200000000000000000000, 288230376151711744000000000000000000000, 180143985094819840000000000000000000000, 225179981368524800000000000000000000000, 281474976710656000000000000000000000000, 175921860444160000000000000000000000000, 219902325555200000000000000000000000000, 274877906944000000000000000000000000000, 171798691840000000000000000000000000000, 214748364800000000000000000000000000000, 268435456000000000000000000000000000000, 335544320000000000000000000000000000000, 209715200000000000000000000000000000000, 262144000000000000000000000000000000000, 327680000000000000000000000000000000000, 204800000000000000000000000000000000000, 256000000000000000000000000000000000000, 320000000000000000000000000000000000000, 200000000000000000000000000000000000000, 250000000000000000000000000000000000000, 312500000000000000000000000000000000000, 195312500000000000000000000000000000000, 244140625000000000000000000000000000000, 305175781250000000000000000000000000000, 190734863281250000000000000000000000000, 238418579101562500000000000000000000000, 298023223876953125000000000000000000000, 186264514923095703125000000000000000000, 232830643653869628906250000000000000000, 291038304567337036132812500000000000000, 181898940354585647583007812500000000000, 227373675443232059478759765625000000000, 284217094304040074348449707031250000000, 177635683940025046467781066894531250000, 222044604925031308084726333618164062500, 277555756156289135105907917022705078125, 173472347597680709441192448139190673829, 216840434497100886801490560173988342286, 271050543121376108501863200217485427857, 338813178901720135627329000271856784821, 211758236813575084767080625169910490513, 264697796016968855958850781462388113142, 330872245021211069948563476827985141427, 206795153138256918717852173017490713392, 258493941422821148397315216271863391740, 323117426778526435496644020339829239675, 201948391736579022185402512712393274797, 252435489670723777731753140890491593496, 315544362088404722164691426113114491870, 197215226305252951352932141320696557419, 246519032881566189191165176650870696773, 308148791101957736488956470813588370967, 192592994438723585305597794258492731854, 240741243048404481631997242823115914818, 300926553810505602039996553528894893522, 188079096131566001274997845955559308451, 235098870164457501593747307444449135564, 293873587705571876992184134305561419455, 183670992315982423120115083940975887160, 229588740394978028900143854926219858949, 286985925493722536125179818657774823687, 179366203433576585078237386661109264804, 224207754291970731347796733326386581005, 280259692864963414184745916657983226257, 175162308040602133865466197911239516411, 218952885050752667331832747389049395513, 273691106313440834164790934236311744391, 171056941445900521352994333897694840245, 213821176807375651691242917372118550306, 267276471009219564614053646715148187882, 334095588761524455767567058393935234852, 208809742975952784854729411496209521783, 261012178719940981068411764370261902229, 326265223399926226335514705462827377786, 203915764624953891459696690914267111116, 254894705781192364324620863642833888895, 318618382226490455405776079553542361119, 199136488891556534628610049720963975699, 248920611114445668285762562151204969624, 311150763893057085357203202689006212030, 194469227433160678348252001680628882519, 243086534291450847935315002100786103149, 303858167864313559919143752625982628936, 189911354915195974949464845391239143085, 237389193643994968686831056739048928856, 296736492054993710858538820923811161070, 185460307534371069286586763077381975669, 231825384417963836608233453846727469586, 289781730522454795760291817308409336982, 181113581576534247350182385817755835614, 226391976970667809187727982272194794518, 282989971213334761484659977840243493147, 176868732008334225927912486150152183217, 221085915010417782409890607687690229021, 276357393763022228012363259609612786276, 172723371101888892507727037256007991423, 215904213877361115634658796570009989278, 269880267346701394543323495712512486598, 337350334183376743179154369640640608247, 210843958864610464486971481025400380155, 263554948580763080608714351281750475193, 329443685725953850760892939102188093991, 205902303578721156725558086938867558745, 257377879473401445906947608673584448431, 321722349341751807383684510841980560539, 201076468338594879614802819276237850337, 251345585423243599518503524095297312921, 314181981779054499398129405119121641151, 196363738611909062123830878199451025720, 245454673264886327654788597749313782149, 306818341581107909568485747186642227686, 191761463488192443480303591991651392304, 239701829360240554350379489989564240380, 299627286700300692937974362486955300475, 187267054187687933086233976554347062797, 234083817734609916357792470692933828496, 292604772168262395447240588366167285620, 182877982605163997154525367728854553513, 228597478256454996443156709661068191891, 285746847820568745553945887076335239863, 178591779887855465971216179422709524915, 223239724859819332464020224278386906143, 279049656074774165580025280347983632679, 174406035046733853487515800217489770425, 218007543808417316859394750271862213031, 272509429760521646074243437839827766288, 170318393600326028796402148649892353930, 212897992000407535995502685812365442413, 266122490000509419994378357265456803016, 332653112500636774992972946581821003770, 207908195312897984370608091613638127356, 259885244141122480463260114517047659195, 324856555176403100579075143146309573994, 203035346985251937861921964466443483746, 253794183731564922327402455583054354683, 317242729664456152909253069478817943353, 198276706040285095568283168424261214596, 247845882550356369460353960530326518245, 309807353187945461825442450662908147806, 193629595742465913640901531664317592379, 242036994678082392051126914580396990474, 302546243347602990063908643225496238092, 189091402092251868789942902015935148808, 236364252615314835987428627519918936009, 295455315769143544984285784399898670012, 184659572355714715615178615249936668757, 230824465444643394518973269062420835947, 288530581805804243148716586328026044933, 180331613628627651967947866455016278083, 225414517035784564959934833068770347604, 281768146294730706199918541335962934505, 176105091434206691374949088334976834066, 220131364292758364218686360418721042582, 275164205365947955273357950523401303228, 171977628353717472045848719077125814518, 214972035442146840057310898846407268147, 268715044302683550071638623558009085183, 335893805378354437589548279447511356479, 209933628361471523493467674654694597800, 262417035451839404366834593318368247249, 328021294314799255458543241647960309062, 205013308946749534661589526029975193164, 256266636183436918326986907537468991454, 320333295229296147908733634421836239318, 200208309518310092442958521513647649574, 250260386897887615553698151892059561967, 312825483622359519442122689865074452459, 195515927263974699651326681165671532787, 244394909079968374564158351457089415984, 305493636349960468205197939321361769979, 190933522718725292628248712075851106237, 238666903398406615785310890094813882797, 298333629248008269731638612618517353496, 186458518280005168582274132886573345935, 233073147850006460727842666108216682419, 291341434812508075909803332635270853023, 182088396757817547443627082897044283140, 227610495947271934304533853621305353924, 284513119934089917880667317026631692405, 177820699958806198675417073141644807754, 222275874948507748344271341427056009692, 277844843685634685430339176783820012115, 173653027303521678393961985489887507572, 217066284129402097992452481862359384465, 271332855161752622490565602327949230581, 339166068952190778113207002909936538226, 211978793095119236320754376818710336391, 264973491368899045400942971023387920489, 331216864211123806751178713779234900611, 207010540131952379219486696112021812882, 258763175164940474024358370140027266102, 323453968956175592530447962675034082628, 202158730597609745331529976671896301643, 252698413247012181664412470839870377053, 315873016558765227080515588549837971316, 197420635349228266925322242843648732073, 246775794186535333656652803554560915091, 308469742733169167070816004443201143864, 192793589208230729419260002777000714915, 240991986510288411774075003471250893644, 301239983137860514717593754339063617054, 188274989461162821698496096461914760659, 235343736826453527123120120577393450824, 294179671033066908903900150721741813530, 183862294395666818064937594201088633456, 229827867994583522581171992751360791820, 287284834993229403226464990939200989775, 179553021870768377016540619337000618610, 224441277338460471270675774171250773262, 280551596673075589088344717714063466577, 175344747920672243180215448571289666611, 219180934900840303975269310714112083264, 273976168626050379969086638392640104079, 171235105391281487480679148995400065050, 214043881739101859350848936244250081312, 267554852173877324188561170305312601640, 334443565217346655235701462881640752050, 209027228260841659522313414301025470031, 261284035326052074402891767876281837539, 326605044157565093003614709845352296924, 204128152598478183127259193653345185578, 255160190748097728909073992066681481972, 318950238435122161136342490083351852465, 199343899021951350710214056302094907791, 249179873777439188387767570377618634738, 311474842221798985484709462972023293422, 194671776388624365927943414357514558389, 243339720485780457409929267946893197986, 304174650607225571762411584933616497483, 190109156629515982351507240583510310927, 237636445786894977939384050729387888659, 297045557233618722424230063411734860823, 185653473271011701515143789632334288015, 232066841588764626893929737040417860018, 290083551985955783617412171300522325023, 181302219991222364760882607062826453139, 226627774989027955951103258828533066424, 283284718736284944938879073535666333030, 177052949210178090586799420959791458144, 221316186512722613233499276199739322680, 276645233140903266541874095249674153350, 172903270713064541588671309531046345844, 216129088391330676985839136913807932304, 270161360489163346232298921142259915380, 337701700611454182790373651427824894225, 211063562882158864243983532142390558891, 263829453602698580304979415177988198614, 329786817003373225381224268972485248267, 206116760627108265863265168107803280167, 257645950783885332329081460134754100209, 322057438479856665411351825168442625261, 201285899049910415882094890730276640788, 251607373812388019852618613412845800985, 314509217265485024815773266766057251231, 196568260790928140509858291728785782020, 245710325988660175637322864660982227524, 307137907485825219546653580826227784405, 191961192178640762216658488016392365254, 239951490223300952770823110020490456567, 299939362779126190963528887525613070708, 187462101736953869352205554703508169193, 234327627171192336690256943379385211491, 292909533963990420862821179224231514364, 183068458727494013039263237015144696478, 228835573409367516299079046268930870597, 286044466761709395373848807836163588246, 178777791726068372108655504897602242654, 223472239657585465135819381122002803317, 279340299571981831419774226402503504146, 174587687232488644637358891501564690092, 218234609040610805796698614376955862614, 272793261300763507245873267971194828268, 170495788312977192028670792481996767668, 213119735391221490035838490602495959584, 266399669239026862544798113253119949480, 332999586548783578180997641566399936850, 208124741592989736363123525978999960532, 260155926991237170453904407473749950665, 325194908739046463067380509342187438331, 203246817961904039417112818338867148957, 254058522452380049271391022923583936196, 317573153065475061589238778654479920245, 198483220665921913493274236659049950153, 248104025832402391866592795823812437691, 310130032290502989833240994779765547114, 193831270181564368645775621737353466946, 242289087726955460807219527171691833683, 302861359658694326009024408964614792103, 189288349786683953755640255602884245065, 236610437233354942194550319503605306331, 295763046541693677743187899379506632914, 184851904088558548589492437112191645571, 231064880110698185736865546390239556964, 288831100138372732171081932987799446205, 180519437586482957606926208117374653878, 225649296983103697008657760146718317347, 282061621228879621260822200183397896684, 176288513268049763288013875114623685428, 220360641585062204110017343893279606785, 275450801981327755137521679866599508481, 172156751238329846960951049916624692801, 215195939047912308701188812395780866001, 268994923809890385876486015494726082501, 336243654762362982345607519368407603126, 210152284226476863966004699605254751954, 262690355283096079957505874506568439942, 328362944103870099946882343133210549928] }>
    %6 = pop.string.address %string_1
    %7 = pop.pointer.bitcast %6 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %8 = kgen.struct.create(%7, %index57) : !kgen.struct<(pointer<none>, index)>
    %9 = kgen.struct.create(%index57, %index6, %8) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %10 = pop.cast_from_builtin %arg0 : index to !pop.scalar<index>
    %11 = pop.cast %10 : !pop.scalar<index> to !pop.scalar<uindex>
    %12 = pop.cmp lt(%11, %simd) : <1, uindex>
    %13 = pop.cast_to_builtin %12 : !pop.scalar<bool> to i1
    %14 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
    pop.stack_alloc.lifetime.start(%14) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
    pop.store %array, %14 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
    %15 = pop.pointer.bitcast %14 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<index>
    %16 = pop.load %15 : !kgen.pointer<index>
    %17 = index.cmp eq(%16, %index-1)
    %18 = pop.select %17, %index0, %index-1 : index
    pop.stack_alloc.lifetime.end(%14) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
    %19 = index.cmp eq(%18, %index-1)
    %20 = hlcf.if %19 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
      %73 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
      pop.stack_alloc.lifetime.start(%73) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
      pop.store %array, %73 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
      %74 = pop.pointer.bitcast %73 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
      %75 = pop.load %74 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
      pop.stack_alloc.lifetime.end(%73) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
      hlcf.yield %75 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    } else {
      hlcf.yield %4 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    }
    %21 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
    pop.stack_alloc.lifetime.start(%21) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %22 = kgen.struct.gep %21[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index6, %22 : !kgen.pointer<index>
    %23 = kgen.struct.gep %21[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %24 = pop.string.address %string_12
    %25 = pop.pointer.bitcast %24 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    pop.store %25, %23 : !kgen.pointer<pointer<none>>
    %26 = kgen.struct.gep %21[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index2305843009213693952, %26 : !kgen.pointer<index>
    %27 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
    pop.stack_alloc.lifetime.start(%27) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %28 = kgen.struct.gep %27[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index39, %28 : !kgen.pointer<index>
    %29 = kgen.struct.gep %27[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %30 = pop.string.address %string_11
    %31 = pop.pointer.bitcast %30 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    pop.store %31, %29 : !kgen.pointer<pointer<none>>
    %32 = kgen.struct.gep %27[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index2305843009213693952, %32 : !kgen.pointer<index>
    %33 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
    %34 = pop.pointer.bitcast %33 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
    pop.store %20, %34 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
    %35 = pop.load %33 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
    %36 = pop.array.get %35[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
    %37 = pop.array.create [%36] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
    %38 = kgen.struct.create(%37) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
    %39 = pop.string.address %string_2
    %40 = pop.pointer.bitcast %39 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %41 = kgen.struct.create(%40, %index53) : !kgen.struct<(pointer<none>, index)>
    %42 = kgen.struct.create(%index330, %index27, %41) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %43 = pop.string.address %string_3
    %44 = pop.pointer.bitcast %43 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %45 = kgen.struct.create(%44, %index53) : !kgen.struct<(pointer<none>, index)>
    %46 = kgen.struct.create(%index610, %index18, %45) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %47 = pop.string.address %string_5
    %48 = pop.pointer.bitcast %47 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %49 = pop.string.address %string_6
    %50 = pop.pointer.bitcast %49 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %51 = pop.string.address %string_7
    %52 = pop.pointer.bitcast %51 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %53 = pop.string.address %string_8
    %54 = pop.pointer.bitcast %53 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %55 = kgen.struct.create(%52, %index0) : !kgen.struct<(pointer<none>, index)>
    %56 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
    %57 = pop.pointer.bitcast %56 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
    pop.store %38, %56 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
    hlcf.if %13 {
      hlcf.yield
    } else {
      %73 = pop.stack_allocation 2048 x scalar<ui8> align 1
      %74 = pop.pointer.bitcast %73 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<none>
      %75 = kgen.struct.create(%74, %index0) : !kgen.struct<(pointer<none>, index) memoryOnly>
      %76 = kgen.call @"std::collections::string::string::String::write_to[::Writer](::String,$0&),writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%21, %75) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
      %77 = kgen.call @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%arg0, %76) : (index, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
      %78 = kgen.call @"std::collections::string::string::String::write_to[::Writer](::String,$0&),writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%27, %77) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
      %79 = kgen.call @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%idx618, %78) : (index, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
      %80 = kgen.struct.extract %79[0] : <(pointer<none>, index) memoryOnly>
      %81 = kgen.struct.extract %79[1] : <(pointer<none>, index) memoryOnly>
      %82 = index.add %81, %index1
      %83 = index.cmp sgt(%82, %index2048)
      hlcf.if %83 {
        kgen.call @"std::io::io::_printf[KGENParamList[::AnyType],::StringSlice[::Bool(False), StaticConstantOrigin, *?],*::AnyType,LITImmutOrigin,::Origin[::Bool(False), $3]](*$0,file:::FileDescriptor),types.values`=[],fmt={ #interp.memref<{[(#interp.memory_handle<16, \22HEAP_BUFFER_BYTES exceeded, increase with: `mojo -D HEAP_BUFFER_BYTES=4096`\\0A\\00\22 string>, const_global, [], [])], []}, 0, 0>, 76 }"(%index1) : (index) -> ()
        llvm.intr.trap
        hlcf.loop "_loop_0" {
          hlcf.continue "_loop_0"
        }
        kgen.unreachable
      } else {
        hlcf.yield
      }
      %84 = pop.pointer.bitcast %80 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %85 = pop.offset %84[%81] : !kgen.pointer<scalar<ui8>>
      pop.store %simd_9, %85 : !kgen.pointer<scalar<ui8>>
      %86 = pop.load %56 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
      %87 = kgen.struct.extract %86[0] : <(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
      %88 = pop.array.get %87[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
      %89 = pop.array.create [%88] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
      %90 = kgen.struct.create(%89) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
      %91 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
      pop.store %90, %91 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
      %92 = pop.pointer.bitcast %91 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
      %93 = pop.load %92 : !kgen.pointer<index>
      %94 = index.cmp eq(%93, %index-1)
      %95 = pop.select %94, %index0, %index-1 : index
      %96 = index.cmp eq(%95, %index-1)
      %97 = hlcf.if %96 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
        %98 = pop.load %56 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
        %99 = kgen.struct.extract %98[0] : <(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
        %100 = pop.array.get %99[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
        %101 = pop.array.create [%100] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
        %102 = kgen.struct.create(%101) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
        %103 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
        pop.store %102, %103 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
        %104 = pop.pointer.bitcast %103 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
        %105 = pop.load %104 : !kgen.pointer<index>
        %106 = index.cmp eq(%105, %index-1)
        %107 = pop.select %106, %index0, %index-1 : index
        %108 = index.cmp eq(%107, %index-1)
        %109 = pop.xor %108, %0
        hlcf.if %109 {
          %111 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
          pop.stack_alloc.lifetime.start(%111) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
          %112 = kgen.struct.gep %111[1] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index192, %112 : !kgen.pointer<index>
          %113 = kgen.struct.gep %111[0] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %54, %113 : !kgen.pointer<pointer<none>>
          %114 = kgen.struct.gep %111[2] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index2305843009213693952, %114 : !kgen.pointer<index>
          %115 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
          %116 = pop.pointer.bitcast %115 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
          pop.store %42, %116 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
          %117 = pop.load %115 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
          %118 = pop.array.get %117[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
          %119 = pop.array.create [%118] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
          %120 = kgen.struct.create(%119) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
          %121 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
          %122 = pop.pointer.bitcast %121 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
          pop.store %120, %121 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
          %123 = pop.pointer.bitcast %121 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
          %124 = pop.load %123 : !kgen.pointer<index>
          %125 = index.cmp eq(%124, %index-1)
          %126 = pop.select %125, %index0, %index-1 : index
          %127 = index.cmp eq(%126, %index-1)
          %128 = hlcf.if %127 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
            %146 = pop.load %122 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
            hlcf.yield %146 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
          } else {
            hlcf.yield %46 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
          }
          %129 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
          pop.stack_alloc.lifetime.start(%129) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
          %130 = kgen.struct.gep %129[1] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index1, %130 : !kgen.pointer<index>
          %131 = kgen.struct.gep %129[0] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %48, %131 : !kgen.pointer<pointer<none>>
          %132 = kgen.struct.gep %129[2] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index2305843009213693952, %132 : !kgen.pointer<index>
          %133 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
          pop.stack_alloc.lifetime.start(%133) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
          %134 = kgen.struct.gep %133[1] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index2, %134 : !kgen.pointer<index>
          %135 = kgen.struct.gep %133[0] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %50, %135 : !kgen.pointer<pointer<none>>
          %136 = kgen.struct.gep %133[2] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index2305843009213693952, %136 : !kgen.pointer<index>
          kgen.call @"std::io::io::print[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?],flush:::Bool,file:::FileDescriptor$),Ts.values`=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::reflection::location::SourceLocation\22>>, struct<(index, index, struct<(pointer<none>, index)>)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]]"(%struct_4, %129, %128, %133, %111, %55, %struct, %0, %index1) : (!kgen.struct<(pointer<none>, index)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index)>, i1, index owned) -> ()
          %137 = pop.load %132 : !kgen.pointer<index>
          %138 = index.and %137, %index4611686018427387904
          %139 = index.cmp ne(%138, %index0)
          hlcf.if %139 {
            %146 = pop.load %131 : !kgen.pointer<pointer<none>>
            %147 = pop.pointer.bitcast %146 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
            %148 = pop.offset %147[%idx-8] : !kgen.pointer<scalar<ui8>>
            %149 = pop.pointer.bitcast %148 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
            %150 = kgen.struct.gep %149[0] : <struct<(scalar<index>) memoryOnly>>
            %151 = pop.atomic.rmw sub(%150, %simd_10) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
            %152 = pop.cmp eq(%151, %simd_10) : <1, index>
            %153 = pop.cast_to_builtin %152 : !pop.scalar<bool> to i1
            hlcf.if %153 {
              pop.fence syncscope("") acquire
              pop.aligned_free %148 : <scalar<ui8>>
              hlcf.yield
            } else {
              hlcf.yield
            }
            hlcf.yield
          } else {
            hlcf.yield
          }
          %140 = pop.load %136 : !kgen.pointer<index>
          %141 = index.and %140, %index4611686018427387904
          %142 = index.cmp ne(%141, %index0)
          hlcf.if %142 {
            %146 = pop.load %135 : !kgen.pointer<pointer<none>>
            %147 = pop.pointer.bitcast %146 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
            %148 = pop.offset %147[%idx-8] : !kgen.pointer<scalar<ui8>>
            %149 = pop.pointer.bitcast %148 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
            %150 = kgen.struct.gep %149[0] : <struct<(scalar<index>) memoryOnly>>
            %151 = pop.atomic.rmw sub(%150, %simd_10) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
            %152 = pop.cmp eq(%151, %simd_10) : <1, index>
            %153 = pop.cast_to_builtin %152 : !pop.scalar<bool> to i1
            hlcf.if %153 {
              pop.fence syncscope("") acquire
              pop.aligned_free %148 : <scalar<ui8>>
              hlcf.yield
            } else {
              hlcf.yield
            }
            hlcf.yield
          } else {
            hlcf.yield
          }
          %143 = pop.load %114 : !kgen.pointer<index>
          %144 = index.and %143, %index4611686018427387904
          %145 = index.cmp ne(%144, %index0)
          hlcf.if %145 {
            %146 = pop.load %113 : !kgen.pointer<pointer<none>>
            %147 = pop.pointer.bitcast %146 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
            %148 = pop.offset %147[%idx-8] : !kgen.pointer<scalar<ui8>>
            %149 = pop.pointer.bitcast %148 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
            %150 = kgen.struct.gep %149[0] : <struct<(scalar<index>) memoryOnly>>
            %151 = pop.atomic.rmw sub(%150, %simd_10) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
            %152 = pop.cmp eq(%151, %simd_10) : <1, index>
            %153 = pop.cast_to_builtin %152 : !pop.scalar<bool> to i1
            hlcf.if %153 {
              pop.fence syncscope("") acquire
              pop.aligned_free %148 : <scalar<ui8>>
              hlcf.yield
            } else {
              hlcf.yield
            }
            hlcf.yield
          } else {
            hlcf.yield
          }
          llvm.intr.trap
          hlcf.loop "_loop_0" {
            hlcf.continue "_loop_0"
          }
          kgen.unreachable
        } else {
          hlcf.yield
        }
        %110 = pop.load %57 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
        hlcf.yield %110 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
      } else {
        hlcf.yield %9 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
      }
      kgen.call @"std::builtin::debug_assert::_debug_assert_msg[LITImmutOrigin,::Origin[::Bool(False), $0]](::UnsafePointer[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1, ::AddressSpace(::Int(0))],::Int,::SourceLocation)"(%80, %97) : (!kgen.pointer<none>, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>) -> ()
      hlcf.yield
    }
    %58 = pop.load %26 : !kgen.pointer<index>
    %59 = index.and %58, %index4611686018427387904
    %60 = index.cmp ne(%59, %index0)
    hlcf.if %60 {
      %73 = pop.load %23 : !kgen.pointer<pointer<none>>
      %74 = pop.pointer.bitcast %73 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %75 = pop.offset %74[%idx-8] : !kgen.pointer<scalar<ui8>>
      %76 = pop.pointer.bitcast %75 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
      %77 = kgen.struct.gep %76[0] : <struct<(scalar<index>) memoryOnly>>
      %78 = pop.atomic.rmw sub(%77, %simd_10) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
      %79 = pop.cmp eq(%78, %simd_10) : <1, index>
      %80 = pop.cast_to_builtin %79 : !pop.scalar<bool> to i1
      hlcf.if %80 {
        pop.fence syncscope("") acquire
        pop.aligned_free %75 : <scalar<ui8>>
        hlcf.yield
      } else {
        hlcf.yield
      }
      hlcf.yield
    } else {
      hlcf.yield
    }
    %61 = pop.load %32 : !kgen.pointer<index>
    %62 = index.and %61, %index4611686018427387904
    %63 = index.cmp ne(%62, %index0)
    hlcf.if %63 {
      %73 = pop.load %29 : !kgen.pointer<pointer<none>>
      %74 = pop.pointer.bitcast %73 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %75 = pop.offset %74[%idx-8] : !kgen.pointer<scalar<ui8>>
      %76 = pop.pointer.bitcast %75 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
      %77 = kgen.struct.gep %76[0] : <struct<(scalar<index>) memoryOnly>>
      %78 = pop.atomic.rmw sub(%77, %simd_10) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
      %79 = pop.cmp eq(%78, %simd_10) : <1, index>
      %80 = pop.cast_to_builtin %79 : !pop.scalar<bool> to i1
      hlcf.if %80 {
        pop.fence syncscope("") acquire
        pop.aligned_free %75 : <scalar<ui8>>
        hlcf.yield
      } else {
        hlcf.yield
      }
      hlcf.yield
    } else {
      hlcf.yield
    }
    %64 = kgen.struct.gep %5[0] : <struct<(array<619, scalar<ui128>>) memoryOnly>>
    %65 = pop.array.gep %64[%arg0] : <array<619, scalar<ui128>>>
    %66 = pop.load %65 : !kgen.pointer<scalar<ui128>>
    %67 = pop.shr %66, %simd_0 : !pop.scalar<ui128>
    %68 = pop.cast fast %67 : !pop.scalar<ui128> to !pop.scalar<ui64>
    %69 = index.sub %index63, %arg1
    %70 = pop.cast_from_builtin %69 : index to !pop.scalar<index>
    %71 = pop.cast %70 : !pop.scalar<index> to !pop.scalar<ui64>
    %72 = pop.shr %68, %71 : !pop.scalar<ui64>
    kgen.return %72 : !pop.scalar<ui64>
  }
  kgen.func @"std::builtin::_format_float::_umul192_upper128[::DType](::SIMD[$0, ::Int(1)],::SIMD[::DType(uint128), ::Int(1)]),CarrierDType=ui64"(%arg0: !pop.scalar<ui64>, %arg1: !pop.scalar<ui128>) -> !pop.scalar<ui128> {
    %simd = kgen.param.constant: scalar<ui128> = <64>
    %simd_0 = kgen.param.constant: scalar<ui64> = <32>
    %0 = pop.shr %arg1, %simd : !pop.scalar<ui128>
    %1 = pop.cast fast %0 : !pop.scalar<ui128> to !pop.scalar<ui64>
    %2 = pop.shr %arg0, %simd_0 : !pop.scalar<ui64>
    %3 = pop.cast fast %2 : !pop.scalar<ui64> to !pop.scalar<ui32>
    %4 = pop.cast fast %arg0 : !pop.scalar<ui64> to !pop.scalar<ui32>
    %5 = pop.shr %1, %simd_0 : !pop.scalar<ui64>
    %6 = pop.cast fast %5 : !pop.scalar<ui64> to !pop.scalar<ui32>
    %7 = pop.cast fast %1 : !pop.scalar<ui64> to !pop.scalar<ui32>
    %8 = pop.cast fast %3 : !pop.scalar<ui32> to !pop.scalar<ui64>
    %9 = pop.cast fast %6 : !pop.scalar<ui32> to !pop.scalar<ui64>
    %10 = pop.mul %8, %9 : !pop.scalar<ui64>
    %11 = pop.cast fast %4 : !pop.scalar<ui32> to !pop.scalar<ui64>
    %12 = pop.mul %11, %9 : !pop.scalar<ui64>
    %13 = pop.cast fast %7 : !pop.scalar<ui32> to !pop.scalar<ui64>
    %14 = pop.mul %8, %13 : !pop.scalar<ui64>
    %15 = pop.mul %11, %13 : !pop.scalar<ui64>
    %16 = pop.shr %15, %simd_0 : !pop.scalar<ui64>
    %17 = pop.cast fast %14 : !pop.scalar<ui64> to !pop.scalar<ui32>
    %18 = pop.cast fast %17 : !pop.scalar<ui32> to !pop.scalar<ui64>
    %19 = pop.add %16, %18 : !pop.scalar<ui64>
    %20 = pop.cast fast %12 : !pop.scalar<ui64> to !pop.scalar<ui32>
    %21 = pop.cast fast %20 : !pop.scalar<ui32> to !pop.scalar<ui64>
    %22 = pop.add %19, %21 : !pop.scalar<ui64>
    %23 = pop.shr %22, %simd_0 : !pop.scalar<ui64>
    %24 = pop.add %10, %23 : !pop.scalar<ui64>
    %25 = pop.shr %14, %simd_0 : !pop.scalar<ui64>
    %26 = pop.add %24, %25 : !pop.scalar<ui64>
    %27 = pop.shr %12, %simd_0 : !pop.scalar<ui64>
    %28 = pop.add %26, %27 : !pop.scalar<ui64>
    %29 = pop.shl %22, %simd_0 : !pop.scalar<ui64>
    %30 = pop.cast fast %15 : !pop.scalar<ui64> to !pop.scalar<ui32>
    %31 = pop.cast fast %30 : !pop.scalar<ui32> to !pop.scalar<ui64>
    %32 = pop.add %29, %31 : !pop.scalar<ui64>
    %33 = pop.cast fast %28 : !pop.scalar<ui64> to !pop.scalar<ui128>
    %34 = pop.shl %33, %simd : !pop.scalar<ui128>
    %35 = pop.cast fast %32 : !pop.scalar<ui64> to !pop.scalar<ui128>
    %36 = pop.simd.or %34, %35 : <1, ui128>
    %37 = pop.cast fast %arg1 : !pop.scalar<ui128> to !pop.scalar<ui64>
    %38 = pop.shr %37, %simd_0 : !pop.scalar<ui64>
    %39 = pop.cast fast %38 : !pop.scalar<ui64> to !pop.scalar<ui32>
    %40 = pop.cast fast %37 : !pop.scalar<ui64> to !pop.scalar<ui32>
    %41 = pop.cast fast %39 : !pop.scalar<ui32> to !pop.scalar<ui64>
    %42 = pop.mul %8, %41 : !pop.scalar<ui64>
    %43 = pop.mul %11, %41 : !pop.scalar<ui64>
    %44 = pop.cast fast %40 : !pop.scalar<ui32> to !pop.scalar<ui64>
    %45 = pop.mul %8, %44 : !pop.scalar<ui64>
    %46 = pop.mul %11, %44 : !pop.scalar<ui64>
    %47 = pop.shr %46, %simd_0 : !pop.scalar<ui64>
    %48 = pop.cast fast %45 : !pop.scalar<ui64> to !pop.scalar<ui32>
    %49 = pop.cast fast %48 : !pop.scalar<ui32> to !pop.scalar<ui64>
    %50 = pop.add %47, %49 : !pop.scalar<ui64>
    %51 = pop.cast fast %43 : !pop.scalar<ui64> to !pop.scalar<ui32>
    %52 = pop.cast fast %51 : !pop.scalar<ui32> to !pop.scalar<ui64>
    %53 = pop.add %50, %52 : !pop.scalar<ui64>
    %54 = pop.shr %53, %simd_0 : !pop.scalar<ui64>
    %55 = pop.add %42, %54 : !pop.scalar<ui64>
    %56 = pop.shr %45, %simd_0 : !pop.scalar<ui64>
    %57 = pop.add %55, %56 : !pop.scalar<ui64>
    %58 = pop.shr %43, %simd_0 : !pop.scalar<ui64>
    %59 = pop.add %57, %58 : !pop.scalar<ui64>
    %60 = pop.cast fast %59 : !pop.scalar<ui64> to !pop.scalar<ui128>
    %61 = pop.add %36, %60 : !pop.scalar<ui128>
    kgen.return %61 : !pop.scalar<ui128>
  }
  kgen.func @"std::builtin::_format_float::_compute_round_up_for_shorter_interval_case[::DType,::Int,::Int,::Int](::Int,::Int),CarrierDType=ui64,total_bits=64,sig_bits=52,cache_bits=128"(%arg0: index, %arg1: index) -> !pop.scalar<ui64> {
    %string = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/builtin/_format_float.mojo">
    %simd = kgen.param.constant: scalar<uindex> = <619>
    %idx618 = index.constant 618
    %index54 = kgen.param.constant = <54>
    %index49 = kgen.param.constant = <49>
    %index837 = kgen.param.constant = <837>
    %simd_0 = kgen.param.constant: scalar<ui128> = <64>
    %string_1 = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/collections/check_bounds.mojo">
    %index57 = kgen.param.constant = <57>
    %string_2 = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/builtin/debug_assert.mojo">
    %index27 = kgen.param.constant = <27>
    %index330 = kgen.param.constant = <330>
    %index53 = kgen.param.constant = <53>
    %string_3 = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/collections/optional.mojo">
    %index18 = kgen.param.constant = <18>
    %index610 = kgen.param.constant = <610>
    %idx-8 = index.constant -8
    %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle, const_global, [], [])], []}, 0, 0>, 1 }>
    %struct_4 = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle4, const_global, [], [])], []}, 0, 0>, 6 }>
    %string_5 = kgen.param.constant: string = <" ">
    %index2 = kgen.param.constant = <2>
    %string_6 = kgen.param.constant: string = <": ">
    %string_7 = kgen.param.constant: string = <"">
    %string_8 = kgen.param.constant: string = <"`Optional.value()` called on empty `Optional`. Consider using `if optional:` to check whether the `Optional` is empty before calling `.value()`, or use `.or_else()` to provide a default value.">
    %index192 = kgen.param.constant = <192>
    %simd_9 = kgen.param.constant: scalar<ui8> = <0>
    %index2048 = kgen.param.constant = <2048>
    %index-1 = kgen.param.constant = <-1>
    %index0 = kgen.param.constant = <0>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %simd_10 = kgen.param.constant: scalar<index> = <1>
    %string_11 = kgen.param.constant: string = <" is out of bounds, valid range is 0 to ">
    %index39 = kgen.param.constant = <39>
    %string_12 = kgen.param.constant: string = <"index ">
    %index6 = kgen.param.constant = <6>
    %index2305843009213693952 = kgen.param.constant = <2305843009213693952>
    %index1 = kgen.param.constant = <1>
    %none = kgen.param.constant: none = <#kgen.none>
    %array = kgen.param.constant: array<1, struct<(index, index, struct<(pointer<none>, index)>)>> = <[{ -1, 0, { 0, 0 } }]>
    %0 = kgen.param.constant: i1 = <1>
    %index10 = kgen.param.constant = <10>
    %simd_13 = kgen.param.constant: scalar<ui64> = <1>
    %1 = pop.string.address %string
    %2 = pop.pointer.bitcast %1 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %3 = kgen.struct.create(%2, %index54) : !kgen.struct<(pointer<none>, index)>
    %4 = kgen.struct.create(%index837, %index49, %3) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %5 = pop.global_constant: struct<(array<619, scalar<ui128>>) memoryOnly> = <{ [339574632262346462319337857089816694651, 212234145163966538949586160681135434157, 265292681454958173686982700851419292696, 331615851818697717108728376064274115870, 207259907386686073192955235040171322419, 259074884233357591491194043800214153024, 323843605291696989363992554750267691280, 202402253307310618352495346718917307050, 253002816634138272940619183398646633812, 316253520792672841175773979248308292265, 197658450495420525734858737030192682666, 247073063119275657168573421287740853332, 308841328899094571460716776609676066665, 193025830561934107162947985381047541666, 241282288202417633953684981726309427083, 301602860253022042442106227157886783853, 188501787658138776526316391973679239908, 235627234572673470657895489967099049885, 294534043215841838322369362458873812356, 184083777009901148951480851536796132723, 230104721262376436189351064420995165904, 287630901577970545236688830526243957379, 179769313486231590772930519078902473362, 224711641857789488466163148848628091703, 280889552322236860582703936060785114628, 175555970201398037864189960037990696643, 219444962751747547330237450047488370803, 274306203439684434162796812559360463504, 171441377149802771351748007849600289690, 214301721437253464189685009812000362113, 267877151796566830237106262265000452641, 334846439745708537796382827831250565801, 209279024841067836122739267394531603626, 261598781051334795153424084243164504532, 326998476314168493941780105303955630665, 204374047696355308713612565814972269166, 255467559620444135892015707268715336457, 319334449525555169865019634085894170571, 199584030953471981165637271303683856607, 249480038691839976457046589129604820759, 311850048364799970571308236412006025949, 194906280227999981607067647757503766218, 243632850284999977008834559696879707772, 304541062856249971261043199621099634715, 190338164285156232038151999763187271697, 237922705356445290047689999703984089622, 297403381695556612559612499629980112027, 185877113559722882849757812268737570017, 232346391949653603562197265335921962521, 290432989937067004452746581669902453151, 181520618710666877782966613543689033220, 226900773388333597228708266929611291524, 283625966735416996535885333662014114405, 177266229209635622834928333538758821504, 221582786512044528543660416923448526879, 276978483140055660679575521154310658599, 173111551962534787924734700721444161625, 216389439953168484905918375901805202031, 270486799941460606132397969877256502538, 338108499926825757665497462346570628173, 211317812454266098540935913966606642608, 264147265567832623176169892458258303260, 330184081959790778970212365572822879075, 206365051224869236856382728483014299422, 257956314031086546070478410603767874277, 322445392538858182588098013254709842846, 201528370336786364117561258284193651779, 251910462920982955146951572855242064724, 314888078651228693933689466069052580905, 196805049157017933708555916293157863066, 246006311446272417135694895366447328832, 307507889307840521419618619208059161040, 192192430817400325887261637005036975650, 240240538521750407359077046256296219562, 300300673152188009198846307820370274453, 187687920720117505749278942387731421533, 234609900900146882186598677984664276916, 293262376125183602733248347480830346145, 183288985078239751708280217175518966341, 229111231347799689635350271469398707926, 286389039184749612044187839336748384908, 178993149490468507527617399585467740568, 223741436863085634409521749481834675709, 279676796078857043011902186852293344636, 174797997549285651882438866782683340398, 218497496936607064853048583478354175497, 273121871170758831066310729347942719372, 170701169481724269416444205842464199607, 213376461852155336770555257303080249509, 266720577315194170963194071628850311886, 333400721643992713703992589536062889858, 208375451027495446064995368460039306161, 260469313784369307581244210575049132701, 325586642230461634476555263218811415877, 203491651394038521547847039511757134923, 254364564242548151934808799389696418654, 317955705303185189918510999237120523317, 198722315814490743699069374523200327073, 248402894768113429623836718154000408842, 310503618460141787029795897692500511052, 194064761537588616893622436057812819408, 242580951921985771117028045072266024259, 303226189902482213896285056340332530324, 189516368689051383685178160212707831453, 236895460861314229606472700265884789316, 296119326076642787008090875332355986645, 185074578797901741880056797082722491653, 231343223497377177350070996353403114566, 289179029371721471687588745441753893208, 180736893357325919804742965901096183255, 225921116696657399755928707376370229069, 282401395870821749694910884220462786336, 176500872419263593559319302637789241460, 220626090524079491949149128297236551825, 275782613155099364936436410371545689781, 172364133221937103085272756482216056113, 215455166527421378856590945602770070141, 269318958159276723570738682003462587677, 336648697699095904463423352504328234596, 210405436061934940289639595315205146623, 263006795077418675362049494144006433278, 328758493846773344202561867680008041597, 205474058654233340126601167300005025999, 256842573317791675158251459125006282498, 321053216647239593947814323906257853122, 200658260404524746217383952441411158202, 250822825505655932771729940551763947752, 313528531882069915964662425689704934690, 195955332426293697477914016056065584181, 244944165532867121847392520070081980227, 306180206916083902309240650087602475283, 191362629322552438943275406304751547052, 239203286653190548679094257880939433815, 299004108316488185848867822351174292269, 186877567697805116155542388969483932668, 233596959622256395194427986211854915835, 291996199527820493993034982764818644794, 182497624704887808745646864228011652996, 228122030881109760932058580285014566245, 285152538601387201165073225356268207806, 178220336625867000728170765847667629879, 222775420782333750910213457309584537349, 278469275977917188637766821636980671686, 174043297486198242898604263523112919804, 217554121857747803623255329403891149755, 271942652322184754529069161754863937193, 339928315402730943161336452193579921491, 212455197126706839475835282620987450932, 265568996408383549344794103276234313665, 331961245510479436680992629095292892081, 207475778444049647925620393184558057551, 259344723055062059907025491480697571939, 324180903818827574883781864350871964923, 202613064886767234302363665219294978077, 253266331108459042877954581524118722596, 316582913885573803597443226905148403245, 197864321178483627248402016815717752029, 247330401473104534060502521019647190036, 309163001841380667575628151274558987544, 193226876150862917234767594546599367215, 241533595188578646543459493183249209019, 301916993985723308179324366479061511274, 188698121241077067612077729049413444546, 235872651551346334515097161311766805683, 294840814439182918143871451639708507103, 184275509024489323839919657274817816940, 230344386280611654799899571593522271175, 287930482850764568499874464491902838968, 179956551781727855312421540307439274355, 224945689727159819140526925384299092944, 281182112158949773925658656730373866180, 175738820099343608703536660456483666363, 219673525124179510879420825570604582953, 274591906405224388599276031963255728691, 171619941503265242874547519977034830432, 214524926879081553593184399971293538040, 268156158598851941991480499964116922550, 335195198248564927489350624955146153187, 209496998905353079680844140596966345742, 261871248631691349601055175746207932178, 327339060789614187001318969682759915222, 204586912993508866875824356051724947014, 255733641241886083594780445064656183767, 319667051552357604493475556330820229709, 199791907220223502808422222706762643568, 249739884025279378510527778383453304460, 312174855031599223138159722979316630575, 195109284394749514461349826862072894110, 243886605493436893076687283577591117637, 304858256866796116345859104471988897046, 190536410541747572716161940294993060654, 238170513177184465895202425368741325818, 297713141471480582369003031710926657272, 186070713419675363980626894819329160795, 232588391774594204975783618524161450994, 290735489718242756219729523155201813742, 181709681073901722637330951972001133589, 227137101342377153296663689965001416986, 283921376677971441620829612456251771232, 177450860423732151013018507785157357020, 221813575529665188766273134731446696275, 277266969412081485957841418414308370344, 173291855882550928723650886508942731465, 216614819853188660904563608136178414331, 270768524816485826130704510170223017914, 338460656020607282663380637712778772393, 211537910012879551664612898570486732746, 264422387516099439580766123213108415932, 330527984395124299475957654016385519915, 206579990246952687172473533760240949947, 258224987808690858965591917200301187433, 322781234760863573706989896500376484292, 201738271725539733566868685312735302683, 252172839656924666958585856640919128353, 315216049571155833698232320801148910441, 197010030981972396061395200500718069026, 246262538727465495076744000625897586282, 307828173409331868845930000782371982853, 192392608380832418028706250488982489283, 240490760476040522535882813111228111604, 300613450595050653169853516389035139505, 187883406621906658231158447743146962191, 234854258277383322788948059678933702738, 293567822846729153486185074598667128422, 183479889279205720928865671624166955264, 229349861599007151161082089530208694080, 286687326998758938951352611912760867600, 179179579374224336844595382445475542250, 223974474217780421055744228056844427813, 279968092772225526319680285071055534766, 174980057982640953949800178169409709229, 218725072478301192437250222711762136536, 273406340597876490546562778389702670670, 170878962873672806591601736493564169169, 213598703592091008239502170616955211461, 266998379490113760299377713271194014326, 333747974362642200374222141588992517907, 208592483976651375233888838493120323692, 260740604970814219042361048116400404615, 325925756213517773802951310145500505769, 203703597633448608626844568840937816106, 254629497041810760783555711051172270132, 318286871302263450979444638813965337665, 198929294563914656862152899258728336041, 248661618204893321077691124073410420051, 310827022756116651347113905091763025063, 194266889222572907091946190682351890665, 242833611528216133864932738352939863331, 303542014410270167331165922941174829163, 189713759006418854581978701838234268227, 237142198758023568227473377297792835284, 296427748447529460284341721622241044105, 185267342779705912677713576013900652566, 231584178474632390847141970017375815707, 289480223093290488558927462521719769634, 180925139433306555349329664076074856021, 226156424291633194186662080095093570026, 282695530364541492733327600118866962533, 176684706477838432958329750074291851583, 220855883097298041197912187592864814479, 276069853871622551497390234491081018099, 172543658669764094685868896556925636312, 215679573337205118357336120696157045390, 269599466671506397946670150870196306737, 336999333339382997433337688587745383421, 210624583337114373395836055367340864638, 263280729171392966744795069209176080798, 329100911464241208430993836511470100997, 205688069665150755269371147819668813123, 257110087081438444086713934774586016404, 321387608851798055108392418468232520505, 200867255532373784442745261542645325316, 251084069415467230553431576928306656645, 313855086769334038191789471160383320806, 196159429230833773869868419475239575504, 245199286538542217337335524344049469379, 306499108173177771671669405430061836724, 191561942608236107294793378393788647953, 239452428260295134118491722992235809941, 299315535325368917648114653740294762426, 187072209578355573530071658587684226516, 233840261972944466912589573234605283145, 292300327466180583640736966543256603932, 182687704666362864775460604089535377457, 228359630832953580969325755111919221822, 285449538541191976211657193889899027277, 178405961588244985132285746181186892048, 223007451985306231415357182726483615060, 278759314981632789269196478408104518825, 174224571863520493293247799005065324266, 217780714829400616616559748756331655332, 272225893536750770770699685945414569165, 170141183460469231731687303715884105728, 212676479325586539664609129644855132160, 265845599156983174580761412056068915200, 332306998946228968225951765070086144000, 207691874341393105141219853168803840000, 259614842926741381426524816461004800000, 324518553658426726783156020576256000000, 202824096036516704239472512860160000000, 253530120045645880299340641075200000000, 316912650057057350374175801344000000000, 198070406285660843983859875840000000000, 247588007857076054979824844800000000000, 309485009821345068724781056000000000000, 193428131138340667952988160000000000000, 241785163922925834941235200000000000000, 302231454903657293676544000000000000000, 188894659314785808547840000000000000000, 236118324143482260684800000000000000000, 295147905179352825856000000000000000000, 184467440737095516160000000000000000000, 230584300921369395200000000000000000000, 288230376151711744000000000000000000000, 180143985094819840000000000000000000000, 225179981368524800000000000000000000000, 281474976710656000000000000000000000000, 175921860444160000000000000000000000000, 219902325555200000000000000000000000000, 274877906944000000000000000000000000000, 171798691840000000000000000000000000000, 214748364800000000000000000000000000000, 268435456000000000000000000000000000000, 335544320000000000000000000000000000000, 209715200000000000000000000000000000000, 262144000000000000000000000000000000000, 327680000000000000000000000000000000000, 204800000000000000000000000000000000000, 256000000000000000000000000000000000000, 320000000000000000000000000000000000000, 200000000000000000000000000000000000000, 250000000000000000000000000000000000000, 312500000000000000000000000000000000000, 195312500000000000000000000000000000000, 244140625000000000000000000000000000000, 305175781250000000000000000000000000000, 190734863281250000000000000000000000000, 238418579101562500000000000000000000000, 298023223876953125000000000000000000000, 186264514923095703125000000000000000000, 232830643653869628906250000000000000000, 291038304567337036132812500000000000000, 181898940354585647583007812500000000000, 227373675443232059478759765625000000000, 284217094304040074348449707031250000000, 177635683940025046467781066894531250000, 222044604925031308084726333618164062500, 277555756156289135105907917022705078125, 173472347597680709441192448139190673829, 216840434497100886801490560173988342286, 271050543121376108501863200217485427857, 338813178901720135627329000271856784821, 211758236813575084767080625169910490513, 264697796016968855958850781462388113142, 330872245021211069948563476827985141427, 206795153138256918717852173017490713392, 258493941422821148397315216271863391740, 323117426778526435496644020339829239675, 201948391736579022185402512712393274797, 252435489670723777731753140890491593496, 315544362088404722164691426113114491870, 197215226305252951352932141320696557419, 246519032881566189191165176650870696773, 308148791101957736488956470813588370967, 192592994438723585305597794258492731854, 240741243048404481631997242823115914818, 300926553810505602039996553528894893522, 188079096131566001274997845955559308451, 235098870164457501593747307444449135564, 293873587705571876992184134305561419455, 183670992315982423120115083940975887160, 229588740394978028900143854926219858949, 286985925493722536125179818657774823687, 179366203433576585078237386661109264804, 224207754291970731347796733326386581005, 280259692864963414184745916657983226257, 175162308040602133865466197911239516411, 218952885050752667331832747389049395513, 273691106313440834164790934236311744391, 171056941445900521352994333897694840245, 213821176807375651691242917372118550306, 267276471009219564614053646715148187882, 334095588761524455767567058393935234852, 208809742975952784854729411496209521783, 261012178719940981068411764370261902229, 326265223399926226335514705462827377786, 203915764624953891459696690914267111116, 254894705781192364324620863642833888895, 318618382226490455405776079553542361119, 199136488891556534628610049720963975699, 248920611114445668285762562151204969624, 311150763893057085357203202689006212030, 194469227433160678348252001680628882519, 243086534291450847935315002100786103149, 303858167864313559919143752625982628936, 189911354915195974949464845391239143085, 237389193643994968686831056739048928856, 296736492054993710858538820923811161070, 185460307534371069286586763077381975669, 231825384417963836608233453846727469586, 289781730522454795760291817308409336982, 181113581576534247350182385817755835614, 226391976970667809187727982272194794518, 282989971213334761484659977840243493147, 176868732008334225927912486150152183217, 221085915010417782409890607687690229021, 276357393763022228012363259609612786276, 172723371101888892507727037256007991423, 215904213877361115634658796570009989278, 269880267346701394543323495712512486598, 337350334183376743179154369640640608247, 210843958864610464486971481025400380155, 263554948580763080608714351281750475193, 329443685725953850760892939102188093991, 205902303578721156725558086938867558745, 257377879473401445906947608673584448431, 321722349341751807383684510841980560539, 201076468338594879614802819276237850337, 251345585423243599518503524095297312921, 314181981779054499398129405119121641151, 196363738611909062123830878199451025720, 245454673264886327654788597749313782149, 306818341581107909568485747186642227686, 191761463488192443480303591991651392304, 239701829360240554350379489989564240380, 299627286700300692937974362486955300475, 187267054187687933086233976554347062797, 234083817734609916357792470692933828496, 292604772168262395447240588366167285620, 182877982605163997154525367728854553513, 228597478256454996443156709661068191891, 285746847820568745553945887076335239863, 178591779887855465971216179422709524915, 223239724859819332464020224278386906143, 279049656074774165580025280347983632679, 174406035046733853487515800217489770425, 218007543808417316859394750271862213031, 272509429760521646074243437839827766288, 170318393600326028796402148649892353930, 212897992000407535995502685812365442413, 266122490000509419994378357265456803016, 332653112500636774992972946581821003770, 207908195312897984370608091613638127356, 259885244141122480463260114517047659195, 324856555176403100579075143146309573994, 203035346985251937861921964466443483746, 253794183731564922327402455583054354683, 317242729664456152909253069478817943353, 198276706040285095568283168424261214596, 247845882550356369460353960530326518245, 309807353187945461825442450662908147806, 193629595742465913640901531664317592379, 242036994678082392051126914580396990474, 302546243347602990063908643225496238092, 189091402092251868789942902015935148808, 236364252615314835987428627519918936009, 295455315769143544984285784399898670012, 184659572355714715615178615249936668757, 230824465444643394518973269062420835947, 288530581805804243148716586328026044933, 180331613628627651967947866455016278083, 225414517035784564959934833068770347604, 281768146294730706199918541335962934505, 176105091434206691374949088334976834066, 220131364292758364218686360418721042582, 275164205365947955273357950523401303228, 171977628353717472045848719077125814518, 214972035442146840057310898846407268147, 268715044302683550071638623558009085183, 335893805378354437589548279447511356479, 209933628361471523493467674654694597800, 262417035451839404366834593318368247249, 328021294314799255458543241647960309062, 205013308946749534661589526029975193164, 256266636183436918326986907537468991454, 320333295229296147908733634421836239318, 200208309518310092442958521513647649574, 250260386897887615553698151892059561967, 312825483622359519442122689865074452459, 195515927263974699651326681165671532787, 244394909079968374564158351457089415984, 305493636349960468205197939321361769979, 190933522718725292628248712075851106237, 238666903398406615785310890094813882797, 298333629248008269731638612618517353496, 186458518280005168582274132886573345935, 233073147850006460727842666108216682419, 291341434812508075909803332635270853023, 182088396757817547443627082897044283140, 227610495947271934304533853621305353924, 284513119934089917880667317026631692405, 177820699958806198675417073141644807754, 222275874948507748344271341427056009692, 277844843685634685430339176783820012115, 173653027303521678393961985489887507572, 217066284129402097992452481862359384465, 271332855161752622490565602327949230581, 339166068952190778113207002909936538226, 211978793095119236320754376818710336391, 264973491368899045400942971023387920489, 331216864211123806751178713779234900611, 207010540131952379219486696112021812882, 258763175164940474024358370140027266102, 323453968956175592530447962675034082628, 202158730597609745331529976671896301643, 252698413247012181664412470839870377053, 315873016558765227080515588549837971316, 197420635349228266925322242843648732073, 246775794186535333656652803554560915091, 308469742733169167070816004443201143864, 192793589208230729419260002777000714915, 240991986510288411774075003471250893644, 301239983137860514717593754339063617054, 188274989461162821698496096461914760659, 235343736826453527123120120577393450824, 294179671033066908903900150721741813530, 183862294395666818064937594201088633456, 229827867994583522581171992751360791820, 287284834993229403226464990939200989775, 179553021870768377016540619337000618610, 224441277338460471270675774171250773262, 280551596673075589088344717714063466577, 175344747920672243180215448571289666611, 219180934900840303975269310714112083264, 273976168626050379969086638392640104079, 171235105391281487480679148995400065050, 214043881739101859350848936244250081312, 267554852173877324188561170305312601640, 334443565217346655235701462881640752050, 209027228260841659522313414301025470031, 261284035326052074402891767876281837539, 326605044157565093003614709845352296924, 204128152598478183127259193653345185578, 255160190748097728909073992066681481972, 318950238435122161136342490083351852465, 199343899021951350710214056302094907791, 249179873777439188387767570377618634738, 311474842221798985484709462972023293422, 194671776388624365927943414357514558389, 243339720485780457409929267946893197986, 304174650607225571762411584933616497483, 190109156629515982351507240583510310927, 237636445786894977939384050729387888659, 297045557233618722424230063411734860823, 185653473271011701515143789632334288015, 232066841588764626893929737040417860018, 290083551985955783617412171300522325023, 181302219991222364760882607062826453139, 226627774989027955951103258828533066424, 283284718736284944938879073535666333030, 177052949210178090586799420959791458144, 221316186512722613233499276199739322680, 276645233140903266541874095249674153350, 172903270713064541588671309531046345844, 216129088391330676985839136913807932304, 270161360489163346232298921142259915380, 337701700611454182790373651427824894225, 211063562882158864243983532142390558891, 263829453602698580304979415177988198614, 329786817003373225381224268972485248267, 206116760627108265863265168107803280167, 257645950783885332329081460134754100209, 322057438479856665411351825168442625261, 201285899049910415882094890730276640788, 251607373812388019852618613412845800985, 314509217265485024815773266766057251231, 196568260790928140509858291728785782020, 245710325988660175637322864660982227524, 307137907485825219546653580826227784405, 191961192178640762216658488016392365254, 239951490223300952770823110020490456567, 299939362779126190963528887525613070708, 187462101736953869352205554703508169193, 234327627171192336690256943379385211491, 292909533963990420862821179224231514364, 183068458727494013039263237015144696478, 228835573409367516299079046268930870597, 286044466761709395373848807836163588246, 178777791726068372108655504897602242654, 223472239657585465135819381122002803317, 279340299571981831419774226402503504146, 174587687232488644637358891501564690092, 218234609040610805796698614376955862614, 272793261300763507245873267971194828268, 170495788312977192028670792481996767668, 213119735391221490035838490602495959584, 266399669239026862544798113253119949480, 332999586548783578180997641566399936850, 208124741592989736363123525978999960532, 260155926991237170453904407473749950665, 325194908739046463067380509342187438331, 203246817961904039417112818338867148957, 254058522452380049271391022923583936196, 317573153065475061589238778654479920245, 198483220665921913493274236659049950153, 248104025832402391866592795823812437691, 310130032290502989833240994779765547114, 193831270181564368645775621737353466946, 242289087726955460807219527171691833683, 302861359658694326009024408964614792103, 189288349786683953755640255602884245065, 236610437233354942194550319503605306331, 295763046541693677743187899379506632914, 184851904088558548589492437112191645571, 231064880110698185736865546390239556964, 288831100138372732171081932987799446205, 180519437586482957606926208117374653878, 225649296983103697008657760146718317347, 282061621228879621260822200183397896684, 176288513268049763288013875114623685428, 220360641585062204110017343893279606785, 275450801981327755137521679866599508481, 172156751238329846960951049916624692801, 215195939047912308701188812395780866001, 268994923809890385876486015494726082501, 336243654762362982345607519368407603126, 210152284226476863966004699605254751954, 262690355283096079957505874506568439942, 328362944103870099946882343133210549928] }>
    %6 = pop.string.address %string_1
    %7 = pop.pointer.bitcast %6 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %8 = kgen.struct.create(%7, %index57) : !kgen.struct<(pointer<none>, index)>
    %9 = kgen.struct.create(%index57, %index6, %8) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %10 = pop.cast_from_builtin %arg0 : index to !pop.scalar<index>
    %11 = pop.cast %10 : !pop.scalar<index> to !pop.scalar<uindex>
    %12 = pop.cmp lt(%11, %simd) : <1, uindex>
    %13 = pop.cast_to_builtin %12 : !pop.scalar<bool> to i1
    %14 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
    pop.stack_alloc.lifetime.start(%14) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
    pop.store %array, %14 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
    %15 = pop.pointer.bitcast %14 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<index>
    %16 = pop.load %15 : !kgen.pointer<index>
    %17 = index.cmp eq(%16, %index-1)
    %18 = pop.select %17, %index0, %index-1 : index
    pop.stack_alloc.lifetime.end(%14) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
    %19 = index.cmp eq(%18, %index-1)
    %20 = hlcf.if %19 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
      %75 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
      pop.stack_alloc.lifetime.start(%75) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
      pop.store %array, %75 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
      %76 = pop.pointer.bitcast %75 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
      %77 = pop.load %76 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
      pop.stack_alloc.lifetime.end(%75) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
      hlcf.yield %77 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    } else {
      hlcf.yield %4 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    }
    %21 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
    pop.stack_alloc.lifetime.start(%21) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %22 = kgen.struct.gep %21[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index6, %22 : !kgen.pointer<index>
    %23 = kgen.struct.gep %21[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %24 = pop.string.address %string_12
    %25 = pop.pointer.bitcast %24 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    pop.store %25, %23 : !kgen.pointer<pointer<none>>
    %26 = kgen.struct.gep %21[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index2305843009213693952, %26 : !kgen.pointer<index>
    %27 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
    pop.stack_alloc.lifetime.start(%27) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %28 = kgen.struct.gep %27[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index39, %28 : !kgen.pointer<index>
    %29 = kgen.struct.gep %27[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %30 = pop.string.address %string_11
    %31 = pop.pointer.bitcast %30 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    pop.store %31, %29 : !kgen.pointer<pointer<none>>
    %32 = kgen.struct.gep %27[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index2305843009213693952, %32 : !kgen.pointer<index>
    %33 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
    %34 = pop.pointer.bitcast %33 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
    pop.store %20, %34 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
    %35 = pop.load %33 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
    %36 = pop.array.get %35[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
    %37 = pop.array.create [%36] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
    %38 = kgen.struct.create(%37) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
    %39 = pop.string.address %string_2
    %40 = pop.pointer.bitcast %39 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %41 = kgen.struct.create(%40, %index53) : !kgen.struct<(pointer<none>, index)>
    %42 = kgen.struct.create(%index330, %index27, %41) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %43 = pop.string.address %string_3
    %44 = pop.pointer.bitcast %43 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %45 = kgen.struct.create(%44, %index53) : !kgen.struct<(pointer<none>, index)>
    %46 = kgen.struct.create(%index610, %index18, %45) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %47 = pop.string.address %string_5
    %48 = pop.pointer.bitcast %47 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %49 = pop.string.address %string_6
    %50 = pop.pointer.bitcast %49 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %51 = pop.string.address %string_7
    %52 = pop.pointer.bitcast %51 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %53 = pop.string.address %string_8
    %54 = pop.pointer.bitcast %53 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %55 = kgen.struct.create(%52, %index0) : !kgen.struct<(pointer<none>, index)>
    %56 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
    %57 = pop.pointer.bitcast %56 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
    pop.store %38, %56 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
    hlcf.if %13 {
      hlcf.yield
    } else {
      %75 = pop.stack_allocation 2048 x scalar<ui8> align 1
      %76 = pop.pointer.bitcast %75 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<none>
      %77 = kgen.struct.create(%76, %index0) : !kgen.struct<(pointer<none>, index) memoryOnly>
      %78 = kgen.call @"std::collections::string::string::String::write_to[::Writer](::String,$0&),writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%21, %77) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
      %79 = kgen.call @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%arg0, %78) : (index, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
      %80 = kgen.call @"std::collections::string::string::String::write_to[::Writer](::String,$0&),writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%27, %79) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
      %81 = kgen.call @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%idx618, %80) : (index, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
      %82 = kgen.struct.extract %81[0] : <(pointer<none>, index) memoryOnly>
      %83 = kgen.struct.extract %81[1] : <(pointer<none>, index) memoryOnly>
      %84 = index.add %83, %index1
      %85 = index.cmp sgt(%84, %index2048)
      hlcf.if %85 {
        kgen.call @"std::io::io::_printf[KGENParamList[::AnyType],::StringSlice[::Bool(False), StaticConstantOrigin, *?],*::AnyType,LITImmutOrigin,::Origin[::Bool(False), $3]](*$0,file:::FileDescriptor),types.values`=[],fmt={ #interp.memref<{[(#interp.memory_handle<16, \22HEAP_BUFFER_BYTES exceeded, increase with: `mojo -D HEAP_BUFFER_BYTES=4096`\\0A\\00\22 string>, const_global, [], [])], []}, 0, 0>, 76 }"(%index1) : (index) -> ()
        llvm.intr.trap
        hlcf.loop "_loop_0" {
          hlcf.continue "_loop_0"
        }
        kgen.unreachable
      } else {
        hlcf.yield
      }
      %86 = pop.pointer.bitcast %82 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %87 = pop.offset %86[%83] : !kgen.pointer<scalar<ui8>>
      pop.store %simd_9, %87 : !kgen.pointer<scalar<ui8>>
      %88 = pop.load %56 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
      %89 = kgen.struct.extract %88[0] : <(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
      %90 = pop.array.get %89[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
      %91 = pop.array.create [%90] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
      %92 = kgen.struct.create(%91) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
      %93 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
      pop.store %92, %93 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
      %94 = pop.pointer.bitcast %93 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
      %95 = pop.load %94 : !kgen.pointer<index>
      %96 = index.cmp eq(%95, %index-1)
      %97 = pop.select %96, %index0, %index-1 : index
      %98 = index.cmp eq(%97, %index-1)
      %99 = hlcf.if %98 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
        %100 = pop.load %56 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
        %101 = kgen.struct.extract %100[0] : <(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
        %102 = pop.array.get %101[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
        %103 = pop.array.create [%102] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
        %104 = kgen.struct.create(%103) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
        %105 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
        pop.store %104, %105 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
        %106 = pop.pointer.bitcast %105 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
        %107 = pop.load %106 : !kgen.pointer<index>
        %108 = index.cmp eq(%107, %index-1)
        %109 = pop.select %108, %index0, %index-1 : index
        %110 = index.cmp eq(%109, %index-1)
        %111 = pop.xor %110, %0
        hlcf.if %111 {
          %113 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
          pop.stack_alloc.lifetime.start(%113) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
          %114 = kgen.struct.gep %113[1] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index192, %114 : !kgen.pointer<index>
          %115 = kgen.struct.gep %113[0] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %54, %115 : !kgen.pointer<pointer<none>>
          %116 = kgen.struct.gep %113[2] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index2305843009213693952, %116 : !kgen.pointer<index>
          %117 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
          %118 = pop.pointer.bitcast %117 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
          pop.store %42, %118 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
          %119 = pop.load %117 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
          %120 = pop.array.get %119[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
          %121 = pop.array.create [%120] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
          %122 = kgen.struct.create(%121) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
          %123 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
          %124 = pop.pointer.bitcast %123 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
          pop.store %122, %123 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
          %125 = pop.pointer.bitcast %123 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
          %126 = pop.load %125 : !kgen.pointer<index>
          %127 = index.cmp eq(%126, %index-1)
          %128 = pop.select %127, %index0, %index-1 : index
          %129 = index.cmp eq(%128, %index-1)
          %130 = hlcf.if %129 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
            %148 = pop.load %124 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
            hlcf.yield %148 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
          } else {
            hlcf.yield %46 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
          }
          %131 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
          pop.stack_alloc.lifetime.start(%131) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
          %132 = kgen.struct.gep %131[1] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index1, %132 : !kgen.pointer<index>
          %133 = kgen.struct.gep %131[0] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %48, %133 : !kgen.pointer<pointer<none>>
          %134 = kgen.struct.gep %131[2] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index2305843009213693952, %134 : !kgen.pointer<index>
          %135 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
          pop.stack_alloc.lifetime.start(%135) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
          %136 = kgen.struct.gep %135[1] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index2, %136 : !kgen.pointer<index>
          %137 = kgen.struct.gep %135[0] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %50, %137 : !kgen.pointer<pointer<none>>
          %138 = kgen.struct.gep %135[2] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index2305843009213693952, %138 : !kgen.pointer<index>
          kgen.call @"std::io::io::print[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?],flush:::Bool,file:::FileDescriptor$),Ts.values`=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::reflection::location::SourceLocation\22>>, struct<(index, index, struct<(pointer<none>, index)>)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]]"(%struct_4, %131, %130, %135, %113, %55, %struct, %0, %index1) : (!kgen.struct<(pointer<none>, index)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index)>, i1, index owned) -> ()
          %139 = pop.load %134 : !kgen.pointer<index>
          %140 = index.and %139, %index4611686018427387904
          %141 = index.cmp ne(%140, %index0)
          hlcf.if %141 {
            %148 = pop.load %133 : !kgen.pointer<pointer<none>>
            %149 = pop.pointer.bitcast %148 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
            %150 = pop.offset %149[%idx-8] : !kgen.pointer<scalar<ui8>>
            %151 = pop.pointer.bitcast %150 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
            %152 = kgen.struct.gep %151[0] : <struct<(scalar<index>) memoryOnly>>
            %153 = pop.atomic.rmw sub(%152, %simd_10) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
            %154 = pop.cmp eq(%153, %simd_10) : <1, index>
            %155 = pop.cast_to_builtin %154 : !pop.scalar<bool> to i1
            hlcf.if %155 {
              pop.fence syncscope("") acquire
              pop.aligned_free %150 : <scalar<ui8>>
              hlcf.yield
            } else {
              hlcf.yield
            }
            hlcf.yield
          } else {
            hlcf.yield
          }
          %142 = pop.load %138 : !kgen.pointer<index>
          %143 = index.and %142, %index4611686018427387904
          %144 = index.cmp ne(%143, %index0)
          hlcf.if %144 {
            %148 = pop.load %137 : !kgen.pointer<pointer<none>>
            %149 = pop.pointer.bitcast %148 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
            %150 = pop.offset %149[%idx-8] : !kgen.pointer<scalar<ui8>>
            %151 = pop.pointer.bitcast %150 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
            %152 = kgen.struct.gep %151[0] : <struct<(scalar<index>) memoryOnly>>
            %153 = pop.atomic.rmw sub(%152, %simd_10) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
            %154 = pop.cmp eq(%153, %simd_10) : <1, index>
            %155 = pop.cast_to_builtin %154 : !pop.scalar<bool> to i1
            hlcf.if %155 {
              pop.fence syncscope("") acquire
              pop.aligned_free %150 : <scalar<ui8>>
              hlcf.yield
            } else {
              hlcf.yield
            }
            hlcf.yield
          } else {
            hlcf.yield
          }
          %145 = pop.load %116 : !kgen.pointer<index>
          %146 = index.and %145, %index4611686018427387904
          %147 = index.cmp ne(%146, %index0)
          hlcf.if %147 {
            %148 = pop.load %115 : !kgen.pointer<pointer<none>>
            %149 = pop.pointer.bitcast %148 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
            %150 = pop.offset %149[%idx-8] : !kgen.pointer<scalar<ui8>>
            %151 = pop.pointer.bitcast %150 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
            %152 = kgen.struct.gep %151[0] : <struct<(scalar<index>) memoryOnly>>
            %153 = pop.atomic.rmw sub(%152, %simd_10) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
            %154 = pop.cmp eq(%153, %simd_10) : <1, index>
            %155 = pop.cast_to_builtin %154 : !pop.scalar<bool> to i1
            hlcf.if %155 {
              pop.fence syncscope("") acquire
              pop.aligned_free %150 : <scalar<ui8>>
              hlcf.yield
            } else {
              hlcf.yield
            }
            hlcf.yield
          } else {
            hlcf.yield
          }
          llvm.intr.trap
          hlcf.loop "_loop_0" {
            hlcf.continue "_loop_0"
          }
          kgen.unreachable
        } else {
          hlcf.yield
        }
        %112 = pop.load %57 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
        hlcf.yield %112 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
      } else {
        hlcf.yield %9 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
      }
      kgen.call @"std::builtin::debug_assert::_debug_assert_msg[LITImmutOrigin,::Origin[::Bool(False), $0]](::UnsafePointer[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1, ::AddressSpace(::Int(0))],::Int,::SourceLocation)"(%82, %99) : (!kgen.pointer<none>, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>) -> ()
      hlcf.yield
    }
    %58 = pop.load %26 : !kgen.pointer<index>
    %59 = index.and %58, %index4611686018427387904
    %60 = index.cmp ne(%59, %index0)
    hlcf.if %60 {
      %75 = pop.load %23 : !kgen.pointer<pointer<none>>
      %76 = pop.pointer.bitcast %75 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %77 = pop.offset %76[%idx-8] : !kgen.pointer<scalar<ui8>>
      %78 = pop.pointer.bitcast %77 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
      %79 = kgen.struct.gep %78[0] : <struct<(scalar<index>) memoryOnly>>
      %80 = pop.atomic.rmw sub(%79, %simd_10) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
      %81 = pop.cmp eq(%80, %simd_10) : <1, index>
      %82 = pop.cast_to_builtin %81 : !pop.scalar<bool> to i1
      hlcf.if %82 {
        pop.fence syncscope("") acquire
        pop.aligned_free %77 : <scalar<ui8>>
        hlcf.yield
      } else {
        hlcf.yield
      }
      hlcf.yield
    } else {
      hlcf.yield
    }
    %61 = pop.load %32 : !kgen.pointer<index>
    %62 = index.and %61, %index4611686018427387904
    %63 = index.cmp ne(%62, %index0)
    hlcf.if %63 {
      %75 = pop.load %29 : !kgen.pointer<pointer<none>>
      %76 = pop.pointer.bitcast %75 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %77 = pop.offset %76[%idx-8] : !kgen.pointer<scalar<ui8>>
      %78 = pop.pointer.bitcast %77 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
      %79 = kgen.struct.gep %78[0] : <struct<(scalar<index>) memoryOnly>>
      %80 = pop.atomic.rmw sub(%79, %simd_10) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
      %81 = pop.cmp eq(%80, %simd_10) : <1, index>
      %82 = pop.cast_to_builtin %81 : !pop.scalar<bool> to i1
      hlcf.if %82 {
        pop.fence syncscope("") acquire
        pop.aligned_free %77 : <scalar<ui8>>
        hlcf.yield
      } else {
        hlcf.yield
      }
      hlcf.yield
    } else {
      hlcf.yield
    }
    %64 = kgen.struct.gep %5[0] : <struct<(array<619, scalar<ui128>>) memoryOnly>>
    %65 = pop.array.gep %64[%arg0] : <array<619, scalar<ui128>>>
    %66 = pop.load %65 : !kgen.pointer<scalar<ui128>>
    %67 = pop.shr %66, %simd_0 : !pop.scalar<ui128>
    %68 = pop.cast fast %67 : !pop.scalar<ui128> to !pop.scalar<ui64>
    %69 = index.sub %index10, %arg1
    %70 = pop.cast_from_builtin %69 : index to !pop.scalar<index>
    %71 = pop.cast %70 : !pop.scalar<index> to !pop.scalar<ui64>
    %72 = pop.shr %68, %71 : !pop.scalar<ui64>
    %73 = pop.add %72, %simd_13 : !pop.scalar<ui64>
    %74 = pop.shr %73, %simd_13 : !pop.scalar<ui64>
    kgen.return %74 : !pop.scalar<ui64>
  }
  kgen.func @"std::builtin::_format_float::_case_shorter_interval_left_endpoint_upper_threshold[::DType,::Int](),CarrierDType=ui64,sig_bits=52"() -> index {
    %idx3 = index.constant 3
    %simd = kgen.param.constant: scalar<ui64> = <10>
    %idx1 = index.constant 1
    %simd_0 = kgen.param.constant: scalar<ui64> = <5>
    %index10 = kgen.param.constant = <10>
    %index1 = kgen.param.constant = <1>
    %simd_1 = kgen.param.constant: scalar<ui64> = <18014398509481983>
    %index0 = kgen.param.constant = <0>
    %simd_2 = kgen.param.constant: scalar<ui64> = <0>
    %simd_3 = kgen.param.constant: scalar<ui64> = <1>
    %index-1 = kgen.param.constant = <-1>
    %0 = kgen.param.constant: i1 = <1>
    hlcf.loop "_loop_0" (%arg0 = %index0 : index, %arg1 = %simd_1 : !pop.scalar<ui64>) {
      %1 = pop.rem %arg1, %simd_0 : !pop.scalar<ui64>
      %2 = pop.cmp eq(%1, %simd_2) : <1, ui64>
      %3 = pop.cast_to_builtin %2 : !pop.scalar<bool> to i1
      hlcf.if %3 {
        hlcf.yield
      } else {
        hlcf.break "_loop_0"
      }
      %4 = pop.div %arg1, %simd_0 : !pop.scalar<ui64>
      %5 = index.add %arg0, %index1
      hlcf.continue "_loop_0" %5, %4 : index, !pop.scalar<ui64>
    }
    hlcf.loop "_loop_0" (%arg0 = %index1 : index, %arg1 = %index10 : index, %arg2 = %idx1 : index) {
      %1 = index.cmp sgt(%arg2, %index0)
      hlcf.if %1 {
        hlcf.yield
      } else {
        hlcf.break "_loop_0"
      }
      %2 = index.and %arg2, %index1
      %3 = index.cmp ne(%2, %index0)
      %4 = index.mul %arg0, %arg1
      %5 = pop.select %3, %4, %arg0 : index
      %6 = index.mul %arg1, %arg1
      %7 = index.shrs %arg2, %index1
      hlcf.continue "_loop_0" %5, %6, %7 : index, index, index
    }
    hlcf.loop "_loop_0" (%arg0 = %index-1 : index, %arg1 = %simd : !pop.scalar<ui64>) {
      %1 = pop.cmp eq(%arg1, %simd_2) : <1, ui64>
      %2 = pop.cast_to_builtin %1 : !pop.scalar<bool> to i1
      %3 = pop.xor %2, %0
      hlcf.if %3 {
        hlcf.yield
      } else {
        hlcf.break "_loop_0"
      }
      %4 = index.add %arg0, %index1
      %5 = pop.shr %arg1, %simd_3 : !pop.scalar<ui64>
      hlcf.continue "_loop_0" %4, %5 : index, !pop.scalar<ui64>
    }
    kgen.return %idx3 : index
  }
  kgen.func @"std::builtin::debug_assert::_debug_assert_msg[LITImmutOrigin,::Origin[::Bool(False), $0]](::UnsafePointer[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1, ::AddressSpace(::Int(0))],::Int,::SourceLocation)"(%arg0: !kgen.pointer<none>, %arg1: !kgen.struct<(index, index, struct<(pointer<none>, index)>)>) no_inline {
    %index1 = kgen.param.constant = <1>
    %0 = kgen.struct.extract %arg1[2] : <(index, index, struct<(pointer<none>, index)>)>
    %1 = kgen.struct.extract %0[0] : <(pointer<none>, index)>
    %2 = kgen.struct.extract %arg1[0] : <(index, index, struct<(pointer<none>, index)>)>
    %3 = kgen.struct.extract %arg1[1] : <(index, index, struct<(pointer<none>, index)>)>
    kgen.call @"std::io::io::_printf[KGENParamList[::AnyType],::StringSlice[::Bool(False), StaticConstantOrigin, *?],*::AnyType,LITImmutOrigin,::Origin[::Bool(False), $3]](*$0,file:::FileDescriptor),types.values`=[[typevalue<#kgen.instref<\1B\22std::memory::unsafe_pointer::UnsafePointer,mut=0,origin._mlir_origin`={  },type=[typevalue<#kgen.instref<\\1B\\22std::builtin::simd::SIMD,dtype=ui8,size=1\\22>>, scalar<ui8>],origin={  },address_space=0\22>>, pointer<none>], [typevalue<#kgen.instref<\1B\22std::builtin::int::Int\22>>, index], [typevalue<#kgen.instref<\1B\22std::builtin::int::Int\22>>, index], [typevalue<#kgen.instref<\1B\22std::memory::unsafe_pointer::UnsafePointer,mut=0,origin._mlir_origin`={  },type=[typevalue<#kgen.instref<\\1B\\22std::builtin::simd::SIMD,dtype=ui8,size=1\\22>>, scalar<ui8>],origin={  },address_space=0\22>>, pointer<none>]],fmt={ #interp.memref<{[(#interp.memory_handle<16, \22At: %s:%llu:%llu: Assert Error: %s\\0A\\00\22 string>, const_global, [], [])], []}, 0, 0>, 35 }"(%1, %2, %3, %arg0, %index1) : (!kgen.pointer<none>, index, index, !kgen.pointer<none>, index) -> ()
    llvm.intr.trap
    hlcf.loop "_loop_0" {
      hlcf.continue "_loop_0"
    }
    kgen.unreachable
  }
  kgen.func @"std::builtin::format_int::_write_int[::DType,::Writer,::Int,::StringSlice[::Bool(False), StaticConstantOrigin, *?]]($1&,::SIMD[$0, ::Int(1)],prefix:::StringSlice[::Bool(False), StaticConstantOrigin, *?]),dtype=si64,W=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>],radix=10,digit_chars={ #interp.memref<{[(#interp.memory_handle<16, \220123456789abcdefghijklmnopqrstuvwxyz\\00\22 string>, const_global, [], [])], []}, 0, 0>, 36 }"(%arg0: !kgen.struct<(pointer<none>, index) memoryOnly>, %arg1: !pop.scalar<si64>, %arg2: !kgen.struct<(pointer<none>, index)>) -> !kgen.struct<(pointer<none>, index) memoryOnly> {
    %index64 = kgen.param.constant = <64>
    %index1 = kgen.param.constant = <1>
    %none = kgen.param.constant: none = <#kgen.none>
    %simd = kgen.param.constant: scalar<ui8> = <0>
    %simd_0 = kgen.param.constant: scalar<si64> = <0>
    %pointer = kgen.param.constant: pointer<none> = <#interp.memref<{[(#memory_handle5, const_global, [], [])], []}, 0, 0>>
    %index65 = kgen.param.constant = <65>
    %index2305843009213693952 = kgen.param.constant = <2305843009213693952>
    %string = kgen.param.constant: string = <"-">
    %simd_1 = kgen.param.constant: scalar<index> = <1>
    %index8 = kgen.param.constant = <8>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %index0 = kgen.param.constant = <0>
    %index2 = kgen.param.constant = <2>
    %index2048 = kgen.param.constant = <2048>
    %idx-8 = index.constant -8
    %simd_2 = kgen.param.constant: scalar<si64> = <10>
    %simd_3 = kgen.param.constant: scalar<bool> = <true>
    %simd_4 = kgen.param.constant: scalar<si64> = <-10>
    %simd_5 = kgen.param.constant: scalar<ui8> = <1>
    %idx-4 = index.constant -4
    %simd_6 = kgen.param.constant: scalar<uindex> = <32>
    %index32 = kgen.param.constant = <32>
    %index5 = kgen.param.constant = <5>
    %index16 = kgen.param.constant = <16>
    %idx63 = index.constant 63
    %idx64 = index.constant 64
    %0 = kgen.struct.extract %arg0[0] : <(pointer<none>, index) memoryOnly>
    %1 = kgen.struct.extract %arg0[1] : <(pointer<none>, index) memoryOnly>
    %2 = pop.string.address %string
    %3 = pop.pointer.bitcast %2 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %4 = pop.pointer.bitcast %pointer : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
    %5 = pop.cmp ge(%arg1, %simd_0) : <1, si64>
    %6 = pop.cast_to_builtin %5 : !pop.scalar<bool> to i1
    %7 = pop.cmp lt(%arg1, %simd_0) : <1, si64>
    %8 = pop.cast_to_builtin %7 : !pop.scalar<bool> to i1
    %9:2 = hlcf.if %8 -> !kgen.pointer<none>, index {
      %18 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
      pop.stack_alloc.lifetime.start(%18) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
      %19 = kgen.struct.gep %18[1] : <struct<(pointer<none>, index, index) memoryOnly>>
      pop.store %index1, %19 : !kgen.pointer<index>
      %20 = kgen.struct.gep %18[0] : <struct<(pointer<none>, index, index) memoryOnly>>
      pop.store %3, %20 : !kgen.pointer<pointer<none>>
      %21 = kgen.struct.gep %18[2] : <struct<(pointer<none>, index, index) memoryOnly>>
      pop.store %index2305843009213693952, %21 : !kgen.pointer<index>
      %22 = kgen.call @"std::collections::string::string::String::write_to[::Writer](::String,$0&),writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%18, %arg0) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
      %23 = kgen.struct.extract %22[0] : <(pointer<none>, index) memoryOnly>
      %24 = kgen.struct.extract %22[1] : <(pointer<none>, index) memoryOnly>
      %25 = pop.load %21 : !kgen.pointer<index>
      %26 = index.and %25, %index4611686018427387904
      %27 = index.cmp ne(%26, %index0)
      hlcf.if %27 {
        %28 = pop.load %20 : !kgen.pointer<pointer<none>>
        %29 = pop.pointer.bitcast %28 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
        %30 = pop.offset %29[%idx-8] : !kgen.pointer<scalar<ui8>>
        %31 = pop.pointer.bitcast %30 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
        %32 = kgen.struct.gep %31[0] : <struct<(scalar<index>) memoryOnly>>
        %33 = pop.atomic.rmw sub(%32, %simd_1) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
        %34 = pop.cmp eq(%33, %simd_1) : <1, index>
        %35 = pop.cast_to_builtin %34 : !pop.scalar<bool> to i1
        hlcf.if %35 {
          pop.fence syncscope("") acquire
          pop.aligned_free %30 : <scalar<ui8>>
          hlcf.yield
        } else {
          hlcf.yield
        }
        hlcf.yield
      } else {
        hlcf.yield
      }
      pop.stack_alloc.lifetime.end(%18) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
      hlcf.yield %23, %24 : !kgen.pointer<none>, index
    } else {
      hlcf.yield %0, %1 : !kgen.pointer<none>, index
    }
    %10 = kgen.struct.create(%9#0, %9#1) : !kgen.struct<(pointer<none>, index) memoryOnly>
    %11 = kgen.call @"std::collections::string::string_slice::StringSlice::write_to[::Writer](::StringSlice[$0, $1, $2],$3&),mut=0,writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%arg2, %10) : (!kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
    %12 = kgen.struct.extract %11[0] : <(pointer<none>, index) memoryOnly>
    %13 = kgen.struct.extract %11[1] : <(pointer<none>, index) memoryOnly>
    %14 = pop.cmp eq(%arg1, %simd_0) : <1, si64>
    %15 = pop.cast_to_builtin %14 : !pop.scalar<bool> to i1
    %16:2 = hlcf.if %15 -> !kgen.pointer<none>, index {
      %18 = pop.load %4 : !kgen.pointer<scalar<ui8>>
      %19 = pop.stack_allocation 1 x array<2, scalar<ui8>> align 1 marked
      pop.stack_alloc.lifetime.start(%19) : !kgen.pointer<array<2, scalar<ui8>>>
      %20 = pop.pointer.bitcast %19 : !kgen.pointer<array<2, scalar<ui8>>> to !kgen.pointer<scalar<ui8>>
      pop.store %18, %20 : !kgen.pointer<scalar<ui8>>
      %21 = pop.offset %20[%index1] : !kgen.pointer<scalar<ui8>>
      pop.store %simd, %21 : !kgen.pointer<scalar<ui8>>
      %22 = pop.pointer.bitcast %19 : !kgen.pointer<array<2, scalar<ui8>>> to !kgen.pointer<none>
      %23 = kgen.struct.create(%22, %index1) : !kgen.struct<(pointer<none>, index)>
      %24 = kgen.call @"std::collections::string::string_slice::StringSlice::write_to[::Writer](::StringSlice[$0, $1, $2],$3&),mut=1,writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%23, %11) : (!kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
      %25 = kgen.struct.extract %24[0] : <(pointer<none>, index) memoryOnly>
      %26 = kgen.struct.extract %24[1] : <(pointer<none>, index) memoryOnly>
      pop.stack_alloc.lifetime.end(%19) : !kgen.pointer<array<2, scalar<ui8>>>
      hlcf.yield %25, %26 : !kgen.pointer<none>, index
    } else {
      %18 = pop.stack_allocation 1 x array<65, scalar<ui8>> align 1 marked
      pop.stack_alloc.lifetime.start(%18) : !kgen.pointer<array<65, scalar<ui8>>>
      %19 = pop.pointer.bitcast %18 : !kgen.pointer<array<65, scalar<ui8>>> to !kgen.pointer<scalar<ui8>>
      %20 = pop.offset %19[%index64] : !kgen.pointer<scalar<ui8>>
      pop.store %simd, %20 : !kgen.pointer<scalar<ui8>>
      %21 = hlcf.if %6 -> index {
        %103 = hlcf.loop "_loop_0" (%arg3 = %idx63 : index, %arg4 = %arg1 : !pop.scalar<si64>) -> index {
          %104 = pop.cmp ne(%arg4, %simd_0) : <1, si64>
          %105 = pop.cast_to_builtin %104 : !pop.scalar<bool> to i1
          hlcf.if %105 {
            hlcf.yield
          } else {
            hlcf.break "_loop_0" %arg3 : index
          }
          %106 = pop.div %arg4, %simd_2 : !pop.scalar<si64>
          %107 = pop.mul %106, %simd_2 : !pop.scalar<si64>
          %108 = pop.sub %arg4, %107 : !pop.scalar<si64>
          %109 = pop.cmp lt(%arg4, %simd_0) : <1, si64>
          %110 = pop.cmp ne(%108, %simd_0) : <1, si64>
          %111 = pop.simd.and %109, %110 : <1, bool>
          %112 = pop.simd.select %111, %simd_2, %simd_0 : <1, si64>
          %113 = pop.add %108, %112 : !pop.scalar<si64>
          %114 = pop.offset %19[%arg3] : !kgen.pointer<scalar<ui8>>
          %115 = pop.cast fast %113 : !pop.scalar<si64> to !pop.scalar<index>
          %116 = pop.cast_to_builtin %115 : !pop.scalar<index> to index
          %117 = pop.offset %4[%116] : !kgen.pointer<scalar<ui8>>
          %118 = pop.load %117 : !kgen.pointer<scalar<ui8>>
          pop.store %118, %114 : !kgen.pointer<scalar<ui8>>
          %119 = index.sub %arg3, %index1
          hlcf.continue "_loop_0" %119, %106 : index, !pop.scalar<si64>
        }
        hlcf.yield %103 : index
      } else {
        %103 = hlcf.loop "_loop_0" (%arg3 = %idx63 : index, %arg4 = %arg1 : !pop.scalar<si64>) -> index {
          %104 = pop.cmp ne(%arg4, %simd_0) : <1, si64>
          %105 = pop.cast_to_builtin %104 : !pop.scalar<bool> to i1
          hlcf.if %105 {
            hlcf.yield
          } else {
            hlcf.break "_loop_0" %arg3 : index
          }
          %106 = pop.div %arg4, %simd_4 : !pop.scalar<si64>
          %107 = pop.mul %106, %simd_4 : !pop.scalar<si64>
          %108 = pop.sub %arg4, %107 : !pop.scalar<si64>
          %109 = pop.cmp lt(%arg4, %simd_0) : <1, si64>
          %110 = pop.simd.xor %109, %simd_3 : <1, bool>
          %111 = pop.cmp ne(%108, %simd_0) : <1, si64>
          %112 = pop.simd.and %110, %111 : <1, bool>
          %113 = pop.simd.select %112, %simd_4, %simd_0 : <1, si64>
          %114 = pop.add %108, %113 : !pop.scalar<si64>
          %115 = pop.abs %114 : !pop.scalar<si64>
          %116 = pop.offset %19[%arg3] : !kgen.pointer<scalar<ui8>>
          %117 = pop.cast fast %115 : !pop.scalar<si64> to !pop.scalar<index>
          %118 = pop.cast_to_builtin %117 : !pop.scalar<index> to index
          %119 = pop.offset %4[%118] : !kgen.pointer<scalar<ui8>>
          %120 = pop.load %119 : !kgen.pointer<scalar<ui8>>
          pop.store %120, %116 : !kgen.pointer<scalar<ui8>>
          %121 = index.sub %arg3, %index1
          %122 = pop.floordiv %arg4, %simd_4 : !pop.scalar<si64>
          %123 = pop.neg %122 : !pop.scalar<si64>
          hlcf.continue "_loop_0" %121, %123 : index, !pop.scalar<si64>
        }
        hlcf.yield %103 : index
      }
      %22 = index.add %21, %index1
      %23 = pop.stack_allocation 1 x union<struct<()>, index>
      %24 = pop.union.bitcast %23 : <union<struct<()>, index>> as <index>
      pop.store %22, %24 : !kgen.pointer<index>
      %25 = pop.load %23 : !kgen.pointer<union<struct<()>, index>>
      %26 = pop.stack_allocation 1 x union<struct<()>, index>
      %27 = pop.union.bitcast %26 : <union<struct<()>, index>> as <index>
      pop.store %idx64, %27 : !kgen.pointer<index>
      %28 = pop.load %26 : !kgen.pointer<union<struct<()>, index>>
      %29 = pop.stack_allocation 1 x union<struct<()>, index>
      %30 = pop.union.bitcast %29 : <union<struct<()>, index>> as <index>
      pop.store %25, %29 : !kgen.pointer<union<struct<()>, index>>
      %31 = pop.stack_allocation 1 x union<struct<()>, index>
      %32 = pop.union.bitcast %31 : <union<struct<()>, index>> as <index>
      %33 = pop.load %30 : !kgen.pointer<index>
      pop.store %33, %32 : !kgen.pointer<index>
      %34 = pop.load %31 : !kgen.pointer<union<struct<()>, index>>
      %35 = kgen.struct.create(%34, %simd_5) : !kgen.struct<(union<struct<()>, index>, scalar<ui8>)>
      %36 = kgen.struct.create(%35) : !kgen.struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>
      %37 = kgen.struct.create(%36) : !kgen.struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>
      %38 = pop.stack_allocation 1 x union<struct<()>, index>
      %39 = pop.union.bitcast %38 : <union<struct<()>, index>> as <index>
      pop.store %28, %38 : !kgen.pointer<union<struct<()>, index>>
      %40 = pop.stack_allocation 1 x union<struct<()>, index>
      %41 = pop.union.bitcast %40 : <union<struct<()>, index>> as <index>
      %42 = pop.load %39 : !kgen.pointer<index>
      pop.store %42, %41 : !kgen.pointer<index>
      %43 = pop.load %40 : !kgen.pointer<union<struct<()>, index>>
      %44 = kgen.struct.create(%43, %simd_5) : !kgen.struct<(union<struct<()>, index>, scalar<ui8>)>
      %45 = kgen.struct.create(%44) : !kgen.struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>
      %46 = kgen.struct.create(%45) : !kgen.struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>
      %47 = kgen.struct.create(%37) : !kgen.struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>
      %48 = kgen.struct.create(%46) : !kgen.struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>
      %49 = kgen.struct.create(%47, %48) : !kgen.struct<(struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>, struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>) memoryOnly>
      %50 = kgen.call @"std::builtin::builtin_slice::ContiguousSlice::indices(::ContiguousSlice,::Int)"(%49, %index65) : (!kgen.struct<(struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>, struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>) memoryOnly>, index) -> !kgen.struct<(struct<(index, index)>)>
      %51 = kgen.struct.extract %50[0] : <(struct<(index, index)>)>
      %52 = kgen.struct.extract %51[0] : <(index, index)>
      %53 = kgen.struct.extract %51[1] : <(index, index)>
      %54 = pop.offset %19[%52] : !kgen.pointer<scalar<ui8>>
      %55 = pop.pointer.bitcast %54 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<none>
      %56 = index.sub %53, %52
      %57 = index.add %56, %13
      %58 = index.cmp sgt(%57, %index2048)
      hlcf.if %58 {
        kgen.call @"std::io::io::_printf[KGENParamList[::AnyType],::StringSlice[::Bool(False), StaticConstantOrigin, *?],*::AnyType,LITImmutOrigin,::Origin[::Bool(False), $3]](*$0,file:::FileDescriptor),types.values`=[],fmt={ #interp.memref<{[(#interp.memory_handle<16, \22HEAP_BUFFER_BYTES exceeded, increase with: `mojo -D HEAP_BUFFER_BYTES=4096`\\0A\\00\22 string>, const_global, [], [])], []}, 0, 0>, 76 }"(%index1) : (index) -> ()
        llvm.intr.trap
        hlcf.loop "_loop_0" {
          hlcf.continue "_loop_0"
        }
        kgen.unreachable
      } else {
        hlcf.yield
      }
      %59 = pop.pointer.bitcast %12 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %60 = pop.offset %59[%13] : !kgen.pointer<scalar<ui8>>
      %61 = pop.pointer.bitcast %60 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<none>
      %62 = pop.stack_allocation 1 x pointer<none>
      %63 = pop.pointer.bitcast %62 : !kgen.pointer<pointer<none>> to !kgen.pointer<struct<(array<1, pointer<none>>)>>
      %64 = kgen.struct.gep %63[0] : <struct<(array<1, pointer<none>>)>>
      %65 = pop.pointer.bitcast %64 : !kgen.pointer<array<1, pointer<none>>> to !kgen.pointer<pointer<none>>
      pop.store %61, %65 : !kgen.pointer<pointer<none>>
      %66 = pop.load %62 : !kgen.pointer<pointer<none>>
      %67 = pop.pointer.bitcast %66 : !kgen.pointer<none> to !kgen.pointer<scalar<ui64>>
      %68 = pop.pointer.bitcast %66 : !kgen.pointer<none> to !kgen.pointer<scalar<ui32>>
      %69 = pop.pointer.bitcast %66 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %70 = pop.offset %69[%index1] : !kgen.pointer<scalar<ui8>>
      %71 = pop.offset %69[%56] : !kgen.pointer<scalar<ui8>>
      %72 = pop.offset %71[%idx-8] : !kgen.pointer<scalar<ui8>>
      %73 = pop.pointer.bitcast %72 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui64>>
      %74 = pop.offset %71[%idx-4] : !kgen.pointer<scalar<ui8>>
      %75 = pop.pointer.bitcast %74 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui32>>
      %76 = pop.stack_allocation 1 x pointer<none>
      %77 = pop.pointer.bitcast %76 : !kgen.pointer<pointer<none>> to !kgen.pointer<struct<(array<1, pointer<none>>)>>
      %78 = kgen.struct.gep %77[0] : <struct<(array<1, pointer<none>>)>>
      %79 = pop.pointer.bitcast %78 : !kgen.pointer<array<1, pointer<none>>> to !kgen.pointer<pointer<none>>
      pop.store %55, %79 : !kgen.pointer<pointer<none>>
      %80 = pop.load %76 : !kgen.pointer<pointer<none>>
      %81 = pop.cast_from_builtin %56 : index to !pop.scalar<index>
      %82 = pop.cast %81 : !pop.scalar<index> to !pop.scalar<uindex>
      %83 = index.cmp sge(%56, %index8)
      %84 = index.cmp slt(%56, %index5)
      %85 = index.cmp sle(%56, %index2)
      %86 = index.sub %56, %index2
      %87 = pop.offset %69[%86] : !kgen.pointer<scalar<ui8>>
      %88 = index.cmp sle(%56, %index16)
      %89 = index.sub %56, %index1
      %90 = pop.offset %69[%89] : !kgen.pointer<scalar<ui8>>
      %91 = pop.pointer.bitcast %80 : !kgen.pointer<none> to !kgen.pointer<scalar<ui64>>
      %92 = pop.pointer.bitcast %80 : !kgen.pointer<none> to !kgen.pointer<scalar<ui32>>
      %93 = pop.pointer.bitcast %80 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %94 = pop.offset %93[%86] : !kgen.pointer<scalar<ui8>>
      %95 = pop.offset %93[%89] : !kgen.pointer<scalar<ui8>>
      %96 = pop.offset %93[%index1] : !kgen.pointer<scalar<ui8>>
      %97 = pop.offset %93[%56] : !kgen.pointer<scalar<ui8>>
      %98 = pop.offset %97[%idx-8] : !kgen.pointer<scalar<ui8>>
      %99 = pop.pointer.bitcast %98 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui64>>
      %100 = pop.offset %97[%idx-4] : !kgen.pointer<scalar<ui8>>
      %101 = pop.pointer.bitcast %100 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui32>>
      %102 = index.cmp eq(%53, %52)
      hlcf.if %102 {
        hlcf.yield
      } else {
        hlcf.if %84 {
          %103 = pop.load %93 : !kgen.pointer<scalar<ui8>>
          pop.store %103, %69 : !kgen.pointer<scalar<ui8>>
          %104 = pop.load %95 : !kgen.pointer<scalar<ui8>>
          pop.store %104, %90 : !kgen.pointer<scalar<ui8>>
          hlcf.if %85 {
            hlcf.yield
          } else {
            %105 = pop.load %96 : !kgen.pointer<scalar<ui8>>
            pop.store %105, %70 : !kgen.pointer<scalar<ui8>>
            %106 = pop.load %94 : !kgen.pointer<scalar<ui8>>
            pop.store %106, %87 : !kgen.pointer<scalar<ui8>>
            hlcf.yield
          }
          hlcf.yield
        } else {
          hlcf.if %88 {
            hlcf.if %83 {
              %103 = pop.load volatile<0> invariant<0> nontemporal<0> %91 align<1> : !kgen.pointer<scalar<ui64>>
              pop.store volatile<0> nontemporal<0> %103, %67 align<1> : !kgen.pointer<scalar<ui64>>
              %104 = pop.load volatile<0> invariant<0> nontemporal<0> %99 align<1> : !kgen.pointer<scalar<ui64>>
              pop.store volatile<0> nontemporal<0> %104, %73 align<1> : !kgen.pointer<scalar<ui64>>
              hlcf.yield
            } else {
              %103 = pop.load volatile<0> invariant<0> nontemporal<0> %92 align<1> : !kgen.pointer<scalar<ui32>>
              pop.store volatile<0> nontemporal<0> %103, %68 align<1> : !kgen.pointer<scalar<ui32>>
              %104 = pop.load volatile<0> invariant<0> nontemporal<0> %101 align<1> : !kgen.pointer<scalar<ui32>>
              pop.store volatile<0> nontemporal<0> %104, %75 align<1> : !kgen.pointer<scalar<ui32>>
              hlcf.yield
            }
            hlcf.yield
          } else {
            %103 = pop.floordiv %82, %simd_6 : !pop.scalar<uindex>
            %104 = pop.mul %103, %simd_6 : !pop.scalar<uindex>
            %105 = pop.cast fast %104 : !pop.scalar<uindex> to !pop.scalar<index>
            %106 = pop.cast_to_builtin %105 : !pop.scalar<index> to index
            hlcf.loop "_loop_0" (%arg3 = %index0 : index) {
              %108 = index.add %arg3, %index32
              %109 = index.sub %106, %arg3
              %110 = index.cmp slt(%arg3, %106)
              %111 = pop.select %110, %109, %index0 : index
              %112 = index.cmp sle(%111, %index0)
              %113 = pop.select %112, %arg3, %108 : index
              %114:2 = lit.try "try0" -> index, index {
                hlcf.if %112 {
                  lit.try.raise "try0" %113, %arg3 : index, index
                } else {
                  hlcf.yield
                }
                lit.try.yield %113, %arg3 : index, index
              } except (%arg4: index, %arg5: index) {
                hlcf.break "_loop_0"
              } else (%arg4: index, %arg5: index) {
                lit.try.yield %arg4, %arg5 : index, index
              }
              %115 = pop.offset %93[%114#1] : !kgen.pointer<scalar<ui8>>
              %116 = pop.pointer.bitcast %115 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<simd<32, ui8>>
              %117 = pop.load volatile<0> invariant<0> nontemporal<0> %116 align<1> : !kgen.pointer<simd<32, ui8>>
              %118 = pop.offset %69[%114#1] : !kgen.pointer<scalar<ui8>>
              %119 = pop.pointer.bitcast %118 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<simd<32, ui8>>
              pop.store volatile<0> nontemporal<0> %117, %119 align<1> : !kgen.pointer<simd<32, ui8>>
              hlcf.continue "_loop_0" %114#0 : index
            }
            %107 = index.maxs %106, %56
            hlcf.loop "_loop_2" (%arg3 = %106 : index) {
              %108 = index.add %arg3, %index1
              %109 = index.cmp eq(%arg3, %107)
              %110 = pop.select %109, %arg3, %108 : index
              %111:2 = lit.try "try2" -> index, index {
                hlcf.if %109 {
                  lit.try.raise "try2" %110, %arg3 : index, index
                } else {
                  hlcf.yield
                }
                lit.try.yield %110, %arg3 : index, index
              } except (%arg4: index, %arg5: index) {
                hlcf.break "_loop_2"
              } else (%arg4: index, %arg5: index) {
                lit.try.yield %arg4, %arg5 : index, index
              }
              %112 = pop.offset %93[%111#1] : !kgen.pointer<scalar<ui8>>
              %113 = pop.load volatile<0> invariant<0> nontemporal<0> %112 align<1> : !kgen.pointer<scalar<ui8>>
              %114 = pop.offset %69[%111#1] : !kgen.pointer<scalar<ui8>>
              pop.store volatile<0> nontemporal<0> %113, %114 align<1> : !kgen.pointer<scalar<ui8>>
              hlcf.continue "_loop_2" %111#0 : index
            }
            hlcf.yield
          }
          hlcf.yield
        }
        hlcf.yield
      }
      pop.stack_alloc.lifetime.end(%18) : !kgen.pointer<array<65, scalar<ui8>>>
      hlcf.yield %12, %57 : !kgen.pointer<none>, index
    }
    %17 = kgen.struct.create(%16#0, %16#1) : !kgen.struct<(pointer<none>, index) memoryOnly>
    kgen.return %17 : !kgen.struct<(pointer<none>, index) memoryOnly>
  }
  kgen.func @"std::builtin::format_int::_write_int[::DType,::Writer,::Int,::StringSlice[::Bool(False), StaticConstantOrigin, *?]]($1&,::SIMD[$0, ::Int(1)],prefix:::StringSlice[::Bool(False), StaticConstantOrigin, *?]),dtype=si64,W=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>],radix=10,digit_chars={ #interp.memref<{[(#interp.memory_handle<16, \220123456789abcdefghijklmnopqrstuvwxyz\\00\22 string>, const_global, [], [])], []}, 0, 0>, 36 }"(%arg0: !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, %arg1: !pop.scalar<si64>, %arg2: !kgen.struct<(pointer<none>, index)>) {
    %idx-8 = index.constant -8
    %simd = kgen.param.constant: scalar<si64> = <10>
    %simd_0 = kgen.param.constant: scalar<bool> = <true>
    %simd_1 = kgen.param.constant: scalar<si64> = <-10>
    %simd_2 = kgen.param.constant: scalar<ui8> = <1>
    %idx63 = index.constant 63
    %idx64 = index.constant 64
    %index2233785415175766016 = kgen.param.constant = <2233785415175766016>
    %index56 = kgen.param.constant = <56>
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %index0 = kgen.param.constant = <0>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %simd_3 = kgen.param.constant: scalar<index> = <1>
    %string = kgen.param.constant: string = <"-">
    %index2305843009213693952 = kgen.param.constant = <2305843009213693952>
    %index65 = kgen.param.constant = <65>
    %pointer = kgen.param.constant: pointer<none> = <#interp.memref<{[(#memory_handle5, const_global, [], [])], []}, 0, 0>>
    %simd_4 = kgen.param.constant: scalar<si64> = <0>
    %simd_5 = kgen.param.constant: scalar<ui8> = <0>
    %index1 = kgen.param.constant = <1>
    %index64 = kgen.param.constant = <64>
    %0 = pop.string.address %string
    %1 = pop.pointer.bitcast %0 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %2 = pop.pointer.bitcast %pointer : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
    %3 = pop.cmp ge(%arg1, %simd_4) : <1, si64>
    %4 = pop.cast_to_builtin %3 : !pop.scalar<bool> to i1
    %5 = pop.cmp lt(%arg1, %simd_4) : <1, si64>
    %6 = pop.cast_to_builtin %5 : !pop.scalar<bool> to i1
    hlcf.if %6 {
      %9 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
      pop.stack_alloc.lifetime.start(%9) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
      %10 = kgen.struct.gep %9[1] : <struct<(pointer<none>, index, index) memoryOnly>>
      pop.store %index1, %10 : !kgen.pointer<index>
      %11 = kgen.struct.gep %9[0] : <struct<(pointer<none>, index, index) memoryOnly>>
      pop.store %1, %11 : !kgen.pointer<pointer<none>>
      %12 = kgen.struct.gep %9[2] : <struct<(pointer<none>, index, index) memoryOnly>>
      pop.store %index2305843009213693952, %12 : !kgen.pointer<index>
      %13 = pop.pointer.bitcast %9 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
      %14 = pop.load %12 : !kgen.pointer<index>
      %15 = index.and %14, %index-9223372036854775808
      %16 = index.cmp ne(%15, %index0)
      %17 = hlcf.if %16 -> !kgen.pointer<none> {
        hlcf.yield %13 : !kgen.pointer<none>
      } else {
        %26 = pop.load %11 : !kgen.pointer<pointer<none>>
        hlcf.yield %26 : !kgen.pointer<none>
      }
      %18 = pop.load %12 : !kgen.pointer<index>
      %19 = index.and %18, %index-9223372036854775808
      %20 = index.cmp ne(%19, %index0)
      %21 = hlcf.if %20 -> index {
        %26 = pop.load %12 : !kgen.pointer<index>
        %27 = index.and %26, %index2233785415175766016
        %28 = index.shrs %27, %index56
        hlcf.yield %28 : index
      } else {
        %26 = pop.load %10 : !kgen.pointer<index>
        hlcf.yield %26 : index
      }
      %22 = kgen.struct.create(%17, %21) : !kgen.struct<(pointer<none>, index)>
      kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg0, %22) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      %23 = pop.load %12 : !kgen.pointer<index>
      %24 = index.and %23, %index4611686018427387904
      %25 = index.cmp ne(%24, %index0)
      hlcf.if %25 {
        %26 = pop.load %11 : !kgen.pointer<pointer<none>>
        %27 = pop.pointer.bitcast %26 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
        %28 = pop.offset %27[%idx-8] : !kgen.pointer<scalar<ui8>>
        %29 = pop.pointer.bitcast %28 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
        %30 = kgen.struct.gep %29[0] : <struct<(scalar<index>) memoryOnly>>
        %31 = pop.atomic.rmw sub(%30, %simd_3) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
        %32 = pop.cmp eq(%31, %simd_3) : <1, index>
        %33 = pop.cast_to_builtin %32 : !pop.scalar<bool> to i1
        hlcf.if %33 {
          pop.fence syncscope("") acquire
          pop.aligned_free %28 : <scalar<ui8>>
          hlcf.yield
        } else {
          hlcf.yield
        }
        hlcf.yield
      } else {
        hlcf.yield
      }
      pop.stack_alloc.lifetime.end(%9) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
      hlcf.yield
    } else {
      hlcf.yield
    }
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg0, %arg2) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    %7 = pop.cmp eq(%arg1, %simd_4) : <1, si64>
    %8 = pop.cast_to_builtin %7 : !pop.scalar<bool> to i1
    hlcf.if %8 {
      %9 = pop.load %2 : !kgen.pointer<scalar<ui8>>
      %10 = pop.stack_allocation 1 x array<2, scalar<ui8>> align 1 marked
      pop.stack_alloc.lifetime.start(%10) : !kgen.pointer<array<2, scalar<ui8>>>
      %11 = pop.pointer.bitcast %10 : !kgen.pointer<array<2, scalar<ui8>>> to !kgen.pointer<scalar<ui8>>
      pop.store %9, %11 : !kgen.pointer<scalar<ui8>>
      %12 = pop.offset %11[%index1] : !kgen.pointer<scalar<ui8>>
      pop.store %simd_5, %12 : !kgen.pointer<scalar<ui8>>
      %13 = pop.pointer.bitcast %10 : !kgen.pointer<array<2, scalar<ui8>>> to !kgen.pointer<none>
      %14 = kgen.struct.create(%13, %index1) : !kgen.struct<(pointer<none>, index)>
      kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=1"(%arg0, %14) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      pop.stack_alloc.lifetime.end(%10) : !kgen.pointer<array<2, scalar<ui8>>>
      hlcf.yield
    } else {
      %9 = pop.stack_allocation 1 x array<65, scalar<ui8>> align 1 marked
      pop.stack_alloc.lifetime.start(%9) : !kgen.pointer<array<65, scalar<ui8>>>
      %10 = pop.pointer.bitcast %9 : !kgen.pointer<array<65, scalar<ui8>>> to !kgen.pointer<scalar<ui8>>
      %11 = pop.offset %10[%index64] : !kgen.pointer<scalar<ui8>>
      pop.store %simd_5, %11 : !kgen.pointer<scalar<ui8>>
      %12 = hlcf.if %4 -> index {
        %49 = hlcf.loop "_loop_0" (%arg3 = %idx63 : index, %arg4 = %arg1 : !pop.scalar<si64>) -> index {
          %50 = pop.cmp ne(%arg4, %simd_4) : <1, si64>
          %51 = pop.cast_to_builtin %50 : !pop.scalar<bool> to i1
          hlcf.if %51 {
            hlcf.yield
          } else {
            hlcf.break "_loop_0" %arg3 : index
          }
          %52 = pop.div %arg4, %simd : !pop.scalar<si64>
          %53 = pop.mul %52, %simd : !pop.scalar<si64>
          %54 = pop.sub %arg4, %53 : !pop.scalar<si64>
          %55 = pop.cmp lt(%arg4, %simd_4) : <1, si64>
          %56 = pop.cmp ne(%54, %simd_4) : <1, si64>
          %57 = pop.simd.and %55, %56 : <1, bool>
          %58 = pop.simd.select %57, %simd, %simd_4 : <1, si64>
          %59 = pop.add %54, %58 : !pop.scalar<si64>
          %60 = pop.offset %10[%arg3] : !kgen.pointer<scalar<ui8>>
          %61 = pop.cast fast %59 : !pop.scalar<si64> to !pop.scalar<index>
          %62 = pop.cast_to_builtin %61 : !pop.scalar<index> to index
          %63 = pop.offset %2[%62] : !kgen.pointer<scalar<ui8>>
          %64 = pop.load %63 : !kgen.pointer<scalar<ui8>>
          pop.store %64, %60 : !kgen.pointer<scalar<ui8>>
          %65 = index.sub %arg3, %index1
          hlcf.continue "_loop_0" %65, %52 : index, !pop.scalar<si64>
        }
        hlcf.yield %49 : index
      } else {
        %49 = hlcf.loop "_loop_0" (%arg3 = %idx63 : index, %arg4 = %arg1 : !pop.scalar<si64>) -> index {
          %50 = pop.cmp ne(%arg4, %simd_4) : <1, si64>
          %51 = pop.cast_to_builtin %50 : !pop.scalar<bool> to i1
          hlcf.if %51 {
            hlcf.yield
          } else {
            hlcf.break "_loop_0" %arg3 : index
          }
          %52 = pop.div %arg4, %simd_1 : !pop.scalar<si64>
          %53 = pop.mul %52, %simd_1 : !pop.scalar<si64>
          %54 = pop.sub %arg4, %53 : !pop.scalar<si64>
          %55 = pop.cmp lt(%arg4, %simd_4) : <1, si64>
          %56 = pop.simd.xor %55, %simd_0 : <1, bool>
          %57 = pop.cmp ne(%54, %simd_4) : <1, si64>
          %58 = pop.simd.and %56, %57 : <1, bool>
          %59 = pop.simd.select %58, %simd_1, %simd_4 : <1, si64>
          %60 = pop.add %54, %59 : !pop.scalar<si64>
          %61 = pop.abs %60 : !pop.scalar<si64>
          %62 = pop.offset %10[%arg3] : !kgen.pointer<scalar<ui8>>
          %63 = pop.cast fast %61 : !pop.scalar<si64> to !pop.scalar<index>
          %64 = pop.cast_to_builtin %63 : !pop.scalar<index> to index
          %65 = pop.offset %2[%64] : !kgen.pointer<scalar<ui8>>
          %66 = pop.load %65 : !kgen.pointer<scalar<ui8>>
          pop.store %66, %62 : !kgen.pointer<scalar<ui8>>
          %67 = index.sub %arg3, %index1
          %68 = pop.floordiv %arg4, %simd_1 : !pop.scalar<si64>
          %69 = pop.neg %68 : !pop.scalar<si64>
          hlcf.continue "_loop_0" %67, %69 : index, !pop.scalar<si64>
        }
        hlcf.yield %49 : index
      }
      %13 = index.add %12, %index1
      %14 = pop.stack_allocation 1 x union<struct<()>, index>
      %15 = pop.union.bitcast %14 : <union<struct<()>, index>> as <index>
      pop.store %13, %15 : !kgen.pointer<index>
      %16 = pop.load %14 : !kgen.pointer<union<struct<()>, index>>
      %17 = pop.stack_allocation 1 x union<struct<()>, index>
      %18 = pop.union.bitcast %17 : <union<struct<()>, index>> as <index>
      pop.store %idx64, %18 : !kgen.pointer<index>
      %19 = pop.load %17 : !kgen.pointer<union<struct<()>, index>>
      %20 = pop.stack_allocation 1 x union<struct<()>, index>
      %21 = pop.union.bitcast %20 : <union<struct<()>, index>> as <index>
      pop.store %16, %20 : !kgen.pointer<union<struct<()>, index>>
      %22 = pop.stack_allocation 1 x union<struct<()>, index>
      %23 = pop.union.bitcast %22 : <union<struct<()>, index>> as <index>
      %24 = pop.load %21 : !kgen.pointer<index>
      pop.store %24, %23 : !kgen.pointer<index>
      %25 = pop.load %22 : !kgen.pointer<union<struct<()>, index>>
      %26 = kgen.struct.create(%25, %simd_2) : !kgen.struct<(union<struct<()>, index>, scalar<ui8>)>
      %27 = kgen.struct.create(%26) : !kgen.struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>
      %28 = kgen.struct.create(%27) : !kgen.struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>
      %29 = pop.stack_allocation 1 x union<struct<()>, index>
      %30 = pop.union.bitcast %29 : <union<struct<()>, index>> as <index>
      pop.store %19, %29 : !kgen.pointer<union<struct<()>, index>>
      %31 = pop.stack_allocation 1 x union<struct<()>, index>
      %32 = pop.union.bitcast %31 : <union<struct<()>, index>> as <index>
      %33 = pop.load %30 : !kgen.pointer<index>
      pop.store %33, %32 : !kgen.pointer<index>
      %34 = pop.load %31 : !kgen.pointer<union<struct<()>, index>>
      %35 = kgen.struct.create(%34, %simd_2) : !kgen.struct<(union<struct<()>, index>, scalar<ui8>)>
      %36 = kgen.struct.create(%35) : !kgen.struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>
      %37 = kgen.struct.create(%36) : !kgen.struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>
      %38 = kgen.struct.create(%28) : !kgen.struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>
      %39 = kgen.struct.create(%37) : !kgen.struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>
      %40 = kgen.struct.create(%38, %39) : !kgen.struct<(struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>, struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>) memoryOnly>
      %41 = kgen.call @"std::builtin::builtin_slice::ContiguousSlice::indices(::ContiguousSlice,::Int)"(%40, %index65) : (!kgen.struct<(struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>, struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>) memoryOnly>, index) -> !kgen.struct<(struct<(index, index)>)>
      %42 = kgen.struct.extract %41[0] : <(struct<(index, index)>)>
      %43 = kgen.struct.extract %42[0] : <(index, index)>
      %44 = kgen.struct.extract %42[1] : <(index, index)>
      %45 = pop.offset %10[%43] : !kgen.pointer<scalar<ui8>>
      %46 = pop.pointer.bitcast %45 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<none>
      %47 = index.sub %44, %43
      %48 = kgen.struct.create(%46, %47) : !kgen.struct<(pointer<none>, index)>
      kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=1"(%arg0, %48) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      pop.stack_alloc.lifetime.end(%9) : !kgen.pointer<array<65, scalar<ui8>>>
      hlcf.yield
    }
    kgen.return
  }
  kgen.func @"std::builtin::format_int::_write_int[::DType,::Writer,::Int,::StringSlice[::Bool(False), StaticConstantOrigin, *?]]($1&,::SIMD[$0, ::Int(1)],prefix:::StringSlice[::Bool(False), StaticConstantOrigin, *?]),dtype=ui64,W=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>],radix=10,digit_chars={ #interp.memref<{[(#interp.memory_handle<16, \220123456789abcdefghijklmnopqrstuvwxyz\\00\22 string>, const_global, [], [])], []}, 0, 0>, 36 }"(%arg0: !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, %arg1: !pop.scalar<ui64>, %arg2: !kgen.struct<(pointer<none>, index)>) {
    %simd = kgen.param.constant: scalar<ui64> = <10>
    %simd_0 = kgen.param.constant: scalar<ui8> = <1>
    %idx63 = index.constant 63
    %idx64 = index.constant 64
    %index65 = kgen.param.constant = <65>
    %pointer = kgen.param.constant: pointer<none> = <#interp.memref<{[(#memory_handle5, const_global, [], [])], []}, 0, 0>>
    %simd_1 = kgen.param.constant: scalar<ui64> = <0>
    %simd_2 = kgen.param.constant: scalar<ui8> = <0>
    %index1 = kgen.param.constant = <1>
    %index64 = kgen.param.constant = <64>
    %0 = pop.pointer.bitcast %pointer : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg0, %arg2) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    %1 = pop.cmp eq(%arg1, %simd_1) : <1, ui64>
    %2 = pop.cast_to_builtin %1 : !pop.scalar<bool> to i1
    hlcf.if %2 {
      %3 = pop.load %0 : !kgen.pointer<scalar<ui8>>
      %4 = pop.stack_allocation 1 x array<2, scalar<ui8>> align 1 marked
      pop.stack_alloc.lifetime.start(%4) : !kgen.pointer<array<2, scalar<ui8>>>
      %5 = pop.pointer.bitcast %4 : !kgen.pointer<array<2, scalar<ui8>>> to !kgen.pointer<scalar<ui8>>
      pop.store %3, %5 : !kgen.pointer<scalar<ui8>>
      %6 = pop.offset %5[%index1] : !kgen.pointer<scalar<ui8>>
      pop.store %simd_2, %6 : !kgen.pointer<scalar<ui8>>
      %7 = pop.pointer.bitcast %4 : !kgen.pointer<array<2, scalar<ui8>>> to !kgen.pointer<none>
      %8 = kgen.struct.create(%7, %index1) : !kgen.struct<(pointer<none>, index)>
      kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=1"(%arg0, %8) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      pop.stack_alloc.lifetime.end(%4) : !kgen.pointer<array<2, scalar<ui8>>>
      hlcf.yield
    } else {
      %3 = pop.stack_allocation 1 x array<65, scalar<ui8>> align 1 marked
      pop.stack_alloc.lifetime.start(%3) : !kgen.pointer<array<65, scalar<ui8>>>
      %4 = pop.pointer.bitcast %3 : !kgen.pointer<array<65, scalar<ui8>>> to !kgen.pointer<scalar<ui8>>
      %5 = pop.offset %4[%index64] : !kgen.pointer<scalar<ui8>>
      pop.store %simd_2, %5 : !kgen.pointer<scalar<ui8>>
      %6 = hlcf.loop "_loop_0" (%arg3 = %idx63 : index, %arg4 = %arg1 : !pop.scalar<ui64>) -> index {
        %43 = pop.cmp ne(%arg4, %simd_1) : <1, ui64>
        %44 = pop.cast_to_builtin %43 : !pop.scalar<bool> to i1
        hlcf.if %44 {
          hlcf.yield
        } else {
          hlcf.break "_loop_0" %arg3 : index
        }
        %45 = pop.rem %arg4, %simd : !pop.scalar<ui64>
        %46 = pop.offset %4[%arg3] : !kgen.pointer<scalar<ui8>>
        %47 = pop.cast fast %45 : !pop.scalar<ui64> to !pop.scalar<index>
        %48 = pop.cast_to_builtin %47 : !pop.scalar<index> to index
        %49 = pop.offset %0[%48] : !kgen.pointer<scalar<ui8>>
        %50 = pop.load %49 : !kgen.pointer<scalar<ui8>>
        pop.store %50, %46 : !kgen.pointer<scalar<ui8>>
        %51 = index.sub %arg3, %index1
        %52 = pop.div %arg4, %simd : !pop.scalar<ui64>
        hlcf.continue "_loop_0" %51, %52 : index, !pop.scalar<ui64>
      }
      %7 = index.add %6, %index1
      %8 = pop.stack_allocation 1 x union<struct<()>, index>
      %9 = pop.union.bitcast %8 : <union<struct<()>, index>> as <index>
      pop.store %7, %9 : !kgen.pointer<index>
      %10 = pop.load %8 : !kgen.pointer<union<struct<()>, index>>
      %11 = pop.stack_allocation 1 x union<struct<()>, index>
      %12 = pop.union.bitcast %11 : <union<struct<()>, index>> as <index>
      pop.store %idx64, %12 : !kgen.pointer<index>
      %13 = pop.load %11 : !kgen.pointer<union<struct<()>, index>>
      %14 = pop.stack_allocation 1 x union<struct<()>, index>
      %15 = pop.union.bitcast %14 : <union<struct<()>, index>> as <index>
      pop.store %10, %14 : !kgen.pointer<union<struct<()>, index>>
      %16 = pop.stack_allocation 1 x union<struct<()>, index>
      %17 = pop.union.bitcast %16 : <union<struct<()>, index>> as <index>
      %18 = pop.load %15 : !kgen.pointer<index>
      pop.store %18, %17 : !kgen.pointer<index>
      %19 = pop.load %16 : !kgen.pointer<union<struct<()>, index>>
      %20 = kgen.struct.create(%19, %simd_0) : !kgen.struct<(union<struct<()>, index>, scalar<ui8>)>
      %21 = kgen.struct.create(%20) : !kgen.struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>
      %22 = kgen.struct.create(%21) : !kgen.struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>
      %23 = pop.stack_allocation 1 x union<struct<()>, index>
      %24 = pop.union.bitcast %23 : <union<struct<()>, index>> as <index>
      pop.store %13, %23 : !kgen.pointer<union<struct<()>, index>>
      %25 = pop.stack_allocation 1 x union<struct<()>, index>
      %26 = pop.union.bitcast %25 : <union<struct<()>, index>> as <index>
      %27 = pop.load %24 : !kgen.pointer<index>
      pop.store %27, %26 : !kgen.pointer<index>
      %28 = pop.load %25 : !kgen.pointer<union<struct<()>, index>>
      %29 = kgen.struct.create(%28, %simd_0) : !kgen.struct<(union<struct<()>, index>, scalar<ui8>)>
      %30 = kgen.struct.create(%29) : !kgen.struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>
      %31 = kgen.struct.create(%30) : !kgen.struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>
      %32 = kgen.struct.create(%22) : !kgen.struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>
      %33 = kgen.struct.create(%31) : !kgen.struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>
      %34 = kgen.struct.create(%32, %33) : !kgen.struct<(struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>, struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>) memoryOnly>
      %35 = kgen.call @"std::builtin::builtin_slice::ContiguousSlice::indices(::ContiguousSlice,::Int)"(%34, %index65) : (!kgen.struct<(struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>, struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>) memoryOnly>, index) -> !kgen.struct<(struct<(index, index)>)>
      %36 = kgen.struct.extract %35[0] : <(struct<(index, index)>)>
      %37 = kgen.struct.extract %36[0] : <(index, index)>
      %38 = kgen.struct.extract %36[1] : <(index, index)>
      %39 = pop.offset %4[%37] : !kgen.pointer<scalar<ui8>>
      %40 = pop.pointer.bitcast %39 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<none>
      %41 = index.sub %38, %37
      %42 = kgen.struct.create(%40, %41) : !kgen.struct<(pointer<none>, index)>
      kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=1"(%arg0, %42) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      pop.stack_alloc.lifetime.end(%3) : !kgen.pointer<array<65, scalar<ui8>>>
      hlcf.yield
    }
    kgen.return
  }
  kgen.func @"std::builtin::format_int::_write_int[::DType,::Writer,::Int,::StringSlice[::Bool(False), StaticConstantOrigin, *?]]($1&,::SIMD[$0, ::Int(1)],prefix:::StringSlice[::Bool(False), StaticConstantOrigin, *?]),dtype=ui8,W=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>],radix=10,digit_chars={ #interp.memref<{[(#interp.memory_handle<16, \220123456789abcdefghijklmnopqrstuvwxyz\\00\22 string>, const_global, [], [])], []}, 0, 0>, 36 }"(%arg0: !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, %arg1: !pop.scalar<ui8>, %arg2: !kgen.struct<(pointer<none>, index)>) {
    %simd = kgen.param.constant: scalar<ui8> = <10>
    %simd_0 = kgen.param.constant: scalar<ui8> = <1>
    %idx63 = index.constant 63
    %idx64 = index.constant 64
    %index65 = kgen.param.constant = <65>
    %pointer = kgen.param.constant: pointer<none> = <#interp.memref<{[(#memory_handle5, const_global, [], [])], []}, 0, 0>>
    %simd_1 = kgen.param.constant: scalar<ui8> = <0>
    %index1 = kgen.param.constant = <1>
    %index64 = kgen.param.constant = <64>
    %0 = pop.pointer.bitcast %pointer : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg0, %arg2) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    %1 = pop.cmp eq(%arg1, %simd_1) : <1, ui8>
    %2 = pop.cast_to_builtin %1 : !pop.scalar<bool> to i1
    hlcf.if %2 {
      %3 = pop.load %0 : !kgen.pointer<scalar<ui8>>
      %4 = pop.stack_allocation 1 x array<2, scalar<ui8>> align 1 marked
      pop.stack_alloc.lifetime.start(%4) : !kgen.pointer<array<2, scalar<ui8>>>
      %5 = pop.pointer.bitcast %4 : !kgen.pointer<array<2, scalar<ui8>>> to !kgen.pointer<scalar<ui8>>
      pop.store %3, %5 : !kgen.pointer<scalar<ui8>>
      %6 = pop.offset %5[%index1] : !kgen.pointer<scalar<ui8>>
      pop.store %simd_1, %6 : !kgen.pointer<scalar<ui8>>
      %7 = pop.pointer.bitcast %4 : !kgen.pointer<array<2, scalar<ui8>>> to !kgen.pointer<none>
      %8 = kgen.struct.create(%7, %index1) : !kgen.struct<(pointer<none>, index)>
      kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=1"(%arg0, %8) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      pop.stack_alloc.lifetime.end(%4) : !kgen.pointer<array<2, scalar<ui8>>>
      hlcf.yield
    } else {
      %3 = pop.stack_allocation 1 x array<65, scalar<ui8>> align 1 marked
      pop.stack_alloc.lifetime.start(%3) : !kgen.pointer<array<65, scalar<ui8>>>
      %4 = pop.pointer.bitcast %3 : !kgen.pointer<array<65, scalar<ui8>>> to !kgen.pointer<scalar<ui8>>
      %5 = pop.offset %4[%index64] : !kgen.pointer<scalar<ui8>>
      pop.store %simd_1, %5 : !kgen.pointer<scalar<ui8>>
      %6 = hlcf.loop "_loop_0" (%arg3 = %idx63 : index, %arg4 = %arg1 : !pop.scalar<ui8>) -> index {
        %43 = pop.cmp ne(%arg4, %simd_1) : <1, ui8>
        %44 = pop.cast_to_builtin %43 : !pop.scalar<bool> to i1
        hlcf.if %44 {
          hlcf.yield
        } else {
          hlcf.break "_loop_0" %arg3 : index
        }
        %45 = pop.rem %arg4, %simd : !pop.scalar<ui8>
        %46 = pop.offset %4[%arg3] : !kgen.pointer<scalar<ui8>>
        %47 = pop.cast %45 : !pop.scalar<ui8> to !pop.scalar<index>
        %48 = pop.cast_to_builtin %47 : !pop.scalar<index> to index
        %49 = pop.offset %0[%48] : !kgen.pointer<scalar<ui8>>
        %50 = pop.load %49 : !kgen.pointer<scalar<ui8>>
        pop.store %50, %46 : !kgen.pointer<scalar<ui8>>
        %51 = index.sub %arg3, %index1
        %52 = pop.div %arg4, %simd : !pop.scalar<ui8>
        hlcf.continue "_loop_0" %51, %52 : index, !pop.scalar<ui8>
      }
      %7 = index.add %6, %index1
      %8 = pop.stack_allocation 1 x union<struct<()>, index>
      %9 = pop.union.bitcast %8 : <union<struct<()>, index>> as <index>
      pop.store %7, %9 : !kgen.pointer<index>
      %10 = pop.load %8 : !kgen.pointer<union<struct<()>, index>>
      %11 = pop.stack_allocation 1 x union<struct<()>, index>
      %12 = pop.union.bitcast %11 : <union<struct<()>, index>> as <index>
      pop.store %idx64, %12 : !kgen.pointer<index>
      %13 = pop.load %11 : !kgen.pointer<union<struct<()>, index>>
      %14 = pop.stack_allocation 1 x union<struct<()>, index>
      %15 = pop.union.bitcast %14 : <union<struct<()>, index>> as <index>
      pop.store %10, %14 : !kgen.pointer<union<struct<()>, index>>
      %16 = pop.stack_allocation 1 x union<struct<()>, index>
      %17 = pop.union.bitcast %16 : <union<struct<()>, index>> as <index>
      %18 = pop.load %15 : !kgen.pointer<index>
      pop.store %18, %17 : !kgen.pointer<index>
      %19 = pop.load %16 : !kgen.pointer<union<struct<()>, index>>
      %20 = kgen.struct.create(%19, %simd_0) : !kgen.struct<(union<struct<()>, index>, scalar<ui8>)>
      %21 = kgen.struct.create(%20) : !kgen.struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>
      %22 = kgen.struct.create(%21) : !kgen.struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>
      %23 = pop.stack_allocation 1 x union<struct<()>, index>
      %24 = pop.union.bitcast %23 : <union<struct<()>, index>> as <index>
      pop.store %13, %23 : !kgen.pointer<union<struct<()>, index>>
      %25 = pop.stack_allocation 1 x union<struct<()>, index>
      %26 = pop.union.bitcast %25 : <union<struct<()>, index>> as <index>
      %27 = pop.load %24 : !kgen.pointer<index>
      pop.store %27, %26 : !kgen.pointer<index>
      %28 = pop.load %25 : !kgen.pointer<union<struct<()>, index>>
      %29 = kgen.struct.create(%28, %simd_0) : !kgen.struct<(union<struct<()>, index>, scalar<ui8>)>
      %30 = kgen.struct.create(%29) : !kgen.struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>
      %31 = kgen.struct.create(%30) : !kgen.struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>
      %32 = kgen.struct.create(%22) : !kgen.struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>
      %33 = kgen.struct.create(%31) : !kgen.struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>
      %34 = kgen.struct.create(%32, %33) : !kgen.struct<(struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>, struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>) memoryOnly>
      %35 = kgen.call @"std::builtin::builtin_slice::ContiguousSlice::indices(::ContiguousSlice,::Int)"(%34, %index65) : (!kgen.struct<(struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>, struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>) memoryOnly>, index) -> !kgen.struct<(struct<(index, index)>)>
      %36 = kgen.struct.extract %35[0] : <(struct<(index, index)>)>
      %37 = kgen.struct.extract %36[0] : <(index, index)>
      %38 = kgen.struct.extract %36[1] : <(index, index)>
      %39 = pop.offset %4[%37] : !kgen.pointer<scalar<ui8>>
      %40 = pop.pointer.bitcast %39 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<none>
      %41 = index.sub %38, %37
      %42 = kgen.struct.create(%40, %41) : !kgen.struct<(pointer<none>, index)>
      kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=1"(%arg0, %42) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      pop.stack_alloc.lifetime.end(%3) : !kgen.pointer<array<65, scalar<ui8>>>
      hlcf.yield
    }
    kgen.return
  }
  kgen.func @"std::builtin::simd::_write_scalar[::DType,::Writer]($1&,::SIMD[$0, ::Int(1)]),dtype=si64,W=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%arg0: !kgen.struct<(pointer<none>, index) memoryOnly>, %arg1: !pop.scalar<si64>) -> !kgen.struct<(pointer<none>, index) memoryOnly> {
    %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle3, const_global, [], [])], []}, 0, 0>, 0 }>
    %0 = kgen.call tail @"std::builtin::format_int::_write_int[::DType,::Writer,::Int,::StringSlice[::Bool(False), StaticConstantOrigin, *?]]($1&,::SIMD[$0, ::Int(1)],prefix:::StringSlice[::Bool(False), StaticConstantOrigin, *?]),dtype=si64,W=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>],radix=10,digit_chars={ #interp.memref<{[(#interp.memory_handle<16, \220123456789abcdefghijklmnopqrstuvwxyz\\00\22 string>, const_global, [], [])], []}, 0, 0>, 36 }"(%arg0, %arg1, %struct) : (!kgen.struct<(pointer<none>, index) memoryOnly>, !pop.scalar<si64>, !kgen.struct<(pointer<none>, index)>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
    kgen.return %0 : !kgen.struct<(pointer<none>, index) memoryOnly>
  }
  kgen.func @"std::builtin::simd::_write_scalar[::DType,::Writer]($1&,::SIMD[$0, ::Int(1)]),dtype=si64,W=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>]"(%arg0: !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, %arg1: !pop.scalar<si64>) {
    %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle3, const_global, [], [])], []}, 0, 0>, 0 }>
    kgen.call tail @"std::builtin::format_int::_write_int[::DType,::Writer,::Int,::StringSlice[::Bool(False), StaticConstantOrigin, *?]]($1&,::SIMD[$0, ::Int(1)],prefix:::StringSlice[::Bool(False), StaticConstantOrigin, *?]),dtype=si64,W=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>],radix=10,digit_chars={ #interp.memref<{[(#interp.memory_handle<16, \220123456789abcdefghijklmnopqrstuvwxyz\\00\22 string>, const_global, [], [])], []}, 0, 0>, 36 }"(%arg0, %arg1, %struct) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !pop.scalar<si64>, !kgen.struct<(pointer<none>, index)>) -> ()
    kgen.return
  }
  kgen.func @"std::format::__init__::Writer::write[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $3]]($0&,*$1),_Self`=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>],Ts.values`2x=[[typevalue<#kgen.instref<\1B\22std::builtin::simd::SIMD,dtype=si64,size=1\22>>, scalar<si64>]]"(%arg0: !kgen.struct<(pointer<none>, index) memoryOnly>, %arg1: !pop.scalar<si64>) -> !kgen.struct<(pointer<none>, index) memoryOnly> {
    %0 = kgen.call @"std::builtin::simd::SIMD::write_to[::Writer](::SIMD[$0, $1],$2&),dtype=si64,size=1,writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferHeap\22>>, struct<(pointer<none>, index) memoryOnly>]"(%arg1, %arg0) : (!pop.scalar<si64>, !kgen.struct<(pointer<none>, index) memoryOnly>) -> !kgen.struct<(pointer<none>, index) memoryOnly>
    kgen.return %0 : !kgen.struct<(pointer<none>, index) memoryOnly>
  }
  kgen.func @"std::format::__init__::Writer::write[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $3]]($0&,*$1),_Self`=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>],Ts.values`2x=[[typevalue<#kgen.instref<\1B\22std::builtin::simd::SIMD,dtype=si64,size=1\22>>, scalar<si64>]]"(%arg0: !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, %arg1: !pop.scalar<si64>) {
    kgen.call @"std::builtin::simd::SIMD::write_to[::Writer](::SIMD[$0, $1],$2&),dtype=si64,size=1,writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>]"(%arg1, %arg0) : (!pop.scalar<si64>, !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut) -> ()
    kgen.return
  }
  kgen.func @"std::format::__init__::Writer::write[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $3]]($0&,*$1),_Self`=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>],Ts.values`2x=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::builtin::int::Int\22>>, index], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::builtin::int::Int\22>>, index]]"(%arg0: !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, %arg1: !kgen.struct<(pointer<none>, index)>, %arg2: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, %arg3: index, %arg4: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, %arg5: index) {
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %index0 = kgen.param.constant = <0>
    %index56 = kgen.param.constant = <56>
    %index2233785415175766016 = kgen.param.constant = <2233785415175766016>
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg0, %arg1) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    %0 = kgen.struct.gep %arg2[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %1 = pop.pointer.bitcast %arg2 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
    %2 = kgen.struct.gep %arg2[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %3 = pop.load %2 : !kgen.pointer<index>
    %4 = index.and %3, %index-9223372036854775808
    %5 = index.cmp ne(%4, %index0)
    %6 = hlcf.if %5 -> !kgen.pointer<none> {
      hlcf.yield %1 : !kgen.pointer<none>
    } else {
      %26 = pop.load %0 : !kgen.pointer<pointer<none>>
      hlcf.yield %26 : !kgen.pointer<none>
    }
    %7 = kgen.struct.gep %arg2[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %8 = pop.load %2 : !kgen.pointer<index>
    %9 = index.and %8, %index-9223372036854775808
    %10 = index.cmp ne(%9, %index0)
    %11 = hlcf.if %10 -> index {
      %26 = pop.load %2 : !kgen.pointer<index>
      %27 = index.and %26, %index2233785415175766016
      %28 = index.shrs %27, %index56
      hlcf.yield %28 : index
    } else {
      %26 = pop.load %7 : !kgen.pointer<index>
      hlcf.yield %26 : index
    }
    %12 = kgen.struct.create(%6, %11) : !kgen.struct<(pointer<none>, index)>
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg0, %12) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    kgen.call @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>]"(%arg3, %arg0) : (index, !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut) -> ()
    %13 = kgen.struct.gep %arg4[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %14 = pop.pointer.bitcast %arg4 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
    %15 = kgen.struct.gep %arg4[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %16 = pop.load %15 : !kgen.pointer<index>
    %17 = index.and %16, %index-9223372036854775808
    %18 = index.cmp ne(%17, %index0)
    %19 = hlcf.if %18 -> !kgen.pointer<none> {
      hlcf.yield %14 : !kgen.pointer<none>
    } else {
      %26 = pop.load %13 : !kgen.pointer<pointer<none>>
      hlcf.yield %26 : !kgen.pointer<none>
    }
    %20 = kgen.struct.gep %arg4[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %21 = pop.load %15 : !kgen.pointer<index>
    %22 = index.and %21, %index-9223372036854775808
    %23 = index.cmp ne(%22, %index0)
    %24 = hlcf.if %23 -> index {
      %26 = pop.load %15 : !kgen.pointer<index>
      %27 = index.and %26, %index2233785415175766016
      %28 = index.shrs %27, %index56
      hlcf.yield %28 : index
    } else {
      %26 = pop.load %20 : !kgen.pointer<index>
      hlcf.yield %26 : index
    }
    %25 = kgen.struct.create(%19, %24) : !kgen.struct<(pointer<none>, index)>
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg0, %25) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    kgen.call @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>]"(%arg5, %arg0) : (index, !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut) -> ()
    kgen.return
  }
  kgen.func @"std::io::io::_flush(::FileDescriptor)"(%arg0: index) no_inline {
    %pointer = kgen.param.constant: pointer<none> = <#interp.memref<{[(#memory_handle6, const_global, [], [])], []}, 0, 0>>
    %pointer_0 = kgen.param.constant: pointer<none> = <0>
    %index0 = kgen.param.constant = <0>
    %index-1 = kgen.param.constant = <-1>
    %0 = pop.cast_from_builtin %arg0 : index to !pop.scalar<index>
    %1 = pop.cast %0 : !pop.scalar<index> to !pop.scalar<si32>
    %2 = pop.external_call @dup(%1) : (!pop.scalar<si32>) -> !pop.scalar<si32>
    %3 = pop.external_call @fdopen(%2, %pointer) : (!pop.scalar<si32>, !kgen.pointer<none>) -> !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>
    %4 = kgen.struct.extract %3[0] : <(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>
    %5 = kgen.struct.extract %4[0] : <(struct<(struct<(pointer<none>) memoryOnly>)>)>
    %6 = kgen.struct.extract %5[0] : <(struct<(pointer<none>) memoryOnly>)>
    %7 = kgen.struct.extract %6[0] : <(pointer<none>) memoryOnly>
    %8 = pop.stack_allocation 1 x pointer<none>
    %9 = pop.pointer.bitcast %8 : !kgen.pointer<pointer<none>> to !kgen.pointer<struct<(array<1, pointer<none>>)>>
    %10 = kgen.struct.gep %9[0] : <struct<(array<1, pointer<none>>)>>
    %11 = pop.pointer.bitcast %10 : !kgen.pointer<array<1, pointer<none>>> to !kgen.pointer<pointer<none>>
    %12 = pop.pointer_to_index %7 : <none>
    %13 = index.cmp eq(%12, %index0)
    %14 = pop.select %13, %index0, %index-1 : index
    %15 = index.cmp eq(%14, %index-1)
    hlcf.if %15 {
      pop.store %7, %11 : !kgen.pointer<pointer<none>>
      hlcf.yield
    } else {
      pop.store %pointer_0, %8 : !kgen.pointer<pointer<none>>
      hlcf.yield
    }
    %16 = pop.load %8 : !kgen.pointer<pointer<none>>
    %17 = kgen.struct.create(%16) : !kgen.struct<(pointer<none>) memoryOnly>
    %18 = kgen.struct.create(%17) : !kgen.struct<(struct<(pointer<none>) memoryOnly>)>
    %19 = kgen.struct.create(%18) : !kgen.struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>
    %20 = kgen.struct.create(%19) : !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>
    %21 = pop.external_call @fflush(%20) : (!kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>) -> !pop.scalar<si32>
    %22 = pop.external_call @fclose(%3) : (!kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>) -> !pop.scalar<si32>
    kgen.return
  }
  kgen.func @"std::io::io::_printf[KGENParamList[::AnyType],::StringSlice[::Bool(False), StaticConstantOrigin, *?],*::AnyType,LITImmutOrigin,::Origin[::Bool(False), $3]](*$0,file:::FileDescriptor),types.values`=[],fmt={ #interp.memref<{[(#interp.memory_handle<16, \22HEAP_BUFFER_BYTES exceeded, increase with: `mojo -D HEAP_BUFFER_BYTES=4096`\\0A\\00\22 string>, const_global, [], [])], []}, 0, 0>, 76 }"(%arg0: index) no_inline {
    %index-1 = kgen.param.constant = <-1>
    %index0 = kgen.param.constant = <0>
    %pointer = kgen.param.constant: pointer<none> = <0>
    %pointer_0 = kgen.param.constant: pointer<none> = <#interp.memref<{[(#memory_handle6, const_global, [], [])], []}, 0, 0>>
    %string = kgen.param.constant: string = <"HEAP_BUFFER_BYTES exceeded, increase with: `mojo -D HEAP_BUFFER_BYTES=4096`\0A">
    %0 = pop.cast_from_builtin %arg0 : index to !pop.scalar<index>
    %1 = pop.cast %0 : !pop.scalar<index> to !pop.scalar<si32>
    %2 = pop.external_call @dup(%1) : (!pop.scalar<si32>) -> !pop.scalar<si32>
    %3 = pop.external_call @fdopen(%2, %pointer_0) : (!pop.scalar<si32>, !kgen.pointer<none>) -> !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>
    %4 = kgen.struct.extract %3[0] : <(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>
    %5 = kgen.struct.extract %4[0] : <(struct<(struct<(pointer<none>) memoryOnly>)>)>
    %6 = kgen.struct.extract %5[0] : <(struct<(pointer<none>) memoryOnly>)>
    %7 = kgen.struct.extract %6[0] : <(pointer<none>) memoryOnly>
    %8 = pop.stack_allocation 1 x pointer<none>
    %9 = pop.pointer.bitcast %8 : !kgen.pointer<pointer<none>> to !kgen.pointer<struct<(array<1, pointer<none>>)>>
    %10 = kgen.struct.gep %9[0] : <struct<(array<1, pointer<none>>)>>
    %11 = pop.pointer.bitcast %10 : !kgen.pointer<array<1, pointer<none>>> to !kgen.pointer<pointer<none>>
    %12 = pop.pointer_to_index %7 : <none>
    %13 = index.cmp eq(%12, %index0)
    %14 = pop.select %13, %index0, %index-1 : index
    %15 = index.cmp eq(%14, %index-1)
    hlcf.if %15 {
      pop.store %7, %11 : !kgen.pointer<pointer<none>>
      hlcf.yield
    } else {
      pop.store %pointer, %8 : !kgen.pointer<pointer<none>>
      hlcf.yield
    }
    %16 = pop.load %8 : !kgen.pointer<pointer<none>>
    %17 = kgen.struct.create(%16) : !kgen.struct<(pointer<none>) memoryOnly>
    %18 = kgen.struct.create(%17) : !kgen.struct<(struct<(pointer<none>) memoryOnly>)>
    %19 = kgen.struct.create(%18) : !kgen.struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>
    %20 = kgen.struct.create(%19) : !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>
    %21 = pop.string.address %string
    %22 = pop.pointer.bitcast %21 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %23 = pop.external_call @KGEN_CompilerRT_fprintf(%20, %22) (!kgen.pointer<none>, !kgen.pointer<scalar<si8>>) -> !pop.scalar<si32> : (!kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>, !kgen.pointer<none>) -> !pop.scalar<si32>
    %24 = pop.external_call @fclose(%3) : (!kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>) -> !pop.scalar<si32>
    kgen.return
  }
  kgen.func @"std::io::io::_printf[KGENParamList[::AnyType],::StringSlice[::Bool(False), StaticConstantOrigin, *?],*::AnyType,LITImmutOrigin,::Origin[::Bool(False), $3]](*$0,file:::FileDescriptor),types.values`=[[typevalue<#kgen.instref<\1B\22std::memory::unsafe_pointer::UnsafePointer,mut=0,origin._mlir_origin`={  },type=[typevalue<#kgen.instref<\\1B\\22std::builtin::simd::SIMD,dtype=ui8,size=1\\22>>, scalar<ui8>],origin={  },address_space=0\22>>, pointer<none>], [typevalue<#kgen.instref<\1B\22std::builtin::int::Int\22>>, index], [typevalue<#kgen.instref<\1B\22std::builtin::int::Int\22>>, index], [typevalue<#kgen.instref<\1B\22std::memory::unsafe_pointer::UnsafePointer,mut=0,origin._mlir_origin`={  },type=[typevalue<#kgen.instref<\\1B\\22std::builtin::simd::SIMD,dtype=ui8,size=1\\22>>, scalar<ui8>],origin={  },address_space=0\22>>, pointer<none>]],fmt={ #interp.memref<{[(#interp.memory_handle<16, \22At: %s:%llu:%llu: Assert Error: %s\\0A\\00\22 string>, const_global, [], [])], []}, 0, 0>, 35 }"(%arg0: !kgen.pointer<none>, %arg1: index, %arg2: index, %arg3: !kgen.pointer<none>, %arg4: index) no_inline {
    %index-1 = kgen.param.constant = <-1>
    %index0 = kgen.param.constant = <0>
    %pointer = kgen.param.constant: pointer<none> = <0>
    %pointer_0 = kgen.param.constant: pointer<none> = <#interp.memref<{[(#memory_handle6, const_global, [], [])], []}, 0, 0>>
    %string = kgen.param.constant: string = <"At: %s:%llu:%llu: Assert Error: %s\0A">
    %0 = pop.cast_from_builtin %arg4 : index to !pop.scalar<index>
    %1 = pop.cast %0 : !pop.scalar<index> to !pop.scalar<si32>
    %2 = pop.external_call @dup(%1) : (!pop.scalar<si32>) -> !pop.scalar<si32>
    %3 = pop.external_call @fdopen(%2, %pointer_0) : (!pop.scalar<si32>, !kgen.pointer<none>) -> !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>
    %4 = kgen.struct.extract %3[0] : <(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>
    %5 = kgen.struct.extract %4[0] : <(struct<(struct<(pointer<none>) memoryOnly>)>)>
    %6 = kgen.struct.extract %5[0] : <(struct<(pointer<none>) memoryOnly>)>
    %7 = kgen.struct.extract %6[0] : <(pointer<none>) memoryOnly>
    %8 = pop.stack_allocation 1 x pointer<none>
    %9 = pop.pointer.bitcast %8 : !kgen.pointer<pointer<none>> to !kgen.pointer<struct<(array<1, pointer<none>>)>>
    %10 = kgen.struct.gep %9[0] : <struct<(array<1, pointer<none>>)>>
    %11 = pop.pointer.bitcast %10 : !kgen.pointer<array<1, pointer<none>>> to !kgen.pointer<pointer<none>>
    %12 = pop.pointer_to_index %7 : <none>
    %13 = index.cmp eq(%12, %index0)
    %14 = pop.select %13, %index0, %index-1 : index
    %15 = index.cmp eq(%14, %index-1)
    hlcf.if %15 {
      pop.store %7, %11 : !kgen.pointer<pointer<none>>
      hlcf.yield
    } else {
      pop.store %pointer, %8 : !kgen.pointer<pointer<none>>
      hlcf.yield
    }
    %16 = pop.load %8 : !kgen.pointer<pointer<none>>
    %17 = kgen.struct.create(%16) : !kgen.struct<(pointer<none>) memoryOnly>
    %18 = kgen.struct.create(%17) : !kgen.struct<(struct<(pointer<none>) memoryOnly>)>
    %19 = kgen.struct.create(%18) : !kgen.struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>
    %20 = kgen.struct.create(%19) : !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>
    %21 = pop.string.address %string
    %22 = pop.pointer.bitcast %21 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %23 = pop.external_call @KGEN_CompilerRT_fprintf(%20, %22, %arg0, %arg1, %arg2, %arg3) (!kgen.pointer<none>, !kgen.pointer<scalar<si8>>) -> !pop.scalar<si32> : (!kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>, !kgen.pointer<none>, !kgen.pointer<none>, index, index, !kgen.pointer<none>) -> !pop.scalar<si32>
    %24 = pop.external_call @fclose(%3) : (!kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>) -> !pop.scalar<si32>
    kgen.return
  }
  kgen.func @"std::io::io::print[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?],flush:::Bool,file:::FileDescriptor$),Ts.values`=[[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::builtin::simd::SIMD,dtype=f64,size=1\22>>, scalar<f64>]]"(%arg0: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, %arg1: !pop.scalar<f64>, %arg2: !kgen.struct<(pointer<none>, index)>, %arg3: !kgen.struct<(pointer<none>, index)>, %arg4: i1, %arg5: index owned) no_inline {
    %index0 = kgen.param.constant = <0>
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %index56 = kgen.param.constant = <56>
    %index2233785415175766016 = kgen.param.constant = <2233785415175766016>
    %0 = pop.stack_allocation 1 x index
    pop.store %arg5, %0 : !kgen.pointer<index>
    %1 = pop.stack_allocation 1 x struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly> align 1 marked
    pop.stack_alloc.lifetime.start(%1) : !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>>
    %2 = kgen.struct.gep %1[1] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>>
    pop.store %index0, %2 : !kgen.pointer<index>
    %3 = kgen.struct.gep %1[2] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>>
    pop.store %0, %3 : !kgen.pointer<pointer<index>>
    %4 = kgen.struct.gep %arg0[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %5 = pop.pointer.bitcast %arg0 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
    %6 = kgen.struct.gep %arg0[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %7 = pop.load %6 : !kgen.pointer<index>
    %8 = index.and %7, %index-9223372036854775808
    %9 = index.cmp ne(%8, %index0)
    %10 = hlcf.if %9 -> !kgen.pointer<none> {
      hlcf.yield %5 : !kgen.pointer<none>
    } else {
      %24 = pop.load %4 : !kgen.pointer<pointer<none>>
      hlcf.yield %24 : !kgen.pointer<none>
    }
    %11 = kgen.struct.gep %arg0[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %12 = pop.load %6 : !kgen.pointer<index>
    %13 = index.and %12, %index-9223372036854775808
    %14 = index.cmp ne(%13, %index0)
    %15 = hlcf.if %14 -> index {
      %24 = pop.load %6 : !kgen.pointer<index>
      %25 = index.and %24, %index2233785415175766016
      %26 = index.shrs %25, %index56
      hlcf.yield %26 : index
    } else {
      %24 = pop.load %11 : !kgen.pointer<index>
      hlcf.yield %24 : index
    }
    %16 = kgen.struct.create(%10, %15) : !kgen.struct<(pointer<none>, index)>
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%1, %16) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%1, %arg2) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    kgen.call @"std::builtin::simd::SIMD::write_to[::Writer](::SIMD[$0, $1],$2&),dtype=f64,size=1,writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>]"(%arg1, %1) : (!pop.scalar<f64>, !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut) -> ()
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%1, %arg3) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    %17 = pop.load %3 : !kgen.pointer<pointer<index>>
    %18 = kgen.struct.gep %1[0] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>>
    %19 = kgen.struct.gep %18[0] : <struct<(array<4096, scalar<ui8>>) memoryOnly>>
    %20 = pop.pointer.bitcast %19 : !kgen.pointer<array<4096, scalar<ui8>>> to !kgen.pointer<none>
    %21 = pop.load %2 : !kgen.pointer<index>
    %22 = pop.load %17 : !kgen.pointer<index>
    %23 = pop.external_call @write(%22, %20, %21) : (index, !kgen.pointer<none>, index) -> index
    pop.store %index0, %2 : !kgen.pointer<index>
    pop.stack_alloc.lifetime.end(%1) : !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>>
    hlcf.if %arg4 {
      %24 = pop.load %0 : !kgen.pointer<index>
      kgen.call tail @"std::io::io::_flush(::FileDescriptor)"(%24) : (index) -> ()
      hlcf.yield
    } else {
      hlcf.yield
    }
    kgen.return
  }
  kgen.func @"std::io::io::print[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?],flush:::Bool,file:::FileDescriptor$),Ts.values`=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::reflection::location::SourceLocation\22>>, struct<(index, index, struct<(pointer<none>, index)>)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]]"(%arg0: !kgen.struct<(pointer<none>, index)>, %arg1: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, %arg2: !kgen.struct<(index, index, struct<(pointer<none>, index)>)>, %arg3: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, %arg4: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, %arg5: !kgen.struct<(pointer<none>, index)>, %arg6: !kgen.struct<(pointer<none>, index)>, %arg7: i1, %arg8: index owned) no_inline {
    %index0 = kgen.param.constant = <0>
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %index56 = kgen.param.constant = <56>
    %index2233785415175766016 = kgen.param.constant = <2233785415175766016>
    %0 = pop.stack_allocation 1 x index
    pop.store %arg8, %0 : !kgen.pointer<index>
    %1 = pop.stack_allocation 1 x struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly> align 1 marked
    pop.stack_alloc.lifetime.start(%1) : !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>>
    %2 = kgen.struct.gep %1[1] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>>
    pop.store %index0, %2 : !kgen.pointer<index>
    %3 = kgen.struct.gep %1[2] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>>
    pop.store %0, %3 : !kgen.pointer<pointer<index>>
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%1, %arg0) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%1, %arg5) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    %4 = kgen.struct.gep %arg1[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %5 = pop.pointer.bitcast %arg1 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
    %6 = kgen.struct.gep %arg1[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %7 = pop.load %6 : !kgen.pointer<index>
    %8 = index.and %7, %index-9223372036854775808
    %9 = index.cmp ne(%8, %index0)
    %10 = hlcf.if %9 -> !kgen.pointer<none> {
      hlcf.yield %5 : !kgen.pointer<none>
    } else {
      %50 = pop.load %4 : !kgen.pointer<pointer<none>>
      hlcf.yield %50 : !kgen.pointer<none>
    }
    %11 = kgen.struct.gep %arg1[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %12 = pop.load %6 : !kgen.pointer<index>
    %13 = index.and %12, %index-9223372036854775808
    %14 = index.cmp ne(%13, %index0)
    %15 = hlcf.if %14 -> index {
      %50 = pop.load %6 : !kgen.pointer<index>
      %51 = index.and %50, %index2233785415175766016
      %52 = index.shrs %51, %index56
      hlcf.yield %52 : index
    } else {
      %50 = pop.load %11 : !kgen.pointer<index>
      hlcf.yield %50 : index
    }
    %16 = kgen.struct.create(%10, %15) : !kgen.struct<(pointer<none>, index)>
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%1, %16) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%1, %arg5) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    kgen.call @"std::reflection::location::SourceLocation::write_to[::Writer](::SourceLocation,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>]"(%arg2, %1) : (!kgen.struct<(index, index, struct<(pointer<none>, index)>)>, !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut) -> ()
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%1, %arg5) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    %17 = kgen.struct.gep %arg3[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %18 = pop.pointer.bitcast %arg3 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
    %19 = kgen.struct.gep %arg3[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %20 = pop.load %19 : !kgen.pointer<index>
    %21 = index.and %20, %index-9223372036854775808
    %22 = index.cmp ne(%21, %index0)
    %23 = hlcf.if %22 -> !kgen.pointer<none> {
      hlcf.yield %18 : !kgen.pointer<none>
    } else {
      %50 = pop.load %17 : !kgen.pointer<pointer<none>>
      hlcf.yield %50 : !kgen.pointer<none>
    }
    %24 = kgen.struct.gep %arg3[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %25 = pop.load %19 : !kgen.pointer<index>
    %26 = index.and %25, %index-9223372036854775808
    %27 = index.cmp ne(%26, %index0)
    %28 = hlcf.if %27 -> index {
      %50 = pop.load %19 : !kgen.pointer<index>
      %51 = index.and %50, %index2233785415175766016
      %52 = index.shrs %51, %index56
      hlcf.yield %52 : index
    } else {
      %50 = pop.load %24 : !kgen.pointer<index>
      hlcf.yield %50 : index
    }
    %29 = kgen.struct.create(%23, %28) : !kgen.struct<(pointer<none>, index)>
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%1, %29) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%1, %arg5) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    %30 = kgen.struct.gep %arg4[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %31 = pop.pointer.bitcast %arg4 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
    %32 = kgen.struct.gep %arg4[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %33 = pop.load %32 : !kgen.pointer<index>
    %34 = index.and %33, %index-9223372036854775808
    %35 = index.cmp ne(%34, %index0)
    %36 = hlcf.if %35 -> !kgen.pointer<none> {
      hlcf.yield %31 : !kgen.pointer<none>
    } else {
      %50 = pop.load %30 : !kgen.pointer<pointer<none>>
      hlcf.yield %50 : !kgen.pointer<none>
    }
    %37 = kgen.struct.gep %arg4[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %38 = pop.load %32 : !kgen.pointer<index>
    %39 = index.and %38, %index-9223372036854775808
    %40 = index.cmp ne(%39, %index0)
    %41 = hlcf.if %40 -> index {
      %50 = pop.load %32 : !kgen.pointer<index>
      %51 = index.and %50, %index2233785415175766016
      %52 = index.shrs %51, %index56
      hlcf.yield %52 : index
    } else {
      %50 = pop.load %37 : !kgen.pointer<index>
      hlcf.yield %50 : index
    }
    %42 = kgen.struct.create(%36, %41) : !kgen.struct<(pointer<none>, index)>
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%1, %42) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%1, %arg6) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    %43 = pop.load %3 : !kgen.pointer<pointer<index>>
    %44 = kgen.struct.gep %1[0] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>>
    %45 = kgen.struct.gep %44[0] : <struct<(array<4096, scalar<ui8>>) memoryOnly>>
    %46 = pop.pointer.bitcast %45 : !kgen.pointer<array<4096, scalar<ui8>>> to !kgen.pointer<none>
    %47 = pop.load %2 : !kgen.pointer<index>
    %48 = pop.load %43 : !kgen.pointer<index>
    %49 = pop.external_call @write(%48, %46, %47) : (index, !kgen.pointer<none>, index) -> index
    pop.store %index0, %2 : !kgen.pointer<index>
    pop.stack_alloc.lifetime.end(%1) : !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>>
    hlcf.if %arg7 {
      %50 = pop.load %0 : !kgen.pointer<index>
      kgen.call tail @"std::io::io::_flush(::FileDescriptor)"(%50) : (index) -> ()
      hlcf.yield
    } else {
      hlcf.yield
    }
    kgen.return
  }
}


