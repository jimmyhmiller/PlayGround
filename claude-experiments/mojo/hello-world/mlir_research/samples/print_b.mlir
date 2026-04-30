# === samples/print_b.mlirbc ===
# MLIR bytecode: 19961 bytes, Modular trailer: 32 bytes
# decoded to 65865 chars of textual MLIR

#memory_handle = #interp.memory_handle<16, "Runtime\00" string>
#memory_handle1 = #interp.memory_handle<16, " \00" string>
#memory_handle2 = #interp.memory_handle<16, "\0A\00" string>
#memory_handle3 = #interp.memory_handle<16, "0123456789abcdefghijklmnopqrstuvwxyz\00" string>
#memory_handle4 = #interp.memory_handle<16, "" string>
#memory_handle5 = #interp.memory_handle<16, "a\00" string>
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
    %index6 = kgen.param.constant = <6>
    %struct_0 = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle1, const_global, [], [])], []}, 0, 0>, 1 }>
    %struct_1 = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle2, const_global, [], [])], []}, 0, 0>, 1 }>
    %0 = kgen.param.constant: i1 = <0>
    %index1 = kgen.param.constant = <1>
    %1 = pop.external_call @KGEN_CompilerRT_AsyncRT_GetCurrentRuntime() : () -> !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>
    %2 = kgen.struct.extract %1[0] : <(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>
    %3 = kgen.struct.extract %2[0] : <(struct<(struct<(pointer<none>) memoryOnly>)>)>
    %4 = kgen.struct.extract %3[0] : <(struct<(pointer<none>) memoryOnly>)>
    %5 = kgen.struct.extract %4[0] : <(pointer<none>) memoryOnly>
    %6 = pop.pointer_to_index %5 : <none>
    %7 = index.cmp eq(%6, %index0)
    %8 = pop.select %7, %index0, %index-1 : index
    %9 = index.cmp eq(%8, %index-1)
    hlcf.if %9 {
      hlcf.yield
    } else {
      %10 = kgen.create_closure[() -> !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>: @main_closure_0]() 
      %11 = kgen.create_closure[(!kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>) -> (): @main_closure_1]() 
      %12 = pop.external_call @KGEN_CompilerRT_GetOrCreateGlobal(%struct, %10, %11) : (!kgen.struct<(pointer<none>, index)>, !kgen.generator<() -> !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>>, !kgen.generator<(!kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>) -> ()>) -> !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>
      hlcf.yield
    }
    pop.external_call @KGEN_CompilerRT_SetArgV(%arg0, %arg1) : (!pop.scalar<si32>, !kgen.pointer<pointer<scalar<ui8>>>) -> ()
    pop.external_call @KGEN_CompilerRT_PrintStackTraceOnFault() : () -> ()
    kgen.call @"std::io::io::print[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?],flush:::Bool,file:::FileDescriptor$),Ts.values`=[[typevalue<#kgen.instref<\1B\22std::builtin::int::Int\22>>, index]]"(%index6, %struct_1, %0, %index1) : (index, !kgen.struct<(pointer<none>, index)>, i1, index owned) -> ()
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
  kgen.func @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>]"(%arg0: index, %arg1: !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut) {
    %0 = pop.cast_from_builtin %arg0 : index to !pop.scalar<index>
    %1 = pop.cast %0 : !pop.scalar<index> to !pop.scalar<si64>
    kgen.call @"std::format::__init__::Writer::write[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $3]]($0&,*$1),_Self`=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>],Ts.values`2x=[[typevalue<#kgen.instref<\1B\22std::builtin::simd::SIMD,dtype=si64,size=1\22>>, scalar<si64>]]"(%arg1, %1) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !pop.scalar<si64>) -> ()
    kgen.return
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
    %pointer = kgen.param.constant: pointer<none> = <#interp.memref<{[(#memory_handle3, const_global, [], [])], []}, 0, 0>>
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
  kgen.func @"std::builtin::simd::_write_scalar[::DType,::Writer]($1&,::SIMD[$0, ::Int(1)]),dtype=si64,W=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>]"(%arg0: !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, %arg1: !pop.scalar<si64>) {
    %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle4, const_global, [], [])], []}, 0, 0>, 0 }>
    kgen.call tail @"std::builtin::format_int::_write_int[::DType,::Writer,::Int,::StringSlice[::Bool(False), StaticConstantOrigin, *?]]($1&,::SIMD[$0, ::Int(1)],prefix:::StringSlice[::Bool(False), StaticConstantOrigin, *?]),dtype=si64,W=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>],radix=10,digit_chars={ #interp.memref<{[(#interp.memory_handle<16, \220123456789abcdefghijklmnopqrstuvwxyz\\00\22 string>, const_global, [], [])], []}, 0, 0>, 36 }"(%arg0, %arg1, %struct) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !pop.scalar<si64>, !kgen.struct<(pointer<none>, index)>) -> ()
    kgen.return
  }
  kgen.func @"std::format::__init__::Writer::write[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $3]]($0&,*$1),_Self`=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>],Ts.values`2x=[[typevalue<#kgen.instref<\1B\22std::builtin::simd::SIMD,dtype=si64,size=1\22>>, scalar<si64>]]"(%arg0: !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, %arg1: !pop.scalar<si64>) {
    kgen.call @"std::builtin::simd::SIMD::write_to[::Writer](::SIMD[$0, $1],$2&),dtype=si64,size=1,writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>]"(%arg1, %arg0) : (!pop.scalar<si64>, !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut) -> ()
    kgen.return
  }
  kgen.func @"std::io::io::_flush(::FileDescriptor)"(%arg0: index) no_inline {
    %pointer = kgen.param.constant: pointer<none> = <#interp.memref<{[(#memory_handle5, const_global, [], [])], []}, 0, 0>>
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
  kgen.func @"std::io::io::print[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?],flush:::Bool,file:::FileDescriptor$),Ts.values`=[[typevalue<#kgen.instref<\1B\22std::builtin::int::Int\22>>, index]]"(%arg0: index, %arg1: !kgen.struct<(pointer<none>, index)>, %arg2: i1, %arg3: index owned) no_inline {
    %index0 = kgen.param.constant = <0>
    %0 = pop.stack_allocation 1 x index
    pop.store %arg3, %0 : !kgen.pointer<index>
    %1 = pop.stack_allocation 1 x struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly> align 1 marked
    pop.stack_alloc.lifetime.start(%1) : !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>>
    %2 = kgen.struct.gep %1[1] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>>
    pop.store %index0, %2 : !kgen.pointer<index>
    %3 = kgen.struct.gep %1[2] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>>
    pop.store %0, %3 : !kgen.pointer<pointer<index>>
    kgen.call @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>]"(%arg0, %1) : (index, !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut) -> ()
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%1, %arg1) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    %4 = pop.load %3 : !kgen.pointer<pointer<index>>
    %5 = kgen.struct.gep %1[0] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>>
    %6 = kgen.struct.gep %5[0] : <struct<(array<4096, scalar<ui8>>) memoryOnly>>
    %7 = pop.pointer.bitcast %6 : !kgen.pointer<array<4096, scalar<ui8>>> to !kgen.pointer<none>
    %8 = pop.load %2 : !kgen.pointer<index>
    %9 = pop.load %4 : !kgen.pointer<index>
    %10 = pop.external_call @write(%9, %7, %8) : (index, !kgen.pointer<none>, index) -> index
    pop.store %index0, %2 : !kgen.pointer<index>
    pop.stack_alloc.lifetime.end(%1) : !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>>
    hlcf.if %arg2 {
      %11 = pop.load %0 : !kgen.pointer<index>
      kgen.call tail @"std::io::io::_flush(::FileDescriptor)"(%11) : (index) -> ()
      hlcf.yield
    } else {
      hlcf.yield
    }
    kgen.return
  }
}


