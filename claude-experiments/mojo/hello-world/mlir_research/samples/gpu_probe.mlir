# === samples/gpu_probe.mlirbc ===
# MLIR bytecode: 95669 bytes, Modular trailer: 32 bytes
# decoded to 504312 chars of textual MLIR

#memory_handle = #interp.memory_handle<16, " \00" string>
#memory_handle1 = #interp.memory_handle<16, "\0A\00" string>
#memory_handle2 = #interp.memory_handle<8, "0x105ED0B20000000003000000000000000000000000000000">
#memory_handle3 = #interp.memory_handle<16, "Radeon 8060S\00" string>
#memory_handle4 = #interp.memory_handle<16, "hip\00" string>
#memory_handle5 = #interp.memory_handle<16, "gfx1151\00" string>
#memory_handle6 = #interp.memory_handle<16, "RDNA3.5\00" string>
#memory_handle7 = #interp.memory_handle<8, "0x005ED0B20000000000000000000000000000000000000020">
#memory_handle8 = #interp.memory_handle<16, "" string>
#memory_handle9 = #interp.memory_handle<16, "ABORT:\00" string>
#memory_handle10 = #interp.memory_handle<16, "Runtime\00" string>
#memory_handle11 = #interp.memory_handle<16, "0123456789abcdefghijklmnopqrstuvwxyz\00" string>
#memory_handle12 = #interp.memory_handle<16, "a\00" string>
module attributes {M.target_info = #M.target<triple = "x86_64-unknown-linux-gnu", arch = "znver5", features = "+adx,+aes,+avx,+avx2,+avx512bf16,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vp2intersect,+avx512vpopcntdq,+avxvnni,+bmi,+bmi2,+clflushopt,+clwb,+clzero,+crc32,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+gfni,+invpcid,+lzcnt,+mmx,+movbe,+movdir64b,+movdiri,+mwaitx,+pclmul,+pku,+popcnt,+prefetchi,+prfchw,+rdpid,+rdpru,+rdrnd,+rdseed,+sahf,+sha,+shstk,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+sse4a,+ssse3,+vaes,+vpclmulqdq,+wbnoinvd,+x87,+xsave,+xsavec,+xsaveopt,+xsaves", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", relocation_model = "pic", simd_bit_width = 512, index_bit_width = 64, accelerator_arch = "amdgpu:gfx1151">, kgen.env = #kgen.env<{__OPTIMIZATION_LEVEL = 3 : index, __SANITIZE_ADDRESS = 0 : index}>} {
  kgen.func @"probe::main()"(%arg0: !kgen.pointer<struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>>) -> i1 {
    %string = kgen.param.constant: string = <"/home/jimmyhmiller/mojo-strix/hello-gpu/probe.mojo">
    %index50 = kgen.param.constant = <50>
    %index20 = kgen.param.constant = <20>
    %index10 = kgen.param.constant = <10>
    %simd = kgen.param.constant: scalar<si32> = <0>
    %index6208 = kgen.param.constant = <6208>
    %string_0 = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/gpu/host/device_context.mojo">
    %index17 = kgen.param.constant = <17>
    %index3112 = kgen.param.constant = <3112>
    %idx-8 = index.constant -8
    %pointer = kgen.param.constant: pointer<none> = <0>
    %pointer_1 = kgen.param.constant: pointer<none> = <#interp.uninitmem>
    %index2233785415175766016 = kgen.param.constant = <2233785415175766016>
    %index56 = kgen.param.constant = <56>
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %simd_2 = kgen.param.constant: scalar<ui8> = <0>
    %struct = kgen.param.constant: struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)> = <{ { { { 0 } } } }>
    %array = kgen.param.constant: array<1, struct<(index, index, struct<(pointer<none>, index)>)>> = <[{ -1, 0, { 0, 0 } }]>
    %index-1 = kgen.param.constant = <-1>
    %string_3 = kgen.param.constant: string = <"fill+sync ok">
    %index12 = kgen.param.constant = <12>
    %simd_4 = kgen.param.constant: scalar<f32> = <"1">
    %string_5 = kgen.param.constant: string = <"allocated 64 bytes ok">
    %index21 = kgen.param.constant = <21>
    %string_6 = kgen.param.constant: string = <"api:">
    %index4 = kgen.param.constant = <4>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %simd_7 = kgen.param.constant: scalar<index> = <1>
    %string_8 = kgen.param.constant: string = <"device:">
    %index7 = kgen.param.constant = <7>
    %index2305843009213693952 = kgen.param.constant = <2305843009213693952>
    %index0 = kgen.param.constant = <0>
    %0 = kgen.param.constant: i1 = <1>
    %struct_9 = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle, const_global, [], [])], []}, 0, 0>, 1 }>
    %struct_10 = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle1, const_global, [], [])], []}, 0, 0>, 1 }>
    %1 = kgen.param.constant: i1 = <0>
    %index1 = kgen.param.constant = <1>
    %index16 = kgen.param.constant = <16>
    %2 = pop.string.address %string
    %3 = pop.pointer.bitcast %2 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %4 = kgen.struct.create(%3, %index50) : !kgen.struct<(pointer<none>, index)>
    %5 = kgen.struct.create(%index10, %index20, %4) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %6 = pop.string.address %string_3
    %7 = pop.pointer.bitcast %6 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %8 = pop.string.address %string_5
    %9 = pop.pointer.bitcast %8 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %10 = pop.string.address %string_6
    %11 = pop.pointer.bitcast %10 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %12 = pop.string.address %string_8
    %13 = pop.pointer.bitcast %12 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %14 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
    %15 = kgen.param.materialize: struct<(pointer<none>, index, index) memoryOnly> = <{ #interp.memref<{[(#memory_handle2, stack, [(0, 2, 0)], []), (#memory_handle3, const_global, [], []), (#memory_handle4, const_global, [], []), (#memory_handle5, const_global, [], []), (#memory_handle6, const_global, [], [])], []}, 2, 0>, 3, 0 }>
    pop.stack_alloc.lifetime.start(%14) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %15, %14 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %16 = pop.string.address %string_0
    %17 = pop.pointer.bitcast %16 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %18 = kgen.struct.create(%17, %index56) : !kgen.struct<(pointer<none>, index)>
    %19 = kgen.struct.create(%index6208, %index17, %18) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %20 = kgen.struct.create(%index3112, %index17, %18) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %21 = pop.stack_allocation 1 x struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)> align 1 marked
    %22 = kgen.struct.gep %21[0] : <struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>>
    pop.stack_alloc.lifetime.start(%21) : !kgen.pointer<struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>>
    pop.store %struct, %21 : !kgen.pointer<struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>>
    %23 = pop.pointer.bitcast %21 : !kgen.pointer<struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>> to !kgen.pointer<none>
    %24 = kgen.struct.gep %14[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %25 = pop.pointer.bitcast %14 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
    %26 = kgen.struct.gep %14[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %27 = kgen.struct.gep %14[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %28 = pop.load %27 : !kgen.pointer<index>
    %29 = index.and %28, %index2305843009213693952
    %30 = index.cmp ne(%29, %index0)
    %31 = pop.xor %30, %0
    hlcf.if %31 {
      %74 = pop.load %27 : !kgen.pointer<index>
      %75 = index.and %74, %index-9223372036854775808
      %76 = index.cmp ne(%75, %index0)
      %77 = hlcf.if %76 -> index {
        %88 = pop.load %27 : !kgen.pointer<index>
        %89 = index.and %88, %index2233785415175766016
        %90 = index.shrs %89, %index56
        hlcf.yield %90 : index
      } else {
        %88 = pop.load %26 : !kgen.pointer<index>
        hlcf.yield %88 : index
      }
      %78 = index.add %77, %index1
      %79 = kgen.call @"std::collections::string::string::String::unsafe_ptr_mut(::String&,::Int$)"(%14, %78) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, index owned) -> !kgen.pointer<none>
      %80 = pop.load %27 : !kgen.pointer<index>
      %81 = index.and %80, %index-9223372036854775808
      %82 = index.cmp ne(%81, %index0)
      %83 = hlcf.if %82 -> index {
        %88 = pop.load %27 : !kgen.pointer<index>
        %89 = index.and %88, %index2233785415175766016
        %90 = index.shrs %89, %index56
        hlcf.yield %90 : index
      } else {
        %88 = pop.load %26 : !kgen.pointer<index>
        hlcf.yield %88 : index
      }
      %84 = pop.pointer.bitcast %79 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %85 = pop.offset %84[%83] : !kgen.pointer<scalar<ui8>>
      pop.store %simd_2, %85 : !kgen.pointer<scalar<ui8>>
      %86 = pop.load %27 : !kgen.pointer<index>
      %87 = index.or %86, %index2305843009213693952
      pop.store %87, %27 : !kgen.pointer<index>
      hlcf.yield
    } else {
      hlcf.yield
    }
    %32 = pop.load %27 : !kgen.pointer<index>
    %33 = index.and %32, %index-9223372036854775808
    %34 = index.cmp ne(%33, %index0)
    %35 = hlcf.if %34 -> !kgen.pointer<none> {
      hlcf.yield %25 : !kgen.pointer<none>
    } else {
      %74 = pop.load %24 : !kgen.pointer<pointer<none>>
      hlcf.yield %74 : !kgen.pointer<none>
    }
    %36 = pop.external_call @AsyncRT_DeviceContext_create(%23, %35, %simd) : (!kgen.pointer<none>, !kgen.pointer<none>, !pop.scalar<si32>) -> !kgen.struct<(struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>)>
    %37 = pop.load %27 : !kgen.pointer<index>
    %38 = index.and %37, %index4611686018427387904
    %39 = index.cmp ne(%38, %index0)
    hlcf.if %39 {
      %74 = pop.load %24 : !kgen.pointer<pointer<none>>
      %75 = pop.pointer.bitcast %74 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %76 = pop.offset %75[%idx-8] : !kgen.pointer<scalar<ui8>>
      %77 = pop.pointer.bitcast %76 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
      %78 = kgen.struct.gep %77[0] : <struct<(scalar<index>) memoryOnly>>
      %79 = pop.atomic.rmw sub(%78, %simd_7) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
      %80 = pop.cmp eq(%79, %simd_7) : <1, index>
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
    %40 = kgen.struct.extract %36[0] : <(struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>)>
    %41 = kgen.struct.extract %40[0] : <(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>
    %42 = kgen.struct.extract %41[0] : <(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>
    %43 = kgen.struct.extract %42[0] : <(struct<(array<1, pointer<none>>)>) memoryOnly>
    %44 = kgen.struct.extract %43[0] : <(array<1, pointer<none>>)>
    %45 = pop.array.get %44[0] : !pop.array<1, pointer<none>>
    %46 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
    %47 = kgen.struct.gep %46[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %48 = kgen.param.materialize: struct<(pointer<none>, index, index) memoryOnly> = <{ #interp.memref<{[(#memory_handle7, stack, [(0, 1, 0)], []), (#memory_handle8, const_global, [], [])], []}, 1, 0>, 0, 2305843009213693952 }>
    pop.stack_alloc.lifetime.start(%46) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %48, %46 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %49 = pop.array.create [%45] : !pop.array<1, pointer<none>>
    %50 = kgen.struct.create(%49) : !kgen.struct<(array<1, pointer<none>>)>
    %51 = kgen.struct.create(%50) : !kgen.struct<(struct<(array<1, pointer<none>>)>) memoryOnly>
    %52 = kgen.struct.create(%51) : !kgen.struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>
    %53 = kgen.struct.create(%52) : !kgen.struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>
    %54 = kgen.struct.create(%53) : !kgen.struct<(struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>)>
    %55 = pop.stack_allocation 1 x struct<(array<1, pointer<none>>)>
    pop.store %50, %55 : !kgen.pointer<struct<(array<1, pointer<none>>)>>
    %56 = pop.pointer.bitcast %55 : !kgen.pointer<struct<(array<1, pointer<none>>)>> to !kgen.pointer<pointer<none>>
    %57 = pop.load %56 : !kgen.pointer<pointer<none>>
    %58 = pop.pointer_to_index %57 : <none>
    %59 = index.cmp eq(%58, %index0)
    %60 = pop.select %59, %index0, %index-1 : index
    %61 = index.cmp eq(%60, %index-1)
    %62 = hlcf.if %61 -> i1 {
      %74 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
      pop.stack_alloc.lifetime.start(%74) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
      pop.store %array, %74 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
      %75 = pop.pointer.bitcast %74 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<index>
      %76 = pop.load %75 : !kgen.pointer<index>
      %77 = index.cmp eq(%76, %index-1)
      %78 = pop.select %77, %index0, %index-1 : index
      pop.stack_alloc.lifetime.end(%74) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
      %79 = index.cmp eq(%78, %index-1)
      %80 = hlcf.if %79 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
        %82 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
        pop.stack_alloc.lifetime.start(%82) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
        pop.store %array, %82 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
        %83 = pop.pointer.bitcast %82 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
        %84 = pop.load %83 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
        pop.stack_alloc.lifetime.end(%82) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
        hlcf.yield %84 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
      } else {
        hlcf.yield %20 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
      }
      %81:2 = kgen.call @"std::gpu::host::device_context::_raise_checked_impl[LITImmutOrigin,::Origin[::Bool(False), $0]](::Optional[::CStringSlice[$0, $1]],::String,::SourceLocation)_REMOVED_ARG"(%54, %46, %80) : (!kgen.struct<(struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>) throws -> (i1, !kgen.struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>)
      pop.store %81#1, %arg0 : !kgen.pointer<struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>>
      hlcf.yield %81#0 : i1
    } else {
      hlcf.yield %1 : i1
    }
    %63 = kgen.struct.gep %46[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %64 = pop.load %63 : !kgen.pointer<index>
    %65 = index.and %64, %index4611686018427387904
    %66 = index.cmp ne(%65, %index0)
    hlcf.if %66 {
      %74 = pop.load %47 : !kgen.pointer<pointer<none>>
      %75 = pop.pointer.bitcast %74 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %76 = pop.offset %75[%idx-8] : !kgen.pointer<scalar<ui8>>
      %77 = pop.pointer.bitcast %76 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
      %78 = kgen.struct.gep %77[0] : <struct<(scalar<index>) memoryOnly>>
      %79 = pop.atomic.rmw sub(%78, %simd_7) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
      %80 = pop.cmp eq(%79, %simd_7) : <1, index>
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
    pop.stack_alloc.lifetime.end(%46) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %67 = hlcf.if %62 -> !kgen.pointer<none> {
      pop.stack_alloc.lifetime.end(%21) : !kgen.pointer<struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>>
      hlcf.yield %pointer_1 : !kgen.pointer<none>
    } else {
      %74 = pop.load %22 : !kgen.pointer<struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>>
      %75 = kgen.struct.extract %74[0] : <(struct<(struct<(pointer<none>) memoryOnly>)>)>
      %76 = kgen.struct.extract %75[0] : <(struct<(pointer<none>) memoryOnly>)>
      %77 = kgen.struct.extract %76[0] : <(pointer<none>) memoryOnly>
      %78 = pop.stack_allocation 1 x pointer<none>
      %79 = pop.pointer.bitcast %78 : !kgen.pointer<pointer<none>> to !kgen.pointer<struct<(array<1, pointer<none>>)>>
      %80 = kgen.struct.gep %79[0] : <struct<(array<1, pointer<none>>)>>
      %81 = pop.pointer.bitcast %80 : !kgen.pointer<array<1, pointer<none>>> to !kgen.pointer<pointer<none>>
      %82 = pop.pointer_to_index %77 : <none>
      %83 = index.cmp eq(%82, %index0)
      %84 = pop.select %83, %index0, %index-1 : index
      %85 = index.cmp eq(%84, %index-1)
      hlcf.if %85 {
        pop.store %77, %81 : !kgen.pointer<pointer<none>>
        hlcf.yield
      } else {
        pop.store %pointer, %78 : !kgen.pointer<pointer<none>>
        hlcf.yield
      }
      %86 = pop.load %78 : !kgen.pointer<pointer<none>>
      pop.stack_alloc.lifetime.end(%21) : !kgen.pointer<struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>>
      hlcf.yield %86 : !kgen.pointer<none>
    }
    %68 = kgen.struct.create(%67) : !kgen.struct<(pointer<none>) memoryOnly>
    %69 = kgen.struct.create(%68) : !kgen.struct<(struct<(pointer<none>) memoryOnly>)>
    %70 = kgen.struct.create(%69) : !kgen.struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>
    %71 = kgen.struct.create(%70) : !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>
    %72 = kgen.struct.create(%71, %0) : !kgen.struct<(struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>, i1)>
    pop.stack_alloc.lifetime.end(%14) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %73 = hlcf.if %62 -> i1 {
      hlcf.yield %0 : i1
    } else {
      %74 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
      %75 = kgen.struct.gep %74[0] : <struct<(pointer<none>, index, index) memoryOnly>>
      pop.stack_alloc.lifetime.start(%74) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
      %76 = pop.external_call @AsyncRT_DeviceContext_deviceName(%71) : (!kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>) -> !kgen.struct<(struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>)>
      %77 = kgen.struct.extract %76[0] : <(struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>)>
      %78 = kgen.struct.extract %77[0] : <(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>
      %79 = kgen.struct.extract %78[0] : <(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>
      %80 = kgen.struct.extract %79[0] : <(struct<(array<1, pointer<none>>)>) memoryOnly>
      %81 = kgen.struct.extract %80[0] : <(array<1, pointer<none>>)>
      %82 = pop.array.get %81[0] : !pop.array<1, pointer<none>>
      %83 = pop.array.create [%82] : !pop.array<1, pointer<none>>
      %84 = kgen.struct.create(%83) : !kgen.struct<(array<1, pointer<none>>)>
      %85 = kgen.struct.create(%84) : !kgen.struct<(struct<(array<1, pointer<none>>)>) memoryOnly>
      %86 = kgen.struct.create(%85) : !kgen.struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>
      %87 = kgen.struct.create(%86) : !kgen.struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>
      %88 = kgen.struct.create(%87) : !kgen.struct<(struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>)>
      %89 = kgen.call @"std::gpu::host::device_context::_string_from_owned_charptr[LITImmutOrigin,::Origin[::Bool(False), $0]](::Optional[::CStringSlice[$0, $1]])"(%88) : (!kgen.struct<(struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>)>) -> !kgen.struct<(pointer<none>, index, index) memoryOnly>
      pop.store %89, %74 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
      %90 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
      pop.stack_alloc.lifetime.start(%90) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
      %91 = kgen.struct.gep %90[1] : <struct<(pointer<none>, index, index) memoryOnly>>
      pop.store %index7, %91 : !kgen.pointer<index>
      %92 = kgen.struct.gep %90[0] : <struct<(pointer<none>, index, index) memoryOnly>>
      pop.store %13, %92 : !kgen.pointer<pointer<none>>
      %93 = kgen.struct.gep %90[2] : <struct<(pointer<none>, index, index) memoryOnly>>
      pop.store %index2305843009213693952, %93 : !kgen.pointer<index>
      kgen.call @"std::io::io::print[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?],flush:::Bool,file:::FileDescriptor$),Ts.values`=[[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]]"(%90, %74, %struct_9, %struct_10, %1, %index1) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index)>, i1, index owned) -> ()
      %94 = pop.load %93 : !kgen.pointer<index>
      %95 = index.and %94, %index4611686018427387904
      %96 = index.cmp ne(%95, %index0)
      hlcf.if %96 {
        %127 = pop.load %92 : !kgen.pointer<pointer<none>>
        %128 = pop.pointer.bitcast %127 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
        %129 = pop.offset %128[%idx-8] : !kgen.pointer<scalar<ui8>>
        %130 = pop.pointer.bitcast %129 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
        %131 = kgen.struct.gep %130[0] : <struct<(scalar<index>) memoryOnly>>
        %132 = pop.atomic.rmw sub(%131, %simd_7) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
        %133 = pop.cmp eq(%132, %simd_7) : <1, index>
        %134 = pop.cast_to_builtin %133 : !pop.scalar<bool> to i1
        hlcf.if %134 {
          pop.fence syncscope("") acquire
          pop.aligned_free %129 : <scalar<ui8>>
          hlcf.yield
        } else {
          hlcf.yield
        }
        hlcf.yield
      } else {
        hlcf.yield
      }
      %97 = kgen.struct.gep %74[2] : <struct<(pointer<none>, index, index) memoryOnly>>
      %98 = pop.load %97 : !kgen.pointer<index>
      %99 = index.and %98, %index4611686018427387904
      %100 = index.cmp ne(%99, %index0)
      hlcf.if %100 {
        %127 = pop.load %75 : !kgen.pointer<pointer<none>>
        %128 = pop.pointer.bitcast %127 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
        %129 = pop.offset %128[%idx-8] : !kgen.pointer<scalar<ui8>>
        %130 = pop.pointer.bitcast %129 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
        %131 = kgen.struct.gep %130[0] : <struct<(scalar<index>) memoryOnly>>
        %132 = pop.atomic.rmw sub(%131, %simd_7) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
        %133 = pop.cmp eq(%132, %simd_7) : <1, index>
        %134 = pop.cast_to_builtin %133 : !pop.scalar<bool> to i1
        hlcf.if %134 {
          pop.fence syncscope("") acquire
          pop.aligned_free %129 : <scalar<ui8>>
          hlcf.yield
        } else {
          hlcf.yield
        }
        hlcf.yield
      } else {
        hlcf.yield
      }
      %101 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
      pop.stack_alloc.lifetime.start(%101) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
      %102 = pop.stack_allocation 1 x index align 1 marked
      pop.stack_alloc.lifetime.start(%102) : !kgen.pointer<index>
      pop.store %index1, %102 : !kgen.pointer<index>
      %103 = pop.pointer.bitcast %102 : !kgen.pointer<index> to !kgen.pointer<pointer<none>>
      %104 = pop.load %103 : !kgen.pointer<pointer<none>>
      pop.stack_alloc.lifetime.end(%102) : !kgen.pointer<index>
      %105 = kgen.struct.create(%104, %index0) : !kgen.struct<(pointer<none>, index)>
      %106 = pop.stack_allocation 1 x struct<(pointer<none>, index)> align 1 marked
      pop.stack_alloc.lifetime.start(%106) : !kgen.pointer<struct<(pointer<none>, index)>>
      pop.store %105, %106 : !kgen.pointer<struct<(pointer<none>, index)>>
      %107 = pop.pointer.bitcast %106 : !kgen.pointer<struct<(pointer<none>, index)>> to !kgen.pointer<none>
      pop.external_call @AsyncRT_DeviceContext_deviceApi(%107, %71) : (!kgen.pointer<none>, !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>) -> ()
      %108 = pop.load %106 : !kgen.pointer<struct<(pointer<none>, index)>>
      pop.stack_alloc.lifetime.end(%106) : !kgen.pointer<struct<(pointer<none>, index)>>
      %109 = kgen.struct.gep %101[1] : <struct<(pointer<none>, index, index) memoryOnly>>
      %110 = kgen.struct.extract %108[1] : <(pointer<none>, index)>
      pop.store %110, %109 : !kgen.pointer<index>
      %111 = kgen.struct.gep %101[0] : <struct<(pointer<none>, index, index) memoryOnly>>
      %112 = kgen.struct.extract %108[0] : <(pointer<none>, index)>
      pop.store %112, %111 : !kgen.pointer<pointer<none>>
      %113 = kgen.struct.gep %101[2] : <struct<(pointer<none>, index, index) memoryOnly>>
      pop.store %index0, %113 : !kgen.pointer<index>
      %114 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
      pop.stack_alloc.lifetime.start(%114) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
      %115 = kgen.struct.gep %114[1] : <struct<(pointer<none>, index, index) memoryOnly>>
      pop.store %index4, %115 : !kgen.pointer<index>
      %116 = kgen.struct.gep %114[0] : <struct<(pointer<none>, index, index) memoryOnly>>
      pop.store %11, %116 : !kgen.pointer<pointer<none>>
      %117 = kgen.struct.gep %114[2] : <struct<(pointer<none>, index, index) memoryOnly>>
      pop.store %index2305843009213693952, %117 : !kgen.pointer<index>
      kgen.call @"std::io::io::print[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?],flush:::Bool,file:::FileDescriptor$),Ts.values`=[[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]]"(%114, %101, %struct_9, %struct_10, %1, %index1) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index)>, i1, index owned) -> ()
      %118 = pop.load %117 : !kgen.pointer<index>
      %119 = index.and %118, %index4611686018427387904
      %120 = index.cmp ne(%119, %index0)
      hlcf.if %120 {
        %127 = pop.load %116 : !kgen.pointer<pointer<none>>
        %128 = pop.pointer.bitcast %127 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
        %129 = pop.offset %128[%idx-8] : !kgen.pointer<scalar<ui8>>
        %130 = pop.pointer.bitcast %129 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
        %131 = kgen.struct.gep %130[0] : <struct<(scalar<index>) memoryOnly>>
        %132 = pop.atomic.rmw sub(%131, %simd_7) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
        %133 = pop.cmp eq(%132, %simd_7) : <1, index>
        %134 = pop.cast_to_builtin %133 : !pop.scalar<bool> to i1
        hlcf.if %134 {
          pop.fence syncscope("") acquire
          pop.aligned_free %129 : <scalar<ui8>>
          hlcf.yield
        } else {
          hlcf.yield
        }
        hlcf.yield
      } else {
        hlcf.yield
      }
      %121 = pop.load %113 : !kgen.pointer<index>
      %122 = index.and %121, %index4611686018427387904
      %123 = index.cmp ne(%122, %index0)
      hlcf.if %123 {
        %127 = pop.load %111 : !kgen.pointer<pointer<none>>
        %128 = pop.pointer.bitcast %127 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
        %129 = pop.offset %128[%idx-8] : !kgen.pointer<scalar<ui8>>
        %130 = pop.pointer.bitcast %129 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
        %131 = kgen.struct.gep %130[0] : <struct<(scalar<index>) memoryOnly>>
        %132 = pop.atomic.rmw sub(%131, %simd_7) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
        %133 = pop.cmp eq(%132, %simd_7) : <1, index>
        %134 = pop.cast_to_builtin %133 : !pop.scalar<bool> to i1
        hlcf.if %134 {
          pop.fence syncscope("") acquire
          pop.aligned_free %129 : <scalar<ui8>>
          hlcf.yield
        } else {
          hlcf.yield
        }
        hlcf.yield
      } else {
        hlcf.yield
      }
      %124:3 = kgen.call @"std::gpu::host::device_context::DeviceContext::enqueue_create_buffer[::DType](::DeviceContext,::Int),dtype=f32"(%72, %index16) : (!kgen.struct<(struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>, i1)>, index) throws -> (i1, !kgen.struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>, !kgen.struct<(pointer<none>, struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>) memoryOnly>)
      pop.store %124#1, %arg0 : !kgen.pointer<struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>>
      %125 = kgen.struct.extract %124#2[1] : <(pointer<none>, struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>) memoryOnly>
      %126 = hlcf.if %124#0 -> i1 {
        pop.external_call @AsyncRT_DeviceContext_release(%71) : (!kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>) -> ()
        hlcf.yield %0 : i1
      } else {
        %127 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
        pop.stack_alloc.lifetime.start(%127) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
        %128 = kgen.struct.gep %127[1] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %index21, %128 : !kgen.pointer<index>
        %129 = kgen.struct.gep %127[0] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %9, %129 : !kgen.pointer<pointer<none>>
        %130 = kgen.struct.gep %127[2] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %index2305843009213693952, %130 : !kgen.pointer<index>
        kgen.call @"std::io::io::print[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?],flush:::Bool,file:::FileDescriptor$),Ts.values`=[[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]]"(%127, %struct_10, %1, %index1) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index)>, i1, index owned) -> ()
        %131 = pop.load %130 : !kgen.pointer<index>
        %132 = index.and %131, %index4611686018427387904
        %133 = index.cmp ne(%132, %index0)
        hlcf.if %133 {
          %136 = pop.load %129 : !kgen.pointer<pointer<none>>
          %137 = pop.pointer.bitcast %136 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
          %138 = pop.offset %137[%idx-8] : !kgen.pointer<scalar<ui8>>
          %139 = pop.pointer.bitcast %138 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
          %140 = kgen.struct.gep %139[0] : <struct<(scalar<index>) memoryOnly>>
          %141 = pop.atomic.rmw sub(%140, %simd_7) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
          %142 = pop.cmp eq(%141, %simd_7) : <1, index>
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
        pop.stack_alloc.lifetime.end(%127) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
        %134 = kgen.call @"std::gpu::host::device_context::DeviceBuffer::enqueue_fill(::DeviceBuffer[$0],::SIMD[$0, ::Int(1)]),dtype=f32"(%124#2, %simd_4, %arg0) : (!kgen.struct<(pointer<none>, struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>) memoryOnly>, !pop.scalar<f32>, !kgen.pointer<struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>>) -> i1
        pop.external_call @AsyncRT_DeviceBuffer_release(%125) : (!kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>) -> ()
        %135 = hlcf.if %134 -> i1 {
          pop.external_call @AsyncRT_DeviceContext_release(%71) : (!kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>) -> ()
          hlcf.yield %0 : i1
        } else {
          %136 = pop.external_call @AsyncRT_DeviceContext_synchronize(%71) : (!kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>) -> !kgen.struct<(struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>)>
          %137 = kgen.struct.extract %136[0] : <(struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>)>
          %138 = kgen.struct.extract %137[0] : <(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>
          %139 = kgen.struct.extract %138[0] : <(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>
          %140 = kgen.struct.extract %139[0] : <(struct<(array<1, pointer<none>>)>) memoryOnly>
          %141 = kgen.struct.extract %140[0] : <(array<1, pointer<none>>)>
          %142 = pop.array.get %141[0] : !pop.array<1, pointer<none>>
          %143 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> align 1 marked
          pop.stack_alloc.lifetime.start(%143) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
          %144 = pop.pointer.bitcast %143 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
          pop.store %5, %144 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
          %145 = pop.load %143 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
          pop.stack_alloc.lifetime.end(%143) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
          %146 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
          %147 = kgen.struct.gep %146[0] : <struct<(pointer<none>, index, index) memoryOnly>>
          %148 = kgen.param.materialize: struct<(pointer<none>, index, index) memoryOnly> = <{ #interp.memref<{[(#memory_handle7, stack, [(0, 1, 0)], []), (#memory_handle8, const_global, [], [])], []}, 1, 0>, 0, 2305843009213693952 }>
          pop.stack_alloc.lifetime.start(%146) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %148, %146 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
          %149 = pop.array.create [%142] : !pop.array<1, pointer<none>>
          %150 = kgen.struct.create(%149) : !kgen.struct<(array<1, pointer<none>>)>
          %151 = kgen.struct.create(%150) : !kgen.struct<(struct<(array<1, pointer<none>>)>) memoryOnly>
          %152 = kgen.struct.create(%151) : !kgen.struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>
          %153 = kgen.struct.create(%152) : !kgen.struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>
          %154 = kgen.struct.create(%153) : !kgen.struct<(struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>)>
          %155 = pop.stack_allocation 1 x struct<(array<1, pointer<none>>)>
          pop.store %150, %155 : !kgen.pointer<struct<(array<1, pointer<none>>)>>
          %156 = pop.pointer.bitcast %155 : !kgen.pointer<struct<(array<1, pointer<none>>)>> to !kgen.pointer<pointer<none>>
          %157 = pop.load %156 : !kgen.pointer<pointer<none>>
          %158 = pop.pointer_to_index %157 : <none>
          %159 = index.cmp eq(%158, %index0)
          %160 = pop.select %159, %index0, %index-1 : index
          %161 = index.cmp eq(%160, %index-1)
          %162 = hlcf.if %161 -> i1 {
            %167 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
            pop.stack_alloc.lifetime.start(%167) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
            pop.store %145, %167 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
            %168 = pop.pointer.bitcast %167 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<index>
            %169 = pop.load %168 : !kgen.pointer<index>
            %170 = index.cmp eq(%169, %index-1)
            %171 = pop.select %170, %index0, %index-1 : index
            pop.stack_alloc.lifetime.end(%167) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
            %172 = index.cmp eq(%171, %index-1)
            %173 = hlcf.if %172 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
              %175 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
              pop.stack_alloc.lifetime.start(%175) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
              pop.store %145, %175 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
              %176 = pop.pointer.bitcast %175 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
              %177 = pop.load %176 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
              pop.stack_alloc.lifetime.end(%175) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
              hlcf.yield %177 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
            } else {
              hlcf.yield %19 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
            }
            %174:2 = kgen.call @"std::gpu::host::device_context::_raise_checked_impl[LITImmutOrigin,::Origin[::Bool(False), $0]](::Optional[::CStringSlice[$0, $1]],::String,::SourceLocation)_REMOVED_ARG"(%154, %146, %173) : (!kgen.struct<(struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>) throws -> (i1, !kgen.struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>)
            pop.store %174#1, %arg0 : !kgen.pointer<struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>>
            hlcf.yield %174#0 : i1
          } else {
            hlcf.yield %1 : i1
          }
          %163 = kgen.struct.gep %146[2] : <struct<(pointer<none>, index, index) memoryOnly>>
          %164 = pop.load %163 : !kgen.pointer<index>
          %165 = index.and %164, %index4611686018427387904
          %166 = index.cmp ne(%165, %index0)
          hlcf.if %166 {
            %167 = pop.load %147 : !kgen.pointer<pointer<none>>
            %168 = pop.pointer.bitcast %167 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
            %169 = pop.offset %168[%idx-8] : !kgen.pointer<scalar<ui8>>
            %170 = pop.pointer.bitcast %169 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
            %171 = kgen.struct.gep %170[0] : <struct<(scalar<index>) memoryOnly>>
            %172 = pop.atomic.rmw sub(%171, %simd_7) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
            %173 = pop.cmp eq(%172, %simd_7) : <1, index>
            %174 = pop.cast_to_builtin %173 : !pop.scalar<bool> to i1
            hlcf.if %174 {
              pop.fence syncscope("") acquire
              pop.aligned_free %169 : <scalar<ui8>>
              hlcf.yield
            } else {
              hlcf.yield
            }
            hlcf.yield
          } else {
            hlcf.yield
          }
          pop.stack_alloc.lifetime.end(%146) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
          pop.external_call @AsyncRT_DeviceContext_release(%71) : (!kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>) -> ()
          hlcf.if %162 {
            hlcf.yield
          } else {
            %167 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
            pop.stack_alloc.lifetime.start(%167) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
            %168 = kgen.struct.gep %167[1] : <struct<(pointer<none>, index, index) memoryOnly>>
            pop.store %index12, %168 : !kgen.pointer<index>
            %169 = kgen.struct.gep %167[0] : <struct<(pointer<none>, index, index) memoryOnly>>
            pop.store %7, %169 : !kgen.pointer<pointer<none>>
            %170 = kgen.struct.gep %167[2] : <struct<(pointer<none>, index, index) memoryOnly>>
            pop.store %index2305843009213693952, %170 : !kgen.pointer<index>
            kgen.call @"std::io::io::print[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?],flush:::Bool,file:::FileDescriptor$),Ts.values`=[[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]]"(%167, %struct_10, %1, %index1) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index)>, i1, index owned) -> ()
            %171 = pop.load %170 : !kgen.pointer<index>
            %172 = index.and %171, %index4611686018427387904
            %173 = index.cmp ne(%172, %index0)
            hlcf.if %173 {
              %174 = pop.load %169 : !kgen.pointer<pointer<none>>
              %175 = pop.pointer.bitcast %174 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
              %176 = pop.offset %175[%idx-8] : !kgen.pointer<scalar<ui8>>
              %177 = pop.pointer.bitcast %176 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
              %178 = kgen.struct.gep %177[0] : <struct<(scalar<index>) memoryOnly>>
              %179 = pop.atomic.rmw sub(%178, %simd_7) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
              %180 = pop.cmp eq(%179, %simd_7) : <1, index>
              %181 = pop.cast_to_builtin %180 : !pop.scalar<bool> to i1
              hlcf.if %181 {
                pop.fence syncscope("") acquire
                pop.aligned_free %176 : <scalar<ui8>>
                hlcf.yield
              } else {
                hlcf.yield
              }
              hlcf.yield
            } else {
              hlcf.yield
            }
            pop.stack_alloc.lifetime.end(%167) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
            hlcf.yield
          }
          hlcf.yield %162 : i1
        }
        hlcf.yield %135 : i1
      }
      hlcf.yield %126 : i1
    }
    kgen.return %73 : i1
  }
  kgen.func export C @main(%arg0: !pop.scalar<si32>, %arg1: !kgen.pointer<pointer<scalar<ui8>>>) -> !pop.scalar<si32> {
    %0 = kgen.call @"std::builtin::_startup::__wrap_and_execute_raising_main[def() raises -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\22probe::main()\22"(%arg0, %arg1) : (!pop.scalar<si32>, !kgen.pointer<pointer<scalar<ui8>>>) -> !pop.scalar<si32>
    kgen.return %0 : !pop.scalar<si32>
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
  kgen.func @"std::builtin::error::StackTrace::collect_if_enabled(::Int)"(%arg0: index) -> !kgen.struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly> no_inline {
    %index0 = kgen.param.constant = <0>
    %simd = kgen.param.constant: scalar<ui8> = <0>
    %struct = kgen.param.constant: struct<()> = <{  }>
    %struct_0 = kgen.param.constant: struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)> = <{ { { { 0 } } } }>
    %simd_1 = kgen.param.constant: scalar<ui8> = <1>
    %0 = pop.stack_allocation 1 x union<struct<()>, struct<(pointer<none>) memoryOnly>>
    %1 = pop.union.bitcast %0 : <union<struct<()>, struct<(pointer<none>) memoryOnly>>> as <struct<(pointer<none>) memoryOnly>>
    %2 = kgen.struct.gep %1[0] : <struct<(pointer<none>) memoryOnly>>
    %3 = pop.union.bitcast %0 : <union<struct<()>, struct<(pointer<none>) memoryOnly>>> as <struct<()>>
    %4 = index.cmp slt(%arg0, %index0)
    %5 = hlcf.if %4 -> !pop.scalar<ui8> {
      pop.store %struct, %3 : !kgen.pointer<struct<()>>
      hlcf.yield %simd : !pop.scalar<ui8>
    } else {
      %11 = pop.stack_allocation 1 x struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)> align 1 marked
      %12 = kgen.struct.gep %11[0] : <struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>>
      %13 = kgen.struct.gep %12[0] : <struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>>
      %14 = kgen.struct.gep %13[0] : <struct<(struct<(pointer<none>) memoryOnly>)>>
      %15 = kgen.struct.gep %14[0] : <struct<(pointer<none>) memoryOnly>>
      pop.stack_alloc.lifetime.start(%11) : !kgen.pointer<struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>>
      pop.store %struct_0, %11 : !kgen.pointer<struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>>
      %16 = pop.pointer.bitcast %11 : !kgen.pointer<struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>> to !kgen.pointer<none>
      %17 = pop.external_call @KGEN_CompilerRT_GetStackTrace(%16, %arg0) : (!kgen.pointer<none>, index) -> index
      %18 = index.cmp eq(%17, %index0)
      %19 = pop.select %18, %simd, %simd_1 : !pop.scalar<ui8>
      hlcf.if %18 {
        pop.stack_alloc.lifetime.end(%11) : !kgen.pointer<struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>>
        pop.store %struct, %3 : !kgen.pointer<struct<()>>
        hlcf.yield
      } else {
        %20 = pop.load %15 : !kgen.pointer<pointer<none>>
        pop.stack_alloc.lifetime.end(%11) : !kgen.pointer<struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>>
        pop.store %20, %2 : !kgen.pointer<pointer<none>>
        hlcf.yield
      }
      hlcf.yield %19 : !pop.scalar<ui8>
    }
    %6 = pop.load %0 : !kgen.pointer<union<struct<()>, struct<(pointer<none>) memoryOnly>>>
    %7 = kgen.struct.create(%6, %5) : !kgen.struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>
    %8 = kgen.struct.create(%7) : !kgen.struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>
    %9 = kgen.struct.create(%8) : !kgen.struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>
    %10 = kgen.struct.create(%9) : !kgen.struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>
    kgen.return %10 : !kgen.struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>
  }
  kgen.func @"std::builtin::error::Error::write_to[::Writer](::Error,$0&),writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>]"(%arg0: !kgen.pointer<struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>> read_mem, %arg1: !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut) no_inline {
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %index0 = kgen.param.constant = <0>
    %index56 = kgen.param.constant = <56>
    %index2233785415175766016 = kgen.param.constant = <2233785415175766016>
    %0 = kgen.struct.gep %arg0[0] : <struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>>
    %1 = kgen.struct.gep %0[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %2 = pop.pointer.bitcast %0 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
    %3 = kgen.struct.gep %0[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %4 = pop.load %3 : !kgen.pointer<index>
    %5 = index.and %4, %index-9223372036854775808
    %6 = index.cmp ne(%5, %index0)
    %7 = hlcf.if %6 -> !kgen.pointer<none> {
      hlcf.yield %2 : !kgen.pointer<none>
    } else {
      %14 = pop.load %1 : !kgen.pointer<pointer<none>>
      hlcf.yield %14 : !kgen.pointer<none>
    }
    %8 = kgen.struct.gep %0[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %9 = pop.load %3 : !kgen.pointer<index>
    %10 = index.and %9, %index-9223372036854775808
    %11 = index.cmp ne(%10, %index0)
    %12 = hlcf.if %11 -> index {
      %14 = pop.load %3 : !kgen.pointer<index>
      %15 = index.and %14, %index2233785415175766016
      %16 = index.shrs %15, %index56
      hlcf.yield %16 : index
    } else {
      %14 = pop.load %8 : !kgen.pointer<index>
      hlcf.yield %14 : index
    }
    %13 = kgen.struct.create(%7, %12) : !kgen.struct<(pointer<none>, index)>
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg1, %13) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    kgen.return
  }
  kgen.func @"std::builtin::error::Error::get_stack_trace(::Error)"(%arg0: !kgen.struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>) -> !kgen.struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>, index, index) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly> {
    %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle8, const_global, [], [])], []}, 0, 0>, 0 }>
    %0 = kgen.param.constant: i1 = <1>
    %simd = kgen.param.constant: scalar<ui8> = <0>
    %index2305843009213693952 = kgen.param.constant = <2305843009213693952>
    %index192 = kgen.param.constant = <192>
    %string = kgen.param.constant: string = <"`Optional.value()` called on empty `Optional`. Consider using `if optional:` to check whether the `Optional` is empty before calling `.value()`, or use `.or_else()` to provide a default value.">
    %simd_0 = kgen.param.constant: scalar<index> = <1>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %index0 = kgen.param.constant = <0>
    %index-1 = kgen.param.constant = <-1>
    %simd_1 = kgen.param.constant: scalar<ui8> = <1>
    %string_2 = kgen.param.constant: string = <"">
    %string_3 = kgen.param.constant: string = <": ">
    %index2 = kgen.param.constant = <2>
    %string_4 = kgen.param.constant: string = <" ">
    %index1 = kgen.param.constant = <1>
    %struct_5 = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle9, const_global, [], [])], []}, 0, 0>, 6 }>
    %struct_6 = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle1, const_global, [], [])], []}, 0, 0>, 1 }>
    %idx-8 = index.constant -8
    %index610 = kgen.param.constant = <610>
    %index18 = kgen.param.constant = <18>
    %string_7 = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/collections/optional.mojo">
    %index53 = kgen.param.constant = <53>
    %struct_8 = kgen.param.constant: struct<()> = <{  }>
    %index261 = kgen.param.constant = <261>
    %index50 = kgen.param.constant = <50>
    %string_9 = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/builtin/error.mojo">
    %index46 = kgen.param.constant = <46>
    %1 = pop.stack_allocation 1 x union<struct<()>, struct<(pointer<none>) memoryOnly>>
    %2 = kgen.struct.extract %arg0[1] : <(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>
    %3 = kgen.struct.extract %2[0] : <(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>
    %4 = kgen.struct.extract %3[0] : <(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>
    %5 = kgen.struct.extract %4[0] : <(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>
    %6 = kgen.struct.extract %5[0] : <(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>
    pop.store %6, %1 : !kgen.pointer<union<struct<()>, struct<(pointer<none>) memoryOnly>>>
    %7 = kgen.struct.extract %5[1] : <(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>
    %8 = pop.stack_allocation 1 x union<struct<()>, struct<(pointer<none>, index, index) memoryOnly>>
    %9 = pop.union.bitcast %8 : <union<struct<()>, struct<(pointer<none>, index, index) memoryOnly>>> as <struct<()>>
    %10 = pop.union.bitcast %8 : <union<struct<()>, struct<(pointer<none>, index, index) memoryOnly>>> as <struct<(pointer<none>, index, index) memoryOnly>>
    %11 = kgen.struct.gep %10[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %12 = kgen.struct.gep %10[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %13 = kgen.struct.gep %10[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %14 = pop.string.address %string_9
    %15 = pop.pointer.bitcast %14 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %16 = kgen.struct.create(%15, %index46) : !kgen.struct<(pointer<none>, index)>
    %17 = kgen.struct.create(%index261, %index50, %16) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %18 = pop.string.address %string_7
    %19 = pop.pointer.bitcast %18 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %20 = kgen.struct.create(%19, %index53) : !kgen.struct<(pointer<none>, index)>
    %21 = kgen.struct.create(%index610, %index18, %20) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %22 = pop.string.address %string_4
    %23 = pop.pointer.bitcast %22 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %24 = pop.string.address %string_3
    %25 = pop.pointer.bitcast %24 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %26 = pop.string.address %string_2
    %27 = pop.pointer.bitcast %26 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %28 = kgen.struct.create(%27, %index0) : !kgen.struct<(pointer<none>, index)>
    %29 = pop.string.address %string
    %30 = pop.pointer.bitcast %29 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %31 = pop.union.bitcast %1 : <union<struct<()>, struct<(pointer<none>) memoryOnly>>> as <struct<(pointer<none>) memoryOnly>>
    %32 = pop.cmp eq(%7, %simd) : <1, ui8>
    %33 = pop.cast_to_builtin %32 : !pop.scalar<bool> to i1
    %34 = pop.xor %33, %0
    %35 = pop.select %34, %simd_1, %simd : !pop.scalar<ui8>
    hlcf.if %34 {
      hlcf.if %33 {
        %49 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
        pop.stack_alloc.lifetime.start(%49) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
        %50 = kgen.struct.gep %49[1] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %index192, %50 : !kgen.pointer<index>
        %51 = kgen.struct.gep %49[0] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %30, %51 : !kgen.pointer<pointer<none>>
        %52 = kgen.struct.gep %49[2] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %index2305843009213693952, %52 : !kgen.pointer<index>
        %53 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
        %54 = pop.pointer.bitcast %53 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
        pop.store %17, %54 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
        %55 = pop.load %53 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
        %56 = pop.array.get %55[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
        %57 = pop.array.create [%56] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
        %58 = kgen.struct.create(%57) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
        %59 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
        %60 = pop.pointer.bitcast %59 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
        pop.store %58, %59 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
        %61 = pop.pointer.bitcast %59 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
        %62 = pop.load %61 : !kgen.pointer<index>
        %63 = index.cmp eq(%62, %index-1)
        %64 = pop.select %63, %index0, %index-1 : index
        %65 = index.cmp eq(%64, %index-1)
        %66 = hlcf.if %65 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
          %84 = pop.load %60 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
          hlcf.yield %84 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
        } else {
          hlcf.yield %21 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
        }
        %67 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
        pop.stack_alloc.lifetime.start(%67) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
        %68 = kgen.struct.gep %67[1] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %index1, %68 : !kgen.pointer<index>
        %69 = kgen.struct.gep %67[0] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %23, %69 : !kgen.pointer<pointer<none>>
        %70 = kgen.struct.gep %67[2] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %index2305843009213693952, %70 : !kgen.pointer<index>
        %71 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
        pop.stack_alloc.lifetime.start(%71) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
        %72 = kgen.struct.gep %71[1] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %index2, %72 : !kgen.pointer<index>
        %73 = kgen.struct.gep %71[0] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %25, %73 : !kgen.pointer<pointer<none>>
        %74 = kgen.struct.gep %71[2] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %index2305843009213693952, %74 : !kgen.pointer<index>
        kgen.call @"std::io::io::print[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?],flush:::Bool,file:::FileDescriptor$),Ts.values`=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::reflection::location::SourceLocation\22>>, struct<(index, index, struct<(pointer<none>, index)>)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]]"(%struct_5, %67, %66, %71, %49, %28, %struct_6, %0, %index1) : (!kgen.struct<(pointer<none>, index)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index)>, i1, index owned) -> ()
        %75 = pop.load %70 : !kgen.pointer<index>
        %76 = index.and %75, %index4611686018427387904
        %77 = index.cmp ne(%76, %index0)
        hlcf.if %77 {
          %84 = pop.load %69 : !kgen.pointer<pointer<none>>
          %85 = pop.pointer.bitcast %84 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
          %86 = pop.offset %85[%idx-8] : !kgen.pointer<scalar<ui8>>
          %87 = pop.pointer.bitcast %86 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
          %88 = kgen.struct.gep %87[0] : <struct<(scalar<index>) memoryOnly>>
          %89 = pop.atomic.rmw sub(%88, %simd_0) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
          %90 = pop.cmp eq(%89, %simd_0) : <1, index>
          %91 = pop.cast_to_builtin %90 : !pop.scalar<bool> to i1
          hlcf.if %91 {
            pop.fence syncscope("") acquire
            pop.aligned_free %86 : <scalar<ui8>>
            hlcf.yield
          } else {
            hlcf.yield
          }
          hlcf.yield
        } else {
          hlcf.yield
        }
        %78 = pop.load %74 : !kgen.pointer<index>
        %79 = index.and %78, %index4611686018427387904
        %80 = index.cmp ne(%79, %index0)
        hlcf.if %80 {
          %84 = pop.load %73 : !kgen.pointer<pointer<none>>
          %85 = pop.pointer.bitcast %84 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
          %86 = pop.offset %85[%idx-8] : !kgen.pointer<scalar<ui8>>
          %87 = pop.pointer.bitcast %86 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
          %88 = kgen.struct.gep %87[0] : <struct<(scalar<index>) memoryOnly>>
          %89 = pop.atomic.rmw sub(%88, %simd_0) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
          %90 = pop.cmp eq(%89, %simd_0) : <1, index>
          %91 = pop.cast_to_builtin %90 : !pop.scalar<bool> to i1
          hlcf.if %91 {
            pop.fence syncscope("") acquire
            pop.aligned_free %86 : <scalar<ui8>>
            hlcf.yield
          } else {
            hlcf.yield
          }
          hlcf.yield
        } else {
          hlcf.yield
        }
        %81 = pop.load %52 : !kgen.pointer<index>
        %82 = index.and %81, %index4611686018427387904
        %83 = index.cmp ne(%82, %index0)
        hlcf.if %83 {
          %84 = pop.load %51 : !kgen.pointer<pointer<none>>
          %85 = pop.pointer.bitcast %84 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
          %86 = pop.offset %85[%idx-8] : !kgen.pointer<scalar<ui8>>
          %87 = pop.pointer.bitcast %86 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
          %88 = kgen.struct.gep %87[0] : <struct<(scalar<index>) memoryOnly>>
          %89 = pop.atomic.rmw sub(%88, %simd_0) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
          %90 = pop.cmp eq(%89, %simd_0) : <1, index>
          %91 = pop.cast_to_builtin %90 : !pop.scalar<bool> to i1
          hlcf.if %91 {
            pop.fence syncscope("") acquire
            pop.aligned_free %86 : <scalar<ui8>>
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
      %41 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
      pop.stack_alloc.lifetime.start(%41) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
      %42 = pop.load %31 : !kgen.pointer<struct<(pointer<none>) memoryOnly>>
      kgen.call @"std::collections::string::string::String::__init__[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?]),Ts.values`2x=[[typevalue<#kgen.instref<\1B\22std::builtin::error::StackTrace\22>>, struct<(pointer<none>) memoryOnly>]]"(%42, %struct, %41) : (!kgen.struct<(pointer<none>) memoryOnly>, !kgen.struct<(pointer<none>, index)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> byref_result) -> ()
      %43 = kgen.struct.gep %41[0] : <struct<(pointer<none>, index, index) memoryOnly>>
      %44 = pop.load %43 : !kgen.pointer<pointer<none>>
      pop.store %44, %13 : !kgen.pointer<pointer<none>>
      %45 = kgen.struct.gep %41[1] : <struct<(pointer<none>, index, index) memoryOnly>>
      %46 = pop.load %45 : !kgen.pointer<index>
      pop.store %46, %12 : !kgen.pointer<index>
      %47 = kgen.struct.gep %41[2] : <struct<(pointer<none>, index, index) memoryOnly>>
      %48 = pop.load %47 : !kgen.pointer<index>
      pop.store %48, %11 : !kgen.pointer<index>
      pop.stack_alloc.lifetime.end(%41) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
      hlcf.yield
    } else {
      pop.store %struct_8, %9 : !kgen.pointer<struct<()>>
      hlcf.yield
    }
    %36 = pop.load %8 : !kgen.pointer<union<struct<()>, struct<(pointer<none>, index, index) memoryOnly>>>
    %37 = kgen.struct.create(%36, %35) : !kgen.struct<(union<struct<()>, struct<(pointer<none>, index, index) memoryOnly>>, scalar<ui8>)>
    %38 = kgen.struct.create(%37) : !kgen.struct<(struct<(union<struct<()>, struct<(pointer<none>, index, index) memoryOnly>>, scalar<ui8>)>) memoryOnly>
    %39 = kgen.struct.create(%38) : !kgen.struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>, index, index) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>
    %40 = kgen.struct.create(%39) : !kgen.struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>, index, index) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>
    kgen.return %40 : !kgen.struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>, index, index) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>
  }
  kgen.func @"std::builtin::int::Int::write_to[::Writer](::Int,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>]"(%arg0: index, %arg1: !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut) {
    %0 = pop.cast_from_builtin %arg0 : index to !pop.scalar<index>
    %1 = pop.cast %0 : !pop.scalar<index> to !pop.scalar<si64>
    kgen.call @"std::format::__init__::Writer::write[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $3]]($0&,*$1),_Self`=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>],Ts.values`2x=[[typevalue<#kgen.instref<\1B\22std::builtin::simd::SIMD,dtype=si64,size=1\22>>, scalar<si64>]]"(%arg1, %1) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !pop.scalar<si64>) -> ()
    kgen.return
  }
  kgen.func @"std::builtin::simd::SIMD::write_to[::Writer](::SIMD[$0, $1],$2&),dtype=si64,size=1,writer.T`2x=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]"(%arg0: !pop.scalar<si64>, %arg1: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut) no_inline {
    %string = kgen.param.constant: string = <", ">
    %simd = kgen.param.constant: scalar<index> = <1>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle8, const_global, [], [])], []}, 0, 0>, 0 }>
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
        kgen.call @"std::collections::string::string::String::write[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](::String&,*$0),Ts.values`2x=[[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]]"(%arg1, %7) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem) -> ()
        %11 = pop.load %10 : !kgen.pointer<index>
        %12 = index.and %11, %index4611686018427387904
        %13 = index.cmp ne(%12, %index0)
        hlcf.if %13 {
          %14 = pop.load %9 : !kgen.pointer<pointer<none>>
          %15 = pop.pointer.bitcast %14 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
          %16 = pop.offset %15[%idx-8] : !kgen.pointer<scalar<ui8>>
          %17 = pop.pointer.bitcast %16 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
          %18 = kgen.struct.gep %17[0] : <struct<(scalar<index>) memoryOnly>>
          %19 = pop.atomic.rmw sub(%18, %simd) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
          %20 = pop.cmp eq(%19, %simd) : <1, index>
          %21 = pop.cast_to_builtin %20 : !pop.scalar<bool> to i1
          hlcf.if %21 {
            pop.fence syncscope("") acquire
            pop.aligned_free %16 : <scalar<ui8>>
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
      kgen.call @"std::builtin::format_int::_write_int[::DType,::Writer,::Int,::StringSlice[::Bool(False), StaticConstantOrigin, *?]]($1&,::SIMD[$0, ::Int(1)],prefix:::StringSlice[::Bool(False), StaticConstantOrigin, *?]),dtype=si64,W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],radix=10,digit_chars={ #interp.memref<{[(#interp.memory_handle<16, \220123456789abcdefghijklmnopqrstuvwxyz\\00\22 string>, const_global, [], [])], []}, 0, 0>, 36 }"(%arg1, %arg0, %struct) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !pop.scalar<si64>, !kgen.struct<(pointer<none>, index)>) -> ()
      hlcf.continue "_loop_0" %5#0 : index
    }
    kgen.return
  }
  kgen.func @"std::builtin::simd::SIMD::write_to[::Writer](::SIMD[$0, $1],$2&),dtype=si64,size=1,writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_TotalWritableBytes\22>>, struct<(index) memoryOnly>]"(%arg0: !pop.scalar<si64>, %arg1: !kgen.struct<(index) memoryOnly>) -> !kgen.struct<(index) memoryOnly> no_inline {
    %idx1 = index.constant 1
    %index0 = kgen.param.constant = <0>
    %index1 = kgen.param.constant = <1>
    %index2 = kgen.param.constant = <2>
    %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle8, const_global, [], [])], []}, 0, 0>, 0 }>
    %0 = kgen.struct.extract %arg1[0] : <(index) memoryOnly>
    %1 = hlcf.loop "_loop_0" (%arg2 = %idx1 : index, %arg3 = %0 : index) -> index {
      %3 = index.sub %idx1, %arg2
      %4 = index.sub %arg2, %index1
      %5 = index.cmp eq(%arg2, %index0)
      %6:2 = lit.try "try0" -> index, index {
        %13 = pop.select %5, %arg2, %4 : index
        hlcf.if %5 {
          lit.try.raise "try0" %13, %3 : index, index
        } else {
          hlcf.yield
        }
        lit.try.yield %13, %3 : index, index
      } except (%arg4: index, %arg5: index) {
        hlcf.break "_loop_0" %arg3 : index
      } else (%arg4: index, %arg5: index) {
        lit.try.yield %arg4, %arg5 : index, index
      }
      %7 = index.cmp ne(%6#1, %index0)
      %8 = index.add %arg3, %index2
      %9 = pop.select %7, %8, %arg3 : index
      %10 = kgen.struct.create(%9) : !kgen.struct<(index) memoryOnly>
      %11 = kgen.call @"std::builtin::format_int::_write_int[::DType,::Writer,::Int,::StringSlice[::Bool(False), StaticConstantOrigin, *?]]($1&,::SIMD[$0, ::Int(1)],prefix:::StringSlice[::Bool(False), StaticConstantOrigin, *?]),dtype=si64,W=[typevalue<#kgen.instref<\1B\22std::format::_utils::_TotalWritableBytes\22>>, struct<(index) memoryOnly>],radix=10,digit_chars={ #interp.memref<{[(#interp.memory_handle<16, \220123456789abcdefghijklmnopqrstuvwxyz\\00\22 string>, const_global, [], [])], []}, 0, 0>, 36 }"(%10, %arg0, %struct) : (!kgen.struct<(index) memoryOnly>, !pop.scalar<si64>, !kgen.struct<(pointer<none>, index)>) -> !kgen.struct<(index) memoryOnly>
      %12 = kgen.struct.extract %11[0] : <(index) memoryOnly>
      hlcf.continue "_loop_0" %6#0, %12 : index, index
    }
    %2 = kgen.struct.create(%1) : !kgen.struct<(index) memoryOnly>
    kgen.return %2 : !kgen.struct<(index) memoryOnly>
  }
  kgen.func @"std::builtin::simd::SIMD::write_to[::Writer](::SIMD[$0, $1],$2&),dtype=si64,size=1,writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::collections::string::string::String\\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>]"(%arg0: !pop.scalar<si64>, %arg1: !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut) no_inline {
    %string = kgen.param.constant: string = <", ">
    %simd = kgen.param.constant: scalar<index> = <1>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %index56 = kgen.param.constant = <56>
    %index2233785415175766016 = kgen.param.constant = <2233785415175766016>
    %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle8, const_global, [], [])], []}, 0, 0>, 0 }>
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
        kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg1, %20) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
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
      kgen.call @"std::builtin::format_int::_write_int[::DType,::Writer,::Int,::StringSlice[::Bool(False), StaticConstantOrigin, *?]]($1&,::SIMD[$0, ::Int(1)],prefix:::StringSlice[::Bool(False), StaticConstantOrigin, *?]),dtype=si64,W=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::collections::string::string::String\\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>],radix=10,digit_chars={ #interp.memref<{[(#interp.memory_handle<16, \220123456789abcdefghijklmnopqrstuvwxyz\\00\22 string>, const_global, [], [])], []}, 0, 0>, 36 }"(%arg1, %arg0, %struct) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !pop.scalar<si64>, !kgen.struct<(pointer<none>, index)>) -> ()
      hlcf.continue "_loop_0" %5#0 : index
    }
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
  kgen.func @"std::collections::string::string::String::__init__[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?]),Ts.values`2x=[[typevalue<#kgen.instref<\1B\22std::builtin::error::StackTrace\22>>, struct<(pointer<none>) memoryOnly>]]"(%arg0: !kgen.struct<(pointer<none>) memoryOnly>, %arg1: !kgen.struct<(pointer<none>, index)>, %arg2: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> byref_result) {
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %index7 = kgen.param.constant = <7>
    %index3 = kgen.param.constant = <3>
    %index0 = kgen.param.constant = <0>
    %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle8, const_global, [], [])], []}, 0, 0>, 0 }>
    %index23 = kgen.param.constant = <23>
    %simd = kgen.param.constant: scalar<uindex> = <18446744073709551615>
    %simd_0 = kgen.param.constant: scalar<uindex> = <1>
    %simd_1 = kgen.param.constant: scalar<uindex> = <0>
    %simd_2 = kgen.param.constant: scalar<ui8> = <0>
    %false = index.bool.constant false
    %0 = kgen.struct.extract %arg0[0] : <(pointer<none>) memoryOnly>
    %1 = kgen.struct.gep %arg2[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %2 = kgen.struct.gep %arg2[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %3 = kgen.struct.gep %arg2[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %4 = pop.pointer.bitcast %0 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
    %5 = hlcf.loop "_loop_0" (%arg3 = %simd_1 : !pop.scalar<uindex>) -> !pop.scalar<uindex> {
      %13 = pop.cast fast %arg3 : !pop.scalar<uindex> to !pop.scalar<index>
      %14 = pop.cast_to_builtin %13 : !pop.scalar<index> to index
      %15 = pop.offset %4[%14] : !kgen.pointer<scalar<ui8>>
      %16 = pop.cmp lt(%arg3, %simd) : <1, uindex>
      %17 = pop.cast_to_builtin %16 : !pop.scalar<bool> to i1
      %18 = hlcf.if %17 -> i1 {
        %20 = pop.load %15 : !kgen.pointer<scalar<ui8>>
        %21 = pop.cmp ne(%20, %simd_2) : <1, ui8>
        %22 = pop.cast_to_builtin %21 : !pop.scalar<bool> to i1
        hlcf.yield %22 : i1
      } else {
        hlcf.yield %false : i1
      }
      hlcf.if %18 {
        hlcf.yield
      } else {
        hlcf.break "_loop_0" %arg3 : !pop.scalar<uindex>
      }
      %19 = pop.add %arg3, %simd_0 : !pop.scalar<uindex>
      hlcf.continue "_loop_0" %19 : !pop.scalar<uindex>
    }
    %6 = pop.cast fast %5 : !pop.scalar<uindex> to !pop.scalar<index>
    %7 = pop.cast_to_builtin %6 : !pop.scalar<index> to index
    %8 = kgen.struct.extract %arg1[1] : <(pointer<none>, index)>
    %9 = index.add %7, %8
    %10 = index.add %9, %index7
    %11 = index.shrs %10, %index3
    %12 = index.cmp sle(%9, %index23)
    hlcf.if %12 {
      pop.store %index-9223372036854775808, %3 : !kgen.pointer<index>
      kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg2, %struct) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      %13 = hlcf.loop "_loop_0" (%arg3 = %simd_1 : !pop.scalar<uindex>) -> !pop.scalar<uindex> {
        %17 = pop.cast fast %arg3 : !pop.scalar<uindex> to !pop.scalar<index>
        %18 = pop.cast_to_builtin %17 : !pop.scalar<index> to index
        %19 = pop.offset %4[%18] : !kgen.pointer<scalar<ui8>>
        %20 = pop.cmp lt(%arg3, %simd) : <1, uindex>
        %21 = pop.cast_to_builtin %20 : !pop.scalar<bool> to i1
        %22 = hlcf.if %21 -> i1 {
          %24 = pop.load %19 : !kgen.pointer<scalar<ui8>>
          %25 = pop.cmp ne(%24, %simd_2) : <1, ui8>
          %26 = pop.cast_to_builtin %25 : !pop.scalar<bool> to i1
          hlcf.yield %26 : i1
        } else {
          hlcf.yield %false : i1
        }
        hlcf.if %22 {
          hlcf.yield
        } else {
          hlcf.break "_loop_0" %arg3 : !pop.scalar<uindex>
        }
        %23 = pop.add %arg3, %simd_0 : !pop.scalar<uindex>
        hlcf.continue "_loop_0" %23 : !pop.scalar<uindex>
      }
      %14 = pop.cast fast %13 : !pop.scalar<uindex> to !pop.scalar<index>
      %15 = pop.cast_to_builtin %14 : !pop.scalar<index> to index
      %16 = kgen.struct.create(%0, %15) : !kgen.struct<(pointer<none>, index)>
      kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg2, %16) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg2, %arg1) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      hlcf.yield
    } else {
      pop.store %11, %3 : !kgen.pointer<index>
      %13 = index.shl %11, %index3
      %14 = kgen.call @"std::collections::string::string::String::_alloc(::Int)"(%13) : (index) -> !kgen.pointer<none>
      pop.store %14, %2 : !kgen.pointer<pointer<none>>
      pop.store %index0, %1 : !kgen.pointer<index>
      %15 = pop.load %3 : !kgen.pointer<index>
      %16 = index.or %15, %index4611686018427387904
      pop.store %16, %3 : !kgen.pointer<index>
      %17 = pop.stack_allocation 1 x struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly> align 1 marked
      pop.stack_alloc.lifetime.start(%17) : !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      %18 = kgen.struct.gep %17[1] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      pop.store %index0, %18 : !kgen.pointer<index>
      %19 = kgen.struct.gep %17[2] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      pop.store %arg2, %19 : !kgen.pointer<pointer<struct<(pointer<none>, index, index) memoryOnly>>>
      kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"(%17, %struct) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      %20 = hlcf.loop "_loop_0" (%arg3 = %simd_1 : !pop.scalar<uindex>) -> !pop.scalar<uindex> {
        %30 = pop.cast fast %arg3 : !pop.scalar<uindex> to !pop.scalar<index>
        %31 = pop.cast_to_builtin %30 : !pop.scalar<index> to index
        %32 = pop.offset %4[%31] : !kgen.pointer<scalar<ui8>>
        %33 = pop.cmp lt(%arg3, %simd) : <1, uindex>
        %34 = pop.cast_to_builtin %33 : !pop.scalar<bool> to i1
        %35 = hlcf.if %34 -> i1 {
          %37 = pop.load %32 : !kgen.pointer<scalar<ui8>>
          %38 = pop.cmp ne(%37, %simd_2) : <1, ui8>
          %39 = pop.cast_to_builtin %38 : !pop.scalar<bool> to i1
          hlcf.yield %39 : i1
        } else {
          hlcf.yield %false : i1
        }
        hlcf.if %35 {
          hlcf.yield
        } else {
          hlcf.break "_loop_0" %arg3 : !pop.scalar<uindex>
        }
        %36 = pop.add %arg3, %simd_0 : !pop.scalar<uindex>
        hlcf.continue "_loop_0" %36 : !pop.scalar<uindex>
      }
      %21 = pop.cast fast %20 : !pop.scalar<uindex> to !pop.scalar<index>
      %22 = pop.cast_to_builtin %21 : !pop.scalar<index> to index
      %23 = kgen.struct.create(%0, %22) : !kgen.struct<(pointer<none>, index)>
      kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"(%17, %23) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"(%17, %arg1) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      %24 = pop.load %19 : !kgen.pointer<pointer<struct<(pointer<none>, index, index) memoryOnly>>>
      %25 = kgen.struct.gep %17[0] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      %26 = kgen.struct.gep %25[0] : <struct<(array<4096, scalar<ui8>>) memoryOnly>>
      %27 = pop.pointer.bitcast %26 : !kgen.pointer<array<4096, scalar<ui8>>> to !kgen.pointer<none>
      %28 = pop.load %18 : !kgen.pointer<index>
      %29 = kgen.struct.create(%27, %28) : !kgen.struct<(pointer<none>, index)>
      kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%24, %29) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      pop.store %index0, %18 : !kgen.pointer<index>
      pop.stack_alloc.lifetime.end(%17) : !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      hlcf.yield
    }
    kgen.return
  }
  kgen.func @"std::collections::string::string::String::__init__[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?]),Ts.values`2x=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>]]"(%arg0: !kgen.struct<(pointer<none>, index)>, %arg1: !kgen.struct<(pointer<none>, index)>, %arg2: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> byref_result) {
    %index23 = kgen.param.constant = <23>
    %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle8, const_global, [], [])], []}, 0, 0>, 0 }>
    %index0 = kgen.param.constant = <0>
    %index3 = kgen.param.constant = <3>
    %index7 = kgen.param.constant = <7>
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %0 = kgen.struct.gep %arg2[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %1 = kgen.struct.gep %arg2[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %2 = kgen.struct.gep %arg2[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %3 = kgen.struct.extract %arg0[1] : <(pointer<none>, index)>
    %4 = kgen.struct.extract %arg1[1] : <(pointer<none>, index)>
    %5 = index.add %3, %4
    %6 = index.add %5, %index7
    %7 = index.shrs %6, %index3
    %8 = index.cmp sle(%5, %index23)
    hlcf.if %8 {
      pop.store %index-9223372036854775808, %2 : !kgen.pointer<index>
      kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg2, %struct) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg2, %arg0) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg2, %arg1) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      hlcf.yield
    } else {
      pop.store %7, %2 : !kgen.pointer<index>
      %9 = index.shl %7, %index3
      %10 = kgen.call @"std::collections::string::string::String::_alloc(::Int)"(%9) : (index) -> !kgen.pointer<none>
      pop.store %10, %1 : !kgen.pointer<pointer<none>>
      pop.store %index0, %0 : !kgen.pointer<index>
      %11 = pop.load %2 : !kgen.pointer<index>
      %12 = index.or %11, %index4611686018427387904
      pop.store %12, %2 : !kgen.pointer<index>
      %13 = pop.stack_allocation 1 x struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly> align 1 marked
      pop.stack_alloc.lifetime.start(%13) : !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      %14 = kgen.struct.gep %13[1] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      pop.store %index0, %14 : !kgen.pointer<index>
      %15 = kgen.struct.gep %13[2] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      pop.store %arg2, %15 : !kgen.pointer<pointer<struct<(pointer<none>, index, index) memoryOnly>>>
      kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"(%13, %struct) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"(%13, %arg0) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"(%13, %arg1) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      %16 = pop.load %15 : !kgen.pointer<pointer<struct<(pointer<none>, index, index) memoryOnly>>>
      %17 = kgen.struct.gep %13[0] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      %18 = kgen.struct.gep %17[0] : <struct<(array<4096, scalar<ui8>>) memoryOnly>>
      %19 = pop.pointer.bitcast %18 : !kgen.pointer<array<4096, scalar<ui8>>> to !kgen.pointer<none>
      %20 = pop.load %14 : !kgen.pointer<index>
      %21 = kgen.struct.create(%19, %20) : !kgen.struct<(pointer<none>, index)>
      kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%16, %21) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      pop.store %index0, %14 : !kgen.pointer<index>
      pop.stack_alloc.lifetime.end(%13) : !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      hlcf.yield
    }
    kgen.return
  }
  kgen.func @"std::collections::string::string::String::write[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](::String&,*$0),Ts.values`2x=[[typevalue<#kgen.instref<\1B\22std::builtin::simd::SIMD,dtype=si64,size=1\22>>, scalar<si64>]]"(%arg0: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, %arg1: !pop.scalar<si64>) {
    %index0 = kgen.param.constant = <0>
    %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle8, const_global, [], [])], []}, 0, 0>, 0 }>
    %index23 = kgen.param.constant = <23>
    %index56 = kgen.param.constant = <56>
    %index2233785415175766016 = kgen.param.constant = <2233785415175766016>
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %index3 = kgen.param.constant = <3>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %0 = kgen.param.constant: i1 = <1>
    %1 = kgen.struct.gep %arg0[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %2 = kgen.struct.gep %arg0[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %3 = pop.load %2 : !kgen.pointer<index>
    %4 = index.and %3, %index-9223372036854775808
    %5 = index.cmp ne(%4, %index0)
    %6 = hlcf.if %5 -> index {
      %11 = pop.load %2 : !kgen.pointer<index>
      %12 = index.and %11, %index2233785415175766016
      %13 = index.shrs %12, %index56
      hlcf.yield %13 : index
    } else {
      %11 = pop.load %1 : !kgen.pointer<index>
      hlcf.yield %11 : index
    }
    %7 = kgen.struct.create(%6) : !kgen.struct<(index) memoryOnly>
    %8 = kgen.call @"std::builtin::simd::SIMD::write_to[::Writer](::SIMD[$0, $1],$2&),dtype=si64,size=1,writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_TotalWritableBytes\22>>, struct<(index) memoryOnly>]"(%arg1, %7) : (!pop.scalar<si64>, !kgen.struct<(index) memoryOnly>) -> !kgen.struct<(index) memoryOnly>
    %9 = kgen.struct.extract %8[0] : <(index) memoryOnly>
    %10 = index.cmp sle(%9, %index23)
    hlcf.if %10 {
      kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg0, %struct) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      kgen.call @"std::builtin::simd::SIMD::write_to[::Writer](::SIMD[$0, $1],$2&),dtype=si64,size=1,writer.T`2x=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]"(%arg1, %arg0) : (!pop.scalar<si64>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut) -> ()
      kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg0, %struct) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      hlcf.yield
    } else {
      %11 = pop.load %2 : !kgen.pointer<index>
      %12 = index.and %11, %index-9223372036854775808
      %13 = index.cmp ne(%12, %index0)
      %14 = hlcf.if %13 -> index {
        hlcf.yield %index23 : index
      } else {
        %25 = pop.load %2 : !kgen.pointer<index>
        %26 = index.and %25, %index4611686018427387904
        %27 = index.cmp ne(%26, %index0)
        %28 = pop.xor %27, %0
        %29 = hlcf.if %28 -> index {
          %30 = pop.load %1 : !kgen.pointer<index>
          hlcf.yield %30 : index
        } else {
          %30 = pop.load %2 : !kgen.pointer<index>
          %31 = index.shl %30, %index3
          hlcf.yield %31 : index
        }
        hlcf.yield %29 : index
      }
      %15 = index.cmp sle(%9, %14)
      hlcf.if %15 {
        hlcf.yield
      } else {
        kgen.call @"std::collections::string::string::String::_realloc_mutable(::String&,::Int)"(%arg0, %9) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, index) -> ()
        hlcf.yield
      }
      %16 = pop.stack_allocation 1 x struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly> align 1 marked
      pop.stack_alloc.lifetime.start(%16) : !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      %17 = kgen.struct.gep %16[1] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      pop.store %index0, %17 : !kgen.pointer<index>
      %18 = kgen.struct.gep %16[2] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      pop.store %arg0, %18 : !kgen.pointer<pointer<struct<(pointer<none>, index, index) memoryOnly>>>
      kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"(%16, %struct) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      kgen.call @"std::builtin::simd::SIMD::write_to[::Writer](::SIMD[$0, $1],$2&),dtype=si64,size=1,writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::collections::string::string::String\\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>]"(%arg1, %16) : (!pop.scalar<si64>, !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut) -> ()
      kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"(%16, %struct) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      %19 = pop.load %18 : !kgen.pointer<pointer<struct<(pointer<none>, index, index) memoryOnly>>>
      %20 = kgen.struct.gep %16[0] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      %21 = kgen.struct.gep %20[0] : <struct<(array<4096, scalar<ui8>>) memoryOnly>>
      %22 = pop.pointer.bitcast %21 : !kgen.pointer<array<4096, scalar<ui8>>> to !kgen.pointer<none>
      %23 = pop.load %17 : !kgen.pointer<index>
      %24 = kgen.struct.create(%22, %23) : !kgen.struct<(pointer<none>, index)>
      kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%19, %24) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      pop.store %index0, %17 : !kgen.pointer<index>
      pop.stack_alloc.lifetime.end(%16) : !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      hlcf.yield
    }
    kgen.return
  }
  kgen.func @"std::collections::string::string::String::write[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](::String&,*$0),Ts.values`2x=[[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]]"(%arg0: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, %arg1: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem) {
    %0 = kgen.param.constant: i1 = <1>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %index3 = kgen.param.constant = <3>
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %index2233785415175766016 = kgen.param.constant = <2233785415175766016>
    %index56 = kgen.param.constant = <56>
    %index23 = kgen.param.constant = <23>
    %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle8, const_global, [], [])], []}, 0, 0>, 0 }>
    %index0 = kgen.param.constant = <0>
    %1 = pop.pointer.bitcast %arg1 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
    %2 = kgen.struct.gep %arg1[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %3 = kgen.struct.gep %arg0[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %4 = kgen.struct.gep %arg0[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %5 = pop.load %4 : !kgen.pointer<index>
    %6 = index.and %5, %index-9223372036854775808
    %7 = index.cmp ne(%6, %index0)
    %8 = hlcf.if %7 -> index {
      %17 = pop.load %4 : !kgen.pointer<index>
      %18 = index.and %17, %index2233785415175766016
      %19 = index.shrs %18, %index56
      hlcf.yield %19 : index
    } else {
      %17 = pop.load %3 : !kgen.pointer<index>
      hlcf.yield %17 : index
    }
    %9 = kgen.struct.gep %arg1[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %10 = kgen.struct.gep %arg1[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %11 = pop.load %9 : !kgen.pointer<index>
    %12 = index.and %11, %index-9223372036854775808
    %13 = index.cmp ne(%12, %index0)
    %14 = hlcf.if %13 -> index {
      %17 = pop.load %9 : !kgen.pointer<index>
      %18 = index.and %17, %index2233785415175766016
      %19 = index.shrs %18, %index56
      hlcf.yield %19 : index
    } else {
      %17 = pop.load %10 : !kgen.pointer<index>
      hlcf.yield %17 : index
    }
    %15 = index.add %8, %14
    %16 = index.cmp sle(%15, %index23)
    hlcf.if %16 {
      kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg0, %struct) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      %17 = pop.load %9 : !kgen.pointer<index>
      %18 = index.and %17, %index-9223372036854775808
      %19 = index.cmp ne(%18, %index0)
      %20 = hlcf.if %19 -> !kgen.pointer<none> {
        hlcf.yield %1 : !kgen.pointer<none>
      } else {
        %26 = pop.load %2 : !kgen.pointer<pointer<none>>
        hlcf.yield %26 : !kgen.pointer<none>
      }
      %21 = pop.load %9 : !kgen.pointer<index>
      %22 = index.and %21, %index-9223372036854775808
      %23 = index.cmp ne(%22, %index0)
      %24 = hlcf.if %23 -> index {
        %26 = pop.load %9 : !kgen.pointer<index>
        %27 = index.and %26, %index2233785415175766016
        %28 = index.shrs %27, %index56
        hlcf.yield %28 : index
      } else {
        %26 = pop.load %10 : !kgen.pointer<index>
        hlcf.yield %26 : index
      }
      %25 = kgen.struct.create(%20, %24) : !kgen.struct<(pointer<none>, index)>
      kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg0, %25) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg0, %struct) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      hlcf.yield
    } else {
      %17 = pop.load %4 : !kgen.pointer<index>
      %18 = index.and %17, %index-9223372036854775808
      %19 = index.cmp ne(%18, %index0)
      %20 = hlcf.if %19 -> index {
        hlcf.yield %index23 : index
      } else {
        %40 = pop.load %4 : !kgen.pointer<index>
        %41 = index.and %40, %index4611686018427387904
        %42 = index.cmp ne(%41, %index0)
        %43 = pop.xor %42, %0
        %44 = hlcf.if %43 -> index {
          %45 = pop.load %3 : !kgen.pointer<index>
          hlcf.yield %45 : index
        } else {
          %45 = pop.load %4 : !kgen.pointer<index>
          %46 = index.shl %45, %index3
          hlcf.yield %46 : index
        }
        hlcf.yield %44 : index
      }
      %21 = index.cmp sle(%15, %20)
      hlcf.if %21 {
        hlcf.yield
      } else {
        kgen.call @"std::collections::string::string::String::_realloc_mutable(::String&,::Int)"(%arg0, %15) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, index) -> ()
        hlcf.yield
      }
      %22 = pop.stack_allocation 1 x struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly> align 1 marked
      pop.stack_alloc.lifetime.start(%22) : !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      %23 = kgen.struct.gep %22[1] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      pop.store %index0, %23 : !kgen.pointer<index>
      %24 = kgen.struct.gep %22[2] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      pop.store %arg0, %24 : !kgen.pointer<pointer<struct<(pointer<none>, index, index) memoryOnly>>>
      kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"(%22, %struct) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      %25 = pop.load %9 : !kgen.pointer<index>
      %26 = index.and %25, %index-9223372036854775808
      %27 = index.cmp ne(%26, %index0)
      %28 = hlcf.if %27 -> !kgen.pointer<none> {
        hlcf.yield %1 : !kgen.pointer<none>
      } else {
        %40 = pop.load %2 : !kgen.pointer<pointer<none>>
        hlcf.yield %40 : !kgen.pointer<none>
      }
      %29 = pop.load %9 : !kgen.pointer<index>
      %30 = index.and %29, %index-9223372036854775808
      %31 = index.cmp ne(%30, %index0)
      %32 = hlcf.if %31 -> index {
        %40 = pop.load %9 : !kgen.pointer<index>
        %41 = index.and %40, %index2233785415175766016
        %42 = index.shrs %41, %index56
        hlcf.yield %42 : index
      } else {
        %40 = pop.load %10 : !kgen.pointer<index>
        hlcf.yield %40 : index
      }
      %33 = kgen.struct.create(%28, %32) : !kgen.struct<(pointer<none>, index)>
      kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"(%22, %33) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"(%22, %struct) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      %34 = pop.load %24 : !kgen.pointer<pointer<struct<(pointer<none>, index, index) memoryOnly>>>
      %35 = kgen.struct.gep %22[0] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      %36 = kgen.struct.gep %35[0] : <struct<(array<4096, scalar<ui8>>) memoryOnly>>
      %37 = pop.pointer.bitcast %36 : !kgen.pointer<array<4096, scalar<ui8>>> to !kgen.pointer<none>
      %38 = pop.load %23 : !kgen.pointer<index>
      %39 = kgen.struct.create(%37, %38) : !kgen.struct<(pointer<none>, index)>
      kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%34, %39) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      pop.store %index0, %23 : !kgen.pointer<index>
      pop.stack_alloc.lifetime.end(%22) : !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      hlcf.yield
    }
    kgen.return
  }
  kgen.func @"std::collections::string::string::String::write[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](::String&,*$0),Ts.values`2x=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::builtin::int::Int\22>>, index], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::builtin::int::Int\22>>, index]]"(%arg0: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, %arg1: !kgen.struct<(pointer<none>, index)>, %arg2: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, %arg3: index, %arg4: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, %arg5: index) {
    %index0 = kgen.param.constant = <0>
    %string = kgen.param.constant: string = <"">
    %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle8, const_global, [], [])], []}, 0, 0>, 0 }>
    %index23 = kgen.param.constant = <23>
    %index56 = kgen.param.constant = <56>
    %index2233785415175766016 = kgen.param.constant = <2233785415175766016>
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %index3 = kgen.param.constant = <3>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %0 = kgen.param.constant: i1 = <1>
    %1 = kgen.struct.gep %arg0[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %2 = kgen.struct.gep %arg0[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %3 = pop.load %2 : !kgen.pointer<index>
    %4 = index.and %3, %index-9223372036854775808
    %5 = index.cmp ne(%4, %index0)
    %6 = hlcf.if %5 -> index {
      %16 = pop.load %2 : !kgen.pointer<index>
      %17 = index.and %16, %index2233785415175766016
      %18 = index.shrs %17, %index56
      hlcf.yield %18 : index
    } else {
      %16 = pop.load %1 : !kgen.pointer<index>
      hlcf.yield %16 : index
    }
    %7 = pop.string.address %string
    %8 = pop.pointer.bitcast %7 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %9 = kgen.struct.create(%8, %index0) : !kgen.struct<(pointer<none>, index)>
    %10 = pop.load %arg2 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %11 = pop.load %arg4 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %12 = kgen.struct.create(%6) : !kgen.struct<(index) memoryOnly>
    %13 = kgen.call @"std::builtin::variadics::VariadicPack::_write_to[LITImmutOrigin,LITImmutOrigin,LITImmutOrigin,::Origin[::Bool(False), $7],::Origin[::Bool(False), $8],::Origin[::Bool(False), $9],::Bool,::Writer](::VariadicPack[$0, $1, $2, $3, $4, $5, $6],$14&,::StringSlice[::Bool(False), $7, $10],::StringSlice[::Bool(False), $8, $11],::StringSlice[::Bool(False), $9, $12]){#kgen.param_list.reduce($4, base=::Bool(True), reducer=[::Bool, KGENParamList[::AnyType], index] ::Bool(conforms_to($1[$2], AnyType & ImplicitlyDestructible & Writable)) if $0 else $0)._mlir_value}_REMOVED_ARG,element_trait=type,element_types.values`2=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::builtin::int::Int\22>>, index], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::builtin::int::Int\22>>, index]],is_repr=0,writer.T`2x4=[typevalue<#kgen.instref<\1B\22std::format::_utils::_TotalWritableBytes\22>>, struct<(index) memoryOnly>]"(%arg1, %10, %arg3, %11, %arg5, %12, %struct, %struct, %9) : (!kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index, index) memoryOnly>, index, !kgen.struct<(pointer<none>, index, index) memoryOnly>, index, !kgen.struct<(index) memoryOnly>, !kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index)>) -> !kgen.struct<(index) memoryOnly>
    %14 = kgen.struct.extract %13[0] : <(index) memoryOnly>
    %15 = index.cmp sle(%14, %index23)
    hlcf.if %15 {
      kgen.call @"std::builtin::variadics::VariadicPack::_write_to[LITImmutOrigin,LITImmutOrigin,LITImmutOrigin,::Origin[::Bool(False), $7],::Origin[::Bool(False), $8],::Origin[::Bool(False), $9],::Bool,::Writer](::VariadicPack[$0, $1, $2, $3, $4, $5, $6],$14&,::StringSlice[::Bool(False), $7, $10],::StringSlice[::Bool(False), $8, $11],::StringSlice[::Bool(False), $9, $12]){#kgen.param_list.reduce($4, base=::Bool(True), reducer=[::Bool, KGENParamList[::AnyType], index] ::Bool(conforms_to($1[$2], AnyType & ImplicitlyDestructible & Writable)) if $0 else $0)._mlir_value}_REMOVED_ARG,element_trait=type,element_types.values`2=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::builtin::int::Int\22>>, index], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::builtin::int::Int\22>>, index]],is_repr=0,writer.T`2x4=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]"(%arg1, %arg2, %arg3, %arg4, %arg5, %arg0, %struct, %struct, %9) : (!kgen.struct<(pointer<none>, index)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, index, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, index, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index)>) -> ()
      hlcf.yield
    } else {
      %16 = pop.load %2 : !kgen.pointer<index>
      %17 = index.and %16, %index-9223372036854775808
      %18 = index.cmp ne(%17, %index0)
      %19 = hlcf.if %18 -> index {
        hlcf.yield %index23 : index
      } else {
        %30 = pop.load %2 : !kgen.pointer<index>
        %31 = index.and %30, %index4611686018427387904
        %32 = index.cmp ne(%31, %index0)
        %33 = pop.xor %32, %0
        %34 = hlcf.if %33 -> index {
          %35 = pop.load %1 : !kgen.pointer<index>
          hlcf.yield %35 : index
        } else {
          %35 = pop.load %2 : !kgen.pointer<index>
          %36 = index.shl %35, %index3
          hlcf.yield %36 : index
        }
        hlcf.yield %34 : index
      }
      %20 = index.cmp sle(%14, %19)
      hlcf.if %20 {
        hlcf.yield
      } else {
        kgen.call @"std::collections::string::string::String::_realloc_mutable(::String&,::Int)"(%arg0, %14) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, index) -> ()
        hlcf.yield
      }
      %21 = pop.stack_allocation 1 x struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly> align 1 marked
      pop.stack_alloc.lifetime.start(%21) : !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      %22 = kgen.struct.gep %21[1] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      pop.store %index0, %22 : !kgen.pointer<index>
      %23 = kgen.struct.gep %21[2] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      pop.store %arg0, %23 : !kgen.pointer<pointer<struct<(pointer<none>, index, index) memoryOnly>>>
      kgen.call @"std::builtin::variadics::VariadicPack::_write_to[LITImmutOrigin,LITImmutOrigin,LITImmutOrigin,::Origin[::Bool(False), $7],::Origin[::Bool(False), $8],::Origin[::Bool(False), $9],::Bool,::Writer](::VariadicPack[$0, $1, $2, $3, $4, $5, $6],$14&,::StringSlice[::Bool(False), $7, $10],::StringSlice[::Bool(False), $8, $11],::StringSlice[::Bool(False), $9, $12]){#kgen.param_list.reduce($4, base=::Bool(True), reducer=[::Bool, KGENParamList[::AnyType], index] ::Bool(conforms_to($1[$2], AnyType & ImplicitlyDestructible & Writable)) if $0 else $0)._mlir_value}_REMOVED_ARG,element_trait=type,element_types.values`2=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::builtin::int::Int\22>>, index], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::builtin::int::Int\22>>, index]],is_repr=0,writer.T`2x4=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::collections::string::string::String\\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>]"(%arg1, %arg2, %arg3, %arg4, %arg5, %21, %struct, %struct, %9) : (!kgen.struct<(pointer<none>, index)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, index, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, index, !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index)>) -> ()
      %24 = pop.load %23 : !kgen.pointer<pointer<struct<(pointer<none>, index, index) memoryOnly>>>
      %25 = kgen.struct.gep %21[0] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      %26 = kgen.struct.gep %25[0] : <struct<(array<4096, scalar<ui8>>) memoryOnly>>
      %27 = pop.pointer.bitcast %26 : !kgen.pointer<array<4096, scalar<ui8>>> to !kgen.pointer<none>
      %28 = pop.load %22 : !kgen.pointer<index>
      %29 = kgen.struct.create(%27, %28) : !kgen.struct<(pointer<none>, index)>
      kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%24, %29) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      pop.store %index0, %22 : !kgen.pointer<index>
      pop.stack_alloc.lifetime.end(%21) : !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      hlcf.yield
    }
    kgen.return
  }
  kgen.func @"std::collections::string::string::String::write[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](::String&,*$0),Ts.values`2x=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>]]"(%arg0: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, %arg1: !kgen.struct<(pointer<none>, index)>) {
    %index0 = kgen.param.constant = <0>
    %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle8, const_global, [], [])], []}, 0, 0>, 0 }>
    %index23 = kgen.param.constant = <23>
    %index56 = kgen.param.constant = <56>
    %index2233785415175766016 = kgen.param.constant = <2233785415175766016>
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %index3 = kgen.param.constant = <3>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %0 = kgen.param.constant: i1 = <1>
    %1 = kgen.struct.extract %arg1[1] : <(pointer<none>, index)>
    %2 = kgen.struct.gep %arg0[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %3 = kgen.struct.gep %arg0[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %4 = pop.load %3 : !kgen.pointer<index>
    %5 = index.and %4, %index-9223372036854775808
    %6 = index.cmp ne(%5, %index0)
    %7 = hlcf.if %6 -> index {
      %10 = pop.load %3 : !kgen.pointer<index>
      %11 = index.and %10, %index2233785415175766016
      %12 = index.shrs %11, %index56
      hlcf.yield %12 : index
    } else {
      %10 = pop.load %2 : !kgen.pointer<index>
      hlcf.yield %10 : index
    }
    %8 = index.add %7, %1
    %9 = index.cmp sle(%8, %index23)
    hlcf.if %9 {
      kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg0, %struct) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg0, %arg1) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg0, %struct) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      hlcf.yield
    } else {
      %10 = pop.load %3 : !kgen.pointer<index>
      %11 = index.and %10, %index-9223372036854775808
      %12 = index.cmp ne(%11, %index0)
      %13 = hlcf.if %12 -> index {
        hlcf.yield %index23 : index
      } else {
        %24 = pop.load %3 : !kgen.pointer<index>
        %25 = index.and %24, %index4611686018427387904
        %26 = index.cmp ne(%25, %index0)
        %27 = pop.xor %26, %0
        %28 = hlcf.if %27 -> index {
          %29 = pop.load %2 : !kgen.pointer<index>
          hlcf.yield %29 : index
        } else {
          %29 = pop.load %3 : !kgen.pointer<index>
          %30 = index.shl %29, %index3
          hlcf.yield %30 : index
        }
        hlcf.yield %28 : index
      }
      %14 = index.cmp sle(%8, %13)
      hlcf.if %14 {
        hlcf.yield
      } else {
        kgen.call @"std::collections::string::string::String::_realloc_mutable(::String&,::Int)"(%arg0, %8) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, index) -> ()
        hlcf.yield
      }
      %15 = pop.stack_allocation 1 x struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly> align 1 marked
      pop.stack_alloc.lifetime.start(%15) : !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      %16 = kgen.struct.gep %15[1] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      pop.store %index0, %16 : !kgen.pointer<index>
      %17 = kgen.struct.gep %15[2] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      pop.store %arg0, %17 : !kgen.pointer<pointer<struct<(pointer<none>, index, index) memoryOnly>>>
      kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"(%15, %struct) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"(%15, %arg1) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"(%15, %struct) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      %18 = pop.load %17 : !kgen.pointer<pointer<struct<(pointer<none>, index, index) memoryOnly>>>
      %19 = kgen.struct.gep %15[0] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      %20 = kgen.struct.gep %19[0] : <struct<(array<4096, scalar<ui8>>) memoryOnly>>
      %21 = pop.pointer.bitcast %20 : !kgen.pointer<array<4096, scalar<ui8>>> to !kgen.pointer<none>
      %22 = pop.load %16 : !kgen.pointer<index>
      %23 = kgen.struct.create(%21, %22) : !kgen.struct<(pointer<none>, index)>
      kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%18, %23) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      pop.store %index0, %16 : !kgen.pointer<index>
      pop.stack_alloc.lifetime.end(%15) : !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      hlcf.yield
    }
    kgen.return
  }
  kgen.func @"std::collections::string::string::String::write[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](::String&,*$0),Ts.values`2x=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=1,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>]]"(%arg0: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, %arg1: !kgen.struct<(pointer<none>, index)>) {
    %index0 = kgen.param.constant = <0>
    %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle8, const_global, [], [])], []}, 0, 0>, 0 }>
    %index23 = kgen.param.constant = <23>
    %index56 = kgen.param.constant = <56>
    %index2233785415175766016 = kgen.param.constant = <2233785415175766016>
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %index3 = kgen.param.constant = <3>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %0 = kgen.param.constant: i1 = <1>
    %1 = kgen.struct.extract %arg1[1] : <(pointer<none>, index)>
    %2 = kgen.struct.gep %arg0[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %3 = kgen.struct.gep %arg0[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %4 = pop.load %3 : !kgen.pointer<index>
    %5 = index.and %4, %index-9223372036854775808
    %6 = index.cmp ne(%5, %index0)
    %7 = hlcf.if %6 -> index {
      %10 = pop.load %3 : !kgen.pointer<index>
      %11 = index.and %10, %index2233785415175766016
      %12 = index.shrs %11, %index56
      hlcf.yield %12 : index
    } else {
      %10 = pop.load %2 : !kgen.pointer<index>
      hlcf.yield %10 : index
    }
    %8 = index.add %7, %1
    %9 = index.cmp sle(%8, %index23)
    hlcf.if %9 {
      kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg0, %struct) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg0, %arg1) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg0, %struct) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      hlcf.yield
    } else {
      %10 = pop.load %3 : !kgen.pointer<index>
      %11 = index.and %10, %index-9223372036854775808
      %12 = index.cmp ne(%11, %index0)
      %13 = hlcf.if %12 -> index {
        hlcf.yield %index23 : index
      } else {
        %24 = pop.load %3 : !kgen.pointer<index>
        %25 = index.and %24, %index4611686018427387904
        %26 = index.cmp ne(%25, %index0)
        %27 = pop.xor %26, %0
        %28 = hlcf.if %27 -> index {
          %29 = pop.load %2 : !kgen.pointer<index>
          hlcf.yield %29 : index
        } else {
          %29 = pop.load %3 : !kgen.pointer<index>
          %30 = index.shl %29, %index3
          hlcf.yield %30 : index
        }
        hlcf.yield %28 : index
      }
      %14 = index.cmp sle(%8, %13)
      hlcf.if %14 {
        hlcf.yield
      } else {
        kgen.call @"std::collections::string::string::String::_realloc_mutable(::String&,::Int)"(%arg0, %8) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, index) -> ()
        hlcf.yield
      }
      %15 = pop.stack_allocation 1 x struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly> align 1 marked
      pop.stack_alloc.lifetime.start(%15) : !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      %16 = kgen.struct.gep %15[1] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      pop.store %index0, %16 : !kgen.pointer<index>
      %17 = kgen.struct.gep %15[2] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      pop.store %arg0, %17 : !kgen.pointer<pointer<struct<(pointer<none>, index, index) memoryOnly>>>
      kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"(%15, %struct) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=1"(%15, %arg1) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"(%15, %struct) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      %18 = pop.load %17 : !kgen.pointer<pointer<struct<(pointer<none>, index, index) memoryOnly>>>
      %19 = kgen.struct.gep %15[0] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      %20 = kgen.struct.gep %19[0] : <struct<(array<4096, scalar<ui8>>) memoryOnly>>
      %21 = pop.pointer.bitcast %20 : !kgen.pointer<array<4096, scalar<ui8>>> to !kgen.pointer<none>
      %22 = pop.load %16 : !kgen.pointer<index>
      %23 = kgen.struct.create(%21, %22) : !kgen.struct<(pointer<none>, index)>
      kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%18, %23) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      pop.store %index0, %16 : !kgen.pointer<index>
      pop.stack_alloc.lifetime.end(%15) : !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
      hlcf.yield
    }
    kgen.return
  }
  kgen.func @"std::collections::string::string::String::_alloc(::Int)"(%arg0: index) -> !kgen.pointer<none> {
    %simd = kgen.param.constant: scalar<index> = <1>
    %index1 = kgen.param.constant = <1>
    %index8 = kgen.param.constant = <8>
    %index0 = kgen.param.constant = <0>
    %index2305843009213693952 = kgen.param.constant = <2305843009213693952>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %0 = kgen.param.constant: i1 = <1>
    %1 = kgen.param.constant: i1 = <0>
    %index37 = kgen.param.constant = <37>
    %string = kgen.param.constant: string = <"alloc failed: returned a null pointer">
    %index-1 = kgen.param.constant = <-1>
    %pointer = kgen.param.constant: pointer<none> = <0>
    %string_0 = kgen.param.constant: string = <"">
    %string_1 = kgen.param.constant: string = <": ">
    %index2 = kgen.param.constant = <2>
    %string_2 = kgen.param.constant: string = <" ">
    %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle9, const_global, [], [])], []}, 0, 0>, 6 }>
    %struct_3 = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle1, const_global, [], [])], []}, 0, 0>, 1 }>
    %idx-8 = index.constant -8
    %index244 = kgen.param.constant = <244>
    %index14 = kgen.param.constant = <14>
    %struct_4 = kgen.param.constant: struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)> = <{ [{ -1, 0, { 0, 0 } }] }>
    %index54 = kgen.param.constant = <54>
    %string_5 = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/memory/unsafe_pointer.mojo">
    %2 = index.add %arg0, %index8
    %3 = pop.string.address %string_5
    %4 = pop.pointer.bitcast %3 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %5 = kgen.struct.create(%4, %index54) : !kgen.struct<(pointer<none>, index)>
    %6 = kgen.struct.create(%index244, %index14, %5) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %7 = pop.string.address %string_2
    %8 = pop.pointer.bitcast %7 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %9 = pop.string.address %string_1
    %10 = pop.pointer.bitcast %9 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %11 = pop.string.address %string_0
    %12 = pop.pointer.bitcast %11 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %13 = pop.string.address %string
    %14 = pop.pointer.bitcast %13 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %15 = kgen.struct.create(%12, %index0) : !kgen.struct<(pointer<none>, index)>
    %16 = pop.stack_allocation 1 x pointer<scalar<ui8>> marked
    %17 = pop.aligned_alloc %index1, %2 : <scalar<ui8>>
    pop.stack_alloc.lifetime.start(%16) : !kgen.pointer<pointer<scalar<ui8>>>
    pop.store %17, %16 : !kgen.pointer<pointer<scalar<ui8>>>
    %18 = pop.pointer.bitcast %16 : !kgen.pointer<pointer<scalar<ui8>>> to !kgen.pointer<struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>>
    %19 = kgen.struct.gep %18[0] : <struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>>
    %20 = pop.load %19 : !kgen.pointer<struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>>
    %21 = kgen.struct.extract %20[0] : <(struct<(struct<(pointer<none>) memoryOnly>)>)>
    %22 = kgen.struct.extract %21[0] : <(struct<(pointer<none>) memoryOnly>)>
    %23 = kgen.struct.extract %22[0] : <(pointer<none>) memoryOnly>
    %24 = pop.stack_allocation 1 x pointer<none>
    %25 = pop.pointer.bitcast %24 : !kgen.pointer<pointer<none>> to !kgen.pointer<struct<(array<1, pointer<none>>)>>
    %26 = kgen.struct.gep %25[0] : <struct<(array<1, pointer<none>>)>>
    %27 = pop.pointer.bitcast %26 : !kgen.pointer<array<1, pointer<none>>> to !kgen.pointer<pointer<none>>
    %28 = pop.pointer_to_index %23 : <none>
    %29 = index.cmp eq(%28, %index0)
    %30 = pop.select %29, %index0, %index-1 : index
    %31 = index.cmp eq(%30, %index-1)
    hlcf.if %31 {
      pop.store %23, %27 : !kgen.pointer<pointer<none>>
      hlcf.yield
    } else {
      pop.store %pointer, %24 : !kgen.pointer<pointer<none>>
      hlcf.yield
    }
    %32 = pop.load %24 : !kgen.pointer<pointer<none>>
    pop.stack_alloc.lifetime.end(%16) : !kgen.pointer<pointer<scalar<ui8>>>
    %33 = pop.pointer_to_index %32 : <none>
    %34 = index.cmp eq(%33, %index0)
    %35 = pop.select %34, %index0, %index-1 : index
    %36 = index.cmp eq(%35, %index-1)
    %37 = pop.xor %36, %0
    %38 = kgen.struct.create(%37, %1) : !kgen.struct<(i1, i1)>
    %39 = pop.call_llvm_intrinsic side_effecting<0> "llvm.expect", (%38) : (!kgen.struct<(i1, i1)>) -> i1
    hlcf.if %39 {
      %45 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
      pop.stack_alloc.lifetime.start(%45) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
      %46 = kgen.struct.gep %45[1] : <struct<(pointer<none>, index, index) memoryOnly>>
      pop.store %index37, %46 : !kgen.pointer<index>
      %47 = kgen.struct.gep %45[0] : <struct<(pointer<none>, index, index) memoryOnly>>
      pop.store %14, %47 : !kgen.pointer<pointer<none>>
      %48 = kgen.struct.gep %45[2] : <struct<(pointer<none>, index, index) memoryOnly>>
      pop.store %index2305843009213693952, %48 : !kgen.pointer<index>
      %49 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
      %50 = pop.pointer.bitcast %49 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
      pop.store %struct_4, %49 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
      %51 = pop.pointer.bitcast %49 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
      %52 = pop.load %51 : !kgen.pointer<index>
      %53 = index.cmp eq(%52, %index-1)
      %54 = pop.select %53, %index0, %index-1 : index
      %55 = index.cmp eq(%54, %index-1)
      %56 = hlcf.if %55 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
        %74 = pop.load %50 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
        hlcf.yield %74 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
      } else {
        hlcf.yield %6 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
      }
      %57 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
      pop.stack_alloc.lifetime.start(%57) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
      %58 = kgen.struct.gep %57[1] : <struct<(pointer<none>, index, index) memoryOnly>>
      pop.store %index1, %58 : !kgen.pointer<index>
      %59 = kgen.struct.gep %57[0] : <struct<(pointer<none>, index, index) memoryOnly>>
      pop.store %8, %59 : !kgen.pointer<pointer<none>>
      %60 = kgen.struct.gep %57[2] : <struct<(pointer<none>, index, index) memoryOnly>>
      pop.store %index2305843009213693952, %60 : !kgen.pointer<index>
      %61 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
      pop.stack_alloc.lifetime.start(%61) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
      %62 = kgen.struct.gep %61[1] : <struct<(pointer<none>, index, index) memoryOnly>>
      pop.store %index2, %62 : !kgen.pointer<index>
      %63 = kgen.struct.gep %61[0] : <struct<(pointer<none>, index, index) memoryOnly>>
      pop.store %10, %63 : !kgen.pointer<pointer<none>>
      %64 = kgen.struct.gep %61[2] : <struct<(pointer<none>, index, index) memoryOnly>>
      pop.store %index2305843009213693952, %64 : !kgen.pointer<index>
      kgen.call @"std::io::io::print[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?],flush:::Bool,file:::FileDescriptor$),Ts.values`=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::reflection::location::SourceLocation\22>>, struct<(index, index, struct<(pointer<none>, index)>)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]]"(%struct, %57, %56, %61, %45, %15, %struct_3, %0, %index1) : (!kgen.struct<(pointer<none>, index)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index)>, i1, index owned) -> ()
      %65 = pop.load %60 : !kgen.pointer<index>
      %66 = index.and %65, %index4611686018427387904
      %67 = index.cmp ne(%66, %index0)
      hlcf.if %67 {
        %74 = pop.load %59 : !kgen.pointer<pointer<none>>
        %75 = pop.pointer.bitcast %74 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
        %76 = pop.offset %75[%idx-8] : !kgen.pointer<scalar<ui8>>
        %77 = pop.pointer.bitcast %76 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
        %78 = kgen.struct.gep %77[0] : <struct<(scalar<index>) memoryOnly>>
        %79 = pop.atomic.rmw sub(%78, %simd) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
        %80 = pop.cmp eq(%79, %simd) : <1, index>
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
      %68 = pop.load %64 : !kgen.pointer<index>
      %69 = index.and %68, %index4611686018427387904
      %70 = index.cmp ne(%69, %index0)
      hlcf.if %70 {
        %74 = pop.load %63 : !kgen.pointer<pointer<none>>
        %75 = pop.pointer.bitcast %74 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
        %76 = pop.offset %75[%idx-8] : !kgen.pointer<scalar<ui8>>
        %77 = pop.pointer.bitcast %76 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
        %78 = kgen.struct.gep %77[0] : <struct<(scalar<index>) memoryOnly>>
        %79 = pop.atomic.rmw sub(%78, %simd) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
        %80 = pop.cmp eq(%79, %simd) : <1, index>
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
      %71 = pop.load %48 : !kgen.pointer<index>
      %72 = index.and %71, %index4611686018427387904
      %73 = index.cmp ne(%72, %index0)
      hlcf.if %73 {
        %74 = pop.load %47 : !kgen.pointer<pointer<none>>
        %75 = pop.pointer.bitcast %74 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
        %76 = pop.offset %75[%idx-8] : !kgen.pointer<scalar<ui8>>
        %77 = pop.pointer.bitcast %76 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
        %78 = kgen.struct.gep %77[0] : <struct<(scalar<index>) memoryOnly>>
        %79 = pop.atomic.rmw sub(%78, %simd) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
        %80 = pop.cmp eq(%79, %simd) : <1, index>
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
      llvm.intr.trap
      hlcf.loop "_loop_0" {
        hlcf.continue "_loop_0"
      }
      kgen.unreachable
    } else {
      hlcf.yield
    }
    %40 = pop.pointer.bitcast %32 : !kgen.pointer<none> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
    %41 = kgen.struct.gep %40[0] : <struct<(scalar<index>) memoryOnly>>
    pop.store %simd, %41 : !kgen.pointer<scalar<index>>
    %42 = pop.pointer.bitcast %32 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
    %43 = pop.offset %42[%index8] : !kgen.pointer<scalar<ui8>>
    %44 = pop.pointer.bitcast %43 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<none>
    kgen.return %44 : !kgen.pointer<none>
  }
  kgen.func @"std::collections::string::string::String::_add[::Bool,LITOrigin[$0._mlir_value],::Origin[$0, $1],::Bool,LITOrigin[$3._mlir_value],::Origin[$3, $4]](::Span[$0, $1, ::SIMD[::DType(uint8), ::Int(1)], $2],::Span[$3, $4, ::SIMD[::DType(uint8), ::Int(1)], $5]),lhs.mut`2x=0,rhs.mut`2x3=0"(%arg0: !kgen.struct<(pointer<none>, index)>, %arg1: !kgen.struct<(pointer<none>, index)>) -> !kgen.struct<(pointer<none>, index, index) memoryOnly> {
    %index0 = kgen.param.constant = <0>
    %index3 = kgen.param.constant = <3>
    %index7 = kgen.param.constant = <7>
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %index23 = kgen.param.constant = <23>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %index8 = kgen.param.constant = <8>
    %index56 = kgen.param.constant = <56>
    %index-2233785415175766017 = kgen.param.constant = <-2233785415175766017>
    %index1 = kgen.param.constant = <1>
    %index16 = kgen.param.constant = <16>
    %index2 = kgen.param.constant = <2>
    %index5 = kgen.param.constant = <5>
    %index32 = kgen.param.constant = <32>
    %simd = kgen.param.constant: scalar<uindex> = <32>
    %idx-8 = index.constant -8
    %idx-4 = index.constant -4
    %0 = kgen.struct.extract %arg0[1] : <(pointer<none>, index)>
    %1 = kgen.struct.extract %arg1[1] : <(pointer<none>, index)>
    %2 = index.add %0, %1
    %3 = index.add %2, %index7
    %4 = index.shrs %3, %index3
    %5 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
    %6 = kgen.struct.gep %5[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %7 = kgen.struct.gep %5[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %8 = kgen.struct.gep %5[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.stack_alloc.lifetime.start(%5) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %9 = index.cmp sle(%2, %index23)
    hlcf.if %9 {
      pop.store %index-9223372036854775808, %8 : !kgen.pointer<index>
      hlcf.yield
    } else {
      pop.store %4, %8 : !kgen.pointer<index>
      %106 = index.shl %4, %index3
      %107 = kgen.call @"std::collections::string::string::String::_alloc(::Int)"(%106) : (index) -> !kgen.pointer<none>
      pop.store %107, %7 : !kgen.pointer<pointer<none>>
      pop.store %index0, %6 : !kgen.pointer<index>
      %108 = pop.load %8 : !kgen.pointer<index>
      %109 = index.or %108, %index4611686018427387904
      pop.store %109, %8 : !kgen.pointer<index>
      hlcf.yield
    }
    %10 = index.shl %2, %index56
    %11 = pop.load %8 : !kgen.pointer<index>
    %12 = index.and %11, %index-9223372036854775808
    %13 = index.cmp ne(%12, %index0)
    hlcf.if %13 {
      %106 = pop.load %8 : !kgen.pointer<index>
      %107 = index.and %106, %index-2233785415175766017
      %108 = index.or %107, %10
      pop.store %108, %8 : !kgen.pointer<index>
      hlcf.yield
    } else {
      pop.store %2, %6 : !kgen.pointer<index>
      hlcf.yield
    }
    %14 = kgen.call @"std::collections::string::string::String::unsafe_ptr_mut(::String&,::Int$)"(%5, %index0) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, index owned) -> !kgen.pointer<none>
    %15 = kgen.struct.extract %arg0[0] : <(pointer<none>, index)>
    %16 = pop.stack_allocation 1 x pointer<none>
    %17 = pop.pointer.bitcast %16 : !kgen.pointer<pointer<none>> to !kgen.pointer<struct<(array<1, pointer<none>>)>>
    %18 = kgen.struct.gep %17[0] : <struct<(array<1, pointer<none>>)>>
    %19 = pop.pointer.bitcast %18 : !kgen.pointer<array<1, pointer<none>>> to !kgen.pointer<pointer<none>>
    pop.store %14, %19 : !kgen.pointer<pointer<none>>
    %20 = pop.load %16 : !kgen.pointer<pointer<none>>
    %21 = pop.pointer.bitcast %20 : !kgen.pointer<none> to !kgen.pointer<scalar<ui64>>
    %22 = pop.pointer.bitcast %20 : !kgen.pointer<none> to !kgen.pointer<scalar<ui32>>
    %23 = pop.pointer.bitcast %20 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
    %24 = pop.offset %23[%index1] : !kgen.pointer<scalar<ui8>>
    %25 = pop.offset %23[%0] : !kgen.pointer<scalar<ui8>>
    %26 = pop.offset %25[%idx-8] : !kgen.pointer<scalar<ui8>>
    %27 = pop.pointer.bitcast %26 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui64>>
    %28 = pop.offset %25[%idx-4] : !kgen.pointer<scalar<ui8>>
    %29 = pop.pointer.bitcast %28 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui32>>
    %30 = pop.stack_allocation 1 x pointer<none>
    %31 = pop.pointer.bitcast %30 : !kgen.pointer<pointer<none>> to !kgen.pointer<struct<(array<1, pointer<none>>)>>
    %32 = kgen.struct.gep %31[0] : <struct<(array<1, pointer<none>>)>>
    %33 = pop.pointer.bitcast %32 : !kgen.pointer<array<1, pointer<none>>> to !kgen.pointer<pointer<none>>
    pop.store %15, %33 : !kgen.pointer<pointer<none>>
    %34 = pop.load %30 : !kgen.pointer<pointer<none>>
    %35 = pop.cast_from_builtin %0 : index to !pop.scalar<index>
    %36 = pop.cast %35 : !pop.scalar<index> to !pop.scalar<uindex>
    %37 = index.cmp sge(%0, %index8)
    %38 = index.cmp slt(%0, %index5)
    %39 = index.cmp sle(%0, %index2)
    %40 = index.sub %0, %index2
    %41 = pop.offset %23[%40] : !kgen.pointer<scalar<ui8>>
    %42 = index.cmp sle(%0, %index16)
    %43 = index.sub %0, %index1
    %44 = pop.offset %23[%43] : !kgen.pointer<scalar<ui8>>
    %45 = pop.pointer.bitcast %34 : !kgen.pointer<none> to !kgen.pointer<scalar<ui64>>
    %46 = pop.pointer.bitcast %34 : !kgen.pointer<none> to !kgen.pointer<scalar<ui32>>
    %47 = pop.pointer.bitcast %34 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
    %48 = pop.offset %47[%40] : !kgen.pointer<scalar<ui8>>
    %49 = pop.offset %47[%43] : !kgen.pointer<scalar<ui8>>
    %50 = pop.offset %47[%index1] : !kgen.pointer<scalar<ui8>>
    %51 = pop.offset %47[%0] : !kgen.pointer<scalar<ui8>>
    %52 = pop.offset %51[%idx-8] : !kgen.pointer<scalar<ui8>>
    %53 = pop.pointer.bitcast %52 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui64>>
    %54 = pop.offset %51[%idx-4] : !kgen.pointer<scalar<ui8>>
    %55 = pop.pointer.bitcast %54 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui32>>
    %56 = index.cmp eq(%0, %index0)
    hlcf.if %56 {
      hlcf.yield
    } else {
      hlcf.if %38 {
        %106 = pop.load %47 : !kgen.pointer<scalar<ui8>>
        pop.store %106, %23 : !kgen.pointer<scalar<ui8>>
        %107 = pop.load %49 : !kgen.pointer<scalar<ui8>>
        pop.store %107, %44 : !kgen.pointer<scalar<ui8>>
        hlcf.if %39 {
          hlcf.yield
        } else {
          %108 = pop.load %50 : !kgen.pointer<scalar<ui8>>
          pop.store %108, %24 : !kgen.pointer<scalar<ui8>>
          %109 = pop.load %48 : !kgen.pointer<scalar<ui8>>
          pop.store %109, %41 : !kgen.pointer<scalar<ui8>>
          hlcf.yield
        }
        hlcf.yield
      } else {
        hlcf.if %42 {
          hlcf.if %37 {
            %106 = pop.load volatile<0> invariant<0> nontemporal<0> %45 align<1> : !kgen.pointer<scalar<ui64>>
            pop.store volatile<0> nontemporal<0> %106, %21 align<1> : !kgen.pointer<scalar<ui64>>
            %107 = pop.load volatile<0> invariant<0> nontemporal<0> %53 align<1> : !kgen.pointer<scalar<ui64>>
            pop.store volatile<0> nontemporal<0> %107, %27 align<1> : !kgen.pointer<scalar<ui64>>
            hlcf.yield
          } else {
            %106 = pop.load volatile<0> invariant<0> nontemporal<0> %46 align<1> : !kgen.pointer<scalar<ui32>>
            pop.store volatile<0> nontemporal<0> %106, %22 align<1> : !kgen.pointer<scalar<ui32>>
            %107 = pop.load volatile<0> invariant<0> nontemporal<0> %55 align<1> : !kgen.pointer<scalar<ui32>>
            pop.store volatile<0> nontemporal<0> %107, %29 align<1> : !kgen.pointer<scalar<ui32>>
            hlcf.yield
          }
          hlcf.yield
        } else {
          %106 = pop.floordiv %36, %simd : !pop.scalar<uindex>
          %107 = pop.mul %106, %simd : !pop.scalar<uindex>
          %108 = pop.cast fast %107 : !pop.scalar<uindex> to !pop.scalar<index>
          %109 = pop.cast_to_builtin %108 : !pop.scalar<index> to index
          hlcf.loop "_loop_0" (%arg2 = %index0 : index) {
            %111 = index.add %arg2, %index32
            %112 = index.sub %109, %arg2
            %113 = index.cmp slt(%arg2, %109)
            %114 = pop.select %113, %112, %index0 : index
            %115 = index.cmp sle(%114, %index0)
            %116 = pop.select %115, %arg2, %111 : index
            %117:2 = lit.try "try0" -> index, index {
              hlcf.if %115 {
                lit.try.raise "try0" %116, %arg2 : index, index
              } else {
                hlcf.yield
              }
              lit.try.yield %116, %arg2 : index, index
            } except (%arg3: index, %arg4: index) {
              hlcf.break "_loop_0"
            } else (%arg3: index, %arg4: index) {
              lit.try.yield %arg3, %arg4 : index, index
            }
            %118 = pop.offset %47[%117#1] : !kgen.pointer<scalar<ui8>>
            %119 = pop.pointer.bitcast %118 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<simd<32, ui8>>
            %120 = pop.load volatile<0> invariant<0> nontemporal<0> %119 align<1> : !kgen.pointer<simd<32, ui8>>
            %121 = pop.offset %23[%117#1] : !kgen.pointer<scalar<ui8>>
            %122 = pop.pointer.bitcast %121 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<simd<32, ui8>>
            pop.store volatile<0> nontemporal<0> %120, %122 align<1> : !kgen.pointer<simd<32, ui8>>
            hlcf.continue "_loop_0" %117#0 : index
          }
          %110 = index.maxs %109, %0
          hlcf.loop "_loop_2" (%arg2 = %109 : index) {
            %111 = index.add %arg2, %index1
            %112 = index.cmp eq(%arg2, %110)
            %113 = pop.select %112, %arg2, %111 : index
            %114:2 = lit.try "try2" -> index, index {
              hlcf.if %112 {
                lit.try.raise "try2" %113, %arg2 : index, index
              } else {
                hlcf.yield
              }
              lit.try.yield %113, %arg2 : index, index
            } except (%arg3: index, %arg4: index) {
              hlcf.break "_loop_2"
            } else (%arg3: index, %arg4: index) {
              lit.try.yield %arg3, %arg4 : index, index
            }
            %115 = pop.offset %47[%114#1] : !kgen.pointer<scalar<ui8>>
            %116 = pop.load volatile<0> invariant<0> nontemporal<0> %115 align<1> : !kgen.pointer<scalar<ui8>>
            %117 = pop.offset %23[%114#1] : !kgen.pointer<scalar<ui8>>
            pop.store volatile<0> nontemporal<0> %116, %117 align<1> : !kgen.pointer<scalar<ui8>>
            hlcf.continue "_loop_2" %114#0 : index
          }
          hlcf.yield
        }
        hlcf.yield
      }
      hlcf.yield
    }
    %57 = pop.pointer.bitcast %14 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
    %58 = pop.offset %57[%0] : !kgen.pointer<scalar<ui8>>
    %59 = pop.pointer.bitcast %58 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<none>
    %60 = kgen.struct.extract %arg1[0] : <(pointer<none>, index)>
    %61 = pop.stack_allocation 1 x pointer<none>
    %62 = pop.pointer.bitcast %61 : !kgen.pointer<pointer<none>> to !kgen.pointer<struct<(array<1, pointer<none>>)>>
    %63 = kgen.struct.gep %62[0] : <struct<(array<1, pointer<none>>)>>
    %64 = pop.pointer.bitcast %63 : !kgen.pointer<array<1, pointer<none>>> to !kgen.pointer<pointer<none>>
    pop.store %59, %64 : !kgen.pointer<pointer<none>>
    %65 = pop.load %61 : !kgen.pointer<pointer<none>>
    %66 = pop.pointer.bitcast %65 : !kgen.pointer<none> to !kgen.pointer<scalar<ui64>>
    %67 = pop.pointer.bitcast %65 : !kgen.pointer<none> to !kgen.pointer<scalar<ui32>>
    %68 = pop.pointer.bitcast %65 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
    %69 = pop.offset %68[%index1] : !kgen.pointer<scalar<ui8>>
    %70 = pop.offset %68[%1] : !kgen.pointer<scalar<ui8>>
    %71 = pop.offset %70[%idx-8] : !kgen.pointer<scalar<ui8>>
    %72 = pop.pointer.bitcast %71 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui64>>
    %73 = pop.offset %70[%idx-4] : !kgen.pointer<scalar<ui8>>
    %74 = pop.pointer.bitcast %73 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui32>>
    %75 = pop.stack_allocation 1 x pointer<none>
    %76 = pop.pointer.bitcast %75 : !kgen.pointer<pointer<none>> to !kgen.pointer<struct<(array<1, pointer<none>>)>>
    %77 = kgen.struct.gep %76[0] : <struct<(array<1, pointer<none>>)>>
    %78 = pop.pointer.bitcast %77 : !kgen.pointer<array<1, pointer<none>>> to !kgen.pointer<pointer<none>>
    pop.store %60, %78 : !kgen.pointer<pointer<none>>
    %79 = pop.load %75 : !kgen.pointer<pointer<none>>
    %80 = pop.cast_from_builtin %1 : index to !pop.scalar<index>
    %81 = pop.cast %80 : !pop.scalar<index> to !pop.scalar<uindex>
    %82 = index.cmp sge(%1, %index8)
    %83 = index.cmp slt(%1, %index5)
    %84 = index.cmp sle(%1, %index2)
    %85 = index.sub %1, %index2
    %86 = pop.offset %68[%85] : !kgen.pointer<scalar<ui8>>
    %87 = index.cmp sle(%1, %index16)
    %88 = index.sub %1, %index1
    %89 = pop.offset %68[%88] : !kgen.pointer<scalar<ui8>>
    %90 = pop.pointer.bitcast %79 : !kgen.pointer<none> to !kgen.pointer<scalar<ui64>>
    %91 = pop.pointer.bitcast %79 : !kgen.pointer<none> to !kgen.pointer<scalar<ui32>>
    %92 = pop.pointer.bitcast %79 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
    %93 = pop.offset %92[%85] : !kgen.pointer<scalar<ui8>>
    %94 = pop.offset %92[%88] : !kgen.pointer<scalar<ui8>>
    %95 = pop.offset %92[%index1] : !kgen.pointer<scalar<ui8>>
    %96 = pop.offset %92[%1] : !kgen.pointer<scalar<ui8>>
    %97 = pop.offset %96[%idx-8] : !kgen.pointer<scalar<ui8>>
    %98 = pop.pointer.bitcast %97 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui64>>
    %99 = pop.offset %96[%idx-4] : !kgen.pointer<scalar<ui8>>
    %100 = pop.pointer.bitcast %99 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui32>>
    %101 = index.cmp eq(%1, %index0)
    hlcf.if %101 {
      hlcf.yield
    } else {
      hlcf.if %83 {
        %106 = pop.load %92 : !kgen.pointer<scalar<ui8>>
        pop.store %106, %68 : !kgen.pointer<scalar<ui8>>
        %107 = pop.load %94 : !kgen.pointer<scalar<ui8>>
        pop.store %107, %89 : !kgen.pointer<scalar<ui8>>
        hlcf.if %84 {
          hlcf.yield
        } else {
          %108 = pop.load %95 : !kgen.pointer<scalar<ui8>>
          pop.store %108, %69 : !kgen.pointer<scalar<ui8>>
          %109 = pop.load %93 : !kgen.pointer<scalar<ui8>>
          pop.store %109, %86 : !kgen.pointer<scalar<ui8>>
          hlcf.yield
        }
        hlcf.yield
      } else {
        hlcf.if %87 {
          hlcf.if %82 {
            %106 = pop.load volatile<0> invariant<0> nontemporal<0> %90 align<1> : !kgen.pointer<scalar<ui64>>
            pop.store volatile<0> nontemporal<0> %106, %66 align<1> : !kgen.pointer<scalar<ui64>>
            %107 = pop.load volatile<0> invariant<0> nontemporal<0> %98 align<1> : !kgen.pointer<scalar<ui64>>
            pop.store volatile<0> nontemporal<0> %107, %72 align<1> : !kgen.pointer<scalar<ui64>>
            hlcf.yield
          } else {
            %106 = pop.load volatile<0> invariant<0> nontemporal<0> %91 align<1> : !kgen.pointer<scalar<ui32>>
            pop.store volatile<0> nontemporal<0> %106, %67 align<1> : !kgen.pointer<scalar<ui32>>
            %107 = pop.load volatile<0> invariant<0> nontemporal<0> %100 align<1> : !kgen.pointer<scalar<ui32>>
            pop.store volatile<0> nontemporal<0> %107, %74 align<1> : !kgen.pointer<scalar<ui32>>
            hlcf.yield
          }
          hlcf.yield
        } else {
          %106 = pop.floordiv %81, %simd : !pop.scalar<uindex>
          %107 = pop.mul %106, %simd : !pop.scalar<uindex>
          %108 = pop.cast fast %107 : !pop.scalar<uindex> to !pop.scalar<index>
          %109 = pop.cast_to_builtin %108 : !pop.scalar<index> to index
          hlcf.loop "_loop_0" (%arg2 = %index0 : index) {
            %111 = index.add %arg2, %index32
            %112 = index.sub %109, %arg2
            %113 = index.cmp slt(%arg2, %109)
            %114 = pop.select %113, %112, %index0 : index
            %115 = index.cmp sle(%114, %index0)
            %116 = pop.select %115, %arg2, %111 : index
            %117:2 = lit.try "try0" -> index, index {
              hlcf.if %115 {
                lit.try.raise "try0" %116, %arg2 : index, index
              } else {
                hlcf.yield
              }
              lit.try.yield %116, %arg2 : index, index
            } except (%arg3: index, %arg4: index) {
              hlcf.break "_loop_0"
            } else (%arg3: index, %arg4: index) {
              lit.try.yield %arg3, %arg4 : index, index
            }
            %118 = pop.offset %92[%117#1] : !kgen.pointer<scalar<ui8>>
            %119 = pop.pointer.bitcast %118 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<simd<32, ui8>>
            %120 = pop.load volatile<0> invariant<0> nontemporal<0> %119 align<1> : !kgen.pointer<simd<32, ui8>>
            %121 = pop.offset %68[%117#1] : !kgen.pointer<scalar<ui8>>
            %122 = pop.pointer.bitcast %121 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<simd<32, ui8>>
            pop.store volatile<0> nontemporal<0> %120, %122 align<1> : !kgen.pointer<simd<32, ui8>>
            hlcf.continue "_loop_0" %117#0 : index
          }
          %110 = index.maxs %109, %1
          hlcf.loop "_loop_2" (%arg2 = %109 : index) {
            %111 = index.add %arg2, %index1
            %112 = index.cmp eq(%arg2, %110)
            %113 = pop.select %112, %arg2, %111 : index
            %114:2 = lit.try "try2" -> index, index {
              hlcf.if %112 {
                lit.try.raise "try2" %113, %arg2 : index, index
              } else {
                hlcf.yield
              }
              lit.try.yield %113, %arg2 : index, index
            } except (%arg3: index, %arg4: index) {
              hlcf.break "_loop_2"
            } else (%arg3: index, %arg4: index) {
              lit.try.yield %arg3, %arg4 : index, index
            }
            %115 = pop.offset %92[%114#1] : !kgen.pointer<scalar<ui8>>
            %116 = pop.load volatile<0> invariant<0> nontemporal<0> %115 align<1> : !kgen.pointer<scalar<ui8>>
            %117 = pop.offset %68[%114#1] : !kgen.pointer<scalar<ui8>>
            pop.store volatile<0> nontemporal<0> %116, %117 align<1> : !kgen.pointer<scalar<ui8>>
            hlcf.continue "_loop_2" %114#0 : index
          }
          hlcf.yield
        }
        hlcf.yield
      }
      hlcf.yield
    }
    %102 = pop.load %7 : !kgen.pointer<pointer<none>>
    %103 = pop.load %6 : !kgen.pointer<index>
    %104 = pop.load %8 : !kgen.pointer<index>
    pop.stack_alloc.lifetime.end(%5) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %105 = kgen.struct.create(%102, %103, %104) : !kgen.struct<(pointer<none>, index, index) memoryOnly>
    kgen.return %105 : !kgen.struct<(pointer<none>, index, index) memoryOnly>
  }
  kgen.func @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg0: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, %arg1: !kgen.struct<(pointer<none>, index)>) {
    %index-2233785415175766017 = kgen.param.constant = <-2233785415175766017>
    %idx-4 = index.constant -4
    %idx-8 = index.constant -8
    %simd = kgen.param.constant: scalar<uindex> = <32>
    %index32 = kgen.param.constant = <32>
    %index8 = kgen.param.constant = <8>
    %index5 = kgen.param.constant = <5>
    %index2 = kgen.param.constant = <2>
    %index16 = kgen.param.constant = <16>
    %index1 = kgen.param.constant = <1>
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %index2233785415175766016 = kgen.param.constant = <2233785415175766016>
    %index56 = kgen.param.constant = <56>
    %index-2305843009213693953 = kgen.param.constant = <-2305843009213693953>
    %index0 = kgen.param.constant = <0>
    %0 = kgen.struct.extract %arg1[0] : <(pointer<none>, index)>
    %1 = kgen.struct.gep %arg0[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %2 = kgen.struct.gep %arg0[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %3 = kgen.struct.extract %arg1[1] : <(pointer<none>, index)>
    %4 = index.sub %3, %index1
    %5 = index.cmp sle(%3, %index16)
    %6 = index.sub %3, %index2
    %7 = index.cmp sle(%3, %index2)
    %8 = index.cmp slt(%3, %index5)
    %9 = index.cmp sge(%3, %index8)
    %10 = pop.cast_from_builtin %3 : index to !pop.scalar<index>
    %11 = pop.cast %10 : !pop.scalar<index> to !pop.scalar<uindex>
    %12 = index.cmp eq(%3, %index0)
    hlcf.if %12 {
      hlcf.yield
    } else {
      %13 = pop.load %1 : !kgen.pointer<index>
      %14 = index.and %13, %index-9223372036854775808
      %15 = index.cmp ne(%14, %index0)
      %16 = hlcf.if %15 -> index {
        %60 = pop.load %1 : !kgen.pointer<index>
        %61 = index.and %60, %index2233785415175766016
        %62 = index.shrs %61, %index56
        hlcf.yield %62 : index
      } else {
        %60 = pop.load %2 : !kgen.pointer<index>
        hlcf.yield %60 : index
      }
      %17 = index.add %16, %3
      %18 = kgen.call @"std::collections::string::string::String::unsafe_ptr_mut(::String&,::Int$)"(%arg0, %17) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, index owned) -> !kgen.pointer<none>
      %19 = pop.pointer.bitcast %18 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %20 = pop.offset %19[%16] : !kgen.pointer<scalar<ui8>>
      %21 = pop.pointer.bitcast %20 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<none>
      %22 = pop.stack_allocation 1 x pointer<none>
      %23 = pop.pointer.bitcast %22 : !kgen.pointer<pointer<none>> to !kgen.pointer<struct<(array<1, pointer<none>>)>>
      %24 = kgen.struct.gep %23[0] : <struct<(array<1, pointer<none>>)>>
      %25 = pop.pointer.bitcast %24 : !kgen.pointer<array<1, pointer<none>>> to !kgen.pointer<pointer<none>>
      pop.store %21, %25 : !kgen.pointer<pointer<none>>
      %26 = pop.load %22 : !kgen.pointer<pointer<none>>
      %27 = pop.pointer.bitcast %26 : !kgen.pointer<none> to !kgen.pointer<scalar<ui64>>
      %28 = pop.pointer.bitcast %26 : !kgen.pointer<none> to !kgen.pointer<scalar<ui32>>
      %29 = pop.pointer.bitcast %26 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %30 = pop.offset %29[%6] : !kgen.pointer<scalar<ui8>>
      %31 = pop.offset %29[%4] : !kgen.pointer<scalar<ui8>>
      %32 = pop.offset %29[%index1] : !kgen.pointer<scalar<ui8>>
      %33 = pop.offset %29[%3] : !kgen.pointer<scalar<ui8>>
      %34 = pop.offset %33[%idx-8] : !kgen.pointer<scalar<ui8>>
      %35 = pop.pointer.bitcast %34 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui64>>
      %36 = pop.offset %33[%idx-4] : !kgen.pointer<scalar<ui8>>
      %37 = pop.pointer.bitcast %36 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui32>>
      %38 = pop.stack_allocation 1 x pointer<none>
      %39 = pop.pointer.bitcast %38 : !kgen.pointer<pointer<none>> to !kgen.pointer<struct<(array<1, pointer<none>>)>>
      %40 = kgen.struct.gep %39[0] : <struct<(array<1, pointer<none>>)>>
      %41 = pop.pointer.bitcast %40 : !kgen.pointer<array<1, pointer<none>>> to !kgen.pointer<pointer<none>>
      pop.store %0, %41 : !kgen.pointer<pointer<none>>
      %42 = pop.load %38 : !kgen.pointer<pointer<none>>
      %43 = pop.pointer.bitcast %42 : !kgen.pointer<none> to !kgen.pointer<scalar<ui64>>
      %44 = pop.pointer.bitcast %42 : !kgen.pointer<none> to !kgen.pointer<scalar<ui32>>
      %45 = pop.pointer.bitcast %42 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %46 = pop.offset %45[%6] : !kgen.pointer<scalar<ui8>>
      %47 = pop.offset %45[%4] : !kgen.pointer<scalar<ui8>>
      %48 = pop.offset %45[%index1] : !kgen.pointer<scalar<ui8>>
      %49 = pop.offset %45[%3] : !kgen.pointer<scalar<ui8>>
      %50 = pop.offset %49[%idx-8] : !kgen.pointer<scalar<ui8>>
      %51 = pop.pointer.bitcast %50 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui64>>
      %52 = pop.offset %49[%idx-4] : !kgen.pointer<scalar<ui8>>
      %53 = pop.pointer.bitcast %52 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui32>>
      hlcf.if %8 {
        %60 = pop.load %45 : !kgen.pointer<scalar<ui8>>
        pop.store %60, %29 : !kgen.pointer<scalar<ui8>>
        %61 = pop.load %47 : !kgen.pointer<scalar<ui8>>
        pop.store %61, %31 : !kgen.pointer<scalar<ui8>>
        hlcf.if %7 {
          hlcf.yield
        } else {
          %62 = pop.load %48 : !kgen.pointer<scalar<ui8>>
          pop.store %62, %32 : !kgen.pointer<scalar<ui8>>
          %63 = pop.load %46 : !kgen.pointer<scalar<ui8>>
          pop.store %63, %30 : !kgen.pointer<scalar<ui8>>
          hlcf.yield
        }
        hlcf.yield
      } else {
        hlcf.if %5 {
          hlcf.if %9 {
            %60 = pop.load volatile<0> invariant<0> nontemporal<0> %43 align<1> : !kgen.pointer<scalar<ui64>>
            pop.store volatile<0> nontemporal<0> %60, %27 align<1> : !kgen.pointer<scalar<ui64>>
            %61 = pop.load volatile<0> invariant<0> nontemporal<0> %51 align<1> : !kgen.pointer<scalar<ui64>>
            pop.store volatile<0> nontemporal<0> %61, %35 align<1> : !kgen.pointer<scalar<ui64>>
            hlcf.yield
          } else {
            %60 = pop.load volatile<0> invariant<0> nontemporal<0> %44 align<1> : !kgen.pointer<scalar<ui32>>
            pop.store volatile<0> nontemporal<0> %60, %28 align<1> : !kgen.pointer<scalar<ui32>>
            %61 = pop.load volatile<0> invariant<0> nontemporal<0> %53 align<1> : !kgen.pointer<scalar<ui32>>
            pop.store volatile<0> nontemporal<0> %61, %37 align<1> : !kgen.pointer<scalar<ui32>>
            hlcf.yield
          }
          hlcf.yield
        } else {
          %60 = pop.floordiv %11, %simd : !pop.scalar<uindex>
          %61 = pop.mul %60, %simd : !pop.scalar<uindex>
          %62 = pop.cast fast %61 : !pop.scalar<uindex> to !pop.scalar<index>
          %63 = pop.cast_to_builtin %62 : !pop.scalar<index> to index
          hlcf.loop "_loop_0" (%arg2 = %index0 : index) {
            %65 = index.add %arg2, %index32
            %66 = index.sub %63, %arg2
            %67 = index.cmp slt(%arg2, %63)
            %68 = pop.select %67, %66, %index0 : index
            %69 = index.cmp sle(%68, %index0)
            %70 = pop.select %69, %arg2, %65 : index
            %71:2 = lit.try "try0" -> index, index {
              hlcf.if %69 {
                lit.try.raise "try0" %70, %arg2 : index, index
              } else {
                hlcf.yield
              }
              lit.try.yield %70, %arg2 : index, index
            } except (%arg3: index, %arg4: index) {
              hlcf.break "_loop_0"
            } else (%arg3: index, %arg4: index) {
              lit.try.yield %arg3, %arg4 : index, index
            }
            %72 = pop.offset %45[%71#1] : !kgen.pointer<scalar<ui8>>
            %73 = pop.pointer.bitcast %72 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<simd<32, ui8>>
            %74 = pop.load volatile<0> invariant<0> nontemporal<0> %73 align<1> : !kgen.pointer<simd<32, ui8>>
            %75 = pop.offset %29[%71#1] : !kgen.pointer<scalar<ui8>>
            %76 = pop.pointer.bitcast %75 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<simd<32, ui8>>
            pop.store volatile<0> nontemporal<0> %74, %76 align<1> : !kgen.pointer<simd<32, ui8>>
            hlcf.continue "_loop_0" %71#0 : index
          }
          %64 = index.maxs %63, %3
          hlcf.loop "_loop_2" (%arg2 = %63 : index) {
            %65 = index.add %arg2, %index1
            %66 = index.cmp eq(%arg2, %64)
            %67 = pop.select %66, %arg2, %65 : index
            %68:2 = lit.try "try2" -> index, index {
              hlcf.if %66 {
                lit.try.raise "try2" %67, %arg2 : index, index
              } else {
                hlcf.yield
              }
              lit.try.yield %67, %arg2 : index, index
            } except (%arg3: index, %arg4: index) {
              hlcf.break "_loop_2"
            } else (%arg3: index, %arg4: index) {
              lit.try.yield %arg3, %arg4 : index, index
            }
            %69 = pop.offset %45[%68#1] : !kgen.pointer<scalar<ui8>>
            %70 = pop.load volatile<0> invariant<0> nontemporal<0> %69 align<1> : !kgen.pointer<scalar<ui8>>
            %71 = pop.offset %29[%68#1] : !kgen.pointer<scalar<ui8>>
            pop.store volatile<0> nontemporal<0> %70, %71 align<1> : !kgen.pointer<scalar<ui8>>
            hlcf.continue "_loop_2" %68#0 : index
          }
          hlcf.yield
        }
        hlcf.yield
      }
      %54 = index.shl %17, %index56
      %55 = pop.load %1 : !kgen.pointer<index>
      %56 = index.and %55, %index-9223372036854775808
      %57 = index.cmp ne(%56, %index0)
      hlcf.if %57 {
        %60 = pop.load %1 : !kgen.pointer<index>
        %61 = index.and %60, %index-2233785415175766017
        %62 = index.or %61, %54
        pop.store %62, %1 : !kgen.pointer<index>
        hlcf.yield
      } else {
        pop.store %17, %2 : !kgen.pointer<index>
        hlcf.yield
      }
      %58 = pop.load %1 : !kgen.pointer<index>
      %59 = index.and %58, %index-2305843009213693953
      pop.store %59, %1 : !kgen.pointer<index>
      hlcf.yield
    }
    kgen.return
  }
  kgen.func @"std::collections::string::string::String::unsafe_ptr_mut(::String&,::Int$)"(%arg0: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, %arg1: index owned) -> !kgen.pointer<none> {
    %idx-8 = index.constant -8
    %true = index.bool.constant true
    %0 = kgen.param.constant: i1 = <1>
    %index0 = kgen.param.constant = <0>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %simd = kgen.param.constant: scalar<index> = <1>
    %1 = kgen.param.constant: i1 = <0>
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %index3 = kgen.param.constant = <3>
    %index23 = kgen.param.constant = <23>
    %2 = kgen.struct.gep %arg0[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %3 = pop.pointer.bitcast %arg0 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
    %4 = kgen.struct.gep %arg0[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %5 = kgen.struct.gep %arg0[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %6 = pop.load %5 : !kgen.pointer<index>
    %7 = index.and %6, %index-9223372036854775808
    %8 = index.cmp ne(%7, %index0)
    %9 = hlcf.if %8 -> index {
      hlcf.yield %index23 : index
    } else {
      %16 = pop.load %5 : !kgen.pointer<index>
      %17 = index.and %16, %index4611686018427387904
      %18 = index.cmp ne(%17, %index0)
      %19 = pop.xor %18, %0
      %20 = hlcf.if %19 -> index {
        %21 = pop.load %4 : !kgen.pointer<index>
        hlcf.yield %21 : index
      } else {
        %21 = pop.load %5 : !kgen.pointer<index>
        %22 = index.shl %21, %index3
        hlcf.yield %22 : index
      }
      hlcf.yield %20 : index
    }
    %10 = index.maxs %9, %arg1
    %11 = index.cmp sle(%10, %index23)
    hlcf.if %11 {
      %16 = pop.load %5 : !kgen.pointer<index>
      %17 = index.and %16, %index-9223372036854775808
      %18 = index.cmp ne(%17, %index0)
      %19 = pop.xor %18, %0
      hlcf.if %19 {
        kgen.call tail @"std::collections::string::string::String::_inline_string(::String&)"(%arg0) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut) -> ()
        hlcf.yield
      } else {
        hlcf.yield
      }
      hlcf.yield
    } else {
      %16 = pop.load %5 : !kgen.pointer<index>
      %17 = index.and %16, %index4611686018427387904
      %18 = index.cmp ne(%17, %index0)
      %19 = hlcf.if %18 -> i1 {
        %22 = pop.load %2 : !kgen.pointer<pointer<none>>
        %23 = pop.pointer.bitcast %22 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
        %24 = pop.offset %23[%idx-8] : !kgen.pointer<scalar<ui8>>
        %25 = pop.pointer.bitcast %24 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
        %26 = kgen.struct.gep %25[0] : <struct<(scalar<index>) memoryOnly>>
        %27 = pop.load atomic syncscope("") monotonic %26 : !kgen.pointer<scalar<index>>
        %28 = pop.cmp eq(%27, %simd) : <1, index>
        %29 = pop.cast_to_builtin %28 : !pop.scalar<bool> to i1
        hlcf.yield %29 : i1
      } else {
        hlcf.yield %1 : i1
      }
      %20 = pop.xor %19, %0
      %21 = hlcf.if %20 -> i1 {
        hlcf.yield %true : i1
      } else {
        %22 = pop.load %5 : !kgen.pointer<index>
        %23 = index.and %22, %index-9223372036854775808
        %24 = index.cmp ne(%23, %index0)
        %25 = hlcf.if %24 -> index {
          hlcf.yield %index23 : index
        } else {
          %27 = pop.load %5 : !kgen.pointer<index>
          %28 = index.and %27, %index4611686018427387904
          %29 = index.cmp ne(%28, %index0)
          %30 = pop.xor %29, %0
          %31 = hlcf.if %30 -> index {
            %32 = pop.load %4 : !kgen.pointer<index>
            hlcf.yield %32 : index
          } else {
            %32 = pop.load %5 : !kgen.pointer<index>
            %33 = index.shl %32, %index3
            hlcf.yield %33 : index
          }
          hlcf.yield %31 : index
        }
        %26 = index.cmp sgt(%10, %25)
        hlcf.yield %26 : i1
      }
      hlcf.if %21 {
        kgen.call tail @"std::collections::string::string::String::_realloc_mutable(::String&,::Int)"(%arg0, %10) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, index) -> ()
        hlcf.yield
      } else {
        hlcf.yield
      }
      hlcf.yield
    }
    %12 = pop.load %5 : !kgen.pointer<index>
    %13 = index.and %12, %index-9223372036854775808
    %14 = index.cmp ne(%13, %index0)
    %15 = hlcf.if %14 -> !kgen.pointer<none> {
      hlcf.yield %3 : !kgen.pointer<none>
    } else {
      %16 = pop.load %2 : !kgen.pointer<pointer<none>>
      hlcf.yield %16 : !kgen.pointer<none>
    }
    kgen.return %15 : !kgen.pointer<none>
  }
  kgen.func @"std::collections::string::string::String::_inline_string(::String&)"(%arg0: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut) {
    %idx-8 = index.constant -8
    %index-2233785415175766017 = kgen.param.constant = <-2233785415175766017>
    %index1 = kgen.param.constant = <1>
    %index0 = kgen.param.constant = <0>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %simd = kgen.param.constant: scalar<index> = <1>
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %index56 = kgen.param.constant = <56>
    %index2233785415175766016 = kgen.param.constant = <2233785415175766016>
    %0 = kgen.struct.gep %arg0[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %1 = pop.pointer.bitcast %arg0 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
    %2 = kgen.struct.gep %arg0[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %3 = kgen.struct.gep %arg0[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %4 = pop.load %3 : !kgen.pointer<index>
    %5 = index.and %4, %index-9223372036854775808
    %6 = index.cmp ne(%5, %index0)
    %7 = hlcf.if %6 -> index {
      %26 = pop.load %3 : !kgen.pointer<index>
      %27 = index.and %26, %index2233785415175766016
      %28 = index.shrs %27, %index56
      hlcf.yield %28 : index
    } else {
      %26 = pop.load %2 : !kgen.pointer<index>
      hlcf.yield %26 : index
    }
    %8 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
    pop.stack_alloc.lifetime.start(%8) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %9 = kgen.struct.gep %8[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index-9223372036854775808, %9 : !kgen.pointer<index>
    %10 = kgen.struct.gep %8[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %11 = index.shl %7, %index56
    %12 = pop.load %9 : !kgen.pointer<index>
    %13 = index.and %12, %index-9223372036854775808
    %14 = index.cmp ne(%13, %index0)
    hlcf.if %14 {
      %26 = pop.load %9 : !kgen.pointer<index>
      %27 = index.and %26, %index-2233785415175766017
      %28 = index.or %27, %11
      pop.store %28, %9 : !kgen.pointer<index>
      hlcf.yield
    } else {
      pop.store %7, %10 : !kgen.pointer<index>
      hlcf.yield
    }
    %15 = pop.pointer.bitcast %8 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<scalar<ui8>>
    %16 = pop.load %3 : !kgen.pointer<index>
    %17 = index.and %16, %index-9223372036854775808
    %18 = index.cmp ne(%17, %index0)
    %19 = hlcf.if %18 -> !kgen.pointer<none> {
      hlcf.yield %1 : !kgen.pointer<none>
    } else {
      %26 = pop.load %0 : !kgen.pointer<pointer<none>>
      hlcf.yield %26 : !kgen.pointer<none>
    }
    %20 = pop.pointer.bitcast %19 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
    %21 = index.maxs %7, %index0
    hlcf.loop "_loop_0" (%arg1 = %21 : index) {
      %26 = index.sub %21, %arg1
      %27 = index.sub %arg1, %index1
      %28 = index.cmp eq(%arg1, %index0)
      %29:2 = lit.try "try0" -> index, index {
        %33 = pop.select %28, %arg1, %27 : index
        hlcf.if %28 {
          %34 = pop.load %3 : !kgen.pointer<index>
          %35 = index.and %34, %index4611686018427387904
          %36 = index.cmp ne(%35, %index0)
          hlcf.if %36 {
            %37 = pop.load %0 : !kgen.pointer<pointer<none>>
            %38 = pop.pointer.bitcast %37 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
            %39 = pop.offset %38[%idx-8] : !kgen.pointer<scalar<ui8>>
            %40 = pop.pointer.bitcast %39 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
            %41 = kgen.struct.gep %40[0] : <struct<(scalar<index>) memoryOnly>>
            %42 = pop.atomic.rmw sub(%41, %simd) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
            %43 = pop.cmp eq(%42, %simd) : <1, index>
            %44 = pop.cast_to_builtin %43 : !pop.scalar<bool> to i1
            hlcf.if %44 {
              pop.fence syncscope("") acquire
              pop.aligned_free %39 : <scalar<ui8>>
              hlcf.yield
            } else {
              hlcf.yield
            }
            hlcf.yield
          } else {
            hlcf.yield
          }
          lit.try.raise "try0" %33, %26 : index, index
        } else {
          hlcf.yield
        }
        lit.try.yield %33, %26 : index, index
      } except (%arg2: index, %arg3: index) {
        hlcf.break "_loop_0"
      } else (%arg2: index, %arg3: index) {
        lit.try.yield %arg2, %arg3 : index, index
      }
      %30 = pop.offset %15[%29#1] : !kgen.pointer<scalar<ui8>>
      %31 = pop.offset %20[%29#1] : !kgen.pointer<scalar<ui8>>
      %32 = pop.load %31 : !kgen.pointer<scalar<ui8>>
      pop.store %32, %30 : !kgen.pointer<scalar<ui8>>
      hlcf.continue "_loop_0" %29#0 : index
    }
    %22 = kgen.struct.gep %8[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %23 = pop.load %22 : !kgen.pointer<pointer<none>>
    pop.store %23, %0 : !kgen.pointer<pointer<none>>
    %24 = pop.load %10 : !kgen.pointer<index>
    pop.store %24, %2 : !kgen.pointer<index>
    %25 = pop.load %9 : !kgen.pointer<index>
    pop.store %25, %3 : !kgen.pointer<index>
    pop.stack_alloc.lifetime.end(%8) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    kgen.return
  }
  kgen.func @"std::collections::string::string::String::_realloc_mutable(::String&,::Int)"(%arg0: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, %arg1: index) {
    %idx-4 = index.constant -4
    %idx-8 = index.constant -8
    %simd = kgen.param.constant: scalar<uindex> = <32>
    %index32 = kgen.param.constant = <32>
    %index5 = kgen.param.constant = <5>
    %index16 = kgen.param.constant = <16>
    %index1 = kgen.param.constant = <1>
    %0 = kgen.param.constant: i1 = <1>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %index8 = kgen.param.constant = <8>
    %simd_0 = kgen.param.constant: scalar<index> = <1>
    %index0 = kgen.param.constant = <0>
    %index23 = kgen.param.constant = <23>
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %index2 = kgen.param.constant = <2>
    %index7 = kgen.param.constant = <7>
    %index3 = kgen.param.constant = <3>
    %index56 = kgen.param.constant = <56>
    %index2233785415175766016 = kgen.param.constant = <2233785415175766016>
    %1 = kgen.struct.gep %arg0[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %2 = pop.pointer.bitcast %arg0 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
    %3 = kgen.struct.gep %arg0[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %4 = kgen.struct.gep %arg0[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %5 = pop.load %4 : !kgen.pointer<index>
    %6 = index.and %5, %index-9223372036854775808
    %7 = index.cmp ne(%6, %index0)
    %8 = hlcf.if %7 -> index {
      %68 = pop.load %4 : !kgen.pointer<index>
      %69 = index.and %68, %index2233785415175766016
      %70 = index.shrs %69, %index56
      hlcf.yield %70 : index
    } else {
      %68 = pop.load %3 : !kgen.pointer<index>
      hlcf.yield %68 : index
    }
    %9 = pop.load %4 : !kgen.pointer<index>
    %10 = index.and %9, %index-9223372036854775808
    %11 = index.cmp ne(%10, %index0)
    %12 = hlcf.if %11 -> !kgen.pointer<none> {
      hlcf.yield %2 : !kgen.pointer<none>
    } else {
      %68 = pop.load %1 : !kgen.pointer<pointer<none>>
      hlcf.yield %68 : !kgen.pointer<none>
    }
    %13 = pop.load %4 : !kgen.pointer<index>
    %14 = index.and %13, %index-9223372036854775808
    %15 = index.cmp ne(%14, %index0)
    %16 = hlcf.if %15 -> index {
      hlcf.yield %index23 : index
    } else {
      %68 = pop.load %4 : !kgen.pointer<index>
      %69 = index.and %68, %index4611686018427387904
      %70 = index.cmp ne(%69, %index0)
      %71 = pop.xor %70, %0
      %72 = hlcf.if %71 -> index {
        %73 = pop.load %3 : !kgen.pointer<index>
        hlcf.yield %73 : index
      } else {
        %73 = pop.load %4 : !kgen.pointer<index>
        %74 = index.shl %73, %index3
        hlcf.yield %74 : index
      }
      hlcf.yield %72 : index
    }
    %17 = index.mul %16, %index2
    %18 = index.maxs %arg1, %17
    %19 = index.add %18, %index7
    %20 = index.shrs %19, %index3
    %21 = index.shl %20, %index3
    %22 = kgen.call tail @"std::collections::string::string::String::_alloc(::Int)"(%21) : (index) -> !kgen.pointer<none>
    %23 = pop.stack_allocation 1 x pointer<none>
    %24 = pop.pointer.bitcast %23 : !kgen.pointer<pointer<none>> to !kgen.pointer<struct<(array<1, pointer<none>>)>>
    %25 = kgen.struct.gep %24[0] : <struct<(array<1, pointer<none>>)>>
    %26 = pop.pointer.bitcast %25 : !kgen.pointer<array<1, pointer<none>>> to !kgen.pointer<pointer<none>>
    pop.store %22, %26 : !kgen.pointer<pointer<none>>
    %27 = pop.load %23 : !kgen.pointer<pointer<none>>
    %28 = pop.pointer.bitcast %27 : !kgen.pointer<none> to !kgen.pointer<scalar<ui64>>
    %29 = pop.pointer.bitcast %27 : !kgen.pointer<none> to !kgen.pointer<scalar<ui32>>
    %30 = pop.pointer.bitcast %27 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
    %31 = pop.offset %30[%index1] : !kgen.pointer<scalar<ui8>>
    %32 = pop.offset %30[%8] : !kgen.pointer<scalar<ui8>>
    %33 = pop.offset %32[%idx-8] : !kgen.pointer<scalar<ui8>>
    %34 = pop.pointer.bitcast %33 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui64>>
    %35 = pop.offset %32[%idx-4] : !kgen.pointer<scalar<ui8>>
    %36 = pop.pointer.bitcast %35 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui32>>
    %37 = pop.stack_allocation 1 x pointer<none>
    %38 = pop.pointer.bitcast %37 : !kgen.pointer<pointer<none>> to !kgen.pointer<struct<(array<1, pointer<none>>)>>
    %39 = kgen.struct.gep %38[0] : <struct<(array<1, pointer<none>>)>>
    %40 = pop.pointer.bitcast %39 : !kgen.pointer<array<1, pointer<none>>> to !kgen.pointer<pointer<none>>
    pop.store %12, %40 : !kgen.pointer<pointer<none>>
    %41 = pop.load %37 : !kgen.pointer<pointer<none>>
    %42 = pop.cast_from_builtin %8 : index to !pop.scalar<index>
    %43 = pop.cast %42 : !pop.scalar<index> to !pop.scalar<uindex>
    %44 = index.cmp sge(%8, %index8)
    %45 = index.cmp slt(%8, %index5)
    %46 = index.cmp sle(%8, %index2)
    %47 = index.sub %8, %index2
    %48 = pop.offset %30[%47] : !kgen.pointer<scalar<ui8>>
    %49 = index.cmp sle(%8, %index16)
    %50 = index.sub %8, %index1
    %51 = pop.offset %30[%50] : !kgen.pointer<scalar<ui8>>
    %52 = pop.pointer.bitcast %41 : !kgen.pointer<none> to !kgen.pointer<scalar<ui64>>
    %53 = pop.pointer.bitcast %41 : !kgen.pointer<none> to !kgen.pointer<scalar<ui32>>
    %54 = pop.pointer.bitcast %41 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
    %55 = pop.offset %54[%47] : !kgen.pointer<scalar<ui8>>
    %56 = pop.offset %54[%50] : !kgen.pointer<scalar<ui8>>
    %57 = pop.offset %54[%index1] : !kgen.pointer<scalar<ui8>>
    %58 = pop.offset %54[%8] : !kgen.pointer<scalar<ui8>>
    %59 = pop.offset %58[%idx-8] : !kgen.pointer<scalar<ui8>>
    %60 = pop.pointer.bitcast %59 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui64>>
    %61 = pop.offset %58[%idx-4] : !kgen.pointer<scalar<ui8>>
    %62 = pop.pointer.bitcast %61 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<scalar<ui32>>
    %63 = index.cmp eq(%8, %index0)
    hlcf.if %63 {
      hlcf.yield
    } else {
      hlcf.if %45 {
        %68 = pop.load %54 : !kgen.pointer<scalar<ui8>>
        pop.store %68, %30 : !kgen.pointer<scalar<ui8>>
        %69 = pop.load %56 : !kgen.pointer<scalar<ui8>>
        pop.store %69, %51 : !kgen.pointer<scalar<ui8>>
        hlcf.if %46 {
          hlcf.yield
        } else {
          %70 = pop.load %57 : !kgen.pointer<scalar<ui8>>
          pop.store %70, %31 : !kgen.pointer<scalar<ui8>>
          %71 = pop.load %55 : !kgen.pointer<scalar<ui8>>
          pop.store %71, %48 : !kgen.pointer<scalar<ui8>>
          hlcf.yield
        }
        hlcf.yield
      } else {
        hlcf.if %49 {
          hlcf.if %44 {
            %68 = pop.load volatile<0> invariant<0> nontemporal<0> %52 align<1> : !kgen.pointer<scalar<ui64>>
            pop.store volatile<0> nontemporal<0> %68, %28 align<1> : !kgen.pointer<scalar<ui64>>
            %69 = pop.load volatile<0> invariant<0> nontemporal<0> %60 align<1> : !kgen.pointer<scalar<ui64>>
            pop.store volatile<0> nontemporal<0> %69, %34 align<1> : !kgen.pointer<scalar<ui64>>
            hlcf.yield
          } else {
            %68 = pop.load volatile<0> invariant<0> nontemporal<0> %53 align<1> : !kgen.pointer<scalar<ui32>>
            pop.store volatile<0> nontemporal<0> %68, %29 align<1> : !kgen.pointer<scalar<ui32>>
            %69 = pop.load volatile<0> invariant<0> nontemporal<0> %62 align<1> : !kgen.pointer<scalar<ui32>>
            pop.store volatile<0> nontemporal<0> %69, %36 align<1> : !kgen.pointer<scalar<ui32>>
            hlcf.yield
          }
          hlcf.yield
        } else {
          %68 = pop.floordiv %43, %simd : !pop.scalar<uindex>
          %69 = pop.mul %68, %simd : !pop.scalar<uindex>
          %70 = pop.cast fast %69 : !pop.scalar<uindex> to !pop.scalar<index>
          %71 = pop.cast_to_builtin %70 : !pop.scalar<index> to index
          hlcf.loop "_loop_0" (%arg2 = %index0 : index) {
            %73 = index.add %arg2, %index32
            %74 = index.sub %71, %arg2
            %75 = index.cmp slt(%arg2, %71)
            %76 = pop.select %75, %74, %index0 : index
            %77 = index.cmp sle(%76, %index0)
            %78 = pop.select %77, %arg2, %73 : index
            %79:2 = lit.try "try0" -> index, index {
              hlcf.if %77 {
                lit.try.raise "try0" %78, %arg2 : index, index
              } else {
                hlcf.yield
              }
              lit.try.yield %78, %arg2 : index, index
            } except (%arg3: index, %arg4: index) {
              hlcf.break "_loop_0"
            } else (%arg3: index, %arg4: index) {
              lit.try.yield %arg3, %arg4 : index, index
            }
            %80 = pop.offset %54[%79#1] : !kgen.pointer<scalar<ui8>>
            %81 = pop.pointer.bitcast %80 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<simd<32, ui8>>
            %82 = pop.load volatile<0> invariant<0> nontemporal<0> %81 align<1> : !kgen.pointer<simd<32, ui8>>
            %83 = pop.offset %30[%79#1] : !kgen.pointer<scalar<ui8>>
            %84 = pop.pointer.bitcast %83 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<simd<32, ui8>>
            pop.store volatile<0> nontemporal<0> %82, %84 align<1> : !kgen.pointer<simd<32, ui8>>
            hlcf.continue "_loop_0" %79#0 : index
          }
          %72 = index.maxs %71, %8
          hlcf.loop "_loop_2" (%arg2 = %71 : index) {
            %73 = index.add %arg2, %index1
            %74 = index.cmp eq(%arg2, %72)
            %75 = pop.select %74, %arg2, %73 : index
            %76:2 = lit.try "try2" -> index, index {
              hlcf.if %74 {
                lit.try.raise "try2" %75, %arg2 : index, index
              } else {
                hlcf.yield
              }
              lit.try.yield %75, %arg2 : index, index
            } except (%arg3: index, %arg4: index) {
              hlcf.break "_loop_2"
            } else (%arg3: index, %arg4: index) {
              lit.try.yield %arg3, %arg4 : index, index
            }
            %77 = pop.offset %54[%76#1] : !kgen.pointer<scalar<ui8>>
            %78 = pop.load volatile<0> invariant<0> nontemporal<0> %77 align<1> : !kgen.pointer<scalar<ui8>>
            %79 = pop.offset %30[%76#1] : !kgen.pointer<scalar<ui8>>
            pop.store volatile<0> nontemporal<0> %78, %79 align<1> : !kgen.pointer<scalar<ui8>>
            hlcf.continue "_loop_2" %76#0 : index
          }
          hlcf.yield
        }
        hlcf.yield
      }
      hlcf.yield
    }
    %64 = pop.load %4 : !kgen.pointer<index>
    %65 = index.and %64, %index4611686018427387904
    %66 = index.cmp ne(%65, %index0)
    hlcf.if %66 {
      %68 = pop.load %1 : !kgen.pointer<pointer<none>>
      %69 = pop.pointer.bitcast %68 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %70 = pop.offset %69[%idx-8] : !kgen.pointer<scalar<ui8>>
      %71 = pop.pointer.bitcast %70 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
      %72 = kgen.struct.gep %71[0] : <struct<(scalar<index>) memoryOnly>>
      %73 = pop.atomic.rmw sub(%72, %simd_0) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
      %74 = pop.cmp eq(%73, %simd_0) : <1, index>
      %75 = pop.cast_to_builtin %74 : !pop.scalar<bool> to i1
      hlcf.if %75 {
        pop.fence syncscope("") acquire
        pop.aligned_free %70 : <scalar<ui8>>
        hlcf.yield
      } else {
        hlcf.yield
      }
      hlcf.yield
    } else {
      hlcf.yield
    }
    pop.store %8, %3 : !kgen.pointer<index>
    pop.store %22, %1 : !kgen.pointer<pointer<none>>
    pop.store %20, %4 : !kgen.pointer<index>
    %67 = index.or %20, %index4611686018427387904
    pop.store %67, %4 : !kgen.pointer<index>
    kgen.return
  }
  kgen.func @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg0: !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, %arg1: !kgen.struct<(pointer<none>, index)>) {
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
    %0 = kgen.struct.gep %arg0[0] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
    %1 = kgen.struct.gep %0[0] : <struct<(array<4096, scalar<ui8>>) memoryOnly>>
    %2 = pop.pointer.bitcast %1 : !kgen.pointer<array<4096, scalar<ui8>>> to !kgen.pointer<none>
    %3 = kgen.struct.gep %arg0[2] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
    %4 = kgen.struct.extract %arg1[0] : <(pointer<none>, index)>
    %5 = pop.pointer.bitcast %1 : !kgen.pointer<array<4096, scalar<ui8>>> to !kgen.pointer<scalar<ui8>>
    %6 = kgen.struct.gep %arg0[1] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
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
      %18 = pop.load %3 : !kgen.pointer<pointer<struct<(pointer<none>, index, index) memoryOnly>>>
      %19 = pop.load %6 : !kgen.pointer<index>
      %20 = kgen.struct.create(%2, %19) : !kgen.struct<(pointer<none>, index)>
      kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%18, %20) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      pop.store %index0, %6 : !kgen.pointer<index>
      %21 = pop.load %3 : !kgen.pointer<pointer<struct<(pointer<none>, index, index) memoryOnly>>>
      kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%21, %arg1) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      hlcf.yield
    } else {
      %18 = pop.load %6 : !kgen.pointer<index>
      %19 = index.add %18, %7
      %20 = index.cmp sgt(%19, %index4096)
      hlcf.if %20 {
        %58 = pop.load %3 : !kgen.pointer<pointer<struct<(pointer<none>, index, index) memoryOnly>>>
        %59 = pop.load %6 : !kgen.pointer<index>
        %60 = kgen.struct.create(%2, %59) : !kgen.struct<(pointer<none>, index)>
        kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%58, %60) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
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
  kgen.func @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=1"(%arg0: !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, %arg1: !kgen.struct<(pointer<none>, index)>) {
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
    %0 = kgen.struct.gep %arg0[0] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
    %1 = kgen.struct.gep %0[0] : <struct<(array<4096, scalar<ui8>>) memoryOnly>>
    %2 = pop.pointer.bitcast %1 : !kgen.pointer<array<4096, scalar<ui8>>> to !kgen.pointer<none>
    %3 = kgen.struct.gep %arg0[2] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
    %4 = kgen.struct.extract %arg1[0] : <(pointer<none>, index)>
    %5 = pop.pointer.bitcast %1 : !kgen.pointer<array<4096, scalar<ui8>>> to !kgen.pointer<scalar<ui8>>
    %6 = kgen.struct.gep %arg0[1] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>>
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
      %18 = pop.load %3 : !kgen.pointer<pointer<struct<(pointer<none>, index, index) memoryOnly>>>
      %19 = pop.load %6 : !kgen.pointer<index>
      %20 = kgen.struct.create(%2, %19) : !kgen.struct<(pointer<none>, index)>
      kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%18, %20) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      pop.store %index0, %6 : !kgen.pointer<index>
      %21 = pop.load %3 : !kgen.pointer<pointer<struct<(pointer<none>, index, index) memoryOnly>>>
      kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%21, %arg1) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      hlcf.yield
    } else {
      %18 = pop.load %6 : !kgen.pointer<index>
      %19 = index.add %18, %7
      %20 = index.cmp sgt(%19, %index4096)
      hlcf.if %20 {
        %58 = pop.load %3 : !kgen.pointer<pointer<struct<(pointer<none>, index, index) memoryOnly>>>
        %59 = pop.load %6 : !kgen.pointer<index>
        %60 = kgen.struct.create(%2, %59) : !kgen.struct<(pointer<none>, index)>
        kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%58, %60) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
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
  kgen.func @"std::format::tstring::TString::write_to[::Writer](::TString[$0, $1, $2, $3, $4],$5&),Ts.values`1=[[typevalue<#kgen.instref<\1B\22std::reflection::location::SourceLocation\22>>, struct<(index, index, struct<(pointer<none>, index)>)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]],format_string={ #interp.memref<{[(#interp.memory_handle<16, \22At {}: {}\\00\22 string>, const_global, [], [])], []}, 0, 0>, 9 },writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]"(%arg0: !kgen.struct<(struct<(pointer<struct<(index, index, struct<(pointer<none>, index)>)>>, pointer<struct<(pointer<none>, index, index) memoryOnly>>)>) memoryOnly>, %arg1: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut) {
    %index0 = kgen.param.constant = <0>
    %index1 = kgen.param.constant = <1>
    %simd = kgen.param.constant: scalar<ui8> = <0>
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %index56 = kgen.param.constant = <56>
    %index2233785415175766016 = kgen.param.constant = <2233785415175766016>
    %0 = kgen.struct.extract %arg0[0] : <(struct<(pointer<struct<(index, index, struct<(pointer<none>, index)>)>>, pointer<struct<(pointer<none>, index, index) memoryOnly>>)>) memoryOnly>
    %1 = kgen.struct.extract %0[0] : <(pointer<struct<(index, index, struct<(pointer<none>, index)>)>>, pointer<struct<(pointer<none>, index, index) memoryOnly>>)>
    %2 = kgen.struct.extract %0[1] : <(pointer<struct<(index, index, struct<(pointer<none>, index)>)>>, pointer<struct<(pointer<none>, index, index) memoryOnly>>)>
    %3 = pop.global_constant: struct<(array<8, scalar<ui8>>) memoryOnly> = <{ [65, 116, 32, 0, 58, 32, 0, 0] }>
    %4 = kgen.struct.gep %3[0] : <struct<(array<8, scalar<ui8>>) memoryOnly>>
    %5 = pop.pointer.bitcast %4 : !kgen.pointer<array<8, scalar<ui8>>> to !kgen.pointer<none>
    %6 = pop.pointer.bitcast %5 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
    %7 = hlcf.loop "_loop_0" (%arg2 = %index0 : index) -> index {
      %34 = pop.offset %6[%arg2] : !kgen.pointer<scalar<ui8>>
      %35 = pop.load %34 : !kgen.pointer<scalar<ui8>>
      %36 = pop.cmp ne(%35, %simd) : <1, ui8>
      %37 = pop.cast_to_builtin %36 : !pop.scalar<bool> to i1
      hlcf.if %37 {
        hlcf.yield
      } else {
        hlcf.break "_loop_0" %arg2 : index
      }
      %38 = index.add %arg2, %index1
      hlcf.continue "_loop_0" %38 : index
    }
    %8 = kgen.struct.create(%5, %7) : !kgen.struct<(pointer<none>, index)>
    kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg1, %8) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    %9 = index.add %7, %index1
    %10 = pop.load %1 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
    kgen.call @"std::reflection::location::SourceLocation::write_to[::Writer](::SourceLocation,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]"(%10, %arg1) : (!kgen.struct<(index, index, struct<(pointer<none>, index)>)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut) -> ()
    %11 = pop.offset %6[%9] : !kgen.pointer<scalar<ui8>>
    %12 = pop.pointer.bitcast %11 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<none>
    %13 = hlcf.loop "_loop_0" (%arg2 = %index0 : index) -> index {
      %34 = pop.offset %11[%arg2] : !kgen.pointer<scalar<ui8>>
      %35 = pop.load %34 : !kgen.pointer<scalar<ui8>>
      %36 = pop.cmp ne(%35, %simd) : <1, ui8>
      %37 = pop.cast_to_builtin %36 : !pop.scalar<bool> to i1
      hlcf.if %37 {
        hlcf.yield
      } else {
        hlcf.break "_loop_0" %arg2 : index
      }
      %38 = index.add %arg2, %index1
      hlcf.continue "_loop_0" %38 : index
    }
    %14 = kgen.struct.create(%12, %13) : !kgen.struct<(pointer<none>, index)>
    kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg1, %14) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    %15 = index.add %13, %index1
    %16 = index.add %9, %15
    %17 = kgen.struct.gep %2[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %18 = pop.pointer.bitcast %2 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
    %19 = kgen.struct.gep %2[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %20 = pop.load %19 : !kgen.pointer<index>
    %21 = index.and %20, %index-9223372036854775808
    %22 = index.cmp ne(%21, %index0)
    %23 = hlcf.if %22 -> !kgen.pointer<none> {
      hlcf.yield %18 : !kgen.pointer<none>
    } else {
      %34 = pop.load %17 : !kgen.pointer<pointer<none>>
      hlcf.yield %34 : !kgen.pointer<none>
    }
    %24 = kgen.struct.gep %2[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %25 = pop.load %19 : !kgen.pointer<index>
    %26 = index.and %25, %index-9223372036854775808
    %27 = index.cmp ne(%26, %index0)
    %28 = hlcf.if %27 -> index {
      %34 = pop.load %19 : !kgen.pointer<index>
      %35 = index.and %34, %index2233785415175766016
      %36 = index.shrs %35, %index56
      hlcf.yield %36 : index
    } else {
      %34 = pop.load %24 : !kgen.pointer<index>
      hlcf.yield %34 : index
    }
    %29 = kgen.struct.create(%23, %28) : !kgen.struct<(pointer<none>, index)>
    kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg1, %29) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    %30 = pop.offset %6[%16] : !kgen.pointer<scalar<ui8>>
    %31 = pop.pointer.bitcast %30 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<none>
    %32 = hlcf.loop "_loop_0" (%arg2 = %index0 : index) -> index {
      %34 = pop.offset %30[%arg2] : !kgen.pointer<scalar<ui8>>
      %35 = pop.load %34 : !kgen.pointer<scalar<ui8>>
      %36 = pop.cmp ne(%35, %simd) : <1, ui8>
      %37 = pop.cast_to_builtin %36 : !pop.scalar<bool> to i1
      hlcf.if %37 {
        hlcf.yield
      } else {
        hlcf.break "_loop_0" %arg2 : index
      }
      %38 = index.add %arg2, %index1
      hlcf.continue "_loop_0" %38 : index
    }
    %33 = kgen.struct.create(%31, %32) : !kgen.struct<(pointer<none>, index)>
    kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg1, %33) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    kgen.return
  }
  kgen.func @"std::gpu::host::device_context::DeviceBuffer::enqueue_fill(::DeviceBuffer[$0],::SIMD[$0, ::Int(1)]),dtype=f32"(%arg0: !kgen.struct<(pointer<none>, struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>) memoryOnly>, %arg1: !pop.scalar<f32>, %arg2: !kgen.pointer<struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>>) -> i1 {
    %string = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/gpu/host/device_context.mojo">
    %index56 = kgen.param.constant = <56>
    %idx-8 = index.constant -8
    %index17 = kgen.param.constant = <17>
    %index5967 = kgen.param.constant = <5967>
    %simd = kgen.param.constant: scalar<uindex> = <4>
    %array = kgen.param.constant: array<1, struct<(index, index, struct<(pointer<none>, index)>)>> = <[{ -1, 0, { 0, 0 } }]>
    %simd_0 = kgen.param.constant: scalar<index> = <1>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %0 = kgen.param.constant: i1 = <0>
    %pointer = kgen.param.constant: pointer<none> = <0>
    %index0 = kgen.param.constant = <0>
    %index-1 = kgen.param.constant = <-1>
    %1 = kgen.struct.extract %arg0[1] : <(pointer<none>, struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>) memoryOnly>
    %2 = pop.external_call @AsyncRT_DeviceBuffer_context(%1) : (!kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>) -> !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>
    %3 = kgen.struct.extract %2[0] : <(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>
    %4 = kgen.struct.extract %3[0] : <(struct<(struct<(pointer<none>) memoryOnly>)>)>
    %5 = kgen.struct.extract %4[0] : <(struct<(pointer<none>) memoryOnly>)>
    %6 = kgen.struct.extract %5[0] : <(pointer<none>) memoryOnly>
    %7 = pop.stack_allocation 1 x pointer<none>
    %8 = pop.pointer.bitcast %7 : !kgen.pointer<pointer<none>> to !kgen.pointer<struct<(array<1, pointer<none>>)>>
    %9 = kgen.struct.gep %8[0] : <struct<(array<1, pointer<none>>)>>
    %10 = pop.pointer.bitcast %9 : !kgen.pointer<array<1, pointer<none>>> to !kgen.pointer<pointer<none>>
    %11 = pop.pointer_to_index %6 : <none>
    %12 = index.cmp eq(%11, %index0)
    %13 = pop.select %12, %index0, %index-1 : index
    %14 = index.cmp eq(%13, %index-1)
    hlcf.if %14 {
      pop.store %6, %10 : !kgen.pointer<pointer<none>>
      hlcf.yield
    } else {
      pop.store %pointer, %7 : !kgen.pointer<pointer<none>>
      hlcf.yield
    }
    %15 = pop.load %7 : !kgen.pointer<pointer<none>>
    %16 = kgen.struct.create(%15) : !kgen.struct<(pointer<none>) memoryOnly>
    %17 = kgen.struct.create(%16) : !kgen.struct<(struct<(pointer<none>) memoryOnly>)>
    %18 = kgen.struct.create(%17) : !kgen.struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>
    %19 = kgen.struct.create(%18) : !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>
    %20 = pop.string.address %string
    %21 = pop.pointer.bitcast %20 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %22 = kgen.struct.create(%21, %index56) : !kgen.struct<(pointer<none>, index)>
    %23 = kgen.struct.create(%index5967, %index17, %22) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %24 = pop.bitcast %arg1 : !pop.scalar<f32> to !pop.scalar<ui32>
    %25 = pop.cast fast %24 : !pop.scalar<ui32> to !pop.scalar<ui64>
    %26 = pop.external_call @AsyncRT_DeviceContext_setMemory_async(%19, %1, %25, %simd) : (!kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>, !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>, !pop.scalar<ui64>, !pop.scalar<uindex>) -> !kgen.struct<(struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>)>
    %27 = kgen.struct.extract %26[0] : <(struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>)>
    %28 = kgen.struct.extract %27[0] : <(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>
    %29 = kgen.struct.extract %28[0] : <(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>
    %30 = kgen.struct.extract %29[0] : <(struct<(array<1, pointer<none>>)>) memoryOnly>
    %31 = kgen.struct.extract %30[0] : <(array<1, pointer<none>>)>
    %32 = pop.array.get %31[0] : !pop.array<1, pointer<none>>
    %33 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
    %34 = kgen.struct.gep %33[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %35 = kgen.param.materialize: struct<(pointer<none>, index, index) memoryOnly> = <{ #interp.memref<{[(#memory_handle7, stack, [(0, 1, 0)], []), (#memory_handle8, const_global, [], [])], []}, 1, 0>, 0, 2305843009213693952 }>
    pop.stack_alloc.lifetime.start(%33) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %35, %33 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %36 = pop.array.create [%32] : !pop.array<1, pointer<none>>
    %37 = kgen.struct.create(%36) : !kgen.struct<(array<1, pointer<none>>)>
    %38 = kgen.struct.create(%37) : !kgen.struct<(struct<(array<1, pointer<none>>)>) memoryOnly>
    %39 = kgen.struct.create(%38) : !kgen.struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>
    %40 = kgen.struct.create(%39) : !kgen.struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>
    %41 = kgen.struct.create(%40) : !kgen.struct<(struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>)>
    %42 = pop.stack_allocation 1 x struct<(array<1, pointer<none>>)>
    pop.store %37, %42 : !kgen.pointer<struct<(array<1, pointer<none>>)>>
    %43 = pop.pointer.bitcast %42 : !kgen.pointer<struct<(array<1, pointer<none>>)>> to !kgen.pointer<pointer<none>>
    %44 = pop.load %43 : !kgen.pointer<pointer<none>>
    %45 = pop.pointer_to_index %44 : <none>
    %46 = index.cmp eq(%45, %index0)
    %47 = pop.select %46, %index0, %index-1 : index
    %48 = index.cmp eq(%47, %index-1)
    %49 = hlcf.if %48 -> i1 {
      %54 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
      pop.stack_alloc.lifetime.start(%54) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
      pop.store %array, %54 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
      %55 = pop.pointer.bitcast %54 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<index>
      %56 = pop.load %55 : !kgen.pointer<index>
      %57 = index.cmp eq(%56, %index-1)
      %58 = pop.select %57, %index0, %index-1 : index
      pop.stack_alloc.lifetime.end(%54) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
      %59 = index.cmp eq(%58, %index-1)
      %60 = hlcf.if %59 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
        %62 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
        pop.stack_alloc.lifetime.start(%62) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
        pop.store %array, %62 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
        %63 = pop.pointer.bitcast %62 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
        %64 = pop.load %63 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
        pop.stack_alloc.lifetime.end(%62) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
        hlcf.yield %64 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
      } else {
        hlcf.yield %23 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
      }
      %61:2 = kgen.call @"std::gpu::host::device_context::_raise_checked_impl[LITImmutOrigin,::Origin[::Bool(False), $0]](::Optional[::CStringSlice[$0, $1]],::String,::SourceLocation)_REMOVED_ARG"(%41, %33, %60) : (!kgen.struct<(struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>) throws -> (i1, !kgen.struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>)
      pop.store %61#1, %arg2 : !kgen.pointer<struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>>
      hlcf.yield %61#0 : i1
    } else {
      hlcf.yield %0 : i1
    }
    %50 = kgen.struct.gep %33[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %51 = pop.load %50 : !kgen.pointer<index>
    %52 = index.and %51, %index4611686018427387904
    %53 = index.cmp ne(%52, %index0)
    hlcf.if %53 {
      %54 = pop.load %34 : !kgen.pointer<pointer<none>>
      %55 = pop.pointer.bitcast %54 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %56 = pop.offset %55[%idx-8] : !kgen.pointer<scalar<ui8>>
      %57 = pop.pointer.bitcast %56 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
      %58 = kgen.struct.gep %57[0] : <struct<(scalar<index>) memoryOnly>>
      %59 = pop.atomic.rmw sub(%58, %simd_0) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
      %60 = pop.cmp eq(%59, %simd_0) : <1, index>
      %61 = pop.cast_to_builtin %60 : !pop.scalar<bool> to i1
      hlcf.if %61 {
        pop.fence syncscope("") acquire
        pop.aligned_free %56 : <scalar<ui8>>
        hlcf.yield
      } else {
        hlcf.yield
      }
      hlcf.yield
    } else {
      hlcf.yield
    }
    pop.stack_alloc.lifetime.end(%33) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    kgen.return %49 : i1
  }
  kgen.func @"std::gpu::host::device_context::DeviceContext::enqueue_create_buffer[::DType](::DeviceContext,::Int),dtype=f32"(%arg0: !kgen.struct<(struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>, i1)>, %arg1: index) throws -> (i1, !kgen.struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>, !kgen.struct<(pointer<none>, struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>) memoryOnly>) {
    %simd = kgen.param.constant: scalar<ui8> = <#interp.uninitmem>
    %union = kgen.param.constant: union<struct<()>, struct<(pointer<none>) memoryOnly>> = <#interp.uninitmem>
    %index = kgen.param.constant = <#interp.uninitmem>
    %pointer = kgen.param.constant: pointer<none> = <#interp.uninitmem>
    %index44 = kgen.param.constant = <44>
    %index917 = kgen.param.constant = <917>
    %index56 = kgen.param.constant = <56>
    %string = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/gpu/host/device_context.mojo">
    %index21 = kgen.param.constant = <21>
    %index897 = kgen.param.constant = <897>
    %pointer_0 = kgen.param.constant: pointer<none> = <0>
    %string_1 = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/collections/optional.mojo">
    %index18 = kgen.param.constant = <18>
    %index610 = kgen.param.constant = <610>
    %idx-8 = index.constant -8
    %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle1, const_global, [], [])], []}, 0, 0>, 1 }>
    %struct_2 = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle9, const_global, [], [])], []}, 0, 0>, 6 }>
    %string_3 = kgen.param.constant: string = <" ">
    %index2 = kgen.param.constant = <2>
    %string_4 = kgen.param.constant: string = <": ">
    %string_5 = kgen.param.constant: string = <"">
    %string_6 = kgen.param.constant: string = <"`Optional.value()` called on empty `Optional`. Consider using `if optional:` to check whether the `Optional` is empty before calling `.value()`, or use `.or_else()` to provide a default value.">
    %index192 = kgen.param.constant = <192>
    %struct_7 = kgen.param.constant: struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)> = <{ { { { 0 } } } }>
    %0 = kgen.param.constant: i1 = <0>
    %1 = kgen.param.constant: i1 = <1>
    %simd_8 = kgen.param.constant: scalar<uindex> = <4>
    %index2305843009213693952 = kgen.param.constant = <2305843009213693952>
    %index53 = kgen.param.constant = <53>
    %simd_9 = kgen.param.constant: scalar<index> = <1>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %index0 = kgen.param.constant = <0>
    %index-1 = kgen.param.constant = <-1>
    %index1 = kgen.param.constant = <1>
    %index3345 = kgen.param.constant = <3345>
    %index35 = kgen.param.constant = <35>
    %2 = pop.cast_from_builtin %arg1 : index to !pop.scalar<index>
    %3 = pop.cast %2 : !pop.scalar<index> to !pop.scalar<uindex>
    %4 = pop.string.address %string
    %5 = pop.pointer.bitcast %4 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %6 = kgen.struct.create(%5, %index56) : !kgen.struct<(pointer<none>, index)>
    %7 = kgen.struct.create(%index917, %index44, %6) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %8 = kgen.struct.create(%index897, %index21, %6) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %9 = pop.string.address %string_1
    %10 = pop.pointer.bitcast %9 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %11 = pop.string.address %string_3
    %12 = pop.pointer.bitcast %11 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %13 = pop.string.address %string_4
    %14 = pop.pointer.bitcast %13 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %15 = pop.string.address %string_5
    %16 = pop.pointer.bitcast %15 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %17 = pop.string.address %string_6
    %18 = pop.pointer.bitcast %17 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %19 = kgen.struct.create(%10, %index53) : !kgen.struct<(pointer<none>, index)>
    %20 = kgen.struct.create(%index610, %index18, %19) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %21 = kgen.struct.create(%16, %index0) : !kgen.struct<(pointer<none>, index)>
    %22 = kgen.struct.extract %arg0[0] : <(struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>, i1)>
    %23 = pop.stack_allocation 1 x struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)> align 1 marked
    %24 = kgen.struct.gep %23[0] : <struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>>
    %25 = pop.pointer.bitcast %23 : !kgen.pointer<struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>> to !kgen.pointer<none>
    pop.stack_alloc.lifetime.start(%23) : !kgen.pointer<struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>>
    pop.store %struct_7, %23 : !kgen.pointer<struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>>
    %26 = pop.stack_allocation 1 x struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)> align 1 marked
    %27 = kgen.struct.gep %26[0] : <struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>>
    %28 = kgen.struct.gep %27[0] : <struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>>
    %29 = kgen.struct.gep %28[0] : <struct<(struct<(pointer<none>) memoryOnly>)>>
    %30 = kgen.struct.gep %29[0] : <struct<(pointer<none>) memoryOnly>>
    %31 = pop.pointer.bitcast %26 : !kgen.pointer<struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>> to !kgen.pointer<none>
    pop.stack_alloc.lifetime.start(%26) : !kgen.pointer<struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>>
    pop.store %struct_7, %26 : !kgen.pointer<struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>>
    %32 = pop.external_call @AsyncRT_DeviceContext_createBuffer_async(%25, %31, %22, %3, %simd_8) : (!kgen.pointer<none>, !kgen.pointer<none>, !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>, !pop.scalar<uindex>, !pop.scalar<uindex>) -> !kgen.struct<(struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>)>
    %33 = kgen.struct.create(%index3345, %index35, %6) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %34 = kgen.struct.extract %32[0] : <(struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>)>
    %35 = kgen.struct.extract %34[0] : <(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>
    %36 = kgen.struct.extract %35[0] : <(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>
    %37 = kgen.struct.extract %36[0] : <(struct<(array<1, pointer<none>>)>) memoryOnly>
    %38 = kgen.struct.extract %37[0] : <(array<1, pointer<none>>)>
    %39 = pop.array.get %38[0] : !pop.array<1, pointer<none>>
    %40 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> align 1 marked
    pop.stack_alloc.lifetime.start(%40) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
    %41 = pop.pointer.bitcast %40 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
    pop.store %33, %41 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
    %42 = pop.load %40 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
    pop.stack_alloc.lifetime.end(%40) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
    %43 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
    %44 = kgen.struct.gep %43[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %45 = kgen.param.materialize: struct<(pointer<none>, index, index) memoryOnly> = <{ #interp.memref<{[(#memory_handle7, stack, [(0, 1, 0)], []), (#memory_handle8, const_global, [], [])], []}, 1, 0>, 0, 2305843009213693952 }>
    pop.stack_alloc.lifetime.start(%43) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %45, %43 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %46 = pop.array.create [%39] : !pop.array<1, pointer<none>>
    %47 = kgen.struct.create(%46) : !kgen.struct<(array<1, pointer<none>>)>
    %48 = kgen.struct.create(%47) : !kgen.struct<(struct<(array<1, pointer<none>>)>) memoryOnly>
    %49 = kgen.struct.create(%48) : !kgen.struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>
    %50 = kgen.struct.create(%49) : !kgen.struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>
    %51 = kgen.struct.create(%50) : !kgen.struct<(struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>)>
    %52 = pop.stack_allocation 1 x struct<(array<1, pointer<none>>)>
    pop.store %47, %52 : !kgen.pointer<struct<(array<1, pointer<none>>)>>
    %53 = pop.pointer.bitcast %52 : !kgen.pointer<struct<(array<1, pointer<none>>)>> to !kgen.pointer<pointer<none>>
    %54 = pop.load %53 : !kgen.pointer<pointer<none>>
    %55 = pop.pointer_to_index %54 : <none>
    %56 = index.cmp eq(%55, %index0)
    %57 = pop.select %56, %index0, %index-1 : index
    %58 = index.cmp eq(%57, %index-1)
    %59:6 = hlcf.if %58 -> i1, !kgen.pointer<none>, index, index, !pop.union<struct<()>, struct<(pointer<none>) memoryOnly>>, !pop.scalar<ui8> {
      %76 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
      pop.stack_alloc.lifetime.start(%76) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
      pop.store %42, %76 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
      %77 = pop.pointer.bitcast %76 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<index>
      %78 = pop.load %77 : !kgen.pointer<index>
      %79 = index.cmp eq(%78, %index-1)
      %80 = pop.select %79, %index0, %index-1 : index
      pop.stack_alloc.lifetime.end(%76) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
      %81 = index.cmp eq(%80, %index-1)
      %82 = hlcf.if %81 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
        %94 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>> marked
        pop.stack_alloc.lifetime.start(%94) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
        pop.store %42, %94 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
        %95 = pop.pointer.bitcast %94 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
        %96 = pop.load %95 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
        pop.stack_alloc.lifetime.end(%94) : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
        hlcf.yield %96 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
      } else {
        hlcf.yield %8 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
      }
      %83:2 = kgen.call @"std::gpu::host::device_context::_raise_checked_impl[LITImmutOrigin,::Origin[::Bool(False), $0]](::Optional[::CStringSlice[$0, $1]],::String,::SourceLocation)_REMOVED_ARG"(%51, %43, %82) : (!kgen.struct<(struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>) throws -> (i1, !kgen.struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>)
      %84 = kgen.struct.extract %83#1[0] : <(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>
      %85 = kgen.struct.extract %84[0] : <(pointer<none>, index, index) memoryOnly>
      %86 = kgen.struct.extract %84[1] : <(pointer<none>, index, index) memoryOnly>
      %87 = kgen.struct.extract %84[2] : <(pointer<none>, index, index) memoryOnly>
      %88 = kgen.struct.extract %83#1[1] : <(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>
      %89 = kgen.struct.extract %88[0] : <(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>
      %90 = kgen.struct.extract %89[0] : <(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>
      %91 = kgen.struct.extract %90[0] : <(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>
      %92 = kgen.struct.extract %91[0] : <(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>
      %93 = kgen.struct.extract %91[1] : <(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>
      hlcf.yield %83#0, %85, %86, %87, %92, %93 : i1, !kgen.pointer<none>, index, index, !pop.union<struct<()>, struct<(pointer<none>) memoryOnly>>, !pop.scalar<ui8>
    } else {
      hlcf.yield %0, %pointer, %index, %index, %union, %simd : i1, !kgen.pointer<none>, index, index, !pop.union<struct<()>, struct<(pointer<none>) memoryOnly>>, !pop.scalar<ui8>
    }
    %60 = kgen.struct.gep %43[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %61 = pop.load %60 : !kgen.pointer<index>
    %62 = index.and %61, %index4611686018427387904
    %63 = index.cmp ne(%62, %index0)
    hlcf.if %63 {
      %76 = pop.load %44 : !kgen.pointer<pointer<none>>
      %77 = pop.pointer.bitcast %76 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %78 = pop.offset %77[%idx-8] : !kgen.pointer<scalar<ui8>>
      %79 = pop.pointer.bitcast %78 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
      %80 = kgen.struct.gep %79[0] : <struct<(scalar<index>) memoryOnly>>
      %81 = pop.atomic.rmw sub(%80, %simd_9) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
      %82 = pop.cmp eq(%81, %simd_9) : <1, index>
      %83 = pop.cast_to_builtin %82 : !pop.scalar<bool> to i1
      hlcf.if %83 {
        pop.fence syncscope("") acquire
        pop.aligned_free %78 : <scalar<ui8>>
        hlcf.yield
      } else {
        hlcf.yield
      }
      hlcf.yield
    } else {
      hlcf.yield
    }
    pop.stack_alloc.lifetime.end(%43) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %64:2 = hlcf.if %59#0 -> !kgen.pointer<none>, !kgen.pointer<none> {
      pop.stack_alloc.lifetime.end(%23) : !kgen.pointer<struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>>
      pop.stack_alloc.lifetime.end(%26) : !kgen.pointer<struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>>
      hlcf.yield %pointer, %pointer : !kgen.pointer<none>, !kgen.pointer<none>
    } else {
      %76 = pop.load %27 : !kgen.pointer<struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>>
      %77 = kgen.struct.extract %76[0] : <(struct<(struct<(pointer<none>) memoryOnly>)>)>
      %78 = kgen.struct.extract %77[0] : <(struct<(pointer<none>) memoryOnly>)>
      %79 = kgen.struct.extract %78[0] : <(pointer<none>) memoryOnly>
      %80 = pop.pointer_to_index %79 : <none>
      %81 = index.cmp eq(%80, %index0)
      %82 = pop.select %81, %index0, %index-1 : index
      %83 = index.cmp eq(%82, %index-1)
      %84 = pop.xor %83, %1
      hlcf.if %84 {
        %99 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
        pop.stack_alloc.lifetime.start(%99) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
        %100 = kgen.struct.gep %99[1] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %index192, %100 : !kgen.pointer<index>
        %101 = kgen.struct.gep %99[0] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %18, %101 : !kgen.pointer<pointer<none>>
        %102 = kgen.struct.gep %99[2] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %index2305843009213693952, %102 : !kgen.pointer<index>
        %103 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
        %104 = pop.pointer.bitcast %103 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
        pop.store %7, %104 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
        %105 = pop.load %103 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
        %106 = pop.array.get %105[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
        %107 = pop.array.create [%106] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
        %108 = kgen.struct.create(%107) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
        %109 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
        %110 = pop.pointer.bitcast %109 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
        pop.store %108, %109 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
        %111 = pop.pointer.bitcast %109 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
        %112 = pop.load %111 : !kgen.pointer<index>
        %113 = index.cmp eq(%112, %index-1)
        %114 = pop.select %113, %index0, %index-1 : index
        %115 = index.cmp eq(%114, %index-1)
        %116 = hlcf.if %115 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
          %134 = pop.load %110 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
          hlcf.yield %134 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
        } else {
          hlcf.yield %20 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
        }
        %117 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
        pop.stack_alloc.lifetime.start(%117) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
        %118 = kgen.struct.gep %117[1] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %index1, %118 : !kgen.pointer<index>
        %119 = kgen.struct.gep %117[0] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %12, %119 : !kgen.pointer<pointer<none>>
        %120 = kgen.struct.gep %117[2] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %index2305843009213693952, %120 : !kgen.pointer<index>
        %121 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
        pop.stack_alloc.lifetime.start(%121) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
        %122 = kgen.struct.gep %121[1] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %index2, %122 : !kgen.pointer<index>
        %123 = kgen.struct.gep %121[0] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %14, %123 : !kgen.pointer<pointer<none>>
        %124 = kgen.struct.gep %121[2] : <struct<(pointer<none>, index, index) memoryOnly>>
        pop.store %index2305843009213693952, %124 : !kgen.pointer<index>
        kgen.call @"std::io::io::print[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?],flush:::Bool,file:::FileDescriptor$),Ts.values`=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::reflection::location::SourceLocation\22>>, struct<(index, index, struct<(pointer<none>, index)>)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]]"(%struct_2, %117, %116, %121, %99, %21, %struct, %1, %index1) : (!kgen.struct<(pointer<none>, index)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index)>, i1, index owned) -> ()
        %125 = pop.load %120 : !kgen.pointer<index>
        %126 = index.and %125, %index4611686018427387904
        %127 = index.cmp ne(%126, %index0)
        hlcf.if %127 {
          %134 = pop.load %119 : !kgen.pointer<pointer<none>>
          %135 = pop.pointer.bitcast %134 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
          %136 = pop.offset %135[%idx-8] : !kgen.pointer<scalar<ui8>>
          %137 = pop.pointer.bitcast %136 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
          %138 = kgen.struct.gep %137[0] : <struct<(scalar<index>) memoryOnly>>
          %139 = pop.atomic.rmw sub(%138, %simd_9) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
          %140 = pop.cmp eq(%139, %simd_9) : <1, index>
          %141 = pop.cast_to_builtin %140 : !pop.scalar<bool> to i1
          hlcf.if %141 {
            pop.fence syncscope("") acquire
            pop.aligned_free %136 : <scalar<ui8>>
            hlcf.yield
          } else {
            hlcf.yield
          }
          hlcf.yield
        } else {
          hlcf.yield
        }
        %128 = pop.load %124 : !kgen.pointer<index>
        %129 = index.and %128, %index4611686018427387904
        %130 = index.cmp ne(%129, %index0)
        hlcf.if %130 {
          %134 = pop.load %123 : !kgen.pointer<pointer<none>>
          %135 = pop.pointer.bitcast %134 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
          %136 = pop.offset %135[%idx-8] : !kgen.pointer<scalar<ui8>>
          %137 = pop.pointer.bitcast %136 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
          %138 = kgen.struct.gep %137[0] : <struct<(scalar<index>) memoryOnly>>
          %139 = pop.atomic.rmw sub(%138, %simd_9) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
          %140 = pop.cmp eq(%139, %simd_9) : <1, index>
          %141 = pop.cast_to_builtin %140 : !pop.scalar<bool> to i1
          hlcf.if %141 {
            pop.fence syncscope("") acquire
            pop.aligned_free %136 : <scalar<ui8>>
            hlcf.yield
          } else {
            hlcf.yield
          }
          hlcf.yield
        } else {
          hlcf.yield
        }
        %131 = pop.load %102 : !kgen.pointer<index>
        %132 = index.and %131, %index4611686018427387904
        %133 = index.cmp ne(%132, %index0)
        hlcf.if %133 {
          %134 = pop.load %101 : !kgen.pointer<pointer<none>>
          %135 = pop.pointer.bitcast %134 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
          %136 = pop.offset %135[%idx-8] : !kgen.pointer<scalar<ui8>>
          %137 = pop.pointer.bitcast %136 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
          %138 = kgen.struct.gep %137[0] : <struct<(scalar<index>) memoryOnly>>
          %139 = pop.atomic.rmw sub(%138, %simd_9) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
          %140 = pop.cmp eq(%139, %simd_9) : <1, index>
          %141 = pop.cast_to_builtin %140 : !pop.scalar<bool> to i1
          hlcf.if %141 {
            pop.fence syncscope("") acquire
            pop.aligned_free %136 : <scalar<ui8>>
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
      %85 = pop.load %30 : !kgen.pointer<pointer<none>>
      pop.stack_alloc.lifetime.end(%26) : !kgen.pointer<struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>>
      %86 = pop.load %24 : !kgen.pointer<struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>>
      %87 = kgen.struct.extract %86[0] : <(struct<(struct<(pointer<none>) memoryOnly>)>)>
      %88 = kgen.struct.extract %87[0] : <(struct<(pointer<none>) memoryOnly>)>
      %89 = kgen.struct.extract %88[0] : <(pointer<none>) memoryOnly>
      %90 = pop.stack_allocation 1 x pointer<none>
      %91 = pop.pointer.bitcast %90 : !kgen.pointer<pointer<none>> to !kgen.pointer<struct<(array<1, pointer<none>>)>>
      %92 = kgen.struct.gep %91[0] : <struct<(array<1, pointer<none>>)>>
      %93 = pop.pointer.bitcast %92 : !kgen.pointer<array<1, pointer<none>>> to !kgen.pointer<pointer<none>>
      %94 = pop.pointer_to_index %89 : <none>
      %95 = index.cmp eq(%94, %index0)
      %96 = pop.select %95, %index0, %index-1 : index
      %97 = index.cmp eq(%96, %index-1)
      hlcf.if %97 {
        pop.store %89, %93 : !kgen.pointer<pointer<none>>
        hlcf.yield
      } else {
        pop.store %pointer_0, %90 : !kgen.pointer<pointer<none>>
        hlcf.yield
      }
      %98 = pop.load %90 : !kgen.pointer<pointer<none>>
      pop.stack_alloc.lifetime.end(%23) : !kgen.pointer<struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>>
      hlcf.yield %85, %98 : !kgen.pointer<none>, !kgen.pointer<none>
    }
    %65 = kgen.struct.create(%59#1, %59#2, %59#3) : !kgen.struct<(pointer<none>, index, index) memoryOnly>
    %66 = kgen.struct.create(%59#4, %59#5) : !kgen.struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>
    %67 = kgen.struct.create(%66) : !kgen.struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>
    %68 = kgen.struct.create(%67) : !kgen.struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>
    %69 = kgen.struct.create(%68) : !kgen.struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>
    %70 = kgen.struct.create(%65, %69) : !kgen.struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>
    %71 = kgen.struct.create(%64#1) : !kgen.struct<(pointer<none>) memoryOnly>
    %72 = kgen.struct.create(%71) : !kgen.struct<(struct<(pointer<none>) memoryOnly>)>
    %73 = kgen.struct.create(%72) : !kgen.struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>
    %74 = kgen.struct.create(%73) : !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>
    %75 = kgen.struct.create(%64#0, %74) : !kgen.struct<(pointer<none>, struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>) memoryOnly>
    kgen.return %59#0, %70, %75 : i1, !kgen.struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>, !kgen.struct<(pointer<none>, struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>) memoryOnly>
  }
  kgen.func @"std::reflection::location::SourceLocation::prefix[::Writable](::SourceLocation,$0),T=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]"(%arg0: !kgen.struct<(index, index, struct<(pointer<none>, index)>)>, %arg1: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, %arg2: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> byref_result) no_inline {
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %0 = pop.stack_allocation 1 x struct<(index, index, struct<(pointer<none>, index)>)>
    pop.store %arg0, %0 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
    %1 = kgen.struct.create(%0, %arg1) : !kgen.struct<(pointer<struct<(index, index, struct<(pointer<none>, index)>)>>, pointer<struct<(pointer<none>, index, index) memoryOnly>>)>
    %2 = kgen.struct.gep %arg2[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index-9223372036854775808, %2 : !kgen.pointer<index>
    %3 = kgen.struct.create(%1) : !kgen.struct<(struct<(pointer<struct<(index, index, struct<(pointer<none>, index)>)>>, pointer<struct<(pointer<none>, index, index) memoryOnly>>)>) memoryOnly>
    kgen.call @"std::format::tstring::TString::write_to[::Writer](::TString[$0, $1, $2, $3, $4],$5&),Ts.values`1=[[typevalue<#kgen.instref<\1B\22std::reflection::location::SourceLocation\22>>, struct<(index, index, struct<(pointer<none>, index)>)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]],format_string={ #interp.memref<{[(#interp.memory_handle<16, \22At {}: {}\\00\22 string>, const_global, [], [])], []}, 0, 0>, 9 },writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]"(%3, %arg2) : (!kgen.struct<(struct<(pointer<struct<(index, index, struct<(pointer<none>, index)>)>>, pointer<struct<(pointer<none>, index, index) memoryOnly>>)>) memoryOnly>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut) -> ()
    kgen.return
  }
  kgen.func @"std::reflection::location::SourceLocation::write_to[::Writer](::SourceLocation,$0&),writer.T`2x=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]"(%arg0: !kgen.struct<(index, index, struct<(pointer<none>, index)>)>, %arg1: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut) {
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
    kgen.call @"std::collections::string::string::String::write[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](::String&,*$0),Ts.values`2x=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::builtin::int::Int\22>>, index], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::builtin::int::Int\22>>, index]]"(%arg1, %0, %3, %1, %9, %2) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, index, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, index) -> ()
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
  kgen.func @"std::builtin::_startup::__wrap_and_execute_raising_main[def() raises -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\22probe::main()\22_closure_0"() -> !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)> {
    %0 = pop.external_call @KGEN_CompilerRT_AsyncRT_GetOrCreateRuntime() : () -> !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>
    kgen.return %0 : !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>
  }
  kgen.func @"std::builtin::_startup::__wrap_and_execute_raising_main[def() raises -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\22probe::main()\22_closure_1"(%arg0: !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>) {
    pop.external_call @KGEN_CompilerRT_AsyncRT_ReleaseRuntime(%arg0) : (!kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>) -> ()
    kgen.return
  }
  kgen.func @"std::builtin::_startup::__wrap_and_execute_raising_main[def() raises -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\22probe::main()\22"(%arg0: !pop.scalar<si32>, %arg1: !kgen.pointer<pointer<scalar<ui8>>>) -> !pop.scalar<si32> {
    %string = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/builtin/_startup.mojo">
    %index49 = kgen.param.constant = <49>
    %index36 = kgen.param.constant = <36>
    %index106 = kgen.param.constant = <106>
    %index53 = kgen.param.constant = <53>
    %string_0 = kgen.param.constant: string = <"oss/modular/mojo/stdlib/std/collections/optional.mojo">
    %index18 = kgen.param.constant = <18>
    %index610 = kgen.param.constant = <610>
    %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle9, const_global, [], [])], []}, 0, 0>, 6 }>
    %string_1 = kgen.param.constant: string = <" ">
    %index2 = kgen.param.constant = <2>
    %string_2 = kgen.param.constant: string = <": ">
    %string_3 = kgen.param.constant: string = <"">
    %idx-8 = index.constant -8
    %simd = kgen.param.constant: scalar<ui8> = <1>
    %string_4 = kgen.param.constant: string = <"`Optional.value()` called on empty `Optional`. Consider using `if optional:` to check whether the `Optional` is empty before calling `.value()`, or use `.or_else()` to provide a default value.">
    %index192 = kgen.param.constant = <192>
    %simd_5 = kgen.param.constant: scalar<ui8> = <0>
    %index-1 = kgen.param.constant = <-1>
    %index0 = kgen.param.constant = <0>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %simd_6 = kgen.param.constant: scalar<index> = <1>
    %string_7 = kgen.param.constant: string = <"Unhandled exception caught during execution:">
    %index44 = kgen.param.constant = <44>
    %index2305843009213693952 = kgen.param.constant = <2305843009213693952>
    %0 = kgen.param.constant: i1 = <1>
    %struct_8 = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle, const_global, [], [])], []}, 0, 0>, 1 }>
    %struct_9 = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle1, const_global, [], [])], []}, 0, 0>, 1 }>
    %1 = kgen.param.constant: i1 = <0>
    %index1 = kgen.param.constant = <1>
    %simd_10 = kgen.param.constant: scalar<si32> = <1>
    %simd_11 = kgen.param.constant: scalar<si32> = <0>
    %struct_12 = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle10, const_global, [], [])], []}, 0, 0>, 7 }>
    %2 = pop.string.address %string
    %3 = pop.pointer.bitcast %2 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %4 = kgen.struct.create(%3, %index49) : !kgen.struct<(pointer<none>, index)>
    %5 = kgen.struct.create(%index106, %index36, %4) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %6 = pop.string.address %string_0
    %7 = pop.pointer.bitcast %6 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %8 = kgen.struct.create(%7, %index53) : !kgen.struct<(pointer<none>, index)>
    %9 = kgen.struct.create(%index610, %index18, %8) : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
    %10 = pop.string.address %string_1
    %11 = pop.pointer.bitcast %10 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %12 = pop.string.address %string_2
    %13 = pop.pointer.bitcast %12 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %14 = pop.string.address %string_3
    %15 = pop.pointer.bitcast %14 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %16 = pop.string.address %string_4
    %17 = pop.pointer.bitcast %16 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %18 = kgen.struct.create(%15, %index0) : !kgen.struct<(pointer<none>, index)>
    %19 = pop.string.address %string_7
    %20 = pop.pointer.bitcast %19 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %21 = pop.external_call @KGEN_CompilerRT_AsyncRT_GetCurrentRuntime() : () -> !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>
    %22 = kgen.struct.extract %21[0] : <(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>
    %23 = kgen.struct.extract %22[0] : <(struct<(struct<(pointer<none>) memoryOnly>)>)>
    %24 = kgen.struct.extract %23[0] : <(struct<(pointer<none>) memoryOnly>)>
    %25 = kgen.struct.extract %24[0] : <(pointer<none>) memoryOnly>
    %26 = pop.pointer_to_index %25 : <none>
    %27 = index.cmp eq(%26, %index0)
    %28 = pop.select %27, %index0, %index-1 : index
    %29 = index.cmp eq(%28, %index-1)
    hlcf.if %29 {
      hlcf.yield
    } else {
      %42 = kgen.create_closure[() -> !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>: @"std::builtin::_startup::__wrap_and_execute_raising_main[def() raises -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\22probe::main()\22_closure_0"]() 
      %43 = kgen.create_closure[(!kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>) -> (): @"std::builtin::_startup::__wrap_and_execute_raising_main[def() raises -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\22probe::main()\22_closure_1"]() 
      %44 = pop.external_call @KGEN_CompilerRT_GetOrCreateGlobal(%struct_12, %42, %43) : (!kgen.struct<(pointer<none>, index)>, !kgen.generator<() -> !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>>, !kgen.generator<(!kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>) -> ()>) -> !kgen.struct<(struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>)>
      hlcf.yield
    }
    pop.external_call @KGEN_CompilerRT_SetArgV(%arg0, %arg1) : (!pop.scalar<si32>, !kgen.pointer<pointer<scalar<ui8>>>) -> ()
    pop.external_call @KGEN_CompilerRT_PrintStackTraceOnFault() : () -> ()
    %30 = pop.stack_allocation 1 x struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly> align 1 marked
    %31 = kgen.struct.gep %30[1] : <struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>>
    %32 = kgen.struct.gep %31[0] : <struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>>
    %33 = kgen.struct.gep %32[0] : <struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>>
    %34 = kgen.struct.gep %33[0] : <struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>>
    %35 = kgen.struct.gep %34[1] : <struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>>
    %36 = kgen.struct.gep %34[0] : <struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>>
    %37 = pop.union.bitcast %36 : <union<struct<()>, struct<(pointer<none>) memoryOnly>>> as <struct<(pointer<none>) memoryOnly>>
    %38 = kgen.struct.gep %37[0] : <struct<(pointer<none>) memoryOnly>>
    %39 = kgen.struct.gep %30[0] : <struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>>
    %40 = kgen.struct.gep %39[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %41 = kgen.struct.gep %39[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    lit.try "try0" {
      pop.stack_alloc.lifetime.start(%30) : !kgen.pointer<struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>>
      %42 = kgen.call @"probe::main()"(%30) : (!kgen.pointer<struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>>) -> i1
      hlcf.if %42 {
        lit.try.raise "try0"
      } else {
        pop.stack_alloc.lifetime.end(%30) : !kgen.pointer<struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>>
        hlcf.yield
      }
      lit.try.yield
    } except {
      %42 = pop.stack_allocation 1 x union<struct<()>, struct<(pointer<none>, index, index) memoryOnly>> align 1 marked
      pop.stack_alloc.lifetime.start(%42) : !kgen.pointer<union<struct<()>, struct<(pointer<none>, index, index) memoryOnly>>>
      %43 = pop.load %30 : !kgen.pointer<struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>>
      %44 = kgen.call @"std::builtin::error::Error::get_stack_trace(::Error)"(%43) : (!kgen.struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>) -> !kgen.struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>, index, index) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>
      %45 = kgen.struct.extract %44[0] : <(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>, index, index) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>
      %46 = kgen.struct.extract %45[0] : <(struct<(struct<(union<struct<()>, struct<(pointer<none>, index, index) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>
      %47 = kgen.struct.extract %46[0] : <(struct<(union<struct<()>, struct<(pointer<none>, index, index) memoryOnly>>, scalar<ui8>)>) memoryOnly>
      %48 = kgen.struct.extract %47[0] : <(union<struct<()>, struct<(pointer<none>, index, index) memoryOnly>>, scalar<ui8>)>
      pop.store %48, %42 : !kgen.pointer<union<struct<()>, struct<(pointer<none>, index, index) memoryOnly>>>
      %49 = kgen.struct.extract %47[1] : <(union<struct<()>, struct<(pointer<none>, index, index) memoryOnly>>, scalar<ui8>)>
      %50 = pop.union.bitcast %42 : <union<struct<()>, struct<(pointer<none>, index, index) memoryOnly>>> as <struct<(pointer<none>, index, index) memoryOnly>>
      %51 = kgen.struct.gep %50[0] : <struct<(pointer<none>, index, index) memoryOnly>>
      %52 = kgen.struct.gep %50[2] : <struct<(pointer<none>, index, index) memoryOnly>>
      %53 = pop.cmp eq(%49, %simd_5) : <1, ui8>
      %54 = pop.cast_to_builtin %53 : !pop.scalar<bool> to i1
      %55 = pop.xor %54, %0
      hlcf.if %55 {
        hlcf.if %54 {
          %69 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
          pop.stack_alloc.lifetime.start(%69) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
          %70 = kgen.struct.gep %69[1] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index192, %70 : !kgen.pointer<index>
          %71 = kgen.struct.gep %69[0] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %17, %71 : !kgen.pointer<pointer<none>>
          %72 = kgen.struct.gep %69[2] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index2305843009213693952, %72 : !kgen.pointer<index>
          %73 = pop.stack_allocation 1 x array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
          %74 = pop.pointer.bitcast %73 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
          pop.store %5, %74 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
          %75 = pop.load %73 : !kgen.pointer<array<1, struct<(index, index, struct<(pointer<none>, index)>)>>>
          %76 = pop.array.get %75[0] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
          %77 = pop.array.create [%76] : !pop.array<1, struct<(index, index, struct<(pointer<none>, index)>)>>
          %78 = kgen.struct.create(%77) : !kgen.struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
          %79 = pop.stack_allocation 1 x struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>
          %80 = pop.pointer.bitcast %79 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
          pop.store %78, %79 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>>
          %81 = pop.pointer.bitcast %79 : !kgen.pointer<struct<(array<1, struct<(index, index, struct<(pointer<none>, index)>)>>)>> to !kgen.pointer<index>
          %82 = pop.load %81 : !kgen.pointer<index>
          %83 = index.cmp eq(%82, %index-1)
          %84 = pop.select %83, %index0, %index-1 : index
          %85 = index.cmp eq(%84, %index-1)
          %86 = hlcf.if %85 -> !kgen.struct<(index, index, struct<(pointer<none>, index)>)> {
            %104 = pop.load %80 : !kgen.pointer<struct<(index, index, struct<(pointer<none>, index)>)>>
            hlcf.yield %104 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
          } else {
            hlcf.yield %9 : !kgen.struct<(index, index, struct<(pointer<none>, index)>)>
          }
          %87 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
          pop.stack_alloc.lifetime.start(%87) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
          %88 = kgen.struct.gep %87[1] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index1, %88 : !kgen.pointer<index>
          %89 = kgen.struct.gep %87[0] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %11, %89 : !kgen.pointer<pointer<none>>
          %90 = kgen.struct.gep %87[2] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index2305843009213693952, %90 : !kgen.pointer<index>
          %91 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
          pop.stack_alloc.lifetime.start(%91) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
          %92 = kgen.struct.gep %91[1] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index2, %92 : !kgen.pointer<index>
          %93 = kgen.struct.gep %91[0] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %13, %93 : !kgen.pointer<pointer<none>>
          %94 = kgen.struct.gep %91[2] : <struct<(pointer<none>, index, index) memoryOnly>>
          pop.store %index2305843009213693952, %94 : !kgen.pointer<index>
          kgen.call @"std::io::io::print[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?],flush:::Bool,file:::FileDescriptor$),Ts.values`=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::reflection::location::SourceLocation\22>>, struct<(index, index, struct<(pointer<none>, index)>)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]]"(%struct, %87, %86, %91, %69, %18, %struct_9, %0, %index1) : (!kgen.struct<(pointer<none>, index)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(index, index, struct<(pointer<none>, index)>)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index)>, i1, index owned) -> ()
          %95 = pop.load %90 : !kgen.pointer<index>
          %96 = index.and %95, %index4611686018427387904
          %97 = index.cmp ne(%96, %index0)
          hlcf.if %97 {
            %104 = pop.load %89 : !kgen.pointer<pointer<none>>
            %105 = pop.pointer.bitcast %104 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
            %106 = pop.offset %105[%idx-8] : !kgen.pointer<scalar<ui8>>
            %107 = pop.pointer.bitcast %106 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
            %108 = kgen.struct.gep %107[0] : <struct<(scalar<index>) memoryOnly>>
            %109 = pop.atomic.rmw sub(%108, %simd_6) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
            %110 = pop.cmp eq(%109, %simd_6) : <1, index>
            %111 = pop.cast_to_builtin %110 : !pop.scalar<bool> to i1
            hlcf.if %111 {
              pop.fence syncscope("") acquire
              pop.aligned_free %106 : <scalar<ui8>>
              hlcf.yield
            } else {
              hlcf.yield
            }
            hlcf.yield
          } else {
            hlcf.yield
          }
          %98 = pop.load %94 : !kgen.pointer<index>
          %99 = index.and %98, %index4611686018427387904
          %100 = index.cmp ne(%99, %index0)
          hlcf.if %100 {
            %104 = pop.load %93 : !kgen.pointer<pointer<none>>
            %105 = pop.pointer.bitcast %104 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
            %106 = pop.offset %105[%idx-8] : !kgen.pointer<scalar<ui8>>
            %107 = pop.pointer.bitcast %106 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
            %108 = kgen.struct.gep %107[0] : <struct<(scalar<index>) memoryOnly>>
            %109 = pop.atomic.rmw sub(%108, %simd_6) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
            %110 = pop.cmp eq(%109, %simd_6) : <1, index>
            %111 = pop.cast_to_builtin %110 : !pop.scalar<bool> to i1
            hlcf.if %111 {
              pop.fence syncscope("") acquire
              pop.aligned_free %106 : <scalar<ui8>>
              hlcf.yield
            } else {
              hlcf.yield
            }
            hlcf.yield
          } else {
            hlcf.yield
          }
          %101 = pop.load %72 : !kgen.pointer<index>
          %102 = index.and %101, %index4611686018427387904
          %103 = index.cmp ne(%102, %index0)
          hlcf.if %103 {
            %104 = pop.load %71 : !kgen.pointer<pointer<none>>
            %105 = pop.pointer.bitcast %104 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
            %106 = pop.offset %105[%idx-8] : !kgen.pointer<scalar<ui8>>
            %107 = pop.pointer.bitcast %106 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
            %108 = kgen.struct.gep %107[0] : <struct<(scalar<index>) memoryOnly>>
            %109 = pop.atomic.rmw sub(%108, %simd_6) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
            %110 = pop.cmp eq(%109, %simd_6) : <1, index>
            %111 = pop.cast_to_builtin %110 : !pop.scalar<bool> to i1
            hlcf.if %111 {
              pop.fence syncscope("") acquire
              pop.aligned_free %106 : <scalar<ui8>>
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
        kgen.call @"std::io::io::print[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?],flush:::Bool,file:::FileDescriptor$),Ts.values`=[[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]]"(%50, %struct_9, %1, %index1) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index)>, i1, index owned) -> ()
        hlcf.if %54 {
          hlcf.yield
        } else {
          %69 = pop.cmp eq(%49, %simd) : <1, ui8>
          %70 = pop.cast_to_builtin %69 : !pop.scalar<bool> to i1
          hlcf.if %70 {
            %71 = pop.load %52 : !kgen.pointer<index>
            %72 = index.and %71, %index4611686018427387904
            %73 = index.cmp ne(%72, %index0)
            hlcf.if %73 {
              %74 = pop.load %51 : !kgen.pointer<pointer<none>>
              %75 = pop.pointer.bitcast %74 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
              %76 = pop.offset %75[%idx-8] : !kgen.pointer<scalar<ui8>>
              %77 = pop.pointer.bitcast %76 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
              %78 = kgen.struct.gep %77[0] : <struct<(scalar<index>) memoryOnly>>
              %79 = pop.atomic.rmw sub(%78, %simd_6) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
              %80 = pop.cmp eq(%79, %simd_6) : <1, index>
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
            hlcf.yield
          } else {
            hlcf.yield
          }
          hlcf.yield
        }
        pop.stack_alloc.lifetime.end(%42) : !kgen.pointer<union<struct<()>, struct<(pointer<none>, index, index) memoryOnly>>>
        hlcf.yield
      } else {
        hlcf.if %54 {
          hlcf.yield
        } else {
          %69 = pop.cmp eq(%49, %simd) : <1, ui8>
          %70 = pop.cast_to_builtin %69 : !pop.scalar<bool> to i1
          hlcf.if %70 {
            %71 = pop.load %52 : !kgen.pointer<index>
            %72 = index.and %71, %index4611686018427387904
            %73 = index.cmp ne(%72, %index0)
            hlcf.if %73 {
              %74 = pop.load %51 : !kgen.pointer<pointer<none>>
              %75 = pop.pointer.bitcast %74 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
              %76 = pop.offset %75[%idx-8] : !kgen.pointer<scalar<ui8>>
              %77 = pop.pointer.bitcast %76 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
              %78 = kgen.struct.gep %77[0] : <struct<(scalar<index>) memoryOnly>>
              %79 = pop.atomic.rmw sub(%78, %simd_6) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
              %80 = pop.cmp eq(%79, %simd_6) : <1, index>
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
            hlcf.yield
          } else {
            hlcf.yield
          }
          hlcf.yield
        }
        pop.stack_alloc.lifetime.end(%42) : !kgen.pointer<union<struct<()>, struct<(pointer<none>, index, index) memoryOnly>>>
        hlcf.yield
      }
      %56 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
      pop.stack_alloc.lifetime.start(%56) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
      %57 = kgen.struct.gep %56[1] : <struct<(pointer<none>, index, index) memoryOnly>>
      pop.store %index44, %57 : !kgen.pointer<index>
      %58 = kgen.struct.gep %56[0] : <struct<(pointer<none>, index, index) memoryOnly>>
      pop.store %20, %58 : !kgen.pointer<pointer<none>>
      %59 = kgen.struct.gep %56[2] : <struct<(pointer<none>, index, index) memoryOnly>>
      pop.store %index2305843009213693952, %59 : !kgen.pointer<index>
      kgen.call @"std::io::io::print[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?],flush:::Bool,file:::FileDescriptor$),Ts.values`=[[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::builtin::error::Error\22>>, struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(variant<struct<()>, struct<(pointer<none>) memoryOnly>>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>]]"(%56, %30, %struct_8, %struct_9, %1, %index1) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.pointer<struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>> read_mem, !kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index)>, i1, index owned) -> ()
      %60 = pop.load %59 : !kgen.pointer<index>
      %61 = index.and %60, %index4611686018427387904
      %62 = index.cmp ne(%61, %index0)
      hlcf.if %62 {
        %69 = pop.load %58 : !kgen.pointer<pointer<none>>
        %70 = pop.pointer.bitcast %69 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
        %71 = pop.offset %70[%idx-8] : !kgen.pointer<scalar<ui8>>
        %72 = pop.pointer.bitcast %71 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
        %73 = kgen.struct.gep %72[0] : <struct<(scalar<index>) memoryOnly>>
        %74 = pop.atomic.rmw sub(%73, %simd_6) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
        %75 = pop.cmp eq(%74, %simd_6) : <1, index>
        %76 = pop.cast_to_builtin %75 : !pop.scalar<bool> to i1
        hlcf.if %76 {
          pop.fence syncscope("") acquire
          pop.aligned_free %71 : <scalar<ui8>>
          hlcf.yield
        } else {
          hlcf.yield
        }
        hlcf.yield
      } else {
        hlcf.yield
      }
      %63 = pop.load %41 : !kgen.pointer<index>
      %64 = index.and %63, %index4611686018427387904
      %65 = index.cmp ne(%64, %index0)
      hlcf.if %65 {
        %69 = pop.load %40 : !kgen.pointer<pointer<none>>
        %70 = pop.pointer.bitcast %69 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
        %71 = pop.offset %70[%idx-8] : !kgen.pointer<scalar<ui8>>
        %72 = pop.pointer.bitcast %71 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
        %73 = kgen.struct.gep %72[0] : <struct<(scalar<index>) memoryOnly>>
        %74 = pop.atomic.rmw sub(%73, %simd_6) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
        %75 = pop.cmp eq(%74, %simd_6) : <1, index>
        %76 = pop.cast_to_builtin %75 : !pop.scalar<bool> to i1
        hlcf.if %76 {
          pop.fence syncscope("") acquire
          pop.aligned_free %71 : <scalar<ui8>>
          hlcf.yield
        } else {
          hlcf.yield
        }
        hlcf.yield
      } else {
        hlcf.yield
      }
      %66 = pop.load %35 : !kgen.pointer<scalar<ui8>>
      %67 = pop.cmp eq(%66, %simd_5) : <1, ui8>
      %68 = pop.cast_to_builtin %67 : !pop.scalar<bool> to i1
      hlcf.if %68 {
        hlcf.yield
      } else {
        %69 = pop.load %35 : !kgen.pointer<scalar<ui8>>
        %70 = pop.cmp eq(%69, %simd) : <1, ui8>
        %71 = pop.cast_to_builtin %70 : !pop.scalar<bool> to i1
        hlcf.if %71 {
          %72 = pop.load %38 : !kgen.pointer<pointer<none>>
          pop.aligned_free %72 : <none>
          hlcf.yield
        } else {
          hlcf.yield
        }
        hlcf.yield
      }
      kgen.return %simd_10 : !pop.scalar<si32>
    } else {
      lit.try.yield
    }
    pop.external_call @KGEN_CompilerRT_DestroyGlobals() : () -> ()
    kgen.return %simd_11 : !pop.scalar<si32>
  }
  kgen.func @"std::builtin::format_int::_write_int[::DType,::Writer,::Int,::StringSlice[::Bool(False), StaticConstantOrigin, *?]]($1&,::SIMD[$0, ::Int(1)],prefix:::StringSlice[::Bool(False), StaticConstantOrigin, *?]),dtype=si64,W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],radix=10,digit_chars={ #interp.memref<{[(#interp.memory_handle<16, \220123456789abcdefghijklmnopqrstuvwxyz\\00\22 string>, const_global, [], [])], []}, 0, 0>, 36 }"(%arg0: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, %arg1: !pop.scalar<si64>, %arg2: !kgen.struct<(pointer<none>, index)>) {
    %idx-8 = index.constant -8
    %simd = kgen.param.constant: scalar<si64> = <10>
    %simd_0 = kgen.param.constant: scalar<bool> = <true>
    %simd_1 = kgen.param.constant: scalar<si64> = <-10>
    %simd_2 = kgen.param.constant: scalar<ui8> = <1>
    %idx63 = index.constant 63
    %idx64 = index.constant 64
    %index0 = kgen.param.constant = <0>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %simd_3 = kgen.param.constant: scalar<index> = <1>
    %string = kgen.param.constant: string = <"-">
    %index2305843009213693952 = kgen.param.constant = <2305843009213693952>
    %index65 = kgen.param.constant = <65>
    %pointer = kgen.param.constant: pointer<none> = <#interp.memref<{[(#memory_handle11, const_global, [], [])], []}, 0, 0>>
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
      kgen.call @"std::collections::string::string::String::write[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](::String&,*$0),Ts.values`2x=[[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]]"(%arg0, %9) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem) -> ()
      %13 = pop.load %12 : !kgen.pointer<index>
      %14 = index.and %13, %index4611686018427387904
      %15 = index.cmp ne(%14, %index0)
      hlcf.if %15 {
        %16 = pop.load %11 : !kgen.pointer<pointer<none>>
        %17 = pop.pointer.bitcast %16 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
        %18 = pop.offset %17[%idx-8] : !kgen.pointer<scalar<ui8>>
        %19 = pop.pointer.bitcast %18 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
        %20 = kgen.struct.gep %19[0] : <struct<(scalar<index>) memoryOnly>>
        %21 = pop.atomic.rmw sub(%20, %simd_3) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
        %22 = pop.cmp eq(%21, %simd_3) : <1, index>
        %23 = pop.cast_to_builtin %22 : !pop.scalar<bool> to i1
        hlcf.if %23 {
          pop.fence syncscope("") acquire
          pop.aligned_free %18 : <scalar<ui8>>
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
    kgen.call @"std::collections::string::string::String::write[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](::String&,*$0),Ts.values`2x=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>]]"(%arg0, %arg2) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
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
      kgen.call @"std::collections::string::string::String::write[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](::String&,*$0),Ts.values`2x=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=1,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>]]"(%arg0, %14) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
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
      kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg0, %48) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      pop.stack_alloc.lifetime.end(%9) : !kgen.pointer<array<65, scalar<ui8>>>
      hlcf.yield
    }
    kgen.return
  }
  kgen.func @"std::builtin::format_int::_write_int[::DType,::Writer,::Int,::StringSlice[::Bool(False), StaticConstantOrigin, *?]]($1&,::SIMD[$0, ::Int(1)],prefix:::StringSlice[::Bool(False), StaticConstantOrigin, *?]),dtype=si64,W=[typevalue<#kgen.instref<\1B\22std::format::_utils::_TotalWritableBytes\22>>, struct<(index) memoryOnly>],radix=10,digit_chars={ #interp.memref<{[(#interp.memory_handle<16, \220123456789abcdefghijklmnopqrstuvwxyz\\00\22 string>, const_global, [], [])], []}, 0, 0>, 36 }"(%arg0: !kgen.struct<(index) memoryOnly>, %arg1: !pop.scalar<si64>, %arg2: !kgen.struct<(pointer<none>, index)>) -> !kgen.struct<(index) memoryOnly> {
    %index1 = kgen.param.constant = <1>
    %simd = kgen.param.constant: scalar<ui8> = <0>
    %simd_0 = kgen.param.constant: scalar<si64> = <0>
    %pointer = kgen.param.constant: pointer<none> = <#interp.memref<{[(#memory_handle11, const_global, [], [])], []}, 0, 0>>
    %index65 = kgen.param.constant = <65>
    %simd_1 = kgen.param.constant: scalar<si64> = <10>
    %simd_2 = kgen.param.constant: scalar<si64> = <-10>
    %simd_3 = kgen.param.constant: scalar<ui8> = <1>
    %idx63 = index.constant 63
    %idx64 = index.constant 64
    %0 = kgen.struct.extract %arg0[0] : <(index) memoryOnly>
    %1 = pop.pointer.bitcast %pointer : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
    %2 = pop.cmp ge(%arg1, %simd_0) : <1, si64>
    %3 = pop.cast_to_builtin %2 : !pop.scalar<bool> to i1
    %4 = pop.cmp lt(%arg1, %simd_0) : <1, si64>
    %5 = pop.cast_to_builtin %4 : !pop.scalar<bool> to i1
    %6 = index.add %0, %index1
    %7 = pop.select %5, %6, %0 : index
    %8 = kgen.struct.extract %arg2[1] : <(pointer<none>, index)>
    %9 = index.add %7, %8
    %10 = pop.cmp eq(%arg1, %simd_0) : <1, si64>
    %11 = pop.cast_to_builtin %10 : !pop.scalar<bool> to i1
    %12 = hlcf.if %11 -> index {
      %14 = pop.load %1 : !kgen.pointer<scalar<ui8>>
      %15 = pop.stack_allocation 1 x array<2, scalar<ui8>> align 1 marked
      pop.stack_alloc.lifetime.start(%15) : !kgen.pointer<array<2, scalar<ui8>>>
      %16 = pop.pointer.bitcast %15 : !kgen.pointer<array<2, scalar<ui8>>> to !kgen.pointer<scalar<ui8>>
      pop.store %14, %16 : !kgen.pointer<scalar<ui8>>
      %17 = pop.offset %16[%index1] : !kgen.pointer<scalar<ui8>>
      pop.store %simd, %17 : !kgen.pointer<scalar<ui8>>
      %18 = index.add %9, %index1
      pop.stack_alloc.lifetime.end(%15) : !kgen.pointer<array<2, scalar<ui8>>>
      hlcf.yield %18 : index
    } else {
      %14 = hlcf.if %3 -> index {
        %49 = hlcf.loop "_loop_0" (%arg3 = %idx63 : index, %arg4 = %arg1 : !pop.scalar<si64>) -> index {
          %50 = pop.cmp ne(%arg4, %simd_0) : <1, si64>
          %51 = pop.cast_to_builtin %50 : !pop.scalar<bool> to i1
          hlcf.if %51 {
            hlcf.yield
          } else {
            hlcf.break "_loop_0" %arg3 : index
          }
          %52 = pop.div %arg4, %simd_1 : !pop.scalar<si64>
          %53 = index.sub %arg3, %index1
          hlcf.continue "_loop_0" %53, %52 : index, !pop.scalar<si64>
        }
        hlcf.yield %49 : index
      } else {
        %49 = hlcf.loop "_loop_0" (%arg3 = %idx63 : index, %arg4 = %arg1 : !pop.scalar<si64>) -> index {
          %50 = pop.cmp ne(%arg4, %simd_0) : <1, si64>
          %51 = pop.cast_to_builtin %50 : !pop.scalar<bool> to i1
          hlcf.if %51 {
            hlcf.yield
          } else {
            hlcf.break "_loop_0" %arg3 : index
          }
          %52 = index.sub %arg3, %index1
          %53 = pop.floordiv %arg4, %simd_2 : !pop.scalar<si64>
          %54 = pop.neg %53 : !pop.scalar<si64>
          hlcf.continue "_loop_0" %52, %54 : index, !pop.scalar<si64>
        }
        hlcf.yield %49 : index
      }
      %15 = index.add %14, %index1
      %16 = pop.stack_allocation 1 x union<struct<()>, index>
      %17 = pop.union.bitcast %16 : <union<struct<()>, index>> as <index>
      pop.store %15, %17 : !kgen.pointer<index>
      %18 = pop.load %16 : !kgen.pointer<union<struct<()>, index>>
      %19 = pop.stack_allocation 1 x union<struct<()>, index>
      %20 = pop.union.bitcast %19 : <union<struct<()>, index>> as <index>
      pop.store %idx64, %20 : !kgen.pointer<index>
      %21 = pop.load %19 : !kgen.pointer<union<struct<()>, index>>
      %22 = pop.stack_allocation 1 x union<struct<()>, index>
      %23 = pop.union.bitcast %22 : <union<struct<()>, index>> as <index>
      pop.store %18, %22 : !kgen.pointer<union<struct<()>, index>>
      %24 = pop.stack_allocation 1 x union<struct<()>, index>
      %25 = pop.union.bitcast %24 : <union<struct<()>, index>> as <index>
      %26 = pop.load %23 : !kgen.pointer<index>
      pop.store %26, %25 : !kgen.pointer<index>
      %27 = pop.load %24 : !kgen.pointer<union<struct<()>, index>>
      %28 = kgen.struct.create(%27, %simd_3) : !kgen.struct<(union<struct<()>, index>, scalar<ui8>)>
      %29 = kgen.struct.create(%28) : !kgen.struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>
      %30 = kgen.struct.create(%29) : !kgen.struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>
      %31 = pop.stack_allocation 1 x union<struct<()>, index>
      %32 = pop.union.bitcast %31 : <union<struct<()>, index>> as <index>
      pop.store %21, %31 : !kgen.pointer<union<struct<()>, index>>
      %33 = pop.stack_allocation 1 x union<struct<()>, index>
      %34 = pop.union.bitcast %33 : <union<struct<()>, index>> as <index>
      %35 = pop.load %32 : !kgen.pointer<index>
      pop.store %35, %34 : !kgen.pointer<index>
      %36 = pop.load %33 : !kgen.pointer<union<struct<()>, index>>
      %37 = kgen.struct.create(%36, %simd_3) : !kgen.struct<(union<struct<()>, index>, scalar<ui8>)>
      %38 = kgen.struct.create(%37) : !kgen.struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>
      %39 = kgen.struct.create(%38) : !kgen.struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>
      %40 = kgen.struct.create(%30) : !kgen.struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>
      %41 = kgen.struct.create(%39) : !kgen.struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>
      %42 = kgen.struct.create(%40, %41) : !kgen.struct<(struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>, struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>) memoryOnly>
      %43 = kgen.call @"std::builtin::builtin_slice::ContiguousSlice::indices(::ContiguousSlice,::Int)"(%42, %index65) : (!kgen.struct<(struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>, struct<(struct<(struct<(struct<(union<struct<()>, index>, scalar<ui8>)>)>)>)>) memoryOnly>, index) -> !kgen.struct<(struct<(index, index)>)>
      %44 = kgen.struct.extract %43[0] : <(struct<(index, index)>)>
      %45 = kgen.struct.extract %44[0] : <(index, index)>
      %46 = kgen.struct.extract %44[1] : <(index, index)>
      %47 = index.sub %46, %45
      %48 = index.add %9, %47
      hlcf.yield %48 : index
    }
    %13 = kgen.struct.create(%12) : !kgen.struct<(index) memoryOnly>
    kgen.return %13 : !kgen.struct<(index) memoryOnly>
  }
  kgen.func @"std::builtin::format_int::_write_int[::DType,::Writer,::Int,::StringSlice[::Bool(False), StaticConstantOrigin, *?]]($1&,::SIMD[$0, ::Int(1)],prefix:::StringSlice[::Bool(False), StaticConstantOrigin, *?]),dtype=si64,W=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::collections::string::string::String\\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>],radix=10,digit_chars={ #interp.memref<{[(#interp.memory_handle<16, \220123456789abcdefghijklmnopqrstuvwxyz\\00\22 string>, const_global, [], [])], []}, 0, 0>, 36 }"(%arg0: !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, %arg1: !pop.scalar<si64>, %arg2: !kgen.struct<(pointer<none>, index)>) {
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
    %pointer = kgen.param.constant: pointer<none> = <#interp.memref<{[(#memory_handle11, const_global, [], [])], []}, 0, 0>>
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
      kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg0, %22) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
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
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg0, %arg2) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
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
      kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=1"(%arg0, %14) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
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
      kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=1"(%arg0, %48) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
      pop.stack_alloc.lifetime.end(%9) : !kgen.pointer<array<65, scalar<ui8>>>
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
    %pointer = kgen.param.constant: pointer<none> = <#interp.memref<{[(#memory_handle11, const_global, [], [])], []}, 0, 0>>
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
    %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle8, const_global, [], [])], []}, 0, 0>, 0 }>
    kgen.call tail @"std::builtin::format_int::_write_int[::DType,::Writer,::Int,::StringSlice[::Bool(False), StaticConstantOrigin, *?]]($1&,::SIMD[$0, ::Int(1)],prefix:::StringSlice[::Bool(False), StaticConstantOrigin, *?]),dtype=si64,W=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>],radix=10,digit_chars={ #interp.memref<{[(#interp.memory_handle<16, \220123456789abcdefghijklmnopqrstuvwxyz\\00\22 string>, const_global, [], [])], []}, 0, 0>, 36 }"(%arg0, %arg1, %struct) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !pop.scalar<si64>, !kgen.struct<(pointer<none>, index)>) -> ()
    kgen.return
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
  kgen.func @"std::gpu::host::device_context::_string_from_owned_charptr[LITImmutOrigin,::Origin[::Bool(False), $0]](::Optional[::CStringSlice[$0, $1]])"(%arg0: !kgen.struct<(struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>)>) -> !kgen.struct<(pointer<none>, index, index) memoryOnly> {
    %index-1 = kgen.param.constant = <-1>
    %index0 = kgen.param.constant = <0>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %simd = kgen.param.constant: scalar<index> = <1>
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ #interp.memref<{[(#memory_handle8, const_global, [], [])], []}, 0, 0>, 0 }>
    %simd_0 = kgen.param.constant: scalar<uindex> = <18446744073709551615>
    %simd_1 = kgen.param.constant: scalar<uindex> = <1>
    %simd_2 = kgen.param.constant: scalar<uindex> = <0>
    %simd_3 = kgen.param.constant: scalar<ui8> = <0>
    %false = index.bool.constant false
    %idx-8 = index.constant -8
    %0 = pop.stack_allocation 1 x struct<(array<1, pointer<none>>)>
    %1 = kgen.struct.extract %arg0[0] : <(struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>)>
    %2 = kgen.struct.extract %1[0] : <(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>
    %3 = kgen.struct.extract %2[0] : <(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>
    %4 = kgen.struct.extract %3[0] : <(struct<(array<1, pointer<none>>)>) memoryOnly>
    pop.store %4, %0 : !kgen.pointer<struct<(array<1, pointer<none>>)>>
    %5 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
    %6 = kgen.struct.gep %5[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.stack_alloc.lifetime.start(%5) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %7 = kgen.struct.gep %5[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.store %index-9223372036854775808, %7 : !kgen.pointer<index>
    %8 = pop.pointer.bitcast %0 : !kgen.pointer<struct<(array<1, pointer<none>>)>> to !kgen.pointer<pointer<none>>
    %9 = pop.load %0 : !kgen.pointer<struct<(array<1, pointer<none>>)>>
    %10 = kgen.struct.extract %9[0] : <(array<1, pointer<none>>)>
    %11 = pop.array.get %10[0] : !pop.array<1, pointer<none>>
    %12 = pop.array.create [%11] : !pop.array<1, pointer<none>>
    %13 = kgen.struct.create(%12) : !kgen.struct<(array<1, pointer<none>>)>
    %14 = pop.stack_allocation 1 x struct<(array<1, pointer<none>>)>
    pop.store %13, %14 : !kgen.pointer<struct<(array<1, pointer<none>>)>>
    %15 = pop.pointer.bitcast %14 : !kgen.pointer<struct<(array<1, pointer<none>>)>> to !kgen.pointer<pointer<none>>
    %16 = pop.load %15 : !kgen.pointer<pointer<none>>
    %17 = pop.pointer_to_index %16 : <none>
    %18 = index.cmp eq(%17, %index0)
    %19 = pop.select %18, %index0, %index-1 : index
    %20 = index.cmp eq(%19, %index-1)
    hlcf.if %20 {
      %31 = pop.load %7 : !kgen.pointer<index>
      %32 = index.and %31, %index4611686018427387904
      %33 = index.cmp ne(%32, %index0)
      hlcf.if %33 {
        %40 = pop.load %6 : !kgen.pointer<pointer<none>>
        %41 = pop.pointer.bitcast %40 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
        %42 = pop.offset %41[%idx-8] : !kgen.pointer<scalar<ui8>>
        %43 = pop.pointer.bitcast %42 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
        %44 = kgen.struct.gep %43[0] : <struct<(scalar<index>) memoryOnly>>
        %45 = pop.atomic.rmw sub(%44, %simd) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
        %46 = pop.cmp eq(%45, %simd) : <1, index>
        %47 = pop.cast_to_builtin %46 : !pop.scalar<bool> to i1
        hlcf.if %47 {
          pop.fence syncscope("") acquire
          pop.aligned_free %42 : <scalar<ui8>>
          hlcf.yield
        } else {
          hlcf.yield
        }
        hlcf.yield
      } else {
        hlcf.yield
      }
      pop.stack_alloc.lifetime.end(%5) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
      %34 = pop.load %8 : !kgen.pointer<pointer<none>>
      pop.stack_alloc.lifetime.start(%5) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
      %35 = pop.pointer.bitcast %34 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %36 = hlcf.loop "_loop_0" (%arg1 = %simd_2 : !pop.scalar<uindex>) -> !pop.scalar<uindex> {
        %40 = pop.cast fast %arg1 : !pop.scalar<uindex> to !pop.scalar<index>
        %41 = pop.cast_to_builtin %40 : !pop.scalar<index> to index
        %42 = pop.offset %35[%41] : !kgen.pointer<scalar<ui8>>
        %43 = pop.cmp lt(%arg1, %simd_0) : <1, uindex>
        %44 = pop.cast_to_builtin %43 : !pop.scalar<bool> to i1
        %45 = hlcf.if %44 -> i1 {
          %47 = pop.load %42 : !kgen.pointer<scalar<ui8>>
          %48 = pop.cmp ne(%47, %simd_3) : <1, ui8>
          %49 = pop.cast_to_builtin %48 : !pop.scalar<bool> to i1
          hlcf.yield %49 : i1
        } else {
          hlcf.yield %false : i1
        }
        hlcf.if %45 {
          hlcf.yield
        } else {
          hlcf.break "_loop_0" %arg1 : !pop.scalar<uindex>
        }
        %46 = pop.add %arg1, %simd_1 : !pop.scalar<uindex>
        hlcf.continue "_loop_0" %46 : !pop.scalar<uindex>
      }
      %37 = pop.cast fast %36 : !pop.scalar<uindex> to !pop.scalar<index>
      %38 = pop.cast_to_builtin %37 : !pop.scalar<index> to index
      %39 = kgen.struct.create(%34, %38) : !kgen.struct<(pointer<none>, index)>
      kgen.call @"std::collections::string::string::String::__init__[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?]),Ts.values`2x=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>]]"(%39, %struct, %5) : (!kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> byref_result) -> ()
      hlcf.yield
    } else {
      hlcf.yield
    }
    %21 = pop.load %0 : !kgen.pointer<struct<(array<1, pointer<none>>)>>
    %22 = kgen.struct.create(%21) : !kgen.struct<(struct<(array<1, pointer<none>>)>) memoryOnly>
    %23 = kgen.struct.create(%22) : !kgen.struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>
    %24 = kgen.struct.create(%23) : !kgen.struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>
    %25 = kgen.struct.create(%24) : !kgen.struct<(struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>)>
    pop.external_call @AsyncRT_DeviceContext_strfree(%25) : (!kgen.struct<(struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>)>) -> ()
    %26 = pop.load %6 : !kgen.pointer<pointer<none>>
    %27 = kgen.struct.gep %5[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %28 = pop.load %27 : !kgen.pointer<index>
    %29 = pop.load %7 : !kgen.pointer<index>
    pop.stack_alloc.lifetime.end(%5) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %30 = kgen.struct.create(%26, %28, %29) : !kgen.struct<(pointer<none>, index, index) memoryOnly>
    kgen.return %30 : !kgen.struct<(pointer<none>, index, index) memoryOnly>
  }
  kgen.func @"std::io::io::_flush(::FileDescriptor)"(%arg0: index) no_inline {
    %pointer = kgen.param.constant: pointer<none> = <#interp.memref<{[(#memory_handle12, const_global, [], [])], []}, 0, 0>>
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
  kgen.func @"std::io::io::print[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?],flush:::Bool,file:::FileDescriptor$),Ts.values`=[[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::builtin::error::Error\22>>, struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(variant<struct<()>, struct<(pointer<none>) memoryOnly>>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>]]"(%arg0: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, %arg1: !kgen.pointer<struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>> read_mem, %arg2: !kgen.struct<(pointer<none>, index)>, %arg3: !kgen.struct<(pointer<none>, index)>, %arg4: i1, %arg5: index owned) no_inline {
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
    kgen.call @"std::builtin::error::Error::write_to[::Writer](::Error,$0&),writer.T`2x1=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::io::file_descriptor::FileDescriptor\\22>>, index],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>]"(%arg1, %1) : (!kgen.pointer<struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>> read_mem, !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut) -> ()
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
  kgen.func @"std::io::io::print[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?],flush:::Bool,file:::FileDescriptor$),Ts.values`=[[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]]"(%arg0: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, %arg1: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, %arg2: !kgen.struct<(pointer<none>, index)>, %arg3: !kgen.struct<(pointer<none>, index)>, %arg4: i1, %arg5: index owned) no_inline {
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
      %37 = pop.load %4 : !kgen.pointer<pointer<none>>
      hlcf.yield %37 : !kgen.pointer<none>
    }
    %11 = kgen.struct.gep %arg0[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %12 = pop.load %6 : !kgen.pointer<index>
    %13 = index.and %12, %index-9223372036854775808
    %14 = index.cmp ne(%13, %index0)
    %15 = hlcf.if %14 -> index {
      %37 = pop.load %6 : !kgen.pointer<index>
      %38 = index.and %37, %index2233785415175766016
      %39 = index.shrs %38, %index56
      hlcf.yield %39 : index
    } else {
      %37 = pop.load %11 : !kgen.pointer<index>
      hlcf.yield %37 : index
    }
    %16 = kgen.struct.create(%10, %15) : !kgen.struct<(pointer<none>, index)>
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%1, %16) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%1, %arg2) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    %17 = kgen.struct.gep %arg1[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %18 = pop.pointer.bitcast %arg1 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
    %19 = kgen.struct.gep %arg1[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %20 = pop.load %19 : !kgen.pointer<index>
    %21 = index.and %20, %index-9223372036854775808
    %22 = index.cmp ne(%21, %index0)
    %23 = hlcf.if %22 -> !kgen.pointer<none> {
      hlcf.yield %18 : !kgen.pointer<none>
    } else {
      %37 = pop.load %17 : !kgen.pointer<pointer<none>>
      hlcf.yield %37 : !kgen.pointer<none>
    }
    %24 = kgen.struct.gep %arg1[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %25 = pop.load %19 : !kgen.pointer<index>
    %26 = index.and %25, %index-9223372036854775808
    %27 = index.cmp ne(%26, %index0)
    %28 = hlcf.if %27 -> index {
      %37 = pop.load %19 : !kgen.pointer<index>
      %38 = index.and %37, %index2233785415175766016
      %39 = index.shrs %38, %index56
      hlcf.yield %39 : index
    } else {
      %37 = pop.load %24 : !kgen.pointer<index>
      hlcf.yield %37 : index
    }
    %29 = kgen.struct.create(%23, %28) : !kgen.struct<(pointer<none>, index)>
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%1, %29) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%1, %arg3) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    %30 = pop.load %3 : !kgen.pointer<pointer<index>>
    %31 = kgen.struct.gep %1[0] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>>
    %32 = kgen.struct.gep %31[0] : <struct<(array<4096, scalar<ui8>>) memoryOnly>>
    %33 = pop.pointer.bitcast %32 : !kgen.pointer<array<4096, scalar<ui8>>> to !kgen.pointer<none>
    %34 = pop.load %2 : !kgen.pointer<index>
    %35 = pop.load %30 : !kgen.pointer<index>
    %36 = pop.external_call @write(%35, %33, %34) : (index, !kgen.pointer<none>, index) -> index
    pop.store %index0, %2 : !kgen.pointer<index>
    pop.stack_alloc.lifetime.end(%1) : !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>>
    hlcf.if %arg4 {
      %37 = pop.load %0 : !kgen.pointer<index>
      kgen.call tail @"std::io::io::_flush(::FileDescriptor)"(%37) : (index) -> ()
      hlcf.yield
    } else {
      hlcf.yield
    }
    kgen.return
  }
  kgen.func @"std::io::io::print[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](*$0,sep:::StringSlice[::Bool(False), StaticConstantOrigin, *?],end:::StringSlice[::Bool(False), StaticConstantOrigin, *?],flush:::Bool,file:::FileDescriptor$),Ts.values`=[[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]]"(%arg0: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, %arg1: !kgen.struct<(pointer<none>, index)>, %arg2: i1, %arg3: index owned) no_inline {
    %index0 = kgen.param.constant = <0>
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %index56 = kgen.param.constant = <56>
    %index2233785415175766016 = kgen.param.constant = <2233785415175766016>
    %0 = pop.stack_allocation 1 x index
    pop.store %arg3, %0 : !kgen.pointer<index>
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
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::io::file_descriptor::FileDescriptor\22>>, index],stack_buffer_bytes=4096,string.mut`2x1=0"(%1, %arg1) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    %17 = pop.load %3 : !kgen.pointer<pointer<index>>
    %18 = kgen.struct.gep %1[0] : <struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>>
    %19 = kgen.struct.gep %18[0] : <struct<(array<4096, scalar<ui8>>) memoryOnly>>
    %20 = pop.pointer.bitcast %19 : !kgen.pointer<array<4096, scalar<ui8>>> to !kgen.pointer<none>
    %21 = pop.load %2 : !kgen.pointer<index>
    %22 = pop.load %17 : !kgen.pointer<index>
    %23 = pop.external_call @write(%22, %20, %21) : (index, !kgen.pointer<none>, index) -> index
    pop.store %index0, %2 : !kgen.pointer<index>
    pop.stack_alloc.lifetime.end(%1) : !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<index>) memoryOnly>>
    hlcf.if %arg2 {
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
  kgen.func @"std::builtin::variadics::VariadicPack::_write_to[LITImmutOrigin,LITImmutOrigin,LITImmutOrigin,::Origin[::Bool(False), $7],::Origin[::Bool(False), $8],::Origin[::Bool(False), $9],::Bool,::Writer](::VariadicPack[$0, $1, $2, $3, $4, $5, $6],$14&,::StringSlice[::Bool(False), $7, $10],::StringSlice[::Bool(False), $8, $11],::StringSlice[::Bool(False), $9, $12]){#kgen.param_list.reduce($4, base=::Bool(True), reducer=[::Bool, KGENParamList[::AnyType], index] ::Bool(conforms_to($1[$2], AnyType & ImplicitlyDestructible & Writable)) if $0 else $0)._mlir_value}_REMOVED_ARG,element_trait=type,element_types.values`2=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::builtin::int::Int\22>>, index], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::builtin::int::Int\22>>, index]],is_repr=0,writer.T`2x4=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]"(%arg0: !kgen.struct<(pointer<none>, index)>, %arg1: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, %arg2: index, %arg3: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, %arg4: index, %arg5: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, %arg6: !kgen.struct<(pointer<none>, index)>, %arg7: !kgen.struct<(pointer<none>, index)>, %arg8: !kgen.struct<(pointer<none>, index)>) {
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %index0 = kgen.param.constant = <0>
    %index56 = kgen.param.constant = <56>
    %index2233785415175766016 = kgen.param.constant = <2233785415175766016>
    kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg5, %arg6) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg5, %arg0) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg5, %arg8) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    %0 = kgen.struct.gep %arg1[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %1 = pop.pointer.bitcast %arg1 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
    %2 = kgen.struct.gep %arg1[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %3 = pop.load %2 : !kgen.pointer<index>
    %4 = index.and %3, %index-9223372036854775808
    %5 = index.cmp ne(%4, %index0)
    %6 = hlcf.if %5 -> !kgen.pointer<none> {
      hlcf.yield %1 : !kgen.pointer<none>
    } else {
      %30 = pop.load %0 : !kgen.pointer<pointer<none>>
      hlcf.yield %30 : !kgen.pointer<none>
    }
    %7 = kgen.struct.gep %arg1[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %8 = pop.load %2 : !kgen.pointer<index>
    %9 = index.and %8, %index-9223372036854775808
    %10 = index.cmp ne(%9, %index0)
    %11 = hlcf.if %10 -> index {
      %30 = pop.load %2 : !kgen.pointer<index>
      %31 = index.and %30, %index2233785415175766016
      %32 = index.shrs %31, %index56
      hlcf.yield %32 : index
    } else {
      %30 = pop.load %7 : !kgen.pointer<index>
      hlcf.yield %30 : index
    }
    %12 = kgen.struct.create(%6, %11) : !kgen.struct<(pointer<none>, index)>
    kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg5, %12) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg5, %arg8) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    %13 = pop.cast_from_builtin %arg2 : index to !pop.scalar<index>
    %14 = pop.cast %13 : !pop.scalar<index> to !pop.scalar<si64>
    kgen.call @"std::collections::string::string::String::write[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](::String&,*$0),Ts.values`2x=[[typevalue<#kgen.instref<\1B\22std::builtin::simd::SIMD,dtype=si64,size=1\22>>, scalar<si64>]]"(%arg5, %14) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !pop.scalar<si64>) -> ()
    kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg5, %arg8) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    %15 = kgen.struct.gep %arg3[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %16 = pop.pointer.bitcast %arg3 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
    %17 = kgen.struct.gep %arg3[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %18 = pop.load %17 : !kgen.pointer<index>
    %19 = index.and %18, %index-9223372036854775808
    %20 = index.cmp ne(%19, %index0)
    %21 = hlcf.if %20 -> !kgen.pointer<none> {
      hlcf.yield %16 : !kgen.pointer<none>
    } else {
      %30 = pop.load %15 : !kgen.pointer<pointer<none>>
      hlcf.yield %30 : !kgen.pointer<none>
    }
    %22 = kgen.struct.gep %arg3[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %23 = pop.load %17 : !kgen.pointer<index>
    %24 = index.and %23, %index-9223372036854775808
    %25 = index.cmp ne(%24, %index0)
    %26 = hlcf.if %25 -> index {
      %30 = pop.load %17 : !kgen.pointer<index>
      %31 = index.and %30, %index2233785415175766016
      %32 = index.shrs %31, %index56
      hlcf.yield %32 : index
    } else {
      %30 = pop.load %22 : !kgen.pointer<index>
      hlcf.yield %30 : index
    }
    %27 = kgen.struct.create(%21, %26) : !kgen.struct<(pointer<none>, index)>
    kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg5, %27) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg5, %arg8) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    %28 = pop.cast_from_builtin %arg4 : index to !pop.scalar<index>
    %29 = pop.cast %28 : !pop.scalar<index> to !pop.scalar<si64>
    kgen.call @"std::collections::string::string::String::write[KGENParamList[::Writable],*::Writable,LITImmutOrigin,::Origin[::Bool(False), $2]](::String&,*$0),Ts.values`2x=[[typevalue<#kgen.instref<\1B\22std::builtin::simd::SIMD,dtype=si64,size=1\22>>, scalar<si64>]]"(%arg5, %29) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !pop.scalar<si64>) -> ()
    kgen.call @"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"(%arg5, %arg7) : (!kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    kgen.return
  }
  kgen.func @"std::builtin::variadics::VariadicPack::_write_to[LITImmutOrigin,LITImmutOrigin,LITImmutOrigin,::Origin[::Bool(False), $7],::Origin[::Bool(False), $8],::Origin[::Bool(False), $9],::Bool,::Writer](::VariadicPack[$0, $1, $2, $3, $4, $5, $6],$14&,::StringSlice[::Bool(False), $7, $10],::StringSlice[::Bool(False), $8, $11],::StringSlice[::Bool(False), $9, $12]){#kgen.param_list.reduce($4, base=::Bool(True), reducer=[::Bool, KGENParamList[::AnyType], index] ::Bool(conforms_to($1[$2], AnyType & ImplicitlyDestructible & Writable)) if $0 else $0)._mlir_value}_REMOVED_ARG,element_trait=type,element_types.values`2=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::builtin::int::Int\22>>, index], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::builtin::int::Int\22>>, index]],is_repr=0,writer.T`2x4=[typevalue<#kgen.instref<\1B\22std::format::_utils::_TotalWritableBytes\22>>, struct<(index) memoryOnly>]"(%arg0: !kgen.struct<(pointer<none>, index)>, %arg1: !kgen.struct<(pointer<none>, index, index) memoryOnly>, %arg2: index, %arg3: !kgen.struct<(pointer<none>, index, index) memoryOnly>, %arg4: index, %arg5: !kgen.struct<(index) memoryOnly>, %arg6: !kgen.struct<(pointer<none>, index)>, %arg7: !kgen.struct<(pointer<none>, index)>, %arg8: !kgen.struct<(pointer<none>, index)>) -> !kgen.struct<(index) memoryOnly> {
    %index2233785415175766016 = kgen.param.constant = <2233785415175766016>
    %index56 = kgen.param.constant = <56>
    %index0 = kgen.param.constant = <0>
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %0 = kgen.struct.extract %arg1[1] : <(pointer<none>, index, index) memoryOnly>
    %1 = kgen.struct.extract %arg1[2] : <(pointer<none>, index, index) memoryOnly>
    %2 = kgen.struct.extract %arg3[1] : <(pointer<none>, index, index) memoryOnly>
    %3 = kgen.struct.extract %arg3[2] : <(pointer<none>, index, index) memoryOnly>
    %4 = kgen.struct.extract %arg5[0] : <(index) memoryOnly>
    %5 = kgen.struct.extract %arg0[1] : <(pointer<none>, index)>
    %6 = kgen.struct.extract %arg6[1] : <(pointer<none>, index)>
    %7 = index.add %4, %6
    %8 = index.add %7, %5
    %9 = kgen.struct.extract %arg8[1] : <(pointer<none>, index)>
    %10 = index.add %8, %9
    %11 = index.and %1, %index-9223372036854775808
    %12 = index.cmp ne(%11, %index0)
    %13 = hlcf.if %12 -> index {
      %35 = index.and %1, %index2233785415175766016
      %36 = index.shrs %35, %index56
      hlcf.yield %36 : index
    } else {
      hlcf.yield %0 : index
    }
    %14 = index.add %10, %13
    %15 = index.add %14, %9
    %16 = pop.cast_from_builtin %arg2 : index to !pop.scalar<index>
    %17 = pop.cast %16 : !pop.scalar<index> to !pop.scalar<si64>
    %18 = kgen.struct.create(%15) : !kgen.struct<(index) memoryOnly>
    %19 = kgen.call @"std::builtin::simd::SIMD::write_to[::Writer](::SIMD[$0, $1],$2&),dtype=si64,size=1,writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_TotalWritableBytes\22>>, struct<(index) memoryOnly>]"(%17, %18) : (!pop.scalar<si64>, !kgen.struct<(index) memoryOnly>) -> !kgen.struct<(index) memoryOnly>
    %20 = kgen.struct.extract %19[0] : <(index) memoryOnly>
    %21 = index.add %20, %9
    %22 = index.and %3, %index-9223372036854775808
    %23 = index.cmp ne(%22, %index0)
    %24 = hlcf.if %23 -> index {
      %35 = index.and %3, %index2233785415175766016
      %36 = index.shrs %35, %index56
      hlcf.yield %36 : index
    } else {
      hlcf.yield %2 : index
    }
    %25 = index.add %21, %24
    %26 = index.add %25, %9
    %27 = pop.cast_from_builtin %arg4 : index to !pop.scalar<index>
    %28 = pop.cast %27 : !pop.scalar<index> to !pop.scalar<si64>
    %29 = kgen.struct.create(%26) : !kgen.struct<(index) memoryOnly>
    %30 = kgen.call @"std::builtin::simd::SIMD::write_to[::Writer](::SIMD[$0, $1],$2&),dtype=si64,size=1,writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_TotalWritableBytes\22>>, struct<(index) memoryOnly>]"(%28, %29) : (!pop.scalar<si64>, !kgen.struct<(index) memoryOnly>) -> !kgen.struct<(index) memoryOnly>
    %31 = kgen.struct.extract %30[0] : <(index) memoryOnly>
    %32 = kgen.struct.extract %arg7[1] : <(pointer<none>, index)>
    %33 = index.add %31, %32
    %34 = kgen.struct.create(%33) : !kgen.struct<(index) memoryOnly>
    kgen.return %34 : !kgen.struct<(index) memoryOnly>
  }
  kgen.func @"std::builtin::variadics::VariadicPack::_write_to[LITImmutOrigin,LITImmutOrigin,LITImmutOrigin,::Origin[::Bool(False), $7],::Origin[::Bool(False), $8],::Origin[::Bool(False), $9],::Bool,::Writer](::VariadicPack[$0, $1, $2, $3, $4, $5, $6],$14&,::StringSlice[::Bool(False), $7, $10],::StringSlice[::Bool(False), $8, $11],::StringSlice[::Bool(False), $9, $12]){#kgen.param_list.reduce($4, base=::Bool(True), reducer=[::Bool, KGENParamList[::AnyType], index] ::Bool(conforms_to($1[$2], AnyType & ImplicitlyDestructible & Writable)) if $0 else $0)._mlir_value}_REMOVED_ARG,element_trait=type,element_types.values`2=[[typevalue<#kgen.instref<\1B\22std::collections::string::string_slice::StringSlice,mut=0,origin._mlir_origin`={  },origin={  }\22>>, struct<(pointer<none>, index)>], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::builtin::int::Int\22>>, index], [typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>], [typevalue<#kgen.instref<\1B\22std::builtin::int::Int\22>>, index]],is_repr=0,writer.T`2x4=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::collections::string::string::String\\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>]"(%arg0: !kgen.struct<(pointer<none>, index)>, %arg1: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, %arg2: index, %arg3: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, %arg4: index, %arg5: !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, %arg6: !kgen.struct<(pointer<none>, index)>, %arg7: !kgen.struct<(pointer<none>, index)>, %arg8: !kgen.struct<(pointer<none>, index)>) {
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %index0 = kgen.param.constant = <0>
    %index56 = kgen.param.constant = <56>
    %index2233785415175766016 = kgen.param.constant = <2233785415175766016>
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg5, %arg6) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg5, %arg0) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg5, %arg8) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    %0 = kgen.struct.gep %arg1[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %1 = pop.pointer.bitcast %arg1 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
    %2 = kgen.struct.gep %arg1[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %3 = pop.load %2 : !kgen.pointer<index>
    %4 = index.and %3, %index-9223372036854775808
    %5 = index.cmp ne(%4, %index0)
    %6 = hlcf.if %5 -> !kgen.pointer<none> {
      hlcf.yield %1 : !kgen.pointer<none>
    } else {
      %30 = pop.load %0 : !kgen.pointer<pointer<none>>
      hlcf.yield %30 : !kgen.pointer<none>
    }
    %7 = kgen.struct.gep %arg1[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %8 = pop.load %2 : !kgen.pointer<index>
    %9 = index.and %8, %index-9223372036854775808
    %10 = index.cmp ne(%9, %index0)
    %11 = hlcf.if %10 -> index {
      %30 = pop.load %2 : !kgen.pointer<index>
      %31 = index.and %30, %index2233785415175766016
      %32 = index.shrs %31, %index56
      hlcf.yield %32 : index
    } else {
      %30 = pop.load %7 : !kgen.pointer<index>
      hlcf.yield %30 : index
    }
    %12 = kgen.struct.create(%6, %11) : !kgen.struct<(pointer<none>, index)>
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg5, %12) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg5, %arg8) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    %13 = pop.cast_from_builtin %arg2 : index to !pop.scalar<index>
    %14 = pop.cast %13 : !pop.scalar<index> to !pop.scalar<si64>
    kgen.call @"std::builtin::simd::SIMD::write_to[::Writer](::SIMD[$0, $1],$2&),dtype=si64,size=1,writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::collections::string::string::String\\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>]"(%14, %arg5) : (!pop.scalar<si64>, !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut) -> ()
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg5, %arg8) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    %15 = kgen.struct.gep %arg3[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %16 = pop.pointer.bitcast %arg3 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
    %17 = kgen.struct.gep %arg3[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %18 = pop.load %17 : !kgen.pointer<index>
    %19 = index.and %18, %index-9223372036854775808
    %20 = index.cmp ne(%19, %index0)
    %21 = hlcf.if %20 -> !kgen.pointer<none> {
      hlcf.yield %16 : !kgen.pointer<none>
    } else {
      %30 = pop.load %15 : !kgen.pointer<pointer<none>>
      hlcf.yield %30 : !kgen.pointer<none>
    }
    %22 = kgen.struct.gep %arg3[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %23 = pop.load %17 : !kgen.pointer<index>
    %24 = index.and %23, %index-9223372036854775808
    %25 = index.cmp ne(%24, %index0)
    %26 = hlcf.if %25 -> index {
      %30 = pop.load %17 : !kgen.pointer<index>
      %31 = index.and %30, %index2233785415175766016
      %32 = index.shrs %31, %index56
      hlcf.yield %32 : index
    } else {
      %30 = pop.load %22 : !kgen.pointer<index>
      hlcf.yield %30 : index
    }
    %27 = kgen.struct.create(%21, %26) : !kgen.struct<(pointer<none>, index)>
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg5, %27) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg5, %arg8) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    %28 = pop.cast_from_builtin %arg4 : index to !pop.scalar<index>
    %29 = pop.cast %28 : !pop.scalar<index> to !pop.scalar<si64>
    kgen.call @"std::builtin::simd::SIMD::write_to[::Writer](::SIMD[$0, $1],$2&),dtype=si64,size=1,writer.T`2x=[typevalue<#kgen.instref<\1B\22std::format::_utils::_WriteBufferStack,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<\\1B\\22std::collections::string::string::String\\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096\22>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>]"(%29, %arg5) : (!pop.scalar<si64>, !kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut) -> ()
    kgen.call @"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"(%arg5, %arg7) : (!kgen.pointer<struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, index, pointer<struct<(pointer<none>, index, index) memoryOnly>>) memoryOnly>> mut, !kgen.struct<(pointer<none>, index)>) -> ()
    kgen.return
  }
  kgen.func @"std::gpu::host::device_context::_raise_checked_impl[LITImmutOrigin,::Origin[::Bool(False), $0]](::Optional[::CStringSlice[$0, $1]],::String,::SourceLocation)_REMOVED_ARG"(%arg0: !kgen.struct<(struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>)>, %arg1: !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, %arg2: !kgen.struct<(index, index, struct<(pointer<none>, index)>)>) throws -> (i1, !kgen.struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>) no_inline {
    %idx-8 = index.constant -8
    %index-9223372036854775808 = kgen.param.constant = <-9223372036854775808>
    %index2233785415175766016 = kgen.param.constant = <2233785415175766016>
    %index56 = kgen.param.constant = <56>
    %string = kgen.param.constant: string = <" ">
    %index1 = kgen.param.constant = <1>
    %string_0 = kgen.param.constant: string = <"">
    %index2305843009213693952 = kgen.param.constant = <2305843009213693952>
    %index4611686018427387904 = kgen.param.constant = <4611686018427387904>
    %simd = kgen.param.constant: scalar<index> = <1>
    %index0 = kgen.param.constant = <0>
    %index-1 = kgen.param.constant = <-1>
    %0 = kgen.param.constant: i1 = <1>
    %1 = pop.pointer.bitcast %arg1 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
    %2 = kgen.struct.gep %arg1[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %3 = pop.string.address %string
    %4 = pop.pointer.bitcast %3 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %5 = pop.string.address %string_0
    %6 = pop.pointer.bitcast %5 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %7 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
    pop.stack_alloc.lifetime.start(%7) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %8 = kgen.call @"std::gpu::host::device_context::_string_from_owned_charptr[LITImmutOrigin,::Origin[::Bool(False), $0]](::Optional[::CStringSlice[$0, $1]])"(%arg0) : (!kgen.struct<(struct<(struct<(struct<(struct<(array<1, pointer<none>>)>) memoryOnly>)>)>)>) -> !kgen.struct<(pointer<none>, index, index) memoryOnly>
    pop.store %8, %7 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %9 = kgen.struct.gep %arg1[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %10 = kgen.struct.gep %arg1[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %11 = pop.load %10 : !kgen.pointer<index>
    %12 = index.and %11, %index-9223372036854775808
    %13 = index.cmp ne(%12, %index0)
    %14 = hlcf.if %13 -> index {
      %66 = pop.load %10 : !kgen.pointer<index>
      %67 = index.and %66, %index2233785415175766016
      %68 = index.shrs %67, %index56
      hlcf.yield %68 : index
    } else {
      %66 = pop.load %9 : !kgen.pointer<index>
      hlcf.yield %66 : index
    }
    %15 = index.cmp sgt(%14, %index0)
    %16 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
    %17 = kgen.struct.gep %16[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %18 = kgen.struct.gep %16[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %19 = kgen.struct.gep %16[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    hlcf.if %15 {
      %66 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
      pop.stack_alloc.lifetime.start(%66) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
      %67 = kgen.struct.gep %66[1] : <struct<(pointer<none>, index, index) memoryOnly>>
      pop.store %index1, %67 : !kgen.pointer<index>
      %68 = kgen.struct.gep %66[0] : <struct<(pointer<none>, index, index) memoryOnly>>
      pop.store %4, %68 : !kgen.pointer<pointer<none>>
      %69 = kgen.struct.gep %66[2] : <struct<(pointer<none>, index, index) memoryOnly>>
      pop.store %index2305843009213693952, %69 : !kgen.pointer<index>
      %70 = pop.load %10 : !kgen.pointer<index>
      %71 = index.and %70, %index-9223372036854775808
      %72 = index.cmp ne(%71, %index0)
      %73 = hlcf.if %72 -> !kgen.pointer<none> {
        hlcf.yield %1 : !kgen.pointer<none>
      } else {
        %96 = pop.load %2 : !kgen.pointer<pointer<none>>
        hlcf.yield %96 : !kgen.pointer<none>
      }
      %74 = pop.load %10 : !kgen.pointer<index>
      %75 = index.and %74, %index-9223372036854775808
      %76 = index.cmp ne(%75, %index0)
      %77 = hlcf.if %76 -> index {
        %96 = pop.load %10 : !kgen.pointer<index>
        %97 = index.and %96, %index2233785415175766016
        %98 = index.shrs %97, %index56
        hlcf.yield %98 : index
      } else {
        %96 = pop.load %9 : !kgen.pointer<index>
        hlcf.yield %96 : index
      }
      %78 = kgen.struct.create(%73, %77) : !kgen.struct<(pointer<none>, index)>
      %79 = pop.pointer.bitcast %66 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
      %80 = pop.load %69 : !kgen.pointer<index>
      %81 = index.and %80, %index-9223372036854775808
      %82 = index.cmp ne(%81, %index0)
      %83 = hlcf.if %82 -> !kgen.pointer<none> {
        hlcf.yield %79 : !kgen.pointer<none>
      } else {
        %96 = pop.load %68 : !kgen.pointer<pointer<none>>
        hlcf.yield %96 : !kgen.pointer<none>
      }
      %84 = pop.load %69 : !kgen.pointer<index>
      %85 = index.and %84, %index-9223372036854775808
      %86 = index.cmp ne(%85, %index0)
      %87 = hlcf.if %86 -> index {
        %96 = pop.load %69 : !kgen.pointer<index>
        %97 = index.and %96, %index2233785415175766016
        %98 = index.shrs %97, %index56
        hlcf.yield %98 : index
      } else {
        %96 = pop.load %67 : !kgen.pointer<index>
        hlcf.yield %96 : index
      }
      %88 = kgen.struct.create(%83, %87) : !kgen.struct<(pointer<none>, index)>
      %89 = kgen.call @"std::collections::string::string::String::_add[::Bool,LITOrigin[$0._mlir_value],::Origin[$0, $1],::Bool,LITOrigin[$3._mlir_value],::Origin[$3, $4]](::Span[$0, $1, ::SIMD[::DType(uint8), ::Int(1)], $2],::Span[$3, $4, ::SIMD[::DType(uint8), ::Int(1)], $5]),lhs.mut`2x=0,rhs.mut`2x3=0"(%88, %78) : (!kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index)>) -> !kgen.struct<(pointer<none>, index, index) memoryOnly>
      %90 = kgen.struct.extract %89[0] : <(pointer<none>, index, index) memoryOnly>
      %91 = kgen.struct.extract %89[1] : <(pointer<none>, index, index) memoryOnly>
      %92 = kgen.struct.extract %89[2] : <(pointer<none>, index, index) memoryOnly>
      %93 = pop.load %69 : !kgen.pointer<index>
      %94 = index.and %93, %index4611686018427387904
      %95 = index.cmp ne(%94, %index0)
      hlcf.if %95 {
        %96 = pop.load %68 : !kgen.pointer<pointer<none>>
        %97 = pop.pointer.bitcast %96 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
        %98 = pop.offset %97[%idx-8] : !kgen.pointer<scalar<ui8>>
        %99 = pop.pointer.bitcast %98 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
        %100 = kgen.struct.gep %99[0] : <struct<(scalar<index>) memoryOnly>>
        %101 = pop.atomic.rmw sub(%100, %simd) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
        %102 = pop.cmp eq(%101, %simd) : <1, index>
        %103 = pop.cast_to_builtin %102 : !pop.scalar<bool> to i1
        hlcf.if %103 {
          pop.fence syncscope("") acquire
          pop.aligned_free %98 : <scalar<ui8>>
          hlcf.yield
        } else {
          hlcf.yield
        }
        hlcf.yield
      } else {
        hlcf.yield
      }
      pop.stack_alloc.lifetime.end(%66) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
      pop.stack_alloc.lifetime.start(%16) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
      pop.store %90, %19 : !kgen.pointer<pointer<none>>
      pop.store %91, %18 : !kgen.pointer<index>
      pop.store %92, %17 : !kgen.pointer<index>
      hlcf.yield
    } else {
      pop.stack_alloc.lifetime.start(%16) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
      pop.store %6, %19 : !kgen.pointer<pointer<none>>
      pop.store %index0, %18 : !kgen.pointer<index>
      pop.store %index2305843009213693952, %17 : !kgen.pointer<index>
      hlcf.yield
    }
    %20 = pop.pointer.bitcast %16 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
    %21 = pop.load %17 : !kgen.pointer<index>
    %22 = index.and %21, %index-9223372036854775808
    %23 = index.cmp ne(%22, %index0)
    %24 = hlcf.if %23 -> !kgen.pointer<none> {
      hlcf.yield %20 : !kgen.pointer<none>
    } else {
      %66 = pop.load %19 : !kgen.pointer<pointer<none>>
      hlcf.yield %66 : !kgen.pointer<none>
    }
    %25 = pop.load %17 : !kgen.pointer<index>
    %26 = index.and %25, %index-9223372036854775808
    %27 = index.cmp ne(%26, %index0)
    %28 = hlcf.if %27 -> index {
      %66 = pop.load %17 : !kgen.pointer<index>
      %67 = index.and %66, %index2233785415175766016
      %68 = index.shrs %67, %index56
      hlcf.yield %68 : index
    } else {
      %66 = pop.load %18 : !kgen.pointer<index>
      hlcf.yield %66 : index
    }
    %29 = kgen.struct.create(%24, %28) : !kgen.struct<(pointer<none>, index)>
    %30 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
    %31 = kgen.struct.gep %30[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    pop.stack_alloc.lifetime.start(%30) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %32 = kgen.struct.gep %7[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %33 = pop.pointer.bitcast %7 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> to !kgen.pointer<none>
    %34 = kgen.struct.gep %7[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %35 = pop.load %34 : !kgen.pointer<index>
    %36 = index.and %35, %index-9223372036854775808
    %37 = index.cmp ne(%36, %index0)
    %38 = hlcf.if %37 -> !kgen.pointer<none> {
      hlcf.yield %33 : !kgen.pointer<none>
    } else {
      %66 = pop.load %32 : !kgen.pointer<pointer<none>>
      hlcf.yield %66 : !kgen.pointer<none>
    }
    %39 = kgen.struct.gep %7[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %40 = pop.load %34 : !kgen.pointer<index>
    %41 = index.and %40, %index-9223372036854775808
    %42 = index.cmp ne(%41, %index0)
    %43 = hlcf.if %42 -> index {
      %66 = pop.load %34 : !kgen.pointer<index>
      %67 = index.and %66, %index2233785415175766016
      %68 = index.shrs %67, %index56
      hlcf.yield %68 : index
    } else {
      %66 = pop.load %39 : !kgen.pointer<index>
      hlcf.yield %66 : index
    }
    %44 = kgen.struct.create(%38, %43) : !kgen.struct<(pointer<none>, index)>
    %45 = kgen.call @"std::collections::string::string::String::_add[::Bool,LITOrigin[$0._mlir_value],::Origin[$0, $1],::Bool,LITOrigin[$3._mlir_value],::Origin[$3, $4]](::Span[$0, $1, ::SIMD[::DType(uint8), ::Int(1)], $2],::Span[$3, $4, ::SIMD[::DType(uint8), ::Int(1)], $5]),lhs.mut`2x=0,rhs.mut`2x3=0"(%44, %29) : (!kgen.struct<(pointer<none>, index)>, !kgen.struct<(pointer<none>, index)>) -> !kgen.struct<(pointer<none>, index, index) memoryOnly>
    pop.store %45, %30 : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %46 = pop.load %34 : !kgen.pointer<index>
    %47 = index.and %46, %index4611686018427387904
    %48 = index.cmp ne(%47, %index0)
    hlcf.if %48 {
      %66 = pop.load %32 : !kgen.pointer<pointer<none>>
      %67 = pop.pointer.bitcast %66 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %68 = pop.offset %67[%idx-8] : !kgen.pointer<scalar<ui8>>
      %69 = pop.pointer.bitcast %68 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
      %70 = kgen.struct.gep %69[0] : <struct<(scalar<index>) memoryOnly>>
      %71 = pop.atomic.rmw sub(%70, %simd) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
      %72 = pop.cmp eq(%71, %simd) : <1, index>
      %73 = pop.cast_to_builtin %72 : !pop.scalar<bool> to i1
      hlcf.if %73 {
        pop.fence syncscope("") acquire
        pop.aligned_free %68 : <scalar<ui8>>
        hlcf.yield
      } else {
        hlcf.yield
      }
      hlcf.yield
    } else {
      hlcf.yield
    }
    pop.stack_alloc.lifetime.end(%7) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %49 = pop.load %17 : !kgen.pointer<index>
    %50 = index.and %49, %index4611686018427387904
    %51 = index.cmp ne(%50, %index0)
    hlcf.if %51 {
      %66 = pop.load %19 : !kgen.pointer<pointer<none>>
      %67 = pop.pointer.bitcast %66 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %68 = pop.offset %67[%idx-8] : !kgen.pointer<scalar<ui8>>
      %69 = pop.pointer.bitcast %68 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
      %70 = kgen.struct.gep %69[0] : <struct<(scalar<index>) memoryOnly>>
      %71 = pop.atomic.rmw sub(%70, %simd) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
      %72 = pop.cmp eq(%71, %simd) : <1, index>
      %73 = pop.cast_to_builtin %72 : !pop.scalar<bool> to i1
      hlcf.if %73 {
        pop.fence syncscope("") acquire
        pop.aligned_free %68 : <scalar<ui8>>
        hlcf.yield
      } else {
        hlcf.yield
      }
      hlcf.yield
    } else {
      hlcf.yield
    }
    pop.stack_alloc.lifetime.end(%16) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %52 = pop.stack_allocation 1 x struct<(pointer<none>, index, index) memoryOnly> align 1 marked
    pop.stack_alloc.lifetime.start(%52) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    kgen.call @"std::reflection::location::SourceLocation::prefix[::Writable](::SourceLocation,$0),T=[typevalue<#kgen.instref<\1B\22std::collections::string::string::String\22>>, struct<(pointer<none>, index, index) memoryOnly>]"(%arg2, %30, %52) : (!kgen.struct<(index, index, struct<(pointer<none>, index)>)>, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> read_mem, !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>> byref_result) -> ()
    %53 = kgen.struct.gep %30[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %54 = pop.load %53 : !kgen.pointer<index>
    %55 = index.and %54, %index4611686018427387904
    %56 = index.cmp ne(%55, %index0)
    hlcf.if %56 {
      %66 = pop.load %31 : !kgen.pointer<pointer<none>>
      %67 = pop.pointer.bitcast %66 : !kgen.pointer<none> to !kgen.pointer<scalar<ui8>>
      %68 = pop.offset %67[%idx-8] : !kgen.pointer<scalar<ui8>>
      %69 = pop.pointer.bitcast %68 : !kgen.pointer<scalar<ui8>> to !kgen.pointer<struct<(scalar<index>) memoryOnly>>
      %70 = kgen.struct.gep %69[0] : <struct<(scalar<index>) memoryOnly>>
      %71 = pop.atomic.rmw sub(%70, %simd) syncscope("") seq_cst : !kgen.pointer<scalar<index>>
      %72 = pop.cmp eq(%71, %simd) : <1, index>
      %73 = pop.cast_to_builtin %72 : !pop.scalar<bool> to i1
      hlcf.if %73 {
        pop.fence syncscope("") acquire
        pop.aligned_free %68 : <scalar<ui8>>
        hlcf.yield
      } else {
        hlcf.yield
      }
      hlcf.yield
    } else {
      hlcf.yield
    }
    pop.stack_alloc.lifetime.end(%30) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %57 = kgen.struct.gep %52[0] : <struct<(pointer<none>, index, index) memoryOnly>>
    %58 = pop.load %57 : !kgen.pointer<pointer<none>>
    %59 = kgen.struct.gep %52[1] : <struct<(pointer<none>, index, index) memoryOnly>>
    %60 = pop.load %59 : !kgen.pointer<index>
    %61 = kgen.struct.gep %52[2] : <struct<(pointer<none>, index, index) memoryOnly>>
    %62 = pop.load %61 : !kgen.pointer<index>
    %63 = kgen.call @"std::builtin::error::StackTrace::collect_if_enabled(::Int)"(%index-1) : (index) -> !kgen.struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>
    pop.stack_alloc.lifetime.end(%52) : !kgen.pointer<struct<(pointer<none>, index, index) memoryOnly>>
    %64 = kgen.struct.create(%58, %60, %62) : !kgen.struct<(pointer<none>, index, index) memoryOnly>
    %65 = kgen.struct.create(%64, %63) : !kgen.struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>
    kgen.return %0, %65 : i1, !kgen.struct<(struct<(pointer<none>, index, index) memoryOnly>, struct<(struct<(struct<(struct<(union<struct<()>, struct<(pointer<none>) memoryOnly>>, scalar<ui8>)>) memoryOnly>) memoryOnly>) memoryOnly>) memoryOnly>
  }
}


