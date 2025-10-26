; ModuleID = 'BitcodeBuffer'
source_filename = "c_api_transform"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-apple-macosx26.0.1-unknown"

%Target.Cpu.Feature.Set = type { [5 x i64] }
%Target.Cpu = type { ptr, %Target.Cpu.Feature.Set, i6, [7 x i8] }
%Target.Cpu.Model = type { { ptr, i64 }, { ptr, i64 }, %Target.Cpu.Feature.Set }
%SemanticVersion.Range = type { %SemanticVersion, %SemanticVersion }
%SemanticVersion = type { i64, i64, i64, { ptr, i64 }, { ptr, i64 } }
%Target.DynamicLinker = type { [255 x i8], i8 }
%builtin.CallingConvention.CommonOptions = type { { i64, i8, [7 x i8] } }
%builtin.StackTrace = type { i64, { ptr, i64 } }

@0 = private unnamed_addr constant { ptr, i16, [6 x i8] } { ptr null, i16 0, [6 x i8] undef }, align 8
@1 = private unnamed_addr constant { ptr, i16, [6 x i8] } { ptr undef, i16 1, [6 x i8] undef }, align 8
@2 = private unnamed_addr constant { ptr, i16, [6 x i8] } { ptr undef, i16 2, [6 x i8] undef }, align 8
@__anon_1817 = internal unnamed_addr constant [10 x i8] c"operation\00", align 1
@3 = private unnamed_addr constant { ptr, i16, [6 x i8] } { ptr undef, i16 3, [6 x i8] undef }, align 8
@__anon_1828 = internal unnamed_addr constant [5 x i8] c"name\00", align 1
@__anon_1831 = internal unnamed_addr constant [10 x i8] c"func.call\00", align 1
@__anon_1835 = internal unnamed_addr constant [16 x i8] c"result-bindings\00", align 1
@__anon_1840 = internal unnamed_addr constant [9 x i8] c"%result0\00", align 1
@__anon_1845 = internal unnamed_addr constant [13 x i8] c"result-types\00", align 1
@__anon_1852 = internal unnamed_addr constant [11 x i8] c"attributes\00", align 1
@__anon_1859 = internal unnamed_addr constant [8 x i8] c":callee\00", align 1
@__anon_1755 = internal unnamed_addr constant [5 x i8] c"call\00", align 1
@__anon_1761 = internal unnamed_addr constant [6 x i8] c"@test\00", align 1
@__anon_1766 = internal unnamed_addr constant [4 x i8] c"i64\00", align 1
@builtin.zig_backend = internal unnamed_addr constant i64 2, align 8, !dbg !0
@start.simplified_logic = internal unnamed_addr constant i1 false, align 1, !dbg !20
@builtin.output_mode = internal unnamed_addr constant i2 -2, align 1, !dbg !24
@Target.Cpu.Feature.Set.empty = internal unnamed_addr constant %Target.Cpu.Feature.Set zeroinitializer, align 8, !dbg !32
@builtin.cpu = internal unnamed_addr constant %Target.Cpu { ptr @Target.aarch64.cpu.apple_m2, %Target.Cpu.Feature.Set { [5 x i64] [i64 1333202426378888154, i64 7658872716159647446, i64 147492888299700224, i64 1565743148175360, i64 0] }, i6 6, [7 x i8] undef }, align 8, !dbg !479
@Target.aarch64.cpu.apple_m2 = internal unnamed_addr constant %Target.Cpu.Model { { ptr, i64 } { ptr @__anon_1885, i64 8 }, { ptr, i64 } { ptr @__anon_1887, i64 8 }, %Target.Cpu.Feature.Set { [5 x i64] [i64 1152922054362661642, i64 2251799813717636, i64 144115188344291328, i64 422214612549632, i64 0] } }, align 8, !dbg !507
@builtin.os = internal unnamed_addr constant { { <{ %SemanticVersion.Range, [64 x i8] }>, i3, [7 x i8] }, i6, [7 x i8] } { { <{ %SemanticVersion.Range, [64 x i8] }>, i3, [7 x i8] } { <{ %SemanticVersion.Range, [64 x i8] }> <{ %SemanticVersion.Range { %SemanticVersion { i64 26, i64 0, i64 1, { ptr, i64 } zeroinitializer, { ptr, i64 } zeroinitializer }, %SemanticVersion { i64 26, i64 0, i64 1, { ptr, i64 } zeroinitializer, { ptr, i64 } zeroinitializer } }, [64 x i8] undef }>, i3 1, [7 x i8] undef }, i6 19, [7 x i8] undef }, align 8, !dbg !481
@builtin.abi = internal unnamed_addr constant i5 0, align 1, !dbg !483
@builtin.object_format = internal unnamed_addr constant i4 5, align 1, !dbg !485
@Target.DynamicLinker.none = internal unnamed_addr constant %Target.DynamicLinker { [255 x i8] undef, i8 0 }, align 1, !dbg !487
@builtin.target = internal unnamed_addr constant { %Target.Cpu, { { <{ %SemanticVersion.Range, [64 x i8] }>, i3, [7 x i8] }, i6, [7 x i8] }, i5, i4, %Target.DynamicLinker, [6 x i8] } { %Target.Cpu { ptr @Target.aarch64.cpu.apple_m2, %Target.Cpu.Feature.Set { [5 x i64] [i64 1333202426378888154, i64 7658872716159647446, i64 147492888299700224, i64 1565743148175360, i64 0] }, i6 6, [7 x i8] undef }, { { <{ %SemanticVersion.Range, [64 x i8] }>, i3, [7 x i8] }, i6, [7 x i8] } { { <{ %SemanticVersion.Range, [64 x i8] }>, i3, [7 x i8] } { <{ %SemanticVersion.Range, [64 x i8] }> <{ %SemanticVersion.Range { %SemanticVersion { i64 26, i64 0, i64 1, { ptr, i64 } zeroinitializer, { ptr, i64 } zeroinitializer }, %SemanticVersion { i64 26, i64 0, i64 1, { ptr, i64 } zeroinitializer, { ptr, i64 } zeroinitializer } }, [64 x i8] undef }>, i3 1, [7 x i8] undef }, i6 19, [7 x i8] undef }, i5 0, i4 5, %Target.DynamicLinker { [255 x i8] [i8 47, i8 117, i8 115, i8 114, i8 47, i8 108, i8 105, i8 98, i8 47, i8 100, i8 121, i8 108, i8 100, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef], i8 13 }, [6 x i8] undef }, align 8, !dbg !496
@builtin.CallingConvention.c = internal unnamed_addr constant { <{ %builtin.CallingConvention.CommonOptions, [8 x i8] }>, i8, [7 x i8] } { <{ %builtin.CallingConvention.CommonOptions, [8 x i8] }> <{ %builtin.CallingConvention.CommonOptions { { i64, i8, [7 x i8] } { i64 undef, i8 0, [7 x i8] undef } }, [8 x i8] undef }>, i8 21, [7 x i8] undef }, align 8, !dbg !505
@__anon_1885 = internal unnamed_addr constant [9 x i8] c"apple_m2\00", align 1
@__anon_1887 = internal unnamed_addr constant [9 x i8] c"apple-m2\00", align 1

; Function Attrs: nounwind uwtable
define internal fastcc void @c_api_transform.transformCallToOperation(ptr noalias nonnull sret({ ptr, i16, [6 x i8] }) %0, ptr nonnull %1, ptr align 1 %2, ptr align 1 %3) unnamed_addr #0 !dbg !518 {
Entry:
  %4 = alloca { ptr, i16, [6 x i8] }, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca ptr, align 8
  %13 = alloca ptr, align 8
  %14 = alloca ptr, align 8
  %15 = alloca ptr, align 8
  %16 = alloca ptr, align 8
  %17 = alloca ptr, align 8
  %18 = alloca ptr, align 8
  %19 = alloca ptr, align 8
  %20 = alloca ptr, align 8
  %21 = alloca ptr, align 8
  %22 = alloca ptr, align 8
  %23 = alloca ptr, align 8
  %24 = alloca ptr, align 8
  %25 = alloca ptr, align 8
  %26 = alloca ptr, align 8
  %27 = alloca ptr, align 8
  %28 = alloca ptr, align 8
  %29 = alloca ptr, align 8
  %30 = alloca ptr, align 8
  %31 = alloca ptr, align 8
  %32 = alloca ptr, align 8
  %33 = alloca ptr, align 8
  %34 = alloca ptr, align 8
  %35 = alloca ptr, align 8
  %36 = alloca ptr, align 8
  %37 = alloca ptr, align 8
  %38 = alloca ptr, align 8
  %39 = alloca ptr, align 8
  %40 = alloca ptr, align 8
  %41 = alloca ptr, align 8
  %42 = alloca ptr, align 8
  %43 = alloca ptr, align 8
  %44 = alloca ptr, align 8
  %45 = alloca ptr, align 8
  %46 = alloca ptr, align 8
  %47 = alloca ptr, align 8
  %48 = alloca i64, align 8
  %49 = alloca ptr, align 8
  %50 = alloca ptr, align 8
  %51 = alloca ptr, align 8
  store ptr %2, ptr %51, align 8, !dbg !532
    #dbg_declare(ptr %51, !533, !DIExpression(), !532)
  store ptr %3, ptr %50, align 8, !dbg !532
    #dbg_declare(ptr %50, !534, !DIExpression(), !532)
  %52 = icmp eq ptr %2, null, !dbg !535
  br i1 %52, label %Then, label %Else, !dbg !537

Block:                                            ; preds = %Else1
  %53 = call ptr @value_get_list(ptr align 1 %2, ptr readonly align 1 %3), !dbg !539
  %54 = icmp ne ptr %53, null, !dbg !539
  br i1 %54, label %Then2, label %Else2, !dbg !539

Block1:                                           ; preds = %Else, %Then
  %55 = phi i1 [ true, %Then ], [ %56, %Else ], !dbg !535
  br i1 %55, label %Then1, label %Else1, !dbg !535

Then:                                             ; preds = %Entry
  br label %Block1, !dbg !541

Else:                                             ; preds = %Entry
  %56 = icmp eq ptr %3, null, !dbg !541
  br label %Block1, !dbg !541

Then1:                                            ; preds = %Block1
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @0, i64 16, i1 false), !dbg !543
  ret void, !dbg !543

Else1:                                            ; preds = %Block1
  br label %Block, !dbg !544

Block2:                                           ; preds = %Then2
  %57 = phi ptr [ %53, %Then2 ], !dbg !546
  store ptr %57, ptr %49, align 8, !dbg !546
    #dbg_declare(ptr %49, !547, !DIExpression(), !546)
  %58 = call i64 @vector_value_len(ptr readonly align 1 %57), !dbg !548
  store i64 %58, ptr %48, align 8, !dbg !548
    #dbg_declare(ptr %48, !549, !DIExpression(), !548)
  %59 = icmp ult i64 %58, 3, !dbg !550
  br i1 %59, label %Then3, label %Else3, !dbg !550

Then2:                                            ; preds = %Block
  br label %Block2, !dbg !552

Else2:                                            ; preds = %Block
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !554
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @1, i64 16, i1 false), !dbg !554
  ret void, !dbg !554

Block3:                                           ; preds = %Else3
  %60 = call ptr @vector_value_at(ptr readonly align 1 %57, i64 0), !dbg !555
  store ptr %60, ptr %47, align 8, !dbg !555
    #dbg_declare(ptr %47, !556, !DIExpression(), !555)
  %61 = call ptr @vector_value_at(ptr readonly align 1 %57, i64 1), !dbg !557
  store ptr %61, ptr %46, align 8, !dbg !557
    #dbg_declare(ptr %46, !558, !DIExpression(), !557)
  %62 = call ptr @vector_value_at(ptr readonly align 1 %57, i64 2), !dbg !559
  store ptr %62, ptr %45, align 8, !dbg !559
    #dbg_declare(ptr %45, !560, !DIExpression(), !559)
  %63 = call ptr @value_get_atom(ptr align 1 %2, ptr readonly align 1 %60), !dbg !561
  store ptr %63, ptr %44, align 8, !dbg !561
    #dbg_declare(ptr %44, !562, !DIExpression(), !561)
  %64 = call ptr @value_create_identifier(ptr align 1 %2, ptr nonnull readonly align 1 @__anon_1817), !dbg !563
  %65 = icmp ne ptr %64, null, !dbg !563
  br i1 %65, label %Then4, label %Else4, !dbg !563

Then3:                                            ; preds = %Block2
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !565
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !567
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @2, i64 16, i1 false), !dbg !567
  ret void, !dbg !567

Else3:                                            ; preds = %Block2
  br label %Block3, !dbg !568

Block4:                                           ; preds = %Then4
  %66 = phi ptr [ %64, %Then4 ], !dbg !570
  store ptr %66, ptr %43, align 8, !dbg !570
    #dbg_declare(ptr %43, !571, !DIExpression(), !570)
  %67 = call ptr @value_create_identifier(ptr align 1 %2, ptr nonnull readonly align 1 @__anon_1828), !dbg !572
  %68 = icmp ne ptr %67, null, !dbg !572
  br i1 %68, label %Then5, label %Else5, !dbg !572

Then4:                                            ; preds = %Block3
  br label %Block4, !dbg !574

Else4:                                            ; preds = %Block3
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !576
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !577
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !578
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !578
  ret void, !dbg !578

Block5:                                           ; preds = %Then5
  %69 = phi ptr [ %67, %Then5 ], !dbg !579
  store ptr %69, ptr %42, align 8, !dbg !579
    #dbg_declare(ptr %42, !580, !DIExpression(), !579)
  %70 = call ptr @value_create_identifier(ptr align 1 %2, ptr nonnull readonly align 1 @__anon_1831), !dbg !581
  %71 = icmp ne ptr %70, null, !dbg !581
  br i1 %71, label %Then6, label %Else6, !dbg !581

Then5:                                            ; preds = %Block4
  br label %Block5, !dbg !583

Else5:                                            ; preds = %Block4
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !585
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !586
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !587
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !587
  ret void, !dbg !587

Block6:                                           ; preds = %Then6
  %72 = phi ptr [ %70, %Then6 ], !dbg !588
  store ptr %72, ptr %41, align 8, !dbg !588
    #dbg_declare(ptr %41, !589, !DIExpression(), !588)
  %73 = call ptr @vector_value_create(ptr align 1 %2), !dbg !590
  %74 = icmp ne ptr %73, null, !dbg !590
  br i1 %74, label %Then7, label %Else7, !dbg !590

Then6:                                            ; preds = %Block5
  br label %Block6, !dbg !592

Else6:                                            ; preds = %Block5
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !594
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !595
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !596
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !596
  ret void, !dbg !596

Block7:                                           ; preds = %Then7
  %75 = phi ptr [ %73, %Then7 ], !dbg !597
  store ptr %75, ptr %40, align 8, !dbg !597
    #dbg_declare(ptr %40, !598, !DIExpression(), !597)
  %76 = call ptr @vector_value_push(ptr align 1 %2, ptr align 1 %75, ptr align 1 %69), !dbg !599
  %77 = icmp ne ptr %76, null, !dbg !599
  br i1 %77, label %Then8, label %Else8, !dbg !599

Then7:                                            ; preds = %Block6
  br label %Block7, !dbg !601

Else7:                                            ; preds = %Block6
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !603
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !604
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !605
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !605
  ret void, !dbg !605

Block8:                                           ; preds = %Then8
  %78 = phi ptr [ %76, %Then8 ], !dbg !606
  store ptr %78, ptr %39, align 8, !dbg !606
    #dbg_declare(ptr %39, !607, !DIExpression(), !606)
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %75), !dbg !608
  %79 = call ptr @vector_value_push(ptr align 1 %2, ptr align 1 %78, ptr align 1 %72), !dbg !609
  %80 = icmp ne ptr %79, null, !dbg !609
  br i1 %80, label %Then9, label %Else9, !dbg !609

Then8:                                            ; preds = %Block7
  br label %Block8, !dbg !611

Else8:                                            ; preds = %Block7
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !613
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !614
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !615
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !615
  ret void, !dbg !615

Block9:                                           ; preds = %Then9
  %81 = phi ptr [ %79, %Then9 ], !dbg !616
  store ptr %81, ptr %38, align 8, !dbg !616
    #dbg_declare(ptr %38, !617, !DIExpression(), !616)
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %78), !dbg !618
  %82 = call ptr @value_create_list(ptr align 1 %2, ptr align 1 %81), !dbg !619
  %83 = icmp ne ptr %82, null, !dbg !619
  br i1 %83, label %Then10, label %Else10, !dbg !619

Then9:                                            ; preds = %Block8
  br label %Block9, !dbg !621

Else9:                                            ; preds = %Block8
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !623
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !624
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !625
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !625
  ret void, !dbg !625

Block10:                                          ; preds = %Then10
  %84 = phi ptr [ %82, %Then10 ], !dbg !626
  store ptr %84, ptr %37, align 8, !dbg !626
    #dbg_declare(ptr %37, !627, !DIExpression(), !626)
  %85 = call ptr @value_create_identifier(ptr align 1 %2, ptr nonnull readonly align 1 @__anon_1835), !dbg !628
  %86 = icmp ne ptr %85, null, !dbg !628
  br i1 %86, label %Then11, label %Else11, !dbg !628

Then10:                                           ; preds = %Block9
  br label %Block10, !dbg !630

Else10:                                           ; preds = %Block9
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !632
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !633
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !634
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !634
  ret void, !dbg !634

Block11:                                          ; preds = %Then11
  %87 = phi ptr [ %85, %Then11 ], !dbg !635
  store ptr %87, ptr %36, align 8, !dbg !635
    #dbg_declare(ptr %36, !636, !DIExpression(), !635)
  %88 = call ptr @value_create_identifier(ptr align 1 %2, ptr nonnull readonly align 1 @__anon_1840), !dbg !637
  %89 = icmp ne ptr %88, null, !dbg !637
  br i1 %89, label %Then12, label %Else12, !dbg !637

Then11:                                           ; preds = %Block10
  br label %Block11, !dbg !639

Else11:                                           ; preds = %Block10
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !641
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !642
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !643
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !643
  ret void, !dbg !643

Block12:                                          ; preds = %Then12
  %90 = phi ptr [ %88, %Then12 ], !dbg !644
  store ptr %90, ptr %35, align 8, !dbg !644
    #dbg_declare(ptr %35, !645, !DIExpression(), !644)
  %91 = call ptr @vector_value_create(ptr align 1 %2), !dbg !646
  %92 = icmp ne ptr %91, null, !dbg !646
  br i1 %92, label %Then13, label %Else13, !dbg !646

Then12:                                           ; preds = %Block11
  br label %Block12, !dbg !648

Else12:                                           ; preds = %Block11
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !650
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !651
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !652
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !652
  ret void, !dbg !652

Block13:                                          ; preds = %Then13
  %93 = phi ptr [ %91, %Then13 ], !dbg !653
  store ptr %93, ptr %34, align 8, !dbg !653
    #dbg_declare(ptr %34, !654, !DIExpression(), !653)
  %94 = call ptr @vector_value_push(ptr align 1 %2, ptr align 1 %93, ptr align 1 %90), !dbg !655
  %95 = icmp ne ptr %94, null, !dbg !655
  br i1 %95, label %Then14, label %Else14, !dbg !655

Then13:                                           ; preds = %Block12
  br label %Block13, !dbg !657

Else13:                                           ; preds = %Block12
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !659
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !660
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !661
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !661
  ret void, !dbg !661

Block14:                                          ; preds = %Then14
  %96 = phi ptr [ %94, %Then14 ], !dbg !662
  store ptr %96, ptr %33, align 8, !dbg !662
    #dbg_declare(ptr %33, !663, !DIExpression(), !662)
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %93), !dbg !664
  %97 = call ptr @value_create_list(ptr align 1 %2, ptr align 1 %96), !dbg !665
  %98 = icmp ne ptr %97, null, !dbg !665
  br i1 %98, label %Then15, label %Else15, !dbg !665

Then14:                                           ; preds = %Block13
  br label %Block14, !dbg !667

Else14:                                           ; preds = %Block13
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !669
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !670
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !671
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !671
  ret void, !dbg !671

Block15:                                          ; preds = %Then15
  %99 = phi ptr [ %97, %Then15 ], !dbg !672
  store ptr %99, ptr %32, align 8, !dbg !672
    #dbg_declare(ptr %32, !673, !DIExpression(), !672)
  %100 = call ptr @vector_value_create(ptr align 1 %2), !dbg !674
  %101 = icmp ne ptr %100, null, !dbg !674
  br i1 %101, label %Then16, label %Else16, !dbg !674

Then15:                                           ; preds = %Block14
  br label %Block15, !dbg !676

Else15:                                           ; preds = %Block14
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !678
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !679
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !680
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !680
  ret void, !dbg !680

Block16:                                          ; preds = %Then16
  %102 = phi ptr [ %100, %Then16 ], !dbg !681
  store ptr %102, ptr %31, align 8, !dbg !681
    #dbg_declare(ptr %31, !682, !DIExpression(), !681)
  %103 = call ptr @vector_value_push(ptr align 1 %2, ptr align 1 %102, ptr align 1 %87), !dbg !683
  %104 = icmp ne ptr %103, null, !dbg !683
  br i1 %104, label %Then17, label %Else17, !dbg !683

Then16:                                           ; preds = %Block15
  br label %Block16, !dbg !685

Else16:                                           ; preds = %Block15
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !687
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !688
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !689
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !689
  ret void, !dbg !689

Block17:                                          ; preds = %Then17
  %105 = phi ptr [ %103, %Then17 ], !dbg !690
  store ptr %105, ptr %30, align 8, !dbg !690
    #dbg_declare(ptr %30, !691, !DIExpression(), !690)
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %102), !dbg !692
  %106 = call ptr @vector_value_push(ptr align 1 %2, ptr align 1 %105, ptr align 1 %99), !dbg !693
  %107 = icmp ne ptr %106, null, !dbg !693
  br i1 %107, label %Then18, label %Else18, !dbg !693

Then17:                                           ; preds = %Block16
  br label %Block17, !dbg !695

Else17:                                           ; preds = %Block16
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !697
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !698
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !699
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !699
  ret void, !dbg !699

Block18:                                          ; preds = %Then18
  %108 = phi ptr [ %106, %Then18 ], !dbg !700
  store ptr %108, ptr %29, align 8, !dbg !700
    #dbg_declare(ptr %29, !701, !DIExpression(), !700)
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %105), !dbg !702
  %109 = call ptr @value_create_list(ptr align 1 %2, ptr align 1 %108), !dbg !703
  %110 = icmp ne ptr %109, null, !dbg !703
  br i1 %110, label %Then19, label %Else19, !dbg !703

Then18:                                           ; preds = %Block17
  br label %Block18, !dbg !705

Else18:                                           ; preds = %Block17
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !707
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !708
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !709
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !709
  ret void, !dbg !709

Block19:                                          ; preds = %Then19
  %111 = phi ptr [ %109, %Then19 ], !dbg !710
  store ptr %111, ptr %28, align 8, !dbg !710
    #dbg_declare(ptr %28, !711, !DIExpression(), !710)
  %112 = call ptr @value_create_identifier(ptr align 1 %2, ptr nonnull readonly align 1 @__anon_1845), !dbg !712
  %113 = icmp ne ptr %112, null, !dbg !712
  br i1 %113, label %Then20, label %Else20, !dbg !712

Then19:                                           ; preds = %Block18
  br label %Block19, !dbg !714

Else19:                                           ; preds = %Block18
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !716
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !717
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !718
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !718
  ret void, !dbg !718

Block20:                                          ; preds = %Then20
  %114 = phi ptr [ %112, %Then20 ], !dbg !719
  store ptr %114, ptr %27, align 8, !dbg !719
    #dbg_declare(ptr %27, !720, !DIExpression(), !719)
  %115 = call ptr @value_create_type_expr(ptr align 1 %2, ptr align 1 %62), !dbg !721
  %116 = icmp ne ptr %115, null, !dbg !721
  br i1 %116, label %Then21, label %Else21, !dbg !721

Then20:                                           ; preds = %Block19
  br label %Block20, !dbg !723

Else20:                                           ; preds = %Block19
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !725
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !726
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !727
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !727
  ret void, !dbg !727

Block21:                                          ; preds = %Then21
  %117 = phi ptr [ %115, %Then21 ], !dbg !728
  store ptr %117, ptr %26, align 8, !dbg !728
    #dbg_declare(ptr %26, !729, !DIExpression(), !728)
  %118 = call ptr @vector_value_create(ptr align 1 %2), !dbg !730
  %119 = icmp ne ptr %118, null, !dbg !730
  br i1 %119, label %Then22, label %Else22, !dbg !730

Then21:                                           ; preds = %Block20
  br label %Block21, !dbg !732

Else21:                                           ; preds = %Block20
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !734
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !735
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !736
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !736
  ret void, !dbg !736

Block22:                                          ; preds = %Then22
  %120 = phi ptr [ %118, %Then22 ], !dbg !737
  store ptr %120, ptr %25, align 8, !dbg !737
    #dbg_declare(ptr %25, !738, !DIExpression(), !737)
  %121 = call ptr @vector_value_push(ptr align 1 %2, ptr align 1 %120, ptr align 1 %114), !dbg !739
  %122 = icmp ne ptr %121, null, !dbg !739
  br i1 %122, label %Then23, label %Else23, !dbg !739

Then22:                                           ; preds = %Block21
  br label %Block22, !dbg !741

Else22:                                           ; preds = %Block21
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !743
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !744
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !745
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !745
  ret void, !dbg !745

Block23:                                          ; preds = %Then23
  %123 = phi ptr [ %121, %Then23 ], !dbg !746
  store ptr %123, ptr %24, align 8, !dbg !746
    #dbg_declare(ptr %24, !747, !DIExpression(), !746)
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %120), !dbg !748
  %124 = call ptr @vector_value_push(ptr align 1 %2, ptr align 1 %123, ptr align 1 %117), !dbg !749
  %125 = icmp ne ptr %124, null, !dbg !749
  br i1 %125, label %Then24, label %Else24, !dbg !749

Then23:                                           ; preds = %Block22
  br label %Block23, !dbg !751

Else23:                                           ; preds = %Block22
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !753
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !754
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !755
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !755
  ret void, !dbg !755

Block24:                                          ; preds = %Then24
  %126 = phi ptr [ %124, %Then24 ], !dbg !756
  store ptr %126, ptr %23, align 8, !dbg !756
    #dbg_declare(ptr %23, !757, !DIExpression(), !756)
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %123), !dbg !758
  %127 = call ptr @value_create_list(ptr align 1 %2, ptr align 1 %126), !dbg !759
  %128 = icmp ne ptr %127, null, !dbg !759
  br i1 %128, label %Then25, label %Else25, !dbg !759

Then24:                                           ; preds = %Block23
  br label %Block24, !dbg !761

Else24:                                           ; preds = %Block23
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !763
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !764
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !765
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !765
  ret void, !dbg !765

Block25:                                          ; preds = %Then25
  %129 = phi ptr [ %127, %Then25 ], !dbg !766
  store ptr %129, ptr %22, align 8, !dbg !766
    #dbg_declare(ptr %22, !767, !DIExpression(), !766)
  %130 = call ptr @value_create_identifier(ptr align 1 %2, ptr nonnull readonly align 1 @__anon_1852), !dbg !768
  %131 = icmp ne ptr %130, null, !dbg !768
  br i1 %131, label %Then26, label %Else26, !dbg !768

Then25:                                           ; preds = %Block24
  br label %Block25, !dbg !770

Else25:                                           ; preds = %Block24
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !772
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !773
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !774
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !774
  ret void, !dbg !774

Block26:                                          ; preds = %Then26
  %132 = phi ptr [ %130, %Then26 ], !dbg !775
  store ptr %132, ptr %21, align 8, !dbg !775
    #dbg_declare(ptr %21, !776, !DIExpression(), !775)
  %133 = call ptr @value_create_keyword(ptr align 1 %2, ptr nonnull readonly align 1 @__anon_1859), !dbg !777
  %134 = icmp ne ptr %133, null, !dbg !777
  br i1 %134, label %Then27, label %Else27, !dbg !777

Then26:                                           ; preds = %Block25
  br label %Block26, !dbg !779

Else26:                                           ; preds = %Block25
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !781
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !782
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !783
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !783
  ret void, !dbg !783

Block27:                                          ; preds = %Then27
  %135 = phi ptr [ %133, %Then27 ], !dbg !784
  store ptr %135, ptr %20, align 8, !dbg !784
    #dbg_declare(ptr %20, !785, !DIExpression(), !784)
  %136 = call ptr @vector_value_create(ptr align 1 %2), !dbg !786
  %137 = icmp ne ptr %136, null, !dbg !786
  br i1 %137, label %Then28, label %Else28, !dbg !786

Then27:                                           ; preds = %Block26
  br label %Block27, !dbg !788

Else27:                                           ; preds = %Block26
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !790
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !791
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !792
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !792
  ret void, !dbg !792

Block28:                                          ; preds = %Then28
  %138 = phi ptr [ %136, %Then28 ], !dbg !793
  store ptr %138, ptr %19, align 8, !dbg !793
    #dbg_declare(ptr %19, !794, !DIExpression(), !793)
  %139 = call ptr @vector_value_push(ptr align 1 %2, ptr align 1 %138, ptr align 1 %135), !dbg !795
  %140 = icmp ne ptr %139, null, !dbg !795
  br i1 %140, label %Then29, label %Else29, !dbg !795

Then28:                                           ; preds = %Block27
  br label %Block28, !dbg !797

Else28:                                           ; preds = %Block27
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !799
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !800
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !801
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !801
  ret void, !dbg !801

Block29:                                          ; preds = %Then29
  %141 = phi ptr [ %139, %Then29 ], !dbg !802
  store ptr %141, ptr %18, align 8, !dbg !802
    #dbg_declare(ptr %18, !803, !DIExpression(), !802)
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %138), !dbg !804
  %142 = call ptr @vector_value_push(ptr align 1 %2, ptr align 1 %141, ptr align 1 %61), !dbg !805
  %143 = icmp ne ptr %142, null, !dbg !805
  br i1 %143, label %Then30, label %Else30, !dbg !805

Then29:                                           ; preds = %Block28
  br label %Block29, !dbg !807

Else29:                                           ; preds = %Block28
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !809
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !810
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !811
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !811
  ret void, !dbg !811

Block30:                                          ; preds = %Then30
  %144 = phi ptr [ %142, %Then30 ], !dbg !812
  store ptr %144, ptr %17, align 8, !dbg !812
    #dbg_declare(ptr %17, !813, !DIExpression(), !812)
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %141), !dbg !814
  %145 = call ptr @value_create_map(ptr align 1 %2, ptr align 1 %144), !dbg !815
  %146 = icmp ne ptr %145, null, !dbg !815
  br i1 %146, label %Then31, label %Else31, !dbg !815

Then30:                                           ; preds = %Block29
  br label %Block30, !dbg !817

Else30:                                           ; preds = %Block29
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !819
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !820
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !821
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !821
  ret void, !dbg !821

Block31:                                          ; preds = %Then31
  %147 = phi ptr [ %145, %Then31 ], !dbg !822
  store ptr %147, ptr %16, align 8, !dbg !822
    #dbg_declare(ptr %16, !823, !DIExpression(), !822)
  %148 = call ptr @vector_value_create(ptr align 1 %2), !dbg !824
  %149 = icmp ne ptr %148, null, !dbg !824
  br i1 %149, label %Then32, label %Else32, !dbg !824

Then31:                                           ; preds = %Block30
  br label %Block31, !dbg !826

Else31:                                           ; preds = %Block30
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !828
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !829
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !830
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !830
  ret void, !dbg !830

Block32:                                          ; preds = %Then32
  %150 = phi ptr [ %148, %Then32 ], !dbg !831
  store ptr %150, ptr %15, align 8, !dbg !831
    #dbg_declare(ptr %15, !832, !DIExpression(), !831)
  %151 = call ptr @vector_value_push(ptr align 1 %2, ptr align 1 %150, ptr align 1 %132), !dbg !833
  %152 = icmp ne ptr %151, null, !dbg !833
  br i1 %152, label %Then33, label %Else33, !dbg !833

Then32:                                           ; preds = %Block31
  br label %Block32, !dbg !835

Else32:                                           ; preds = %Block31
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !837
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !838
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !839
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !839
  ret void, !dbg !839

Block33:                                          ; preds = %Then33
  %153 = phi ptr [ %151, %Then33 ], !dbg !840
  store ptr %153, ptr %14, align 8, !dbg !840
    #dbg_declare(ptr %14, !841, !DIExpression(), !840)
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %150), !dbg !842
  %154 = call ptr @vector_value_push(ptr align 1 %2, ptr align 1 %153, ptr align 1 %147), !dbg !843
  %155 = icmp ne ptr %154, null, !dbg !843
  br i1 %155, label %Then34, label %Else34, !dbg !843

Then33:                                           ; preds = %Block32
  br label %Block33, !dbg !845

Else33:                                           ; preds = %Block32
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !847
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !848
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !849
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !849
  ret void, !dbg !849

Block34:                                          ; preds = %Then34
  %156 = phi ptr [ %154, %Then34 ], !dbg !850
  store ptr %156, ptr %13, align 8, !dbg !850
    #dbg_declare(ptr %13, !851, !DIExpression(), !850)
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %153), !dbg !852
  %157 = call ptr @value_create_list(ptr align 1 %2, ptr align 1 %156), !dbg !853
  %158 = icmp ne ptr %157, null, !dbg !853
  br i1 %158, label %Then35, label %Else35, !dbg !853

Then34:                                           ; preds = %Block33
  br label %Block34, !dbg !855

Else34:                                           ; preds = %Block33
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !857
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !858
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !859
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !859
  ret void, !dbg !859

Block35:                                          ; preds = %Then35
  %159 = phi ptr [ %157, %Then35 ], !dbg !860
  store ptr %159, ptr %12, align 8, !dbg !860
    #dbg_declare(ptr %12, !861, !DIExpression(), !860)
  %160 = call ptr @vector_value_create(ptr align 1 %2), !dbg !862
  %161 = icmp ne ptr %160, null, !dbg !862
  br i1 %161, label %Then36, label %Else36, !dbg !862

Then35:                                           ; preds = %Block34
  br label %Block35, !dbg !864

Else35:                                           ; preds = %Block34
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !866
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !867
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !868
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !868
  ret void, !dbg !868

Block36:                                          ; preds = %Then36
  %162 = phi ptr [ %160, %Then36 ], !dbg !869
  store ptr %162, ptr %11, align 8, !dbg !869
    #dbg_declare(ptr %11, !870, !DIExpression(), !869)
  %163 = call ptr @vector_value_push(ptr align 1 %2, ptr align 1 %162, ptr align 1 %66), !dbg !871
  %164 = icmp ne ptr %163, null, !dbg !871
  br i1 %164, label %Then37, label %Else37, !dbg !871

Then36:                                           ; preds = %Block35
  br label %Block36, !dbg !873

Else36:                                           ; preds = %Block35
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !875
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !876
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !877
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !877
  ret void, !dbg !877

Block37:                                          ; preds = %Then37
  %165 = phi ptr [ %163, %Then37 ], !dbg !878
  store ptr %165, ptr %10, align 8, !dbg !878
    #dbg_declare(ptr %10, !879, !DIExpression(), !878)
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %162), !dbg !880
  %166 = call ptr @vector_value_push(ptr align 1 %2, ptr align 1 %165, ptr align 1 %84), !dbg !881
  %167 = icmp ne ptr %166, null, !dbg !881
  br i1 %167, label %Then38, label %Else38, !dbg !881

Then37:                                           ; preds = %Block36
  br label %Block37, !dbg !883

Else37:                                           ; preds = %Block36
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !885
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !886
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !887
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !887
  ret void, !dbg !887

Block38:                                          ; preds = %Then38
  %168 = phi ptr [ %166, %Then38 ], !dbg !888
  store ptr %168, ptr %9, align 8, !dbg !888
    #dbg_declare(ptr %9, !889, !DIExpression(), !888)
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %165), !dbg !890
  %169 = call ptr @vector_value_push(ptr align 1 %2, ptr align 1 %168, ptr align 1 %111), !dbg !891
  %170 = icmp ne ptr %169, null, !dbg !891
  br i1 %170, label %Then39, label %Else39, !dbg !891

Then38:                                           ; preds = %Block37
  br label %Block38, !dbg !893

Else38:                                           ; preds = %Block37
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !895
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !896
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !897
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !897
  ret void, !dbg !897

Block39:                                          ; preds = %Then39
  %171 = phi ptr [ %169, %Then39 ], !dbg !898
  store ptr %171, ptr %8, align 8, !dbg !898
    #dbg_declare(ptr %8, !899, !DIExpression(), !898)
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %168), !dbg !900
  %172 = call ptr @vector_value_push(ptr align 1 %2, ptr align 1 %171, ptr align 1 %129), !dbg !901
  %173 = icmp ne ptr %172, null, !dbg !901
  br i1 %173, label %Then40, label %Else40, !dbg !901

Then39:                                           ; preds = %Block38
  br label %Block39, !dbg !903

Else39:                                           ; preds = %Block38
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !905
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !906
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !907
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !907
  ret void, !dbg !907

Block40:                                          ; preds = %Then40
  %174 = phi ptr [ %172, %Then40 ], !dbg !908
  store ptr %174, ptr %7, align 8, !dbg !908
    #dbg_declare(ptr %7, !909, !DIExpression(), !908)
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %171), !dbg !910
  %175 = call ptr @vector_value_push(ptr align 1 %2, ptr align 1 %174, ptr align 1 %159), !dbg !911
  %176 = icmp ne ptr %175, null, !dbg !911
  br i1 %176, label %Then41, label %Else41, !dbg !911

Then40:                                           ; preds = %Block39
  br label %Block40, !dbg !913

Else40:                                           ; preds = %Block39
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !915
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !916
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !917
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !917
  ret void, !dbg !917

Block41:                                          ; preds = %Then41
  %177 = phi ptr [ %175, %Then41 ], !dbg !918
  store ptr %177, ptr %6, align 8, !dbg !918
    #dbg_declare(ptr %6, !919, !DIExpression(), !918)
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %174), !dbg !920
  %178 = call ptr @value_create_list(ptr align 1 %2, ptr align 1 %177), !dbg !921
  %179 = icmp ne ptr %178, null, !dbg !921
  br i1 %179, label %Then42, label %Else42, !dbg !921

Then41:                                           ; preds = %Block40
  br label %Block41, !dbg !923

Else41:                                           ; preds = %Block40
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !925
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !926
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !927
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !927
  ret void, !dbg !927

Block42:                                          ; preds = %Then42
  %180 = phi ptr [ %178, %Then42 ], !dbg !928
  store ptr %180, ptr %5, align 8, !dbg !928
    #dbg_declare(ptr %5, !929, !DIExpression(), !928)
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !930
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !931
  %181 = getelementptr inbounds { ptr, i16, [6 x i8] }, ptr %4, i32 0, i32 1, !dbg !932
  store i16 0, ptr %181, align 2, !dbg !932
  %182 = getelementptr inbounds { ptr, i16, [6 x i8] }, ptr %4, i32 0, i32 0, !dbg !932
  store ptr %180, ptr %182, align 8, !dbg !932
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 %4, i64 16, i1 false), !dbg !932
  ret void, !dbg !932

Then42:                                           ; preds = %Block41
  br label %Block42, !dbg !933

Else42:                                           ; preds = %Block41
  call void @value_free_atom(ptr align 1 %2, ptr readonly align 1 %63), !dbg !935
  call void @vector_value_destroy(ptr align 1 %2, ptr align 1 %57), !dbg !936
  notail call fastcc void @builtin.returnError(ptr %1) #4, !dbg !937
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @3, i64 16, i1 false), !dbg !937
  ret void, !dbg !937
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #1

; Function Attrs: nounwind uwtable
declare ptr @value_get_list(ptr align 1, ptr readonly align 1) #0

; Function Attrs: noinline nounwind uwtable
define internal fastcc void @builtin.returnError(ptr nonnull %0) unnamed_addr #2 !dbg !938 {
Entry:
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8, !dbg !941
  store ptr %0, ptr %1, align 8, !dbg !941
    #dbg_declare(ptr %1, !942, !DIExpression(), !941)
  %3 = getelementptr inbounds %builtin.StackTrace, ptr %0, i32 0, i32 0, !dbg !952
  %4 = load i64, ptr %3, align 8, !dbg !952
  %5 = getelementptr inbounds %builtin.StackTrace, ptr %0, i32 0, i32 1, !dbg !954
  %6 = load { ptr, i64 }, ptr %5, align 8, !dbg !954
  %7 = extractvalue { ptr, i64 } %6, 1, !dbg !955
  %8 = icmp ult i64 %4, %7, !dbg !955
  br i1 %8, label %Then, label %Else, !dbg !955

Block:                                            ; preds = %Else, %Then
  %9 = load ptr, ptr %2, align 8, !dbg !956
  %10 = getelementptr inbounds %builtin.StackTrace, ptr %9, i32 0, i32 0, !dbg !956
  %11 = load i64, ptr %10, align 8, !dbg !956
  %12 = add nuw i64 %11, 1, !dbg !957
  store i64 %12, ptr %10, align 8, !dbg !957
  ret void, !dbg !957

Then:                                             ; preds = %Entry
  %13 = load ptr, ptr %2, align 8, !dbg !958
  %14 = getelementptr inbounds %builtin.StackTrace, ptr %13, i32 0, i32 1, !dbg !958
  %15 = getelementptr inbounds %builtin.StackTrace, ptr %0, i32 0, i32 0, !dbg !960
  %16 = load i64, ptr %15, align 8, !dbg !960
  %17 = load { ptr, i64 }, ptr %14, align 8, !dbg !961
  %18 = extractvalue { ptr, i64 } %17, 0, !dbg !961
  %19 = getelementptr inbounds i64, ptr %18, i64 %16, !dbg !961
  %20 = call ptr @llvm.returnaddress(i32 0), !dbg !961
  %21 = ptrtoint ptr %20 to i64, !dbg !961
  store i64 %21, ptr %19, align 8, !dbg !961
  br label %Block, !dbg !961

Else:                                             ; preds = %Entry
  br label %Block, !dbg !962
}

; Function Attrs: nounwind uwtable
declare i64 @vector_value_len(ptr readonly align 1) #0

; Function Attrs: nounwind uwtable
declare void @vector_value_destroy(ptr align 1, ptr align 1) #0

; Function Attrs: nounwind uwtable
declare ptr @vector_value_at(ptr readonly align 1, i64) #0

; Function Attrs: nounwind uwtable
declare ptr @value_get_atom(ptr align 1, ptr readonly align 1) #0

; Function Attrs: nounwind uwtable
declare ptr @value_create_identifier(ptr align 1, ptr nonnull readonly align 1) #0

; Function Attrs: nounwind uwtable
declare void @value_free_atom(ptr align 1, ptr readonly align 1) #0

; Function Attrs: nounwind uwtable
declare ptr @vector_value_create(ptr align 1) #0

; Function Attrs: nounwind uwtable
declare ptr @vector_value_push(ptr align 1, ptr align 1, ptr align 1) #0

; Function Attrs: nounwind uwtable
declare ptr @value_create_list(ptr align 1, ptr align 1) #0

; Function Attrs: nounwind uwtable
declare ptr @value_create_type_expr(ptr align 1, ptr align 1) #0

; Function Attrs: nounwind uwtable
declare ptr @value_create_keyword(ptr align 1, ptr nonnull readonly align 1) #0

; Function Attrs: nounwind uwtable
declare ptr @value_create_map(ptr align 1, ptr align 1) #0

; Function Attrs: nounwind uwtable
define dso_local ptr @exampleTransformCallToOperation() #0 !dbg !964 {
Entry:
  %0 = alloca ptr, align 8
  %1 = alloca { ptr, i16, [6 x i8] }, align 8
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca %builtin.StackTrace, align 8
  %12 = alloca [32 x i64], align 8
  %13 = getelementptr inbounds %builtin.StackTrace, ptr %11, i32 0, i32 1
  %14 = getelementptr inbounds [32 x i64], ptr %12, i64 0, i64 0
  %15 = insertvalue { ptr, i64 } poison, ptr %14, 0
  %16 = insertvalue { ptr, i64 } %15, i64 32, 1
  store { ptr, i64 } %16, ptr %13, align 8
  %17 = getelementptr inbounds %builtin.StackTrace, ptr %11, i32 0, i32 0
  store i64 0, ptr %17, align 8
  %18 = getelementptr inbounds %builtin.StackTrace, ptr %11, i32 0, i32 0
  %19 = load i64, ptr %18, align 8
  %20 = call ptr @allocator_create_c(), !dbg !967
  %21 = icmp ne ptr %20, null, !dbg !967
  br i1 %21, label %Then, label %Else, !dbg !967

Block:                                            ; preds = %Then
  %22 = phi ptr [ %20, %Then ], !dbg !969
  store ptr %22, ptr %10, align 8, !dbg !969
    #dbg_declare(ptr %10, !970, !DIExpression(), !969)
  %23 = call ptr @value_create_identifier(ptr align 1 %22, ptr nonnull readonly align 1 @__anon_1755), !dbg !971
  %24 = icmp ne ptr %23, null, !dbg !971
  br i1 %24, label %Then1, label %Else1, !dbg !971

Then:                                             ; preds = %Entry
  br label %Block, !dbg !973

Else:                                             ; preds = %Entry
  ret ptr null, !dbg !975

Block1:                                           ; preds = %Then1
  %25 = phi ptr [ %23, %Then1 ], !dbg !976
  store ptr %25, ptr %9, align 8, !dbg !976
    #dbg_declare(ptr %9, !977, !DIExpression(), !976)
  %26 = call ptr @value_create_symbol(ptr align 1 %22, ptr nonnull readonly align 1 @__anon_1761), !dbg !978
  %27 = icmp ne ptr %26, null, !dbg !978
  br i1 %27, label %Then2, label %Else2, !dbg !978

Then1:                                            ; preds = %Block
  br label %Block1, !dbg !980

Else1:                                            ; preds = %Block
  ret ptr null, !dbg !982

Block2:                                           ; preds = %Then2
  %28 = phi ptr [ %26, %Then2 ], !dbg !983
  store ptr %28, ptr %8, align 8, !dbg !983
    #dbg_declare(ptr %8, !984, !DIExpression(), !983)
  %29 = call ptr @value_create_identifier(ptr align 1 %22, ptr nonnull readonly align 1 @__anon_1766), !dbg !985
  %30 = icmp ne ptr %29, null, !dbg !985
  br i1 %30, label %Then3, label %Else3, !dbg !985

Then2:                                            ; preds = %Block1
  br label %Block2, !dbg !987

Else2:                                            ; preds = %Block1
  ret ptr null, !dbg !989

Block3:                                           ; preds = %Then3
  %31 = phi ptr [ %29, %Then3 ], !dbg !990
  store ptr %31, ptr %7, align 8, !dbg !990
    #dbg_declare(ptr %7, !991, !DIExpression(), !990)
  %32 = call ptr @vector_value_create(ptr align 1 %22), !dbg !992
  %33 = icmp ne ptr %32, null, !dbg !992
  br i1 %33, label %Then4, label %Else4, !dbg !992

Then3:                                            ; preds = %Block2
  br label %Block3, !dbg !994

Else3:                                            ; preds = %Block2
  ret ptr null, !dbg !996

Block4:                                           ; preds = %Then4
  %34 = phi ptr [ %32, %Then4 ], !dbg !997
  store ptr %34, ptr %6, align 8, !dbg !997
    #dbg_declare(ptr %6, !998, !DIExpression(), !997)
  %35 = call ptr @vector_value_push(ptr align 1 %22, ptr align 1 %34, ptr align 1 %25), !dbg !999
  %36 = icmp ne ptr %35, null, !dbg !999
  br i1 %36, label %Then5, label %Else5, !dbg !999

Then4:                                            ; preds = %Block3
  br label %Block4, !dbg !1001

Else4:                                            ; preds = %Block3
  ret ptr null, !dbg !1003

Block5:                                           ; preds = %Then5
  %37 = phi ptr [ %35, %Then5 ], !dbg !1004
  store ptr %37, ptr %5, align 8, !dbg !1004
    #dbg_declare(ptr %5, !1005, !DIExpression(), !1004)
  call void @vector_value_destroy(ptr align 1 %22, ptr align 1 %34), !dbg !1006
  %38 = call ptr @vector_value_push(ptr align 1 %22, ptr align 1 %37, ptr align 1 %28), !dbg !1007
  %39 = icmp ne ptr %38, null, !dbg !1007
  br i1 %39, label %Then6, label %Else6, !dbg !1007

Then5:                                            ; preds = %Block4
  br label %Block5, !dbg !1009

Else5:                                            ; preds = %Block4
  ret ptr null, !dbg !1011

Block6:                                           ; preds = %Then6
  %40 = phi ptr [ %38, %Then6 ], !dbg !1012
  store ptr %40, ptr %4, align 8, !dbg !1012
    #dbg_declare(ptr %4, !1013, !DIExpression(), !1012)
  call void @vector_value_destroy(ptr align 1 %22, ptr align 1 %37), !dbg !1014
  %41 = call ptr @vector_value_push(ptr align 1 %22, ptr align 1 %40, ptr align 1 %31), !dbg !1015
  %42 = icmp ne ptr %41, null, !dbg !1015
  br i1 %42, label %Then7, label %Else7, !dbg !1015

Then6:                                            ; preds = %Block5
  br label %Block6, !dbg !1017

Else6:                                            ; preds = %Block5
  ret ptr null, !dbg !1019

Block7:                                           ; preds = %Then7
  %43 = phi ptr [ %41, %Then7 ], !dbg !1020
  store ptr %43, ptr %3, align 8, !dbg !1020
    #dbg_declare(ptr %3, !1021, !DIExpression(), !1020)
  call void @vector_value_destroy(ptr align 1 %22, ptr align 1 %40), !dbg !1022
  %44 = call ptr @value_create_list(ptr align 1 %22, ptr align 1 %43), !dbg !1023
  %45 = icmp ne ptr %44, null, !dbg !1023
  br i1 %45, label %Then8, label %Else8, !dbg !1023

Then7:                                            ; preds = %Block6
  br label %Block7, !dbg !1025

Else7:                                            ; preds = %Block6
  ret ptr null, !dbg !1027

Block8:                                           ; preds = %Then8
  %46 = phi ptr [ %44, %Then8 ], !dbg !1028
  store ptr %46, ptr %2, align 8, !dbg !1028
    #dbg_declare(ptr %2, !1029, !DIExpression(), !1028)
  call fastcc void @c_api_transform.transformCallToOperation(ptr sret({ ptr, i16, [6 x i8] }) %1, ptr %11, ptr align 1 %22, ptr align 1 %46), !dbg !1030
  %47 = getelementptr inbounds { ptr, i16, [6 x i8] }, ptr %1, i32 0, i32 1, !dbg !1030
  %48 = load i16, ptr %47, align 2, !dbg !1030
  %49 = icmp eq i16 %48, 0, !dbg !1030
  br i1 %49, label %Then9, label %Else9, !dbg !1030

Then8:                                            ; preds = %Block7
  br label %Block8, !dbg !1032

Else8:                                            ; preds = %Block7
  ret ptr null, !dbg !1034

Block9:                                           ; preds = %Then9
  %50 = phi ptr [ %52, %Then9 ], !dbg !1035
  store ptr %50, ptr %0, align 8, !dbg !1035
    #dbg_declare(ptr %0, !1036, !DIExpression(), !1035)
  call void @value_destroy(ptr align 1 %22, ptr align 1 %46), !dbg !1037
  ret ptr %50, !dbg !1038

Then9:                                            ; preds = %Block8
  %51 = getelementptr inbounds { ptr, i16, [6 x i8] }, ptr %1, i32 0, i32 0, !dbg !1039
  %52 = load ptr, ptr %51, align 8, !dbg !1039
  br label %Block9, !dbg !1039

Else9:                                            ; preds = %Block8
  %53 = getelementptr inbounds %builtin.StackTrace, ptr %11, i32 0, i32 0, !dbg !1041
  store i64 %19, ptr %53, align 8, !dbg !1041
  ret ptr null, !dbg !1041
}

; Function Attrs: nounwind uwtable
declare ptr @allocator_create_c() #0

; Function Attrs: nounwind uwtable
declare ptr @value_create_symbol(ptr align 1, ptr nonnull readonly align 1) #0

; Function Attrs: nounwind uwtable
declare void @value_destroy(ptr align 1, ptr align 1) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.returnaddress(i32 immarg) #3

attributes #0 = { nounwind uwtable "frame-pointer"="all" "target-cpu"="apple-m2" "target-features"="+aes,+alternate-sextload-cvt-f32-pattern,+altnzcv,+am,+amvs,+arith-bcc-fusion,+arith-cbz-fusion,+bf16,+bti,+ccdp,+ccidx,+ccpp,+complxnum,+CONTEXTIDREL2,+crc,+disable-latency-sched-heuristic,+dit,+dotprod,+ecv,+el2vmsa,+el3,+fgt,+flagm,+fp16fml,+fp-armv8,+fpac,+fptoint,+fullfp16,+fuse-address,+fuse-adrp-add,+fuse-aes,+fuse-arith-logic,+fuse-crypto-eor,+fuse-csel,+fuse-literals,+i8mm,+jsconv,+lor,+lse,+lse2,+mpam,+neon,+nv,+pan,+pan-rwv,+pauth,+perfmon,+predres,+ras,+rcpc,+rcpc-immo,+rdm,+sb,+sel2,+sha2,+sha3,+specrestrict,+ssbs,+store-pair-suppress,+tlb-rmi,+tracev8.4,+uaops,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8.6a,+v8a,+vh,+zcm,+zcz,+zcz-gp,-addr-lsl-slow-14,-aggressive-fma,-alu-lsl-fast,-ascend-store-address,-avoid-ldapur,-balance-fp-ops,-brbe,-call-saved-x10,-call-saved-x11,-call-saved-x12,-call-saved-x13,-call-saved-x14,-call-saved-x15,-call-saved-x18,-call-saved-x8,-call-saved-x9,-chk,-clrbhb,-cmp-bcc-fusion,-cmpbr,-cpa,-crypto,-cssc,-d128,-disable-ldp,-disable-stp,-enable-select-opt,-ete,-exynos-cheap-as-move,-f32mm,-f64mm,-f8f16mm,-f8f32mm,-faminmax,-fix-cortex-a53-835769,-fmv,-force-32bit-jump-tables,-fp8,-fp8dot2,-fp8dot4,-fp8fma,-fprcvt,-fujitsu-monaka,-fuse-addsub-2reg-const1,-gcs,-harden-sls-blr,-harden-sls-nocomdat,-harden-sls-retbr,-hbc,-hcx,-ite,-ldp-aligned-only,-ls64,-lse128,-lsfe,-lsui,-lut,-mec,-mops,-mte,-nmi,-no-bti-at-return-twice,-no-neg-immediates,-no-sve-fp-ld1r,-no-zcz-fp,-occmo,-outline-atomics,-pauth-lr,-pcdphint,-pops,-predictable-select-expensive,-prfm-slc-target,-rand,-rasv2,-rcpc3,-reserve-lr-for-ra,-reserve-x1,-reserve-x10,-reserve-x11,-reserve-x12,-reserve-x13,-reserve-x14,-reserve-x15,-reserve-x18,-reserve-x2,-reserve-x20,-reserve-x21,-reserve-x22,-reserve-x23,-reserve-x24,-reserve-x25,-reserve-x26,-reserve-x27,-reserve-x28,-reserve-x3,-reserve-x4,-reserve-x5,-reserve-x6,-reserve-x7,-reserve-x9,-rme,-slow-misaligned-128store,-slow-paired-128,-slow-strqro-store,-sm4,-sme,-sme2,-sme2p1,-sme2p2,-sme-b16b16,-sme-f16f16,-sme-f64f64,-sme-f8f16,-sme-f8f32,-sme-fa64,-sme-i16i64,-sme-lutv2,-sme-mop4,-sme-tmop,-spe,-spe-eef,-specres2,-ssve-aes,-ssve-bitperm,-ssve-fp8dot2,-ssve-fp8dot4,-ssve-fp8fma,-stp-aligned-only,-strict-align,-sve,-sve2,-sve2-aes,-sve2-bitperm,-sve2-sha3,-sve2-sm4,-sve2p1,-sve2p2,-sve-aes,-sve-aes2,-sve-b16b16,-sve-bfscale,-sve-bitperm,-sve-f16f32mm,-tagged-globals,-the,-tlbiw,-tme,-tpidr-el1,-tpidr-el2,-tpidr-el3,-tpidrro-el0,-trbe,-use-experimental-zeroing-pseudos,-use-fixed-over-scalable-if-equal-cost,-use-postra-scheduler,-use-reciprocal-square-root,-v8.7a,-v8.8a,-v8.9a,-v8r,-v9.1a,-v9.2a,-v9.3a,-v9.4a,-v9.5a,-v9.6a,-v9a,-wfxt,-xs,-zcz-fp-workaround" }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { noinline nounwind uwtable "frame-pointer"="all" "target-cpu"="apple-m2" "target-features"="+aes,+alternate-sextload-cvt-f32-pattern,+altnzcv,+am,+amvs,+arith-bcc-fusion,+arith-cbz-fusion,+bf16,+bti,+ccdp,+ccidx,+ccpp,+complxnum,+CONTEXTIDREL2,+crc,+disable-latency-sched-heuristic,+dit,+dotprod,+ecv,+el2vmsa,+el3,+fgt,+flagm,+fp16fml,+fp-armv8,+fpac,+fptoint,+fullfp16,+fuse-address,+fuse-adrp-add,+fuse-aes,+fuse-arith-logic,+fuse-crypto-eor,+fuse-csel,+fuse-literals,+i8mm,+jsconv,+lor,+lse,+lse2,+mpam,+neon,+nv,+pan,+pan-rwv,+pauth,+perfmon,+predres,+ras,+rcpc,+rcpc-immo,+rdm,+sb,+sel2,+sha2,+sha3,+specrestrict,+ssbs,+store-pair-suppress,+tlb-rmi,+tracev8.4,+uaops,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8.6a,+v8a,+vh,+zcm,+zcz,+zcz-gp,-addr-lsl-slow-14,-aggressive-fma,-alu-lsl-fast,-ascend-store-address,-avoid-ldapur,-balance-fp-ops,-brbe,-call-saved-x10,-call-saved-x11,-call-saved-x12,-call-saved-x13,-call-saved-x14,-call-saved-x15,-call-saved-x18,-call-saved-x8,-call-saved-x9,-chk,-clrbhb,-cmp-bcc-fusion,-cmpbr,-cpa,-crypto,-cssc,-d128,-disable-ldp,-disable-stp,-enable-select-opt,-ete,-exynos-cheap-as-move,-f32mm,-f64mm,-f8f16mm,-f8f32mm,-faminmax,-fix-cortex-a53-835769,-fmv,-force-32bit-jump-tables,-fp8,-fp8dot2,-fp8dot4,-fp8fma,-fprcvt,-fujitsu-monaka,-fuse-addsub-2reg-const1,-gcs,-harden-sls-blr,-harden-sls-nocomdat,-harden-sls-retbr,-hbc,-hcx,-ite,-ldp-aligned-only,-ls64,-lse128,-lsfe,-lsui,-lut,-mec,-mops,-mte,-nmi,-no-bti-at-return-twice,-no-neg-immediates,-no-sve-fp-ld1r,-no-zcz-fp,-occmo,-outline-atomics,-pauth-lr,-pcdphint,-pops,-predictable-select-expensive,-prfm-slc-target,-rand,-rasv2,-rcpc3,-reserve-lr-for-ra,-reserve-x1,-reserve-x10,-reserve-x11,-reserve-x12,-reserve-x13,-reserve-x14,-reserve-x15,-reserve-x18,-reserve-x2,-reserve-x20,-reserve-x21,-reserve-x22,-reserve-x23,-reserve-x24,-reserve-x25,-reserve-x26,-reserve-x27,-reserve-x28,-reserve-x3,-reserve-x4,-reserve-x5,-reserve-x6,-reserve-x7,-reserve-x9,-rme,-slow-misaligned-128store,-slow-paired-128,-slow-strqro-store,-sm4,-sme,-sme2,-sme2p1,-sme2p2,-sme-b16b16,-sme-f16f16,-sme-f64f64,-sme-f8f16,-sme-f8f32,-sme-fa64,-sme-i16i64,-sme-lutv2,-sme-mop4,-sme-tmop,-spe,-spe-eef,-specres2,-ssve-aes,-ssve-bitperm,-ssve-fp8dot2,-ssve-fp8dot4,-ssve-fp8fma,-stp-aligned-only,-strict-align,-sve,-sve2,-sve2-aes,-sve2-bitperm,-sve2-sha3,-sve2-sm4,-sve2p1,-sve2p2,-sve-aes,-sve-aes2,-sve-b16b16,-sve-bfscale,-sve-bitperm,-sve-f16f32mm,-tagged-globals,-the,-tlbiw,-tme,-tpidr-el1,-tpidr-el2,-tpidr-el3,-tpidrro-el0,-trbe,-use-experimental-zeroing-pseudos,-use-fixed-over-scalable-if-equal-cost,-use-postra-scheduler,-use-reciprocal-square-root,-v8.7a,-v8.8a,-v8.9a,-v8r,-v9.1a,-v9.2a,-v9.3a,-v9.4a,-v9.5a,-v9.6a,-v9a,-wfxt,-xs,-zcz-fp-workaround" }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #4 = { noinline }

!llvm.dbg.cu = !{!36}
!llvm.module.flags = !{!515, !516, !517}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "zig_backend", linkageName: "builtin.zig_backend", scope: !2, file: !2, line: 6, type: !3, isLocal: true, isDefinition: true)
!2 = !DIFile(filename: "builtin.zig", directory: "/Users/jimmyhmiller/.cache/zig/b/3bc568b18a992326da880d092ec756b1")
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "builtin.CompilerBackend", scope: !4, file: !4, line: 1027, baseType: !5, size: 64, align: 64, elements: !6)
!4 = !DIFile(filename: "builtin.zig", directory: "/opt/homebrew/Cellar/zig/0.15.1/lib/zig/std")
!5 = !DIBasicType(name: "u64", size: 64, encoding: DW_ATE_unsigned)
!6 = !{!7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19}
!7 = !DIEnumerator(name: "other", value: 0, isUnsigned: true)
!8 = !DIEnumerator(name: "stage1", value: 1, isUnsigned: true)
!9 = !DIEnumerator(name: "stage2_llvm", value: 2, isUnsigned: true)
!10 = !DIEnumerator(name: "stage2_c", value: 3, isUnsigned: true)
!11 = !DIEnumerator(name: "stage2_wasm", value: 4, isUnsigned: true)
!12 = !DIEnumerator(name: "stage2_arm", value: 5, isUnsigned: true)
!13 = !DIEnumerator(name: "stage2_x86_64", value: 6, isUnsigned: true)
!14 = !DIEnumerator(name: "stage2_aarch64", value: 7, isUnsigned: true)
!15 = !DIEnumerator(name: "stage2_x86", value: 8, isUnsigned: true)
!16 = !DIEnumerator(name: "stage2_riscv64", value: 9, isUnsigned: true)
!17 = !DIEnumerator(name: "stage2_sparc64", value: 10, isUnsigned: true)
!18 = !DIEnumerator(name: "stage2_spirv", value: 11, isUnsigned: true)
!19 = !DIEnumerator(name: "stage2_powerpc", value: 12, isUnsigned: true)
!20 = !DIGlobalVariableExpression(var: !21, expr: !DIExpression())
!21 = distinct !DIGlobalVariable(name: "simplified_logic", linkageName: "start.simplified_logic", scope: !22, file: !22, line: 17, type: !23, isLocal: true, isDefinition: true)
!22 = !DIFile(filename: "start.zig", directory: "/opt/homebrew/Cellar/zig/0.15.1/lib/zig/std")
!23 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!24 = !DIGlobalVariableExpression(var: !25, expr: !DIExpression())
!25 = distinct !DIGlobalVariable(name: "output_mode", linkageName: "builtin.output_mode", scope: !2, file: !2, line: 8, type: !26, isLocal: true, isDefinition: true)
!26 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "builtin.OutputMode", scope: !4, file: !4, line: 782, baseType: !27, size: 8, align: 8, elements: !28)
!27 = !DIBasicType(name: "u2", size: 8, encoding: DW_ATE_unsigned)
!28 = !{!29, !30, !31}
!29 = !DIEnumerator(name: "Exe", value: 0, isUnsigned: true)
!30 = !DIEnumerator(name: "Lib", value: 1, isUnsigned: true)
!31 = !DIEnumerator(name: "Obj", value: 2, isUnsigned: true)
!32 = !DIGlobalVariableExpression(var: !33, expr: !DIExpression())
!33 = distinct !DIGlobalVariable(name: "empty", linkageName: "Target.Cpu.Feature.Set.empty", scope: !34, file: !34, line: 1153, type: !35, isLocal: true, isDefinition: true)
!34 = !DIFile(filename: "Target.zig", directory: "/opt/homebrew/Cellar/zig/0.15.1/lib/zig/std")
!35 = !DICompositeType(tag: DW_TAG_structure_type, name: "Target.Cpu.Feature.Set", scope: !36, size: 320, align: 64, elements: !510)
!36 = distinct !DICompileUnit(language: DW_LANG_C99, file: !37, producer: "zig 0.15.1", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !38, globals: !478, splitDebugInlining: false)
!37 = !DIFile(filename: "c_api_transform", directory: "/Users/jimmyhmiller/Documents/Code/PlayGround/zig/mlir-lisp/src")
!38 = !{!3, !26, !39, !105, !144, !153, !222, !255, !269, !285, !300, !308}
!39 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Target.Cpu.Arch", scope: !40, file: !34, line: 1777, baseType: !58, size: 8, align: 8, elements: !59)
!40 = !DICompositeType(tag: DW_TAG_structure_type, name: "Target.Cpu", scope: !36, size: 448, align: 64, elements: !41)
!41 = !{!42, !56, !57}
!42 = !DIDerivedType(tag: DW_TAG_member, name: "model", scope: !40, baseType: !43, size: 64, align: 64)
!43 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*Target.Cpu.Model", baseType: !44, size: 64, align: 64)
!44 = !DICompositeType(tag: DW_TAG_structure_type, name: "Target.Cpu.Model", scope: !36, size: 576, align: 64, elements: !45)
!45 = !{!46, !54, !55}
!46 = !DIDerivedType(tag: DW_TAG_member, name: "name", scope: !44, baseType: !47, size: 128, align: 64)
!47 = !DICompositeType(tag: DW_TAG_structure_type, name: "[]u8", scope: !36, size: 128, align: 64, elements: !48)
!48 = !{!49, !52}
!49 = !DIDerivedType(tag: DW_TAG_member, name: "ptr", scope: !47, baseType: !50, size: 64, align: 64)
!50 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*u8", baseType: !51, size: 64, align: 8)
!51 = !DIBasicType(name: "u8", size: 8, encoding: DW_ATE_unsigned)
!52 = !DIDerivedType(tag: DW_TAG_member, name: "len", scope: !47, baseType: !53, size: 64, align: 64, offset: 64)
!53 = !DIBasicType(name: "usize", size: 64, encoding: DW_ATE_unsigned)
!54 = !DIDerivedType(tag: DW_TAG_member, name: "llvm_name", scope: !44, baseType: !47, size: 128, align: 64, offset: 128)
!55 = !DIDerivedType(tag: DW_TAG_member, name: "features", scope: !44, baseType: !35, size: 320, align: 64, offset: 256)
!56 = !DIDerivedType(tag: DW_TAG_member, name: "features", scope: !40, baseType: !35, size: 320, align: 64, offset: 64)
!57 = !DIDerivedType(tag: DW_TAG_member, name: "arch", scope: !40, baseType: !39, size: 8, align: 8, offset: 384)
!58 = !DIBasicType(name: "u6", size: 8, encoding: DW_ATE_unsigned)
!59 = !{!60, !61, !62, !63, !64, !65, !66, !67, !68, !69, !70, !71, !72, !73, !74, !75, !76, !77, !78, !79, !80, !81, !82, !83, !84, !85, !86, !87, !88, !89, !90, !91, !92, !93, !94, !95, !96, !97, !98, !99, !100, !101, !102, !103, !104}
!60 = !DIEnumerator(name: "amdgcn", value: 0, isUnsigned: true)
!61 = !DIEnumerator(name: "arc", value: 1, isUnsigned: true)
!62 = !DIEnumerator(name: "arm", value: 2, isUnsigned: true)
!63 = !DIEnumerator(name: "armeb", value: 3, isUnsigned: true)
!64 = !DIEnumerator(name: "thumb", value: 4, isUnsigned: true)
!65 = !DIEnumerator(name: "thumbeb", value: 5, isUnsigned: true)
!66 = !DIEnumerator(name: "aarch64", value: 6, isUnsigned: true)
!67 = !DIEnumerator(name: "aarch64_be", value: 7, isUnsigned: true)
!68 = !DIEnumerator(name: "avr", value: 8, isUnsigned: true)
!69 = !DIEnumerator(name: "bpfel", value: 9, isUnsigned: true)
!70 = !DIEnumerator(name: "bpfeb", value: 10, isUnsigned: true)
!71 = !DIEnumerator(name: "csky", value: 11, isUnsigned: true)
!72 = !DIEnumerator(name: "hexagon", value: 12, isUnsigned: true)
!73 = !DIEnumerator(name: "kalimba", value: 13, isUnsigned: true)
!74 = !DIEnumerator(name: "lanai", value: 14, isUnsigned: true)
!75 = !DIEnumerator(name: "loongarch32", value: 15, isUnsigned: true)
!76 = !DIEnumerator(name: "loongarch64", value: 16, isUnsigned: true)
!77 = !DIEnumerator(name: "m68k", value: 17, isUnsigned: true)
!78 = !DIEnumerator(name: "mips", value: 18, isUnsigned: true)
!79 = !DIEnumerator(name: "mipsel", value: 19, isUnsigned: true)
!80 = !DIEnumerator(name: "mips64", value: 20, isUnsigned: true)
!81 = !DIEnumerator(name: "mips64el", value: 21, isUnsigned: true)
!82 = !DIEnumerator(name: "msp430", value: 22, isUnsigned: true)
!83 = !DIEnumerator(name: "or1k", value: 23, isUnsigned: true)
!84 = !DIEnumerator(name: "nvptx", value: 24, isUnsigned: true)
!85 = !DIEnumerator(name: "nvptx64", value: 25, isUnsigned: true)
!86 = !DIEnumerator(name: "powerpc", value: 26, isUnsigned: true)
!87 = !DIEnumerator(name: "powerpcle", value: 27, isUnsigned: true)
!88 = !DIEnumerator(name: "powerpc64", value: 28, isUnsigned: true)
!89 = !DIEnumerator(name: "powerpc64le", value: 29, isUnsigned: true)
!90 = !DIEnumerator(name: "propeller", value: 30, isUnsigned: true)
!91 = !DIEnumerator(name: "riscv32", value: 31, isUnsigned: true)
!92 = !DIEnumerator(name: "riscv64", value: 32, isUnsigned: true)
!93 = !DIEnumerator(name: "s390x", value: 33, isUnsigned: true)
!94 = !DIEnumerator(name: "sparc", value: 34, isUnsigned: true)
!95 = !DIEnumerator(name: "sparc64", value: 35, isUnsigned: true)
!96 = !DIEnumerator(name: "spirv32", value: 36, isUnsigned: true)
!97 = !DIEnumerator(name: "spirv64", value: 37, isUnsigned: true)
!98 = !DIEnumerator(name: "ve", value: 38, isUnsigned: true)
!99 = !DIEnumerator(name: "wasm32", value: 39, isUnsigned: true)
!100 = !DIEnumerator(name: "wasm64", value: 40, isUnsigned: true)
!101 = !DIEnumerator(name: "x86", value: 41, isUnsigned: true)
!102 = !DIEnumerator(name: "x86_64", value: 42, isUnsigned: true)
!103 = !DIEnumerator(name: "xcore", value: 43, isUnsigned: true)
!104 = !DIEnumerator(name: "xtensa", value: 44, isUnsigned: true)
!105 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Target.Os.WindowsVersion", scope: !106, file: !34, line: 311, baseType: !137, size: 32, align: 32, elements: !195)
!106 = !DICompositeType(tag: DW_TAG_structure_type, name: "Target.Os", scope: !36, size: 1536, align: 64, elements: !107)
!107 = !{!108, !152}
!108 = !DIDerivedType(tag: DW_TAG_member, name: "version_range", scope: !106, baseType: !109, size: 1472, align: 64)
!109 = !DICompositeType(tag: DW_TAG_structure_type, name: "Target.Os.VersionRange", scope: !36, size: 1472, align: 64, elements: !110)
!110 = !{!111, !143}
!111 = !DIDerivedType(tag: DW_TAG_member, name: "payload", scope: !109, baseType: !112, size: 1408, align: 64)
!112 = !DICompositeType(tag: DW_TAG_union_type, name: "Target.Os.VersionRange:Payload", scope: !36, size: 1472, align: 64, elements: !113)
!113 = !{!114, !126, !131, !138}
!114 = !DIDerivedType(tag: DW_TAG_member, name: "semver", scope: !112, baseType: !115, size: 896, align: 64)
!115 = !DICompositeType(tag: DW_TAG_structure_type, name: "SemanticVersion.Range", scope: !36, size: 896, align: 64, elements: !116)
!116 = !{!117, !125}
!117 = !DIDerivedType(tag: DW_TAG_member, name: "min", scope: !115, baseType: !118, size: 448, align: 64)
!118 = !DICompositeType(tag: DW_TAG_structure_type, name: "SemanticVersion", scope: !36, size: 448, align: 64, elements: !119)
!119 = !{!120, !121, !122, !123, !124}
!120 = !DIDerivedType(tag: DW_TAG_member, name: "major", scope: !118, baseType: !53, size: 64, align: 64)
!121 = !DIDerivedType(tag: DW_TAG_member, name: "minor", scope: !118, baseType: !53, size: 64, align: 64, offset: 64)
!122 = !DIDerivedType(tag: DW_TAG_member, name: "patch", scope: !118, baseType: !53, size: 64, align: 64, offset: 128)
!123 = !DIDerivedType(tag: DW_TAG_member, name: "pre", scope: !118, baseType: !47, size: 128, align: 64, offset: 192)
!124 = !DIDerivedType(tag: DW_TAG_member, name: "build", scope: !118, baseType: !47, size: 128, align: 64, offset: 320)
!125 = !DIDerivedType(tag: DW_TAG_member, name: "max", scope: !115, baseType: !118, size: 448, align: 64, offset: 448)
!126 = !DIDerivedType(tag: DW_TAG_member, name: "hurd", scope: !112, baseType: !127, size: 1344, align: 64)
!127 = !DICompositeType(tag: DW_TAG_structure_type, name: "Target.Os.HurdVersionRange", scope: !36, size: 1344, align: 64, elements: !128)
!128 = !{!129, !130}
!129 = !DIDerivedType(tag: DW_TAG_member, name: "range", scope: !127, baseType: !115, size: 896, align: 64)
!130 = !DIDerivedType(tag: DW_TAG_member, name: "glibc", scope: !127, baseType: !118, size: 448, align: 64, offset: 896)
!131 = !DIDerivedType(tag: DW_TAG_member, name: "linux", scope: !112, baseType: !132, size: 1408, align: 64)
!132 = !DICompositeType(tag: DW_TAG_structure_type, name: "Target.Os.LinuxVersionRange", scope: !36, size: 1408, align: 64, elements: !133)
!133 = !{!134, !135, !136}
!134 = !DIDerivedType(tag: DW_TAG_member, name: "range", scope: !132, baseType: !115, size: 896, align: 64)
!135 = !DIDerivedType(tag: DW_TAG_member, name: "glibc", scope: !132, baseType: !118, size: 448, align: 64, offset: 896)
!136 = !DIDerivedType(tag: DW_TAG_member, name: "android", scope: !132, baseType: !137, size: 32, align: 32, offset: 1344)
!137 = !DIBasicType(name: "u32", size: 32, encoding: DW_ATE_unsigned)
!138 = !DIDerivedType(tag: DW_TAG_member, name: "windows", scope: !112, baseType: !139, size: 64, align: 32)
!139 = !DICompositeType(tag: DW_TAG_structure_type, name: "Target.Os.WindowsVersion.Range", scope: !36, size: 64, align: 32, elements: !140)
!140 = !{!141, !142}
!141 = !DIDerivedType(tag: DW_TAG_member, name: "min", scope: !139, baseType: !105, size: 32, align: 32)
!142 = !DIDerivedType(tag: DW_TAG_member, name: "max", scope: !139, baseType: !105, size: 32, align: 32, offset: 32)
!143 = !DIDerivedType(tag: DW_TAG_member, name: "tag", scope: !109, baseType: !144, size: 8, align: 8, offset: 1408)
!144 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "@typeInfo(Target.Os.VersionRange).@\22union\22.tag_type.?", scope: !109, file: !34, line: 653, baseType: !145, size: 8, align: 8, elements: !146)
!145 = !DIBasicType(name: "u3", size: 8, encoding: DW_ATE_unsigned)
!146 = !{!147, !148, !149, !150, !151}
!147 = !DIEnumerator(name: "none", value: 0, isUnsigned: true)
!148 = !DIEnumerator(name: "semver", value: 1, isUnsigned: true)
!149 = !DIEnumerator(name: "hurd", value: 2, isUnsigned: true)
!150 = !DIEnumerator(name: "linux", value: 3, isUnsigned: true)
!151 = !DIEnumerator(name: "windows", value: 4, isUnsigned: true)
!152 = !DIDerivedType(tag: DW_TAG_member, name: "tag", scope: !106, baseType: !153, size: 8, align: 8, offset: 1472)
!153 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Target.Os.Tag", scope: !106, file: !34, line: 213, baseType: !58, size: 8, align: 8, elements: !154)
!154 = !{!155, !156, !157, !158, !159, !160, !161, !162, !163, !164, !165, !166, !167, !168, !169, !170, !171, !172, !173, !174, !175, !176, !177, !178, !179, !180, !181, !182, !183, !184, !185, !186, !187, !188, !189, !190, !191, !192, !193, !194}
!155 = !DIEnumerator(name: "freestanding", value: 0, isUnsigned: true)
!156 = !DIEnumerator(name: "other", value: 1, isUnsigned: true)
!157 = !DIEnumerator(name: "contiki", value: 2, isUnsigned: true)
!158 = !DIEnumerator(name: "fuchsia", value: 3, isUnsigned: true)
!159 = !DIEnumerator(name: "hermit", value: 4, isUnsigned: true)
!160 = !DIEnumerator(name: "aix", value: 5, isUnsigned: true)
!161 = !DIEnumerator(name: "haiku", value: 6, isUnsigned: true)
!162 = !DIEnumerator(name: "hurd", value: 7, isUnsigned: true)
!163 = !DIEnumerator(name: "linux", value: 8, isUnsigned: true)
!164 = !DIEnumerator(name: "plan9", value: 9, isUnsigned: true)
!165 = !DIEnumerator(name: "rtems", value: 10, isUnsigned: true)
!166 = !DIEnumerator(name: "serenity", value: 11, isUnsigned: true)
!167 = !DIEnumerator(name: "zos", value: 12, isUnsigned: true)
!168 = !DIEnumerator(name: "dragonfly", value: 13, isUnsigned: true)
!169 = !DIEnumerator(name: "freebsd", value: 14, isUnsigned: true)
!170 = !DIEnumerator(name: "netbsd", value: 15, isUnsigned: true)
!171 = !DIEnumerator(name: "openbsd", value: 16, isUnsigned: true)
!172 = !DIEnumerator(name: "driverkit", value: 17, isUnsigned: true)
!173 = !DIEnumerator(name: "ios", value: 18, isUnsigned: true)
!174 = !DIEnumerator(name: "macos", value: 19, isUnsigned: true)
!175 = !DIEnumerator(name: "tvos", value: 20, isUnsigned: true)
!176 = !DIEnumerator(name: "visionos", value: 21, isUnsigned: true)
!177 = !DIEnumerator(name: "watchos", value: 22, isUnsigned: true)
!178 = !DIEnumerator(name: "illumos", value: 23, isUnsigned: true)
!179 = !DIEnumerator(name: "solaris", value: 24, isUnsigned: true)
!180 = !DIEnumerator(name: "windows", value: 25, isUnsigned: true)
!181 = !DIEnumerator(name: "uefi", value: 26, isUnsigned: true)
!182 = !DIEnumerator(name: "ps3", value: 27, isUnsigned: true)
!183 = !DIEnumerator(name: "ps4", value: 28, isUnsigned: true)
!184 = !DIEnumerator(name: "ps5", value: 29, isUnsigned: true)
!185 = !DIEnumerator(name: "emscripten", value: 30, isUnsigned: true)
!186 = !DIEnumerator(name: "wasi", value: 31, isUnsigned: true)
!187 = !DIEnumerator(name: "amdhsa", value: 32, isUnsigned: true)
!188 = !DIEnumerator(name: "amdpal", value: 33, isUnsigned: true)
!189 = !DIEnumerator(name: "cuda", value: 34, isUnsigned: true)
!190 = !DIEnumerator(name: "mesa3d", value: 35, isUnsigned: true)
!191 = !DIEnumerator(name: "nvcl", value: 36, isUnsigned: true)
!192 = !DIEnumerator(name: "opencl", value: 37, isUnsigned: true)
!193 = !DIEnumerator(name: "opengl", value: 38, isUnsigned: true)
!194 = !DIEnumerator(name: "vulkan", value: 39, isUnsigned: true)
!195 = !{!196, !197, !198, !199, !200, !201, !202, !203, !204, !205, !206, !207, !208, !209, !210, !211, !212, !213, !214, !215, !216, !217, !218, !219, !220, !221}
!196 = !DIEnumerator(name: "nt4", value: 67108864, isUnsigned: true)
!197 = !DIEnumerator(name: "win2k", value: 83886080, isUnsigned: true)
!198 = !DIEnumerator(name: "xp", value: 83951616, isUnsigned: true)
!199 = !DIEnumerator(name: "ws2003", value: 84017152, isUnsigned: true)
!200 = !DIEnumerator(name: "vista", value: 100663296, isUnsigned: true)
!201 = !DIEnumerator(name: "win7", value: 100728832, isUnsigned: true)
!202 = !DIEnumerator(name: "win8", value: 100794368, isUnsigned: true)
!203 = !DIEnumerator(name: "win8_1", value: 100859904, isUnsigned: true)
!204 = !DIEnumerator(name: "win10", value: 167772160, isUnsigned: true)
!205 = !DIEnumerator(name: "win10_th2", value: 167772161, isUnsigned: true)
!206 = !DIEnumerator(name: "win10_rs1", value: 167772162, isUnsigned: true)
!207 = !DIEnumerator(name: "win10_rs2", value: 167772163, isUnsigned: true)
!208 = !DIEnumerator(name: "win10_rs3", value: 167772164, isUnsigned: true)
!209 = !DIEnumerator(name: "win10_rs4", value: 167772165, isUnsigned: true)
!210 = !DIEnumerator(name: "win10_rs5", value: 167772166, isUnsigned: true)
!211 = !DIEnumerator(name: "win10_19h1", value: 167772167, isUnsigned: true)
!212 = !DIEnumerator(name: "win10_vb", value: 167772168, isUnsigned: true)
!213 = !DIEnumerator(name: "win10_mn", value: 167772169, isUnsigned: true)
!214 = !DIEnumerator(name: "win10_fe", value: 167772170, isUnsigned: true)
!215 = !DIEnumerator(name: "win10_co", value: 167772171, isUnsigned: true)
!216 = !DIEnumerator(name: "win10_ni", value: 167772172, isUnsigned: true)
!217 = !DIEnumerator(name: "win10_cu", value: 167772173, isUnsigned: true)
!218 = !DIEnumerator(name: "win11_zn", value: 167772174, isUnsigned: true)
!219 = !DIEnumerator(name: "win11_ga", value: 167772175, isUnsigned: true)
!220 = !DIEnumerator(name: "win11_ge", value: 167772176, isUnsigned: true)
!221 = !DIEnumerator(name: "win11_dt", value: 167772177, isUnsigned: true)
!222 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Target.Abi", scope: !34, file: !34, line: 952, baseType: !223, size: 8, align: 8, elements: !224)
!223 = !DIBasicType(name: "u5", size: 8, encoding: DW_ATE_unsigned)
!224 = !{!225, !226, !227, !228, !229, !230, !231, !232, !233, !234, !235, !236, !237, !238, !239, !240, !241, !242, !243, !244, !245, !246, !247, !248, !249, !250, !251, !252, !253, !254}
!225 = !DIEnumerator(name: "none", value: 0, isUnsigned: true)
!226 = !DIEnumerator(name: "gnu", value: 1, isUnsigned: true)
!227 = !DIEnumerator(name: "gnuabin32", value: 2, isUnsigned: true)
!228 = !DIEnumerator(name: "gnuabi64", value: 3, isUnsigned: true)
!229 = !DIEnumerator(name: "gnueabi", value: 4, isUnsigned: true)
!230 = !DIEnumerator(name: "gnueabihf", value: 5, isUnsigned: true)
!231 = !DIEnumerator(name: "gnuf32", value: 6, isUnsigned: true)
!232 = !DIEnumerator(name: "gnusf", value: 7, isUnsigned: true)
!233 = !DIEnumerator(name: "gnux32", value: 8, isUnsigned: true)
!234 = !DIEnumerator(name: "code16", value: 9, isUnsigned: true)
!235 = !DIEnumerator(name: "eabi", value: 10, isUnsigned: true)
!236 = !DIEnumerator(name: "eabihf", value: 11, isUnsigned: true)
!237 = !DIEnumerator(name: "ilp32", value: 12, isUnsigned: true)
!238 = !DIEnumerator(name: "android", value: 13, isUnsigned: true)
!239 = !DIEnumerator(name: "androideabi", value: 14, isUnsigned: true)
!240 = !DIEnumerator(name: "musl", value: 15, isUnsigned: true)
!241 = !DIEnumerator(name: "muslabin32", value: 16, isUnsigned: true)
!242 = !DIEnumerator(name: "muslabi64", value: 17, isUnsigned: true)
!243 = !DIEnumerator(name: "musleabi", value: 18, isUnsigned: true)
!244 = !DIEnumerator(name: "musleabihf", value: 19, isUnsigned: true)
!245 = !DIEnumerator(name: "muslf32", value: 20, isUnsigned: true)
!246 = !DIEnumerator(name: "muslsf", value: 21, isUnsigned: true)
!247 = !DIEnumerator(name: "muslx32", value: 22, isUnsigned: true)
!248 = !DIEnumerator(name: "msvc", value: 23, isUnsigned: true)
!249 = !DIEnumerator(name: "itanium", value: 24, isUnsigned: true)
!250 = !DIEnumerator(name: "cygnus", value: 25, isUnsigned: true)
!251 = !DIEnumerator(name: "simulator", value: 26, isUnsigned: true)
!252 = !DIEnumerator(name: "macabi", value: 27, isUnsigned: true)
!253 = !DIEnumerator(name: "ohos", value: 28, isUnsigned: true)
!254 = !DIEnumerator(name: "ohoseabi", value: 29, isUnsigned: true)
!255 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Target.ObjectFormat", scope: !34, file: !34, line: 1007, baseType: !256, size: 8, align: 8, elements: !257)
!256 = !DIBasicType(name: "u4", size: 8, encoding: DW_ATE_unsigned)
!257 = !{!258, !259, !260, !261, !262, !263, !264, !265, !266, !267, !268}
!258 = !DIEnumerator(name: "c", value: 0, isUnsigned: true)
!259 = !DIEnumerator(name: "coff", value: 1, isUnsigned: true)
!260 = !DIEnumerator(name: "elf", value: 2, isUnsigned: true)
!261 = !DIEnumerator(name: "goff", value: 3, isUnsigned: true)
!262 = !DIEnumerator(name: "hex", value: 4, isUnsigned: true)
!263 = !DIEnumerator(name: "macho", value: 5, isUnsigned: true)
!264 = !DIEnumerator(name: "plan9", value: 6, isUnsigned: true)
!265 = !DIEnumerator(name: "raw", value: 7, isUnsigned: true)
!266 = !DIEnumerator(name: "spirv", value: 8, isUnsigned: true)
!267 = !DIEnumerator(name: "wasm", value: 9, isUnsigned: true)
!268 = !DIEnumerator(name: "xcoff", value: 10, isUnsigned: true)
!269 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "builtin.CallingConvention.ArmInterruptOptions.InterruptType", scope: !270, file: !4, line: 382, baseType: !145, size: 8, align: 8, elements: !278)
!270 = !DICompositeType(tag: DW_TAG_structure_type, name: "builtin.CallingConvention.ArmInterruptOptions", scope: !36, size: 192, align: 64, elements: !271)
!271 = !{!272, !277}
!272 = !DIDerivedType(tag: DW_TAG_member, name: "incoming_stack_alignment", scope: !270, baseType: !273, size: 128, align: 64)
!273 = !DICompositeType(tag: DW_TAG_structure_type, name: "?u64", scope: !36, size: 128, align: 64, elements: !274)
!274 = !{!275, !276}
!275 = !DIDerivedType(tag: DW_TAG_member, name: "data", scope: !273, baseType: !5, size: 64, align: 64)
!276 = !DIDerivedType(tag: DW_TAG_member, name: "some", scope: !273, baseType: !51, size: 8, align: 8, offset: 64)
!277 = !DIDerivedType(tag: DW_TAG_member, name: "type", scope: !270, baseType: !269, size: 8, align: 8, offset: 128)
!278 = !{!279, !280, !281, !282, !283, !284}
!279 = !DIEnumerator(name: "generic", value: 0, isUnsigned: true)
!280 = !DIEnumerator(name: "irq", value: 1, isUnsigned: true)
!281 = !DIEnumerator(name: "fiq", value: 2, isUnsigned: true)
!282 = !DIEnumerator(name: "swi", value: 3, isUnsigned: true)
!283 = !DIEnumerator(name: "abort", value: 4, isUnsigned: true)
!284 = !DIEnumerator(name: "undef", value: 5, isUnsigned: true)
!285 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "builtin.CallingConvention.MipsInterruptOptions.InterruptMode", scope: !286, file: !4, line: 400, baseType: !256, size: 8, align: 8, elements: !290)
!286 = !DICompositeType(tag: DW_TAG_structure_type, name: "builtin.CallingConvention.MipsInterruptOptions", scope: !36, size: 192, align: 64, elements: !287)
!287 = !{!288, !289}
!288 = !DIDerivedType(tag: DW_TAG_member, name: "incoming_stack_alignment", scope: !286, baseType: !273, size: 128, align: 64)
!289 = !DIDerivedType(tag: DW_TAG_member, name: "mode", scope: !286, baseType: !285, size: 8, align: 8, offset: 128)
!290 = !{!291, !292, !293, !294, !295, !296, !297, !298, !299}
!291 = !DIEnumerator(name: "eic", value: 0, isUnsigned: true)
!292 = !DIEnumerator(name: "sw0", value: 1, isUnsigned: true)
!293 = !DIEnumerator(name: "sw1", value: 2, isUnsigned: true)
!294 = !DIEnumerator(name: "hw0", value: 3, isUnsigned: true)
!295 = !DIEnumerator(name: "hw1", value: 4, isUnsigned: true)
!296 = !DIEnumerator(name: "hw2", value: 5, isUnsigned: true)
!297 = !DIEnumerator(name: "hw3", value: 6, isUnsigned: true)
!298 = !DIEnumerator(name: "hw4", value: 7, isUnsigned: true)
!299 = !DIEnumerator(name: "hw5", value: 8, isUnsigned: true)
!300 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "builtin.CallingConvention.RiscvInterruptOptions.PrivilegeMode", scope: !301, file: !4, line: 421, baseType: !27, size: 8, align: 8, elements: !305)
!301 = !DICompositeType(tag: DW_TAG_structure_type, name: "builtin.CallingConvention.RiscvInterruptOptions", scope: !36, size: 192, align: 64, elements: !302)
!302 = !{!303, !304}
!303 = !DIDerivedType(tag: DW_TAG_member, name: "incoming_stack_alignment", scope: !301, baseType: !273, size: 128, align: 64)
!304 = !DIDerivedType(tag: DW_TAG_member, name: "mode", scope: !301, baseType: !300, size: 8, align: 8, offset: 128)
!305 = !{!306, !307}
!306 = !DIEnumerator(name: "supervisor", value: 0, isUnsigned: true)
!307 = !DIEnumerator(name: "machine", value: 1, isUnsigned: true)
!308 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "@typeInfo(builtin.CallingConvention).@\22union\22.tag_type.?", scope: !309, file: !4, line: 442, baseType: !51, size: 8, align: 8, elements: !392)
!309 = !DICompositeType(tag: DW_TAG_structure_type, name: "builtin.CallingConvention", scope: !36, size: 256, align: 64, elements: !310)
!310 = !{!311, !391}
!311 = !DIDerivedType(tag: DW_TAG_member, name: "payload", scope: !309, baseType: !312, size: 192, align: 64)
!312 = !DICompositeType(tag: DW_TAG_union_type, name: "builtin.CallingConvention:Payload", scope: !36, size: 256, align: 64, elements: !313)
!313 = !{!314, !318, !319, !320, !321, !322, !323, !328, !329, !330, !331, !332, !333, !334, !335, !336, !337, !338, !339, !340, !341, !342, !343, !344, !345, !346, !347, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !368, !369, !370, !371, !372, !373, !374, !375, !376, !377, !378, !379, !380, !381, !382, !383, !384, !385, !386, !387, !388, !389, !390}
!314 = !DIDerivedType(tag: DW_TAG_member, name: "x86_64_sysv", scope: !312, baseType: !315, size: 128, align: 64)
!315 = !DICompositeType(tag: DW_TAG_structure_type, name: "builtin.CallingConvention.CommonOptions", scope: !36, size: 128, align: 64, elements: !316)
!316 = !{!317}
!317 = !DIDerivedType(tag: DW_TAG_member, name: "incoming_stack_alignment", scope: !315, baseType: !273, size: 128, align: 64)
!318 = !DIDerivedType(tag: DW_TAG_member, name: "x86_64_win", scope: !312, baseType: !315, size: 128, align: 64)
!319 = !DIDerivedType(tag: DW_TAG_member, name: "x86_64_regcall_v3_sysv", scope: !312, baseType: !315, size: 128, align: 64)
!320 = !DIDerivedType(tag: DW_TAG_member, name: "x86_64_regcall_v4_win", scope: !312, baseType: !315, size: 128, align: 64)
!321 = !DIDerivedType(tag: DW_TAG_member, name: "x86_64_vectorcall", scope: !312, baseType: !315, size: 128, align: 64)
!322 = !DIDerivedType(tag: DW_TAG_member, name: "x86_64_interrupt", scope: !312, baseType: !315, size: 128, align: 64)
!323 = !DIDerivedType(tag: DW_TAG_member, name: "x86_sysv", scope: !312, baseType: !324, size: 192, align: 64)
!324 = !DICompositeType(tag: DW_TAG_structure_type, name: "builtin.CallingConvention.X86RegparmOptions", scope: !36, size: 192, align: 64, elements: !325)
!325 = !{!326, !327}
!326 = !DIDerivedType(tag: DW_TAG_member, name: "incoming_stack_alignment", scope: !324, baseType: !273, size: 128, align: 64)
!327 = !DIDerivedType(tag: DW_TAG_member, name: "register_params", scope: !324, baseType: !27, size: 8, align: 8, offset: 128)
!328 = !DIDerivedType(tag: DW_TAG_member, name: "x86_win", scope: !312, baseType: !324, size: 192, align: 64)
!329 = !DIDerivedType(tag: DW_TAG_member, name: "x86_stdcall", scope: !312, baseType: !324, size: 192, align: 64)
!330 = !DIDerivedType(tag: DW_TAG_member, name: "x86_fastcall", scope: !312, baseType: !315, size: 128, align: 64)
!331 = !DIDerivedType(tag: DW_TAG_member, name: "x86_thiscall", scope: !312, baseType: !315, size: 128, align: 64)
!332 = !DIDerivedType(tag: DW_TAG_member, name: "x86_thiscall_mingw", scope: !312, baseType: !315, size: 128, align: 64)
!333 = !DIDerivedType(tag: DW_TAG_member, name: "x86_regcall_v3", scope: !312, baseType: !315, size: 128, align: 64)
!334 = !DIDerivedType(tag: DW_TAG_member, name: "x86_regcall_v4_win", scope: !312, baseType: !315, size: 128, align: 64)
!335 = !DIDerivedType(tag: DW_TAG_member, name: "x86_vectorcall", scope: !312, baseType: !315, size: 128, align: 64)
!336 = !DIDerivedType(tag: DW_TAG_member, name: "x86_interrupt", scope: !312, baseType: !315, size: 128, align: 64)
!337 = !DIDerivedType(tag: DW_TAG_member, name: "aarch64_aapcs", scope: !312, baseType: !315, size: 128, align: 64)
!338 = !DIDerivedType(tag: DW_TAG_member, name: "aarch64_aapcs_darwin", scope: !312, baseType: !315, size: 128, align: 64)
!339 = !DIDerivedType(tag: DW_TAG_member, name: "aarch64_aapcs_win", scope: !312, baseType: !315, size: 128, align: 64)
!340 = !DIDerivedType(tag: DW_TAG_member, name: "aarch64_vfabi", scope: !312, baseType: !315, size: 128, align: 64)
!341 = !DIDerivedType(tag: DW_TAG_member, name: "aarch64_vfabi_sve", scope: !312, baseType: !315, size: 128, align: 64)
!342 = !DIDerivedType(tag: DW_TAG_member, name: "arm_aapcs", scope: !312, baseType: !315, size: 128, align: 64)
!343 = !DIDerivedType(tag: DW_TAG_member, name: "arm_aapcs_vfp", scope: !312, baseType: !315, size: 128, align: 64)
!344 = !DIDerivedType(tag: DW_TAG_member, name: "arm_interrupt", scope: !312, baseType: !270, size: 192, align: 64)
!345 = !DIDerivedType(tag: DW_TAG_member, name: "mips64_n64", scope: !312, baseType: !315, size: 128, align: 64)
!346 = !DIDerivedType(tag: DW_TAG_member, name: "mips64_n32", scope: !312, baseType: !315, size: 128, align: 64)
!347 = !DIDerivedType(tag: DW_TAG_member, name: "mips64_interrupt", scope: !312, baseType: !286, size: 192, align: 64)
!348 = !DIDerivedType(tag: DW_TAG_member, name: "mips_o32", scope: !312, baseType: !315, size: 128, align: 64)
!349 = !DIDerivedType(tag: DW_TAG_member, name: "mips_interrupt", scope: !312, baseType: !286, size: 192, align: 64)
!350 = !DIDerivedType(tag: DW_TAG_member, name: "riscv64_lp64", scope: !312, baseType: !315, size: 128, align: 64)
!351 = !DIDerivedType(tag: DW_TAG_member, name: "riscv64_lp64_v", scope: !312, baseType: !315, size: 128, align: 64)
!352 = !DIDerivedType(tag: DW_TAG_member, name: "riscv64_interrupt", scope: !312, baseType: !301, size: 192, align: 64)
!353 = !DIDerivedType(tag: DW_TAG_member, name: "riscv32_ilp32", scope: !312, baseType: !315, size: 128, align: 64)
!354 = !DIDerivedType(tag: DW_TAG_member, name: "riscv32_ilp32_v", scope: !312, baseType: !315, size: 128, align: 64)
!355 = !DIDerivedType(tag: DW_TAG_member, name: "riscv32_interrupt", scope: !312, baseType: !301, size: 192, align: 64)
!356 = !DIDerivedType(tag: DW_TAG_member, name: "sparc64_sysv", scope: !312, baseType: !315, size: 128, align: 64)
!357 = !DIDerivedType(tag: DW_TAG_member, name: "sparc_sysv", scope: !312, baseType: !315, size: 128, align: 64)
!358 = !DIDerivedType(tag: DW_TAG_member, name: "powerpc64_elf", scope: !312, baseType: !315, size: 128, align: 64)
!359 = !DIDerivedType(tag: DW_TAG_member, name: "powerpc64_elf_altivec", scope: !312, baseType: !315, size: 128, align: 64)
!360 = !DIDerivedType(tag: DW_TAG_member, name: "powerpc64_elf_v2", scope: !312, baseType: !315, size: 128, align: 64)
!361 = !DIDerivedType(tag: DW_TAG_member, name: "powerpc_sysv", scope: !312, baseType: !315, size: 128, align: 64)
!362 = !DIDerivedType(tag: DW_TAG_member, name: "powerpc_sysv_altivec", scope: !312, baseType: !315, size: 128, align: 64)
!363 = !DIDerivedType(tag: DW_TAG_member, name: "powerpc_aix", scope: !312, baseType: !315, size: 128, align: 64)
!364 = !DIDerivedType(tag: DW_TAG_member, name: "powerpc_aix_altivec", scope: !312, baseType: !315, size: 128, align: 64)
!365 = !DIDerivedType(tag: DW_TAG_member, name: "wasm_mvp", scope: !312, baseType: !315, size: 128, align: 64)
!366 = !DIDerivedType(tag: DW_TAG_member, name: "arc_sysv", scope: !312, baseType: !315, size: 128, align: 64)
!367 = !DIDerivedType(tag: DW_TAG_member, name: "bpf_std", scope: !312, baseType: !315, size: 128, align: 64)
!368 = !DIDerivedType(tag: DW_TAG_member, name: "csky_sysv", scope: !312, baseType: !315, size: 128, align: 64)
!369 = !DIDerivedType(tag: DW_TAG_member, name: "csky_interrupt", scope: !312, baseType: !315, size: 128, align: 64)
!370 = !DIDerivedType(tag: DW_TAG_member, name: "hexagon_sysv", scope: !312, baseType: !315, size: 128, align: 64)
!371 = !DIDerivedType(tag: DW_TAG_member, name: "hexagon_sysv_hvx", scope: !312, baseType: !315, size: 128, align: 64)
!372 = !DIDerivedType(tag: DW_TAG_member, name: "lanai_sysv", scope: !312, baseType: !315, size: 128, align: 64)
!373 = !DIDerivedType(tag: DW_TAG_member, name: "loongarch64_lp64", scope: !312, baseType: !315, size: 128, align: 64)
!374 = !DIDerivedType(tag: DW_TAG_member, name: "loongarch32_ilp32", scope: !312, baseType: !315, size: 128, align: 64)
!375 = !DIDerivedType(tag: DW_TAG_member, name: "m68k_sysv", scope: !312, baseType: !315, size: 128, align: 64)
!376 = !DIDerivedType(tag: DW_TAG_member, name: "m68k_gnu", scope: !312, baseType: !315, size: 128, align: 64)
!377 = !DIDerivedType(tag: DW_TAG_member, name: "m68k_rtd", scope: !312, baseType: !315, size: 128, align: 64)
!378 = !DIDerivedType(tag: DW_TAG_member, name: "m68k_interrupt", scope: !312, baseType: !315, size: 128, align: 64)
!379 = !DIDerivedType(tag: DW_TAG_member, name: "msp430_eabi", scope: !312, baseType: !315, size: 128, align: 64)
!380 = !DIDerivedType(tag: DW_TAG_member, name: "or1k_sysv", scope: !312, baseType: !315, size: 128, align: 64)
!381 = !DIDerivedType(tag: DW_TAG_member, name: "propeller_sysv", scope: !312, baseType: !315, size: 128, align: 64)
!382 = !DIDerivedType(tag: DW_TAG_member, name: "s390x_sysv", scope: !312, baseType: !315, size: 128, align: 64)
!383 = !DIDerivedType(tag: DW_TAG_member, name: "s390x_sysv_vx", scope: !312, baseType: !315, size: 128, align: 64)
!384 = !DIDerivedType(tag: DW_TAG_member, name: "ve_sysv", scope: !312, baseType: !315, size: 128, align: 64)
!385 = !DIDerivedType(tag: DW_TAG_member, name: "xcore_xs1", scope: !312, baseType: !315, size: 128, align: 64)
!386 = !DIDerivedType(tag: DW_TAG_member, name: "xcore_xs2", scope: !312, baseType: !315, size: 128, align: 64)
!387 = !DIDerivedType(tag: DW_TAG_member, name: "xtensa_call0", scope: !312, baseType: !315, size: 128, align: 64)
!388 = !DIDerivedType(tag: DW_TAG_member, name: "xtensa_windowed", scope: !312, baseType: !315, size: 128, align: 64)
!389 = !DIDerivedType(tag: DW_TAG_member, name: "amdgcn_device", scope: !312, baseType: !315, size: 128, align: 64)
!390 = !DIDerivedType(tag: DW_TAG_member, name: "amdgcn_cs", scope: !312, baseType: !315, size: 128, align: 64)
!391 = !DIDerivedType(tag: DW_TAG_member, name: "tag", scope: !309, baseType: !308, size: 8, align: 8, offset: 192)
!392 = !{!393, !394, !395, !396, !397, !398, !399, !400, !401, !402, !403, !404, !405, !406, !407, !408, !409, !410, !411, !412, !413, !414, !415, !416, !417, !418, !419, !420, !421, !422, !423, !424, !425, !426, !427, !428, !429, !430, !431, !432, !433, !434, !435, !436, !437, !438, !439, !440, !441, !442, !443, !444, !445, !446, !447, !448, !449, !450, !451, !452, !453, !454, !455, !456, !457, !458, !459, !460, !461, !462, !463, !464, !465, !466, !467, !468, !469, !470, !471, !472, !473, !474, !475, !476, !477}
!393 = !DIEnumerator(name: "auto", value: 0, isUnsigned: true)
!394 = !DIEnumerator(name: "async", value: 1, isUnsigned: true)
!395 = !DIEnumerator(name: "naked", value: 2, isUnsigned: true)
!396 = !DIEnumerator(name: "inline", value: 3, isUnsigned: true)
!397 = !DIEnumerator(name: "x86_64_sysv", value: 4, isUnsigned: true)
!398 = !DIEnumerator(name: "x86_64_win", value: 5, isUnsigned: true)
!399 = !DIEnumerator(name: "x86_64_regcall_v3_sysv", value: 6, isUnsigned: true)
!400 = !DIEnumerator(name: "x86_64_regcall_v4_win", value: 7, isUnsigned: true)
!401 = !DIEnumerator(name: "x86_64_vectorcall", value: 8, isUnsigned: true)
!402 = !DIEnumerator(name: "x86_64_interrupt", value: 9, isUnsigned: true)
!403 = !DIEnumerator(name: "x86_sysv", value: 10, isUnsigned: true)
!404 = !DIEnumerator(name: "x86_win", value: 11, isUnsigned: true)
!405 = !DIEnumerator(name: "x86_stdcall", value: 12, isUnsigned: true)
!406 = !DIEnumerator(name: "x86_fastcall", value: 13, isUnsigned: true)
!407 = !DIEnumerator(name: "x86_thiscall", value: 14, isUnsigned: true)
!408 = !DIEnumerator(name: "x86_thiscall_mingw", value: 15, isUnsigned: true)
!409 = !DIEnumerator(name: "x86_regcall_v3", value: 16, isUnsigned: true)
!410 = !DIEnumerator(name: "x86_regcall_v4_win", value: 17, isUnsigned: true)
!411 = !DIEnumerator(name: "x86_vectorcall", value: 18, isUnsigned: true)
!412 = !DIEnumerator(name: "x86_interrupt", value: 19, isUnsigned: true)
!413 = !DIEnumerator(name: "aarch64_aapcs", value: 20, isUnsigned: true)
!414 = !DIEnumerator(name: "aarch64_aapcs_darwin", value: 21, isUnsigned: true)
!415 = !DIEnumerator(name: "aarch64_aapcs_win", value: 22, isUnsigned: true)
!416 = !DIEnumerator(name: "aarch64_vfabi", value: 23, isUnsigned: true)
!417 = !DIEnumerator(name: "aarch64_vfabi_sve", value: 24, isUnsigned: true)
!418 = !DIEnumerator(name: "arm_aapcs", value: 25, isUnsigned: true)
!419 = !DIEnumerator(name: "arm_aapcs_vfp", value: 26, isUnsigned: true)
!420 = !DIEnumerator(name: "arm_interrupt", value: 27, isUnsigned: true)
!421 = !DIEnumerator(name: "mips64_n64", value: 28, isUnsigned: true)
!422 = !DIEnumerator(name: "mips64_n32", value: 29, isUnsigned: true)
!423 = !DIEnumerator(name: "mips64_interrupt", value: 30, isUnsigned: true)
!424 = !DIEnumerator(name: "mips_o32", value: 31, isUnsigned: true)
!425 = !DIEnumerator(name: "mips_interrupt", value: 32, isUnsigned: true)
!426 = !DIEnumerator(name: "riscv64_lp64", value: 33, isUnsigned: true)
!427 = !DIEnumerator(name: "riscv64_lp64_v", value: 34, isUnsigned: true)
!428 = !DIEnumerator(name: "riscv64_interrupt", value: 35, isUnsigned: true)
!429 = !DIEnumerator(name: "riscv32_ilp32", value: 36, isUnsigned: true)
!430 = !DIEnumerator(name: "riscv32_ilp32_v", value: 37, isUnsigned: true)
!431 = !DIEnumerator(name: "riscv32_interrupt", value: 38, isUnsigned: true)
!432 = !DIEnumerator(name: "sparc64_sysv", value: 39, isUnsigned: true)
!433 = !DIEnumerator(name: "sparc_sysv", value: 40, isUnsigned: true)
!434 = !DIEnumerator(name: "powerpc64_elf", value: 41, isUnsigned: true)
!435 = !DIEnumerator(name: "powerpc64_elf_altivec", value: 42, isUnsigned: true)
!436 = !DIEnumerator(name: "powerpc64_elf_v2", value: 43, isUnsigned: true)
!437 = !DIEnumerator(name: "powerpc_sysv", value: 44, isUnsigned: true)
!438 = !DIEnumerator(name: "powerpc_sysv_altivec", value: 45, isUnsigned: true)
!439 = !DIEnumerator(name: "powerpc_aix", value: 46, isUnsigned: true)
!440 = !DIEnumerator(name: "powerpc_aix_altivec", value: 47, isUnsigned: true)
!441 = !DIEnumerator(name: "wasm_mvp", value: 48, isUnsigned: true)
!442 = !DIEnumerator(name: "arc_sysv", value: 49, isUnsigned: true)
!443 = !DIEnumerator(name: "avr_gnu", value: 50, isUnsigned: true)
!444 = !DIEnumerator(name: "avr_builtin", value: 51, isUnsigned: true)
!445 = !DIEnumerator(name: "avr_signal", value: 52, isUnsigned: true)
!446 = !DIEnumerator(name: "avr_interrupt", value: 53, isUnsigned: true)
!447 = !DIEnumerator(name: "bpf_std", value: 54, isUnsigned: true)
!448 = !DIEnumerator(name: "csky_sysv", value: 55, isUnsigned: true)
!449 = !DIEnumerator(name: "csky_interrupt", value: 56, isUnsigned: true)
!450 = !DIEnumerator(name: "hexagon_sysv", value: 57, isUnsigned: true)
!451 = !DIEnumerator(name: "hexagon_sysv_hvx", value: 58, isUnsigned: true)
!452 = !DIEnumerator(name: "lanai_sysv", value: 59, isUnsigned: true)
!453 = !DIEnumerator(name: "loongarch64_lp64", value: 60, isUnsigned: true)
!454 = !DIEnumerator(name: "loongarch32_ilp32", value: 61, isUnsigned: true)
!455 = !DIEnumerator(name: "m68k_sysv", value: 62, isUnsigned: true)
!456 = !DIEnumerator(name: "m68k_gnu", value: 63, isUnsigned: true)
!457 = !DIEnumerator(name: "m68k_rtd", value: 64, isUnsigned: true)
!458 = !DIEnumerator(name: "m68k_interrupt", value: 65, isUnsigned: true)
!459 = !DIEnumerator(name: "msp430_eabi", value: 66, isUnsigned: true)
!460 = !DIEnumerator(name: "or1k_sysv", value: 67, isUnsigned: true)
!461 = !DIEnumerator(name: "propeller_sysv", value: 68, isUnsigned: true)
!462 = !DIEnumerator(name: "s390x_sysv", value: 69, isUnsigned: true)
!463 = !DIEnumerator(name: "s390x_sysv_vx", value: 70, isUnsigned: true)
!464 = !DIEnumerator(name: "ve_sysv", value: 71, isUnsigned: true)
!465 = !DIEnumerator(name: "xcore_xs1", value: 72, isUnsigned: true)
!466 = !DIEnumerator(name: "xcore_xs2", value: 73, isUnsigned: true)
!467 = !DIEnumerator(name: "xtensa_call0", value: 74, isUnsigned: true)
!468 = !DIEnumerator(name: "xtensa_windowed", value: 75, isUnsigned: true)
!469 = !DIEnumerator(name: "amdgcn_device", value: 76, isUnsigned: true)
!470 = !DIEnumerator(name: "amdgcn_kernel", value: 77, isUnsigned: true)
!471 = !DIEnumerator(name: "amdgcn_cs", value: 78, isUnsigned: true)
!472 = !DIEnumerator(name: "nvptx_device", value: 79, isUnsigned: true)
!473 = !DIEnumerator(name: "nvptx_kernel", value: 80, isUnsigned: true)
!474 = !DIEnumerator(name: "spirv_device", value: 81, isUnsigned: true)
!475 = !DIEnumerator(name: "spirv_kernel", value: 82, isUnsigned: true)
!476 = !DIEnumerator(name: "spirv_fragment", value: 83, isUnsigned: true)
!477 = !DIEnumerator(name: "spirv_vertex", value: 84, isUnsigned: true)
!478 = !{!0, !20, !24, !32, !479, !481, !483, !485, !487, !496, !505, !507}
!479 = !DIGlobalVariableExpression(var: !480, expr: !DIExpression())
!480 = distinct !DIGlobalVariable(name: "cpu", linkageName: "builtin.cpu", scope: !2, file: !2, line: 14, type: !40, isLocal: true, isDefinition: true)
!481 = !DIGlobalVariableExpression(var: !482, expr: !DIExpression())
!482 = distinct !DIGlobalVariable(name: "os", linkageName: "builtin.os", scope: !2, file: !2, line: 93, type: !106, isLocal: true, isDefinition: true)
!483 = !DIGlobalVariableExpression(var: !484, expr: !DIExpression())
!484 = distinct !DIGlobalVariable(name: "abi", linkageName: "builtin.abi", scope: !2, file: !2, line: 13, type: !222, isLocal: true, isDefinition: true)
!485 = !DIGlobalVariableExpression(var: !486, expr: !DIExpression())
!486 = distinct !DIGlobalVariable(name: "object_format", linkageName: "builtin.object_format", scope: !2, file: !2, line: 115, type: !255, isLocal: true, isDefinition: true)
!487 = !DIGlobalVariableExpression(var: !488, expr: !DIExpression())
!488 = distinct !DIGlobalVariable(name: "none", linkageName: "Target.DynamicLinker.none", scope: !34, file: !34, line: 2072, type: !489, isLocal: true, isDefinition: true)
!489 = !DICompositeType(tag: DW_TAG_structure_type, name: "Target.DynamicLinker", scope: !36, size: 2048, align: 8, elements: !490)
!490 = !{!491, !495}
!491 = !DIDerivedType(tag: DW_TAG_member, name: "buffer", scope: !489, baseType: !492, size: 2040, align: 8)
!492 = !DICompositeType(tag: DW_TAG_array_type, baseType: !51, size: 2040, align: 8, elements: !493)
!493 = !{!494}
!494 = !DISubrange(count: 255, lowerBound: 0)
!495 = !DIDerivedType(tag: DW_TAG_member, name: "len", scope: !489, baseType: !51, size: 8, align: 8, offset: 2040)
!496 = !DIGlobalVariableExpression(var: !497, expr: !DIExpression())
!497 = distinct !DIGlobalVariable(name: "target", linkageName: "builtin.target", scope: !2, file: !2, line: 108, type: !498, isLocal: true, isDefinition: true)
!498 = !DICompositeType(tag: DW_TAG_structure_type, name: "Target", scope: !36, size: 4096, align: 64, elements: !499)
!499 = !{!500, !501, !502, !503, !504}
!500 = !DIDerivedType(tag: DW_TAG_member, name: "cpu", scope: !498, baseType: !40, size: 448, align: 64)
!501 = !DIDerivedType(tag: DW_TAG_member, name: "os", scope: !498, baseType: !106, size: 1536, align: 64, offset: 448)
!502 = !DIDerivedType(tag: DW_TAG_member, name: "abi", scope: !498, baseType: !222, size: 8, align: 8, offset: 1984)
!503 = !DIDerivedType(tag: DW_TAG_member, name: "ofmt", scope: !498, baseType: !255, size: 8, align: 8, offset: 1992)
!504 = !DIDerivedType(tag: DW_TAG_member, name: "dynamic_linker", scope: !498, baseType: !489, size: 2048, align: 8, offset: 2000)
!505 = !DIGlobalVariableExpression(var: !506, expr: !DIExpression())
!506 = distinct !DIGlobalVariable(name: "c", linkageName: "builtin.CallingConvention.c", scope: !4, file: !4, line: 172, type: !309, isLocal: true, isDefinition: true)
!507 = !DIGlobalVariableExpression(var: !508, expr: !DIExpression())
!508 = distinct !DIGlobalVariable(name: "apple_m2", linkageName: "Target.aarch64.cpu.apple_m2", scope: !509, file: !509, line: 2165, type: !44, isLocal: true, isDefinition: true)
!509 = !DIFile(filename: "aarch64.zig", directory: "/opt/homebrew/Cellar/zig/0.15.1/lib/zig/std/Target")
!510 = !{!511}
!511 = !DIDerivedType(tag: DW_TAG_member, name: "ints", scope: !35, baseType: !512, size: 320, align: 64)
!512 = !DICompositeType(tag: DW_TAG_array_type, baseType: !53, size: 320, align: 64, elements: !513)
!513 = !{!514}
!514 = !DISubrange(count: 5, lowerBound: 0)
!515 = !{i32 8, !"PIC Level", i32 2}
!516 = !{i32 2, !"Debug Info Version", i32 3}
!517 = !{i32 7, !"Dwarf Version", i32 4}
!518 = distinct !DISubprogram(name: "transformCallToOperation", linkageName: "c_api_transform.transformCallToOperation", scope: !519, file: !519, line: 51, type: !520, scopeLine: 51, flags: DIFlagStaticMember, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !36)
!519 = !DIFile(filename: "c_api_transform.zig", directory: "/Users/jimmyhmiller/Documents/Code/PlayGround/zig/mlir-lisp/src")
!520 = !DISubroutineType(types: !521)
!521 = !{!522, !523, !531, !527, !527}
!522 = !DIBasicType(name: "void", encoding: DW_ATE_signed)
!523 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*@typeInfo(@typeInfo(@TypeOf(c_api_transform.transformCallToOperation)).@\22fn\22.return_type.?).error_union.error_set!?*anyopaque", baseType: !524, size: 64, align: 64)
!524 = !DICompositeType(tag: DW_TAG_structure_type, name: "@typeInfo(@typeInfo(@TypeOf(c_api_transform.transformCallToOperation)).@\22fn\22.return_type.?).error_union.error_set!?*anyopaque", scope: !36, size: 128, align: 64, elements: !525)
!525 = !{!526, !529}
!526 = !DIDerivedType(tag: DW_TAG_member, name: "value", scope: !524, baseType: !527, size: 64, align: 64)
!527 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*anyopaque", baseType: !528, size: 64, align: 8)
!528 = !DIBasicType(name: "anyopaque", encoding: DW_ATE_signed)
!529 = !DIDerivedType(tag: DW_TAG_member, name: "tag", scope: !524, baseType: !530, size: 16, align: 16, offset: 64)
!530 = !DIBasicType(name: "anyerror", size: 16, encoding: DW_ATE_unsigned)
!531 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*usize", baseType: !53, size: 64, align: 64)
!532 = !DILocation(line: 51, column: 90, scope: !518)
!533 = !DILocalVariable(name: "allocator", arg: 1, scope: !518, file: !519, line: 51, type: !527)
!534 = !DILocalVariable(name: "call_expr", arg: 2, scope: !518, file: !519, line: 51, type: !527)
!535 = !DILocation(line: 52, column: 9, scope: !536)
!536 = !DILexicalBlock(scope: !518, file: !519, line: 52, column: 9)
!537 = !DILocation(line: 52, column: 9, scope: !538)
!538 = !DILexicalBlock(scope: !536, file: !519, line: 52, column: 9)
!539 = !DILocation(line: 55, column: 37, scope: !540)
!540 = !DILexicalBlock(scope: !518, file: !519, line: 55, column: 5)
!541 = !DILocation(line: 52, column: 9, scope: !542)
!542 = !DILexicalBlock(scope: !538, file: !519, line: 52, column: 9)
!543 = !DILocation(line: 52, column: 49, scope: !538)
!544 = !DILocation(line: 52, column: 49, scope: !545)
!545 = !DILexicalBlock(scope: !536, file: !519, line: 52, column: 49)
!546 = !DILocation(line: 55, column: 67, scope: !518)
!547 = !DILocalVariable(name: "call_list", scope: !518, file: !519, line: 55, type: !527)
!548 = !DILocation(line: 58, column: 38, scope: !518)
!549 = !DILocalVariable(name: "list_len", scope: !518, file: !519, line: 58, type: !53)
!550 = !DILocation(line: 59, column: 9, scope: !551)
!551 = !DILexicalBlock(scope: !518, file: !519, line: 59, column: 9)
!552 = !DILocation(line: 55, column: 37, scope: !553)
!553 = !DILexicalBlock(scope: !540, file: !519, line: 55, column: 37)
!554 = !DILocation(line: 55, column: 67, scope: !553)
!555 = !DILocation(line: 62, column: 39, scope: !518)
!556 = !DILocalVariable(name: "call_ident", scope: !518, file: !519, line: 62, type: !527)
!557 = !DILocation(line: 63, column: 42, scope: !518)
!558 = !DILocalVariable(name: "callee_symbol", scope: !518, file: !519, line: 63, type: !527)
!559 = !DILocation(line: 64, column: 40, scope: !518)
!560 = !DILocalVariable(name: "return_type", scope: !518, file: !519, line: 64, type: !527)
!561 = !DILocation(line: 67, column: 37, scope: !518)
!562 = !DILocalVariable(name: "call_atom", scope: !518, file: !519, line: 67, type: !50)
!563 = !DILocation(line: 74, column: 52, scope: !564)
!564 = !DILexicalBlock(scope: !518, file: !519, line: 74, column: 5)
!565 = !DILocation(line: 56, column: 31, scope: !566)
!566 = !DILexicalBlock(scope: !551, file: !519, line: 59, column: 9)
!567 = !DILocation(line: 59, column: 23, scope: !566)
!568 = !DILocation(line: 59, column: 23, scope: !569)
!569 = !DILexicalBlock(scope: !551, file: !519, line: 59, column: 23)
!570 = !DILocation(line: 74, column: 84, scope: !518)
!571 = !DILocalVariable(name: "operation_ident", scope: !518, file: !519, line: 74, type: !527)
!572 = !DILocation(line: 77, column: 47, scope: !573)
!573 = !DILexicalBlock(scope: !518, file: !519, line: 77, column: 5)
!574 = !DILocation(line: 74, column: 52, scope: !575)
!575 = !DILexicalBlock(scope: !564, file: !519, line: 74, column: 52)
!576 = !DILocation(line: 68, column: 26, scope: !575)
!577 = !DILocation(line: 56, column: 31, scope: !575)
!578 = !DILocation(line: 74, column: 84, scope: !575)
!579 = !DILocation(line: 77, column: 74, scope: !518)
!580 = !DILocalVariable(name: "name_ident", scope: !518, file: !519, line: 77, type: !527)
!581 = !DILocation(line: 78, column: 52, scope: !582)
!582 = !DILexicalBlock(scope: !518, file: !519, line: 78, column: 5)
!583 = !DILocation(line: 77, column: 47, scope: !584)
!584 = !DILexicalBlock(scope: !573, file: !519, line: 77, column: 47)
!585 = !DILocation(line: 68, column: 26, scope: !584)
!586 = !DILocation(line: 56, column: 31, scope: !584)
!587 = !DILocation(line: 77, column: 74, scope: !584)
!588 = !DILocation(line: 78, column: 84, scope: !518)
!589 = !DILocalVariable(name: "func_call_ident", scope: !518, file: !519, line: 78, type: !527)
!590 = !DILocation(line: 80, column: 46, scope: !591)
!591 = !DILexicalBlock(scope: !518, file: !519, line: 80, column: 5)
!592 = !DILocation(line: 78, column: 52, scope: !593)
!593 = !DILexicalBlock(scope: !582, file: !519, line: 78, column: 52)
!594 = !DILocation(line: 68, column: 26, scope: !593)
!595 = !DILocation(line: 56, column: 31, scope: !593)
!596 = !DILocation(line: 78, column: 84, scope: !593)
!597 = !DILocation(line: 80, column: 65, scope: !518)
!598 = !DILocalVariable(name: "name_list_vec", scope: !518, file: !519, line: 80, type: !527)
!599 = !DILocation(line: 81, column: 41, scope: !600)
!600 = !DILexicalBlock(scope: !518, file: !519, line: 81, column: 5)
!601 = !DILocation(line: 80, column: 46, scope: !602)
!602 = !DILexicalBlock(scope: !591, file: !519, line: 80, column: 46)
!603 = !DILocation(line: 68, column: 26, scope: !602)
!604 = !DILocation(line: 56, column: 31, scope: !602)
!605 = !DILocation(line: 80, column: 65, scope: !602)
!606 = !DILocation(line: 81, column: 87, scope: !518)
!607 = !DILocalVariable(name: "name_vec_1", scope: !518, file: !519, line: 81, type: !527)
!608 = !DILocation(line: 82, column: 25, scope: !518)
!609 = !DILocation(line: 83, column: 41, scope: !610)
!610 = !DILexicalBlock(scope: !518, file: !519, line: 83, column: 5)
!611 = !DILocation(line: 81, column: 41, scope: !612)
!612 = !DILexicalBlock(scope: !600, file: !519, line: 81, column: 41)
!613 = !DILocation(line: 68, column: 26, scope: !612)
!614 = !DILocation(line: 56, column: 31, scope: !612)
!615 = !DILocation(line: 81, column: 87, scope: !612)
!616 = !DILocation(line: 83, column: 89, scope: !518)
!617 = !DILocalVariable(name: "name_vec_2", scope: !518, file: !519, line: 83, type: !527)
!618 = !DILocation(line: 84, column: 25, scope: !518)
!619 = !DILocation(line: 86, column: 42, scope: !620)
!620 = !DILexicalBlock(scope: !518, file: !519, line: 86, column: 5)
!621 = !DILocation(line: 83, column: 41, scope: !622)
!622 = !DILexicalBlock(scope: !610, file: !519, line: 83, column: 41)
!623 = !DILocation(line: 68, column: 26, scope: !622)
!624 = !DILocation(line: 56, column: 31, scope: !622)
!625 = !DILocation(line: 83, column: 89, scope: !622)
!626 = !DILocation(line: 86, column: 73, scope: !518)
!627 = !DILocalVariable(name: "name_clause", scope: !518, file: !519, line: 86, type: !527)
!628 = !DILocation(line: 90, column: 58, scope: !629)
!629 = !DILexicalBlock(scope: !518, file: !519, line: 90, column: 5)
!630 = !DILocation(line: 86, column: 42, scope: !631)
!631 = !DILexicalBlock(scope: !620, file: !519, line: 86, column: 42)
!632 = !DILocation(line: 68, column: 26, scope: !631)
!633 = !DILocation(line: 56, column: 31, scope: !631)
!634 = !DILocation(line: 86, column: 73, scope: !631)
!635 = !DILocation(line: 90, column: 96, scope: !518)
!636 = !DILocalVariable(name: "result_bindings_ident", scope: !518, file: !519, line: 90, type: !527)
!637 = !DILocation(line: 91, column: 52, scope: !638)
!638 = !DILexicalBlock(scope: !518, file: !519, line: 91, column: 5)
!639 = !DILocation(line: 90, column: 58, scope: !640)
!640 = !DILexicalBlock(scope: !629, file: !519, line: 90, column: 58)
!641 = !DILocation(line: 68, column: 26, scope: !640)
!642 = !DILocation(line: 56, column: 31, scope: !640)
!643 = !DILocation(line: 90, column: 96, scope: !640)
!644 = !DILocation(line: 91, column: 83, scope: !518)
!645 = !DILocalVariable(name: "gensym_value_id", scope: !518, file: !519, line: 91, type: !527)
!646 = !DILocation(line: 93, column: 43, scope: !647)
!647 = !DILexicalBlock(scope: !518, file: !519, line: 93, column: 5)
!648 = !DILocation(line: 91, column: 52, scope: !649)
!649 = !DILexicalBlock(scope: !638, file: !519, line: 91, column: 52)
!650 = !DILocation(line: 68, column: 26, scope: !649)
!651 = !DILocation(line: 56, column: 31, scope: !649)
!652 = !DILocation(line: 91, column: 83, scope: !649)
!653 = !DILocation(line: 93, column: 62, scope: !518)
!654 = !DILocalVariable(name: "gensym_vec", scope: !518, file: !519, line: 93, type: !527)
!655 = !DILocation(line: 94, column: 43, scope: !656)
!656 = !DILexicalBlock(scope: !518, file: !519, line: 94, column: 5)
!657 = !DILocation(line: 93, column: 43, scope: !658)
!658 = !DILexicalBlock(scope: !647, file: !519, line: 93, column: 43)
!659 = !DILocation(line: 68, column: 26, scope: !658)
!660 = !DILocation(line: 56, column: 31, scope: !658)
!661 = !DILocation(line: 93, column: 62, scope: !658)
!662 = !DILocation(line: 94, column: 91, scope: !518)
!663 = !DILocalVariable(name: "gensym_vec_1", scope: !518, file: !519, line: 94, type: !527)
!664 = !DILocation(line: 95, column: 25, scope: !518)
!665 = !DILocation(line: 98, column: 46, scope: !666)
!666 = !DILexicalBlock(scope: !518, file: !519, line: 98, column: 5)
!667 = !DILocation(line: 94, column: 43, scope: !668)
!668 = !DILexicalBlock(scope: !656, file: !519, line: 94, column: 43)
!669 = !DILocation(line: 68, column: 26, scope: !668)
!670 = !DILocation(line: 56, column: 31, scope: !668)
!671 = !DILocation(line: 94, column: 91, scope: !668)
!672 = !DILocation(line: 98, column: 79, scope: !518)
!673 = !DILocalVariable(name: "bindings_vector", scope: !518, file: !519, line: 98, type: !527)
!674 = !DILocation(line: 100, column: 45, scope: !675)
!675 = !DILexicalBlock(scope: !518, file: !519, line: 100, column: 5)
!676 = !DILocation(line: 98, column: 46, scope: !677)
!677 = !DILexicalBlock(scope: !666, file: !519, line: 98, column: 46)
!678 = !DILocation(line: 68, column: 26, scope: !677)
!679 = !DILocation(line: 56, column: 31, scope: !677)
!680 = !DILocation(line: 98, column: 79, scope: !677)
!681 = !DILocation(line: 100, column: 64, scope: !518)
!682 = !DILocalVariable(name: "bindings_vec", scope: !518, file: !519, line: 100, type: !527)
!683 = !DILocation(line: 101, column: 45, scope: !684)
!684 = !DILexicalBlock(scope: !518, file: !519, line: 101, column: 5)
!685 = !DILocation(line: 100, column: 45, scope: !686)
!686 = !DILexicalBlock(scope: !675, file: !519, line: 100, column: 45)
!687 = !DILocation(line: 68, column: 26, scope: !686)
!688 = !DILocation(line: 56, column: 31, scope: !686)
!689 = !DILocation(line: 100, column: 64, scope: !686)
!690 = !DILocation(line: 101, column: 101, scope: !518)
!691 = !DILocalVariable(name: "bindings_vec_1", scope: !518, file: !519, line: 101, type: !527)
!692 = !DILocation(line: 102, column: 25, scope: !518)
!693 = !DILocation(line: 103, column: 45, scope: !694)
!694 = !DILexicalBlock(scope: !518, file: !519, line: 103, column: 5)
!695 = !DILocation(line: 101, column: 45, scope: !696)
!696 = !DILexicalBlock(scope: !684, file: !519, line: 101, column: 45)
!697 = !DILocation(line: 68, column: 26, scope: !696)
!698 = !DILocation(line: 56, column: 31, scope: !696)
!699 = !DILocation(line: 101, column: 101, scope: !696)
!700 = !DILocation(line: 103, column: 97, scope: !518)
!701 = !DILocalVariable(name: "bindings_vec_2", scope: !518, file: !519, line: 103, type: !527)
!702 = !DILocation(line: 104, column: 25, scope: !518)
!703 = !DILocation(line: 106, column: 46, scope: !704)
!704 = !DILexicalBlock(scope: !518, file: !519, line: 106, column: 5)
!705 = !DILocation(line: 103, column: 45, scope: !706)
!706 = !DILexicalBlock(scope: !694, file: !519, line: 103, column: 45)
!707 = !DILocation(line: 68, column: 26, scope: !706)
!708 = !DILocation(line: 56, column: 31, scope: !706)
!709 = !DILocation(line: 103, column: 97, scope: !706)
!710 = !DILocation(line: 106, column: 81, scope: !518)
!711 = !DILocalVariable(name: "bindings_clause", scope: !518, file: !519, line: 106, type: !527)
!712 = !DILocation(line: 109, column: 55, scope: !713)
!713 = !DILexicalBlock(scope: !518, file: !519, line: 109, column: 5)
!714 = !DILocation(line: 106, column: 46, scope: !715)
!715 = !DILexicalBlock(scope: !704, file: !519, line: 106, column: 46)
!716 = !DILocation(line: 68, column: 26, scope: !715)
!717 = !DILocation(line: 56, column: 31, scope: !715)
!718 = !DILocation(line: 106, column: 81, scope: !715)
!719 = !DILocation(line: 109, column: 90, scope: !518)
!720 = !DILocalVariable(name: "result_types_ident", scope: !518, file: !519, line: 109, type: !527)
!721 = !DILocation(line: 112, column: 45, scope: !722)
!722 = !DILexicalBlock(scope: !518, file: !519, line: 112, column: 5)
!723 = !DILocation(line: 109, column: 55, scope: !724)
!724 = !DILexicalBlock(scope: !713, file: !519, line: 109, column: 55)
!725 = !DILocation(line: 68, column: 26, scope: !724)
!726 = !DILocation(line: 56, column: 31, scope: !724)
!727 = !DILocation(line: 109, column: 90, scope: !724)
!728 = !DILocation(line: 112, column: 77, scope: !518)
!729 = !DILocalVariable(name: "type_expr", scope: !518, file: !519, line: 112, type: !527)
!730 = !DILocation(line: 114, column: 42, scope: !731)
!731 = !DILexicalBlock(scope: !518, file: !519, line: 114, column: 5)
!732 = !DILocation(line: 112, column: 45, scope: !733)
!733 = !DILexicalBlock(scope: !722, file: !519, line: 112, column: 45)
!734 = !DILocation(line: 68, column: 26, scope: !733)
!735 = !DILocation(line: 56, column: 31, scope: !733)
!736 = !DILocation(line: 112, column: 77, scope: !733)
!737 = !DILocation(line: 114, column: 61, scope: !518)
!738 = !DILocalVariable(name: "types_vec", scope: !518, file: !519, line: 114, type: !527)
!739 = !DILocation(line: 115, column: 42, scope: !740)
!740 = !DILexicalBlock(scope: !518, file: !519, line: 115, column: 5)
!741 = !DILocation(line: 114, column: 42, scope: !742)
!742 = !DILexicalBlock(scope: !731, file: !519, line: 114, column: 42)
!743 = !DILocation(line: 68, column: 26, scope: !742)
!744 = !DILocation(line: 56, column: 31, scope: !742)
!745 = !DILocation(line: 114, column: 61, scope: !742)
!746 = !DILocation(line: 115, column: 92, scope: !518)
!747 = !DILocalVariable(name: "types_vec_1", scope: !518, file: !519, line: 115, type: !527)
!748 = !DILocation(line: 116, column: 25, scope: !518)
!749 = !DILocation(line: 117, column: 42, scope: !750)
!750 = !DILexicalBlock(scope: !518, file: !519, line: 117, column: 5)
!751 = !DILocation(line: 115, column: 42, scope: !752)
!752 = !DILexicalBlock(scope: !740, file: !519, line: 115, column: 42)
!753 = !DILocation(line: 68, column: 26, scope: !752)
!754 = !DILocation(line: 56, column: 31, scope: !752)
!755 = !DILocation(line: 115, column: 92, scope: !752)
!756 = !DILocation(line: 117, column: 85, scope: !518)
!757 = !DILocalVariable(name: "types_vec_2", scope: !518, file: !519, line: 117, type: !527)
!758 = !DILocation(line: 118, column: 25, scope: !518)
!759 = !DILocation(line: 120, column: 43, scope: !760)
!760 = !DILexicalBlock(scope: !518, file: !519, line: 120, column: 5)
!761 = !DILocation(line: 117, column: 42, scope: !762)
!762 = !DILexicalBlock(scope: !750, file: !519, line: 117, column: 42)
!763 = !DILocation(line: 68, column: 26, scope: !762)
!764 = !DILocation(line: 56, column: 31, scope: !762)
!765 = !DILocation(line: 117, column: 85, scope: !762)
!766 = !DILocation(line: 120, column: 75, scope: !518)
!767 = !DILocalVariable(name: "types_clause", scope: !518, file: !519, line: 120, type: !527)
!768 = !DILocation(line: 123, column: 53, scope: !769)
!769 = !DILexicalBlock(scope: !518, file: !519, line: 123, column: 5)
!770 = !DILocation(line: 120, column: 43, scope: !771)
!771 = !DILexicalBlock(scope: !760, file: !519, line: 120, column: 43)
!772 = !DILocation(line: 68, column: 26, scope: !771)
!773 = !DILocation(line: 56, column: 31, scope: !771)
!774 = !DILocation(line: 120, column: 75, scope: !771)
!775 = !DILocation(line: 123, column: 86, scope: !518)
!776 = !DILocalVariable(name: "attributes_ident", scope: !518, file: !519, line: 123, type: !527)
!777 = !DILocation(line: 126, column: 48, scope: !778)
!778 = !DILexicalBlock(scope: !518, file: !519, line: 126, column: 5)
!779 = !DILocation(line: 123, column: 53, scope: !780)
!780 = !DILexicalBlock(scope: !769, file: !519, line: 123, column: 53)
!781 = !DILocation(line: 68, column: 26, scope: !780)
!782 = !DILocation(line: 56, column: 31, scope: !780)
!783 = !DILocation(line: 123, column: 86, scope: !780)
!784 = !DILocation(line: 126, column: 78, scope: !518)
!785 = !DILocalVariable(name: "callee_keyword", scope: !518, file: !519, line: 126, type: !527)
!786 = !DILocation(line: 128, column: 40, scope: !787)
!787 = !DILexicalBlock(scope: !518, file: !519, line: 128, column: 5)
!788 = !DILocation(line: 126, column: 48, scope: !789)
!789 = !DILexicalBlock(scope: !778, file: !519, line: 126, column: 48)
!790 = !DILocation(line: 68, column: 26, scope: !789)
!791 = !DILocation(line: 56, column: 31, scope: !789)
!792 = !DILocation(line: 126, column: 78, scope: !789)
!793 = !DILocation(line: 128, column: 59, scope: !518)
!794 = !DILocalVariable(name: "map_vec", scope: !518, file: !519, line: 128, type: !527)
!795 = !DILocation(line: 129, column: 40, scope: !796)
!796 = !DILexicalBlock(scope: !518, file: !519, line: 129, column: 5)
!797 = !DILocation(line: 128, column: 40, scope: !798)
!798 = !DILexicalBlock(scope: !787, file: !519, line: 128, column: 40)
!799 = !DILocation(line: 68, column: 26, scope: !798)
!800 = !DILocation(line: 56, column: 31, scope: !798)
!801 = !DILocation(line: 128, column: 59, scope: !798)
!802 = !DILocation(line: 129, column: 84, scope: !518)
!803 = !DILocalVariable(name: "map_vec_1", scope: !518, file: !519, line: 129, type: !527)
!804 = !DILocation(line: 130, column: 25, scope: !518)
!805 = !DILocation(line: 131, column: 40, scope: !806)
!806 = !DILexicalBlock(scope: !518, file: !519, line: 131, column: 5)
!807 = !DILocation(line: 129, column: 40, scope: !808)
!808 = !DILexicalBlock(scope: !796, file: !519, line: 129, column: 40)
!809 = !DILocation(line: 68, column: 26, scope: !808)
!810 = !DILocation(line: 56, column: 31, scope: !808)
!811 = !DILocation(line: 129, column: 84, scope: !808)
!812 = !DILocation(line: 131, column: 85, scope: !518)
!813 = !DILocalVariable(name: "map_vec_2", scope: !518, file: !519, line: 131, type: !527)
!814 = !DILocation(line: 132, column: 25, scope: !518)
!815 = !DILocation(line: 134, column: 44, scope: !816)
!816 = !DILexicalBlock(scope: !518, file: !519, line: 134, column: 5)
!817 = !DILocation(line: 131, column: 40, scope: !818)
!818 = !DILexicalBlock(scope: !806, file: !519, line: 131, column: 40)
!819 = !DILocation(line: 68, column: 26, scope: !818)
!820 = !DILocation(line: 56, column: 31, scope: !818)
!821 = !DILocation(line: 131, column: 85, scope: !818)
!822 = !DILocation(line: 134, column: 74, scope: !518)
!823 = !DILocalVariable(name: "attributes_map", scope: !518, file: !519, line: 134, type: !527)
!824 = !DILocation(line: 136, column: 42, scope: !825)
!825 = !DILexicalBlock(scope: !518, file: !519, line: 136, column: 5)
!826 = !DILocation(line: 134, column: 44, scope: !827)
!827 = !DILexicalBlock(scope: !816, file: !519, line: 134, column: 44)
!828 = !DILocation(line: 68, column: 26, scope: !827)
!829 = !DILocation(line: 56, column: 31, scope: !827)
!830 = !DILocation(line: 134, column: 74, scope: !827)
!831 = !DILocation(line: 136, column: 61, scope: !518)
!832 = !DILocalVariable(name: "attrs_vec", scope: !518, file: !519, line: 136, type: !527)
!833 = !DILocation(line: 137, column: 42, scope: !834)
!834 = !DILexicalBlock(scope: !518, file: !519, line: 137, column: 5)
!835 = !DILocation(line: 136, column: 42, scope: !836)
!836 = !DILexicalBlock(scope: !825, file: !519, line: 136, column: 42)
!837 = !DILocation(line: 68, column: 26, scope: !836)
!838 = !DILocation(line: 56, column: 31, scope: !836)
!839 = !DILocation(line: 136, column: 61, scope: !836)
!840 = !DILocation(line: 137, column: 90, scope: !518)
!841 = !DILocalVariable(name: "attrs_vec_1", scope: !518, file: !519, line: 137, type: !527)
!842 = !DILocation(line: 138, column: 25, scope: !518)
!843 = !DILocation(line: 139, column: 42, scope: !844)
!844 = !DILexicalBlock(scope: !518, file: !519, line: 139, column: 5)
!845 = !DILocation(line: 137, column: 42, scope: !846)
!846 = !DILexicalBlock(scope: !834, file: !519, line: 137, column: 42)
!847 = !DILocation(line: 68, column: 26, scope: !846)
!848 = !DILocation(line: 56, column: 31, scope: !846)
!849 = !DILocation(line: 137, column: 90, scope: !846)
!850 = !DILocation(line: 139, column: 90, scope: !518)
!851 = !DILocalVariable(name: "attrs_vec_2", scope: !518, file: !519, line: 139, type: !527)
!852 = !DILocation(line: 140, column: 25, scope: !518)
!853 = !DILocation(line: 142, column: 43, scope: !854)
!854 = !DILexicalBlock(scope: !518, file: !519, line: 142, column: 5)
!855 = !DILocation(line: 139, column: 42, scope: !856)
!856 = !DILexicalBlock(scope: !844, file: !519, line: 139, column: 42)
!857 = !DILocation(line: 68, column: 26, scope: !856)
!858 = !DILocation(line: 56, column: 31, scope: !856)
!859 = !DILocation(line: 139, column: 90, scope: !856)
!860 = !DILocation(line: 142, column: 75, scope: !518)
!861 = !DILocalVariable(name: "attrs_clause", scope: !518, file: !519, line: 142, type: !527)
!862 = !DILocation(line: 145, column: 39, scope: !863)
!863 = !DILexicalBlock(scope: !518, file: !519, line: 145, column: 5)
!864 = !DILocation(line: 142, column: 43, scope: !865)
!865 = !DILexicalBlock(scope: !854, file: !519, line: 142, column: 43)
!866 = !DILocation(line: 68, column: 26, scope: !865)
!867 = !DILocation(line: 56, column: 31, scope: !865)
!868 = !DILocation(line: 142, column: 75, scope: !865)
!869 = !DILocation(line: 145, column: 58, scope: !518)
!870 = !DILocalVariable(name: "op_vec", scope: !518, file: !519, line: 145, type: !527)
!871 = !DILocation(line: 146, column: 39, scope: !872)
!872 = !DILexicalBlock(scope: !518, file: !519, line: 146, column: 5)
!873 = !DILocation(line: 145, column: 39, scope: !874)
!874 = !DILexicalBlock(scope: !863, file: !519, line: 145, column: 39)
!875 = !DILocation(line: 68, column: 26, scope: !874)
!876 = !DILocation(line: 56, column: 31, scope: !874)
!877 = !DILocation(line: 145, column: 58, scope: !874)
!878 = !DILocation(line: 146, column: 83, scope: !518)
!879 = !DILocalVariable(name: "op_vec_1", scope: !518, file: !519, line: 146, type: !527)
!880 = !DILocation(line: 147, column: 25, scope: !518)
!881 = !DILocation(line: 148, column: 39, scope: !882)
!882 = !DILexicalBlock(scope: !518, file: !519, line: 148, column: 5)
!883 = !DILocation(line: 146, column: 39, scope: !884)
!884 = !DILexicalBlock(scope: !872, file: !519, line: 146, column: 39)
!885 = !DILocation(line: 68, column: 26, scope: !884)
!886 = !DILocation(line: 56, column: 31, scope: !884)
!887 = !DILocation(line: 146, column: 83, scope: !884)
!888 = !DILocation(line: 148, column: 81, scope: !518)
!889 = !DILocalVariable(name: "op_vec_2", scope: !518, file: !519, line: 148, type: !527)
!890 = !DILocation(line: 149, column: 25, scope: !518)
!891 = !DILocation(line: 150, column: 39, scope: !892)
!892 = !DILexicalBlock(scope: !518, file: !519, line: 150, column: 5)
!893 = !DILocation(line: 148, column: 39, scope: !894)
!894 = !DILexicalBlock(scope: !882, file: !519, line: 148, column: 39)
!895 = !DILocation(line: 68, column: 26, scope: !894)
!896 = !DILocation(line: 56, column: 31, scope: !894)
!897 = !DILocation(line: 148, column: 81, scope: !894)
!898 = !DILocation(line: 150, column: 85, scope: !518)
!899 = !DILocalVariable(name: "op_vec_3", scope: !518, file: !519, line: 150, type: !527)
!900 = !DILocation(line: 151, column: 25, scope: !518)
!901 = !DILocation(line: 152, column: 39, scope: !902)
!902 = !DILexicalBlock(scope: !518, file: !519, line: 152, column: 5)
!903 = !DILocation(line: 150, column: 39, scope: !904)
!904 = !DILexicalBlock(scope: !892, file: !519, line: 150, column: 39)
!905 = !DILocation(line: 68, column: 26, scope: !904)
!906 = !DILocation(line: 56, column: 31, scope: !904)
!907 = !DILocation(line: 150, column: 85, scope: !904)
!908 = !DILocation(line: 152, column: 82, scope: !518)
!909 = !DILocalVariable(name: "op_vec_4", scope: !518, file: !519, line: 152, type: !527)
!910 = !DILocation(line: 153, column: 25, scope: !518)
!911 = !DILocation(line: 154, column: 39, scope: !912)
!912 = !DILexicalBlock(scope: !518, file: !519, line: 154, column: 5)
!913 = !DILocation(line: 152, column: 39, scope: !914)
!914 = !DILexicalBlock(scope: !902, file: !519, line: 152, column: 39)
!915 = !DILocation(line: 68, column: 26, scope: !914)
!916 = !DILocation(line: 56, column: 31, scope: !914)
!917 = !DILocation(line: 152, column: 82, scope: !914)
!918 = !DILocation(line: 154, column: 82, scope: !518)
!919 = !DILocalVariable(name: "op_vec_5", scope: !518, file: !519, line: 154, type: !527)
!920 = !DILocation(line: 155, column: 25, scope: !518)
!921 = !DILocation(line: 157, column: 40, scope: !922)
!922 = !DILexicalBlock(scope: !518, file: !519, line: 157, column: 5)
!923 = !DILocation(line: 154, column: 39, scope: !924)
!924 = !DILexicalBlock(scope: !912, file: !519, line: 154, column: 39)
!925 = !DILocation(line: 68, column: 26, scope: !924)
!926 = !DILocation(line: 56, column: 31, scope: !924)
!927 = !DILocation(line: 154, column: 82, scope: !924)
!928 = !DILocation(line: 157, column: 69, scope: !518)
!929 = !DILocalVariable(name: "operation", scope: !518, file: !519, line: 157, type: !527)
!930 = !DILocation(line: 68, column: 26, scope: !518)
!931 = !DILocation(line: 56, column: 31, scope: !518)
!932 = !DILocation(line: 159, column: 5, scope: !518)
!933 = !DILocation(line: 157, column: 40, scope: !934)
!934 = !DILexicalBlock(scope: !922, file: !519, line: 157, column: 40)
!935 = !DILocation(line: 68, column: 26, scope: !934)
!936 = !DILocation(line: 56, column: 31, scope: !934)
!937 = !DILocation(line: 157, column: 69, scope: !934)
!938 = distinct !DISubprogram(name: "returnError", linkageName: "builtin.returnError", scope: !4, file: !4, line: 1105, type: !939, scopeLine: 1105, flags: DIFlagStaticMember, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !36)
!939 = !DISubroutineType(types: !940)
!940 = !{!522, !531}
!941 = !DILocation(line: 1108, column: 35, scope: !938)
!942 = !DILocalVariable(name: "st", scope: !938, file: !4, line: 1108, type: !943)
!943 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*builtin.StackTrace", baseType: !944, size: 64, align: 64)
!944 = !DICompositeType(tag: DW_TAG_structure_type, name: "builtin.StackTrace", scope: !36, size: 192, align: 64, elements: !945)
!945 = !{!946, !947}
!946 = !DIDerivedType(tag: DW_TAG_member, name: "index", scope: !944, baseType: !53, size: 64, align: 64)
!947 = !DIDerivedType(tag: DW_TAG_member, name: "instruction_addresses", scope: !944, baseType: !948, size: 128, align: 64, offset: 64)
!948 = !DICompositeType(tag: DW_TAG_structure_type, name: "[]usize", scope: !36, size: 128, align: 64, elements: !949)
!949 = !{!950, !951}
!950 = !DIDerivedType(tag: DW_TAG_member, name: "ptr", scope: !948, baseType: !531, size: 64, align: 64)
!951 = !DIDerivedType(tag: DW_TAG_member, name: "len", scope: !948, baseType: !53, size: 64, align: 64, offset: 64)
!952 = !DILocation(line: 1109, column: 11, scope: !953)
!953 = !DILexicalBlock(scope: !938, file: !4, line: 1109, column: 9)
!954 = !DILocation(line: 1109, column: 22, scope: !953)
!955 = !DILocation(line: 1109, column: 44, scope: !953)
!956 = !DILocation(line: 1111, column: 7, scope: !938)
!957 = !DILocation(line: 1111, column: 14, scope: !938)
!958 = !DILocation(line: 1110, column: 11, scope: !959)
!959 = !DILexicalBlock(scope: !953, file: !4, line: 1109, column: 44)
!960 = !DILocation(line: 1110, column: 36, scope: !959)
!961 = !DILocation(line: 1110, column: 33, scope: !959)
!962 = !DILocation(line: 1110, column: 33, scope: !963)
!963 = !DILexicalBlock(scope: !953, file: !4, line: 1110, column: 33)
!964 = distinct !DISubprogram(name: "exampleTransformCallToOperation", linkageName: "c_api_transform.exampleTransformCallToOperation", scope: !519, file: !519, line: 171, type: !965, scopeLine: 171, flags: DIFlagStaticMember, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !36)
!965 = !DISubroutineType(types: !966)
!966 = !{!527}
!967 = !DILocation(line: 173, column: 41, scope: !968)
!968 = !DILexicalBlock(scope: !964, file: !519, line: 173, column: 5)
!969 = !DILocation(line: 173, column: 51, scope: !964)
!970 = !DILocalVariable(name: "allocator", scope: !964, file: !519, line: 173, type: !527)
!971 = !DILocation(line: 176, column: 47, scope: !972)
!972 = !DILexicalBlock(scope: !964, file: !519, line: 176, column: 5)
!973 = !DILocation(line: 173, column: 41, scope: !974)
!974 = !DILexicalBlock(scope: !968, file: !519, line: 173, column: 41)
!975 = !DILocation(line: 173, column: 51, scope: !974)
!976 = !DILocation(line: 176, column: 74, scope: !964)
!977 = !DILocalVariable(name: "call_ident", scope: !964, file: !519, line: 176, type: !527)
!978 = !DILocation(line: 177, column: 44, scope: !979)
!979 = !DILexicalBlock(scope: !964, file: !519, line: 177, column: 5)
!980 = !DILocation(line: 176, column: 47, scope: !981)
!981 = !DILexicalBlock(scope: !972, file: !519, line: 176, column: 47)
!982 = !DILocation(line: 176, column: 74, scope: !981)
!983 = !DILocation(line: 177, column: 72, scope: !964)
!984 = !DILocalVariable(name: "test_symbol", scope: !964, file: !519, line: 177, type: !527)
!985 = !DILocation(line: 178, column: 46, scope: !986)
!986 = !DILexicalBlock(scope: !964, file: !519, line: 178, column: 5)
!987 = !DILocation(line: 177, column: 44, scope: !988)
!988 = !DILexicalBlock(scope: !979, file: !519, line: 177, column: 44)
!989 = !DILocation(line: 177, column: 72, scope: !988)
!990 = !DILocation(line: 178, column: 72, scope: !964)
!991 = !DILocalVariable(name: "i64_ident", scope: !964, file: !519, line: 178, type: !527)
!992 = !DILocation(line: 180, column: 42, scope: !993)
!993 = !DILexicalBlock(scope: !964, file: !519, line: 180, column: 5)
!994 = !DILocation(line: 178, column: 46, scope: !995)
!995 = !DILexicalBlock(scope: !986, file: !519, line: 178, column: 46)
!996 = !DILocation(line: 178, column: 72, scope: !995)
!997 = !DILocation(line: 180, column: 61, scope: !964)
!998 = !DILocalVariable(name: "input_vec", scope: !964, file: !519, line: 180, type: !527)
!999 = !DILocation(line: 181, column: 42, scope: !1000)
!1000 = !DILexicalBlock(scope: !964, file: !519, line: 181, column: 5)
!1001 = !DILocation(line: 180, column: 42, scope: !1002)
!1002 = !DILexicalBlock(scope: !993, file: !519, line: 180, column: 42)
!1003 = !DILocation(line: 180, column: 61, scope: !1002)
!1004 = !DILocation(line: 181, column: 84, scope: !964)
!1005 = !DILocalVariable(name: "input_vec_1", scope: !964, file: !519, line: 181, type: !527)
!1006 = !DILocation(line: 182, column: 25, scope: !964)
!1007 = !DILocation(line: 183, column: 42, scope: !1008)
!1008 = !DILexicalBlock(scope: !964, file: !519, line: 183, column: 5)
!1009 = !DILocation(line: 181, column: 42, scope: !1010)
!1010 = !DILexicalBlock(scope: !1000, file: !519, line: 181, column: 42)
!1011 = !DILocation(line: 181, column: 84, scope: !1010)
!1012 = !DILocation(line: 183, column: 87, scope: !964)
!1013 = !DILocalVariable(name: "input_vec_2", scope: !964, file: !519, line: 183, type: !527)
!1014 = !DILocation(line: 184, column: 25, scope: !964)
!1015 = !DILocation(line: 185, column: 42, scope: !1016)
!1016 = !DILexicalBlock(scope: !964, file: !519, line: 185, column: 5)
!1017 = !DILocation(line: 183, column: 42, scope: !1018)
!1018 = !DILexicalBlock(scope: !1008, file: !519, line: 183, column: 42)
!1019 = !DILocation(line: 183, column: 87, scope: !1018)
!1020 = !DILocation(line: 185, column: 85, scope: !964)
!1021 = !DILocalVariable(name: "input_vec_3", scope: !964, file: !519, line: 185, type: !527)
!1022 = !DILocation(line: 186, column: 25, scope: !964)
!1023 = !DILocation(line: 188, column: 40, scope: !1024)
!1024 = !DILexicalBlock(scope: !964, file: !519, line: 188, column: 5)
!1025 = !DILocation(line: 185, column: 42, scope: !1026)
!1026 = !DILexicalBlock(scope: !1016, file: !519, line: 185, column: 42)
!1027 = !DILocation(line: 185, column: 85, scope: !1026)
!1028 = !DILocation(line: 188, column: 72, scope: !964)
!1029 = !DILocalVariable(name: "call_expr", scope: !964, file: !519, line: 188, type: !527)
!1030 = !DILocation(line: 191, column: 47, scope: !1031)
!1031 = !DILexicalBlock(scope: !964, file: !519, line: 191, column: 5)
!1032 = !DILocation(line: 188, column: 40, scope: !1033)
!1033 = !DILexicalBlock(scope: !1024, file: !519, line: 188, column: 40)
!1034 = !DILocation(line: 188, column: 72, scope: !1033)
!1035 = !DILocation(line: 191, column: 76, scope: !964)
!1036 = !DILocalVariable(name: "operation", scope: !964, file: !519, line: 191, type: !527)
!1037 = !DILocation(line: 194, column: 18, scope: !964)
!1038 = !DILocation(line: 198, column: 5, scope: !964)
!1039 = !DILocation(line: 191, column: 47, scope: !1040)
!1040 = !DILexicalBlock(scope: !1031, file: !519, line: 191, column: 47)
!1041 = !DILocation(line: 191, column: 76, scope: !1040)
