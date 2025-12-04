; ModuleID = '/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/iongraph-rust-redux/examples/llvm/sum_array.c'
source_filename = "/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/iongraph-rust-redux/examples/llvm/sum_array.c"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx16.0.0"

; Function Attrs: nofree norecurse nosync nounwind ssp memory(argmem: read) uwtable(sync)
define i32 @sum_array(ptr nocapture noundef readonly %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = icmp sgt i32 %1, 0
  br i1 %3, label %4, label %59

4:                                                ; preds = %2
  %5 = zext nneg i32 %1 to i64
  %6 = icmp ult i32 %1, 4
  br i1 %6, label %7, label %10

7:                                                ; preds = %40, %56, %4
  %8 = phi i64 [ %13, %40 ], [ 0, %4 ], [ %46, %56 ]
  %9 = phi i32 [ %38, %40 ], [ 0, %4 ], [ %57, %56 ]
  br label %61

10:                                               ; preds = %4
  %11 = icmp ult i32 %1, 16
  br i1 %11, label %43, label %12

12:                                               ; preds = %10
  %13 = and i64 %5, 2147483632
  br label %14

14:                                               ; preds = %14, %12
  %15 = phi i64 [ 0, %12 ], [ %32, %14 ]
  %16 = phi <4 x i32> [ zeroinitializer, %12 ], [ %28, %14 ]
  %17 = phi <4 x i32> [ zeroinitializer, %12 ], [ %29, %14 ]
  %18 = phi <4 x i32> [ zeroinitializer, %12 ], [ %30, %14 ]
  %19 = phi <4 x i32> [ zeroinitializer, %12 ], [ %31, %14 ]
  %20 = getelementptr inbounds nuw i32, ptr %0, i64 %15
  %21 = getelementptr inbounds nuw i8, ptr %20, i64 16
  %22 = getelementptr inbounds nuw i8, ptr %20, i64 32
  %23 = getelementptr inbounds nuw i8, ptr %20, i64 48
  %24 = load <4 x i32>, ptr %20, align 4, !tbaa !6
  %25 = load <4 x i32>, ptr %21, align 4, !tbaa !6
  %26 = load <4 x i32>, ptr %22, align 4, !tbaa !6
  %27 = load <4 x i32>, ptr %23, align 4, !tbaa !6
  %28 = add <4 x i32> %24, %16
  %29 = add <4 x i32> %25, %17
  %30 = add <4 x i32> %26, %18
  %31 = add <4 x i32> %27, %19
  %32 = add nuw i64 %15, 16
  %33 = icmp eq i64 %32, %13
  br i1 %33, label %34, label %14, !llvm.loop !10

34:                                               ; preds = %14
  %35 = add <4 x i32> %29, %28
  %36 = add <4 x i32> %30, %35
  %37 = add <4 x i32> %31, %36
  %38 = tail call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %37)
  %39 = icmp eq i64 %13, %5
  br i1 %39, label %59, label %40

40:                                               ; preds = %34
  %41 = and i64 %5, 12
  %42 = icmp eq i64 %41, 0
  br i1 %42, label %7, label %43

43:                                               ; preds = %40, %10
  %44 = phi i64 [ %13, %40 ], [ 0, %10 ]
  %45 = phi i32 [ %38, %40 ], [ 0, %10 ]
  %46 = and i64 %5, 2147483644
  %47 = insertelement <4 x i32> <i32 poison, i32 0, i32 0, i32 0>, i32 %45, i64 0
  br label %48

48:                                               ; preds = %48, %43
  %49 = phi i64 [ %44, %43 ], [ %54, %48 ]
  %50 = phi <4 x i32> [ %47, %43 ], [ %53, %48 ]
  %51 = getelementptr inbounds nuw i32, ptr %0, i64 %49
  %52 = load <4 x i32>, ptr %51, align 4, !tbaa !6
  %53 = add <4 x i32> %52, %50
  %54 = add nuw i64 %49, 4
  %55 = icmp eq i64 %54, %46
  br i1 %55, label %56, label %48, !llvm.loop !14

56:                                               ; preds = %48
  %57 = tail call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %53)
  %58 = icmp eq i64 %46, %5
  br i1 %58, label %59, label %7

59:                                               ; preds = %61, %34, %56, %2
  %60 = phi i32 [ 0, %2 ], [ %38, %34 ], [ %57, %56 ], [ %66, %61 ]
  ret i32 %60

61:                                               ; preds = %7, %61
  %62 = phi i64 [ %67, %61 ], [ %8, %7 ]
  %63 = phi i32 [ %66, %61 ], [ %9, %7 ]
  %64 = getelementptr inbounds nuw i32, ptr %0, i64 %62
  %65 = load i32, ptr %64, align 4, !tbaa !6
  %66 = add nsw i32 %65, %63
  %67 = add nuw nsw i64 %62, 1
  %68 = icmp eq i64 %67, %5
  br i1 %68, label %59, label %61, !llvm.loop !15
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.vector.reduce.add.v4i32(<4 x i32>) #1

attributes #0 = { nofree norecurse nosync nounwind ssp memory(argmem: read) uwtable(sync) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+ccpp,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a,+zcm,+zcz" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 0]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 1}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"Homebrew clang version 20.1.7"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = distinct !{!10, !11, !12, !13}
!11 = !{!"llvm.loop.mustprogress"}
!12 = !{!"llvm.loop.isvectorized", i32 1}
!13 = !{!"llvm.loop.unroll.runtime.disable"}
!14 = distinct !{!14, !11, !12, !13}
!15 = distinct !{!15, !11, !13, !12}
