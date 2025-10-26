; ModuleID = 'BitcodeBuffer'
source_filename = "c_api_transform"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-apple-macosx26.0.1-unknown"

@__anon_1817 = internal unnamed_addr constant [10 x i8] c"operation\00", align 1
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

; Function Attrs: minsize nounwind optsize uwtable
declare ptr @value_get_list(ptr align 1, ptr readonly align 1) local_unnamed_addr #0

; Function Attrs: minsize nounwind optsize uwtable
declare i64 @vector_value_len(ptr readonly align 1) local_unnamed_addr #0

; Function Attrs: minsize nounwind optsize uwtable
declare void @vector_value_destroy(ptr align 1, ptr align 1) local_unnamed_addr #0

; Function Attrs: minsize nounwind optsize uwtable
declare ptr @vector_value_at(ptr readonly align 1, i64) local_unnamed_addr #0

; Function Attrs: minsize nounwind optsize uwtable
declare ptr @value_get_atom(ptr align 1, ptr readonly align 1) local_unnamed_addr #0

; Function Attrs: minsize nounwind optsize uwtable
declare ptr @value_create_identifier(ptr align 1, ptr nonnull readonly align 1) local_unnamed_addr #0

; Function Attrs: minsize nounwind optsize uwtable
declare void @value_free_atom(ptr align 1, ptr readonly align 1) local_unnamed_addr #0

; Function Attrs: minsize nounwind optsize uwtable
declare ptr @vector_value_create(ptr align 1) local_unnamed_addr #0

; Function Attrs: minsize nounwind optsize uwtable
declare ptr @vector_value_push(ptr align 1, ptr align 1, ptr align 1) local_unnamed_addr #0

; Function Attrs: minsize nounwind optsize uwtable
declare ptr @value_create_list(ptr align 1, ptr align 1) local_unnamed_addr #0

; Function Attrs: minsize nounwind optsize uwtable
declare ptr @value_create_type_expr(ptr align 1, ptr align 1) local_unnamed_addr #0

; Function Attrs: minsize nounwind optsize uwtable
declare ptr @value_create_keyword(ptr align 1, ptr nonnull readonly align 1) local_unnamed_addr #0

; Function Attrs: minsize nounwind optsize uwtable
declare ptr @value_create_map(ptr align 1, ptr align 1) local_unnamed_addr #0

; Function Attrs: minsize nounwind optsize uwtable
define dso_local ptr @exampleTransformCallToOperation() local_unnamed_addr #0 {
  %1 = tail call ptr @allocator_create_c()
  %.not = icmp eq ptr %1, null
  br i1 %.not, label %common.ret, label %2

2:                                                ; preds = %0
  %3 = tail call ptr @value_create_identifier(ptr nonnull align 1 %1, ptr nonnull readonly align 1 @__anon_1755)
  %.not9 = icmp eq ptr %3, null
  br i1 %.not9, label %common.ret, label %4

common.ret:                                       ; preds = %141, %18, %143, %140, %137, %134, %131, %128, %125, %122, %119, %116, %113, %110, %107, %104, %101, %98, %95, %92, %89, %86, %83, %80, %77, %74, %71, %68, %65, %62, %59, %56, %53, %50, %47, %44, %41, %38, %35, %32, %23, %16, %14, %12, %10, %8, %6, %4, %2, %0, %c_api_transform.transformCallToOperation.exit
  %common.ret.op = phi ptr [ %142, %c_api_transform.transformCallToOperation.exit ], [ null, %0 ], [ null, %2 ], [ null, %4 ], [ null, %6 ], [ null, %8 ], [ null, %10 ], [ null, %12 ], [ null, %14 ], [ null, %16 ], [ null, %23 ], [ null, %32 ], [ null, %35 ], [ null, %38 ], [ null, %41 ], [ null, %44 ], [ null, %47 ], [ null, %50 ], [ null, %53 ], [ null, %56 ], [ null, %59 ], [ null, %62 ], [ null, %65 ], [ null, %68 ], [ null, %71 ], [ null, %74 ], [ null, %77 ], [ null, %80 ], [ null, %83 ], [ null, %86 ], [ null, %89 ], [ null, %92 ], [ null, %95 ], [ null, %98 ], [ null, %101 ], [ null, %104 ], [ null, %107 ], [ null, %110 ], [ null, %113 ], [ null, %116 ], [ null, %119 ], [ null, %122 ], [ null, %125 ], [ null, %128 ], [ null, %131 ], [ null, %134 ], [ null, %137 ], [ null, %140 ], [ null, %143 ], [ null, %18 ], [ null, %141 ]
  ret ptr %common.ret.op

4:                                                ; preds = %2
  %5 = tail call ptr @value_create_symbol(ptr nonnull align 1 %1, ptr nonnull readonly align 1 @__anon_1761)
  %.not10 = icmp eq ptr %5, null
  br i1 %.not10, label %common.ret, label %6

6:                                                ; preds = %4
  %7 = tail call ptr @value_create_identifier(ptr nonnull align 1 %1, ptr nonnull readonly align 1 @__anon_1766)
  %.not11 = icmp eq ptr %7, null
  br i1 %.not11, label %common.ret, label %8

8:                                                ; preds = %6
  %9 = tail call ptr @vector_value_create(ptr nonnull align 1 %1)
  %.not12 = icmp eq ptr %9, null
  br i1 %.not12, label %common.ret, label %10

10:                                               ; preds = %8
  %11 = tail call ptr @vector_value_push(ptr nonnull align 1 %1, ptr nonnull align 1 %9, ptr nonnull align 1 %3)
  %.not13 = icmp eq ptr %11, null
  br i1 %.not13, label %common.ret, label %12

12:                                               ; preds = %10
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %9)
  %13 = tail call ptr @vector_value_push(ptr nonnull align 1 %1, ptr nonnull align 1 %11, ptr nonnull align 1 %5)
  %.not14 = icmp eq ptr %13, null
  br i1 %.not14, label %common.ret, label %14

14:                                               ; preds = %12
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %11)
  %15 = tail call ptr @vector_value_push(ptr nonnull align 1 %1, ptr nonnull align 1 %13, ptr nonnull align 1 %7)
  %.not15 = icmp eq ptr %15, null
  br i1 %.not15, label %common.ret, label %16

16:                                               ; preds = %14
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %13)
  %17 = tail call ptr @value_create_list(ptr nonnull align 1 %1, ptr nonnull align 1 %15)
  %.not16 = icmp eq ptr %17, null
  br i1 %.not16, label %common.ret, label %18

18:                                               ; preds = %16
  %19 = tail call ptr @value_get_list(ptr nonnull align 1 %1, ptr nonnull readonly align 1 %17), !noalias !1
  %.not.i = icmp eq ptr %19, null
  br i1 %.not.i, label %common.ret, label %20

20:                                               ; preds = %18
  %21 = tail call i64 @vector_value_len(ptr nonnull readonly align 1 %19), !noalias !1
  %22 = icmp ult i64 %21, 3
  br i1 %22, label %23, label %24

23:                                               ; preds = %20
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

24:                                               ; preds = %20
  %25 = tail call ptr @vector_value_at(ptr nonnull readonly align 1 %19, i64 0), !noalias !1
  %26 = tail call ptr @vector_value_at(ptr nonnull readonly align 1 %19, i64 1), !noalias !1
  %27 = tail call ptr @vector_value_at(ptr nonnull readonly align 1 %19, i64 2), !noalias !1
  %28 = tail call ptr @value_get_atom(ptr nonnull align 1 %1, ptr readonly align 1 %25), !noalias !1
  %29 = tail call ptr @value_create_identifier(ptr nonnull align 1 %1, ptr nonnull readonly align 1 @__anon_1817), !noalias !1
  %.not42.i = icmp eq ptr %29, null
  br i1 %.not42.i, label %32, label %30

30:                                               ; preds = %24
  %31 = tail call ptr @value_create_identifier(ptr nonnull align 1 %1, ptr nonnull readonly align 1 @__anon_1828), !noalias !1
  %.not43.i = icmp eq ptr %31, null
  br i1 %.not43.i, label %35, label %33

32:                                               ; preds = %24
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

33:                                               ; preds = %30
  %34 = tail call ptr @value_create_identifier(ptr nonnull align 1 %1, ptr nonnull readonly align 1 @__anon_1831), !noalias !1
  %.not44.i = icmp eq ptr %34, null
  br i1 %.not44.i, label %38, label %36

35:                                               ; preds = %30
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

36:                                               ; preds = %33
  %37 = tail call ptr @vector_value_create(ptr nonnull align 1 %1), !noalias !1
  %.not45.i = icmp eq ptr %37, null
  br i1 %.not45.i, label %41, label %39

38:                                               ; preds = %33
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

39:                                               ; preds = %36
  %40 = tail call ptr @vector_value_push(ptr nonnull align 1 %1, ptr nonnull align 1 %37, ptr nonnull align 1 %31), !noalias !1
  %.not46.i = icmp eq ptr %40, null
  br i1 %.not46.i, label %44, label %42

41:                                               ; preds = %36
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

42:                                               ; preds = %39
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %37), !noalias !1
  %43 = tail call ptr @vector_value_push(ptr nonnull align 1 %1, ptr nonnull align 1 %40, ptr nonnull align 1 %34), !noalias !1
  %.not47.i = icmp eq ptr %43, null
  br i1 %.not47.i, label %47, label %45

44:                                               ; preds = %39
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

45:                                               ; preds = %42
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %40), !noalias !1
  %46 = tail call ptr @value_create_list(ptr nonnull align 1 %1, ptr nonnull align 1 %43), !noalias !1
  %.not48.i = icmp eq ptr %46, null
  br i1 %.not48.i, label %50, label %48

47:                                               ; preds = %42
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

48:                                               ; preds = %45
  %49 = tail call ptr @value_create_identifier(ptr nonnull align 1 %1, ptr nonnull readonly align 1 @__anon_1835), !noalias !1
  %.not49.i = icmp eq ptr %49, null
  br i1 %.not49.i, label %53, label %51

50:                                               ; preds = %45
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

51:                                               ; preds = %48
  %52 = tail call ptr @value_create_identifier(ptr nonnull align 1 %1, ptr nonnull readonly align 1 @__anon_1840), !noalias !1
  %.not50.i = icmp eq ptr %52, null
  br i1 %.not50.i, label %56, label %54

53:                                               ; preds = %48
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

54:                                               ; preds = %51
  %55 = tail call ptr @vector_value_create(ptr nonnull align 1 %1), !noalias !1
  %.not51.i = icmp eq ptr %55, null
  br i1 %.not51.i, label %59, label %57

56:                                               ; preds = %51
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

57:                                               ; preds = %54
  %58 = tail call ptr @vector_value_push(ptr nonnull align 1 %1, ptr nonnull align 1 %55, ptr nonnull align 1 %52), !noalias !1
  %.not52.i = icmp eq ptr %58, null
  br i1 %.not52.i, label %62, label %60

59:                                               ; preds = %54
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

60:                                               ; preds = %57
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %55), !noalias !1
  %61 = tail call ptr @value_create_list(ptr nonnull align 1 %1, ptr nonnull align 1 %58), !noalias !1
  %.not53.i = icmp eq ptr %61, null
  br i1 %.not53.i, label %65, label %63

62:                                               ; preds = %57
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

63:                                               ; preds = %60
  %64 = tail call ptr @vector_value_create(ptr nonnull align 1 %1), !noalias !1
  %.not54.i = icmp eq ptr %64, null
  br i1 %.not54.i, label %68, label %66

65:                                               ; preds = %60
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

66:                                               ; preds = %63
  %67 = tail call ptr @vector_value_push(ptr nonnull align 1 %1, ptr nonnull align 1 %64, ptr nonnull align 1 %49), !noalias !1
  %.not55.i = icmp eq ptr %67, null
  br i1 %.not55.i, label %71, label %69

68:                                               ; preds = %63
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

69:                                               ; preds = %66
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %64), !noalias !1
  %70 = tail call ptr @vector_value_push(ptr nonnull align 1 %1, ptr nonnull align 1 %67, ptr nonnull align 1 %61), !noalias !1
  %.not56.i = icmp eq ptr %70, null
  br i1 %.not56.i, label %74, label %72

71:                                               ; preds = %66
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

72:                                               ; preds = %69
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %67), !noalias !1
  %73 = tail call ptr @value_create_list(ptr nonnull align 1 %1, ptr nonnull align 1 %70), !noalias !1
  %.not57.i = icmp eq ptr %73, null
  br i1 %.not57.i, label %77, label %75

74:                                               ; preds = %69
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

75:                                               ; preds = %72
  %76 = tail call ptr @value_create_identifier(ptr nonnull align 1 %1, ptr nonnull readonly align 1 @__anon_1845), !noalias !1
  %.not58.i = icmp eq ptr %76, null
  br i1 %.not58.i, label %80, label %78

77:                                               ; preds = %72
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

78:                                               ; preds = %75
  %79 = tail call ptr @value_create_type_expr(ptr nonnull align 1 %1, ptr align 1 %27), !noalias !1
  %.not59.i = icmp eq ptr %79, null
  br i1 %.not59.i, label %83, label %81

80:                                               ; preds = %75
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

81:                                               ; preds = %78
  %82 = tail call ptr @vector_value_create(ptr nonnull align 1 %1), !noalias !1
  %.not60.i = icmp eq ptr %82, null
  br i1 %.not60.i, label %86, label %84

83:                                               ; preds = %78
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

84:                                               ; preds = %81
  %85 = tail call ptr @vector_value_push(ptr nonnull align 1 %1, ptr nonnull align 1 %82, ptr nonnull align 1 %76), !noalias !1
  %.not61.i = icmp eq ptr %85, null
  br i1 %.not61.i, label %89, label %87

86:                                               ; preds = %81
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

87:                                               ; preds = %84
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %82), !noalias !1
  %88 = tail call ptr @vector_value_push(ptr nonnull align 1 %1, ptr nonnull align 1 %85, ptr nonnull align 1 %79), !noalias !1
  %.not62.i = icmp eq ptr %88, null
  br i1 %.not62.i, label %92, label %90

89:                                               ; preds = %84
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

90:                                               ; preds = %87
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %85), !noalias !1
  %91 = tail call ptr @value_create_list(ptr nonnull align 1 %1, ptr nonnull align 1 %88), !noalias !1
  %.not63.i = icmp eq ptr %91, null
  br i1 %.not63.i, label %95, label %93

92:                                               ; preds = %87
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

93:                                               ; preds = %90
  %94 = tail call ptr @value_create_identifier(ptr nonnull align 1 %1, ptr nonnull readonly align 1 @__anon_1852), !noalias !1
  %.not64.i = icmp eq ptr %94, null
  br i1 %.not64.i, label %98, label %96

95:                                               ; preds = %90
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

96:                                               ; preds = %93
  %97 = tail call ptr @value_create_keyword(ptr nonnull align 1 %1, ptr nonnull readonly align 1 @__anon_1859), !noalias !1
  %.not65.i = icmp eq ptr %97, null
  br i1 %.not65.i, label %101, label %99

98:                                               ; preds = %93
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

99:                                               ; preds = %96
  %100 = tail call ptr @vector_value_create(ptr nonnull align 1 %1), !noalias !1
  %.not66.i = icmp eq ptr %100, null
  br i1 %.not66.i, label %104, label %102

101:                                              ; preds = %96
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

102:                                              ; preds = %99
  %103 = tail call ptr @vector_value_push(ptr nonnull align 1 %1, ptr nonnull align 1 %100, ptr nonnull align 1 %97), !noalias !1
  %.not67.i = icmp eq ptr %103, null
  br i1 %.not67.i, label %107, label %105

104:                                              ; preds = %99
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

105:                                              ; preds = %102
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %100), !noalias !1
  %106 = tail call ptr @vector_value_push(ptr nonnull align 1 %1, ptr nonnull align 1 %103, ptr align 1 %26), !noalias !1
  %.not68.i = icmp eq ptr %106, null
  br i1 %.not68.i, label %110, label %108

107:                                              ; preds = %102
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

108:                                              ; preds = %105
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %103), !noalias !1
  %109 = tail call ptr @value_create_map(ptr nonnull align 1 %1, ptr nonnull align 1 %106), !noalias !1
  %.not69.i = icmp eq ptr %109, null
  br i1 %.not69.i, label %113, label %111

110:                                              ; preds = %105
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

111:                                              ; preds = %108
  %112 = tail call ptr @vector_value_create(ptr nonnull align 1 %1), !noalias !1
  %.not70.i = icmp eq ptr %112, null
  br i1 %.not70.i, label %116, label %114

113:                                              ; preds = %108
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

114:                                              ; preds = %111
  %115 = tail call ptr @vector_value_push(ptr nonnull align 1 %1, ptr nonnull align 1 %112, ptr nonnull align 1 %94), !noalias !1
  %.not71.i = icmp eq ptr %115, null
  br i1 %.not71.i, label %119, label %117

116:                                              ; preds = %111
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

117:                                              ; preds = %114
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %112), !noalias !1
  %118 = tail call ptr @vector_value_push(ptr nonnull align 1 %1, ptr nonnull align 1 %115, ptr nonnull align 1 %109), !noalias !1
  %.not72.i = icmp eq ptr %118, null
  br i1 %.not72.i, label %122, label %120

119:                                              ; preds = %114
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

120:                                              ; preds = %117
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %115), !noalias !1
  %121 = tail call ptr @value_create_list(ptr nonnull align 1 %1, ptr nonnull align 1 %118), !noalias !1
  %.not73.i = icmp eq ptr %121, null
  br i1 %.not73.i, label %125, label %123

122:                                              ; preds = %117
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

123:                                              ; preds = %120
  %124 = tail call ptr @vector_value_create(ptr nonnull align 1 %1), !noalias !1
  %.not74.i = icmp eq ptr %124, null
  br i1 %.not74.i, label %128, label %126

125:                                              ; preds = %120
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

126:                                              ; preds = %123
  %127 = tail call ptr @vector_value_push(ptr nonnull align 1 %1, ptr nonnull align 1 %124, ptr nonnull align 1 %29), !noalias !1
  %.not75.i = icmp eq ptr %127, null
  br i1 %.not75.i, label %131, label %129

128:                                              ; preds = %123
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

129:                                              ; preds = %126
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %124), !noalias !1
  %130 = tail call ptr @vector_value_push(ptr nonnull align 1 %1, ptr nonnull align 1 %127, ptr nonnull align 1 %46), !noalias !1
  %.not76.i = icmp eq ptr %130, null
  br i1 %.not76.i, label %134, label %132

131:                                              ; preds = %126
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

132:                                              ; preds = %129
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %127), !noalias !1
  %133 = tail call ptr @vector_value_push(ptr nonnull align 1 %1, ptr nonnull align 1 %130, ptr nonnull align 1 %73), !noalias !1
  %.not77.i = icmp eq ptr %133, null
  br i1 %.not77.i, label %137, label %135

134:                                              ; preds = %129
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

135:                                              ; preds = %132
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %130), !noalias !1
  %136 = tail call ptr @vector_value_push(ptr nonnull align 1 %1, ptr nonnull align 1 %133, ptr nonnull align 1 %91), !noalias !1
  %.not78.i = icmp eq ptr %136, null
  br i1 %.not78.i, label %140, label %138

137:                                              ; preds = %132
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

138:                                              ; preds = %135
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %133), !noalias !1
  %139 = tail call ptr @vector_value_push(ptr nonnull align 1 %1, ptr nonnull align 1 %136, ptr nonnull align 1 %121), !noalias !1
  %.not79.i = icmp eq ptr %139, null
  br i1 %.not79.i, label %143, label %141

140:                                              ; preds = %135
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

141:                                              ; preds = %138
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %136), !noalias !1
  %142 = tail call ptr @value_create_list(ptr nonnull align 1 %1, ptr nonnull align 1 %139), !noalias !1
  %.not80.i = icmp eq ptr %142, null
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br i1 %.not80.i, label %common.ret, label %c_api_transform.transformCallToOperation.exit

143:                                              ; preds = %138
  tail call void @value_free_atom(ptr nonnull align 1 %1, ptr readonly align 1 %28), !noalias !1
  tail call void @vector_value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %19), !noalias !1
  br label %common.ret

c_api_transform.transformCallToOperation.exit:    ; preds = %141
  tail call void @value_destroy(ptr nonnull align 1 %1, ptr nonnull align 1 %17)
  br label %common.ret
}

; Function Attrs: minsize nounwind optsize uwtable
declare ptr @allocator_create_c() local_unnamed_addr #0

; Function Attrs: minsize nounwind optsize uwtable
declare ptr @value_create_symbol(ptr align 1, ptr nonnull readonly align 1) local_unnamed_addr #0

; Function Attrs: minsize nounwind optsize uwtable
declare void @value_destroy(ptr align 1, ptr align 1) local_unnamed_addr #0

; Function Attrs: minsize nounwind optsize uwtable
define internal void @OUTLINED_FUNCTION_0() unnamed_addr #1 {
entry:
  ret void
}

; Function Attrs: minsize nounwind optsize uwtable
define internal void @OUTLINED_FUNCTION_1() unnamed_addr #1 {
entry:
  ret void
}

; Function Attrs: minsize nounwind optsize uwtable
define internal void @OUTLINED_FUNCTION_2() unnamed_addr #1 {
entry:
  ret void
}

; Function Attrs: minsize nounwind optsize uwtable
define internal void @OUTLINED_FUNCTION_3() unnamed_addr #1 {
entry:
  ret void
}

; Function Attrs: minsize nounwind optsize uwtable
define internal void @OUTLINED_FUNCTION_4() unnamed_addr #1 {
entry:
  ret void
}

; Function Attrs: minsize nounwind optsize uwtable
define internal void @OUTLINED_FUNCTION_5() unnamed_addr #1 {
entry:
  ret void
}

; Function Attrs: minsize nounwind optsize uwtable
define internal void @OUTLINED_FUNCTION_6() unnamed_addr #1 {
entry:
  ret void
}

; Function Attrs: minsize nounwind optsize uwtable
define internal void @OUTLINED_FUNCTION_7() unnamed_addr #1 {
entry:
  ret void
}

; Function Attrs: minsize nounwind optsize uwtable
define internal void @OUTLINED_FUNCTION_8() unnamed_addr #1 {
entry:
  ret void
}

; Function Attrs: minsize nounwind optsize uwtable
define internal void @OUTLINED_FUNCTION_9() unnamed_addr #1 {
entry:
  ret void
}

; Function Attrs: minsize nounwind optsize uwtable
define internal void @OUTLINED_FUNCTION_10() unnamed_addr #1 {
entry:
  ret void
}

; Function Attrs: minsize nounwind optsize uwtable
define internal void @OUTLINED_FUNCTION_11() unnamed_addr #1 {
entry:
  ret void
}

; Function Attrs: minsize nounwind optsize uwtable
define internal void @OUTLINED_FUNCTION_12() unnamed_addr #1 {
entry:
  ret void
}

; Function Attrs: minsize nounwind optsize uwtable
define internal void @OUTLINED_FUNCTION_13() unnamed_addr #1 {
entry:
  ret void
}

; Function Attrs: minsize nounwind optsize uwtable
define internal void @OUTLINED_FUNCTION_14() unnamed_addr #1 {
entry:
  ret void
}

attributes #0 = { minsize nounwind optsize uwtable "frame-pointer"="none" "target-cpu"="apple-m2" "target-features"="+aes,+alternate-sextload-cvt-f32-pattern,+altnzcv,+am,+amvs,+arith-bcc-fusion,+arith-cbz-fusion,+bf16,+bti,+ccdp,+ccidx,+ccpp,+complxnum,+CONTEXTIDREL2,+crc,+disable-latency-sched-heuristic,+dit,+dotprod,+ecv,+el2vmsa,+el3,+fgt,+flagm,+fp16fml,+fp-armv8,+fpac,+fptoint,+fullfp16,+fuse-address,+fuse-adrp-add,+fuse-aes,+fuse-arith-logic,+fuse-crypto-eor,+fuse-csel,+fuse-literals,+i8mm,+jsconv,+lor,+lse,+lse2,+mpam,+neon,+nv,+pan,+pan-rwv,+pauth,+perfmon,+predres,+ras,+rcpc,+rcpc-immo,+rdm,+sb,+sel2,+sha2,+sha3,+specrestrict,+ssbs,+store-pair-suppress,+tlb-rmi,+tracev8.4,+uaops,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8.6a,+v8a,+vh,+zcm,+zcz,+zcz-gp,-addr-lsl-slow-14,-aggressive-fma,-alu-lsl-fast,-ascend-store-address,-avoid-ldapur,-balance-fp-ops,-brbe,-call-saved-x10,-call-saved-x11,-call-saved-x12,-call-saved-x13,-call-saved-x14,-call-saved-x15,-call-saved-x18,-call-saved-x8,-call-saved-x9,-chk,-clrbhb,-cmp-bcc-fusion,-cmpbr,-cpa,-crypto,-cssc,-d128,-disable-ldp,-disable-stp,-enable-select-opt,-ete,-exynos-cheap-as-move,-f32mm,-f64mm,-f8f16mm,-f8f32mm,-faminmax,-fix-cortex-a53-835769,-fmv,-force-32bit-jump-tables,-fp8,-fp8dot2,-fp8dot4,-fp8fma,-fprcvt,-fujitsu-monaka,-fuse-addsub-2reg-const1,-gcs,-harden-sls-blr,-harden-sls-nocomdat,-harden-sls-retbr,-hbc,-hcx,-ite,-ldp-aligned-only,-ls64,-lse128,-lsfe,-lsui,-lut,-mec,-mops,-mte,-nmi,-no-bti-at-return-twice,-no-neg-immediates,-no-sve-fp-ld1r,-no-zcz-fp,-occmo,-outline-atomics,-pauth-lr,-pcdphint,-pops,-predictable-select-expensive,-prfm-slc-target,-rand,-rasv2,-rcpc3,-reserve-lr-for-ra,-reserve-x1,-reserve-x10,-reserve-x11,-reserve-x12,-reserve-x13,-reserve-x14,-reserve-x15,-reserve-x18,-reserve-x2,-reserve-x20,-reserve-x21,-reserve-x22,-reserve-x23,-reserve-x24,-reserve-x25,-reserve-x26,-reserve-x27,-reserve-x28,-reserve-x3,-reserve-x4,-reserve-x5,-reserve-x6,-reserve-x7,-reserve-x9,-rme,-slow-misaligned-128store,-slow-paired-128,-slow-strqro-store,-sm4,-sme,-sme2,-sme2p1,-sme2p2,-sme-b16b16,-sme-f16f16,-sme-f64f64,-sme-f8f16,-sme-f8f32,-sme-fa64,-sme-i16i64,-sme-lutv2,-sme-mop4,-sme-tmop,-spe,-spe-eef,-specres2,-ssve-aes,-ssve-bitperm,-ssve-fp8dot2,-ssve-fp8dot4,-ssve-fp8fma,-stp-aligned-only,-strict-align,-sve,-sve2,-sve2-aes,-sve2-bitperm,-sve2-sha3,-sve2-sm4,-sve2p1,-sve2p2,-sve-aes,-sve-aes2,-sve-b16b16,-sve-bfscale,-sve-bitperm,-sve-f16f32mm,-tagged-globals,-the,-tlbiw,-tme,-tpidr-el1,-tpidr-el2,-tpidr-el3,-tpidrro-el0,-trbe,-use-experimental-zeroing-pseudos,-use-fixed-over-scalable-if-equal-cost,-use-postra-scheduler,-use-reciprocal-square-root,-v8.7a,-v8.8a,-v8.9a,-v8r,-v9.1a,-v9.2a,-v9.3a,-v9.4a,-v9.5a,-v9.6a,-v9a,-wfxt,-xs,-zcz-fp-workaround" }
attributes #1 = { minsize nounwind optsize uwtable "target-cpu"="apple-m2" "target-features"="+aes,+alternate-sextload-cvt-f32-pattern,+altnzcv,+am,+amvs,+arith-bcc-fusion,+arith-cbz-fusion,+bf16,+bti,+ccdp,+ccidx,+ccpp,+complxnum,+CONTEXTIDREL2,+crc,+disable-latency-sched-heuristic,+dit,+dotprod,+ecv,+el2vmsa,+el3,+fgt,+flagm,+fp16fml,+fp-armv8,+fpac,+fptoint,+fullfp16,+fuse-address,+fuse-adrp-add,+fuse-aes,+fuse-arith-logic,+fuse-crypto-eor,+fuse-csel,+fuse-literals,+i8mm,+jsconv,+lor,+lse,+lse2,+mpam,+neon,+nv,+pan,+pan-rwv,+pauth,+perfmon,+predres,+ras,+rcpc,+rcpc-immo,+rdm,+sb,+sel2,+sha2,+sha3,+specrestrict,+ssbs,+store-pair-suppress,+tlb-rmi,+tracev8.4,+uaops,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8.6a,+v8a,+vh,+zcm,+zcz,+zcz-gp,-addr-lsl-slow-14,-aggressive-fma,-alu-lsl-fast,-ascend-store-address,-avoid-ldapur,-balance-fp-ops,-brbe,-call-saved-x10,-call-saved-x11,-call-saved-x12,-call-saved-x13,-call-saved-x14,-call-saved-x15,-call-saved-x18,-call-saved-x8,-call-saved-x9,-chk,-clrbhb,-cmp-bcc-fusion,-cmpbr,-cpa,-crypto,-cssc,-d128,-disable-ldp,-disable-stp,-enable-select-opt,-ete,-exynos-cheap-as-move,-f32mm,-f64mm,-f8f16mm,-f8f32mm,-faminmax,-fix-cortex-a53-835769,-fmv,-force-32bit-jump-tables,-fp8,-fp8dot2,-fp8dot4,-fp8fma,-fprcvt,-fujitsu-monaka,-fuse-addsub-2reg-const1,-gcs,-harden-sls-blr,-harden-sls-nocomdat,-harden-sls-retbr,-hbc,-hcx,-ite,-ldp-aligned-only,-ls64,-lse128,-lsfe,-lsui,-lut,-mec,-mops,-mte,-nmi,-no-bti-at-return-twice,-no-neg-immediates,-no-sve-fp-ld1r,-no-zcz-fp,-occmo,-outline-atomics,-pauth-lr,-pcdphint,-pops,-predictable-select-expensive,-prfm-slc-target,-rand,-rasv2,-rcpc3,-reserve-lr-for-ra,-reserve-x1,-reserve-x10,-reserve-x11,-reserve-x12,-reserve-x13,-reserve-x14,-reserve-x15,-reserve-x18,-reserve-x2,-reserve-x20,-reserve-x21,-reserve-x22,-reserve-x23,-reserve-x24,-reserve-x25,-reserve-x26,-reserve-x27,-reserve-x28,-reserve-x3,-reserve-x4,-reserve-x5,-reserve-x6,-reserve-x7,-reserve-x9,-rme,-slow-misaligned-128store,-slow-paired-128,-slow-strqro-store,-sm4,-sme,-sme2,-sme2p1,-sme2p2,-sme-b16b16,-sme-f16f16,-sme-f64f64,-sme-f8f16,-sme-f8f32,-sme-fa64,-sme-i16i64,-sme-lutv2,-sme-mop4,-sme-tmop,-spe,-spe-eef,-specres2,-ssve-aes,-ssve-bitperm,-ssve-fp8dot2,-ssve-fp8dot4,-ssve-fp8fma,-stp-aligned-only,-strict-align,-sve,-sve2,-sve2-aes,-sve2-bitperm,-sve2-sha3,-sve2-sm4,-sve2p1,-sve2p2,-sve-aes,-sve-aes2,-sve-b16b16,-sve-bfscale,-sve-bitperm,-sve-f16f32mm,-tagged-globals,-the,-tlbiw,-tme,-tpidr-el1,-tpidr-el2,-tpidr-el3,-tpidrro-el0,-trbe,-use-experimental-zeroing-pseudos,-use-fixed-over-scalable-if-equal-cost,-use-postra-scheduler,-use-reciprocal-square-root,-v8.7a,-v8.8a,-v8.9a,-v8r,-v9.1a,-v9.2a,-v9.3a,-v9.4a,-v9.5a,-v9.6a,-v9a,-wfxt,-xs,-zcz-fp-workaround" }

!llvm.module.flags = !{!0}

!0 = !{i32 8, !"PIC Level", i32 2}
!1 = !{!2}
!2 = distinct !{!2, !3, !"c_api_transform.transformCallToOperation: argument 0"}
!3 = distinct !{!3, !"c_api_transform.transformCallToOperation"}
