; ModuleID = 'vector_add_dump.mojo'
source_filename = "vector_add_dump.mojo"
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none, target_mem: none)
define dso_local amdgpu_kernel void @def_LITImmutOrigin_Origin_BcBsAgA6A6AcB6A6A_07fe4b6ad261ea64(ptr noundef nonnull readonly byref({ ptr, { { {} }, { {} } } }) captures(none) %0, ptr noundef nonnull readonly byref({ ptr, { { {} }, { {} } } }) captures(none) %1, ptr noundef nonnull readonly byref({ ptr, { { {} }, { {} } } }) captures(none) %2) #0 {
  %4 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %5 = zext i32 %4 to i64
  %6 = tail call dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %7 = getelementptr inbounds nuw i8, ptr addrspace(4) %6, i64 12
  %8 = load i16, ptr addrspace(4) %7, align 4, !range !2
  %9 = zext nneg i16 %8 to i64
  %10 = mul nuw nsw i64 %9, %5
  %11 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %12 = zext nneg i32 %11 to i64
  %13 = add nuw nsw i64 %10, %12
  %14 = icmp samesign ult i64 %13, 1024
  br i1 %14, label %15, label %22

15:                                               ; preds = %3
  %.global26 = addrspacecast ptr %0 to ptr addrspace(1)
  %.global24 = addrspacecast ptr %1 to ptr addrspace(1)
  %.global = addrspacecast ptr %2 to ptr addrspace(1)
  %.unpack = load ptr, ptr addrspace(1) %.global26, align 8, !amdgpu.noclobber !3
  %.unpack.global = addrspacecast ptr %.unpack to ptr addrspace(1)
  %16 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %.unpack.global, i64 %13
  %.unpack8 = load ptr, ptr addrspace(1) %.global24, align 8, !amdgpu.noclobber !3
  %.unpack8.global = addrspacecast ptr %.unpack8 to ptr addrspace(1)
  %17 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %.unpack8.global, i64 %13
  %.unpack16 = load ptr, ptr addrspace(1) %.global, align 8, !amdgpu.noclobber !3
  %.unpack16.global = addrspacecast ptr %.unpack16 to ptr addrspace(1)
  %18 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %.unpack16.global, i64 %13
  %19 = load float, ptr addrspace(1) %16, align 4, !amdgpu.noclobber !3
  %20 = load float, ptr addrspace(1) %17, align 4, !amdgpu.noclobber !3
  %21 = fadd contract float %19, %20
  store float %21, ptr addrspace(1) %18, align 4
  br label %22

22:                                               ; preds = %3, %15
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef align 4 ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none, target_mem: none) "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" "target-cpu"="gfx1151" "target-features" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"amdhsa_code_object_version", i32 600}
!2 = !{i16 1, i16 1025}
!3 = !{}
