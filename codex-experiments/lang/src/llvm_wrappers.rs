//! Thin `#[no_mangle] pub extern "C"` wrappers around llvm-sys.
//! All LLVM opaque refs are `*mut u8` (maps to `RawPointer<I8>` in .lang).
//! Strings are C strings (null-terminated `*const u8`).

use llvm_sys::core::*;
use llvm_sys::prelude::*;
use llvm_sys::target::*;
use llvm_sys::target_machine::*;
use std::ffi::CStr;

// ── Context / Module / Builder ──────────────────────────────────────────

#[no_mangle]
pub extern "C" fn llvm_context_create() -> *mut u8 {
    unsafe { LLVMContextCreate() as *mut u8 }
}

#[no_mangle]
pub extern "C" fn llvm_module_create(name: *const u8, ctx: *mut u8) -> *mut u8 {
    unsafe { LLVMModuleCreateWithNameInContext(name as *const i8, ctx as LLVMContextRef) as *mut u8 }
}

#[no_mangle]
pub extern "C" fn llvm_create_builder(ctx: *mut u8) -> *mut u8 {
    unsafe { LLVMCreateBuilderInContext(ctx as LLVMContextRef) as *mut u8 }
}

#[no_mangle]
pub extern "C" fn llvm_dispose_builder(b: *mut u8) -> i64 {
    unsafe { LLVMDisposeBuilder(b as LLVMBuilderRef) }
    0
}

#[no_mangle]
pub extern "C" fn llvm_dispose_module(m: *mut u8) -> i64 {
    unsafe { LLVMDisposeModule(m as LLVMModuleRef) }
    0
}

#[no_mangle]
pub extern "C" fn llvm_context_dispose(ctx: *mut u8) -> i64 {
    unsafe { LLVMContextDispose(ctx as LLVMContextRef) }
    0
}

// ── Types ───────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn llvm_int1_type(ctx: *mut u8) -> *mut u8 {
    unsafe { LLVMInt1TypeInContext(ctx as LLVMContextRef) as *mut u8 }
}

#[no_mangle]
pub extern "C" fn llvm_int8_type(ctx: *mut u8) -> *mut u8 {
    unsafe { LLVMInt8TypeInContext(ctx as LLVMContextRef) as *mut u8 }
}

#[no_mangle]
pub extern "C" fn llvm_int16_type(ctx: *mut u8) -> *mut u8 {
    unsafe { LLVMInt16TypeInContext(ctx as LLVMContextRef) as *mut u8 }
}

#[no_mangle]
pub extern "C" fn llvm_int32_type(ctx: *mut u8) -> *mut u8 {
    unsafe { LLVMInt32TypeInContext(ctx as LLVMContextRef) as *mut u8 }
}

#[no_mangle]
pub extern "C" fn llvm_int64_type(ctx: *mut u8) -> *mut u8 {
    unsafe { LLVMInt64TypeInContext(ctx as LLVMContextRef) as *mut u8 }
}

#[no_mangle]
pub extern "C" fn llvm_float_type(ctx: *mut u8) -> *mut u8 {
    unsafe { LLVMFloatTypeInContext(ctx as LLVMContextRef) as *mut u8 }
}

#[no_mangle]
pub extern "C" fn llvm_double_type(ctx: *mut u8) -> *mut u8 {
    unsafe { LLVMDoubleTypeInContext(ctx as LLVMContextRef) as *mut u8 }
}

#[no_mangle]
pub extern "C" fn llvm_ptr_type(ctx: *mut u8) -> *mut u8 {
    unsafe { LLVMPointerTypeInContext(ctx as LLVMContextRef, 0) as *mut u8 }
}

#[no_mangle]
pub extern "C" fn llvm_void_type(ctx: *mut u8) -> *mut u8 {
    unsafe { LLVMVoidTypeInContext(ctx as LLVMContextRef) as *mut u8 }
}

#[no_mangle]
pub extern "C" fn llvm_function_type(
    ret: *mut u8,
    params: *mut u8,
    param_count: i64,
    is_vararg: i64,
) -> *mut u8 {
    unsafe {
        LLVMFunctionType(
            ret as LLVMTypeRef,
            params as *mut LLVMTypeRef,
            param_count as u32,
            if is_vararg != 0 { 1 } else { 0 },
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_struct_type(
    ctx: *mut u8,
    elements: *mut u8,
    count: i64,
    packed: i64,
) -> *mut u8 {
    unsafe {
        LLVMStructTypeInContext(
            ctx as LLVMContextRef,
            elements as *mut LLVMTypeRef,
            count as u32,
            if packed != 0 { 1 } else { 0 },
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_array_type(elem_ty: *mut u8, count: i64) -> *mut u8 {
    unsafe { LLVMArrayType2(elem_ty as LLVMTypeRef, count as u64) as *mut u8 }
}

// ── Constants ───────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn llvm_const_int(ty: *mut u8, val: i64, sign_extend: i64) -> *mut u8 {
    unsafe {
        LLVMConstInt(
            ty as LLVMTypeRef,
            val as u64,
            if sign_extend != 0 { 1 } else { 0 },
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_const_real(ty: *mut u8, val: f64) -> *mut u8 {
    unsafe { LLVMConstReal(ty as LLVMTypeRef, val) as *mut u8 }
}

#[no_mangle]
pub extern "C" fn llvm_const_null(ty: *mut u8) -> *mut u8 {
    unsafe { LLVMConstNull(ty as LLVMTypeRef) as *mut u8 }
}

#[no_mangle]
pub extern "C" fn llvm_const_string(
    ctx: *mut u8,
    s: *const u8,
    len: i64,
    null_terminate: i64,
) -> *mut u8 {
    unsafe {
        LLVMConstStringInContext(
            ctx as LLVMContextRef,
            s as *const i8,
            len as u32,
            if null_terminate != 0 { 0 } else { 1 }, // DontNullTerminate is inverted
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_const_named_struct(
    struct_ty: *mut u8,
    vals: *mut u8,
    count: i64,
) -> *mut u8 {
    unsafe {
        LLVMConstNamedStruct(
            struct_ty as LLVMTypeRef,
            vals as *mut LLVMValueRef,
            count as u32,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_const_array(
    elem_ty: *mut u8,
    vals: *mut u8,
    count: i64,
) -> *mut u8 {
    unsafe {
        LLVMConstArray2(
            elem_ty as LLVMTypeRef,
            vals as *mut LLVMValueRef,
            count as u64,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_const_gep2(
    ty: *mut u8,
    val: *mut u8,
    indices: *mut u8,
    count: i64,
) -> *mut u8 {
    unsafe {
        LLVMConstGEP2(
            ty as LLVMTypeRef,
            val as LLVMValueRef,
            indices as *mut LLVMValueRef,
            count as u32,
        ) as *mut u8
    }
}

// ── Module ops ──────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn llvm_add_function(
    module: *mut u8,
    name: *const u8,
    fn_ty: *mut u8,
) -> *mut u8 {
    unsafe {
        LLVMAddFunction(
            module as LLVMModuleRef,
            name as *const i8,
            fn_ty as LLVMTypeRef,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_get_named_function(module: *mut u8, name: *const u8) -> *mut u8 {
    unsafe {
        LLVMGetNamedFunction(module as LLVMModuleRef, name as *const i8) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_add_global(
    module: *mut u8,
    ty: *mut u8,
    name: *const u8,
) -> *mut u8 {
    unsafe {
        LLVMAddGlobal(
            module as LLVMModuleRef,
            ty as LLVMTypeRef,
            name as *const i8,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_set_initializer(global: *mut u8, val: *mut u8) -> i64 {
    unsafe {
        LLVMSetInitializer(global as LLVMValueRef, val as LLVMValueRef);
    }
    0
}

#[no_mangle]
pub extern "C" fn llvm_get_value_type(val: *mut u8) -> *mut u8 {
    unsafe { LLVMTypeOf(val as LLVMValueRef) as *mut u8 }
}

#[no_mangle]
pub extern "C" fn llvm_print_module_to_string(module: *mut u8) -> *const u8 {
    unsafe { LLVMPrintModuleToString(module as LLVMModuleRef) as *const u8 }
}

// ── Function / BB ops ───────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn llvm_get_param(f: *mut u8, index: i64) -> *mut u8 {
    unsafe { LLVMGetParam(f as LLVMValueRef, index as u32) as *mut u8 }
}

#[no_mangle]
pub extern "C" fn llvm_count_params(f: *mut u8) -> i64 {
    unsafe { LLVMCountParams(f as LLVMValueRef) as i64 }
}

#[no_mangle]
pub extern "C" fn llvm_append_basic_block(
    ctx: *mut u8,
    f: *mut u8,
    name: *const u8,
) -> *mut u8 {
    unsafe {
        LLVMAppendBasicBlockInContext(
            ctx as LLVMContextRef,
            f as LLVMValueRef,
            name as *const i8,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_get_first_basic_block(f: *mut u8) -> *mut u8 {
    unsafe { LLVMGetFirstBasicBlock(f as LLVMValueRef) as *mut u8 }
}

#[no_mangle]
pub extern "C" fn llvm_get_bb_terminator(bb: *mut u8) -> *mut u8 {
    unsafe { LLVMGetBasicBlockTerminator(bb as LLVMBasicBlockRef) as *mut u8 }
}

#[no_mangle]
pub extern "C" fn llvm_get_insert_block(builder: *mut u8) -> *mut u8 {
    unsafe { LLVMGetInsertBlock(builder as LLVMBuilderRef) as *mut u8 }
}

// ── Builder positioning ─────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn llvm_position_at_end(builder: *mut u8, bb: *mut u8) -> i64 {
    unsafe {
        LLVMPositionBuilderAtEnd(builder as LLVMBuilderRef, bb as LLVMBasicBlockRef);
    }
    0
}

#[no_mangle]
pub extern "C" fn llvm_position_before(builder: *mut u8, instr: *mut u8) -> i64 {
    unsafe {
        LLVMPositionBuilderBefore(builder as LLVMBuilderRef, instr as LLVMValueRef);
    }
    0
}

// ── Builder instructions ────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn llvm_build_alloca(
    builder: *mut u8,
    ty: *mut u8,
    name: *const u8,
) -> *mut u8 {
    unsafe {
        LLVMBuildAlloca(
            builder as LLVMBuilderRef,
            ty as LLVMTypeRef,
            name as *const i8,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_build_load(
    builder: *mut u8,
    ty: *mut u8,
    ptr: *mut u8,
    name: *const u8,
) -> *mut u8 {
    unsafe {
        LLVMBuildLoad2(
            builder as LLVMBuilderRef,
            ty as LLVMTypeRef,
            ptr as LLVMValueRef,
            name as *const i8,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_build_store(
    builder: *mut u8,
    val: *mut u8,
    ptr: *mut u8,
) -> *mut u8 {
    unsafe {
        LLVMBuildStore(
            builder as LLVMBuilderRef,
            val as LLVMValueRef,
            ptr as LLVMValueRef,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_build_call(
    builder: *mut u8,
    fn_ty: *mut u8,
    f: *mut u8,
    args: *mut u8,
    arg_count: i64,
    name: *const u8,
) -> *mut u8 {
    unsafe {
        LLVMBuildCall2(
            builder as LLVMBuilderRef,
            fn_ty as LLVMTypeRef,
            f as LLVMValueRef,
            args as *mut LLVMValueRef,
            arg_count as u32,
            name as *const i8,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_build_ret(builder: *mut u8, val: *mut u8) -> *mut u8 {
    unsafe {
        LLVMBuildRet(builder as LLVMBuilderRef, val as LLVMValueRef) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_build_ret_void(builder: *mut u8) -> *mut u8 {
    unsafe { LLVMBuildRetVoid(builder as LLVMBuilderRef) as *mut u8 }
}

#[no_mangle]
pub extern "C" fn llvm_build_br(builder: *mut u8, dest_bb: *mut u8) -> *mut u8 {
    unsafe {
        LLVMBuildBr(builder as LLVMBuilderRef, dest_bb as LLVMBasicBlockRef) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_build_cond_br(
    builder: *mut u8,
    cond: *mut u8,
    then_bb: *mut u8,
    else_bb: *mut u8,
) -> *mut u8 {
    unsafe {
        LLVMBuildCondBr(
            builder as LLVMBuilderRef,
            cond as LLVMValueRef,
            then_bb as LLVMBasicBlockRef,
            else_bb as LLVMBasicBlockRef,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_build_switch(
    builder: *mut u8,
    val: *mut u8,
    default_bb: *mut u8,
    num_cases: i64,
) -> *mut u8 {
    unsafe {
        LLVMBuildSwitch(
            builder as LLVMBuilderRef,
            val as LLVMValueRef,
            default_bb as LLVMBasicBlockRef,
            num_cases as u32,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_build_phi(
    builder: *mut u8,
    ty: *mut u8,
    name: *const u8,
) -> *mut u8 {
    unsafe {
        LLVMBuildPhi(
            builder as LLVMBuilderRef,
            ty as LLVMTypeRef,
            name as *const i8,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_build_gep2(
    builder: *mut u8,
    ty: *mut u8,
    ptr: *mut u8,
    indices: *mut u8,
    count: i64,
    name: *const u8,
) -> *mut u8 {
    unsafe {
        LLVMBuildGEP2(
            builder as LLVMBuilderRef,
            ty as LLVMTypeRef,
            ptr as LLVMValueRef,
            indices as *mut LLVMValueRef,
            count as u32,
            name as *const i8,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_build_inbounds_gep2(
    builder: *mut u8,
    ty: *mut u8,
    ptr: *mut u8,
    indices: *mut u8,
    count: i64,
    name: *const u8,
) -> *mut u8 {
    unsafe {
        LLVMBuildInBoundsGEP2(
            builder as LLVMBuilderRef,
            ty as LLVMTypeRef,
            ptr as LLVMValueRef,
            indices as *mut LLVMValueRef,
            count as u32,
            name as *const i8,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_build_bitcast(
    builder: *mut u8,
    val: *mut u8,
    dest_ty: *mut u8,
    name: *const u8,
) -> *mut u8 {
    unsafe {
        LLVMBuildBitCast(
            builder as LLVMBuilderRef,
            val as LLVMValueRef,
            dest_ty as LLVMTypeRef,
            name as *const i8,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_build_struct_gep2(
    builder: *mut u8,
    struct_ty: *mut u8,
    ptr: *mut u8,
    idx: i64,
    name: *const u8,
) -> *mut u8 {
    unsafe {
        LLVMBuildStructGEP2(
            builder as LLVMBuilderRef,
            struct_ty as LLVMTypeRef,
            ptr as LLVMValueRef,
            idx as u32,
            name as *const i8,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_build_memset(
    builder: *mut u8,
    ptr: *mut u8,
    val: *mut u8,
    len: *mut u8,
    align: i64,
) -> *mut u8 {
    unsafe {
        LLVMBuildMemSet(
            builder as LLVMBuilderRef,
            ptr as LLVMValueRef,
            val as LLVMValueRef,
            len as LLVMValueRef,
            align as u32,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_build_global_string_ptr(
    builder: *mut u8,
    s: *const u8,
    name: *const u8,
) -> *mut u8 {
    unsafe {
        LLVMBuildGlobalStringPtr(
            builder as LLVMBuilderRef,
            s as *const i8,
            name as *const i8,
        ) as *mut u8
    }
}

// ── Integer ops ─────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn llvm_build_add(
    builder: *mut u8,
    l: *mut u8,
    r: *mut u8,
    name: *const u8,
) -> *mut u8 {
    unsafe {
        LLVMBuildAdd(
            builder as LLVMBuilderRef,
            l as LLVMValueRef,
            r as LLVMValueRef,
            name as *const i8,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_build_sub(
    builder: *mut u8,
    l: *mut u8,
    r: *mut u8,
    name: *const u8,
) -> *mut u8 {
    unsafe {
        LLVMBuildSub(
            builder as LLVMBuilderRef,
            l as LLVMValueRef,
            r as LLVMValueRef,
            name as *const i8,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_build_mul(
    builder: *mut u8,
    l: *mut u8,
    r: *mut u8,
    name: *const u8,
) -> *mut u8 {
    unsafe {
        LLVMBuildMul(
            builder as LLVMBuilderRef,
            l as LLVMValueRef,
            r as LLVMValueRef,
            name as *const i8,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_build_sdiv(
    builder: *mut u8,
    l: *mut u8,
    r: *mut u8,
    name: *const u8,
) -> *mut u8 {
    unsafe {
        LLVMBuildSDiv(
            builder as LLVMBuilderRef,
            l as LLVMValueRef,
            r as LLVMValueRef,
            name as *const i8,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_build_srem(
    builder: *mut u8,
    l: *mut u8,
    r: *mut u8,
    name: *const u8,
) -> *mut u8 {
    unsafe {
        LLVMBuildSRem(
            builder as LLVMBuilderRef,
            l as LLVMValueRef,
            r as LLVMValueRef,
            name as *const i8,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_build_icmp(
    builder: *mut u8,
    predicate: i64,
    l: *mut u8,
    r: *mut u8,
    name: *const u8,
) -> *mut u8 {
    unsafe {
        LLVMBuildICmp(
            builder as LLVMBuilderRef,
            std::mem::transmute::<u32, llvm_sys::LLVMIntPredicate>(predicate as u32),
            l as LLVMValueRef,
            r as LLVMValueRef,
            name as *const i8,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_build_and(
    builder: *mut u8,
    l: *mut u8,
    r: *mut u8,
    name: *const u8,
) -> *mut u8 {
    unsafe {
        LLVMBuildAnd(
            builder as LLVMBuilderRef,
            l as LLVMValueRef,
            r as LLVMValueRef,
            name as *const i8,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_build_or(
    builder: *mut u8,
    l: *mut u8,
    r: *mut u8,
    name: *const u8,
) -> *mut u8 {
    unsafe {
        LLVMBuildOr(
            builder as LLVMBuilderRef,
            l as LLVMValueRef,
            r as LLVMValueRef,
            name as *const i8,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_build_xor(
    builder: *mut u8,
    l: *mut u8,
    r: *mut u8,
    name: *const u8,
) -> *mut u8 {
    unsafe {
        LLVMBuildXor(
            builder as LLVMBuilderRef,
            l as LLVMValueRef,
            r as LLVMValueRef,
            name as *const i8,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_build_zext(
    builder: *mut u8,
    val: *mut u8,
    dest_ty: *mut u8,
    name: *const u8,
) -> *mut u8 {
    unsafe {
        LLVMBuildZExt(
            builder as LLVMBuilderRef,
            val as LLVMValueRef,
            dest_ty as LLVMTypeRef,
            name as *const i8,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_build_trunc(
    builder: *mut u8,
    val: *mut u8,
    dest_ty: *mut u8,
    name: *const u8,
) -> *mut u8 {
    unsafe {
        LLVMBuildTrunc(
            builder as LLVMBuilderRef,
            val as LLVMValueRef,
            dest_ty as LLVMTypeRef,
            name as *const i8,
        ) as *mut u8
    }
}

// ── Float ops ───────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn llvm_build_fadd(
    builder: *mut u8,
    l: *mut u8,
    r: *mut u8,
    name: *const u8,
) -> *mut u8 {
    unsafe {
        LLVMBuildFAdd(
            builder as LLVMBuilderRef,
            l as LLVMValueRef,
            r as LLVMValueRef,
            name as *const i8,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_build_fsub(
    builder: *mut u8,
    l: *mut u8,
    r: *mut u8,
    name: *const u8,
) -> *mut u8 {
    unsafe {
        LLVMBuildFSub(
            builder as LLVMBuilderRef,
            l as LLVMValueRef,
            r as LLVMValueRef,
            name as *const i8,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_build_fmul(
    builder: *mut u8,
    l: *mut u8,
    r: *mut u8,
    name: *const u8,
) -> *mut u8 {
    unsafe {
        LLVMBuildFMul(
            builder as LLVMBuilderRef,
            l as LLVMValueRef,
            r as LLVMValueRef,
            name as *const i8,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_build_fdiv(
    builder: *mut u8,
    l: *mut u8,
    r: *mut u8,
    name: *const u8,
) -> *mut u8 {
    unsafe {
        LLVMBuildFDiv(
            builder as LLVMBuilderRef,
            l as LLVMValueRef,
            r as LLVMValueRef,
            name as *const i8,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_build_frem(
    builder: *mut u8,
    l: *mut u8,
    r: *mut u8,
    name: *const u8,
) -> *mut u8 {
    unsafe {
        LLVMBuildFRem(
            builder as LLVMBuilderRef,
            l as LLVMValueRef,
            r as LLVMValueRef,
            name as *const i8,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_build_fcmp(
    builder: *mut u8,
    predicate: i64,
    l: *mut u8,
    r: *mut u8,
    name: *const u8,
) -> *mut u8 {
    unsafe {
        LLVMBuildFCmp(
            builder as LLVMBuilderRef,
            std::mem::transmute::<u32, llvm_sys::LLVMRealPredicate>(predicate as u32),
            l as LLVMValueRef,
            r as LLVMValueRef,
            name as *const i8,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_build_fneg(
    builder: *mut u8,
    val: *mut u8,
    name: *const u8,
) -> *mut u8 {
    unsafe {
        LLVMBuildFNeg(
            builder as LLVMBuilderRef,
            val as LLVMValueRef,
            name as *const i8,
        ) as *mut u8
    }
}

// ── Phi / Switch ────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn llvm_add_incoming(
    phi: *mut u8,
    vals: *mut u8,
    blocks: *mut u8,
    count: i64,
) -> i64 {
    unsafe {
        LLVMAddIncoming(
            phi as LLVMValueRef,
            vals as *mut LLVMValueRef,
            blocks as *mut LLVMBasicBlockRef,
            count as u32,
        );
    }
    0
}

#[no_mangle]
pub extern "C" fn llvm_add_case(switch: *mut u8, val: *mut u8, dest_bb: *mut u8) -> i64 {
    unsafe {
        LLVMAddCase(
            switch as LLVMValueRef,
            val as LLVMValueRef,
            dest_bb as LLVMBasicBlockRef,
        );
    }
    0
}

// ── Volatile ────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn llvm_set_volatile(instr: *mut u8, is_volatile: i64) -> i64 {
    unsafe {
        LLVMSetVolatile(
            instr as LLVMValueRef,
            if is_volatile != 0 { 1 } else { 0 },
        );
    }
    0
}

// ── Target / AOT ────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn llvm_init_native_target() -> i64 {
    unsafe {
        LLVM_InitializeNativeTarget() as i64
    }
}

#[no_mangle]
pub extern "C" fn llvm_init_native_asm_printer() -> i64 {
    unsafe {
        LLVM_InitializeNativeAsmPrinter() as i64
    }
}

#[no_mangle]
pub extern "C" fn llvm_get_default_triple() -> *const u8 {
    unsafe { LLVMGetDefaultTargetTriple() as *const u8 }
}

#[no_mangle]
pub extern "C" fn llvm_get_target_from_triple(triple: *const u8) -> *mut u8 {
    unsafe {
        let mut target: LLVMTargetRef = std::ptr::null_mut();
        let mut error: *mut i8 = std::ptr::null_mut();
        let result = LLVMGetTargetFromTriple(
            triple as *const i8,
            &mut target,
            &mut error,
        );
        if result != 0 {
            if !error.is_null() {
                LLVMDisposeMessage(error);
            }
            return std::ptr::null_mut();
        }
        target as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_get_host_cpu_name() -> *const u8 {
    unsafe { LLVMGetHostCPUName() as *const u8 }
}

#[no_mangle]
pub extern "C" fn llvm_get_host_cpu_features() -> *const u8 {
    unsafe { LLVMGetHostCPUFeatures() as *const u8 }
}

#[no_mangle]
pub extern "C" fn llvm_create_target_machine(
    target: *mut u8,
    triple: *const u8,
    cpu: *const u8,
    features: *const u8,
) -> *mut u8 {
    unsafe {
        LLVMCreateTargetMachine(
            target as LLVMTargetRef,
            triple as *const i8,
            cpu as *const i8,
            features as *const i8,
            LLVMCodeGenOptLevel::LLVMCodeGenLevelNone,
            LLVMRelocMode::LLVMRelocDefault,
            LLVMCodeModel::LLVMCodeModelDefault,
        ) as *mut u8
    }
}

#[no_mangle]
pub extern "C" fn llvm_set_module_target(module: *mut u8, machine: *mut u8) -> i64 {
    unsafe {
        let triple = LLVMGetTargetMachineTriple(machine as LLVMTargetMachineRef);
        LLVMSetTarget(module as LLVMModuleRef, triple);
        let data = LLVMCreateTargetDataLayout(machine as LLVMTargetMachineRef);
        LLVMSetModuleDataLayout(module as LLVMModuleRef, data);
        LLVMDisposeMessage(triple);
    }
    0
}

#[no_mangle]
pub extern "C" fn llvm_emit_to_file(
    machine: *mut u8,
    module: *mut u8,
    filename: *const u8,
) -> i64 {
    unsafe {
        let mut error: *mut i8 = std::ptr::null_mut();
        let result = LLVMTargetMachineEmitToFile(
            machine as LLVMTargetMachineRef,
            module as LLVMModuleRef,
            filename as *mut i8,
            LLVMCodeGenFileType::LLVMObjectFile,
            &mut error,
        );
        if result != 0 {
            if !error.is_null() {
                // Print error then dispose
                let err_str = CStr::from_ptr(error);
                eprintln!("LLVM emit error: {:?}", err_str);
                LLVMDisposeMessage(error);
            }
            return 1;
        }
        0
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn llvm_get_first_instruction(bb: *mut u8) -> *mut u8 {
    unsafe { LLVMGetFirstInstruction(bb as LLVMBasicBlockRef) as *mut u8 }
}

#[no_mangle]
pub extern "C" fn llvm_is_null(val: *mut u8) -> i64 {
    if val.is_null() { 1 } else { 0 }
}

#[no_mangle]
pub extern "C" fn llvm_get_global_value_type(global: *mut u8) -> *mut u8 {
    unsafe { LLVMGlobalGetValueType(global as LLVMValueRef) as *mut u8 }
}

#[no_mangle]
pub extern "C" fn llvm_get_element_type(ty: *mut u8) -> *mut u8 {
    unsafe { LLVMGetElementType(ty as LLVMTypeRef) as *mut u8 }
}

#[no_mangle]
pub extern "C" fn llvm_pointer_type(ty: *mut u8) -> *mut u8 {
    unsafe { LLVMPointerType(ty as LLVMTypeRef, 0) as *mut u8 }
}

#[no_mangle]
pub extern "C" fn llvm_get_type_kind(ty: *mut u8) -> i64 {
    unsafe { LLVMGetTypeKind(ty as LLVMTypeRef) as i64 }
}
