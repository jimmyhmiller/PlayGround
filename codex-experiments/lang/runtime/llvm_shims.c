// Thin C shims for LLVM C API calls that need parameter type conversion.
// .lang only has i64/ptr types; these shims convert i64→unsigned, handle void returns, etc.

#include <llvm-c/Core.h>
#include <llvm-c/Target.h>
#include <llvm-c/TargetMachine.h>
#include <stdint.h>
#include <stdio.h>

// =============================================================================
// 1. Non-trivial logic shims
// =============================================================================

int64_t lang_llvm_init_native_target(void) {
    return (int64_t)LLVMInitializeNativeTarget();
}

int64_t lang_llvm_init_native_asm_printer(void) {
    return (int64_t)LLVMInitializeNativeAsmPrinter();
}

void* lang_llvm_get_target_from_triple(const char* triple) {
    LLVMTargetRef target = NULL;
    char* error = NULL;
    if (LLVMGetTargetFromTriple(triple, &target, &error) != 0) {
        if (error) LLVMDisposeMessage(error);
        return NULL;
    }
    return (void*)target;
}

int64_t lang_llvm_set_module_target(void* module, void* machine) {
    char* triple = LLVMGetTargetMachineTriple((LLVMTargetMachineRef)machine);
    LLVMSetTarget((LLVMModuleRef)module, triple);
    LLVMTargetDataRef data = LLVMCreateTargetDataLayout((LLVMTargetMachineRef)machine);
    LLVMSetModuleDataLayout((LLVMModuleRef)module, data);
    LLVMDisposeMessage(triple);
    return 0;
}

int64_t lang_llvm_emit_to_file(void* machine, void* module, const char* filename) {
    char* error = NULL;
    if (LLVMTargetMachineEmitToFile(
            (LLVMTargetMachineRef)machine,
            (LLVMModuleRef)module,
            (char*)filename,
            LLVMObjectFile,
            &error) != 0) {
        if (error) {
            fprintf(stderr, "LLVM emit error: %s\n", error);
            LLVMDisposeMessage(error);
        }
        return 1;
    }
    return 0;
}

// =============================================================================
// 2. i64→unsigned / LLVMBool parameter conversion shims
// =============================================================================

void* lang_llvm_function_type(void* ret, void* params, int64_t count, int64_t vararg) {
    return (void*)LLVMFunctionType(
        (LLVMTypeRef)ret,
        (LLVMTypeRef*)params,
        (unsigned)count,
        vararg != 0 ? 1 : 0);
}

void* lang_llvm_struct_type(void* ctx, void* elems, int64_t count, int64_t packed) {
    return (void*)LLVMStructTypeInContext(
        (LLVMContextRef)ctx,
        (LLVMTypeRef*)elems,
        (unsigned)count,
        packed != 0 ? 1 : 0);
}

void* lang_llvm_array_type(void* elem, int64_t count) {
    return (void*)LLVMArrayType2((LLVMTypeRef)elem, (uint64_t)count);
}

void* lang_llvm_const_int(void* ty, int64_t val, int64_t sign_extend) {
    return (void*)LLVMConstInt(
        (LLVMTypeRef)ty,
        (unsigned long long)val,
        sign_extend != 0 ? 1 : 0);
}

void* lang_llvm_const_string(void* ctx, const char* s, int64_t len, int64_t null_terminate) {
    // Our wrapper takes null_terminate=1 meaning "yes add null".
    // LLVM's DontNullTerminate=1 means "don't add null". So invert.
    return (void*)LLVMConstStringInContext(
        (LLVMContextRef)ctx,
        s,
        (unsigned)len,
        null_terminate != 0 ? 0 : 1);
}

void* lang_llvm_const_named_struct(void* ty, void* vals, int64_t count) {
    return (void*)LLVMConstNamedStruct(
        (LLVMTypeRef)ty,
        (LLVMValueRef*)vals,
        (unsigned)count);
}

void* lang_llvm_const_array(void* elem, void* vals, int64_t count) {
    return (void*)LLVMConstArray2(
        (LLVMTypeRef)elem,
        (LLVMValueRef*)vals,
        (uint64_t)count);
}

void* lang_llvm_const_gep2(void* ty, void* val, void* indices, int64_t count) {
    return (void*)LLVMConstGEP2(
        (LLVMTypeRef)ty,
        (LLVMValueRef)val,
        (LLVMValueRef*)indices,
        (unsigned)count);
}

void* lang_llvm_get_param(void* fn, int64_t index) {
    return (void*)LLVMGetParam((LLVMValueRef)fn, (unsigned)index);
}

int64_t lang_llvm_count_params(void* fn) {
    return (int64_t)LLVMCountParams((LLVMValueRef)fn);
}

void* lang_llvm_build_call(void* b, void* ty, void* fn, void* args, int64_t count, const char* name) {
    return (void*)LLVMBuildCall2(
        (LLVMBuilderRef)b,
        (LLVMTypeRef)ty,
        (LLVMValueRef)fn,
        (LLVMValueRef*)args,
        (unsigned)count,
        name);
}

void* lang_llvm_build_switch(void* b, void* val, void* def, int64_t cases) {
    return (void*)LLVMBuildSwitch(
        (LLVMBuilderRef)b,
        (LLVMValueRef)val,
        (LLVMBasicBlockRef)def,
        (unsigned)cases);
}

void* lang_llvm_build_gep2(void* b, void* ty, void* ptr, void* idx, int64_t count, const char* name) {
    return (void*)LLVMBuildGEP2(
        (LLVMBuilderRef)b,
        (LLVMTypeRef)ty,
        (LLVMValueRef)ptr,
        (LLVMValueRef*)idx,
        (unsigned)count,
        name);
}

void* lang_llvm_build_inbounds_gep2(void* b, void* ty, void* ptr, void* idx, int64_t count, const char* name) {
    return (void*)LLVMBuildInBoundsGEP2(
        (LLVMBuilderRef)b,
        (LLVMTypeRef)ty,
        (LLVMValueRef)ptr,
        (LLVMValueRef*)idx,
        (unsigned)count,
        name);
}

void* lang_llvm_build_struct_gep2(void* b, void* ty, void* ptr, int64_t idx, const char* name) {
    return (void*)LLVMBuildStructGEP2(
        (LLVMBuilderRef)b,
        (LLVMTypeRef)ty,
        (LLVMValueRef)ptr,
        (unsigned)idx,
        name);
}

void* lang_llvm_build_icmp(void* b, int64_t pred, void* l, void* r, const char* name) {
    return (void*)LLVMBuildICmp(
        (LLVMBuilderRef)b,
        (LLVMIntPredicate)pred,
        (LLVMValueRef)l,
        (LLVMValueRef)r,
        name);
}

void* lang_llvm_build_fcmp(void* b, int64_t pred, void* l, void* r, const char* name) {
    return (void*)LLVMBuildFCmp(
        (LLVMBuilderRef)b,
        (LLVMRealPredicate)pred,
        (LLVMValueRef)l,
        (LLVMValueRef)r,
        name);
}

void* lang_llvm_build_memset(void* b, void* ptr, void* val, void* len, int64_t align) {
    return (void*)LLVMBuildMemSet(
        (LLVMBuilderRef)b,
        (LLVMValueRef)ptr,
        (LLVMValueRef)val,
        (LLVMValueRef)len,
        (unsigned)align);
}

void* lang_llvm_create_target_machine(void* target, const char* triple, const char* cpu, const char* features) {
    return (void*)LLVMCreateTargetMachine(
        (LLVMTargetRef)target,
        triple,
        cpu,
        features,
        LLVMCodeGenLevelNone,
        LLVMRelocDefault,
        LLVMCodeModelDefault);
}

// =============================================================================
// 3. void→i64 return conversion shims
// =============================================================================

int64_t lang_llvm_dispose_builder(void* b) {
    LLVMDisposeBuilder((LLVMBuilderRef)b);
    return 0;
}

int64_t lang_llvm_dispose_module(void* m) {
    LLVMDisposeModule((LLVMModuleRef)m);
    return 0;
}

int64_t lang_llvm_context_dispose(void* ctx) {
    LLVMContextDispose((LLVMContextRef)ctx);
    return 0;
}

int64_t lang_llvm_set_initializer(void* global, void* val) {
    LLVMSetInitializer((LLVMValueRef)global, (LLVMValueRef)val);
    return 0;
}

int64_t lang_llvm_position_at_end(void* builder, void* bb) {
    LLVMPositionBuilderAtEnd((LLVMBuilderRef)builder, (LLVMBasicBlockRef)bb);
    return 0;
}

int64_t lang_llvm_position_before(void* builder, void* instr) {
    LLVMPositionBuilderBefore((LLVMBuilderRef)builder, (LLVMValueRef)instr);
    return 0;
}

int64_t lang_llvm_add_incoming(void* phi, void* vals, void* blocks, int64_t count) {
    LLVMAddIncoming(
        (LLVMValueRef)phi,
        (LLVMValueRef*)vals,
        (LLVMBasicBlockRef*)blocks,
        (unsigned)count);
    return 0;
}

int64_t lang_llvm_add_case(void* sw, void* val, void* dest) {
    LLVMAddCase((LLVMValueRef)sw, (LLVMValueRef)val, (LLVMBasicBlockRef)dest);
    return 0;
}

int64_t lang_llvm_set_volatile(void* instr, int64_t is_volatile) {
    LLVMSetVolatile((LLVMValueRef)instr, is_volatile != 0 ? 1 : 0);
    return 0;
}

void* lang_llvm_ptr_type(void* ctx) {
    return (void*)LLVMPointerTypeInContext((LLVMContextRef)ctx, 0);
}

void* lang_llvm_pointer_type(void* ty) {
    return (void*)LLVMPointerType((LLVMTypeRef)ty, 0);
}

int64_t lang_llvm_get_type_kind(void* ty) {
    return (int64_t)LLVMGetTypeKind((LLVMTypeRef)ty);
}

const char* lang_llvm_print_module_to_string(void* m) {
    return LLVMPrintModuleToString((LLVMModuleRef)m);
}
