use crate::backend::{Arm64Backend, X64Backend};
use crate::{
    DefaultJitConfig, FrameReifyKind, JitFunction, JitModule, JitOutcome,
    SafepointHandlerPayloadKind, call_jit, call_jit_with_reg_limit,
};
use dynexec::{
    AArch64CAbi, AArch64InternalCc, CallbackSafepoints, CodegenConfig, FrameLayout, FrameStrategy,
    NanBoxConfig, PreciseStackRoots, ShadowStackFrames, ShadowStackRoots, StackFrameLayout,
    StackMapFrames, StackMapRoots, StackMapSafepoints, StackSlotFrames, X64SysVCAbi,
};
use dynir::builder::{FunctionBuilder, ModuleBuilder};
use dynir::ir::*;
use dynir::types::Type;
use dynvalue::{LowBit, NanBox, TagScheme};

#[cfg(target_arch = "aarch64")]
core::arch::global_asm!(
    ".globl _dynlower_test_throw_exception_stub",
    "_dynlower_test_throw_exception_stub:",
    "mov x1, #2",
    "mov x2, #1234",
    "ret",
);

#[cfg(target_arch = "aarch64")]
unsafe extern "C" {
    fn dynlower_test_throw_exception_stub();
}

fn run_jit(func: &dynir::Function, args: &[u64]) -> u64 {
    // Use NanBox as default scheme for tests that don't use tag operations
    let jit = JitFunction::compile::<NanBox>(func, &[]);
    unsafe { call_jit(jit.as_ptr(), args) }
}

fn run_jit_outcome(func: &dynir::Function, args: &[u64]) -> JitOutcome {
    let jit = JitFunction::compile::<NanBox>(func, &[]);
    jit.call_outcome(args)
}

#[test]
fn capture_slice_returns_frame_reify_outcome() {
    let mut b = FunctionBuilder::new("capture", &[Type::I64], Some(Type::FrameSlice));
    let entry = b.entry_block();
    let arg = b.block_param(entry, 0);
    let handler_bb = b.create_block(&[Type::FrameSlice]);
    let prompt = b.create_prompt();
    b.push_prompt(prompt, handler_bb);
    let slice = b.capture_slice(prompt, &[arg]);
    b.pop_prompt(prompt);
    b.jump(handler_bb, &[slice]);
    b.switch_to_block(handler_bb);
    let _handler_param = b.block_param(handler_bb, 0);
    b.unreachable();
    let func = b.build();

    let jit = JitFunction::compile::<NanBox>(&func, &[]);
    match jit.call_outcome(&[55]) {
        JitOutcome::CaptureSlice {
            record_idx, values, ..
        } => {
            assert_eq!(record_idx, 0);
            // arg=55, handler_param=0 (block param), slice=0 (not yet assigned)
            assert_eq!(values, vec![55, 0, 0]);
        }
        other => panic!("expected capture outcome, got {other:?}"),
    }

    let records = jit.frame_reify_records();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].kind, FrameReifyKind::CaptureSlice);
    assert_eq!(records[0].prompt, Some(prompt));
    assert_eq!(records[0].active_prompts, vec![prompt]);
    assert_eq!(records[0].frame_value_count, func.value_types.len());
    assert_eq!(
        records[0].value_indices,
        vec![arg.index(), _handler_param.index(), slice.index()]
    );
    assert_eq!(
        records[0].value_types,
        vec![Type::I64, Type::FrameSlice, Type::FrameSlice]
    );
    assert!(records[0].root_payload_indices.is_empty());
    assert_eq!(records[0].return_dest, Some(slice.index()));
}

#[test]
fn capture_slice_marks_gc_payload_positions_as_roots() {
    let mut b = FunctionBuilder::new(
        "capture_gc",
        &[Type::GcPtr, Type::I64],
        Some(Type::FrameSlice),
    );
    let entry = b.entry_block();
    let obj = b.block_param(entry, 0);
    let num = b.block_param(entry, 1);
    let handler_bb = b.create_block(&[Type::FrameSlice]);
    let prompt = b.create_prompt();
    b.push_prompt(prompt, handler_bb);
    let slice = b.capture_slice(prompt, &[num, obj]);
    b.pop_prompt(prompt);
    b.jump(handler_bb, &[slice]);
    b.switch_to_block(handler_bb);
    let _handler_param = b.block_param(handler_bb, 0);
    b.unreachable();
    let func = b.build();

    let jit = JitFunction::compile::<NanBox>(&func, &[]);
    let records = jit.frame_reify_records();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].frame_value_count, func.value_types.len());
    assert_eq!(
        records[0].value_indices,
        vec![
            obj.index(),
            num.index(),
            _handler_param.index(),
            slice.index()
        ]
    );
    assert_eq!(
        records[0].value_types,
        vec![Type::GcPtr, Type::I64, Type::FrameSlice, Type::FrameSlice]
    );
    assert_eq!(records[0].root_payload_indices, vec![0]);
}

#[test]
fn abort_to_prompt_returns_frame_reify_outcome() {
    let mut b = FunctionBuilder::new("abort", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let arg = b.block_param(entry, 0);
    let prompt = b.create_prompt();
    b.abort_to_prompt(prompt, &[arg]);
    let func = b.build();

    let jit = JitFunction::compile::<NanBox>(&func, &[]);
    match jit.call_outcome(&[77]) {
        JitOutcome::AbortToPrompt {
            record_idx, values, ..
        } => {
            assert_eq!(record_idx, 0);
            assert_eq!(values, vec![77]);
        }
        other => panic!("expected abort outcome, got {other:?}"),
    }

    let records = jit.frame_reify_records();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].kind, FrameReifyKind::AbortToPrompt);
    assert_eq!(records[0].prompt, Some(prompt));
    assert_eq!(records[0].active_prompts, Vec::<PromptId>::new());
    assert_eq!(records[0].frame_value_count, func.value_types.len());
    assert_eq!(records[0].value_indices, vec![arg.index()]);
    assert_eq!(records[0].value_types, vec![Type::I64]);
    assert!(records[0].root_payload_indices.is_empty());
    assert_eq!(records[0].return_dest, None);
}

#[test]
fn clone_slice_returns_frame_reify_outcome() {
    let mut b = FunctionBuilder::new("clone_slice", &[Type::FrameSlice], Some(Type::FrameSlice));
    let entry = b.entry_block();
    let slice = b.block_param(entry, 0);
    let cloned = b.clone_slice(slice);
    b.unreachable();
    let func = b.build();

    let jit = JitFunction::compile::<NanBox>(&func, &[]);
    match jit.call_outcome(&[123]) {
        JitOutcome::CloneSlice {
            record_idx, values, ..
        } => {
            assert_eq!(record_idx, 0);
            assert_eq!(values, vec![123, 0]);
        }
        other => panic!("expected clone outcome, got {other:?}"),
    }

    let records = jit.frame_reify_records();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].kind, FrameReifyKind::CloneSlice);
    assert_eq!(records[0].prompt, None);
    assert_eq!(records[0].active_prompts, Vec::<PromptId>::new());
    assert_eq!(records[0].frame_value_count, func.value_types.len());
    assert_eq!(
        records[0].value_indices,
        vec![slice.index(), cloned.index()]
    );
    assert_eq!(records[0].control_value_indices, vec![slice.index()]);
    assert_eq!(
        records[0].value_types,
        vec![Type::FrameSlice, Type::FrameSlice]
    );
    assert_eq!(records[0].root_payload_indices, vec![0]);
    assert_eq!(records[0].return_dest, Some(cloned.index()));
}

#[test]
fn resume_slice_returns_frame_reify_outcome() {
    let mut b = FunctionBuilder::new(
        "resume_slice",
        &[Type::FrameSlice, Type::I64],
        Some(Type::I64),
    );
    let entry = b.entry_block();
    let slice = b.block_param(entry, 0);
    let value = b.block_param(entry, 1);
    let ret_bb = b.create_block(&[Type::I64]);
    b.resume_slice(slice, &[value], ret_bb, &[]);
    b.switch_to_block(ret_bb);
    let ret_val = b.block_param(ret_bb, 0);
    b.ret(ret_val);
    let func = b.build();

    let jit = JitFunction::compile::<NanBox>(&func, &[]);
    match jit.call_outcome(&[456, 99]) {
        JitOutcome::ResumeSlice {
            record_idx, values, ..
        } => {
            assert_eq!(record_idx, 0);
            assert_eq!(values, vec![456, 99]);
        }
        other => panic!("expected resume outcome, got {other:?}"),
    }

    let records = jit.frame_reify_records();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].kind, FrameReifyKind::ResumeSlice);
    assert_eq!(records[0].prompt, None);
    assert_eq!(records[0].active_prompts, Vec::<PromptId>::new());
    assert_eq!(records[0].frame_value_count, func.value_types.len());
    assert_eq!(records[0].value_indices, vec![slice.index(), value.index()]);
    assert_eq!(
        records[0].control_value_indices,
        vec![slice.index(), value.index()]
    );
    assert_eq!(records[0].value_types, vec![Type::FrameSlice, Type::I64]);
    assert_eq!(records[0].root_payload_indices, vec![0]);
    assert_eq!(records[0].return_dest, None);
}

#[test]
fn capture_slice_records_nested_active_prompts() {
    let mut b = FunctionBuilder::new("capture_nested", &[Type::I64], Some(Type::FrameSlice));
    let entry = b.entry_block();
    let arg = b.block_param(entry, 0);
    let outer_handler = b.create_block(&[Type::FrameSlice]);
    let inner_handler = b.create_block(&[Type::FrameSlice]);
    let outer = b.create_prompt();
    let inner = b.create_prompt();
    b.push_prompt(outer, outer_handler);
    b.push_prompt(inner, inner_handler);
    let slice = b.capture_slice(inner, &[arg]);
    b.pop_prompt(inner);
    b.jump(inner_handler, &[slice]);

    b.switch_to_block(inner_handler);
    let inner_result = b.block_param(inner_handler, 0);
    b.pop_prompt(outer);
    b.jump(outer_handler, &[inner_result]);

    b.switch_to_block(outer_handler);
    let _outer_result = b.block_param(outer_handler, 0);
    b.unreachable();
    let func = b.build();

    let jit = JitFunction::compile::<NanBox>(&func, &[]);
    let records = jit.frame_reify_records();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].prompt, Some(inner));
    assert_eq!(records[0].active_prompts, vec![outer, inner]);
}

fn run_jit_with_config<Cfg: CodegenConfig>(func: &dynir::Function, args: &[u64]) -> u64
where
    Cfg::Frames: dynexec::FrameStrategy<Cfg::Layout, Cfg::Roots, Cfg::CallingConvention>,
{
    let jit = JitFunction::compile_with_config::<Cfg>(func, &[]);
    unsafe { call_jit(jit.as_ptr(), args) }
}

struct TestInternalConfig;
#[cfg(target_arch = "aarch64")]
type TestInternalCc = AArch64InternalCc;
#[cfg(target_arch = "x86_64")]
type TestInternalCc = X64SysVCAbi;
#[cfg(target_arch = "aarch64")]
type TestCAbiCc = AArch64CAbi;
#[cfg(target_arch = "x86_64")]
type TestCAbiCc = X64SysVCAbi;

impl CodegenConfig for TestInternalConfig {
    type Layout = LowBit<3>;
    type Roots = PreciseStackRoots;
    type RootTransport = StackMapRoots;
    type CallingConvention = TestInternalCc;
    type Frames = StackMapFrames;
    type Safepoints = StackMapSafepoints;
}

struct TestCAbiConfig;
impl CodegenConfig for TestCAbiConfig {
    type Layout = LowBit<3>;
    type Roots = PreciseStackRoots;
    type RootTransport = ShadowStackRoots;
    type CallingConvention = TestCAbiCc;
    type Frames = ShadowStackFrames;
    type Safepoints = CallbackSafepoints;
}

struct TestStackMapConfig;
impl CodegenConfig for TestStackMapConfig {
    type Layout = LowBit<3>;
    type Roots = PreciseStackRoots;
    type RootTransport = StackMapRoots;
    type CallingConvention = TestInternalCc;
    type Frames = StackMapFrames;
    type Safepoints = StackMapSafepoints;
}

struct TestShadowStackConfig;
impl CodegenConfig for TestShadowStackConfig {
    type Layout = LowBit<3>;
    type Roots = PreciseStackRoots;
    type RootTransport = ShadowStackRoots;
    type CallingConvention = TestInternalCc;
    type Frames = ShadowStackFrames;
    type Safepoints = CallbackSafepoints;
}

#[derive(Debug, Clone)]
struct WrappedFrameLayout(StackFrameLayout);

impl FrameLayout for WrappedFrameLayout {
    fn alloc_local_slot(&mut self) -> i32 {
        self.0.alloc_local_slot()
    }

    fn alloc_local_slot_bytes(&mut self, size_bytes: i32) -> i32 {
        self.0.alloc_local_slot_bytes(size_bytes)
    }

    fn alloc_root_slot(&mut self) -> i32 {
        self.0.alloc_root_slot()
    }

    fn alloc_root_slot_bytes(&mut self, size_bytes: i32) -> i32 {
        self.0.alloc_root_slot_bytes(size_bytes)
    }

    fn alloc_shadow_root_slot(&mut self) -> i32 {
        self.0.alloc_shadow_root_slot()
    }

    fn reserve_outgoing_arg_bytes(&mut self, bytes: i32) {
        self.0.reserve_outgoing_arg_bytes(bytes);
    }

    fn total_frame_size(&self, stack_align: usize) -> i32 {
        self.0.total_frame_size(stack_align)
    }

    fn add_block_param_slot(&mut self, block_idx: usize, offset: i32) {
        self.0.add_block_param_slot(block_idx, offset);
    }

    fn block_param_slots(&self, block_idx: usize) -> &[i32] {
        self.0.block_param_slots(block_idx)
    }

    fn root_scan_size(&self) -> i32 {
        self.0.root_scan_size()
    }

    fn root_slots(&self) -> &[i32] {
        self.0.root_slots()
    }

    fn shadow_root_slots(&self) -> &[i32] {
        self.0.shadow_root_slots()
    }

    fn slot_access(&self, slot: i32) -> dynexec::FrameSlotAccess {
        self.0.slot_access(slot)
    }
}

struct WrappedFrameStrategy;

impl<L, R, C> FrameStrategy<L, R, C> for WrappedFrameStrategy
where
    L: dynexec::ValueLayout,
    R: dynexec::RootStrategy<L>,
    C: dynexec::CallingConvention<L>,
{
    const NAME: &'static str = "wrapped-frame-strategy";
    type Layout = WrappedFrameLayout;

    fn new_layout(block_count: usize) -> Self::Layout {
        WrappedFrameLayout(StackFrameLayout::new(block_count))
    }

    fn supports_stack_maps() -> bool {
        true
    }
}

struct TestWrappedFrameConfig;

impl CodegenConfig for TestWrappedFrameConfig {
    type Layout = LowBit<3>;
    type Roots = PreciseStackRoots;
    type RootTransport = StackMapRoots;
    type CallingConvention = TestInternalCc;
    type Frames = WrappedFrameStrategy;
    type Safepoints = StackMapSafepoints;
}

// ── Phase 1: return_const ──────────────────────────────────────────

#[test]
fn return_const() {
    let mut b = FunctionBuilder::new("ret42", &[], Some(Type::I64));
    let v = b.iconst(Type::I64, 42);
    b.ret(v);
    let func = b.build();
    assert_eq!(run_jit(&func, &[]), 42);
}

#[test]
fn compile_with_explicit_internal_config() {
    let mut b = FunctionBuilder::new("ret7", &[], Some(Type::I64));
    let v = b.iconst(Type::I64, 7);
    b.ret(v);
    let func = b.build();
    let jit = JitFunction::compile_with_config::<TestInternalConfig>(&func, &[]);
    assert_eq!(unsafe { call_jit(jit.as_ptr(), &[]) }, 7);
}

#[test]
fn compile_with_explicit_backend_type() {
    let mut b = FunctionBuilder::new("ret13", &[], Some(Type::I64));
    let v = b.iconst(Type::I64, 13);
    b.ret(v);
    let func = b.build();
    #[cfg(target_arch = "aarch64")]
    let jit = JitFunction::compile_with_backend_and_config::<TestInternalConfig, Arm64Backend>(
        &func,
        &[],
        None,
    );
    #[cfg(target_arch = "x86_64")]
    let jit = JitFunction::compile_with_backend_and_config::<TestInternalConfig, X64Backend>(
        &func,
        &[],
        None,
    );
    assert_eq!(unsafe { call_jit(jit.as_ptr(), &[]) }, 13);
}

#[test]
fn compile_with_explicit_alternate_config_shape() {
    let mut b = FunctionBuilder::new("ret9", &[], Some(Type::I64));
    let v = b.iconst(Type::I64, 9);
    b.ret(v);
    let func = b.build();
    let jit = JitFunction::compile_with_config::<TestCAbiConfig>(&func, &[]);
    assert_eq!(unsafe { call_jit(jit.as_ptr(), &[]) }, 9);
}

#[test]
fn compile_with_wrapped_frame_layout() {
    let mut b = FunctionBuilder::new("ret19", &[], Some(Type::I64));
    let v = b.iconst(Type::I64, 19);
    b.ret(v);
    let func = b.build();
    let jit = JitFunction::compile_with_config::<TestWrappedFrameConfig>(&func, &[]);
    assert_eq!(unsafe { call_jit(jit.as_ptr(), &[]) }, 19);
}

#[test]
fn x64_backend_stub_is_present() {
    assert_eq!(X64Backend::name(), "x86_64");
}

#[test]
fn x64_backend_can_compile_simple_integer_function() {
    let mut b = FunctionBuilder::new("ret17_x64", &[], Some(Type::I64));
    let v = b.iconst(Type::I64, 17);
    b.ret(v);
    let func = b.build();
    let jit = JitFunction::compile_with_backend_and_config::<TestInternalConfig, X64Backend>(
        &func,
        &[],
        None,
    );
    assert!(!jit.as_ptr().is_null());
}

#[test]
fn x64_backend_can_compile_simple_integer_add() {
    let mut b = FunctionBuilder::new("add2_x64", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let c = b.block_param(entry, 1);
    let sum = b.add(a, c);
    b.ret(sum);
    let func = b.build();
    let jit = JitFunction::compile_with_backend_and_config::<TestInternalConfig, X64Backend>(
        &func,
        &[],
        None,
    );
    assert!(!jit.as_ptr().is_null());
}

#[test]
fn x64_backend_can_compile_integer_compare() {
    let mut b = FunctionBuilder::new("eq_x64", &[Type::I64, Type::I64], Some(Type::I8));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let c = b.block_param(entry, 1);
    let eq = b.icmp(CmpOp::Eq, a, c);
    b.ret(eq);
    let func = b.build();
    let jit = JitFunction::compile_with_backend_and_config::<TestInternalConfig, X64Backend>(
        &func,
        &[],
        None,
    );
    assert!(!jit.as_ptr().is_null());
}

#[test]
fn x64_backend_can_compile_integer_negation() {
    let mut b = FunctionBuilder::new("neg_x64", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let neg = b.neg(a);
    b.ret(neg);
    let func = b.build();
    let jit = JitFunction::compile_with_backend_and_config::<TestInternalConfig, X64Backend>(
        &func,
        &[],
        None,
    );
    assert!(!jit.as_ptr().is_null());
}

#[test]
fn stack_map_transport_records_safepoint_metadata() {
    let mut b = FunctionBuilder::new("stackmap_meta", &[Type::GcPtr], Some(Type::GcPtr));
    let entry = b.entry_block();
    let arg = b.block_param(entry, 0);
    b.safepoint(&[arg]);
    b.ret(arg);
    let func = b.build();
    let jit = JitFunction::compile_with_config::<TestStackMapConfig>(&func, &[]);
    let safepoints = jit.safepoints();
    assert_eq!(safepoints.len(), 1);
    assert!(!safepoints[0].root_slots.is_empty());
}

#[test]
fn shadow_stack_transport_records_shadow_root_slots() {
    let mut b = FunctionBuilder::new("shadow_meta", &[Type::GcPtr], Some(Type::GcPtr));
    let entry = b.entry_block();
    let arg = b.block_param(entry, 0);
    b.safepoint(&[arg]);
    b.ret(arg);
    let func = b.build();
    let jit = JitFunction::compile_with_config::<TestShadowStackConfig>(&func, &[]);
    let safepoints = jit.safepoints();
    assert_eq!(safepoints.len(), 1);
    assert!(!safepoints[0].root_slots.is_empty());
}

#[test]
fn stack_map_transport_uses_safepoint_index_payloads() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static LAST_PAYLOAD: AtomicUsize = AtomicUsize::new(usize::MAX);

    extern "C" fn test_handler(_frame_ptr: *mut u8, payload: usize) {
        LAST_PAYLOAD.store(payload, Ordering::SeqCst);
    }

    let mut b = FunctionBuilder::new("stackmap_handler", &[Type::GcPtr], Some(Type::GcPtr));
    let entry = b.entry_block();
    let arg = b.block_param(entry, 0);
    b.safepoint(&[arg]);
    b.ret(arg);
    let func = b.build();

    LAST_PAYLOAD.store(usize::MAX, Ordering::SeqCst);
    let jit = JitFunction::compile_with_config_and_gc::<TestStackMapConfig>(
        &func,
        &[],
        Some(test_handler as u64),
    );
    assert_eq!(
        jit.handler_payload_kind(),
        SafepointHandlerPayloadKind::SafepointIndex
    );
    assert_eq!(unsafe { call_jit(jit.as_ptr(), &[7]) }, 7);
    assert_eq!(LAST_PAYLOAD.load(Ordering::SeqCst), 0);
}

#[test]
fn alternate_calling_convention_controls_arg_window() {
    let params = vec![Type::I64; 10];
    let mut b = FunctionBuilder::new("sum10", &params, Some(Type::I64));
    let entry = b.entry_block();
    let mut acc = b.block_param(entry, 0);
    for i in 1..10 {
        let p = b.block_param(entry, i);
        acc = b.add(acc, p);
    }
    b.ret(acc);
    let func = b.build();
    let jit = JitFunction::compile_with_config::<TestCAbiConfig>(&func, &[]);
    let args = [1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    #[cfg(target_arch = "aarch64")]
    let reg_limit = 8;
    #[cfg(target_arch = "x86_64")]
    let reg_limit = 6;
    assert_eq!(
        unsafe { call_jit_with_reg_limit(jit.as_ptr(), &args, reg_limit) },
        55
    );
}

#[test]
fn internal_calling_convention_handles_more_than_16_args() {
    let params = vec![Type::I64; 20];
    let mut b = FunctionBuilder::new("sum20", &params, Some(Type::I64));
    let entry = b.entry_block();
    let mut acc = b.block_param(entry, 0);
    for i in 1..20 {
        let p = b.block_param(entry, i);
        acc = b.add(acc, p);
    }
    b.ret(acc);
    let func = b.build();
    let args = [
        1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    ];
    assert_eq!(run_jit_with_config::<TestInternalConfig>(&func, &args), 210);
}

#[test]
fn block_param_assignment_works_under_configured_lowering() {
    let mut b = FunctionBuilder::new("diamond_cfg", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let arg = b.block_param(entry, 0);
    let cond = b.iconst(Type::I8, 1);
    let then_bb = b.create_block(&[]);
    let else_bb = b.create_block(&[]);
    let merge = b.create_block(&[Type::I64]);

    b.br_if(cond, then_bb, &[], else_bb, &[]);

    b.switch_to_block(then_bb);
    let one = b.iconst(Type::I64, 41);
    b.jump(merge, &[one]);

    b.switch_to_block(else_bb);
    let hundred = b.iconst(Type::I64, 100);
    let two = b.add(arg, hundred);
    b.jump(merge, &[two]);

    b.switch_to_block(merge);
    let result = b.block_param(merge, 0);
    let one_more = b.iconst(Type::I64, 1);
    let final_result = b.add(result, one_more);
    b.ret(final_result);

    let func = b.build();
    assert_eq!(run_jit_with_config::<TestInternalConfig>(&func, &[7]), 42);
}

// Test removed: NanBox + PreciseStackRoots is now a valid config
// (NanBox uses precise stack maps with runtime tag checking).

#[test]
fn overflow_check_overflows_in_jit() {
    let mut b = FunctionBuilder::new("ov", &[Type::I64, Type::I64], Some(Type::I8));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let r = b.overflow_check(OverflowOp::SAdd, a, bb);
    b.ret(r);
    let func = b.build();

    assert_eq!(run_jit(&func, &[1, 2]), 0);
    assert_eq!(run_jit(&func, &[i64::MAX as u64, 1]), 1);
}

#[test]
fn guard_deopt_returns_jit_outcome() {
    let mut b = FunctionBuilder::new("guard", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let zero = b.iconst(Type::I8, 0);
    let deopt = b.create_deopt(42, "jit guard");
    b.guard(zero, deopt, &[a, bb]);
    b.ret(a);
    let func = b.build();

    let jit = JitFunction::compile::<LowBit<3>>(&func, &[]);
    assert_eq!(
        jit.call_outcome(&[10, 20]),
        JitOutcome::Deopt {
            deopt_id: deopt,
            resume_point: 42,
            live_values: vec![10, 20],
        }
    );
}

#[test]
fn invoke_internal_declared_late_via_import_module_func() {
    // Mirrors clojure's lower_try shape: outer is defined first;
    // callee is declared and defined DURING outer compilation;
    // outer's `extern_funcs` snapshot doesn't see it; the new
    // `import_module_func` API teaches the outer fb about it.
    let mut mb = ModuleBuilder::new();
    let f_main = mb.declare_func("main", &[], Some(Type::I64));

    // Start defining main first.
    let mut main_fb = mb.define_func(f_main);
    let _entry = main_fb.entry_block();

    // NOW (mid-stream) declare and define callee.
    let f_callee = mb.declare_func("callee", &[], Some(Type::I64));
    main_fb.import_module_func(
        f_callee,
        "callee",
        dynir::types::Signature {
            params: vec![],
            ret: Some(Type::I64),
        },
    );
    let mut callee_fb = mb.define_func(f_callee);
    let _e = callee_fb.entry_block();
    let body = callee_fb.iconst(Type::I64, 42);
    callee_fb.ret(body);
    mb.finish_func(f_callee, callee_fb);

    // Back in main: invoke callee.
    let normal = main_fb.create_block(&[Type::I64]);
    let exception = main_fb.create_block(&[Type::I64]);
    main_fb.invoke(f_callee, &[], normal, &[], exception, &[]);
    main_fb.switch_to_block(normal);
    let rv = main_fb.block_param(normal, 0);
    main_fb.ret(rv);
    main_fb.switch_to_block(exception);
    let _exc = main_fb.block_param(exception, 0);
    let sentinel = main_fb.iconst(Type::I64, -1);
    main_fb.ret(sentinel);
    mb.finish_func(f_main, main_fb);

    let module = mb.build();
    let jit = JitModule::compile::<NanBox>(&module, &[]);
    assert_eq!(jit.call_outcome(f_main, &[]), JitOutcome::Value(42));
}

#[test]
fn invoke_internal_declared_late_with_safepoint_inst_and_linscan() {
    // Closer to clojure's setup: emits a Safepoint([]) inst inside
    // main's entry block before the invoke, AND uses the linear-scan
    // register allocator.
    let mut mb = ModuleBuilder::new();
    let f_main = mb.declare_func("main", &[], Some(Type::I64));
    // BISECT: declare callee BEFORE define_func(main) to test if
    // late-import is the issue (vs declaration order itself).
    let f_callee = mb.declare_func("callee", &[], Some(Type::I64));

    let mut main_fb = mb.define_func(f_main);
    let _entry = main_fb.entry_block();

    // No-op: callee was already declared before define_func, so the
    // outer fb's snapshot already includes it. No import needed.
    let _f_callee_check = f_callee;
    let mut callee_fb = mb.define_func(f_callee);
    let _e = callee_fb.entry_block();
    let body = callee_fb.iconst(Type::I64, 0x4045000000000000_i64); // 42.0 NaN-boxed
    callee_fb.ret(body);
    mb.finish_func(f_callee, callee_fb);

    let normal = main_fb.create_block(&[Type::I64]);
    let exception = main_fb.create_block(&[Type::I64]);
    main_fb.invoke(f_callee, &[], normal, &[], exception, &[]);
    main_fb.switch_to_block(normal);
    let rv = main_fb.block_param(normal, 0);
    main_fb.ret(rv);
    main_fb.switch_to_block(exception);
    let _exc = main_fb.block_param(exception, 0);
    let neg = main_fb.iconst(Type::I64, -1);
    main_fb.ret(neg);
    mb.finish_func(f_main, main_fb);

    let module = mb.build();
    // BISECT: GreedyRegState (default) instead of LinearScan.
    let jit = JitModule::compile_with_regalloc::<
        DefaultJitConfig<NanBox>,
        crate::backend::Arm64Backend,
        crate::regalloc::GreedyRegState,
    >(&module, &[], None);
    assert_eq!(
        jit.call_outcome(f_main, &[]),
        JitOutcome::Value(0x4045000000000000)
    );
}

#[test]
fn invoke_internal_declared_late_with_control_aware_safepoint() {
    // Reproducer for clojure's setup: control_aware mode (with
    // safepoint handler), late-declared callee imported via
    // import_module_func, then invoked.
    let mut mb = ModuleBuilder::new();
    let f_main = mb.declare_func("main", &[], Some(Type::I64));

    let mut main_fb = mb.define_func(f_main);
    let _entry = main_fb.entry_block();

    let f_callee = mb.declare_func("callee", &[], Some(Type::I64));
    main_fb.import_module_func(
        f_callee,
        "callee",
        dynir::types::Signature {
            params: vec![],
            ret: Some(Type::I64),
        },
    );
    let mut callee_fb = mb.define_func(f_callee);
    let _e = callee_fb.entry_block();
    let body = callee_fb.iconst(Type::I64, 42);
    callee_fb.ret(body);
    mb.finish_func(f_callee, callee_fb);

    let normal = main_fb.create_block(&[Type::I64]);
    let exception = main_fb.create_block(&[]);
    main_fb.invoke(f_callee, &[], normal, &[], exception, &[]);
    main_fb.switch_to_block(normal);
    let rv = main_fb.block_param(normal, 0);
    main_fb.ret(rv);
    main_fb.switch_to_block(exception);
    let neg = main_fb.iconst(Type::I64, -1);
    main_fb.ret(neg);
    mb.finish_func(f_main, main_fb);

    let module = mb.build();
    // Control-aware mode (safepoint handler set), like clojure.
    let dummy_safepoint_handler = 0u64;
    let jit = JitModule::compile_with_config_and_gc::<DefaultJitConfig<NanBox>>(
        &module,
        &[],
        Some(dummy_safepoint_handler),
    );
    assert_eq!(jit.call_outcome(f_main, &[]), JitOutcome::Value(42));
}

#[test]
fn invoke_internal_normal_path_in_jit_module() {
    let mut mb = ModuleBuilder::new();
    let f_callee = mb.declare_func("callee", &[Type::I64], Some(Type::I64));
    let f_main = mb.declare_func("main", &[Type::I64], Some(Type::I64));

    let mut fb = mb.define_func(f_callee);
    let entry = fb.entry_block();
    let x = fb.block_param(entry, 0);
    let three = fb.iconst(Type::I64, 3);
    let r = fb.mul(x, three);
    fb.ret(r);
    mb.finish_func(f_callee, fb);

    let mut fb = mb.define_func(f_main);
    let entry = fb.entry_block();
    let x = fb.block_param(entry, 0);
    let normal = fb.create_block(&[Type::I64]);
    let exception = fb.create_block(&[]);
    fb.invoke(f_callee, &[x], normal, &[], exception, &[]);
    fb.switch_to_block(normal);
    let rv = fb.block_param(normal, 0);
    fb.ret(rv);
    fb.switch_to_block(exception);
    let neg = fb.iconst(Type::I64, -1);
    fb.ret(neg);
    mb.finish_func(f_main, fb);

    let module = mb.build();
    let jit = JitModule::compile::<LowBit<3>>(&module, &[]);
    assert_eq!(jit.call_outcome(f_main, &[14]), JitOutcome::Value(42));
}

#[test]
fn invoke_indirect_normal_path_in_jit() {
    let mut callee_builder = FunctionBuilder::new("callee", &[Type::I64], Some(Type::I64));
    let callee_entry = callee_builder.entry_block();
    let x = callee_builder.block_param(callee_entry, 0);
    let ten = callee_builder.iconst(Type::I64, 10);
    let mul = callee_builder.mul(x, ten);
    callee_builder.ret(mul);
    let callee = callee_builder.build();
    let callee_jit = JitFunction::compile::<LowBit<3>>(&callee, &[]);

    let mut b = FunctionBuilder::new("inv_ind", &[Type::Ptr], Some(Type::I64));
    let entry = b.entry_block();
    let callee_ptr = b.block_param(entry, 0);
    let arg = b.iconst(Type::I64, 5);
    let normal = b.create_block(&[Type::I64]);
    let exception = b.create_block(&[]);
    b.invoke_indirect(
        callee_ptr,
        &[arg],
        Some(Type::I64),
        normal,
        &[],
        exception,
        &[],
    );
    b.switch_to_block(normal);
    let ret = b.block_param(normal, 0);
    b.ret(ret);
    b.switch_to_block(exception);
    let neg = b.iconst(Type::I64, -1);
    b.ret(neg);
    let func = b.build();

    let jit = JitFunction::compile::<LowBit<3>>(&func, &[]);
    assert_eq!(
        jit.call_outcome(&[callee_jit.as_ptr() as u64]),
        JitOutcome::Value(50)
    );
}

#[test]
fn return_const_i32() {
    let mut b = FunctionBuilder::new("ret99", &[], Some(Type::I32));
    let v = b.iconst(Type::I32, 99);
    b.ret(v);
    let func = b.build();
    assert_eq!(run_jit(&func, &[]) as i32, 99);
}

#[test]
fn return_zero() {
    let mut b = FunctionBuilder::new("ret0", &[], Some(Type::I64));
    let v = b.iconst(Type::I64, 0);
    b.ret(v);
    let func = b.build();
    assert_eq!(run_jit(&func, &[]), 0);
}

#[test]
fn return_large_const() {
    let mut b = FunctionBuilder::new("retlarge", &[], Some(Type::I64));
    let v = b.iconst(Type::I64, 0x1234_5678_9ABC_DEF0u64 as i64);
    b.ret(v);
    let func = b.build();
    assert_eq!(run_jit(&func, &[]), 0x1234_5678_9ABC_DEF0);
}

// ── Phase 2: Arithmetic + args ─────────────────────────────────────

#[test]
fn add_two() {
    let mut b = FunctionBuilder::new("add", &[Type::I32, Type::I32], Some(Type::I32));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let sum = b.add(a, bb);
    b.ret(sum);
    let func = b.build();
    assert_eq!(run_jit(&func, &[10, 32]) as i32, 42);
}

#[test]
fn arithmetic() {
    let mut b = FunctionBuilder::new("calc", &[Type::I32, Type::I32], Some(Type::I32));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let prod = b.mul(a, bb);
    let sum = b.add(prod, a);
    b.ret(sum);
    let func = b.build();
    // 5 * 7 + 5 = 40
    assert_eq!(run_jit(&func, &[5, 7]) as i32, 40);
}

#[test]
fn sub_mul() {
    let mut b = FunctionBuilder::new("arith", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let sum = b.add(a, bb);
    let diff = b.sub(sum, bb);
    let prod = b.mul(diff, bb);
    b.ret(prod);
    let func = b.build();
    // (10 + 3 - 3) * 3 = 30
    assert_eq!(run_jit(&func, &[10, 3]), 30);
}

#[test]
fn sdiv_udiv() {
    let mut b = FunctionBuilder::new("sdiv", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let q = b.sdiv(a, bb);
    b.ret(q);
    let func = b.build();
    assert_eq!(run_jit(&func, &[(-10i64) as u64, 3]), (-3i64) as u64);

    let mut b = FunctionBuilder::new("udiv", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let q = b.udiv(a, bb);
    b.ret(q);
    let func = b.build();
    assert_eq!(run_jit(&func, &[100, 5]), 20);
}

#[test]
fn bitwise_ops() {
    let mut b = FunctionBuilder::new("bits", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let r = b.and(a, bb);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_jit(&func, &[0xFF, 0x0F]), 0x0F);

    let mut b = FunctionBuilder::new("or", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let r = b.or(a, bb);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_jit(&func, &[0xF0, 0x0F]), 0xFF);

    let mut b = FunctionBuilder::new("xor", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let r = b.xor(a, bb);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_jit(&func, &[0xFF, 0xFF]), 0);
}

#[test]
fn shift_ops() {
    let mut b = FunctionBuilder::new("shl", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let r = b.shl(a, bb);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_jit(&func, &[1, 4]), 16);
}

#[test]
fn ashr_sign_extension() {
    let mut b = FunctionBuilder::new("ashr", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let r = b.ashr(a, bb);
    b.ret(r);
    let func = b.build();
    let neg_eight = (-8i64) as u64;
    let result = run_jit(&func, &[neg_eight, 1]) as i64;
    assert_eq!(result, -4);
}

#[test]
fn unary_neg() {
    let mut b = FunctionBuilder::new("neg", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let r = b.neg(a);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_jit(&func, &[5]), (-5i64) as u64);
}

// ── Phase 3: Control flow ──────────────────────────────────────────

#[test]
fn if_else() {
    // if arg0 > arg1 then arg0 else arg1
    let wat = r#"(module
        (func (export "max") (param i32) (param i32) (result i32)
            local.get 0
            local.get 1
            i32.gt_s
            if (result i32)
                local.get 0
            else
                local.get 1
            end))"#;
    let wasm = wat::parse_str(wat).expect("parse WAT");
    let (func, _) = wasm2dynir::translate_wasm(&wasm).expect("translate");
    assert_eq!(run_jit(&func, &[10, 20]) as i32, 20);
    assert_eq!(run_jit(&func, &[30, 20]) as i32, 30);
}

#[test]
fn diamond_br_if() {
    let mut b = FunctionBuilder::new("diamond", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let arg = b.block_param(entry, 0);

    let then_bb = b.create_block(&[]);
    let else_bb = b.create_block(&[]);
    let merge = b.create_block(&[Type::I64]);
    let zero = b.iconst(Type::I64, 0);
    let cond = b.icmp(CmpOp::Ne, arg, zero);
    b.br_if(cond, then_bb, &[], else_bb, &[]);

    b.switch_to_block(then_bb);
    let one = b.iconst(Type::I64, 1);
    b.jump(merge, &[one]);

    b.switch_to_block(else_bb);
    let two = b.iconst(Type::I64, 2);
    b.jump(merge, &[two]);

    b.switch_to_block(merge);
    let result = b.block_param(merge, 0);
    b.ret(result);

    let func = b.build();
    assert_eq!(run_jit(&func, &[1]), 1);
    assert_eq!(run_jit(&func, &[0]), 2);
}

// ── Phase 4: Loops + spilling ──────────────────────────────────────

#[test]
fn simple_loop() {
    let wat = r#"(module
        (func (export "sum") (param $n i32) (result i32)
            (local $i i32)
            (local $acc i32)
            i32.const 0
            local.set $i
            i32.const 0
            local.set $acc
            block $exit
                loop $loop
                    local.get $i
                    local.get $n
                    i32.ge_s
                    br_if $exit
                    local.get $acc
                    local.get $i
                    i32.add
                    local.set $acc
                    local.get $i
                    i32.const 1
                    i32.add
                    local.set $i
                    br $loop
                end
            end
            local.get $acc))"#;
    let wasm = wat::parse_str(wat).expect("parse WAT");
    let (func, _) = wasm2dynir::translate_wasm(&wasm).expect("translate");
    assert_eq!(run_jit(&func, &[10]) as i32, 45);
    assert_eq!(run_jit(&func, &[0]) as i32, 0);
    assert_eq!(run_jit(&func, &[100]) as i32, 4950);
}

#[test]
fn fibonacci() {
    let wat = r#"(module
        (func (export "fib") (param $n i32) (result i32)
            (local $a i32)
            (local $b i32)
            (local $i i32)
            (local $tmp i32)
            i32.const 0
            local.set $a
            i32.const 1
            local.set $b
            i32.const 0
            local.set $i
            block $exit
                loop $loop
                    local.get $i
                    local.get $n
                    i32.ge_s
                    br_if $exit
                    local.get $a
                    local.get $b
                    i32.add
                    local.set $tmp
                    local.get $b
                    local.set $a
                    local.get $tmp
                    local.set $b
                    local.get $i
                    i32.const 1
                    i32.add
                    local.set $i
                    br $loop
                end
            end
            local.get $a))"#;
    let wasm = wat::parse_str(wat).expect("parse WAT");
    let (func, _) = wasm2dynir::translate_wasm(&wasm).expect("translate");
    assert_eq!(run_jit(&func, &[0]) as i32, 0);
    assert_eq!(run_jit(&func, &[1]) as i32, 1);
    assert_eq!(run_jit(&func, &[10]) as i32, 55);
    assert_eq!(run_jit(&func, &[20]) as i32, 6765);
}

#[test]
fn factorial() {
    let wat = r#"(module
        (func (export "fact") (param $n i32) (result i32)
            (local $result i32)
            i32.const 1
            local.set $result
            block $exit
                loop $loop
                    local.get $n
                    i32.const 1
                    i32.le_s
                    br_if $exit
                    local.get $result
                    local.get $n
                    i32.mul
                    local.set $result
                    local.get $n
                    i32.const 1
                    i32.sub
                    local.set $n
                    br $loop
                end
            end
            local.get $result))"#;
    let wasm = wat::parse_str(wat).expect("parse WAT");
    let (func, _) = wasm2dynir::translate_wasm(&wasm).expect("translate");
    assert_eq!(run_jit(&func, &[1]) as i32, 1);
    assert_eq!(run_jit(&func, &[5]) as i32, 120);
    assert_eq!(run_jit(&func, &[10]) as i32, 3628800);
}

#[test]
fn nested_if() {
    let wat = r#"(module
        (func (export "clamp") (param $x i32) (param $lo i32) (param $hi i32) (result i32)
            local.get $x
            local.get $lo
            i32.lt_s
            if (result i32)
                local.get $lo
            else
                local.get $x
                local.get $hi
                i32.gt_s
                if (result i32)
                    local.get $hi
                else
                    local.get $x
                end
            end))"#;
    let wasm = wat::parse_str(wat).expect("parse WAT");
    let (func, _) = wasm2dynir::translate_wasm(&wasm).expect("translate");
    assert_eq!(run_jit(&func, &[5, 0, 10]) as i32, 5);
    assert_eq!(run_jit(&func, &[(-5i32) as u64, 0, 10]) as i32, 0);
    assert_eq!(run_jit(&func, &[15, 0, 10]) as i32, 10);
}

#[test]
fn void_if() {
    let wat = r#"(module
        (func (export "abs") (param $x i32) (result i32)
            local.get $x
            i32.const 0
            i32.lt_s
            if
                i32.const 0
                local.get $x
                i32.sub
                local.set $x
            end
            local.get $x))"#;
    let wasm = wat::parse_str(wat).expect("parse WAT");
    let (func, _) = wasm2dynir::translate_wasm(&wasm).expect("translate");
    assert_eq!(run_jit(&func, &[5]) as i32, 5);
    assert_eq!(run_jit(&func, &[(-5i32) as u64 & 0xFFFFFFFF]) as i32, 5);
}

#[test]
fn local_tee() {
    let wat = r#"(module
        (func (export "test") (param $x i32) (result i32)
            local.get $x
            i32.const 10
            i32.add
            local.tee $x
            local.get $x
            i32.add))"#;
    let wasm = wat::parse_str(wat).expect("parse WAT");
    let (func, _) = wasm2dynir::translate_wasm(&wasm).expect("translate");
    assert_eq!(run_jit(&func, &[5]) as i32, 30);
}

// ── Phase 5: 64-bit ────────────────────────────────────────────────

#[test]
fn i64_arithmetic() {
    let wat = r#"(module
        (func (export "add64") (param i64) (param i64) (result i64)
            local.get 0
            local.get 1
            i64.add))"#;
    let wasm = wat::parse_str(wat).expect("parse WAT");
    let (func, _) = wasm2dynir::translate_wasm(&wasm).expect("translate");
    assert_eq!(
        run_jit(&func, &[1_000_000_000_000, 2_000_000_000_000]) as i64,
        3_000_000_000_000
    );
}

// ── Phase 6: Floats ────────────────────────────────────────────────

#[test]
fn float_arithmetic() {
    let mut b = FunctionBuilder::new("fadd", &[Type::F64, Type::F64], Some(Type::F64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let sum = b.fadd(a, bb);
    b.ret(sum);
    let func = b.build();
    let result = f64::from_bits(run_jit(&func, &[1.5f64.to_bits(), 2.5f64.to_bits()]));
    assert_eq!(result, 4.0);
}

#[test]
fn float_sub_mul_div() {
    let mut b = FunctionBuilder::new("fops", &[Type::F64, Type::F64], Some(Type::F64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let diff = b.fsub(a, bb);
    let prod = b.fmul(diff, bb);
    let quot = b.fdiv(prod, a);
    b.ret(quot);
    let func = b.build();
    let result = f64::from_bits(run_jit(&func, &[10.0f64.to_bits(), 3.0f64.to_bits()]));
    assert!((result - 2.1).abs() < 1e-10);
}

#[test]
fn float_neg() {
    let mut b = FunctionBuilder::new("fneg", &[Type::F64], Some(Type::F64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let r = b.fneg(a);
    b.ret(r);
    let func = b.build();
    let result = f64::from_bits(run_jit(&func, &[3.0f64.to_bits()]));
    assert_eq!(result, -3.0);
}

#[test]
fn int_to_float_and_back() {
    let mut b = FunctionBuilder::new("itof", &[Type::I64], Some(Type::F64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let r = b.int_to_float(a);
    b.ret(r);
    let func = b.build();
    let result = f64::from_bits(run_jit(&func, &[42]));
    assert_eq!(result, 42.0);

    let mut b = FunctionBuilder::new("ftoi", &[Type::F64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let r = b.float_to_int(a);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_jit(&func, &[42.9f64.to_bits()]), 42);
}

#[test]
fn f64_const() {
    let mut b = FunctionBuilder::new("fconst", &[], Some(Type::F64));
    let v = b.f64const(3.14);
    b.ret(v);
    let func = b.build();
    let result = f64::from_bits(run_jit(&func, &[]));
    assert!((result - 3.14).abs() < 1e-10);
}

// ── Conversions ────────────────────────────────────────────────────

#[test]
fn sext_zext_trunc() {
    // sext i8 -1 -> i64 -1
    let mut b = FunctionBuilder::new("sext", &[Type::I8], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let r = b.sext(a, Type::I64);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_jit(&func, &[0xFF]), (-1i64) as u64);

    // zext i8 0xFF -> i64 255
    let mut b = FunctionBuilder::new("zext", &[Type::I8], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let r = b.zext(a, Type::I64);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_jit(&func, &[0xFF]), 0xFF);

    // trunc i64 -> i8
    let mut b = FunctionBuilder::new("trunc", &[Type::I64], Some(Type::I8));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let r = b.trunc(a, Type::I8);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_jit(&func, &[0x1234]), 0x34);
}

// ── Select ─────────────────────────────────────────────────────────

#[test]
fn select_true_false() {
    let mut b = FunctionBuilder::new("sel", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let one = b.iconst(Type::I8, 1);
    let r = b.select(one, a, bb);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_jit(&func, &[10, 20]), 10);

    let mut b = FunctionBuilder::new("sel", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let zero = b.iconst(Type::I8, 0);
    let r = b.select(zero, a, bb);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_jit(&func, &[10, 20]), 20);
}

// ── Comparison ─────────────────────────────────────────────────────

#[test]
fn icmp_all_ops() {
    fn test_icmp(op: CmpOp, a: i64, b_val: i64) -> u64 {
        let mut b = FunctionBuilder::new("cmp", &[Type::I64, Type::I64], Some(Type::I8));
        let entry = b.entry_block();
        let va = b.block_param(entry, 0);
        let vb = b.block_param(entry, 1);
        let r = b.icmp(op, va, vb);
        b.ret(r);
        let func = b.build();
        run_jit(&func, &[a as u64, b_val as u64])
    }

    assert_eq!(test_icmp(CmpOp::Eq, 5, 5), 1);
    assert_eq!(test_icmp(CmpOp::Eq, 5, 6), 0);
    assert_eq!(test_icmp(CmpOp::Ne, 5, 6), 1);
    assert_eq!(test_icmp(CmpOp::Slt, -1, 1), 1);
    assert_eq!(test_icmp(CmpOp::Sle, 5, 5), 1);
    assert_eq!(test_icmp(CmpOp::Sgt, 1, -1), 1);
    assert_eq!(test_icmp(CmpOp::Sge, 5, 5), 1);
    assert_eq!(test_icmp(CmpOp::Ult, 1, 2), 1);
    assert_eq!(test_icmp(CmpOp::Ule, 2, 2), 1);
    assert_eq!(test_icmp(CmpOp::Ugt, 3, 2), 1);
    assert_eq!(test_icmp(CmpOp::Uge, 2, 2), 1);
}

// ── Switch ─────────────────────────────────────────────────────────

#[test]
fn switch_dispatch() {
    let mut b = FunctionBuilder::new("sw", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let val = b.block_param(entry, 0);

    let case1 = b.create_block(&[]);
    let case2 = b.create_block(&[]);
    let default = b.create_block(&[]);
    let merge = b.create_block(&[Type::I64]);

    b.switch(val, &[(1, case1, &[]), (2, case2, &[])], default, &[]);

    b.switch_to_block(case1);
    let ten = b.iconst(Type::I64, 10);
    b.jump(merge, &[ten]);

    b.switch_to_block(case2);
    let twenty = b.iconst(Type::I64, 20);
    b.jump(merge, &[twenty]);

    b.switch_to_block(default);
    let nn = b.iconst(Type::I64, 99);
    b.jump(merge, &[nn]);

    b.switch_to_block(merge);
    let r = b.block_param(merge, 0);
    b.ret(r);

    let func = b.build();
    assert_eq!(run_jit(&func, &[1]), 10);
    assert_eq!(run_jit(&func, &[2]), 20);
    assert_eq!(run_jit(&func, &[3]), 99);
}

#[test]
fn overflow_check_signed_add_overflow() {
    let mut b = FunctionBuilder::new("ov", &[Type::I64, Type::I64], Some(Type::I8));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let c = b.block_param(entry, 1);
    let overflowed = b.overflow_check(OverflowOp::SAdd, a, c);
    b.ret(overflowed);
    let func = b.build();

    assert_eq!(run_jit(&func, &[i64::MAX as u64, 1]), 1);
    assert_eq!(run_jit(&func, &[3, 4]), 0);
}

#[test]
fn guard_deopt_returns_outcome() {
    let mut b = FunctionBuilder::new("guard", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let c = b.block_param(entry, 1);
    let deopt = b.create_deopt(200, "guard fail");
    let zero = b.iconst(Type::I8, 0);
    b.guard(zero, deopt, &[a, c]);
    b.ret(a);
    let func = b.build();

    assert_eq!(
        run_jit_outcome(&func, &[10, 20]),
        JitOutcome::Deopt {
            deopt_id: DeoptId::from_index(0),
            resume_point: 200,
            live_values: vec![10, 20],
        }
    );
}

#[test]
fn internal_call_propagates_deopt_outcome() {
    let mut mb = ModuleBuilder::new();
    let callee = mb.declare_func("callee", &[Type::I64], Some(Type::I64));
    let main = mb.declare_func("main", &[Type::I64], Some(Type::I64));

    let mut fb = mb.define_func(callee);
    let entry = fb.entry_block();
    let x = fb.block_param(entry, 0);
    let deopt = fb.create_deopt(77, "callee deopt");
    let zero = fb.iconst(Type::I8, 0);
    fb.guard(zero, deopt, &[x]);
    fb.ret(x);
    mb.finish_func(callee, fb);

    let mut fb = mb.define_func(main);
    let entry = fb.entry_block();
    let x = fb.block_param(entry, 0);
    let rv = fb.call(callee, &[x]).unwrap();
    fb.ret(rv);
    mb.finish_func(main, fb);

    let module = mb.build();
    let jit = JitModule::compile::<NanBox>(&module, &[]);
    assert_eq!(
        jit.call_outcome(main, &[55]),
        JitOutcome::Deopt {
            deopt_id: DeoptId::from_index(0),
            resume_point: 77,
            live_values: vec![55],
        }
    );
}

#[test]
fn invoke_internal_normal_path() {
    let mut mb = ModuleBuilder::new();
    let callee = mb.declare_func("callee", &[Type::I64], Some(Type::I64));
    let main = mb.declare_func("main", &[Type::I64], Some(Type::I64));

    let mut fb = mb.define_func(callee);
    let entry = fb.entry_block();
    let x = fb.block_param(entry, 0);
    let three = fb.iconst(Type::I64, 3);
    let rv = fb.mul(x, three);
    fb.ret(rv);
    mb.finish_func(callee, fb);

    let mut fb = mb.define_func(main);
    let entry = fb.entry_block();
    let x = fb.block_param(entry, 0);
    let normal = fb.create_block(&[Type::I64]);
    let exception = fb.create_block(&[]);
    fb.invoke(callee, &[x], normal, &[], exception, &[]);
    fb.switch_to_block(normal);
    let rv = fb.block_param(normal, 0);
    fb.ret(rv);
    fb.switch_to_block(exception);
    let neg = fb.iconst(Type::I64, -1);
    fb.ret(neg);
    mb.finish_func(main, fb);

    let module = mb.build();
    let jit = JitModule::compile::<NanBox>(&module, &[]);
    assert_eq!(jit.call_outcome(main, &[14]), JitOutcome::Value(42));
}

#[test]
fn invoke_internal_propagates_deopt() {
    let mut mb = ModuleBuilder::new();
    let callee = mb.declare_func("callee", &[Type::I64], Some(Type::I64));
    let main = mb.declare_func("main", &[Type::I64], Some(Type::I64));

    let mut fb = mb.define_func(callee);
    let entry = fb.entry_block();
    let x = fb.block_param(entry, 0);
    let deopt = fb.create_deopt(33, "invoke deopt");
    let zero = fb.iconst(Type::I8, 0);
    fb.guard(zero, deopt, &[x]);
    fb.ret(x);
    mb.finish_func(callee, fb);

    let mut fb = mb.define_func(main);
    let entry = fb.entry_block();
    let x = fb.block_param(entry, 0);
    let normal = fb.create_block(&[Type::I64]);
    let exception = fb.create_block(&[]);
    fb.invoke(callee, &[x], normal, &[], exception, &[]);
    fb.switch_to_block(normal);
    let rv = fb.block_param(normal, 0);
    fb.ret(rv);
    fb.switch_to_block(exception);
    let neg = fb.iconst(Type::I64, -1);
    fb.ret(neg);
    mb.finish_func(main, fb);

    let module = mb.build();
    let jit = JitModule::compile::<NanBox>(&module, &[]);
    assert_eq!(
        jit.call_outcome(main, &[9]),
        JitOutcome::Deopt {
            deopt_id: DeoptId::from_index(0),
            resume_point: 33,
            live_values: vec![9],
        }
    );
}

#[cfg(target_arch = "aarch64")]
#[test]
fn invoke_internal_exception_path() {
    let mut mb = ModuleBuilder::new();
    let callee = mb.declare_func("callee", &[Type::Ptr], Some(Type::I64));
    let main = mb.declare_func("main", &[Type::Ptr], Some(Type::I64));

    let mut fb = mb.define_func(callee);
    let entry = fb.entry_block();
    let throw_ptr = fb.block_param(entry, 0);
    let rv = fb.call_indirect(throw_ptr, &[], Some(Type::I64)).unwrap();
    fb.ret(rv);
    mb.finish_func(callee, fb);

    let mut fb = mb.define_func(main);
    let entry = fb.entry_block();
    let throw_ptr = fb.block_param(entry, 0);
    let normal = fb.create_block(&[Type::I64]);
    let exception = fb.create_block(&[]);
    fb.invoke(callee, &[throw_ptr], normal, &[], exception, &[]);
    fb.switch_to_block(normal);
    let rv = fb.block_param(normal, 0);
    fb.ret(rv);
    fb.switch_to_block(exception);
    let sentinel = fb.iconst(Type::I64, 999);
    fb.ret(sentinel);
    mb.finish_func(main, fb);

    let module = mb.build();
    let jit = JitModule::compile::<NanBox>(&module, &[]);
    let throw_ptr = dynlower_test_throw_exception_stub as usize as u64;
    assert_eq!(jit.call_outcome(main, &[throw_ptr]), JitOutcome::Value(999));
}

// ── Loop with dynir builder ────────────────────────────────────────

#[test]
fn loop_sum_dynir() {
    let mut b = FunctionBuilder::new("sum", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let n = b.block_param(entry, 0);

    let loop_bb = b.create_block(&[Type::I64, Type::I64]); // (i, acc)
    let exit_bb = b.create_block(&[Type::I64]);

    let zero = b.iconst(Type::I64, 0);
    b.jump(loop_bb, &[zero, zero]);

    b.switch_to_block(loop_bb);
    let i = b.block_param(loop_bb, 0);
    let acc = b.block_param(loop_bb, 1);
    let cond = b.icmp(CmpOp::Slt, i, n);
    let new_acc = b.add(acc, i);
    let one = b.iconst(Type::I64, 1);
    let new_i = b.add(i, one);
    b.br_if(cond, loop_bb, &[new_i, new_acc], exit_bb, &[acc]);

    b.switch_to_block(exit_bb);
    let result = b.block_param(exit_bb, 0);
    b.ret(result);

    let func = b.build();
    assert_eq!(run_jit(&func, &[10]), 45);
    assert_eq!(run_jit(&func, &[0]), 0);
}

#[test]
fn as_fib() {
    let wasm = include_bytes!("../../wasm2dynir/as-programs/fib.wasm");
    let (func, _) = wasm2dynir::translate_wasm(wasm).expect("translate");
    assert_eq!(run_jit(&func, &[0]) as i32, 0);
    assert_eq!(run_jit(&func, &[1]) as i32, 1);
    assert_eq!(run_jit(&func, &[10]) as i32, 55);
    assert_eq!(run_jit(&func, &[20]) as i32, 6765);
    assert_eq!(run_jit(&func, &[30]) as i32, 832040);
}

#[test]
fn as_collatz() {
    let wasm = include_bytes!("../../wasm2dynir/as-programs/collatz.wasm");
    let (func, _) = wasm2dynir::translate_wasm(wasm).expect("translate");
    assert_eq!(run_jit(&func, &[1]) as i32, 0);
    assert_eq!(run_jit(&func, &[2]) as i32, 1);
    assert_eq!(run_jit(&func, &[3]) as i32, 7);
    assert_eq!(run_jit(&func, &[6]) as i32, 8);
    assert_eq!(run_jit(&func, &[27]) as i32, 111);
}

#[test]
fn as_gcd() {
    let wasm = include_bytes!("../../wasm2dynir/as-programs/gcd.wasm");
    let (func, _) = wasm2dynir::translate_wasm(wasm).expect("translate");
    assert_eq!(run_jit(&func, &[12, 8]) as i32, 4);
    assert_eq!(run_jit(&func, &[100, 75]) as i32, 25);
    assert_eq!(run_jit(&func, &[17, 13]) as i32, 1);
    assert_eq!(run_jit(&func, &[0, 5]) as i32, 5);
}

#[test]
fn as_power() {
    let wasm = include_bytes!("../../wasm2dynir/as-programs/power.wasm");
    let (func, _) = wasm2dynir::translate_wasm(wasm).expect("translate");
    assert_eq!(run_jit(&func, &[2, 0]) as i64, 1);
    assert_eq!(run_jit(&func, &[2, 10]) as i64, 1024);
    assert_eq!(run_jit(&func, &[3, 5]) as i64, 243);
    assert_eq!(run_jit(&func, &[10, 9]) as i64, 1_000_000_000);
}

#[test]
fn as_primes() {
    let wasm = include_bytes!("../../wasm2dynir/as-programs/primes.wasm");
    let (func, _) = wasm2dynir::translate_wasm(wasm).expect("translate");
    assert_eq!(run_jit(&func, &[1]) as i32, 0);
    assert_eq!(run_jit(&func, &[10]) as i32, 4);
    assert_eq!(run_jit(&func, &[100]) as i32, 25);
    assert_eq!(run_jit(&func, &[1000]) as i32, 168);
}

#[test]
fn fib_loop_dynir() {
    let mut b = FunctionBuilder::new("fib", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let n = b.block_param(entry, 0);

    let loop_bb = b.create_block(&[Type::I64, Type::I64, Type::I64]); // (i, a, b)
    let exit = b.create_block(&[Type::I64]);

    let zero = b.iconst(Type::I64, 0);
    let one = b.iconst(Type::I64, 1);
    b.jump(loop_bb, &[zero, zero, one]);

    b.switch_to_block(loop_bb);
    let i = b.block_param(loop_bb, 0);
    let a = b.block_param(loop_bb, 1);
    let fib_b = b.block_param(loop_bb, 2);
    let cond = b.icmp(CmpOp::Slt, i, n);
    let next = b.add(a, fib_b);
    let i_plus = b.add(i, one);
    b.br_if(cond, loop_bb, &[i_plus, fib_b, next], exit, &[a]);

    b.switch_to_block(exit);
    let result = b.block_param(exit, 0);
    b.ret(result);

    let func = b.build();
    assert_eq!(run_jit(&func, &[0]), 0);
    assert_eq!(run_jit(&func, &[1]), 1);
    assert_eq!(run_jit(&func, &[10]), 55);
    assert_eq!(run_jit(&func, &[20]), 6765);
}

#[test]
fn bench_primes() {
    let wasm = include_bytes!("../../wasm2dynir/as-programs/primes.wasm");
    let (func, _) = wasm2dynir::translate_wasm(wasm).expect("translate");

    // JIT compile
    let start = std::time::Instant::now();
    let jit = crate::JitFunction::compile::<NanBox>(&func, &[]);
    let compile_time = start.elapsed();

    // JIT run
    let start = std::time::Instant::now();
    let jit_result = unsafe { crate::call_jit(jit.as_ptr(), &[10000]) } as i32;
    let jit_time = start.elapsed();

    // Interpreter run
    use dynalloc::LowBitPtrPolicy;
    use dynir::gc_runtime::GcInterpCtx;
    use dynir::interp::*;
    use dynir::ir::Module;
    use dynobj::Compact;
    use dynvalue::LowBit;
    let (module, entry) = Module::from_function(func.clone());
    let roots: GcInterpCtx<Compact, LowBitPtrPolicy<3>> = GcInterpCtx::new_unallocating();
    let interp = ModuleInterpreter::<LowBit<3>, _>::new(&module, &roots);
    let start = std::time::Instant::now();
    let interp_result = match interp.run(entry, &[10000]).unwrap() {
        InterpResult::Value(v) => v as i32,
        other => panic!("{:?}", other),
    };
    let interp_time = start.elapsed();

    assert_eq!(jit_result, interp_result);
    eprintln!("count_primes(10000) = {}", jit_result);
    eprintln!("  compile:     {:?}", compile_time);
    eprintln!("  JIT run:     {:?}", jit_time);
    eprintln!("  interp run:  {:?}", interp_time);
    eprintln!(
        "  speedup:     {:.1}x",
        interp_time.as_secs_f64() / jit_time.as_secs_f64()
    );
}

#[test]
fn test_payload_nanbox() {
    let mut b = FunctionBuilder::new("test", &[Type::I64], Some(Type::I64));
    let x = b.block_param(b.entry_block(), 0);
    let p = b.payload(x);
    b.ret(p);
    let func = b.build();
    let jit = JitFunction::compile::<NanBox>(&func, &[]);
    let input = 0x7FFC_0001_0000_1234u64;
    let result = unsafe { call_jit(jit.as_ptr(), &[input]) };
    assert_eq!(result, 0x0000_0001_0000_1234u64);
}

#[test]
fn test_payload_lowbit() {
    use dynvalue::LowBit;
    let mut b = FunctionBuilder::new("test", &[Type::I64], Some(Type::I64));
    let x = b.block_param(b.entry_block(), 0);
    let p = b.payload(x);
    b.ret(p);
    let func = b.build();
    let jit = JitFunction::compile::<LowBit<3>>(&func, &[]);
    // LowBit<3>: encode_tagged(2, 0x1234) = (0x1234 << 3) | 2 = 0x91A2
    let input = LowBit::<3>::encode_tagged(2, 0x1234);
    let result = unsafe { call_jit(jit.as_ptr(), &[input]) };
    assert_eq!(result, 0x1234);
}

#[test]
fn test_payload_then_load() {
    // fn(ptr_nanbox: I64) -> I64
    // result = Load(I64, Payload(ptr_nanbox), 0)
    let mut b = FunctionBuilder::new("test", &[Type::I64], Some(Type::I64));
    let x = b.block_param(b.entry_block(), 0);
    let p = b.payload(x);
    let loaded = b.load(Type::I64, p, 0);
    b.ret(loaded);
    let func = b.build();
    let jit = JitFunction::compile::<NanBox>(&func, &[]);

    // Allocate a u64 on the heap, encode as NanBox TAG_PTR
    let val: Box<u64> = Box::new(0xDEAD_BEEF_CAFE_BABEu64);
    let ptr = Box::into_raw(val) as u64;
    let nanbox = NanBox::encode_tagged(0, ptr & ((1u64 << NanBox::PAYLOAD_BITS) - 1));
    let result = unsafe { call_jit(jit.as_ptr(), &[nanbox]) };
    assert_eq!(result, 0xDEAD_BEEF_CAFE_BABEu64);
    unsafe {
        drop(Box::from_raw(ptr as *mut u64));
    }
}

#[test]
fn store_then_load_round_trip() {
    let mut b = FunctionBuilder::new("store_load", &[Type::Ptr, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let ptr = b.block_param(entry, 0);
    let value = b.block_param(entry, 1);
    b.store(value, ptr, 0);
    let loaded = b.load(Type::I64, ptr, 0);
    b.ret(loaded);
    let func = b.build();

    let mut slot = 0u64;
    let result = run_jit(
        &func,
        &[(&mut slot as *mut u64) as u64, 0xDEAD_BEEF_CAFE_BABE],
    );
    assert_eq!(result, 0xDEAD_BEEF_CAFE_BABE);
    assert_eq!(slot, 0xDEAD_BEEF_CAFE_BABE);
}

#[test]
fn test_extern_with_payload() {
    use dynir::types::Signature;

    // fn(nanbox: I64) -> I64
    // p = Payload(nanbox)
    // result = call double_extern(p)
    extern "C" fn double_val(x: u64) -> u64 {
        x * 2
    }

    let mut b = FunctionBuilder::new("test", &[Type::I64], Some(Type::I64));
    let fref = b.declare_func(
        "double",
        Signature {
            params: vec![Type::I64],
            ret: Some(Type::I64),
        },
    );
    let x = b.block_param(b.entry_block(), 0);
    let p = b.payload(x);
    let result = b.call(fref, &[p]).unwrap();
    b.ret(result);
    let func = b.build();
    let externs = vec![double_val as *const u8];
    let jit = JitFunction::compile::<NanBox>(&func, &externs);
    let input = NanBox::encode_tagged(0, 5); // payload = 5
    let result = unsafe { call_jit(jit.as_ptr(), &[input]) };
    assert_eq!(result, 10);
}

#[test]
fn test_extern_two_params() {
    use dynir::types::Signature;

    // fn(a: I64, b: I64) -> I64
    // Call extern sub(a, b)
    extern "C" fn sub_fn(a: u64, b: u64) -> u64 {
        a.wrapping_sub(b)
    }

    let mut b = FunctionBuilder::new("test", &[Type::I64, Type::I64], Some(Type::I64));
    let fref = b.declare_func(
        "sub",
        Signature {
            params: vec![Type::I64, Type::I64],
            ret: Some(Type::I64),
        },
    );
    let a = b.block_param(b.entry_block(), 0);
    let bv = b.block_param(b.entry_block(), 1);
    let result = b.call(fref, &[a, bv]).unwrap();
    b.ret(result);
    let func = b.build();
    let externs = vec![sub_fn as *const u8];
    let jit = JitFunction::compile::<NanBox>(&func, &externs);
    let result = unsafe { call_jit(jit.as_ptr(), &[10, 3]) };
    assert_eq!(result, 7);
}

#[test]
fn test_extern_skip_first_param() {
    use dynir::types::Signature;

    // fn(unused: I64, a: I64, b: I64) -> I64
    // Call extern sub(a, b)  -- skipping param 0
    extern "C" fn sub_fn(a: u64, b: u64) -> u64 {
        a.wrapping_sub(b)
    }

    let mut b = FunctionBuilder::new("test", &[Type::I64, Type::I64, Type::I64], Some(Type::I64));
    let fref = b.declare_func(
        "sub",
        Signature {
            params: vec![Type::I64, Type::I64],
            ret: Some(Type::I64),
        },
    );
    let _unused = b.block_param(b.entry_block(), 0);
    let a = b.block_param(b.entry_block(), 1);
    let bv = b.block_param(b.entry_block(), 2);
    let result = b.call(fref, &[a, bv]).unwrap();
    b.ret(result);
    let func = b.build();
    let externs = vec![sub_fn as *const u8];
    let jit = JitFunction::compile::<NanBox>(&func, &externs);
    // Pass unused=99, a=10, b=3
    let result = unsafe { call_jit(jit.as_ptr(), &[99, 10, 3]) };
    assert_eq!(result, 7);
}

#[test]
fn test_branch_preserves_params() {
    // Mimics: fn(closure: I64, n: I64) -> I64
    //   result = call extern_check(n)
    //   if result == 1 then return 42
    //   else return call extern_identity(n)  <-- n must survive the branch
    use dynir::types::Signature;

    extern "C" fn check_fn(x: u64) -> u64 {
        if x > 5 { 1 } else { 0 }
    }
    extern "C" fn identity_fn(x: u64) -> u64 {
        x
    }

    let mut b = FunctionBuilder::new("test", &[Type::I64, Type::I64], Some(Type::I64));
    let check = b.declare_func(
        "check",
        Signature {
            params: vec![Type::I64],
            ret: Some(Type::I64),
        },
    );
    let ident = b.declare_func(
        "identity",
        Signature {
            params: vec![Type::I64],
            ret: Some(Type::I64),
        },
    );

    let closure = b.block_param(b.entry_block(), 0);
    let n = b.block_param(b.entry_block(), 1);

    // Call extern check(n)
    let result = b.call(check, &[n]).unwrap();
    let one = b.iconst(Type::I64, 1);
    let cond = b.icmp(CmpOp::Eq, result, one);

    // Create then/else blocks with 2 params each (closure, n)
    let then_block = b.create_block(&[Type::I64, Type::I64]);
    let else_block = b.create_block(&[Type::I64, Type::I64]);

    b.br_if(cond, then_block, &[closure, n], else_block, &[closure, n]);

    // Then block: return 42
    b.switch_to_block(then_block);
    let val42 = b.iconst(Type::I64, 42);
    b.ret(val42);

    // Else block: return identity(n)
    b.switch_to_block(else_block);
    let else_n = b.block_param(else_block, 1);
    let else_result = b.call(ident, &[else_n]).unwrap();
    b.ret(else_result);

    let func = b.build();
    let externs = vec![check_fn as *const u8, identity_fn as *const u8];
    let jit = JitFunction::compile::<NanBox>(&func, &externs);

    // n=10 > 5, so check returns 1, should go to then_block and return 42
    assert_eq!(unsafe { call_jit(jit.as_ptr(), &[99, 10]) }, 42);
    // n=3 <= 5, so check returns 0, should go to else_block and return 3
    assert_eq!(unsafe { call_jit(jit.as_ptr(), &[99, 3]) }, 3);
}

#[test]
fn test_simple_branch_params_no_extern() {
    // fn(a: I64, b: I64) -> I64
    // if a == 0 then return 42 else return b
    // No extern calls - tests pure branching with params

    let mut b = FunctionBuilder::new("test", &[Type::I64, Type::I64], Some(Type::I64));
    let a = b.block_param(b.entry_block(), 0);
    let bv = b.block_param(b.entry_block(), 1);

    let zero = b.iconst(Type::I64, 0);
    let cond = b.icmp(CmpOp::Eq, a, zero);

    let then_block = b.create_block(&[Type::I64, Type::I64]);
    let else_block = b.create_block(&[Type::I64, Type::I64]);
    b.br_if(cond, then_block, &[a, bv], else_block, &[a, bv]);

    b.switch_to_block(then_block);
    let val42 = b.iconst(Type::I64, 42);
    b.ret(val42);

    b.switch_to_block(else_block);
    let else_b = b.block_param(else_block, 1);
    b.ret(else_b);

    let func = b.build();
    let jit = JitFunction::compile::<NanBox>(&func, &[]);

    assert_eq!(unsafe { call_jit(jit.as_ptr(), &[0, 99]) }, 42); // a==0 → return 42
    assert_eq!(unsafe { call_jit(jit.as_ptr(), &[1, 77]) }, 77); // a!=0 → return b=77
}

// ── TagScheme-generic tag operation tests ──────────────────────────

fn build_is_tag_func() -> dynir::Function {
    let mut b = FunctionBuilder::new("is_tag", &[Type::I64], Some(Type::I64));
    let x = b.block_param(b.entry_block(), 0);
    let result = b.is_tag(x, 1);
    let r64 = b.zext(result, Type::I64);
    b.ret(r64);
    b.build()
}

fn build_make_tagged_func() -> dynir::Function {
    let mut b = FunctionBuilder::new("make_tagged", &[Type::I64], Some(Type::I64));
    let payload = b.block_param(b.entry_block(), 0);
    let tagged = b.make_tagged(2, payload);
    b.ret(tagged);
    b.build()
}

fn build_tag_of_func() -> dynir::Function {
    let mut b = FunctionBuilder::new("tag_of", &[Type::I64], Some(Type::I64));
    let x = b.block_param(b.entry_block(), 0);
    let tag = b.tag_of(x);
    let tag64 = b.zext(tag, Type::I64);
    b.ret(tag64);
    b.build()
}

#[test]
fn test_is_tag_nanbox() {
    let func = build_is_tag_func();
    let jit = JitFunction::compile::<NanBox>(&func, &[]);
    let tagged1 = NanBox::encode_tagged(1, 42);
    let tagged0 = NanBox::encode_tagged(0, 42);
    assert_eq!(unsafe { call_jit(jit.as_ptr(), &[tagged1]) }, 1);
    assert_eq!(unsafe { call_jit(jit.as_ptr(), &[tagged0]) }, 0);
}

#[test]
fn test_is_tag_lowbit() {
    use dynvalue::LowBit;
    let func = build_is_tag_func();
    let jit = JitFunction::compile::<LowBit<3>>(&func, &[]);
    let tagged1 = LowBit::<3>::encode_tagged(1, 42);
    let tagged0 = LowBit::<3>::encode_tagged(0, 42);
    assert_eq!(unsafe { call_jit(jit.as_ptr(), &[tagged1]) }, 1);
    assert_eq!(unsafe { call_jit(jit.as_ptr(), &[tagged0]) }, 0);
}

#[test]
fn test_make_tagged_nanbox() {
    let func = build_make_tagged_func();
    let jit = JitFunction::compile::<NanBox>(&func, &[]);
    let result = unsafe { call_jit(jit.as_ptr(), &[0x1234]) };
    assert_eq!(result, NanBox::encode_tagged(2, 0x1234));
}

#[test]
fn test_make_tagged_lowbit() {
    use dynvalue::LowBit;
    let func = build_make_tagged_func();
    let jit = JitFunction::compile::<LowBit<3>>(&func, &[]);
    let result = unsafe { call_jit(jit.as_ptr(), &[0x1234]) };
    assert_eq!(result, LowBit::<3>::encode_tagged(2, 0x1234));
}

#[test]
fn test_tag_of_nanbox() {
    let func = build_tag_of_func();
    let jit = JitFunction::compile::<NanBox>(&func, &[]);
    for tag in 0..4u32 {
        let tagged = NanBox::encode_tagged(tag, 0xABC);
        assert_eq!(unsafe { call_jit(jit.as_ptr(), &[tagged]) }, tag as u64);
    }
}

#[test]
fn test_tag_of_lowbit() {
    use dynvalue::LowBit;
    let func = build_tag_of_func();
    let jit = JitFunction::compile::<LowBit<3>>(&func, &[]);
    for tag in 0..8u32 {
        let tagged = LowBit::<3>::encode_tagged(tag, 0xABC);
        assert_eq!(unsafe { call_jit(jit.as_ptr(), &[tagged]) }, tag as u64);
    }
}

#[test]
fn test_roundtrip_lowbit() {
    // make_tagged then payload should recover the original payload
    use dynvalue::LowBit;
    let mut b = FunctionBuilder::new("roundtrip", &[Type::I64], Some(Type::I64));
    let x = b.block_param(b.entry_block(), 0);
    let tagged = b.make_tagged(1, x);
    let recovered = b.payload(tagged);
    b.ret(recovered);
    let func = b.build();
    let jit = JitFunction::compile::<LowBit<3>>(&func, &[]);
    let result = unsafe { call_jit(jit.as_ptr(), &[0x1234_5678]) };
    assert_eq!(result, 0x1234_5678);
}

#[test]
fn test_roundtrip_nanbox() {
    let mut b = FunctionBuilder::new("roundtrip", &[Type::I64], Some(Type::I64));
    let x = b.block_param(b.entry_block(), 0);
    let tagged = b.make_tagged(1, x);
    let recovered = b.payload(tagged);
    b.ret(recovered);
    let func = b.build();
    let jit = JitFunction::compile::<NanBox>(&func, &[]);
    let result = unsafe { call_jit(jit.as_ptr(), &[0x1234_5678]) };
    assert_eq!(result, 0x1234_5678);
}

// ── JitModule tests ────────────────────────────────────────────────

#[test]
fn jit_module_two_funcs() {
    // func double(x: I32) -> I32 = x * 2
    // func main(x: I32) -> I32 = double(x) + 1
    let mut mb = ModuleBuilder::new();
    let double = mb.declare_func("double", &[Type::I32], Some(Type::I32));
    let main_fn = mb.declare_func("main", &[Type::I32], Some(Type::I32));

    {
        let mut fb = mb.define_func(double);
        let entry = fb.entry_block();
        let x = fb.block_param(entry, 0);
        let two = fb.iconst(Type::I32, 2);
        let result = fb.mul(x, two);
        fb.ret(result);
        mb.finish_func(double, fb);
    }
    {
        let mut fb = mb.define_func(main_fn);
        let entry = fb.entry_block();
        let x = fb.block_param(entry, 0);
        let doubled = fb.call(double, &[x]).unwrap();
        let one = fb.iconst(Type::I32, 1);
        let result = fb.add(doubled, one);
        fb.ret(result);
        mb.finish_func(main_fn, fb);
    }

    let module = mb.build();
    let jit = JitModule::compile::<NanBox>(&module, &[]);
    assert_eq!(jit.call(main_fn, &[10]) as i32, 21);
    assert_eq!(jit.call(main_fn, &[0]) as i32, 1);
}

#[test]
fn jit_module_recursive_factorial() {
    // func fact(n: I32) -> I32 = if n <= 1 then 1 else n * fact(n-1)
    // func main(n: I32) -> I32 = fact(n)
    let mut mb = ModuleBuilder::new();
    let fact = mb.declare_func("fact", &[Type::I32], Some(Type::I32));
    let main_fn = mb.declare_func("main", &[Type::I32], Some(Type::I32));

    {
        let mut fb = mb.define_func(fact);
        let entry = fb.entry_block();
        let n = fb.block_param(entry, 0);
        let one = fb.iconst(Type::I32, 1);
        let cond = fb.icmp(CmpOp::Sle, n, one);

        let then_bb = fb.create_block(&[]);
        let else_bb = fb.create_block(&[]);
        fb.br_if(cond, then_bb, &[], else_bb, &[]);

        fb.switch_to_block(then_bb);
        let ret_one = fb.iconst(Type::I32, 1);
        fb.ret(ret_one);

        fb.switch_to_block(else_bb);
        let one2 = fb.iconst(Type::I32, 1);
        let n_minus_1 = fb.sub(n, one2);
        let sub_result = fb.call(fact, &[n_minus_1]).unwrap();
        let result = fb.mul(n, sub_result);
        fb.ret(result);

        mb.finish_func(fact, fb);
    }
    {
        let mut fb = mb.define_func(main_fn);
        let entry = fb.entry_block();
        let n = fb.block_param(entry, 0);
        let result = fb.call(fact, &[n]).unwrap();
        fb.ret(result);
        mb.finish_func(main_fn, fb);
    }

    let module = mb.build();
    let jit = JitModule::compile::<NanBox>(&module, &[]);
    assert_eq!(jit.call(main_fn, &[1]) as i32, 1);
    assert_eq!(jit.call(main_fn, &[5]) as i32, 120);
    assert_eq!(jit.call(main_fn, &[10]) as i32, 3628800);
}

#[test]
fn jit_module_with_extern() {
    use dynir::types::Signature;

    // extern fn add_ten(x: I64) -> I64
    // func main(x: I64) -> I64 = add_ten(x) * 2
    extern "C" fn add_ten(x: u64) -> u64 {
        x + 10
    }

    let mut mb = ModuleBuilder::new();
    let ext = mb.declare_extern(
        "add_ten",
        Signature {
            params: vec![Type::I64],
            ret: Some(Type::I64),
        },
    );
    let main_fn = mb.declare_func("main", &[Type::I64], Some(Type::I64));

    {
        let mut fb = mb.define_func(main_fn);
        let entry = fb.entry_block();
        let x = fb.block_param(entry, 0);
        let added = fb.call(ext, &[x]).unwrap();
        let two = fb.iconst(Type::I64, 2);
        let result = fb.mul(added, two);
        fb.ret(result);
        mb.finish_func(main_fn, fb);
    }

    let module = mb.build();
    let externs = vec![add_ten as *const u8];
    let jit = JitModule::compile::<NanBox>(&module, &externs);
    assert_eq!(jit.call(main_fn, &[5]), 30); // (5 + 10) * 2
}

#[test]
fn jit_module_fifty_nested() {
    // 50 functions chained: func_0(x) = x + 1, func_i(x) = func_{i-1}(x) + 1
    // main(0) should return 50
    let n = 50;
    let mut wat = String::from("(module\n");
    wat.push_str("  (func $func_0 (param i32) (result i32)\n");
    wat.push_str("    local.get 0\n    i32.const 1\n    i32.add)\n");
    for i in 1..n {
        wat.push_str(&format!("  (func $func_{i} (param i32) (result i32)\n"));
        wat.push_str("    local.get 0\n");
        wat.push_str(&format!("    call $func_{}\n", i - 1));
        wat.push_str("    i32.const 1\n    i32.add)\n");
    }
    wat.push_str(&format!("  (export \"main\" (func $func_{})))\n", n - 1));

    let wasm = wat::parse_str(&wat).expect("parse WAT");
    let (module, entry) = wasm2dynir::translate_wasm_module(&wasm).expect("translate");

    let jit = JitModule::compile::<NanBox>(&module, &[]);
    assert_eq!(jit.call(entry, &[0]) as i32, 50);
    assert_eq!(jit.call(entry, &[100]) as i32, 150);
}

#[test]
fn jit_module_wasm_recursive_factorial() {
    let wat = r#"(module
        (func $fact (param $n i32) (result i32)
            local.get $n
            i32.const 1
            i32.le_s
            if (result i32)
                i32.const 1
            else
                local.get $n
                local.get $n
                i32.const 1
                i32.sub
                call $fact
                i32.mul
            end)
        (func (export "main") (param i32) (result i32)
            local.get 0
            call $fact))"#;
    let wasm = wat::parse_str(wat).expect("parse WAT");
    let (module, entry) = wasm2dynir::translate_wasm_module(&wasm).expect("translate");

    let jit = JitModule::compile::<NanBox>(&module, &[]);
    assert_eq!(jit.call(entry, &[1]) as i32, 1);
    assert_eq!(jit.call(entry, &[5]) as i32, 120);
    assert_eq!(jit.call(entry, &[10]) as i32, 3628800);
}

#[test]
fn jit_module_from_single_function() {
    // Wrap a single function in a module and run it through JitModule
    let mut b = FunctionBuilder::new("add1", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let x = b.block_param(entry, 0);
    let one = b.iconst(Type::I64, 1);
    let result = b.add(x, one);
    b.ret(result);
    let func = b.build();

    let (module, entry_ref) = Module::from_function(func);
    let jit = JitModule::compile::<NanBox>(&module, &[]);
    assert_eq!(jit.call(entry_ref, &[41]), 42);
}

#[test]
fn jit_module_safepoint_no_handler() {
    // Safepoint with no handler should be silently skipped
    // Use GcPtr type since safepoint requires it
    let mut b = FunctionBuilder::new("test", &[Type::GcPtr], Some(Type::GcPtr));
    let entry = b.entry_block();
    let x = b.block_param(entry, 0);
    b.safepoint(&[x]);
    // GcPtr is pointer-sized (I64 on 64-bit), so add works
    let one = b.iconst(Type::I64, 1);
    let result = b.add(x, one);
    b.ret(result);
    let func = b.build();

    let (module, entry_ref) = Module::from_function(func);
    let jit = JitModule::compile::<NanBox>(&module, &[]);
    assert_eq!(jit.call(entry_ref, &[10]), 11);
}

#[test]
fn jit_module_safepoint_with_handler() {
    use std::sync::atomic::{AtomicU32, Ordering};
    static CALL_COUNT: AtomicU32 = AtomicU32::new(0);

    extern "C" fn test_handler(_frame_ptr: *mut u8, _frame_size: usize) {
        CALL_COUNT.fetch_add(1, Ordering::SeqCst);
    }

    let mut b = FunctionBuilder::new("test", &[Type::GcPtr], Some(Type::GcPtr));
    let entry = b.entry_block();
    let x = b.block_param(entry, 0);
    b.safepoint(&[x]);
    let one = b.iconst(Type::I64, 1);
    let result = b.add(x, one);
    b.ret(result);
    let func = b.build();

    let (module, entry_ref) = Module::from_function(func);
    CALL_COUNT.store(0, Ordering::SeqCst);
    let jit = JitModule::compile_with_gc::<NanBox>(&module, &[], test_handler);
    assert_eq!(jit.call(entry_ref, &[10]), 11);
    assert_eq!(CALL_COUNT.load(Ordering::SeqCst), 1);
}

#[test]
fn jit_entry_fp_fence_stops_walker_at_rust_boundary() {
    use crate::{pop_jit_entry_fp, push_jit_entry_fp, walk_jit_ancestor_roots};
    // Without a fence, the walker would try to crawl arbitrarily far up
    // the FP chain. Establish a fence and verify that walking from a
    // synthetic JIT-frame FP stops at the fence (zero ancestor visits).
    //
    // We construct a fake "JIT frame" on the stack: just two u64 slots
    // representing [saved_fp, saved_lr]. Set saved_fp to a sentinel
    // value, install that sentinel as the fence, and confirm the walker
    // sees no ancestors.

    // Fake frame: saved_fp = 0xDEAD_BEEF (a value we'll fence), saved_lr = 0
    let fake_frame: [u64; 2] = [0xDEAD_BEEF_DEAD_BEEFu64, 0];
    let synthetic_jit_fp = fake_frame.as_ptr() as *const u8;

    let visited = std::cell::Cell::new(0usize);
    unsafe { push_jit_entry_fp(0xDEAD_BEEF_DEAD_BEEFu64 as *const u8) };
    walk_jit_ancestor_roots(synthetic_jit_fp, &mut |_slot| {
        visited.set(visited.get() + 1);
    });
    pop_jit_entry_fp();

    // The first hop walks: saved_fp = 0xDEAD_BEEF == fence → stop
    // before visiting any roots.
    assert_eq!(visited.get(), 0, "walker visited a root past the fence");
}

#[test]
fn parked_walker_traverses_interleaved_host_frames() {
    // Regression: `walk_parked_thread_jit_roots` (used by minor GC and
    // alloc-path major GC) must scan EVERY JIT frame in the chain, even
    // when JIT frames are separated by an intervening runtime (non-JIT)
    // frame — e.g. clojure.core `str`'s variadic path recurses
    // `JIT → runtime helper → JIT → …`. An earlier version stopped at the
    // first such non-JIT frame ("Phase B break"), scanning only the
    // deepest JIT frame and silently dropping every outer recursion
    // frame's roots, so an alloc-path collection left those spill slots
    // un-forwarded → dangling pointer → crash.
    use crate::{
        SafepointRecord, register_jit_code, unregister_jit_code, walk_parked_thread_jit_roots,
    };

    // A fake JIT code range. The walker only does address arithmetic on
    // it (no execution), so any non-overlapping window works. Use a high
    // base to avoid colliding with real registrations from other tests.
    const CODE_START: usize = 0x7000_0000_0000;
    const CODE_END: usize = CODE_START + 0x1000;
    const RET_OFFSET: usize = 0x10; // safepoint return offset within the fn
    let jit_lr = CODE_START + RET_OFFSET;
    let host_lr = 0x1usize; // deliberately outside any registered range

    let safepoints: std::sync::Arc<[SafepointRecord]> =
        std::sync::Arc::from(vec![SafepointRecord {
            code_offset: RET_OFFSET,
            return_offset: RET_OFFSET,
            root_slots: vec![16], // one root at +16 in the caller frame
        }]);
    register_jit_code(CODE_START, CODE_END, safepoints);

    // Synthetic FP chain (each frame = [saved_fp, saved_lr, root_slot]):
    //   start → f_inner(JIT) → f_host(non-JIT) → f_outer(JIT) → f_end(null)
    // The walker scans `saved_fp + 16` for each JIT frame it finds.
    let mut f_end: [u64; 3] = [0, 0, 0];
    let f_end_ptr = f_end.as_mut_ptr() as u64;
    let mut f_outer: [u64; 3] = [f_end_ptr, jit_lr as u64, 0];
    let f_outer_ptr = f_outer.as_mut_ptr() as u64;
    let mut f_host: [u64; 3] = [f_outer_ptr, host_lr as u64, 0];
    let f_host_ptr = f_host.as_mut_ptr() as u64;
    let mut f_inner: [u64; 3] = [f_host_ptr, jit_lr as u64, 0];
    let start_fp = f_inner.as_mut_ptr() as *const u8;

    let visited = std::cell::Cell::new(0usize);
    // No fence pushed: the chain terminates on the null saved_fp in f_end.
    walk_parked_thread_jit_roots(start_fp, &mut |_slot| {
        visited.set(visited.get() + 1);
    });

    unregister_jit_code(CODE_START);

    // Both JIT frames (inner AND outer) must be scanned: 2 roots total.
    // The pre-fix walker stopped at f_host and reported only 1.
    assert_eq!(
        visited.get(),
        2,
        "parked walker did not traverse through the intervening non-JIT \
         frame to reach the outer JIT frame"
    );

    // touch the frame arrays so they stay live for the whole walk
    std::hint::black_box((&f_inner, &f_host, &f_outer, &f_end));
}

#[test]
fn jit_module_bench_fifty_nested() {
    let n = 50;
    let mut wat = String::from("(module\n");
    wat.push_str("  (func $func_0 (param i32) (result i32)\n");
    wat.push_str("    local.get 0\n    i32.const 1\n    i32.add)\n");
    for i in 1..n {
        wat.push_str(&format!("  (func $func_{i} (param i32) (result i32)\n"));
        wat.push_str("    local.get 0\n");
        wat.push_str(&format!("    call $func_{}\n", i - 1));
        wat.push_str("    i32.const 1\n    i32.add)\n");
    }
    wat.push_str(&format!("  (export \"main\" (func $func_{})))\n", n - 1));

    let wasm = wat::parse_str(&wat).expect("parse WAT");
    let (module, entry) = wasm2dynir::translate_wasm_module(&wasm).expect("translate");

    // JIT compile + run
    let start = std::time::Instant::now();
    let jit = JitModule::compile::<NanBox>(&module, &[]);
    let compile_time = start.elapsed();

    let start = std::time::Instant::now();
    let jit_result = jit.call(entry, &[0]) as i32;
    let jit_time = start.elapsed();

    // Interpreter run
    use dynalloc::LowBitPtrPolicy;
    use dynir::gc_runtime::GcInterpCtx;
    use dynir::interp::*;
    use dynobj::Compact;
    use dynvalue::LowBit;
    let roots: GcInterpCtx<Compact, LowBitPtrPolicy<3>> = GcInterpCtx::new_unallocating();
    let interp = ModuleInterpreter::<LowBit<3>, _>::new(&module, &roots);
    let start = std::time::Instant::now();
    let interp_result = match interp.run(entry, &[0]).unwrap() {
        InterpResult::Value(v) => v as i32,
        other => panic!("{:?}", other),
    };
    let interp_time = start.elapsed();

    assert_eq!(jit_result, interp_result);
    assert_eq!(jit_result, n as i32);
    eprintln!("50-nested JIT module:");
    eprintln!("  compile:     {:?}", compile_time);
    eprintln!("  JIT run:     {:?}", jit_time);
    eprintln!("  interp run:  {:?}", interp_time);
    eprintln!(
        "  speedup:     {:.1}x",
        interp_time.as_secs_f64() / jit_time.as_secs_f64()
    );
}

// ─── Linear Scan Register Allocator Tests ─────────────────────────

fn run_jit_linear_scan(func: &dynir::Function, args: &[u64]) -> u64 {
    #[cfg(target_arch = "aarch64")]
    let jit = JitFunction::compile_with_regalloc::<
        DefaultJitConfig<NanBox>,
        crate::backend::Arm64Backend,
        crate::regalloc::LinearScanAllocator,
    >(func, &[], None);
    #[cfg(target_arch = "x86_64")]
    let jit = JitFunction::compile_with_regalloc::<
        DefaultJitConfig<NanBox>,
        crate::backend::X64Backend,
        crate::regalloc::LinearScanAllocator,
    >(func, &[], None);
    unsafe { call_jit(jit.as_ptr(), args) }
}

#[test]
fn linear_scan_return_const() {
    let mut b = FunctionBuilder::new("const", &[], Some(Type::I64));
    let entry = b.entry_block();
    let _ = entry;
    let v = b.iconst(Type::I64, 42);
    b.ret(v);
    let func = b.build();
    assert_eq!(run_jit_linear_scan(&func, &[]), 42);
}

#[test]
fn linear_scan_add() {
    let mut b = FunctionBuilder::new("add", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let result = b.add(a, bb);
    b.ret(result);
    let func = b.build();
    assert_eq!(run_jit_linear_scan(&func, &[10, 32]), 42);
}

#[test]
fn linear_scan_if_else() {
    let mut b = FunctionBuilder::new("max", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);

    let then_bb = b.create_block(&[]);
    let else_bb = b.create_block(&[]);
    let merge = b.create_block(&[Type::I64]);

    let cond = b.icmp(CmpOp::Sgt, a, bb);
    b.br_if(cond, then_bb, &[], else_bb, &[]);

    b.switch_to_block(then_bb);
    b.jump(merge, &[a]);

    b.switch_to_block(else_bb);
    b.jump(merge, &[bb]);

    b.switch_to_block(merge);
    let result = b.block_param(merge, 0);
    b.ret(result);

    let func = b.build();
    assert_eq!(run_jit_linear_scan(&func, &[10, 20]), 20);
    assert_eq!(run_jit_linear_scan(&func, &[30, 20]), 30);
}

#[test]
fn linear_scan_loop_sum() {
    let mut b = FunctionBuilder::new("sum", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let n = b.block_param(entry, 0);

    let loop_bb = b.create_block(&[Type::I64, Type::I64]); // (i, acc)
    let exit = b.create_block(&[Type::I64]);

    let zero = b.iconst(Type::I64, 0);
    b.jump(loop_bb, &[zero, zero]);

    b.switch_to_block(loop_bb);
    let i = b.block_param(loop_bb, 0);
    let acc = b.block_param(loop_bb, 1);
    let cond = b.icmp(CmpOp::Slt, i, n);
    let new_acc = b.add(acc, i);
    let one = b.iconst(Type::I64, 1);
    let i_plus = b.add(i, one);
    b.br_if(cond, loop_bb, &[i_plus, new_acc], exit, &[acc]);

    b.switch_to_block(exit);
    let result = b.block_param(exit, 0);
    b.ret(result);

    let func = b.build();
    assert_eq!(run_jit_linear_scan(&func, &[0]), 0);
    assert_eq!(run_jit_linear_scan(&func, &[10]), 45);
    assert_eq!(run_jit_linear_scan(&func, &[100]), 4950);
}

#[test]
fn linear_scan_fib_loop() {
    let mut b = FunctionBuilder::new("fib", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let n = b.block_param(entry, 0);

    let loop_bb = b.create_block(&[Type::I64, Type::I64, Type::I64]); // (i, a, b)
    let exit = b.create_block(&[Type::I64]);

    let zero = b.iconst(Type::I64, 0);
    let one = b.iconst(Type::I64, 1);
    b.jump(loop_bb, &[zero, zero, one]);

    b.switch_to_block(loop_bb);
    let i = b.block_param(loop_bb, 0);
    let a = b.block_param(loop_bb, 1);
    let fib_b = b.block_param(loop_bb, 2);
    let cond = b.icmp(CmpOp::Slt, i, n);
    let next = b.add(a, fib_b);
    let i_plus = b.add(i, one);
    b.br_if(cond, loop_bb, &[i_plus, fib_b, next], exit, &[a]);

    b.switch_to_block(exit);
    let result = b.block_param(exit, 0);
    b.ret(result);

    let func = b.build();
    assert_eq!(run_jit_linear_scan(&func, &[0]), 0);
    assert_eq!(run_jit_linear_scan(&func, &[1]), 1);
    assert_eq!(run_jit_linear_scan(&func, &[10]), 55);
    assert_eq!(run_jit_linear_scan(&func, &[20]), 6765);
}

// ─── Incremental extend tests (microlisp prerequisite) ────────────

#[test]
fn extend_compile_then_extend_calls_first() {
    // Step 1: declare and compile `f(x) = x * 2`. Run it.
    // Step 2: extend with `g(x) = f(x) + 1`. Run g (which calls f via the
    // stable call table). Run f again to confirm it still works.
    let mut mb = ModuleBuilder::new();
    let f = mb.declare_func("f", &[Type::I64], Some(Type::I64));
    {
        let mut fb = mb.define_func(f);
        let entry = fb.entry_block();
        let x = fb.block_param(entry, 0);
        let two = fb.iconst(Type::I64, 2);
        let r = fb.mul(x, two);
        fb.ret(r);
        mb.finish_func(f, fb);
    }

    let mut jit = JitModule::new_empty::<
        DefaultJitConfig<NanBox>,
        crate::backend::Arm64Backend,
        crate::regalloc::GreedyRegState,
    >(64, 16, crate::CallMode::FastCall);

    let snap1 = mb.snapshot();
    let new1 = jit.extend::<
        DefaultJitConfig<NanBox>,
        crate::backend::Arm64Backend,
        crate::regalloc::GreedyRegState,
    >(&snap1, &[]);
    assert_eq!(new1, vec![f]);
    assert_eq!(jit.call(f, &[7]), 14);

    // Step 2: declare g, define it (it calls f), snapshot, extend.
    let g = mb.declare_func("g", &[Type::I64], Some(Type::I64));
    {
        let mut fb = mb.define_func(g);
        let entry = fb.entry_block();
        let x = fb.block_param(entry, 0);
        let doubled = fb.call(f, &[x]).unwrap();
        let one = fb.iconst(Type::I64, 1);
        let r = fb.add(doubled, one);
        fb.ret(r);
        mb.finish_func(g, fb);
    }

    let snap2 = mb.snapshot();
    let new2 = jit.extend::<
        DefaultJitConfig<NanBox>,
        crate::backend::Arm64Backend,
        crate::regalloc::GreedyRegState,
    >(&snap2, &[]);
    assert_eq!(new2, vec![g]);

    // g(10) = f(10) + 1 = 21
    assert_eq!(jit.call(g, &[10]), 21);
    // f still works after extend
    assert_eq!(jit.call(f, &[3]), 6);
    assert_eq!(jit.call(g, &[0]), 1);
}

#[test]
fn extend_three_batches_chained_calls() {
    // Compile f, then g calling f, then h calling g calling f — three extends.
    let mut mb = ModuleBuilder::new();
    let f = mb.declare_func("f", &[Type::I64], Some(Type::I64));
    {
        let mut fb = mb.define_func(f);
        let e = fb.entry_block();
        let x = fb.block_param(e, 0);
        let one = fb.iconst(Type::I64, 1);
        let r = fb.add(x, one);
        fb.ret(r);
        mb.finish_func(f, fb);
    }

    let mut jit = JitModule::new_empty::<
        DefaultJitConfig<NanBox>,
        crate::backend::Arm64Backend,
        crate::regalloc::GreedyRegState,
    >(64, 16, crate::CallMode::FastCall);
    let _ = jit.extend::<
        DefaultJitConfig<NanBox>,
        crate::backend::Arm64Backend,
        crate::regalloc::GreedyRegState,
    >(&mb.snapshot(), &[]);
    assert_eq!(jit.call(f, &[10]), 11);

    let g = mb.declare_func("g", &[Type::I64], Some(Type::I64));
    {
        let mut fb = mb.define_func(g);
        let e = fb.entry_block();
        let x = fb.block_param(e, 0);
        let r = fb.call(f, &[x]).unwrap();
        let r2 = fb.call(f, &[r]).unwrap();
        fb.ret(r2);
        mb.finish_func(g, fb);
    }
    let _ = jit.extend::<
        DefaultJitConfig<NanBox>,
        crate::backend::Arm64Backend,
        crate::regalloc::GreedyRegState,
    >(&mb.snapshot(), &[]);
    assert_eq!(jit.call(g, &[5]), 7);

    let h = mb.declare_func("h", &[Type::I64], Some(Type::I64));
    {
        let mut fb = mb.define_func(h);
        let e = fb.entry_block();
        let x = fb.block_param(e, 0);
        let r = fb.call(g, &[x]).unwrap();
        fb.ret(r);
        mb.finish_func(h, fb);
    }
    let _ = jit.extend::<
        DefaultJitConfig<NanBox>,
        crate::backend::Arm64Backend,
        crate::regalloc::GreedyRegState,
    >(&mb.snapshot(), &[]);
    assert_eq!(jit.call(h, &[100]), 102);

    // All three still callable directly.
    assert_eq!(jit.call(f, &[0]), 1);
    assert_eq!(jit.call(g, &[0]), 2);
    assert_eq!(jit.call(h, &[0]), 2);
}

#[test]
fn literal_pool_load_returns_slot_value() {
    // Function: just returns gc_literal(0). Push a value into slot 0,
    // verify the call reads it. Then mutate the slot (simulating a GC
    // relocation in place) and verify the call sees the new value.
    let mut mb = ModuleBuilder::new();
    let f = mb.declare_func("read_lit", &[], Some(Type::I64));
    let lit = LiteralRef::from_u32(0);
    {
        let mut fb = mb.define_func(f);
        let v = fb.gc_literal(lit);
        fb.ret(v);
        mb.finish_func(f, fb);
    }

    let mut jit = JitModule::new_empty::<
        DefaultJitConfig<NanBox>,
        crate::backend::Arm64Backend,
        crate::regalloc::GreedyRegState,
    >(64, 16, crate::CallMode::FastCall);

    let idx = jit.literal_pool().push(0xDEAD_BEEF);
    assert_eq!(idx, 0);
    let _ = jit.extend::<
        DefaultJitConfig<NanBox>,
        crate::backend::Arm64Backend,
        crate::regalloc::GreedyRegState,
    >(&mb.snapshot(), &[]);

    assert_eq!(jit.call(f, &[]), 0xDEAD_BEEF);

    // Mutate the slot — the next call must see the new value because the
    // emitted code reads through the pool, never bakes the value in.
    jit.literal_pool().set(0, 0x1234_5678);
    assert_eq!(jit.call(f, &[]), 0x1234_5678);

    jit.literal_pool().set(0, 42);
    assert_eq!(jit.call(f, &[]), 42);
}

#[test]
fn literal_pool_scan_roots_can_rewrite_slots_in_place() {
    // Simulates a moving GC: scan_roots gives us *mut u64 to each live slot,
    // and we rewrite it. The next call must reflect the new value.
    use dynobj::RootSource;

    let mut mb = ModuleBuilder::new();
    let f = mb.declare_func("read_lit", &[], Some(Type::I64));
    {
        let mut fb = mb.define_func(f);
        let v0 = fb.gc_literal(LiteralRef::from_u32(0));
        let v1 = fb.gc_literal(LiteralRef::from_u32(1));
        let s = fb.add(v0, v1);
        fb.ret(s);
        mb.finish_func(f, fb);
    }

    let mut jit = JitModule::new_empty::<
        DefaultJitConfig<NanBox>,
        crate::backend::Arm64Backend,
        crate::regalloc::GreedyRegState,
    >(64, 16, crate::CallMode::FastCall);
    jit.literal_pool().push(10);
    jit.literal_pool().push(20);
    let _ = jit.extend::<
        DefaultJitConfig<NanBox>,
        crate::backend::Arm64Backend,
        crate::regalloc::GreedyRegState,
    >(&mb.snapshot(), &[]);
    assert_eq!(jit.call(f, &[]), 30);

    // Pretend the GC ran and "relocated" each value, doubling it.
    let pool = jit.literal_pool();
    pool.scan_roots(&mut |slot_ptr| unsafe {
        let old = *slot_ptr;
        *slot_ptr = old * 2;
    });
    assert_eq!(jit.call(f, &[]), 60); // (10*2) + (20*2)
}

#[test]
fn literal_pool_survives_extend() {
    // Push a literal, compile a function that reads it, extend, run.
    // Then push ANOTHER literal, compile a second function that reads slot 1,
    // extend again, run. Both functions resolve correctly through the same
    // stable pool base.
    let mut mb = ModuleBuilder::new();
    let f = mb.declare_func("read0", &[], Some(Type::I64));
    {
        let mut fb = mb.define_func(f);
        let v = fb.gc_literal(LiteralRef::from_u32(0));
        fb.ret(v);
        mb.finish_func(f, fb);
    }

    let mut jit = JitModule::new_empty::<
        DefaultJitConfig<NanBox>,
        crate::backend::Arm64Backend,
        crate::regalloc::GreedyRegState,
    >(64, 16, crate::CallMode::FastCall);
    jit.literal_pool().push(100);
    let _ = jit.extend::<
        DefaultJitConfig<NanBox>,
        crate::backend::Arm64Backend,
        crate::regalloc::GreedyRegState,
    >(&mb.snapshot(), &[]);
    assert_eq!(jit.call(f, &[]), 100);

    let g = mb.declare_func("read1", &[], Some(Type::I64));
    {
        let mut fb = mb.define_func(g);
        let v = fb.gc_literal(LiteralRef::from_u32(1));
        fb.ret(v);
        mb.finish_func(g, fb);
    }
    jit.literal_pool().push(200);
    let _ = jit.extend::<
        DefaultJitConfig<NanBox>,
        crate::backend::Arm64Backend,
        crate::regalloc::GreedyRegState,
    >(&mb.snapshot(), &[]);

    assert_eq!(jit.call(f, &[]), 100);
    assert_eq!(jit.call(g, &[]), 200);

    // Update both, verify each function tracks its own slot.
    jit.literal_pool().set(0, 999);
    jit.literal_pool().set(1, 1001);
    assert_eq!(jit.call(f, &[]), 999);
    assert_eq!(jit.call(g, &[]), 1001);
}

#[test]
fn extend_with_extern_added_in_second_batch() {
    use dynir::types::Signature;

    extern "C" fn add_seven(x: u64) -> u64 {
        x + 7
    }

    let mut mb = ModuleBuilder::new();
    let f = mb.declare_func("f", &[Type::I64], Some(Type::I64));
    {
        let mut fb = mb.define_func(f);
        let e = fb.entry_block();
        let x = fb.block_param(e, 0);
        let two = fb.iconst(Type::I64, 2);
        let r = fb.mul(x, two);
        fb.ret(r);
        mb.finish_func(f, fb);
    }

    let mut jit = JitModule::new_empty::<
        DefaultJitConfig<NanBox>,
        crate::backend::Arm64Backend,
        crate::regalloc::GreedyRegState,
    >(64, 16, crate::CallMode::FastCall);
    let _ = jit.extend::<
        DefaultJitConfig<NanBox>,
        crate::backend::Arm64Backend,
        crate::regalloc::GreedyRegState,
    >(&mb.snapshot(), &[]);
    assert_eq!(jit.call(f, &[3]), 6);

    // Add an extern + a function that uses it.
    let ext = mb.declare_extern(
        "add_seven",
        Signature {
            params: vec![Type::I64],
            ret: Some(Type::I64),
        },
    );
    let g = mb.declare_func("g", &[Type::I64], Some(Type::I64));
    {
        let mut fb = mb.define_func(g);
        let e = fb.entry_block();
        let x = fb.block_param(e, 0);
        let doubled = fb.call(f, &[x]).unwrap();
        let plus_seven = fb.call(ext, &[doubled]).unwrap();
        fb.ret(plus_seven);
        mb.finish_func(g, fb);
    }
    let externs: &[*const u8] = &[add_seven as *const u8];
    let _ = jit.extend::<
        DefaultJitConfig<NanBox>,
        crate::backend::Arm64Backend,
        crate::regalloc::GreedyRegState,
    >(&mb.snapshot(), externs);

    // g(4) = f(4) + 7 = 8 + 7 = 15
    assert_eq!(jit.call(g, &[4]), 15);
    assert_eq!(jit.call(f, &[10]), 20);
}

// ────────── Raise + push_handler (exception primitive) ──────────

#[test]
fn raise_without_handler_surfaces_as_jit_outcome_exception() {
    let mut b = FunctionBuilder::new("raise_top", &[], Some(Type::I64));
    let v = b.iconst(Type::I64, 99);
    b.raise(v);
    let func = b.build();

    match run_jit_outcome(&func, &[]) {
        JitOutcome::Exception(v) => assert_eq!(v, 99),
        other => panic!("expected Exception(99), got {other:?}"),
    }
}

#[test]
fn raise_inside_local_handler_jumps_to_catch_block() {
    // push_handler h ; raise 42 ; h: ret(param + 1) → 43
    let mut b = FunctionBuilder::new("local_catch", &[], Some(Type::I64));
    let handler_bb = b.create_block(&[Type::I64]);
    b.push_handler(handler_bb);
    let v = b.iconst(Type::I64, 42);
    b.raise(v);

    b.switch_to_block(handler_bb);
    let caught = b.block_param(handler_bb, 0);
    let one = b.iconst(Type::I64, 1);
    let r = b.add(caught, one);
    b.ret(r);

    let func = b.build();
    assert_eq!(run_jit(&func, &[]), 43);
}

#[test]
fn raise_from_callee_caught_by_caller_handler() {
    // f_throw(x): raise(x)
    // f_main(x): push_handler h; call f_throw(x); unreachable.
    //            h: ret(param + 1)        → main(41) = 42
    let mut mb = ModuleBuilder::new();
    let f_throw = mb.declare_func("throw_v", &[Type::I64], Some(Type::I64));
    let f_main = mb.declare_func("main", &[Type::I64], Some(Type::I64));

    let mut fb = mb.define_func(f_throw);
    let entry = fb.entry_block();
    let v = fb.block_param(entry, 0);
    fb.raise(v);
    mb.finish_func(f_throw, fb);

    let mut fb = mb.define_func(f_main);
    let entry = fb.entry_block();
    let x = fb.block_param(entry, 0);
    let handler_bb = fb.create_block(&[Type::I64]);
    fb.push_handler(handler_bb);
    let _ = fb.call(f_throw, &[x]);
    fb.unreachable();

    fb.switch_to_block(handler_bb);
    let caught = fb.block_param(handler_bb, 0);
    let one = fb.iconst(Type::I64, 1);
    let r = fb.add(caught, one);
    fb.ret(r);
    mb.finish_func(f_main, fb);

    let module = mb.build();
    let jit = JitModule::compile::<NanBox>(&module, &[]);
    assert_eq!(jit.call(f_main, &[41]), 42);
}

#[test]
fn call_with_handler_active_returns_normally_when_callee_doesnt_throw() {
    // f_id(x): ret(x)
    // f_main(x): push_handler h; ret(call f_id(x)); h: ret(param + 1000)
    // Calling main(7) should land on the *normal* path → 7, NOT 1007.
    let mut mb = ModuleBuilder::new();
    let f_id = mb.declare_func("id", &[Type::I64], Some(Type::I64));
    let f_main = mb.declare_func("main", &[Type::I64], Some(Type::I64));

    let mut fb = mb.define_func(f_id);
    let entry = fb.entry_block();
    let v = fb.block_param(entry, 0);
    fb.ret(v);
    mb.finish_func(f_id, fb);

    let mut fb = mb.define_func(f_main);
    let entry = fb.entry_block();
    let x = fb.block_param(entry, 0);
    let handler_bb = fb.create_block(&[Type::I64]);
    fb.push_handler(handler_bb);
    let r = fb.call(f_id, &[x]).unwrap();
    fb.ret(r);

    fb.switch_to_block(handler_bb);
    let caught = fb.block_param(handler_bb, 0);
    let k = fb.iconst(Type::I64, 1000);
    let r2 = fb.add(caught, k);
    fb.ret(r2);
    mb.finish_func(f_main, fb);

    let module = mb.build();
    let jit = JitModule::compile::<NanBox>(&module, &[]);
    assert_eq!(jit.call(f_main, &[7]), 7);
}

#[test]
fn raise_in_callee_caught_by_caller_invoke_directly() {
    // Same as raise_from_callee_caught_by_caller_handler but uses
    // explicit `fb.invoke` instead of push_handler + plain Call. If
    // this works but the push_handler version doesn't, my push_handler
    // routing in lower_call has a bug.
    let mut mb = ModuleBuilder::new();
    let f_throw = mb.declare_func("throw_v", &[Type::I64], Some(Type::I64));
    let f_main = mb.declare_func("main", &[Type::I64], Some(Type::I64));

    let mut fb = mb.define_func(f_throw);
    let entry = fb.entry_block();
    let v = fb.block_param(entry, 0);
    fb.raise(v);
    mb.finish_func(f_throw, fb);

    let mut fb = mb.define_func(f_main);
    let entry = fb.entry_block();
    let x = fb.block_param(entry, 0);
    let normal_bb = fb.create_block(&[Type::I64]);
    let handler_bb = fb.create_block(&[Type::I64]);
    fb.invoke(f_throw, &[x], normal_bb, &[], handler_bb, &[]);

    fb.switch_to_block(normal_bb);
    let _ = fb.block_param(normal_bb, 0);
    fb.unreachable();

    fb.switch_to_block(handler_bb);
    let caught = fb.block_param(handler_bb, 0);
    let one = fb.iconst(Type::I64, 1);
    let r = fb.add(caught, one);
    fb.ret(r);
    mb.finish_func(f_main, fb);

    let module = mb.build();
    let jit = JitModule::compile::<NanBox>(&module, &[]);
    assert_eq!(jit.call(f_main, &[41]), 42);
}

#[test]
fn raise_propagates_through_no_handler_frame_to_outer_handler() {
    // f_inner(x): raise(x)
    // f_mid(x): call f_inner(x); unreachable.   (no handler; propagates)
    // f_outer(x): push_handler h; call f_mid(x); unreachable.
    //             h: ret(param + 100)            → outer(5) = 105
    let mut mb = ModuleBuilder::new();
    let f_inner = mb.declare_func("inner", &[Type::I64], Some(Type::I64));
    let f_mid = mb.declare_func("mid", &[Type::I64], Some(Type::I64));
    let f_outer = mb.declare_func("outer", &[Type::I64], Some(Type::I64));

    let mut fb = mb.define_func(f_inner);
    let entry = fb.entry_block();
    let v = fb.block_param(entry, 0);
    fb.raise(v);
    mb.finish_func(f_inner, fb);

    let mut fb = mb.define_func(f_mid);
    let entry = fb.entry_block();
    let x = fb.block_param(entry, 0);
    let _ = fb.call(f_inner, &[x]);
    fb.unreachable();
    mb.finish_func(f_mid, fb);

    let mut fb = mb.define_func(f_outer);
    let entry = fb.entry_block();
    let x = fb.block_param(entry, 0);
    let handler_bb = fb.create_block(&[Type::I64]);
    fb.push_handler(handler_bb);
    let _ = fb.call(f_mid, &[x]);
    fb.unreachable();

    fb.switch_to_block(handler_bb);
    let caught = fb.block_param(handler_bb, 0);
    let hundred = fb.iconst(Type::I64, 100);
    let r = fb.add(caught, hundred);
    fb.ret(r);
    mb.finish_func(f_outer, fb);

    let module = mb.build();
    let jit = JitModule::compile::<NanBox>(&module, &[]);
    assert_eq!(jit.call(f_outer, &[5]), 105);
}

#[test]
fn nested_handlers_innermost_catches_first_jit() {
    let mut b = FunctionBuilder::new("nested", &[], Some(Type::I64));
    let outer_bb = b.create_block(&[Type::I64]);
    let inner_bb = b.create_block(&[Type::I64]);

    b.push_handler(outer_bb);
    b.push_handler(inner_bb);
    let v = b.iconst(Type::I64, 5);
    b.raise(v);

    b.switch_to_block(inner_bb);
    let caught = b.block_param(inner_bb, 0);
    let ten = b.iconst(Type::I64, 10);
    let r = b.add(caught, ten);
    b.ret(r);

    b.switch_to_block(outer_bb);
    let caught = b.block_param(outer_bb, 0);
    let hundred = b.iconst(Type::I64, 100);
    let r = b.add(caught, hundred);
    b.ret(r);

    let func = b.build();
    assert_eq!(run_jit(&func, &[]), 15);
}

#[test]
fn nine_arg_call_last_arg() {
    // The real bug shape: a call with N args where N exceeds the
    // allocatable register count. Callee returns its LAST arg. main
    // computes N distinct values (via id calls so they're live across the
    // big call's arg setup), then calls the N-ary callee. Expect last arg.
    for n in [7usize, 8, 9, 10, 11] {
        let mut mb = ModuleBuilder::new();
        let f_main = mb.declare_func("main", &[], Some(Type::I64));
        let f_id = mb.declare_func("id", &[Type::I64], Some(Type::I64));
        let params = vec![Type::I64; n];
        let f_nary = mb.declare_func("nary", &params, Some(Type::I64));

        let mut mfb = mb.define_func(f_main);
        let _e = mfb.entry_block();
        let mut vs = Vec::new();
        for i in 0..n {
            let c = mfb.iconst(Type::I64, (i as i64 + 1) * 100);
            vs.push(mfb.call(f_id, &[c]).unwrap());
        }
        let r = mfb.call(f_nary, &vs).unwrap();
        mfb.ret(r);
        mb.finish_func(f_main, mfb);

        let mut idfb = mb.define_func(f_id);
        let _ = idfb.entry_block();
        let p = idfb.block_param(idfb.entry_block(), 0);
        idfb.ret(p);
        mb.finish_func(f_id, idfb);

        // nary returns its LAST param.
        let mut nfb = mb.define_func(f_nary);
        let eb = nfb.entry_block();
        let last = nfb.block_param(eb, n - 1);
        nfb.ret(last);
        mb.finish_func(f_nary, nfb);

        let module = mb.build();
        let jit = JitModule::compile_with_regalloc::<
            NanBoxConfig,
            Arm64Backend,
            crate::regalloc::LinearScanAllocator,
        >(&module, &[], None);
        let want = (n as u64) * 100;
        let got = match jit.call_outcome(f_main, &[]) {
            JitOutcome::Value(v) => v,
            other => panic!("n={n}: {other:?}"),
        };
        eprintln!(
            "n={n}: last-arg got={got} want={want} {}",
            if got == want { "ok" } else { "CORRUPT" }
        );
        assert_eq!(got, want, "n={n}");
    }
}

#[test]
fn eight_call_results_folded_across_calls() {
    // Reproducer for the regalloc spill bug: N call-results held live
    // across a fold of further calls. With 7 allocatable GP regs, N=8
    // forces spilling; the spilled value must round-trip correctly.
    // id(x) = x ; addfn(a,b) = a+b. main: vi = id(i) for i in 0..N,
    // then acc = id(v0); acc = addfn(acc, vi)... ; ret acc.
    // Expected = 0+1+...+(N-1).
    for n in [7usize, 8, 9, 10] {
        let mut mb = ModuleBuilder::new();
        let f_main = mb.declare_func("main", &[], Some(Type::I64));
        let f_id = mb.declare_func("id", &[Type::I64], Some(Type::I64));
        let f_add = mb.declare_func("addfn", &[Type::I64, Type::I64], Some(Type::I64));

        let mut mfb = mb.define_func(f_main);
        let _e = mfb.entry_block();
        // Compute v0..v_{n-1} via id calls.
        let mut vs = Vec::new();
        for i in 0..n {
            let c = mfb.iconst(Type::I64, i as i64);
            vs.push(mfb.call(f_id, &[c]).unwrap());
        }
        // Fold RIGHT-TO-LEFT (consume last-defined value first), mirroring
        // the cons-fold in variadic packing. Start from a fresh const acc.
        let mut acc = mfb.iconst(Type::I64, 0);
        for i in (0..n).rev() {
            acc = mfb.call(f_add, &[vs[i], acc]).unwrap();
        }
        // Trailing call consuming the folded acc (mirrors the `list` call).
        let r = mfb.call(f_id, &[acc]).unwrap();
        mfb.ret(r);
        mb.finish_func(f_main, mfb);

        let mut idfb = mb.define_func(f_id);
        let _ = idfb.entry_block();
        let p = idfb.block_param(idfb.entry_block(), 0);
        idfb.ret(p);
        mb.finish_func(f_id, idfb);

        let mut afb = mb.define_func(f_add);
        let _ = afb.entry_block();
        let a = afb.block_param(afb.entry_block(), 0);
        let b = afb.block_param(afb.entry_block(), 1);
        let s = afb.add(a, b);
        afb.ret(s);
        mb.finish_func(f_add, afb);

        let module = mb.build();
        // Use the EXACT config clojure-jvm uses: NanBoxConfig (internal CC,
        // 7 allocatable GP regs, StackMap roots) + LinearScan.
        let jit = JitModule::compile_with_regalloc::<
            NanBoxConfig,
            Arm64Backend,
            crate::regalloc::LinearScanAllocator,
        >(&module, &[], None);
        let want = (0..n as u64).sum::<u64>();
        let got = match jit.call_outcome(f_main, &[]) {
            JitOutcome::Value(v) => v,
            other => panic!("n={n}: unexpected {other:?}"),
        };
        eprintln!(
            "n={n}: got={got} want={want} {}",
            if got == want { "ok" } else { "CORRUPT" }
        );
        assert_eq!(got, want, "n={n}");
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn test_ret_9th(
    _a: u64,
    _b: u64,
    _c: u64,
    _d: u64,
    _e: u64,
    _f: u64,
    _g: u64,
    _h: u64,
    i: u64,
) -> u64 {
    i
}

#[test]
fn jit_calls_extern_c_with_nine_args() {
    use dynir::types::Signature;
    // The JIT (internal CC) calls an `extern "C"` fn with 9 args. The C ABI
    // passes args 0-7 in X0-X7 and arg 8 on the stack; the internal CC would
    // put arg 8 in X8. If the JIT doesn't switch to the C ABI for extern
    // calls, the callee reads its 9th arg from the stack → garbage.
    let mut mb = ModuleBuilder::new();
    let sig9 = Signature {
        params: vec![Type::I64; 9],
        ret: Some(Type::I64),
    };
    let f_ext = mb.declare_extern("test_ret_9th", sig9);
    let f_main = mb.declare_func("main", &[], Some(Type::I64));

    let mut mfb = mb.define_func(f_main);
    let _e = mfb.entry_block();
    let mut vs = Vec::new();
    for i in 0..9 {
        vs.push(mfb.iconst(Type::I64, (i as i64 + 1) * 11));
    }
    let r = mfb.call(f_ext, &vs).unwrap();
    mfb.ret(r);
    mb.finish_func(f_main, mfb);

    let module = mb.build();
    let externs: &[*const u8] = &[test_ret_9th as *const u8];
    let jit = JitModule::compile_with_regalloc::<
        NanBoxConfig,
        Arm64Backend,
        crate::regalloc::LinearScanAllocator,
    >(&module, externs, None);
    let got = match jit.call_outcome(f_main, &[]) {
        JitOutcome::Value(v) => v,
        other => panic!("{other:?}"),
    };
    eprintln!(
        "9th arg: got={got} want=99 {}",
        if got == 99 { "ok" } else { "CORRUPT" }
    );
    assert_eq!(got, 99, "9th extern arg corrupted");
}

#[test]
fn cross_block_live_values_under_pressure() {
    // Reproducer for the LinearScan cross-block-liveness bug: values
    // defined in the entry block are used after a branch merge, staying
    // live across a branch arm that itself has calls + register pressure.
    // Mirrors the `ns`/`mmin` macro IR shape.
    for n in [6usize, 8, 10, 12] {
        let mut mb = ModuleBuilder::new();
        let f_main = mb.declare_func("main", &[], Some(Type::I64));
        let f_id = mb.declare_func("id", &[Type::I64], Some(Type::I64));
        let f_add = mb.declare_func("addfn", &[Type::I64, Type::I64], Some(Type::I64));

        let mut mfb = mb.define_func(f_main);
        let entry = mfb.entry_block();
        // Cross-block values defined in entry.
        let mut vs = Vec::new();
        for i in 0..n {
            let c = mfb.iconst(Type::I64, (i as i64 + 1) * 100);
            vs.push(mfb.call(f_id, &[c]).unwrap());
        }
        let then_b = mfb.create_block(&[]);
        let else_b = mfb.create_block(&[]);
        let merge = mfb.create_block(&[Type::I64]);
        let cond = mfb.iconst(Type::I8, 1);
        mfb.br_if(cond, then_b, &[], else_b, &[]);

        // then: extra calls (pressure) while vs are live across them.
        mfb.switch_to_block(then_b);
        let mut t = mfb.iconst(Type::I64, 0);
        for _ in 0..3 {
            t = mfb.call(f_id, &[t]).unwrap();
        }
        mfb.jump(merge, &[t]);

        mfb.switch_to_block(else_b);
        let z = mfb.iconst(Type::I64, 0);
        mfb.jump(merge, &[z]);

        // merge: sum all cross-block vs + the merge param.
        mfb.switch_to_block(merge);
        let p = mfb.block_param(merge, 0);
        let mut acc = p;
        for v in &vs {
            acc = mfb.call(f_add, &[acc, *v]).unwrap();
        }
        mfb.ret(acc);
        mb.finish_func(f_main, mfb);

        let mut idfb = mb.define_func(f_id);
        let e = idfb.entry_block();
        let pp = idfb.block_param(e, 0);
        idfb.ret(pp);
        mb.finish_func(f_id, idfb);

        let mut afb = mb.define_func(f_add);
        let ae = afb.entry_block();
        let aa = afb.block_param(ae, 0);
        let ab = afb.block_param(ae, 1);
        let asum = afb.add(aa, ab);
        afb.ret(asum);
        mb.finish_func(f_add, afb);

        let module = mb.build();
        let jit = JitModule::compile_with_regalloc::<
            NanBoxConfig,
            Arm64Backend,
            crate::regalloc::LinearScanAllocator,
        >(&module, &[], None);
        // then-arm taken (cond=1): t = id(id(id(0))) = 0. acc = 0 + sum(vs).
        let want: u64 = (0..n as u64).map(|i| (i + 1) * 100).sum();
        let got = match jit.call_outcome(f_main, &[]) {
            JitOutcome::Value(v) => v,
            other => panic!("n={n}: {other:?}"),
        };
        eprintln!(
            "n={n}: got={got} want={want} {}",
            if got == want { "ok" } else { "CORRUPT" }
        );
        assert_eq!(got, want, "n={n}");
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn test_id1(x: u64) -> u64 {
    x
}
#[unsafe(no_mangle)]
pub extern "C" fn test_add2(a: u64, b: u64) -> u64 {
    a.wrapping_add(b)
}

#[test]
fn cross_block_live_values_extern_calls() {
    use dynir::types::Signature;
    // Like cross_block_live_values_under_pressure but the producing and
    // consuming calls are `extern "C"` (mirrors clojure's dynamic-invoke
    // externs). Cross-block values held across an extern-call-heavy arm.
    for n in [6usize, 8, 10, 12] {
        let mut mb = ModuleBuilder::new();
        let f_main = mb.declare_func("main", &[], Some(Type::I64));
        let f_id = mb.declare_extern(
            "test_id1",
            Signature {
                params: vec![Type::I64],
                ret: Some(Type::I64),
            },
        );
        let f_add = mb.declare_extern(
            "test_add2",
            Signature {
                params: vec![Type::I64, Type::I64],
                ret: Some(Type::I64),
            },
        );

        let mut mfb = mb.define_func(f_main);
        let _entry = mfb.entry_block();
        let mut vs = Vec::new();
        for i in 0..n {
            let c = mfb.iconst(Type::I64, (i as i64 + 1) * 100);
            vs.push(mfb.call(f_id, &[c]).unwrap());
        }
        let then_b = mfb.create_block(&[]);
        let else_b = mfb.create_block(&[]);
        let merge = mfb.create_block(&[Type::I64]);
        // Mirror `when`'s cond: icmp on i64 values + or/xor producing an i8,
        // mixed in with the live i64 cross-block values.
        let k1 = mfb.iconst(Type::I64, 1);
        let e1 = mfb.icmp(dynir::ir::CmpOp::Eq, vs[0], k1);
        let e2 = mfb.icmp(dynir::ir::CmpOp::Eq, vs[1], k1);
        let orr = mfb.or(e1, e2);
        let one = mfb.iconst(Type::I8, 1);
        let cond = mfb.xor(orr, one);
        mfb.br_if(cond, then_b, &[], else_b, &[]);

        mfb.switch_to_block(then_b);
        let mut t = mfb.iconst(Type::I64, 0);
        for _ in 0..3 {
            t = mfb.call(f_id, &[t]).unwrap();
        }
        mfb.jump(merge, &[t]);

        mfb.switch_to_block(else_b);
        let z = mfb.iconst(Type::I64, 0);
        mfb.jump(merge, &[z]);

        mfb.switch_to_block(merge);
        let p = mfb.block_param(merge, 0);
        let mut acc = p;
        for v in &vs {
            acc = mfb.call(f_add, &[acc, *v]).unwrap();
        }
        mfb.ret(acc);
        mb.finish_func(f_main, mfb);

        let module = mb.build();
        let externs: &[*const u8] = &[test_id1 as *const u8, test_add2 as *const u8];
        let jit = JitModule::compile_with_regalloc::<
            NanBoxConfig,
            Arm64Backend,
            crate::regalloc::LinearScanAllocator,
        >(&module, externs, None);
        let want: u64 = (0..n as u64).map(|i| (i + 1) * 100).sum();
        let got = match jit.call_outcome(f_main, &[]) {
            JitOutcome::Value(v) => v,
            other => panic!("n={n}: {other:?}"),
        };
        eprintln!(
            "n={n}: got={got} want={want} {}",
            if got == want { "ok" } else { "CORRUPT" }
        );
        assert_eq!(got, want, "n={n}");
    }
}
