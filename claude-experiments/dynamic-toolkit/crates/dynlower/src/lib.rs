use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::io::Write;
use std::marker::PhantomData;
use std::sync::RwLock;
use std::sync::atomic::{AtomicUsize, Ordering};

pub mod backend;
pub mod batch_lower;
pub mod regalloc;
pub mod regalloc_bridge;

pub use backend::Arm64Backend;

#[cfg(target_arch = "x86_64")]
use backend::X64Backend;
use backend::{
    LoweringBackend, MachineFpBinOp, MachineGpBinOp, MachineLocation, MachineWordSize,
};
use dynasm::buffer::{CodeBuffer, Label};
use dynasm::code_memory::{CodeMemory, PagedCodeMemory};
use dynexec::{
    BuilderFrame, CallArgLocation, CallingConvention, CodegenConfig, DefaultCodegenConfig,
    FrameLayout, FrameResume, FrameResumePoint, FrameSlotAccess, FrameSlotBase, FrameStrategy,
    LayoutConfigDefaults, RootTransport, RootTransportKind, SoundRoots, SoundTransport,
    validate_codegen_config,
};
use dynir::ir::*;
use dynir::types::Type;
use dynvalue::TagScheme;
use regalloc::{
    GreedyRegState, RegisterAllocator, ValueLoc, is_float_type, machine_fp, machine_gp,
};

#[cfg(test)]
mod tests;

// ─── JIT Code Registry & FP-Chain Root Scanner ───────────────────
//
// For multi-frame GC root scanning: instead of per-call push/pop overhead,
// each JitFunction/JitModule registers its code range and frame scan size
// at compile time. At GC time, the safepoint handler walks the native
// ARM64 frame pointer chain (FP → [FP] → [[FP]] → ...) and looks up each
// return address to identify JIT frames and their root scan regions.
//
// Zero per-call overhead. All cost is at compile time (register) and GC
// time (walk + binary search).

/// A registered JIT code region.
///
/// Holds enough information for the ancestor-frame walker to scan only
/// the slots live at the specific PC of an ancestor — not the entire
/// function's root-slot region.
#[derive(Clone, Debug)]
struct JitCodeEntry {
    code_start: usize, // inclusive
    code_end: usize,   // exclusive
    /// Per-PC live-slot map. Every internal `BLR` in this function
    /// (both explicit `Inst::Safepoint` calls and ordinary
    /// `Inst::Call`s) pushes a record here whose `return_offset`
    /// equals the offset of the byte immediately after the `BLR`.
    /// Records are sorted by `return_offset`, so lookup is O(log n).
    safepoints: std::sync::Arc<[SafepointRecord]>,
}

static JIT_CODE_REGISTRY: RwLock<Vec<JitCodeEntry>> = RwLock::new(Vec::new());

fn register_jit_code(
    code_start: usize,
    code_end: usize,
    safepoints: std::sync::Arc<[SafepointRecord]>,
) {
    let mut registry = JIT_CODE_REGISTRY.write().unwrap();
    let pos = registry.partition_point(|e| e.code_start < code_start);
    registry.insert(
        pos,
        JitCodeEntry {
            code_start,
            code_end,
            safepoints,
        },
    );
}

fn unregister_jit_code(code_start: usize) {
    let mut registry = JIT_CODE_REGISTRY.write().unwrap();
    if let Ok(pos) = registry.binary_search_by_key(&code_start, |e| e.code_start) {
        registry.remove(pos);
    }
}

fn lookup_code_entry(registry: &[JitCodeEntry], addr: usize) -> Option<&JitCodeEntry> {
    let idx = registry.partition_point(|e| e.code_start <= addr);
    if idx == 0 {
        return None;
    }
    let entry = &registry[idx - 1];
    if addr < entry.code_end {
        Some(entry)
    } else {
        None
    }
}

// ─── perf-pid.map support ──────────────────────────────────────────
//
// Write entries to /tmp/perf-<pid>.map so profilers (perf, samply, etc.)
// can resolve JIT-compiled function addresses to symbolic names.
// Format per line: `<hex_start_addr> <hex_size> <name>\n`

fn write_perf_map_entries(entries: &[(usize, usize, &str)]) {
    let path = format!("/tmp/perf-{}.map", std::process::id());
    let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    else {
        return;
    };
    for &(addr, size, name) in entries {
        let _ = writeln!(file, "{addr:x} {size:x} {name}");
    }
}

// ─── JIT-entry FP fence ────────────────────────────────────────────
//
// The FP-chain walker (`walk_jit_ancestor_roots`) walks UP from a JIT
// frame's saved FP, looking up each saved LR in the JIT code registry.
// Without a stop condition it would happily walk into Rust frames above
// the JIT call boundary. There it could:
//   - Pattern-match a Rust-saved LR against the JIT registry by accident
//     (vanishingly unlikely, but possible if JIT code lives near Rust
//     code in address space and the LR happens to fall inside a JIT range).
//   - Follow a Rust frame's saved-FP into compiler-elided territory
//     (Rust doesn't always preserve x29 along the chain).
//
// The fix: every Rust → JIT trampoline records the Rust caller's FP onto
// a per-thread fence stack just before the BLR, and pops on return. The
// walker stops when it reaches the topmost fence — which is exactly the
// FP the JIT prologue saved when the trampoline entered JIT.
//
// Nested calls (Rust → JIT → extern → Rust → JIT → …) are supported via
// the stack: each nested entry pushes its own fence; the walker uses the
// topmost. Outer JIT frames above an intervening Rust segment are not
// scanned by an inner-call collection — those values are the Rust
// segment's responsibility (it must hold them in a `RootSet` or
// `FrameChain`).

thread_local! {
    static JIT_ENTRY_FP_STACK: RefCell<Vec<*const u8>> = const { RefCell::new(Vec::new()) };
}

/// Push the current JIT-entry FP fence. Must be paired with `pop_jit_entry_fp`
/// in LIFO order. Typical use: the trampoline (`call_jit_regs_with_reg_limit`)
/// captures `x29` immediately before the BLR and pushes it.
///
/// # Safety
/// `fp` must point to a real Rust caller frame whose stack memory remains
/// valid until the matching pop. The trampoline guarantees this by
/// capturing `x29` of its own frame and popping before the trampoline
/// returns.
pub unsafe fn push_jit_entry_fp(fp: *const u8) {
    JIT_ENTRY_FP_STACK.with(|c| c.borrow_mut().push(fp));
}

/// Pop the top of the fence stack. Must match a prior `push_jit_entry_fp`.
pub fn pop_jit_entry_fp() {
    JIT_ENTRY_FP_STACK.with(|c| {
        let popped = c.borrow_mut().pop();
        debug_assert!(popped.is_some(), "pop_jit_entry_fp on empty stack");
    });
}

/// The topmost fence FP, or null if no JIT call is active on this thread.
fn current_jit_entry_fp() -> *const u8 {
    JIT_ENTRY_FP_STACK.with(|c| c.borrow().last().copied().unwrap_or(std::ptr::null()))
}

/// RAII guard that pushes a fence on construction and pops on drop.
pub struct JitEntryFpGuard {
    _phantom: PhantomData<*const ()>, // !Send + !Sync; thread-local
}

impl JitEntryFpGuard {
    /// # Safety
    /// See `push_jit_entry_fp`.
    pub unsafe fn new(fp: *const u8) -> Self {
        unsafe { push_jit_entry_fp(fp) };
        JitEntryFpGuard { _phantom: PhantomData }
    }
}

impl Drop for JitEntryFpGuard {
    fn drop(&mut self) {
        pop_jit_entry_fp();
    }
}

/// Walk ancestor JIT frames starting from `jit_fp` (the FP of the frame
/// that hit the safepoint). The current frame is skipped — the safepoint
/// handler already scans it via the `(frame_ptr, payload)` arguments.
///
/// For each ancestor, `[fp+8]` is the saved return address into that
/// ancestor. We look up the `SafepointRecord` for that exact PC and
/// scan *only* the root slots recorded as live at that point —
/// conservative whole-frame sweeping would resurrect stale spill slots
/// as phantom roots.
///
/// Stops when `saved_fp` reaches the topmost JIT-entry FP fence: that's
/// the frame the Rust caller of `call_jit_outcome` was in. Anything
/// above is host-side stack, not JIT — its roots (if any) are managed
/// by the host's `FrameChain`.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub fn walk_jit_ancestor_roots(jit_fp: *const u8, visitor: &mut dyn FnMut(*mut u64)) {
    let registry = JIT_CODE_REGISTRY.read().unwrap();
    if registry.is_empty() {
        return;
    }
    let fence = current_jit_entry_fp();

    let mut fp = jit_fp as *const u64;
    loop {
        if fp.is_null() {
            break;
        }
        let saved_fp = unsafe { *fp } as *const u64;
        let saved_lr = unsafe { *fp.add(1) } as usize;

        if saved_fp.is_null() {
            break;
        }
        // Stop at the JIT-entry boundary: `saved_fp` here is the FP of
        // the Rust frame that called `call_jit_outcome`. Above it is
        // host code, not JIT.
        if !fence.is_null() && (saved_fp as *const u8) == fence {
            break;
        }

        if let Some(entry) = lookup_code_entry(&registry, saved_lr) {
            let return_offset = saved_lr - entry.code_start;
            // Binary-search the safepoints by return_offset.
            match entry
                .safepoints
                .binary_search_by_key(&return_offset, |sp| sp.return_offset)
            {
                Ok(idx) => {
                    let record = &entry.safepoints[idx];
                    for &slot_offset in &record.root_slots {
                        let slot = unsafe {
                            (saved_fp as *mut u8).offset(slot_offset as isize) as *mut u64
                        };
                        visitor(slot);
                    }
                }
                Err(_) => {
                    // No exact match: the saved_lr came from code we
                    // didn't register as a safepoint (e.g. an
                    // internal block branch). That can't happen under
                    // the soundness invariant — calls and explicit
                    // safepoints always record here — so flag it.
                    debug_assert!(
                        false,
                        "no SafepointRecord for return_offset={:#x} in JIT function \
                         [{:#x}..{:#x})",
                        return_offset, entry.code_start, entry.code_end,
                    );
                }
            }
        }

        fp = saved_fp;
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub fn walk_jit_ancestor_roots(_jit_fp: *const u8, _visitor: &mut dyn FnMut(*mut u64)) {
    // FP-chain walking is architecture-specific; no-op on unsupported targets.
}

/// Root source that walks all ancestor JIT frames via the FP chain.
/// Construct with the frame pointer of the JIT frame at the safepoint.
pub struct JitFrameRoots {
    pub jit_fp: *const u8,
}

// Safety: the pointer is only dereferenced during scan_roots, which
// happens at a safepoint when all mutator threads are stopped.
unsafe impl Send for JitFrameRoots {}
unsafe impl Sync for JitFrameRoots {}

impl dynobj::RootSource for JitFrameRoots {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        walk_jit_ancestor_roots(self.jit_fp, visitor);
    }
}

// ─── Public API ────────────────────────────────────────────────────

pub type DefaultJitConfig<L> = DefaultCodegenConfig<L>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafepointRecord {
    /// Offset (within the enclosing function) of the first instruction
    /// of the safepoint's handler-call sequence. Used as the payload
    /// passed to `active_jit_safepoint_handler` under `StackMapRoots`
    /// (via its index into the module's safepoint list).
    pub code_offset: usize,
    /// Offset (within the enclosing function) of the *return address*
    /// that would be saved in LR if execution is suspended at this
    /// safepoint — i.e. the byte just after the `BLR` of the safepoint
    /// call, or after a `BLR` to another JIT function. Used by the FP
    /// chain walker to find the right record for an ancestor frame
    /// whose `saved_lr` falls within this function's code range.
    pub return_offset: usize,
    pub root_slots: Vec<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameReifyKind {
    CaptureSlice,
    CloneSlice,
    ResumeSlice,
    AbortToPrompt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameReifyRecord {
    pub kind: FrameReifyKind,
    pub prompt: Option<PromptId>,
    pub active_prompts: Vec<PromptId>,
    pub resume: FrameResumePoint,
    pub native_resume_offset: Option<usize>,
    pub frame_value_count: usize,
    pub value_indices: Vec<usize>,
    pub control_value_indices: Vec<usize>,
    pub value_types: Vec<Type>,
    pub root_payload_indices: Vec<usize>,
    pub return_dest: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafepointHandlerPayloadKind {
    FrameSize,
    SafepointIndex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum JitOutcomeKind {
    ReturnValue = 0,
    ReturnVoid = 1,
    Exception = 2,
    Deopt = 3,
    CaptureSlice = 4,
    AbortToPrompt = 5,
    CloneSlice = 6,
    ResumeSlice = 7,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JitOutcome {
    Value(u64),
    Void,
    Exception(u64),
    Deopt {
        deopt_id: DeoptId,
        resume_point: u64,
        live_values: Vec<u64>,
    },
    CaptureSlice {
        func_idx: usize,
        record_idx: usize,
        values: Vec<u64>,
    },
    CloneSlice {
        func_idx: usize,
        record_idx: usize,
        values: Vec<u64>,
    },
    ResumeSlice {
        func_idx: usize,
        record_idx: usize,
        values: Vec<u64>,
    },
    AbortToPrompt {
        func_idx: usize,
        record_idx: usize,
        values: Vec<u64>,
    },
}

/// A JIT frame that was on the native stack when a deeper frame decided
/// to reify (capture / clone / resume / abort). The `frame` carries the
/// same data a `BuilderFrame` does — except its `caller_resume` is a
/// placeholder. `build_capture_builder` fills in each frame's real
/// `caller_resume` from the `callee_caller_resume` of the frame above it
/// (or `TopLevel` at the outermost).
#[derive(Debug, Clone)]
pub struct SuspendedJitFrame {
    pub frame: BuilderFrame,
    pub callee_caller_resume: FrameResume,
}

#[derive(Debug)]
struct CallSuspendRecord {
    resume: FrameResumePoint,
    native_resume_offset: Option<usize>,
    native_exception_resume_offset: Option<usize>,
    value_accesses: Vec<Option<FrameSlotAccess>>,
    root_value_indices: Vec<usize>,
    active_prompts: Vec<u32>,
    callee_caller_resume: SuspendCallerResume,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SuspendCallerResume {
    FromCall {
        return_dest: Option<usize>,
    },
    FromInvoke {
        normal_block: usize,
        normal_arg_indices: Vec<usize>,
        exception_block: usize,
        exception_arg_indices: Vec<usize>,
        has_ret_param: bool,
    },
}

impl SuspendCallerResume {
    fn materialize(&self, values: &[u64]) -> FrameResume {
        match self {
            SuspendCallerResume::FromCall { return_dest } => FrameResume::FromCall {
                return_dest: *return_dest,
            },
            SuspendCallerResume::FromInvoke {
                normal_block,
                normal_arg_indices,
                exception_block,
                exception_arg_indices,
                has_ret_param,
            } => FrameResume::FromInvoke {
                normal_block: *normal_block,
                normal_args_vals: normal_arg_indices
                    .iter()
                    .map(|&idx| values.get(idx).copied().unwrap_or(0))
                    .collect(),
                exception_block: *exception_block,
                exception_args_vals: exception_arg_indices
                    .iter()
                    .map(|&idx| values.get(idx).copied().unwrap_or(0))
                    .collect(),
                has_ret_param: *has_ret_param,
            },
        }
    }
}

thread_local! {
    static ACTIVE_SUSPENDED_JIT_FRAMES: RefCell<Vec<SuspendedJitFrame>> = const { RefCell::new(Vec::new()) };
}

pub fn take_suspended_frames() -> Vec<SuspendedJitFrame> {
    ACTIVE_SUSPENDED_JIT_FRAMES.with(|cell| std::mem::take(&mut *cell.borrow_mut()))
}

extern "C" fn jit_push_suspended_frame(
    record: *const CallSuspendRecord,
    frame_ptr: *mut u8,
    stack_ptr: *mut u8,
) {
    let record = unsafe { &*record };
    let mut values = vec![0u64; record.value_accesses.len()];
    for (idx, access) in record.value_accesses.iter().enumerate() {
        let Some(access) = access else {
            continue;
        };
        let base_ptr = match access.base {
            FrameSlotBase::FramePointer => frame_ptr,
            FrameSlotBase::StackPointer => stack_ptr,
        };
        let slot = unsafe { base_ptr.byte_offset(access.offset as isize).cast::<u64>() };
        values[idx] = unsafe { slot.read_unaligned() };
    }
    ACTIVE_SUSPENDED_JIT_FRAMES.with(|cell| {
        let callee_caller_resume = record.callee_caller_resume.materialize(&values);
        let root_indices: Vec<u16> = record
            .root_value_indices
            .iter()
            .map(|&i| i as u16)
            .collect();
        cell.borrow_mut().push(SuspendedJitFrame {
            frame: BuilderFrame {
                func_idx: record.resume.func_idx as u32,
                block_idx: record.resume.block_idx as u32,
                inst_idx: record.resume.inst_idx as u32,
                values,
                active_prompts: record.active_prompts.clone(),
                root_indices,
                resume_arg_slot: None,
                // Placeholder — `build_capture_builder` fills this in from
                // the `callee_caller_resume` of the next outer frame.
                caller_resume: FrameResume::TopLevel,
            },
            callee_caller_resume,
        });
    });
}

extern "C" fn jit_pop_suspended_frame() {
    ACTIVE_SUSPENDED_JIT_FRAMES.with(|cell| {
        let _ = cell.borrow_mut().pop();
    });
}

#[repr(C)]
struct JitControlContext {
    live_values_ptr: *mut u64,
    live_values_len: usize,
}

pub struct JitFunction {
    memory: PagedCodeMemory,
    safepoints: Vec<SafepointRecord>,
    frame_reify_records: Vec<FrameReifyRecord>,
    suspend_records: Vec<Box<CallSuspendRecord>>,
    handler_payload_kind: SafepointHandlerPayloadKind,
    max_deopt_live_values: usize,
    #[allow(dead_code)]
    root_scan_size: usize,
}

impl Drop for JitFunction {
    fn drop(&mut self) {
        unregister_jit_code(self.memory.base_ptr() as usize);
    }
}

impl JitFunction {
    pub fn compile<L>(func: &Function, externs: &[*const u8]) -> Self
    where
        L: LayoutConfigDefaults,
        L::DefaultRoots: SoundRoots<L>,
        L::DefaultRootTransport: SoundTransport<L, L::DefaultRoots>,
    {
        Self::compile_with_config::<DefaultJitConfig<L>>(func, externs)
    }

    pub fn compile_with_gc<L>(
        func: &Function,
        externs: &[*const u8],
        handler: extern "C" fn(*mut u8, usize),
    ) -> Self
    where
        L: LayoutConfigDefaults,
        L::DefaultRoots: SoundRoots<L>,
        L::DefaultRootTransport: SoundTransport<L, L::DefaultRoots>,
    {
        Self::compile_with_config_and_gc::<DefaultJitConfig<L>>(func, externs, Some(handler as u64))
    }

    /// Compile with GC support using the linear scan register allocator.
    pub fn compile_with_gc_linear_scan<L>(
        func: &Function,
        externs: &[*const u8],
        handler: extern "C" fn(*mut u8, usize),
    ) -> Self
    where
        L: LayoutConfigDefaults,
        L::DefaultRoots: SoundRoots<L>,
        L::DefaultRootTransport: SoundTransport<L, L::DefaultRoots>,
    {
        #[cfg(target_arch = "aarch64")]
        {
            Self::compile_with_regalloc::<
                DefaultJitConfig<L>,
                Arm64Backend,
                regalloc::LinearScanAllocator,
            >(func, externs, Some(handler as u64))
        }
        #[cfg(target_arch = "x86_64")]
        {
            Self::compile_with_regalloc::<
                DefaultJitConfig<L>,
                X64Backend,
                regalloc::LinearScanAllocator,
            >(func, externs, Some(handler as u64))
        }
    }

    pub fn compile_with_config<Cfg: CodegenConfig>(func: &Function, externs: &[*const u8]) -> Self
    where
        Cfg::Frames: FrameStrategy<Cfg::Layout, Cfg::Roots, Cfg::CallingConvention>,
    {
        #[cfg(target_arch = "aarch64")]
        {
            Self::compile_with_backend_and_config::<Cfg, Arm64Backend>(func, externs, None)
        }
        #[cfg(target_arch = "x86_64")]
        {
            Self::compile_with_backend_and_config::<Cfg, X64Backend>(func, externs, None)
        }
    }

    pub fn compile_with_config_and_gc<Cfg: CodegenConfig>(
        func: &Function,
        externs: &[*const u8],
        safepoint_handler: Option<u64>,
    ) -> Self
    where
        Cfg::Frames: FrameStrategy<Cfg::Layout, Cfg::Roots, Cfg::CallingConvention>,
    {
        #[cfg(target_arch = "aarch64")]
        {
            Self::compile_with_backend_and_config::<Cfg, Arm64Backend>(
                func,
                externs,
                safepoint_handler,
            )
        }
        #[cfg(target_arch = "x86_64")]
        {
            Self::compile_with_backend_and_config::<Cfg, X64Backend>(
                func,
                externs,
                safepoint_handler,
            )
        }
    }

    pub fn compile_with_config_and_gc_linear_scan<Cfg: CodegenConfig>(
        func: &Function,
        externs: &[*const u8],
        safepoint_handler: Option<u64>,
    ) -> Self
    where
        Cfg::Frames: FrameStrategy<Cfg::Layout, Cfg::Roots, Cfg::CallingConvention>,
    {
        #[cfg(target_arch = "aarch64")]
        {
            Self::compile_with_regalloc::<Cfg, Arm64Backend, regalloc::LinearScanAllocator>(
                func,
                externs,
                safepoint_handler,
            )
        }
        #[cfg(target_arch = "x86_64")]
        {
            Self::compile_with_regalloc::<Cfg, X64Backend, regalloc::LinearScanAllocator>(
                func,
                externs,
                safepoint_handler,
            )
        }
    }

    pub fn compile_with_backend_and_config<Cfg: CodegenConfig, B: LoweringBackend>(
        func: &Function,
        externs: &[*const u8],
        safepoint_handler: Option<u64>,
    ) -> Self
    where
        Cfg::Frames: FrameStrategy<Cfg::Layout, Cfg::Roots, Cfg::CallingConvention>,
    {
        Self::compile_with_regalloc::<Cfg, B, GreedyRegState>(func, externs, safepoint_handler)
    }

    pub fn compile_with_regalloc<Cfg: CodegenConfig, B: LoweringBackend, R: RegisterAllocator>(
        func: &Function,
        externs: &[*const u8],
        safepoint_handler: Option<u64>,
    ) -> Self
    where
        Cfg::Frames: FrameStrategy<Cfg::Layout, Cfg::Roots, Cfg::CallingConvention>,
    {
        let mut lowerer =
            Lowerer::<Cfg, B, R>::new_inner(0, func, externs, None, None, None, safepoint_handler);
        lowerer.run();
        let safepoints = std::mem::take(&mut lowerer.safepoints);
        let frame_reify_records = std::mem::take(&mut lowerer.frame_reify_records);
        let suspend_records = std::mem::take(&mut lowerer.suspend_records);
        let max_deopt_live_values = lowerer.max_deopt_live_values;
        let root_scan_size = lowerer.frame.root_scan_size() as usize;
        let code = lowerer.buf.into_code();
        let handler_payload_kind = match Cfg::RootTransport::kind() {
            RootTransportKind::FrameScan => SafepointHandlerPayloadKind::FrameSize,
            RootTransportKind::ShadowStack | RootTransportKind::StackMap => {
                SafepointHandlerPayloadKind::SafepointIndex
            }
        };

        let mut memory = PagedCodeMemory::new();
        memory.push(&code);
        memory.finalize();

        let code_start = memory.base_ptr() as usize;
        let code_end = code_start + memory.len();
        let safepoints_arc: std::sync::Arc<[SafepointRecord]> = safepoints.clone().into();
        register_jit_code(code_start, code_end, safepoints_arc);

        write_perf_map_entries(&[(code_start, code_end - code_start, &func.name)]);

        JitFunction {
            memory,
            safepoints,
            frame_reify_records,
            suspend_records,
            handler_payload_kind,
            max_deopt_live_values,
            root_scan_size,
        }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.memory.base_ptr()
    }

    pub fn safepoints(&self) -> &[SafepointRecord] {
        &self.safepoints
    }

    pub fn frame_reify_records(&self) -> &[FrameReifyRecord] {
        &self.frame_reify_records
    }

    pub fn handler_payload_kind(&self) -> SafepointHandlerPayloadKind {
        self.handler_payload_kind
    }

    pub fn call_outcome(&self, args: &[u64]) -> JitOutcome {
        unsafe { call_jit_outcome(self.as_ptr(), args, self.max_deopt_live_values) }
    }

    /// Dump the raw bytes as hex for external disassembly, and print
    /// a shell command to disassemble with llvm-objdump.
    pub fn dump_code(&self) {
        let base = self.memory.base_ptr();
        let len = self.memory.len();
        eprintln!("JIT code: {:?} ({} bytes)", base, len);
        let bytes = unsafe { std::slice::from_raw_parts(base, len) };
        // Write to a temp file and print the objdump command
        let path = "/tmp/jit_dump.bin";
        std::fs::write(path, bytes).unwrap();
        eprintln!("Disassemble with:");
        eprintln!(
            "  llvm-objdump -d -m aarch64 -b binary {} | head -200",
            path
        );
    }

    pub fn native_resume_ptr(&self, record: &FrameReifyRecord) -> Option<*const u8> {
        record
            .native_resume_offset
            .map(|offset| unsafe { self.as_ptr().add(offset) })
    }

    pub fn call_resume_outcome(
        &self,
        record: &FrameReifyRecord,
        frame_values_ptr: *const u64,
        resume_args: &[u64],
    ) -> JitOutcome {
        let ptr = self
            .native_resume_ptr(record)
            .expect("record does not have a native resume entry");
        let args_ptr = if resume_args.is_empty() {
            std::ptr::null()
        } else {
            resume_args.as_ptr()
        };
        let args = [
            frame_values_ptr as u64,
            args_ptr as u64,
            resume_args.len() as u64,
        ];
        unsafe { call_jit_outcome(ptr, &args, self.max_deopt_live_values) }
    }

    /// View-based resume entry point. Takes the resume point and frame
    /// values directly.
    pub fn call_view_resume_outcome(
        &self,
        resume: &FrameResumePoint,
        values: &[u64],
        resume_args: &[u64],
    ) -> Option<JitOutcome> {
        if let Some(record) = self.frame_reify_records.iter().find(|record| {
            record.kind == FrameReifyKind::CaptureSlice
                && record.native_resume_offset.is_some()
                && record.resume == *resume
        }) {
            return Some(self.call_resume_outcome(record, values.as_ptr(), resume_args));
        }

        let suspend = self
            .suspend_records
            .iter()
            .find(|record| record.native_resume_offset.is_some() && record.resume == *resume)?;
        let ptr = unsafe {
            self.as_ptr()
                .add(suspend.native_resume_offset.expect("checked above"))
        };
        let args_ptr = if resume_args.is_empty() {
            std::ptr::null()
        } else {
            resume_args.as_ptr()
        };
        let args = [
            values.as_ptr() as u64,
            args_ptr as u64,
            resume_args.len() as u64,
        ];
        Some(unsafe { call_jit_outcome(ptr, &args, self.max_deopt_live_values) })
    }
}

// ─── CallTable ─────────────────────────────────────────────────────

/// Pointer-stable call table.
///
/// JIT'd code reads function pointers via `ldr [base + idx*8]` where `base`
/// is baked into the instruction stream as an immediate. The base must stay
/// valid for the entire life of the JIT image — pushing past a `Vec`'s
/// capacity reallocates and silently invalidates every emitted call site.
///
/// `CallTable` solves that by allocating a fixed-size `Box<[Cell<*const u8>]>`
/// up front. Pushes never reallocate (panicking on overflow is preferable to
/// undefined behavior). `Cell<*const u8>` is `#[repr(transparent)]` so the
/// emitted load reads the inner pointer directly.
pub struct CallTable {
    slots: Box<[Cell<*const u8>]>,
    len: AtomicUsize,
}

// Single-threaded compilation; raw reads from JIT code are not synchronized.
unsafe impl Sync for CallTable {}
unsafe impl Send for CallTable {}

impl CallTable {
    pub fn new(capacity: usize) -> Self {
        let slots: Box<[Cell<*const u8>]> = (0..capacity)
            .map(|_| Cell::new(std::ptr::null()))
            .collect();
        Self {
            slots,
            len: AtomicUsize::new(0),
        }
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    pub fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Stable base address baked into emitted call sites as an immediate.
    pub fn base(&self) -> *const Cell<*const u8> {
        self.slots.as_ptr()
    }

    pub fn base_addr(&self) -> u64 {
        self.base() as u64
    }

    /// Append a new slot, returning its index. Panics if capacity exhausted.
    pub fn push(&self, ptr: *const u8) -> usize {
        let idx = self.len.fetch_add(1, Ordering::AcqRel);
        assert!(
            idx < self.slots.len(),
            "CallTable: capacity {} exhausted",
            self.slots.len()
        );
        self.slots[idx].set(ptr);
        idx
    }

    /// Update an existing slot — used to late-patch internal function entries
    /// after their machine code is finalized.
    pub fn set(&self, idx: usize, ptr: *const u8) {
        assert!(
            idx < self.len(),
            "CallTable: set({}) past len {}",
            idx,
            self.len()
        );
        self.slots[idx].set(ptr);
    }

    pub fn get(&self, idx: usize) -> *const u8 {
        assert!(
            idx < self.len(),
            "CallTable: get({}) past len {}",
            idx,
            self.len()
        );
        self.slots[idx].get()
    }

    /// Snapshot the live slots into a Vec (for callers that need to copy out).
    pub fn snapshot(&self) -> Vec<*const u8> {
        let len = self.len();
        self.slots[..len].iter().map(|c| c.get()).collect()
    }
}

// ─── LiteralPool ──────────────────────────────────────────────────

/// Pointer-stable, GC-traced array of NanBox-encoded literal slots.
///
/// `Inst::GcLiteral(idx)` lowers to a load of `slots[idx]`. Slots can hold
/// any 64-bit NanBox value, but the point of the indirection is to support
/// GC-managed payloads: when a moving collector relocates a heap object,
/// it walks the pool (registered as a root set) and rewrites the affected
/// slot — the next execution of the emitted `ldr` reads the new pointer.
///
/// Same invariants as [`CallTable`]: fixed-capacity backing storage, base
/// address stable across pushes, baked into emitted code as an immediate.
pub struct LiteralPool {
    slots: Box<[Cell<u64>]>,
    len: AtomicUsize,
}

unsafe impl Sync for LiteralPool {}
unsafe impl Send for LiteralPool {}

impl LiteralPool {
    pub fn new(capacity: usize) -> Self {
        let slots: Box<[Cell<u64>]> = (0..capacity).map(|_| Cell::new(0)).collect();
        Self { slots, len: AtomicUsize::new(0) }
    }

    pub fn capacity(&self) -> usize { self.slots.len() }

    pub fn len(&self) -> usize { self.len.load(Ordering::Acquire) }

    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Stable base address baked into emitted `GcLiteral` loads.
    pub fn base(&self) -> *const Cell<u64> { self.slots.as_ptr() }

    pub fn base_addr(&self) -> u64 { self.base() as u64 }

    /// Append a slot, return its index. Panics if capacity exhausted.
    pub fn push(&self, value: u64) -> usize {
        let idx = self.len.fetch_add(1, Ordering::AcqRel);
        assert!(
            idx < self.slots.len(),
            "LiteralPool: capacity {} exhausted",
            self.slots.len()
        );
        self.slots[idx].set(value);
        idx
    }

    pub fn get(&self, idx: usize) -> u64 {
        assert!(idx < self.len(), "LiteralPool: get({}) past len {}", idx, self.len());
        self.slots[idx].get()
    }

    /// Update an existing slot. Used by the GC after relocating an object.
    pub fn set(&self, idx: usize, value: u64) {
        assert!(idx < self.len(), "LiteralPool: set({}) past len {}", idx, self.len());
        self.slots[idx].set(value);
    }

    /// Iterate `(slot_addr, current_value)` pairs over live slots — used by
    /// GC root-tracing impls so they can decode each NanBox, follow it, and
    /// rewrite the slot in place.
    pub fn iter_addrs(&self) -> impl Iterator<Item = (*mut u64, u64)> + '_ {
        let len = self.len();
        self.slots[..len].iter().map(|c| {
            // Cell<u64> is repr(transparent) over UnsafeCell<u64>; raw pointer
            // to the inner u64 is sound for read+write under the GC's STW
            // contract.
            (c.as_ptr(), c.get())
        })
    }

    pub fn snapshot(&self) -> Vec<u64> {
        let len = self.len();
        self.slots[..len].iter().map(|c| c.get()).collect()
    }
}

/// `LiteralPool` is a GC root set: every live slot may hold a NanBox-encoded
/// heap pointer that the collector must trace and may rewrite during a
/// moving collection. The visitor receives `*mut u64` pointing directly at
/// the underlying `Cell<u64>`'s storage so a moving GC can replace the
/// pointer in place — emitted code reads the new value on next access
/// without any code patching.
impl dynobj::RootSource for LiteralPool {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        let len = self.len();
        for cell in &self.slots[..len] {
            visitor(cell.as_ptr());
        }
    }
}

// ─── JitModule ─────────────────────────────────────────────────────

/// Default capacity (slot count) for `JitModule::literal_pool`. 4096 slots
/// = 32 KiB. The legacy batch entry points use this; `new_empty` takes the
/// capacity explicitly.
pub const DEFAULT_LITERAL_POOL_CAPACITY: usize = 4096;

/// How internal calls are emitted in this `JitModule`, and whether the GC
/// can fire from inside JIT-executed code.
///
/// Pinned at construction. The choice has to be uniform across the whole
/// module because emitted call sites bake the sequence in — you can't mix
/// fast and control-aware calls in the same image.
///
/// Coupling the safepoint handler with the call mode is deliberate. The
/// previous API exposed an `Option<u64>` that let frontends pass `None`
/// alongside a moving GC — a silent footgun where allocations could trigger
/// without any stack-map metadata, corrupting whichever live values weren't
/// re-rooted by hand. This enum makes that combination unrepresentable: if
/// you want JIT-time GC, you provide a handler; if you don't want one, you
/// get `FastCall` and any `Inst::Safepoint` in the IR fails at extend time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallMode {
    /// Control-aware calls plus a GC-aware safepoint handler.
    ///
    /// `safepoint_handler` is the address of the C-ABI function the JIT
    /// invokes at every `Inst::Safepoint` — typically
    /// `dynruntime::active_jit_safepoint_handler` (or your own root
    /// scanner). Required for any frontend that lets allocations happen
    /// during JIT execution, since the handler is what reads the live
    /// values out of the current stack frame and hands them to the GC.
    ///
    /// Required for delimited continuations, deopts, `Invoke` terminators,
    /// and any moving GC. Roughly 2x call-site code size compared to
    /// `FastCall`; small constant runtime overhead per call.
    ControlAware { safepoint_handler: u64 },
    /// Plain call/return, no safepoints.
    ///
    /// `Inst::Safepoint`, `Terminator::Invoke`, deopts, and prompt usage
    /// all fail at extend time. Suitable for stateless numeric code or
    /// frontends with no GC interaction.
    FastCall,
}

impl CallMode {
    pub fn is_control_aware(self) -> bool {
        matches!(self, CallMode::ControlAware { .. })
    }

    pub fn safepoint_handler(self) -> Option<u64> {
        match self {
            CallMode::ControlAware { safepoint_handler } => Some(safepoint_handler),
            CallMode::FastCall => None,
        }
    }
}

/// JIT-compiled module: multiple functions that can call each other.
///
/// Internal calls go through an indirect call table so all function
/// pointers are resolved after compilation.
pub struct JitModule {
    memory: PagedCodeMemory,
    /// One entry per `Module::func_table` slot. Extern entries hold the
    /// provided extern pointers; internal entries are filled in after
    /// compilation with pointers into `memory`.
    call_table: CallTable,
    /// GC-traced NanBox literal slots. Emitted `GcLiteral` instructions load
    /// from `literal_pool.base() + idx*8`.
    literal_pool: LiteralPool,
    function_entry_offsets: Vec<usize>,
    function_suspend_records: Vec<Vec<Box<CallSuspendRecord>>>,
    function_safepoints: Vec<Vec<SafepointRecord>>,
    function_frame_reify_records: Vec<Vec<FrameReifyRecord>>,
    handler_payload_kind: SafepointHandlerPayloadKind,
    max_deopt_live_values: usize,
    /// Pinned at construction; uniform across all calls in the module.
    call_mode: CallMode,
    /// Pinned at construction; passed to every Lowerer.
    safepoint_handler: Option<u64>,
    /// How many extern declarations from `module.func_table` we've already
    /// pulled pointers for. Used by `extend` to index into the externs slice.
    extern_count_seen: usize,
}

impl Drop for JitModule {
    fn drop(&mut self) {
        let base = self.memory.base_ptr() as usize;
        for &offset in &self.function_entry_offsets {
            unregister_jit_code(base + offset);
        }
    }
}

impl JitModule {
    /// Compile a module with no GC safepoint handler.
    pub fn compile<L>(module: &Module, externs: &[*const u8]) -> Self
    where
        L: LayoutConfigDefaults,
        L::DefaultRoots: SoundRoots<L>,
        L::DefaultRootTransport: SoundTransport<L, L::DefaultRoots>,
    {
        Self::compile_with_config::<DefaultJitConfig<L>>(module, externs)
    }

    /// Compile a module using the linear scan register allocator.
    pub fn compile_linear_scan<L>(module: &Module, externs: &[*const u8]) -> Self
    where
        L: LayoutConfigDefaults,
        L::DefaultRoots: SoundRoots<L>,
        L::DefaultRootTransport: SoundTransport<L, L::DefaultRoots>,
    {
        #[cfg(target_arch = "aarch64")]
        {
            Self::compile_with_regalloc::<
                DefaultJitConfig<L>,
                Arm64Backend,
                regalloc::LinearScanAllocator,
            >(module, externs, None)
        }
        #[cfg(target_arch = "x86_64")]
        {
            Self::compile_with_regalloc::<
                DefaultJitConfig<L>,
                X64Backend,
                regalloc::LinearScanAllocator,
            >(module, externs, None)
        }
    }

    /// Compile a module with fast calls: internal calls are treated as plain
    /// calls without suspend/resume bookkeeping. Disables delimited continuation
    /// support but significantly reduces call overhead.
    pub fn compile_fast<L>(module: &Module, externs: &[*const u8]) -> Self
    where
        L: LayoutConfigDefaults,
        L::DefaultRoots: SoundRoots<L>,
        L::DefaultRootTransport: SoundTransport<L, L::DefaultRoots>,
    {
        #[cfg(target_arch = "aarch64")]
        {
            Self::compile_fast_with_regalloc::<
                DefaultJitConfig<L>,
                Arm64Backend,
                regalloc::LinearScanAllocator,
            >(module, externs)
        }
        #[cfg(target_arch = "x86_64")]
        {
            Self::compile_fast_with_regalloc::<
                DefaultJitConfig<L>,
                X64Backend,
                regalloc::LinearScanAllocator,
            >(module, externs)
        }
    }

    fn compile_fast_with_regalloc<Cfg: CodegenConfig, B: LoweringBackend, R: RegisterAllocator>(
        module: &Module,
        externs: &[*const u8],
    ) -> Self
    where
        Cfg::Frames: FrameStrategy<Cfg::Layout, Cfg::Roots, Cfg::CallingConvention>,
    {
        let call_table = CallTable::new(module.func_table.len());
        let mut extern_idx = 0usize;
        for def in &module.func_table {
            match def {
                FuncDef::Extern(_) => {
                    call_table.push(externs[extern_idx]);
                    extern_idx += 1;
                }
                FuncDef::Internal(_) => {
                    call_table.push(std::ptr::null());
                }
            }
        }
        let call_table_base = call_table.base_addr();

        let mut memory = PagedCodeMemory::new();
        let mut entry_offsets: Vec<usize> = Vec::new();
        let mut function_suspend_records: Vec<Vec<Box<CallSuspendRecord>>> = Vec::new();
        let mut function_safepoints: Vec<Vec<SafepointRecord>> = Vec::new();
        let mut function_frame_reify_records: Vec<Vec<FrameReifyRecord>> = Vec::new();
        let mut max_deopt_live_values = 0usize;
        let handler_payload_kind = match Cfg::RootTransport::kind() {
            RootTransportKind::FrameScan => SafepointHandlerPayloadKind::FrameSize,
            RootTransportKind::ShadowStack | RootTransportKind::StackMap => {
                SafepointHandlerPayloadKind::SafepointIndex
            }
        };

        // All false → no call is control-aware → no suspend/resume overhead
        let direct_call_is_internal: Vec<bool> = vec![false; module.func_table.len()];

        let mut function_root_scan_sizes: Vec<usize> = Vec::new();
        for (func_idx, func) in module.functions.iter().enumerate() {
            let mut lowerer = Lowerer::<Cfg, B, R>::new_module(
                func_idx,
                func,
                &direct_call_is_internal,
                call_table_base,
                None, // legacy batch path: no GC literal pool
                None,
            );
            lowerer.run();
            function_suspend_records.push(std::mem::take(&mut lowerer.suspend_records));
            function_safepoints.push(std::mem::take(&mut lowerer.safepoints));
            function_frame_reify_records.push(std::mem::take(&mut lowerer.frame_reify_records));
            max_deopt_live_values = max_deopt_live_values.max(lowerer.max_deopt_live_values);
            function_root_scan_sizes.push(lowerer.frame.root_scan_size() as usize);

            let code = lowerer.buf.into_code();
            let offset = memory.push(&code);
            entry_offsets.push(offset);
        }

        memory.finalize();

        let base = memory.base_ptr() as usize;
        let total_len = memory.len();
        let mut perf_entries: Vec<(usize, usize, &str)> = Vec::new();
        for (i, &offset) in entry_offsets.iter().enumerate() {
            let code_start = base + offset;
            let code_end = if i + 1 < entry_offsets.len() {
                base + entry_offsets[i + 1]
            } else {
                base + total_len
            };
            let safepoints_arc: std::sync::Arc<[SafepointRecord]> =
                function_safepoints[i].clone().into();
            register_jit_code(code_start, code_end, safepoints_arc);
            perf_entries.push((code_start, code_end - code_start, &module.functions[i].name));
        }
        let _ = function_root_scan_sizes; // no longer used at runtime
        write_perf_map_entries(&perf_entries);

        let base = memory.base_ptr();
        for (ft_idx, def) in module.func_table.iter().enumerate() {
            if let FuncDef::Internal(func_idx) = def {
                let ptr = unsafe { base.add(entry_offsets[*func_idx]) };
                call_table.set(ft_idx, ptr);
            }
        }

        JitModule {
            memory,
            call_table,
            literal_pool: LiteralPool::new(DEFAULT_LITERAL_POOL_CAPACITY),
            function_entry_offsets: entry_offsets,
            function_suspend_records,
            function_safepoints,
            function_frame_reify_records,
            handler_payload_kind,
            max_deopt_live_values,
            call_mode: CallMode::FastCall,
            safepoint_handler: None,
            extern_count_seen: module
                .func_table
                .iter()
                .filter(|d| matches!(d, FuncDef::Extern(_)))
                .count(),
        }
    }

    /// Compile a module with a GC safepoint handler (greedy register allocator).
    ///
    /// At each `Inst::Safepoint`, the JIT will spill all live values and
    /// call `handler(frame_ptr, frame_size)`. The handler can scan the
    /// frame for GC pointers using `PtrPolicy::try_decode_ptr`.
    pub fn compile_with_gc<L>(
        module: &Module,
        externs: &[*const u8],
        handler: extern "C" fn(*mut u8, usize),
    ) -> Self
    where
        L: LayoutConfigDefaults,
        L::DefaultRoots: SoundRoots<L>,
        L::DefaultRootTransport: SoundTransport<L, L::DefaultRoots>,
    {
        Self::compile_with_config_and_gc::<DefaultJitConfig<L>>(
            module,
            externs,
            Some(handler as u64),
        )
    }

    /// Compile a module with GC safepoint handler + linear scan register allocator.
    pub fn compile_with_gc_linear_scan<L>(
        module: &Module,
        externs: &[*const u8],
        handler: extern "C" fn(*mut u8, usize),
    ) -> Self
    where
        L: LayoutConfigDefaults,
        L::DefaultRoots: SoundRoots<L>,
        L::DefaultRootTransport: SoundTransport<L, L::DefaultRoots>,
    {
        #[cfg(target_arch = "aarch64")]
        {
            Self::compile_with_regalloc::<
                DefaultJitConfig<L>,
                Arm64Backend,
                regalloc::LinearScanAllocator,
            >(module, externs, Some(handler as u64))
        }
        #[cfg(target_arch = "x86_64")]
        {
            Self::compile_with_regalloc::<
                DefaultJitConfig<L>,
                X64Backend,
                regalloc::LinearScanAllocator,
            >(module, externs, Some(handler as u64))
        }
    }

    /// Compile a module using the **batch** register allocator from the
    /// `regalloc` crate. This uses a proper SSA-aware linear scan with
    /// callee-saved registers, producing better code for large functions.
    ///
    /// Compile a module with the batch register allocator (linear scan).
    /// Supports GC safepoints via conservative frame scanning when a
    /// safepoint_handler is provided.
    pub fn compile_batch<L: dynvalue::TagScheme>(
        module: &Module,
        externs: &[*const u8],
        safepoint_handler: Option<u64>,
    ) -> Self {
        use batch_lower::{TagConfig, compile_function_batch};

        let tags = TagConfig {
            has_unboxed_float: L::HAS_UNBOXED_FLOAT,
            payload_bits: L::PAYLOAD_BITS as u8,
            tag_count: L::TAG_COUNT,
            encode_tagged: L::encode_tagged,
        };

        // 1. Build call table
        let call_table = CallTable::new(module.func_table.len());
        let mut extern_ptrs: Vec<*const u8> = Vec::new();
        for def in &module.func_table {
            match def {
                FuncDef::Extern(_) => {
                    let ptr = externs[extern_ptrs.len()];
                    extern_ptrs.push(ptr);
                    call_table.push(ptr);
                }
                FuncDef::Internal(_) => {
                    call_table.push(std::ptr::null());
                }
            }
        }
        let call_table_base = call_table.base_addr();

        // 2. Compile each function with the batch allocator
        let mut memory = PagedCodeMemory::new();
        let mut entry_offsets: Vec<usize> = Vec::new();
        let mut frame_sizes: Vec<u32> = Vec::new();
        let mut function_safepoints: Vec<Vec<SafepointRecord>> = Vec::new();

        for (_func_idx, func) in module.functions.iter().enumerate() {
            let (code, frame_size, safepoints) = compile_function_batch(
                func,
                &extern_ptrs,
                call_table_base,
                TagConfig {
                    has_unboxed_float: L::HAS_UNBOXED_FLOAT,
                    payload_bits: L::PAYLOAD_BITS as u8,
                    tag_count: L::TAG_COUNT,
                    encode_tagged: L::encode_tagged,
                },
                safepoint_handler,
            )
            .unwrap_or_else(|e| panic!("batch compile failed: {e}"));
            let offset = memory.push(&code);
            entry_offsets.push(offset);
            frame_sizes.push(frame_size);
            function_safepoints.push(safepoints);
        }

        memory.finalize();

        // 3. Register code ranges
        let base = memory.base_ptr() as usize;
        let total_len = memory.len();
        let mut perf_entries: Vec<(usize, usize, &str)> = Vec::new();
        for (i, &offset) in entry_offsets.iter().enumerate() {
            let code_start = base + offset;
            let code_end = if i + 1 < entry_offsets.len() {
                base + entry_offsets[i + 1]
            } else {
                base + total_len
            };
            let safepoints_arc: std::sync::Arc<[SafepointRecord]> =
                function_safepoints[i].clone().into();
            register_jit_code(code_start, code_end, safepoints_arc);
            perf_entries.push((code_start, code_end - code_start, &module.functions[i].name));
        }
        let _ = frame_sizes; // historical (was passed to registry for conservative scan)
        write_perf_map_entries(&perf_entries);

        // 4. Patch internal function pointers
        let base = memory.base_ptr();
        for (ft_idx, def) in module.func_table.iter().enumerate() {
            if let FuncDef::Internal(func_idx) = def {
                let ptr = unsafe { base.add(entry_offsets[*func_idx]) };
                call_table.set(ft_idx, ptr);
            }
        }

        let call_mode = match safepoint_handler {
            Some(h) => CallMode::ControlAware { safepoint_handler: h },
            None => CallMode::FastCall,
        };
        JitModule {
            memory,
            call_table,
            literal_pool: LiteralPool::new(DEFAULT_LITERAL_POOL_CAPACITY),
            function_entry_offsets: entry_offsets,
            function_suspend_records: module.functions.iter().map(|_| vec![]).collect(),
            function_safepoints,
            function_frame_reify_records: module.functions.iter().map(|_| vec![]).collect(),
            handler_payload_kind: SafepointHandlerPayloadKind::SafepointIndex,
            max_deopt_live_values: 0,
            call_mode,
            safepoint_handler,
            extern_count_seen: module
                .func_table
                .iter()
                .filter(|d| matches!(d, FuncDef::Extern(_)))
                .count(),
        }
    }

    pub fn compile_with_config<Cfg: CodegenConfig>(module: &Module, externs: &[*const u8]) -> Self
    where
        Cfg::Frames: FrameStrategy<Cfg::Layout, Cfg::Roots, Cfg::CallingConvention>,
    {
        #[cfg(target_arch = "aarch64")]
        {
            Self::compile_with_backend_and_config::<Cfg, Arm64Backend>(module, externs, None)
        }
        #[cfg(target_arch = "x86_64")]
        {
            Self::compile_with_backend_and_config::<Cfg, X64Backend>(module, externs, None)
        }
    }

    pub fn compile_with_config_and_gc<Cfg: CodegenConfig>(
        module: &Module,
        externs: &[*const u8],
        safepoint_handler: Option<u64>,
    ) -> Self
    where
        Cfg::Frames: FrameStrategy<Cfg::Layout, Cfg::Roots, Cfg::CallingConvention>,
    {
        #[cfg(target_arch = "aarch64")]
        {
            Self::compile_with_backend_and_config::<Cfg, Arm64Backend>(
                module,
                externs,
                safepoint_handler,
            )
        }
        #[cfg(target_arch = "x86_64")]
        {
            Self::compile_with_backend_and_config::<Cfg, X64Backend>(
                module,
                externs,
                safepoint_handler,
            )
        }
    }

    pub fn compile_with_backend_and_config<Cfg: CodegenConfig, B: LoweringBackend>(
        module: &Module,
        externs: &[*const u8],
        safepoint_handler: Option<u64>,
    ) -> Self
    where
        Cfg::Frames: FrameStrategy<Cfg::Layout, Cfg::Roots, Cfg::CallingConvention>,
    {
        Self::compile_with_regalloc::<Cfg, B, GreedyRegState>(module, externs, safepoint_handler)
    }

    pub fn compile_with_regalloc<Cfg: CodegenConfig, B: LoweringBackend, R: RegisterAllocator>(
        module: &Module,
        externs: &[*const u8],
        safepoint_handler: Option<u64>,
    ) -> Self
    where
        Cfg::Frames: FrameStrategy<Cfg::Layout, Cfg::Roots, Cfg::CallingConvention>,
    {
        // 1. Build call table: fill externs, leave internals as null
        let call_table = CallTable::new(module.func_table.len());
        let mut extern_idx = 0usize;
        for def in &module.func_table {
            match def {
                FuncDef::Extern(_) => {
                    call_table.push(externs[extern_idx]);
                    extern_idx += 1;
                }
                FuncDef::Internal(_) => {
                    call_table.push(std::ptr::null());
                }
            }
        }

        let call_table_base = call_table.base_addr();

        // 2. Compile each internal function
        let mut memory = PagedCodeMemory::new();
        let mut entry_offsets: Vec<usize> = Vec::new();
        let mut function_suspend_records: Vec<Vec<Box<CallSuspendRecord>>> = Vec::new();
        let mut function_safepoints: Vec<Vec<SafepointRecord>> = Vec::new();
        let mut function_frame_reify_records: Vec<Vec<FrameReifyRecord>> = Vec::new();
        let mut max_deopt_live_values = 0usize;
        let handler_payload_kind = match Cfg::RootTransport::kind() {
            RootTransportKind::FrameScan => SafepointHandlerPayloadKind::FrameSize,
            RootTransportKind::ShadowStack | RootTransportKind::StackMap => {
                SafepointHandlerPayloadKind::SafepointIndex
            }
        };

        // Mark internal calls as control-aware per function.
        // A call is control-aware when:
        //   - The module uses continuations (prompts) — ALL internal calls need it
        //   - The CALLEE has guards/deopts that might propagate non-standard outcomes
        //   - The CALLER uses Invoke terminators (exception dispatch needs X1 check)
        // Plain call/return is the fast path with no outcome checking.
        let uses_continuations = module.functions.iter().any(|f| f.prompt_count > 0);

        // Per-function: does this function have deopts that callers need to handle?
        let func_has_deopts: Vec<bool> = module
            .functions
            .iter()
            .map(|f| !f.deopt_info.is_empty())
            .collect();

        // Any caller uses Invoke terminators?
        let any_invoke = module.functions.iter().any(|f| {
            f.blocks.iter().any(|b| {
                matches!(
                    b.terminator,
                    Terminator::Invoke { .. } | Terminator::InvokeIndirect { .. }
                )
            })
        });

        let direct_call_is_internal: Vec<bool> = module
            .func_table
            .iter()
            .enumerate()
            .map(|(idx, def)| match def {
                FuncDef::Internal(func_idx) => {
                    uses_continuations || func_has_deopts[*func_idx] || any_invoke
                }
                _ => false,
            })
            .collect();

        let mut function_root_scan_sizes: Vec<usize> = Vec::new();
        for (func_idx, func) in module.functions.iter().enumerate() {
            let mut lowerer = Lowerer::<Cfg, B, R>::new_module(
                func_idx,
                func,
                &direct_call_is_internal,
                call_table_base,
                None, // legacy compile_with_regalloc: no literal pool
                safepoint_handler,
            );
            lowerer.run();
            function_suspend_records.push(std::mem::take(&mut lowerer.suspend_records));
            function_safepoints.push(std::mem::take(&mut lowerer.safepoints));
            function_frame_reify_records.push(std::mem::take(&mut lowerer.frame_reify_records));
            max_deopt_live_values = max_deopt_live_values.max(lowerer.max_deopt_live_values);
            function_root_scan_sizes.push(lowerer.frame.root_scan_size() as usize);
            let code = lowerer.buf.into_code();
            let offset = memory.push(&code);
            entry_offsets.push(offset);
        }

        memory.finalize();

        // Register each function's code range in the global registry.
        let base = memory.base_ptr() as usize;
        let total_len = memory.len();
        let mut perf_entries: Vec<(usize, usize, &str)> = Vec::new();
        for (i, &offset) in entry_offsets.iter().enumerate() {
            let code_start = base + offset;
            let code_end = if i + 1 < entry_offsets.len() {
                base + entry_offsets[i + 1]
            } else {
                base + total_len
            };
            let safepoints_arc: std::sync::Arc<[SafepointRecord]> =
                function_safepoints[i].clone().into();
            register_jit_code(code_start, code_end, safepoints_arc);
            perf_entries.push((code_start, code_end - code_start, &module.functions[i].name));
        }
        let _ = function_root_scan_sizes; // historical (was passed to registry for conservative scan)
        write_perf_map_entries(&perf_entries);

        // 3. Patch internal function pointers into the call table
        let base = memory.base_ptr();
        for (ft_idx, def) in module.func_table.iter().enumerate() {
            if let FuncDef::Internal(func_idx) = def {
                let ptr = unsafe { base.add(entry_offsets[*func_idx]) };
                call_table.set(ft_idx, ptr);
            }
        }

        let call_mode = match safepoint_handler {
            Some(h) => CallMode::ControlAware { safepoint_handler: h },
            None => CallMode::FastCall,
        };
        JitModule {
            memory,
            call_table,
            literal_pool: LiteralPool::new(DEFAULT_LITERAL_POOL_CAPACITY),
            function_entry_offsets: entry_offsets,
            function_suspend_records,
            function_safepoints,
            function_frame_reify_records,
            handler_payload_kind,
            max_deopt_live_values,
            call_mode,
            safepoint_handler,
            extern_count_seen: module
                .func_table
                .iter()
                .filter(|d| matches!(d, FuncDef::Extern(_)))
                .count(),
        }
    }

    /// Create an empty `JitModule` ready to be incrementally extended via
    /// [`extend`](Self::extend).
    ///
    /// `call_table_capacity` is the maximum number of `func_table` slots the
    /// module can ever hold — fixed because emitted call sites bake the
    /// call table base address as an immediate. 64K (= 512 KiB) is a
    /// sensible default for REPL-scale workloads.
    ///
    /// `literal_pool_capacity` is the maximum number of GC-managed literal
    /// slots. Quote-style literals from frontends with a moving GC go here;
    /// see [`LiteralPool`]. Pass [`DEFAULT_LITERAL_POOL_CAPACITY`] (4096
    /// slots, 32 KiB) for a sane default.
    ///
    /// `call_mode` carries the safepoint handler when GC is in play; see
    /// [`CallMode`]. The previous separate `Option<u64>` for handler was
    /// removed deliberately — pairing "moving GC" with "no safepoints" is
    /// no longer a representable state.
    ///
    /// Type parameters fix the codegen choice for the lifetime of the
    /// module. `extend` must be called with the same parameters.
    pub fn new_empty<Cfg: CodegenConfig, B: LoweringBackend, R: RegisterAllocator>(
        call_table_capacity: usize,
        literal_pool_capacity: usize,
        call_mode: CallMode,
    ) -> Self
    where
        Cfg::Frames: FrameStrategy<Cfg::Layout, Cfg::Roots, Cfg::CallingConvention>,
    {
        let _phantom: PhantomData<(Cfg, B, R)> = PhantomData;
        let handler_payload_kind = match Cfg::RootTransport::kind() {
            RootTransportKind::FrameScan => SafepointHandlerPayloadKind::FrameSize,
            RootTransportKind::ShadowStack | RootTransportKind::StackMap => {
                SafepointHandlerPayloadKind::SafepointIndex
            }
        };
        let safepoint_handler = call_mode.safepoint_handler();
        JitModule {
            memory: PagedCodeMemory::new(),
            call_table: CallTable::new(call_table_capacity),
            literal_pool: LiteralPool::new(literal_pool_capacity),
            function_entry_offsets: Vec::new(),
            function_suspend_records: Vec::new(),
            function_safepoints: Vec::new(),
            function_frame_reify_records: Vec::new(),
            handler_payload_kind,
            max_deopt_live_values: 0,
            call_mode,
            safepoint_handler,
            extern_count_seen: 0,
        }
    }

    /// Compile and link the functions in `module` that aren't already in this
    /// `JitModule`, returning the [`FuncRef`]s of the newly added internal
    /// functions in declaration order.
    ///
    /// `module` is expected to be a superset of whatever was last passed in:
    /// the existing `func_table` slots and `functions` must match (by index)
    /// what we've already compiled. Anything beyond is treated as new.
    ///
    /// `externs` is the full extern pointer slice for `module` — the new call
    /// table slots pull their pointers starting at `extern_count_seen`.
    ///
    /// Cross-batch calls work for free: the call table base is stable, slots
    /// filled in earlier extends keep their addresses, and newly emitted
    /// `ldr + blr` against the same base resolves either old or new entries.
    pub fn extend<Cfg: CodegenConfig, B: LoweringBackend, R: RegisterAllocator>(
        &mut self,
        module: &Module,
        externs: &[*const u8],
    ) -> Vec<FuncRef>
    where
        Cfg::Frames: FrameStrategy<Cfg::Layout, Cfg::Roots, Cfg::CallingConvention>,
    {
        // 1. Append new func_table slots to the call table.
        let old_table_len = self.call_table.len();
        let new_table_len = module.func_table.len();
        assert!(
            new_table_len >= old_table_len,
            "extend: module has fewer func_table entries ({}) than already compiled ({})",
            new_table_len,
            old_table_len
        );
        for ft_idx in old_table_len..new_table_len {
            match &module.func_table[ft_idx] {
                FuncDef::Extern(_) => {
                    let ext_idx = self.extern_count_seen;
                    self.extern_count_seen += 1;
                    assert!(
                        ext_idx < externs.len(),
                        "extend: externs slice too short ({}) for extern slot {}",
                        externs.len(),
                        ext_idx
                    );
                    self.call_table.push(externs[ext_idx]);
                }
                FuncDef::Internal(_) => {
                    self.call_table.push(std::ptr::null());
                }
            }
        }

        // 2. Build direct_call_is_internal from the pinned CallMode.
        let direct_call_is_internal: Vec<bool> = if self.call_mode.is_control_aware() {
            module
                .func_table
                .iter()
                .map(|d| matches!(d, FuncDef::Internal(_)))
                .collect()
        } else {
            vec![false; module.func_table.len()]
        };

        // 3. Compile each new function.
        let new_func_start = self.function_entry_offsets.len();
        let new_func_end = module.functions.len();
        assert!(
            new_func_end >= new_func_start,
            "extend: module has fewer internal functions ({}) than already compiled ({})",
            new_func_end,
            new_func_start
        );

        if matches!(self.call_mode, CallMode::FastCall) {
            for func in &module.functions[new_func_start..new_func_end] {
                assert!(
                    func.prompt_count == 0,
                    "FastCall mode rejects continuations in '{}'",
                    func.name
                );
                assert!(
                    func.deopt_info.is_empty(),
                    "FastCall mode rejects deopts in '{}'",
                    func.name
                );
                for b in &func.blocks {
                    assert!(
                        !matches!(
                            b.terminator,
                            Terminator::Invoke { .. } | Terminator::InvokeIndirect { .. }
                        ),
                        "FastCall mode rejects Invoke terminator in '{}'",
                        func.name
                    );
                }
            }
        }

        let call_table_base = self.call_table.base_addr();
        let literal_pool_base = self.literal_pool.base_addr();
        for func_idx in new_func_start..new_func_end {
            let func = &module.functions[func_idx];
            let mut lowerer = Lowerer::<Cfg, B, R>::new_module(
                func_idx,
                func,
                &direct_call_is_internal,
                call_table_base,
                Some(literal_pool_base),
                self.safepoint_handler,
            );
            lowerer.run();
            self.function_suspend_records
                .push(std::mem::take(&mut lowerer.suspend_records));
            self.function_safepoints
                .push(std::mem::take(&mut lowerer.safepoints));
            self.function_frame_reify_records
                .push(std::mem::take(&mut lowerer.frame_reify_records));
            self.max_deopt_live_values = self
                .max_deopt_live_values
                .max(lowerer.max_deopt_live_values);
            let code = lowerer.buf.into_code();
            let offset = self.memory.push(&code);
            self.function_entry_offsets.push(offset);
        }

        // 4. Finalize new pages (existing pages are already RX, no-op for them).
        self.memory.finalize();

        // 5. Patch the new internal slots and register code ranges.
        let base_addr = self.memory.base_ptr() as usize;
        let total_len = self.memory.len();
        let mut perf_entries: Vec<(usize, usize, &str)> = Vec::new();
        for ft_idx in old_table_len..new_table_len {
            if let FuncDef::Internal(func_idx) = &module.func_table[ft_idx] {
                let ptr = unsafe {
                    self.memory
                        .base_ptr()
                        .add(self.function_entry_offsets[*func_idx])
                };
                self.call_table.set(ft_idx, ptr);
            }
        }
        for func_idx in new_func_start..new_func_end {
            let code_start = base_addr + self.function_entry_offsets[func_idx];
            let code_end = if func_idx + 1 < self.function_entry_offsets.len() {
                base_addr + self.function_entry_offsets[func_idx + 1]
            } else {
                base_addr + total_len
            };
            let safepoints_arc: std::sync::Arc<[SafepointRecord]> =
                self.function_safepoints[func_idx].clone().into();
            register_jit_code(code_start, code_end, safepoints_arc);
            perf_entries.push((
                code_start,
                code_end - code_start,
                &module.functions[func_idx].name,
            ));
        }
        write_perf_map_entries(&perf_entries);

        // 6. Collect FuncRefs of newly added internals.
        let mut new_refs = Vec::new();
        for ft_idx in old_table_len..new_table_len {
            if let FuncDef::Internal(_) = &module.func_table[ft_idx] {
                new_refs.push(FuncRef::from_u32(ft_idx as u32));
            }
        }
        new_refs
    }

    /// The pinned [`CallMode`] for this module.
    pub fn call_mode(&self) -> CallMode {
        self.call_mode
    }

    /// The GC-traced literal pool. Frontends push values here and emit
    /// `Inst::GcLiteral(idx)` to load them at runtime.
    pub fn literal_pool(&self) -> &LiteralPool {
        &self.literal_pool
    }

    /// Call a function in the module by its `FuncRef`.
    pub fn call(&self, func_ref: FuncRef, args: &[u64]) -> u64 {
        let ptr = self.call_table.get(func_ref.index());
        assert!(!ptr.is_null(), "call to unresolved function");
        match self.call_outcome(func_ref, args) {
            JitOutcome::Value(v) => v,
            JitOutcome::Void => 0,
            other => panic!("unexpected non-return outcome from JIT module call: {other:?}"),
        }
    }

    pub fn call_outcome(&self, func_ref: FuncRef, args: &[u64]) -> JitOutcome {
        let ptr = self.call_table.get(func_ref.index());
        assert!(!ptr.is_null(), "call to unresolved function");
        unsafe { call_jit_outcome(ptr, args, self.max_deopt_live_values) }
    }

    /// Snapshot the call table (func_table_index → code pointer).
    pub fn call_table(&self) -> Vec<*const u8> {
        self.call_table.snapshot()
    }

    pub fn function_ptr(&self, func_ref: FuncRef) -> *const u8 {
        let ptr = self.call_table.get(func_ref.index());
        assert!(!ptr.is_null(), "call to unresolved function");
        ptr
    }

    pub fn safepoints_for_function(&self, func_idx: usize) -> &[SafepointRecord] {
        &self.function_safepoints[func_idx]
    }

    /// All safepoint records across all functions, flattened.
    pub fn all_safepoints(&self) -> Vec<SafepointRecord> {
        self.function_safepoints
            .iter()
            .flat_map(|v| v.iter().cloned())
            .collect()
    }

    /// Dump all JIT code to a temp file and print disassembly commands.
    /// Each function's entry offset is printed for correlation.
    pub fn dump_code(&self) {
        let base = self.memory.base_ptr();
        let len = self.memory.len();
        eprintln!(
            "JIT module: {:?} ({} bytes, {} functions)",
            base,
            len,
            self.function_entry_offsets.len()
        );
        for (i, &off) in self.function_entry_offsets.iter().enumerate() {
            eprintln!("  func[{}] at offset {:#x}", i, off);
        }
        let bytes = unsafe { std::slice::from_raw_parts(base, len) };
        let path = "/tmp/jit_dump.bin";
        std::fs::write(path, bytes).unwrap();
        eprintln!("Disassemble with:");
        eprintln!(
            "  llvm-objdump -d -m aarch64 -b binary {} | head -500",
            path
        );
    }

    pub fn frame_reify_records_for_function(&self, func_idx: usize) -> &[FrameReifyRecord] {
        &self.function_frame_reify_records[func_idx]
    }

    pub fn native_resume_ptr(
        &self,
        func_idx: usize,
        record: &FrameReifyRecord,
    ) -> Option<*const u8> {
        record.native_resume_offset.map(|offset| unsafe {
            self.memory
                .base_ptr()
                .add(self.function_entry_offsets[func_idx] + offset)
        })
    }

    pub fn call_resume_outcome(
        &self,
        func_idx: usize,
        record: &FrameReifyRecord,
        frame_values_ptr: *const u64,
        resume_args: &[u64],
    ) -> JitOutcome {
        let ptr = self
            .native_resume_ptr(func_idx, record)
            .expect("record does not have a native resume entry");
        let args_ptr = if resume_args.is_empty() {
            std::ptr::null()
        } else {
            resume_args.as_ptr()
        };
        let args = [
            frame_values_ptr as u64,
            args_ptr as u64,
            resume_args.len() as u64,
        ];
        unsafe { call_jit_outcome(ptr, &args, self.max_deopt_live_values) }
    }

    /// View-based resume entry point. Takes the resume point and frame
    /// values directly.
    pub fn call_view_resume_outcome(
        &self,
        resume: &FrameResumePoint,
        values: &[u64],
        resume_args: &[u64],
    ) -> Option<JitOutcome> {
        let func_idx = resume.func_idx;
        if let Some(record) = self.function_frame_reify_records[func_idx]
            .iter()
            .find(|record| {
                record.kind == FrameReifyKind::CaptureSlice
                    && record.native_resume_offset.is_some()
                    && record.resume == *resume
            })
        {
            return Some(self.call_resume_outcome(func_idx, record, values.as_ptr(), resume_args));
        }

        let suspend = self.function_suspend_records[func_idx]
            .iter()
            .find(|record| record.native_resume_offset.is_some() && record.resume == *resume)?;
        let ptr = unsafe {
            self.memory.base_ptr().add(
                self.function_entry_offsets[func_idx]
                    + suspend.native_resume_offset.expect("checked above"),
            )
        };
        let args_ptr = if resume_args.is_empty() {
            std::ptr::null()
        } else {
            resume_args.as_ptr()
        };
        let args = [
            values.as_ptr() as u64,
            args_ptr as u64,
            resume_args.len() as u64,
        ];
        Some(unsafe { call_jit_outcome(ptr, &args, self.max_deopt_live_values) })
    }

    /// View-based invoke-resume entry point.
    pub fn call_view_invoke_resume_outcome(
        &self,
        resume: &FrameResumePoint,
        values: &[u64],
        is_exception: bool,
        resume_args: &[u64],
    ) -> Option<JitOutcome> {
        let func_idx = resume.func_idx;
        let suspend = self.function_suspend_records[func_idx]
            .iter()
            .find(|record| record.resume == *resume)?;
        let offset = if is_exception {
            suspend.native_exception_resume_offset?
        } else {
            suspend.native_resume_offset?
        };
        let ptr = unsafe {
            self.memory
                .base_ptr()
                .add(self.function_entry_offsets[func_idx] + offset)
        };
        let args_ptr = if resume_args.is_empty() {
            std::ptr::null()
        } else {
            resume_args.as_ptr()
        };
        let args = [
            values.as_ptr() as u64,
            args_ptr as u64,
            resume_args.len() as u64,
        ];
        Some(unsafe { call_jit_outcome(ptr, &args, self.max_deopt_live_values) })
    }

    pub fn handler_payload_kind(&self) -> SafepointHandlerPayloadKind {
        self.handler_payload_kind
    }
}

/// Call a JIT-compiled function with arbitrary 64-bit argument slots.
///
/// The JIT's entry convention is:
/// - slots 0..15 arrive in X0..X15
/// - remaining slots are passed on the stack at the incoming SP
///
/// This matches the lowering path used for JIT-to-JIT calls.
#[cfg(target_arch = "aarch64")]
unsafe fn call_jit_regs_with_reg_limit(
    ptr: *const u8,
    args: &[u64],
    reg_limit: usize,
    ctx: *mut JitControlContext,
) -> (u64, u64, u64, u64) {
    let padded_len = args.len().max(reg_limit);
    let mut padded = vec![0u64; padded_len];
    padded[..args.len()].copy_from_slice(args);
    let overflow = args.len().saturating_sub(reg_limit);
    let overflow_bytes = align_up(overflow * 8, 16);
    let overflow_count = overflow;
    let overflow_src = unsafe { padded.as_ptr().add(reg_limit) };
    let result: u64;
    let kind: u64;
    let payload0: u64;
    let payload1: u64;

    // Capture this Rust frame's FP (x29). The JIT prologue, on entry via
    // BLR, will save *this exact* value as the saved-FP slot of its
    // first frame. The FP-chain walker uses it as a stop sentinel so
    // GC-time root scanning doesn't follow the FP chain past JIT into
    // host (Rust) frames whose layout the JIT registry doesn't describe.
    let fence_fp: *const u8;
    unsafe {
        core::arch::asm!("mov {0}, x29", out(reg) fence_fp);
    }
    let _fence_guard = unsafe { JitEntryFpGuard::new(fence_fp) };

    unsafe {
        core::arch::asm!(
            "mov x23, {ctx}",
            "sub sp, sp, x20",
            "mov x9, sp",
            "cbz x21, 2f",
            "1:",
            "ldr x10, [x22], #8",
            "str x10, [x9], #8",
            "sub x21, x21, #1",
            "cbnz x21, 1b",
            "2:",
            "ldp x0, x1, [x17]",
            "ldp x2, x3, [x17, #16]",
            "ldp x4, x5, [x17, #32]",
            "ldp x6, x7, [x17, #48]",
            "ldp x8, x9, [x17, #64]",
            "ldp x10, x11, [x17, #80]",
            "ldp x12, x13, [x17, #96]",
            "ldp x14, x15, [x17, #112]",
            "blr x16",
            "add sp, sp, x20",
            ctx = in(reg) ctx,
            inlateout("x21") overflow_count => _,
            inlateout("x22") overflow_src => _,
            in("x20") overflow_bytes,
            in("x16") ptr,
            in("x17") padded.as_ptr(),
            lateout("x0") result,
            lateout("x1") kind,
            lateout("x2") payload0,
            lateout("x3") payload1,
            lateout("x23") _,
            lateout("x9") _,
            lateout("x10") _,
            clobber_abi("C"),
        );
    }
    (result, kind, payload0, payload1)
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn call_jit_with_reg_limit(ptr: *const u8, args: &[u64], reg_limit: usize) -> u64 {
    match unsafe { call_jit_outcome_with_reg_limit(ptr, args, reg_limit, 0) } {
        JitOutcome::Value(v) => v,
        JitOutcome::Void => 0,
        other => panic!("unexpected non-return outcome from JIT entry: {other:?}"),
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn call_jit(ptr: *const u8, args: &[u64]) -> u64 {
    unsafe { call_jit_with_reg_limit(ptr, args, 16) }
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn call_jit_outcome_with_reg_limit(
    ptr: *const u8,
    args: &[u64],
    reg_limit: usize,
    max_deopt_live_values: usize,
) -> JitOutcome {
    let mut live_values = vec![0u64; max_deopt_live_values];
    let mut ctx = JitControlContext {
        live_values_ptr: live_values.as_mut_ptr(),
        live_values_len: 0,
    };
    let (result, kind, payload0, payload1) =
        unsafe { call_jit_regs_with_reg_limit(ptr, args, reg_limit, &mut ctx) };
    match kind {
        x if x == JitOutcomeKind::ReturnValue as u64 => JitOutcome::Value(result),
        x if x == JitOutcomeKind::ReturnVoid as u64 => JitOutcome::Void,
        x if x == JitOutcomeKind::Exception as u64 => JitOutcome::Exception(payload0),
        x if x == JitOutcomeKind::Deopt as u64 => {
            live_values.truncate(ctx.live_values_len);
            JitOutcome::Deopt {
                deopt_id: DeoptId::from_index(payload0 as usize),
                resume_point: payload1,
                live_values,
            }
        }
        x if x == JitOutcomeKind::CaptureSlice as u64 => {
            live_values.truncate(ctx.live_values_len);
            JitOutcome::CaptureSlice {
                func_idx: payload1 as usize,
                record_idx: payload0 as usize,
                values: live_values,
            }
        }
        x if x == JitOutcomeKind::CloneSlice as u64 => {
            live_values.truncate(ctx.live_values_len);
            JitOutcome::CloneSlice {
                func_idx: payload1 as usize,
                record_idx: payload0 as usize,
                values: live_values,
            }
        }
        x if x == JitOutcomeKind::ResumeSlice as u64 => {
            live_values.truncate(ctx.live_values_len);
            JitOutcome::ResumeSlice {
                func_idx: payload1 as usize,
                record_idx: payload0 as usize,
                values: live_values,
            }
        }
        x if x == JitOutcomeKind::AbortToPrompt as u64 => {
            live_values.truncate(ctx.live_values_len);
            JitOutcome::AbortToPrompt {
                func_idx: payload1 as usize,
                record_idx: payload0 as usize,
                values: live_values,
            }
        }
        _ => panic!("unknown JIT outcome kind: {kind}"),
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn call_jit_outcome(
    ptr: *const u8,
    args: &[u64],
    max_deopt_live_values: usize,
) -> JitOutcome {
    unsafe { call_jit_outcome_with_reg_limit(ptr, args, 16, max_deopt_live_values) }
}

#[cfg(target_arch = "x86_64")]
unsafe fn call_jit_regs_with_reg_limit(
    ptr: *const u8,
    args: &[u64],
    reg_limit: usize,
    ctx: *mut JitControlContext,
) -> (u64, u64, u64, u64) {
    let reg_count = reg_limit.min(6);
    let padded_len = args.len().max(6);
    let mut padded = vec![0u64; padded_len];
    padded[..args.len()].copy_from_slice(args);
    let overflow = args.len().saturating_sub(reg_count);
    let overflow_bytes = align_up(overflow * 8, 16);
    let overflow_src = unsafe { padded.as_ptr().add(reg_count) };
    let result: u64;
    let kind: u64;
    let payload0: u64;
    let payload1: u64;
    unsafe {
        core::arch::asm!(
            "push r12",
            "push r13",
            "push r14",
            "push r15",
            "mov r15, {ctx}",
            "mov r12, {arg_ptr}",
            "mov r13, {target}",
            "mov r14, {overflow_bytes}",
            "sub rsp, r14",
            "mov r10, {overflow_count}",
            "mov r11, {overflow_src}",
            "test r10, r10",
            "jz 4f",
            "mov rdi, rsp",
            "3:",
            "mov rax, [r11]",
            "mov [rdi], rax",
            "add r11, 8",
            "add rdi, 8",
            "dec r10",
            "jnz 3b",
            "4:",
            "mov rdi, [r12 + 0]",
            "mov rsi, [r12 + 8]",
            "mov rdx, [r12 + 16]",
            "mov rcx, [r12 + 24]",
            "mov r8,  [r12 + 32]",
            "mov r9,  [r12 + 40]",
            "call r13",
            "add rsp, r14",
            "pop r15",
            "pop r14",
            "pop r13",
            "pop r12",
            ctx = in(reg) ctx,
            overflow_bytes = in(reg) overflow_bytes,
            overflow_count = in(reg) overflow,
            overflow_src = in(reg) overflow_src,
            arg_ptr = in(reg) padded.as_ptr(),
            target = in(reg) ptr,
            lateout("rax") result,
            lateout("rcx") kind,
            lateout("rdx") payload0,
            lateout("rsi") payload1,
            lateout("r10") _,
            lateout("r11") _,
            lateout("r12") _,
            lateout("r13") _,
            lateout("r14") _,
            lateout("r15") _,
            clobber_abi("C"),
        );
    }
    (result, kind, payload0, payload1)
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn call_jit_with_reg_limit(ptr: *const u8, args: &[u64], reg_limit: usize) -> u64 {
    match unsafe { call_jit_outcome_with_reg_limit(ptr, args, reg_limit, 0) } {
        JitOutcome::Value(v) => v,
        JitOutcome::Void => 0,
        other => panic!("unexpected non-return outcome from JIT entry: {other:?}"),
    }
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn call_jit(ptr: *const u8, args: &[u64]) -> u64 {
    unsafe { call_jit_with_reg_limit(ptr, args, 6) }
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn call_jit_outcome_with_reg_limit(
    ptr: *const u8,
    args: &[u64],
    reg_limit: usize,
    max_deopt_live_values: usize,
) -> JitOutcome {
    let mut live_values = vec![0u64; max_deopt_live_values];
    let mut ctx = JitControlContext {
        live_values_ptr: live_values.as_mut_ptr(),
        live_values_len: 0,
    };
    let (result, kind, payload0, payload1) =
        unsafe { call_jit_regs_with_reg_limit(ptr, args, reg_limit, &mut ctx) };
    match kind {
        x if x == JitOutcomeKind::ReturnValue as u64 => JitOutcome::Value(result),
        x if x == JitOutcomeKind::ReturnVoid as u64 => JitOutcome::Void,
        x if x == JitOutcomeKind::Exception as u64 => JitOutcome::Exception(payload0),
        x if x == JitOutcomeKind::Deopt as u64 => {
            live_values.truncate(ctx.live_values_len);
            JitOutcome::Deopt {
                deopt_id: DeoptId::from_index(payload0 as usize),
                resume_point: payload1,
                live_values,
            }
        }
        x if x == JitOutcomeKind::CaptureSlice as u64 => {
            live_values.truncate(ctx.live_values_len);
            JitOutcome::CaptureSlice {
                func_idx: payload1 as usize,
                record_idx: payload0 as usize,
                values: live_values,
            }
        }
        x if x == JitOutcomeKind::CloneSlice as u64 => {
            live_values.truncate(ctx.live_values_len);
            JitOutcome::CloneSlice {
                func_idx: payload1 as usize,
                record_idx: payload0 as usize,
                values: live_values,
            }
        }
        x if x == JitOutcomeKind::ResumeSlice as u64 => {
            live_values.truncate(ctx.live_values_len);
            JitOutcome::ResumeSlice {
                func_idx: payload1 as usize,
                record_idx: payload0 as usize,
                values: live_values,
            }
        }
        x if x == JitOutcomeKind::AbortToPrompt as u64 => {
            live_values.truncate(ctx.live_values_len);
            JitOutcome::AbortToPrompt {
                func_idx: payload1 as usize,
                record_idx: payload0 as usize,
                values: live_values,
            }
        }
        _ => panic!("unknown JIT outcome kind: {kind}"),
    }
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn call_jit_outcome(
    ptr: *const u8,
    args: &[u64],
    max_deopt_live_values: usize,
) -> JitOutcome {
    unsafe { call_jit_outcome_with_reg_limit(ptr, args, 6, max_deopt_live_values) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AssignmentTarget {
    CallArg(CallArgLocation),
    FrameSlot(FrameSlotAccess),
}

impl AssignmentTarget {
    fn as_machine_location(self) -> MachineLocation {
        match self {
            AssignmentTarget::CallArg(CallArgLocation::Register(reg_idx)) => {
                MachineLocation::Reg(machine_gp(reg_idx as u8))
            }
            AssignmentTarget::CallArg(CallArgLocation::Stack(offset)) => {
                MachineLocation::StackArg(offset)
            }
            AssignmentTarget::FrameSlot(access) => MachineLocation::FrameSlot(access),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ValueAssignment {
    value: Value,
    ty: Type,
    target: AssignmentTarget,
}

// ─── Frame Layout State ────────────────────────────────────────────

// ─── Register State ────────────────────────────────────────────────

// ─── Block metadata ────────────────────────────────────────────────

struct BlockMeta {
    label: Label,
    active_prompts: Vec<PromptId>,
}

struct PendingResumeStub {
    target: ResumeStubTarget,
    continue_label: Label,
    resume_inject: ResumeInjectTarget,
    slot_offsets: Vec<Option<i32>>,
}

enum ResumeStubTarget {
    FrameReifyRecord(usize),
    SuspendRecord(usize),
    SuspendRecordException(usize),
}

enum ResumeInjectTarget {
    None,
    FrameSlot(i32),
    ReturnReg,
}

struct RootTransportState {
    shadow_slots_by_value: Vec<Option<i32>>,
}

impl RootTransportState {
    fn new(num_values: usize) -> Self {
        Self {
            shadow_slots_by_value: vec![None; num_values],
        }
    }
}

fn simulate_block_prompt_stack(block: &Block, stack: &[PromptId]) -> Vec<PromptId> {
    let mut prompts = stack.to_vec();
    for inst_node in &block.insts {
        match &inst_node.inst {
            Inst::PushPrompt(prompt, _) => prompts.push(*prompt),
            Inst::PopPrompt(prompt) => {
                let popped = prompts.pop().unwrap_or_else(|| {
                    panic!("pop_prompt without matching active prompt in block")
                });
                assert_eq!(
                    popped, *prompt,
                    "mismatched prompt nesting in block: expected {:?}, got {:?}",
                    popped, prompt
                );
            }
            _ => {}
        }
    }
    prompts
}

fn terminator_successors(terminator: &Terminator) -> Vec<BlockId> {
    match terminator {
        Terminator::Jump(target, _) => vec![*target],
        Terminator::BrIf {
            then_block,
            else_block,
            ..
        } => vec![*then_block, *else_block],
        Terminator::Switch {
            cases,
            default_block,
            ..
        } => {
            let mut succs = Vec::with_capacity(cases.len() + 1);
            succs.extend(cases.iter().map(|(_, block, _)| *block));
            succs.push(*default_block);
            succs
        }
        Terminator::Invoke {
            normal, exception, ..
        }
        | Terminator::InvokeIndirect {
            normal, exception, ..
        } => vec![*normal, *exception],
        Terminator::ResumeSlice { return_block, .. } => vec![*return_block],
        Terminator::CaptureSlice {
            handler_block,
            resume_block,
            ..
        } => {
            vec![*handler_block, *resume_block]
        }
        Terminator::Ret(_)
        | Terminator::RetVoid
        | Terminator::Unreachable
        | Terminator::AbortToPrompt { .. } => Vec::new(),
    }
}

fn assign_block_prompt_stacks(func: &Function, block_meta: &mut [BlockMeta]) {
    if func.blocks.is_empty() {
        return;
    }
    let mut incoming: Vec<Option<Vec<PromptId>>> = vec![None; func.blocks.len()];
    incoming[0] = Some(Vec::new());
    let mut queue = VecDeque::from([0usize]);

    while let Some(block_idx) = queue.pop_front() {
        let entry_prompts = incoming[block_idx]
            .clone()
            .expect("queued block missing prompt stack");
        let exit_prompts = simulate_block_prompt_stack(&func.blocks[block_idx], &entry_prompts);
        for succ in terminator_successors(&func.blocks[block_idx].terminator) {
            let succ_idx = succ.index();
            match &incoming[succ_idx] {
                Some(existing) => {
                    assert_eq!(
                        existing, &exit_prompts,
                        "inconsistent prompt stack entering block bb{}",
                        succ_idx
                    );
                }
                None => {
                    incoming[succ_idx] = Some(exit_prompts.clone());
                    queue.push_back(succ_idx);
                }
            }
        }
    }

    for (idx, meta) in block_meta.iter_mut().enumerate() {
        meta.active_prompts = incoming[idx].clone().unwrap_or_default();
    }
}

// ─── Lowerer ───────────────────────────────────────────────────────

struct Lowerer<'a, Cfg: CodegenConfig, B: LoweringBackend, R: RegisterAllocator = GreedyRegState>
where
    Cfg::Frames: FrameStrategy<Cfg::Layout, Cfg::Roots, Cfg::CallingConvention>,
{
    func: &'a Function,
    current_func_idx: usize,
    externs: &'a [*const u8],
    direct_call_is_internal: Option<&'a [bool]>,
    /// When Some, all calls go through an indirect call table at this
    /// address. `call_table[func_ref.index()]` holds the function pointer.
    call_table_base: Option<u64>,
    /// When Some, `Inst::GcLiteral(idx)` loads from this base address.
    /// Pointer is the JitModule's `literal_pool.base()`.
    literal_pool_base: Option<u64>,
    /// When Some, safepoints emit a call to this handler function.
    /// Signature: `extern "C" fn(frame_ptr: *mut u8, frame_size: usize)`.
    safepoint_handler: Option<u64>,
    buf: CodeBuffer<B::Arch>,
    regs: R,
    frame: <Cfg::Frames as FrameStrategy<Cfg::Layout, Cfg::Roots, Cfg::CallingConvention>>::Layout,
    block_meta: Vec<BlockMeta>,
    root_transport: RootTransportState,
    safepoints: Vec<SafepointRecord>,
    frame_reify_records: Vec<FrameReifyRecord>,
    suspend_records: Vec<Box<CallSuspendRecord>>,
    pending_resume_stubs: Vec<PendingResumeStub>,
    max_deopt_live_values: usize,
    prologue_stp_offset: usize,
    epilogue_ldp_offsets: Vec<usize>, // offsets of LDP instructions to patch
    /// Pre-allocated frame offsets for each StackSlot declared by the function.
    stack_slot_offsets: Vec<i32>,
    _config: PhantomData<Cfg>,
    _backend: PhantomData<B>,
}

impl<'a, Cfg: CodegenConfig, B: LoweringBackend, R: RegisterAllocator> Lowerer<'a, Cfg, B, R>
where
    Cfg::Frames: FrameStrategy<Cfg::Layout, Cfg::Roots, Cfg::CallingConvention>,
{
    fn direct_call_is_control_aware(&self, func_ref: FuncRef) -> bool {
        self.direct_call_is_internal
            .map(|kinds| kinds[func_ref.index()])
            .unwrap_or(false)
    }

    fn is_gc_ptr_value(&self, val: Value) -> bool {
        self.regs.value_type(val) == Type::GcPtr
    }

    fn assign_gp_value(&mut self, val: Value, reg: u8) {
        self.regs.assign_gp(val, reg);
        self.track_value_in_gp(val, reg);
    }

    fn assign_fp_value(&mut self, val: Value, reg: u8) {
        self.regs.assign_fp(val, reg);
    }

    fn assign_spill_value(&mut self, val: Value, offset: i32) {
        self.regs.set_spill_slot(val, offset);
        self.regs.set_value_loc(val, ValueLoc::Spill(offset));
        self.track_value_in_frame_slot(val, offset);
    }

    fn ensure_shadow_slot_for_value(&mut self, val: Value) -> Option<i32> {
        if Cfg::RootTransport::kind() != RootTransportKind::ShadowStack
            || !self.is_gc_ptr_value(val)
        {
            return None;
        }
        let slot = &mut self.root_transport.shadow_slots_by_value[val.index()];
        if slot.is_none() {
            *slot = Some(self.frame.alloc_shadow_root_slot());
        }
        *slot
    }

    fn track_value_in_gp(&mut self, val: Value, reg: u8) {
        if let Some(shadow_slot) = self.ensure_shadow_slot_for_value(val) {
            B::emit_store_gp_to_frame(
                &mut self.buf,
                machine_gp(reg),
                self.frame.slot_access(shadow_slot),
            );
        }
    }

    fn track_value_in_frame_slot(&mut self, val: Value, slot_offset: i32) {
        if let Some(shadow_slot) = self.ensure_shadow_slot_for_value(val) {
            if shadow_slot != slot_offset {
                B::emit_load_gp_from_frame(
                    &mut self.buf,
                    machine_gp(27),
                    self.frame.slot_access(slot_offset),
                );
                B::emit_store_gp_to_frame(
                    &mut self.buf,
                    machine_gp(27),
                    self.frame.slot_access(shadow_slot),
                );
            }
        }
    }

    fn ensure_shadow_value_materialized(&mut self, val: Value) {
        if Cfg::RootTransport::kind() != RootTransportKind::ShadowStack
            || !self.is_gc_ptr_value(val)
        {
            return;
        }
        if self.root_transport.shadow_slots_by_value[val.index()].is_some() {
            return;
        }
        match self.regs.value_loc(val) {
            ValueLoc::GpReg(r) => self.track_value_in_gp(val, r),
            ValueLoc::Spill(offset) => self.track_value_in_frame_slot(val, offset),
            ValueLoc::FpReg(_) | ValueLoc::Unassigned => {}
        }
    }

    fn emit_set_outcome_kind(&mut self, kind: JitOutcomeKind) {
        B::emit_mov_imm(&mut self.buf, machine_gp(1), kind as u64);
    }

    fn emit_return_current_outcome(&mut self) {
        self.emit_epilogue();
    }

    fn emit_write_deopt_live_values(&mut self, live: &[Value]) {
        self.emit_write_outcome_values(live);
    }

    fn emit_write_full_frame_values(&mut self) {
        B::emit_load_gp(
            &mut self.buf,
            machine_gp(28),
            machine_gp(23),
            0,
            MachineWordSize::W64,
        );
        for value_idx in 0..self.func.value_types.len() {
            let value = Value::from_index(value_idx);
            let ty = self.regs.value_type(value);
            match self.regs.value_loc(value) {
                ValueLoc::FpReg(src) if is_float_type(ty) => {
                    B::emit_fp_to_gp_move(&mut self.buf, machine_gp(27), machine_fp(src));
                    B::emit_store_gp(
                        &mut self.buf,
                        machine_gp(27),
                        machine_gp(28),
                        (value_idx as i32) * 8,
                        MachineWordSize::W64,
                    );
                }
                ValueLoc::GpReg(src) => {
                    B::emit_store_gp(
                        &mut self.buf,
                        machine_gp(src),
                        machine_gp(28),
                        (value_idx as i32) * 8,
                        MachineWordSize::W64,
                    );
                }
                ValueLoc::Spill(offset) => {
                    if is_float_type(ty) {
                        B::emit_load_fp_from_frame(
                            &mut self.buf,
                            machine_fp(31),
                            self.frame.slot_access(offset),
                        );
                        B::emit_fp_to_gp_move(&mut self.buf, machine_gp(27), machine_fp(31));
                        B::emit_store_gp(
                            &mut self.buf,
                            machine_gp(27),
                            machine_gp(28),
                            (value_idx as i32) * 8,
                            MachineWordSize::W64,
                        );
                    } else {
                        B::emit_load_gp_from_frame(
                            &mut self.buf,
                            machine_gp(27),
                            self.frame.slot_access(offset),
                        );
                        B::emit_store_gp(
                            &mut self.buf,
                            machine_gp(27),
                            machine_gp(28),
                            (value_idx as i32) * 8,
                            MachineWordSize::W64,
                        );
                    }
                }
                ValueLoc::Unassigned => {
                    let _ = value;
                    B::emit_mov_imm(&mut self.buf, machine_gp(27), 0);
                    B::emit_store_gp(
                        &mut self.buf,
                        machine_gp(27),
                        machine_gp(28),
                        (value_idx as i32) * 8,
                        MachineWordSize::W64,
                    );
                }
                ValueLoc::FpReg(_) => unreachable!("non-float value in FP register"),
            }
        }
        B::emit_mov_imm(
            &mut self.buf,
            machine_gp(28),
            self.func.value_types.len() as u64,
        );
        B::emit_store_gp(
            &mut self.buf,
            machine_gp(28),
            machine_gp(23),
            8,
            MachineWordSize::W64,
        );
    }

    fn emit_write_outcome_values(&mut self, values: &[Value]) {
        B::emit_load_gp(
            &mut self.buf,
            machine_gp(28),
            machine_gp(23),
            0,
            MachineWordSize::W64,
        );
        for (idx, &value) in values.iter().enumerate() {
            let ty = self.regs.value_type(value);
            if is_float_type(ty) {
                let src = self
                    .regs
                    .ensure_in_fp_reg::<B>(&mut self.buf, &mut self.frame, value);
                B::emit_fp_to_gp_move(&mut self.buf, machine_gp(27), machine_fp(src));
                B::emit_store_gp(
                    &mut self.buf,
                    machine_gp(27),
                    machine_gp(28),
                    (idx as i32) * 8,
                    MachineWordSize::W64,
                );
            } else {
                let src = self
                    .regs
                    .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, value);
                B::emit_store_gp(
                    &mut self.buf,
                    machine_gp(src),
                    machine_gp(28),
                    (idx as i32) * 8,
                    MachineWordSize::W64,
                );
            }
        }
        B::emit_mov_imm(&mut self.buf, machine_gp(28), values.len() as u64);
        B::emit_store_gp(
            &mut self.buf,
            machine_gp(28),
            machine_gp(23),
            8,
            MachineWordSize::W64,
        );
    }

    fn emit_deopt_return(&mut self, deopt_id: DeoptId, resume_point: u64, live: &[Value]) {
        self.regs
            .spill_all_live::<B>(&mut self.buf, &mut self.frame);
        self.emit_write_deopt_live_values(live);
        self.emit_set_outcome_kind(JitOutcomeKind::Deopt);
        B::emit_mov_imm(&mut self.buf, machine_gp(2), deopt_id.index() as u64);
        B::emit_mov_imm(&mut self.buf, machine_gp(3), resume_point);
        self.emit_return_current_outcome();
    }

    fn push_frame_reify_record(
        &mut self,
        kind: FrameReifyKind,
        prompt: Option<PromptId>,
        active_prompts: &[PromptId],
        values: &[Value],
        control_values: &[Value],
        return_dest: Option<usize>,
        resume: FrameResumePoint,
    ) -> usize {
        let value_indices: Vec<usize> = values.iter().map(|value| value.index()).collect();
        let control_value_indices: Vec<usize> =
            control_values.iter().map(|value| value.index()).collect();
        let value_types: Vec<Type> = values
            .iter()
            .map(|value| self.regs.value_type(*value))
            .collect();
        let root_payload_indices: Vec<usize> = value_types
            .iter()
            .enumerate()
            .filter_map(|(idx, ty)| {
                (ty.is_gc()
                    && self.regs.value_loc(Value::from_index(value_indices[idx]))
                        != ValueLoc::Unassigned)
                    .then_some(idx)
            })
            .collect();
        let idx = self.frame_reify_records.len();
        self.frame_reify_records.push(FrameReifyRecord {
            kind,
            prompt,
            active_prompts: active_prompts.to_vec(),
            resume,
            native_resume_offset: None,
            frame_value_count: self.func.value_types.len(),
            value_indices,
            control_value_indices,
            value_types,
            root_payload_indices,
            return_dest,
        });
        idx
    }

    fn emit_frame_reify_outcome(
        &mut self,
        kind: JitOutcomeKind,
        record_idx: usize,
        values: &[Value],
    ) {
        self.regs
            .spill_all_live::<B>(&mut self.buf, &mut self.frame);
        self.emit_write_outcome_values(values);
        self.emit_set_outcome_kind(kind);
        B::emit_mov_imm(&mut self.buf, machine_gp(2), record_idx as u64);
        B::emit_mov_imm(&mut self.buf, machine_gp(3), self.current_func_idx as u64);
        self.emit_return_current_outcome();
    }

    fn push_call_suspend_record(
        &mut self,
        active_prompts: &[PromptId],
        callee_caller_resume: SuspendCallerResume,
        resume: FrameResumePoint,
    ) -> *const CallSuspendRecord {
        let snapshot = self.regs.value_info_snapshot();
        let value_accesses = snapshot
            .iter()
            .map(|(spill_slot, _ty)| spill_slot.map(|slot| self.frame.slot_access(slot)))
            .collect::<Vec<_>>();
        let root_value_indices = snapshot
            .iter()
            .enumerate()
            .filter_map(|(idx, (spill_slot, ty))| {
                (ty.is_gc() && spill_slot.is_some()).then_some(idx)
            })
            .collect::<Vec<_>>();
        let record = Box::new(CallSuspendRecord {
            resume,
            native_resume_offset: None,
            native_exception_resume_offset: None,
            value_accesses,
            root_value_indices,
            active_prompts: active_prompts.iter().map(|p| p.index() as u32).collect(),
            callee_caller_resume,
        });
        let ptr = (&*record) as *const CallSuspendRecord;
        self.suspend_records.push(record);
        ptr
    }

    fn emit_push_suspended_frame(&mut self, record_ptr: *const CallSuspendRecord) {
        B::emit_mov_imm(&mut self.buf, B::call_arg_gp(0), record_ptr as usize as u64);
        B::emit_gp_move(&mut self.buf, B::call_arg_gp(1), machine_gp(29));
        B::emit_stack_pointer_to_gp(&mut self.buf, B::call_arg_gp(2));
        B::emit_mov_imm(
            &mut self.buf,
            machine_gp(28),
            jit_push_suspended_frame as usize as u64,
        );
        B::emit_call_reg(&mut self.buf, machine_gp(28));
    }

    fn emit_pop_suspended_frame(&mut self) {
        B::emit_mov_imm(
            &mut self.buf,
            machine_gp(28),
            jit_pop_suspended_frame as usize as u64,
        );
        B::emit_call_reg(&mut self.buf, machine_gp(28));
    }

    fn new_module(
        current_func_idx: usize,
        func: &'a Function,
        direct_call_is_internal: &'a [bool],
        call_table_base: u64,
        literal_pool_base: Option<u64>,
        safepoint_handler: Option<u64>,
    ) -> Self {
        Self::new_inner(
            current_func_idx,
            func,
            &[],
            Some(direct_call_is_internal),
            Some(call_table_base),
            literal_pool_base,
            safepoint_handler,
        )
    }

    fn new_inner(
        current_func_idx: usize,
        func: &'a Function,
        externs: &'a [*const u8],
        direct_call_is_internal: Option<&'a [bool]>,
        call_table_base: Option<u64>,
        literal_pool_base: Option<u64>,
        safepoint_handler: Option<u64>,
    ) -> Self {
        let num_values = func.value_types.len();
        let mut buf = CodeBuffer::<B::Arch>::new();

        // Create labels for all blocks
        let mut block_meta = Vec::new();
        for (_i, _block) in func.blocks.iter().enumerate() {
            let label = buf.create_label();
            block_meta.push(BlockMeta {
                label,
                active_prompts: Vec::new(),
            });
        }
        assign_block_prompt_stacks(func, &mut block_meta);

        let mut regs = R::new(num_values);
        let mut frame = Cfg::Frames::new_layout(func.blocks.len());

        // Liveness pass: count uses
        for block in &func.blocks {
            for inst_node in &block.insts {
                inst_node.inst.for_each_value(|v| {
                    regs.inc_use(v);
                });
            }
            block.terminator.for_each_value(|v| {
                regs.inc_use(v);
            });
        }

        // Set types for all values
        for (i, ty) in func.value_types.iter().enumerate() {
            regs.set_type(Value::from_index(i), *ty);
        }

        // Run allocator-specific pre-analysis (e.g., linear scan interval computation).
        regs.prepare::<B>(func);

        // Allocate canonical spill slots for ALL blocks with params.
        // This simplifies block transitions: every block param has a
        // canonical spill location, regardless of predecessor count.
        for (i, block) in func.blocks.iter().enumerate() {
            if i > 0 && !block.params.is_empty() {
                for _ in &block.params {
                    let offset = frame.alloc_root_slot();
                    frame.add_block_param_slot(i, offset);
                }
            }
        }

        Lowerer {
            func,
            current_func_idx,
            externs,
            direct_call_is_internal,
            call_table_base,
            literal_pool_base,
            safepoint_handler,
            buf,
            regs,
            frame,
            block_meta,
            root_transport: RootTransportState::new(num_values),
            safepoints: Vec::new(),
            frame_reify_records: Vec::new(),
            suspend_records: Vec::new(),
            pending_resume_stubs: Vec::new(),
            max_deopt_live_values: func
                .blocks
                .iter()
                .flat_map(|block| block.insts.iter())
                .filter_map(|node| match &node.inst {
                    Inst::Guard(_, _, live) => Some(live.len()),
                    Inst::CaptureSlice(_, _) => Some(func.value_types.len()),
                    Inst::CloneSlice(_) => Some(func.value_types.len()),
                    _ => None,
                })
                .chain(
                    func.blocks
                        .iter()
                        .filter_map(|block| match &block.terminator {
                            Terminator::AbortToPrompt { args, .. } => Some(args.len()),
                            Terminator::ResumeSlice { args, .. } => Some(args.len() + 1),
                            _ => None,
                        }),
                )
                .max()
                .unwrap_or(0),
            prologue_stp_offset: 0,
            epilogue_ldp_offsets: Vec::new(),
            stack_slot_offsets: Vec::new(),
            _config: PhantomData,
            _backend: PhantomData,
        }
    }

    fn run(&mut self) {
        // Pre-allocate frame slots for all declared stack slots.
        self.stack_slot_offsets = self
            .func
            .stack_slots
            .iter()
            .map(|slot_data| {
                if slot_data.is_gc_root {
                    self.frame.alloc_root_slot()
                } else {
                    self.frame.alloc_local_slot()
                }
            })
            .collect();

        self.emit_prologue();

        for block_idx in 0..self.func.blocks.len() {
            self.lower_block(block_idx);
        }

        self.patch_prologue();
        self.emit_resume_stubs();
    }

    fn emit_prologue(&mut self) {
        self.prologue_stp_offset = B::emit_prologue(&mut self.buf);
    }

    fn emit_epilogue(&mut self) {
        B::emit_epilogue(&mut self.buf, &mut self.epilogue_ldp_offsets);
    }

    fn patch_prologue(&mut self) {
        let total_frame = self
            .frame
            .total_frame_size(Cfg::Frames::stack_align())
            .max(16);

        B::emit_frame_size_patch(
            &mut self.buf,
            self.prologue_stp_offset,
            &self.epilogue_ldp_offsets,
            total_frame,
        );
    }

    fn emit_resume_stubs(&mut self) {
        let total_frame = self
            .frame
            .total_frame_size(Cfg::Frames::stack_align())
            .max(16);
        let pending = std::mem::take(&mut self.pending_resume_stubs);
        for stub in pending {
            let code_offset = self.buf.current_offset();
            match stub.target {
                ResumeStubTarget::FrameReifyRecord(record_idx) => {
                    self.frame_reify_records[record_idx].native_resume_offset = Some(code_offset);
                }
                ResumeStubTarget::SuspendRecord(record_idx) => {
                    self.suspend_records[record_idx].native_resume_offset = Some(code_offset);
                }
                ResumeStubTarget::SuspendRecordException(record_idx) => {
                    self.suspend_records[record_idx].native_exception_resume_offset =
                        Some(code_offset);
                }
            }
            let patch_offset = B::emit_prologue(&mut self.buf);
            B::emit_frame_size_patch(&mut self.buf, patch_offset, &[], total_frame);

            let value_types: Vec<Type> = match stub.target {
                ResumeStubTarget::FrameReifyRecord(record_idx) => {
                    self.frame_reify_records[record_idx].value_types.clone()
                }
                ResumeStubTarget::SuspendRecord(_)
                | ResumeStubTarget::SuspendRecordException(_) => self.func.value_types.clone(),
            };
            for (payload_idx, slot_opt) in stub.slot_offsets.iter().enumerate() {
                let Some(slot) = slot_opt else {
                    continue;
                };
                let ty = value_types[payload_idx];
                B::emit_load_gp(
                    &mut self.buf,
                    machine_gp(27),
                    machine_gp(0),
                    (payload_idx as i32) * 8,
                    MachineWordSize::W64,
                );
                if is_float_type(ty) {
                    B::emit_gp_to_fp_move(&mut self.buf, machine_fp(31), machine_gp(27));
                    B::emit_store_fp_to_frame(
                        &mut self.buf,
                        machine_fp(31),
                        self.frame.slot_access(*slot),
                    );
                } else {
                    B::emit_store_gp_to_frame(
                        &mut self.buf,
                        machine_gp(27),
                        self.frame.slot_access(*slot),
                    );
                }
            }

            match stub.resume_inject {
                ResumeInjectTarget::None => {}
                ResumeInjectTarget::FrameSlot(return_slot) => {
                    let skip = self.buf.create_label();
                    B::emit_cbz_to_label(&mut self.buf, machine_gp(2), skip);
                    B::emit_load_gp(
                        &mut self.buf,
                        machine_gp(27),
                        machine_gp(1),
                        0,
                        MachineWordSize::W64,
                    );
                    B::emit_store_gp_to_frame(
                        &mut self.buf,
                        machine_gp(27),
                        self.frame.slot_access(return_slot),
                    );
                    B::bind_label(&mut self.buf, skip);
                }
                ResumeInjectTarget::ReturnReg => {
                    let skip = self.buf.create_label();
                    B::emit_cbz_to_label(&mut self.buf, machine_gp(2), skip);
                    B::emit_load_gp(
                        &mut self.buf,
                        machine_gp(0),
                        machine_gp(1),
                        0,
                        MachineWordSize::W64,
                    );
                    B::bind_label(&mut self.buf, skip);
                }
            }

            B::emit_branch_to_label(&mut self.buf, stub.continue_label);
        }
    }

    fn lower_block(&mut self, block_idx: usize) {
        let block = &self.func.blocks[block_idx];
        let mut active_prompts = self.block_meta[block_idx].active_prompts.clone();

        // Bind label
        B::bind_label(&mut self.buf, self.block_meta[block_idx].label);

        // Handle block params: entry block uses calling convention,
        // other blocks delegate to the register allocator.
        if block_idx == 0 {
            for (slot, &(val, ty)) in block.params.iter().enumerate() {
                self.read_entry_arg_into_value(val, ty, slot);
            }
        } else {
            self.regs
                .enter_block::<B>(block_idx, self.func, &mut self.buf, &mut self.frame);
        }

        for (inst_idx, inst_node) in block.insts.iter().enumerate() {
            self.lower_inst(block_idx, inst_idx, inst_node, &mut active_prompts);
        }

        // Lower terminator
        self.lower_terminator(block_idx, &active_prompts);
    }

    fn lower_inst(
        &mut self,
        block_idx: usize,
        inst_idx: usize,
        inst_node: &InstNode,
        active_prompts: &mut Vec<PromptId>,
    ) {
        let result_val = inst_node.value;

        match &inst_node.inst {
            Inst::Iconst(ty, imm) => {
                let val = result_val.unwrap();
                let r = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                let imm_val = match ty {
                    Type::I8 => (*imm as u8) as u64,
                    Type::I32 => (*imm as u32) as u64,
                    _ => *imm as u64,
                };
                B::emit_mov_imm(&mut self.buf, machine_gp(r), imm_val);
                self.assign_gp_value(val, r);
            }

            Inst::F64Const(f) => {
                let val = result_val.unwrap();
                let bits = f.to_bits();
                let fr = self.regs.alloc_fp::<B>(&mut self.buf, &mut self.frame);
                B::emit_f64_const(&mut self.buf, machine_fp(fr), bits);
                self.assign_fp_value(val, fr);
            }

            Inst::GcLiteral(lit) => {
                let val = result_val.unwrap();
                let pool_base = self
                    .literal_pool_base
                    .expect("Inst::GcLiteral requires a literal_pool_base; new_module/extend should have provided one");
                let r = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                // Load pool base into x27 (scratch — same convention used by
                // call-table-indirect call lowering).
                B::emit_mov_imm(&mut self.buf, machine_gp(27), pool_base);
                let offset = (lit.index() * 8) as i32;
                B::emit_load_gp(
                    &mut self.buf,
                    machine_gp(r),
                    machine_gp(27),
                    offset,
                    MachineWordSize::W64,
                );
                self.assign_gp_value(val, r);
            }

            Inst::Add(a, b) => self.lower_gp_binop(result_val.unwrap(), *a, *b, BinOp::Add),
            Inst::Sub(a, b) => self.lower_gp_binop(result_val.unwrap(), *a, *b, BinOp::Sub),
            Inst::Mul(a, b) => self.lower_gp_binop(result_val.unwrap(), *a, *b, BinOp::Mul),
            Inst::SDiv(a, b) => self.lower_gp_binop(result_val.unwrap(), *a, *b, BinOp::SDiv),
            Inst::UDiv(a, b) => self.lower_gp_binop(result_val.unwrap(), *a, *b, BinOp::UDiv),
            Inst::And(a, b) => self.lower_gp_binop(result_val.unwrap(), *a, *b, BinOp::And),
            Inst::Or(a, b) => self.lower_gp_binop(result_val.unwrap(), *a, *b, BinOp::Or),
            Inst::Xor(a, b) => self.lower_gp_binop(result_val.unwrap(), *a, *b, BinOp::Xor),
            Inst::Shl(a, b) => self.lower_gp_binop(result_val.unwrap(), *a, *b, BinOp::Shl),
            Inst::LShr(a, b) => self.lower_gp_binop(result_val.unwrap(), *a, *b, BinOp::LShr),
            Inst::AShr(a, b) => self.lower_gp_binop(result_val.unwrap(), *a, *b, BinOp::AShr),

            Inst::Neg(a) => {
                let val = result_val.unwrap();
                let ra = self
                    .regs
                    .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, *a);
                let ty = self.regs.value_type(*a);
                self.regs.dec_use(*a);
                let rd_idx = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                B::emit_gp_neg(
                    &mut self.buf,
                    machine_gp(rd_idx),
                    machine_gp(ra),
                    type_to_machine_word_size(ty),
                );
                self.assign_gp_value(val, rd_idx);
            }

            Inst::Not(a) => {
                let val = result_val.unwrap();
                let ra = self
                    .regs
                    .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, *a);
                let ty = self.regs.value_type(*a);
                self.regs.dec_use(*a);
                let rd_idx = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                B::emit_gp_not(
                    &mut self.buf,
                    machine_gp(rd_idx),
                    machine_gp(ra),
                    type_to_machine_word_size(ty),
                );
                self.assign_gp_value(val, rd_idx);
            }

            Inst::FAdd(a, b) => self.lower_fp_binop(result_val.unwrap(), *a, *b, FpBinOp::Add),
            Inst::FSub(a, b) => self.lower_fp_binop(result_val.unwrap(), *a, *b, FpBinOp::Sub),
            Inst::FMul(a, b) => self.lower_fp_binop(result_val.unwrap(), *a, *b, FpBinOp::Mul),
            Inst::FDiv(a, b) => self.lower_fp_binop(result_val.unwrap(), *a, *b, FpBinOp::Div),

            Inst::FNeg(a) => {
                let val = result_val.unwrap();
                let ra = self
                    .regs
                    .ensure_in_fp_reg::<B>(&mut self.buf, &mut self.frame, *a);
                self.regs.dec_use(*a);
                let rd_idx = self.regs.alloc_fp::<B>(&mut self.buf, &mut self.frame);
                B::emit_fp_neg(&mut self.buf, machine_fp(rd_idx), machine_fp(ra));
                self.assign_fp_value(val, rd_idx);
            }

            Inst::Icmp(op, a, b) => {
                let val = result_val.unwrap();
                let ra = self
                    .regs
                    .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, *a);
                let rb = self
                    .regs
                    .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, *b);
                let ty = self.regs.value_type(*a);
                self.regs.dec_use(*a);
                self.regs.dec_use(*b);
                let size = type_to_machine_word_size(ty);
                let rd_idx = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                B::emit_icmp_set(
                    &mut self.buf,
                    *op,
                    machine_gp(rd_idx),
                    machine_gp(ra),
                    machine_gp(rb),
                    size,
                );
                self.assign_gp_value(val, rd_idx);
            }

            Inst::Fcmp(op, a, b) => {
                let val = result_val.unwrap();
                let ra = self
                    .regs
                    .ensure_in_fp_reg::<B>(&mut self.buf, &mut self.frame, *a);
                let rb = self
                    .regs
                    .ensure_in_fp_reg::<B>(&mut self.buf, &mut self.frame, *b);
                self.regs.dec_use(*a);
                self.regs.dec_use(*b);
                let rd_idx = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                B::emit_fcmp_set(
                    &mut self.buf,
                    *op,
                    machine_gp(rd_idx),
                    machine_fp(ra),
                    machine_fp(rb),
                );
                self.assign_gp_value(val, rd_idx);
            }

            Inst::Select(cond, t, f) => {
                let val = result_val.unwrap();
                let rc = self
                    .regs
                    .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, *cond);
                let ty = self.regs.value_type(*t);

                if is_float_type(ty) {
                    let rt = self
                        .regs
                        .ensure_in_fp_reg::<B>(&mut self.buf, &mut self.frame, *t);
                    let rf = self
                        .regs
                        .ensure_in_fp_reg::<B>(&mut self.buf, &mut self.frame, *f);
                    self.regs.dec_use(*cond);
                    self.regs.dec_use(*t);
                    self.regs.dec_use(*f);
                    let rd_idx = self.regs.alloc_fp::<B>(&mut self.buf, &mut self.frame);
                    B::emit_fp_select(
                        &mut self.buf,
                        machine_gp(rc),
                        machine_fp(rd_idx),
                        machine_fp(rt),
                        machine_fp(rf),
                    );
                    self.assign_fp_value(val, rd_idx);
                } else {
                    let rt = self
                        .regs
                        .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, *t);
                    let rf = self
                        .regs
                        .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, *f);
                    let size = type_to_machine_word_size(ty);
                    self.regs.dec_use(*cond);
                    self.regs.dec_use(*t);
                    self.regs.dec_use(*f);
                    let rd_idx = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                    B::emit_gp_select(
                        &mut self.buf,
                        machine_gp(rd_idx),
                        machine_gp(rc),
                        machine_gp(rt),
                        machine_gp(rf),
                        size,
                    );
                    self.assign_gp_value(val, rd_idx);
                }
            }

            Inst::OverflowCheck(op, a, b) => {
                let val = result_val.unwrap();
                let ra = self
                    .regs
                    .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, *a);
                let rb = self
                    .regs
                    .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, *b);
                let ty = self.regs.value_type(*a);
                if let Some(reg) = self.try_lower_overflow_check_inline(*op, ty, ra, rb) {
                    self.regs.dec_use(*a);
                    self.regs.dec_use(*b);
                    self.assign_gp_value(val, reg);
                } else {
                    self.regs
                        .spill_caller_saved::<B>(&mut self.buf, &mut self.frame);
                    B::emit_gp_move(&mut self.buf, machine_gp(26), machine_gp(ra));
                    B::emit_gp_move(&mut self.buf, machine_gp(27), machine_gp(rb));
                    B::emit_mov_imm(&mut self.buf, B::call_arg_gp(0), overflow_op_code(*op));
                    B::emit_mov_imm(&mut self.buf, B::call_arg_gp(1), overflow_type_code(ty));
                    B::emit_gp_move(&mut self.buf, B::call_arg_gp(2), machine_gp(26));
                    B::emit_gp_move(&mut self.buf, B::call_arg_gp(3), machine_gp(27));
                    B::emit_mov_imm(
                        &mut self.buf,
                        machine_gp(28),
                        jit_overflow_check_helper as usize as u64,
                    );
                    B::emit_call_reg(&mut self.buf, machine_gp(28));
                    self.regs.dec_use(*a);
                    self.regs.dec_use(*b);
                    self.regs.clear_regs();
                    self.assign_gp_value(val, 0);
                }
            }

            Inst::Sext(a, target_ty) => {
                let val = result_val.unwrap();
                let ra = self
                    .regs
                    .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, *a);
                let src_ty = self.regs.value_type(*a);
                self.regs.dec_use(*a);
                let rd_idx = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                B::emit_sign_extend(
                    &mut self.buf,
                    machine_gp(rd_idx),
                    machine_gp(ra),
                    src_ty,
                    *target_ty,
                );
                self.assign_gp_value(val, rd_idx);
            }

            Inst::Zext(a, target_ty) => {
                let val = result_val.unwrap();
                let ra = self
                    .regs
                    .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, *a);
                let src_ty = self.regs.value_type(*a);
                self.regs.dec_use(*a);
                let rd_idx = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                B::emit_zero_extend(
                    &mut self.buf,
                    machine_gp(rd_idx),
                    machine_gp(ra),
                    src_ty,
                    *target_ty,
                );
                self.assign_gp_value(val, rd_idx);
            }

            Inst::Trunc(a, target_ty) => {
                let val = result_val.unwrap();
                let ra = self
                    .regs
                    .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, *a);
                self.regs.dec_use(*a);
                let rd_idx = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                B::emit_trunc(
                    &mut self.buf,
                    machine_gp(rd_idx),
                    machine_gp(ra),
                    *target_ty,
                );
                self.assign_gp_value(val, rd_idx);
            }

            Inst::IntToFloat(a) => {
                let val = result_val.unwrap();
                let ra = self
                    .regs
                    .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, *a);
                let src_ty = self.regs.value_type(*a);
                self.regs.dec_use(*a);
                let rd_idx = self.regs.alloc_fp::<B>(&mut self.buf, &mut self.frame);
                B::emit_int_to_float(&mut self.buf, machine_fp(rd_idx), machine_gp(ra), src_ty);
                self.assign_fp_value(val, rd_idx);
            }

            Inst::FloatToInt(a) => {
                let val = result_val.unwrap();
                let ra = self
                    .regs
                    .ensure_in_fp_reg::<B>(&mut self.buf, &mut self.frame, *a);
                self.regs.dec_use(*a);
                let rd_idx = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                B::emit_float_to_int(&mut self.buf, machine_gp(rd_idx), machine_fp(ra));
                self.assign_gp_value(val, rd_idx);
            }

            Inst::Bitcast(a, _target_ty) => {
                let val = result_val.unwrap();
                let src_ty = self.regs.value_type(*a);
                let dst_ty = self.regs.value_type(val);

                if is_float_type(src_ty) && !is_float_type(dst_ty) {
                    // FP -> GP: FMOV Xd, Dn
                    let ra = self
                        .regs
                        .ensure_in_fp_reg::<B>(&mut self.buf, &mut self.frame, *a);
                    self.regs.dec_use(*a);
                    let rd_idx = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                    B::emit_fp_to_gp_move(&mut self.buf, machine_gp(rd_idx), machine_fp(ra));
                    self.assign_gp_value(val, rd_idx);
                } else if !is_float_type(src_ty) && is_float_type(dst_ty) {
                    // GP -> FP: FMOV Dd, Xn
                    let ra = self
                        .regs
                        .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, *a);
                    self.regs.dec_use(*a);
                    let rd_idx = self.regs.alloc_fp::<B>(&mut self.buf, &mut self.frame);
                    B::emit_gp_to_fp_move(&mut self.buf, machine_fp(rd_idx), machine_gp(ra));
                    self.assign_fp_value(val, rd_idx);
                } else {
                    // Same class: just rename
                    if is_float_type(src_ty) {
                        let ra =
                            self.regs
                                .ensure_in_fp_reg::<B>(&mut self.buf, &mut self.frame, *a);
                        self.regs.dec_use(*a);
                        self.assign_fp_value(val, ra);
                    } else {
                        let ra =
                            self.regs
                                .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, *a);
                        self.regs.dec_use(*a);
                        self.assign_gp_value(val, ra);
                    }
                }
            }

            Inst::StackAddr(stack_slot) => {
                let val = result_val.unwrap();
                // Look up the pre-allocated frame offset for this stack slot.
                let slot_offset = self.stack_slot_offsets[stack_slot.index()];
                let slot = self.frame.slot_access(slot_offset);
                let rd_idx = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                B::emit_lea_frame_slot(&mut self.buf, machine_gp(rd_idx), slot);
                self.assign_gp_value(val, rd_idx);
            }

            Inst::Load(ty, addr, offset) => {
                let val = result_val.unwrap();
                let ra = self
                    .regs
                    .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, *addr);
                self.regs.dec_use(*addr);

                if is_float_type(*ty) {
                    let rd_idx = self.regs.alloc_fp::<B>(&mut self.buf, &mut self.frame);
                    B::emit_load_fp(&mut self.buf, machine_fp(rd_idx), machine_gp(ra), *offset);
                    self.assign_fp_value(val, rd_idx);
                } else {
                    let rd_idx = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                    let size = type_to_machine_word_size(*ty);
                    B::emit_load_gp(
                        &mut self.buf,
                        machine_gp(rd_idx),
                        machine_gp(ra),
                        *offset,
                        size,
                    );
                    self.assign_gp_value(val, rd_idx);
                }
            }

            Inst::Store(val_to_store, addr, offset) => {
                let val_ty = self.regs.value_type(*val_to_store);

                // Load the value FIRST, then the address. This ensures the
                // address register isn't evicted when loading the value.
                if is_float_type(val_ty) {
                    let rv = self.regs.ensure_in_fp_reg::<B>(
                        &mut self.buf,
                        &mut self.frame,
                        *val_to_store,
                    );
                    let ra = self
                        .regs
                        .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, *addr);
                    B::emit_store_fp(&mut self.buf, machine_fp(rv), machine_gp(ra), *offset);
                } else {
                    let rv = self.regs.ensure_in_gp_reg::<B>(
                        &mut self.buf,
                        &mut self.frame,
                        *val_to_store,
                    );
                    let ra = self
                        .regs
                        .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, *addr);
                    let size = type_to_machine_word_size(val_ty);
                    B::emit_store_gp(&mut self.buf, machine_gp(rv), machine_gp(ra), *offset, size);
                }
                self.regs.dec_use(*val_to_store);
                self.regs.dec_use(*addr);
            }

            Inst::Call(func_ref, args) => {
                self.lower_call(
                    *func_ref,
                    args,
                    result_val,
                    block_idx,
                    inst_idx,
                    active_prompts,
                );
            }

            Inst::CallIndirect(callee, args, _ret_ty) => {
                self.lower_call_indirect(*callee, args, result_val);
            }

            // ── Tagged value operations (TagScheme-generic) ─────────
            //
            // Uses the selected ValueLayout constants to emit the correct bit
            // manipulation for any tagging scheme (NanBox, LowBit, etc).
            Inst::Payload(a) => {
                // Layout::extract_payload(bits)
                let val = result_val.unwrap();
                let ra = self
                    .regs
                    .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, *a);
                self.regs.dec_use(*a);
                let rd_idx = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                B::emit_extract_payload(
                    &mut self.buf,
                    machine_gp(rd_idx),
                    machine_gp(ra),
                    Cfg::Layout::HAS_UNBOXED_FLOAT,
                    Cfg::Layout::PAYLOAD_BITS as u8,
                );
                self.assign_gp_value(val, rd_idx);
            }

            Inst::IsTag(a, tag) => {
                // Layout::has_tag(bits, tag)
                let val = result_val.unwrap();
                let ra = self
                    .regs
                    .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, *a);
                self.regs.dec_use(*a);
                let rd_idx = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                let expected = if Cfg::Layout::HAS_UNBOXED_FLOAT {
                    Cfg::Layout::encode_tagged(*tag, 0) >> Cfg::Layout::PAYLOAD_BITS
                } else {
                    *tag as u64
                };
                B::emit_is_tag(
                    &mut self.buf,
                    machine_gp(rd_idx),
                    machine_gp(ra),
                    Cfg::Layout::HAS_UNBOXED_FLOAT,
                    Cfg::Layout::PAYLOAD_BITS as u8,
                    Cfg::Layout::TAG_COUNT as u64 - 1,
                    expected,
                );
                self.assign_gp_value(val, rd_idx);
            }

            Inst::MakeTagged(tag, payload) => {
                // Layout::encode_tagged(tag, payload)
                let val = result_val.unwrap();
                let ra = self
                    .regs
                    .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, *payload);
                self.regs.dec_use(*payload);
                let rd_idx = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                B::emit_make_tagged(
                    &mut self.buf,
                    machine_gp(rd_idx),
                    machine_gp(ra),
                    Cfg::Layout::HAS_UNBOXED_FLOAT,
                    Cfg::Layout::PAYLOAD_BITS as u8,
                    Cfg::Layout::encode_tagged(*tag, 0),
                    *tag as u64,
                );
                self.assign_gp_value(val, rd_idx);
            }

            Inst::TagOf(a) => {
                // Extract tag from bits
                let val = result_val.unwrap();
                let ra = self
                    .regs
                    .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, *a);
                self.regs.dec_use(*a);
                let rd_idx = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                B::emit_tag_of(
                    &mut self.buf,
                    machine_gp(rd_idx),
                    machine_gp(ra),
                    Cfg::Layout::HAS_UNBOXED_FLOAT,
                    Cfg::Layout::PAYLOAD_BITS as u8,
                    Cfg::Layout::TAG_COUNT as u64 - 1,
                );
                self.assign_gp_value(val, rd_idx);
            }

            Inst::Safepoint(live) => {
                self.emit_safepoint(live);
                // Dec uses for live values
                for v in live.iter() {
                    self.regs.dec_use(*v);
                }
            }

            Inst::InvokeDynamic { .. } => {
                // The streaming lowerer is deprecated. InvokeDynamic is handled
                // by the batch lowerer only. This arm exists for exhaustiveness.
                panic!("InvokeDynamic requires the batch lowerer");
            }

            Inst::Guard(cond, deopt_id, live) => {
                let resume_point = self.func.deopt_info[deopt_id.index()].resume_point;
                let cond_reg =
                    self.regs
                        .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, *cond);
                let continue_label = self.buf.create_label();
                B::emit_cbnz_to_label(&mut self.buf, machine_gp(cond_reg), continue_label);
                self.emit_deopt_return(*deopt_id, resume_point, live);
                B::bind_label(&mut self.buf, continue_label);
                self.regs.dec_use(*cond);
                for &value in live {
                    self.regs.dec_use(value);
                }
            }

            Inst::CaptureSlice(prompt, live) => {
                let return_dest = result_val
                    .expect("capture_slice must define a destination value for later resume");
                let frame_values: Vec<Value> = (0..self.func.value_types.len())
                    .map(Value::from_index)
                    .collect();
                let record_idx = self.push_frame_reify_record(
                    FrameReifyKind::CaptureSlice,
                    Some(*prompt),
                    active_prompts,
                    &frame_values,
                    &[],
                    Some(return_dest.index()),
                    FrameResumePoint {
                        func_idx: self.current_func_idx,
                        block_idx,
                        inst_idx,
                    },
                );
                self.regs
                    .spill_all_live::<B>(&mut self.buf, &mut self.frame);
                self.emit_write_full_frame_values();
                self.emit_set_outcome_kind(JitOutcomeKind::CaptureSlice);
                B::emit_mov_imm(&mut self.buf, machine_gp(2), record_idx as u64);
                B::emit_mov_imm(&mut self.buf, machine_gp(3), self.current_func_idx as u64);
                self.emit_return_current_outcome();
                let continue_label = self.buf.create_label();
                B::bind_label(&mut self.buf, continue_label);
                let slot_offsets = self.frame_reify_records[record_idx]
                    .value_indices
                    .iter()
                    .map(|&value_idx| self.regs.value_spill_slot(Value::from_index(value_idx)))
                    .collect();
                let return_slot = self.frame.alloc_root_slot();
                self.assign_spill_value(return_dest, return_slot);
                self.pending_resume_stubs.push(PendingResumeStub {
                    target: ResumeStubTarget::FrameReifyRecord(record_idx),
                    continue_label,
                    resume_inject: ResumeInjectTarget::FrameSlot(return_slot),
                    slot_offsets,
                });
                for &value in live {
                    self.regs.dec_use(value);
                }
            }

            Inst::PushPrompt(prompt, _handler) => {
                active_prompts.push(*prompt);
            }

            Inst::PopPrompt(prompt) => {
                let popped = active_prompts
                    .pop()
                    .unwrap_or_else(|| panic!("pop_prompt without active prompt in JIT lowering"));
                assert_eq!(
                    popped, *prompt,
                    "mismatched prompt stack in JIT lowering: expected {:?}, got {:?}",
                    popped, prompt
                );
            }

            Inst::CloneSlice(slice) => {
                let clone_dest = result_val.expect("clone_slice must define a destination value");
                let frame_values: Vec<Value> = (0..self.func.value_types.len())
                    .map(Value::from_index)
                    .collect();
                let record_idx = self.push_frame_reify_record(
                    FrameReifyKind::CloneSlice,
                    None,
                    active_prompts,
                    &frame_values,
                    &[*slice],
                    Some(clone_dest.index()),
                    FrameResumePoint {
                        func_idx: self.current_func_idx,
                        block_idx,
                        inst_idx,
                    },
                );
                self.regs
                    .spill_all_live::<B>(&mut self.buf, &mut self.frame);
                self.emit_write_full_frame_values();
                self.emit_set_outcome_kind(JitOutcomeKind::CloneSlice);
                B::emit_mov_imm(&mut self.buf, machine_gp(2), record_idx as u64);
                B::emit_mov_imm(&mut self.buf, machine_gp(3), self.current_func_idx as u64);
                self.emit_return_current_outcome();
                let continue_label = self.buf.create_label();
                B::bind_label(&mut self.buf, continue_label);
                let slot_offsets = self.frame_reify_records[record_idx]
                    .value_indices
                    .iter()
                    .map(|&value_idx| self.regs.value_spill_slot(Value::from_index(value_idx)))
                    .collect();
                let return_slot = self.frame.alloc_root_slot();
                self.assign_spill_value(clone_dest, return_slot);
                self.pending_resume_stubs.push(PendingResumeStub {
                    target: ResumeStubTarget::FrameReifyRecord(record_idx),
                    continue_label,
                    resume_inject: ResumeInjectTarget::FrameSlot(return_slot),
                    slot_offsets,
                });
                self.regs.dec_use(*slice);
            }
        }
    }

    fn lower_gp_binop(&mut self, val: Value, a: Value, b: Value, op: BinOp) {
        let ra = self
            .regs
            .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, a);
        let rb = self
            .regs
            .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, b);
        let ty = self.regs.value_type(a);
        self.regs.dec_use(a);
        self.regs.dec_use(b);

        let rd_idx = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
        let size = type_to_machine_word_size(ty);
        let backend_op = match op {
            BinOp::Add => MachineGpBinOp::Add,
            BinOp::Sub => MachineGpBinOp::Sub,
            BinOp::Mul => MachineGpBinOp::Mul,
            BinOp::SDiv => MachineGpBinOp::SDiv,
            BinOp::UDiv => MachineGpBinOp::UDiv,
            BinOp::And => MachineGpBinOp::And,
            BinOp::Or => MachineGpBinOp::Or,
            BinOp::Xor => MachineGpBinOp::Xor,
            BinOp::Shl => MachineGpBinOp::Shl,
            BinOp::LShr => MachineGpBinOp::LShr,
            BinOp::AShr => MachineGpBinOp::AShr,
        };

        B::emit_gp_binop(
            &mut self.buf,
            backend_op,
            machine_gp(rd_idx),
            machine_gp(ra),
            machine_gp(rb),
            size,
        );

        self.assign_gp_value(val, rd_idx);
    }

    fn lower_fp_binop(&mut self, val: Value, a: Value, b: Value, op: FpBinOp) {
        let ra = self
            .regs
            .ensure_in_fp_reg::<B>(&mut self.buf, &mut self.frame, a);
        let rb = self
            .regs
            .ensure_in_fp_reg::<B>(&mut self.buf, &mut self.frame, b);
        self.regs.dec_use(a);
        self.regs.dec_use(b);

        let rd_idx = self.regs.alloc_fp::<B>(&mut self.buf, &mut self.frame);
        let backend_op = match op {
            FpBinOp::Add => MachineFpBinOp::Add,
            FpBinOp::Sub => MachineFpBinOp::Sub,
            FpBinOp::Mul => MachineFpBinOp::Mul,
            FpBinOp::Div => MachineFpBinOp::Div,
        };

        B::emit_fp_binop(
            &mut self.buf,
            backend_op,
            machine_fp(rd_idx),
            machine_fp(ra),
            machine_fp(rb),
        );

        self.assign_fp_value(val, rd_idx);
    }

    fn try_lower_overflow_check_inline(
        &mut self,
        op: OverflowOp,
        ty: Type,
        ra: u8,
        rb: u8,
    ) -> Option<u8> {
        let size = type_to_machine_word_size(ty);
        match op {
            OverflowOp::UAdd => {
                let sum = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                B::emit_gp_binop(
                    &mut self.buf,
                    MachineGpBinOp::Add,
                    machine_gp(sum),
                    machine_gp(ra),
                    machine_gp(rb),
                    size,
                );
                let out = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                B::emit_icmp_set(
                    &mut self.buf,
                    CmpOp::Ult,
                    machine_gp(out),
                    machine_gp(sum),
                    machine_gp(ra),
                    size,
                );
                Some(out)
            }
            OverflowOp::USub => {
                let out = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                B::emit_icmp_set(
                    &mut self.buf,
                    CmpOp::Ult,
                    machine_gp(out),
                    machine_gp(ra),
                    machine_gp(rb),
                    size,
                );
                Some(out)
            }
            OverflowOp::SAdd if ty == Type::I64 => {
                let sum = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                B::emit_gp_binop(
                    &mut self.buf,
                    MachineGpBinOp::Add,
                    machine_gp(sum),
                    machine_gp(ra),
                    machine_gp(rb),
                    size,
                );
                let x1 = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                B::emit_gp_binop(
                    &mut self.buf,
                    MachineGpBinOp::Xor,
                    machine_gp(x1),
                    machine_gp(sum),
                    machine_gp(ra),
                    MachineWordSize::W64,
                );
                let x2 = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                B::emit_gp_binop(
                    &mut self.buf,
                    MachineGpBinOp::Xor,
                    machine_gp(x2),
                    machine_gp(sum),
                    machine_gp(rb),
                    MachineWordSize::W64,
                );
                let x3 = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                B::emit_gp_binop(
                    &mut self.buf,
                    MachineGpBinOp::And,
                    machine_gp(x3),
                    machine_gp(x1),
                    machine_gp(x2),
                    MachineWordSize::W64,
                );
                let shift = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                B::emit_mov_imm(&mut self.buf, machine_gp(27), 63);
                B::emit_gp_binop(
                    &mut self.buf,
                    MachineGpBinOp::AShr,
                    machine_gp(shift),
                    machine_gp(x3),
                    machine_gp(27),
                    MachineWordSize::W64,
                );
                let out = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                B::emit_mov_imm(&mut self.buf, machine_gp(27), 1);
                B::emit_gp_binop(
                    &mut self.buf,
                    MachineGpBinOp::And,
                    machine_gp(out),
                    machine_gp(shift),
                    machine_gp(27),
                    MachineWordSize::W64,
                );
                Some(out)
            }
            OverflowOp::SSub if ty == Type::I64 => {
                let diff = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                B::emit_gp_binop(
                    &mut self.buf,
                    MachineGpBinOp::Sub,
                    machine_gp(diff),
                    machine_gp(ra),
                    machine_gp(rb),
                    size,
                );
                let x1 = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                B::emit_gp_binop(
                    &mut self.buf,
                    MachineGpBinOp::Xor,
                    machine_gp(x1),
                    machine_gp(ra),
                    machine_gp(rb),
                    MachineWordSize::W64,
                );
                let x2 = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                B::emit_gp_binop(
                    &mut self.buf,
                    MachineGpBinOp::Xor,
                    machine_gp(x2),
                    machine_gp(diff),
                    machine_gp(ra),
                    MachineWordSize::W64,
                );
                let x3 = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                B::emit_gp_binop(
                    &mut self.buf,
                    MachineGpBinOp::And,
                    machine_gp(x3),
                    machine_gp(x1),
                    machine_gp(x2),
                    MachineWordSize::W64,
                );
                let shift = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                B::emit_mov_imm(&mut self.buf, machine_gp(27), 63);
                B::emit_gp_binop(
                    &mut self.buf,
                    MachineGpBinOp::AShr,
                    machine_gp(shift),
                    machine_gp(x3),
                    machine_gp(27),
                    MachineWordSize::W64,
                );
                let out = self.regs.alloc_gp::<B>(&mut self.buf, &mut self.frame);
                B::emit_mov_imm(&mut self.buf, machine_gp(27), 1);
                B::emit_gp_binop(
                    &mut self.buf,
                    MachineGpBinOp::And,
                    machine_gp(out),
                    machine_gp(shift),
                    machine_gp(27),
                    MachineWordSize::W64,
                );
                Some(out)
            }
            _ => None,
        }
    }

    fn write_value_to_frame_slot(&mut self, arg: Value, ty: Type, slot: FrameSlotAccess) {
        if is_float_type(ty) {
            let src = self
                .regs
                .ensure_in_fp_reg::<B>(&mut self.buf, &mut self.frame, arg);
            B::emit_store_fp_to_frame(&mut self.buf, machine_fp(src), slot);
        } else {
            let src = self
                .regs
                .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, arg);
            B::emit_store_gp_to_frame(&mut self.buf, machine_gp(src), slot);
        }
    }

    fn write_value_to_assignment_target(&mut self, arg: Value, ty: Type, target: AssignmentTarget) {
        let machine_location = match target {
            AssignmentTarget::CallArg(CallArgLocation::Register(reg_idx)) => {
                MachineLocation::Reg(B::call_arg_gp(reg_idx))
            }
            other => other.as_machine_location(),
        };
        match machine_location {
            MachineLocation::Reg(dst) => {
                if matches!(
                    target,
                    AssignmentTarget::CallArg(CallArgLocation::Register(_))
                ) {
                    if let Some(slot) = self.regs.value_spill_slot(arg) {
                        B::emit_load_gp_from_frame(
                            &mut self.buf,
                            dst,
                            self.frame.slot_access(slot),
                        );
                        self.regs.mark_gp_occupied(dst.index, arg);
                        return;
                    }
                }
                if is_float_type(ty) {
                    let src = self
                        .regs
                        .ensure_in_fp_reg::<B>(&mut self.buf, &mut self.frame, arg);
                    B::emit_fp_to_gp_move(&mut self.buf, dst, machine_fp(src));
                } else {
                    let src = self
                        .regs
                        .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, arg);
                    if src != dst.index {
                        B::emit_gp_move(&mut self.buf, dst, machine_gp(src));
                        self.regs.clear_gp_occupied(src, arg);
                        self.regs.set_value_loc(arg, ValueLoc::GpReg(dst.index));
                        // Mark the destination register as occupied so that
                        // subsequent alloc_gp calls don't clobber it.  This is
                        // critical when the same SSA value appears in multiple
                        // argument slots (e.g. after GVN merges two iconst(0)).
                        self.regs.mark_gp_occupied(dst.index, arg);
                    }
                }
            }
            MachineLocation::StackArg(stack_offset) => {
                if is_float_type(ty) {
                    let src = self
                        .regs
                        .ensure_in_fp_reg::<B>(&mut self.buf, &mut self.frame, arg);
                    B::emit_fp_to_gp_move(&mut self.buf, machine_gp(27), machine_fp(src));
                    B::emit_store_gp_to_stack_arg(&mut self.buf, machine_gp(27), stack_offset);
                } else {
                    let src = self
                        .regs
                        .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, arg);
                    if src != 27 {
                        B::emit_gp_move(&mut self.buf, machine_gp(27), machine_gp(src));
                    }
                    B::emit_store_gp_to_stack_arg(&mut self.buf, machine_gp(27), stack_offset);
                }
            }
            MachineLocation::FrameSlot(slot) => {
                self.write_value_to_frame_slot(arg, ty, slot);
            }
        }
    }

    fn emit_assignments(&mut self, assignments: &[ValueAssignment]) {
        for assignment in assignments {
            self.write_value_to_assignment_target(
                assignment.value,
                assignment.ty,
                assignment.target,
            );
        }
    }

    fn read_entry_arg_into_value(&mut self, val: Value, ty: Type, slot: usize) {
        match Cfg::CallingConvention::arg_location(slot) {
            CallArgLocation::Register(reg_idx) => {
                let arg_reg = B::call_arg_gp(reg_idx);
                if is_float_type(ty) {
                    let fp_idx = self.regs.alloc_fp::<B>(&mut self.buf, &mut self.frame);
                    B::emit_gp_to_fp_move(&mut self.buf, machine_fp(fp_idx), arg_reg);
                    self.assign_fp_value(val, fp_idx);
                } else {
                    self.assign_gp_value(val, arg_reg.index);
                }
            }
            CallArgLocation::Stack(incoming_offset) => {
                let slot_offset = self.frame.alloc_root_slot();
                self.assign_spill_value(val, slot_offset);
                if is_float_type(ty) {
                    B::emit_load_incoming_stack_arg(&mut self.buf, machine_gp(27), incoming_offset);
                    B::emit_gp_to_fp_move(&mut self.buf, machine_fp(31), machine_gp(27));
                    B::emit_store_fp_to_frame(
                        &mut self.buf,
                        machine_fp(31),
                        self.frame.slot_access(slot_offset),
                    );
                } else {
                    B::emit_load_incoming_stack_arg(&mut self.buf, machine_gp(27), incoming_offset);
                    B::emit_store_gp_to_frame(
                        &mut self.buf,
                        machine_gp(27),
                        self.frame.slot_access(slot_offset),
                    );
                }
            }
        }
    }

    fn prepare_call_args(&mut self, args: &[Value]) -> i32 {
        // Materialize all live values before assigning ABI registers.  The
        // assignment pass is sequential, so an early argument can overwrite
        // the register holding a later argument unless that later value has a
        // frame home to reload from.
        self.regs
            .spill_all_live::<B>(&mut self.buf, &mut self.frame);

        let outgoing_size = Cfg::CallingConvention::outgoing_stack_size(args.len());
        self.frame.reserve_outgoing_arg_bytes(outgoing_size);
        if outgoing_size > 0 {
            B::emit_stack_adjust(&mut self.buf, outgoing_size);
        }

        let assignments: Vec<ValueAssignment> = args
            .iter()
            .enumerate()
            .map(|(slot, &arg)| ValueAssignment {
                value: arg,
                ty: self.regs.value_type(arg),
                target: AssignmentTarget::CallArg(Cfg::CallingConvention::arg_location(slot)),
            })
            .collect();
        self.emit_assignments(&assignments);
        outgoing_size
    }

    fn finish_call_cleanup(&mut self, args: &[Value], outgoing_size: i32) {
        if outgoing_size > 0 {
            B::emit_stack_adjust(&mut self.buf, -outgoing_size);
        }

        for &arg in args {
            self.regs.dec_use(arg);
        }

        // Values loaded into caller-saved regs during arg setup must have
        // their locs restored to their spill slots (BLR clobbered the regs).
        self.regs.clobber_caller_saved::<B>();
    }

    fn assign_call_result(&mut self, result_val: Option<Value>) {
        if let Some(val) = result_val {
            let ty = self.regs.value_type(val);
            if is_float_type(ty) {
                self.assign_fp_value(val, 0);
            } else {
                self.assign_gp_value(val, 0);
            }
        }
    }

    fn lower_call(
        &mut self,
        func_ref: FuncRef,
        args: &[Value],
        result_val: Option<Value>,
        block_idx: usize,
        inst_idx: usize,
        active_prompts: &[PromptId],
    ) {
        let control_aware = self.direct_call_is_control_aware(func_ref);
        let suspend_record = if control_aware {
            self.regs
                .spill_all_live::<B>(&mut self.buf, &mut self.frame);
            Some(self.push_call_suspend_record(
                active_prompts,
                SuspendCallerResume::FromCall {
                    return_dest: result_val.map(|value| value.index()),
                },
                FrameResumePoint {
                    func_idx: self.current_func_idx,
                    block_idx,
                    inst_idx: inst_idx + 1,
                },
            ))
        } else {
            None
        };

        if let Some(record_ptr) = suspend_record {
            self.emit_push_suspended_frame(record_ptr);
        }

        let outgoing_size = self.prepare_call_args(args);

        // Load function pointer into the backend's call scratch register and call through it.
        let func_idx = func_ref.index();
        if let Some(table_base) = self.call_table_base {
            // Module mode: load from indirect call table.
            B::emit_mov_imm(&mut self.buf, machine_gp(27), table_base);
            let offset = (func_idx * 8) as i32;
            B::emit_load_gp(
                &mut self.buf,
                machine_gp(28),
                machine_gp(27),
                offset,
                MachineWordSize::W64,
            );
        } else if func_idx < self.externs.len() {
            let ptr = self.externs[func_idx] as u64;
            B::emit_mov_imm(&mut self.buf, machine_gp(28), ptr);
        } else {
            // null pointer - will crash, which is better than silently wrong behavior
            B::emit_mov_imm(&mut self.buf, machine_gp(28), 0);
        }
        B::emit_call_reg(&mut self.buf, machine_gp(28));
        // Any GC inside the callee will walk our frame via the FP
        // chain; the ancestor walker looks up a SafepointRecord keyed
        // by the saved LR (== current buffer offset after the BLR).
        self.record_call_return_safepoint();

        if control_aware {
            let continue_label = self.buf.create_label();
            let post_pop_label = self.buf.create_label();
            let expected_kind = if result_val.is_some() {
                JitOutcomeKind::ReturnValue as u64
            } else {
                JitOutcomeKind::ReturnVoid as u64
            };
            B::emit_cmp_gp_imm(&mut self.buf, machine_gp(1), expected_kind);
            B::emit_branch_eq_to_label(&mut self.buf, continue_label);
            self.emit_return_current_outcome();
            B::bind_label(&mut self.buf, continue_label);
            // Save return value (X0) across the pop call — it follows
            // the C ABI and clobbers all caller-saved registers.
            let save_slot = self.frame.alloc_local_slot();
            let save_access = self.frame.slot_access(save_slot);
            B::emit_store_gp_to_frame(&mut self.buf, machine_gp(0), save_access);
            self.emit_pop_suspended_frame();
            B::emit_load_gp_from_frame(&mut self.buf, machine_gp(0), save_access);
            B::bind_label(&mut self.buf, post_pop_label);

            let slot_offsets = self
                .regs
                .value_info_snapshot()
                .iter()
                .map(|(spill_slot, _ty)| *spill_slot)
                .collect::<Vec<_>>();
            let suspend_idx = self.suspend_records.len() - 1;
            self.pending_resume_stubs.push(PendingResumeStub {
                target: ResumeStubTarget::SuspendRecord(suspend_idx),
                continue_label: post_pop_label,
                resume_inject: if result_val.is_some() {
                    ResumeInjectTarget::ReturnReg
                } else {
                    ResumeInjectTarget::None
                },
                slot_offsets,
            });
        }

        self.finish_call_cleanup(args, outgoing_size);
        self.assign_call_result(result_val);
    }

    fn lower_call_indirect(&mut self, callee: Value, args: &[Value], result_val: Option<Value>) {
        // Get callee pointer first
        let callee_reg = self
            .regs
            .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, callee);
        // Move to the backend's call scratch register before spilling.
        B::emit_gp_move(&mut self.buf, machine_gp(28), machine_gp(callee_reg));
        self.regs.dec_use(callee);

        let outgoing_size = self.prepare_call_args(args);

        B::emit_call_reg(&mut self.buf, machine_gp(28));
        self.record_call_return_safepoint();

        let continue_label = self.buf.create_label();
        let expected_kind = if result_val.is_some() {
            JitOutcomeKind::ReturnValue as u64
        } else {
            JitOutcomeKind::ReturnVoid as u64
        };
        B::emit_cmp_gp_imm(&mut self.buf, machine_gp(1), expected_kind);
        B::emit_branch_eq_to_label(&mut self.buf, continue_label);
        self.emit_return_current_outcome();
        B::bind_label(&mut self.buf, continue_label);

        self.finish_call_cleanup(args, outgoing_size);
        self.assign_call_result(result_val);
    }

    fn write_invoke_return_to_target(&mut self, target: BlockId) {
        let block_idx = target.index();
        let slots = self.frame.block_param_slots(block_idx);
        if slots.is_empty() {
            return;
        }
        B::emit_store_gp_to_frame(
            &mut self.buf,
            machine_gp(0),
            self.frame.slot_access(slots[0]),
        );
    }

    fn lower_invoke_common(
        &mut self,
        call_args: &[Value],
        outgoing_size: i32,
        normal: BlockId,
        normal_args: &[Value],
        has_ret_param: bool,
        exception: BlockId,
        exception_args: &[Value],
        control_aware: bool,
        suspended_invoke_idx: Option<usize>,
    ) {
        if control_aware {
            let exception_label = self.buf.create_label();
            let normal_label = self.buf.create_label();
            let normal_post_pop_label = self.buf.create_label();
            let exception_post_pop_label = self.buf.create_label();
            B::emit_cmp_gp_imm(
                &mut self.buf,
                machine_gp(1),
                JitOutcomeKind::Exception as u64,
            );
            B::emit_branch_eq_to_label(&mut self.buf, exception_label);
            let expected_kind = if has_ret_param {
                JitOutcomeKind::ReturnValue as u64
            } else {
                JitOutcomeKind::ReturnVoid as u64
            };
            B::emit_cmp_gp_imm(&mut self.buf, machine_gp(1), expected_kind);
            B::emit_branch_eq_to_label(&mut self.buf, normal_label);
            self.emit_return_current_outcome();

            B::bind_label(&mut self.buf, exception_label);
            {
                let save_slot = self.frame.alloc_local_slot();
                let save_access = self.frame.slot_access(save_slot);
                B::emit_store_gp_to_frame(&mut self.buf, machine_gp(0), save_access);
                self.emit_pop_suspended_frame();
                B::emit_load_gp_from_frame(&mut self.buf, machine_gp(0), save_access);
            }
            B::bind_label(&mut self.buf, exception_post_pop_label);
            self.finish_call_cleanup(call_args, outgoing_size);
            self.emit_block_args(exception, exception_args);
            B::emit_branch_to_label(&mut self.buf, self.block_meta[exception.index()].label);

            B::bind_label(&mut self.buf, normal_label);
            {
                let save_slot = self.frame.alloc_local_slot();
                let save_access = self.frame.slot_access(save_slot);
                B::emit_store_gp_to_frame(&mut self.buf, machine_gp(0), save_access);
                self.emit_pop_suspended_frame();
                B::emit_load_gp_from_frame(&mut self.buf, machine_gp(0), save_access);
            }
            B::bind_label(&mut self.buf, normal_post_pop_label);

            if let Some(suspend_idx) = suspended_invoke_idx {
                let slot_offsets = self
                    .regs
                    .value_info_snapshot()
                    .iter()
                    .map(|(spill_slot, _ty)| *spill_slot)
                    .collect::<Vec<_>>();
                self.pending_resume_stubs.push(PendingResumeStub {
                    target: ResumeStubTarget::SuspendRecord(suspend_idx),
                    continue_label: normal_post_pop_label,
                    resume_inject: if has_ret_param {
                        ResumeInjectTarget::ReturnReg
                    } else {
                        ResumeInjectTarget::None
                    },
                    slot_offsets: slot_offsets.clone(),
                });
                self.pending_resume_stubs.push(PendingResumeStub {
                    target: ResumeStubTarget::SuspendRecordException(suspend_idx),
                    continue_label: exception_post_pop_label,
                    resume_inject: ResumeInjectTarget::None,
                    slot_offsets,
                });
            }
        }

        self.finish_call_cleanup(call_args, outgoing_size);
        if has_ret_param {
            self.write_invoke_return_to_target(normal);
        }
        self.store_block_args_to_canonical_with_param_offset(
            normal,
            normal_args,
            usize::from(has_ret_param),
        );
        B::emit_branch_to_label(&mut self.buf, self.block_meta[normal.index()].label);
    }

    fn lower_terminator(&mut self, block_idx: usize, active_prompts: &[PromptId]) {
        let block = &self.func.blocks[block_idx];
        match &block.terminator {
            Terminator::Ret(v) => {
                let ty = self.regs.value_type(*v);
                if is_float_type(ty) {
                    // Return float bits in X0 (call_jit reads X0 as u64)
                    let r = self
                        .regs
                        .ensure_in_fp_reg::<B>(&mut self.buf, &mut self.frame, *v);
                    B::emit_return_fp_bits(&mut self.buf, machine_fp(r));
                } else {
                    let r = self
                        .regs
                        .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, *v);
                    B::emit_return_gp(&mut self.buf, machine_gp(r));
                }
                self.regs.dec_use(*v);
                self.emit_set_outcome_kind(JitOutcomeKind::ReturnValue);
                self.emit_epilogue();
            }

            Terminator::RetVoid => {
                self.emit_set_outcome_kind(JitOutcomeKind::ReturnVoid);
                self.emit_epilogue();
            }

            Terminator::Jump(target, args) => {
                // Spill live-through values before assigning successor params.
                // Linear-scan block args are register-to-register moves, so
                // doing them first can overwrite unrelated live values before
                // they are materialized to their frame homes.
                self.regs
                    .spill_all_live::<B>(&mut self.buf, &mut self.frame);
                self.emit_block_args(*target, args);
                let label = self.block_meta[target.index()].label;
                B::emit_branch_to_label(&mut self.buf, label);
            }

            Terminator::BrIf {
                cond,
                then_block,
                then_args,
                else_block,
                else_args,
            } => {
                let rc = self
                    .regs
                    .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, *cond);
                // Don't dec_use cond yet - keep its register occupied so
                // spill_all_live won't allocate it for something else.
                // spill_all_live will spill it (harmless, just a redundant store).

                // Spill all live values before the conditional branch
                self.regs
                    .spill_all_live::<B>(&mut self.buf, &mut self.frame);

                // Now safe to use rc - spill_all_live only emits STR (reads regs)
                self.regs.dec_use(*cond);

                let else_tramp = self.buf.create_label();
                B::emit_cbz_to_label(&mut self.buf, machine_gp(rc), else_tramp);

                // Save register state before the then path, so we can
                // restore for the else path. At runtime only one path
                // executes, so they must each start from the post-spill state.
                let saved = self.regs.save_state();

                // Then path: store args and branch
                self.store_block_args_for_branch(*then_block, then_args);
                let then_label = self.block_meta[then_block.index()].label;
                B::emit_branch_to_label(&mut self.buf, then_label);

                // Restore register state for else path
                self.regs.restore_state(saved);

                // Else trampoline: store args and branch
                B::bind_label(&mut self.buf, else_tramp);
                self.store_block_args_for_branch(*else_block, else_args);
                let else_label = self.block_meta[else_block.index()].label;
                if else_block.index() != block_idx + 1 {
                    B::emit_branch_to_label(&mut self.buf, else_label);
                }
                // If else is next block, fall through
            }

            Terminator::Switch {
                val,
                cases,
                default_block,
                default_args,
            } => {
                // Spill everything first
                let rv = self
                    .regs
                    .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, *val);
                self.regs
                    .spill_all_live::<B>(&mut self.buf, &mut self.frame);

                // For each case: CMP + B.EQ
                for (case_val, target, args) in cases {
                    B::emit_cmp_gp_imm(&mut self.buf, machine_gp(rv), *case_val as u64);

                    if !args.is_empty() {
                        self.emit_block_args(*target, args);
                    }

                    let label = self.block_meta[target.index()].label;
                    B::emit_branch_eq_to_label(&mut self.buf, label);
                }

                self.regs.dec_use(*val);

                // Default
                self.emit_block_args(*default_block, default_args);
                let label = self.block_meta[default_block.index()].label;
                B::emit_branch_to_label(&mut self.buf, label);
            }

            Terminator::Invoke {
                func,
                args,
                normal,
                normal_args,
                exception,
                exception_args,
            } => {
                let control_aware = self.direct_call_is_control_aware(*func);
                let suspended_invoke_idx = if control_aware {
                    self.regs
                        .spill_all_live::<B>(&mut self.buf, &mut self.frame);
                    let record_ptr = self.push_call_suspend_record(
                        active_prompts,
                        SuspendCallerResume::FromInvoke {
                            normal_block: normal.index(),
                            normal_arg_indices: normal_args
                                .iter()
                                .map(|value| value.index())
                                .collect(),
                            exception_block: exception.index(),
                            exception_arg_indices: exception_args
                                .iter()
                                .map(|value| value.index())
                                .collect(),
                            has_ret_param: !self.func.blocks[normal.index()].params.is_empty(),
                        },
                        FrameResumePoint {
                            func_idx: self.current_func_idx,
                            block_idx,
                            inst_idx: block.insts.len(),
                        },
                    );
                    self.emit_push_suspended_frame(record_ptr);
                    Some(self.suspend_records.len() - 1)
                } else {
                    None
                };
                let outgoing_size = self.prepare_call_args(args);

                let func_idx = func.index();
                if let Some(table_base) = self.call_table_base {
                    B::emit_mov_imm(&mut self.buf, machine_gp(27), table_base);
                    let offset = (func_idx * 8) as i32;
                    B::emit_load_gp(
                        &mut self.buf,
                        machine_gp(28),
                        machine_gp(27),
                        offset,
                        MachineWordSize::W64,
                    );
                } else if func_idx < self.externs.len() {
                    let ptr = self.externs[func_idx] as u64;
                    B::emit_mov_imm(&mut self.buf, machine_gp(28), ptr);
                } else {
                    B::emit_mov_imm(&mut self.buf, machine_gp(28), 0);
                }
                B::emit_call_reg(&mut self.buf, machine_gp(28));

                let has_ret_param = !self.func.blocks[normal.index()].params.is_empty();
                self.lower_invoke_common(
                    args,
                    outgoing_size,
                    *normal,
                    normal_args,
                    has_ret_param,
                    *exception,
                    exception_args,
                    control_aware,
                    suspended_invoke_idx,
                );
            }

            Terminator::InvokeIndirect {
                callee,
                args,
                ret_ty,
                normal,
                normal_args,
                exception,
                exception_args,
            } => {
                let suspended_invoke_idx = {
                    self.regs
                        .spill_all_live::<B>(&mut self.buf, &mut self.frame);
                    let record_ptr = self.push_call_suspend_record(
                        active_prompts,
                        SuspendCallerResume::FromInvoke {
                            normal_block: normal.index(),
                            normal_arg_indices: normal_args
                                .iter()
                                .map(|value| value.index())
                                .collect(),
                            exception_block: exception.index(),
                            exception_arg_indices: exception_args
                                .iter()
                                .map(|value| value.index())
                                .collect(),
                            has_ret_param: ret_ty.is_some(),
                        },
                        FrameResumePoint {
                            func_idx: self.current_func_idx,
                            block_idx,
                            inst_idx: block.insts.len(),
                        },
                    );
                    self.emit_push_suspended_frame(record_ptr);
                    Some(self.suspend_records.len() - 1)
                };

                let outgoing_size = self.prepare_call_args(args);
                if let Some(slot) = self.regs.value_spill_slot(*callee) {
                    B::emit_load_gp_from_frame(
                        &mut self.buf,
                        machine_gp(28),
                        self.frame.slot_access(slot),
                    );
                } else {
                    let callee_reg =
                        self.regs
                            .ensure_in_gp_reg::<B>(&mut self.buf, &mut self.frame, *callee);
                    B::emit_gp_move(&mut self.buf, machine_gp(28), machine_gp(callee_reg));
                }
                self.regs.dec_use(*callee);
                B::emit_call_reg(&mut self.buf, machine_gp(28));

                self.lower_invoke_common(
                    args,
                    outgoing_size,
                    *normal,
                    normal_args,
                    ret_ty.is_some(),
                    *exception,
                    exception_args,
                    true,
                    suspended_invoke_idx,
                );
            }

            Terminator::Unreachable => {
                B::emit_trap(&mut self.buf);
            }

            Terminator::AbortToPrompt { prompt, args } => {
                let record_idx = self.push_frame_reify_record(
                    FrameReifyKind::AbortToPrompt,
                    Some(*prompt),
                    active_prompts,
                    args,
                    args,
                    None,
                    FrameResumePoint {
                        func_idx: self.current_func_idx,
                        block_idx,
                        inst_idx: block.insts.len(),
                    },
                );
                self.emit_frame_reify_outcome(JitOutcomeKind::AbortToPrompt, record_idx, args);
                for &value in args {
                    self.regs.dec_use(value);
                }
            }

            Terminator::ResumeSlice { slice, args, .. } => {
                let mut values = Vec::with_capacity(args.len() + 1);
                values.push(*slice);
                values.extend(args.iter().copied());
                let record_idx = self.push_frame_reify_record(
                    FrameReifyKind::ResumeSlice,
                    None,
                    active_prompts,
                    &values,
                    &values,
                    None,
                    FrameResumePoint {
                        func_idx: self.current_func_idx,
                        block_idx,
                        inst_idx: block.insts.len(),
                    },
                );
                self.emit_frame_reify_outcome(JitOutcomeKind::ResumeSlice, record_idx, &values);
                self.regs.dec_use(*slice);
                for &value in args {
                    self.regs.dec_use(value);
                }
            }

            Terminator::CaptureSlice { .. } => {
                // The JIT does not yet lower the new shift/reset form.
                // Interpreter-only for now.
                panic!("CaptureSlice terminator not supported in JIT backend yet");
            }
        }
    }

    /// Store block args to canonical spill slots for multi-pred blocks,
    /// or set up renaming for single-pred blocks.
    fn emit_block_args(&mut self, target: BlockId, args: &[Value]) {
        self.regs.emit_block_args::<B>(
            target.index(),
            args,
            self.func,
            &mut self.buf,
            &mut self.frame,
        );
    }

    fn store_block_args_for_branch(&mut self, target: BlockId, args: &[Value]) {
        self.regs.emit_block_args::<B>(
            target.index(),
            args,
            self.func,
            &mut self.buf,
            &mut self.frame,
        );
    }

    fn store_block_args_to_canonical_with_param_offset(
        &mut self,
        target: BlockId,
        args: &[Value],
        param_offset: usize,
    ) {
        // This is still needed for Invoke normal/exception args where
        // the first param is the return value (offset by 1).
        let target_idx = target.index();
        let assignments: Vec<ValueAssignment> = args
            .iter()
            .enumerate()
            .map(|(i, &arg)| ValueAssignment {
                value: arg,
                ty: self.regs.value_type(arg),
                target: AssignmentTarget::FrameSlot(
                    self.frame
                        .slot_access(self.frame.block_param_slots(target_idx)[param_offset + i]),
                ),
            })
            .collect();
        self.emit_assignments(&assignments);
        for &arg in args {
            self.regs.dec_use(arg);
        }
    }

    fn collect_live_root_slots(&mut self, live: &[Value]) -> Vec<i32> {
        let mut root_slots: Vec<i32> = live
            .iter()
            .filter_map(|&v| match Cfg::RootTransport::kind() {
                RootTransportKind::FrameScan | RootTransportKind::StackMap => {
                    self.regs.value_spill_slot(v)
                }
                RootTransportKind::ShadowStack => {
                    self.ensure_shadow_value_materialized(v);
                    self.root_transport.shadow_slots_by_value[v.index()]
                }
            })
            .collect();
        root_slots.extend(
            self.func
                .stack_slots
                .iter()
                .enumerate()
                .filter_map(|(idx, slot)| slot.is_gc_root.then_some(self.stack_slot_offsets[idx])),
        );
        root_slots.sort_unstable();
        root_slots.dedup();
        root_slots
    }

    /// After emitting a `BLR` to another JIT function, push a
    /// `SafepointRecord` whose `return_offset` matches the LR the
    /// ancestor-frame walker will see if a GC fires inside the callee.
    ///
    /// The record's `root_slots` covers (a) the spill slots of every
    /// still-live value (spilled into root slots by the regalloc) and
    /// (b) every `is_gc_root = true` stack slot (the user-visible
    /// variables emitted by the frontend's `def_var`). Under the
    /// `SoundRoots` / `SoundTransport` contract nothing else in the
    /// frame can contain a heap reference while the callee is running.
    ///
    /// No-op for `FrameScan` transport, which doesn't consult the
    /// per-PC records (it scans by `root_scan_size`).
    fn record_call_return_safepoint(&mut self) {
        if !matches!(
            Cfg::RootTransport::kind(),
            RootTransportKind::StackMap | RootTransportKind::ShadowStack,
        ) {
            return;
        }
        let return_offset = self.buf.current_offset();
        let code_offset = return_offset;

        // Every still-live regalloc value whose home is a spill slot
        // plus every user-declared `is_gc_root = true` stack slot.
        // `spill_caller_saved` before the BLR has already flushed live
        // values into their spill slots, so reading the spill slot is
        // guaranteed to see a live heap reference (or a non-pointer
        // payload that `PtrPolicy` will filter).
        let mut root_slots: Vec<i32> = (0..self.regs.num_values())
            .filter_map(|i| {
                let v = Value::from_index(i);
                if self.regs.remaining_uses(v) == 0 {
                    return None;
                }
                self.regs.value_spill_slot(v)
            })
            .collect();
        root_slots.extend(
            self.func
                .stack_slots
                .iter()
                .enumerate()
                .filter_map(|(idx, slot)| {
                    slot.is_gc_root.then_some(self.stack_slot_offsets[idx])
                }),
        );
        root_slots.sort_unstable();
        root_slots.dedup();

        self.safepoints.push(SafepointRecord {
            code_offset,
            return_offset,
            root_slots,
        });
    }

    fn emit_safepoint(&mut self, live: &[Value]) {
        match Cfg::RootTransport::kind() {
            RootTransportKind::FrameScan => {
                if let Some(handler) = self.safepoint_handler {
                    self.regs
                        .spill_all_live::<B>(&mut self.buf, &mut self.frame);
                    let frame_size = self.frame.root_scan_size() as u64;
                    B::emit_call_safepoint_handler(&mut self.buf, handler, frame_size);
                    self.regs.clear_regs();
                }
            }
            RootTransportKind::StackMap => {
                self.regs
                    .spill_all_live::<B>(&mut self.buf, &mut self.frame);
                let code_offset = self.buf.current_offset();
                let root_slots = self.collect_live_root_slots(live);
                let safepoint_index = self.safepoints.len() as u64;
                if let Some(handler) = self.safepoint_handler {
                    B::emit_call_safepoint_handler(&mut self.buf, handler, safepoint_index);
                }
                let return_offset = self.buf.current_offset();
                self.safepoints.push(SafepointRecord {
                    code_offset,
                    return_offset,
                    root_slots,
                });
                self.regs.clear_regs();
            }
            RootTransportKind::ShadowStack => {
                self.regs
                    .spill_all_live::<B>(&mut self.buf, &mut self.frame);
                let code_offset = self.buf.current_offset();
                let root_slots = self.collect_live_root_slots(live);
                let safepoint_index = self.safepoints.len() as u64;
                if let Some(handler) = self.safepoint_handler {
                    B::emit_call_safepoint_handler(&mut self.buf, handler, safepoint_index);
                }
                let return_offset = self.buf.current_offset();
                self.safepoints.push(SafepointRecord {
                    code_offset,
                    return_offset,
                    root_slots,
                });
                self.regs.clear_regs();
            }
        }
    }
}

// ─── Helpers ───────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum BinOp {
    Add,
    Sub,
    Mul,
    SDiv,
    UDiv,
    And,
    Or,
    Xor,
    Shl,
    LShr,
    AShr,
}

#[derive(Clone, Copy)]
enum FpBinOp {
    Add,
    Sub,
    Mul,
    Div,
}

fn type_to_machine_word_size(ty: Type) -> MachineWordSize {
    match ty {
        Type::I8 | Type::I32 => MachineWordSize::W32,
        Type::I64 | Type::Ptr | Type::GcPtr | Type::F64 | Type::FrameSlice => MachineWordSize::W64,
    }
}

fn overflow_op_code(op: OverflowOp) -> u64 {
    match op {
        OverflowOp::SAdd => 0,
        OverflowOp::SSub => 1,
        OverflowOp::SMul => 2,
        OverflowOp::UAdd => 3,
        OverflowOp::USub => 4,
        OverflowOp::UMul => 5,
    }
}

fn overflow_type_code(ty: Type) -> u64 {
    match ty {
        Type::I8 => 8,
        Type::I32 => 32,
        Type::I64 | Type::Ptr | Type::GcPtr => 64,
        Type::F64 | Type::FrameSlice => panic!("overflow check on non-integer type"),
    }
}

extern "C" fn jit_overflow_check_helper(op_code: u64, ty_code: u64, a: u64, b: u64) -> u64 {
    match (op_code, ty_code) {
        (0, 8) => (a as u8 as i8).overflowing_add(b as u8 as i8).1 as u64,
        (0, 32) => (a as u32 as i32).overflowing_add(b as u32 as i32).1 as u64,
        (0, 64) => (a as i64).overflowing_add(b as i64).1 as u64,
        (1, 8) => (a as u8 as i8).overflowing_sub(b as u8 as i8).1 as u64,
        (1, 32) => (a as u32 as i32).overflowing_sub(b as u32 as i32).1 as u64,
        (1, 64) => (a as i64).overflowing_sub(b as i64).1 as u64,
        (2, 8) => (a as u8 as i8).overflowing_mul(b as u8 as i8).1 as u64,
        (2, 32) => (a as u32 as i32).overflowing_mul(b as u32 as i32).1 as u64,
        (2, 64) => (a as i64).overflowing_mul(b as i64).1 as u64,
        (3, 8) => (a as u8).overflowing_add(b as u8).1 as u64,
        (3, 32) => (a as u32).overflowing_add(b as u32).1 as u64,
        (3, 64) => a.overflowing_add(b).1 as u64,
        (4, 8) => (a as u8).overflowing_sub(b as u8).1 as u64,
        (4, 32) => (a as u32).overflowing_sub(b as u32).1 as u64,
        (4, 64) => a.overflowing_sub(b).1 as u64,
        (5, 8) => (a as u8).overflowing_mul(b as u8).1 as u64,
        (5, 32) => (a as u32).overflowing_mul(b as u32).1 as u64,
        (5, 64) => a.overflowing_mul(b).1 as u64,
        _ => panic!("invalid overflow helper inputs: op_code={op_code}, ty_code={ty_code}"),
    }
}

fn align_up(n: usize, align: usize) -> usize {
    (n + align - 1) & !(align - 1)
}
