use std::error::Error;
use std::fmt::{Display, Formatter};
use std::marker::PhantomData;

use dynvalue::{LowBit, NanBox, TagScheme};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootPrecision {
    PreciseSlots,
    ConservativeWords,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootTransportKind {
    FrameScan,
    ShadowStack,
    StackMap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameSliceMode {
    OneShot,
    MultiShot,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapturedCallerResume {
    TopLevel,
    FromCall {
        return_dest: Option<usize>,
    },
    FromInvoke {
        normal_block: usize,
        normal_args_vals: Vec<u64>,
        exception_block: usize,
        exception_args_vals: Vec<u64>,
        has_ret_param: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameResumePoint {
    pub func_idx: usize,
    pub block_idx: usize,
    pub inst_idx: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapturedFrame {
    pub resume: FrameResumePoint,
    pub values: Vec<u64>,
    pub root_value_indices: Vec<usize>,
    pub resume_arg_value_indices: Vec<usize>,
    pub active_prompts: Vec<u32>,
    pub caller_resume: CapturedCallerResume,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameSliceSnapshot {
    pub prompt_id: u32,
    pub mode: FrameSliceMode,
    pub frames: Vec<CapturedFrame>,
    pub consumed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameSliceError {
    InvalidRootIndex { frame_idx: usize, root_idx: usize },
    MissingSlice,
    Consumed,
}

impl Display for FrameSliceError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            FrameSliceError::InvalidRootIndex { frame_idx, root_idx } => {
                write!(
                    f,
                    "frame slice root index {root_idx} is out of bounds for captured frame {frame_idx}"
                )
            }
            FrameSliceError::MissingSlice => f.write_str("frame slice handle does not exist"),
            FrameSliceError::Consumed => f.write_str("frame slice has already been consumed"),
        }
    }
}

impl Error for FrameSliceError {}

impl FrameSliceSnapshot {
    pub fn validate(&self) -> Result<(), FrameSliceError> {
        for (frame_idx, frame) in self.frames.iter().enumerate() {
            for &root_idx in &frame.root_value_indices {
                if root_idx >= frame.values.len() {
                    return Err(FrameSliceError::InvalidRootIndex { frame_idx, root_idx });
                }
            }
        }
        Ok(())
    }

    pub fn root_word_count(&self) -> usize {
        self.frames
            .iter()
            .map(|frame| frame.root_value_indices.len())
            .sum()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotClass {
    IncomingArg,
    Local,
    Spill,
    OutgoingArg,
    Root,
    SavedReg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallArgLocation {
    Register(usize),
    Stack(i32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameSlotBase {
    FramePointer,
    StackPointer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameSlotAccess {
    pub base: FrameSlotBase,
    pub offset: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConfigError {
    pub message: &'static str,
}

impl Display for ConfigError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message)
    }
}

impl Error for ConfigError {}

pub trait FrameSliceStore {
    type Handle: Clone;

    fn insert_slice(&mut self, slice: FrameSliceSnapshot) -> Result<Self::Handle, FrameSliceError>;
    fn clone_slice(&mut self, handle: &Self::Handle) -> Result<Self::Handle, FrameSliceError>;
    fn slice(&self, handle: &Self::Handle) -> Result<&FrameSliceSnapshot, FrameSliceError>;
    fn slice_mut(
        &mut self,
        handle: &Self::Handle,
    ) -> Result<&mut FrameSliceSnapshot, FrameSliceError>;
    fn encode_handle(handle: &Self::Handle) -> u64;
    fn decode_handle(bits: u64) -> Result<Self::Handle, FrameSliceError>;

    fn mark_consumed(&mut self, handle: &Self::Handle) -> Result<(), FrameSliceError> {
        let slice = self.slice_mut(handle)?;
        if slice.consumed {
            return Err(FrameSliceError::Consumed);
        }
        slice.consumed = true;
        Ok(())
    }
}

#[derive(Default)]
pub struct InMemoryFrameSliceStore {
    slices: Vec<FrameSliceSnapshot>,
}

impl InMemoryFrameSliceStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl FrameSliceStore for InMemoryFrameSliceStore {
    type Handle = usize;

    fn insert_slice(&mut self, slice: FrameSliceSnapshot) -> Result<Self::Handle, FrameSliceError> {
        slice.validate()?;
        let handle = self.slices.len();
        self.slices.push(slice);
        Ok(handle)
    }

    fn clone_slice(&mut self, handle: &Self::Handle) -> Result<Self::Handle, FrameSliceError> {
        let mut cloned = self.slice(handle)?.clone();
        cloned.consumed = false;
        self.insert_slice(cloned)
    }

    fn slice(&self, handle: &Self::Handle) -> Result<&FrameSliceSnapshot, FrameSliceError> {
        self.slices.get(*handle).ok_or(FrameSliceError::MissingSlice)
    }

    fn slice_mut(
        &mut self,
        handle: &Self::Handle,
    ) -> Result<&mut FrameSliceSnapshot, FrameSliceError> {
        self.slices
            .get_mut(*handle)
            .ok_or(FrameSliceError::MissingSlice)
    }

    fn encode_handle(handle: &Self::Handle) -> u64 {
        *handle as u64
    }

    fn decode_handle(bits: u64) -> Result<Self::Handle, FrameSliceError> {
        usize::try_from(bits).map_err(|_| FrameSliceError::MissingSlice)
    }
}

// ─── Stack Strategy ────────────────────────────────────────

// ─── Unified Stack Strategy ────────────────────────────────
//
// A StackStrategy is THE pluggable component that language authors choose.
// It determines how the stack works for both the interpreter AND the JIT:
//   - Compile-time: what codegen the JIT emits (prologue checks, segment sizes)
//   - Runtime: how the interpreter pushes frames, captures continuations, etc.
//
// Language authors either pick a prebuilt strategy (ContiguousVecStack,
// GCSegmentedStack) or implement their own.

/// Resume information for a frame's caller.
#[derive(Debug, Clone)]
pub enum FrameResume {
    TopLevel,
    FromCall { return_dest: Option<usize> },
}

/// Configuration for creating a stack runtime.
#[derive(Debug, Clone)]
pub struct StackConfig {
    pub heap_size: usize,
}

impl Default for StackConfig {
    fn default() -> Self { StackConfig { heap_size: 1024 * 1024 } }
}

/// The runtime side of a stack strategy — what the interpreter calls.
///
/// Owns the entire stack. The interpreter is a thin dispatch loop over
/// this trait. Different implementations give different stack models
/// (Vec-backed, GC-segmented, etc.) with different continuation behavior.
pub trait StackRuntime {
    // ── Frame management ────────────────────────────────────────

    /// Push a new frame. `args` are written into the frame's param slots.
    fn push_frame(&mut self, func_idx: usize, val_count: usize, args: &[(usize, u64)],
                  resume: FrameResume);
    /// Pop the current frame. Returns the caller resume info.
    fn pop_frame(&mut self) -> FrameResume;
    /// Is the stack empty?
    fn is_empty(&self) -> bool;

    // ── Value access (current frame) ────────────────────────────

    fn get(&self, idx: usize) -> u64;
    fn set(&mut self, idx: usize, val: u64);

    // ── Execution state (current frame) ─────────────────────────

    fn func_idx(&self) -> usize;
    fn block_idx(&self) -> usize;
    fn set_block(&mut self, block: usize);
    fn inst_idx(&self) -> usize;
    fn set_inst(&mut self, inst: usize);
    fn advance_inst(&mut self);

    // ── Prompts ─────────────────────────────────────────────────

    fn push_prompt(&mut self, prompt: u32);
    fn pop_prompt(&mut self, prompt: u32);
    /// Find the depth (distance from bottom) of the frame owning this prompt.
    fn find_prompt_depth(&self, prompt: u32) -> Option<usize>;
    /// Pop all frames above `depth`, keeping the frame at `depth` as the new top.
    fn pop_frames_above(&mut self, depth: usize);

    // ── Continuations ───────────────────────────────────────────

    /// Capture frames from the prompt to the top of the stack.
    /// `resume_dest` is the value slot in the top frame where resume args go.
    /// Returns a handle (u64) that can be stored in a value slot.
    fn capture(&mut self, prompt: u32, resume_dest: usize) -> u64;
    /// Resume a captured continuation with the given args.
    /// Replaces the current stack entirely.
    fn resume(&mut self, handle: u64, args: &[u64]);
    /// Clone a continuation for multi-shot use. Returns new handle.
    fn clone_continuation(&mut self, handle: u64) -> u64;

    // ── GC integration ─────────────────────────────────────────

    /// Whether the runtime wants a GC collection before the next allocation.
    fn needs_gc(&self) -> bool { false }
    /// Run a GC collection. Only called when `needs_gc()` returns true.
    fn collect_gc(&mut self) {}
}

/// The unified stack strategy trait. Language authors implement this
/// (or use a prebuilt one) to control stack behavior for both
/// the interpreter and the JIT.
pub trait UnifiedStackStrategy: Sized + 'static {
    const NAME: &'static str;

    // ── Compile-time configuration (JIT reads these) ─────────────

    /// Does the JIT prologue need a segment overflow check?
    fn needs_prologue_check() -> bool { false }
    /// Default segment size for segmented modes.
    fn segment_size() -> Option<usize> { None }
    /// Whether continuation IR instructions are supported.
    fn supports_continuations() -> bool { true }
    /// Whether prompts create segment boundaries.
    fn prompt_creates_segment_boundary() -> bool { false }
    /// Whether captured continuation roots need stack maps.
    fn captured_roots_need_stack_maps() -> bool { false }

    // ── Runtime ──────────────────────────────────────────────────

    /// The interpreter's runtime state.
    type Runtime: StackRuntime;
    /// Create the runtime.
    fn create_runtime(config: StackConfig) -> Self::Runtime;

    // ── Validation ───────────────────────────────────────────────

    fn validate() -> Result<(), ConfigError> { Ok(()) }
}

// ─── Prebuilt Strategies ───────────────────────────────────

/// Contiguous Vec-backed stack. No GC integration, values stored in
/// `Vec<u64>`. Continuations captured by cloning values. The simplest
/// strategy — good default for languages that don't need GC-managed stacks.
pub struct ContiguousVecStack;

impl UnifiedStackStrategy for ContiguousVecStack {
    const NAME: &'static str = "contiguous-vec";
    fn supports_continuations() -> bool { true }
    type Runtime = VecStackRuntime;
    fn create_runtime(_: StackConfig) -> VecStackRuntime { VecStackRuntime::new() }
}

struct VecFrame {
    func_idx: usize,
    vals: Vec<u64>,
    block_idx: usize,
    inst_idx: usize,
    resume: FrameResume,
    active_prompts: Vec<u32>,
}

#[derive(Clone)]
struct VecCont {
    frames: Vec<VecContFrame>,
}

#[derive(Clone)]
struct VecContFrame {
    func_idx: usize,
    vals: Vec<u64>,
    block_idx: usize,
    inst_idx: usize,
    resume: FrameResume,
    active_prompts: Vec<u32>,
    resume_arg_idx: Option<usize>,
}

pub struct VecStackRuntime {
    stack: Vec<VecFrame>,
    conts: Vec<VecCont>,
}

impl VecStackRuntime {
    pub fn new() -> Self {
        VecStackRuntime { stack: Vec::new(), conts: Vec::new() }
    }
}

impl Default for VecStackRuntime {
    fn default() -> Self { Self::new() }
}

impl StackRuntime for VecStackRuntime {
    fn push_frame(&mut self, func_idx: usize, val_count: usize, args: &[(usize, u64)], resume: FrameResume) {
        let mut vals = vec![0u64; val_count];
        for &(idx, val) in args {
            vals[idx] = val;
        }
        self.stack.push(VecFrame {
            func_idx, vals, block_idx: 0, inst_idx: 0, resume,
            active_prompts: Vec::new(),
        });
    }

    fn pop_frame(&mut self) -> FrameResume {
        self.stack.pop().expect("pop_frame on empty stack").resume
    }

    fn is_empty(&self) -> bool { self.stack.is_empty() }

    fn get(&self, idx: usize) -> u64 { self.stack.last().unwrap().vals[idx] }
    fn set(&mut self, idx: usize, val: u64) { self.stack.last_mut().unwrap().vals[idx] = val; }

    fn func_idx(&self) -> usize { self.stack.last().unwrap().func_idx }
    fn block_idx(&self) -> usize { self.stack.last().unwrap().block_idx }
    fn set_block(&mut self, block: usize) { self.stack.last_mut().unwrap().block_idx = block; }
    fn inst_idx(&self) -> usize { self.stack.last().unwrap().inst_idx }
    fn set_inst(&mut self, inst: usize) { self.stack.last_mut().unwrap().inst_idx = inst; }
    fn advance_inst(&mut self) { self.stack.last_mut().unwrap().inst_idx += 1; }

    fn push_prompt(&mut self, prompt: u32) {
        self.stack.last_mut().unwrap().active_prompts.push(prompt);
    }
    fn pop_prompt(&mut self, prompt: u32) {
        let popped = self.stack.last_mut().unwrap().active_prompts.pop();
        assert_eq!(popped, Some(prompt));
    }
    fn find_prompt_depth(&self, prompt: u32) -> Option<usize> {
        self.stack.iter().rposition(|f| f.active_prompts.contains(&prompt))
    }
    fn pop_frames_above(&mut self, depth: usize) {
        while self.stack.len() > depth + 1 {
            self.stack.pop();
        }
    }

    fn capture(&mut self, prompt: u32, resume_dest: usize) -> u64 {
        let start = self.stack.iter().rposition(|f| f.active_prompts.contains(&prompt))
            .expect("capture: prompt not found");
        let frame_count = self.stack.len() - start;
        let mut frames = Vec::with_capacity(frame_count);
        for (i, f) in self.stack[start..].iter().enumerate() {
            let is_top = i + 1 == frame_count;
            frames.push(VecContFrame {
                func_idx: f.func_idx,
                vals: f.vals.clone(),
                block_idx: f.block_idx,
                inst_idx: f.inst_idx,
                resume: f.resume.clone(),
                active_prompts: f.active_prompts.clone(),
                resume_arg_idx: if is_top { Some(resume_dest) } else { None },
            });
        }
        let handle = self.conts.len() as u64;
        self.conts.push(VecCont { frames });
        handle
    }

    fn resume(&mut self, handle: u64, args: &[u64]) {
        let cont = self.conts[handle as usize].clone();
        self.stack.clear();
        let frame_count = cont.frames.len();
        for (i, cf) in cont.frames.into_iter().enumerate() {
            let mut vals = cf.vals;
            if i + 1 == frame_count {
                if let Some(idx) = cf.resume_arg_idx {
                    if let Some(&arg) = args.first() {
                        vals[idx] = arg;
                    }
                }
            }
            self.stack.push(VecFrame {
                func_idx: cf.func_idx,
                vals,
                block_idx: cf.block_idx,
                inst_idx: cf.inst_idx,
                resume: cf.resume,
                active_prompts: cf.active_prompts,
            });
        }
    }

    fn clone_continuation(&mut self, handle: u64) -> u64 {
        let cloned = self.conts[handle as usize].clone();
        let new_handle = self.conts.len() as u64;
        self.conts.push(cloned);
        new_handle
    }
}

// ─── Value Layout ──────────────────────────────────────────

pub trait ValueLayout: TagScheme {
    const NAME: &'static str;

    fn root_precision_hint() -> RootPrecision;
}

pub trait LayoutConfigDefaults: ValueLayout {
    type DefaultRoots: RootStrategy<Self>;
    type DefaultRootTransport: RootTransport<Self, Self::DefaultRoots>;
}

pub trait RootStrategy<L: ValueLayout> {
    const NAME: &'static str;

    fn precision() -> RootPrecision;

    fn supports_layout() -> bool {
        match (Self::precision(), L::root_precision_hint()) {
            (RootPrecision::PreciseSlots, RootPrecision::ConservativeWords) => false,
            _ => true,
        }
    }

    fn validate() -> Result<(), ConfigError> {
        if Self::supports_layout() {
            Ok(())
        } else {
            Err(ConfigError {
                message: "root strategy is incompatible with value layout",
            })
        }
    }
}

pub trait RootTransport<L: ValueLayout, R: RootStrategy<L>> {
    const NAME: &'static str;

    fn kind() -> RootTransportKind;

    fn requires_frame_roots() -> bool {
        matches!(Self::kind(), RootTransportKind::FrameScan)
    }

    fn requires_shadow_slots() -> bool {
        matches!(Self::kind(), RootTransportKind::ShadowStack)
    }

    fn emits_stack_maps() -> bool {
        matches!(Self::kind(), RootTransportKind::StackMap)
    }

    fn validate_frame<F: FrameStrategy<L, R, C>, C: CallingConvention<L>>() -> Result<(), ConfigError> {
        if Self::requires_frame_roots() && !F::exposes_slot_class(SlotClass::Root) {
            return Err(ConfigError {
                message: "root transport requires frame-visible root slots",
            });
        }
        if Self::requires_shadow_slots() && !F::supports_shadow_roots() {
            return Err(ConfigError {
                message: "root transport requires shadow-root frame support",
            });
        }
        Ok(())
    }
}

pub trait CallingConvention<L: ValueLayout> {
    const NAME: &'static str;

    fn stack_align() -> usize;
    fn register_arg_limit() -> usize;
    fn stack_arg_stride() -> usize {
        8
    }

    fn arg_location(slot: usize) -> CallArgLocation {
        let reg_limit = Self::register_arg_limit();
        if slot < reg_limit {
            CallArgLocation::Register(slot)
        } else {
            CallArgLocation::Stack(Self::outgoing_stack_arg_offset(slot))
        }
    }

    fn incoming_stack_arg_offset(slot: usize) -> i32 {
        let reg_limit = Self::register_arg_limit();
        if slot < reg_limit {
            0
        } else {
            ((slot - reg_limit) * Self::stack_arg_stride()) as i32
        }
    }

    fn outgoing_stack_arg_offset(slot: usize) -> i32 {
        Self::incoming_stack_arg_offset(slot)
    }

    fn outgoing_stack_size(total_args: usize) -> i32 {
        let overflow = total_args.saturating_sub(Self::register_arg_limit());
        let bytes = overflow * Self::stack_arg_stride();
        (((bytes + (Self::stack_align() - 1)) / Self::stack_align()) * Self::stack_align()) as i32
    }
}

#[derive(Debug, Clone)]
pub struct StackFrameLayout {
    next_local_offset: i32,
    max_outgoing_arg_bytes: i32,
    block_param_slots: Vec<Vec<i32>>,
    root_slots: Vec<i32>,
    shadow_root_slots: Vec<i32>,
}

impl StackFrameLayout {
    pub fn new(block_count: usize) -> Self {
        StackFrameLayout {
            next_local_offset: 16,
            max_outgoing_arg_bytes: 0,
            block_param_slots: vec![Vec::new(); block_count],
            root_slots: Vec::new(),
            shadow_root_slots: Vec::new(),
        }
    }

    pub fn alloc_local_slot(&mut self) -> i32 {
        let offset = self.next_local_offset;
        self.next_local_offset += 8;
        offset
    }

    pub fn alloc_root_slot(&mut self) -> i32 {
        let offset = self.alloc_local_slot();
        self.root_slots.push(offset);
        offset
    }

    pub fn alloc_shadow_root_slot(&mut self) -> i32 {
        let offset = self.alloc_local_slot();
        self.shadow_root_slots.push(offset);
        offset
    }

    pub fn reserve_outgoing_arg_bytes(&mut self, bytes: i32) {
        self.max_outgoing_arg_bytes = self.max_outgoing_arg_bytes.max(bytes);
    }

    pub fn total_frame_size(&self, stack_align: usize) -> i32 {
        (((self.next_local_offset + self.max_outgoing_arg_bytes) as usize + (stack_align - 1))
            / stack_align
            * stack_align) as i32
    }

    pub fn add_block_param_slot(&mut self, block_idx: usize, offset: i32) {
        self.block_param_slots[block_idx].push(offset);
    }

    pub fn block_param_slots(&self, block_idx: usize) -> &[i32] {
        &self.block_param_slots[block_idx]
    }

    pub fn root_scan_size(&self) -> i32 {
        self.root_slots
            .iter()
            .copied()
            .max()
            .map(|offset| offset + 8)
            .unwrap_or(0)
    }

    pub fn shadow_root_slots(&self) -> &[i32] {
        &self.shadow_root_slots
    }
}

pub trait FrameLayout {
    fn alloc_local_slot(&mut self) -> i32;
    fn alloc_root_slot(&mut self) -> i32;
    fn alloc_shadow_root_slot(&mut self) -> i32;
    fn reserve_outgoing_arg_bytes(&mut self, bytes: i32);
    fn total_frame_size(&self, stack_align: usize) -> i32;
    fn add_block_param_slot(&mut self, block_idx: usize, offset: i32);
    fn block_param_slots(&self, block_idx: usize) -> &[i32];
    fn root_scan_size(&self) -> i32;
    fn shadow_root_slots(&self) -> &[i32];
    fn slot_access(&self, slot: i32) -> FrameSlotAccess;
}

impl FrameLayout for StackFrameLayout {
    fn alloc_local_slot(&mut self) -> i32 {
        Self::alloc_local_slot(self)
    }

    fn alloc_root_slot(&mut self) -> i32 {
        Self::alloc_root_slot(self)
    }

    fn alloc_shadow_root_slot(&mut self) -> i32 {
        Self::alloc_shadow_root_slot(self)
    }

    fn reserve_outgoing_arg_bytes(&mut self, bytes: i32) {
        Self::reserve_outgoing_arg_bytes(self, bytes)
    }

    fn total_frame_size(&self, stack_align: usize) -> i32 {
        Self::total_frame_size(self, stack_align)
    }

    fn add_block_param_slot(&mut self, block_idx: usize, offset: i32) {
        Self::add_block_param_slot(self, block_idx, offset)
    }

    fn block_param_slots(&self, block_idx: usize) -> &[i32] {
        Self::block_param_slots(self, block_idx)
    }

    fn root_scan_size(&self) -> i32 {
        Self::root_scan_size(self)
    }

    fn shadow_root_slots(&self) -> &[i32] {
        Self::shadow_root_slots(self)
    }

    fn slot_access(&self, slot: i32) -> FrameSlotAccess {
        FrameSlotAccess {
            base: FrameSlotBase::FramePointer,
            offset: slot,
        }
    }
}

pub trait FrameStrategy<L: ValueLayout, R: RootStrategy<L>, C: CallingConvention<L>> {
    const NAME: &'static str;
    type Layout: FrameLayout;

    fn stack_align() -> usize {
        C::stack_align()
    }

    fn new_layout(block_count: usize) -> Self::Layout;

    fn exposes_slot_class(_class: SlotClass) -> bool {
        true
    }

    fn supports_shadow_roots() -> bool {
        false
    }

    fn supports_stack_maps() -> bool {
        false
    }

    fn validate() -> Result<(), ConfigError> {
        Ok(())
    }
}

pub trait SafepointStrategy<
    L: ValueLayout,
    R: RootStrategy<L>,
    T: RootTransport<L, R>,
    F: FrameStrategy<L, R, C>,
    C: CallingConvention<L>,
> {
    const NAME: &'static str;

    fn validates_frame() -> bool {
        if T::requires_frame_roots() {
            F::exposes_slot_class(SlotClass::Root) || R::precision() == RootPrecision::ConservativeWords
        } else if T::requires_shadow_slots() {
            F::supports_shadow_roots()
        } else if T::emits_stack_maps() {
            F::supports_stack_maps()
        } else {
            true
        }
    }

    fn validate() -> Result<(), ConfigError> {
        if Self::validates_frame() {
            Ok(())
        } else {
            Err(ConfigError {
                message: "safepoint strategy requires frame-visible root slots",
            })
        }
    }
}

/// Codegen configuration for JIT lowering. Controls value layout, root
/// strategy, calling convention, frame strategy, and safepoints.
pub trait CodegenConfig {
    type Layout: ValueLayout;
    type Roots: RootStrategy<Self::Layout>;
    type RootTransport: RootTransport<Self::Layout, Self::Roots>;
    type CallingConvention: CallingConvention<Self::Layout>;
    type Frames: FrameStrategy<Self::Layout, Self::Roots, Self::CallingConvention>;
    type Safepoints:
        SafepointStrategy<
            Self::Layout,
            Self::Roots,
            Self::RootTransport,
            Self::Frames,
            Self::CallingConvention,
        >;

    fn validate() -> Result<(), ConfigError> {
        Self::Roots::validate()?;
        Self::Frames::validate()?;
        Self::RootTransport::validate_frame::<Self::Frames, Self::CallingConvention>()?;
        Self::Safepoints::validate()?;
        Ok(())
    }
}

pub fn validate_codegen_config<C: CodegenConfig>() -> Result<(), ConfigError> {
    <C as CodegenConfig>::validate()
}

pub struct PreciseStackRoots;

impl<L: ValueLayout> RootStrategy<L> for PreciseStackRoots {
    const NAME: &'static str = "precise-stack-roots";

    fn precision() -> RootPrecision {
        RootPrecision::PreciseSlots
    }
}

pub struct ConservativeWordRoots;

impl<L: ValueLayout> RootStrategy<L> for ConservativeWordRoots {
    const NAME: &'static str = "conservative-word-roots";

    fn precision() -> RootPrecision {
        RootPrecision::ConservativeWords
    }
}

pub struct AArch64InternalCc;

impl<L: ValueLayout> CallingConvention<L> for AArch64InternalCc {
    const NAME: &'static str = "aarch64-internal";

    fn stack_align() -> usize {
        16
    }

    fn register_arg_limit() -> usize {
        16
    }
}

pub struct AArch64CAbi;

impl<L: ValueLayout> CallingConvention<L> for AArch64CAbi {
    const NAME: &'static str = "aarch64-c-abi";

    fn stack_align() -> usize {
        16
    }

    fn register_arg_limit() -> usize {
        8
    }
}

pub struct StackSlotFrames;

impl<L: ValueLayout, R: RootStrategy<L>, C: CallingConvention<L>> FrameStrategy<L, R, C>
    for StackSlotFrames
{
    const NAME: &'static str = "stack-slot-frames";
    type Layout = StackFrameLayout;

    fn new_layout(block_count: usize) -> Self::Layout {
        StackFrameLayout::new(block_count)
    }
}

pub struct ShadowStackFrames;

impl<L: ValueLayout, R: RootStrategy<L>, C: CallingConvention<L>> FrameStrategy<L, R, C>
    for ShadowStackFrames
{
    const NAME: &'static str = "shadow-stack-frames";
    type Layout = StackFrameLayout;

    fn new_layout(block_count: usize) -> Self::Layout {
        StackFrameLayout::new(block_count)
    }

    fn supports_shadow_roots() -> bool {
        true
    }
}

pub struct StackMapFrames;

impl<L: ValueLayout, R: RootStrategy<L>, C: CallingConvention<L>> FrameStrategy<L, R, C>
    for StackMapFrames
{
    const NAME: &'static str = "stack-map-frames";
    type Layout = StackFrameLayout;

    fn new_layout(block_count: usize) -> Self::Layout {
        StackFrameLayout::new(block_count)
    }

    fn supports_stack_maps() -> bool {
        true
    }
}

pub struct FrameScanRoots;

impl<L: ValueLayout, R: RootStrategy<L>> RootTransport<L, R> for FrameScanRoots {
    const NAME: &'static str = "frame-scan-roots";

    fn kind() -> RootTransportKind {
        RootTransportKind::FrameScan
    }
}

pub struct ShadowStackRoots;

impl<L: ValueLayout, R: RootStrategy<L>> RootTransport<L, R> for ShadowStackRoots {
    const NAME: &'static str = "shadow-stack-roots";

    fn kind() -> RootTransportKind {
        RootTransportKind::ShadowStack
    }
}

pub struct StackMapRoots;

impl<L: ValueLayout, R: RootStrategy<L>> RootTransport<L, R> for StackMapRoots {
    const NAME: &'static str = "stack-map-roots";

    fn kind() -> RootTransportKind {
        RootTransportKind::StackMap
    }
}

pub struct CallbackSafepoints;

impl<
        L: ValueLayout,
        R: RootStrategy<L>,
        T: RootTransport<L, R>,
        F: FrameStrategy<L, R, C>,
        C: CallingConvention<L>,
    > SafepointStrategy<L, R, T, F, C> for CallbackSafepoints
{
    const NAME: &'static str = "callback-safepoints";
}

pub struct StackMapSafepoints;

impl<
        L: ValueLayout,
        R: RootStrategy<L>,
        T: RootTransport<L, R>,
        F: FrameStrategy<L, R, C>,
        C: CallingConvention<L>,
    > SafepointStrategy<L, R, T, F, C> for StackMapSafepoints
{
    const NAME: &'static str = "stack-map-safepoints";

    fn validate() -> Result<(), ConfigError> {
        if !T::emits_stack_maps() {
            return Err(ConfigError {
                message: "stack-map safepoints require stack-map root transport",
            });
        }
        if F::supports_stack_maps() {
            Ok(())
        } else {
            Err(ConfigError {
                message: "stack-map safepoints require stack-map-capable frames",
            })
        }
    }
}

pub struct DefaultCodegenConfig<L: LayoutConfigDefaults>(PhantomData<L>);

impl<L: LayoutConfigDefaults> CodegenConfig for DefaultCodegenConfig<L> {
    type Layout = L;
    type Roots = L::DefaultRoots;
    type RootTransport = L::DefaultRootTransport;
    type CallingConvention = AArch64InternalCc;
    type Frames = StackSlotFrames;
    type Safepoints = CallbackSafepoints;
}


impl<const TAG_BITS: u32> ValueLayout for LowBit<TAG_BITS> {
    const NAME: &'static str = "low-bit";

    fn root_precision_hint() -> RootPrecision {
        RootPrecision::PreciseSlots
    }
}

impl<const TAG_BITS: u32> LayoutConfigDefaults for LowBit<TAG_BITS> {
    type DefaultRoots = PreciseStackRoots;
    type DefaultRootTransport = FrameScanRoots;
}

impl ValueLayout for NanBox {
    const NAME: &'static str = "nan-box";

    fn root_precision_hint() -> RootPrecision {
        RootPrecision::ConservativeWords
    }
}

impl LayoutConfigDefaults for NanBox {
    type DefaultRoots = ConservativeWordRoots;
    type DefaultRootTransport = FrameScanRoots;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct InvalidNanBoxConfig;
    impl CodegenConfig for InvalidNanBoxConfig {
        type Layout = NanBox;
        type Roots = PreciseStackRoots;
        type RootTransport = FrameScanRoots;
        type CallingConvention = AArch64InternalCc;
        type Frames = StackSlotFrames;
        type Safepoints = CallbackSafepoints;
    }

    struct InvalidShadowTransportConfig;
    impl CodegenConfig for InvalidShadowTransportConfig {
        type Layout = LowBit<3>;
        type Roots = PreciseStackRoots;
        type RootTransport = ShadowStackRoots;
        type CallingConvention = AArch64InternalCc;
        type Frames = StackSlotFrames;
        type Safepoints = CallbackSafepoints;
    }

    struct InvalidStackMapSafepointConfig;
    impl CodegenConfig for InvalidStackMapSafepointConfig {
        type Layout = LowBit<3>;
        type Roots = PreciseStackRoots;
        type RootTransport = FrameScanRoots;
        type CallingConvention = AArch64InternalCc;
        type Frames = StackSlotFrames;
        type Safepoints = StackMapSafepoints;
    }

    #[test]
    fn low_bit_default_config_is_valid() {
        assert!(validate_codegen_config::<DefaultCodegenConfig<LowBit<3>>>().is_ok());
    }

    #[test]
    fn nan_box_default_config_is_valid() {
        assert!(validate_codegen_config::<DefaultCodegenConfig<NanBox>>().is_ok());
    }

    #[test]
    fn precise_roots_reject_nan_box_layout() {
        let err = validate_codegen_config::<InvalidNanBoxConfig>().unwrap_err();
        assert_eq!(err.message, "root strategy is incompatible with value layout");
    }

    #[test]
    fn shadow_stack_transport_requires_shadow_frame_support() {
        let err = validate_codegen_config::<InvalidShadowTransportConfig>().unwrap_err();
        assert_eq!(err.message, "root transport requires shadow-root frame support");
    }

    #[test]
    fn stack_map_safepoints_require_stack_map_transport() {
        let err = validate_codegen_config::<InvalidStackMapSafepointConfig>().unwrap_err();
        assert_eq!(err.message, "stack-map safepoints require stack-map root transport");
    }

    #[test]
    fn frame_slice_snapshot_validates_root_indices() {
        let good = FrameSliceSnapshot {
            prompt_id: 0,
            mode: FrameSliceMode::OneShot,
            frames: vec![CapturedFrame {
                resume: FrameResumePoint {
                    func_idx: 1,
                    block_idx: 2,
                    inst_idx: 3,
                },
                values: vec![10, 20, 30],
                root_value_indices: vec![0, 2],
                resume_arg_value_indices: vec![1],
                active_prompts: vec![0],
                caller_resume: CapturedCallerResume::TopLevel,
            }],
            consumed: false,
        };
        assert_eq!(good.root_word_count(), 2);
        assert!(good.validate().is_ok());

        let mut bad = good.clone();
        bad.frames[0].root_value_indices.push(9);
        assert_eq!(
            bad.validate(),
            Err(FrameSliceError::InvalidRootIndex {
                frame_idx: 0,
                root_idx: 9,
            })
        );
    }

}
