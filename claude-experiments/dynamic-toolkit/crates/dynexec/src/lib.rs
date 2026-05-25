use std::error::Error;
use std::fmt::{Display, Formatter};
use std::marker::PhantomData;

use dynvalue::{LowBit, NanBox, TagScheme};

pub mod cont_heap;
pub mod cont_ops;
pub use cont_heap::{
    BuilderFrame, CapturedStackBuilder, ContinuationContext, ContinuationTypes, ContinuationView,
    FrameView, NoContinuations, capture_continuation, read_continuation,
};
pub use cont_ops::{
    FrameCapture, FrameRestorable, PromptBoundaryAction, do_capture, do_clone, do_resume,
    resolve_prompt_boundary,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootTransportKind {
    ShadowStack,
    StackMap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameResumePoint {
    pub func_idx: usize,
    pub block_idx: usize,
    pub inst_idx: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameSliceError {
    InvalidRootIndex { frame_idx: usize, root_idx: usize },
    MissingSlice,
}

impl Display for FrameSliceError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            FrameSliceError::InvalidRootIndex {
                frame_idx,
                root_idx,
            } => {
                write!(
                    f,
                    "frame slice root index {root_idx} is out of bounds for captured frame {frame_idx}"
                )
            }
            FrameSliceError::MissingSlice => f.write_str("frame slice handle does not exist"),
        }
    }
}

impl Error for FrameSliceError {}

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameResume {
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
    /// This frame is the bottom of a captured slice that was spliced on top
    /// of the resumer's stack by `resume_slice`. When it pops, the runtime
    /// should transfer control to `return_block` in the resumer's current
    /// frame, delivering the returned value as that block's first param
    /// (written at `return_param_dest`).
    FromResume {
        return_block: usize,
        return_param_dest: Option<usize>,
    },
}

/// Configuration for creating a stack runtime.
#[derive(Debug, Clone)]
pub struct StackConfig {
    pub heap_size: usize,
}

impl Default for StackConfig {
    fn default() -> Self {
        StackConfig {
            heap_size: 1024 * 1024,
        }
    }
}

/// The interpreter's frame store — what the interpreter dispatch loop calls.
///
/// Manages frames, values, execution state, and prompts. Different
/// implementations give different stack models (Vec-backed, GC-segmented, etc.).
///
/// Continuation capture/resume and GC are handled externally via
/// `ContinuationStore` and separate GC traits, NOT inside this trait.
pub trait InterpFrameStore {
    // ── Frame management ────────────────────────────────────────

    /// Push a new frame. `args` are written into the frame's param slots.
    /// `slot_count` is the number of stack-allocated slots for StackAddr instructions.
    fn push_frame(
        &mut self,
        func_idx: usize,
        val_count: usize,
        slot_count: usize,
        args: &[(usize, u64)],
        resume: FrameResume,
    );
    /// Pop the current frame. Returns the caller resume info.
    fn pop_frame(&mut self) -> FrameResume;
    /// Read the current top frame's resume info without popping.
    /// Used by the dispatcher's `abort_to_prompt` handler to detect
    /// spliced captured frames (whose `resume` is `FromResume`) and
    /// trampoline back to the resumer instead of running the
    /// prompt's handler block.
    fn peek_top_resume(&self) -> FrameResume;
    /// Is the stack empty?
    fn is_empty(&self) -> bool;

    // ── Value access (current frame) ────────────────────────────

    fn get(&self, idx: usize) -> u64;
    fn set(&mut self, idx: usize, val: u64);

    // ── Stack-allocated slots (current frame) ──────────────────

    /// Return a raw pointer to a stack-allocated slot in the current frame.
    /// Used by `StackAddr` instructions that need stable pointers to frame memory.
    /// Default: panics (only needed by interpreters that support StackAddr).
    fn slot_ptr(&self, _slot_idx: usize) -> *const u64 {
        panic!("slot_ptr not supported by this InterpFrameStore")
    }

    // ── GC root synchronization ──────────────────────────────────

    /// Sync live values from the current frame to GC root slots.
    /// Called before extern calls that may trigger a moving GC.
    fn sync_top_to_roots(&self) {}
    /// Reload values from GC root slots into the current frame.
    /// Called after extern calls that may have triggered a moving GC.
    fn sync_top_from_roots(&mut self) {}

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

    // ── Exception handlers ──────────────────────────────────────

    /// Push an exception handler onto the current top frame. The block
    /// is jumped to (with the thrown value as its first block param)
    /// when `Terminator::Raise` fires or when a callee's exception
    /// propagates up to this frame.
    ///
    /// Default impl panics — only interpreter implementations that
    /// honor `Inst::PushHandler` need to override.
    fn push_handler(&mut self, _handler_block_idx: usize) {
        panic!("push_handler not supported by this InterpFrameStore")
    }
    /// Pop the most-recently-pushed exception handler from the current
    /// top frame. Default impl panics — see `push_handler`.
    fn pop_handler(&mut self) {
        panic!("pop_handler not supported by this InterpFrameStore")
    }
    /// Return the topmost active exception handler in the current
    /// frame, or `None` if no handler is active. Default returns
    /// `None` so stores without handler support don't see catches.
    fn top_handler(&self) -> Option<usize> {
        None
    }

    // Heap-backed continuation support lives in the `FrameCapture` and
    // `FrameRestorable` traits (in `cont_ops`). Stores that support
    // heap-backed continuations implement those in addition to
    // `InterpFrameStore`. The core frame-store trait stays focused on
    // mechanics that every store must provide.

    // ── GC integration ─────────────────────────────────────────

    /// Whether the runtime wants a GC collection before the next allocation.
    fn needs_gc(&self) -> bool {
        false
    }
    /// Run a GC collection. Only called when `needs_gc()` returns true.
    fn collect_gc(&mut self) {}
    /// Precise safepoint: only the given value indices are live GC roots.
    /// Default: falls back to `collect_gc()`.
    fn safepoint(&mut self, _live_indices: &[usize]) {
        self.collect_gc()
    }
}

/// The unified stack strategy trait. Language authors implement this
/// (or use a prebuilt one) to control stack behavior for both
/// the interpreter and the JIT.
pub trait UnifiedStackStrategy: Sized + 'static {
    const NAME: &'static str;

    // ── Compile-time configuration (JIT reads these) ─────────────

    /// Does the JIT prologue need a segment overflow check?
    fn needs_prologue_check() -> bool {
        false
    }
    /// Default segment size for segmented modes.
    fn segment_size() -> Option<usize> {
        None
    }
    /// Whether continuation IR instructions are supported.
    fn supports_continuations() -> bool {
        true
    }
    /// Whether prompts create segment boundaries.
    fn prompt_creates_segment_boundary() -> bool {
        false
    }
    /// Whether captured continuation roots need stack maps.
    fn captured_roots_need_stack_maps() -> bool {
        false
    }

    // ── Runtime ──────────────────────────────────────────────────

    /// The interpreter's runtime state.
    type Runtime: InterpFrameStore;
    /// Create the runtime.
    fn create_runtime(config: StackConfig) -> Self::Runtime;

    // ── Validation ───────────────────────────────────────────────

    fn validate() -> Result<(), ConfigError> {
        Ok(())
    }
}

// ─── Prebuilt Strategies ───────────────────────────────────

/// Contiguous Vec-backed stack. No GC integration, values stored in
/// `Vec<u64>`. Continuations captured by cloning values. The simplest
/// strategy — good default for languages that don't need GC-managed stacks.
pub struct ContiguousVecStack;

impl UnifiedStackStrategy for ContiguousVecStack {
    const NAME: &'static str = "contiguous-vec";
    fn supports_continuations() -> bool {
        true
    }
    type Runtime = VecFrameStore;
    fn create_runtime(_: StackConfig) -> VecFrameStore {
        VecFrameStore::new()
    }
}

struct VecFrame {
    func_idx: usize,
    vals: Vec<u64>,
    slots: Vec<u64>,
    block_idx: usize,
    inst_idx: usize,
    resume: FrameResume,
    active_prompts: Vec<u32>,
}

pub struct VecFrameStore {
    stack: Vec<VecFrame>,
}

impl VecFrameStore {
    pub fn new() -> Self {
        VecFrameStore { stack: Vec::new() }
    }
}

impl Default for VecFrameStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InterpFrameStore for VecFrameStore {
    fn push_frame(
        &mut self,
        func_idx: usize,
        val_count: usize,
        slot_count: usize,
        args: &[(usize, u64)],
        resume: FrameResume,
    ) {
        let mut vals = vec![0u64; val_count];
        for &(idx, val) in args {
            vals[idx] = val;
        }
        self.stack.push(VecFrame {
            func_idx,
            vals,
            slots: vec![0u64; slot_count],
            block_idx: 0,
            inst_idx: 0,
            resume,
            active_prompts: Vec::new(),
        });
    }

    fn pop_frame(&mut self) -> FrameResume {
        self.stack.pop().expect("pop_frame on empty stack").resume
    }

    fn peek_top_resume(&self) -> FrameResume {
        self.stack
            .last()
            .expect("peek_top_resume on empty stack")
            .resume
            .clone()
    }

    fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    fn get(&self, idx: usize) -> u64 {
        self.stack.last().unwrap().vals[idx]
    }
    fn set(&mut self, idx: usize, val: u64) {
        self.stack.last_mut().unwrap().vals[idx] = val;
    }

    fn slot_ptr(&self, slot_idx: usize) -> *const u64 {
        let frame = self.stack.last().unwrap();
        unsafe { frame.slots.as_ptr().add(slot_idx) }
    }

    fn func_idx(&self) -> usize {
        self.stack.last().unwrap().func_idx
    }
    fn block_idx(&self) -> usize {
        self.stack.last().unwrap().block_idx
    }
    fn set_block(&mut self, block: usize) {
        self.stack.last_mut().unwrap().block_idx = block;
    }
    fn inst_idx(&self) -> usize {
        self.stack.last().unwrap().inst_idx
    }
    fn set_inst(&mut self, inst: usize) {
        self.stack.last_mut().unwrap().inst_idx = inst;
    }
    fn advance_inst(&mut self) {
        self.stack.last_mut().unwrap().inst_idx += 1;
    }

    fn push_prompt(&mut self, prompt: u32) {
        self.stack.last_mut().unwrap().active_prompts.push(prompt);
    }
    fn pop_prompt(&mut self, prompt: u32) {
        let popped = self.stack.last_mut().unwrap().active_prompts.pop();
        assert_eq!(popped, Some(prompt));
    }
    fn find_prompt_depth(&self, prompt: u32) -> Option<usize> {
        self.stack
            .iter()
            .rposition(|f| f.active_prompts.contains(&prompt))
    }
    fn pop_frames_above(&mut self, depth: usize) {
        while self.stack.len() > depth + 1 {
            self.stack.pop();
        }
    }
}

// ─── Value Layout ──────────────────────────────────────────

pub trait ValueLayout: TagScheme {
    const NAME: &'static str;
}

pub trait LayoutConfigDefaults: ValueLayout {
    type DefaultRoots: RootStrategy<Self>;
    type DefaultRootTransport: RootTransport<Self, Self::DefaultRoots>;
}

pub trait RootStrategy<L: ValueLayout> {
    const NAME: &'static str;

    fn validate() -> Result<(), ConfigError> {
        Ok(())
    }
}

pub trait RootTransport<L: ValueLayout, R: RootStrategy<L>> {
    const NAME: &'static str;

    fn kind() -> RootTransportKind;

    fn requires_shadow_slots() -> bool {
        matches!(Self::kind(), RootTransportKind::ShadowStack)
    }

    fn emits_stack_maps() -> bool {
        matches!(Self::kind(), RootTransportKind::StackMap)
    }

    fn validate_frame<F: FrameStrategy<L, R, C>, C: CallingConvention<L>>()
    -> Result<(), ConfigError> {
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
        self.alloc_local_slot_bytes(8)
    }

    /// Allocate `size_bytes` of local stack space (rounded up to a
    /// multiple of 8) and return the offset of the first byte. Used by
    /// frontends that spill multi-word buffers (closure capture
    /// arrays, packed argument slots, etc.) into a contiguous slot.
    pub fn alloc_local_slot_bytes(&mut self, size_bytes: i32) -> i32 {
        let aligned = ((size_bytes.max(8) as i32 + 7) & !7) as i32;
        let offset = self.next_local_offset;
        self.next_local_offset += aligned;
        offset
    }

    pub fn alloc_root_slot(&mut self) -> i32 {
        let offset = self.alloc_local_slot();
        self.root_slots.push(offset);
        offset
    }

    /// Like `alloc_root_slot` but for `size_bytes` of contiguous
    /// space; **every** word in the range is registered as a GC root.
    pub fn alloc_root_slot_bytes(&mut self, size_bytes: i32) -> i32 {
        let aligned = ((size_bytes.max(8) as i32 + 7) & !7) as i32;
        let offset = self.alloc_local_slot_bytes(aligned);
        let mut o = offset;
        while o < offset + aligned {
            self.root_slots.push(o);
            o += 8;
        }
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
    fn alloc_local_slot_bytes(&mut self, size_bytes: i32) -> i32;
    fn alloc_root_slot(&mut self) -> i32;
    fn alloc_root_slot_bytes(&mut self, size_bytes: i32) -> i32;
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

    fn alloc_local_slot_bytes(&mut self, size_bytes: i32) -> i32 {
        Self::alloc_local_slot_bytes(self, size_bytes)
    }

    fn alloc_root_slot(&mut self) -> i32 {
        Self::alloc_root_slot(self)
    }

    fn alloc_root_slot_bytes(&mut self, size_bytes: i32) -> i32 {
        Self::alloc_root_slot_bytes(self, size_bytes)
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
>
{
    const NAME: &'static str;

    fn validates_frame() -> bool {
        if T::requires_shadow_slots() {
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

/// Sealing module — only this crate can implement `SoundRoots` and
/// `SoundTransport`. This makes the set of valid GC configurations
/// closed: downstream code cannot declare new "sound" combinations.
mod sound {
    pub trait Sealed {}
}

/// Marker trait: `Self` is a sound root strategy for `L`.
///
/// Sound here means: at every safepoint, the GC can identify the exact
/// set of heap references the program depends on — no stale slots,
/// no conservative false-retention. Implementations are sealed; only
/// combinations verified correct in this crate are admitted.
pub trait SoundRoots<L: ValueLayout>: RootStrategy<L> + sound::Sealed {}

/// Marker trait: `Self` is a sound root transport for `(L, R)`.
pub trait SoundTransport<L: ValueLayout, R: RootStrategy<L>>:
    RootTransport<L, R> + sound::Sealed {}

impl sound::Sealed for PreciseStackRoots {}
impl sound::Sealed for StackMapRoots {}
impl sound::Sealed for ShadowStackRoots {}

impl<const TAG_BITS: u32> SoundRoots<LowBit<TAG_BITS>> for PreciseStackRoots {}
impl SoundRoots<NanBox> for PreciseStackRoots {}

impl<L, R> SoundTransport<L, R> for StackMapRoots
where
    L: ValueLayout,
    R: RootStrategy<L> + SoundRoots<L>,
{
}
impl<L, R> SoundTransport<L, R> for ShadowStackRoots
where
    L: ValueLayout,
    R: RootStrategy<L> + SoundRoots<L>,
{
}

/// Codegen configuration for JIT lowering. Controls value layout, root
/// strategy, calling convention, frame strategy, and safepoints.
///
/// `Roots` / `RootTransport` are gated by [`SoundRoots`] / [`SoundTransport`]
/// so that the only `CodegenConfig` types that typecheck are those whose
/// GC contract is verified safe — there is no runtime "oops" path.
pub trait CodegenConfig {
    type Layout: ValueLayout;
    type Roots: RootStrategy<Self::Layout> + SoundRoots<Self::Layout>;
    type RootTransport:
        RootTransport<Self::Layout, Self::Roots>
        + SoundTransport<Self::Layout, Self::Roots>;
    type CallingConvention: CallingConvention<Self::Layout>;
    type Frames: FrameStrategy<Self::Layout, Self::Roots, Self::CallingConvention>;
    type Safepoints: SafepointStrategy<
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

pub struct PreciseStackRoots;

impl<L: ValueLayout> RootStrategy<L> for PreciseStackRoots {
    const NAME: &'static str = "precise-stack-roots";
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

pub struct X64SysVCAbi;

impl<L: ValueLayout> CallingConvention<L> for X64SysVCAbi {
    const NAME: &'static str = "x86_64-sysv-c-abi";

    fn stack_align() -> usize {
        16
    }

    fn register_arg_limit() -> usize {
        6
    }
}

#[cfg(target_arch = "x86_64")]
pub type PlatformDefaultCc = X64SysVCAbi;

#[cfg(not(target_arch = "x86_64"))]
pub type PlatformDefaultCc = AArch64InternalCc;

pub struct StackSlotFrames;

impl<L: ValueLayout, R: RootStrategy<L>, C: CallingConvention<L>> FrameStrategy<L, R, C>
    for StackSlotFrames
{
    const NAME: &'static str = "stack-slot-frames";
    type Layout = StackFrameLayout;

    fn new_layout(block_count: usize) -> Self::Layout {
        StackFrameLayout::new(block_count)
    }

    fn supports_stack_maps() -> bool {
        true
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

impl<L> CodegenConfig for DefaultCodegenConfig<L>
where
    L: LayoutConfigDefaults,
    L::DefaultRoots: SoundRoots<L>,
    L::DefaultRootTransport: SoundTransport<L, L::DefaultRoots>,
{
    type Layout = L;
    type Roots = L::DefaultRoots;
    type RootTransport = L::DefaultRootTransport;
    type CallingConvention = PlatformDefaultCc;
    type Frames = StackSlotFrames;
    type Safepoints = CallbackSafepoints;
}

/// The blessed GC configuration for NaN-boxed dynamic languages.
///
/// Uses `PreciseStackRoots` + `StackMapRoots` + `StackSlotFrames` +
/// `StackMapSafepoints`. Together they guarantee the GC sees exactly the
/// slots live at each safepoint — no conservative over-retention from
/// stale spill slots. `StackSlotFrames` matches the interpreter-side
/// layout, so the same `NanBoxConfig` can feed both the JIT and
/// `ModuleInterpreter`.
pub struct NanBoxConfig;

impl CodegenConfig for NanBoxConfig {
    type Layout = NanBox;
    type Roots = PreciseStackRoots;
    type RootTransport = StackMapRoots;
    type CallingConvention = AArch64InternalCc;
    type Frames = StackSlotFrames;
    type Safepoints = StackMapSafepoints;
}

pub fn validate_codegen_config<C: CodegenConfig>() -> Result<(), ConfigError> {
    <C as CodegenConfig>::validate()
}

impl<const TAG_BITS: u32> ValueLayout for LowBit<TAG_BITS> {
    const NAME: &'static str = "low-bit";
}

impl<const TAG_BITS: u32> LayoutConfigDefaults for LowBit<TAG_BITS> {
    type DefaultRoots = PreciseStackRoots;
    type DefaultRootTransport = StackMapRoots;
}

impl ValueLayout for NanBox {
    const NAME: &'static str = "nan-box";
}

impl LayoutConfigDefaults for NanBox {
    type DefaultRoots = PreciseStackRoots;
    type DefaultRootTransport = StackMapRoots;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Compile-time assertion: the sound combinations we ship typecheck.
    fn _assert_sound<C: CodegenConfig>() {}
    #[test]
    fn sound_configs_typecheck() {
        _assert_sound::<DefaultCodegenConfig<LowBit<3>>>();
        _assert_sound::<NanBoxConfig>();
    }

    #[test]
    fn nan_box_default_config_is_valid() {
        assert!(validate_codegen_config::<DefaultCodegenConfig<NanBox>>().is_ok());
    }
}
