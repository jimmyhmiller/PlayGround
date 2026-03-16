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

pub trait ExecutionConfig {
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

pub fn validate_execution_config<C: ExecutionConfig>() -> Result<(), ConfigError> {
    C::validate()
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

pub struct DefaultExecutionConfig<L: LayoutConfigDefaults>(PhantomData<L>);

impl<L: LayoutConfigDefaults> ExecutionConfig for DefaultExecutionConfig<L> {
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

    impl ExecutionConfig for InvalidNanBoxConfig {
        type Layout = NanBox;
        type Roots = PreciseStackRoots;
        type RootTransport = FrameScanRoots;
        type CallingConvention = AArch64InternalCc;
        type Frames = StackSlotFrames;
        type Safepoints = CallbackSafepoints;
    }

    struct InvalidShadowTransportConfig;

    impl ExecutionConfig for InvalidShadowTransportConfig {
        type Layout = LowBit<3>;
        type Roots = PreciseStackRoots;
        type RootTransport = ShadowStackRoots;
        type CallingConvention = AArch64InternalCc;
        type Frames = StackSlotFrames;
        type Safepoints = CallbackSafepoints;
    }

    struct InvalidStackMapSafepointConfig;

    impl ExecutionConfig for InvalidStackMapSafepointConfig {
        type Layout = LowBit<3>;
        type Roots = PreciseStackRoots;
        type RootTransport = FrameScanRoots;
        type CallingConvention = AArch64InternalCc;
        type Frames = StackSlotFrames;
        type Safepoints = StackMapSafepoints;
    }

    #[test]
    fn low_bit_default_config_is_valid() {
        assert!(validate_execution_config::<DefaultExecutionConfig<LowBit<3>>>().is_ok());
    }

    #[test]
    fn nan_box_default_config_is_valid() {
        assert!(validate_execution_config::<DefaultExecutionConfig<NanBox>>().is_ok());
    }

    #[test]
    fn precise_roots_reject_nan_box_layout() {
        let err = validate_execution_config::<InvalidNanBoxConfig>().unwrap_err();
        assert_eq!(err.message, "root strategy is incompatible with value layout");
    }

    #[test]
    fn shadow_stack_transport_requires_shadow_frame_support() {
        let err = validate_execution_config::<InvalidShadowTransportConfig>().unwrap_err();
        assert_eq!(err.message, "root transport requires shadow-root frame support");
    }

    #[test]
    fn stack_map_safepoints_require_stack_map_transport() {
        let err = validate_execution_config::<InvalidStackMapSafepointConfig>().unwrap_err();
        assert_eq!(err.message, "stack-map safepoints require stack-map root transport");
    }
}
