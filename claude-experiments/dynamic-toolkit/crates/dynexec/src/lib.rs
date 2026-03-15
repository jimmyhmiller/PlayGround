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
pub enum SlotClass {
    IncomingArg,
    Local,
    Spill,
    OutgoingArg,
    Root,
    SavedReg,
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

pub trait CallingConvention<L: ValueLayout> {
    const NAME: &'static str;

    fn stack_align() -> usize;
    fn register_arg_limit() -> usize;
}

pub trait FrameStrategy<L: ValueLayout, R: RootStrategy<L>, C: CallingConvention<L>> {
    const NAME: &'static str;

    fn stack_align() -> usize {
        C::stack_align()
    }

    fn exposes_slot_class(_class: SlotClass) -> bool {
        true
    }

    fn validate() -> Result<(), ConfigError> {
        Ok(())
    }
}

pub trait SafepointStrategy<
    L: ValueLayout,
    R: RootStrategy<L>,
    F: FrameStrategy<L, R, C>,
    C: CallingConvention<L>,
> {
    const NAME: &'static str;

    fn validates_frame() -> bool {
        F::exposes_slot_class(SlotClass::Root) || R::precision() == RootPrecision::ConservativeWords
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
    type CallingConvention: CallingConvention<Self::Layout>;
    type Frames: FrameStrategy<Self::Layout, Self::Roots, Self::CallingConvention>;
    type Safepoints:
        SafepointStrategy<Self::Layout, Self::Roots, Self::Frames, Self::CallingConvention>;

    fn validate() -> Result<(), ConfigError> {
        Self::Roots::validate()?;
        Self::Frames::validate()?;
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
}

pub struct ShadowStackFrames;

impl<L: ValueLayout, R: RootStrategy<L>, C: CallingConvention<L>> FrameStrategy<L, R, C>
    for ShadowStackFrames
{
    const NAME: &'static str = "shadow-stack-frames";
}

pub struct CallbackSafepoints;

impl<L: ValueLayout, R: RootStrategy<L>, F: FrameStrategy<L, R, C>, C: CallingConvention<L>>
    SafepointStrategy<L, R, F, C> for CallbackSafepoints
{
    const NAME: &'static str = "callback-safepoints";
}

pub struct DefaultExecutionConfig<L: LayoutConfigDefaults>(PhantomData<L>);

impl<L: LayoutConfigDefaults> ExecutionConfig for DefaultExecutionConfig<L> {
    type Layout = L;
    type Roots = L::DefaultRoots;
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
}

impl ValueLayout for NanBox {
    const NAME: &'static str = "nan-box";

    fn root_precision_hint() -> RootPrecision {
        RootPrecision::ConservativeWords
    }
}

impl LayoutConfigDefaults for NanBox {
    type DefaultRoots = ConservativeWordRoots;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct InvalidNanBoxConfig;

    impl ExecutionConfig for InvalidNanBoxConfig {
        type Layout = NanBox;
        type Roots = PreciseStackRoots;
        type CallingConvention = AArch64InternalCc;
        type Frames = StackSlotFrames;
        type Safepoints = CallbackSafepoints;
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
}
