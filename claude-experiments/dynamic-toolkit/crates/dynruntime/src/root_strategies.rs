use dynexec::{RootPrecision, RootStrategy, ValueLayout};

pub struct MutatorPreciseRoots;
pub struct MutatorConservativeRoots;
pub struct FrameChainPreciseRoots;

impl<L: ValueLayout> RootStrategy<L> for MutatorPreciseRoots {
    const NAME: &'static str = "mutator-precise-roots";

    fn precision() -> RootPrecision {
        RootPrecision::PreciseSlots
    }
}

impl<L: ValueLayout> RootStrategy<L> for MutatorConservativeRoots {
    const NAME: &'static str = "mutator-conservative-roots";

    fn precision() -> RootPrecision {
        RootPrecision::ConservativeWords
    }
}

impl<L: ValueLayout> RootStrategy<L> for FrameChainPreciseRoots {
    const NAME: &'static str = "framechain-precise-roots";

    fn precision() -> RootPrecision {
        RootPrecision::PreciseSlots
    }
}
