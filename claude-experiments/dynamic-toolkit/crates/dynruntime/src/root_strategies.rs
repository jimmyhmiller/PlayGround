use dynexec::{RootStrategy, ValueLayout};

pub struct MutatorPreciseRoots;
pub struct FrameChainPreciseRoots;

impl<L: ValueLayout> RootStrategy<L> for MutatorPreciseRoots {
    const NAME: &'static str = "mutator-precise-roots";
}

impl<L: ValueLayout> RootStrategy<L> for FrameChainPreciseRoots {
    const NAME: &'static str = "framechain-precise-roots";
}
