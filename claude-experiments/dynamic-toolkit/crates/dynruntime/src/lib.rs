mod framechain;
mod ptr_policy;
mod root_strategies;
mod stackmap;

pub use dynalloc::PtrPolicy;
pub use framechain::FrameChainRootManager;
pub use ptr_policy::{LowBitPtrPolicy, NanBoxPtrPolicy};
pub use root_strategies::{
    FrameChainPreciseRoots, MutatorConservativeRoots, MutatorPreciseRoots,
};
pub use stackmap::MutatorRootManager;

#[cfg(test)]
mod tests;
