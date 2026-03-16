mod framechain;
mod jit;
mod ptr_policy;
mod root_strategies;
mod stackmap;

pub use dynalloc::PtrPolicy;
pub use framechain::FrameChainRootManager;
pub use jit::{
    FrameScanJitTransport, JitSafepointSession, ShadowStackJitTransport, StackMapJitTransport,
    active_jit_safepoint_handler,
};
pub use ptr_policy::{LowBitPtrPolicy, NanBoxPtrPolicy};
pub use root_strategies::{
    FrameChainPreciseRoots, MutatorConservativeRoots, MutatorPreciseRoots,
};
pub use stackmap::MutatorRootManager;

#[cfg(test)]
mod tests;
