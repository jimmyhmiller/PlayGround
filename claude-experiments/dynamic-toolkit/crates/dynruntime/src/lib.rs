mod engine;
mod framechain;
mod jit;
mod ptr_policy;
mod root_strategies;
mod stackmap;

pub use dynalloc::PtrPolicy;
pub use engine::{
    AlwaysCompile, CallCountTier, ExecutionEngine, ExecutionError, ExecutionResult, NeverCompile,
    TierPolicy, default_engine,
};
pub use framechain::FrameChainRootManager;
pub use jit::{
    GcPolicy, JitExecutionResult, JitFrameControl, JitFrameControlError,
    JitFrameSliceRuntime, JitRootTransportRuntime, JitSafepointSession, ScopedJitRoot,
    ScopedJitRoots,
    ResumeWithInterpreterError, ShadowStackJitTransport, StackMapJitTransport,
    active_jit_safepoint_handler, decode_frame_control_outcome,
    execute_jit_function, execute_jit_function_to_terminal, execute_jit_module_function,
    execute_jit_module_function_to_terminal, materialize_capture_slice,
    resume_stored_slice_with_interpreter, resume_stored_slice_with_jit,
    resume_stored_slice_with_jit_module, with_registered_active_jit_roots,
};
pub use ptr_policy::{LowBitPtrPolicy, NanBoxPtrPolicy};
pub use root_strategies::{
    FrameChainPreciseRoots, MutatorPreciseRoots,
};
pub use stackmap::MutatorRootManager;

#[cfg(test)]
mod tests;
