//! Generic SSA Optimization Framework
//!
//! This module provides a trait-based optimization framework that allows consumers
//! to implement interfaces and receive generic optimization passes "for free."
//!
//! # Overview
//!
//! To use optimizations with your IR, implement these additional traits:
//!
//! - [`OptimizableValue`] - Extends `SsaValue` with constant handling
//! - [`OptimizableInstruction`] - Extends `SsaInstruction` with optimization metadata
//!
//! Then use the [`Optimizer`] for safe, easy optimization:
//!
//! ```ignore
//! use ssa_lib::optim::Optimizer;
//!
//! // Safe API with verification ON and DCE auto-included:
//! Optimizer::new(&mut translator)
//!     .copy_propagation()
//!     .constant_folding()
//!     .run()?;  // Returns Result - errors hard to ignore
//!
//! // Or use convenience functions:
//! use ssa_lib::optim::optimize;
//! optimize(&mut translator)?;
//! ```
//!
//! For more control, use [`OptimizationPipeline`] directly.
//!
//! # Available Passes
//!
//! - **Dead Code Elimination (DCE)**: Removes unused definitions
//! - **Copy Propagation**: Replaces copies with original values
//! - **Constant Folding**: Evaluates constant expressions at compile time
//! - **Constant Propagation**: Replaces variables with known constant values
//! - **Common Subexpression Elimination (CSE)**: Reuses previously computed expressions

pub mod traits;
pub mod pass;
pub mod pipeline;
pub mod analysis;
pub mod passes;
pub mod regalloc;
pub mod optimizer;
pub mod lowering;

// Re-export main types
pub use traits::{OptimizableValue, OptimizableInstruction, InstructionMutator, ExpressionKey, BranchHint};
pub use pass::{OptimizationPass, PassResult, PassStats, Invalidations};
pub use pipeline::{OptimizationPipeline, PipelineResult};
pub use analysis::{AnalysisCache, LivenessAnalysis, UseDefChains};
pub use optimizer::{Optimizer, OptimizationError, OptimizationResult, optimize, optimize_aggressive};

// Re-export regalloc types
pub use regalloc::{
    PhysicalRegister, RegisterClass, TargetArchitecture,
    RegisterConstraint, OperandConstraints, HasRegisterConstraints,
    ProgramPoint, LiveRange, LiveInterval, Location, IntervalAnalysis,
    PhiElimination, eliminate_trampolines,
    LinearScanAllocator, LinearScanConfig, AllocationResult, AllocationStats,
    SpillCodeFactory,
    LoweredOperand, LoweredInstruction, LoweredBlock, LoweredFunction,
};

// Re-export lowering types
pub use lowering::{
    BlockLayoutStrategy, DefaultBlockLayout, ExtTspBlockLayout, LoweringContext,
    CodeEmitter, EmitContext, LoweredTerminator, FallThroughChoice,
    FallThroughDecision, EdgeKind, DebugEmitter, LoweredBlockInfo,
};

/// Prelude for convenient imports
pub mod prelude {
    pub use super::traits::{OptimizableValue, OptimizableInstruction, InstructionMutator, ExpressionKey, BranchHint};
    pub use super::pass::{OptimizationPass, PassResult, PassStats, Invalidations};
    pub use super::pipeline::{OptimizationPipeline, PipelineResult};
    pub use super::analysis::{AnalysisCache, LivenessAnalysis, UseDefChains};
    pub use super::optimizer::{Optimizer, OptimizationError, OptimizationResult, optimize, optimize_aggressive};
    pub use super::passes::*;
    pub use super::regalloc::*;
    pub use super::lowering::*;
}
