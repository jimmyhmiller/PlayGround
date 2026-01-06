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
//! Then use the [`OptimizationPipeline`] to run passes:
//!
//! ```ignore
//! use ssa_lib::optim::prelude::*;
//!
//! let mut pipeline = OptimizationPipeline::standard();
//! pipeline.run(&mut translator);
//! ```
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

// Re-export main types
pub use traits::{OptimizableValue, OptimizableInstruction, InstructionMutator, ExpressionKey};
pub use pass::{OptimizationPass, PassResult, PassStats, Invalidations};
pub use pipeline::OptimizationPipeline;
pub use analysis::{AnalysisCache, LivenessAnalysis, UseDefChains};

// Re-export regalloc types
pub use regalloc::{
    PhysicalRegister, RegisterClass, TargetArchitecture,
    RegisterConstraint, OperandConstraints, HasRegisterConstraints,
    ProgramPoint, LiveRange, LiveInterval, Location, IntervalAnalysis,
    PhiElimination,
    LinearScanAllocator, LinearScanConfig, AllocationResult, AllocationStats,
    SpillCodeFactory,
    LoweredOperand, LoweredInstruction, LoweredBlock, LoweredFunction,
};

/// Prelude for convenient imports
pub mod prelude {
    pub use super::traits::{OptimizableValue, OptimizableInstruction, InstructionMutator, ExpressionKey};
    pub use super::pass::{OptimizationPass, PassResult, PassStats, Invalidations};
    pub use super::pipeline::OptimizationPipeline;
    pub use super::analysis::{AnalysisCache, LivenessAnalysis, UseDefChains};
    pub use super::passes::*;
    pub use super::regalloc::*;
}
