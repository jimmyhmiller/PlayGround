//! Safe, ergonomic optimizer API.
//!
//! This module provides a builder-style API that makes it easy to do the right thing:
//! - Verification is ON by default
//! - Dead code elimination runs automatically at the end (unless disabled)
//! - Returns `Result` so validation errors are hard to ignore
//!
//! # Example
//!
//! ```ignore
//! use ssa_lib::optim::Optimizer;
//!
//! // The easy, correct path - DCE auto-included, verification on:
//! Optimizer::new(&mut translator)
//!     .copy_propagation()
//!     .constant_folding()
//!     .run()?;
//!
//! // Run until fixed point:
//! Optimizer::new(&mut translator)
//!     .copy_propagation()
//!     .constant_propagation()
//!     .constant_folding()
//!     .cse()
//!     .run_to_fixed_point(10)?;
//!
//! // Opt out of automatic cleanup (must be explicit):
//! Optimizer::new(&mut translator)
//!     .copy_propagation()
//!     .no_auto_cleanup()  // Explicit: I know what I'm doing
//!     .run()?;
//!
//! // Opt out of verification (must be explicit):
//! Optimizer::new(&mut translator)
//!     .copy_propagation()
//!     .no_verification()  // Explicit: I accept the risk
//!     .run_unchecked();   // Returns PipelineResult, not Result
//! ```

use std::fmt;

use crate::traits::InstructionFactory;
use crate::translator::SSATranslator;

use super::pass::OptimizationPass;
use super::passes::{
    CommonSubexpressionElimination, ConstantFolding, ConstantPropagation, CopyPropagation,
    DeadCodeElimination,
};
use super::pipeline::{OptimizationPipeline, PipelineResult};
use super::traits::{InstructionMutator, OptimizableInstruction, OptimizableValue};

/// Error returned when optimization validation fails.
#[derive(Debug, Clone)]
pub struct OptimizationError {
    /// Which pass (or "final") produced the error
    pub pass: String,
    /// The validation errors found
    pub errors: Vec<String>,
}

impl fmt::Display for OptimizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Optimization validation failed after '{}':", self.pass)?;
        for err in &self.errors {
            writeln!(f, "  - {}", err)?;
        }
        Ok(())
    }
}

impl std::error::Error for OptimizationError {}

/// Result type for optimization operations.
pub type OptimizationResult = Result<PipelineResult, OptimizationError>;

/// Safe optimizer builder with good defaults.
///
/// By default:
/// - Verification is **ON**
/// - Dead code elimination runs **automatically** at the end
///
/// These can be explicitly disabled, but you have to be intentional about it.
pub struct Optimizer<'a, V, I, F>
where
    V: OptimizableValue,
    I: OptimizableInstruction<Value = V>,
    F: InstructionFactory<Instr = I>,
{
    translator: &'a mut SSATranslator<V, I, F>,
    pipeline: OptimizationPipeline<V, I, F>,
    auto_cleanup: bool,
    verify: bool,
}

impl<'a, V, I, F> Optimizer<'a, V, I, F>
where
    V: OptimizableValue,
    I: OptimizableInstruction<Value = V>,
    F: InstructionFactory<Instr = I>,
{
    /// Create a new optimizer for the given translator.
    ///
    /// By default:
    /// - Verification is ON
    /// - DCE runs automatically at the end
    pub fn new(translator: &'a mut SSATranslator<V, I, F>) -> Self {
        let mut pipeline = OptimizationPipeline::new();
        pipeline.set_verify(true); // ON by default

        Optimizer {
            translator,
            pipeline,
            auto_cleanup: true, // DCE auto-included by default
            verify: true,
        }
    }

    /// Add copy propagation pass.
    pub fn copy_propagation(mut self) -> Self
    where
        CopyPropagation: OptimizationPass<V, I, F>,
    {
        self.pipeline.add_pass(CopyPropagation::new());
        self
    }

    /// Add constant propagation pass.
    pub fn constant_propagation(mut self) -> Self
    where
        ConstantPropagation: OptimizationPass<V, I, F>,
    {
        self.pipeline.add_pass(ConstantPropagation::new());
        self
    }

    /// Add constant folding pass.
    pub fn constant_folding(mut self) -> Self
    where
        F: InstructionMutator,
        ConstantFolding: OptimizationPass<V, I, F>,
    {
        self.pipeline.add_pass(ConstantFolding::new());
        self
    }

    /// Add common subexpression elimination pass.
    pub fn cse(mut self) -> Self
    where
        CommonSubexpressionElimination: OptimizationPass<V, I, F>,
    {
        self.pipeline.add_pass(CommonSubexpressionElimination::new());
        self
    }

    /// Add dead code elimination pass explicitly.
    ///
    /// Note: DCE is automatically added at the end unless `no_auto_cleanup()` is called.
    /// Use this if you want DCE to run at a specific point in the pipeline.
    pub fn dce(mut self) -> Self
    where
        DeadCodeElimination: OptimizationPass<V, I, F>,
    {
        self.pipeline.add_pass(DeadCodeElimination::new());
        self
    }

    /// Add a custom optimization pass.
    pub fn pass<P: OptimizationPass<V, I, F> + 'static>(mut self, pass: P) -> Self {
        self.pipeline.add_pass(pass);
        self
    }

    /// Disable automatic cleanup (DCE at end).
    ///
    /// **Warning**: Without DCE, dead code and dead phis may remain,
    /// which can cause issues in later compilation stages.
    ///
    /// Only use this if you:
    /// - Are adding DCE manually at a specific point
    /// - Know your passes don't create dead code
    /// - Are debugging and want to see intermediate state
    pub fn no_auto_cleanup(mut self) -> Self {
        self.auto_cleanup = false;
        self
    }

    /// Disable verification.
    ///
    /// **Warning**: Without verification, invalid IR may not be detected.
    /// Use `run_unchecked()` when verification is disabled.
    ///
    /// Only use this if you:
    /// - Are debugging and want maximum performance
    /// - Have already verified correctness another way
    pub fn no_verification(mut self) -> Self {
        self.verify = false;
        self.pipeline.set_verify(false);
        self
    }

    /// Run all passes once and return Result.
    ///
    /// Returns `Err` if validation fails, making errors hard to ignore.
    pub fn run(mut self) -> OptimizationResult
    where
        DeadCodeElimination: OptimizationPass<V, I, F>,
    {
        // Add DCE at the end if auto_cleanup is enabled
        if self.auto_cleanup {
            self.pipeline.add_pass(DeadCodeElimination::new());
        }

        let result = self.pipeline.run(self.translator);
        Self::check_result(result)
    }

    /// Run passes until fixed point and return Result.
    ///
    /// Returns `Err` if validation fails, making errors hard to ignore.
    pub fn run_to_fixed_point(mut self, max_iterations: usize) -> OptimizationResult
    where
        DeadCodeElimination: OptimizationPass<V, I, F>,
    {
        // Add DCE at the end if auto_cleanup is enabled
        if self.auto_cleanup {
            self.pipeline.add_pass(DeadCodeElimination::new());
        }

        let result = self.pipeline.run_until_fixed_point(self.translator, max_iterations);
        Self::check_result(result)
    }

    /// Run without checking results (for use with `no_verification()`).
    ///
    /// Returns `PipelineResult` directly without converting errors to `Result`.
    /// Use this when you've disabled verification and want to inspect the raw result.
    pub fn run_unchecked(mut self) -> PipelineResult
    where
        DeadCodeElimination: OptimizationPass<V, I, F>,
    {
        if self.auto_cleanup {
            self.pipeline.add_pass(DeadCodeElimination::new());
        }
        self.pipeline.run(self.translator)
    }

    /// Run to fixed point without checking results.
    pub fn run_to_fixed_point_unchecked(mut self, max_iterations: usize) -> PipelineResult
    where
        DeadCodeElimination: OptimizationPass<V, I, F>,
    {
        if self.auto_cleanup {
            self.pipeline.add_pass(DeadCodeElimination::new());
        }
        self.pipeline.run_until_fixed_point(self.translator, max_iterations)
    }

    /// Convert PipelineResult to Result, extracting first error if any.
    fn check_result(result: PipelineResult) -> OptimizationResult {
        if let Some((pass, errors)) = result.validation_errors.first() {
            Err(OptimizationError {
                pass: pass.clone(),
                errors: errors.clone(),
            })
        } else {
            Ok(result)
        }
    }
}

/// Convenience function for the most common optimization setup.
///
/// Runs copy propagation, constant propagation, and DCE with verification.
///
/// # Example
/// ```ignore
/// use ssa_lib::optim::optimize;
///
/// optimize(&mut translator)?;
/// ```
pub fn optimize<V, I, F>(translator: &mut SSATranslator<V, I, F>) -> OptimizationResult
where
    V: OptimizableValue,
    I: OptimizableInstruction<Value = V>,
    F: InstructionFactory<Instr = I>,
    CopyPropagation: OptimizationPass<V, I, F>,
    ConstantPropagation: OptimizationPass<V, I, F>,
    DeadCodeElimination: OptimizationPass<V, I, F>,
{
    Optimizer::new(translator)
        .copy_propagation()
        .constant_propagation()
        .run()
}

/// Convenience function for aggressive optimization.
///
/// Runs all standard passes until fixed point with verification.
///
/// # Example
/// ```ignore
/// use ssa_lib::optim::optimize_aggressive;
///
/// optimize_aggressive(&mut translator, 10)?;
/// ```
pub fn optimize_aggressive<V, I, F>(
    translator: &mut SSATranslator<V, I, F>,
    max_iterations: usize,
) -> OptimizationResult
where
    V: OptimizableValue,
    I: OptimizableInstruction<Value = V>,
    F: InstructionFactory<Instr = I> + InstructionMutator,
    CopyPropagation: OptimizationPass<V, I, F>,
    ConstantPropagation: OptimizationPass<V, I, F>,
    ConstantFolding: OptimizationPass<V, I, F>,
    CommonSubexpressionElimination: OptimizationPass<V, I, F>,
    DeadCodeElimination: OptimizationPass<V, I, F>,
{
    Optimizer::new(translator)
        .copy_propagation()
        .constant_propagation()
        .constant_folding()
        .cse()
        .run_to_fixed_point(max_iterations)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_error_display() {
        let err = OptimizationError {
            pass: "test_pass".to_string(),
            errors: vec!["Error 1".to_string(), "Error 2".to_string()],
        };
        let display = format!("{}", err);
        assert!(display.contains("test_pass"));
        assert!(display.contains("Error 1"));
        assert!(display.contains("Error 2"));
    }
}
