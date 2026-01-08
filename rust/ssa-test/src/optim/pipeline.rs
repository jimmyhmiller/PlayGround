//! Optimization pipeline for running multiple passes.

use crate::traits::InstructionFactory;
use crate::translator::SSATranslator;
use crate::validation::{validate_ssa, SSAViolation};

use super::analysis::AnalysisCache;
use super::pass::{OptimizationPass, PassResult, PassStats};
use super::passes::{
    DeadCodeElimination, CopyPropagation, ConstantFolding, ConstantPropagation,
    CommonSubexpressionElimination, ControlFlowSimplificationPass, JumpThreading, CfgCleanup,
};
use super::traits::{OptimizableValue, OptimizableInstruction, InstructionMutator};

/// Result from running the optimization pipeline.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// Whether any pass modified the IR
    pub changed: bool,
    /// Number of iterations run (for fixed-point mode)
    pub iterations: usize,
    /// Combined statistics from all passes
    pub stats: PassStats,
    /// Per-pass results (pass name, result)
    pub pass_results: Vec<(String, PassResult)>,
    /// Validation violations found after passes (if verification enabled)
    pub validation_errors: Vec<(String, Vec<String>)>,
}

impl PipelineResult {
    fn new() -> Self {
        PipelineResult {
            changed: false,
            iterations: 0,
            stats: PassStats::new(),
            pass_results: Vec::new(),
            validation_errors: Vec::new(),
        }
    }

    /// Check if any validation errors occurred.
    pub fn has_validation_errors(&self) -> bool {
        !self.validation_errors.is_empty()
    }

    /// Panic if validation errors were found, with detailed messages.
    pub fn assert_valid(&self) {
        if self.has_validation_errors() {
            let mut msg = String::from("SSA validation failed after optimization passes:\n");
            for (pass_name, errors) in &self.validation_errors {
                msg.push_str(&format!("\nAfter pass '{}':\n", pass_name));
                for err in errors {
                    msg.push_str(&format!("  - {}\n", err));
                }
            }
            panic!("{}", msg);
        }
    }
}

/// Optimization pipeline that runs multiple passes.
///
/// # Example
/// ```ignore
/// use ssa_lib::optim::prelude::*;
///
/// // Create pipeline with standard passes
/// let mut pipeline = OptimizationPipeline::standard();
///
/// // Or build custom pipeline
/// let mut pipeline = OptimizationPipeline::new();
/// pipeline.add_pass(DeadCodeElimination::new());
/// pipeline.add_pass(CopyPropagation::new());
///
/// // Run once
/// let result = pipeline.run(&mut translator);
///
/// // Or run until fixed point
/// let result = pipeline.run_until_fixed_point(&mut translator, 10);
/// ```
pub struct OptimizationPipeline<V, I, F>
where
    V: OptimizableValue,
    I: OptimizableInstruction<Value = V>,
    F: InstructionFactory<Instr = I>,
{
    passes: Vec<Box<dyn OptimizationPass<V, I, F>>>,
    cache: AnalysisCache<V, I>,
    /// Whether to validate SSA properties after each pass
    verify: bool,
}

impl<V, I, F> OptimizationPipeline<V, I, F>
where
    V: OptimizableValue,
    I: OptimizableInstruction<Value = V>,
    F: InstructionFactory<Instr = I>,
{
    /// Create an empty pipeline.
    pub fn new() -> Self {
        OptimizationPipeline {
            passes: Vec::new(),
            cache: AnalysisCache::new(),
            verify: false,
        }
    }

    /// Create an empty pipeline with verification enabled.
    ///
    /// When verification is enabled, SSA properties are validated after
    /// each optimization pass. This helps catch bugs in optimization passes
    /// but adds overhead.
    pub fn new_with_verify() -> Self {
        OptimizationPipeline {
            passes: Vec::new(),
            cache: AnalysisCache::new(),
            verify: true,
        }
    }

    /// Enable or disable verification after each pass.
    pub fn set_verify(&mut self, verify: bool) {
        self.verify = verify;
    }

    /// Check if verification is enabled.
    pub fn verify_enabled(&self) -> bool {
        self.verify
    }

    /// Add a pass to the pipeline.
    pub fn add_pass<P: OptimizationPass<V, I, F> + 'static>(&mut self, pass: P) {
        self.passes.push(Box::new(pass));
    }

    /// Run all passes once.
    pub fn run(&mut self, translator: &mut SSATranslator<V, I, F>) -> PipelineResult {
        self.run_internal(translator, true)
    }

    /// Internal run that optionally skips final validation (for use in fixed-point iteration)
    fn run_internal(&mut self, translator: &mut SSATranslator<V, I, F>, do_final_validation: bool) -> PipelineResult {
        let mut result = PipelineResult::new();
        result.iterations = 1;

        for pass in &mut self.passes {
            let pass_name = pass.name().to_string();
            let pass_result = pass.run(translator, &mut self.cache);

            if pass_result.changed {
                result.changed = true;
                // Invalidate cached analyses
                self.cache.invalidate(&pass.invalidates());
            }

            // Verify SSA properties after each pass if enabled
            if self.verify {
                let violations = validate_ssa(translator);
                // During intermediate passes, filter out DeadPhi (DCE should handle)
                // The final validation will catch any remaining issues
                let critical_violations: Vec<_> = violations
                    .into_iter()
                    .filter(|v| !matches!(v, SSAViolation::DeadPhi { .. }))
                    .collect();

                if !critical_violations.is_empty() {
                    let error_strings: Vec<String> = critical_violations
                        .iter()
                        .map(|v| format!("{}", v))
                        .collect();
                    result.validation_errors.push((pass_name.clone(), error_strings));
                }
            }

            result.stats.merge(&pass_result.stats);
            result.pass_results.push((pass_name, pass_result));
        }

        // Final validation - check EVERYTHING including things we filtered during passes
        if self.verify && do_final_validation {
            let final_violations = Self::final_validation(translator);
            if !final_violations.is_empty() {
                result.validation_errors.push(("final".to_string(), final_violations));
            }
        }

        result
    }

    /// Comprehensive final validation after all passes complete.
    /// This checks everything - no filtering.
    fn final_validation(translator: &SSATranslator<V, I, F>) -> Vec<String> {
        let mut errors = Vec::new();

        // Check SSA properties (including dead phis - they should be gone now)
        let ssa_violations = validate_ssa(translator);
        for v in ssa_violations {
            errors.push(format!("SSA: {}", v));
        }

        // Check for empty blocks
        for block in &translator.blocks {
            if block.instructions.is_empty() {
                errors.push(format!("Empty block: Block {:?} has no instructions", block.id));
            }
        }

        // Check for unreachable blocks (except entry)
        let entry = crate::types::BlockId(0);
        for block in &translator.blocks {
            if block.id != entry && block.predecessors.is_empty() {
                errors.push(format!("Unreachable block: Block {:?} has no predecessors", block.id));
            }
        }

        errors
    }

    /// Run passes repeatedly until no changes are made or max iterations reached.
    pub fn run_until_fixed_point(
        &mut self,
        translator: &mut SSATranslator<V, I, F>,
        max_iterations: usize,
    ) -> PipelineResult {
        let mut result = PipelineResult::new();

        for iteration in 0..max_iterations {
            result.iterations = iteration + 1;

            // Don't do final validation on intermediate iterations
            let iter_result = self.run_internal(translator, false);
            result.stats.merge(&iter_result.stats);
            result.pass_results.extend(iter_result.pass_results);
            result.validation_errors.extend(iter_result.validation_errors);

            if !iter_result.changed {
                // Fixed point reached
                break;
            }

            result.changed = true;
        }

        // Do final validation only once at the end
        if self.verify {
            let final_violations = Self::final_validation(translator);
            if !final_violations.is_empty() {
                result.validation_errors.push(("final".to_string(), final_violations));
            }
        }

        result
    }

    /// Get mutable access to the analysis cache.
    pub fn cache_mut(&mut self) -> &mut AnalysisCache<V, I> {
        &mut self.cache
    }

    /// Get read access to the analysis cache.
    pub fn cache(&self) -> &AnalysisCache<V, I> {
        &self.cache
    }

    /// Clear all passes from the pipeline.
    pub fn clear(&mut self) {
        self.passes.clear();
    }

    /// Get the number of passes in the pipeline.
    pub fn len(&self) -> usize {
        self.passes.len()
    }

    /// Check if the pipeline is empty.
    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
    }

    /// Create a standard optimization pipeline.
    ///
    /// Includes: Copy Propagation, Constant Propagation, DCE
    ///
    /// This is a good default for most use cases.
    pub fn standard() -> Self
    where
        DeadCodeElimination: OptimizationPass<V, I, F>,
        CopyPropagation: OptimizationPass<V, I, F>,
        ConstantPropagation: OptimizationPass<V, I, F>,
    {
        let mut pipeline = Self::new();
        pipeline.add_pass(CopyPropagation::new());
        pipeline.add_pass(ConstantPropagation::new());
        pipeline.add_pass(DeadCodeElimination::new());
        pipeline
    }

    /// Create an aggressive optimization pipeline.
    ///
    /// Includes all passes: Copy Prop, Const Prop, Const Fold, Control Flow Simplify,
    /// Jump Threading, CFG Cleanup, CSE, DCE
    ///
    /// Use `run_until_fixed_point` for best results.
    pub fn aggressive() -> Self
    where
        F: InstructionMutator,
        DeadCodeElimination: OptimizationPass<V, I, F>,
        CopyPropagation: OptimizationPass<V, I, F>,
        ConstantPropagation: OptimizationPass<V, I, F>,
        ConstantFolding: OptimizationPass<V, I, F>,
        ControlFlowSimplificationPass: OptimizationPass<V, I, F>,
        JumpThreading: OptimizationPass<V, I, F>,
        CfgCleanup: OptimizationPass<V, I, F>,
        CommonSubexpressionElimination: OptimizationPass<V, I, F>,
    {
        let mut pipeline = Self::new();
        // Order matters: propagate first, then fold, simplify control flow,
        // thread jumps, cleanup CFG, then eliminate
        pipeline.add_pass(CopyPropagation::new());
        pipeline.add_pass(ConstantPropagation::new());
        pipeline.add_pass(ConstantFolding::new());
        pipeline.add_pass(ControlFlowSimplificationPass::new());
        pipeline.add_pass(JumpThreading::new());
        pipeline.add_pass(CfgCleanup::new());
        pipeline.add_pass(CommonSubexpressionElimination::new());
        pipeline.add_pass(DeadCodeElimination::new());
        pipeline
    }
}

impl<V, I, F> Default for OptimizationPipeline<V, I, F>
where
    V: OptimizableValue,
    I: OptimizableInstruction<Value = V>,
    F: InstructionFactory<Instr = I>,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_result() {
        let result = PipelineResult::new();
        assert!(!result.changed);
        assert_eq!(result.iterations, 0);
        assert!(result.pass_results.is_empty());
        assert!(!result.has_validation_errors());
    }

    #[test]
    fn test_stats_merge() {
        let mut stats1 = PassStats::new();
        stats1.instructions_removed = 5;

        let mut stats2 = PassStats::new();
        stats2.instructions_removed = 3;
        stats2.values_propagated = 2;

        stats1.merge(&stats2);

        assert_eq!(stats1.instructions_removed, 8);
        assert_eq!(stats1.values_propagated, 2);
    }

    #[test]
    fn test_validation_error_detection() {
        let mut result = PipelineResult::new();
        assert!(!result.has_validation_errors());

        result.validation_errors.push((
            "test_pass".to_string(),
            vec!["Some error".to_string()],
        ));
        assert!(result.has_validation_errors());
    }
}
