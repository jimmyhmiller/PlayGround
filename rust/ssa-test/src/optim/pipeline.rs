//! Optimization pipeline for running multiple passes.

use crate::traits::InstructionFactory;
use crate::translator::SSATranslator;

use super::analysis::AnalysisCache;
use super::pass::{OptimizationPass, PassResult, PassStats};
use super::passes::{
    DeadCodeElimination, CopyPropagation, ConstantFolding, ConstantPropagation,
    CommonSubexpressionElimination,
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
}

impl PipelineResult {
    fn new() -> Self {
        PipelineResult {
            changed: false,
            iterations: 0,
            stats: PassStats::new(),
            pass_results: Vec::new(),
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
        }
    }

    /// Add a pass to the pipeline.
    pub fn add_pass<P: OptimizationPass<V, I, F> + 'static>(&mut self, pass: P) {
        self.passes.push(Box::new(pass));
    }

    /// Run all passes once.
    pub fn run(&mut self, translator: &mut SSATranslator<V, I, F>) -> PipelineResult {
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

            result.stats.merge(&pass_result.stats);
            result.pass_results.push((pass_name, pass_result));
        }

        result
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

            let iter_result = self.run(translator);
            result.stats.merge(&iter_result.stats);
            result.pass_results.extend(iter_result.pass_results);

            if !iter_result.changed {
                // Fixed point reached
                break;
            }

            result.changed = true;
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
    /// Includes all passes: Copy Prop, Const Prop, Const Fold, CSE, DCE
    ///
    /// Use `run_until_fixed_point` for best results.
    pub fn aggressive() -> Self
    where
        F: InstructionMutator,
        DeadCodeElimination: OptimizationPass<V, I, F>,
        CopyPropagation: OptimizationPass<V, I, F>,
        ConstantPropagation: OptimizationPass<V, I, F>,
        ConstantFolding: OptimizationPass<V, I, F>,
        CommonSubexpressionElimination: OptimizationPass<V, I, F>,
    {
        let mut pipeline = Self::new();
        // Order matters: propagate first, then fold, then eliminate
        pipeline.add_pass(CopyPropagation::new());
        pipeline.add_pass(ConstantPropagation::new());
        pipeline.add_pass(ConstantFolding::new());
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
}
