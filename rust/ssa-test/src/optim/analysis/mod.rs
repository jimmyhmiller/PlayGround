//! Analysis infrastructure for optimization passes.
//!
//! This module provides:
//! - [`AnalysisCache`] - Cached analyses with invalidation
//! - [`LivenessAnalysis`] - Live variable analysis
//! - [`UseDefChains`] - Use-def chain tracking

pub mod liveness;
pub mod use_def;

pub use liveness::LivenessAnalysis;
pub use use_def::UseDefChains;

use std::marker::PhantomData;

use crate::traits::InstructionFactory;
use crate::translator::SSATranslator;

use super::traits::{OptimizableValue, OptimizableInstruction};
use super::pass::Invalidations;

/// Cache for analysis results.
///
/// Analyses are computed on-demand and cached. When the IR is modified,
/// the relevant analyses are invalidated.
///
/// # Example
/// ```ignore
/// let mut cache = AnalysisCache::new();
///
/// // First access computes the analysis
/// let liveness = cache.liveness(&translator);
///
/// // Second access returns cached result
/// let liveness2 = cache.liveness(&translator);
///
/// // After modifying IR, invalidate
/// cache.invalidate(&Invalidations::instructions_only());
/// ```
#[derive(Debug)]
pub struct AnalysisCache<V, I>
where
    V: OptimizableValue,
    I: OptimizableInstruction<Value = V>,
{
    liveness: Option<LivenessAnalysis>,
    use_def: Option<UseDefChains<V>>,
    _phantom: PhantomData<I>,
}

impl<V, I> Default for AnalysisCache<V, I>
where
    V: OptimizableValue,
    I: OptimizableInstruction<Value = V>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<V, I> AnalysisCache<V, I>
where
    V: OptimizableValue,
    I: OptimizableInstruction<Value = V>,
{
    /// Create a new empty cache.
    pub fn new() -> Self {
        AnalysisCache {
            liveness: None,
            use_def: None,
            _phantom: PhantomData,
        }
    }

    /// Invalidate analyses based on what a pass modifies.
    pub fn invalidate(&mut self, invalidations: &Invalidations) {
        if invalidations.liveness {
            self.liveness = None;
        }
        if invalidations.use_def {
            self.use_def = None;
        }
        // Note: dominators not yet implemented
    }

    /// Invalidate all cached analyses.
    pub fn invalidate_all(&mut self) {
        self.liveness = None;
        self.use_def = None;
    }

    /// Get or compute liveness analysis.
    pub fn liveness<F>(&mut self, translator: &SSATranslator<V, I, F>) -> &LivenessAnalysis
    where
        F: InstructionFactory<Instr = I>,
    {
        if self.liveness.is_none() {
            self.liveness = Some(LivenessAnalysis::compute(translator));
        }
        self.liveness.as_ref().unwrap()
    }

    /// Get or compute use-def chains.
    pub fn use_def<F>(&mut self, translator: &SSATranslator<V, I, F>) -> &UseDefChains<V>
    where
        F: InstructionFactory<Instr = I>,
    {
        if self.use_def.is_none() {
            self.use_def = Some(UseDefChains::compute(translator));
        }
        self.use_def.as_ref().unwrap()
    }

    /// Check if liveness analysis is cached.
    pub fn has_liveness(&self) -> bool {
        self.liveness.is_some()
    }

    /// Check if use-def chains are cached.
    pub fn has_use_def(&self) -> bool {
        self.use_def.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Basic tests for cache invalidation
    #[test]
    fn test_invalidations() {
        // This would need concrete types to fully test
        // Just verify the API compiles
        let inv = Invalidations::all();
        assert!(inv.liveness);
        assert!(inv.use_def);
    }
}
