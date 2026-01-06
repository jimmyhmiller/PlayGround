//! Optimization pass infrastructure.
//!
//! Defines the `OptimizationPass` trait and supporting types.

use std::fmt::Debug;

use crate::traits::InstructionFactory;
use crate::translator::SSATranslator;

use super::analysis::AnalysisCache;
use super::traits::{OptimizableValue, OptimizableInstruction};

/// What analyses a pass invalidates when it modifies the IR.
#[derive(Debug, Clone, Default)]
pub struct Invalidations {
    /// Invalidates liveness analysis (live-in/live-out sets)
    pub liveness: bool,
    /// Invalidates use-def chains
    pub use_def: bool,
    /// Invalidates dominator tree
    pub dominators: bool,
}

impl Invalidations {
    /// No analyses invalidated (read-only pass)
    pub fn none() -> Self {
        Invalidations::default()
    }

    /// All analyses invalidated
    pub fn all() -> Self {
        Invalidations {
            liveness: true,
            use_def: true,
            dominators: true,
        }
    }

    /// Instruction-level changes (no control flow changes)
    pub fn instructions_only() -> Self {
        Invalidations {
            liveness: true,
            use_def: true,
            dominators: false,
        }
    }
}

/// Statistics from running an optimization pass.
#[derive(Debug, Clone, Default)]
pub struct PassStats {
    /// Number of instructions removed
    pub instructions_removed: usize,
    /// Number of instructions added
    pub instructions_added: usize,
    /// Number of values propagated (copy prop, const prop)
    pub values_propagated: usize,
    /// Number of expressions folded (const fold)
    pub expressions_folded: usize,
    /// Number of common subexpressions eliminated
    pub cse_eliminations: usize,
    /// Number of phi nodes simplified
    pub phis_simplified: usize,
}

impl PassStats {
    pub fn new() -> Self {
        PassStats::default()
    }

    /// Merge stats from another pass
    pub fn merge(&mut self, other: &PassStats) {
        self.instructions_removed += other.instructions_removed;
        self.instructions_added += other.instructions_added;
        self.values_propagated += other.values_propagated;
        self.expressions_folded += other.expressions_folded;
        self.cse_eliminations += other.cse_eliminations;
        self.phis_simplified += other.phis_simplified;
    }

    /// Returns true if any work was done
    pub fn any_changes(&self) -> bool {
        self.instructions_removed > 0
            || self.instructions_added > 0
            || self.values_propagated > 0
            || self.expressions_folded > 0
            || self.cse_eliminations > 0
            || self.phis_simplified > 0
    }
}

/// Result of running an optimization pass.
#[derive(Debug, Clone)]
pub struct PassResult {
    /// Whether the IR was modified
    pub changed: bool,
    /// Statistics about what the pass did
    pub stats: PassStats,
}

impl PassResult {
    /// No changes made
    pub fn unchanged() -> Self {
        PassResult {
            changed: false,
            stats: PassStats::new(),
        }
    }

    /// Changes were made with the given stats
    pub fn changed(stats: PassStats) -> Self {
        PassResult {
            changed: true,
            stats,
        }
    }
}

/// Trait for optimization passes.
///
/// Each pass transforms the SSA IR in some way (DCE, copy propagation, etc.).
/// Passes receive the translator and an analysis cache, and return whether
/// they modified the IR.
///
/// # Example
/// ```ignore
/// struct MyPass;
///
/// impl<V, I, F> OptimizationPass<V, I, F> for MyPass
/// where
///     V: OptimizableValue,
///     I: OptimizableInstruction<Value = V>,
///     F: InstructionFactory<Instr = I>,
/// {
///     fn name(&self) -> &'static str { "my-pass" }
///
///     fn run(
///         &mut self,
///         translator: &mut SSATranslator<V, I, F>,
///         cache: &mut AnalysisCache<V, I>,
///     ) -> PassResult {
///         // ... transform IR ...
///         PassResult::unchanged()
///     }
///
///     fn invalidates(&self) -> Invalidations {
///         Invalidations::instructions_only()
///     }
/// }
/// ```
pub trait OptimizationPass<V, I, F>: Debug
where
    V: OptimizableValue,
    I: OptimizableInstruction<Value = V>,
    F: InstructionFactory<Instr = I>,
{
    /// Human-readable name of this pass (e.g., "dce", "copy-prop")
    fn name(&self) -> &'static str;

    /// Run the optimization pass.
    ///
    /// Returns `PassResult` indicating whether the IR was modified and statistics.
    fn run(
        &mut self,
        translator: &mut SSATranslator<V, I, F>,
        cache: &mut AnalysisCache<V, I>,
    ) -> PassResult;

    /// What analyses this pass invalidates when it modifies the IR.
    ///
    /// The pipeline uses this to invalidate cached analyses after a pass
    /// that returns `changed: true`.
    fn invalidates(&self) -> Invalidations;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pass_stats_merge() {
        let mut stats1 = PassStats {
            instructions_removed: 5,
            values_propagated: 3,
            ..Default::default()
        };
        let stats2 = PassStats {
            instructions_removed: 2,
            expressions_folded: 4,
            ..Default::default()
        };

        stats1.merge(&stats2);

        assert_eq!(stats1.instructions_removed, 7);
        assert_eq!(stats1.values_propagated, 3);
        assert_eq!(stats1.expressions_folded, 4);
    }

    #[test]
    fn test_invalidations() {
        let none = Invalidations::none();
        assert!(!none.liveness && !none.use_def && !none.dominators);

        let all = Invalidations::all();
        assert!(all.liveness && all.use_def && all.dominators);

        let instr = Invalidations::instructions_only();
        assert!(instr.liveness && instr.use_def && !instr.dominators);
    }
}
