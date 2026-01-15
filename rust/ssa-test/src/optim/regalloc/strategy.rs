//! Allocation strategies for the register allocator.
//!
//! This module provides a trait-based strategy system that controls how the
//! register allocator selects registers and makes spill decisions. Different
//! strategies can be used for different use cases:
//!
//! - Standard compiled languages using platform ABIs
//! - JIT compilers with custom calling conventions
//! - Languages that handle register preservation explicitly
//!
//! # Built-in Strategies
//!
//! - [`StandardCallingConvention`]: Traditional behavior where call-crossing
//!   intervals must use callee-saved registers
//! - [`IgnoreCallingConvention`]: Treats all registers equally, ignoring calls
//! - [`PreferCallerSavedForShortLived`]: Smart strategy that reserves callee-saved
//!   registers for intervals that actually need them
//! - [`PreferredRegistersStrategy`]: Configurable strategy for custom scenarios
//!
//! # Example
//!
//! ```ignore
//! use ssa_lib::optim::regalloc::{LinearScanAllocator, IgnoreCallingConvention};
//!
//! // Create allocator that ignores calling conventions
//! let allocator = LinearScanAllocator::with_strategy(
//!     my_arch,
//!     IgnoreCallingConvention,
//! );
//! ```

use std::collections::HashSet;
use std::fmt::Debug;

use super::interval::{LiveInterval, ProgramPoint};
use super::target::{PhysicalRegister, RegisterClass, TargetArchitecture};

/// Context provided to allocation strategies for making decisions.
///
/// Contains information about the current state of allocation that strategies
/// can use to make register selection and spill decisions.
#[derive(Debug)]
pub struct AllocationContext<'a, T: TargetArchitecture> {
    /// The target architecture
    pub target: &'a T,
    /// Set of currently free register IDs (indexed by register ID)
    pub free_registers: &'a [bool],
}

impl<'a, T: TargetArchitecture> AllocationContext<'a, T>
where
    T::Register: 'static,
{
    /// Create a new allocation context.
    pub fn new(target: &'a T, free_registers: &'a [bool]) -> Self {
        Self { target, free_registers }
    }

    /// Check if a register is currently free.
    pub fn is_register_free(&self, reg_id: usize) -> bool {
        self.free_registers.get(reg_id).copied().unwrap_or(false)
    }

    /// Get the set of callee-saved register IDs.
    pub fn callee_saved_ids(&self) -> HashSet<usize> {
        self.target.callee_saved().iter().map(|r| r.id()).collect()
    }

    /// Get the set of caller-saved register IDs.
    pub fn caller_saved_ids(&self) -> HashSet<usize> {
        self.target.caller_saved().iter().map(|r| r.id()).collect()
    }

    /// Iterate over free register IDs.
    pub fn free_register_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.free_registers
            .iter()
            .enumerate()
            .filter_map(|(id, &is_free)| if is_free { Some(id) } else { None })
    }
}

/// Information about an active interval for spill decisions.
#[derive(Debug, Clone, Copy)]
pub struct ActiveInterval {
    /// Index into the intervals array
    pub interval_idx: usize,
    /// Currently assigned register ID
    pub register: usize,
    /// End point of the interval
    pub end: ProgramPoint,
}

/// Trait for customizing register allocation decisions.
///
/// Implement this trait to control:
/// - Which registers are preferred for different kinds of intervals
/// - How to select spill candidates when registers are exhausted
/// - Whether to consider calling convention semantics
///
/// # Design Philosophy
///
/// The strategy pattern separates policy from mechanism:
/// - The `LinearScanAllocator` handles the mechanics of tracking intervals,
///   maintaining the active list, and performing the actual allocation
/// - The `AllocationStrategy` makes policy decisions about which registers
///   to prefer and which intervals to spill
///
/// # Example: Custom Strategy
///
/// ```ignore
/// #[derive(Debug, Clone)]
/// struct MyStrategy;
///
/// impl<T: TargetArchitecture> AllocationStrategy<T> for MyStrategy
/// where
///     T::Register: 'static,
/// {
///     fn select_register(
///         &self,
///         interval: &LiveInterval,
///         ctx: &AllocationContext<T>,
///     ) -> Option<usize> {
///         // Custom register selection logic
///         ctx.free_register_ids().next()
///     }
///
///     fn select_spill_candidate(
///         &self,
///         current: &LiveInterval,
///         active: &[ActiveInterval],
///         all_intervals: &[LiveInterval],
///         ctx: &AllocationContext<T>,
///     ) -> Option<usize> {
///         // Custom spill selection logic
///         None // Spill current interval
///     }
/// }
/// ```
pub trait AllocationStrategy<T: TargetArchitecture>: Debug {
    /// Select a register for the given interval.
    ///
    /// Called when allocating a new interval. Should return `Some(reg_id)` if
    /// a suitable free register is found, or `None` if all registers are occupied.
    ///
    /// # Arguments
    /// - `interval`: The interval being allocated (includes crosses_call, register_hint, etc.)
    /// - `ctx`: Current allocation context with free register information
    ///
    /// # Implementation Guidelines
    /// - Check `interval.register_hint` first if hints should be respected
    /// - Consider `interval.crosses_call` when choosing between register categories
    /// - Only return registers that are actually free (use `ctx.is_register_free()`)
    fn select_register(
        &self,
        interval: &LiveInterval,
        ctx: &AllocationContext<T>,
    ) -> Option<usize>;

    /// Select a spill candidate from the active intervals.
    ///
    /// Called when no free register is available. Should return the index
    /// into the active list of the best interval to spill, or `None` if
    /// the current interval should be spilled instead.
    ///
    /// # Arguments
    /// - `current_interval`: The interval we're trying to allocate
    /// - `active_intervals`: Slice of active interval information
    /// - `all_intervals`: All intervals (for looking up properties like crosses_call)
    /// - `ctx`: Current allocation context
    ///
    /// # Returns
    /// - `Some(active_idx)`: Spill the interval at this index in active_intervals
    /// - `None`: Spill the current interval instead
    fn select_spill_candidate(
        &self,
        current_interval: &LiveInterval,
        active_intervals: &[ActiveInterval],
        all_intervals: &[LiveInterval],
        ctx: &AllocationContext<T>,
    ) -> Option<usize>;

    /// Returns true if this strategy cares about calling conventions.
    ///
    /// When false, the allocator may skip some overhead related to
    /// tracking call sites and callee-saved register preferences.
    fn respects_calling_convention(&self) -> bool {
        true
    }
}

// ============================================================================
// Built-in Strategies
// ============================================================================

/// Standard calling convention strategy.
///
/// This is the default strategy that respects traditional calling conventions:
/// - Call-crossing intervals MUST use callee-saved registers (or spill)
/// - Non-call-crossing intervals can use any register
/// - Spill decisions respect the callee-saved requirement
///
/// # Use Cases
/// - Standard compiled languages (C, Rust, etc.)
/// - Any language using platform ABI
#[derive(Debug, Clone, Copy, Default)]
pub struct StandardCallingConvention;

impl<T: TargetArchitecture> AllocationStrategy<T> for StandardCallingConvention
where
    T::Register: 'static,
{
    fn select_register(
        &self,
        interval: &LiveInterval,
        ctx: &AllocationContext<T>,
    ) -> Option<usize> {
        let hint = interval.register_hint;

        if interval.crosses_call {
            // Must use callee-saved register
            let callee_saved = ctx.callee_saved_ids();

            // Check hint first (only if it's callee-saved)
            if let Some(h) = hint {
                if callee_saved.contains(&h) && ctx.is_register_free(h) {
                    return Some(h);
                }
            }

            // Find any free callee-saved register
            for reg in ctx.target.callee_saved() {
                let id = reg.id();
                if ctx.is_register_free(id) {
                    return Some(id);
                }
            }

            None
        } else {
            // Can use any register - check hint first
            if let Some(h) = hint {
                if ctx.is_register_free(h) {
                    return Some(h);
                }
            }

            // Find any free register
            ctx.free_register_ids().next()
        }
    }

    fn select_spill_candidate(
        &self,
        current_interval: &LiveInterval,
        active_intervals: &[ActiveInterval],
        all_intervals: &[LiveInterval],
        ctx: &AllocationContext<T>,
    ) -> Option<usize> {
        let callee_saved = ctx.callee_saved_ids();
        let current_end = current_interval.end();

        // Find candidate with furthest end point
        let candidate = active_intervals
            .iter()
            .enumerate()
            .filter(|(_, active)| {
                let interval = &all_intervals[active.interval_idx];

                // Don't spill fixed-register intervals
                if interval.fixed_register.is_some() {
                    return false;
                }

                // If current interval crosses a call, we can only evict from
                // callee-saved registers (since we need a callee-saved register)
                if current_interval.crosses_call && !callee_saved.contains(&active.register) {
                    return false;
                }

                true
            })
            .max_by_key(|(_, active)| active.end);

        // Only spill if the candidate lives longer than current
        candidate.and_then(|(active_idx, active)| {
            if Some(active.end) > current_end {
                Some(active_idx)
            } else {
                None
            }
        })
    }

    fn respects_calling_convention(&self) -> bool {
        true
    }
}

/// Ignore calling conventions entirely.
///
/// All registers are treated equally regardless of whether an interval
/// crosses a call. Useful for:
/// - Languages that handle register saving/restoring explicitly
/// - JIT compilers with non-standard calling conventions
/// - Testing and debugging
///
/// # Use Cases
/// - Beagle language (uses CallWithSaves for explicit save/restore)
/// - Interpreters with custom register management
#[derive(Debug, Clone, Copy, Default)]
pub struct IgnoreCallingConvention;

impl<T: TargetArchitecture> AllocationStrategy<T> for IgnoreCallingConvention
where
    T::Register: 'static,
{
    fn select_register(
        &self,
        interval: &LiveInterval,
        ctx: &AllocationContext<T>,
    ) -> Option<usize> {
        let callee_saved_ids = ctx.callee_saved_ids();

        // Check hint first
        if let Some(h) = interval.register_hint {
            if ctx.is_register_free(h) {
                return Some(h);
            }
        }

        if interval.crosses_call {
            // For call-crossing intervals, PREFER callee-saved registers to minimize
            // save/restore overhead, but allow caller-saved as fallback.
            for reg in ctx.target.callee_saved() {
                let id = reg.id();
                if ctx.is_register_free(id) {
                    return Some(id);
                }
            }
            // Fall back to any free register
            for reg in ctx.target.default_class().allocatable_registers() {
                let id = reg.id();
                if ctx.is_register_free(id) {
                    return Some(id);
                }
            }
        } else {
            // For non-call-crossing intervals, PREFER caller-saved registers to
            // reserve callee-saved for intervals that actually need them.
            for reg in ctx.target.default_class().allocatable_registers() {
                let id = reg.id();
                // Skip callee-saved registers on first pass
                if callee_saved_ids.contains(&id) {
                    continue;
                }
                if ctx.is_register_free(id) {
                    return Some(id);
                }
            }
            // Fall back to callee-saved if no caller-saved available
            for reg in ctx.target.callee_saved() {
                let id = reg.id();
                if ctx.is_register_free(id) {
                    return Some(id);
                }
            }
        }
        None
    }

    fn select_spill_candidate(
        &self,
        current_interval: &LiveInterval,
        active_intervals: &[ActiveInterval],
        all_intervals: &[LiveInterval],
        _ctx: &AllocationContext<T>,
    ) -> Option<usize> {
        let current_end = current_interval.end();

        // Find candidate with furthest end point (no filtering by register type)
        let candidate = active_intervals
            .iter()
            .enumerate()
            .filter(|(_, active)| {
                // Don't spill fixed-register intervals
                all_intervals[active.interval_idx].fixed_register.is_none()
            })
            .max_by_key(|(_, active)| active.end);

        // Only spill if the candidate lives longer
        candidate.and_then(|(active_idx, active)| {
            if Some(active.end) > current_end {
                Some(active_idx)
            } else {
                None
            }
        })
    }

    fn respects_calling_convention(&self) -> bool {
        false
    }
}

/// Smart strategy that prefers caller-saved for short-lived intervals.
///
/// This is an improved version of StandardCallingConvention that:
/// - Prefers caller-saved registers for non-call-crossing intervals
/// - Reserves callee-saved registers for call-crossing intervals
/// - Can evict non-call-crossing intervals from callee-saved registers
///   when a call-crossing interval needs one
///
/// This solves the problem where non-call-crossing intervals "steal"
/// callee-saved registers before call-crossing intervals are processed.
///
/// # Algorithm
/// For non-call-crossing intervals:
///   1. Try hint (if caller-saved)
///   2. Try any caller-saved register
///   3. Try hint (if callee-saved) - only as fallback
///   4. Try any callee-saved register - only as last resort
///
/// For call-crossing intervals:
///   1. Try hint (if callee-saved)
///   2. Try any free callee-saved register
///   3. Consider evicting a non-call-crossing interval from a callee-saved register
#[derive(Debug, Clone, Copy, Default)]
pub struct PreferCallerSavedForShortLived;

impl<T: TargetArchitecture> AllocationStrategy<T> for PreferCallerSavedForShortLived
where
    T::Register: 'static,
{
    fn select_register(
        &self,
        interval: &LiveInterval,
        ctx: &AllocationContext<T>,
    ) -> Option<usize> {
        let callee_saved = ctx.callee_saved_ids();
        let hint = interval.register_hint;

        if interval.crosses_call {
            // Must use callee-saved register
            // Check hint first (only if it's callee-saved)
            if let Some(h) = hint {
                if callee_saved.contains(&h) && ctx.is_register_free(h) {
                    return Some(h);
                }
            }

            // Find any free callee-saved register
            for reg in ctx.target.callee_saved() {
                let id = reg.id();
                if ctx.is_register_free(id) {
                    return Some(id);
                }
            }

            None
        } else {
            // Prefer caller-saved to leave callee-saved available

            // 1. Try hint if it's caller-saved
            if let Some(h) = hint {
                if !callee_saved.contains(&h) && ctx.is_register_free(h) {
                    return Some(h);
                }
            }

            // 2. Try any free caller-saved register
            for reg in ctx.target.caller_saved() {
                let id = reg.id();
                if ctx.is_register_free(id) {
                    return Some(id);
                }
            }

            // 3. Try hint if it's callee-saved (fallback)
            if let Some(h) = hint {
                if callee_saved.contains(&h) && ctx.is_register_free(h) {
                    return Some(h);
                }
            }

            // 4. Try any free callee-saved register (last resort)
            for reg in ctx.target.callee_saved() {
                let id = reg.id();
                if ctx.is_register_free(id) {
                    return Some(id);
                }
            }

            None
        }
    }

    fn select_spill_candidate(
        &self,
        current_interval: &LiveInterval,
        active_intervals: &[ActiveInterval],
        all_intervals: &[LiveInterval],
        ctx: &AllocationContext<T>,
    ) -> Option<usize> {
        let callee_saved = ctx.callee_saved_ids();
        let current_end = current_interval.end();

        if current_interval.crosses_call {
            // We need a callee-saved register.
            // First, try to evict a non-call-crossing interval from a callee-saved reg
            // (This is the key improvement over StandardCallingConvention!)

            let evict_candidate = active_intervals
                .iter()
                .enumerate()
                .filter(|(_, active)| {
                    let interval = &all_intervals[active.interval_idx];

                    // Don't spill fixed-register intervals
                    if interval.fixed_register.is_some() {
                        return false;
                    }

                    // Must be in a callee-saved register
                    if !callee_saved.contains(&active.register) {
                        return false;
                    }

                    // Prefer to evict non-call-crossing intervals
                    // (they can go to caller-saved or stack)
                    !interval.crosses_call
                })
                .max_by_key(|(_, active)| active.end);

            // If we found a non-call-crossing candidate, evict it
            if let Some((active_idx, _)) = evict_candidate {
                return Some(active_idx);
            }

            // Otherwise, fall back to standard behavior: evict any callee-saved
            let standard_candidate = active_intervals
                .iter()
                .enumerate()
                .filter(|(_, active)| {
                    let interval = &all_intervals[active.interval_idx];
                    interval.fixed_register.is_none() && callee_saved.contains(&active.register)
                })
                .max_by_key(|(_, active)| active.end);

            standard_candidate.and_then(|(active_idx, active)| {
                if Some(active.end) > current_end {
                    Some(active_idx)
                } else {
                    None
                }
            })
        } else {
            // Non-call-crossing interval - can evict from any register
            let candidate = active_intervals
                .iter()
                .enumerate()
                .filter(|(_, active)| {
                    all_intervals[active.interval_idx].fixed_register.is_none()
                })
                .max_by_key(|(_, active)| active.end);

            candidate.and_then(|(active_idx, active)| {
                if Some(active.end) > current_end {
                    Some(active_idx)
                } else {
                    None
                }
            })
        }
    }

    fn respects_calling_convention(&self) -> bool {
        true
    }
}

/// Strategy that prefers specific registers for call-crossing intervals.
///
/// This is designed for languages like Beagle that:
/// - Don't use standard calling conventions
/// - Explicitly save/restore registers via special instructions
/// - Want to minimize save/restore overhead by preferring certain registers
///   for values that live across calls
///
/// # Configuration
/// - `preferred_for_call_crossing`: Register IDs to prefer for call-crossing intervals
/// - `evict_non_crossing_from_preferred`: If true, will evict non-call-crossing
///   intervals from preferred registers to make room for call-crossing ones
///
/// # Example
/// ```ignore
/// // Prefer X19-X28 for call-crossing, allow eviction
/// let strategy = PreferredRegistersStrategy::new(
///     (19..=28).collect(),
///     true,
/// );
/// let allocator = LinearScanAllocator::with_strategy(arch, strategy);
/// ```
#[derive(Debug, Clone)]
pub struct PreferredRegistersStrategy {
    /// Register IDs to prefer for call-crossing intervals
    pub preferred_for_call_crossing: HashSet<usize>,
    /// Whether to evict non-call-crossing intervals from preferred registers
    pub evict_non_crossing_from_preferred: bool,
}

impl PreferredRegistersStrategy {
    /// Create a new preferred registers strategy.
    ///
    /// # Arguments
    /// - `preferred_for_call_crossing`: Set of register IDs to prefer for call-crossing intervals
    /// - `evict_non_crossing_from_preferred`: If true, non-call-crossing intervals
    ///   may be evicted from preferred registers to make room for call-crossing ones
    pub fn new(
        preferred_for_call_crossing: HashSet<usize>,
        evict_non_crossing_from_preferred: bool,
    ) -> Self {
        Self {
            preferred_for_call_crossing,
            evict_non_crossing_from_preferred,
        }
    }

    /// Create a strategy with the given preferred registers and no eviction.
    pub fn without_eviction(preferred_for_call_crossing: HashSet<usize>) -> Self {
        Self::new(preferred_for_call_crossing, false)
    }

    /// Create a strategy with the given preferred registers and eviction enabled.
    pub fn with_eviction(preferred_for_call_crossing: HashSet<usize>) -> Self {
        Self::new(preferred_for_call_crossing, true)
    }
}

impl<T: TargetArchitecture> AllocationStrategy<T> for PreferredRegistersStrategy
where
    T::Register: 'static,
{
    fn select_register(
        &self,
        interval: &LiveInterval,
        ctx: &AllocationContext<T>,
    ) -> Option<usize> {
        let hint = interval.register_hint;

        if interval.crosses_call {
            // Prefer registers in the preferred set

            // Check hint first (only if it's in preferred set)
            if let Some(h) = hint {
                if self.preferred_for_call_crossing.contains(&h) && ctx.is_register_free(h) {
                    return Some(h);
                }
            }

            // Try any free preferred register
            for &reg_id in &self.preferred_for_call_crossing {
                if ctx.is_register_free(reg_id) {
                    return Some(reg_id);
                }
            }

            // Fall back to any free register (not ideal but better than failing)
            if let Some(h) = hint {
                if ctx.is_register_free(h) {
                    return Some(h);
                }
            }
            ctx.free_register_ids().next()
        } else {
            // Non-call-crossing: prefer non-preferred registers to leave preferred ones free

            // Check hint first if it's NOT in preferred set
            if let Some(h) = hint {
                if !self.preferred_for_call_crossing.contains(&h) && ctx.is_register_free(h) {
                    return Some(h);
                }
            }

            // Try any free non-preferred register
            for reg_id in ctx.free_register_ids() {
                if !self.preferred_for_call_crossing.contains(&reg_id) {
                    return Some(reg_id);
                }
            }

            // Fall back to preferred registers if nothing else available
            if let Some(h) = hint {
                if ctx.is_register_free(h) {
                    return Some(h);
                }
            }
            ctx.free_register_ids().next()
        }
    }

    fn select_spill_candidate(
        &self,
        current_interval: &LiveInterval,
        active_intervals: &[ActiveInterval],
        all_intervals: &[LiveInterval],
        _ctx: &AllocationContext<T>,
    ) -> Option<usize> {
        let current_end = current_interval.end();

        if current_interval.crosses_call && self.evict_non_crossing_from_preferred {
            // Try to evict a non-call-crossing interval from a preferred register

            let evict_candidate = active_intervals
                .iter()
                .enumerate()
                .filter(|(_, active)| {
                    let interval = &all_intervals[active.interval_idx];

                    // Don't spill fixed-register intervals
                    if interval.fixed_register.is_some() {
                        return false;
                    }

                    // Must be in a preferred register
                    if !self.preferred_for_call_crossing.contains(&active.register) {
                        return false;
                    }

                    // Prefer to evict non-call-crossing intervals
                    !interval.crosses_call
                })
                .max_by_key(|(_, active)| active.end);

            if let Some((active_idx, _)) = evict_candidate {
                return Some(active_idx);
            }
        }

        // Standard behavior: evict interval with furthest end
        let candidate = active_intervals
            .iter()
            .enumerate()
            .filter(|(_, active)| {
                all_intervals[active.interval_idx].fixed_register.is_none()
            })
            .max_by_key(|(_, active)| active.end);

        candidate.and_then(|(active_idx, active)| {
            if Some(active.end) > current_end {
                Some(active_idx)
            } else {
                None
            }
        })
    }

    fn respects_calling_convention(&self) -> bool {
        false // Custom handling
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::target::{PhysicalRegister, RegisterClass, TargetArchitecture};
    use super::super::interval::LiveRange;
    use crate::types::SsaVariable;

    // Test architecture with 6 registers: R0-R2 caller-saved, R3-R5 callee-saved
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum TestReg { R0, R1, R2, R3, R4, R5 }

    impl PhysicalRegister for TestReg {
        fn id(&self) -> usize {
            match self {
                TestReg::R0 => 0,
                TestReg::R1 => 1,
                TestReg::R2 => 2,
                TestReg::R3 => 3,
                TestReg::R4 => 4,
                TestReg::R5 => 5,
            }
        }
        fn name(&self) -> &'static str {
            match self {
                TestReg::R0 => "r0",
                TestReg::R1 => "r1",
                TestReg::R2 => "r2",
                TestReg::R3 => "r3",
                TestReg::R4 => "r4",
                TestReg::R5 => "r5",
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum TestClass { GP }

    impl RegisterClass for TestClass {
        type Register = TestReg;
        fn name(&self) -> &'static str { "gp" }
        fn allocatable_registers(&self) -> &'static [TestReg] {
            &[TestReg::R0, TestReg::R1, TestReg::R2, TestReg::R3, TestReg::R4, TestReg::R5]
        }
    }

    #[derive(Debug, Clone)]
    struct TestArch;

    impl TargetArchitecture for TestArch {
        type Register = TestReg;
        type Class = TestClass;
        fn register_classes(&self) -> &'static [TestClass] { &[TestClass::GP] }
        fn default_class(&self) -> TestClass { TestClass::GP }
        fn stack_slot_size(&self) -> usize { 8 }
        fn caller_saved(&self) -> &'static [TestReg] {
            &[TestReg::R0, TestReg::R1, TestReg::R2]
        }
        fn callee_saved(&self) -> &'static [TestReg] {
            &[TestReg::R3, TestReg::R4, TestReg::R5]
        }
    }

    fn make_interval(name: &str, start: usize, end: usize, crosses_call: bool) -> LiveInterval {
        let mut interval = LiveInterval::new(SsaVariable::new(name));
        interval.add_range(LiveRange::new(ProgramPoint(start), ProgramPoint(end)));
        interval.crosses_call = crosses_call;
        interval
    }

    #[test]
    fn test_standard_convention_call_crossing() {
        let arch = TestArch;
        let free_regs = vec![true; 6];
        let ctx = AllocationContext::new(&arch, &free_regs);

        let strategy = StandardCallingConvention;
        let interval = make_interval("x", 0, 10, true);

        let reg = strategy.select_register(&interval, &ctx);

        // Should select a callee-saved register (R3, R4, or R5)
        assert!(reg.is_some());
        assert!(reg.unwrap() >= 3);
    }

    #[test]
    fn test_standard_convention_non_call_crossing() {
        let arch = TestArch;
        let free_regs = vec![true; 6];
        let ctx = AllocationContext::new(&arch, &free_regs);

        let strategy = StandardCallingConvention;
        let interval = make_interval("x", 0, 10, false);

        let reg = strategy.select_register(&interval, &ctx);

        // Should get any register (first free, which is R0)
        assert_eq!(reg, Some(0));
    }

    #[test]
    fn test_ignore_convention_call_crossing() {
        let arch = TestArch;
        let free_regs = vec![true; 6];
        let ctx = AllocationContext::new(&arch, &free_regs);

        let strategy = IgnoreCallingConvention;
        let interval = make_interval("x", 0, 10, true);

        let reg = strategy.select_register(&interval, &ctx);

        // For call-crossing intervals, should PREFER callee-saved (R3-R5) to minimize saves
        assert!(reg.is_some());
        assert!(reg.unwrap() >= 3, "Call-crossing should prefer callee-saved registers");
    }

    #[test]
    fn test_ignore_convention_non_call_crossing() {
        let arch = TestArch;
        let free_regs = vec![true; 6];
        let ctx = AllocationContext::new(&arch, &free_regs);

        let strategy = IgnoreCallingConvention;
        let interval = make_interval("x", 0, 10, false);

        let reg = strategy.select_register(&interval, &ctx);

        // For non-call-crossing intervals, PREFER caller-saved to reserve callee-saved
        assert!(reg.is_some());
        assert!(reg.unwrap() < 3, "Non-call-crossing should prefer caller-saved registers");
    }

    #[test]
    fn test_prefer_caller_saved_non_crossing() {
        let arch = TestArch;
        let free_regs = vec![true; 6];
        let ctx = AllocationContext::new(&arch, &free_regs);

        let strategy = PreferCallerSavedForShortLived;
        let interval = make_interval("x", 0, 10, false);

        let reg = strategy.select_register(&interval, &ctx);

        // Should prefer caller-saved (R0, R1, or R2)
        assert!(reg.is_some());
        assert!(reg.unwrap() < 3);
    }

    #[test]
    fn test_prefer_caller_saved_crossing() {
        let arch = TestArch;
        let free_regs = vec![true; 6];
        let ctx = AllocationContext::new(&arch, &free_regs);

        let strategy = PreferCallerSavedForShortLived;
        let interval = make_interval("x", 0, 10, true);

        let reg = strategy.select_register(&interval, &ctx);

        // Should get callee-saved for call-crossing
        assert!(reg.is_some());
        assert!(reg.unwrap() >= 3);
    }

    #[test]
    fn test_preferred_registers_strategy() {
        let arch = TestArch;
        let free_regs = vec![true; 6];
        let ctx = AllocationContext::new(&arch, &free_regs);

        // Prefer R4 and R5 for call-crossing
        let strategy = PreferredRegistersStrategy::new(
            [4, 5].into_iter().collect(),
            true,
        );

        let call_crossing = make_interval("x", 0, 10, true);
        let non_crossing = make_interval("y", 0, 10, false);

        // Call-crossing should get R4 or R5
        let reg1 = strategy.select_register(&call_crossing, &ctx);
        assert!(reg1 == Some(4) || reg1 == Some(5));

        // Non-crossing should prefer non-preferred registers
        let reg2 = strategy.select_register(&non_crossing, &ctx);
        assert!(reg2.is_some());
        assert!(!strategy.preferred_for_call_crossing.contains(&reg2.unwrap()));
    }

    #[test]
    fn test_respects_calling_convention() {
        assert!(<StandardCallingConvention as AllocationStrategy<TestArch>>::respects_calling_convention(&StandardCallingConvention));
        assert!(!<IgnoreCallingConvention as AllocationStrategy<TestArch>>::respects_calling_convention(&IgnoreCallingConvention));
        assert!(<PreferCallerSavedForShortLived as AllocationStrategy<TestArch>>::respects_calling_convention(&PreferCallerSavedForShortLived));

        let strategy = PreferredRegistersStrategy::new([4, 5].into_iter().collect(), true);
        assert!(!<PreferredRegistersStrategy as AllocationStrategy<TestArch>>::respects_calling_convention(&strategy));
    }
}
