//! Linear Scan Register Allocator.
//!
//! This module implements the linear scan algorithm by Poletto & Sarkar.
//! The algorithm processes live intervals in order of their start points
//! and assigns registers greedily.
//!
//! # Algorithm Overview
//!
//! 1. Sort all live intervals by start point
//! 2. For each interval:
//!    a. Expire intervals that have ended (return registers to pool)
//!    b. If a register is available, assign it
//!    c. Otherwise, spill (either this interval or one from active)
//!
//! # Register Constraints
//!
//! Fixed register constraints are handled by pre-coloring: intervals with
//! fixed constraints are assigned their required register before the main
//! algorithm runs. When a regular interval needs a register that's occupied
//! by a pre-colored interval, it must be spilled or use a different register.

use std::collections::{HashMap, HashSet};

use crate::types::SsaVariable;

use super::interval::{LiveInterval, Location, ProgramPoint};
use super::target::{PhysicalRegister, TargetArchitecture};

/// Configuration for the linear scan allocator.
#[derive(Debug, Clone)]
pub struct LinearScanConfig {
    /// Whether to use register hints when available.
    pub use_hints: bool,

    /// Whether to prefer callee-saved registers for long-lived intervals.
    pub prefer_callee_saved_for_long_lived: bool,

    /// Threshold for considering an interval "long-lived" (in program points).
    pub long_lived_threshold: usize,
}

impl Default for LinearScanConfig {
    fn default() -> Self {
        LinearScanConfig {
            use_hints: true,
            prefer_callee_saved_for_long_lived: true,
            long_lived_threshold: 50,
        }
    }
}

/// Statistics from register allocation.
#[derive(Debug, Clone, Default)]
pub struct AllocationStats {
    /// Number of variables assigned to registers.
    pub registers_assigned: usize,

    /// Number of variables spilled to stack.
    pub variables_spilled: usize,

    /// Number of fixed register constraints satisfied.
    pub fixed_constraints_satisfied: usize,

    /// Number of register hints followed.
    pub hints_followed: usize,
}

/// Result of register allocation.
#[derive(Debug, Clone)]
pub struct AllocationResult {
    /// Assignment for each variable.
    pub assignments: HashMap<SsaVariable, Location>,

    /// Number of stack slots used for spills.
    pub stack_slots_used: usize,

    /// Statistics about the allocation.
    pub stats: AllocationStats,
}

/// Entry in the active list, tracking an interval and its register.
#[derive(Debug, Clone)]
struct ActiveEntry {
    /// Index into the intervals array.
    interval_idx: usize,
    /// Assigned register.
    register: usize,
    /// End point (for sorting).
    end: ProgramPoint,
}

/// The linear scan register allocator.
#[derive(Debug)]
pub struct LinearScanAllocator<T>
where
    T: TargetArchitecture,
    T::Register: 'static,
    T::Class: 'static,
{
    /// Target architecture.
    target: T,

    /// Configuration.
    config: LinearScanConfig,

    /// Currently active intervals (sorted by end point).
    active: Vec<ActiveEntry>,

    /// Set of free registers (by ID).
    free_registers: Vec<bool>,

    /// Next stack slot to allocate.
    next_stack_slot: usize,

    /// Statistics.
    stats: AllocationStats,
}

impl<T> LinearScanAllocator<T>
where
    T: TargetArchitecture,
    T::Register: 'static,
    T::Class: 'static,
{
    /// Create a new allocator for the given target.
    pub fn new(target: T) -> Self {
        Self::with_config(target, LinearScanConfig::default())
    }

    /// Create a new allocator with custom configuration.
    pub fn with_config(target: T, config: LinearScanConfig) -> Self {
        let num_regs = target.total_registers();
        LinearScanAllocator {
            target,
            config,
            active: Vec::new(),
            free_registers: vec![true; num_regs],
            next_stack_slot: 0,
            stats: AllocationStats::default(),
        }
    }

    /// Reset the allocator state for a new allocation.
    fn reset(&mut self) {
        self.active.clear();
        self.free_registers.fill(true);
        self.next_stack_slot = 0;
        self.stats = AllocationStats::default();
    }

    /// Run the linear scan algorithm on the given intervals.
    pub fn allocate(&mut self, intervals: &mut [LiveInterval]) -> AllocationResult {
        self.reset();

        // Handle empty case
        if intervals.is_empty() {
            return AllocationResult {
                assignments: HashMap::new(),
                stack_slots_used: 0,
                stats: self.stats.clone(),
            };
        }

        // Step 1: Sort intervals by start point
        let mut sorted_indices: Vec<usize> = (0..intervals.len()).collect();
        sorted_indices.sort_by_key(|&i| intervals[i].start());

        // Step 2: Process intervals with fixed constraints first
        // This ensures fixed registers are reserved before regular allocation
        for &idx in &sorted_indices {
            if let Some(fixed_reg) = intervals[idx].fixed_register {
                intervals[idx].assignment = Some(Location::Register(fixed_reg));
                self.stats.fixed_constraints_satisfied += 1;
            }
        }

        // Step 3: Process remaining intervals in order
        for &idx in &sorted_indices {
            // Skip already-assigned (fixed constraint) intervals
            if intervals[idx].assignment.is_some() {
                continue;
            }

            let start = match intervals[idx].start() {
                Some(s) => s,
                None => continue,  // Empty interval
            };

            // Expire old intervals
            self.expire_old_intervals(start);

            // Try to allocate a register
            let register_hint = intervals[idx].register_hint;
            let end = intervals[idx].end().unwrap_or(start);
            let crosses_call = intervals[idx].crosses_call;

            // Choose allocation strategy based on whether interval crosses a call
            let allocated_reg = if crosses_call {
                // Must use a callee-saved register (preserved across calls)
                self.try_allocate_callee_saved(register_hint)
            } else {
                // Can use any register
                self.try_allocate_register_with_hint(register_hint)
            };

            if let Some(reg) = allocated_reg {
                intervals[idx].assignment = Some(Location::Register(reg));
                self.stats.registers_assigned += 1;

                if self.config.use_hints && register_hint == Some(reg) {
                    self.stats.hints_followed += 1;
                }

                // Add to active list
                self.add_to_active(idx, reg, end);
            } else {
                // Must spill
                self.spill_at_interval(intervals, idx);
            }
        }

        // Build result
        let mut assignments = HashMap::new();
        for interval in intervals.iter() {
            if let Some(loc) = interval.assignment {
                assignments.insert(interval.variable.clone(), loc);
            }
        }

        AllocationResult {
            assignments,
            stack_slots_used: self.next_stack_slot,
            stats: self.stats.clone(),
        }
    }

    /// Remove intervals from active list that have expired.
    fn expire_old_intervals(&mut self, current: ProgramPoint) {
        // Remove intervals whose end point is before the current point
        let mut i = 0;
        while i < self.active.len() {
            if self.active[i].end <= current {
                let entry = self.active.remove(i);
                // Return register to free pool
                self.free_registers[entry.register] = true;
            } else {
                i += 1;
            }
        }
    }

    /// Try to allocate a register, optionally respecting a hint.
    fn try_allocate_register_with_hint(&mut self, hint: Option<usize>) -> Option<usize> {
        // Check hint first
        if self.config.use_hints {
            if let Some(h) = hint {
                if self.free_registers.get(h).copied().unwrap_or(false) {
                    self.free_registers[h] = false;
                    return Some(h);
                }
            }
        }

        // Find any free register
        // TODO: Use register class to filter
        for (reg, is_free) in self.free_registers.iter_mut().enumerate() {
            if *is_free {
                *is_free = false;
                return Some(reg);
            }
        }

        None
    }

    /// Handle spilling when no register is available.
    fn spill_at_interval(&mut self, intervals: &mut [LiveInterval], current_idx: usize) {
        let current_interval = &intervals[current_idx];
        let current_end = current_interval.end();
        let current_crosses_call = current_interval.crosses_call;

        // Get callee-saved register IDs for filtering
        let callee_saved = self.callee_saved_ids();

        // Find the interval with the furthest end point among active intervals
        let spill_candidate = self.active.iter()
            .enumerate()
            .filter(|(_, entry)| {
                // Don't spill fixed-register intervals
                if intervals[entry.interval_idx].fixed_register.is_some() {
                    return false;
                }
                // If current interval crosses a call, we can only evict from
                // callee-saved registers (since we need a callee-saved register)
                if current_crosses_call && !callee_saved.contains(&entry.register) {
                    return false;
                }
                true
            })
            .max_by_key(|(_, entry)| entry.end);

        match spill_candidate {
            Some((active_idx, entry)) if Some(entry.end) > current_end => {
                // Spill the active interval (it has a longer lifetime)
                let spill_idx = entry.interval_idx;
                let reg = entry.register;

                // Assign the register to current interval
                intervals[current_idx].assignment = Some(Location::Register(reg));
                self.stats.registers_assigned += 1;

                // Update active list
                self.active.remove(active_idx);
                if let Some(end) = current_end {
                    self.add_to_active(current_idx, reg, end);
                }

                // Spill the evicted interval
                let slot = self.allocate_stack_slot();
                intervals[spill_idx].assignment = Some(Location::StackSlot(slot));
                self.stats.variables_spilled += 1;
            }
            _ => {
                // Spill the current interval (no benefit to evicting)
                let slot = self.allocate_stack_slot();
                intervals[current_idx].assignment = Some(Location::StackSlot(slot));
                self.stats.variables_spilled += 1;
            }
        }
    }

    /// Add an interval to the active list, maintaining sorted order by end.
    fn add_to_active(&mut self, interval_idx: usize, register: usize, end: ProgramPoint) {
        let entry = ActiveEntry { interval_idx, register, end };
        let pos = self.active.iter()
            .position(|e| e.end > end)
            .unwrap_or(self.active.len());
        self.active.insert(pos, entry);
    }

    /// Allocate a new stack slot.
    fn allocate_stack_slot(&mut self) -> usize {
        let slot = self.next_stack_slot;
        self.next_stack_slot += 1;
        slot
    }

    /// Get the set of callee-saved register IDs.
    fn callee_saved_ids(&self) -> HashSet<usize> {
        self.target.callee_saved().iter().map(|r| r.id()).collect()
    }

    /// Try to allocate a register for an interval that crosses a call.
    ///
    /// Only callee-saved registers can be used for such intervals.
    fn try_allocate_callee_saved(&mut self, hint: Option<usize>) -> Option<usize> {
        let callee_saved = self.callee_saved_ids();

        // Check hint first (only if it's callee-saved)
        if self.config.use_hints {
            if let Some(h) = hint {
                if callee_saved.contains(&h)
                    && self.free_registers.get(h).copied().unwrap_or(false)
                {
                    self.free_registers[h] = false;
                    return Some(h);
                }
            }
        }

        // Find any free callee-saved register
        for &reg in self.target.callee_saved() {
            let id = reg.id();
            if self.free_registers.get(id).copied().unwrap_or(false) {
                self.free_registers[id] = false;
                return Some(id);
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::target::{PhysicalRegister, RegisterClass, TargetArchitecture};
    use super::super::interval::LiveRange;

    // Test architecture with 3 registers
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum TestReg { R0, R1, R2 }

    impl PhysicalRegister for TestReg {
        fn id(&self) -> usize {
            match self {
                TestReg::R0 => 0,
                TestReg::R1 => 1,
                TestReg::R2 => 2,
            }
        }
        fn name(&self) -> &'static str {
            match self {
                TestReg::R0 => "r0",
                TestReg::R1 => "r1",
                TestReg::R2 => "r2",
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum TestClass { GP }

    impl RegisterClass for TestClass {
        type Register = TestReg;
        fn name(&self) -> &'static str { "gp" }
        fn allocatable_registers(&self) -> &'static [TestReg] {
            &[TestReg::R0, TestReg::R1, TestReg::R2]
        }
    }

    #[derive(Debug, Clone)]
    struct TestArch;

    impl TargetArchitecture for TestArch {
        type Register = TestReg;
        type Class = TestClass;
        fn register_classes(&self) -> &'static [TestClass] { &[TestClass::GP] }
        fn default_class(&self) -> TestClass { TestClass::GP }
        fn stack_slot_size(&self) -> usize { 4 }
    }

    fn make_interval(name: &str, start: usize, end: usize) -> LiveInterval {
        let mut interval = LiveInterval::new(SsaVariable::new(name));
        interval.add_range(LiveRange::new(ProgramPoint(start), ProgramPoint(end)));
        interval
    }

    #[test]
    fn test_empty_allocation() {
        let mut allocator = LinearScanAllocator::new(TestArch);
        let result = allocator.allocate(&mut []);

        assert_eq!(result.assignments.len(), 0);
        assert_eq!(result.stack_slots_used, 0);
    }

    #[test]
    fn test_single_interval() {
        let mut allocator = LinearScanAllocator::new(TestArch);
        let mut intervals = vec![make_interval("x", 0, 10)];

        let result = allocator.allocate(&mut intervals);

        assert_eq!(result.assignments.len(), 1);
        assert!(matches!(
            result.assignments.get(&SsaVariable::new("x")),
            Some(Location::Register(_))
        ));
        assert_eq!(result.stats.registers_assigned, 1);
        assert_eq!(result.stats.variables_spilled, 0);
    }

    #[test]
    fn test_non_overlapping_intervals() {
        let mut allocator = LinearScanAllocator::new(TestArch);
        let mut intervals = vec![
            make_interval("a", 0, 5),
            make_interval("b", 5, 10),
            make_interval("c", 10, 15),
        ];

        let result = allocator.allocate(&mut intervals);

        // All should get registers (can reuse after expiration)
        assert_eq!(result.stats.registers_assigned, 3);
        assert_eq!(result.stats.variables_spilled, 0);
    }

    #[test]
    fn test_overlapping_within_capacity() {
        let mut allocator = LinearScanAllocator::new(TestArch);
        let mut intervals = vec![
            make_interval("a", 0, 10),
            make_interval("b", 2, 8),
            make_interval("c", 4, 12),
        ];

        let result = allocator.allocate(&mut intervals);

        // All 3 overlap but we have 3 registers
        assert_eq!(result.stats.registers_assigned, 3);
        assert_eq!(result.stats.variables_spilled, 0);
    }

    #[test]
    fn test_spilling_required() {
        let mut allocator = LinearScanAllocator::new(TestArch);
        let mut intervals = vec![
            make_interval("a", 0, 10),
            make_interval("b", 0, 10),
            make_interval("c", 0, 10),
            make_interval("d", 0, 10),  // 4th interval, only 3 registers
        ];

        let result = allocator.allocate(&mut intervals);

        assert_eq!(result.stats.registers_assigned, 3);
        assert_eq!(result.stats.variables_spilled, 1);
        assert_eq!(result.stack_slots_used, 1);
    }

    #[test]
    fn test_fixed_register_constraint() {
        let mut allocator = LinearScanAllocator::new(TestArch);
        let mut intervals = vec![
            make_interval("a", 0, 10),
            {
                let mut i = make_interval("b", 0, 10);
                i.fixed_register = Some(1);  // Must be R1
                i
            },
        ];

        let result = allocator.allocate(&mut intervals);

        // b must be in register 1
        assert_eq!(
            result.assignments.get(&SsaVariable::new("b")),
            Some(&Location::Register(1))
        );
        assert_eq!(result.stats.fixed_constraints_satisfied, 1);
    }

    #[test]
    fn test_register_hint() {
        let config = LinearScanConfig {
            use_hints: true,
            ..Default::default()
        };
        let mut allocator = LinearScanAllocator::with_config(TestArch, config);

        let mut intervals = vec![{
            let mut i = make_interval("a", 0, 10);
            i.register_hint = Some(2);  // Prefer R2
            i
        }];

        let result = allocator.allocate(&mut intervals);

        // Should follow the hint if available
        assert_eq!(
            result.assignments.get(&SsaVariable::new("a")),
            Some(&Location::Register(2))
        );
        assert_eq!(result.stats.hints_followed, 1);
    }
}
