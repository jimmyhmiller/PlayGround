//! Live interval computation for register allocation.
//!
//! Live intervals represent the range of program points where a variable
//! is live (its value may be used). The linear scan allocator uses these
//! intervals to determine when registers can be reused.

use std::collections::HashMap;

use crate::traits::InstructionFactory;
use crate::translator::SSATranslator;
use crate::types::{BlockId, SsaVariable};
use crate::optim::analysis::LivenessAnalysis;
use crate::optim::traits::{OptimizableValue, OptimizableInstruction};

/// A program point in the linearized instruction stream.
///
/// Program points are assigned sequentially to instructions as blocks
/// are traversed in order. Each instruction gets a unique program point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProgramPoint(pub usize);

impl ProgramPoint {
    /// Create a new program point.
    pub fn new(index: usize) -> Self {
        ProgramPoint(index)
    }

    /// Get the underlying index.
    pub fn index(&self) -> usize {
        self.0
    }
}

/// A contiguous range of program points where a variable is live.
///
/// The range is half-open: [start, end), meaning start is included
/// and end is excluded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveRange {
    /// First program point where the variable is live.
    pub start: ProgramPoint,
    /// First program point where the variable is no longer live.
    pub end: ProgramPoint,
}

impl LiveRange {
    /// Create a new live range.
    pub fn new(start: ProgramPoint, end: ProgramPoint) -> Self {
        debug_assert!(start <= end, "LiveRange start must be <= end");
        LiveRange { start, end }
    }

    /// Check if this range contains a program point.
    pub fn contains(&self, point: ProgramPoint) -> bool {
        point >= self.start && point < self.end
    }

    /// Check if two ranges overlap.
    pub fn overlaps(&self, other: &LiveRange) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// Merge with another overlapping or adjacent range.
    pub fn merge(&self, other: &LiveRange) -> Option<LiveRange> {
        // Check if they're adjacent or overlapping
        if self.end >= other.start && other.end >= self.start {
            Some(LiveRange {
                start: std::cmp::min(self.start, other.start),
                end: std::cmp::max(self.end, other.end),
            })
        } else {
            None
        }
    }

    /// Get the length of this range.
    pub fn len(&self) -> usize {
        self.end.0.saturating_sub(self.start.0)
    }

    /// Check if the range is empty.
    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }
}

/// Location assigned to a variable after register allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Location {
    /// Assigned to a physical register (by ID).
    Register(usize),
    /// Spilled to a stack slot (by slot number).
    StackSlot(usize),
}

/// A live interval for a single SSA variable.
///
/// A live interval tracks the range of program points where a variable
/// is live, along with allocation preferences and the final assignment.
#[derive(Debug, Clone)]
pub struct LiveInterval {
    /// The SSA variable this interval represents.
    pub variable: SsaVariable,

    /// Live ranges for this interval (sorted by start point).
    ///
    /// Most variables have a single range, but some may have multiple
    /// disjoint ranges (e.g., after interval splitting).
    pub ranges: Vec<LiveRange>,

    /// Weight for spilling decisions.
    ///
    /// Higher weight means we prefer to keep this value in a register.
    /// Can be based on loop depth, use frequency, etc.
    pub spill_weight: f64,

    /// Fixed register requirement (pre-coloring).
    ///
    /// If Some, this interval MUST be assigned to this specific register.
    /// Used for instructions with fixed register constraints.
    pub fixed_register: Option<usize>,

    /// Preferred register (hint).
    ///
    /// Unlike fixed_register, this is just a preference that can be
    /// ignored if the register is not available. Used for coalescing.
    pub register_hint: Option<usize>,

    /// Register class for this interval.
    ///
    /// Determines which registers are valid for this interval.
    pub reg_class: Option<usize>,

    /// Final assignment after register allocation.
    pub assignment: Option<Location>,

    /// Whether this interval crosses a call site.
    ///
    /// If true, this interval must be assigned to a callee-saved register
    /// or spilled to the stack. Caller-saved registers would be clobbered
    /// by the call.
    pub crosses_call: bool,
}

impl LiveInterval {
    /// Create a new live interval for a variable.
    pub fn new(variable: SsaVariable) -> Self {
        LiveInterval {
            variable,
            ranges: Vec::new(),
            spill_weight: 1.0,
            fixed_register: None,
            register_hint: None,
            reg_class: None,
            assignment: None,
            crosses_call: false,
        }
    }

    /// Add a live range to this interval.
    pub fn add_range(&mut self, range: LiveRange) {
        // Try to merge with existing ranges
        let mut i = 0;
        while i < self.ranges.len() {
            if let Some(merged) = self.ranges[i].merge(&range) {
                self.ranges[i] = merged;
                // Try to merge with subsequent ranges
                while i + 1 < self.ranges.len() {
                    if let Some(further_merged) = self.ranges[i].merge(&self.ranges[i + 1]) {
                        self.ranges[i] = further_merged;
                        self.ranges.remove(i + 1);
                    } else {
                        break;
                    }
                }
                return;
            }
            i += 1;
        }

        // No merge possible, insert in sorted order
        let pos = self.ranges.iter().position(|r| r.start > range.start)
            .unwrap_or(self.ranges.len());
        self.ranges.insert(pos, range);
    }

    /// Get the start point of the first range.
    pub fn start(&self) -> Option<ProgramPoint> {
        self.ranges.first().map(|r| r.start)
    }

    /// Get the end point of the last range.
    pub fn end(&self) -> Option<ProgramPoint> {
        self.ranges.last().map(|r| r.end)
    }

    /// Check if this interval is live at a program point.
    pub fn is_live_at(&self, point: ProgramPoint) -> bool {
        self.ranges.iter().any(|r| r.contains(point))
    }

    /// Check if this interval overlaps with another.
    pub fn overlaps(&self, other: &LiveInterval) -> bool {
        for r1 in &self.ranges {
            for r2 in &other.ranges {
                if r1.overlaps(r2) {
                    return true;
                }
            }
        }
        false
    }

    /// Total length of all ranges.
    pub fn total_length(&self) -> usize {
        self.ranges.iter().map(|r| r.len()).sum()
    }

    /// Check if this interval has a fixed register constraint.
    pub fn is_fixed(&self) -> bool {
        self.fixed_register.is_some()
    }

    /// Set a fixed register constraint.
    pub fn set_fixed(&mut self, reg: usize) {
        self.fixed_register = Some(reg);
    }

    /// Set a register hint.
    pub fn set_hint(&mut self, reg: usize) {
        self.register_hint = Some(reg);
    }
}

/// Result of interval analysis: all live intervals and program point mappings.
#[derive(Debug, Clone)]
pub struct IntervalAnalysis {
    /// All live intervals, indexed by an internal ID.
    pub intervals: Vec<LiveInterval>,

    /// Map from variable to interval index.
    pub variable_to_interval: HashMap<SsaVariable, usize>,

    /// Map from block ID to its starting program point.
    pub block_starts: HashMap<BlockId, ProgramPoint>,

    /// Map from block ID to its ending program point.
    pub block_ends: HashMap<BlockId, ProgramPoint>,

    /// Block order used for linearization.
    pub block_order: Vec<BlockId>,

    /// Total number of program points.
    pub num_program_points: usize,

    /// Program points that are call sites (where caller-saved registers are clobbered).
    ///
    /// Intervals that are live across any of these points must be assigned
    /// to callee-saved registers or spilled.
    pub call_sites: Vec<ProgramPoint>,
}

impl IntervalAnalysis {
    /// Compute live intervals from liveness analysis.
    ///
    /// This linearizes the blocks in order and computes the program point
    /// range for each variable based on its definition and uses.
    pub fn compute<V, I, F>(
        translator: &SSATranslator<V, I, F>,
        liveness: &LivenessAnalysis,
    ) -> Self
    where
        V: OptimizableValue,
        I: OptimizableInstruction<Value = V>,
        F: InstructionFactory<Instr = I>,
    {
        // Step 1: Order blocks (for now, just use the natural order)
        // TODO: Use reverse post-order for better allocation
        let block_order: Vec<BlockId> = translator.blocks.iter()
            .map(|b| b.id)
            .collect();

        // Step 2: Assign program points to blocks
        let mut block_starts: HashMap<BlockId, ProgramPoint> = HashMap::new();
        let mut block_ends: HashMap<BlockId, ProgramPoint> = HashMap::new();
        let mut current_point = 0usize;

        for &block_id in &block_order {
            block_starts.insert(block_id, ProgramPoint(current_point));

            let block = &translator.blocks[block_id.0];
            // Each instruction gets one program point
            // We also add points for phi nodes at the start
            let num_phis = translator.phis.values()
                .filter(|phi| phi.block_id == block_id)
                .count();
            current_point += num_phis;
            current_point += block.instructions.len();

            block_ends.insert(block_id, ProgramPoint(current_point));
        }

        let num_program_points = current_point;

        // Step 3: Build live intervals
        let mut intervals: Vec<LiveInterval> = Vec::new();
        let mut variable_to_interval: HashMap<SsaVariable, usize> = HashMap::new();

        // For each block, create intervals based on liveness
        for &block_id in &block_order {
            let block_start = block_starts[&block_id];
            let block_end = block_ends[&block_id];

            // Variables live at block entry are live from block start
            if let Some(live_in) = liveness.live_in.get(&block_id) {
                for var in live_in {
                    let interval_idx = *variable_to_interval
                        .entry(var.clone())
                        .or_insert_with(|| {
                            let idx = intervals.len();
                            intervals.push(LiveInterval::new(var.clone()));
                            idx
                        });

                    // Extend interval to cover this block
                    intervals[interval_idx].add_range(LiveRange::new(block_start, block_end));
                }
            }

            // Variables live at block exit are live to block end
            if let Some(live_out) = liveness.live_out.get(&block_id) {
                for var in live_out {
                    let interval_idx = *variable_to_interval
                        .entry(var.clone())
                        .or_insert_with(|| {
                            let idx = intervals.len();
                            intervals.push(LiveInterval::new(var.clone()));
                            idx
                        });

                    // Extend interval to cover this block
                    intervals[interval_idx].add_range(LiveRange::new(block_start, block_end));
                }
            }

            // Process phi definitions
            let mut point_offset = 0usize;
            for phi in translator.phis.values() {
                if phi.block_id == block_id {
                    if let Some(dest) = &phi.dest {
                        let def_point = ProgramPoint(block_start.0 + point_offset);
                        let interval_idx = *variable_to_interval
                            .entry(dest.clone())
                            .or_insert_with(|| {
                                let idx = intervals.len();
                                intervals.push(LiveInterval::new(dest.clone()));
                                idx
                            });

                        // Definition starts the interval
                        intervals[interval_idx].add_range(LiveRange::new(def_point, block_end));
                    }
                    point_offset += 1;
                }
            }

            // Process instruction definitions and uses
            let block = &translator.blocks[block_id.0];
            for (instr_idx, instr) in block.instructions.iter().enumerate() {
                let instr_point = ProgramPoint(block_start.0 + point_offset + instr_idx);

                // Uses: variable must be live up to this point
                instr.visit_values(|value| {
                    if let Some(var) = value.as_var() {
                        if let Some(&interval_idx) = variable_to_interval.get(var) {
                            // Extend to include the use
                            let current_start = intervals[interval_idx].start()
                                .unwrap_or(instr_point);
                            intervals[interval_idx].add_range(
                                LiveRange::new(current_start, ProgramPoint(instr_point.0 + 1))
                            );
                        }
                    }
                });

                // Definition: starts a new live range
                if let Some(dest) = instr.destination() {
                    let interval_idx = *variable_to_interval
                        .entry(dest.clone())
                        .or_insert_with(|| {
                            let idx = intervals.len();
                            intervals.push(LiveInterval::new(dest.clone()));
                            idx
                        });

                    // Definition creates a new range starting at this point
                    intervals[interval_idx].add_range(
                        LiveRange::new(instr_point, ProgramPoint(instr_point.0 + 1))
                    );
                }
            }
        }

        IntervalAnalysis {
            intervals,
            variable_to_interval,
            block_starts,
            block_ends,
            block_order,
            num_program_points,
            call_sites: Vec::new(),
        }
    }

    /// Compute live intervals from liveness analysis, with call site detection.
    ///
    /// This is like `compute()` but also tracks which program points are call
    /// sites (instructions that clobber caller-saved registers). Intervals that
    /// are live across call sites will have `crosses_call` set to true.
    ///
    /// The `is_call` closure should return `true` for instructions that clobber
    /// caller-saved registers (function calls, etc.).
    pub fn compute_with_call_sites<V, I, F, IsCall>(
        translator: &SSATranslator<V, I, F>,
        liveness: &LivenessAnalysis,
        is_call: IsCall,
    ) -> Self
    where
        V: OptimizableValue,
        I: OptimizableInstruction<Value = V>,
        F: InstructionFactory<Instr = I>,
        IsCall: Fn(&I) -> bool,
    {
        // Step 1: Order blocks (for now, just use the natural order)
        let block_order: Vec<BlockId> = translator.blocks.iter()
            .map(|b| b.id)
            .collect();

        // Step 2: Assign program points to blocks and track call sites
        let mut block_starts: HashMap<BlockId, ProgramPoint> = HashMap::new();
        let mut block_ends: HashMap<BlockId, ProgramPoint> = HashMap::new();
        let mut call_sites: Vec<ProgramPoint> = Vec::new();
        let mut current_point = 0usize;

        for &block_id in &block_order {
            block_starts.insert(block_id, ProgramPoint(current_point));

            let block = &translator.blocks[block_id.0];
            // Each instruction gets one program point
            // We also add points for phi nodes at the start
            let num_phis = translator.phis.values()
                .filter(|phi| phi.block_id == block_id)
                .count();
            let _phi_point_start = current_point;
            current_point += num_phis;

            // Track call sites while processing instructions
            for (instr_idx, instr) in block.instructions.iter().enumerate() {
                let instr_point = ProgramPoint(current_point + instr_idx);
                if is_call(instr) {
                    call_sites.push(instr_point);
                }
            }
            current_point += block.instructions.len();

            block_ends.insert(block_id, ProgramPoint(current_point));
        }

        let num_program_points = current_point;

        // Step 3: Build live intervals
        let mut intervals: Vec<LiveInterval> = Vec::new();
        let mut variable_to_interval: HashMap<SsaVariable, usize> = HashMap::new();

        // For each block, create intervals based on liveness
        for &block_id in &block_order {
            let block_start = block_starts[&block_id];
            let block_end = block_ends[&block_id];

            // Variables live at block entry are live from block start
            if let Some(live_in) = liveness.live_in.get(&block_id) {
                for var in live_in {
                    let interval_idx = *variable_to_interval
                        .entry(var.clone())
                        .or_insert_with(|| {
                            let idx = intervals.len();
                            intervals.push(LiveInterval::new(var.clone()));
                            idx
                        });

                    // Extend interval to cover this block
                    intervals[interval_idx].add_range(LiveRange::new(block_start, block_end));
                }
            }

            // Variables live at block exit are live to block end
            if let Some(live_out) = liveness.live_out.get(&block_id) {
                for var in live_out {
                    let interval_idx = *variable_to_interval
                        .entry(var.clone())
                        .or_insert_with(|| {
                            let idx = intervals.len();
                            intervals.push(LiveInterval::new(var.clone()));
                            idx
                        });

                    // Extend interval to cover this block
                    intervals[interval_idx].add_range(LiveRange::new(block_start, block_end));
                }
            }

            // Process phi definitions
            let mut point_offset = 0usize;
            for phi in translator.phis.values() {
                if phi.block_id == block_id {
                    if let Some(dest) = &phi.dest {
                        let def_point = ProgramPoint(block_start.0 + point_offset);
                        let interval_idx = *variable_to_interval
                            .entry(dest.clone())
                            .or_insert_with(|| {
                                let idx = intervals.len();
                                intervals.push(LiveInterval::new(dest.clone()));
                                idx
                            });

                        // Definition starts the interval
                        intervals[interval_idx].add_range(LiveRange::new(def_point, block_end));
                    }
                    point_offset += 1;
                }
            }

            // Process instruction definitions and uses
            let block = &translator.blocks[block_id.0];
            for (instr_idx, instr) in block.instructions.iter().enumerate() {
                let instr_point = ProgramPoint(block_start.0 + point_offset + instr_idx);

                // Uses: variable must be live up to this point
                instr.visit_values(|value| {
                    if let Some(var) = value.as_var() {
                        if let Some(&interval_idx) = variable_to_interval.get(var) {
                            // Extend to include the use
                            let current_start = intervals[interval_idx].start()
                                .unwrap_or(instr_point);
                            intervals[interval_idx].add_range(
                                LiveRange::new(current_start, ProgramPoint(instr_point.0 + 1))
                            );
                        }
                    }
                });

                // Definition: starts a new live range
                if let Some(dest) = instr.destination() {
                    let interval_idx = *variable_to_interval
                        .entry(dest.clone())
                        .or_insert_with(|| {
                            let idx = intervals.len();
                            intervals.push(LiveInterval::new(dest.clone()));
                            idx
                        });

                    // Definition creates a new range starting at this point
                    intervals[interval_idx].add_range(
                        LiveRange::new(instr_point, ProgramPoint(instr_point.0 + 1))
                    );
                }
            }
        }

        // Step 4: Mark intervals that cross call sites
        for interval in &mut intervals {
            for &call_point in &call_sites {
                // An interval crosses a call if:
                // 1. It is live at the call point
                // 2. It continues after the call (doesn't end at the call)
                if interval.is_live_at(call_point) {
                    if let Some(end) = interval.end() {
                        if end.0 > call_point.0 + 1 {
                            interval.crosses_call = true;
                            break;
                        }
                    }
                }
            }
        }

        IntervalAnalysis {
            intervals,
            variable_to_interval,
            block_starts,
            block_ends,
            block_order,
            num_program_points,
            call_sites,
        }
    }

    /// Get the interval for a variable.
    pub fn get_interval(&self, var: &SsaVariable) -> Option<&LiveInterval> {
        self.variable_to_interval.get(var)
            .map(|&idx| &self.intervals[idx])
    }

    /// Get a mutable reference to the interval for a variable.
    pub fn get_interval_mut(&mut self, var: &SsaVariable) -> Option<&mut LiveInterval> {
        self.variable_to_interval.get(var)
            .copied()
            .map(move |idx| &mut self.intervals[idx])
    }

    /// Get all intervals sorted by start point.
    pub fn intervals_sorted_by_start(&self) -> Vec<&LiveInterval> {
        let mut intervals: Vec<_> = self.intervals.iter().collect();
        intervals.sort_by_key(|i| i.start());
        intervals
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_program_point_ordering() {
        let p1 = ProgramPoint(0);
        let p2 = ProgramPoint(5);
        let p3 = ProgramPoint(10);

        assert!(p1 < p2);
        assert!(p2 < p3);
        assert!(p1 < p3);
    }

    #[test]
    fn test_live_range_contains() {
        let range = LiveRange::new(ProgramPoint(5), ProgramPoint(10));

        assert!(!range.contains(ProgramPoint(4)));
        assert!(range.contains(ProgramPoint(5)));
        assert!(range.contains(ProgramPoint(7)));
        assert!(range.contains(ProgramPoint(9)));
        assert!(!range.contains(ProgramPoint(10)));  // end is exclusive
    }

    #[test]
    fn test_live_range_overlaps() {
        let r1 = LiveRange::new(ProgramPoint(0), ProgramPoint(5));
        let r2 = LiveRange::new(ProgramPoint(3), ProgramPoint(8));
        let r3 = LiveRange::new(ProgramPoint(5), ProgramPoint(10));
        let r4 = LiveRange::new(ProgramPoint(10), ProgramPoint(15));

        assert!(r1.overlaps(&r2));  // overlap at 3-5
        assert!(!r1.overlaps(&r3)); // r1 ends where r3 starts (no overlap)
        assert!(r2.overlaps(&r3));  // overlap at 5-8
        assert!(!r3.overlaps(&r4)); // adjacent but not overlapping
    }

    #[test]
    fn test_live_range_merge() {
        let r1 = LiveRange::new(ProgramPoint(0), ProgramPoint(5));
        let r2 = LiveRange::new(ProgramPoint(5), ProgramPoint(10));
        let r3 = LiveRange::new(ProgramPoint(3), ProgramPoint(8));
        let r4 = LiveRange::new(ProgramPoint(15), ProgramPoint(20));

        // Adjacent ranges can merge
        let merged = r1.merge(&r2);
        assert!(merged.is_some());
        let m = merged.unwrap();
        assert_eq!(m.start, ProgramPoint(0));
        assert_eq!(m.end, ProgramPoint(10));

        // Overlapping ranges can merge
        let merged2 = r1.merge(&r3);
        assert!(merged2.is_some());
        let m2 = merged2.unwrap();
        assert_eq!(m2.start, ProgramPoint(0));
        assert_eq!(m2.end, ProgramPoint(8));

        // Non-adjacent ranges cannot merge
        assert!(r1.merge(&r4).is_none());
    }

    #[test]
    fn test_live_interval_add_range() {
        let mut interval = LiveInterval::new(SsaVariable::new("x"));

        interval.add_range(LiveRange::new(ProgramPoint(0), ProgramPoint(5)));
        interval.add_range(LiveRange::new(ProgramPoint(10), ProgramPoint(15)));
        interval.add_range(LiveRange::new(ProgramPoint(5), ProgramPoint(10)));

        // All ranges should be merged into one
        assert_eq!(interval.ranges.len(), 1);
        assert_eq!(interval.start(), Some(ProgramPoint(0)));
        assert_eq!(interval.end(), Some(ProgramPoint(15)));
    }

    #[test]
    fn test_live_interval_overlaps() {
        let mut i1 = LiveInterval::new(SsaVariable::new("x"));
        i1.add_range(LiveRange::new(ProgramPoint(0), ProgramPoint(10)));

        let mut i2 = LiveInterval::new(SsaVariable::new("y"));
        i2.add_range(LiveRange::new(ProgramPoint(5), ProgramPoint(15)));

        let mut i3 = LiveInterval::new(SsaVariable::new("z"));
        i3.add_range(LiveRange::new(ProgramPoint(10), ProgramPoint(20)));

        assert!(i1.overlaps(&i2));
        assert!(!i1.overlaps(&i3));  // i1 ends where i3 starts
        assert!(i2.overlaps(&i3));
    }
}
