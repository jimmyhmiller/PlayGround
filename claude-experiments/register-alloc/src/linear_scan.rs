//! Linear Scan Register Allocation
//!
//! Implementation of the algorithm from:
//!   Poletto & Sarkar, "Linear Scan Register Allocation",
//!   ACM TOPLAS 21(5), September 1999.
//!
//! The algorithm:
//! 1. Compute live intervals [start, end] for each vreg.
//! 2. Sort intervals by increasing start point.
//! 3. Scan intervals left to right, maintaining an `active` list
//!    (sorted by increasing end point) of currently-allocated intervals.
//! 4. For each new interval:
//!    a. ExpireOldIntervals — free registers of intervals that ended.
//!    b. If active is full (|active| = R), SpillAtInterval — spill
//!       the interval ending furthest in the future (Belady's heuristic).
//!    c. Otherwise, assign a free register.
//! 5. For spilled vregs, insert reload-before-use and store-after-def moves.

use std::collections::{HashMap, HashSet};

use crate::allocator::*;
use crate::ir::{Function, SafepointAction};
use crate::liveness::{self, LiveInterval, LivenessInfo};
use crate::target::Target;
use crate::types::*;

/// Linear scan register allocator (Poletto & Sarkar 1999).
pub struct LinearScanAllocator;

/// Internal state during allocation, per register class.
struct ClassState {
    /// Pool of free physical registers for this class.
    free_regs: Vec<PReg>,
    /// Active intervals, sorted by increasing end point.
    /// Each entry: (end_point, interval_index, assigned PReg).
    active: Vec<(u32, usize, PReg)>,
}

/// Per-interval allocation result.
#[derive(Clone, Debug)]
enum IntervalAlloc {
    /// Assigned to a physical register for its entire lifetime.
    Reg(PReg),
    /// Spilled to a stack slot.
    Spilled(SpillSlot),
    /// Rematerializable: the value is a constant that can be recomputed
    /// with a MOV immediate before each use. No register or spill slot needed.
    Remat(u64),
}

impl RegisterAllocator for LinearScanAllocator {
    fn name(&self) -> &str {
        "linear-scan"
    }

    fn allocate<F: Function, T: Target>(
        &mut self,
        func: &F,
        target: &T,
    ) -> Result<Allocation, AllocError> {
        // Step 1: Compute liveness.
        let liveness = liveness::compute_liveness(func);

        // Step 2: Run the linear scan core, per register class.
        let mut alloc_result: HashMap<VReg, IntervalAlloc> = HashMap::new();
        let mut num_spill_slots: u32 = 0;

        // Group intervals by register class.
        let mut class_intervals: HashMap<RegClass, Vec<usize>> = HashMap::new();
        for (idx, interval) in liveness.intervals.iter().enumerate() {
            class_intervals
                .entry(interval.class)
                .or_default()
                .push(idx);
        }

        for (&class, interval_indices) in &class_intervals {
            // Get allocatable registers for this class.
            let all_regs = target.allocatable_regs(class);
            if all_regs.is_empty() {
                return Err(AllocError::OutOfRegisters {
                    inst: InstId(0),
                    class,
                });
            }

            // Sort intervals by increasing start point (paper: "foreach
            // live interval i, in order of increasing start point").
            let mut sorted: Vec<usize> = interval_indices.clone();
            sorted.sort_by_key(|&idx| liveness.intervals[idx].start);

            let mut state = ClassState {
                free_regs: all_regs.into_iter().rev().collect(), // reversed so pop() gives first-preference
                active: Vec::new(),
            };

            for &idx in &sorted {
                let interval = &liveness.intervals[idx];

                // Skip rematerializable values — they don't need registers.
                if let Some(imm) = func.remat_value(interval.vreg) {
                    alloc_result.insert(interval.vreg, IntervalAlloc::Remat(imm));
                    continue;
                }

                // EXPIREOLDINTERVALS(i)
                expire_old_intervals(&mut state, interval.start);

                let num_allocatable = state.free_regs.len() + state.active.len();

                if state.active.len() == num_allocatable {
                    // All registers in use: SPILLATINTERVAL(i)
                    let spill_slot = spill_at_interval(
                        &mut state,
                        idx,
                        interval,
                        &liveness,
                        &mut alloc_result,
                        &mut num_spill_slots,
                    );
                    if let Some(slot) = spill_slot {
                        alloc_result.insert(interval.vreg, IntervalAlloc::Spilled(slot));
                    }
                    // else: interval got a register (stolen from the spilled one)
                } else {
                    // Try to assign a register from the free pool.
                    // pick_free_reg returns None if all free regs are
                    // clobbered within this interval — must spill instead.
                    match pick_free_reg(&mut state, interval, &liveness) {
                        Some(preg) => {
                            alloc_result.insert(interval.vreg, IntervalAlloc::Reg(preg));
                            insert_active(&mut state, interval.end, idx, preg);
                        }
                        None => {
                            // All free registers would be clobbered. Spill this interval.
                            let slot = SpillSlot(num_spill_slots);
                            num_spill_slots += 1;
                            alloc_result.insert(interval.vreg, IntervalAlloc::Spilled(slot));
                        }
                    }
                }
            }
        }

        // Step 3: Build the Allocation from the per-interval results.
        let mut alloc = build_allocation(func, target, &liveness, &alloc_result, num_spill_slots)?;

        // Step 4: Build stackmaps for safepoint instructions.
        build_stackmaps(func, &liveness, &alloc_result, &mut alloc);

        Ok(alloc)
    }
}

/// EXPIREOLDINTERVALS(i) from the paper.
///
/// "foreach interval j in active, in order of increasing end point:
///    if endpoint[j] > startpoint[i] then return
///    remove j from active
///    add register[j] to pool of free registers"
fn expire_old_intervals(state: &mut ClassState, current_start: u32) {
    // active is sorted by increasing end point, so we can stop early.
    let i = 0;
    while i < state.active.len() {
        let (end, _, preg) = state.active[i];
        if end >= current_start {
            // This interval (and all following) still overlap.
            break;
        }
        // This interval has expired.
        state.free_regs.push(preg);
        state.active.remove(i);
        // Don't increment i — next element slid into position i.
    }
}

/// SPILLATINTERVAL(i) from the paper.
///
/// "spill ← last interval in active
///  if endpoint[spill] > endpoint[i] then
///      register[i] ← register[spill]
///      location[spill] ← new stack location
///      remove spill from active
///      add i to active, sorted by increasing end point
///  else
///      location[i] ← new stack location"
///
/// Returns Some(slot) if the NEW interval is spilled,
/// or None if it stole a register (the old interval was spilled internally).
fn spill_at_interval(
    state: &mut ClassState,
    new_idx: usize,
    new_interval: &LiveInterval,
    liveness: &LivenessInfo,
    alloc_result: &mut HashMap<VReg, IntervalAlloc>,
    num_spill_slots: &mut u32,
) -> Option<SpillSlot> {
    // "spill ← last interval in active" (last = longest end point)
    // active is sorted by increasing end point, so last is at the back.
    let (spill_end, spill_idx, spill_preg) = *state.active.last().unwrap();

    if spill_end > new_interval.end {
        // The interval in active ends later — spill IT, give its register
        // to the new interval.
        let spill_vreg = liveness.intervals[spill_idx].vreg;
        let slot = SpillSlot(*num_spill_slots);
        *num_spill_slots += 1;

        // register[i] ← register[spill]
        alloc_result.insert(new_interval.vreg, IntervalAlloc::Reg(spill_preg));
        // location[spill] ← new stack location
        alloc_result.insert(spill_vreg, IntervalAlloc::Spilled(slot));

        // remove spill from active
        state.active.pop();
        // add i to active, sorted by increasing end point
        insert_active(state, new_interval.end, new_idx, spill_preg);

        None // new interval got a register
    } else {
        // The new interval ends later (or same) — spill the new interval.
        let slot = SpillSlot(*num_spill_slots);
        *num_spill_slots += 1;
        Some(slot) // new interval is spilled
    }
}

/// Check if a physical register is clobbered at any point within [start, end].
/// Uses sorted clobber positions for fast range queries.
fn is_clobbered_in_range(
    preg: PReg,
    start: u32,
    end: u32,
    liveness: &LivenessInfo,
) -> bool {
    let positions = &liveness.sorted_clobber_positions;
    let lo = positions.partition_point(|&p| p < start);
    for &pos in &positions[lo..] {
        if pos > end { break; }
        if let Some(clobbers) = liveness.clobber_points.get(&pos) {
            if clobbers.contains(&preg) {
                return true;
            }
        }
    }
    false
}

/// Pick a free register, preferring:
/// 1. Fixed-hint register (if available)
/// 2. A register NOT clobbered during this interval's lifetime
/// 3. Any available register (caller-saved fallback)
fn pick_free_reg(
    state: &mut ClassState,
    interval: &LiveInterval,
    liveness: &LivenessInfo,
) -> Option<PReg> {
    // If there's a fixed-register hint, try it — but only if it's not clobbered
    // during this interval's lifetime (otherwise we'd lose the value at a call site).
    if let Some(hint) = interval.fixed_hint {
        if let Some(pos) = state.free_regs.iter().position(|&r| r == hint) {
            if !is_clobbered_in_range(hint, interval.start, interval.end, liveness) {
                return Some(state.free_regs.remove(pos));
            }
        }
    }

    // Prefer a register that isn't clobbered during this interval's range.
    // free_regs is stored reversed, so we search from the back (highest preference first).
    if !liveness.clobber_points.is_empty() {
        for i in (0..state.free_regs.len()).rev() {
            if !is_clobbered_in_range(state.free_regs[i], interval.start, interval.end, liveness) {
                return Some(state.free_regs.remove(i));
            }
        }
        // All free registers are clobbered within this interval.
        // Return None to force a spill.
        return None;
    }

    // No clobber points at all — any register is fine.
    state.free_regs.pop()
}

/// Try to steal a non-clobbered (callee-saved) register from an active
/// interval that doesn't need clobber protection. The evicted interval
/// gets a caller-saved register (which is safe for it since it doesn't
/// span any clobber points). Returns the stolen PReg, or None if no
/// suitable swap exists.
fn steal_safe_reg(
    state: &mut ClassState,
    _new_idx: usize,
    new_interval: &LiveInterval,
    liveness: &LivenessInfo,
    alloc_result: &mut HashMap<VReg, IntervalAlloc>,
    num_spill_slots: &mut u32,
) -> Option<PReg> {
    // Find an active interval whose register is NOT clobbered within
    // the new interval's range AND whose own interval doesn't span
    // a clobber of the free caller-saved regs we'd give it.
    for i in 0..state.active.len() {
        let (_, _, active_preg) = state.active[i];

        // Is this register safe for the new interval?
        if is_clobbered_in_range(active_preg, new_interval.start, new_interval.end, liveness) {
            continue;
        }

        // Can the evicted interval use a caller-saved register?
        // Check if any free register (all are clobbered for new_interval,
        // but might not be clobbered for the active interval).
        let active_vreg = liveness.intervals[state.active[i].1].vreg;
        let active_start = liveness.intervals[state.active[i].1].start;
        let active_end = state.active[i].0;

        let replacement = state.free_regs.iter().position(|&r| {
            !is_clobbered_in_range(r, active_start, active_end, liveness)
        });

        if let Some(free_idx) = replacement {
            let replacement_preg = state.free_regs.remove(free_idx);

            // Steal: give the active interval the caller-saved reg,
            // and take its callee-saved reg for the new interval.
            alloc_result.insert(active_vreg, IntervalAlloc::Reg(replacement_preg));
            state.active[i].2 = replacement_preg;

            // The stolen register is NOT returned to free_regs;
            // it's given to the new interval by the caller.
            return Some(active_preg);
        }
    }

    // No swap found. Last resort: spill the new interval.
    // (Caller handles this by inserting a spill slot.)
    let _ = num_spill_slots;
    None
}

/// Insert into the active list maintaining sort by increasing end point.
fn insert_active(state: &mut ClassState, end: u32, idx: usize, preg: PReg) {
    let pos = state
        .active
        .binary_search_by_key(&end, |&(e, _, _)| e)
        .unwrap_or_else(|p| p);
    state.active.insert(pos, (end, idx, preg));
}

/// Build the final Allocation from the per-vreg allocation decisions.
///
/// For each instruction, look at its operands and fill in the physical
/// register. For spilled vregs, insert reload/store moves.
///
/// When a spilled vreg needs a temp register but all registers are
/// occupied by non-spilled intervals, we "borrow" a register by
/// saving its current occupant to a scratch spill slot, using the
/// register for the reload, and restoring after the instruction.
fn build_allocation<F: Function, T: Target>(
    func: &F,
    target: &T,
    _liveness: &LivenessInfo,
    alloc_result: &HashMap<VReg, IntervalAlloc>,
    num_spill_slots: u32,
) -> Result<Allocation, AllocError> {
    let mut alloc = Allocation::new();
    alloc.num_spill_slots = num_spill_slots;

    // Record spill slots and vreg homes.
    for (vreg, result) in alloc_result {
        match result {
            IntervalAlloc::Spilled(slot) => {
                alloc.spill_slots.insert(*vreg, *slot);
                alloc.vreg_homes.insert(*vreg, MoveOperand::SpillSlot(*slot));
            }
            IntervalAlloc::Reg(preg) => {
                alloc.vreg_homes.insert(*vreg, MoveOperand::Reg(*preg));
            }
            IntervalAlloc::Remat(imm) => {
                alloc.vreg_homes.insert(*vreg, MoveOperand::Remat(*imm));
            }
        }
    }

    // Build a map from PReg → VReg for register-allocated intervals,
    // so we know who occupies each register when we need to evict.
    let mut preg_owner: HashMap<PReg, VReg> = HashMap::new();
    for (&vreg, result) in alloc_result {
        if let IntervalAlloc::Reg(preg) = result {
            preg_owner.insert(*preg, vreg);
        }
    }

    for block in func.blocks() {
        for inst in func.block_insts(block) {
            let operands: Vec<Operand> = func.inst_operands(inst).collect();
            let mut op_assignments: Vec<Option<PReg>> = vec![None; operands.len()];

            // Track which pregs are used BY OPERANDS of this instruction.
            let mut operand_pregs: HashSet<PReg> = HashSet::new();
            let clobbers = func.inst_clobbers(inst);
            for &p in clobbers {
                operand_pregs.insert(p);
            }

            // First pass: assign non-spilled USE operands first, so that
            // Reuse/Tied defs can reference them.
            for (op_idx, operand) in operands.iter().enumerate() {
                if operand.kind != OperandKind::Use {
                    continue;
                }
                if let Reg::Virtual(vreg) = operand.reg {
                    if let Some(IntervalAlloc::Reg(preg)) = alloc_result.get(&vreg) {
                        let final_preg = resolve_constraint(
                            operand, *preg, &op_assignments,
                        );
                        // If the constraint requires a different register than
                        // the interval's home, insert a move before the instruction.
                        if final_preg != *preg {
                            alloc.moves.push(InsertedMove {
                                at: MovePosition::Before(inst),
                                from: MoveOperand::Reg(*preg),
                                to: MoveOperand::Reg(final_preg),
                                class: func.vreg_class(vreg),
                            });
                        }
                        op_assignments[op_idx] = Some(final_preg);
                        operand_pregs.insert(final_preg);
                    }
                }
            }
            // Then assign non-spilled DEF/USEDEF/EARLYDEF operands.
            for (op_idx, operand) in operands.iter().enumerate() {
                if operand.kind == OperandKind::Use {
                    continue;
                }
                if op_assignments[op_idx].is_some() {
                    continue;
                }
                if let Reg::Virtual(vreg) = operand.reg {
                    if let Some(IntervalAlloc::Reg(preg)) = alloc_result.get(&vreg) {
                        let final_preg = resolve_constraint(
                            operand, *preg, &op_assignments,
                        );
                        // If the constraint requires a different register than
                        // the interval's home, insert a move after the instruction.
                        if final_preg != *preg {
                            alloc.moves.push(InsertedMove {
                                at: MovePosition::After(inst),
                                from: MoveOperand::Reg(final_preg),
                                to: MoveOperand::Reg(*preg),
                                class: func.vreg_class(vreg),
                            });
                        }
                        op_assignments[op_idx] = Some(final_preg);
                        operand_pregs.insert(final_preg);
                    }
                }
            }

            // Second pass: handle rematerialized and spilled operands.
            for (op_idx, operand) in operands.iter().enumerate() {
                if op_assignments[op_idx].is_some() {
                    continue;
                }
                // Remat: record a Remat move. The emitter materializes the
                // constant directly into the required register.
                if let Reg::Virtual(vreg) = operand.reg {
                    if let Some(IntervalAlloc::Remat(imm)) = alloc_result.get(&vreg) {
                        let class = func.vreg_class(vreg);
                        let temp_preg = match &operand.constraint {
                            OperandConstraint::FixedReg(preg) => *preg,
                            _ => {
                                // Pick any non-conflicting register
                                let available = target.allocatable_regs(class);
                                *available.iter()
                                    .find(|r| !operand_pregs.contains(r))
                                    .unwrap_or(&available[0])
                            }
                        };
                        op_assignments[op_idx] = Some(temp_preg);
                        operand_pregs.insert(temp_preg);
                        if operand.kind == OperandKind::Use || operand.kind == OperandKind::UseDef {
                            alloc.moves.push(InsertedMove {
                                at: MovePosition::Before(inst),
                                from: MoveOperand::Remat(*imm),
                                to: MoveOperand::Reg(temp_preg),
                                class,
                            });
                        }
                        continue;
                    }
                }
            }
            // Third pass: handle spilled operands.
            // For each, we need a temp register. If all are occupied by
            // non-spilled operands of THIS instruction, we evict a register
            // that holds a value NOT used by this instruction.
            for (op_idx, operand) in operands.iter().enumerate() {
                if op_assignments[op_idx].is_some() {
                    continue;
                }
                if let Reg::Virtual(vreg) = operand.reg {
                    if let Some(IntervalAlloc::Spilled(slot)) = alloc_result.get(&vreg) {
                        let class = func.vreg_class(vreg);

                        let temp_preg = match &operand.constraint {
                            OperandConstraint::FixedReg(preg) => *preg,
                            OperandConstraint::Reuse(reuse_idx) => {
                                op_assignments[*reuse_idx].unwrap()
                            }
                            OperandConstraint::Tied(tied_idx) => {
                                op_assignments[*tied_idx].unwrap()
                            }
                            _ => {
                                let available = target.allocatable_regs(class);

                                // For a regular Def (not EarlyDef), the output is
                                // written after inputs are read, so it CAN reuse a
                                // register occupied by a Use operand at this instruction.
                                let is_late_def = operand.kind == OperandKind::Def;

                                // Try to find a register not conflicting with this operand.
                                let conflicts = |r: &PReg| -> bool {
                                    if is_late_def {
                                        // Late def: only conflicts with other defs and
                                        // spilled uses (which are also loaded before the inst).
                                        // Does NOT conflict with non-spilled uses.
                                        operands.iter().enumerate().any(|(oi, op)| {
                                            if oi == op_idx { return false; }
                                            if let Some(assigned) = op_assignments[oi] {
                                                if assigned != *r { return false; }
                                                // Conflicts with defs, early_defs, usedefs
                                                // but NOT with pure uses (they read before we write).
                                                !matches!(op.kind, OperandKind::Use)
                                            } else {
                                                false
                                            }
                                        })
                                    } else {
                                        // EarlyDef, Use, UseDef: conflicts with all assigned operands.
                                        operand_pregs.contains(r)
                                    }
                                };

                                // Find a temp register. Prefer one not owned by
                                // another live vreg. If we must borrow, save/restore
                                // the owner around this instruction.
                                let non_conflicting: Vec<PReg> = available.iter()
                                    .filter(|r| !conflicts(r))
                                    .copied()
                                    .collect();

                                if non_conflicting.is_empty() {
                                    // All registers conflict with this inst's operands.
                                    // Evict a register not used by THIS instruction.
                                    let evict_preg = available.iter()
                                        .find(|r| {
                                            !operands.iter().enumerate().any(|(oi, _)| {
                                                if oi == op_idx { return false; }
                                                op_assignments[oi] == Some(**r)
                                            })
                                        })
                                        .ok_or(AllocError::OutOfRegisters { inst, class })?;
                                    let evict_preg = *evict_preg;
                                    if let Some(&evicted_vreg) = preg_owner.get(&evict_preg) {
                                        let evicted_slot = alloc.spill_slots
                                            .get(&evicted_vreg)
                                            .copied()
                                            .unwrap_or_else(|| alloc.add_spill(evicted_vreg));
                                        alloc.moves.push(InsertedMove {
                                            at: MovePosition::Before(inst),
                                            from: MoveOperand::Reg(evict_preg),
                                            to: MoveOperand::SpillSlot(evicted_slot),
                                            class,
                                        });
                                        alloc.moves.push(InsertedMove {
                                            at: MovePosition::After(inst),
                                            from: MoveOperand::SpillSlot(evicted_slot),
                                            to: MoveOperand::Reg(evict_preg),
                                            class,
                                        });
                                    }
                                    evict_preg
                                } else {
                                    // Prefer a register that's truly free (no owner).
                                    let truly_free = non_conflicting.iter()
                                        .find(|r| !preg_owner.contains_key(r));
                                    if let Some(&free) = truly_free {
                                        free
                                    } else {
                                        // Must borrow from a live owner. Pick one and
                                        // save/restore it.
                                        let borrow = non_conflicting[0];
                                        if let Some(&evicted_vreg) = preg_owner.get(&borrow) {
                                            let evicted_slot = alloc.spill_slots
                                                .get(&evicted_vreg)
                                                .copied()
                                                .unwrap_or_else(|| alloc.add_spill(evicted_vreg));
                                            alloc.moves.push(InsertedMove {
                                                at: MovePosition::Before(inst),
                                                from: MoveOperand::Reg(borrow),
                                                to: MoveOperand::SpillSlot(evicted_slot),
                                                class,
                                            });
                                            alloc.moves.push(InsertedMove {
                                                at: MovePosition::After(inst),
                                                from: MoveOperand::SpillSlot(evicted_slot),
                                                to: MoveOperand::Reg(borrow),
                                                class,
                                            });
                                        }
                                        borrow
                                    }
                                }
                            }
                        };

                        op_assignments[op_idx] = Some(temp_preg);
                        operand_pregs.insert(temp_preg);

                        // Insert spill code moves.
                        match operand.kind {
                            OperandKind::Use => {
                                alloc.moves.push(InsertedMove {
                                    at: MovePosition::Before(inst),
                                    from: MoveOperand::SpillSlot(*slot),
                                    to: MoveOperand::Reg(temp_preg),
                                    class,
                                });
                            }
                            OperandKind::Def | OperandKind::EarlyDef => {
                                alloc.moves.push(InsertedMove {
                                    at: MovePosition::After(inst),
                                    from: MoveOperand::Reg(temp_preg),
                                    to: MoveOperand::SpillSlot(*slot),
                                    class,
                                });
                            }
                            OperandKind::UseDef => {
                                alloc.moves.push(InsertedMove {
                                    at: MovePosition::Before(inst),
                                    from: MoveOperand::SpillSlot(*slot),
                                    to: MoveOperand::Reg(temp_preg),
                                    class,
                                });
                                alloc.moves.push(InsertedMove {
                                    at: MovePosition::After(inst),
                                    from: MoveOperand::Reg(temp_preg),
                                    to: MoveOperand::SpillSlot(*slot),
                                    class,
                                });
                            }
                        }
                    }
                }
            }

            // Write final assignments to the allocation.
            for (op_idx, _operand) in operands.iter().enumerate() {
                if let Some(preg) = op_assignments[op_idx] {
                    alloc.set(inst, op_idx, preg);
                }
            }
        }
    }

    // ── Block-edge moves (phi resolution) ─────────────────────
    //
    // For each branch, match the branch_args (source vregs) to the
    // successor's block_params (destination vregs). If their allocated
    // registers differ, insert a BlockEdge move.
    for block in func.blocks() {
        for inst in func.block_insts(block) {
            if !func.is_branch(inst) {
                continue;
            }
            let succs: Vec<BlockId> = func.block_succs(block).collect();
            for (succ_idx, &succ) in succs.iter().enumerate() {
                let args = func.branch_args(inst, succ_idx);
                let params = func.block_params(succ);
                for (arg_vreg, param_vreg) in args.iter().zip(params.iter()) {
                    let arg_loc = alloc_result.get(arg_vreg);
                    let param_loc = alloc_result.get(param_vreg);

                    let from = match arg_loc {
                        Some(IntervalAlloc::Reg(preg)) => MoveOperand::Reg(*preg),
                        Some(IntervalAlloc::Spilled(slot)) => MoveOperand::SpillSlot(*slot),
                        Some(IntervalAlloc::Remat(imm)) => MoveOperand::Remat(*imm),
                        None => continue,
                    };
                    let to = match param_loc {
                        Some(IntervalAlloc::Reg(preg)) => MoveOperand::Reg(*preg),
                        Some(IntervalAlloc::Spilled(slot)) => MoveOperand::SpillSlot(*slot),
                        Some(IntervalAlloc::Remat(imm)) => MoveOperand::Remat(*imm),
                        None => continue,
                    };

                    if from != to {
                        alloc.moves.push(InsertedMove {
                            at: MovePosition::BlockEdge { from: block, to: succ },
                            from,
                            to,
                            class: func.vreg_class(*arg_vreg),
                        });
                    }
                }
            }
        }
    }

    Ok(alloc)
}

/// Build stackmaps for all safepoint instructions.
///
/// For each safepoint, finds all live vregs, queries the function for the
/// desired `SafepointAction`, and records their locations. For `SpillAndRecord`
/// values that are currently in registers, inserts spill moves and records
/// the stack slot.
fn build_stackmaps<F: Function>(
    func: &F,
    liveness: &LivenessInfo,
    alloc_result: &HashMap<VReg, IntervalAlloc>,
    alloc: &mut Allocation,
) {
    for block in func.blocks() {
        for inst in func.block_insts(block) {
            if !func.is_safepoint(inst) {
                continue;
            }

            let pos = match liveness.inst_position.get(&inst) {
                Some(&p) => p,
                None => continue,
            };

            let mut entries = Vec::new();

            for interval in &liveness.intervals {
                // Is this vreg live at this instruction?
                if pos < interval.start || pos > interval.end {
                    continue;
                }

                let action = func.safepoint_action(inst, interval.vreg);

                match action {
                    SafepointAction::CallingConvention | SafepointAction::Ignore => {
                        // Nothing to record.
                        continue;
                    }
                    SafepointAction::Record => {
                        // Record current location without moving anything.
                        let location = match alloc_result.get(&interval.vreg) {
                            Some(IntervalAlloc::Reg(preg)) => MoveOperand::Reg(*preg),
                            Some(IntervalAlloc::Spilled(slot)) => MoveOperand::SpillSlot(*slot),
                            Some(IntervalAlloc::Remat(imm)) => MoveOperand::Remat(*imm),
                            None => continue,
                        };
                        entries.push(StackmapEntry {
                            vreg: interval.vreg,
                            location,
                            action,
                        });
                    }
                    SafepointAction::SpillAndRecord => {
                        // Value must be on the stack at this safepoint.
                        match alloc_result.get(&interval.vreg) {
                            Some(IntervalAlloc::Spilled(slot)) => {
                                // Already on the stack.
                                entries.push(StackmapEntry {
                                    vreg: interval.vreg,
                                    location: MoveOperand::SpillSlot(*slot),
                                    action,
                                });
                            }
                            Some(IntervalAlloc::Reg(preg)) => {
                                // In a register — need to spill before the safepoint
                                // and reload after.
                                let slot = *alloc.spill_slots
                                    .entry(interval.vreg)
                                    .or_insert_with(|| {
                                        let s = SpillSlot(alloc.num_spill_slots);
                                        alloc.num_spill_slots += 1;
                                        s
                                    });
                                let class = interval.class;
                                alloc.moves.push(InsertedMove {
                                    at: MovePosition::Before(inst),
                                    from: MoveOperand::Reg(*preg),
                                    to: MoveOperand::SpillSlot(slot),
                                    class,
                                });
                                alloc.moves.push(InsertedMove {
                                    at: MovePosition::After(inst),
                                    from: MoveOperand::SpillSlot(slot),
                                    to: MoveOperand::Reg(*preg),
                                    class,
                                });
                                entries.push(StackmapEntry {
                                    vreg: interval.vreg,
                                    location: MoveOperand::SpillSlot(slot),
                                    action,
                                });
                            }
                            Some(IntervalAlloc::Remat(_)) => {
                                // Constants don't need GC tracking
                                continue;
                            }
                            None => continue,
                        }
                    }
                }
            }

            if !entries.is_empty() {
                alloc.stackmaps.insert(inst, entries);
            }
        }
    }
}

/// Resolve a constraint: return the physical register this operand should use.
fn resolve_constraint(
    operand: &Operand,
    allocated: PReg,
    prior_assignments: &[Option<PReg>],
) -> PReg {
    match &operand.constraint {
        OperandConstraint::FixedReg(preg) => *preg,
        OperandConstraint::Tied(tied_idx) => {
            prior_assignments[*tied_idx].unwrap_or(allocated)
        }
        OperandConstraint::Reuse(reuse_idx) => {
            prior_assignments[*reuse_idx].unwrap_or(allocated)
        }
        _ => allocated,
    }
}
