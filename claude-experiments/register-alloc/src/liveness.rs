//! Liveness analysis: compute live intervals for each virtual register.
//!
//! Uses standard backward dataflow to compute live-in/live-out sets per block,
//! then builds a single [start, end] interval per vreg (conservative
//! approximation as in the Poletto & Sarkar paper).

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::ir::Function;
use crate::types::*;

/// A live interval for a virtual register: the vreg is live from
/// `start` to `end` (inclusive on both ends, as in the paper).
#[derive(Clone, Debug)]
pub struct LiveInterval {
    pub vreg: VReg,
    pub class: RegClass,
    pub start: u32,
    pub end: u32,
    /// Instructions where this vreg is used (read).
    pub use_points: Vec<u32>,
    /// Instructions where this vreg is defined (written).
    pub def_points: Vec<u32>,
    /// If this operand has a fixed-register constraint at any point,
    /// record it. (first, last_resort hint for the allocator)
    pub fixed_hint: Option<PReg>,
}

/// Result of liveness analysis.
pub struct LivenessInfo {
    /// Live intervals, one per vreg that is actually used.
    pub intervals: Vec<LiveInterval>,
    /// Mapping from vreg index to interval index.
    pub vreg_to_interval: HashMap<VReg, usize>,
    /// Live-in sets per block.
    pub live_in: Vec<HashSet<VReg>>,
    /// Live-out sets per block.
    pub live_out: Vec<HashSet<VReg>>,
    /// Linearized instruction order: inst_order[position] = InstId.
    pub inst_order: Vec<InstId>,
    /// Reverse map: InstId -> position in the linear order.
    pub inst_position: HashMap<InstId, u32>,
    /// Clobber points: position → set of physical registers clobbered at that point.
    /// Only populated for instructions that have non-empty clobber lists.
    pub clobber_points: HashMap<u32, Vec<PReg>>,
}

/// Compute live intervals for all virtual registers in a function.
///
/// Instructions are numbered in the order they appear when walking
/// blocks in layout order (depth-first order as suggested by the paper).
pub fn compute_liveness<F: Function>(func: &F) -> LivenessInfo {
    // Step 1: Linearize instructions — assign each a sequential number.
    let mut inst_order: Vec<InstId> = Vec::new();
    let mut inst_position: HashMap<InstId, u32> = HashMap::new();
    let mut block_ranges: Vec<(u32, u32)> = Vec::new(); // (first_inst_pos, last_inst_pos) per block

    let blocks: Vec<BlockId> = func.blocks().collect();

    for &block in &blocks {
        let first = inst_order.len() as u32;
        let mut count = 0u32;
        for inst in func.block_insts(block) {
            let pos = inst_order.len() as u32;
            inst_position.insert(inst, pos);
            inst_order.push(inst);
            count += 1;
        }
        if count == 0 {
            // Empty block — use a sentinel range where first > last.
            // Use (u32::MAX, 0) as the sentinel so first > last is always true.
            block_ranges.push((u32::MAX, 0));
        } else {
            let last = (inst_order.len() - 1) as u32;
            block_ranges.push((first, last));
        }
    }

    let num_blocks = blocks.len();

    // Step 2: Compute live-in and live-out sets using backward dataflow.
    // live_in[b] = use[b] ∪ (live_out[b] - def[b])
    // live_out[b] = ∪ {live_in[s] : s ∈ successors(b)} ∪ branch_args flowing to successors

    // Compute use/def sets per block.
    let mut block_use: Vec<HashSet<VReg>> = vec![HashSet::new(); num_blocks];
    let mut block_def: Vec<HashSet<VReg>> = vec![HashSet::new(); num_blocks];

    for (bi, &block) in blocks.iter().enumerate() {
        // Block params are defs at block entry.
        for &vreg in func.block_params(block) {
            block_def[bi].insert(vreg);
        }

        for inst in func.block_insts(block) {
            let operands: Vec<Operand> = func.inst_operands(inst).collect();
            for op in &operands {
                if let Reg::Virtual(vreg) = op.reg {
                    match op.kind {
                        OperandKind::Use => {
                            if !block_def[bi].contains(&vreg) {
                                block_use[bi].insert(vreg);
                            }
                        }
                        OperandKind::Def | OperandKind::EarlyDef => {
                            block_def[bi].insert(vreg);
                        }
                        OperandKind::UseDef => {
                            if !block_def[bi].contains(&vreg) {
                                block_use[bi].insert(vreg);
                            }
                            block_def[bi].insert(vreg);
                        }
                    }
                }
            }
        }
    }

    let mut live_in: Vec<HashSet<VReg>> = vec![HashSet::new(); num_blocks];
    let mut live_out: Vec<HashSet<VReg>> = vec![HashSet::new(); num_blocks];

    // Iterate until fixed point.
    let mut changed = true;
    while changed {
        changed = false;
        // Process blocks in reverse order for faster convergence.
        for bi in (0..num_blocks).rev() {
            let block = blocks[bi];

            // live_out[b] = ∪ live_in[s] for each successor s
            // plus: branch args flowing to successor block params
            let mut new_live_out = HashSet::new();
            let succs: Vec<BlockId> = func.block_succs(block).collect();
            for &succ in &succs {
                let si = succ.0 as usize;
                for &vreg in &live_in[si] {
                    new_live_out.insert(vreg);
                }
            }
            // Branch args: the last instruction in the block might be a branch
            // that passes args to successors. Those args are live-out.
            if !block_ranges.is_empty() {
                let (first, last) = block_ranges[bi];
                if first <= last {
                    let last_inst = inst_order[last as usize];
                    if func.is_branch(last_inst) {
                        for (succ_idx, _) in succs.iter().enumerate() {
                            let args = func.branch_args(last_inst, succ_idx);
                            for &vreg in args {
                                new_live_out.insert(vreg);
                            }
                        }
                    }
                }
            }

            if new_live_out != live_out[bi] {
                live_out[bi] = new_live_out;
                changed = true;
            }

            // live_in[b] = use[b] ∪ (live_out[b] - def[b])
            let mut new_live_in = block_use[bi].clone();
            for &vreg in &live_out[bi] {
                if !block_def[bi].contains(&vreg) {
                    new_live_in.insert(vreg);
                }
            }
            // Also include block params' sources — but those are handled via
            // branch_args in predecessors' live_out, not here.

            if new_live_in != live_in[bi] {
                live_in[bi] = new_live_in;
                changed = true;
            }
        }
    }

    // Step 3: Build live intervals.
    // For each vreg, the interval is [first_def_or_live_in, last_use_or_live_out].
    // This is the conservative single-interval approach from the paper.

    let mut vreg_start: HashMap<VReg, u32> = HashMap::new();
    let mut vreg_end: HashMap<VReg, u32> = HashMap::new();
    let mut vreg_uses: HashMap<VReg, Vec<u32>> = HashMap::new();
    let mut vreg_defs: HashMap<VReg, Vec<u32>> = HashMap::new();
    let mut vreg_fixed_hint: HashMap<VReg, PReg> = HashMap::new();

    for (bi, &block) in blocks.iter().enumerate() {
        let (first_pos, last_pos) = block_ranges[bi];
        let block_has_insts = first_pos <= last_pos;

        // Vregs live-in to this block: their interval starts at (or extends to)
        // the beginning of this block.
        if block_has_insts {
            for &vreg in &live_in[bi] {
                let entry = vreg_start.entry(vreg).or_insert(first_pos);
                *entry = (*entry).min(first_pos);
                let entry = vreg_end.entry(vreg).or_insert(first_pos);
                *entry = (*entry).max(first_pos);
            }

            // Vregs live-out of this block: their interval extends to the end.
            for &vreg in &live_out[bi] {
                let entry = vreg_start.entry(vreg).or_insert(last_pos);
                *entry = (*entry).min(last_pos);
                let entry = vreg_end.entry(vreg).or_insert(last_pos);
                *entry = (*entry).max(last_pos);
            }
        }

        // Block params: defined at block entry.
        for &vreg in func.block_params(block) {
            let entry = vreg_start.entry(vreg).or_insert(first_pos);
            *entry = (*entry).min(first_pos);
            let entry = vreg_end.entry(vreg).or_insert(first_pos);
            *entry = (*entry).max(first_pos);
            vreg_defs.entry(vreg).or_default().push(first_pos);
        }

        // Process instructions.
        for inst in func.block_insts(block) {
            let pos = inst_position[&inst];
            let operands: Vec<Operand> = func.inst_operands(inst).collect();

            for op in &operands {
                if let Reg::Virtual(vreg) = op.reg {
                    match op.kind {
                        OperandKind::Use => {
                            let entry = vreg_end.entry(vreg).or_insert(pos);
                            *entry = (*entry).max(pos);
                            let entry = vreg_start.entry(vreg).or_insert(pos);
                            *entry = (*entry).min(pos);
                            vreg_uses.entry(vreg).or_default().push(pos);
                        }
                        OperandKind::Def | OperandKind::EarlyDef => {
                            let entry = vreg_start.entry(vreg).or_insert(pos);
                            *entry = (*entry).min(pos);
                            let entry = vreg_end.entry(vreg).or_insert(pos);
                            *entry = (*entry).max(pos);
                            vreg_defs.entry(vreg).or_default().push(pos);
                        }
                        OperandKind::UseDef => {
                            let entry = vreg_start.entry(vreg).or_insert(pos);
                            *entry = (*entry).min(pos);
                            let entry = vreg_end.entry(vreg).or_insert(pos);
                            *entry = (*entry).max(pos);
                            vreg_uses.entry(vreg).or_default().push(pos);
                            vreg_defs.entry(vreg).or_default().push(pos);
                        }
                    }

                    // Record fixed-register hints.
                    if let OperandConstraint::FixedReg(preg) = op.constraint {
                        vreg_fixed_hint.entry(vreg).or_insert(preg);
                    }
                }
            }
        }
    }

    // Build the interval list.
    let mut intervals: Vec<LiveInterval> = Vec::new();
    let mut vreg_to_interval: HashMap<VReg, usize> = HashMap::new();

    // Collect all vregs that appear.
    let mut all_vregs: BTreeSet<VReg> = BTreeSet::new();
    all_vregs.extend(vreg_start.keys());

    for vreg in all_vregs {
        let start = vreg_start[&vreg];
        let end = vreg_end[&vreg];
        let class = func.vreg_class(vreg);

        let idx = intervals.len();
        vreg_to_interval.insert(vreg, idx);

        intervals.push(LiveInterval {
            vreg,
            class,
            start,
            end,
            use_points: vreg_uses.remove(&vreg).unwrap_or_default(),
            def_points: vreg_defs.remove(&vreg).unwrap_or_default(),
            fixed_hint: vreg_fixed_hint.get(&vreg).copied(),
        });
    }

    // Step 4: Collect clobber points.
    let mut clobber_points: HashMap<u32, Vec<PReg>> = HashMap::new();
    for (bi, &block) in blocks.iter().enumerate() {
        let _ = bi;
        for inst in func.block_insts(block) {
            let clobbers = func.inst_clobbers(inst);
            if !clobbers.is_empty() {
                if let Some(&pos) = inst_position.get(&inst) {
                    clobber_points.insert(pos, clobbers.to_vec());
                }
            }
        }
    }

    LivenessInfo {
        intervals,
        vreg_to_interval,
        live_in,
        live_out,
        inst_order,
        inst_position,
        clobber_points,
    }
}
