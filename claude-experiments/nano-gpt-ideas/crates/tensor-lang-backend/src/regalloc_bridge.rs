//! Bridge between the `regalloc` crate and our MachIR.
//!
//! Implements `regalloc::flat::FlatFunction` for `&[MachInst]` and provides
//! a custom ARM64 target matching our reserved register set.  The entry
//! point is [`regalloc_allocate`], which replaces the old `linear_scan_alloc`.

use std::collections::HashMap;

use regalloc::flat::{FlatFunction, FlatOperand, allocate_flat};
use regalloc::allocator::{MoveOperand, MovePosition, InsertedMove};
use regalloc::types::{
    VReg as RVReg, PReg, RegClass as RRegClass, InstId,
};
use regalloc::aarch64::{AArch64Target, GPR, FP_SIMD};
use regalloc::target::{CallingConvention, RegInfo};

use super::mach_ir::{MachInst, VReg, RegClass, Allocation, defs_uses};

// ---------------------------------------------------------------------------
// FlatFunction implementation for MachInst slices
// ---------------------------------------------------------------------------

/// Wrapper that implements `FlatFunction` for a MachIR instruction stream.
pub(crate) struct MachFlatFunc<'a> {
    insts: &'a [MachInst],
    num_vregs: usize,
    vreg_classes: Vec<RegClass>,
}

impl<'a> MachFlatFunc<'a> {
    pub fn new(insts: &'a [MachInst], num_vregs: u32) -> Self {
        // Collect vreg classes from the instruction stream.
        let mut classes = vec![RegClass::GP; num_vregs as usize];
        for inst in insts {
            let (defs, uses) = defs_uses(inst);
            for vr in defs.iter().chain(uses.iter()) {
                if (vr.id as usize) < classes.len() {
                    classes[vr.id as usize] = vr.class;
                }
            }
        }
        MachFlatFunc {
            insts,
            num_vregs: num_vregs as usize,
            vreg_classes: classes,
        }
    }
}

/// Convert a nano-gpt RegClass to a regalloc RegClass.
fn to_rclass(c: RegClass) -> RRegClass {
    match c {
        RegClass::GP => GPR,
        RegClass::FpScalar | RegClass::Vec128 => FP_SIMD,
    }
}

/// Convert a nano-gpt VReg to a regalloc VReg.
fn to_rvreg(v: VReg) -> RVReg {
    RVReg(v.id)
}

impl<'a> FlatFunction for MachFlatFunc<'a> {
    fn num_vregs(&self) -> usize {
        self.num_vregs
    }

    fn vreg_class(&self, vreg: RVReg) -> RRegClass {
        to_rclass(self.vreg_classes[vreg.0 as usize])
    }

    fn num_insts(&self) -> usize {
        self.insts.len()
    }

    fn inst_operands(&self, index: usize) -> Vec<FlatOperand> {
        let inst = &self.insts[index];
        let (defs, uses) = defs_uses(inst);

        // Detect tied operands: a vreg that appears in both defs and uses.
        let mut tied: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for d in &defs {
            for u in &uses {
                if d.id == u.id {
                    tied.insert(d.id);
                }
            }
        }

        // CallFpUnary: lower_one already handles moving arg to S0 and
        // result from S0. We just describe the operands normally — no
        // fixed-reg constraints needed. This avoids the problem where
        // two vregs both constrained to S0 create conflicts at instructions
        // that use both.
        if let MachInst::CallFpUnary { func_ptr, arg, result } = inst {
            return vec![
                FlatOperand::Def(to_rvreg(*result)),
                FlatOperand::Use(to_rvreg(*arg)),
                FlatOperand::Use(to_rvreg(*func_ptr)),
            ];
        }

        let mut ops = Vec::new();

        // Emit defs first.
        for d in &defs {
            if tied.contains(&d.id) {
                ops.push(FlatOperand::UseDef(to_rvreg(*d)));
            } else {
                ops.push(FlatOperand::Def(to_rvreg(*d)));
            }
        }

        // Emit uses (skip those already emitted as UseDef).
        for u in &uses {
            if !tied.contains(&u.id) {
                ops.push(FlatOperand::Use(to_rvreg(*u)));
            }
        }

        ops
    }

    fn inst_label(&self, index: usize) -> Option<usize> {
        if let MachInst::Label { label } = &self.insts[index] {
            Some(*label)
        } else {
            None
        }
    }

    fn inst_branch_targets(&self, index: usize) -> Option<Vec<usize>> {
        match &self.insts[index] {
            MachInst::B { label } => Some(vec![*label]),
            MachInst::BCond { label, .. } => Some(vec![*label]),
            _ => None,
        }
    }

    fn is_conditional_branch(&self, index: usize) -> bool {
        matches!(&self.insts[index], MachInst::BCond { .. })
    }

    fn is_return(&self, _index: usize) -> bool {
        // MachIR doesn't have an explicit return instruction;
        // the stream just ends.
        false
    }

    fn is_call(&self, index: usize) -> bool {
        matches!(&self.insts[index], MachInst::CallFpUnary { .. })
    }

    fn inst_clobbers(&self, index: usize) -> Vec<PReg> {
        if self.is_call(index) {
            // Function calls (CallFpUnary for exp2/log2) destroy all
            // caller-saved registers. The allocator must spill any live
            // values in these registers across the call.
            let mut clobbers = Vec::with_capacity(39);
            clobbers.extend_from_slice(&GP_CALLER_SAVED);
            clobbers.extend_from_slice(&FP_CALLER_SAVED);
            clobbers
        } else {
            Vec::new()
        }
    }
}

// ---------------------------------------------------------------------------
// Custom ARM64 target matching nano-gpt's register pools
// ---------------------------------------------------------------------------

/// nano-gpt reserves: X8 (scratch for SP-large offsets), X16-X18 (platform),
/// X19 (memory base ptr), X20 (heap ptr), X29 (FP), X30 (LR), SP.
///
/// GP_POOL: callee-saved X21-X28 first, then caller-saved X0-X7, X9-X15.
/// FP_VEC_POOL: callee-saved V8-V15 first, then caller-saved V0-V7, V16-V31.
///
/// We create a custom target that only exposes these registers as allocatable,
/// with callee-saved registers ordered first (matching nano-gpt's preference).
pub(crate) struct NanoGptTarget;

/// Singleton instance.
pub(crate) static NANO_GPT_TARGET: NanoGptTarget = NanoGptTarget;

/// GP registers available, callee-saved first (matches GP_POOL ordering).
static NANO_GP_REGS: [PReg; 23] = [
    // Callee-saved: X21-X28
    PReg(21), PReg(22), PReg(23), PReg(24), PReg(25), PReg(26), PReg(27), PReg(28),
    // Caller-saved: X0-X7, X9-X15
    PReg(0), PReg(1), PReg(2), PReg(3), PReg(4), PReg(5), PReg(6), PReg(7),
    PReg(9), PReg(10), PReg(11), PReg(12), PReg(13), PReg(14), PReg(15),
];

/// FP/Vec registers available, callee-saved first (matches FP_VEC_POOL ordering).
static NANO_FP_REGS: [PReg; 32] = [
    // Callee-saved: V8-V15
    PReg(40), PReg(41), PReg(42), PReg(43), PReg(44), PReg(45), PReg(46), PReg(47),
    // Caller-saved: V0-V7, V16-V31
    PReg(32), PReg(33), PReg(34), PReg(35), PReg(36), PReg(37), PReg(38), PReg(39),
    PReg(48), PReg(49), PReg(50), PReg(51), PReg(52), PReg(53), PReg(54), PReg(55),
    PReg(56), PReg(57), PReg(58), PReg(59), PReg(60), PReg(61), PReg(62), PReg(63),
];

static REG_CLASSES: [RRegClass; 2] = [GPR, FP_SIMD];

static GP_CALLEE_SAVED: [PReg; 8] = [
    PReg(21), PReg(22), PReg(23), PReg(24), PReg(25), PReg(26), PReg(27), PReg(28),
];

static FP_CALLEE_SAVED: [PReg; 8] = [
    PReg(40), PReg(41), PReg(42), PReg(43), PReg(44), PReg(45), PReg(46), PReg(47),
];

static GP_CALLER_SAVED: [PReg; 15] = [
    PReg(0), PReg(1), PReg(2), PReg(3), PReg(4), PReg(5), PReg(6), PReg(7),
    PReg(9), PReg(10), PReg(11), PReg(12), PReg(13), PReg(14), PReg(15),
];

static FP_CALLER_SAVED: [PReg; 24] = [
    PReg(32), PReg(33), PReg(34), PReg(35), PReg(36), PReg(37), PReg(38), PReg(39),
    PReg(48), PReg(49), PReg(50), PReg(51), PReg(52), PReg(53), PReg(54), PReg(55),
    PReg(56), PReg(57), PReg(58), PReg(59), PReg(60), PReg(61), PReg(62), PReg(63),
];

impl RegInfo for NanoGptTarget {
    type RegIter<'a> = std::iter::Copied<std::slice::Iter<'a, PReg>>;

    fn reg_classes(&self) -> &[RRegClass] {
        &REG_CLASSES
    }

    /// Returns ONLY the allocatable registers (in preference order).
    fn class_regs(&self, class: RRegClass) -> Self::RegIter<'_> {
        match class.0 {
            0 => NANO_GP_REGS.iter().copied(),
            1 => NANO_FP_REGS.iter().copied(),
            _ => [].iter().copied(),
        }
    }

    fn class_size(&self, class: RRegClass) -> usize {
        match class.0 {
            0 => NANO_GP_REGS.len(),
            1 => NANO_FP_REGS.len(),
            _ => 0,
        }
    }

    fn reg_class_of(&self, reg: PReg) -> RRegClass {
        if reg.0 < 32 { GPR } else { FP_SIMD }
    }

    fn reg_name(&self, reg: PReg) -> &str {
        // Delegate to the standard target for names.
        static STD: AArch64Target = AArch64Target {
            use_frame_pointer: true,
            reserve_x18: true,
        };
        STD.reg_name(reg)
    }

    fn class_name(&self, class: RRegClass) -> &str {
        match class.0 {
            0 => "GP",
            1 => "FP/Vec",
            _ => "unknown",
        }
    }

    fn spill_size(&self, class: RRegClass) -> u32 {
        match class.0 {
            0 => 8,   // 64-bit GP
            1 => 16,  // 128-bit Vec (conservative)
            _ => 8,
        }
    }

    fn spill_align(&self, class: RRegClass) -> u32 {
        match class.0 {
            0 => 8,
            1 => 16,
            _ => 8,
        }
    }
}

impl CallingConvention for NanoGptTarget {
    fn callee_saved(&self) -> &[PReg] {
        // Both GP and FP callee-saved.
        static ALL_CALLEE: [PReg; 16] = {
            let mut arr = [PReg(0); 16];
            let mut i = 0;
            while i < 8 {
                arr[i] = GP_CALLEE_SAVED[i];
                i += 1;
            }
            let mut j = 0;
            while j < 8 {
                arr[8 + j] = FP_CALLEE_SAVED[j];
                j += 1;
            }
            arr
        };
        &ALL_CALLEE
    }

    fn caller_saved(&self) -> &[PReg] {
        // Combine GP and FP caller-saved.
        static ALL_CALLER: [PReg; 39] = {
            let mut arr = [PReg(0); 39];
            let mut i = 0;
            while i < 15 {
                arr[i] = GP_CALLER_SAVED[i];
                i += 1;
            }
            let mut j = 0;
            while j < 24 {
                arr[15 + j] = FP_CALLER_SAVED[j];
                j += 1;
            }
            arr
        };
        &ALL_CALLER
    }

    fn arg_regs(&self, _class: RRegClass) -> &[PReg] {
        &[] // nano-gpt doesn't use standard arg passing
    }

    fn ret_regs(&self, _class: RRegClass) -> &[PReg] {
        &[]
    }

    fn reserved_regs(&self) -> &[PReg] {
        // Nothing extra to reserve — class_regs already returns only
        // the allocatable subset.
        &[]
    }
}

// ---------------------------------------------------------------------------
// Conversion: regalloc Allocation → nano-gpt Allocation
// ---------------------------------------------------------------------------

/// Convert a PReg to nano-gpt's u8 physical register number.
/// GPR: PReg(n) → n.  FP: PReg(32+n) → n.
fn preg_to_hw(preg: PReg) -> u8 {
    if preg.0 < 32 {
        preg.0 as u8
    } else {
        (preg.0 - 32) as u8
    }
}

/// Run the regalloc crate's allocator on a MachIR instruction stream,
/// returning a nano-gpt `Allocation` plus any spill-rewritten instructions.
///
/// This replaces the old `linear_scan_alloc` + `rewrite_spills` loop.
pub(crate) fn regalloc_allocate(
    insts: &[MachInst],
    num_vregs: u32,
    alloc_slot: &mut dyn FnMut() -> u32,
) -> Result<(Vec<MachInst>, Allocation), String> {
    let flat = MachFlatFunc::new(insts, num_vregs);
    let ralloc = allocate_flat(&flat, &NANO_GPT_TARGET)
        .map_err(|e| format!("register allocation failed: {}", e))?;

    // Debug: print allocation for diagnostics.
    if std::env::var("REGALLOC_DEBUG").is_ok() {
        eprintln!("=== regalloc ({} insts, {} vregs, {} spills, {} moves) ===",
            insts.len(), num_vregs, ralloc.spill_slots.len(), ralloc.moves.len());
        for i in 0..insts.len() {
            let flat_ops = flat.inst_operands(i);
            let ops_str: Vec<String> = flat_ops.iter().enumerate().map(|(op_idx, fop)| {
                let preg = ralloc.get(InstId(i as u32), op_idx);
                format!("{:?}->{:?}", fop, preg)
            }).collect();
            let clob = flat.inst_clobbers(i);
            let clob_str = if clob.is_empty() { String::new() } else { format!(" CLOB={}", clob.len()) };
            eprintln!("  {:3}: {:?}  [{}]{}", i, &insts[i], ops_str.join(", "), clob_str);
        }
        eprintln!("  moves: {:?}", ralloc.moves);
        eprintln!("===");
    }

    // Build vreg → physical register map from the per-operand allocation.
    // For non-spilled vregs, every operand gets the same PReg.
    // For spilled vregs, we need to rewrite the instruction stream.
    let mut vreg_map: HashMap<u32, u8> = HashMap::new();
    let mut class_map: HashMap<u32, RegClass> = HashMap::new();

    // Collect all vreg classes.
    for inst in insts {
        let (defs, uses) = defs_uses(inst);
        for vr in defs.iter().chain(uses.iter()) {
            class_map.entry(vr.id).or_insert(vr.class);
        }
    }

    // Build per-instruction vreg→preg assignments from the allocator output.
    // For each instruction, the allocator tells us exactly which physical
    // register each operand should use (even for spilled vregs that get
    // temporary registers at each use/def point).
    let mut inst_vreg_map: Vec<HashMap<u32, u8>> = Vec::with_capacity(insts.len());
    for i in 0..insts.len() {
        let flat_ops = flat.inst_operands(i);
        let mut imap = HashMap::new();
        for (op_idx, fop) in flat_ops.iter().enumerate() {
            let vreg = fop.vreg();
            if let Some(preg) = ralloc.get(InstId(i as u32), op_idx) {
                let hw = preg_to_hw(preg);
                imap.insert(vreg.0, hw);
                // Also populate global map (first assignment wins for non-spilled).
                vreg_map.entry(vreg.0).or_insert(hw);
            }
        }
        inst_vreg_map.push(imap);
    }

    // Now lower_to_emitter uses get_at(inst_idx, vreg) which checks
    // inst_maps first. For spills/moves, we insert move instructions
    // and give them their own per-instruction maps.
    let has_moves = !ralloc.moves.is_empty();

    if !has_moves {
        // No moves needed: return original instructions with per-inst maps.
        let alloc = Allocation {
            map: vreg_map,
            classes: class_map,
            inst_maps: inst_vreg_map,
        };
        return Ok((insts.to_vec(), alloc));
    }

    // With moves: build rewritten instruction stream with move instructions
    // interleaved, and a matching inst_maps for the expanded stream.
    let mut out_insts: Vec<MachInst> = Vec::with_capacity(insts.len() * 2);
    let mut out_inst_maps: Vec<HashMap<u32, u8>> = Vec::with_capacity(insts.len() * 2);
    let mut spill_slot_to_mach_slot: HashMap<u32, u32> = HashMap::new();

    // Allocate real stack frame slots for all referenced spill slots.
    // Vec128 spills need 16 bytes (2 × 8-byte slots), so always allocate
    // 16 bytes to be safe for any register class.
    let alloc_spill_slot = |alloc_slot: &mut dyn FnMut() -> u32| -> u32 {
        let s = alloc_slot();
        alloc_slot(); // allocate a second 8-byte slot for 16-byte alignment
        s
    };
    for (_, &slot) in &ralloc.spill_slots {
        spill_slot_to_mach_slot.entry(slot.0).or_insert_with(|| alloc_spill_slot(alloc_slot));
    }
    for m in &ralloc.moves {
        if let MoveOperand::SpillSlot(s) = &m.from {
            spill_slot_to_mach_slot.entry(s.0).or_insert_with(|| alloc_spill_slot(alloc_slot));
        }
        if let MoveOperand::SpillSlot(s) = &m.to {
            spill_slot_to_mach_slot.entry(s.0).or_insert_with(|| alloc_spill_slot(alloc_slot));
        }
    }

    // Group moves by instruction position.
    let mut before_moves: HashMap<u32, Vec<&InsertedMove>> = HashMap::new();
    let mut after_moves: HashMap<u32, Vec<&InsertedMove>> = HashMap::new();
    for m in &ralloc.moves {
        match m.at {
            MovePosition::Before(inst) => before_moves.entry(inst.0).or_default().push(m),
            MovePosition::After(inst) => after_moves.entry(inst.0).or_default().push(m),
            MovePosition::BlockEdge { .. } => {}
        }
    }

    let mut next_vreg = num_vregs;

    /// Emit a move instruction, creating fresh vregs mapped to physical regs.
    fn emit_move(
        out: &mut Vec<MachInst>,
        out_maps: &mut Vec<HashMap<u32, u8>>,
        m: &InsertedMove,
        vreg_map: &mut HashMap<u32, u8>,
        class_map: &mut HashMap<u32, RegClass>,
        next_vreg: &mut u32,
        slots: &HashMap<u32, u32>,
    ) {
        match (&m.from, &m.to) {
            (MoveOperand::SpillSlot(slot), MoveOperand::Reg(preg)) => {
                let hw = preg_to_hw(*preg);
                let mach_slot = slots[&slot.0];
                let cls = if preg.0 < 32 { RegClass::GP } else { RegClass::Vec128 };
                let v = VReg { id: *next_vreg, class: cls };
                *next_vreg += 1;
                vreg_map.insert(v.id, hw);
                class_map.insert(v.id, cls);
                let mut imap = HashMap::new();
                imap.insert(v.id, hw);
                out_maps.push(imap);
                match cls {
                    RegClass::GP => out.push(MachInst::SpLoad { dst: v, slot: mach_slot }),
                    // Always use Vec128 spill/restore for FP regs — this is safe
                    // for both FpScalar and Vec128 (saves all 128 bits).
                    _ => out.push(MachInst::SpLoadVec128 { dst: v, slot: mach_slot }),
                }
            }
            (MoveOperand::Reg(preg), MoveOperand::SpillSlot(slot)) => {
                let hw = preg_to_hw(*preg);
                let mach_slot = slots[&slot.0];
                let cls = if preg.0 < 32 { RegClass::GP } else { RegClass::Vec128 };
                let v = VReg { id: *next_vreg, class: cls };
                *next_vreg += 1;
                vreg_map.insert(v.id, hw);
                class_map.insert(v.id, cls);
                let mut imap = HashMap::new();
                imap.insert(v.id, hw);
                out_maps.push(imap);
                match cls {
                    RegClass::GP => out.push(MachInst::SpStore { src: v, slot: mach_slot }),
                    _ => out.push(MachInst::SpStoreVec128 { src: v, slot: mach_slot }),
                }
            }
            (MoveOperand::Reg(from), MoveOperand::Reg(to)) => {
                let cls = if from.0 < 32 { RegClass::GP } else { RegClass::Vec128 };
                let src = VReg { id: *next_vreg, class: cls };
                *next_vreg += 1;
                let dst = VReg { id: *next_vreg, class: cls };
                *next_vreg += 1;
                let from_hw = preg_to_hw(*from);
                let to_hw = preg_to_hw(*to);
                vreg_map.insert(src.id, from_hw);
                vreg_map.insert(dst.id, to_hw);
                class_map.insert(src.id, cls);
                class_map.insert(dst.id, cls);
                let mut imap = HashMap::new();
                imap.insert(src.id, from_hw);
                imap.insert(dst.id, to_hw);
                out_maps.push(imap);
                match cls {
                    RegClass::GP => out.push(MachInst::MovReg { dst, src }),
                    // Use MovVec for FP reg-to-reg moves to preserve all 128 bits.
                    _ => out.push(MachInst::MovVec { dst, src }),
                }
            }
            _ => {}
        }
    }

    for i in 0..insts.len() {
        // Emit "before" moves.
        if let Some(moves) = before_moves.get(&(i as u32)) {
            for m in moves {
                emit_move(&mut out_insts, &mut out_inst_maps, m,
                          &mut vreg_map, &mut class_map,
                          &mut next_vreg, &spill_slot_to_mach_slot);
            }
        }

        // Emit original instruction with its per-instruction map.
        out_insts.push(insts[i].clone());
        out_inst_maps.push(inst_vreg_map[i].clone());

        // Emit "after" moves.
        if let Some(moves) = after_moves.get(&(i as u32)) {
            for m in moves {
                emit_move(&mut out_insts, &mut out_inst_maps, m,
                          &mut vreg_map, &mut class_map,
                          &mut next_vreg, &spill_slot_to_mach_slot);
            }
        }
    }

    let alloc = Allocation {
        map: vreg_map,
        classes: class_map,
        inst_maps: out_inst_maps,
    };
    Ok((out_insts, alloc))
}
