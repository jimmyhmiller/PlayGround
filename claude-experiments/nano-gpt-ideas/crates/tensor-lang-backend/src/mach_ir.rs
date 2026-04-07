//! Machine IR with virtual registers and linear-scan register allocation
//! for the ARM64 backend.
//!
//! Virtual registers are tagged by class (GP, FpScalar, Vec128) and get
//! mapped to physical ARM64 registers by `linear_scan_alloc`.  The
//! `lower_to_emitter` pass then emits real ArmEmitter calls.

use std::collections::HashMap;
use super::arm::ArmEmitter;

// ---------------------------------------------------------------------------
// Virtual register
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegClass {
    GP,
    /// Scalar f32 — shares the physical S0..S31 file with Vec128 (ARM64
    /// S-regs alias the low 32 bits of V-regs).
    FpScalar,
    /// 128-bit NEON .4S — shares physical V0..V31 file with FpScalar.
    Vec128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VReg {
    pub id: u32,
    pub class: RegClass,
}

// ---------------------------------------------------------------------------
// Machine instructions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum MachInst {
    // --- GP arithmetic ---
    /// Xd = Xn + Xm
    AddReg { dst: VReg, lhs: VReg, rhs: VReg },
    /// Xd = Xn + #imm12
    AddImm { dst: VReg, src: VReg, imm: u32 },
    /// Xd = Xn - Xm
    SubReg { dst: VReg, lhs: VReg, rhs: VReg },
    /// Xd = Xn - #imm12
    SubImm { dst: VReg, src: VReg, imm: u32 },
    /// Xd = Xn * Xm
    MulReg { dst: VReg, lhs: VReg, rhs: VReg },
    /// Xd = Xa + Xn * Xm
    Madd { dst: VReg, mul_lhs: VReg, mul_rhs: VReg, add: VReg },
    /// Xd = Xn / Xm (signed)
    Sdiv { dst: VReg, lhs: VReg, rhs: VReg },
    /// Xd = #imm64
    MovImm64 { dst: VReg, val: u64 },
    /// Xd = Xm  (GP move)
    MovReg { dst: VReg, src: VReg },
    /// Xd = Xn << #shift
    LslImm { dst: VReg, src: VReg, shift: u32 },
    /// CMP Xn, Xm  (sets flags, no destination)
    CmpReg { lhs: VReg, rhs: VReg },
    /// CMP Xn, #imm12
    CmpImm { lhs: VReg, imm: u32 },
    /// CSEL Xd, Xn, Xm, cond
    Csel { dst: VReg, t_val: VReg, f_val: VReg, cond: u8 },

    // --- SP load/store (stack frame access) ---
    /// Xd = [SP + #slot]  (64-bit load from stack)
    SpLoad { dst: VReg, slot: u32 },
    /// [SP + #slot] = Xn  (64-bit store to stack)
    SpStore { src: VReg, slot: u32 },

    // --- Scalar FP ---
    /// Sd = [Xbase + Xoff, LSL #2]  (f32 register-offset load)
    LdrSRegScaled { dst: VReg, base: VReg, offset: VReg },
    /// [Xbase + Xoff, LSL #2] = Ss
    StrSRegScaled { src: VReg, base: VReg, offset: VReg },
    /// Sd = [Xn + #imm]  (imm scaled by 4)
    LdrSImm { dst: VReg, base: VReg, imm: u32 },
    /// [Xn + #imm] = Ss
    StrSImm { src: VReg, base: VReg, imm: u32 },
    /// FMOV Wd, Sn  (FP -> GP)
    FmovWFromS { dst_gp: VReg, src_fp: VReg },
    /// FMOV Sd, Wn  (GP -> FP)
    FmovSFromW { dst_fp: VReg, src_gp: VReg },
    /// FMADD Sd, Sn, Sm, Sa = Sa + Sn*Sm
    Fmadd { dst: VReg, mul_lhs: VReg, mul_rhs: VReg, add: VReg },
    /// FADD Sd, Sn, Sm
    FaddS { dst: VReg, lhs: VReg, rhs: VReg },
    /// FSUB Sd, Sn, Sm
    FsubS { dst: VReg, lhs: VReg, rhs: VReg },
    /// FMUL Sd, Sn, Sm
    FmulS { dst: VReg, lhs: VReg, rhs: VReg },
    /// FDIV Sd, Sn, Sm
    FdivS { dst: VReg, lhs: VReg, rhs: VReg },
    /// FNEG Sd, Sn
    FnegS { dst: VReg, src: VReg },
    /// FSQRT Sd, Sn
    FsqrtS { dst: VReg, src: VReg },
    /// SCVTF Sd, Xn  (signed int64 -> f32)
    ScvtfSX { dst: VReg, src: VReg },
    /// FCMP Sn, Sm  (sets flags)
    FcmpS { lhs: VReg, rhs: VReg },
    /// FCSEL Sd, Sn, Sm, cond
    FcselS { dst: VReg, t_val: VReg, f_val: VReg, cond: u8 },
    /// FMAX Sd, Sn, Sm
    FmaxS { dst: VReg, lhs: VReg, rhs: VReg },
    /// FMOV Sd, Sn
    FmovS { dst: VReg, src: VReg },
    /// LDR Sd, [Xbase + Xoff] (unscaled byte offset in GP reg)
    LdrSReg { dst: VReg, base: VReg, offset: VReg },
    /// Xd = Xa - Xn * Xm  (MSUB)
    Msub { dst: VReg, mul_lhs: VReg, mul_rhs: VReg, sub_from: VReg },

    /// Call a function pointer. arg in S0, result in S0.
    /// The allocator treats this as clobbering all caller-saved regs.
    CallFpUnary { func_ptr: VReg, arg: VReg, result: VReg },
    /// STR Ss, [SP, #slot]  (store f32 to stack)
    SpStoreF32 { src: VReg, slot: u32 },
    /// LDR Ss, [SP, #slot]  (load f32 from stack)
    SpLoadF32 { dst: VReg, slot: u32 },

    // --- NEON .4S ---
    /// Vd.4S = 0
    Movi4sZero { dst: VReg },
    /// DUP Vd.4S, Wn  (broadcast GP -> all 4 lanes)
    Dup4sGp { dst: VReg, src_gp: VReg },
    /// LDR Qt, [Xn, #imm]  (128-bit, imm scaled by 16)
    LdrQImm { dst: VReg, base: VReg, imm: u32 },
    /// STR Qt, [Xn, #imm]
    StrQImm { src: VReg, base: VReg, imm: u32 },
    /// FMLA Vd.4S, Vn.4S, Vm.4S  —  Vd += Vn * Vm
    /// `acc` is both read AND written (tied operand).
    Fmla4s { acc: VReg, lhs: VReg, rhs: VReg },
    /// MOV Vd.16B, Vn.16B  (128-bit register move)
    MovVec { dst: VReg, src: VReg },

    // --- NEON .4S elementwise ---
    /// FADD Vd.4S, Vn.4S, Vm.4S
    Fadd4s { dst: VReg, lhs: VReg, rhs: VReg },
    /// FMUL Vd.4S, Vn.4S, Vm.4S
    Fmul4s { dst: VReg, lhs: VReg, rhs: VReg },
    /// FNEG Vd.4S, Vn.4S
    Fneg4s { dst: VReg, src: VReg },
    /// FMAX Vd.4S, Vn.4S, Vm.4S
    Fmax4s { dst: VReg, lhs: VReg, rhs: VReg },
    /// FSQRT Vd.4S, Vn.4S
    Fsqrt4s { dst: VReg, src: VReg },
    /// FDIV Vd.4S, Vn.4S, Vm.4S (for Recip: div 1.0 by src)
    Fdiv4s { dst: VReg, lhs: VReg, rhs: VReg },
    /// FCMGT Vd.4S, Vm.4S, Vn.4S  (true if Vm > Vn, per lane)
    Fcmgt4s { dst: VReg, lhs: VReg, rhs: VReg },
    /// AND Vd.16B, Vn.16B, Vm.16B  (bitwise AND, for masking with compare results)
    And16b { dst: VReg, lhs: VReg, rhs: VReg },
    /// DUP Vd.4S, Sn  (broadcast scalar S-reg to all 4 lanes)
    Dup4sScalar { dst: VReg, src: VReg },
    /// LDR Qt, [Xn, Xm]  (128-bit register-offset load)
    LdrQReg { dst: VReg, base: VReg, offset: VReg },
    /// STR Qt, [Xn, Xm]  (128-bit register-offset store)
    StrQReg { src: VReg, base: VReg, offset: VReg },
    /// FMOV Vd.4S, #1.0  (broadcast immediate 1.0 to all lanes)
    Fmov4sOne { dst: VReg },

    // --- Control flow ---
    /// Unconditional branch to label
    B { label: usize },
    /// Conditional branch
    BCond { cond: u8, label: usize },
    /// Bind a label at the current position
    Label { label: usize },
}

// ---------------------------------------------------------------------------
// MachBuilder — construct instruction sequences with virtual registers
// ---------------------------------------------------------------------------

pub struct MachBuilder {
    insts: Vec<MachInst>,
    next_vreg: u32,
}

impl MachBuilder {
    pub fn new() -> Self {
        MachBuilder {
            insts: Vec::new(),
            next_vreg: 0,
        }
    }

    fn alloc(&mut self, class: RegClass) -> VReg {
        let id = self.next_vreg;
        self.next_vreg += 1;
        VReg { id, class }
    }

    pub fn new_gp(&mut self) -> VReg {
        self.alloc(RegClass::GP)
    }

    pub fn new_fp(&mut self) -> VReg {
        self.alloc(RegClass::FpScalar)
    }

    pub fn new_vec(&mut self) -> VReg {
        self.alloc(RegClass::Vec128)
    }

    pub fn push(&mut self, inst: MachInst) {
        self.insts.push(inst);
    }

    pub fn finish(&self) -> Vec<MachInst> {
        self.insts.clone()
    }

    pub fn next_vreg_id(&self) -> u32 {
        self.next_vreg
    }
}

// ---------------------------------------------------------------------------
// Physical register pools
// ---------------------------------------------------------------------------

/// GP registers available for allocation.
/// Avoids: X8 (scratch for SP-large offset helpers),
/// X16-X18 (platform reserved), X19 (memory base ptr), X20 (heap ptr),
/// X29 (FP), X30 (LR), XZR/SP.
/// X0-X2 are free after prologue (ABI args already saved to callee-saved regs).
/// Callee-saved regs (X21-X28) are listed FIRST so the linear-scan allocator
/// assigns them to long-lived values (loop counters, pointers), reducing the
/// number of caller-saved registers that need saving around BLR calls.
const GP_POOL: &[u8] = &[
    21, 22, 23, 24, 25, 26, 27, 28,
    0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15,
];

/// FP/Vec registers available for allocation.
/// Callee-saved V8-V15 listed FIRST so the allocator assigns them to
/// long-lived values (values that must survive function calls).
/// Then caller-saved V0-V7 and V16-V31.
const FP_VEC_POOL: &[u8] = &[
    8, 9, 10, 11, 12, 13, 14, 15,
    0, 1, 2, 3, 4, 5, 6, 7,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
];

// ---------------------------------------------------------------------------
// Live-range analysis & linear-scan allocator
// ---------------------------------------------------------------------------

/// Allocation result: maps VReg id -> physical register number.
/// Supports both global (one register per vreg) and per-instruction
/// (different register at different instructions, for spilled vregs).
pub struct Allocation {
    /// Global vreg→preg map (used when no per-instruction overrides exist).
    pub map: HashMap<u32, u8>,
    pub classes: HashMap<u32, RegClass>,
    /// Per-instruction overrides. If inst_maps[i] has an entry for a vreg,
    /// it takes precedence over the global map.
    pub inst_maps: Vec<HashMap<u32, u8>>,
}

impl Allocation {
    /// Look up the register for a vreg (global map only).
    pub fn get(&self, vr: VReg) -> u8 {
        self.map[&vr.id]
    }

    /// Look up the register for a vreg at a specific instruction index.
    /// Falls back to the global map if no per-instruction override exists.
    pub fn get_at(&self, inst_idx: usize, vr: VReg) -> u8 {
        if inst_idx < self.inst_maps.len() {
            if let Some(&hw) = self.inst_maps[inst_idx].get(&vr.id) {
                return hw;
            }
        }
        self.map[&vr.id]
    }
}

struct LiveRange {
    start: usize,
    end: usize,
    vreg: VReg,
}

/// Compute live ranges and allocate physical registers via linear scan.
pub fn linear_scan_alloc(insts: &[MachInst]) -> Result<Allocation, SpillRequest> {
    // 1. Compute def and last-use for every VReg.
    let mut first_def: HashMap<u32, usize> = HashMap::new();
    let mut last_use: HashMap<u32, usize> = HashMap::new();
    let mut classes: HashMap<u32, RegClass> = HashMap::new();

    for (i, inst) in insts.iter().enumerate() {
        let (defs, uses) = defs_uses(inst);
        for vr in &defs {
            classes.entry(vr.id).or_insert(vr.class);
            first_def.entry(vr.id).or_insert(i);
            last_use.entry(vr.id).or_insert(i);
        }
        for vr in &uses {
            classes.entry(vr.id).or_insert(vr.class);
            first_def.entry(vr.id).or_insert(i);
            last_use.insert(vr.id, i);
        }
    }

    // 2. Handle backward branches (loops).
    // Find label positions, then for each backward branch (B/BCond to an
    // earlier label), extend the live range of any vreg that is alive at the
    // label target to at least the branch source position.
    let mut label_pos: HashMap<usize, usize> = HashMap::new();
    for (i, inst) in insts.iter().enumerate() {
        if let MachInst::Label { label } = inst {
            label_pos.insert(*label, i);
        }
    }

    // Collect backward edges: (branch_pos, target_pos) where target < branch
    let mut back_edges: Vec<(usize, usize)> = Vec::new();
    for (i, inst) in insts.iter().enumerate() {
        let target_label = match inst {
            MachInst::B { label } => Some(*label),
            MachInst::BCond { label, .. } => Some(*label),
            _ => None,
        };
        if let Some(lbl) = target_label {
            if let Some(&target_pos) = label_pos.get(&lbl) {
                if target_pos <= i {
                    back_edges.push((i, target_pos));
                }
            }
        }
    }

    // For each back edge, extend vregs that are truly loop-carried:
    // A vreg is loop-carried if it's defined at or before the loop header
    // and used within the loop. Vregs defined AFTER the loop header are
    // iteration-local temporaries and should NOT be extended.
    for &(branch_pos, target_pos) in &back_edges {
        for (&id, start) in &first_def {
            let end = last_use.get_mut(&id).unwrap();
            // Only extend if defined at or before the loop target (loop-carried)
            if *start <= target_pos && *end >= target_pos && *end < branch_pos {
                *end = branch_pos;
            }
        }
    }

    // 3. Build sorted live ranges.
    let mut gp_ranges: Vec<LiveRange> = Vec::new();
    let mut fp_ranges: Vec<LiveRange> = Vec::new();

    for (&id, &cls) in &classes {
        let start = first_def[&id];
        let end = last_use[&id];
        let lr = LiveRange {
            start,
            end,
            vreg: VReg { id, class: cls },
        };
        match cls {
            RegClass::GP => gp_ranges.push(lr),
            RegClass::FpScalar | RegClass::Vec128 => fp_ranges.push(lr),
        }
    }

    gp_ranges.sort_by_key(|r| r.start);
    fp_ranges.sort_by_key(|r| r.start);

    let mut map = HashMap::new();

    if let Err(spills) = try_allocate_pool(&gp_ranges, GP_POOL, &mut map) {
        return Err(SpillRequest { vreg_ids: spills, classes: classes.clone() });
    }
    if let Err(spills) = try_allocate_pool(&fp_ranges, FP_VEC_POOL, &mut map) {
        return Err(SpillRequest { vreg_ids: spills, classes: classes.clone() });
    }

    Ok(Allocation { map, classes, inst_maps: Vec::new() })
}

/// Information about which vregs need to be spilled.
pub struct SpillRequest {
    pub vreg_ids: Vec<u32>,
    pub classes: HashMap<u32, RegClass>,
}

/// Try to allocate physical registers for the given live ranges.
/// Returns Ok(()) on success, Err(spill_set) with vreg IDs to spill on failure.
fn try_allocate_pool(ranges: &[LiveRange], pool: &[u8], map: &mut HashMap<u32, u8>) -> Result<(), Vec<u32>> {
    let mut active: Vec<(usize, u8, u32)> = Vec::new(); // (end, phys_reg, vreg_id)
    let mut free: Vec<u8> = pool.iter().copied().rev().collect();

    for lr in ranges {
        active.retain(|&(end, preg, _)| {
            if end < lr.start {
                free.push(preg);
                false
            } else {
                true
            }
        });

        if let Some(preg) = free.pop() {
            map.insert(lr.vreg.id, preg);
            active.push((lr.end, preg, lr.vreg.id));
            active.sort_by_key(|&(end, _, _)| end);
        } else {
            // Out of registers. Pick the vreg with the longest live range
            // among all currently active + the current one as the spill victim.
            let current_len = lr.end - lr.start;
            let (worst_idx, worst_len) = active.iter().enumerate()
                .map(|(i, (end, _, _))| (i, *end - lr.start))
                .max_by_key(|(_, len)| *len)
                .unwrap_or((0, 0));

            if worst_len > current_len {
                // Spill the active vreg with the longest remaining range
                let spill_id = active[worst_idx].2;
                return Err(vec![spill_id]);
            } else {
                // Spill the current vreg
                return Err(vec![lr.vreg.id]);
            }
        }
    }
    Ok(())
}

/// Rewrite the instruction stream to spill the given vregs.
/// For each spilled vreg:
///   - After each def: insert SpStore (GP) or SpStoreF32 (FP) to save to a stack slot
///   - Before each use: insert SpLoad/SpLoadF32 with a fresh short-lived vreg,
///     and replace the spilled vreg with the new one in the instruction.
fn rewrite_spills(
    insts: &[MachInst],
    spill_ids: &[u32],
    classes: &HashMap<u32, RegClass>,
    next_vreg_id: &mut u32,
    alloc_slot: &mut dyn FnMut() -> u32,
) -> Vec<MachInst> {
    // Allocate a stack slot for each spilled vreg
    let mut spill_slots: HashMap<u32, u32> = HashMap::new();
    for &id in spill_ids {
        spill_slots.insert(id, alloc_slot());
    }

    let mut out = Vec::with_capacity(insts.len() + spill_ids.len() * 4);

    for inst in insts {
        let (defs, uses) = defs_uses(inst);

        // Check if any USE in this instruction is a spilled vreg.
        // If so, insert a reload before this instruction and rewrite the use.
        let spilled_uses: Vec<(u32, u32)> = uses.iter()
            .filter(|vr| spill_slots.contains_key(&vr.id))
            .map(|vr| (vr.id, spill_slots[&vr.id]))
            .collect();

        // Create reload vregs and insert reload instructions
        let mut remap: HashMap<u32, VReg> = HashMap::new();
        for &(old_id, slot) in &spilled_uses {
            // Skip if we already created a reload for this vreg in this instruction
            if remap.contains_key(&old_id) { continue; }
            let cls = classes[&old_id];
            let new_id = *next_vreg_id;
            *next_vreg_id += 1;
            let new_vreg = VReg { id: new_id, class: cls };
            remap.insert(old_id, new_vreg);

            // Insert reload
            match cls {
                RegClass::GP => out.push(MachInst::SpLoad { dst: new_vreg, slot }),
                RegClass::FpScalar => out.push(MachInst::SpLoadF32 { dst: new_vreg, slot }),
                RegClass::Vec128 => {
                    // No SpLoadVec in MachIR yet — use SpLoad into GP then...
                    // Actually for Vec128 spills we'd need LDR Q from stack.
                    // For now, Vec128 spills are unlikely (we have 24 FP/Vec regs).
                    // Just panic if it happens.
                    panic!("Vec128 spill not implemented");
                }
            }
        }

        // Emit the instruction with spilled uses remapped to reload vregs
        let remapped = if remap.is_empty() {
            inst.clone()
        } else {
            remap_vreg_uses(inst, &remap)
        };
        out.push(remapped);

        // After each def of a spilled vreg, insert a spill store
        for vr in &defs {
            if let Some(&slot) = spill_slots.get(&vr.id) {
                match vr.class {
                    RegClass::GP => out.push(MachInst::SpStore { src: *vr, slot }),
                    RegClass::FpScalar => out.push(MachInst::SpStoreF32 { src: *vr, slot }),
                    RegClass::Vec128 => panic!("Vec128 spill not implemented"),
                }
            }
        }
    }

    out
}

/// Remap ALL vregs (both defs and uses) in an instruction.
pub fn remap_vreg_uses_and_defs(inst: &MachInst, remap: &HashMap<u32, VReg>) -> MachInst {
    let r = |vr: &VReg| -> VReg {
        remap.get(&vr.id).copied().unwrap_or(*vr)
    };
    use MachInst::*;
    match inst {
        AddReg { dst, lhs, rhs } => AddReg { dst: r(dst), lhs: r(lhs), rhs: r(rhs) },
        AddImm { dst, src, imm } => AddImm { dst: r(dst), src: r(src), imm: *imm },
        SubReg { dst, lhs, rhs } => SubReg { dst: r(dst), lhs: r(lhs), rhs: r(rhs) },
        SubImm { dst, src, imm } => SubImm { dst: r(dst), src: r(src), imm: *imm },
        MulReg { dst, lhs, rhs } => MulReg { dst: r(dst), lhs: r(lhs), rhs: r(rhs) },
        Madd { dst, mul_lhs, mul_rhs, add } => Madd { dst: r(dst), mul_lhs: r(mul_lhs), mul_rhs: r(mul_rhs), add: r(add) },
        Sdiv { dst, lhs, rhs } => Sdiv { dst: r(dst), lhs: r(lhs), rhs: r(rhs) },
        MovImm64 { dst, val } => MovImm64 { dst: r(dst), val: *val },
        MovReg { dst, src } => MovReg { dst: r(dst), src: r(src) },
        LslImm { dst, src, shift } => LslImm { dst: r(dst), src: r(src), shift: *shift },
        CmpReg { lhs, rhs } => CmpReg { lhs: r(lhs), rhs: r(rhs) },
        CmpImm { lhs, imm } => CmpImm { lhs: r(lhs), imm: *imm },
        Csel { dst, t_val, f_val, cond } => Csel { dst: r(dst), t_val: r(t_val), f_val: r(f_val), cond: *cond },
        SpLoad { dst, slot } => SpLoad { dst: r(dst), slot: *slot },
        SpStore { src, slot } => SpStore { src: r(src), slot: *slot },
        LdrSRegScaled { dst, base, offset } => LdrSRegScaled { dst: r(dst), base: r(base), offset: r(offset) },
        StrSRegScaled { src, base, offset } => StrSRegScaled { src: r(src), base: r(base), offset: r(offset) },
        LdrSImm { dst, base, imm } => LdrSImm { dst: r(dst), base: r(base), imm: *imm },
        StrSImm { src, base, imm } => StrSImm { src: r(src), base: r(base), imm: *imm },
        FmovWFromS { dst_gp, src_fp } => FmovWFromS { dst_gp: r(dst_gp), src_fp: r(src_fp) },
        FmovSFromW { dst_fp, src_gp } => FmovSFromW { dst_fp: r(dst_fp), src_gp: r(src_gp) },
        Fmadd { dst, mul_lhs, mul_rhs, add } => Fmadd { dst: r(dst), mul_lhs: r(mul_lhs), mul_rhs: r(mul_rhs), add: r(add) },
        FaddS { dst, lhs, rhs } => FaddS { dst: r(dst), lhs: r(lhs), rhs: r(rhs) },
        FsubS { dst, lhs, rhs } => FsubS { dst: r(dst), lhs: r(lhs), rhs: r(rhs) },
        FmulS { dst, lhs, rhs } => FmulS { dst: r(dst), lhs: r(lhs), rhs: r(rhs) },
        FdivS { dst, lhs, rhs } => FdivS { dst: r(dst), lhs: r(lhs), rhs: r(rhs) },
        FnegS { dst, src } => FnegS { dst: r(dst), src: r(src) },
        FsqrtS { dst, src } => FsqrtS { dst: r(dst), src: r(src) },
        ScvtfSX { dst, src } => ScvtfSX { dst: r(dst), src: r(src) },
        FcmpS { lhs, rhs } => FcmpS { lhs: r(lhs), rhs: r(rhs) },
        FcselS { dst, t_val, f_val, cond } => FcselS { dst: r(dst), t_val: r(t_val), f_val: r(f_val), cond: *cond },
        FmaxS { dst, lhs, rhs } => FmaxS { dst: r(dst), lhs: r(lhs), rhs: r(rhs) },
        FmovS { dst, src } => FmovS { dst: r(dst), src: r(src) },
        LdrSReg { dst, base, offset } => LdrSReg { dst: r(dst), base: r(base), offset: r(offset) },
        Msub { dst, mul_lhs, mul_rhs, sub_from } => Msub { dst: r(dst), mul_lhs: r(mul_lhs), mul_rhs: r(mul_rhs), sub_from: r(sub_from) },
        CallFpUnary { func_ptr, arg, result } => CallFpUnary { func_ptr: r(func_ptr), arg: r(arg), result: r(result) },
        SpStoreF32 { src, slot } => SpStoreF32 { src: r(src), slot: *slot },
        SpLoadF32 { dst, slot } => SpLoadF32 { dst: r(dst), slot: *slot },
        Movi4sZero { dst } => Movi4sZero { dst: r(dst) },
        Dup4sGp { dst, src_gp } => Dup4sGp { dst: r(dst), src_gp: r(src_gp) },
        LdrQImm { dst, base, imm } => LdrQImm { dst: r(dst), base: r(base), imm: *imm },
        StrQImm { src, base, imm } => StrQImm { src: r(src), base: r(base), imm: *imm },
        Fmla4s { acc, lhs, rhs } => Fmla4s { acc: r(acc), lhs: r(lhs), rhs: r(rhs) },
        MovVec { dst, src } => MovVec { dst: r(dst), src: r(src) },
        Fadd4s { dst, lhs, rhs } => Fadd4s { dst: r(dst), lhs: r(lhs), rhs: r(rhs) },
        Fmul4s { dst, lhs, rhs } => Fmul4s { dst: r(dst), lhs: r(lhs), rhs: r(rhs) },
        Fneg4s { dst, src } => Fneg4s { dst: r(dst), src: r(src) },
        Fmax4s { dst, lhs, rhs } => Fmax4s { dst: r(dst), lhs: r(lhs), rhs: r(rhs) },
        Fsqrt4s { dst, src } => Fsqrt4s { dst: r(dst), src: r(src) },
        Fdiv4s { dst, lhs, rhs } => Fdiv4s { dst: r(dst), lhs: r(lhs), rhs: r(rhs) },
        Fcmgt4s { dst, lhs, rhs } => Fcmgt4s { dst: r(dst), lhs: r(lhs), rhs: r(rhs) },
        And16b { dst, lhs, rhs } => And16b { dst: r(dst), lhs: r(lhs), rhs: r(rhs) },
        Dup4sScalar { dst, src } => Dup4sScalar { dst: r(dst), src: r(src) },
        LdrQReg { dst, base, offset } => LdrQReg { dst: r(dst), base: r(base), offset: r(offset) },
        StrQReg { src, base, offset } => StrQReg { src: r(src), base: r(base), offset: r(offset) },
        Fmov4sOne { dst } => Fmov4sOne { dst: r(dst) },
        B { .. } | BCond { .. } | Label { .. } => inst.clone(),
    }
}

/// Create a copy of the instruction with certain vregs replaced (both defs and uses).
fn remap_vreg_uses(inst: &MachInst, remap: &HashMap<u32, VReg>) -> MachInst {
    let r = |vr: &VReg| -> VReg {
        remap.get(&vr.id).copied().unwrap_or(*vr)
    };
    use MachInst::*;
    match inst {
        AddReg { dst, lhs, rhs } => AddReg { dst: *dst, lhs: r(lhs), rhs: r(rhs) },
        AddImm { dst, src, imm } => AddImm { dst: *dst, src: r(src), imm: *imm },
        SubReg { dst, lhs, rhs } => SubReg { dst: *dst, lhs: r(lhs), rhs: r(rhs) },
        SubImm { dst, src, imm } => SubImm { dst: *dst, src: r(src), imm: *imm },
        MulReg { dst, lhs, rhs } => MulReg { dst: *dst, lhs: r(lhs), rhs: r(rhs) },
        Madd { dst, mul_lhs, mul_rhs, add } => Madd { dst: *dst, mul_lhs: r(mul_lhs), mul_rhs: r(mul_rhs), add: r(add) },
        Sdiv { dst, lhs, rhs } => Sdiv { dst: *dst, lhs: r(lhs), rhs: r(rhs) },
        Msub { dst, mul_lhs, mul_rhs, sub_from } => Msub { dst: *dst, mul_lhs: r(mul_lhs), mul_rhs: r(mul_rhs), sub_from: r(sub_from) },
        MovReg { dst, src } => MovReg { dst: *dst, src: r(src) },
        LslImm { dst, src, shift } => LslImm { dst: *dst, src: r(src), shift: *shift },
        CmpReg { lhs, rhs } => CmpReg { lhs: r(lhs), rhs: r(rhs) },
        CmpImm { lhs, imm } => CmpImm { lhs: r(lhs), imm: *imm },
        Csel { dst, t_val, f_val, cond } => Csel { dst: *dst, t_val: r(t_val), f_val: r(f_val), cond: *cond },
        SpStore { src, slot } => SpStore { src: r(src), slot: *slot },
        LdrSRegScaled { dst, base, offset } => LdrSRegScaled { dst: *dst, base: r(base), offset: r(offset) },
        StrSRegScaled { src, base, offset } => StrSRegScaled { src: r(src), base: r(base), offset: r(offset) },
        LdrSImm { dst, base, imm } => LdrSImm { dst: *dst, base: r(base), imm: *imm },
        StrSImm { src, base, imm } => StrSImm { src: r(src), base: r(base), imm: *imm },
        FmovWFromS { dst_gp, src_fp } => FmovWFromS { dst_gp: *dst_gp, src_fp: r(src_fp) },
        FmovSFromW { dst_fp, src_gp } => FmovSFromW { dst_fp: *dst_fp, src_gp: r(src_gp) },
        Fmadd { dst, mul_lhs, mul_rhs, add } => Fmadd { dst: *dst, mul_lhs: r(mul_lhs), mul_rhs: r(mul_rhs), add: r(add) },
        FaddS { dst, lhs, rhs } => FaddS { dst: *dst, lhs: r(lhs), rhs: r(rhs) },
        FsubS { dst, lhs, rhs } => FsubS { dst: *dst, lhs: r(lhs), rhs: r(rhs) },
        FmulS { dst, lhs, rhs } => FmulS { dst: *dst, lhs: r(lhs), rhs: r(rhs) },
        FdivS { dst, lhs, rhs } => FdivS { dst: *dst, lhs: r(lhs), rhs: r(rhs) },
        FnegS { dst, src } => FnegS { dst: *dst, src: r(src) },
        FsqrtS { dst, src } => FsqrtS { dst: *dst, src: r(src) },
        ScvtfSX { dst, src } => ScvtfSX { dst: *dst, src: r(src) },
        FcmpS { lhs, rhs } => FcmpS { lhs: r(lhs), rhs: r(rhs) },
        FcselS { dst, t_val, f_val, cond } => FcselS { dst: *dst, t_val: r(t_val), f_val: r(f_val), cond: *cond },
        FmaxS { dst, lhs, rhs } => FmaxS { dst: *dst, lhs: r(lhs), rhs: r(rhs) },
        FmovS { dst, src } => FmovS { dst: *dst, src: r(src) },
        LdrSReg { dst, base, offset } => LdrSReg { dst: *dst, base: r(base), offset: r(offset) },
        CallFpUnary { func_ptr, arg, result } => CallFpUnary { func_ptr: r(func_ptr), arg: r(arg), result: *result },
        SpStoreF32 { src, slot } => SpStoreF32 { src: r(src), slot: *slot },
        Dup4sGp { dst, src_gp } => Dup4sGp { dst: *dst, src_gp: r(src_gp) },
        StrQImm { src, base, imm } => StrQImm { src: r(src), base: r(base), imm: *imm },
        LdrQImm { dst, base, imm } => LdrQImm { dst: *dst, base: r(base), imm: *imm },
        Fmla4s { acc, lhs, rhs } => Fmla4s { acc: r(acc), lhs: r(lhs), rhs: r(rhs) },
        MovVec { dst, src } => MovVec { dst: *dst, src: r(src) },
        Fadd4s { dst, lhs, rhs } => Fadd4s { dst: *dst, lhs: r(lhs), rhs: r(rhs) },
        Fmul4s { dst, lhs, rhs } => Fmul4s { dst: *dst, lhs: r(lhs), rhs: r(rhs) },
        Fneg4s { dst, src } => Fneg4s { dst: *dst, src: r(src) },
        Fmax4s { dst, lhs, rhs } => Fmax4s { dst: *dst, lhs: r(lhs), rhs: r(rhs) },
        Fsqrt4s { dst, src } => Fsqrt4s { dst: *dst, src: r(src) },
        Fdiv4s { dst, lhs, rhs } => Fdiv4s { dst: *dst, lhs: r(lhs), rhs: r(rhs) },
        Fcmgt4s { dst, lhs, rhs } => Fcmgt4s { dst: *dst, lhs: r(lhs), rhs: r(rhs) },
        And16b { dst, lhs, rhs } => And16b { dst: *dst, lhs: r(lhs), rhs: r(rhs) },
        Dup4sScalar { dst, src } => Dup4sScalar { dst: *dst, src: r(src) },
        LdrQReg { dst, base, offset } => LdrQReg { dst: *dst, base: r(base), offset: r(offset) },
        StrQReg { src, base, offset } => StrQReg { src: r(src), base: r(base), offset: r(offset) },
        // These have no vreg uses to remap
        MovImm64 { .. } | SpLoad { .. } | SpLoadF32 { .. } | Movi4sZero { .. }
        | Fmov4sOne { .. } | B { .. } | BCond { .. } | Label { .. } => inst.clone(),
    }
}

/// Return (defs, uses) for a MachInst.
pub fn defs_uses(inst: &MachInst) -> (Vec<VReg>, Vec<VReg>) {
    use MachInst::*;
    match inst {
        AddReg { dst, lhs, rhs } => (vec![*dst], vec![*lhs, *rhs]),
        AddImm { dst, src, .. } => (vec![*dst], vec![*src]),
        SubReg { dst, lhs, rhs } => (vec![*dst], vec![*lhs, *rhs]),
        SubImm { dst, src, .. } => (vec![*dst], vec![*src]),
        MulReg { dst, lhs, rhs } => (vec![*dst], vec![*lhs, *rhs]),
        Madd { dst, mul_lhs, mul_rhs, add } => (vec![*dst], vec![*mul_lhs, *mul_rhs, *add]),
        Sdiv { dst, lhs, rhs } => (vec![*dst], vec![*lhs, *rhs]),
        MovImm64 { dst, .. } => (vec![*dst], vec![]),
        MovReg { dst, src } => (vec![*dst], vec![*src]),
        LslImm { dst, src, .. } => (vec![*dst], vec![*src]),
        CmpReg { lhs, rhs } => (vec![], vec![*lhs, *rhs]),
        CmpImm { lhs, .. } => (vec![], vec![*lhs]),
        Csel { dst, t_val, f_val, .. } => (vec![*dst], vec![*t_val, *f_val]),
        SpLoad { dst, .. } => (vec![*dst], vec![]),
        SpStore { src, .. } => (vec![], vec![*src]),
        LdrSRegScaled { dst, base, offset } => (vec![*dst], vec![*base, *offset]),
        StrSRegScaled { src, base, offset } => (vec![], vec![*src, *base, *offset]),
        LdrSImm { dst, base, .. } => (vec![*dst], vec![*base]),
        StrSImm { src, base, .. } => (vec![], vec![*src, *base]),
        FmovWFromS { dst_gp, src_fp } => (vec![*dst_gp], vec![*src_fp]),
        FmovSFromW { dst_fp, src_gp } => (vec![*dst_fp], vec![*src_gp]),
        Fmadd { dst, mul_lhs, mul_rhs, add } => (vec![*dst], vec![*mul_lhs, *mul_rhs, *add]),
        FaddS { dst, lhs, rhs } => (vec![*dst], vec![*lhs, *rhs]),
        FsubS { dst, lhs, rhs } => (vec![*dst], vec![*lhs, *rhs]),
        FmulS { dst, lhs, rhs } => (vec![*dst], vec![*lhs, *rhs]),
        FdivS { dst, lhs, rhs } => (vec![*dst], vec![*lhs, *rhs]),
        FnegS { dst, src } => (vec![*dst], vec![*src]),
        FsqrtS { dst, src } => (vec![*dst], vec![*src]),
        ScvtfSX { dst, src } => (vec![*dst], vec![*src]),
        FcmpS { lhs, rhs } => (vec![], vec![*lhs, *rhs]),
        FcselS { dst, t_val, f_val, .. } => (vec![*dst], vec![*t_val, *f_val]),
        FmaxS { dst, lhs, rhs } => (vec![*dst], vec![*lhs, *rhs]),
        FmovS { dst, src } => (vec![*dst], vec![*src]),
        LdrSReg { dst, base, offset } => (vec![*dst], vec![*base, *offset]),
        Msub { dst, mul_lhs, mul_rhs, sub_from } => (vec![*dst], vec![*mul_lhs, *mul_rhs, *sub_from]),
        CallFpUnary { func_ptr, arg, result } => (vec![*result], vec![*func_ptr, *arg]),
        SpStoreF32 { src, .. } => (vec![], vec![*src]),
        SpLoadF32 { dst, .. } => (vec![*dst], vec![]),
        Movi4sZero { dst } => (vec![*dst], vec![]),
        Dup4sGp { dst, src_gp } => (vec![*dst], vec![*src_gp]),
        LdrQImm { dst, base, .. } => (vec![*dst], vec![*base]),
        StrQImm { src, base, .. } => (vec![], vec![*src, *base]),
        // FMLA: acc is BOTH read and written (tied operand)
        Fmla4s { acc, lhs, rhs } => (vec![*acc], vec![*acc, *lhs, *rhs]),
        MovVec { dst, src } => (vec![*dst], vec![*src]),
        Fadd4s { dst, lhs, rhs } => (vec![*dst], vec![*lhs, *rhs]),
        Fmul4s { dst, lhs, rhs } => (vec![*dst], vec![*lhs, *rhs]),
        Fneg4s { dst, src } => (vec![*dst], vec![*src]),
        Fmax4s { dst, lhs, rhs } => (vec![*dst], vec![*lhs, *rhs]),
        Fsqrt4s { dst, src } => (vec![*dst], vec![*src]),
        Fdiv4s { dst, lhs, rhs } => (vec![*dst], vec![*lhs, *rhs]),
        Fcmgt4s { dst, lhs, rhs } => (vec![*dst], vec![*lhs, *rhs]),
        And16b { dst, lhs, rhs } => (vec![*dst], vec![*lhs, *rhs]),
        Dup4sScalar { dst, src } => (vec![*dst], vec![*src]),
        LdrQReg { dst, base, offset } => (vec![*dst], vec![*base, *offset]),
        StrQReg { src, base, offset } => (vec![], vec![*src, *base, *offset]),
        Fmov4sOne { dst } => (vec![*dst], vec![]),
        B { .. } | BCond { .. } | Label { .. } => (vec![], vec![]),
    }
}

// ---------------------------------------------------------------------------
// Lowering: MachInst + Allocation -> ArmEmitter
// ---------------------------------------------------------------------------

/// Lower allocated MachInsts to real ARM64 instructions via ArmEmitter.
pub(crate) fn lower_to_emitter(insts: &[MachInst], alloc: &Allocation, e: &mut ArmEmitter) {
    for (idx, inst) in insts.iter().enumerate() {
        lower_one(inst, idx, alloc, e);
    }
}

fn lower_one(inst: &MachInst, idx: usize, a: &Allocation, e: &mut ArmEmitter) {
    // Use per-instruction allocation (handles spilled vregs correctly).
    let g = |vr: VReg| -> u8 { a.get_at(idx, vr) };
    use MachInst::*;
    match inst {
        AddReg { dst, lhs, rhs } => e.add_reg(g(*dst), g(*lhs), g(*rhs)),
        AddImm { dst, src, imm } => e.add_imm(g(*dst), g(*src), *imm),
        SubReg { dst, lhs, rhs } => e.sub_reg(g(*dst), g(*lhs), g(*rhs)),
        SubImm { dst, src, imm } => e.sub_imm(g(*dst), g(*src), *imm),
        MulReg { dst, lhs, rhs } => e.mul_reg(g(*dst), g(*lhs), g(*rhs)),
        Madd { dst, mul_lhs, mul_rhs, add } => {
            e.madd(g(*dst), g(*mul_lhs), g(*mul_rhs), g(*add));
        }
        Sdiv { dst, lhs, rhs } => e.sdiv(g(*dst), g(*lhs), g(*rhs)),
        MovImm64 { dst, val } => e.mov_imm64(g(*dst), *val),
        MovReg { dst, src } => e.mov_reg(g(*dst), g(*src)),
        LslImm { dst, src, shift } => e.lsl_imm(g(*dst), g(*src), *shift),
        CmpReg { lhs, rhs } => e.cmp_reg(g(*lhs), g(*rhs)),
        CmpImm { lhs, imm } => e.cmp_imm(g(*lhs), *imm),
        Csel { dst, t_val, f_val, cond } => {
            let rd = g(*dst);
            let rn = g(*t_val);
            let rm = g(*f_val);
            // CSEL Xd, Xn, Xm, cond
            e.emit(0x9A800000 | (rm as u32) << 16 | (*cond as u32) << 12 | (rn as u32) << 5 | rd as u32);
        }
        SpLoad { dst, slot } => {
            super::arm::load_from_sp_large(e, g(*dst), *slot);
        }
        SpStore { src, slot } => {
            super::arm::store_to_sp_large(e, g(*src), *slot);
        }
        LdrSRegScaled { dst, base, offset } => {
            e.ldr_s_reg_scaled(g(*dst), g(*base), g(*offset));
        }
        StrSRegScaled { src, base, offset } => {
            e.str_s_reg_scaled(g(*src), g(*base), g(*offset));
        }
        LdrSImm { dst, base, imm } => {
            e.ldr_s_imm(g(*dst), g(*base), *imm);
        }
        StrSImm { src, base, imm } => {
            e.str_s_imm(g(*src), g(*base), *imm);
        }
        FmovWFromS { dst_gp, src_fp } => {
            e.fmov_w_from_s(g(*dst_gp), g(*src_fp));
        }
        FmovSFromW { dst_fp, src_gp } => {
            e.fmov_s_from_w(g(*dst_fp), g(*src_gp));
        }
        Fmadd { dst, mul_lhs, mul_rhs, add } => {
            let rd = g(*dst);
            let rn = g(*mul_lhs);
            let rm = g(*mul_rhs);
            let ra = g(*add);
            // FMADD Sd, Sn, Sm, Sa = Sa + Sn*Sm  =>  Rd = Ra + Rn*Rm
            e.emit(0x1F000000 | (rm as u32) << 16 | (ra as u32) << 10 | (rn as u32) << 5 | rd as u32);
        }
        FaddS { dst, lhs, rhs } => e.fadd_s(g(*dst), g(*lhs), g(*rhs)),
        FsubS { dst, lhs, rhs } => e.fsub_s(g(*dst), g(*lhs), g(*rhs)),
        FmulS { dst, lhs, rhs } => e.fmul_s(g(*dst), g(*lhs), g(*rhs)),
        FdivS { dst, lhs, rhs } => e.fdiv_s(g(*dst), g(*lhs), g(*rhs)),
        FnegS { dst, src } => e.fneg_s(g(*dst), g(*src)),
        FsqrtS { dst, src } => e.fsqrt_s(g(*dst), g(*src)),
        ScvtfSX { dst, src } => e.scvtf_s_x(g(*dst), g(*src)),
        FcmpS { lhs, rhs } => e.fcmp_s(g(*lhs), g(*rhs)),
        FcselS { dst, t_val, f_val, cond } => {
            e.fcsel_s(g(*dst), g(*t_val), g(*f_val), *cond);
        }
        FmaxS { dst, lhs, rhs } => e.fmax_s(g(*dst), g(*lhs), g(*rhs)),
        FmovS { dst, src } => e.fmov_s(g(*dst), g(*src)),
        LdrSReg { dst, base, offset } => {
            // LDR Sd, [Xbase, Xoff]  (byte offset, no scale)
            let rd = g(*dst);
            let rn = g(*base);
            let rm = g(*offset);
            e.emit(0xBC606800 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
        }
        Msub { dst, mul_lhs, mul_rhs, sub_from } => {
            e.msub(g(*dst), g(*mul_lhs), g(*mul_rhs), g(*sub_from));
        }
        CallFpUnary { func_ptr, arg, result } => {
            let arg_phys = g(*arg);
            let res_phys = g(*result);
            let ptr_phys = g(*func_ptr);

            // FP save/restore is handled at the MachIR level (SpStoreF32/SpLoadF32
            // emitted around the call). GP regs are safe because the callee-saved-first
            // pool ordering puts long-lived values in X21-X28 (callee-saved).

            // Move arg to S0 if needed
            if arg_phys != 0 {
                e.fmov_s(0, arg_phys);
            }

            // Call
            e.blr(ptr_phys);

            // Move result from S0
            if res_phys != 0 {
                e.fmov_s(res_phys, 0);
            }
        }
        SpStoreF32 { src, slot } => {
            super::arm::store_s_to_sp_seq(e, g(*src), *slot);
        }
        SpLoadF32 { dst, slot } => {
            super::arm::load_s_from_sp_seq(e, g(*dst), *slot);
        }
        Movi4sZero { dst } => e.movi_4s_zero(g(*dst)),
        Dup4sGp { dst, src_gp } => e.dup_4s_gp(g(*dst), g(*src_gp)),
        LdrQImm { dst, base, imm } => e.ldr_q_imm(g(*dst), g(*base), *imm),
        StrQImm { src, base, imm } => e.str_q_imm(g(*src), g(*base), *imm),
        Fmla4s { acc, lhs, rhs } => {
            e.fmla_4s(g(*acc), g(*lhs), g(*rhs));
        }
        MovVec { dst, src } => {
            let vd = g(*dst);
            let vs = g(*src);
            // ORR Vd.16B, Vs.16B, Vs.16B
            e.emit(0x4EA01C00 | (vs as u32) << 16 | (vs as u32) << 5 | vd as u32);
        }
        Fadd4s { dst, lhs, rhs } => {
            // FADD Vd.4S, Vn.4S, Vm.4S: 0x4E20D400
            e.emit(0x4E20D400 | (g(*rhs) as u32) << 16 | (g(*lhs) as u32) << 5 | g(*dst) as u32);
        }
        Fmul4s { dst, lhs, rhs } => {
            // FMUL Vd.4S, Vn.4S, Vm.4S: 0x6E20DC00
            e.emit(0x6E20DC00 | (g(*rhs) as u32) << 16 | (g(*lhs) as u32) << 5 | g(*dst) as u32);
        }
        Fneg4s { dst, src } => {
            // FNEG Vd.4S, Vn.4S: 0x6EA0F800
            e.emit(0x6EA0F800 | (g(*src) as u32) << 5 | g(*dst) as u32);
        }
        Fmax4s { dst, lhs, rhs } => {
            e.fmax_4s(g(*dst), g(*lhs), g(*rhs));
        }
        Fsqrt4s { dst, src } => {
            // FSQRT Vd.4S, Vn.4S: 0x6EA1F800
            e.emit(0x6EA1F800 | (g(*src) as u32) << 5 | g(*dst) as u32);
        }
        Fdiv4s { dst, lhs, rhs } => {
            // FDIV Vd.4S, Vn.4S, Vm.4S: 0x6E20FC00
            e.emit(0x6E20FC00 | (g(*rhs) as u32) << 16 | (g(*lhs) as u32) << 5 | g(*dst) as u32);
        }
        Fcmgt4s { dst, lhs, rhs } => {
            // FCMGT Vd.4S, Vn.4S, Vm.4S: 0x6E20E400
            e.emit(0x6E20E400 | (g(*rhs) as u32) << 16 | (g(*lhs) as u32) << 5 | g(*dst) as u32);
        }
        And16b { dst, lhs, rhs } => {
            // AND Vd.16B, Vn.16B, Vm.16B: 0x4E201C00
            e.emit(0x4E201C00 | (g(*rhs) as u32) << 16 | (g(*lhs) as u32) << 5 | g(*dst) as u32);
        }
        Dup4sScalar { dst, src } => {
            // DUP Vd.4S, Vn.S[0]: 0x4E040400 | (Rn<<5) | Rd
            e.emit(0x4E040400 | (g(*src) as u32) << 5 | g(*dst) as u32);
        }
        LdrQReg { dst, base, offset } => {
            // LDR Qt, [Xn, Xm]: 0x3CE06800
            e.emit(0x3CE06800 | (g(*offset) as u32) << 16 | (g(*base) as u32) << 5 | g(*dst) as u32);
        }
        StrQReg { src, base, offset } => {
            // STR Qt, [Xn, Xm]: 0x3CA06800
            e.emit(0x3CA06800 | (g(*offset) as u32) << 16 | (g(*base) as u32) << 5 | g(*src) as u32);
        }
        Fmov4sOne { dst } => {
            // FMOV Vd.4S, #1.0: 0x4F03F400 | Rd
            // imm8 for 1.0 = 0b01110000 = 0x70, encoded as: a=0,b=1,c=1,d=1,e=0,f=0,g=0,h=0
            // FMOV (vector, immediate): 0x4F00F400 | (a:b:c:d:e:f:g:h encoding)
            // For 1.0f: abcdefgh = 01110000, op=0, cmode=1111
            e.emit(0x4F03F400 | g(*dst) as u32);
        }
        B { label } => e.b_label(*label),
        BCond { cond, label } => e.b_cond_label(*cond, *label),
        Label { label } => e.bind_label(*label),
    }
}

// ---------------------------------------------------------------------------
// Convenience: allocate + emit in one call
// ---------------------------------------------------------------------------

/// Compute the next vreg ID from an instruction stream (one past the max used).
fn compute_num_vregs(insts: &[MachInst]) -> u32 {
    let mut max_id: u32 = 0;
    for inst in insts {
        let (defs, uses) = defs_uses(inst);
        for vr in defs.iter().chain(uses.iter()) {
            max_id = max_id.max(vr.id + 1);
        }
    }
    max_id
}

/// Build, allocate, and emit MachInsts to an ArmEmitter.
/// `alloc_slot` provides stack slots for spilled vregs.
pub(crate) fn allocate_and_emit_with_spills(
    insts: &[MachInst],
    e: &mut ArmEmitter,
    next_vreg_id: u32,
    alloc_slot: &mut dyn FnMut() -> u32,
) {
    use crate::regalloc_bridge::regalloc_allocate;
    let (rewritten, alloc) = regalloc_allocate(insts, next_vreg_id, alloc_slot)
        .expect("register allocation failed");
    lower_to_emitter(&rewritten, &alloc, e);
}

/// Simple version without spill support (panics if allocation fails).
pub(crate) fn allocate_and_emit(insts: &[MachInst], e: &mut ArmEmitter) {
    use crate::regalloc_bridge::regalloc_allocate;
    let num_vregs = compute_num_vregs(insts);
    let mut next_slot = 0u32;
    let (rewritten, alloc) = regalloc_allocate(insts, num_vregs, &mut || {
        let s = next_slot;
        next_slot += 1;
        s
    }).expect("register allocation failed");
    lower_to_emitter(&rewritten, &alloc, e);
}
