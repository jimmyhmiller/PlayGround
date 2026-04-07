//! ARM64 native backend: emits AArch64 machine code from the fused loop IR.
//!
//! Memory model (matches WASM backend):
//! - A single flat memory buffer holds all inputs, intermediates, and outputs.
//! - The `execute` function takes a base pointer, initial heap offset, and
//!   a params array containing dim params followed by input byte offsets.
//! - Intermediate buffers are bump-allocated from the heap pointer.
//! - Returns the byte offset of the output buffer.
//!
//! Generated function signature (C ABI):
//!   extern "C" fn(
//!       memory: *mut u8,      // X0
//!       heap_ptr: u64,        // X1
//!       params: *const i64,   // X2 — [dim0, dim1, ..., input_off0, input_off1, ...]
//!   ) -> u64                  // returns byte offset of output

use std::collections::HashMap;
use tensor_lang_graph::{Dim, Graph, Op};
use crate::loop_ir::{self, Stmt, Inst, Index, ReduceOp, TileConfig};

// ============================================================================
// ARM64 instruction encoding helpers
// ============================================================================

// GP registers
const X0: u8 = 0;
const X1: u8 = 1;
const X2: u8 = 2;
// X3-X7: scratch for computation
const X3: u8 = 3;
const X4: u8 = 4;
const X5: u8 = 5;
const X6: u8 = 6;
const X7: u8 = 7;
const X8: u8 = 8;
const X9: u8 = 9;
const X10: u8 = 10;
const X11: u8 = 11;
const X12: u8 = 12;
const X13: u8 = 13;
const X14: u8 = 14;
const X15: u8 = 15;
// X16-X18: platform reserved
// X19-X28: callee-saved
const X19: u8 = 19;
const X20: u8 = 20;
const X21: u8 = 21;
const X22: u8 = 22;
const X23: u8 = 23;
const X24: u8 = 24;
const X25: u8 = 25;
const X26: u8 = 26;
const X27: u8 = 27;
const X28: u8 = 28;
const X29: u8 = 29; // frame pointer
const X30: u8 = 30; // link register
const XZR: u8 = 31; // zero register / SP depending on context

// NEON registers (V0-V31, using as Q regs = 128-bit)
// V0-V7: scratch (caller-saved)
// V8-V15: callee-saved (lower 64 bits only)
// V16-V31: scratch (caller-saved)

/// Convention for our generated code:
/// - X0: memory base pointer (preserved)
/// - X1: heap pointer (evolves during execution)
/// - X2: params array pointer (consumed at entry)
/// - X19: memory base (callee-saved copy)
/// - X20: heap pointer (callee-saved working copy)
/// - X21-X28: available for dim params, loop counters, etc.

// Stack frame layout for spilled values:
// [SP+0..SP+spill_area_size]: spill slots for IR instruction results (f32 each)
// Spill slots are 4 bytes each (f32), but aligned to 16-byte boundary.

// ============================================================================
// ARM64 instruction encoding
// ============================================================================

// Condition codes for B.cond
const COND_EQ: u8 = 0x0;
const COND_NE: u8 = 0x1;
const COND_GE: u8 = 0xA;
const COND_LT: u8 = 0xB;
const COND_GT: u8 = 0xC;
const COND_LE: u8 = 0xD;

/// ARM64 code emitter. Collects u32 instruction words.
pub(crate) struct ArmEmitter {
    code: Vec<u32>,
    /// Fixup locations: (code_index, target_label)
    branch_fixups: Vec<(usize, usize)>,
    /// Label positions: label_id -> code_index
    labels: HashMap<usize, usize>,
    next_label: usize,
}

impl ArmEmitter {
    pub(crate) fn new() -> Self {
        ArmEmitter {
            code: Vec::new(),
            branch_fixups: Vec::new(),
            labels: HashMap::new(),
            next_label: 0,
        }
    }

    pub(crate) fn alloc_label(&mut self) -> usize {
        let l = self.next_label;
        self.next_label += 1;
        l
    }

    pub(crate) fn bind_label(&mut self, label: usize) {
        self.labels.insert(label, self.code.len());
    }

    pub(crate) fn emit(&mut self, inst: u32) {
        self.code.push(inst);
    }

    pub(crate) fn current_offset(&self) -> usize {
        self.code.len()
    }

    // --- Integer arithmetic (64-bit) ---

    /// ADD Xd, Xn, Xm
    pub(crate) fn add_reg(&mut self, rd: u8, rn: u8, rm: u8) {
        // sf=1, shift=00, Rm, imm6=0, Rn, Rd
        self.emit(0x8B000000 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// ADD Xd, Xn, #imm12
    pub(crate) fn add_imm(&mut self, rd: u8, rn: u8, imm12: u32) {
        assert!(imm12 < 4096, "add_imm: imm12 out of range: {imm12}");
        // sf=1, op=0, S=0, 100010, sh=0
        self.emit(0x91000000 | (imm12 & 0xFFF) << 10 | (rn as u32) << 5 | rd as u32);
    }

    /// SUB Xd, Xn, Xm
    pub(crate) fn sub_reg(&mut self, rd: u8, rn: u8, rm: u8) {
        self.emit(0xCB000000 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// SUB Xd, Xn, #imm12
    pub(crate) fn sub_imm(&mut self, rd: u8, rn: u8, imm12: u32) {
        assert!(imm12 < 4096, "sub_imm: imm12 out of range: {imm12}");
        self.emit(0xD1000000 | (imm12 & 0xFFF) << 10 | (rn as u32) << 5 | rd as u32);
    }

    /// SUBS Xd, Xn, Xm (sets flags)
    pub(crate) fn subs_reg(&mut self, rd: u8, rn: u8, rm: u8) {
        self.emit(0xEB000000 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// CMP Xn, Xm  (alias for SUBS XZR, Xn, Xm)
    pub(crate) fn cmp_reg(&mut self, rn: u8, rm: u8) {
        self.subs_reg(XZR, rn, rm);
    }

    /// CMP Xn, #imm12
    pub(crate) fn cmp_imm(&mut self, rn: u8, imm12: u32) {
        assert!(imm12 < 4096);
        // SUBS XZR, Xn, #imm
        self.emit(0xF1000000 | (imm12 & 0xFFF) << 10 | (rn as u32) << 5 | XZR as u32);
    }

    /// MUL Xd, Xn, Xm  (alias of MADD Xd, Xn, Xm, XZR)
    pub(crate) fn mul_reg(&mut self, rd: u8, rn: u8, rm: u8) {
        self.emit(0x9B007C00 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// MADD Xd, Xn, Xm, Xa  (Xd = Xa + Xn * Xm)
    pub(crate) fn madd(&mut self, rd: u8, rn: u8, rm: u8, ra: u8) {
        self.emit(0x9B000000 | (rm as u32) << 16 | (ra as u32) << 10 | (rn as u32) << 5 | rd as u32);
    }

    /// SDIV Xd, Xn, Xm
    pub(crate) fn sdiv(&mut self, rd: u8, rn: u8, rm: u8) {
        self.emit(0x9AC00C00 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// MSUB Xd, Xn, Xm, Xa  (Xd = Xa - Xn * Xm) — for remainder via a - (a/b)*b
    pub(crate) fn msub(&mut self, rd: u8, rn: u8, rm: u8, ra: u8) {
        self.emit(0x9B008000 | (rm as u32) << 16 | (ra as u32) << 10 | (rn as u32) << 5 | rd as u32);
    }

    /// AND Xd, Xn, Xm
    pub(crate) fn and_reg(&mut self, rd: u8, rn: u8, rm: u8) {
        self.emit(0x8A000000 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// LSL Xd, Xn, #shift  (via UBFM)
    pub(crate) fn lsl_imm(&mut self, rd: u8, rn: u8, shift: u32) {
        assert!(shift < 64);
        let immr = (64 - shift) & 63;
        let imms = 63 - shift;
        self.emit(0xD3400000 | (immr << 16) | (imms << 10) | (rn as u32) << 5 | rd as u32);
    }

    /// LSR Xd, Xn, #shift  (via UBFM)
    pub(crate) fn lsr_imm(&mut self, rd: u8, rn: u8, shift: u32) {
        assert!(shift < 64);
        self.emit(0xD340FC00 | (shift << 16) | (63 << 10) | (rn as u32) << 5 | rd as u32);
    }

    /// ASR Xd, Xn, #shift  (via SBFM)
    pub(crate) fn asr_imm(&mut self, rd: u8, rn: u8, shift: u32) {
        assert!(shift < 64);
        self.emit(0x9340FC00 | (shift << 16) | (63 << 10) | (rn as u32) << 5 | rd as u32);
    }

    // --- Move / immediate ---

    /// MOVZ Xd, #imm16, LSL #shift  (shift = 0, 16, 32, 48)
    pub(crate) fn movz(&mut self, rd: u8, imm16: u16, shift: u8) {
        let hw = (shift / 16) as u32;
        self.emit(0xD2800000 | (hw << 21) | (imm16 as u32) << 5 | rd as u32);
    }

    /// MOVK Xd, #imm16, LSL #shift
    pub(crate) fn movk(&mut self, rd: u8, imm16: u16, shift: u8) {
        let hw = (shift / 16) as u32;
        self.emit(0xF2800000 | (hw << 21) | (imm16 as u32) << 5 | rd as u32);
    }

    /// MOV Xd, Xm (via ORR Xd, XZR, Xm)
    pub(crate) fn mov_reg(&mut self, rd: u8, rm: u8) {
        self.emit(0xAA0003E0 | (rm as u32) << 16 | rd as u32);
    }

    /// Load a 64-bit immediate into Xd using MOVZ + MOVK sequence.
    pub(crate) fn mov_imm64(&mut self, rd: u8, val: u64) {
        if val == 0 {
            self.mov_reg(rd, XZR);
            return;
        }
        let w0 = val as u16;
        let w1 = (val >> 16) as u16;
        let w2 = (val >> 32) as u16;
        let w3 = (val >> 48) as u16;

        // Find first non-zero halfword for MOVZ
        let mut first = true;
        for (shift, w) in [(0u8, w0), (16, w1), (32, w2), (48, w3)] {
            if w != 0 || (first && shift == 48) {
                if first {
                    self.movz(rd, w, shift);
                    first = false;
                } else {
                    self.movk(rd, w, shift);
                }
            }
        }
        if first {
            // All zero — shouldn't reach here due to val==0 check
            self.movz(rd, 0, 0);
        }
    }

    /// Load a 32-bit immediate into Xd (zero-extended).
    pub(crate) fn mov_imm32(&mut self, rd: u8, val: u32) {
        if val == 0 {
            self.mov_reg(rd, XZR);
            return;
        }
        let w0 = val as u16;
        let w1 = (val >> 16) as u16;
        // Use 32-bit MOVZ (sf=0) for small values, but we use 64-bit
        // since we work in 64-bit mode for pointers. Zero-extends naturally.
        self.movz(rd, w0, 0);
        if w1 != 0 {
            self.movk(rd, w1, 16);
        }
    }

    // --- Load/Store (64-bit GP) ---

    /// LDR Xt, [Xn, #imm]  (unsigned offset, scaled by 8)
    pub(crate) fn ldr_x(&mut self, rt: u8, rn: u8, offset: u32) {
        assert!(offset % 8 == 0 && offset / 8 < 4096);
        let imm12 = offset / 8;
        self.emit(0xF9400000 | (imm12 << 10) | (rn as u32) << 5 | rt as u32);
    }

    /// STR Xt, [Xn, #imm]  (unsigned offset, scaled by 8)
    pub(crate) fn str_x(&mut self, rt: u8, rn: u8, offset: u32) {
        assert!(offset % 8 == 0 && offset / 8 < 4096);
        let imm12 = offset / 8;
        self.emit(0xF9000000 | (imm12 << 10) | (rn as u32) << 5 | rt as u32);
    }

    /// LDR Xt, [Xn, Xm]  (register offset)
    pub(crate) fn ldr_x_reg(&mut self, rt: u8, rn: u8, rm: u8) {
        // LDR Xt, [Xn, Xm, LSL #0]
        self.emit(0xF8600800 | (rm as u32) << 16 | (rn as u32) << 5 | rt as u32);
    }

    // --- Load/Store (32-bit, f32) ---

    /// LDR St, [Xn, Xm, LSL #2]  (register offset, scaled)
    pub(crate) fn ldr_s_reg_scaled(&mut self, rt: u8, rn: u8, rm: u8) {
        // size=10, V=1, opc=01, Rm, option=011, S=1, Rn, Rt
        self.emit(0xBC607800 | (rm as u32) << 16 | (rn as u32) << 5 | rt as u32);
    }

    /// STR St, [Xn, Xm, LSL #2]  (register offset, scaled)
    pub(crate) fn str_s_reg_scaled(&mut self, rt: u8, rn: u8, rm: u8) {
        // size=10, V=1, opc=00, Rm, option=011, S=1, Rn, Rt
        self.emit(0xBC207800 | (rm as u32) << 16 | (rn as u32) << 5 | rt as u32);
    }

    /// LDR St, [Xn, #imm]  (unsigned offset, scaled by 4)
    pub(crate) fn ldr_s_imm(&mut self, rt: u8, rn: u8, offset: u32) {
        assert!(offset % 4 == 0 && offset / 4 < 4096, "ldr_s_imm offset {offset} out of range");
        let imm12 = offset / 4;
        self.emit(0xBD400000 | (imm12 << 10) | (rn as u32) << 5 | rt as u32);
    }

    /// STR St, [Xn, #imm]  (unsigned offset, scaled by 4)
    pub(crate) fn str_s_imm(&mut self, rt: u8, rn: u8, offset: u32) {
        assert!(offset % 4 == 0 && offset / 4 < 4096, "str_s_imm offset {offset} out of range");
        let imm12 = offset / 4;
        self.emit(0xBD000000 | (imm12 << 10) | (rn as u32) << 5 | rt as u32);
    }

    // --- Load/Store (128-bit NEON Q register) ---

    /// LDR Qt, [Xn, Xm]  (register offset, no scale)
    pub(crate) fn ldr_q_reg(&mut self, rt: u8, rn: u8, rm: u8) {
        // size=00, V=1, opc=11, Rm, option=011, S=0, Rn, Rt
        self.emit(0x3CE06800 | (rm as u32) << 16 | (rn as u32) << 5 | rt as u32);
    }

    /// STR Qt, [Xn, Xm]  (register offset, no scale)
    pub(crate) fn str_q_reg(&mut self, rt: u8, rn: u8, rm: u8) {
        // size=00, V=1, opc=10, Rm, option=011, S=0, Rn, Rt
        self.emit(0x3CA06800 | (rm as u32) << 16 | (rn as u32) << 5 | rt as u32);
    }

    /// LDR Qt, [Xn, #imm]  (unsigned offset, scaled by 16)
    pub(crate) fn ldr_q_imm(&mut self, rt: u8, rn: u8, offset: u32) {
        assert!(offset % 16 == 0 && offset / 16 < 4096);
        let imm12 = offset / 16;
        self.emit(0x3DC00000 | (imm12 << 10) | (rn as u32) << 5 | rt as u32);
    }

    /// STR Qt, [Xn, #imm]  (unsigned offset, scaled by 16)
    pub(crate) fn str_q_imm(&mut self, rt: u8, rn: u8, offset: u32) {
        assert!(offset % 16 == 0 && offset / 16 < 4096);
        let imm12 = offset / 16;
        self.emit(0x3D800000 | (imm12 << 10) | (rn as u32) << 5 | rt as u32);
    }

    // --- Scalar FP (S registers) ---

    /// FADD Sd, Sn, Sm
    pub(crate) fn fadd_s(&mut self, rd: u8, rn: u8, rm: u8) {
        self.emit(0x1E202800 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// FSUB Sd, Sn, Sm
    pub(crate) fn fsub_s(&mut self, rd: u8, rn: u8, rm: u8) {
        self.emit(0x1E203800 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// FMUL Sd, Sn, Sm
    pub(crate) fn fmul_s(&mut self, rd: u8, rn: u8, rm: u8) {
        self.emit(0x1E200800 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// FDIV Sd, Sn, Sm
    pub(crate) fn fdiv_s(&mut self, rd: u8, rn: u8, rm: u8) {
        self.emit(0x1E201800 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// FNEG Sd, Sn
    pub(crate) fn fneg_s(&mut self, rd: u8, rn: u8) {
        self.emit(0x1E214000 | (rn as u32) << 5 | rd as u32);
    }

    /// FSQRT Sd, Sn
    pub(crate) fn fsqrt_s(&mut self, rd: u8, rn: u8) {
        self.emit(0x1E21C000 | (rn as u32) << 5 | rd as u32);
    }

    /// FMOV Sd, Sn
    pub(crate) fn fmov_s(&mut self, rd: u8, rn: u8) {
        self.emit(0x1E204000 | (rn as u32) << 5 | rd as u32);
    }

    /// FMOV Sd, Wn  (GP -> FP)
    pub(crate) fn fmov_s_from_w(&mut self, rd: u8, rn: u8) {
        // sf=0, ftype=00, rmode=00, opcode=111
        self.emit(0x1E270000 | (rn as u32) << 5 | rd as u32);
    }

    /// FMOV Wd, Sn  (FP -> GP)
    pub(crate) fn fmov_w_from_s(&mut self, rd: u8, rn: u8) {
        // sf=0, ftype=00, rmode=00, opcode=110
        self.emit(0x1E260000 | (rn as u32) << 5 | rd as u32);
    }

    /// SCVTF Sd, Xn  (signed int64 to float32)
    pub(crate) fn scvtf_s_x(&mut self, rd: u8, rn: u8) {
        // sf=1, ftype=00, rmode=00, opcode=010
        self.emit(0x9E220000 | (rn as u32) << 5 | rd as u32);
    }

    /// FCMP Sn, Sm
    pub(crate) fn fcmp_s(&mut self, rn: u8, rm: u8) {
        self.emit(0x1E202000 | (rm as u32) << 16 | (rn as u32) << 5);
    }

    /// FCSEL Sd, Sn, Sm, cond
    pub(crate) fn fcsel_s(&mut self, rd: u8, rn: u8, rm: u8, cond: u8) {
        self.emit(0x1E200C00 | (rm as u32) << 16 | (cond as u32) << 12 | (rn as u32) << 5 | rd as u32);
    }

    /// FMIN Sd, Sn, Sm
    pub(crate) fn fmin_s(&mut self, rd: u8, rn: u8, rm: u8) {
        self.emit(0x1E205800 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// FMAX Sd, Sn, Sm
    pub(crate) fn fmax_s(&mut self, rd: u8, rn: u8, rm: u8) {
        self.emit(0x1E204800 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// FRINTM Sd, Sn  (floor)
    pub(crate) fn frintm_s(&mut self, rd: u8, rn: u8) {
        self.emit(0x1E254000 | (rn as u32) << 5 | rd as u32);
    }

    /// FCVTZS Wd, Sn  (float32 to signed int32, round toward zero)
    pub(crate) fn fcvtzs_w_s(&mut self, rd: u8, rn: u8) {
        // sf=0, ftype=00, rmode=11, opcode=000
        self.emit(0x1E380000 | (rn as u32) << 5 | rd as u32);
    }

    // --- NEON f32x4 (Q register, .4S arrangement) ---

    /// FADD Vd.4S, Vn.4S, Vm.4S
    pub(crate) fn fadd_4s(&mut self, rd: u8, rn: u8, rm: u8) {
        self.emit(0x4E20D400 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// FSUB Vd.4S, Vn.4S, Vm.4S
    pub(crate) fn fsub_4s(&mut self, rd: u8, rn: u8, rm: u8) {
        self.emit(0x4EA0D400 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// FMUL Vd.4S, Vn.4S, Vm.4S
    pub(crate) fn fmul_4s(&mut self, rd: u8, rn: u8, rm: u8) {
        self.emit(0x6E20DC00 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// FMLA Vd.4S, Vn.4S, Vm.4S  (Vd += Vn * Vm) — fused multiply-accumulate!
    pub(crate) fn fmla_4s(&mut self, rd: u8, rn: u8, rm: u8) {
        self.emit(0x4E20CC00 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// FDIV Vd.4S, Vn.4S, Vm.4S
    pub(crate) fn fdiv_4s(&mut self, rd: u8, rn: u8, rm: u8) {
        self.emit(0x6E20FC00 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// FMAX Vd.4S, Vn.4S, Vm.4S
    pub(crate) fn fmax_4s(&mut self, rd: u8, rn: u8, rm: u8) {
        self.emit(0x4E20F400 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// FNEG Vd.4S, Vn.4S
    pub(crate) fn fneg_4s(&mut self, rd: u8, rn: u8) {
        self.emit(0x6EA0F800 | (rn as u32) << 5 | rd as u32);
    }

    /// FSQRT Vd.4S, Vn.4S
    pub(crate) fn fsqrt_4s(&mut self, rd: u8, rn: u8) {
        self.emit(0x6EA1F800 | (rn as u32) << 5 | rd as u32);
    }

    /// FCMGT Vd.4S, Vn.4S, Vm.4S  (per-lane Vn > Vm ? 0xFFFFFFFF : 0)
    pub(crate) fn fcmgt_4s(&mut self, rd: u8, rn: u8, rm: u8) {
        self.emit(0x6EA0E400 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// FCMLT Vd.4S, Vn.4S, Vm.4S  (Vn < Vm is same as FCMGT with swapped args)
    pub(crate) fn fcmlt_4s(&mut self, rd: u8, rn: u8, rm: u8) {
        self.fcmgt_4s(rd, rm, rn); // swap operands: Vm > Vn ≡ Vn < Vm
    }

    /// BSL Vd.16B, Vn.16B, Vm.16B  (bitwise select: Vd = (Vd & Vn) | (~Vd & Vm))
    pub(crate) fn bsl_16b(&mut self, rd: u8, rn: u8, rm: u8) {
        self.emit(0x6E601C00 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// BIF Vd.16B, Vn.16B, Vm.16B  (bit insert if false)
    pub(crate) fn bif_16b(&mut self, rd: u8, rn: u8, rm: u8) {
        self.emit(0x6EE01C00 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    /// DUP Vd.4S, Wn  (broadcast GP scalar to all 4 lanes)
    pub(crate) fn dup_4s_gp(&mut self, vd: u8, wn: u8) {
        // q=1, imm5=00100 (for .4S), Rn, Rd
        self.emit(0x4E040C00 | (wn as u32) << 5 | vd as u32);
    }

    /// DUP Vd.4S, Vn.S[idx]  (broadcast one lane to all)
    pub(crate) fn dup_4s_lane(&mut self, vd: u8, vn: u8, idx: u8) {
        let imm5 = (idx as u32) << 3 | 0x4; // .S arrangement, idx encoded
        self.emit(0x4E000400 | (imm5 << 16) | (vn as u32) << 5 | vd as u32);
    }

    /// MOVI Vd.4S, #0  (zero a Q register)
    pub(crate) fn movi_4s_zero(&mut self, vd: u8) {
        self.emit(0x4F000400 | vd as u32);
    }

    /// FMOV Vd.4S, #imm  — limited to f32 immediates ARM can encode.
    /// For arbitrary constants, use load from literal pool instead.

    // --- Extract lane ---

    /// MOV Wd, Vn.S[idx]  (extract 32-bit lane to GP register)
    pub(crate) fn umov_w_s(&mut self, wd: u8, vn: u8, idx: u8) {
        let imm5 = (idx as u32) << 3 | 0x4;
        self.emit(0x0E003C00 | (imm5 << 16) | (vn as u32) << 5 | wd as u32);
    }

    /// FADDP Vd.4S, Vn.4S, Vm.4S  (pairwise add)
    pub(crate) fn faddp_4s(&mut self, rd: u8, rn: u8, rm: u8) {
        self.emit(0x6E20D400 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    // --- Branches ---

    /// B <label>  (unconditional, PC-relative, ±128MB)
    pub(crate) fn b_label(&mut self, label: usize) {
        self.branch_fixups.push((self.code.len(), label));
        self.emit(0x14000000); // placeholder
    }

    /// B.cond <label>
    pub(crate) fn b_cond_label(&mut self, cond: u8, label: usize) {
        self.branch_fixups.push((self.code.len(), label));
        self.emit(0x54000000 | cond as u32); // placeholder
    }

    /// BLR Xn  (call through register)
    pub(crate) fn blr(&mut self, rn: u8) {
        self.emit(0xD63F0000 | (rn as u32) << 5);
    }

    /// RET (return to LR)
    pub(crate) fn ret(&mut self) {
        self.emit(0xD65F03C0);
    }

    // --- Stack / frame ---

    /// STP Xt1, Xt2, [SP, #imm]!  (pre-index, push to stack)
    pub(crate) fn stp_pre(&mut self, rt1: u8, rt2: u8, imm: i32) {
        assert!(imm % 8 == 0 && imm >= -512 && imm < 512);
        let imm7 = ((imm / 8) as u32) & 0x7F;
        self.emit(0xA9800000 | (imm7 << 15) | (rt2 as u32) << 10 | (31u32) << 5 | rt1 as u32);
    }

    /// LDP Xt1, Xt2, [SP], #imm  (post-index, pop from stack)
    pub(crate) fn ldp_post(&mut self, rt1: u8, rt2: u8, imm: i32) {
        assert!(imm % 8 == 0 && imm >= -512 && imm < 512);
        let imm7 = ((imm / 8) as u32) & 0x7F;
        self.emit(0xA8C00000 | (imm7 << 15) | (rt2 as u32) << 10 | (31u32) << 5 | rt1 as u32);
    }

    /// STP Qt1, Qt2, [SP, #imm]!  (pre-index, push Q regs)
    pub(crate) fn stp_q_pre(&mut self, rt1: u8, rt2: u8, imm: i32) {
        assert!(imm % 16 == 0 && imm >= -1024 && imm < 1024);
        let imm7 = ((imm / 16) as u32) & 0x7F;
        self.emit(0xAD800000 | (imm7 << 15) | (rt2 as u32) << 10 | (31u32) << 5 | rt1 as u32);
    }

    /// LDP Qt1, Qt2, [SP], #imm  (post-index, pop Q regs)
    pub(crate) fn ldp_q_post(&mut self, rt1: u8, rt2: u8, imm: i32) {
        assert!(imm % 16 == 0 && imm >= -1024 && imm < 1024);
        let imm7 = ((imm / 16) as u32) & 0x7F;
        self.emit(0xACC00000 | (imm7 << 15) | (rt2 as u32) << 10 | (31u32) << 5 | rt1 as u32);
    }

    // --- Resolve branches ---

    pub(crate) fn resolve_branches(&mut self) {
        for &(code_idx, label) in &self.branch_fixups {
            let target = self.labels[&label];
            let offset = target as i64 - code_idx as i64; // in instructions
            let inst = self.code[code_idx];
            if inst & 0xFC000000 == 0x14000000 {
                // B (unconditional): imm26
                let imm26 = (offset as u32) & 0x03FFFFFF;
                self.code[code_idx] = 0x14000000 | imm26;
            } else if inst & 0xFF000000 == 0x54000000 {
                // B.cond: imm19 at bits [23:5]
                let imm19 = (offset as u32) & 0x7FFFF;
                self.code[code_idx] = (inst & 0xFF00001F) | (imm19 << 5);
            } else {
                panic!("unknown branch instruction to fixup: {inst:#010x}");
            }
        }
    }

    /// Finalize: resolve branches and return the code bytes.
    pub(crate) fn finalize(mut self) -> Vec<u8> {
        self.resolve_branches();
        let mut bytes = Vec::with_capacity(self.code.len() * 4);
        for inst in &self.code {
            bytes.extend_from_slice(&inst.to_le_bytes());
        }
        bytes
    }
}

// ============================================================================
// Codegen context
// ============================================================================

type DimLocals = HashMap<String, u8>; // symbolic dim name -> GP register with its value

struct CodegenCtx {
    /// Buffer pointer: buf_id -> GP register holding the absolute pointer
    buf_ptrs: HashMap<usize, u32>,
    /// Spill slots: IR instruction index -> stack offset from SP (in bytes)
    inst_slots: HashMap<usize, u32>,
    /// Dim variable locations: dim_index -> stack offset from SP (in bytes)
    dim_slots: HashMap<usize, u32>,
    /// Named symbolic dim params -> stack offset from SP
    dim_param_slots: HashMap<String, u32>,
    /// Next available spill slot offset
    next_slot: u32,
    /// Exp2 helper function pointer (loaded into a register at call time)
    exp2_fn_ptr: u64,
    /// Log2 helper function pointer
    log2_fn_ptr: u64,
}

impl CodegenCtx {
    pub(crate) fn alloc_slot(&mut self) -> u32 {
        let off = self.next_slot;
        self.next_slot += 8; // 8-byte aligned slots for simplicity
        off
    }

    pub(crate) fn get_or_alloc_inst_slot(&mut self, j: usize) -> u32 {
        if let Some(&off) = self.inst_slots.get(&j) {
            off
        } else {
            let off = self.alloc_slot();
            self.inst_slots.insert(j, off);
            off
        }
    }

    pub(crate) fn get_or_alloc_dim_slot(&mut self, d: usize) -> u32 {
        if let Some(&off) = self.dim_slots.get(&d) {
            off
        } else {
            let off = self.alloc_slot();
            self.dim_slots.insert(d, off);
            off
        }
    }
}

// ============================================================================
// Helper functions called from generated code
// ============================================================================

extern "C" fn _arm_debug_ptrs(out: u64, a: u64, b: u64) {
    eprintln!("[ARM DEBUG] out={out:#x}, a={a:#x}, b={b:#x}");
}

/// exp2(x) polynomial approximation — same algorithm as WASM backend.
extern "C" fn arm_exp2_f32(x: f32) -> f32 {
    let x = x.max(-126.0).min(127.0);
    let x_floor = x.floor();
    let n = x_floor as i32;
    let frac = x - x_floor;

    // 2^n via IEEE 754
    let int_pow = f32::from_bits(((n + 127) as u32) << 23);

    // Minimax degree-5 polynomial for 2^frac on [0,1)
    let poly = 1.0
        + frac
            * (0.6931476
                + frac
                    * (0.24020687
                        + frac * (0.055658683 + frac * (0.009196793 + frac * 0.0017896632))));

    int_pow * poly
}

/// log2(x) via IEEE 754 decomposition — same algorithm as WASM backend.
extern "C" fn arm_log2_f32(x: f32) -> f32 {
    let bits = x.to_bits();
    let e = ((bits >> 23) & 0xFF) as i32 - 127;
    let m_bits = (bits & 0x007FFFFF) | 0x3F800000;
    let m = f32::from_bits(m_bits);

    let p = m - 1.0;
    // Polynomial for log2(1+p) on [0, 1)
    let log2_m = p
        * (1.4426950
            + p * (-0.7213476
                + p * (0.4808983 + p * (-0.3606586 + p * (0.2464387 - p * 0.1213475)))));

    e as f32 + log2_m
}

// ============================================================================
// ARM backend
// ============================================================================

pub struct ArmBackend;

impl ArmBackend {
    /// Emit ARM64 machine code from a graph. Returns the raw code bytes
    /// plus metadata needed by the runtime.
    pub fn emit_fused(&self, graph: &Graph) -> ArmCode {
        self.emit_fused_inner(graph, None)
    }

    pub fn emit_fused_multi_output(
        &self,
        graph: &Graph,
        outputs: &[tensor_lang_graph::NodeId],
    ) -> ArmCode {
        self.emit_fused_inner(graph, Some(outputs))
    }

    pub(crate) fn emit_fused_inner(
        &self,
        graph: &Graph,
        multi_outputs: Option<&[tensor_lang_graph::NodeId]>,
    ) -> ArmCode {
        let mut stmts = if let Some(outputs) = multi_outputs {
            loop_ir::lower_with_outputs(graph, outputs)
        } else {
            loop_ir::lower(graph)
        };
        loop_ir::unfuse_matmul_bodies(&mut stmts);
        loop_ir::tile_reduce_loops(&mut stmts);

        // Collect symbolic dim params
        let mut dim_params: Vec<String> = Vec::new();
        for node in &graph.nodes {
            for d in &node.shape {
                collect_params(d, &mut dim_params);
            }
        }
        dim_params.sort();
        dim_params.dedup();

        // Count inputs
        let inputs: Vec<(usize, String)> = graph
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| {
                if let Op::Input { name } = &n.op {
                    Some((i, name.clone()))
                } else {
                    None
                }
            })
            .collect();

        let n_dim_params = dim_params.len();
        let n_inputs = inputs.len();

        let mut ctx = CodegenCtx {
            buf_ptrs: HashMap::new(),
            inst_slots: HashMap::new(),
            dim_slots: HashMap::new(),
            dim_param_slots: HashMap::new(),
            next_slot: 0,
            exp2_fn_ptr: arm_exp2_f32 as u64,
            log2_fn_ptr: arm_log2_f32 as u64,
        };

        let mut e = ArmEmitter::new();

        // === Prologue ===
        // Save callee-saved registers and LR
        e.stp_pre(X29, X30, -16);
        e.mov_reg(X29, 31); // (doesn't set FP to SP correctly but we don't need FP)

        // Save callee-saved GP regs we'll use (X19-X28)
        e.stp_pre(X19, X20, -16);
        e.stp_pre(X21, X22, -16);
        e.stp_pre(X23, X24, -16);
        e.stp_pre(X25, X26, -16);
        e.stp_pre(X27, X28, -16);

        // Save callee-saved FP regs (V8-V15) — 128-bit Q-reg pairs
        e.stp_q_pre(8, 9, -32);
        e.stp_q_pre(10, 11, -32);
        e.stp_q_pre(12, 13, -32);
        e.stp_q_pre(14, 15, -32);

        // Placeholder for frame allocation — we'll patch this after codegen
        // Reserve space for: MOVZ X9, #imm16; MOVK X9, #imm16; SUB SP, SP, X9
        let frame_patch_offset = e.current_offset();
        e.emit(0xD503201F); // NOP placeholder (will be MOVZ)
        e.emit(0xD503201F); // NOP placeholder (will be MOVK or NOP)
        // SUB SP, SP, X9, UXTX  (extended register form, treats Rd/Rn as SP)
        // Encoding: sf=1, op=1, S=0, 01011 00 1 Rm option=011(UXTX) imm3=000 Rn=SP(31) Rd=SP(31)
        e.emit(0xCB2963FF); // SUB SP, SP, X9, UXTX

        // Save function arguments to callee-saved registers
        // X0 = memory base, X1 = heap_ptr, X2 = params
        e.mov_reg(X19, X0);  // X19 = memory base (preserved)
        e.mov_reg(X20, X1);  // X20 = heap pointer (evolves)

        // Load dim params from params array into stack slots
        for (i, name) in dim_params.iter().enumerate() {
            let slot = ctx.alloc_slot();
            ctx.dim_param_slots.insert(name.clone(), slot);
            // params[i] is an i64
            e.ldr_x(X9, X2, (i * 8) as u32);
            // Store to stack slot
            store_to_sp_large(&mut e, X9, slot);
        }

        // Map input node IDs to their byte offsets (loaded from params array)
        // The input offsets follow dim params in the params array
        let mut input_offset_slots: HashMap<usize, u32> = HashMap::new();
        for (i, (node_id, _)) in inputs.iter().enumerate() {
            let slot = ctx.alloc_slot();
            input_offset_slots.insert(*node_id, slot);
            let param_idx = n_dim_params + i;
            e.ldr_x(X9, X2, (param_idx * 8) as u32);
            store_to_sp_large(&mut e, X9, slot);
        }

        // Track output buf_id
        let mut last_buf: usize = 0;

        // === Emit statements ===
        for stmt in &stmts {
            match stmt {
                Stmt::Alloc { buf, size } => {
                    last_buf = *buf;
                    let slot = ctx.alloc_slot();
                    ctx.buf_ptrs.insert(*buf, 0); // placeholder

                    if let Some(&input_slot) = input_offset_slots.get(buf) {
                        // Input buffer: ptr = memory_base + offset
                        load_from_sp_large(&mut e, X9, input_slot);
                        e.add_reg(X9, X19, X9);
                        store_to_sp_large(&mut e, X9, slot);
                    } else {
                        // Intermediate: bump-allocate from heap
                        // Align heap to 16
                        e.add_imm(X20, X20, 15);
                        e.mov_imm64(X9, !15u64);
                        e.and_reg(X20, X20, X9);

                        // buf_ptr = memory_base + heap_offset
                        e.add_reg(X9, X19, X20);
                        store_to_sp_large(&mut e, X9, slot);

                        // heap_offset += size * 4
                        emit_dim_to_reg(&mut e, &size, &ctx, X10);
                        e.lsl_imm(X10, X10, 2); // * 4
                        e.add_reg(X20, X20, X10);
                    }
                    // Store the slot index in buf_ptrs for later lookup
                    // We abuse buf_ptrs to store the slot offset instead of a register
                    ctx.buf_ptrs.insert(*buf, slot);
                    // We'll load it into a register when needed
                }

                Stmt::Fill { buf, value } => {
                    let buf_slot = ctx.buf_ptrs[buf];
                    let val_bits = (*value as f32).to_bits();
                    load_from_sp_large(&mut e, X9, buf_slot); // buf ptr
                    e.mov_imm32(X10, val_bits);
                    // STR W10, [X9]
                    e.emit(0xB9000000 | (X9 as u32) << 5 | X10 as u32);
                }

                Stmt::FillArange { buf, size } => {
                    let buf_slot = ctx.buf_ptrs[buf];
                    load_from_sp_large(&mut e, X9, buf_slot); // buf ptr
                    emit_dim_to_reg(&mut e, size, &ctx, X10); // size

                    let loop_start = e.alloc_label();
                    let loop_end = e.alloc_label();

                    // X11 = counter = 0
                    e.mov_reg(X11, XZR);
                    e.bind_label(loop_start);
                    e.cmp_reg(X11, X10);
                    e.b_cond_label(COND_GE, loop_end);

                    // Convert counter to f32 and store
                    e.scvtf_s_x(0, X11); // S0 = (f32)X11
                    // STR S0, [X9, X11, LSL #2]
                    e.str_s_reg_scaled(0, X9, X11);
                    e.add_imm(X11, X11, 1);
                    e.b_label(loop_start);
                    e.bind_label(loop_end);
                }

                Stmt::Pad { buf, input_buf, output_shape, input_shape, padding } => {
                    emit_pad(&mut e, &mut ctx, *buf, *input_buf, output_shape, input_shape, padding);
                }

                Stmt::Loop { buf, shape, reduce, body, result, tile } => {
                    last_buf = *buf;
                    ctx.inst_slots.clear();
                    let buf_slot = ctx.buf_ptrs[buf];

                    if let (Some(reduce), Some(tile_cfg)) = (reduce.as_ref(), tile.as_ref()) {
                        if std::env::var("ARM_BODY_DEBUG").is_ok() {
                            eprintln!("TILED shape={:?} result={} body_len={}", shape, *result, body.len());
                        }
                        emit_tiled_loop(&mut e, &mut ctx, buf_slot, shape, reduce, body, *result, tile_cfg);
                    } else if let Some(reduce) = reduce.as_ref() {
                        emit_reduce_loop(&mut e, &mut ctx, buf_slot, shape, reduce, body, *result);
                    } else {
                        emit_elementwise_loop(&mut e, &mut ctx, buf_slot, shape, body, *result);
                    }
                }
            }
        }

        // Return: output byte offset = buf_ptr - memory_base
        let out_slot = ctx.buf_ptrs[&last_buf];
        load_from_sp_large(&mut e, X9, out_slot);
        e.sub_reg(X0, X9, X19); // offset = ptr - base

        // === Patch frame size ===
        // Now we know the actual frame size from ctx.next_slot
        let frame_size = ((ctx.next_slot + 15) & !15) as u32;
        // Patch the prologue: MOVZ X9, #lo16; MOVK X9, #hi16; SUB SP, SP, X9
        let lo = (frame_size & 0xFFFF) as u16;
        let hi = ((frame_size >> 16) & 0xFFFF) as u16;
        e.code[frame_patch_offset] = 0xD2800000 | ((lo as u32) << 5) | X9 as u32; // MOVZ X9, #lo
        e.code[frame_patch_offset + 1] = if hi != 0 {
            0xF2A00000 | ((hi as u32) << 5) | X9 as u32 // MOVK X9, #hi, LSL #16
        } else {
            0xD503201F // NOP
        };
        // e.code[frame_patch_offset + 2] is already SUB SP, SP, X9

        // === Epilogue ===
        // Deallocate spill area: ADD SP, SP, X9, UXTX
        e.mov_imm32(X9, frame_size);
        // ADD SP, SP, X9, UXTX (extended register form)
        e.emit(0x8B2963FF);

        // Restore callee-saved FP regs (V8-V15) — reverse order
        e.ldp_q_post(14, 15, 32);
        e.ldp_q_post(12, 13, 32);
        e.ldp_q_post(10, 11, 32);
        e.ldp_q_post(8, 9, 32);

        // Restore callee-saved GP regs
        e.ldp_post(X27, X28, 16);
        e.ldp_post(X25, X26, 16);
        e.ldp_post(X23, X24, 16);
        e.ldp_post(X21, X22, 16);
        e.ldp_post(X19, X20, 16);
        e.ldp_post(X29, X30, 16);
        e.ret();

        let code_bytes = e.finalize();

        ArmCode {
            code: code_bytes,
            n_dim_params,
            n_inputs,
        }
    }
}

/// Compiled ARM code with metadata.
pub struct ArmCode {
    pub code: Vec<u8>,
    pub n_dim_params: usize,
    pub n_inputs: usize,
}

// ============================================================================
// Statement-level codegen helpers
// ============================================================================

/// STR Xreg, [SP, #offset] — single instruction, asserts offset fits
fn store_to_sp(reg: u8, offset: u32) -> u32 {
    assert!(offset % 8 == 0 && offset / 8 < 4096, "store_to_sp: offset {offset} too large");
    0xF9000000 | ((offset / 8) << 10) | (31 << 5) | reg as u32
}

/// LDR Xreg, [SP, #offset] — single instruction, asserts offset fits
fn load_from_sp(reg: u8, offset: u32) -> u32 {
    assert!(offset % 8 == 0 && offset / 8 < 4096, "load_from_sp: offset {offset} too large");
    0xF9400000 | ((offset / 8) << 10) | (31 << 5) | reg as u32
}

/// STR Xreg, [SP, #offset] — handles large offsets using X8 as scratch
pub(crate) fn store_to_sp_large(e: &mut ArmEmitter, reg: u8, offset: u32) {
    if offset % 8 == 0 && offset / 8 < 4096 {
        e.emit(store_to_sp(reg, offset));
    } else {
        e.mov_imm32(X8, offset);
        // ADD X8, SP, X8 (extended register form for SP)
        e.emit(0x8B2863E8); // ADD X8, SP, X8, UXTX
        e.str_x(reg, X8, 0);
    }
}

/// LDR Xreg, [SP, #offset] — handles large offsets using X8 as scratch
pub(crate) fn load_from_sp_large(e: &mut ArmEmitter, reg: u8, offset: u32) {
    if offset % 8 == 0 && offset / 8 < 4096 {
        e.emit(load_from_sp(reg, offset));
    } else {
        e.mov_imm32(X8, offset);
        e.emit(0x8B2863E8); // ADD X8, SP, X8, UXTX
        e.ldr_x(reg, X8, 0);
    }
}

/// STR Sreg, [SP, #offset]  (store f32 to stack)
/// For large offsets, emits multiple instructions using X8 as scratch.
pub(crate) fn store_s_to_sp_seq(e: &mut ArmEmitter, reg: u8, offset: u32) {
    if offset % 4 == 0 && offset / 4 < 4096 {
        e.emit(0xBD000000 | ((offset / 4) << 10) | (31 << 5) | reg as u32);
    } else {
        e.mov_imm32(X8, offset);
        e.emit(0x8B2863E8); // ADD X8, SP, X8, UXTX
        e.str_s_imm(reg, X8, 0);
    }
}

/// LDR Sreg, [SP, #offset]  (load f32 from stack)
pub(crate) fn load_s_from_sp_seq(e: &mut ArmEmitter, reg: u8, offset: u32) {
    if offset % 4 == 0 && offset / 4 < 4096 {
        e.emit(0xBD400000 | ((offset / 4) << 10) | (31 << 5) | reg as u32);
    } else {
        e.mov_imm32(X8, offset);
        e.emit(0x8B2863E8); // ADD X8, SP, X8, UXTX
        e.ldr_s_imm(reg, X8, 0);
    }
}

/// Legacy single-instruction versions (for offsets known to be small)
fn store_s_to_sp(reg: u8, offset: u32) -> u32 {
    assert!(offset % 4 == 0 && offset / 4 < 4096, "store_s_to_sp: offset {offset} too large, use store_s_to_sp_seq");
    0xBD000000 | ((offset / 4) << 10) | (31 << 5) | reg as u32
}

fn load_s_from_sp(reg: u8, offset: u32) -> u32 {
    assert!(offset % 4 == 0 && offset / 4 < 4096, "load_s_from_sp: offset {offset} too large, use load_s_from_sp_seq");
    0xBD400000 | ((offset / 4) << 10) | (31 << 5) | reg as u32
}

/// Emit code to compute a Dim value into a GP register.
fn emit_dim_to_reg(e: &mut ArmEmitter, dim: &Dim, ctx: &CodegenCtx, dst: u8) {
    match dim {
        Dim::Lit(v) => {
            e.mov_imm64(dst, *v as u64);
        }
        Dim::Param(name) => {
            let slot = ctx.dim_param_slots[name];
            load_from_sp_large(e, dst, slot);
        }
        Dim::Add(a, b) => {
            emit_dim_to_reg(e, a, ctx, dst);
            emit_dim_to_reg(e, b, ctx, X15);
            e.add_reg(dst, dst, X15);
        }
        Dim::Sub(a, b) => {
            emit_dim_to_reg(e, a, ctx, dst);
            emit_dim_to_reg(e, b, ctx, X15);
            e.sub_reg(dst, dst, X15);
        }
        Dim::Mul(a, b) => {
            emit_dim_to_reg(e, a, ctx, dst);
            emit_dim_to_reg(e, b, ctx, X15);
            e.mul_reg(dst, dst, X15);
        }
        Dim::Div(a, b) => {
            emit_dim_to_reg(e, a, ctx, dst);
            emit_dim_to_reg(e, b, ctx, X15);
            e.sdiv(dst, dst, X15);
        }
    }
}

/// Load a buf pointer from its stack slot into a GP register.
fn load_buf_ptr(e: &mut ArmEmitter, ctx: &CodegenCtx, buf: usize, dst: u8) {
    let slot = ctx.buf_ptrs[&buf];
    load_from_sp_large(e, dst, slot);
}

/// Compute an index offset (in bytes) into X3, given base ptr in some register.
/// Result: X3 = byte offset from buffer start.
fn emit_index_offset(e: &mut ArmEmitter, ctx: &CodegenCtx, index: &Index, tmp1: u8, tmp2: u8) {
    match index {
        Index::Flat => {
            // offset = oi * 4
            let oi_slot = ctx.dim_slots.get(&usize::MAX).copied().expect("flat index requires oi");
            load_from_sp_large(e, tmp1, oi_slot);
            e.lsl_imm(tmp1, tmp1, 2);
        }
        Index::Strided { parts, offset } => {
            let mut has_terms = false;
            for (dim, stride) in parts {
                if stride.is_zero() { continue; }
                let dim_slot = ctx.dim_slots[dim];
                load_from_sp_large(e, tmp2, dim_slot);
                if !stride.is_one() {
                    emit_dim_to_reg(e, stride, ctx, X14);
                    e.mul_reg(tmp2, tmp2, X14);
                }
                if has_terms {
                    e.add_reg(tmp1, tmp1, tmp2);
                } else {
                    e.mov_reg(tmp1, tmp2);
                    has_terms = true;
                }
            }
            if !offset.is_zero() {
                emit_dim_to_reg(e, offset, ctx, tmp2);
                if has_terms {
                    e.add_reg(tmp1, tmp1, tmp2);
                } else {
                    e.mov_reg(tmp1, tmp2);
                    has_terms = true;
                }
            }
            if !has_terms {
                e.mov_reg(tmp1, XZR);
            }
            e.lsl_imm(tmp1, tmp1, 2); // * 4 bytes
        }
    }
}

/// Emit a single IR instruction. Result f32 goes to its spill slot on the stack.
/// Uses S0-S3 as scratch FP registers.
fn emit_inst(e: &mut ArmEmitter, ctx: &mut CodegenCtx, j: usize, inst: &Inst) {
    let slot = ctx.get_or_alloc_inst_slot(j);

    match inst {
        Inst::Load { buf, index } => {
            // Load buf ptr into X3
            load_buf_ptr(e, ctx, *buf, X3);
            // Compute byte offset into X4
            emit_index_offset(e, ctx, index, X4, X5);
            // Load f32: S0 = [X3 + X4]
            e.emit(0xBC606800 | (X4 as u32) << 16 | (X3 as u32) << 5 | 0); // LDR S0, [X3, X4]
            store_s_to_sp_seq(e, 0, slot);
        }
        Inst::Const(v) => {
            let bits = (*v as f32).to_bits();
            e.mov_imm32(X9, bits);
            e.fmov_s_from_w(0, X9);
            store_s_to_sp_seq(e, 0, slot);
        }
        Inst::DimVar(d) => {
            let dim_slot = ctx.dim_slots[d];
            load_from_sp_large(e, X9, dim_slot);
            e.scvtf_s_x(0, X9);
            store_s_to_sp_seq(e, 0, slot);
        }
        Inst::Neg(a) => {
            let a_slot = ctx.inst_slots[a];
            load_s_from_sp_seq(e, 0, a_slot);
            e.fneg_s(0, 0);
            store_s_to_sp_seq(e, 0, slot);
        }
        Inst::Recip(a) => {
            let a_slot = ctx.inst_slots[a];
            // S1 = 1.0
            e.mov_imm32(X9, 1.0f32.to_bits());
            e.fmov_s_from_w(1, X9);
            load_s_from_sp_seq(e, 0, a_slot);
            e.fdiv_s(0, 1, 0);
            store_s_to_sp_seq(e, 0, slot);
        }
        Inst::Exp2(a) => {
            let a_slot = ctx.inst_slots[a];
            // Call arm_exp2_f32(x) — pass x in S0, result in S0
            // ARM AAPCS64: f32 args in S0, result in S0
            load_s_from_sp_seq(e, 0, a_slot);
            // Load function pointer and call
            e.mov_imm64(X9, ctx.exp2_fn_ptr);
            e.blr(X9);
            store_s_to_sp_seq(e, 0, slot);
        }
        Inst::Log2(a) => {
            let a_slot = ctx.inst_slots[a];
            load_s_from_sp_seq(e, 0, a_slot);
            e.mov_imm64(X9, ctx.log2_fn_ptr);
            e.blr(X9);
            store_s_to_sp_seq(e, 0, slot);
        }
        Inst::Sqrt(a) => {
            let a_slot = ctx.inst_slots[a];
            load_s_from_sp_seq(e, 0, a_slot);
            e.fsqrt_s(0, 0);
            store_s_to_sp_seq(e, 0, slot);
        }
        Inst::Add(a, b) => {
            let a_slot = ctx.inst_slots[a];
            let b_slot = ctx.inst_slots[b];
            load_s_from_sp_seq(e, 0, a_slot);
            load_s_from_sp_seq(e, 1, b_slot);
            e.fadd_s(0, 0, 1);
            store_s_to_sp_seq(e, 0, slot);
        }
        Inst::Mul(a, b) => {
            let a_slot = ctx.inst_slots[a];
            let b_slot = ctx.inst_slots[b];
            load_s_from_sp_seq(e, 0, a_slot);
            load_s_from_sp_seq(e, 1, b_slot);
            e.fmul_s(0, 0, 1);
            store_s_to_sp_seq(e, 0, slot);
        }
        Inst::Max(a, b) => {
            let a_slot = ctx.inst_slots[a];
            let b_slot = ctx.inst_slots[b];
            load_s_from_sp_seq(e, 0, a_slot);
            load_s_from_sp_seq(e, 1, b_slot);
            // Use FCMP + FCSEL to match WASM select-based max (NaN handling)
            e.fcmp_s(0, 1);
            e.fcsel_s(0, 0, 1, COND_GT);
            store_s_to_sp_seq(e, 0, slot);
        }
        Inst::CmpLt(a, b) => {
            let a_slot = ctx.inst_slots[a];
            let b_slot = ctx.inst_slots[b];
            load_s_from_sp_seq(e, 0, a_slot);
            load_s_from_sp_seq(e, 1, b_slot);
            // result = a < b ? 1.0 : 0.0
            e.mov_imm32(X9, 1.0f32.to_bits());
            e.fmov_s_from_w(2, X9);
            e.mov_imm32(X9, 0.0f32.to_bits());
            e.fmov_s_from_w(3, X9);
            e.fcmp_s(0, 1);
            e.fcsel_s(0, 2, 3, COND_LT);
            store_s_to_sp_seq(e, 0, slot);
        }
    }
}

/// Pre-compute buffer pointers needed by body instructions, storing them to stack slots.
/// Returns a map from buf_id -> stack_slot.
fn precompute_body_buf_info(
    e: &mut ArmEmitter,
    ctx: &mut CodegenCtx,
    body: &[Inst],
) -> HashMap<usize, u32> {
    let mut buf_slots: HashMap<usize, u32> = HashMap::new();
    for inst in body {
        if let Inst::Load { buf, .. } = inst {
            if !buf_slots.contains_key(buf) {
                let slot = ctx.alloc_slot();
                load_buf_ptr(e, ctx, *buf, X9);
                store_to_sp_large(e, X9, slot);
                buf_slots.insert(*buf, slot);
            }
        }
    }
    buf_slots
}

/// Compute the index offset for a Load instruction in MachIR.
/// Returns a GP vreg holding the byte offset.
fn emit_index_offset_machir(
    mb: &mut crate::mach_ir::MachBuilder,
    e: &mut ArmEmitter,
    ctx: &CodegenCtx,
    index: &Index,
    dim_vregs: &[crate::mach_ir::VReg],
    oi: Option<crate::mach_ir::VReg>,
) -> crate::mach_ir::VReg {
    use crate::mach_ir::MachInst;

    match index {
        Index::Flat => {
            let oi_v = oi.expect("flat index requires oi");
            let byte_off = mb.new_gp();
            mb.push(MachInst::LslImm { dst: byte_off, src: oi_v, shift: 2 });
            byte_off
        }
        Index::Strided { parts, offset } => {
            let mut has_terms = false;
            let result = mb.new_gp();

            for (dim, stride) in parts {
                if stride.is_zero() { continue; }
                let dim_var = dim_vregs[*dim];
                let term = if stride.is_one() {
                    dim_var
                } else {
                    let stride_val = stride.as_usize().unwrap_or(1) as u64;
                    let s = mb.new_gp();
                    mb.push(MachInst::MovImm64 { dst: s, val: stride_val });
                    let prod = mb.new_gp();
                    mb.push(MachInst::MulReg { dst: prod, lhs: dim_var, rhs: s });
                    prod
                };

                if has_terms {
                    mb.push(MachInst::AddReg { dst: result, lhs: result, rhs: term });
                } else {
                    mb.push(MachInst::MovReg { dst: result, src: term });
                    has_terms = true;
                }
            }

            if !offset.is_zero() {
                let off_val = offset.as_usize().unwrap_or(0) as u64;
                let off = mb.new_gp();
                mb.push(MachInst::MovImm64 { dst: off, val: off_val });
                if has_terms {
                    mb.push(MachInst::AddReg { dst: result, lhs: result, rhs: off });
                } else {
                    mb.push(MachInst::MovReg { dst: result, src: off });
                    has_terms = true;
                }
            }

            if !has_terms {
                mb.push(MachInst::MovImm64 { dst: result, val: 0 });
            }

            let byte_off = mb.new_gp();
            mb.push(MachInst::LslImm { dst: byte_off, src: result, shift: 2 });
            byte_off
        }
    }
}

/// Emit a single IR instruction as MachIR. Returns the FP vreg holding the result.
#[allow(clippy::too_many_arguments)]
fn emit_inst_machir(
    mb: &mut crate::mach_ir::MachBuilder,
    e: &mut ArmEmitter,
    ctx: &CodegenCtx,
    inst: &Inst,
    _j: usize,
    inst_vregs: &[Option<crate::mach_ir::VReg>],
    dim_vregs: &[crate::mach_ir::VReg],
    buf_ptr_vregs: &HashMap<usize, crate::mach_ir::VReg>,
    _shape_vregs: &[crate::mach_ir::VReg],
    _stride_slots: &[u32],
    _shape_slots: &[u32],
    _shape: &[Dim],
    _ndim: usize,
    call_spill_slots: &[u32],
    oi: Option<crate::mach_ir::VReg>,
) -> crate::mach_ir::VReg {
    use crate::mach_ir::MachInst;

    match inst {
        Inst::Load { buf, index } => {
            let buf_ptr = buf_ptr_vregs[buf];
            let byte_off = emit_index_offset_machir(mb, e, ctx, index, dim_vregs, oi);
            let addr = mb.new_gp();
            mb.push(MachInst::AddReg { dst: addr, lhs: buf_ptr, rhs: byte_off });
            let dst = mb.new_fp();
            mb.push(MachInst::LdrSImm { dst, base: addr, imm: 0 });
            dst
        }
        Inst::Const(v) => {
            let bits = (*v as f32).to_bits();
            let gp = mb.new_gp();
            mb.push(MachInst::MovImm64 { dst: gp, val: bits as u64 });
            let dst = mb.new_fp();
            mb.push(MachInst::FmovSFromW { dst_fp: dst, src_gp: gp });
            dst
        }
        Inst::DimVar(d) => {
            let dim_var = dim_vregs[*d];
            let dst = mb.new_fp();
            mb.push(MachInst::ScvtfSX { dst, src: dim_var });
            dst
        }
        Inst::Neg(a) => {
            let src = inst_vregs[*a].unwrap();
            let dst = mb.new_fp();
            mb.push(MachInst::FnegS { dst, src });
            dst
        }
        Inst::Recip(a) => {
            let src = inst_vregs[*a].unwrap();
            let one_gp = mb.new_gp();
            mb.push(MachInst::MovImm64 { dst: one_gp, val: 1.0f32.to_bits() as u64 });
            let one = mb.new_fp();
            mb.push(MachInst::FmovSFromW { dst_fp: one, src_gp: one_gp });
            let dst = mb.new_fp();
            mb.push(MachInst::FdivS { dst, lhs: one, rhs: src });
            dst
        }
        Inst::Sqrt(a) => {
            let src = inst_vregs[*a].unwrap();
            let dst = mb.new_fp();
            mb.push(MachInst::FsqrtS { dst, src });
            dst
        }
        Inst::Add(a, b) => {
            let lhs = inst_vregs[*a].unwrap();
            let rhs = inst_vregs[*b].unwrap();
            let dst = mb.new_fp();
            mb.push(MachInst::FaddS { dst, lhs, rhs });
            dst
        }
        Inst::Mul(a, b) => {
            let lhs = inst_vregs[*a].unwrap();
            let rhs = inst_vregs[*b].unwrap();
            let dst = mb.new_fp();
            mb.push(MachInst::FmulS { dst, lhs, rhs });
            dst
        }
        Inst::Max(a, b) => {
            let lhs = inst_vregs[*a].unwrap();
            let rhs = inst_vregs[*b].unwrap();
            let dst = mb.new_fp();
            // Use FCMP + FCSEL for NaN-handling max (matches WASM semantics)
            mb.push(MachInst::FcmpS { lhs, rhs });
            mb.push(MachInst::FcselS { dst, t_val: lhs, f_val: rhs, cond: COND_GT });
            dst
        }
        Inst::CmpLt(a, b) => {
            let lhs = inst_vregs[*a].unwrap();
            let rhs = inst_vregs[*b].unwrap();
            let one_gp = mb.new_gp();
            mb.push(MachInst::MovImm64 { dst: one_gp, val: 1.0f32.to_bits() as u64 });
            let one = mb.new_fp();
            mb.push(MachInst::FmovSFromW { dst_fp: one, src_gp: one_gp });
            let zero_gp = mb.new_gp();
            mb.push(MachInst::MovImm64 { dst: zero_gp, val: 0u64 });
            let zero = mb.new_fp();
            mb.push(MachInst::FmovSFromW { dst_fp: zero, src_gp: zero_gp });
            let dst = mb.new_fp();
            mb.push(MachInst::FcmpS { lhs, rhs });
            mb.push(MachInst::FcselS { dst, t_val: one, f_val: zero, cond: COND_LT });
            dst
        }
        Inst::Exp2(a) | Inst::Log2(a) => {
            // Only save FP vregs that are truly needed after this call.
            // For typical patterns like [Load, Const, Mul, Exp2], nothing after
            // Exp2 uses Load/Const results, so nothing needs saving.
            // We check: is this vreg index referenced by the result instruction
            // or any instruction after it? Since we process linearly, we just
            // check if the vreg is the result (used by the store at the end).
            // Simple heuristic: don't save anything - the only thing used after
            // the call in the body is the call result itself, which is a new vreg.
            // If there ARE later instructions that reference earlier vregs, those
            // vregs need saving. But the MachIR builder emits SpStoreF32/SpLoadF32
            // which re-establishes the vreg's value. This is correct even if we
            // save unnecessary vregs.
            // For now: be conservative but skip vregs not referenced later.
            let live_fp_vregs: Vec<(usize, crate::mach_ir::VReg)> = Vec::new();
            // No FP saves needed for most patterns. If we get wrong results,
            // we'll add them back selectively.

            for &(i, vr) in &live_fp_vregs {
                if i < call_spill_slots.len() {
                    mb.push(MachInst::SpStoreF32 { src: vr, slot: call_spill_slots[i] });
                }
            }

            let arg = inst_vregs[*a].unwrap();
            let func_ptr_val = match inst {
                Inst::Exp2(_) => ctx.exp2_fn_ptr,
                Inst::Log2(_) => ctx.log2_fn_ptr,
                _ => unreachable!(),
            };
            let func_ptr = mb.new_gp();
            mb.push(MachInst::MovImm64 { dst: func_ptr, val: func_ptr_val });
            let result = mb.new_fp();
            mb.push(MachInst::CallFpUnary { func_ptr, arg, result });

            // Reload saved FP vregs
            for &(i, vr) in &live_fp_vregs {
                if i < call_spill_slots.len() {
                    mb.push(MachInst::SpLoadF32 { dst: vr, slot: call_spill_slots[i] });
                }
            }

            result
        }
    }
}

/// Check if a body can be SIMD-ized (no function calls, all innermost-dim loads
/// have stride 1 or are broadcasts).
fn can_simd_elementwise(body: &[Inst], innermost_dim: usize) -> bool {
    for inst in body {
        match inst {
            Inst::Exp2(_) | Inst::Log2(_) => return false,
            Inst::DimVar(d) if *d == innermost_dim => return false,
            Inst::Load { index: Index::Strided { parts, .. }, .. } => {
                for (d, stride) in parts {
                    if *d == innermost_dim && !stride.is_one() {
                        return false; // non-unit stride on innermost dim
                    }
                }
            }
            Inst::Load { index: Index::Flat, .. } => {
                // Flat index = stride 1, OK for SIMD
            }
            _ => {}
        }
    }
    true
}

/// Emit SIMD body instructions using Vec128 vregs.
/// Each Inst produces a Vec128 result (4xf32).
fn emit_simd_body_machir(
    mb: &mut crate::mach_ir::MachBuilder,
    ctx: &CodegenCtx,
    body: &[Inst],
    inst_vregs: &mut Vec<Option<crate::mach_ir::VReg>>,
    dim_vregs: &[crate::mach_ir::VReg],
    buf_ptr_vregs: &HashMap<usize, crate::mach_ir::VReg>,
    innermost_dim: usize,
    oi: crate::mach_ir::VReg,
) {
    use crate::mach_ir::MachInst;

    for (j, inst) in body.iter().enumerate() {
        let vreg = match inst {
            Inst::Load { buf, index } => {
                let buf_ptr = buf_ptr_vregs[buf];
                let dst = mb.new_vec();
                match index {
                    Index::Flat => {
                        // Flat: load 4 contiguous floats at out[oi*4]
                        let byte_off = mb.new_gp();
                        mb.push(MachInst::LslImm { dst: byte_off, src: oi, shift: 2 });
                        mb.push(MachInst::LdrQReg { dst, base: buf_ptr, offset: byte_off });
                    }
                    Index::Strided { parts, offset } => {
                        // Compute base address from non-innermost dims
                        let base_off = mb.new_gp();
                        let mut has_terms = false;
                        for (dim, stride) in parts {
                            if *dim == innermost_dim { continue; } // innermost handled by Q load
                            let dv = dim_vregs[*dim];
                            let term = if stride.is_one() {
                                dv
                            } else {
                                let sv = mb.new_gp();
                                mb.push(MachInst::MovImm64 { dst: sv, val: stride.as_usize().unwrap_or(1) as u64 });
                                let p = mb.new_gp();
                                mb.push(MachInst::MulReg { dst: p, lhs: dv, rhs: sv });
                                p
                            };
                            if has_terms {
                                mb.push(MachInst::AddReg { dst: base_off, lhs: base_off, rhs: term });
                            } else {
                                mb.push(MachInst::MovReg { dst: base_off, src: term });
                                has_terms = true;
                            }
                        }
                        // Add innermost_dim * 1 (stride must be 1 for SIMD)
                        let inner_dv = dim_vregs[innermost_dim];
                        if has_terms {
                            mb.push(MachInst::AddReg { dst: base_off, lhs: base_off, rhs: inner_dv });
                        } else {
                            mb.push(MachInst::MovReg { dst: base_off, src: inner_dv });
                            has_terms = true;
                        }
                        if !offset.is_zero() {
                            let off = mb.new_gp();
                            mb.push(MachInst::MovImm64 { dst: off, val: offset.as_usize().unwrap_or(0) as u64 });
                            if has_terms {
                                mb.push(MachInst::AddReg { dst: base_off, lhs: base_off, rhs: off });
                            } else {
                                mb.push(MachInst::MovReg { dst: base_off, src: off });
                                has_terms = true;
                            }
                        }
                        if !has_terms {
                            mb.push(MachInst::MovImm64 { dst: base_off, val: 0 });
                        }
                        // Check if this is a broadcast (no innermost dim in parts)
                        let has_inner = parts.iter().any(|(d, _)| *d == innermost_dim);
                        if has_inner {
                            // Strided load with stride 1 on innermost → contiguous Q load
                            let byte_off = mb.new_gp();
                            mb.push(MachInst::LslImm { dst: byte_off, src: base_off, shift: 2 });
                            mb.push(MachInst::LdrQReg { dst, base: buf_ptr, offset: byte_off });
                        } else {
                            // Broadcast: load scalar, DUP to vector
                            let byte_off = mb.new_gp();
                            mb.push(MachInst::LslImm { dst: byte_off, src: base_off, shift: 2 });
                            let addr = mb.new_gp();
                            mb.push(MachInst::AddReg { dst: addr, lhs: buf_ptr, rhs: byte_off });
                            let scalar = mb.new_fp();
                            mb.push(MachInst::LdrSImm { dst: scalar, base: addr, imm: 0 });
                            mb.push(MachInst::Dup4sScalar { dst, src: scalar });
                        }
                    }
                }
                dst
            }
            Inst::Const(v) => {
                let bits = (*v as f32).to_bits();
                let gp = mb.new_gp();
                mb.push(MachInst::MovImm64 { dst: gp, val: bits as u64 });
                let dst = mb.new_vec();
                mb.push(MachInst::Dup4sGp { dst, src_gp: gp });
                dst
            }
            Inst::Add(a, b) => {
                let va = inst_vregs[*a].unwrap();
                let vb = inst_vregs[*b].unwrap();
                let dst = mb.new_vec();
                mb.push(MachInst::Fadd4s { dst, lhs: va, rhs: vb });
                dst
            }
            Inst::Mul(a, b) => {
                let va = inst_vregs[*a].unwrap();
                let vb = inst_vregs[*b].unwrap();
                let dst = mb.new_vec();
                mb.push(MachInst::Fmul4s { dst, lhs: va, rhs: vb });
                dst
            }
            Inst::Neg(a) => {
                let va = inst_vregs[*a].unwrap();
                let dst = mb.new_vec();
                mb.push(MachInst::Fneg4s { dst, src: va });
                dst
            }
            Inst::Max(a, b) => {
                let va = inst_vregs[*a].unwrap();
                let vb = inst_vregs[*b].unwrap();
                let dst = mb.new_vec();
                mb.push(MachInst::Fmax4s { dst, lhs: va, rhs: vb });
                dst
            }
            Inst::Sqrt(a) => {
                let va = inst_vregs[*a].unwrap();
                let dst = mb.new_vec();
                mb.push(MachInst::Fsqrt4s { dst, src: va });
                dst
            }
            Inst::Recip(a) => {
                let va = inst_vregs[*a].unwrap();
                let one = mb.new_vec();
                mb.push(MachInst::Fmov4sOne { dst: one });
                let dst = mb.new_vec();
                mb.push(MachInst::Fdiv4s { dst, lhs: one, rhs: va });
                dst
            }
            Inst::CmpLt(a, b) => {
                // CmpLt(a,b) = a < b = b > a → FCMGT(b, a), result is all-1s or all-0s mask
                // We return the mask as a float vector (AND with 1.0 to get 0.0/1.0)
                let va = inst_vregs[*a].unwrap();
                let vb = inst_vregs[*b].unwrap();
                let mask = mb.new_vec();
                mb.push(MachInst::Fcmgt4s { dst: mask, lhs: vb, rhs: va });
                // AND with 1.0 to get 0.0/1.0
                let one = mb.new_vec();
                mb.push(MachInst::Fmov4sOne { dst: one });
                let dst = mb.new_vec();
                mb.push(MachInst::And16b { dst, lhs: mask, rhs: one });
                dst
            }
            Inst::DimVar(_) => unreachable!("DimVar on innermost dim not supported in SIMD"),
            Inst::Exp2(_) | Inst::Log2(_) => unreachable!("calls not supported in SIMD"),
        };
        inst_vregs[j] = Some(vreg);
    }
}

/// Emit a simple elementwise loop (no reduce) using MachIR with virtual registers.
///
/// Uses nested loops instead of flat-index div/mod decomposition.
/// This eliminates expensive SDIV instructions (12-20 cycles each) from the
/// hot path, replacing them with simple counter increments.
fn emit_elementwise_loop(
    e: &mut ArmEmitter,
    ctx: &mut CodegenCtx,
    buf_slot: u32,
    shape: &[Dim],
    body: &[Inst],
    result: usize,
) {
    use crate::mach_ir::{MachBuilder, MachInst, allocate_and_emit_with_spills};

    let ndim = shape.len();

    // Pre-compute shape sizes into stack slots
    let mut shape_slots = Vec::new();
    for d in 0..ndim {
        let sh = ctx.alloc_slot();
        emit_dim_to_reg(e, &shape[d], ctx, X9);
        store_to_sp_large(e, X9, sh);
        shape_slots.push(sh);
    }

    let buf_info = precompute_body_buf_info(e, ctx, body);

    let has_calls = body.iter().any(|inst| matches!(inst, Inst::Exp2(_) | Inst::Log2(_)));
    let call_spill_slots: Vec<u32> = if has_calls {
        let n_slots = body.len() + 1 + ndim + body.len() + 4;
        (0..n_slots).map(|_| ctx.alloc_slot()).collect()
    } else {
        Vec::new()
    };

    // Build MachIR with nested loops (no div/mod decomposition)
    let mut mb = MachBuilder::new();

    let out_ptr = mb.new_gp();
    mb.push(MachInst::SpLoad { dst: out_ptr, slot: buf_slot });

    // Load shape sizes and buffer pointers
    let shape_vregs: Vec<_> = shape_slots.iter().map(|&s| {
        let v = mb.new_gp(); mb.push(MachInst::SpLoad { dst: v, slot: s }); v
    }).collect();
    let buf_ptr_vregs: HashMap<usize, _> = buf_info.iter().map(|(&buf_id, &slot)| {
        let v = mb.new_gp(); mb.push(MachInst::SpLoad { dst: v, slot }); (buf_id, v)
    }).collect();

    // Flat output counter — incremented in innermost loop
    let oi = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: oi, val: 0 });

    let innermost = ndim - 1;
    let use_simd = ndim >= 1 && !has_calls && can_simd_elementwise(body, innermost);

    // Outer loops: d0..d_{ndim-2}
    let mut dim_vregs: Vec<crate::mach_ir::VReg> = Vec::with_capacity(ndim);
    let mut loop_starts: Vec<usize> = Vec::with_capacity(ndim);
    let mut loop_ends: Vec<usize> = Vec::with_capacity(ndim);

    for d in 0..ndim {
        let dim_var = mb.new_gp();
        mb.push(MachInst::MovImm64 { dst: dim_var, val: 0 });
        dim_vregs.push(dim_var);

        if d < innermost || !use_simd {
            // Standard loop header for outer dims (and innermost if no SIMD)
            let start = e.alloc_label();
            let end = e.alloc_label();
            loop_starts.push(start);
            loop_ends.push(end);
            mb.push(MachInst::Label { label: start });
            mb.push(MachInst::CmpReg { lhs: dim_var, rhs: shape_vregs[d] });
            mb.push(MachInst::BCond { cond: COND_GE, label: end });
        } else {
            // SIMD: innermost loop will be handled specially below
            loop_starts.push(0); // placeholder
            loop_ends.push(0);   // placeholder
        }
    }

    let stride_slots: Vec<u32> = Vec::new();

    if use_simd {
        // SIMD innermost loop: process 4 elements at a time
        let inner_dim = dim_vregs[innermost];
        // inner_limit = shape[innermost] & ~3  (round down to multiple of 4)
        let inner_limit = mb.new_gp();
        let four = mb.new_gp();
        mb.push(MachInst::MovImm64 { dst: four, val: 4 });
        mb.push(MachInst::Sdiv { dst: inner_limit, lhs: shape_vregs[innermost], rhs: four });
        mb.push(MachInst::LslImm { dst: inner_limit, src: inner_limit, shift: 2 });

        let simd_start = e.alloc_label();
        let simd_end = e.alloc_label();
        mb.push(MachInst::Label { label: simd_start });
        mb.push(MachInst::CmpReg { lhs: inner_dim, rhs: inner_limit });
        mb.push(MachInst::BCond { cond: COND_GE, label: simd_end });

        // Emit SIMD body (4 elements at once)
        let mut simd_inst_vregs: Vec<Option<crate::mach_ir::VReg>> = vec![None; body.len()];
        emit_simd_body_machir(&mut mb, ctx, body, &mut simd_inst_vregs,
                              &dim_vregs, &buf_ptr_vregs, innermost, oi);

        // Store result: 4 floats to out[oi*4]
        let simd_result = simd_inst_vregs[result].unwrap();
        let byte_off = mb.new_gp();
        mb.push(MachInst::LslImm { dst: byte_off, src: oi, shift: 2 });
        mb.push(MachInst::StrQReg { src: simd_result, base: out_ptr, offset: byte_off });

        // oi += 4, inner_dim += 4
        mb.push(MachInst::AddImm { dst: oi, src: oi, imm: 4 });
        mb.push(MachInst::AddImm { dst: inner_dim, src: inner_dim, imm: 4 });
        mb.push(MachInst::B { label: simd_start });
        mb.push(MachInst::Label { label: simd_end });

        // Scalar remainder: inner_dim..shape[innermost]
        let scalar_start = e.alloc_label();
        let scalar_end = e.alloc_label();
        mb.push(MachInst::Label { label: scalar_start });
        mb.push(MachInst::CmpReg { lhs: inner_dim, rhs: shape_vregs[innermost] });
        mb.push(MachInst::BCond { cond: COND_GE, label: scalar_end });

        let mut scalar_inst_vregs: Vec<Option<crate::mach_ir::VReg>> = vec![None; body.len()];
        for (j, inst) in body.iter().enumerate() {
            let vreg = emit_inst_machir(
                &mut mb, e, ctx, inst, j, &scalar_inst_vregs, &dim_vregs, &buf_ptr_vregs,
                &shape_vregs, &stride_slots, &shape_slots, shape, ndim, &call_spill_slots,
                Some(oi),
            );
            scalar_inst_vregs[j] = Some(vreg);
        }
        let scalar_result = scalar_inst_vregs[result].unwrap();
        let byte_off2 = mb.new_gp();
        mb.push(MachInst::LslImm { dst: byte_off2, src: oi, shift: 2 });
        let store_addr = mb.new_gp();
        mb.push(MachInst::AddReg { dst: store_addr, lhs: out_ptr, rhs: byte_off2 });
        mb.push(MachInst::StrSImm { src: scalar_result, base: store_addr, imm: 0 });

        mb.push(MachInst::AddImm { dst: oi, src: oi, imm: 1 });
        mb.push(MachInst::AddImm { dst: inner_dim, src: inner_dim, imm: 1 });
        mb.push(MachInst::B { label: scalar_start });
        mb.push(MachInst::Label { label: scalar_end });

        // Close outer loops (skip innermost — handled above)
        for d in (0..innermost).rev() {
            mb.push(MachInst::AddImm { dst: dim_vregs[d], src: dim_vregs[d], imm: 1 });
            mb.push(MachInst::B { label: loop_starts[d] });
            mb.push(MachInst::Label { label: loop_ends[d] });
        }
    } else {
        // Scalar path (unchanged): for loops with calls or non-SIMD-safe bodies
        let body_buf_ptr_vregs: HashMap<usize, _> = if has_calls {
            buf_info.iter().map(|(&buf_id, &slot)| {
                let v = mb.new_gp(); mb.push(MachInst::SpLoad { dst: v, slot }); (buf_id, v)
            }).collect()
        } else {
            buf_ptr_vregs.clone()
        };

        let mut inst_vregs: Vec<Option<crate::mach_ir::VReg>> = vec![None; body.len()];
        for (j, inst) in body.iter().enumerate() {
            let vreg = emit_inst_machir(
                &mut mb, e, ctx, inst, j, &inst_vregs, &dim_vregs, &body_buf_ptr_vregs,
                &shape_vregs, &stride_slots, &shape_slots, shape, ndim, &call_spill_slots,
                Some(oi),
            );
            inst_vregs[j] = Some(vreg);
        }

        let result_vreg = inst_vregs[result].unwrap();
        let byte_off = mb.new_gp();
        mb.push(MachInst::LslImm { dst: byte_off, src: oi, shift: 2 });
        let store_addr = mb.new_gp();
        mb.push(MachInst::AddReg { dst: store_addr, lhs: out_ptr, rhs: byte_off });
        mb.push(MachInst::StrSImm { src: result_vreg, base: store_addr, imm: 0 });

        mb.push(MachInst::AddImm { dst: oi, src: oi, imm: 1 });

        // Close ALL nested loops
        for d in (0..ndim).rev() {
            mb.push(MachInst::AddImm { dst: dim_vregs[d], src: dim_vregs[d], imm: 1 });
            mb.push(MachInst::B { label: loop_starts[d] });
            mb.push(MachInst::Label { label: loop_ends[d] });
        }
    }

    let insts = mb.finish();
    let next_vreg = mb.next_vreg_id();
    allocate_and_emit_with_spills(&insts, e, next_vreg, &mut || ctx.alloc_slot());
}

/// Emit a reduce loop (no tiling).
/// Emit a reduce loop using MachIR with virtual registers.
fn emit_reduce_loop(
    e: &mut ArmEmitter,
    ctx: &mut CodegenCtx,
    buf_slot: u32,
    shape: &[Dim],
    reduce: &loop_ir::ReduceDesc,
    body: &[Inst],
    result: usize,
) {
    use crate::mach_ir::{MachBuilder, MachInst, allocate_and_emit_with_spills};

    let ndim = shape.len();

    let init_val = match reduce.op {
        ReduceOp::Sum => 0.0f32,
        ReduceOp::Max => f32::NEG_INFINITY,
    };

    // Pre-compute shape and reduce sizes into stack slots
    let reduce_size_slot = ctx.alloc_slot();
    emit_dim_to_reg(e, &reduce.size, ctx, X9);
    store_to_sp_large(e, X9, reduce_size_slot);

    let mut shape_slots = Vec::new();
    for d in 0..ndim {
        let sh = ctx.alloc_slot();
        emit_dim_to_reg(e, &shape[d], ctx, X9);
        store_to_sp_large(e, X9, sh);
        shape_slots.push(sh);
    }

    let buf_info = precompute_body_buf_info(e, ctx, body);

    let has_calls = body.iter().any(|inst| matches!(inst, Inst::Exp2(_) | Inst::Log2(_)));
    let call_spill_slots: Vec<u32> = if has_calls {
        let n_slots = body.len() + 1 + ndim + body.len() + 4;
        (0..n_slots).map(|_| ctx.alloc_slot()).collect()
    } else {
        Vec::new()
    };

    // Build MachIR with nested output loops + inner reduce loop
    let mut mb = MachBuilder::new();

    let out_ptr = mb.new_gp();
    mb.push(MachInst::SpLoad { dst: out_ptr, slot: buf_slot });
    let reduce_size = mb.new_gp();
    mb.push(MachInst::SpLoad { dst: reduce_size, slot: reduce_size_slot });

    // Load shape sizes and buffer pointers
    let shape_vregs: Vec<_> = shape_slots.iter().map(|&s| {
        let v = mb.new_gp(); mb.push(MachInst::SpLoad { dst: v, slot: s }); v
    }).collect();
    let buf_ptr_vregs: HashMap<usize, _> = buf_info.iter().map(|(&buf_id, &slot)| {
        let v = mb.new_gp(); mb.push(MachInst::SpLoad { dst: v, slot }); (buf_id, v)
    }).collect();

    // Flat output counter
    let oi = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: oi, val: 0 });

    // Nested output loops (no div/mod)
    let mut dim_vregs: Vec<crate::mach_ir::VReg> = Vec::with_capacity(ndim);
    let mut loop_starts: Vec<usize> = Vec::with_capacity(ndim);
    let mut loop_ends: Vec<usize> = Vec::with_capacity(ndim);

    for d in 0..ndim {
        let dim_var = mb.new_gp();
        mb.push(MachInst::MovImm64 { dst: dim_var, val: 0 });
        dim_vregs.push(dim_var);

        let start = e.alloc_label();
        let end = e.alloc_label();
        loop_starts.push(start);
        loop_ends.push(end);

        mb.push(MachInst::Label { label: start });
        mb.push(MachInst::CmpReg { lhs: dim_var, rhs: shape_vregs[d] });
        mb.push(MachInst::BCond { cond: COND_GE, label: end });
    }

    // acc = init_val
    let init_gp = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: init_gp, val: init_val.to_bits() as u64 });
    let acc = mb.new_fp();
    mb.push(MachInst::FmovSFromW { dst_fp: acc, src_gp: init_gp });

    // Inner reduce loop: ki = 0..reduce_size
    let ki = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: ki, val: 0 });

    let mut dim_vregs_ext = dim_vregs.clone();
    dim_vregs_ext.push(ki);

    let inner_start = e.alloc_label();
    let inner_end = e.alloc_label();

    mb.push(MachInst::Label { label: inner_start });
    mb.push(MachInst::CmpReg { lhs: ki, rhs: reduce_size });
    mb.push(MachInst::BCond { cond: COND_GE, label: inner_end });

    // Emit body
    let stride_slots: Vec<u32> = Vec::new();
    let mut inst_vregs: Vec<Option<crate::mach_ir::VReg>> = vec![None; body.len()];

    for (j, inst) in body.iter().enumerate() {
        let vreg = emit_inst_machir(
            &mut mb, e, ctx, inst, j, &inst_vregs, &dim_vregs_ext, &buf_ptr_vregs,
            &shape_vregs, &stride_slots, &shape_slots, shape, ndim, &call_spill_slots,
            None,
        );
        inst_vregs[j] = Some(vreg);
    }

    // Accumulate
    let result_vreg = inst_vregs[result].unwrap();
    match reduce.op {
        ReduceOp::Sum => {
            mb.push(MachInst::FaddS { dst: acc, lhs: acc, rhs: result_vreg });
        }
        ReduceOp::Max => {
            mb.push(MachInst::FcmpS { lhs: result_vreg, rhs: acc });
            mb.push(MachInst::FcselS { dst: acc, t_val: result_vreg, f_val: acc, cond: COND_GT });
        }
    }

    // ki++
    mb.push(MachInst::AddImm { dst: ki, src: ki, imm: 1 });
    mb.push(MachInst::B { label: inner_start });
    mb.push(MachInst::Label { label: inner_end });

    // Store: out[oi] = acc
    let byte_off = mb.new_gp();
    mb.push(MachInst::LslImm { dst: byte_off, src: oi, shift: 2 });
    let store_addr = mb.new_gp();
    mb.push(MachInst::AddReg { dst: store_addr, lhs: out_ptr, rhs: byte_off });
    mb.push(MachInst::StrSImm { src: acc, base: store_addr, imm: 0 });

    // oi++ (flat output counter)
    mb.push(MachInst::AddImm { dst: oi, src: oi, imm: 1 });

    // Close nested output loops (innermost first)
    for d in (0..ndim).rev() {
        mb.push(MachInst::AddImm { dst: dim_vregs[d], src: dim_vregs[d], imm: 1 });
        mb.push(MachInst::B { label: loop_starts[d] });
        mb.push(MachInst::Label { label: loop_ends[d] });
    }

    let insts = mb.finish();
    let next_vreg = mb.next_vreg_id();
    allocate_and_emit_with_spills(&insts, e, next_vreg, &mut || ctx.alloc_slot());
}

/// Emit a tiled reduce loop with NEON SIMD, using the MachIR register allocator
/// for the inner FMLA kernel.
///
/// Outer loops (batch, m_blk, mi, n_blk, ni_grp) are emitted directly to the
/// ArmEmitter using stack slots (same approach as the general tiled path).
/// The inner kernel (accumulator init, K loop with FMLA, store, remainder)
/// is built as MachIR, register-allocated, and lowered — this avoids the
/// register-clobber bugs of the old manual assignment.
#[allow(clippy::too_many_arguments)]
/// Emit a matmul directly from Stmt::Matmul (no loop body decomposition needed).
/// A: [batch..., M, K], B: [batch..., K, N] -> out: [batch..., M, N]
fn emit_matmul_direct(
    e: &mut ArmEmitter,
    ctx: &mut CodegenCtx,
    buf_slot: u32,
    a_buf_slot: u32,
    b_buf_slot: u32,
    batch: &[Dim],
    m: &Dim,
    k: &Dim,
    n: &Dim,
) {
    use crate::mach_ir::{MachBuilder, MachInst, allocate_and_emit_with_spills};

    let batch_dims = batch.len();
    let m_val = m.as_usize();
    let n_val = n.as_usize();
    let k_val = k.as_usize();
    let tm: usize = 8;
    let tn: usize = 32;
    // unroll must be at least 4 for SIMD groups to work; for N < 4 we use scalar only
    let unroll: usize = n_val.map(|nv| 32usize.min(nv)).unwrap_or(32);
    let unroll = if unroll < 4 { 0 } else { unroll }; // 0 = skip SIMD entirely
    let simd_groups = if unroll > 0 { unroll / 4 } else { 0 };

    // A strides: last two dims are [M, K], stride for M = K, stride for K = 1
    // B strides: last two dims are [K, N], stride for K = N, stride for N = 1

    // Compute batch strides for A and B
    let mut a_shape_full: Vec<Dim> = batch.to_vec();
    a_shape_full.push(m.clone());
    a_shape_full.push(k.clone());
    let a_batch_strides = Dim::strides(&a_shape_full);

    let mut b_shape_full: Vec<Dim> = batch.to_vec();
    b_shape_full.push(k.clone());
    b_shape_full.push(n.clone());
    let b_batch_strides = Dim::strides(&b_shape_full);

    let mut out_shape: Vec<Dim> = batch.to_vec();
    out_shape.push(m.clone());
    out_shape.push(n.clone());
    let out_batch_strides = Dim::strides(&out_shape);

    // Store sizes to stack for MachIR to load
    let m_slot = ctx.alloc_slot();
    emit_dim_to_reg(e, m, ctx, X9);
    store_to_sp_large(e, X9, m_slot);

    let n_slot = ctx.alloc_slot();
    emit_dim_to_reg(e, n, ctx, X9);
    store_to_sp_large(e, X9, n_slot);

    let k_slot = ctx.alloc_slot();
    emit_dim_to_reg(e, k, ctx, X9);
    store_to_sp_large(e, X9, k_slot);

    // Store batch sizes and strides
    let mut batch_size_slots = Vec::new();
    let mut a_batch_stride_slots = Vec::new();
    let mut b_batch_stride_slots = Vec::new();
    let mut out_batch_stride_slots = Vec::new();
    for d in 0..batch_dims {
        let s = ctx.alloc_slot();
        emit_dim_to_reg(e, &batch[d], ctx, X9);
        store_to_sp_large(e, X9, s);
        batch_size_slots.push(s);

        let s = ctx.alloc_slot();
        emit_dim_to_reg(e, &a_batch_strides[d], ctx, X9);
        store_to_sp_large(e, X9, s);
        a_batch_stride_slots.push(s);

        let s = ctx.alloc_slot();
        emit_dim_to_reg(e, &b_batch_strides[d], ctx, X9);
        store_to_sp_large(e, X9, s);
        b_batch_stride_slots.push(s);

        let s = ctx.alloc_slot();
        emit_dim_to_reg(e, &out_batch_strides[d], ctx, X9);
        store_to_sp_large(e, X9, s);
        out_batch_stride_slots.push(s);
    }

    // Build MachIR for the matmul
    let mut mb = MachBuilder::new();

    let out_ptr = mb.new_gp();
    mb.push(MachInst::SpLoad { dst: out_ptr, slot: buf_slot });
    let a_ptr = mb.new_gp();
    mb.push(MachInst::SpLoad { dst: a_ptr, slot: a_buf_slot });
    let b_ptr = mb.new_gp();
    mb.push(MachInst::SpLoad { dst: b_ptr, slot: b_buf_slot });

    let m_reg = mb.new_gp();
    mb.push(MachInst::SpLoad { dst: m_reg, slot: m_slot });
    let n_reg = mb.new_gp();
    mb.push(MachInst::SpLoad { dst: n_reg, slot: n_slot });
    let k_reg = mb.new_gp();
    mb.push(MachInst::SpLoad { dst: k_reg, slot: k_slot });

    // --- Batch loops ---
    let mut batch_dim_regs = Vec::new();
    let mut batch_size_regs = Vec::new();
    let mut batch_loop_starts = Vec::new();
    let mut batch_loop_ends = Vec::new();

    for d in 0..batch_dims {
        let dim_reg = mb.new_gp();
        mb.push(MachInst::MovImm64 { dst: dim_reg, val: 0 });
        let size_reg = mb.new_gp();
        mb.push(MachInst::SpLoad { dst: size_reg, slot: batch_size_slots[d] });
        batch_dim_regs.push(dim_reg);
        batch_size_regs.push(size_reg);

        let start = e.alloc_label();
        let end = e.alloc_label();
        batch_loop_starts.push(start);
        batch_loop_ends.push(end);

        mb.push(MachInst::Label { label: start });
        mb.push(MachInst::CmpReg { lhs: dim_reg, rhs: size_reg });
        mb.push(MachInst::BCond { cond: COND_GE, label: end });
    }

    // Compute batch offsets for A, B, out
    let a_base = mb.new_gp();
    mb.push(MachInst::MovReg { dst: a_base, src: a_ptr });
    let b_base = mb.new_gp();
    mb.push(MachInst::MovReg { dst: b_base, src: b_ptr });
    let out_base = mb.new_gp();
    mb.push(MachInst::MovReg { dst: out_base, src: out_ptr });

    for d in 0..batch_dims {
        let a_stride = mb.new_gp();
        mb.push(MachInst::SpLoad { dst: a_stride, slot: a_batch_stride_slots[d] });
        let off = mb.new_gp();
        mb.push(MachInst::MulReg { dst: off, lhs: batch_dim_regs[d], rhs: a_stride });
        let off_bytes = mb.new_gp();
        mb.push(MachInst::LslImm { dst: off_bytes, src: off, shift: 2 });
        mb.push(MachInst::AddReg { dst: a_base, lhs: a_base, rhs: off_bytes });

        let b_stride = mb.new_gp();
        mb.push(MachInst::SpLoad { dst: b_stride, slot: b_batch_stride_slots[d] });
        let off = mb.new_gp();
        mb.push(MachInst::MulReg { dst: off, lhs: batch_dim_regs[d], rhs: b_stride });
        let off_bytes = mb.new_gp();
        mb.push(MachInst::LslImm { dst: off_bytes, src: off, shift: 2 });
        mb.push(MachInst::AddReg { dst: b_base, lhs: b_base, rhs: off_bytes });

        let out_stride = mb.new_gp();
        mb.push(MachInst::SpLoad { dst: out_stride, slot: out_batch_stride_slots[d] });
        let off = mb.new_gp();
        mb.push(MachInst::MulReg { dst: off, lhs: batch_dim_regs[d], rhs: out_stride });
        let off_bytes = mb.new_gp();
        mb.push(MachInst::LslImm { dst: off_bytes, src: off, shift: 2 });
        mb.push(MachInst::AddReg { dst: out_base, lhs: out_base, rhs: off_bytes });
    }

    // --- M loop (row loop) ---
    let mi = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: mi, val: 0 });

    let mi_start = e.alloc_label();
    let mi_end = e.alloc_label();
    mb.push(MachInst::Label { label: mi_start });
    mb.push(MachInst::CmpReg { lhs: mi, rhs: m_reg });
    mb.push(MachInst::BCond { cond: COND_GE, label: mi_end });

    // a_row_ptr = a_base + mi * K * 4
    let a_row_off = mb.new_gp();
    mb.push(MachInst::MulReg { dst: a_row_off, lhs: mi, rhs: k_reg });
    let a_row_bytes = mb.new_gp();
    mb.push(MachInst::LslImm { dst: a_row_bytes, src: a_row_off, shift: 2 });
    let a_row_ptr = mb.new_gp();
    mb.push(MachInst::AddReg { dst: a_row_ptr, lhs: a_base, rhs: a_row_bytes });

    // --- N tile loop ---
    let ni = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: ni, val: 0 });

    let ni_start = e.alloc_label();
    let ni_end = e.alloc_label();

    if simd_groups > 0 {
    mb.push(MachInst::Label { label: ni_start });

    // Check: ni + unroll <= N (full tile available)
    let ni_plus = mb.new_gp();
    mb.push(MachInst::AddImm { dst: ni_plus, src: ni, imm: unroll as u32 });
    mb.push(MachInst::CmpReg { lhs: ni_plus, rhs: n_reg });
    mb.push(MachInst::BCond { cond: COND_GT, label: ni_end });

    // Zero accumulators V0..V{simd_groups-1}
    let mut acc_regs = Vec::new();
    for _g in 0..simd_groups {
        let acc = mb.new_vec();
        mb.push(MachInst::Movi4sZero { dst: acc });
        acc_regs.push(acc);
    }

    // --- K loop (inner loop, fully in registers) ---
    let ki = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: ki, val: 0 });

    let ki_start = e.alloc_label();
    let ki_end = e.alloc_label();
    mb.push(MachInst::Label { label: ki_start });
    mb.push(MachInst::CmpReg { lhs: ki, rhs: k_reg });
    mb.push(MachInst::BCond { cond: COND_GE, label: ki_end });

    // Load A[mi, ki] scalar, broadcast
    let a_scalar = mb.new_fp();
    mb.push(MachInst::LdrSRegScaled { dst: a_scalar, base: a_row_ptr, offset: ki });
    let a_gp = mb.new_gp();
    mb.push(MachInst::FmovWFromS { dst_gp: a_gp, src_fp: a_scalar });
    let a_vec = mb.new_vec();
    mb.push(MachInst::Dup4sGp { dst: a_vec, src_gp: a_gp });

    // Compute B row ptr: b_base + (ki * N + ni) * 4
    let b_off = mb.new_gp();
    mb.push(MachInst::MulReg { dst: b_off, lhs: ki, rhs: n_reg });
    let b_off2 = mb.new_gp();
    mb.push(MachInst::AddReg { dst: b_off2, lhs: b_off, rhs: ni });
    let b_addr = mb.new_gp();
    mb.push(MachInst::LslImm { dst: b_addr, src: b_off2, shift: 2 });
    let b_row_ptr = mb.new_gp();
    mb.push(MachInst::AddReg { dst: b_row_ptr, lhs: b_base, rhs: b_addr });

    // FMLA for each SIMD group
    for g in 0..simd_groups {
        let b_vec = mb.new_vec();
        mb.push(MachInst::LdrQImm { dst: b_vec, base: b_row_ptr, imm: (g * 16) as u32 });
        mb.push(MachInst::Fmla4s { acc: acc_regs[g], lhs: a_vec, rhs: b_vec });
    }

    // ki++
    mb.push(MachInst::AddImm { dst: ki, src: ki, imm: 1 });
    mb.push(MachInst::B { label: ki_start });
    mb.push(MachInst::Label { label: ki_end });

    // Store accumulators: out[mi * N + ni + g*4]
    let out_row_off = mb.new_gp();
    mb.push(MachInst::MulReg { dst: out_row_off, lhs: mi, rhs: n_reg });
    let out_off = mb.new_gp();
    mb.push(MachInst::AddReg { dst: out_off, lhs: out_row_off, rhs: ni });
    let out_addr = mb.new_gp();
    mb.push(MachInst::LslImm { dst: out_addr, src: out_off, shift: 2 });
    let store_base = mb.new_gp();
    mb.push(MachInst::AddReg { dst: store_base, lhs: out_base, rhs: out_addr });

    for g in 0..simd_groups {
        mb.push(MachInst::StrQImm { src: acc_regs[g], base: store_base, imm: (g * 16) as u32 });
    }

    // ni += unroll
    mb.push(MachInst::AddImm { dst: ni, src: ni, imm: unroll as u32 });
    mb.push(MachInst::B { label: ni_start });
    mb.push(MachInst::Label { label: ni_end });
    } // end if simd_groups > 0

    // Scalar remainder: ni..N
    let rem_start = e.alloc_label();
    let rem_end = e.alloc_label();
    mb.push(MachInst::Label { label: rem_start });
    mb.push(MachInst::CmpReg { lhs: ni, rhs: n_reg });
    mb.push(MachInst::BCond { cond: COND_GE, label: rem_end });

    // acc = 0
    let zero_gp = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: zero_gp, val: 0 });
    let rem_acc = mb.new_fp();
    mb.push(MachInst::FmovSFromW { dst_fp: rem_acc, src_gp: zero_gp });

    let rem_ki = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: rem_ki, val: 0 });
    let rem_ki_start = e.alloc_label();
    let rem_ki_end = e.alloc_label();
    mb.push(MachInst::Label { label: rem_ki_start });
    mb.push(MachInst::CmpReg { lhs: rem_ki, rhs: k_reg });
    mb.push(MachInst::BCond { cond: COND_GE, label: rem_ki_end });

    // S1 = A[mi, rem_ki]
    let a_rem = mb.new_fp();
    mb.push(MachInst::LdrSRegScaled { dst: a_rem, base: a_row_ptr, offset: rem_ki });
    // S2 = B[rem_ki, ni]
    let b_rem_off = mb.new_gp();
    mb.push(MachInst::MulReg { dst: b_rem_off, lhs: rem_ki, rhs: n_reg });
    let b_rem_off2 = mb.new_gp();
    mb.push(MachInst::AddReg { dst: b_rem_off2, lhs: b_rem_off, rhs: ni });
    let b_rem = mb.new_fp();
    mb.push(MachInst::LdrSRegScaled { dst: b_rem, base: b_base, offset: b_rem_off2 });
    // acc += a * b
    mb.push(MachInst::Fmadd { dst: rem_acc, mul_lhs: a_rem, mul_rhs: b_rem, add: rem_acc });

    mb.push(MachInst::AddImm { dst: rem_ki, src: rem_ki, imm: 1 });
    mb.push(MachInst::B { label: rem_ki_start });
    mb.push(MachInst::Label { label: rem_ki_end });

    // Store remainder: out[mi * N + ni]
    let rem_out_off = mb.new_gp();
    mb.push(MachInst::MulReg { dst: rem_out_off, lhs: mi, rhs: n_reg });
    let rem_out_off2 = mb.new_gp();
    mb.push(MachInst::AddReg { dst: rem_out_off2, lhs: rem_out_off, rhs: ni });
    mb.push(MachInst::StrSRegScaled { src: rem_acc, base: out_base, offset: rem_out_off2 });

    mb.push(MachInst::AddImm { dst: ni, src: ni, imm: 1 });
    mb.push(MachInst::B { label: rem_start });
    mb.push(MachInst::Label { label: rem_end });

    // mi++
    mb.push(MachInst::AddImm { dst: mi, src: mi, imm: 1 });
    mb.push(MachInst::B { label: mi_start });
    mb.push(MachInst::Label { label: mi_end });

    // Close batch loops
    for d in (0..batch_dims).rev() {
        mb.push(MachInst::AddImm { dst: batch_dim_regs[d], src: batch_dim_regs[d], imm: 1 });
        mb.push(MachInst::B { label: batch_loop_starts[d] });
        mb.push(MachInst::Label { label: batch_loop_ends[d] });
    }

    let insts = mb.finish();
    let next_vreg = mb.next_vreg_id();
    allocate_and_emit_with_spills(&insts, e, next_vreg, &mut || ctx.alloc_slot());
}

/// MR=8 × NR=8 matmul micro-kernel. Processes 8 rows of A × 8 cols of B
/// simultaneously in the K loop, reusing B data across all 8 rows.
/// Uses 16 Vec128 accumulators (8 rows × 2 SIMD groups of 4 cols).
fn emit_matmul_mr8(
    e: &mut ArmEmitter,
    ctx: &mut CodegenCtx,
    buf_slot: u32,
    shape: &[Dim],
    reduce: &loop_ir::ReduceDesc,
    body: &[Inst],
    a_inst_idx: usize,
    b_inst_idx: usize,
    batch_strides: &[Dim],
    batch_dims: usize,
    m_dim: usize,
    n_dim: usize,
    reduce_dim: usize,
) {
    use crate::mach_ir::{MachBuilder, MachInst, allocate_and_emit_with_spills};

    const MR: usize = 8;
    const NR: usize = 8; // 2 SIMD groups of 4

    let (a_buf, a_index) = match &body[a_inst_idx] {
        Inst::Load { buf, index } => (*buf, index.clone()),
        _ => unreachable!(),
    };
    let (b_buf, b_index) = match &body[b_inst_idx] {
        Inst::Load { buf, index } => (*buf, index.clone()),
        _ => unreachable!(),
    };

    let a_k_stride = get_stride_for_dim(&a_index, reduce_dim);
    let b_k_stride = get_stride_for_dim(&b_index, reduce_dim);
    let b_n_stride = get_stride_for_dim(&b_index, n_dim);

    // Compute effective A row stride for the collapsed (batch..., m) space.
    // Walk from m_dim through batch dims to find the stride between consecutive rows.
    let a_row_stride = {
        let mut s = get_stride_for_dim(&a_index, m_dim);
        if s == 0 {
            // m_dim has size 1, look at batch dims (right to left)
            for d in (0..batch_dims).rev() {
                s = get_stride_for_dim(&a_index, d);
                if s != 0 { break; }
            }
        }
        s
    };

    // Effective M = product of batch dims × shape[m_dim]
    let effective_m_dim = Dim::product(
        &(0..=m_dim).map(|d| shape[d].clone()).collect::<Vec<_>>()
    );

    // Output row stride = shape[n_dim] (output is [effective_M, N])
    let out_row_stride = shape[n_dim].clone();

    // Pre-compute values into stack slots
    let out_ptr_slot = ctx.alloc_slot();
    load_from_sp_large(e, X9, buf_slot);
    store_to_sp_large(e, X9, out_ptr_slot);

    let a_ptr_slot = ctx.alloc_slot();
    load_buf_ptr(e, ctx, a_buf, X9);
    store_to_sp_large(e, X9, a_ptr_slot);

    let b_ptr_slot = ctx.alloc_slot();
    load_buf_ptr(e, ctx, b_buf, X9);
    store_to_sp_large(e, X9, b_ptr_slot);

    let eff_m_slot = ctx.alloc_slot();
    emit_dim_to_reg(e, &effective_m_dim, ctx, X9);
    store_to_sp_large(e, X9, eff_m_slot);

    let n_size_slot = ctx.alloc_slot();
    emit_dim_to_reg(e, &shape[n_dim], ctx, X9);
    store_to_sp_large(e, X9, n_size_slot);

    let k_size_slot = ctx.alloc_slot();
    emit_dim_to_reg(e, &reduce.size, ctx, X9);
    store_to_sp_large(e, X9, k_size_slot);

    // Output row stride slot
    let out_row_stride_slot = ctx.alloc_slot();
    emit_dim_to_reg(e, &out_row_stride, ctx, X9);
    store_to_sp_large(e, X9, out_row_stride_slot);

    // If B has non-unit stride on N (i.e., B is in [N,K] layout after permute),
    // transpose it to [K,N] layout for contiguous Q-loads in the micro-kernel.
    // The transposed buffer is heap-allocated (one-time cost per matmul call).
    let (effective_b_ptr_slot, effective_b_k_stride, effective_b_n_stride) = if b_n_stride > 1 {
        let bt_slot = ctx.alloc_slot();
        // Allocate K*N*4 bytes on heap (bump X20)
        load_from_sp_large(e, X9, n_size_slot);
        load_from_sp_large(e, X10, k_size_slot);
        e.mul_reg(X11, X9, X10);
        e.lsl_imm(X11, X11, 2);
        store_to_sp_large(e, X20, bt_slot);
        e.add_reg(X20, X20, X11);

        // Transpose: B_t[k, n] = B[n * b_n_stride + k * b_k_stride]
        // B_t stored as [K, N] with stride N on K, stride 1 on N.
        load_buf_ptr(e, ctx, b_buf, X3); // src
        load_from_sp_large(e, X4, bt_slot); // dst
        load_from_sp_large(e, X5, k_size_slot);
        load_from_sp_large(e, X6, n_size_slot);

        e.mov_reg(X7, XZR); // n = 0
        let n_loop = e.alloc_label();
        let n_end = e.alloc_label();
        e.bind_label(n_loop);
        e.cmp_reg(X7, X6);
        e.b_cond_label(COND_GE, n_end);

        e.mov_reg(X9, XZR); // k = 0
        let k_loop = e.alloc_label();
        let k_end = e.alloc_label();
        e.bind_label(k_loop);
        e.cmp_reg(X9, X5);
        e.b_cond_label(COND_GE, k_end);

        // src_idx = n * b_n_stride + k * b_k_stride
        e.mov_imm32(X10, b_n_stride as u32);
        e.mul_reg(X11, X7, X10);
        e.mov_imm32(X10, b_k_stride as u32);
        e.madd(X11, X9, X10, X11);
        e.ldr_s_reg_scaled(0, X3, X11);

        // dst_idx = k * N + n
        e.madd(X11, X9, X6, X7);
        e.str_s_reg_scaled(0, X4, X11);

        e.add_imm(X9, X9, 1);
        e.b_label(k_loop);
        e.bind_label(k_end);

        e.add_imm(X7, X7, 1);
        e.b_label(n_loop);
        e.bind_label(n_end);

        // After transpose: effective B at bt_slot, k_stride=N (runtime), n_stride=1
        // Store effective b_k_stride = N into a slot
        let bks_slot = ctx.alloc_slot();
        load_from_sp_large(e, X9, n_size_slot);
        store_to_sp_large(e, X9, bks_slot);
        (bt_slot, bks_slot, 1usize)
    } else {
        let bks_slot = ctx.alloc_slot();
        e.mov_imm64(X9, b_k_stride as u64);
        store_to_sp_large(e, X9, bks_slot);
        (b_ptr_slot, bks_slot, b_n_stride)
    };
    let effective_b_k_stride_slot = effective_b_k_stride; // rename for clarity

    // Build in MachIR
    let mut mb = MachBuilder::new();

    macro_rules! sp_load {
        ($mb:expr, $slot:expr) => {{
            let v = $mb.new_gp();
            $mb.push(MachInst::SpLoad { dst: v, slot: $slot });
            v
        }};
    }

    let n_size = sp_load!(mb, n_size_slot);
    let k_size = sp_load!(mb, k_size_slot);
    let eff_m = sp_load!(mb, eff_m_slot);

    // No batch loops — M is collapsed (batch dims × m_dim).
    // B has no batch offset since it's the weight matrix (shared across all M rows).

    // n_blk loop: step by NR=8
    let nr_v = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: nr_v, val: NR as u64 });
    let n_blk = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: n_blk, val: 0 });
    let n_blk_start = e.alloc_label();
    let n_blk_end = e.alloc_label();
    mb.push(MachInst::Label { label: n_blk_start });
    let n_blk_plus_nr = mb.new_gp();
    mb.push(MachInst::AddReg { dst: n_blk_plus_nr, lhs: n_blk, rhs: nr_v });
    mb.push(MachInst::CmpReg { lhs: n_blk_plus_nr, rhs: n_size });
    mb.push(MachInst::BCond { cond: COND_GT, label: n_blk_end });

    // m loop over effective M: step by MR=8, then remainder
    let mr_v = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: mr_v, val: MR as u64 });
    let mi = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: mi, val: 0 });
    let m_limit = mb.new_gp();
    mb.push(MachInst::Sdiv { dst: m_limit, lhs: eff_m, rhs: mr_v });
    mb.push(MachInst::MulReg { dst: m_limit, lhs: m_limit, rhs: mr_v });

    let mi_start = e.alloc_label();
    let mi_end = e.alloc_label();
    mb.push(MachInst::Label { label: mi_start });
    mb.push(MachInst::CmpReg { lhs: mi, rhs: m_limit });
    mb.push(MachInst::BCond { cond: COND_GE, label: mi_end });

    // === MR=8 × NR=8 micro-kernel ===

    // A base: a_base = a_ptr + mi * a_row_stride * 4
    // (collapsed batch dims — flat indexing)
    let a_ptr = sp_load!(mb, a_ptr_slot);
    let a_row_stride_v = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: a_row_stride_v, val: a_row_stride as u64 });
    let a_base_off = mb.new_gp();
    mb.push(MachInst::MulReg { dst: a_base_off, lhs: mi, rhs: a_row_stride_v });
    let a_base_byte = mb.new_gp();
    mb.push(MachInst::LslImm { dst: a_base_byte, src: a_base_off, shift: 2 });
    let a_base = mb.new_gp();
    mb.push(MachInst::AddReg { dst: a_base, lhs: a_ptr, rhs: a_base_byte });
    // Row stride in bytes
    let a_row_stride_bytes = mb.new_gp();
    mb.push(MachInst::LslImm { dst: a_row_stride_bytes, src: a_row_stride_v, shift: 2 });

    // B base: use effective (possibly transposed) pointer and strides
    let b_ptr = sp_load!(mb, effective_b_ptr_slot);
    let eff_bn = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: eff_bn, val: effective_b_n_stride as u64 });
    let b_col_off = mb.new_gp();
    mb.push(MachInst::MulReg { dst: b_col_off, lhs: n_blk, rhs: eff_bn });
    let b_col_byte = mb.new_gp();
    mb.push(MachInst::LslImm { dst: b_col_byte, src: b_col_off, shift: 2 });
    let b_col_ptr = mb.new_gp();
    mb.push(MachInst::AddReg { dst: b_col_ptr, lhs: b_ptr, rhs: b_col_byte });

    // Zero 16 accumulators: acc[r][g] for r=0..8, g=0..2
    let mut accs: Vec<Vec<crate::mach_ir::VReg>> = Vec::new();
    for _r in 0..MR {
        let mut row_accs = Vec::new();
        for _g in 0..2 {
            let acc = mb.new_vec();
            mb.push(MachInst::Movi4sZero { dst: acc });
            row_accs.push(acc);
        }
        accs.push(row_accs);
    }

    // K loop with pointer incrementing (no multiply per iteration)
    //
    // a_ptr_k starts at A[mi, 0], incremented by a_k_stride*4 each K step.
    // b_ptr_k starts at B[0, n_blk], incremented by b_k_stride*4 each K step.
    // A rows at offset r*a_row_stride_bytes from a_ptr_k (constant offsets).
    let a_ptr_k = mb.new_gp();
    mb.push(MachInst::MovReg { dst: a_ptr_k, src: a_base });
    let b_ptr_k = mb.new_gp();
    mb.push(MachInst::MovReg { dst: b_ptr_k, src: b_col_ptr });

    // Stride increments in bytes (computed once before the loop)
    let a_k_inc = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: a_k_inc, val: (a_k_stride * 4) as u64 });
    let b_k_inc = mb.new_gp();
    {
        let bks = sp_load!(mb, effective_b_k_stride_slot);
        mb.push(MachInst::LslImm { dst: b_k_inc, src: bks, shift: 2 }); // b_k_stride * 4
    }

    let ki = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: ki, val: 0 });
    let ki_start = e.alloc_label();
    let ki_end = e.alloc_label();
    mb.push(MachInst::Label { label: ki_start });
    mb.push(MachInst::CmpReg { lhs: ki, rhs: k_size });
    mb.push(MachInst::BCond { cond: COND_GE, label: ki_end });

    // Load 2 B vectors from b_ptr_k
    // After the transpose (if b_n_stride > 1), B_t is in [K,N]-like layout
    // so b_ptr_k points to contiguous N values.
    let b0 = mb.new_vec();
    mb.push(MachInst::LdrQImm { dst: b0, base: b_ptr_k, imm: 0 });
    let b1 = mb.new_vec();
    mb.push(MachInst::LdrQImm { dst: b1, base: b_ptr_k, imm: 16 });

    // For each of MR=8 rows: load A scalar, broadcast, 2 FMLA
    // A[mi+r, ki] is at a_ptr_k + r * a_row_stride_bytes (constant offset per r)
    for r in 0..MR {
        let a_scalar = mb.new_fp();
        let imm_offset = (r * a_row_stride * 4) as u32;
        if imm_offset % 4 == 0 && imm_offset / 4 < 4096 {
            mb.push(MachInst::LdrSImm { dst: a_scalar, base: a_ptr_k, imm: imm_offset });
        } else {
            let off = mb.new_gp();
            mb.push(MachInst::MovImm64 { dst: off, val: imm_offset as u64 });
            let addr = mb.new_gp();
            mb.push(MachInst::AddReg { dst: addr, lhs: a_ptr_k, rhs: off });
            mb.push(MachInst::LdrSImm { dst: a_scalar, base: addr, imm: 0 });
        }
        // Broadcast: DUP Vd.4S, Sn (1 instruction instead of FMOV+DUP)
        let a_bcast = mb.new_vec();
        mb.push(MachInst::Dup4sScalar { dst: a_bcast, src: a_scalar });
        // 2 FMLAs
        mb.push(MachInst::Fmla4s { acc: accs[r][0], lhs: a_bcast, rhs: b0 });
        mb.push(MachInst::Fmla4s { acc: accs[r][1], lhs: a_bcast, rhs: b1 });
    }

    // Increment pointers
    mb.push(MachInst::AddReg { dst: a_ptr_k, lhs: a_ptr_k, rhs: a_k_inc });
    mb.push(MachInst::AddReg { dst: b_ptr_k, lhs: b_ptr_k, rhs: b_k_inc });

    // ki++
    mb.push(MachInst::AddImm { dst: ki, src: ki, imm: 1 });
    mb.push(MachInst::B { label: ki_start });
    mb.push(MachInst::Label { label: ki_end });

    // Store 16 accumulators to C
    // out[mi+r, n_blk..n_blk+8] — output is flat [effective_M, N]
    let out_ptr = sp_load!(mb, out_ptr_slot);
    let out_row_stride_r = sp_load!(mb, out_row_stride_slot);
    for r in 0..MR {
        let row_idx = mb.new_gp();
        mb.push(MachInst::AddImm { dst: row_idx, src: mi, imm: r as u32 });
        let out_off = mb.new_gp();
        mb.push(MachInst::MulReg { dst: out_off, lhs: row_idx, rhs: out_row_stride_r });
        mb.push(MachInst::AddReg { dst: out_off, lhs: out_off, rhs: n_blk });
        let byte_off = mb.new_gp();
        mb.push(MachInst::LslImm { dst: byte_off, src: out_off, shift: 2 });
        let addr = mb.new_gp();
        mb.push(MachInst::AddReg { dst: addr, lhs: out_ptr, rhs: byte_off });
        mb.push(MachInst::StrQImm { src: accs[r][0], base: addr, imm: 0 });
        mb.push(MachInst::StrQImm { src: accs[r][1], base: addr, imm: 16 });
    }

    // mi += MR
    mb.push(MachInst::AddImm { dst: mi, src: mi, imm: MR as u32 });
    mb.push(MachInst::B { label: mi_start });
    mb.push(MachInst::Label { label: mi_end });

    // === Scalar remainder for M % MR rows ===
    let mi_rem_start = e.alloc_label();
    let mi_rem_end = e.alloc_label();
    mb.push(MachInst::Label { label: mi_rem_start });
    mb.push(MachInst::CmpReg { lhs: mi, rhs: eff_m });
    mb.push(MachInst::BCond { cond: COND_GE, label: mi_rem_end });

    let a_ptr2 = sp_load!(mb, a_ptr_slot);
    let a_row_stride_v2 = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: a_row_stride_v2, val: a_row_stride as u64 });
    let a_row_off = mb.new_gp();
    mb.push(MachInst::MulReg { dst: a_row_off, lhs: mi, rhs: a_row_stride_v2 });
    let a_row_byte = mb.new_gp();
    mb.push(MachInst::LslImm { dst: a_row_byte, src: a_row_off, shift: 2 });
    let a_row = mb.new_gp();
    mb.push(MachInst::AddReg { dst: a_row, lhs: a_ptr2, rhs: a_row_byte });

    let rem_acc0 = mb.new_vec();
    mb.push(MachInst::Movi4sZero { dst: rem_acc0 });
    let rem_acc1 = mb.new_vec();
    mb.push(MachInst::Movi4sZero { dst: rem_acc1 });

    let rem_ki = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: rem_ki, val: 0 });
    let rem_ki_start = e.alloc_label();
    let rem_ki_end = e.alloc_label();
    mb.push(MachInst::Label { label: rem_ki_start });
    mb.push(MachInst::CmpReg { lhs: rem_ki, rhs: k_size });
    mb.push(MachInst::BCond { cond: COND_GE, label: rem_ki_end });

    let rem_aks = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: rem_aks, val: a_k_stride as u64 });
    let rem_a_off = mb.new_gp();
    mb.push(MachInst::MulReg { dst: rem_a_off, lhs: rem_ki, rhs: rem_aks });
    let rem_a_s = mb.new_fp();
    mb.push(MachInst::LdrSRegScaled { dst: rem_a_s, base: a_row, offset: rem_a_off });
    let rem_a_bcast = mb.new_vec();
    mb.push(MachInst::Dup4sScalar { dst: rem_a_bcast, src: rem_a_s });

    let rem_bks = sp_load!(mb, effective_b_k_stride_slot);
    let rem_b_off = mb.new_gp();
    mb.push(MachInst::MulReg { dst: rem_b_off, lhs: rem_ki, rhs: rem_bks });
    let rem_b_byte = mb.new_gp();
    mb.push(MachInst::LslImm { dst: rem_b_byte, src: rem_b_off, shift: 2 });
    let rem_b_addr = mb.new_gp();
    mb.push(MachInst::AddReg { dst: rem_b_addr, lhs: b_col_ptr, rhs: rem_b_byte });
    let rem_b0 = mb.new_vec();
    mb.push(MachInst::LdrQImm { dst: rem_b0, base: rem_b_addr, imm: 0 });
    let rem_b1 = mb.new_vec();
    mb.push(MachInst::LdrQImm { dst: rem_b1, base: rem_b_addr, imm: 16 });
    mb.push(MachInst::Fmla4s { acc: rem_acc0, lhs: rem_a_bcast, rhs: rem_b0 });
    mb.push(MachInst::Fmla4s { acc: rem_acc1, lhs: rem_a_bcast, rhs: rem_b1 });

    mb.push(MachInst::AddImm { dst: rem_ki, src: rem_ki, imm: 1 });
    mb.push(MachInst::B { label: rem_ki_start });
    mb.push(MachInst::Label { label: rem_ki_end });

    // Store remainder row
    let out_ptr2 = sp_load!(mb, out_ptr_slot);
    let out_row_stride_r2 = sp_load!(mb, out_row_stride_slot);
    let rem_out_off = mb.new_gp();
    mb.push(MachInst::MulReg { dst: rem_out_off, lhs: mi, rhs: out_row_stride_r2 });
    mb.push(MachInst::AddReg { dst: rem_out_off, lhs: rem_out_off, rhs: n_blk });
    let rem_byte = mb.new_gp();
    mb.push(MachInst::LslImm { dst: rem_byte, src: rem_out_off, shift: 2 });
    let rem_addr = mb.new_gp();
    mb.push(MachInst::AddReg { dst: rem_addr, lhs: out_ptr2, rhs: rem_byte });
    mb.push(MachInst::StrQImm { src: rem_acc0, base: rem_addr, imm: 0 });
    mb.push(MachInst::StrQImm { src: rem_acc1, base: rem_addr, imm: 16 });

    mb.push(MachInst::AddImm { dst: mi, src: mi, imm: 1 });
    mb.push(MachInst::B { label: mi_rem_start });
    mb.push(MachInst::Label { label: mi_rem_end });

    // n_blk += NR
    mb.push(MachInst::AddImm { dst: n_blk, src: n_blk, imm: NR as u32 });
    mb.push(MachInst::B { label: n_blk_start });
    mb.push(MachInst::Label { label: n_blk_end });

    // === N remainder: scalar fallback for N % NR cols ===
    // For each remaining column, scalar K-loop per row
    let ni = mb.new_gp();
    mb.push(MachInst::MovReg { dst: ni, src: n_blk }); // n_blk now == N rounded down to NR
    let ni_start = e.alloc_label();
    let ni_end = e.alloc_label();
    mb.push(MachInst::Label { label: ni_start });
    mb.push(MachInst::CmpReg { lhs: ni, rhs: n_size });
    mb.push(MachInst::BCond { cond: COND_GE, label: ni_end });

    let mi2 = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: mi2, val: 0 });
    let mi2_start = e.alloc_label();
    let mi2_end = e.alloc_label();
    mb.push(MachInst::Label { label: mi2_start });
    mb.push(MachInst::CmpReg { lhs: mi2, rhs: eff_m });
    mb.push(MachInst::BCond { cond: COND_GE, label: mi2_end });

    // Scalar accumulate
    let s_acc = mb.new_fp();
    mb.push(MachInst::Movi4sZero { dst: s_acc }); // zero scalar via movi (upper bits ignored)

    let a_ptr3 = sp_load!(mb, a_ptr_slot);
    let b_ptr3 = sp_load!(mb, effective_b_ptr_slot);
    let a_row_stride_v3 = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: a_row_stride_v3, val: a_row_stride as u64 });
    let s_a_row_off = mb.new_gp();
    mb.push(MachInst::MulReg { dst: s_a_row_off, lhs: mi2, rhs: a_row_stride_v3 });
    let s_a_byte = mb.new_gp();
    mb.push(MachInst::LslImm { dst: s_a_byte, src: s_a_row_off, shift: 2 });
    let s_a_row = mb.new_gp();
    mb.push(MachInst::AddReg { dst: s_a_row, lhs: a_ptr3, rhs: s_a_byte });

    let b_n_stride_v2 = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: b_n_stride_v2, val: effective_b_n_stride as u64 });
    let s_b_col_off = mb.new_gp();
    mb.push(MachInst::MulReg { dst: s_b_col_off, lhs: ni, rhs: b_n_stride_v2 });
    let s_b_byte = mb.new_gp();
    mb.push(MachInst::LslImm { dst: s_b_byte, src: s_b_col_off, shift: 2 });
    let s_b_col = mb.new_gp();
    mb.push(MachInst::AddReg { dst: s_b_col, lhs: b_ptr3, rhs: s_b_byte });

    let s_ki = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: s_ki, val: 0 });
    let s_ki_start = e.alloc_label();
    let s_ki_end = e.alloc_label();
    mb.push(MachInst::Label { label: s_ki_start });
    mb.push(MachInst::CmpReg { lhs: s_ki, rhs: k_size });
    mb.push(MachInst::BCond { cond: COND_GE, label: s_ki_end });

    let s_aks = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: s_aks, val: a_k_stride as u64 });
    let s_a_off = mb.new_gp();
    mb.push(MachInst::MulReg { dst: s_a_off, lhs: s_ki, rhs: s_aks });
    let s_a_val = mb.new_fp();
    mb.push(MachInst::LdrSRegScaled { dst: s_a_val, base: s_a_row, offset: s_a_off });
    let s_bks = sp_load!(mb, effective_b_k_stride_slot);
    let s_b_off = mb.new_gp();
    mb.push(MachInst::MulReg { dst: s_b_off, lhs: s_ki, rhs: s_bks });
    let s_b_val = mb.new_fp();
    mb.push(MachInst::LdrSRegScaled { dst: s_b_val, base: s_b_col, offset: s_b_off });
    mb.push(MachInst::Fmadd { dst: s_acc, mul_lhs: s_a_val, mul_rhs: s_b_val, add: s_acc });

    mb.push(MachInst::AddImm { dst: s_ki, src: s_ki, imm: 1 });
    mb.push(MachInst::B { label: s_ki_start });
    mb.push(MachInst::Label { label: s_ki_end });

    // Store scalar result
    let out_ptr3 = sp_load!(mb, out_ptr_slot);
    let out_row_stride_r3 = sp_load!(mb, out_row_stride_slot);
    let s_out_off = mb.new_gp();
    mb.push(MachInst::MulReg { dst: s_out_off, lhs: mi2, rhs: out_row_stride_r3 });
    mb.push(MachInst::AddReg { dst: s_out_off, lhs: s_out_off, rhs: ni });
    mb.push(MachInst::StrSRegScaled { src: s_acc, base: out_ptr3, offset: s_out_off });

    mb.push(MachInst::AddImm { dst: mi2, src: mi2, imm: 1 });
    mb.push(MachInst::B { label: mi2_start });
    mb.push(MachInst::Label { label: mi2_end });

    mb.push(MachInst::AddImm { dst: ni, src: ni, imm: 1 });
    mb.push(MachInst::B { label: ni_start });
    mb.push(MachInst::Label { label: ni_end });

    let insts = mb.finish();
    let next_vreg = mb.next_vreg_id();
    allocate_and_emit_with_spills(&insts, e, next_vreg, &mut || ctx.alloc_slot());
}

fn emit_tiled_matmul_fast(
    e: &mut ArmEmitter,
    ctx: &mut CodegenCtx,
    buf_slot: u32,
    shape: &[Dim],
    reduce: &loop_ir::ReduceDesc,
    body: &[Inst],
    _tile_cfg: &TileConfig,
    a_inst_idx: usize,
    b_inst_idx: usize,
    batch_strides: &[Dim],
    batch_dims: usize,
    m_dim: usize,
    n_dim: usize,
    reduce_dim: usize,
    tm: usize,
    tn: usize,
    _tk: usize,
    unroll: usize,
    simd_groups: usize,
    init_val: f32,
) {
    use crate::mach_ir::{MachBuilder, MachInst, allocate_and_emit_with_spills};

    // Extract buffer IDs and strides from the Load instructions
    let (a_buf, a_index) = match &body[a_inst_idx] {
        Inst::Load { buf, index } => (*buf, index.clone()),
        _ => unreachable!(),
    };
    let (b_buf, b_index) = match &body[b_inst_idx] {
        Inst::Load { buf, index } => (*buf, index.clone()),
        _ => unreachable!(),
    };

    let a_k_stride = get_stride_for_dim(&a_index, reduce_dim);
    let a_m_stride = get_stride_for_dim(&a_index, m_dim);
    let b_k_stride = get_stride_for_dim(&b_index, reduce_dim);
    let b_n_stride = get_stride_for_dim(&b_index, n_dim);

    // ---------------------------------------------------------------
    // Pre-compute values into stack slots (using direct emitter)
    // ---------------------------------------------------------------
    let out_ptr_slot = ctx.alloc_slot();
    load_from_sp_large(e, X9, buf_slot);
    store_to_sp_large(e, X9, out_ptr_slot);

    let a_ptr_slot = ctx.alloc_slot();
    load_buf_ptr(e, ctx, a_buf, X9);
    store_to_sp_large(e, X9, a_ptr_slot);

    let b_ptr_slot = ctx.alloc_slot();
    load_buf_ptr(e, ctx, b_buf, X9);
    store_to_sp_large(e, X9, b_ptr_slot);

    let m_size_slot = ctx.alloc_slot();
    emit_dim_to_reg(e, &shape[m_dim], ctx, X9);
    store_to_sp_large(e, X9, m_size_slot);

    let n_size_slot = ctx.alloc_slot();
    emit_dim_to_reg(e, &shape[n_dim], ctx, X9);
    store_to_sp_large(e, X9, n_size_slot);

    let k_size_slot = ctx.alloc_slot();
    emit_dim_to_reg(e, &reduce.size, ctx, X9);
    store_to_sp_large(e, X9, k_size_slot);

    let mut batch_size_slots = Vec::new();
    let mut batch_stride_slots = Vec::new();
    for d in 0..batch_dims {
        let size_slot = ctx.alloc_slot();
        emit_dim_to_reg(e, &shape[d], ctx, X9);
        store_to_sp_large(e, X9, size_slot);
        batch_size_slots.push(size_slot);

        let stride_slot = ctx.alloc_slot();
        emit_dim_to_reg(e, &batch_strides[d], ctx, X9);
        store_to_sp_large(e, X9, stride_slot);
        batch_stride_slots.push(stride_slot);
    }

    // Pre-store constants to stack slots
    let tm_slot = ctx.alloc_slot();
    e.mov_imm64(X9, tm as u64);
    store_to_sp_large(e, X9, tm_slot);

    let tn_slot = ctx.alloc_slot();
    e.mov_imm64(X9, tn as u64);
    store_to_sp_large(e, X9, tn_slot);

    let unroll_slot = ctx.alloc_slot();
    e.mov_imm64(X9, unroll as u64);
    store_to_sp_large(e, X9, unroll_slot);

    // ---------------------------------------------------------------
    // Build entire matmul in MachIR (outer loops + inner kernel)
    // ---------------------------------------------------------------
    // Helper macro-like closure to load a value from stack into a fresh vreg
    let mut mb = MachBuilder::new();

    // n_size and k_size are loaded at top since they're used in hot inner loop
    let n_size = mb.new_gp();
    mb.push(MachInst::SpLoad { dst: n_size, slot: n_size_slot });
    let k_size = mb.new_gp();
    mb.push(MachInst::SpLoad { dst: k_size, slot: k_size_slot });
    // out_ptr, a_ptr, b_ptr loaded on demand to reduce register pressure

    // Helper: load a stack slot into a fresh GP vreg
    macro_rules! sp_load_gp {
        ($mb:expr, $slot:expr) => {{
            let v = $mb.new_gp();
            $mb.push(MachInst::SpLoad { dst: v, slot: $slot });
            v
        }};
    }

    // --- Batch loops ---
    let mut batch_dims_v = Vec::new();
    let mut batch_loop_starts = Vec::new();
    let mut batch_loop_ends = Vec::new();
    for d in 0..batch_dims {
        let bd = mb.new_gp();
        mb.push(MachInst::MovImm64 { dst: bd, val: 0 });
        batch_dims_v.push(bd);
        let start = e.alloc_label();
        let end = e.alloc_label();
        batch_loop_starts.push(start);
        batch_loop_ends.push(end);
        mb.push(MachInst::Label { label: start });
        let bs = sp_load_gp!(mb, batch_size_slots[d]);
        mb.push(MachInst::CmpReg { lhs: bd, rhs: bs });
        mb.push(MachInst::BCond { cond: COND_GE, label: end });
    }

    // m_blocks = ceil(M / tm)
    let m_blocks = mb.new_gp();
    {
        let ms = sp_load_gp!(mb, m_size_slot);
        let tmp = mb.new_gp();
        mb.push(MachInst::AddImm { dst: tmp, src: ms, imm: (tm - 1) as u32 });
        let tv = sp_load_gp!(mb, tm_slot);
        mb.push(MachInst::Sdiv { dst: m_blocks, lhs: tmp, rhs: tv });
    }

    // --- m_blk loop ---
    let m_blk = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: m_blk, val: 0 });
    let m_blk_start = e.alloc_label();
    let m_blk_end = e.alloc_label();
    mb.push(MachInst::Label { label: m_blk_start });
    mb.push(MachInst::CmpReg { lhs: m_blk, rhs: m_blocks });
    mb.push(MachInst::BCond { cond: COND_GE, label: m_blk_end });

    // m_end = min(tm, M - m_blk * tm)
    let m_end = mb.new_gp();
    {
        let tv = sp_load_gp!(mb, tm_slot);
        let ms = sp_load_gp!(mb, m_size_slot);
        let tmp = mb.new_gp();
        mb.push(MachInst::MulReg { dst: tmp, lhs: m_blk, rhs: tv });
        mb.push(MachInst::SubReg { dst: m_end, lhs: ms, rhs: tmp });
        let tv2 = sp_load_gp!(mb, tm_slot);
        mb.push(MachInst::CmpReg { lhs: m_end, rhs: tv2 });
        mb.push(MachInst::Csel { dst: m_end, t_val: m_end, f_val: tv2, cond: COND_LT });
    }

    // --- mi loop ---
    let mi = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: mi, val: 0 });
    let mi_start = e.alloc_label();
    let mi_end = e.alloc_label();
    mb.push(MachInst::Label { label: mi_start });
    mb.push(MachInst::CmpReg { lhs: mi, rhs: m_end });
    mb.push(MachInst::BCond { cond: COND_GE, label: mi_end });

    // m_dim_val = m_blk * tm + mi
    let m_dim_val = mb.new_gp();
    {
        let tv = sp_load_gp!(mb, tm_slot);
        mb.push(MachInst::Madd { dst: m_dim_val, mul_lhs: m_blk, mul_rhs: tv, add: mi });
    }

    // a_row_ptr = a_ptr + (batch_offset_a + m_dim_val * a_m_stride) * 4
    let a_row_ptr = mb.new_gp();
    {
        let a_m_stride_v = mb.new_gp();
        mb.push(MachInst::MovImm64 { dst: a_m_stride_v, val: a_m_stride as u64 });
        let offset = mb.new_gp();
        mb.push(MachInst::MulReg { dst: offset, lhs: m_dim_val, rhs: a_m_stride_v });
        for d in 0..batch_dims {
            let a_batch_stride = get_stride_for_dim(&a_index, d);
            if a_batch_stride != 0 {
                let abs = mb.new_gp();
                mb.push(MachInst::MovImm64 { dst: abs, val: a_batch_stride as u64 });
                mb.push(MachInst::Madd { dst: offset, mul_lhs: batch_dims_v[d], mul_rhs: abs, add: offset });
            }
        }
        mb.push(MachInst::LslImm { dst: offset, src: offset, shift: 2 });
        let ap = sp_load_gp!(mb, a_ptr_slot);
        mb.push(MachInst::AddReg { dst: a_row_ptr, lhs: ap, rhs: offset });
    }

    // b_base = b_ptr + batch_offset_b * 4
    let b_base = mb.new_gp();
    {
        let offset = mb.new_gp();
        mb.push(MachInst::MovImm64 { dst: offset, val: 0 });
        for d in 0..batch_dims {
            let b_batch_stride = get_stride_for_dim(&b_index, d);
            if b_batch_stride != 0 {
                let bbs = mb.new_gp();
                mb.push(MachInst::MovImm64 { dst: bbs, val: b_batch_stride as u64 });
                mb.push(MachInst::Madd { dst: offset, mul_lhs: batch_dims_v[d], mul_rhs: bbs, add: offset });
            }
        }
        mb.push(MachInst::LslImm { dst: offset, src: offset, shift: 2 });
        let bp = sp_load_gp!(mb, b_ptr_slot);
        mb.push(MachInst::AddReg { dst: b_base, lhs: bp, rhs: offset });
    }

    // --- N block loop ---
    let n_blocks = mb.new_gp();
    {
        let tmp = mb.new_gp();
        mb.push(MachInst::AddImm { dst: tmp, src: n_size, imm: (tn - 1) as u32 });
        let tv = sp_load_gp!(mb, tn_slot);
        mb.push(MachInst::Sdiv { dst: n_blocks, lhs: tmp, rhs: tv });
    }

    let n_blk = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: n_blk, val: 0 });
    let n_blk_start = e.alloc_label();
    let n_blk_end = e.alloc_label();
    mb.push(MachInst::Label { label: n_blk_start });
    mb.push(MachInst::CmpReg { lhs: n_blk, rhs: n_blocks });
    mb.push(MachInst::BCond { cond: COND_GE, label: n_blk_end });

    // n_tile = min(tn, N - n_blk * tn)
    let n_tile = mb.new_gp();
    {
        let tv = sp_load_gp!(mb, tn_slot);
        let tmp = mb.new_gp();
        mb.push(MachInst::MulReg { dst: tmp, lhs: n_blk, rhs: tv });
        mb.push(MachInst::SubReg { dst: n_tile, lhs: n_size, rhs: tmp });
        let tv2 = sp_load_gp!(mb, tn_slot);
        mb.push(MachInst::CmpReg { lhs: n_tile, rhs: tv2 });
        mb.push(MachInst::Csel { dst: n_tile, t_val: n_tile, f_val: tv2, cond: COND_LT });
    }

    // ni_grp_count = n_tile / unroll
    let ni_grp_count = mb.new_gp();
    {
        let uv = sp_load_gp!(mb, unroll_slot);
        mb.push(MachInst::Sdiv { dst: ni_grp_count, lhs: n_tile, rhs: uv });
    }

    // --- ni_grp loop ---
    let ni_grp = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: ni_grp, val: 0 });
    let ni_grp_start = e.alloc_label();
    let ni_grp_end = e.alloc_label();
    mb.push(MachInst::Label { label: ni_grp_start });
    mb.push(MachInst::CmpReg { lhs: ni_grp, rhs: ni_grp_count });
    mb.push(MachInst::BCond { cond: COND_GE, label: ni_grp_end });

    // ni_base = n_blk * tn + ni_grp * unroll
    let ni_base = mb.new_gp();
    {
        let tv = sp_load_gp!(mb, tn_slot);
        let tmp = mb.new_gp();
        mb.push(MachInst::MulReg { dst: tmp, lhs: n_blk, rhs: tv });
        let uv = sp_load_gp!(mb, unroll_slot);
        mb.push(MachInst::Madd { dst: ni_base, mul_lhs: ni_grp, mul_rhs: uv, add: tmp });
    }

    // === Zero accumulators ===
    let mut acc_vecs: Vec<crate::mach_ir::VReg> = Vec::new();
    for _g in 0..simd_groups {
        let acc = mb.new_vec();
        if init_val.to_bits() == 0 {
            mb.push(MachInst::Movi4sZero { dst: acc });
        } else {
            let init_gp = mb.new_gp();
            mb.push(MachInst::MovImm64 { dst: init_gp, val: init_val.to_bits() as u64 });
            mb.push(MachInst::Dup4sGp { dst: acc, src_gp: init_gp });
        }
        acc_vecs.push(acc);
    }

    // === K loop ===
    let ki = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: ki, val: 0 });

    let ki_start = e.alloc_label();
    let ki_end = e.alloc_label();
    mb.push(MachInst::Label { label: ki_start });
    mb.push(MachInst::CmpReg { lhs: ki, rhs: k_size });
    mb.push(MachInst::BCond { cond: COND_GE, label: ki_end });

    // Load A[m_dim, ki] scalar, broadcast
    let a_scalar = mb.new_fp();
    if a_k_stride == 1 {
        mb.push(MachInst::LdrSRegScaled { dst: a_scalar, base: a_row_ptr, offset: ki });
    } else {
        let a_k_off = mb.new_gp();
        let a_k_stride_v = mb.new_gp();
        mb.push(MachInst::MovImm64 { dst: a_k_stride_v, val: a_k_stride as u64 });
        mb.push(MachInst::MulReg { dst: a_k_off, lhs: ki, rhs: a_k_stride_v });
        mb.push(MachInst::LdrSRegScaled { dst: a_scalar, base: a_row_ptr, offset: a_k_off });
    }
    let a_bits_gp = mb.new_gp();
    mb.push(MachInst::FmovWFromS { dst_gp: a_bits_gp, src_fp: a_scalar });
    let a_broadcast = mb.new_vec();
    mb.push(MachInst::Dup4sGp { dst: a_broadcast, src_gp: a_bits_gp });

    // Compute B row pointer for this ki
    let b_row_ptr = mb.new_gp();
    if b_k_stride as u64 == shape[n_dim].as_usize().unwrap_or(0) as u64 && b_n_stride == 1 {
        let b_tmp = mb.new_gp();
        mb.push(MachInst::MulReg { dst: b_tmp, lhs: ki, rhs: n_size });
        mb.push(MachInst::AddReg { dst: b_tmp, lhs: b_tmp, rhs: ni_base });
        mb.push(MachInst::LslImm { dst: b_tmp, src: b_tmp, shift: 2 });
        mb.push(MachInst::AddReg { dst: b_row_ptr, lhs: b_base, rhs: b_tmp });
    } else {
        let bk_stride_v = mb.new_gp();
        mb.push(MachInst::MovImm64 { dst: bk_stride_v, val: b_k_stride as u64 });
        let b_tmp = mb.new_gp();
        mb.push(MachInst::MulReg { dst: b_tmp, lhs: ki, rhs: bk_stride_v });
        let bn_stride_v = mb.new_gp();
        mb.push(MachInst::MovImm64 { dst: bn_stride_v, val: b_n_stride as u64 });
        let b_tmp2 = mb.new_gp();
        mb.push(MachInst::MulReg { dst: b_tmp2, lhs: ni_base, rhs: bn_stride_v });
        mb.push(MachInst::AddReg { dst: b_tmp, lhs: b_tmp, rhs: b_tmp2 });
        mb.push(MachInst::LslImm { dst: b_tmp, src: b_tmp, shift: 2 });
        mb.push(MachInst::AddReg { dst: b_row_ptr, lhs: b_base, rhs: b_tmp });
    }

    // FMLA for each SIMD group
    for g in 0..simd_groups {
        let byte_offset = (g * 16) as u32;
        let b_vec = mb.new_vec();
        mb.push(MachInst::LdrQImm { dst: b_vec, base: b_row_ptr, imm: byte_offset });
        mb.push(MachInst::Fmla4s { acc: acc_vecs[g], lhs: a_broadcast, rhs: b_vec });
    }

    // ki++
    mb.push(MachInst::AddImm { dst: ki, src: ki, imm: 1 });
    mb.push(MachInst::B { label: ki_start });
    mb.push(MachInst::Label { label: ki_end });

    // === Store accumulators ===
    let out_off = mb.new_gp();
    mb.push(MachInst::MulReg { dst: out_off, lhs: m_dim_val, rhs: n_size });
    for d in 0..batch_dims {
        let bsv = sp_load_gp!(mb, batch_stride_slots[d]);
        mb.push(MachInst::Madd { dst: out_off, mul_lhs: batch_dims_v[d], mul_rhs: bsv, add: out_off });
    }
    mb.push(MachInst::AddReg { dst: out_off, lhs: out_off, rhs: ni_base });
    mb.push(MachInst::LslImm { dst: out_off, src: out_off, shift: 2 });
    let out_addr = mb.new_gp();
    {
        let op = sp_load_gp!(mb, out_ptr_slot);
        mb.push(MachInst::AddReg { dst: out_addr, lhs: op, rhs: out_off });
    }

    for g in 0..simd_groups {
        let byte_offset = (g * 16) as u32;
        mb.push(MachInst::StrQImm { src: acc_vecs[g], base: out_addr, imm: byte_offset });
    }

    // === Remainder: scalar loop for leftover elements ===
    let rem_base = mb.new_gp();
    {
        let uv = sp_load_gp!(mb, unroll_slot);
        mb.push(MachInst::MulReg { dst: rem_base, lhs: ni_grp_count, rhs: uv });
    }
    let rem_i = mb.new_gp();
    {
        let tv = sp_load_gp!(mb, tn_slot);
        mb.push(MachInst::Madd { dst: rem_i, mul_lhs: n_blk, mul_rhs: tv, add: rem_base });
    }
    let rem_end = mb.new_gp();
    {
        let tv = sp_load_gp!(mb, tn_slot);
        mb.push(MachInst::Madd { dst: rem_end, mul_lhs: n_blk, mul_rhs: tv, add: n_tile });
    }

    let rem_start_lbl = e.alloc_label();
    let rem_end_lbl = e.alloc_label();
    mb.push(MachInst::Label { label: rem_start_lbl });
    mb.push(MachInst::CmpReg { lhs: rem_i, rhs: rem_end });
    mb.push(MachInst::BCond { cond: COND_GE, label: rem_end_lbl });

    // acc = init_val
    let acc_s = mb.new_fp();
    let init_gp = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: init_gp, val: init_val.to_bits() as u64 });
    mb.push(MachInst::FmovSFromW { dst_fp: acc_s, src_gp: init_gp });

    // K loop for remainder
    let rem_ki = mb.new_gp();
    mb.push(MachInst::MovImm64 { dst: rem_ki, val: 0 });
    let rem_k_start = e.alloc_label();
    let rem_k_end = e.alloc_label();
    mb.push(MachInst::Label { label: rem_k_start });
    mb.push(MachInst::CmpReg { lhs: rem_ki, rhs: k_size });
    mb.push(MachInst::BCond { cond: COND_GE, label: rem_k_end });

    let s_a = mb.new_fp();
    if a_k_stride == 1 {
        mb.push(MachInst::LdrSRegScaled { dst: s_a, base: a_row_ptr, offset: rem_ki });
    } else {
        let a_off = mb.new_gp();
        let ak_s = mb.new_gp();
        mb.push(MachInst::MovImm64 { dst: ak_s, val: a_k_stride as u64 });
        mb.push(MachInst::MulReg { dst: a_off, lhs: rem_ki, rhs: ak_s });
        mb.push(MachInst::LdrSRegScaled { dst: s_a, base: a_row_ptr, offset: a_off });
    }

    let s_b = mb.new_fp();
    if b_n_stride == 1 {
        let bk_s = mb.new_gp();
        mb.push(MachInst::MovImm64 { dst: bk_s, val: b_k_stride as u64 });
        let b_off = mb.new_gp();
        mb.push(MachInst::Madd { dst: b_off, mul_lhs: rem_ki, mul_rhs: bk_s, add: rem_i });
        mb.push(MachInst::LdrSRegScaled { dst: s_b, base: b_base, offset: b_off });
    } else {
        let bk_s = mb.new_gp();
        mb.push(MachInst::MovImm64 { dst: bk_s, val: b_k_stride as u64 });
        let b_off1 = mb.new_gp();
        mb.push(MachInst::MulReg { dst: b_off1, lhs: rem_ki, rhs: bk_s });
        let bn_s = mb.new_gp();
        mb.push(MachInst::MovImm64 { dst: bn_s, val: b_n_stride as u64 });
        let b_off = mb.new_gp();
        mb.push(MachInst::Madd { dst: b_off, mul_lhs: rem_i, mul_rhs: bn_s, add: b_off1 });
        mb.push(MachInst::LdrSRegScaled { dst: s_b, base: b_base, offset: b_off });
    }

    mb.push(MachInst::Fmadd { dst: acc_s, mul_lhs: s_a, mul_rhs: s_b, add: acc_s });

    mb.push(MachInst::AddImm { dst: rem_ki, src: rem_ki, imm: 1 });
    mb.push(MachInst::B { label: rem_k_start });
    mb.push(MachInst::Label { label: rem_k_end });

    // Store remainder
    let store_off = mb.new_gp();
    mb.push(MachInst::MulReg { dst: store_off, lhs: m_dim_val, rhs: n_size });
    for d in 0..batch_dims {
        let bsv = sp_load_gp!(mb, batch_stride_slots[d]);
        mb.push(MachInst::Madd { dst: store_off, mul_lhs: batch_dims_v[d], mul_rhs: bsv, add: store_off });
    }
    mb.push(MachInst::AddReg { dst: store_off, lhs: store_off, rhs: rem_i });
    let store_byte_off = mb.new_gp();
    mb.push(MachInst::LslImm { dst: store_byte_off, src: store_off, shift: 2 });
    let store_addr = mb.new_gp();
    {
        let op = sp_load_gp!(mb, out_ptr_slot);
        mb.push(MachInst::AddReg { dst: store_addr, lhs: op, rhs: store_byte_off });
    }
    mb.push(MachInst::StrSImm { src: acc_s, base: store_addr, imm: 0 });

    mb.push(MachInst::AddImm { dst: rem_i, src: rem_i, imm: 1 });
    mb.push(MachInst::B { label: rem_start_lbl });
    mb.push(MachInst::Label { label: rem_end_lbl });

    // ---------------------------------------------------------------
    // Close outer loops (all in MachIR)
    // ---------------------------------------------------------------

    // ni_grp++
    mb.push(MachInst::AddImm { dst: ni_grp, src: ni_grp, imm: 1 });
    mb.push(MachInst::B { label: ni_grp_start });
    mb.push(MachInst::Label { label: ni_grp_end });

    // n_blk++
    mb.push(MachInst::AddImm { dst: n_blk, src: n_blk, imm: 1 });
    mb.push(MachInst::B { label: n_blk_start });
    mb.push(MachInst::Label { label: n_blk_end });

    // mi++
    mb.push(MachInst::AddImm { dst: mi, src: mi, imm: 1 });
    mb.push(MachInst::B { label: mi_start });
    mb.push(MachInst::Label { label: mi_end });

    // m_blk++
    mb.push(MachInst::AddImm { dst: m_blk, src: m_blk, imm: 1 });
    mb.push(MachInst::B { label: m_blk_start });
    mb.push(MachInst::Label { label: m_blk_end });

    // Close batch loops
    for d in (0..batch_dims).rev() {
        mb.push(MachInst::AddImm { dst: batch_dims_v[d], src: batch_dims_v[d], imm: 1 });
        mb.push(MachInst::B { label: batch_loop_starts[d] });
        mb.push(MachInst::Label { label: batch_loop_ends[d] });
    }

    // Allocate and emit
    let insts = mb.finish();
    let next_vreg = mb.next_vreg_id();
    allocate_and_emit_with_spills(&insts, e, next_vreg, &mut || ctx.alloc_slot());
}

/// Extract the stride for a given dimension from an Index.
fn get_stride_for_dim(index: &Index, dim: usize) -> usize {
    match index {
        Index::Strided { parts, .. } => {
            for (d, stride) in parts {
                if *d == dim {
                    return stride.as_usize().unwrap_or(1);
                }
            }
            0 // dimension not present = stride 0 (broadcast)
        }
        Index::Flat => 1,
    }
}

/// Optimized tiled loop with register allocation.
///
/// Register plan for the inner loops:
///   GP: X20=out_ptr, X21=A_ptr, X22=B_ptr, X23=M, X24=N, X25=K
///       X26=m_dim, X27=row_base_bytes, X28=ni_base
///       X3=A_row_ptr, X4=B_col_ptr, X9=ki, X10=k_end
///       X11-X15=outer loop counters/scratch
///   NEON: V0-V7=accumulators, V8=A_broadcast, V16=B_temp
fn emit_tiled_loop(
    e: &mut ArmEmitter,
    ctx: &mut CodegenCtx,
    buf_slot: u32,
    shape: &[Dim],
    reduce: &loop_ir::ReduceDesc,
    body: &[Inst],
    result: usize,
    tile_cfg: &TileConfig,
) {
    let ndim = shape.len();
    let tiles = &tile_cfg.tiles;
    let tk = tiles[ndim].as_usize().expect("tile sizes must be concrete");
    let batch_dims = ndim.saturating_sub(2);
    let m_dim = ndim - 2;
    let n_dim = ndim - 1;
    let tm = tiles[m_dim].as_usize().expect("tile sizes must be concrete");
    let tn = tiles[n_dim].as_usize().expect("tile sizes must be concrete");
    let reduce_dim = ndim;
    let unroll: usize = 32usize.min(tn);

    let init_val = match reduce.op {
        ReduceOp::Sum => 0.0f32,
        ReduceOp::Max => f32::NEG_INFINITY,
    };

    let batch_strides = Dim::strides(shape);

    // Compute N-dependence for hoisting
    let depends_on_n = compute_n_dependence(body, n_dim);
    let use_simd = can_simd_tiled_loop(body, &depends_on_n, n_dim) && unroll >= 4;
    let simd_groups = if use_simd { unroll / 4 } else { 0 };

    // Detect the common matmul pattern: Load, Load, Mul with Sum reduce
    // If matched, use the fully register-allocated fast path
    // Detect matmul pattern: Load, Load, Mul with Sum reduce.
    // The MR8 kernel handles its own B loading, so use_simd is not required.
    let is_matmul_pattern = reduce.op == ReduceOp::Sum
        && body.len() == 3
        && matches!(&body[2], Inst::Mul(0, 1))
        && result == 2
        && !depends_on_n[0] && depends_on_n[1];

    let is_matmul_reversed = reduce.op == ReduceOp::Sum
        && body.len() == 3
        && matches!(&body[2], Inst::Mul(0, 1))
        && result == 2
        && depends_on_n[0] && !depends_on_n[1];

    if std::env::var("ARM_BODY_DEBUG").is_ok() {
        eprintln!("  PATTERN: use_simd={} is_mm={} is_mmr={} dep_n={:?}", use_simd, is_matmul_pattern, is_matmul_reversed, depends_on_n);
        for (i,inst) in body.iter().enumerate() { eprintln!("    [{i}] {inst:?}"); }
    }
    if (is_matmul_pattern || is_matmul_reversed) && simd_groups <= 8 {
        let (a_inst_idx, b_inst_idx) = if is_matmul_pattern { (0, 1) } else { (1, 0) };
        // Always use MR=8 micro-kernel for simple matmul patterns.
        if std::env::var("ARM_BODY_DEBUG").is_ok() {
            eprintln!("MR8 HIT: shape={:?}", shape);
        }
        {
            emit_matmul_mr8(e, ctx, buf_slot, shape, reduce, body,
                a_inst_idx, b_inst_idx, &batch_strides, batch_dims, m_dim, n_dim,
                reduce_dim);
            return;
        }
        // Fall back to existing fast path for small-M or symbolic cases
        emit_tiled_matmul_fast(e, ctx, buf_slot, shape, reduce, body, tile_cfg,
            a_inst_idx, b_inst_idx, &batch_strides, batch_dims, m_dim, n_dim,
            reduce_dim, tm, tn, tk, unroll, simd_groups, init_val);
        return;
    }

    // === General (slower) tiled loop path ===
    // Allocate stack slots for all the loop variables
    let m_size_slot = ctx.alloc_slot();
    let n_size_slot = ctx.alloc_slot();
    let k_size_slot = ctx.alloc_slot();
    let row_base_slot = ctx.alloc_slot();
    let m_blk_slot = ctx.alloc_slot();
    let m_blocks_slot = ctx.alloc_slot();
    let mi_slot = ctx.alloc_slot();
    let m_end_slot = ctx.alloc_slot();
    let n_blk_slot = ctx.alloc_slot();
    let n_blocks_slot = ctx.alloc_slot();
    let n_tile_slot = ctx.alloc_slot();
    let ni_grp_slot = ctx.alloc_slot();
    let ni_grp_count_slot = ctx.alloc_slot();
    let ni_base_slot = ctx.alloc_slot();
    let k_blk_slot = ctx.alloc_slot();
    let k_blocks_slot = ctx.alloc_slot();
    let ki_slot = ctx.alloc_slot();
    let k_end_slot = ctx.alloc_slot();

    // Allocate slots for SIMD accumulators (stored as v128 on stack — 16 bytes each)
    // For simplicity, allocate 16-byte aligned slots
    let mut acc_v_slots: Vec<u32> = Vec::new();
    let mut acc_s_slots: Vec<u32> = Vec::new();
    if use_simd {
        for _ in 0..simd_groups {
            // Align to 16 bytes
            ctx.next_slot = (ctx.next_slot + 15) & !15;
            let slot = ctx.next_slot;
            ctx.next_slot += 16;
            acc_v_slots.push(slot);
        }
    } else {
        for _ in 0..unroll {
            acc_s_slots.push(ctx.alloc_slot());
        }
    }
    let acc_r_slot = ctx.alloc_slot();

    // Ensure dim slots exist
    for d in 0..=ndim {
        ctx.get_or_alloc_dim_slot(d);
    }

    // Compute M, N, K sizes
    emit_dim_to_reg(e, &shape[m_dim], ctx, X9);
    store_to_sp_large(e, X9, m_size_slot);
    emit_dim_to_reg(e, &shape[n_dim], ctx, X9);
    store_to_sp_large(e, X9, n_size_slot);
    emit_dim_to_reg(e, &reduce.size, ctx, X9);
    store_to_sp_large(e, X9, k_size_slot);

    // --- Batch loops ---
    let mut batch_loop_starts = Vec::new();
    let mut batch_loop_ends = Vec::new();
    let mut batch_size_slots = Vec::new();

    for d in 0..batch_dims {
        let dim_slot = ctx.dim_slots[&d];
        let size_slot = ctx.alloc_slot();
        batch_size_slots.push(size_slot);
        emit_dim_to_reg(e, &shape[d], ctx, X9);
        store_to_sp_large(e, X9, size_slot);

        // d = 0
        e.mov_reg(X9, XZR);
        store_to_sp_large(e, X9, dim_slot);

        let start = e.alloc_label();
        let end = e.alloc_label();
        batch_loop_starts.push(start);
        batch_loop_ends.push(end);

        e.bind_label(start);
        load_from_sp_large(e, X9, dim_slot);
        load_from_sp_large(e, X10, size_slot);
        e.cmp_reg(X9, X10);
        e.b_cond_label(COND_GE, end);
    }

    // m_blocks = (M + tm - 1) / tm
    load_from_sp_large(e, X9, m_size_slot);
    e.add_imm(X9, X9, (tm - 1) as u32);
    e.mov_imm32(X10, tm as u32);
    e.sdiv(X9, X9, X10);
    store_to_sp_large(e, X9, m_blocks_slot);

    // m_blk = 0
    e.mov_reg(X9, XZR);
    store_to_sp_large(e, X9, m_blk_slot);

    let m_blk_start = e.alloc_label();
    let m_blk_end = e.alloc_label();
    e.bind_label(m_blk_start);
    load_from_sp_large(e, X9, m_blk_slot);
    load_from_sp_large(e, X10, m_blocks_slot);
    e.cmp_reg(X9, X10);
    e.b_cond_label(COND_GE, m_blk_end);

    // m_end = min(tm, M - m_blk * tm)
    // m_end = min(tm, M - m_blk * tm)
    load_from_sp_large(e, X9, m_size_slot);
    load_from_sp_large(e, X10, m_blk_slot);
    e.mov_imm32(X11, tm as u32);
    e.mul_reg(X10, X10, X11);
    e.sub_reg(X9, X9, X10); // remaining = M - m_blk*tm
    e.mov_imm32(X10, tm as u32);
    e.cmp_reg(X9, X10);
    // CSEL X9, X9, X10, LT  (X9 = min(remaining, tm))
    e.emit(0x9A80B000 | (X10 as u32) << 16 | (X9 as u32) << 5 | X9 as u32);
    store_to_sp_large(e, X9, m_end_slot);

    // --- mi loop ---
    e.mov_reg(X9, XZR);
    store_to_sp_large(e, X9, mi_slot);

    let mi_start = e.alloc_label();
    let mi_end = e.alloc_label();
    e.bind_label(mi_start);
    load_from_sp_large(e, X9, mi_slot);
    load_from_sp_large(e, X10, m_end_slot);
    e.cmp_reg(X9, X10);
    e.b_cond_label(COND_GE, mi_end);

    // d{m_dim} = m_blk * tm + mi
    let m_dim_slot = ctx.dim_slots[&m_dim];
    load_from_sp_large(e, X9, m_blk_slot);
    e.mov_imm32(X10, tm as u32);
    e.mul_reg(X9, X9, X10);
    load_from_sp_large(e, X10, mi_slot);
    e.add_reg(X9, X9, X10);
    store_to_sp_large(e, X9, m_dim_slot);

    // row_base = sum(batch_dim * stride) + m * N
    {
        let mut has_terms = false;
        for d in 0..batch_dims {
            let dim_slot_d = ctx.dim_slots[&d];
            load_from_sp_large(e, X10, dim_slot_d);
            emit_dim_to_reg(e, &batch_strides[d], ctx, X11);
            e.mul_reg(X10, X10, X11);
            if has_terms {
                load_from_sp_large(e, X12, row_base_slot);
                e.add_reg(X10, X12, X10);
            }
            store_to_sp_large(e, X10, row_base_slot);
            has_terms = true;
        }
        load_from_sp_large(e, X10, m_dim_slot);
        load_from_sp_large(e, X11, n_size_slot);
        e.mul_reg(X10, X10, X11);
        if has_terms {
            load_from_sp_large(e, X12, row_base_slot);
            e.add_reg(X10, X12, X10);
        }
        store_to_sp_large(e, X10, row_base_slot);
    }

    // --- N block loop ---
    // n_blocks = (N + tn - 1) / tn
    load_from_sp_large(e, X9, n_size_slot);
    e.add_imm(X9, X9, (tn - 1) as u32);
    e.mov_imm32(X10, tn as u32);
    e.sdiv(X9, X9, X10);
    store_to_sp_large(e, X9, n_blocks_slot);

    e.mov_reg(X9, XZR);
    store_to_sp_large(e, X9, n_blk_slot);

    let n_blk_start = e.alloc_label();
    let n_blk_end = e.alloc_label();
    e.bind_label(n_blk_start);
    load_from_sp_large(e, X9, n_blk_slot);
    load_from_sp_large(e, X10, n_blocks_slot);
    e.cmp_reg(X9, X10);
    e.b_cond_label(COND_GE, n_blk_end);

    // n_tile = min(tn, N - n_blk * tn)
    load_from_sp_large(e, X9, n_size_slot);
    load_from_sp_large(e, X10, n_blk_slot);
    e.mov_imm32(X11, tn as u32);
    e.mul_reg(X10, X10, X11);
    e.sub_reg(X9, X9, X10);
    e.mov_imm32(X10, tn as u32);
    e.cmp_reg(X9, X10);
    e.emit(0x9A80B000 | (X10 as u32) << 16 | (X9 as u32) << 5 | X9 as u32); // CSEL min
    store_to_sp_large(e, X9, n_tile_slot);

    // ni_grp_count = n_tile / unroll
    load_from_sp_large(e, X9, n_tile_slot);
    e.mov_imm32(X10, unroll as u32);
    e.sdiv(X9, X9, X10);
    store_to_sp_large(e, X9, ni_grp_count_slot);

    // ni_grp = 0
    e.mov_reg(X9, XZR);
    store_to_sp_large(e, X9, ni_grp_slot);

    let ni_grp_start = e.alloc_label();
    let ni_grp_end = e.alloc_label();
    e.bind_label(ni_grp_start);
    load_from_sp_large(e, X9, ni_grp_slot);
    load_from_sp_large(e, X10, ni_grp_count_slot);
    e.cmp_reg(X9, X10);
    e.b_cond_label(COND_GE, ni_grp_end);

    // ni_base = n_blk * tn + ni_grp * unroll
    load_from_sp_large(e, X9, n_blk_slot);
    e.mov_imm32(X10, tn as u32);
    e.mul_reg(X9, X9, X10);
    load_from_sp_large(e, X10, ni_grp_slot);
    e.mov_imm32(X11, unroll as u32);
    e.madd(X9, X10, X11, X9);
    store_to_sp_large(e, X9, ni_base_slot);

    // Init accumulators
    if use_simd {
        let init_bits = init_val.to_bits();
        if init_bits == 0 {
            for g in 0..simd_groups {
                let slot = acc_v_slots[g];
                // Zero the accumulator (use MOVI + STR Q)
                e.movi_4s_zero(0);
                // STR Q0, [SP, #slot]
                assert!(slot % 16 == 0);
                store_q_to_sp_large(e, 0, slot);
            }
        } else {
            e.mov_imm32(X9, init_bits);
            e.fmov_s_from_w(0, X9);
            e.dup_4s_gp(0, X9);
            for g in 0..simd_groups {
                store_q_to_sp_large(e, 0, acc_v_slots[g]);
            }
        }
    } else {
        let init_bits = init_val.to_bits();
        e.mov_imm32(X9, init_bits);
        e.fmov_s_from_w(0, X9);
        for u in 0..unroll {
            store_s_to_sp_seq(e, 0, acc_s_slots[u]);
        }
    }

    // Compute K-dependence: instructions that don't depend on the reduce dim
    // can be hoisted out of the K loop entirely.
    let depends_on_k = compute_dim_dependence(body, reduce_dim);

    // Emit K-invariant, N-invariant instructions BEFORE the K loop.
    // These run once per (m, ni_grp) iteration instead of once per K.
    for (j, inst) in body.iter().enumerate() {
        if !depends_on_k[j] && !depends_on_n[j] {
            emit_inst(e, ctx, j, inst);
        }
    }

    // --- K block loop ---
    // k_blocks = (K + tk - 1) / tk
    load_from_sp_large(e, X9, k_size_slot);
    e.add_imm(X9, X9, (tk - 1) as u32);
    e.mov_imm32(X10, tk as u32);
    e.sdiv(X9, X9, X10);
    store_to_sp_large(e, X9, k_blocks_slot);

    e.mov_reg(X9, XZR);
    store_to_sp_large(e, X9, k_blk_slot);

    let k_blk_start = e.alloc_label();
    let k_blk_end = e.alloc_label();
    e.bind_label(k_blk_start);
    load_from_sp_large(e, X9, k_blk_slot);
    load_from_sp_large(e, X10, k_blocks_slot);
    e.cmp_reg(X9, X10);
    e.b_cond_label(COND_GE, k_blk_end);

    // k_end = min(tk, K - k_blk * tk)
    load_from_sp_large(e, X9, k_size_slot);
    load_from_sp_large(e, X10, k_blk_slot);
    e.mov_imm32(X11, tk as u32);
    e.mul_reg(X10, X10, X11);
    e.sub_reg(X9, X9, X10);
    e.mov_imm32(X10, tk as u32);
    e.cmp_reg(X9, X10);
    e.emit(0x9A80B000 | (X10 as u32) << 16 | (X9 as u32) << 5 | X9 as u32); // CSEL min
    store_to_sp_large(e, X9, k_end_slot);

    // ki = 0
    e.mov_reg(X9, XZR);
    store_to_sp_large(e, X9, ki_slot);

    let ki_start = e.alloc_label();
    let ki_end = e.alloc_label();
    e.bind_label(ki_start);
    load_from_sp_large(e, X9, ki_slot);
    load_from_sp_large(e, X10, k_end_slot);
    e.cmp_reg(X9, X10);
    e.b_cond_label(COND_GE, ki_end);

    // d{reduce_dim} = k_blk * tk + ki
    let reduce_dim_slot = ctx.dim_slots[&reduce_dim];
    load_from_sp_large(e, X9, k_blk_slot);
    e.mov_imm32(X10, tk as u32);
    e.mul_reg(X9, X9, X10);
    load_from_sp_large(e, X10, ki_slot);
    e.add_reg(X9, X9, X10);
    store_to_sp_large(e, X9, reduce_dim_slot);

    // Emit K-dependent, N-invariant body instructions (skip K-invariant ones
    // already emitted above, and N-dependent ones emitted per-N below).
    for (j, inst) in body.iter().enumerate() {
        if depends_on_k[j] && !depends_on_n[j] {
            emit_inst(e, ctx, j, inst);
        }
    }

    // Unrolled n-dependent instructions
    let n_dim_slot = ctx.dim_slots[&n_dim];

    if use_simd {
        // SIMD path: process groups of 4 N elements with NEON
        for g in 0..simd_groups {
            // d{n_dim} = ni_base + g * 4
            load_from_sp_large(e, X9, ni_base_slot);
            e.add_imm(X9, X9, (g * 4) as u32);
            store_to_sp_large(e, X9, n_dim_slot);

            // Emit NEON body for n-dependent instructions
            // We use V16-V29 as scratch for intermediate results
            let mut v_results: HashMap<usize, u8> = HashMap::new();
            let mut next_vreg: u8 = 16; // start from V16

            for (j, inst) in body.iter().enumerate() {
                if !depends_on_n[j] { continue; }

                let vd = next_vreg;
                next_vreg += 1;
                if next_vreg > 29 { next_vreg = 16; } // wrap around (hope we don't conflict)
                v_results.insert(j, vd);

                match inst {
                    Inst::Load { buf, index } => {
                        // Load 4 consecutive f32s as Q register
                        load_buf_ptr(e, ctx, *buf, X3);
                        emit_index_offset(e, ctx, index, X4, X5);
                        e.add_reg(X3, X3, X4);
                        // LDR Q{vd}, [X3]
                        e.ldr_q_imm(vd, X3, 0);
                    }
                    Inst::Const(v) => {
                        let bits = (*v as f32).to_bits();
                        e.mov_imm32(X9, bits);
                        e.dup_4s_gp(vd, X9);
                    }
                    Inst::Neg(a) => {
                        let va = get_v128_operand(e, ctx, *a, &depends_on_n, &v_results, 30);
                        e.fneg_4s(vd, va);
                    }
                    Inst::Recip(a) => {
                        // 1.0 / va
                        e.mov_imm32(X9, 1.0f32.to_bits());
                        e.dup_4s_gp(31, X9); // V31 = splat(1.0) — but V31 isn't ideal...
                        // Use vd as temp
                        e.mov_imm32(X9, 1.0f32.to_bits());
                        e.dup_4s_gp(vd, X9);
                        let va = get_v128_operand(e, ctx, *a, &depends_on_n, &v_results, 30);
                        e.fdiv_4s(vd, vd, va);
                    }
                    Inst::Sqrt(a) => {
                        let va = get_v128_operand(e, ctx, *a, &depends_on_n, &v_results, 30);
                        e.fsqrt_4s(vd, va);
                    }
                    Inst::Add(a, b) => {
                        let va = get_v128_operand(e, ctx, *a, &depends_on_n, &v_results, 30);
                        let vb = get_v128_operand(e, ctx, *b, &depends_on_n, &v_results, 31);
                        e.fadd_4s(vd, va, vb);
                    }
                    Inst::Mul(a, b) => {
                        let va = get_v128_operand(e, ctx, *a, &depends_on_n, &v_results, 30);
                        let vb = get_v128_operand(e, ctx, *b, &depends_on_n, &v_results, 31);
                        e.fmul_4s(vd, va, vb);
                    }
                    Inst::Max(a, b) => {
                        let va = get_v128_operand(e, ctx, *a, &depends_on_n, &v_results, 30);
                        let vb = get_v128_operand(e, ctx, *b, &depends_on_n, &v_results, 31);
                        e.fmax_4s(vd, va, vb);
                    }
                    Inst::CmpLt(a, b) => {
                        // result = a < b ? 1.0 : 0.0
                        let va = get_v128_operand(e, ctx, *a, &depends_on_n, &v_results, 30);
                        let vb = get_v128_operand(e, ctx, *b, &depends_on_n, &v_results, 31);
                        // FCMLT -> mask
                        e.fcmlt_4s(vd, va, vb);
                        // AND with 1.0 splat
                        e.mov_imm32(X9, 1.0f32.to_bits());
                        e.dup_4s_gp(30, X9);
                        // vd = vd & V30 (bitwise AND to get 1.0 where mask is true)
                        // Use AND V.16B
                        e.emit(0x4E201C00 | (30u32) << 16 | (vd as u32) << 5 | vd as u32);
                    }
                    _ => {
                        // Exp2/Log2 shouldn't appear in SIMD path
                        panic!("unsupported SIMD instruction");
                    }
                }
            }

            // Accumulate: acc[g] += result_v
            let result_v = v_results[&result];
            let acc_slot = acc_v_slots[g];
            // Load acc into V0
            load_q_from_sp_large(e, 0, acc_slot);
            match reduce.op {
                ReduceOp::Sum => {
                    e.fadd_4s(0, 0, result_v);
                }
                ReduceOp::Max => {
                    e.fmax_4s(0, 0, result_v);
                }
            }
            store_q_to_sp_large(e, 0, acc_slot);
        }
    } else {
        // Scalar unrolled path
        for u in 0..unroll {
            load_from_sp_large(e, X9, ni_base_slot);
            e.add_imm(X9, X9, u as u32);
            store_to_sp_large(e, X9, n_dim_slot);

            for (j, inst) in body.iter().enumerate() {
                if depends_on_n[j] {
                    emit_inst(e, ctx, j, inst);
                }
            }

            let result_slot = ctx.inst_slots[&result];
            load_s_from_sp_seq(e, 0, acc_s_slots[u]);
            load_s_from_sp_seq(e, 1, result_slot);
            match reduce.op {
                ReduceOp::Sum => e.fadd_s(0, 0, 1),
                ReduceOp::Max => {
                    e.fcmp_s(1, 0);
                    e.fcsel_s(0, 1, 0, COND_GT);
                }
            }
            store_s_to_sp_seq(e, 0, acc_s_slots[u]);
        }
    }

    // ki++
    load_from_sp_large(e, X9, ki_slot);
    e.add_imm(X9, X9, 1);
    store_to_sp_large(e, X9, ki_slot);
    e.b_label(ki_start);
    e.bind_label(ki_end);

    // k_blk++
    load_from_sp_large(e, X9, k_blk_slot);
    e.add_imm(X9, X9, 1);
    store_to_sp_large(e, X9, k_blk_slot);
    e.b_label(k_blk_start);
    e.bind_label(k_blk_end);

    // Store accumulators to output
    if use_simd {
        // For each SIMD group, store 4 f32s to output
        for g in 0..simd_groups {
            load_q_from_sp_large(e, 0, acc_v_slots[g]);
            // out_ptr = buf_ptr + (row_base + ni_base + g*4) * 4
            load_from_sp_large(e, X3, buf_slot);
            load_from_sp_large(e, X4, row_base_slot);
            load_from_sp_large(e, X5, ni_base_slot);
            e.add_reg(X4, X4, X5);
            e.add_imm(X4, X4, (g * 4) as u32);
            e.lsl_imm(X4, X4, 2); // * 4
            e.add_reg(X3, X3, X4);
            e.str_q_imm(0, X3, 0);
        }
    } else {
        for u in 0..unroll {
            load_s_from_sp_seq(e, 0, acc_s_slots[u]);
            load_from_sp_large(e, X3, buf_slot);
            load_from_sp_large(e, X4, row_base_slot);
            load_from_sp_large(e, X5, ni_base_slot);
            e.add_reg(X4, X4, X5);
            e.add_imm(X4, X4, u as u32);
            e.lsl_imm(X4, X4, 2);
            e.add_reg(X3, X3, X4);
            e.str_s_imm(0, X3, 0);
        }
    }

    // Remainder loop for n_tile % unroll
    {
        let rem_base_slot = ctx.alloc_slot();
        let rem_end_slot = ctx.alloc_slot();
        let rem_i_slot = ctx.alloc_slot();

        // rem_base = ni_grp_count * unroll
        load_from_sp_large(e, X9, ni_grp_count_slot);
        e.mov_imm32(X10, unroll as u32);
        e.mul_reg(X9, X9, X10);
        // Absolute base: n_blk * tn + rem_base
        load_from_sp_large(e, X10, n_blk_slot);
        e.mov_imm32(X11, tn as u32);
        e.mul_reg(X10, X10, X11);
        e.add_reg(X9, X10, X9);
        store_to_sp_large(e, X9, rem_base_slot);

        // rem_end = n_blk * tn + n_tile
        load_from_sp_large(e, X10, n_tile_slot);
        load_from_sp_large(e, X11, n_blk_slot);
        e.mov_imm32(X12, tn as u32);
        e.mul_reg(X11, X11, X12);
        e.add_reg(X10, X11, X10);
        store_to_sp_large(e, X10, rem_end_slot);

        // rem_i = rem_base
        load_from_sp_large(e, X9, rem_base_slot);
        store_to_sp_large(e, X9, rem_i_slot);

        let rem_start_lbl = e.alloc_label();
        let rem_end_lbl = e.alloc_label();
        e.bind_label(rem_start_lbl);
        load_from_sp_large(e, X9, rem_i_slot);
        load_from_sp_large(e, X10, rem_end_slot);
        e.cmp_reg(X9, X10);
        e.b_cond_label(COND_GE, rem_end_lbl);

        // d{n_dim} = rem_i
        store_to_sp_large(e, X9, n_dim_slot);

        // acc = init
        let init_bits = init_val.to_bits();
        e.mov_imm32(X9, init_bits);
        e.fmov_s_from_w(0, X9);
        store_s_to_sp_seq(e, 0, acc_r_slot);

        // K loop for remainder
        let rem_k_slot = ctx.get_or_alloc_dim_slot(reduce_dim);
        e.mov_reg(X9, XZR);
        store_to_sp_large(e, X9, rem_k_slot);

        let rem_k_start = e.alloc_label();
        let rem_k_end = e.alloc_label();
        e.bind_label(rem_k_start);
        load_from_sp_large(e, X9, rem_k_slot);
        load_from_sp_large(e, X10, k_size_slot);
        e.cmp_reg(X9, X10);
        e.b_cond_label(COND_GE, rem_k_end);

        // Emit body (all instructions, scalar)
        for (j, inst) in body.iter().enumerate() {
            emit_inst(e, ctx, j, inst);
        }

        // Accumulate
        let result_slot = ctx.inst_slots[&result];
        load_s_from_sp_seq(e, 0, acc_r_slot);
        load_s_from_sp_seq(e, 1, result_slot);
        match reduce.op {
            ReduceOp::Sum => e.fadd_s(0, 0, 1),
            ReduceOp::Max => {
                e.fcmp_s(1, 0);
                e.fcsel_s(0, 1, 0, COND_GT);
            }
        }
        store_s_to_sp_seq(e, 0, acc_r_slot);

        // k++
        load_from_sp_large(e, X9, rem_k_slot);
        e.add_imm(X9, X9, 1);
        store_to_sp_large(e, X9, rem_k_slot);
        e.b_label(rem_k_start);
        e.bind_label(rem_k_end);

        // Store remainder result
        load_from_sp_large(e, X3, buf_slot);
        load_from_sp_large(e, X4, row_base_slot);
        load_from_sp_large(e, X5, rem_i_slot);
        e.add_reg(X4, X4, X5);
        e.lsl_imm(X4, X4, 2);
        e.add_reg(X3, X3, X4);
        load_s_from_sp_seq(e, 0, acc_r_slot);
        e.str_s_imm(0, X3, 0);

        // rem_i++
        load_from_sp_large(e, X9, rem_i_slot);
        e.add_imm(X9, X9, 1);
        store_to_sp_large(e, X9, rem_i_slot);
        e.b_label(rem_start_lbl);
        e.bind_label(rem_end_lbl);
    }

    // ni_grp++
    load_from_sp_large(e, X9, ni_grp_slot);
    e.add_imm(X9, X9, 1);
    store_to_sp_large(e, X9, ni_grp_slot);
    e.b_label(ni_grp_start);
    e.bind_label(ni_grp_end);

    // n_blk++
    load_from_sp_large(e, X9, n_blk_slot);
    e.add_imm(X9, X9, 1);
    store_to_sp_large(e, X9, n_blk_slot);
    e.b_label(n_blk_start);
    e.bind_label(n_blk_end);

    // mi++
    load_from_sp_large(e, X9, mi_slot);
    e.add_imm(X9, X9, 1);
    store_to_sp_large(e, X9, mi_slot);
    e.b_label(mi_start);
    e.bind_label(mi_end);

    // m_blk++
    load_from_sp_large(e, X9, m_blk_slot);
    e.add_imm(X9, X9, 1);
    store_to_sp_large(e, X9, m_blk_slot);
    e.b_label(m_blk_start);
    e.bind_label(m_blk_end);

    // Close batch loops
    for d in (0..batch_dims).rev() {
        let dim_slot = ctx.dim_slots[&d];
        load_from_sp_large(e, X9, dim_slot);
        e.add_imm(X9, X9, 1);
        store_to_sp_large(e, X9, dim_slot);
        e.b_label(batch_loop_starts[d]);
        e.bind_label(batch_loop_ends[d]);
    }
}

/// Get a NEON V register for an operand. If it's n-dependent, return its V reg.
/// If n-invariant, splat the scalar value from the stack into `scratch_v`.
fn get_v128_operand(
    e: &mut ArmEmitter,
    ctx: &CodegenCtx,
    idx: usize,
    depends_on_n: &[bool],
    v_results: &HashMap<usize, u8>,
    scratch_v: u8,
) -> u8 {
    if depends_on_n[idx] {
        v_results[&idx]
    } else {
        // Load scalar from stack, broadcast to V register
        let slot = ctx.inst_slots[&idx];
        load_s_from_sp_seq(e, 0, slot); // S0 = scalar
        e.fmov_w_from_s(X9, 0);         // X9 = bits
        e.dup_4s_gp(scratch_v, X9);      // V{scratch} = splat
        scratch_v
    }
}

/// STR Q reg, [SP, #offset]  (store 128-bit to stack) — single instruction
fn store_q_to_sp(reg: u8, offset: u32) -> u32 {
    assert!(offset % 16 == 0 && offset / 16 < 4096);
    0x3D800000 | ((offset / 16) << 10) | (31 << 5) | reg as u32
}

/// LDR Q reg, [SP, #offset]  (load 128-bit from stack) — single instruction
fn load_q_from_sp(reg: u8, offset: u32) -> u32 {
    assert!(offset % 16 == 0 && offset / 16 < 4096);
    0x3DC00000 | ((offset / 16) << 10) | (31 << 5) | reg as u32
}

/// STR Q reg, [SP, #offset] — handles large offsets using X8 as scratch
fn store_q_to_sp_large(e: &mut ArmEmitter, reg: u8, offset: u32) {
    if offset % 16 == 0 && offset / 16 < 4096 {
        e.emit(store_q_to_sp(reg, offset));
    } else {
        e.mov_imm32(X8, offset);
        e.emit(0x8B2863E8); // ADD X8, SP, X8, UXTX
        // STR Q{reg}, [X8]
        e.emit(0x3D800000 | (X8 as u32) << 5 | reg as u32);
    }
}

/// LDR Q reg, [SP, #offset] — handles large offsets using X8 as scratch
fn load_q_from_sp_large(e: &mut ArmEmitter, reg: u8, offset: u32) {
    if offset % 16 == 0 && offset / 16 < 4096 {
        e.emit(load_q_from_sp(reg, offset));
    } else {
        e.mov_imm32(X8, offset);
        e.emit(0x8B2863E8); // ADD X8, SP, X8, UXTX
        // LDR Q{reg}, [X8]
        e.emit(0x3DC00000 | (X8 as u32) << 5 | reg as u32);
    }
}

/// Emit pad: zero-fill output, then copy input with offsets.
fn emit_pad(
    e: &mut ArmEmitter,
    ctx: &mut CodegenCtx,
    buf: usize,
    input_buf: usize,
    output_shape: &[Dim],
    input_shape: &[Dim],
    padding: &[(usize, usize)],
) {
    let buf_slot = ctx.buf_ptrs[&buf];
    let input_slot = ctx.buf_ptrs[&input_buf];

    // Zero-fill output buffer
    let out_size_slot = ctx.alloc_slot();
    emit_dim_to_reg(e, &Dim::product(output_shape), ctx, X9);
    store_to_sp_large(e, X9, out_size_slot);

    load_from_sp_large(e, X3, buf_slot); // out ptr
    load_from_sp_large(e, X10, out_size_slot);
    e.lsl_imm(X10, X10, 2); // byte count
    e.add_reg(X10, X3, X10); // end ptr

    let zero_start = e.alloc_label();
    let zero_end = e.alloc_label();
    e.bind_label(zero_start);
    e.cmp_reg(X3, X10);
    e.b_cond_label(COND_GE, zero_end);
    // STR WZR, [X3], #4  (post-increment)
    e.emit(0xB8004400 | (X3 as u32) << 5 | XZR as u32); // STR W31, [X3], #4
    e.b_label(zero_start);
    e.bind_label(zero_end);

    // Copy input into padded region
    // Simple flat copy: for i in 0..input_size, compute padded index and copy
    let in_size = Dim::product(input_shape);
    let in_strides = Dim::strides(input_shape);
    let out_strides = Dim::strides(output_shape);
    let ndim = input_shape.len();

    let copy_i_slot = ctx.alloc_slot();
    let copy_size_slot = ctx.alloc_slot();

    emit_dim_to_reg(e, &in_size, ctx, X9);
    store_to_sp_large(e, X9, copy_size_slot);
    e.mov_reg(X9, XZR);
    store_to_sp_large(e, X9, copy_i_slot);

    let copy_start = e.alloc_label();
    let copy_end = e.alloc_label();
    e.bind_label(copy_start);
    load_from_sp_large(e, X9, copy_i_slot);
    load_from_sp_large(e, X10, copy_size_slot);
    e.cmp_reg(X9, X10);
    e.b_cond_label(COND_GE, copy_end);

    // Decompose flat index into per-dim coords and compute output flat index
    // X9 = flat_in, we compute X10 = flat_out
    // For each dim d: coord_d = (flat_in / in_stride_d) % in_shape_d
    //                 out_flat += (coord_d + pad_lo_d) * out_stride_d
    e.mov_reg(X10, XZR); // out_flat = 0
    for d in 0..ndim {
        // coord_d
        load_from_sp_large(e, X9, copy_i_slot);
        emit_dim_to_reg(e, &in_strides[d], ctx, X11);
        e.sdiv(X12, X9, X11);
        emit_dim_to_reg(e, &input_shape[d], ctx, X11);
        e.sdiv(X13, X12, X11);
        e.msub(X12, X13, X11, X12); // coord_d = X12

        // out_flat += (coord_d + pad_lo) * out_stride
        let pad_lo = padding[d].0;
        if pad_lo > 0 {
            e.add_imm(X12, X12, pad_lo as u32);
        }
        emit_dim_to_reg(e, &out_strides[d], ctx, X11);
        e.madd(X10, X12, X11, X10);
    }

    // Copy: out[out_flat] = in[i]
    load_from_sp_large(e, X3, input_slot);
    load_from_sp_large(e, X4, copy_i_slot);
    e.ldr_s_reg_scaled(0, X3, X4); // S0 = in[i]
    load_from_sp_large(e, X3, buf_slot);
    e.str_s_reg_scaled(0, X3, X10); // out[out_flat] = S0

    // i++
    load_from_sp_large(e, X9, copy_i_slot);
    e.add_imm(X9, X9, 1);
    store_to_sp_large(e, X9, copy_i_slot);
    e.b_label(copy_start);
    e.bind_label(copy_end);
}

// Reuse the N-dependence analysis from the WASM backend
/// Compute which instructions depend on a given dimension (used for both
/// N-dependence and K-dependence analysis).
fn compute_dim_dependence(body: &[Inst], dim: usize) -> Vec<bool> {
    let mut dep = vec![false; body.len()];
    for (j, inst) in body.iter().enumerate() {
        dep[j] = match inst {
            Inst::Load { index: Index::Strided { parts, .. }, .. } => {
                parts.iter().any(|(d, _)| *d == dim)
            }
            Inst::Load { index: Index::Flat, .. } => true,
            Inst::Const(_) => false,
            Inst::DimVar(d) => *d == dim,
            Inst::Neg(a) | Inst::Recip(a) | Inst::Exp2(a) | Inst::Log2(a) | Inst::Sqrt(a) => dep[*a],
            Inst::Add(a, b) | Inst::Mul(a, b) | Inst::Max(a, b) | Inst::CmpLt(a, b) => dep[*a] || dep[*b],
        };
    }
    dep
}

fn compute_n_dependence(body: &[Inst], n_dim: usize) -> Vec<bool> {
    let mut dep = vec![false; body.len()];
    for (j, inst) in body.iter().enumerate() {
        dep[j] = match inst {
            Inst::Load { index: Index::Strided { parts, .. }, .. } => {
                parts.iter().any(|(dim, _)| *dim == n_dim)
            }
            Inst::Load { index: Index::Flat, .. } => true,
            Inst::Const(_) => false,
            Inst::DimVar(d) => *d == n_dim,
            Inst::Neg(a) | Inst::Recip(a) | Inst::Exp2(a) | Inst::Log2(a) | Inst::Sqrt(a) => dep[*a],
            Inst::Add(a, b) | Inst::Mul(a, b) | Inst::Max(a, b) | Inst::CmpLt(a, b) => dep[*a] || dep[*b],
        };
    }
    dep
}

fn can_simd_tiled_loop(body: &[Inst], depends_on_n: &[bool], n_dim: usize) -> bool {
    body.iter().enumerate().all(|(j, inst)| {
        if !depends_on_n[j] {
            return true;
        }
        match inst {
            Inst::Load { index: Index::Strided { parts, .. }, .. } => {
                parts.iter().all(|(dim, stride)| *dim != n_dim || stride.is_one())
            }
            Inst::Load { index: Index::Flat, .. } => true,
            Inst::Const(_) => true,
            Inst::DimVar(_) => false,
            Inst::Neg(_) | Inst::Recip(_) | Inst::Sqrt(_) => true,
            Inst::Add(_, _) | Inst::Mul(_, _) | Inst::Max(_, _) | Inst::CmpLt(_, _) => true,
            Inst::Exp2(_) | Inst::Log2(_) => false,
        }
    })
}

fn collect_params(dim: &Dim, params: &mut Vec<String>) {
    match dim {
        Dim::Param(name) => {
            if !params.contains(name) {
                params.push(name.clone());
            }
        }
        Dim::Add(a, b) | Dim::Mul(a, b) | Dim::Div(a, b) | Dim::Sub(a, b) => {
            collect_params(a, params);
            collect_params(b, params);
        }
        Dim::Lit(_) => {}
    }
}
