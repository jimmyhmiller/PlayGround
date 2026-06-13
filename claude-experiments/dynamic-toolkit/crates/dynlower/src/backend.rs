use dynasm::Arch;
use dynasm::arm64::Arm64;
use dynasm::arm64::inst::*;
use dynasm::arm64::reg::*;
use dynasm::arm64::reloc::Arm64RelocKind;
use dynasm::buffer::{CodeBuffer, Label};
use dynasm::x86_64::cond::Condition as X64Cond;
use dynasm::x86_64::inst::X64Inst;
use dynasm::x86_64::reg::{
    R8, R9, R10, R11, R12, R13, R14, R15, RAX, RBP, RBX, RCX, RDI, RDX, RSI, RSP, X64Reg,
};
use dynasm::x86_64::{X64, X64RelocKind};
use dynexec::{FrameSlotAccess, FrameSlotBase};
use dynir::ir::CmpOp;
use dynir::types::Type;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MachineRegClass {
    Gp,
    Fp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MachineReg {
    pub class: MachineRegClass,
    pub index: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MachineLocation {
    Reg(MachineReg),
    FrameSlot(FrameSlotAccess),
    StackArg(i32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MachineGpBinOp {
    Add,
    Sub,
    Mul,
    SDiv,
    UDiv,
    And,
    Or,
    Xor,
    Shl,
    LShr,
    AShr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MachineFpBinOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MachineWordSize {
    W32,
    W64,
}

pub trait LoweringBackend {
    type Arch: Arch;

    fn allocatable_gp() -> &'static [u8] {
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    }

    fn allocatable_fp() -> &'static [u8] {
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        ]
    }

    fn caller_saved_gp() -> &'static [u8] {
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    }

    fn caller_saved_fp() -> &'static [u8] {
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        ]
    }

    fn call_arg_gp(slot: usize) -> MachineReg {
        MachineReg {
            class: MachineRegClass::Gp,
            index: slot as u8,
        }
    }

    /// Number of GP argument registers in the platform C ABI (AAPCS / SysV).
    /// Calls to `extern "C"` functions must place args beyond this count on
    /// the stack, regardless of the internal CC's wider `register_arg_limit`.
    /// The register mapping for slots below this limit is `call_arg_gp`,
    /// which already matches the C ABI for the low slots on both arches.
    /// Default is AAPCS (8 GP arg registers, X0–X7).
    fn c_abi_gp_arg_limit() -> usize {
        8
    }

    fn emit_prologue(buf: &mut CodeBuffer<Self::Arch>) -> usize;
    fn emit_epilogue(buf: &mut CodeBuffer<Self::Arch>, patch_offsets: &mut Vec<usize>);
    fn emit_frame_size_patch(
        buf: &mut CodeBuffer<Self::Arch>,
        prologue_offset: usize,
        epilogue_offsets: &[usize],
        frame_size: i32,
    );

    fn emit_gp_move(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, src: MachineReg);
    fn emit_stack_pointer_to_gp(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg);
    fn emit_fp_to_gp_move(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, src: MachineReg);
    fn emit_gp_to_fp_move(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, src: MachineReg);
    fn emit_store_gp_to_frame(
        buf: &mut CodeBuffer<Self::Arch>,
        src: MachineReg,
        slot: FrameSlotAccess,
    );
    fn emit_load_gp_from_frame(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        slot: FrameSlotAccess,
    );
    fn emit_store_fp_to_frame(
        buf: &mut CodeBuffer<Self::Arch>,
        src: MachineReg,
        slot: FrameSlotAccess,
    );
    fn emit_load_fp_from_frame(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        slot: FrameSlotAccess,
    );
    fn emit_store_gp_to_stack_arg(buf: &mut CodeBuffer<Self::Arch>, src: MachineReg, offset: i32);

    fn emit_gp_binop(
        buf: &mut CodeBuffer<Self::Arch>,
        op: MachineGpBinOp,
        dst: MachineReg,
        lhs: MachineReg,
        rhs: MachineReg,
        size: MachineWordSize,
    );

    /// Compute the address of a frame slot: `dst = base_reg + offset`.
    fn emit_lea_frame_slot(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        slot: FrameSlotAccess,
    );

    fn emit_fp_binop(
        buf: &mut CodeBuffer<Self::Arch>,
        op: MachineFpBinOp,
        dst: MachineReg,
        lhs: MachineReg,
        rhs: MachineReg,
    );

    fn emit_icmp_set(
        buf: &mut CodeBuffer<Self::Arch>,
        op: CmpOp,
        dst: MachineReg,
        lhs: MachineReg,
        rhs: MachineReg,
        size: MachineWordSize,
    );

    fn emit_fcmp_set(
        buf: &mut CodeBuffer<Self::Arch>,
        op: CmpOp,
        dst: MachineReg,
        lhs: MachineReg,
        rhs: MachineReg,
    );

    fn emit_gp_select(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        cond: MachineReg,
        when_true: MachineReg,
        when_false: MachineReg,
        size: MachineWordSize,
    );

    fn emit_fp_select(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        cond: MachineReg,
        when_true: MachineReg,
        when_false: MachineReg,
    );

    fn emit_load_gp(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        base: MachineReg,
        offset: i32,
        size: MachineWordSize,
    );

    fn emit_load_fp(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        base: MachineReg,
        offset: i32,
    );

    fn emit_store_gp(
        buf: &mut CodeBuffer<Self::Arch>,
        src: MachineReg,
        base: MachineReg,
        offset: i32,
        size: MachineWordSize,
    );

    fn emit_store_fp(
        buf: &mut CodeBuffer<Self::Arch>,
        src: MachineReg,
        base: MachineReg,
        offset: i32,
    );
    fn emit_gp_neg(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        src: MachineReg,
        size: MachineWordSize,
    );
    fn emit_gp_not(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        src: MachineReg,
        size: MachineWordSize,
    );
    fn emit_fp_neg(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, src: MachineReg);
    fn emit_sign_extend(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        src: MachineReg,
        src_ty: Type,
        target_ty: Type,
    );
    fn emit_zero_extend(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        src: MachineReg,
        src_ty: Type,
        target_ty: Type,
    );
    fn emit_trunc(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        src: MachineReg,
        target_ty: Type,
    );
    fn emit_int_to_float(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        src: MachineReg,
        src_ty: Type,
    );
    fn emit_float_to_int(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, src: MachineReg);
    fn emit_mov_imm(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, value: u64);
    fn emit_f64_const(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, bits: u64);
    fn emit_load_incoming_stack_arg(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        incoming_offset: i32,
    );
    fn emit_cmp_gp_imm(buf: &mut CodeBuffer<Self::Arch>, reg: MachineReg, imm: u64);
    fn emit_extract_payload(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        src: MachineReg,
        has_unboxed_float: bool,
        payload_bits: u8,
    );
    fn emit_is_tag(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        src: MachineReg,
        has_unboxed_float: bool,
        payload_bits: u8,
        tag_mask: u64,
        expected_tag: u64,
    );
    fn emit_make_tagged(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        payload: MachineReg,
        has_unboxed_float: bool,
        payload_bits: u8,
        encoded_tag_pattern: u64,
        tag: u64,
    );
    fn emit_tag_of(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        src: MachineReg,
        has_unboxed_float: bool,
        payload_bits: u8,
        tag_mask: u64,
    );
    fn emit_call_safepoint_handler(buf: &mut CodeBuffer<Self::Arch>, handler: u64, frame_size: u64);

    fn bind_label(buf: &mut CodeBuffer<Self::Arch>, label: Label);
    fn emit_branch_to_label(buf: &mut CodeBuffer<Self::Arch>, label: Label);
    fn emit_cbz_to_label(buf: &mut CodeBuffer<Self::Arch>, reg: MachineReg, label: Label);
    fn emit_cbnz_to_label(buf: &mut CodeBuffer<Self::Arch>, reg: MachineReg, label: Label);
    fn emit_branch_eq_to_label(buf: &mut CodeBuffer<Self::Arch>, label: Label);
    fn emit_call_reg(buf: &mut CodeBuffer<Self::Arch>, reg: MachineReg);
    fn emit_return_gp(buf: &mut CodeBuffer<Self::Arch>, src: MachineReg);
    fn emit_return_fp_bits(buf: &mut CodeBuffer<Self::Arch>, src: MachineReg);
    fn emit_stack_adjust(buf: &mut CodeBuffer<Self::Arch>, amount: i32);
    fn emit_trap(buf: &mut CodeBuffer<Self::Arch>);
}

pub struct Arm64Backend;

#[allow(dead_code)]
pub struct X64Backend;

impl Arm64Backend {
    fn size_to_regsize(size: MachineWordSize) -> RegSize {
        match size {
            MachineWordSize::W32 => RegSize::W32,
            MachineWordSize::W64 => RegSize::X64,
        }
    }

    fn emit_mov_imm64(buf: &mut CodeBuffer<Arm64>, rd: Arm64Reg, value: u64) {
        for inst in Arm64Inst::mov_imm64(rd, value) {
            buf.emit(inst);
        }
    }

    fn cmpop_to_cond(op: CmpOp) -> Arm64Cond {
        match op {
            CmpOp::Eq => Arm64Cond::EQ,
            CmpOp::Ne => Arm64Cond::NE,
            CmpOp::Slt => Arm64Cond::LT,
            CmpOp::Sle => Arm64Cond::LE,
            CmpOp::Sgt => Arm64Cond::GT,
            CmpOp::Sge => Arm64Cond::GE,
            CmpOp::Ult => Arm64Cond::CC,
            CmpOp::Ule => Arm64Cond::LS,
            CmpOp::Ugt => Arm64Cond::HI,
            CmpOp::Uge => Arm64Cond::CS,
        }
    }

    pub fn gp(index: u8) -> MachineReg {
        MachineReg {
            class: MachineRegClass::Gp,
            index,
        }
    }

    #[allow(dead_code)]
    pub fn fp(index: u8) -> MachineReg {
        MachineReg {
            class: MachineRegClass::Fp,
            index,
        }
    }

    pub fn gp_hw(reg: MachineReg, size: RegSize) -> Arm64Reg {
        debug_assert_eq!(reg.class, MachineRegClass::Gp);
        Arm64Reg::new(reg.index, size)
    }

    pub fn fp_hw(reg: MachineReg) -> Arm64Reg {
        debug_assert_eq!(reg.class, MachineRegClass::Fp);
        Arm64Reg::new(reg.index, RegSize::X64)
    }

    pub fn emit_prologue(buf: &mut CodeBuffer<Arm64>) -> usize {
        // Save AAPCS callee-saved GP regs X19-X28. The body clobbers X27/X28
        // unconditionally (GcLiteral lowering, emit_call, prologue scratch),
        // and the batch_lower path may also assign live values to X19-X26.
        // Save X28's caller value FIRST (via X16 scratch) so the upcoming
        // `mov X28, SP` doesn't lose it.
        buf.emit(Arm64Inst::mov(X16, SP));
        buf.emit(Arm64Inst::stp(X19, X20, SP, -16, StpMode::PreIndex));
        buf.emit(Arm64Inst::stp(X21, X22, SP, -16, StpMode::PreIndex));
        buf.emit(Arm64Inst::stp(X23, X24, SP, -16, StpMode::PreIndex));
        buf.emit(Arm64Inst::stp(X25, X26, SP, -16, StpMode::PreIndex));
        buf.emit(Arm64Inst::stp(X27, X28, SP, -16, StpMode::PreIndex));
        // X28 = caller_SP (used by emit_load_incoming_stack_arg).
        buf.emit(Arm64Inst::mov(X28, X16));
        // Reserve 2 instructions for frame size adjustment (patched later).
        // This supports frame sizes > 4095 bytes by splitting into hi/lo parts.
        let patch_offset = buf.emit(Arm64Inst::sub_imm(SP, SP, 16));
        buf.emit(Arm64Inst::sub_imm(SP, SP, 0)); // placeholder for low bits
        buf.emit(Arm64Inst::stp(X29, X30, SP, 0, StpMode::SignedOffset));
        buf.emit(Arm64Inst::mov(X29, SP));
        // Zero the frame's locals region [FP+16, FP+frame_size) — every
        // spill slot and stack slot. REQUIRED FOR GC SOUNDNESS: safepoint
        // records over-approximate (emission-order liveness +
        // `record_materialized_gc_spill_slots` keeps every ever-written
        // GC-capable slot recorded), so a recorded slot whose defining
        // store sits on a branch the execution didn't take would otherwise
        // hold leftover stack junk. If that junk bit-patterns like a
        // NaN-boxed heap pointer, a moving collection chases it and reads
        // garbage ("to-space exhausted", type_id corruption). Zeroed slots
        // decode as non-pointers and are skipped by the root walk.
        //
        // The byte count is unknown until lowering finishes, so emit a
        // fixed-shape loop with a movz/movk placeholder patched by
        // `emit_frame_size_patch` (count = frame_size - 16, always a
        // multiple of 16). X16/X17 are scratch (never argument or
        // allocatable registers at function entry).
        buf.emit(Arm64Inst::movz(X16, 0, 0)); // patched: zero_bytes[15:0]
        buf.emit(Arm64Inst::movk(X16, 0, 16)); // patched: zero_bytes[31:16]
        buf.emit(Arm64Inst::add_imm(X17, X29, 16));
        buf.emit(Arm64Inst::cbz(X16, 4)); // empty locals region: skip loop
        buf.emit(Arm64Inst::stp(XZR, XZR, X17, 16, StpMode::PostIndex));
        buf.emit(Arm64Inst::SubsImm {
            sf: 1,
            sh: 0,
            imm12: 16,
            rn: X16,
            rd: X16,
        });
        buf.emit(Arm64Inst::cbnz(X16, -2)); // back to the stp
        patch_offset
    }

    pub fn emit_epilogue(buf: &mut CodeBuffer<Arm64>, patch_offsets: &mut Vec<usize>) {
        buf.emit(Arm64Inst::mov(SP, X29));
        buf.emit(Arm64Inst::ldp(X29, X30, SP, 0, LdpMode::SignedOffset));
        // Reserve 2 instructions for frame size restore (patched later).
        let add_offset = buf.emit(Arm64Inst::add_imm(SP, SP, 16));
        buf.emit(Arm64Inst::add_imm(SP, SP, 0)); // placeholder for low bits
        patch_offsets.push(add_offset);
        // Restore AAPCS callee-saved GP regs in reverse order.
        buf.emit(Arm64Inst::ldp(X27, X28, SP, 16, LdpMode::PostIndex));
        buf.emit(Arm64Inst::ldp(X25, X26, SP, 16, LdpMode::PostIndex));
        buf.emit(Arm64Inst::ldp(X23, X24, SP, 16, LdpMode::PostIndex));
        buf.emit(Arm64Inst::ldp(X21, X22, SP, 16, LdpMode::PostIndex));
        buf.emit(Arm64Inst::ldp(X19, X20, SP, 16, LdpMode::PostIndex));
        buf.emit(Arm64Inst::ret());
    }

    pub fn emit_frame_size_patch(
        buf: &mut CodeBuffer<Arm64>,
        prologue_offset: usize,
        epilogue_offsets: &[usize],
        frame_size: i32,
    ) {
        let hi = (frame_size >> 12) & 0xFFF;
        let lo = frame_size & 0xFFF;

        // Prologue: sub SP, SP, #hi, LSL #12 ; sub SP, SP, #lo
        let sub_hi = Arm64Inst::SubImm {
            sf: 1,
            sh: 1,
            imm12: hi,
            rn: SP,
            rd: SP,
        };
        let sub_lo = Arm64Inst::SubImm {
            sf: 1,
            sh: 0,
            imm12: lo,
            rn: SP,
            rd: SP,
        };
        buf.patch_bytes(prologue_offset, &sub_hi.encode().to_le_bytes());
        buf.patch_bytes(prologue_offset + 4, &sub_lo.encode().to_le_bytes());

        // Patch the locals-zeroing loop count (see emit_prologue). The
        // movz/movk pair sits 2 instructions after the frame-size subs
        // (stp X29,X30 + mov X29,SP are between). Count = everything above
        // the FP/LR pair; frame_size is 16-aligned so this is too.
        let zero_bytes = (frame_size - 16).max(0) as u32;
        let movz = Arm64Inst::movz(X16, (zero_bytes & 0xFFFF) as u16, 0);
        let movk = Arm64Inst::movk(X16, (zero_bytes >> 16) as u16, 16);
        buf.patch_bytes(prologue_offset + 16, &movz.encode().to_le_bytes());
        buf.patch_bytes(prologue_offset + 20, &movk.encode().to_le_bytes());

        // Epilogue: add SP, SP, #hi, LSL #12 ; add SP, SP, #lo
        let add_hi = Arm64Inst::AddImm {
            sf: 1,
            sh: 1,
            imm12: hi,
            rn: SP,
            rd: SP,
        };
        let add_lo = Arm64Inst::AddImm {
            sf: 1,
            sh: 0,
            imm12: lo,
            rn: SP,
            rd: SP,
        };
        let add_hi_bytes = add_hi.encode().to_le_bytes();
        let add_lo_bytes = add_lo.encode().to_le_bytes();
        for &offset in epilogue_offsets {
            buf.patch_bytes(offset, &add_hi_bytes);
            buf.patch_bytes(offset + 4, &add_lo_bytes);
        }
    }

    pub fn emit_gp_move(buf: &mut CodeBuffer<Arm64>, dst: MachineReg, src: MachineReg) {
        buf.emit(Arm64Inst::mov(
            Self::gp_hw(dst, RegSize::X64),
            Self::gp_hw(src, RegSize::X64),
        ));
    }

    pub fn emit_stack_pointer_to_gp(buf: &mut CodeBuffer<Arm64>, dst: MachineReg) {
        buf.emit(Arm64Inst::mov(Self::gp_hw(dst, RegSize::X64), SP));
    }

    pub fn emit_fp_to_gp_move(buf: &mut CodeBuffer<Arm64>, dst: MachineReg, src: MachineReg) {
        buf.emit(Arm64Inst::fmov_fp_to_gp(
            Self::gp_hw(dst, RegSize::X64),
            Self::fp_hw(src),
        ));
    }

    pub fn emit_gp_to_fp_move(buf: &mut CodeBuffer<Arm64>, dst: MachineReg, src: MachineReg) {
        buf.emit(Arm64Inst::fmov_gp_to_fp(
            Self::fp_hw(dst),
            Self::gp_hw(src, RegSize::X64),
        ));
    }

    pub(crate) fn slot_base_reg(slot: FrameSlotAccess) -> Arm64Reg {
        match slot.base {
            FrameSlotBase::FramePointer => X29,
            FrameSlotBase::StackPointer => SP,
        }
    }

    pub fn emit_store_gp_to_frame(
        buf: &mut CodeBuffer<Arm64>,
        src: MachineReg,
        slot: FrameSlotAccess,
    ) {
        debug_assert!(slot.offset >= 0 && slot.offset % 8 == 0);
        buf.emit(Arm64Inst::str(
            Self::gp_hw(src, RegSize::X64),
            Self::slot_base_reg(slot),
            slot.offset,
        ));
    }

    pub fn emit_load_gp_from_frame(
        buf: &mut CodeBuffer<Arm64>,
        dst: MachineReg,
        slot: FrameSlotAccess,
    ) {
        debug_assert!(slot.offset >= 0 && slot.offset % 8 == 0);
        buf.emit(Arm64Inst::ldr(
            Self::gp_hw(dst, RegSize::X64),
            Self::slot_base_reg(slot),
            slot.offset,
        ));
    }

    pub fn emit_store_fp_to_frame(
        buf: &mut CodeBuffer<Arm64>,
        src: MachineReg,
        slot: FrameSlotAccess,
    ) {
        debug_assert!(slot.offset >= 0 && slot.offset % 8 == 0);
        buf.emit(Arm64Inst::str_fp(
            Self::fp_hw(src),
            Self::slot_base_reg(slot),
            slot.offset,
        ));
    }

    pub fn emit_load_fp_from_frame(
        buf: &mut CodeBuffer<Arm64>,
        dst: MachineReg,
        slot: FrameSlotAccess,
    ) {
        debug_assert!(slot.offset >= 0 && slot.offset % 8 == 0);
        buf.emit(Arm64Inst::ldr_fp(
            Self::fp_hw(dst),
            Self::slot_base_reg(slot),
            slot.offset,
        ));
    }

    pub fn emit_store_gp_to_stack_arg(buf: &mut CodeBuffer<Arm64>, src: MachineReg, offset: i32) {
        buf.emit(Arm64Inst::str(Self::gp_hw(src, RegSize::X64), SP, offset));
    }

    pub fn emit_cmp_imm_zero(buf: &mut CodeBuffer<Arm64>, reg: MachineReg) {
        buf.emit(Arm64Inst::cmp_imm(Self::gp_hw(reg, RegSize::W32), 0));
    }

    pub fn emit_cbz(buf: &mut CodeBuffer<Arm64>, reg: MachineReg) -> usize {
        buf.emit(Arm64Inst::cbz(Self::gp_hw(reg, RegSize::W32), 0))
    }

    pub fn emit_cbnz(buf: &mut CodeBuffer<Arm64>, reg: MachineReg) -> usize {
        buf.emit(Arm64Inst::cbnz(Self::gp_hw(reg, RegSize::W32), 0))
    }

    pub fn emit_branch(buf: &mut CodeBuffer<Arm64>) -> usize {
        buf.emit(Arm64Inst::b(0))
    }

    pub fn emit_branch_eq(buf: &mut CodeBuffer<Arm64>) -> usize {
        buf.emit(Arm64Inst::b_cond(Arm64Cond::EQ, 0))
    }

    pub fn emit_call_reg(buf: &mut CodeBuffer<Arm64>, reg: MachineReg) {
        buf.emit(Arm64Inst::blr(Self::gp_hw(reg, RegSize::X64)));
    }

    pub fn emit_return_gp(buf: &mut CodeBuffer<Arm64>, src: MachineReg) {
        if src.index != 0 {
            Self::emit_gp_move(buf, Self::gp(0), src);
        }
    }

    pub fn emit_return_fp_bits(buf: &mut CodeBuffer<Arm64>, src: MachineReg) {
        Self::emit_fp_to_gp_move(buf, Self::gp(0), src);
    }

    pub fn bind_label(buf: &mut CodeBuffer<Arm64>, label: Label) {
        buf.bind_label(label);
    }

    pub fn emit_branch_to_label(buf: &mut CodeBuffer<Arm64>, label: Label) {
        let offset = Self::emit_branch(buf);
        buf.add_reloc(offset, label, Arm64RelocKind::Branch26);
    }

    pub fn emit_cbz_to_label(buf: &mut CodeBuffer<Arm64>, reg: MachineReg, label: Label) {
        let offset = Self::emit_cbz(buf, reg);
        buf.add_reloc(offset, label, Arm64RelocKind::Cond19);
    }

    pub fn emit_cbnz_to_label(buf: &mut CodeBuffer<Arm64>, reg: MachineReg, label: Label) {
        let offset = Self::emit_cbnz(buf, reg);
        buf.add_reloc(offset, label, Arm64RelocKind::Cond19);
    }

    pub fn emit_branch_eq_to_label(buf: &mut CodeBuffer<Arm64>, label: Label) {
        let offset = Self::emit_branch_eq(buf);
        buf.add_reloc(offset, label, Arm64RelocKind::Cond19);
    }
}

impl LoweringBackend for Arm64Backend {
    type Arch = Arm64;

    fn emit_prologue(buf: &mut CodeBuffer<Self::Arch>) -> usize {
        Arm64Backend::emit_prologue(buf)
    }

    fn emit_epilogue(buf: &mut CodeBuffer<Self::Arch>, patch_offsets: &mut Vec<usize>) {
        Arm64Backend::emit_epilogue(buf, patch_offsets);
    }

    fn emit_frame_size_patch(
        buf: &mut CodeBuffer<Self::Arch>,
        prologue_offset: usize,
        epilogue_offsets: &[usize],
        frame_size: i32,
    ) {
        Arm64Backend::emit_frame_size_patch(buf, prologue_offset, epilogue_offsets, frame_size);
    }

    fn emit_lea_frame_slot(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        slot: FrameSlotAccess,
    ) {
        let base = Arm64Backend::slot_base_reg(slot);
        if slot.offset == 0 {
            buf.emit(Arm64Inst::mov(Arm64Backend::gp_hw(dst, RegSize::X64), base));
        } else {
            buf.emit(Arm64Inst::add_imm(
                Arm64Backend::gp_hw(dst, RegSize::X64),
                base,
                slot.offset,
            ));
        }
    }

    fn emit_gp_move(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, src: MachineReg) {
        Arm64Backend::emit_gp_move(buf, dst, src);
    }

    fn emit_stack_pointer_to_gp(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg) {
        Arm64Backend::emit_stack_pointer_to_gp(buf, dst);
    }

    fn emit_fp_to_gp_move(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, src: MachineReg) {
        Arm64Backend::emit_fp_to_gp_move(buf, dst, src);
    }

    fn emit_gp_to_fp_move(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, src: MachineReg) {
        Arm64Backend::emit_gp_to_fp_move(buf, dst, src);
    }

    fn emit_store_gp_to_frame(
        buf: &mut CodeBuffer<Self::Arch>,
        src: MachineReg,
        slot: FrameSlotAccess,
    ) {
        Arm64Backend::emit_store_gp_to_frame(buf, src, slot);
    }

    fn emit_load_gp_from_frame(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        slot: FrameSlotAccess,
    ) {
        Arm64Backend::emit_load_gp_from_frame(buf, dst, slot);
    }

    fn emit_store_fp_to_frame(
        buf: &mut CodeBuffer<Self::Arch>,
        src: MachineReg,
        slot: FrameSlotAccess,
    ) {
        Arm64Backend::emit_store_fp_to_frame(buf, src, slot);
    }

    fn emit_load_fp_from_frame(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        slot: FrameSlotAccess,
    ) {
        Arm64Backend::emit_load_fp_from_frame(buf, dst, slot);
    }

    fn emit_store_gp_to_stack_arg(buf: &mut CodeBuffer<Self::Arch>, src: MachineReg, offset: i32) {
        Arm64Backend::emit_store_gp_to_stack_arg(buf, src, offset);
    }

    fn emit_gp_binop(
        buf: &mut CodeBuffer<Self::Arch>,
        op: MachineGpBinOp,
        dst: MachineReg,
        lhs: MachineReg,
        rhs: MachineReg,
        size: MachineWordSize,
    ) {
        let size = Arm64Backend::size_to_regsize(size);
        let rd = Arm64Backend::gp_hw(dst, size);
        let rn = Arm64Backend::gp_hw(lhs, size);
        let rm = Arm64Backend::gp_hw(rhs, size);

        match op {
            MachineGpBinOp::Add => buf.emit(Arm64Inst::add(rd, rn, rm)),
            MachineGpBinOp::Sub => buf.emit(Arm64Inst::sub(rd, rn, rm)),
            MachineGpBinOp::Mul => buf.emit(Arm64Inst::mul(rd, rn, rm)),
            MachineGpBinOp::SDiv => buf.emit(Arm64Inst::sdiv(rd, rn, rm)),
            MachineGpBinOp::UDiv => buf.emit(Arm64Inst::udiv(rd, rn, rm)),
            MachineGpBinOp::And => buf.emit(Arm64Inst::and(rd, rn, rm)),
            MachineGpBinOp::Or => buf.emit(Arm64Inst::orr(rd, rn, rm)),
            MachineGpBinOp::Xor => buf.emit(Arm64Inst::eor(rd, rn, rm)),
            MachineGpBinOp::Shl => buf.emit(Arm64Inst::LslReg {
                sf: rd.sf(),
                rm,
                rn,
                rd,
            }),
            MachineGpBinOp::LShr => buf.emit(Arm64Inst::LsrReg {
                sf: rd.sf(),
                rm,
                rn,
                rd,
            }),
            MachineGpBinOp::AShr => buf.emit(Arm64Inst::AsrReg {
                sf: rd.sf(),
                rm,
                rn,
                rd,
            }),
        };
    }

    fn emit_fp_binop(
        buf: &mut CodeBuffer<Self::Arch>,
        op: MachineFpBinOp,
        dst: MachineReg,
        lhs: MachineReg,
        rhs: MachineReg,
    ) {
        let rd = Arm64Backend::fp_hw(dst);
        let rn = Arm64Backend::fp_hw(lhs);
        let rm = Arm64Backend::fp_hw(rhs);

        match op {
            MachineFpBinOp::Add => buf.emit(Arm64Inst::fadd(rd, rn, rm)),
            MachineFpBinOp::Sub => buf.emit(Arm64Inst::fsub(rd, rn, rm)),
            MachineFpBinOp::Mul => buf.emit(Arm64Inst::fmul(rd, rn, rm)),
            MachineFpBinOp::Div => buf.emit(Arm64Inst::fdiv(rd, rn, rm)),
        };
    }

    fn emit_icmp_set(
        buf: &mut CodeBuffer<Self::Arch>,
        op: CmpOp,
        dst: MachineReg,
        lhs: MachineReg,
        rhs: MachineReg,
        size: MachineWordSize,
    ) {
        let size = Arm64Backend::size_to_regsize(size);
        let rn = Arm64Backend::gp_hw(lhs, size);
        let rm = Arm64Backend::gp_hw(rhs, size);
        buf.emit(Arm64Inst::cmp(rn, rm));
        buf.emit(Arm64Inst::cset(
            Arm64Backend::gp_hw(dst, RegSize::W32),
            Arm64Backend::cmpop_to_cond(op),
        ));
    }

    fn emit_fcmp_set(
        buf: &mut CodeBuffer<Self::Arch>,
        op: CmpOp,
        dst: MachineReg,
        lhs: MachineReg,
        rhs: MachineReg,
    ) {
        buf.emit(Arm64Inst::fcmp_double(
            Arm64Backend::fp_hw(lhs),
            Arm64Backend::fp_hw(rhs),
        ));
        buf.emit(Arm64Inst::cset(
            Arm64Backend::gp_hw(dst, RegSize::W32),
            Arm64Backend::cmpop_to_cond(op),
        ));
    }

    fn emit_gp_select(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        cond: MachineReg,
        when_true: MachineReg,
        when_false: MachineReg,
        size: MachineWordSize,
    ) {
        let size = Arm64Backend::size_to_regsize(size);
        Arm64Backend::emit_cmp_imm_zero(buf, cond);
        buf.emit(Arm64Inst::csel(
            Arm64Backend::gp_hw(dst, size),
            Arm64Backend::gp_hw(when_true, size),
            Arm64Backend::gp_hw(when_false, size),
            Arm64Cond::NE,
        ));
    }

    fn emit_fp_select(
        buf: &mut CodeBuffer<Self::Arch>,
        cond: MachineReg,
        dst: MachineReg,
        when_true: MachineReg,
        when_false: MachineReg,
    ) {
        Arm64Backend::emit_cmp_imm_zero(buf, cond);
        buf.emit(Arm64Inst::fcsel(
            Arm64Backend::fp_hw(dst),
            Arm64Backend::fp_hw(when_true),
            Arm64Backend::fp_hw(when_false),
            Arm64Cond::NE,
        ));
    }

    fn emit_load_gp(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        base: MachineReg,
        offset: i32,
        size: MachineWordSize,
    ) {
        let size = Arm64Backend::size_to_regsize(size);
        let rd = Arm64Backend::gp_hw(dst, size);
        let rd_x64 = Arm64Backend::gp_hw(dst, RegSize::X64);
        let rn = Arm64Backend::gp_hw(base, RegSize::X64);
        if (-256..=255).contains(&offset) {
            buf.emit(Arm64Inst::ldur(rd, rn, offset));
        } else {
            // Use rd as scratch to avoid clobbering rn (which might be the
            // same physical register as X28, the usual scratch).
            Arm64Backend::emit_mov_imm64(buf, rd_x64, offset as u64);
            buf.emit(Arm64Inst::add(rd_x64, rn, rd_x64));
            buf.emit(Arm64Inst::ldur(rd, rd_x64, 0));
        }
    }

    fn emit_load_fp(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        base: MachineReg,
        offset: i32,
    ) {
        let rd = Arm64Backend::fp_hw(dst);
        let rn = Arm64Backend::gp_hw(base, RegSize::X64);
        if offset != 0 {
            let scratch = if rn != X28 { X28 } else { X16 };
            Arm64Backend::emit_mov_imm64(buf, scratch, offset as u64);
            buf.emit(Arm64Inst::add(scratch, rn, scratch));
            buf.emit(Arm64Inst::ldur_fp(rd, scratch, 0));
        } else {
            buf.emit(Arm64Inst::ldur_fp(rd, rn, 0));
        }
    }

    fn emit_store_gp(
        buf: &mut CodeBuffer<Self::Arch>,
        src: MachineReg,
        base: MachineReg,
        offset: i32,
        size: MachineWordSize,
    ) {
        let size = Arm64Backend::size_to_regsize(size);
        let rt = Arm64Backend::gp_hw(src, size);
        let rn = Arm64Backend::gp_hw(base, RegSize::X64);
        if (-256..=255).contains(&offset) {
            buf.emit(Arm64Inst::stur(rt, rn, offset));
        } else {
            // Large offset: need a scratch register for address computation.
            // Use X28 (primary scratch) unless it conflicts with src/base,
            // in which case fall back to X16 (secondary scratch / IP0).
            let src_hw = Arm64Backend::gp_hw(src, RegSize::X64);
            let base_hw = Arm64Backend::gp_hw(base, RegSize::X64);
            let scratch = if src_hw != X28 && base_hw != X28 {
                X28
            } else if src_hw != X16 && base_hw != X16 {
                X16
            } else {
                panic!(
                    "emit_store_gp: large offset ({offset}) needs scratch but both \
                     X28 and X16 conflict with src/base"
                );
            };
            Arm64Backend::emit_mov_imm64(buf, scratch, offset as u64);
            buf.emit(Arm64Inst::add(scratch, rn, scratch));
            buf.emit(Arm64Inst::stur(rt, scratch, 0));
        }
    }

    fn emit_store_fp(
        buf: &mut CodeBuffer<Self::Arch>,
        src: MachineReg,
        base: MachineReg,
        offset: i32,
    ) {
        let rt = Arm64Backend::fp_hw(src);
        let rn = Arm64Backend::gp_hw(base, RegSize::X64);
        if offset != 0 {
            let scratch = if rn != X28 { X28 } else { X16 };
            Arm64Backend::emit_mov_imm64(buf, scratch, offset as u64);
            buf.emit(Arm64Inst::add(scratch, rn, scratch));
            buf.emit(Arm64Inst::stur_fp(rt, scratch, 0));
        } else {
            buf.emit(Arm64Inst::stur_fp(rt, rn, 0));
        }
    }

    fn emit_gp_neg(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        src: MachineReg,
        size: MachineWordSize,
    ) {
        let size = Arm64Backend::size_to_regsize(size);
        let rd = Arm64Backend::gp_hw(dst, size);
        let rn = Arm64Backend::gp_hw(src, size);
        let zr = if size == RegSize::W32 { WZR } else { XZR };
        buf.emit(Arm64Inst::sub(rd, zr, rn));
    }

    fn emit_gp_not(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        src: MachineReg,
        size: MachineWordSize,
    ) {
        let size = Arm64Backend::size_to_regsize(size);
        buf.emit(Arm64Inst::mvn(
            Arm64Backend::gp_hw(dst, size),
            Arm64Backend::gp_hw(src, size),
        ));
    }

    fn emit_fp_neg(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, src: MachineReg) {
        buf.emit(Arm64Inst::fneg(
            Arm64Backend::fp_hw(dst),
            Arm64Backend::fp_hw(src),
        ));
    }

    fn emit_sign_extend(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        src: MachineReg,
        src_ty: Type,
        target_ty: Type,
    ) {
        let rd = Arm64Backend::gp_hw(
            dst,
            match target_ty {
                Type::I32 => RegSize::W32,
                _ => RegSize::X64,
            },
        );
        let rn = Arm64Backend::gp_hw(src, RegSize::W32);
        match (src_ty, target_ty) {
            (Type::I32, Type::I64) => buf.emit(Arm64Inst::sxtw(rd, rn)),
            (Type::I8, Type::I32) | (Type::I8, Type::I64) => buf.emit(Arm64Inst::sxtb(rd, rn)),
            _ => buf.emit(Arm64Inst::mov(rd, rn)),
        };
    }

    fn emit_zero_extend(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        src: MachineReg,
        src_ty: Type,
        target_ty: Type,
    ) {
        match (src_ty, target_ty) {
            (Type::I8, _) => {
                let rd = Arm64Backend::gp_hw(dst, RegSize::W32);
                let rn = Arm64Backend::gp_hw(src, RegSize::W32);
                buf.emit(Arm64Inst::AndImm {
                    sf: 0,
                    n: 0,
                    immr: 0,
                    imms: 7,
                    rn,
                    rd,
                });
            }
            (Type::I32, Type::I64) => {
                let rd = Arm64Backend::gp_hw(dst, RegSize::W32);
                let rn = Arm64Backend::gp_hw(src, RegSize::W32);
                buf.emit(Arm64Inst::mov(rd, rn));
            }
            _ => {
                let rd = Arm64Backend::gp_hw(
                    dst,
                    match target_ty {
                        Type::I32 => RegSize::W32,
                        _ => RegSize::X64,
                    },
                );
                let rn = Arm64Backend::gp_hw(
                    src,
                    match src_ty {
                        Type::I32 | Type::I8 => RegSize::W32,
                        _ => RegSize::X64,
                    },
                );
                buf.emit(Arm64Inst::mov(rd, rn));
            }
        }
    }

    fn emit_trunc(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        src: MachineReg,
        target_ty: Type,
    ) {
        match target_ty {
            Type::I8 => {
                let rd = Arm64Backend::gp_hw(dst, RegSize::W32);
                let rn = Arm64Backend::gp_hw(src, RegSize::W32);
                buf.emit(Arm64Inst::AndImm {
                    sf: 0,
                    n: 0,
                    immr: 0,
                    imms: 7,
                    rn,
                    rd,
                });
            }
            Type::I32 => {
                let rd = Arm64Backend::gp_hw(dst, RegSize::W32);
                let rn = Arm64Backend::gp_hw(src, RegSize::W32);
                buf.emit(Arm64Inst::mov(rd, rn));
            }
            _ => {
                let rd = Arm64Backend::gp_hw(dst, RegSize::X64);
                let rn = Arm64Backend::gp_hw(src, RegSize::X64);
                buf.emit(Arm64Inst::mov(rd, rn));
            }
        }
    }

    fn emit_int_to_float(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        src: MachineReg,
        src_ty: Type,
    ) {
        let rn = Arm64Backend::gp_hw(
            src,
            match src_ty {
                Type::I32 | Type::I8 => RegSize::W32,
                _ => RegSize::X64,
            },
        );
        buf.emit(Arm64Inst::scvtf_to_double(Arm64Backend::fp_hw(dst), rn));
    }

    fn emit_float_to_int(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, src: MachineReg) {
        buf.emit(Arm64Inst::fcvtzs_from_double(
            Arm64Backend::gp_hw(dst, RegSize::X64),
            Arm64Backend::fp_hw(src),
        ));
    }

    fn emit_mov_imm(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, value: u64) {
        Arm64Backend::emit_mov_imm64(buf, Arm64Backend::gp_hw(dst, RegSize::X64), value);
    }

    fn emit_f64_const(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, bits: u64) {
        Arm64Backend::emit_mov_imm64(buf, X28, bits);
        buf.emit(Arm64Inst::fmov_gp_to_fp(Arm64Backend::fp_hw(dst), X28));
    }

    fn emit_load_incoming_stack_arg(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        incoming_offset: i32,
    ) {
        buf.emit(Arm64Inst::ldr(
            Arm64Backend::gp_hw(dst, RegSize::X64),
            X28,
            incoming_offset,
        ));
    }

    fn emit_cmp_gp_imm(buf: &mut CodeBuffer<Self::Arch>, reg: MachineReg, imm: u64) {
        Arm64Backend::emit_mov_imm64(buf, X28, imm);
        buf.emit(Arm64Inst::cmp(Arm64Backend::gp_hw(reg, RegSize::X64), X28));
    }

    fn emit_extract_payload(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        src: MachineReg,
        has_unboxed_float: bool,
        payload_bits: u8,
    ) {
        let rd = Arm64Backend::gp_hw(dst, RegSize::X64);
        let rn = Arm64Backend::gp_hw(src, RegSize::X64);
        if has_unboxed_float {
            let mask = (1u64 << payload_bits) - 1;
            Arm64Backend::emit_mov_imm64(buf, X28, mask);
            buf.emit(Arm64Inst::and(rd, rn, X28));
        } else {
            let tag_bits = 64 - payload_bits;
            Arm64Backend::emit_mov_imm64(buf, X28, tag_bits as u64);
            buf.emit(Arm64Inst::LsrReg {
                sf: 1,
                rm: X28,
                rn,
                rd,
            });
        }
    }

    fn emit_is_tag(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        src: MachineReg,
        has_unboxed_float: bool,
        payload_bits: u8,
        tag_mask: u64,
        expected_tag: u64,
    ) {
        let rn = Arm64Backend::gp_hw(src, RegSize::X64);
        if has_unboxed_float {
            Arm64Backend::emit_mov_imm64(buf, X28, payload_bits as u64);
            buf.emit(Arm64Inst::LsrReg {
                sf: 1,
                rm: X28,
                rn,
                rd: X28,
            });
            Arm64Backend::emit_mov_imm64(buf, Arm64Backend::gp_hw(dst, RegSize::X64), expected_tag);
            buf.emit(Arm64Inst::cmp(X28, Arm64Backend::gp_hw(dst, RegSize::X64)));
        } else {
            Arm64Backend::emit_mov_imm64(buf, X28, tag_mask);
            buf.emit(Arm64Inst::and(X28, rn, X28));
            Arm64Backend::emit_mov_imm64(buf, Arm64Backend::gp_hw(dst, RegSize::X64), expected_tag);
            buf.emit(Arm64Inst::cmp(X28, Arm64Backend::gp_hw(dst, RegSize::X64)));
        }
        buf.emit(Arm64Inst::cset(
            Arm64Backend::gp_hw(dst, RegSize::W32),
            Arm64Cond::EQ,
        ));
    }

    fn emit_make_tagged(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        payload: MachineReg,
        has_unboxed_float: bool,
        payload_bits: u8,
        encoded_tag_pattern: u64,
        tag: u64,
    ) {
        let rd = Arm64Backend::gp_hw(dst, RegSize::X64);
        let rn = Arm64Backend::gp_hw(payload, RegSize::X64);
        if has_unboxed_float {
            Arm64Backend::emit_mov_imm64(buf, X28, encoded_tag_pattern);
            buf.emit(Arm64Inst::orr(rd, rn, X28));
        } else {
            let tag_bits = 64 - payload_bits;
            Arm64Backend::emit_mov_imm64(buf, X28, tag_bits as u64);
            buf.emit(Arm64Inst::LslReg {
                sf: 1,
                rm: X28,
                rn,
                rd,
            });
            Arm64Backend::emit_mov_imm64(buf, X28, tag);
            buf.emit(Arm64Inst::orr(rd, rd, X28));
        }
    }

    fn emit_tag_of(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        src: MachineReg,
        has_unboxed_float: bool,
        payload_bits: u8,
        tag_mask: u64,
    ) {
        let rd = Arm64Backend::gp_hw(dst, RegSize::X64);
        let rn = Arm64Backend::gp_hw(src, RegSize::X64);
        if has_unboxed_float {
            Arm64Backend::emit_mov_imm64(buf, X28, payload_bits as u64);
            buf.emit(Arm64Inst::LsrReg {
                sf: 1,
                rm: X28,
                rn,
                rd,
            });
            Arm64Backend::emit_mov_imm64(buf, X28, tag_mask);
            buf.emit(Arm64Inst::and(rd, rd, X28));
        } else {
            Arm64Backend::emit_mov_imm64(buf, X28, tag_mask);
            buf.emit(Arm64Inst::and(rd, rn, X28));
        }
    }

    fn emit_call_safepoint_handler(
        buf: &mut CodeBuffer<Self::Arch>,
        handler: u64,
        frame_size: u64,
    ) {
        buf.emit(Arm64Inst::mov(X0, X29));
        Arm64Backend::emit_mov_imm64(buf, X1, frame_size);
        Arm64Backend::emit_mov_imm64(buf, X28, handler);
        buf.emit(Arm64Inst::blr(X28));
    }

    fn bind_label(buf: &mut CodeBuffer<Self::Arch>, label: Label) {
        Arm64Backend::bind_label(buf, label);
    }

    fn emit_branch_to_label(buf: &mut CodeBuffer<Self::Arch>, label: Label) {
        Arm64Backend::emit_branch_to_label(buf, label);
    }

    fn emit_cbz_to_label(buf: &mut CodeBuffer<Self::Arch>, reg: MachineReg, label: Label) {
        Arm64Backend::emit_cbz_to_label(buf, reg, label);
    }

    fn emit_cbnz_to_label(buf: &mut CodeBuffer<Self::Arch>, reg: MachineReg, label: Label) {
        Arm64Backend::emit_cbnz_to_label(buf, reg, label);
    }

    fn emit_branch_eq_to_label(buf: &mut CodeBuffer<Self::Arch>, label: Label) {
        Arm64Backend::emit_branch_eq_to_label(buf, label);
    }

    fn emit_call_reg(buf: &mut CodeBuffer<Self::Arch>, reg: MachineReg) {
        Arm64Backend::emit_call_reg(buf, reg);
    }

    fn emit_return_gp(buf: &mut CodeBuffer<Self::Arch>, src: MachineReg) {
        Arm64Backend::emit_return_gp(buf, src);
    }

    fn emit_return_fp_bits(buf: &mut CodeBuffer<Self::Arch>, src: MachineReg) {
        Arm64Backend::emit_return_fp_bits(buf, src);
    }

    fn emit_stack_adjust(buf: &mut CodeBuffer<Self::Arch>, amount: i32) {
        if amount >= 0 {
            let hi = (amount >> 12) & 0xFFF;
            let lo = amount & 0xFFF;
            if hi > 0 {
                buf.emit(Arm64Inst::SubImm {
                    sf: 1,
                    sh: 1,
                    imm12: hi,
                    rn: SP,
                    rd: SP,
                });
            }
            if lo > 0 || hi == 0 {
                buf.emit(Arm64Inst::sub_imm(SP, SP, lo));
            }
        } else {
            let abs = -amount;
            let hi = (abs >> 12) & 0xFFF;
            let lo = abs & 0xFFF;
            if hi > 0 {
                buf.emit(Arm64Inst::AddImm {
                    sf: 1,
                    sh: 1,
                    imm12: hi,
                    rn: SP,
                    rd: SP,
                });
            }
            if lo > 0 || hi == 0 {
                buf.emit(Arm64Inst::add_imm(SP, SP, lo));
            }
        }
    }

    fn emit_trap(buf: &mut CodeBuffer<Self::Arch>) {
        buf.emit(Arm64Inst::brk(1));
    }
}

#[allow(dead_code)]
impl X64Backend {
    pub fn name() -> &'static str {
        "x86_64"
    }

    fn gp_hw(reg: MachineReg) -> X64Reg {
        debug_assert_eq!(reg.class, MachineRegClass::Gp);
        let hw = match reg.index {
            0 => RAX,
            1 => RCX,
            2 => RDX,
            3 => RSI,
            4 => RDI,
            5 => R8,
            6 => R9,
            7 | 27 => R10,
            8 | 28 => R11,
            9 => RBX,
            10 => R12,
            11 => R13,
            22 => R14,
            23 => R15,
            24 => R14,
            26 => R11,
            29 => RBP,
            other => panic!("x64 backend has no physical GP mapping for machine register {other}"),
        };
        X64Reg::new(hw.index, dynasm::x86_64::reg::Size::S64)
    }

    fn gp_hw8(reg: MachineReg) -> X64Reg {
        debug_assert_eq!(reg.class, MachineRegClass::Gp);
        let hw = Self::gp_hw(reg);
        X64Reg::new(hw.index, dynasm::x86_64::reg::Size::S8)
    }

    fn gp_hw_sized(reg: MachineReg, size: MachineWordSize) -> X64Reg {
        debug_assert_eq!(reg.class, MachineRegClass::Gp);
        let hw = Self::gp_hw(reg);
        let reg_size = match size {
            MachineWordSize::W32 => dynasm::x86_64::reg::Size::S32,
            MachineWordSize::W64 => dynasm::x86_64::reg::Size::S64,
        };
        X64Reg::new(hw.index, reg_size)
    }

    fn fp_hw(reg: MachineReg) -> X64Reg {
        debug_assert_eq!(reg.class, MachineRegClass::Fp);
        X64Reg::new(reg.index, dynasm::x86_64::reg::Size::S64)
    }

    fn cond(op: CmpOp) -> X64Cond {
        match op {
            CmpOp::Eq => X64Cond::E,
            CmpOp::Ne => X64Cond::NE,
            CmpOp::Slt => X64Cond::L,
            CmpOp::Sle => X64Cond::LE,
            CmpOp::Sgt => X64Cond::G,
            CmpOp::Sge => X64Cond::GE,
            CmpOp::Ult => X64Cond::B,
            CmpOp::Ule => X64Cond::BE,
            CmpOp::Ugt => X64Cond::A,
            CmpOp::Uge => X64Cond::AE,
        }
    }

    fn fcond(op: CmpOp) -> X64Cond {
        match op {
            CmpOp::Eq => X64Cond::E,
            CmpOp::Ne => X64Cond::NE,
            CmpOp::Slt | CmpOp::Ult => X64Cond::B,
            CmpOp::Sle | CmpOp::Ule => X64Cond::BE,
            CmpOp::Sgt | CmpOp::Ugt => X64Cond::A,
            CmpOp::Sge | CmpOp::Uge => X64Cond::AE,
        }
    }

    fn slot_base_reg(slot: FrameSlotAccess) -> X64Reg {
        match slot.base {
            FrameSlotBase::FramePointer => RBP,
            FrameSlotBase::StackPointer => RSP,
        }
    }

    fn emit_mov_imm_to_hw(buf: &mut CodeBuffer<X64>, dst: X64Reg, value: u64) {
        let imm = i64::from_le_bytes(value.to_le_bytes());
        buf.emit(X64Inst::MovRI { dest: dst, imm });
    }

    fn emit_shift_by_reg(
        buf: &mut CodeBuffer<X64>,
        op: MachineGpBinOp,
        dst: X64Reg,
        lhs: X64Reg,
        rhs: X64Reg,
    ) {
        if rhs.index != RCX.index {
            // Save the shift count before clobbering dst; dst may be rhs.
            buf.emit(X64Inst::MovRR {
                dest: RCX,
                src: rhs,
            });
        } else if dst.index == RCX.index {
            // CL already holds the count, so compute through a scratch result.
            buf.emit(X64Inst::MovRR {
                dest: R11,
                src: lhs,
            });
            match op {
                MachineGpBinOp::Shl => buf.emit(X64Inst::ShlRCL { dest: R11 }),
                MachineGpBinOp::LShr => buf.emit(X64Inst::ShrRCL { dest: R11 }),
                MachineGpBinOp::AShr => buf.emit(X64Inst::SarRCL { dest: R11 }),
                _ => unreachable!(),
            };
            buf.emit(X64Inst::MovRR {
                dest: dst,
                src: R11,
            });
            return;
        }

        if dst.index != lhs.index {
            buf.emit(X64Inst::MovRR {
                dest: dst,
                src: lhs,
            });
        }
        match op {
            MachineGpBinOp::Shl => buf.emit(X64Inst::ShlRCL { dest: dst }),
            MachineGpBinOp::LShr => buf.emit(X64Inst::ShrRCL { dest: dst }),
            MachineGpBinOp::AShr => buf.emit(X64Inst::SarRCL { dest: dst }),
            _ => unreachable!(),
        };
    }

    fn emit_zero_gp(buf: &mut CodeBuffer<X64>, reg: X64Reg) {
        buf.emit(X64Inst::XorRR {
            dest: reg,
            src: reg,
        });
    }
}

impl LoweringBackend for X64Backend {
    type Arch = X64;

    fn allocatable_gp() -> &'static [u8] {
        &[3, 4, 5, 6, 9, 10, 11]
    }

    fn c_abi_gp_arg_limit() -> usize {
        // SysV x86-64: 6 GP argument registers (RDI, RSI, RDX, RCX, R8, R9).
        6
    }

    fn allocatable_fp() -> &'static [u8] {
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    }

    fn caller_saved_gp() -> &'static [u8] {
        &[0, 1, 2, 3, 4, 5, 6, 7, 8]
    }

    fn caller_saved_fp() -> &'static [u8] {
        Self::allocatable_fp()
    }

    fn call_arg_gp(slot: usize) -> MachineReg {
        let reg = match slot {
            0 => 4,   // RDI
            1 => 3,   // RSI
            2 => 2,   // RDX
            3 => 1,   // RCX
            4 => 5,   // R8
            5 => 6,   // R9
            6 => 7,   // R10, internal-only extension
            7 => 8,   // R11, internal-only extension
            8 => 9,   // RBX, internal-only extension
            9 => 10,  // R12, internal-only extension
            10 => 11, // R13, internal-only extension
            other => other as u8,
        };
        MachineReg {
            class: MachineRegClass::Gp,
            index: reg,
        }
    }

    fn emit_prologue(buf: &mut CodeBuffer<Self::Arch>) -> usize {
        buf.emit(X64Inst::Push { reg: RBP });
        buf.emit(X64Inst::Push { reg: RBX });
        buf.emit(X64Inst::Push { reg: R12 });
        buf.emit(X64Inst::Push { reg: R13 });
        buf.emit(X64Inst::Push { reg: R14 });
        buf.emit(X64Inst::Push { reg: R15 });
        buf.emit(X64Inst::Lea {
            dest: R14,
            base: RSP,
            offset: 48,
        });
        let patch = buf.emit(X64Inst::SubRI { dest: RSP, imm: 24 });
        // Zero-loop count placeholder, patched by `emit_frame_size_patch`.
        // MUST immediately follow the SubRI so the patch site is at
        // `prologue_offset + len(SubRI)`. See the Arm64 prologue for why
        // zeroing the locals region is required for GC soundness.
        buf.emit(X64Inst::MovRI32 { dest: R11, imm: 0 });
        buf.emit(X64Inst::MovRR {
            dest: RBP,
            src: RSP,
        });
        buf.emit(X64Inst::MovRM {
            dest: R10,
            base: R14,
            offset: -8,
        });
        buf.emit(X64Inst::MovMR {
            base: RBP,
            offset: 0,
            src: R10,
        });
        buf.emit(X64Inst::MovRM {
            dest: R10,
            base: R14,
            offset: 0,
        });
        buf.emit(X64Inst::MovMR {
            base: RBP,
            offset: 8,
            src: R10,
        });
        // Zero the locals region [RBP+16, RBP+16+R11). R10/R11/RAX are
        // scratch at function entry (args arrive in RDI/RSI/RDX/RCX/R8/R9
        // and on the stack).
        buf.emit(X64Inst::Lea {
            dest: R10,
            base: RBP,
            offset: 16,
        });
        buf.emit(X64Inst::MovRI32 { dest: RAX, imm: 0 });
        buf.emit(X64Inst::TestRR { a: R11, b: R11 });
        let skip = buf.create_label();
        let jz = buf.emit(X64Inst::Jcc {
            offset: 0,
            cond: X64Cond::E,
        });
        buf.add_reloc(jz + 2, skip, X64RelocKind::Rel32);
        let loop_top = buf.create_label();
        buf.bind_label(loop_top);
        buf.emit(X64Inst::MovMR {
            base: R10,
            offset: 0,
            src: RAX,
        });
        buf.emit(X64Inst::AddRI { dest: R10, imm: 8 });
        buf.emit(X64Inst::SubRI { dest: R11, imm: 8 });
        let jnz = buf.emit(X64Inst::Jcc {
            offset: 0,
            cond: X64Cond::NE,
        });
        buf.add_reloc(jnz + 2, loop_top, X64RelocKind::Rel32);
        buf.bind_label(skip);
        patch
    }

    fn emit_epilogue(buf: &mut CodeBuffer<Self::Arch>, patch_offsets: &mut Vec<usize>) {
        buf.emit(X64Inst::MovRR {
            dest: RSP,
            src: RBP,
        });
        let add_offset = buf.emit(X64Inst::AddRI { dest: RSP, imm: 24 });
        patch_offsets.push(add_offset);
        buf.emit(X64Inst::Pop { reg: R15 });
        buf.emit(X64Inst::Pop { reg: R14 });
        buf.emit(X64Inst::Pop { reg: R13 });
        buf.emit(X64Inst::Pop { reg: R12 });
        buf.emit(X64Inst::Pop { reg: RBX });
        buf.emit(X64Inst::Pop { reg: RBP });
        buf.emit(X64Inst::Ret);
    }

    fn emit_frame_size_patch(
        buf: &mut CodeBuffer<Self::Arch>,
        prologue_offset: usize,
        epilogue_offsets: &[usize],
        frame_size: i32,
    ) {
        let aligned_frame_size = frame_size + 8;
        let sub = X64Inst::SubRI {
            dest: RSP,
            imm: aligned_frame_size,
        }
        .encode();
        let sub_len = sub.len();
        buf.patch_bytes(prologue_offset, &sub);
        // Patch the locals-zeroing loop count (the MovRI32 immediately
        // after the SubRI — see emit_prologue).
        let zero_bytes = (frame_size - 16).max(0);
        let zero_mov = X64Inst::MovRI32 {
            dest: R11,
            imm: zero_bytes,
        }
        .encode();
        buf.patch_bytes(prologue_offset + sub_len, &zero_mov);
        let add = X64Inst::AddRI {
            dest: RSP,
            imm: aligned_frame_size,
        }
        .encode();
        for &offset in epilogue_offsets {
            buf.patch_bytes(offset, &add);
        }
    }

    fn emit_gp_move(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, src: MachineReg) {
        if dst.index != src.index {
            buf.emit(X64Inst::MovRR {
                dest: Self::gp_hw(dst),
                src: Self::gp_hw(src),
            });
        }
    }

    fn emit_stack_pointer_to_gp(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg) {
        buf.emit(X64Inst::MovRR {
            dest: Self::gp_hw(dst),
            src: RSP,
        });
    }

    fn emit_fp_to_gp_move(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, src: MachineReg) {
        buf.emit(X64Inst::MovqRX {
            dest: Self::gp_hw(dst),
            src: Self::fp_hw(src),
        });
    }

    fn emit_gp_to_fp_move(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, src: MachineReg) {
        buf.emit(X64Inst::MovqXR {
            dest: Self::fp_hw(dst),
            src: Self::gp_hw(src),
        });
    }

    fn emit_store_gp_to_frame(
        buf: &mut CodeBuffer<Self::Arch>,
        src: MachineReg,
        slot: FrameSlotAccess,
    ) {
        buf.emit(X64Inst::MovMR {
            base: Self::slot_base_reg(slot),
            offset: slot.offset,
            src: Self::gp_hw(src),
        });
    }
    fn emit_load_gp_from_frame(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        slot: FrameSlotAccess,
    ) {
        buf.emit(X64Inst::MovRM {
            dest: Self::gp_hw(dst),
            base: Self::slot_base_reg(slot),
            offset: slot.offset,
        });
    }
    fn emit_store_fp_to_frame(
        buf: &mut CodeBuffer<Self::Arch>,
        src: MachineReg,
        slot: FrameSlotAccess,
    ) {
        buf.emit(X64Inst::MovsdMR {
            base: Self::slot_base_reg(slot),
            offset: slot.offset,
            src: Self::fp_hw(src),
        });
    }
    fn emit_load_fp_from_frame(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        slot: FrameSlotAccess,
    ) {
        buf.emit(X64Inst::MovsdRM {
            dest: Self::fp_hw(dst),
            base: Self::slot_base_reg(slot),
            offset: slot.offset,
        });
    }
    fn emit_store_gp_to_stack_arg(buf: &mut CodeBuffer<Self::Arch>, src: MachineReg, offset: i32) {
        buf.emit(X64Inst::MovMR {
            base: RSP,
            offset,
            src: Self::gp_hw(src),
        });
    }

    fn emit_gp_binop(
        buf: &mut CodeBuffer<Self::Arch>,
        op: MachineGpBinOp,
        dst: MachineReg,
        lhs: MachineReg,
        rhs: MachineReg,
        _size: MachineWordSize,
    ) {
        let dst_hw = Self::gp_hw(dst);
        let lhs_hw = Self::gp_hw(lhs);
        let rhs_hw = Self::gp_hw(rhs);
        match op {
            MachineGpBinOp::Shl | MachineGpBinOp::LShr | MachineGpBinOp::AShr => {
                Self::emit_shift_by_reg(buf, op, dst_hw, lhs_hw, rhs_hw);
                return;
            }
            MachineGpBinOp::SDiv | MachineGpBinOp::UDiv => {
                let divisor = if rhs_hw.index == RAX.index || rhs_hw.index == RDX.index {
                    buf.emit(X64Inst::MovRR {
                        dest: R11,
                        src: rhs_hw,
                    });
                    R11
                } else {
                    rhs_hw
                };
                if lhs_hw.index != RAX.index {
                    buf.emit(X64Inst::MovRR {
                        dest: RAX,
                        src: lhs_hw,
                    });
                }
                if matches!(op, MachineGpBinOp::SDiv) {
                    buf.emit(X64Inst::Cqo);
                } else {
                    Self::emit_zero_gp(buf, RDX);
                }
                buf.emit(X64Inst::Idiv { divisor });
                if dst_hw.index != RAX.index {
                    buf.emit(X64Inst::MovRR {
                        dest: dst_hw,
                        src: RAX,
                    });
                }
                return;
            }
            _ => {}
        }
        if dst_hw.index != lhs_hw.index {
            if dst_hw.index == rhs_hw.index {
                match op {
                    MachineGpBinOp::Add
                    | MachineGpBinOp::Mul
                    | MachineGpBinOp::And
                    | MachineGpBinOp::Or
                    | MachineGpBinOp::Xor => {
                        match op {
                            MachineGpBinOp::Add => {
                                buf.emit(X64Inst::AddRR {
                                    dest: dst_hw,
                                    src: lhs_hw,
                                });
                            }
                            MachineGpBinOp::Mul => {
                                buf.emit(X64Inst::ImulRR {
                                    dest: dst_hw,
                                    src: lhs_hw,
                                });
                            }
                            MachineGpBinOp::And => {
                                buf.emit(X64Inst::AndRR {
                                    dest: dst_hw,
                                    src: lhs_hw,
                                });
                            }
                            MachineGpBinOp::Or => {
                                buf.emit(X64Inst::OrRR {
                                    dest: dst_hw,
                                    src: lhs_hw,
                                });
                            }
                            MachineGpBinOp::Xor => {
                                buf.emit(X64Inst::XorRR {
                                    dest: dst_hw,
                                    src: lhs_hw,
                                });
                            }
                            _ => unreachable!(),
                        }
                        return;
                    }
                    _ => {
                        buf.emit(X64Inst::MovRR {
                            dest: R11,
                            src: rhs_hw,
                        });
                    }
                }
            }
            buf.emit(X64Inst::MovRR {
                dest: dst_hw,
                src: lhs_hw,
            });
        }
        match op {
            MachineGpBinOp::Add => {
                let src = if dst_hw.index == rhs_hw.index && lhs_hw.index != rhs_hw.index {
                    lhs_hw
                } else {
                    rhs_hw
                };
                buf.emit(X64Inst::AddRR { dest: dst_hw, src });
            }
            MachineGpBinOp::Sub => {
                let src = if dst_hw.index == rhs_hw.index && lhs_hw.index != rhs_hw.index {
                    R11
                } else {
                    rhs_hw
                };
                buf.emit(X64Inst::SubRR { dest: dst_hw, src });
            }
            MachineGpBinOp::Mul => {
                let src = if dst_hw.index == rhs_hw.index && lhs_hw.index != rhs_hw.index {
                    lhs_hw
                } else {
                    rhs_hw
                };
                buf.emit(X64Inst::ImulRR { dest: dst_hw, src });
            }
            MachineGpBinOp::And => {
                let src = if dst_hw.index == rhs_hw.index && lhs_hw.index != rhs_hw.index {
                    lhs_hw
                } else {
                    rhs_hw
                };
                buf.emit(X64Inst::AndRR { dest: dst_hw, src });
            }
            MachineGpBinOp::Or => {
                let src = if dst_hw.index == rhs_hw.index && lhs_hw.index != rhs_hw.index {
                    lhs_hw
                } else {
                    rhs_hw
                };
                buf.emit(X64Inst::OrRR { dest: dst_hw, src });
            }
            MachineGpBinOp::Xor => {
                let src = if dst_hw.index == rhs_hw.index && lhs_hw.index != rhs_hw.index {
                    lhs_hw
                } else {
                    rhs_hw
                };
                buf.emit(X64Inst::XorRR { dest: dst_hw, src });
            }
            MachineGpBinOp::SDiv | MachineGpBinOp::UDiv => unreachable!(),
            MachineGpBinOp::Shl | MachineGpBinOp::LShr | MachineGpBinOp::AShr => unreachable!(),
        };
    }

    fn emit_fp_binop(
        buf: &mut CodeBuffer<Self::Arch>,
        op: MachineFpBinOp,
        dst: MachineReg,
        lhs: MachineReg,
        rhs: MachineReg,
    ) {
        let dst_hw = Self::fp_hw(dst);
        let lhs_hw = Self::fp_hw(lhs);
        let rhs_hw = Self::fp_hw(rhs);
        let fp_scratch = X64Reg::new(15, dynasm::x86_64::reg::Size::S64);
        if dst_hw.index != lhs_hw.index {
            if dst_hw.index == rhs_hw.index {
                match op {
                    MachineFpBinOp::Add => {
                        buf.emit(X64Inst::Addsd {
                            dest: dst_hw,
                            src: lhs_hw,
                        });
                        return;
                    }
                    MachineFpBinOp::Mul => {
                        buf.emit(X64Inst::Mulsd {
                            dest: dst_hw,
                            src: lhs_hw,
                        });
                        return;
                    }
                    MachineFpBinOp::Sub | MachineFpBinOp::Div => {
                        buf.emit(X64Inst::MovsdRR {
                            dest: fp_scratch,
                            src: rhs_hw,
                        });
                    }
                }
            }
            buf.emit(X64Inst::MovsdRR {
                dest: dst_hw,
                src: lhs_hw,
            });
        }
        match op {
            MachineFpBinOp::Add => {
                let src = if dst_hw.index == rhs_hw.index && lhs_hw.index != rhs_hw.index {
                    lhs_hw
                } else {
                    rhs_hw
                };
                buf.emit(X64Inst::Addsd { dest: dst_hw, src });
            }
            MachineFpBinOp::Sub => {
                let src = if dst_hw.index == rhs_hw.index && lhs_hw.index != rhs_hw.index {
                    fp_scratch
                } else {
                    rhs_hw
                };
                buf.emit(X64Inst::Subsd { dest: dst_hw, src });
            }
            MachineFpBinOp::Mul => {
                let src = if dst_hw.index == rhs_hw.index && lhs_hw.index != rhs_hw.index {
                    lhs_hw
                } else {
                    rhs_hw
                };
                buf.emit(X64Inst::Mulsd { dest: dst_hw, src });
            }
            MachineFpBinOp::Div => {
                let src = if dst_hw.index == rhs_hw.index && lhs_hw.index != rhs_hw.index {
                    fp_scratch
                } else {
                    rhs_hw
                };
                buf.emit(X64Inst::Divsd { dest: dst_hw, src });
            }
        };
    }

    fn emit_icmp_set(
        buf: &mut CodeBuffer<Self::Arch>,
        op: CmpOp,
        dst: MachineReg,
        lhs: MachineReg,
        rhs: MachineReg,
        size: MachineWordSize,
    ) {
        buf.emit(X64Inst::CmpRR {
            a: Self::gp_hw_sized(lhs, size),
            b: Self::gp_hw_sized(rhs, size),
        });
        buf.emit(X64Inst::MovRI32 {
            dest: Self::gp_hw(dst),
            imm: 0,
        });
        buf.emit(X64Inst::Setcc {
            dest: Self::gp_hw8(dst),
            cond: Self::cond(op),
        });
    }

    fn emit_fcmp_set(
        buf: &mut CodeBuffer<Self::Arch>,
        op: CmpOp,
        dst: MachineReg,
        lhs: MachineReg,
        rhs: MachineReg,
    ) {
        buf.emit(X64Inst::Ucomisd {
            a: Self::fp_hw(lhs),
            b: Self::fp_hw(rhs),
        });
        buf.emit(X64Inst::MovRI32 {
            dest: Self::gp_hw(dst),
            imm: 0,
        });
        buf.emit(X64Inst::Setcc {
            dest: Self::gp_hw8(dst),
            cond: Self::fcond(op),
        });
    }
    fn emit_gp_select(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        cond: MachineReg,
        when_true: MachineReg,
        when_false: MachineReg,
        _size: MachineWordSize,
    ) {
        let dst_hw = Self::gp_hw(dst);
        buf.emit(X64Inst::TestRR {
            a: Self::gp_hw(cond),
            b: Self::gp_hw(cond),
        });
        if dst_hw.index == Self::gp_hw(when_true).index
            && dst_hw.index != Self::gp_hw(when_false).index
        {
            buf.emit(X64Inst::Cmovcc {
                dest: dst_hw,
                src: Self::gp_hw(when_false),
                cond: X64Cond::E,
            });
        } else {
            buf.emit(X64Inst::MovRR {
                dest: dst_hw,
                src: Self::gp_hw(when_false),
            });
            buf.emit(X64Inst::Cmovcc {
                dest: dst_hw,
                src: Self::gp_hw(when_true),
                cond: X64Cond::NE,
            });
        }
    }
    fn emit_fp_select(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        cond: MachineReg,
        when_true: MachineReg,
        when_false: MachineReg,
    ) {
        let false_label = buf.create_label();
        let end = buf.create_label();
        buf.emit(X64Inst::TestRR {
            a: Self::gp_hw(cond),
            b: Self::gp_hw(cond),
        });
        let off = buf.emit(X64Inst::Jcc {
            offset: 0,
            cond: X64Cond::E,
        });
        buf.add_reloc(off + 2, false_label, X64RelocKind::Rel32);
        buf.emit(X64Inst::MovsdRR {
            dest: Self::fp_hw(dst),
            src: Self::fp_hw(when_true),
        });
        let jmp = buf.emit(X64Inst::Jmp { offset: 0 });
        buf.add_reloc(jmp + 1, end, X64RelocKind::Rel32);
        buf.bind_label(false_label);
        buf.emit(X64Inst::MovsdRR {
            dest: Self::fp_hw(dst),
            src: Self::fp_hw(when_false),
        });
        buf.bind_label(end);
    }
    fn emit_load_gp(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        base: MachineReg,
        offset: i32,
        _size: MachineWordSize,
    ) {
        buf.emit(X64Inst::MovRM {
            dest: Self::gp_hw(dst),
            base: Self::gp_hw(base),
            offset,
        });
    }
    fn emit_load_fp(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        base: MachineReg,
        offset: i32,
    ) {
        buf.emit(X64Inst::MovsdRM {
            dest: Self::fp_hw(dst),
            base: Self::gp_hw(base),
            offset,
        });
    }
    fn emit_store_gp(
        buf: &mut CodeBuffer<Self::Arch>,
        src: MachineReg,
        base: MachineReg,
        offset: i32,
        _size: MachineWordSize,
    ) {
        buf.emit(X64Inst::MovMR {
            base: Self::gp_hw(base),
            offset,
            src: Self::gp_hw(src),
        });
    }
    fn emit_store_fp(
        buf: &mut CodeBuffer<Self::Arch>,
        src: MachineReg,
        base: MachineReg,
        offset: i32,
    ) {
        buf.emit(X64Inst::MovsdMR {
            base: Self::gp_hw(base),
            offset,
            src: Self::fp_hw(src),
        });
    }
    fn emit_gp_neg(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        src: MachineReg,
        _size: MachineWordSize,
    ) {
        if dst.index != src.index {
            buf.emit(X64Inst::MovRR {
                dest: Self::gp_hw(dst),
                src: Self::gp_hw(src),
            });
        }
        buf.emit(X64Inst::Neg {
            reg: Self::gp_hw(dst),
        });
    }
    fn emit_gp_not(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        src: MachineReg,
        _size: MachineWordSize,
    ) {
        if dst.index != src.index {
            buf.emit(X64Inst::MovRR {
                dest: Self::gp_hw(dst),
                src: Self::gp_hw(src),
            });
        }
        buf.emit(X64Inst::Not {
            reg: Self::gp_hw(dst),
        });
    }
    fn emit_fp_neg(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, src: MachineReg) {
        buf.emit(X64Inst::MovqRX {
            dest: R11,
            src: Self::fp_hw(src),
        });
        Self::emit_mov_imm_to_hw(buf, R10, 0x8000_0000_0000_0000);
        buf.emit(X64Inst::XorRR {
            dest: R11,
            src: R10,
        });
        buf.emit(X64Inst::MovqXR {
            dest: Self::fp_hw(dst),
            src: R11,
        });
    }
    fn emit_sign_extend(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        src: MachineReg,
        src_ty: Type,
        target_ty: Type,
    ) {
        let dst_hw = Self::gp_hw(dst);
        if dst_hw.index != Self::gp_hw(src).index {
            buf.emit(X64Inst::MovRR {
                dest: dst_hw,
                src: Self::gp_hw(src),
            });
        }
        match (src_ty, target_ty) {
            (Type::I8, Type::I32 | Type::I64) => {
                buf.emit(X64Inst::ShlRI {
                    dest: dst_hw,
                    imm: 56,
                });
                buf.emit(X64Inst::SarRI {
                    dest: dst_hw,
                    imm: 56,
                });
            }
            (Type::I32, Type::I64) => {
                buf.emit(X64Inst::ShlRI {
                    dest: dst_hw,
                    imm: 32,
                });
                buf.emit(X64Inst::SarRI {
                    dest: dst_hw,
                    imm: 32,
                });
            }
            _ => {}
        }
    }
    fn emit_zero_extend(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        src: MachineReg,
        src_ty: Type,
        _target_ty: Type,
    ) {
        let dst_hw = Self::gp_hw(dst);
        if dst_hw.index != Self::gp_hw(src).index {
            buf.emit(X64Inst::MovRR {
                dest: dst_hw,
                src: Self::gp_hw(src),
            });
        }
        match src_ty {
            Type::I8 => {
                buf.emit(X64Inst::AndRI {
                    dest: dst_hw,
                    imm: 0xFF,
                });
            }
            Type::I32 => {
                buf.emit(X64Inst::ShlRI {
                    dest: dst_hw,
                    imm: 32,
                });
                buf.emit(X64Inst::ShrRI {
                    dest: dst_hw,
                    imm: 32,
                });
            }
            _ => {}
        }
    }
    fn emit_trunc(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        src: MachineReg,
        target_ty: Type,
    ) {
        Self::emit_zero_extend(buf, dst, src, target_ty, target_ty);
    }
    fn emit_int_to_float(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        src: MachineReg,
        _src_ty: Type,
    ) {
        buf.emit(X64Inst::Cvtsi2sd {
            dest: Self::fp_hw(dst),
            src: Self::gp_hw(src),
        });
    }
    fn emit_float_to_int(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, src: MachineReg) {
        buf.emit(X64Inst::Cvttsd2si {
            dest: Self::gp_hw(dst),
            src: Self::fp_hw(src),
        });
    }
    fn emit_mov_imm(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, value: u64) {
        let imm = i64::from_le_bytes(value.to_le_bytes());
        buf.emit(X64Inst::MovRI {
            dest: Self::gp_hw(dst),
            imm,
        });
    }
    fn emit_f64_const(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, bits: u64) {
        Self::emit_mov_imm_to_hw(buf, R11, bits);
        buf.emit(X64Inst::MovqXR {
            dest: Self::fp_hw(dst),
            src: R11,
        });
    }
    fn emit_load_incoming_stack_arg(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        incoming_offset: i32,
    ) {
        buf.emit(X64Inst::MovRM {
            dest: Self::gp_hw(dst),
            base: R14,
            offset: incoming_offset + 8,
        });
    }
    fn emit_cmp_gp_imm(buf: &mut CodeBuffer<Self::Arch>, reg: MachineReg, imm: u64) {
        if i32::try_from(imm as i64).is_ok() {
            buf.emit(X64Inst::CmpRI {
                reg: Self::gp_hw(reg),
                imm: imm as i32,
            });
        } else {
            Self::emit_mov_imm_to_hw(buf, R11, imm);
            buf.emit(X64Inst::CmpRR {
                a: Self::gp_hw(reg),
                b: R11,
            });
        }
    }
    fn emit_extract_payload(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        src: MachineReg,
        has_unboxed_float: bool,
        payload_bits: u8,
    ) {
        let dst_hw = Self::gp_hw(dst);
        if dst_hw.index != Self::gp_hw(src).index {
            buf.emit(X64Inst::MovRR {
                dest: dst_hw,
                src: Self::gp_hw(src),
            });
        }
        if has_unboxed_float {
            let mask = (1u64 << payload_bits) - 1;
            Self::emit_mov_imm_to_hw(buf, R11, mask);
            buf.emit(X64Inst::AndRR {
                dest: dst_hw,
                src: R11,
            });
        } else {
            buf.emit(X64Inst::ShrRI {
                dest: dst_hw,
                imm: 64 - payload_bits,
            });
        }
    }
    fn emit_is_tag(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        src: MachineReg,
        has_unboxed_float: bool,
        payload_bits: u8,
        tag_mask: u64,
        expected_tag: u64,
    ) {
        let dst_hw = Self::gp_hw(dst);
        buf.emit(X64Inst::MovRR {
            dest: dst_hw,
            src: Self::gp_hw(src),
        });
        if has_unboxed_float {
            buf.emit(X64Inst::ShrRI {
                dest: dst_hw,
                imm: payload_bits,
            });
        } else {
            Self::emit_mov_imm_to_hw(buf, R11, tag_mask);
            buf.emit(X64Inst::AndRR {
                dest: dst_hw,
                src: R11,
            });
        }
        Self::emit_cmp_gp_imm(buf, dst, expected_tag);
        buf.emit(X64Inst::MovRI32 {
            dest: dst_hw,
            imm: 0,
        });
        buf.emit(X64Inst::Setcc {
            dest: Self::gp_hw8(dst),
            cond: X64Cond::E,
        });
    }
    fn emit_make_tagged(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        payload: MachineReg,
        has_unboxed_float: bool,
        payload_bits: u8,
        encoded_tag_pattern: u64,
        tag: u64,
    ) {
        let dst_hw = Self::gp_hw(dst);
        buf.emit(X64Inst::MovRR {
            dest: dst_hw,
            src: Self::gp_hw(payload),
        });
        if has_unboxed_float {
            Self::emit_mov_imm_to_hw(buf, R11, encoded_tag_pattern);
            buf.emit(X64Inst::OrRR {
                dest: dst_hw,
                src: R11,
            });
        } else {
            buf.emit(X64Inst::ShlRI {
                dest: dst_hw,
                imm: 64 - payload_bits,
            });
            if tag != 0 {
                Self::emit_mov_imm_to_hw(buf, R11, tag);
                buf.emit(X64Inst::OrRR {
                    dest: dst_hw,
                    src: R11,
                });
            }
        }
    }
    fn emit_tag_of(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        src: MachineReg,
        has_unboxed_float: bool,
        payload_bits: u8,
        tag_mask: u64,
    ) {
        let dst_hw = Self::gp_hw(dst);
        buf.emit(X64Inst::MovRR {
            dest: dst_hw,
            src: Self::gp_hw(src),
        });
        if has_unboxed_float {
            buf.emit(X64Inst::ShrRI {
                dest: dst_hw,
                imm: payload_bits,
            });
        }
        Self::emit_mov_imm_to_hw(buf, R11, tag_mask);
        buf.emit(X64Inst::AndRR {
            dest: dst_hw,
            src: R11,
        });
    }
    fn emit_call_safepoint_handler(
        buf: &mut CodeBuffer<Self::Arch>,
        handler: u64,
        frame_size: u64,
    ) {
        buf.emit(X64Inst::MovRR {
            dest: RDI,
            src: RBP,
        });
        Self::emit_mov_imm_to_hw(buf, RSI, frame_size);
        Self::emit_mov_imm_to_hw(buf, R11, handler);
        buf.emit(X64Inst::CallR { target: R11 });
    }
    fn emit_lea_frame_slot(
        buf: &mut CodeBuffer<Self::Arch>,
        dst: MachineReg,
        slot: FrameSlotAccess,
    ) {
        buf.emit(X64Inst::Lea {
            dest: Self::gp_hw(dst),
            base: Self::slot_base_reg(slot),
            offset: slot.offset,
        });
    }

    fn bind_label(buf: &mut CodeBuffer<Self::Arch>, label: Label) {
        buf.bind_label(label);
    }
    fn emit_branch_to_label(buf: &mut CodeBuffer<Self::Arch>, label: Label) {
        let off = buf.emit(X64Inst::Jmp { offset: 0 });
        buf.add_reloc(off + 1, label, X64RelocKind::Rel32);
    }
    fn emit_cbz_to_label(buf: &mut CodeBuffer<Self::Arch>, reg: MachineReg, label: Label) {
        buf.emit(X64Inst::TestRR {
            a: Self::gp_hw(reg),
            b: Self::gp_hw(reg),
        });
        let off = buf.emit(X64Inst::Jcc {
            offset: 0,
            cond: X64Cond::E,
        });
        buf.add_reloc(off + 2, label, X64RelocKind::Rel32);
    }
    fn emit_cbnz_to_label(buf: &mut CodeBuffer<Self::Arch>, reg: MachineReg, label: Label) {
        buf.emit(X64Inst::TestRR {
            a: Self::gp_hw(reg),
            b: Self::gp_hw(reg),
        });
        let off = buf.emit(X64Inst::Jcc {
            offset: 0,
            cond: X64Cond::NE,
        });
        buf.add_reloc(off + 2, label, X64RelocKind::Rel32);
    }
    fn emit_branch_eq_to_label(buf: &mut CodeBuffer<Self::Arch>, label: Label) {
        let off = buf.emit(X64Inst::Jcc {
            offset: 0,
            cond: X64Cond::E,
        });
        buf.add_reloc(off + 2, label, X64RelocKind::Rel32);
    }
    fn emit_call_reg(buf: &mut CodeBuffer<Self::Arch>, reg: MachineReg) {
        buf.emit(X64Inst::CallR {
            target: Self::gp_hw(reg),
        });
    }
    fn emit_return_gp(buf: &mut CodeBuffer<Self::Arch>, src: MachineReg) {
        if src.index != 0 {
            buf.emit(X64Inst::MovRR {
                dest: RAX,
                src: Self::gp_hw(src),
            });
        }
    }
    fn emit_return_fp_bits(buf: &mut CodeBuffer<Self::Arch>, src: MachineReg) {
        buf.emit(X64Inst::MovqRX {
            dest: RAX,
            src: Self::fp_hw(src),
        });
    }
    fn emit_stack_adjust(buf: &mut CodeBuffer<Self::Arch>, amount: i32) {
        if amount >= 0 {
            buf.emit(X64Inst::SubRI {
                dest: RSP,
                imm: amount,
            });
        } else {
            buf.emit(X64Inst::AddRI {
                dest: RSP,
                imm: -amount,
            });
        }
    }
    fn emit_trap(buf: &mut CodeBuffer<Self::Arch>) {
        buf.emit(X64Inst::Int3);
    }
}
