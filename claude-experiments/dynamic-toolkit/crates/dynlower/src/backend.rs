use dynasm::arm64::inst::*;
use dynasm::arm64::reg::*;
use dynasm::arm64::reloc::Arm64RelocKind;
use dynasm::arm64::Arm64;
use dynasm::buffer::{CodeBuffer, Label};
use dynasm::x86_64::cond::Condition as X64Cond;
use dynasm::x86_64::inst::X64Inst;
use dynasm::x86_64::reg::{R14, RAX, RBP, RSP, X64Reg};
use dynasm::x86_64::{X64, X64RelocKind};
use dynasm::Arch;
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
    fn emit_store_gp_to_frame(buf: &mut CodeBuffer<Self::Arch>, src: MachineReg, slot: FrameSlotAccess);
    fn emit_load_gp_from_frame(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, slot: FrameSlotAccess);
    fn emit_store_fp_to_frame(buf: &mut CodeBuffer<Self::Arch>, src: MachineReg, slot: FrameSlotAccess);
    fn emit_load_fp_from_frame(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, slot: FrameSlotAccess);
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
    fn emit_int_to_float(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, src: MachineReg, src_ty: Type);
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
    fn emit_call_safepoint_handler(
        buf: &mut CodeBuffer<Self::Arch>,
        handler: u64,
        frame_size: u64,
    );

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
        buf.emit(Arm64Inst::mov(X28, SP));
        // Reserve 2 instructions for frame size adjustment (patched later).
        // This supports frame sizes > 4095 bytes by splitting into hi/lo parts.
        let patch_offset = buf.emit(Arm64Inst::sub_imm(SP, SP, 16));
        buf.emit(Arm64Inst::sub_imm(SP, SP, 0)); // placeholder for low bits
        buf.emit(Arm64Inst::stp(X29, X30, SP, 0, StpMode::SignedOffset));
        buf.emit(Arm64Inst::mov(X29, SP));
        patch_offset
    }

    pub fn emit_epilogue(buf: &mut CodeBuffer<Arm64>, patch_offsets: &mut Vec<usize>) {
        buf.emit(Arm64Inst::mov(SP, X29));
        buf.emit(Arm64Inst::ldp(X29, X30, SP, 0, LdpMode::SignedOffset));
        // Reserve 2 instructions for frame size restore (patched later).
        let add_offset = buf.emit(Arm64Inst::add_imm(SP, SP, 16));
        buf.emit(Arm64Inst::add_imm(SP, SP, 0)); // placeholder for low bits
        patch_offsets.push(add_offset);
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
        let sub_hi = Arm64Inst::SubImm { sf: 1, sh: 1, imm12: hi, rn: SP, rd: SP };
        let sub_lo = Arm64Inst::SubImm { sf: 1, sh: 0, imm12: lo, rn: SP, rd: SP };
        buf.patch_bytes(prologue_offset, &sub_hi.encode().to_le_bytes());
        buf.patch_bytes(prologue_offset + 4, &sub_lo.encode().to_le_bytes());

        // Epilogue: add SP, SP, #hi, LSL #12 ; add SP, SP, #lo
        let add_hi = Arm64Inst::AddImm { sf: 1, sh: 1, imm12: hi, rn: SP, rd: SP };
        let add_lo = Arm64Inst::AddImm { sf: 1, sh: 0, imm12: lo, rn: SP, rd: SP };
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

    pub fn emit_store_gp_to_frame(buf: &mut CodeBuffer<Arm64>, src: MachineReg, slot: FrameSlotAccess) {
        debug_assert!(slot.offset >= 0 && slot.offset % 8 == 0);
        buf.emit(Arm64Inst::str(
            Self::gp_hw(src, RegSize::X64),
            Self::slot_base_reg(slot),
            slot.offset,
        ));
    }

    pub fn emit_load_gp_from_frame(buf: &mut CodeBuffer<Arm64>, dst: MachineReg, slot: FrameSlotAccess) {
        debug_assert!(slot.offset >= 0 && slot.offset % 8 == 0);
        buf.emit(Arm64Inst::ldr(
            Self::gp_hw(dst, RegSize::X64),
            Self::slot_base_reg(slot),
            slot.offset,
        ));
    }

    pub fn emit_store_fp_to_frame(buf: &mut CodeBuffer<Arm64>, src: MachineReg, slot: FrameSlotAccess) {
        debug_assert!(slot.offset >= 0 && slot.offset % 8 == 0);
        buf.emit(Arm64Inst::str_fp(
            Self::fp_hw(src),
            Self::slot_base_reg(slot),
            slot.offset,
        ));
    }

    pub fn emit_load_fp_from_frame(buf: &mut CodeBuffer<Arm64>, dst: MachineReg, slot: FrameSlotAccess) {
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

    fn emit_store_gp_to_frame(buf: &mut CodeBuffer<Self::Arch>, src: MachineReg, slot: FrameSlotAccess) {
        Arm64Backend::emit_store_gp_to_frame(buf, src, slot);
    }

    fn emit_load_gp_from_frame(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, slot: FrameSlotAccess) {
        Arm64Backend::emit_load_gp_from_frame(buf, dst, slot);
    }

    fn emit_store_fp_to_frame(buf: &mut CodeBuffer<Self::Arch>, src: MachineReg, slot: FrameSlotAccess) {
        Arm64Backend::emit_store_fp_to_frame(buf, src, slot);
    }

    fn emit_load_fp_from_frame(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, slot: FrameSlotAccess) {
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
        let rd = Arm64Backend::gp_hw(dst, match target_ty { Type::I32 => RegSize::W32, _ => RegSize::X64 });
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
                let rd = Arm64Backend::gp_hw(dst, match target_ty { Type::I32 => RegSize::W32, _ => RegSize::X64 });
                let rn = Arm64Backend::gp_hw(src, match src_ty { Type::I32 | Type::I8 => RegSize::W32, _ => RegSize::X64 });
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
        let rn = Arm64Backend::gp_hw(src, match src_ty { Type::I32 | Type::I8 => RegSize::W32, _ => RegSize::X64 });
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
            buf.emit(Arm64Inst::LsrReg { sf: 1, rm: X28, rn, rd });
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
        buf.emit(Arm64Inst::cset(Arm64Backend::gp_hw(dst, RegSize::W32), Arm64Cond::EQ));
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
            buf.emit(Arm64Inst::LslReg { sf: 1, rm: X28, rn, rd });
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
            buf.emit(Arm64Inst::LsrReg { sf: 1, rm: X28, rn, rd });
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
                buf.emit(Arm64Inst::SubImm { sf: 1, sh: 1, imm12: hi, rn: SP, rd: SP });
            }
            if lo > 0 || hi == 0 {
                buf.emit(Arm64Inst::sub_imm(SP, SP, lo));
            }
        } else {
            let abs = -amount;
            let hi = (abs >> 12) & 0xFFF;
            let lo = abs & 0xFFF;
            if hi > 0 {
                buf.emit(Arm64Inst::AddImm { sf: 1, sh: 1, imm12: hi, rn: SP, rd: SP });
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
        X64Reg::new(reg.index, dynasm::x86_64::reg::Size::S64)
    }

    fn gp_hw8(reg: MachineReg) -> X64Reg {
        debug_assert_eq!(reg.class, MachineRegClass::Gp);
        X64Reg::new(reg.index, dynasm::x86_64::reg::Size::S8)
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
}

impl LoweringBackend for X64Backend {
    type Arch = X64;

    fn emit_prologue(buf: &mut CodeBuffer<Self::Arch>) -> usize {
        buf.emit(X64Inst::Push { reg: RBP });
        buf.emit(X64Inst::MovRR { dest: RBP, src: RSP });
        buf.emit(X64Inst::MovRR { dest: R14, src: RSP });
        buf.emit(X64Inst::SubRI { dest: RSP, imm: 16 })
    }

    fn emit_epilogue(buf: &mut CodeBuffer<Self::Arch>, patch_offsets: &mut Vec<usize>) {
        let add_offset = buf.emit(X64Inst::AddRI { dest: RSP, imm: 16 });
        patch_offsets.push(add_offset);
        buf.emit(X64Inst::Pop { reg: RBP });
        buf.emit(X64Inst::Ret);
    }

    fn emit_frame_size_patch(
        buf: &mut CodeBuffer<Self::Arch>,
        prologue_offset: usize,
        epilogue_offsets: &[usize],
        frame_size: i32,
    ) {
        let sub = X64Inst::SubRI { dest: RSP, imm: frame_size }.encode();
        buf.patch_bytes(prologue_offset, &sub);
        let add = X64Inst::AddRI { dest: RSP, imm: frame_size }.encode();
        for &offset in epilogue_offsets {
            buf.patch_bytes(offset, &add);
        }
    }

    fn emit_gp_move(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, src: MachineReg) {
        if dst.index != src.index {
            buf.emit(X64Inst::MovRR { dest: Self::gp_hw(dst), src: Self::gp_hw(src) });
        }
    }

    fn emit_stack_pointer_to_gp(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg) {
        buf.emit(X64Inst::MovRR {
            dest: Self::gp_hw(dst),
            src: RSP,
        });
    }

    fn emit_fp_to_gp_move(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, src: MachineReg) {
        buf.emit(X64Inst::MovqRX { dest: Self::gp_hw(dst), src: Self::fp_hw(src) });
    }

    fn emit_gp_to_fp_move(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, src: MachineReg) {
        buf.emit(X64Inst::MovqXR { dest: Self::fp_hw(dst), src: Self::gp_hw(src) });
    }

    fn emit_store_gp_to_frame(_buf: &mut CodeBuffer<Self::Arch>, _src: MachineReg, _slot: FrameSlotAccess) { todo!("x64 frame stores") }
    fn emit_load_gp_from_frame(_buf: &mut CodeBuffer<Self::Arch>, _dst: MachineReg, _slot: FrameSlotAccess) { todo!("x64 frame loads") }
    fn emit_store_fp_to_frame(_buf: &mut CodeBuffer<Self::Arch>, _src: MachineReg, _slot: FrameSlotAccess) { todo!("x64 fp frame stores") }
    fn emit_load_fp_from_frame(_buf: &mut CodeBuffer<Self::Arch>, _dst: MachineReg, _slot: FrameSlotAccess) { todo!("x64 fp frame loads") }
    fn emit_store_gp_to_stack_arg(_buf: &mut CodeBuffer<Self::Arch>, _src: MachineReg, _offset: i32) { todo!("x64 outgoing stack args") }

    fn emit_gp_binop(buf: &mut CodeBuffer<Self::Arch>, op: MachineGpBinOp, dst: MachineReg, lhs: MachineReg, rhs: MachineReg, _size: MachineWordSize) {
        if dst.index != lhs.index {
            buf.emit(X64Inst::MovRR { dest: Self::gp_hw(dst), src: Self::gp_hw(lhs) });
        }
        match op {
            MachineGpBinOp::Add => buf.emit(X64Inst::AddRR { dest: Self::gp_hw(dst), src: Self::gp_hw(rhs) }),
            MachineGpBinOp::Sub => buf.emit(X64Inst::SubRR { dest: Self::gp_hw(dst), src: Self::gp_hw(rhs) }),
            MachineGpBinOp::Mul => buf.emit(X64Inst::ImulRR { dest: Self::gp_hw(dst), src: Self::gp_hw(rhs) }),
            _ => todo!("x64 gp binop {:?}", op),
        };
    }

    fn emit_fp_binop(_buf: &mut CodeBuffer<Self::Arch>, _op: MachineFpBinOp, _dst: MachineReg, _lhs: MachineReg, _rhs: MachineReg) { todo!("x64 fp binops") }

    fn emit_icmp_set(buf: &mut CodeBuffer<Self::Arch>, op: CmpOp, dst: MachineReg, lhs: MachineReg, rhs: MachineReg, _size: MachineWordSize) {
        buf.emit(X64Inst::CmpRR { a: Self::gp_hw(lhs), b: Self::gp_hw(rhs) });
        buf.emit(X64Inst::Setcc { dest: Self::gp_hw8(dst), cond: Self::cond(op) });
    }

    fn emit_fcmp_set(_buf: &mut CodeBuffer<Self::Arch>, _op: CmpOp, _dst: MachineReg, _lhs: MachineReg, _rhs: MachineReg) { todo!("x64 fp cmp") }
    fn emit_gp_select(_buf: &mut CodeBuffer<Self::Arch>, _dst: MachineReg, _cond: MachineReg, _when_true: MachineReg, _when_false: MachineReg, _size: MachineWordSize) { todo!("x64 select") }
    fn emit_fp_select(_buf: &mut CodeBuffer<Self::Arch>, _dst: MachineReg, _cond: MachineReg, _when_true: MachineReg, _when_false: MachineReg) { todo!("x64 fp select") }
    fn emit_load_gp(_buf: &mut CodeBuffer<Self::Arch>, _dst: MachineReg, _base: MachineReg, _offset: i32, _size: MachineWordSize) { todo!("x64 load gp") }
    fn emit_load_fp(_buf: &mut CodeBuffer<Self::Arch>, _dst: MachineReg, _base: MachineReg, _offset: i32) { todo!("x64 load fp") }
    fn emit_store_gp(_buf: &mut CodeBuffer<Self::Arch>, _src: MachineReg, _base: MachineReg, _offset: i32, _size: MachineWordSize) { todo!("x64 store gp") }
    fn emit_store_fp(_buf: &mut CodeBuffer<Self::Arch>, _src: MachineReg, _base: MachineReg, _offset: i32) { todo!("x64 store fp") }
    fn emit_gp_neg(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, src: MachineReg, _size: MachineWordSize) {
        if dst.index != src.index {
            buf.emit(X64Inst::MovRR { dest: Self::gp_hw(dst), src: Self::gp_hw(src) });
        }
        buf.emit(X64Inst::Neg { reg: Self::gp_hw(dst) });
    }
    fn emit_gp_not(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, src: MachineReg, _size: MachineWordSize) {
        if dst.index != src.index {
            buf.emit(X64Inst::MovRR { dest: Self::gp_hw(dst), src: Self::gp_hw(src) });
        }
        buf.emit(X64Inst::Not { reg: Self::gp_hw(dst) });
    }
    fn emit_fp_neg(_buf: &mut CodeBuffer<Self::Arch>, _dst: MachineReg, _src: MachineReg) { todo!("x64 fp neg") }
    fn emit_sign_extend(_buf: &mut CodeBuffer<Self::Arch>, _dst: MachineReg, _src: MachineReg, _src_ty: Type, _target_ty: Type) { todo!("x64 sext") }
    fn emit_zero_extend(_buf: &mut CodeBuffer<Self::Arch>, _dst: MachineReg, _src: MachineReg, _src_ty: Type, _target_ty: Type) { todo!("x64 zext") }
    fn emit_trunc(_buf: &mut CodeBuffer<Self::Arch>, _dst: MachineReg, _src: MachineReg, _target_ty: Type) { todo!("x64 trunc") }
    fn emit_int_to_float(_buf: &mut CodeBuffer<Self::Arch>, _dst: MachineReg, _src: MachineReg, _src_ty: Type) { todo!("x64 int->float") }
    fn emit_float_to_int(_buf: &mut CodeBuffer<Self::Arch>, _dst: MachineReg, _src: MachineReg) { todo!("x64 float->int") }
    fn emit_mov_imm(buf: &mut CodeBuffer<Self::Arch>, dst: MachineReg, value: u64) {
        let imm = i64::from_le_bytes(value.to_le_bytes());
        buf.emit(X64Inst::MovRI { dest: Self::gp_hw(dst), imm });
    }
    fn emit_f64_const(_buf: &mut CodeBuffer<Self::Arch>, _dst: MachineReg, _bits: u64) { todo!("x64 f64 const") }
    fn emit_load_incoming_stack_arg(_buf: &mut CodeBuffer<Self::Arch>, _dst: MachineReg, _incoming_offset: i32) { todo!("x64 incoming stack arg") }
    fn emit_cmp_gp_imm(buf: &mut CodeBuffer<Self::Arch>, reg: MachineReg, imm: u64) {
        buf.emit(X64Inst::CmpRI { reg: Self::gp_hw(reg), imm: imm as i32 });
    }
    fn emit_extract_payload(_buf: &mut CodeBuffer<Self::Arch>, _dst: MachineReg, _src: MachineReg, _has_unboxed_float: bool, _payload_bits: u8) { todo!("x64 payload") }
    fn emit_is_tag(_buf: &mut CodeBuffer<Self::Arch>, _dst: MachineReg, _src: MachineReg, _has_unboxed_float: bool, _payload_bits: u8, _tag_mask: u64, _expected_tag: u64) { todo!("x64 is_tag") }
    fn emit_make_tagged(_buf: &mut CodeBuffer<Self::Arch>, _dst: MachineReg, _payload: MachineReg, _has_unboxed_float: bool, _payload_bits: u8, _encoded_tag_pattern: u64, _tag: u64) { todo!("x64 make_tagged") }
    fn emit_tag_of(_buf: &mut CodeBuffer<Self::Arch>, _dst: MachineReg, _src: MachineReg, _has_unboxed_float: bool, _payload_bits: u8, _tag_mask: u64) { todo!("x64 tag_of") }
    fn emit_call_safepoint_handler(_buf: &mut CodeBuffer<Self::Arch>, _handler: u64, _frame_size: u64) { todo!("x64 safepoint handler") }
    fn emit_lea_frame_slot(_buf: &mut CodeBuffer<Self::Arch>, _dst: MachineReg, _slot: FrameSlotAccess) { todo!("x64 lea frame slot") }

    fn bind_label(buf: &mut CodeBuffer<Self::Arch>, label: Label) { buf.bind_label(label); }
    fn emit_branch_to_label(buf: &mut CodeBuffer<Self::Arch>, label: Label) {
        let off = buf.emit(X64Inst::Jmp { offset: 0 });
        buf.add_reloc(off + 1, label, X64RelocKind::Rel32);
    }
    fn emit_cbz_to_label(buf: &mut CodeBuffer<Self::Arch>, reg: MachineReg, label: Label) {
        buf.emit(X64Inst::TestRR { a: Self::gp_hw(reg), b: Self::gp_hw(reg) });
        let off = buf.emit(X64Inst::Jcc { offset: 0, cond: X64Cond::E });
        buf.add_reloc(off + 2, label, X64RelocKind::Rel32);
    }
    fn emit_cbnz_to_label(buf: &mut CodeBuffer<Self::Arch>, reg: MachineReg, label: Label) {
        buf.emit(X64Inst::TestRR { a: Self::gp_hw(reg), b: Self::gp_hw(reg) });
        let off = buf.emit(X64Inst::Jcc { offset: 0, cond: X64Cond::NE });
        buf.add_reloc(off + 2, label, X64RelocKind::Rel32);
    }
    fn emit_branch_eq_to_label(buf: &mut CodeBuffer<Self::Arch>, label: Label) {
        let off = buf.emit(X64Inst::Jcc { offset: 0, cond: X64Cond::E });
        buf.add_reloc(off + 2, label, X64RelocKind::Rel32);
    }
    fn emit_call_reg(buf: &mut CodeBuffer<Self::Arch>, reg: MachineReg) { buf.emit(X64Inst::CallR { target: Self::gp_hw(reg) }); }
    fn emit_return_gp(buf: &mut CodeBuffer<Self::Arch>, src: MachineReg) {
        if src.index != 0 {
            buf.emit(X64Inst::MovRR { dest: RAX, src: Self::gp_hw(src) });
        }
    }
    fn emit_return_fp_bits(buf: &mut CodeBuffer<Self::Arch>, src: MachineReg) {
        buf.emit(X64Inst::MovqRX { dest: RAX, src: Self::fp_hw(src) });
    }
    fn emit_stack_adjust(buf: &mut CodeBuffer<Self::Arch>, amount: i32) {
        if amount >= 0 {
            buf.emit(X64Inst::SubRI { dest: RSP, imm: amount });
        } else {
            buf.emit(X64Inst::AddRI { dest: RSP, imm: -amount });
        }
    }
    fn emit_trap(buf: &mut CodeBuffer<Self::Arch>) { buf.emit(X64Inst::Int3); }
}
