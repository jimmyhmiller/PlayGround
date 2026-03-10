use smallvec::{SmallVec, smallvec};
use super::cond::Condition;
use super::encoding::*;
use super::reg::X64Reg;

/// x86-64 instruction set.
///
/// All branch/jump instructions take raw `i32` offsets (not label indices).
/// Label-based jumps should use `CodeBuffer::emit_with_reloc()` through
/// the `X64` arch impl.
#[derive(Debug, Clone)]
pub enum X64Inst {
    // === Move instructions ===
    /// MOV r64, r64
    MovRR { dest: X64Reg, src: X64Reg },
    /// MOV r64, imm64
    MovRI { dest: X64Reg, imm: i64 },
    /// MOV r/m64, imm32 (sign-extended)
    MovRI32 { dest: X64Reg, imm: i32 },
    /// MOV r64, [base + offset]
    MovRM { dest: X64Reg, base: X64Reg, offset: i32 },
    /// MOV [base + offset], r64
    MovMR { base: X64Reg, offset: i32, src: X64Reg },
    /// MOV r64, [base + index*1]
    MovRMIndexed { dest: X64Reg, base: X64Reg, index: X64Reg },
    /// MOV [base + index*1], r64
    MovMRIndexed { base: X64Reg, index: X64Reg, src: X64Reg },
    /// LEA r64, [base + offset]
    Lea { dest: X64Reg, base: X64Reg, offset: i32 },
    /// LEA r64, [RIP + disp32]
    LeaRipRel { dest: X64Reg, offset: i32 },

    // === Arithmetic instructions ===
    /// ADD r64, r64
    AddRR { dest: X64Reg, src: X64Reg },
    /// ADD r/m64, imm32
    AddRI { dest: X64Reg, imm: i32 },
    /// SUB r64, r64
    SubRR { dest: X64Reg, src: X64Reg },
    /// SUB r/m64, imm32
    SubRI { dest: X64Reg, imm: i32 },
    /// IMUL r64, r/m64
    ImulRR { dest: X64Reg, src: X64Reg },
    /// IMUL r64, r/m64, imm32
    ImulRRI { dest: X64Reg, src: X64Reg, imm: i32 },
    /// IDIV r/m64
    Idiv { divisor: X64Reg },
    /// CQO
    Cqo,
    /// NEG r/m64
    Neg { reg: X64Reg },

    // === Bitwise instructions ===
    /// AND r64, r64
    AndRR { dest: X64Reg, src: X64Reg },
    /// AND r/m64, imm32
    AndRI { dest: X64Reg, imm: i32 },
    /// OR r64, r64
    OrRR { dest: X64Reg, src: X64Reg },
    /// OR r/m64, imm32
    OrRI { dest: X64Reg, imm: i32 },
    /// XOR r64, r64
    XorRR { dest: X64Reg, src: X64Reg },
    /// XOR r/m64, imm32
    XorRI { dest: X64Reg, imm: i32 },
    /// NOT r/m64
    Not { reg: X64Reg },

    // === Shift instructions ===
    /// SHL r/m64, imm8
    ShlRI { dest: X64Reg, imm: u8 },
    /// SHL r/m64, CL
    ShlRCL { dest: X64Reg },
    /// SHR r/m64, imm8
    ShrRI { dest: X64Reg, imm: u8 },
    /// SHR r/m64, CL
    ShrRCL { dest: X64Reg },
    /// SAR r/m64, imm8
    SarRI { dest: X64Reg, imm: u8 },
    /// SAR r/m64, CL
    SarRCL { dest: X64Reg },

    // === Comparison instructions ===
    /// CMP r64, r64
    CmpRR { a: X64Reg, b: X64Reg },
    /// CMP r/m64, imm32
    CmpRI { reg: X64Reg, imm: i32 },
    /// TEST r64, r64
    TestRR { a: X64Reg, b: X64Reg },
    /// TEST r/m64, imm32
    TestRI { reg: X64Reg, imm: i32 },
    /// SETcc r/m8
    Setcc { dest: X64Reg, cond: Condition },

    // === Control flow instructions ===
    /// JMP rel32
    Jmp { offset: i32 },
    /// Jcc rel32
    Jcc { offset: i32, cond: Condition },
    /// CALL r64
    CallR { target: X64Reg },
    /// CALL rel32
    CallRel { offset: i32 },
    /// RET
    Ret,

    // === Stack instructions ===
    /// PUSH r64
    Push { reg: X64Reg },
    /// POP r64
    Pop { reg: X64Reg },

    // === Floating-point (SSE2) ===
    /// ADDSD xmm, xmm
    Addsd { dest: X64Reg, src: X64Reg },
    /// SUBSD xmm, xmm
    Subsd { dest: X64Reg, src: X64Reg },
    /// MULSD xmm, xmm
    Mulsd { dest: X64Reg, src: X64Reg },
    /// DIVSD xmm, xmm
    Divsd { dest: X64Reg, src: X64Reg },
    /// ROUNDSD xmm, xmm, imm8
    Roundsd { dest: X64Reg, src: X64Reg, mode: u8 },
    /// MOVSD xmm, xmm
    MovsdRR { dest: X64Reg, src: X64Reg },
    /// MOVSD xmm, [base + offset]
    MovsdRM { dest: X64Reg, base: X64Reg, offset: i32 },
    /// MOVSD [base + offset], xmm
    MovsdMR { base: X64Reg, offset: i32, src: X64Reg },
    /// MOVQ r64, xmm
    MovqRX { dest: X64Reg, src: X64Reg },
    /// MOVQ xmm, r64
    MovqXR { dest: X64Reg, src: X64Reg },
    /// CVTSI2SD xmm, r64
    Cvtsi2sd { dest: X64Reg, src: X64Reg },
    /// UCOMISD xmm, xmm
    Ucomisd { a: X64Reg, b: X64Reg },

    // === Atomic instructions ===
    /// MFENCE
    Mfence,
    /// LOCK CMPXCHG [base], r64
    LockCmpxchg { base: X64Reg, src: X64Reg },

    // === Misc instructions ===
    /// INT3
    Int3,
    /// NOP
    Nop,
    /// JMP r64 (indirect)
    JmpR { target: X64Reg },
}

impl X64Inst {
    /// Encode this instruction to bytes.
    pub fn encode(&self) -> SmallVec<[u8; 15]> {
        match self {
            X64Inst::MovRR { dest, src } => {
                smallvec![rex_w(src.index, dest.index), 0x89, modrm(0b11, src.index, dest.index)]
            }
            X64Inst::MovRI { dest, imm } => {
                let mut bytes: SmallVec<[u8; 15]> = smallvec![rex_w(0, dest.index), 0xB8 + (dest.index & 0x7)];
                bytes.extend_from_slice(&imm.to_le_bytes());
                bytes
            }
            X64Inst::MovRI32 { dest, imm } => {
                let mut bytes: SmallVec<[u8; 15]> = smallvec![rex_w(0, dest.index), 0xC7, modrm(0b11, 0, dest.index)];
                bytes.extend_from_slice(&imm.to_le_bytes());
                bytes
            }
            X64Inst::MovRM { dest, base, offset } => {
                SmallVec::from_vec(encode_mem_op(0x8B, dest.index, base.index, *offset, true))
            }
            X64Inst::MovMR { base, offset, src } => {
                SmallVec::from_vec(encode_mem_op(0x89, src.index, base.index, *offset, true))
            }
            X64Inst::MovRMIndexed { dest, base, index } => {
                let base_idx = base.index;
                let index_idx = index.index;
                let dest_idx = dest.index;
                let rex_byte = 0x48 | ((dest_idx >> 3) << 2) | ((index_idx >> 3) << 1) | (base_idx >> 3);
                let modrm_byte = ((dest_idx & 0b111) << 3) | 0b100;
                let sib_byte = ((index_idx & 0b111) << 3) | (base_idx & 0b111);
                if (base_idx & 0b111) == 5 {
                    let modrm_byte = 0b01_000_100 | ((dest_idx & 0b111) << 3);
                    smallvec![rex_byte, 0x8B, modrm_byte, sib_byte, 0x00]
                } else {
                    smallvec![rex_byte, 0x8B, modrm_byte, sib_byte]
                }
            }
            X64Inst::MovMRIndexed { base, index, src } => {
                let base_idx = base.index;
                let index_idx = index.index;
                let src_idx = src.index;
                let rex_byte = 0x48 | ((src_idx >> 3) << 2) | ((index_idx >> 3) << 1) | (base_idx >> 3);
                let modrm_byte = ((src_idx & 0b111) << 3) | 0b100;
                let sib_byte = ((index_idx & 0b111) << 3) | (base_idx & 0b111);
                if (base_idx & 0b111) == 5 {
                    let modrm_byte = 0b01_000_100 | ((src_idx & 0b111) << 3);
                    smallvec![rex_byte, 0x89, modrm_byte, sib_byte, 0x00]
                } else {
                    smallvec![rex_byte, 0x89, modrm_byte, sib_byte]
                }
            }
            X64Inst::Lea { dest, base, offset } => {
                SmallVec::from_vec(encode_mem_op(0x8D, dest.index, base.index, *offset, true))
            }
            X64Inst::LeaRipRel { dest, offset } => {
                let reg = dest.index;
                let rex_byte = 0x48 | ((reg >> 3) << 2);
                let modrm_byte = ((reg & 0b111) << 3) | 0b101;
                let mut bytes: SmallVec<[u8; 15]> = smallvec![rex_byte, 0x8D, modrm_byte];
                bytes.extend_from_slice(&offset.to_le_bytes());
                bytes
            }

            // === Arithmetic ===
            X64Inst::AddRR { dest, src } => {
                smallvec![rex_w(src.index, dest.index), 0x01, modrm(0b11, src.index, dest.index)]
            }
            X64Inst::AddRI { dest, imm } => {
                let mut bytes: SmallVec<[u8; 15]> = smallvec![rex_w(0, dest.index), 0x81, modrm(0b11, 0, dest.index)];
                bytes.extend_from_slice(&imm.to_le_bytes());
                bytes
            }
            X64Inst::SubRR { dest, src } => {
                smallvec![rex_w(src.index, dest.index), 0x29, modrm(0b11, src.index, dest.index)]
            }
            X64Inst::SubRI { dest, imm } => {
                let mut bytes: SmallVec<[u8; 15]> = smallvec![rex_w(0, dest.index), 0x81, modrm(0b11, 5, dest.index)];
                bytes.extend_from_slice(&imm.to_le_bytes());
                bytes
            }
            X64Inst::ImulRR { dest, src } => {
                smallvec![rex_w(dest.index, src.index), 0x0F, 0xAF, modrm(0b11, dest.index, src.index)]
            }
            X64Inst::ImulRRI { dest, src, imm } => {
                let mut bytes: SmallVec<[u8; 15]> = smallvec![rex_w(dest.index, src.index), 0x69, modrm(0b11, dest.index, src.index)];
                bytes.extend_from_slice(&imm.to_le_bytes());
                bytes
            }
            X64Inst::Idiv { divisor } => {
                smallvec![rex_w(0, divisor.index), 0xF7, modrm(0b11, 7, divisor.index)]
            }
            X64Inst::Cqo => {
                smallvec![0x48, 0x99]
            }
            X64Inst::Neg { reg } => {
                smallvec![rex_w(0, reg.index), 0xF7, modrm(0b11, 3, reg.index)]
            }

            // === Bitwise ===
            X64Inst::AndRR { dest, src } => {
                smallvec![rex_w(src.index, dest.index), 0x21, modrm(0b11, src.index, dest.index)]
            }
            X64Inst::AndRI { dest, imm } => {
                let mut bytes: SmallVec<[u8; 15]> = smallvec![rex_w(0, dest.index), 0x81, modrm(0b11, 4, dest.index)];
                bytes.extend_from_slice(&imm.to_le_bytes());
                bytes
            }
            X64Inst::OrRR { dest, src } => {
                smallvec![rex_w(src.index, dest.index), 0x09, modrm(0b11, src.index, dest.index)]
            }
            X64Inst::OrRI { dest, imm } => {
                let mut bytes: SmallVec<[u8; 15]> = smallvec![rex_w(0, dest.index), 0x81, modrm(0b11, 1, dest.index)];
                bytes.extend_from_slice(&imm.to_le_bytes());
                bytes
            }
            X64Inst::XorRR { dest, src } => {
                smallvec![rex_w(src.index, dest.index), 0x31, modrm(0b11, src.index, dest.index)]
            }
            X64Inst::XorRI { dest, imm } => {
                let mut bytes: SmallVec<[u8; 15]> = smallvec![rex_w(0, dest.index), 0x81, modrm(0b11, 6, dest.index)];
                bytes.extend_from_slice(&imm.to_le_bytes());
                bytes
            }
            X64Inst::Not { reg } => {
                smallvec![rex_w(0, reg.index), 0xF7, modrm(0b11, 2, reg.index)]
            }

            // === Shift ===
            X64Inst::ShlRI { dest, imm } => {
                smallvec![rex_w(0, dest.index), 0xC1, modrm(0b11, 4, dest.index), *imm]
            }
            X64Inst::ShlRCL { dest } => {
                smallvec![rex_w(0, dest.index), 0xD3, modrm(0b11, 4, dest.index)]
            }
            X64Inst::ShrRI { dest, imm } => {
                smallvec![rex_w(0, dest.index), 0xC1, modrm(0b11, 5, dest.index), *imm]
            }
            X64Inst::ShrRCL { dest } => {
                smallvec![rex_w(0, dest.index), 0xD3, modrm(0b11, 5, dest.index)]
            }
            X64Inst::SarRI { dest, imm } => {
                smallvec![rex_w(0, dest.index), 0xC1, modrm(0b11, 7, dest.index), *imm]
            }
            X64Inst::SarRCL { dest } => {
                smallvec![rex_w(0, dest.index), 0xD3, modrm(0b11, 7, dest.index)]
            }

            // === Comparison ===
            X64Inst::CmpRR { a, b } => {
                smallvec![rex_w(b.index, a.index), 0x39, modrm(0b11, b.index, a.index)]
            }
            X64Inst::CmpRI { reg, imm } => {
                let mut bytes: SmallVec<[u8; 15]> = smallvec![rex_w(0, reg.index), 0x81, modrm(0b11, 7, reg.index)];
                bytes.extend_from_slice(&imm.to_le_bytes());
                bytes
            }
            X64Inst::TestRR { a, b } => {
                smallvec![rex_w(b.index, a.index), 0x85, modrm(0b11, b.index, a.index)]
            }
            X64Inst::TestRI { reg, imm } => {
                let mut bytes: SmallVec<[u8; 15]> = smallvec![rex_w(0, reg.index), 0xF7, modrm(0b11, 0, reg.index)];
                bytes.extend_from_slice(&imm.to_le_bytes());
                bytes
            }
            X64Inst::Setcc { dest, cond } => {
                let mut bytes: SmallVec<[u8; 15]> = SmallVec::new();
                if let Some(r) = rex_opt(0, dest.index) {
                    bytes.push(r);
                }
                bytes.push(0x0F);
                bytes.push(0x90 + (*cond as u8));
                bytes.push(modrm(0b11, 0, dest.index));
                bytes
            }

            // === Control flow ===
            X64Inst::Jmp { offset } => {
                let mut bytes: SmallVec<[u8; 15]> = smallvec![0xE9];
                bytes.extend_from_slice(&offset.to_le_bytes());
                bytes
            }
            X64Inst::Jcc { offset, cond } => {
                let mut bytes: SmallVec<[u8; 15]> = smallvec![0x0F, 0x80 + (*cond as u8)];
                bytes.extend_from_slice(&offset.to_le_bytes());
                bytes
            }
            X64Inst::CallR { target } => {
                let mut bytes: SmallVec<[u8; 15]> = SmallVec::new();
                if target.needs_rex_ext() {
                    bytes.push(rex(false, false, false, true));
                }
                bytes.push(0xFF);
                bytes.push(modrm(0b11, 2, target.index));
                bytes
            }
            X64Inst::CallRel { offset } => {
                let mut bytes: SmallVec<[u8; 15]> = smallvec![0xE8];
                bytes.extend_from_slice(&offset.to_le_bytes());
                bytes
            }
            X64Inst::Ret => {
                smallvec![0xC3]
            }

            // === Stack ===
            X64Inst::Push { reg } => {
                if reg.needs_rex_ext() {
                    smallvec![rex(false, false, false, true), 0x50 + (reg.index & 0x7)]
                } else {
                    smallvec![0x50 + reg.index]
                }
            }
            X64Inst::Pop { reg } => {
                if reg.needs_rex_ext() {
                    smallvec![rex(false, false, false, true), 0x58 + (reg.index & 0x7)]
                } else {
                    smallvec![0x58 + reg.index]
                }
            }

            // === Floating-point (SSE2) ===
            X64Inst::Addsd { dest, src } => {
                encode_sse_rr(0xF2, &[0x0F, 0x58], dest.index, src.index)
            }
            X64Inst::Subsd { dest, src } => {
                encode_sse_rr(0xF2, &[0x0F, 0x5C], dest.index, src.index)
            }
            X64Inst::Mulsd { dest, src } => {
                encode_sse_rr(0xF2, &[0x0F, 0x59], dest.index, src.index)
            }
            X64Inst::Divsd { dest, src } => {
                encode_sse_rr(0xF2, &[0x0F, 0x5E], dest.index, src.index)
            }
            X64Inst::Roundsd { dest, src, mode } => {
                let mut bytes: SmallVec<[u8; 15]> = smallvec![0x66];
                if dest.needs_rex_ext() || src.needs_rex_ext() {
                    bytes.push(rex(false, dest.index >= 8, false, src.index >= 8));
                }
                bytes.extend_from_slice(&[0x0F, 0x3A, 0x0B]);
                bytes.push(modrm(0b11, dest.index, src.index));
                bytes.push(*mode);
                bytes
            }
            X64Inst::MovsdRR { dest, src } => {
                encode_sse_rr(0xF2, &[0x0F, 0x10], dest.index, src.index)
            }
            X64Inst::MovsdRM { dest, base, offset } => {
                let mut bytes: SmallVec<[u8; 15]> = smallvec![0xF2];
                bytes.extend_from_slice(&encode_mem_op_no_rex(0x10, dest.index, base.index, *offset));
                bytes
            }
            X64Inst::MovsdMR { base, offset, src } => {
                let mut bytes: SmallVec<[u8; 15]> = smallvec![0xF2];
                bytes.extend_from_slice(&encode_mem_op_no_rex(0x11, src.index, base.index, *offset));
                bytes
            }
            X64Inst::MovqRX { dest, src } => {
                smallvec![0x66, rex_w(src.index, dest.index), 0x0F, 0x7E, modrm(0b11, src.index, dest.index)]
            }
            X64Inst::MovqXR { dest, src } => {
                smallvec![0x66, rex_w(dest.index, src.index), 0x0F, 0x6E, modrm(0b11, dest.index, src.index)]
            }
            X64Inst::Cvtsi2sd { dest, src } => {
                smallvec![0xF2, rex_w(dest.index, src.index), 0x0F, 0x2A, modrm(0b11, dest.index, src.index)]
            }
            X64Inst::Ucomisd { a, b } => {
                let mut bytes: SmallVec<[u8; 15]> = smallvec![0x66];
                if a.needs_rex_ext() || b.needs_rex_ext() {
                    bytes.push(rex(false, a.index >= 8, false, b.index >= 8));
                }
                bytes.extend_from_slice(&[0x0F, 0x2E, modrm(0b11, a.index, b.index)]);
                bytes
            }

            // === Atomic ===
            X64Inst::Mfence => {
                smallvec![0x0F, 0xAE, 0xF0]
            }
            X64Inst::LockCmpxchg { base, src } => {
                let mut bytes: SmallVec<[u8; 15]> = smallvec![0xF0, rex_w(src.index, base.index), 0x0F, 0xB1];
                let mut modrm_bytes = Vec::new();
                encode_modrm_mem(&mut modrm_bytes, src.index, base.index, 0);
                bytes.extend_from_slice(&modrm_bytes);
                bytes
            }

            // === Misc ===
            X64Inst::Int3 => smallvec![0xCC],
            X64Inst::Nop => smallvec![0x90],
            X64Inst::JmpR { target } => {
                let mut bytes: SmallVec<[u8; 15]> = SmallVec::new();
                if target.needs_rex_ext() {
                    bytes.push(rex(false, false, false, true));
                }
                bytes.push(0xFF);
                bytes.push(modrm(0b11, 4, target.index));
                bytes
            }
        }
    }

    /// Get the size of this instruction in bytes.
    pub fn size(&self) -> usize {
        self.encode().len()
    }
}

/// Helper to encode SSE register-register instructions.
fn encode_sse_rr(prefix: u8, opcode: &[u8], dest: u8, src: u8) -> SmallVec<[u8; 15]> {
    let mut bytes: SmallVec<[u8; 15]> = smallvec![prefix];
    if dest >= 8 || src >= 8 {
        bytes.push(rex(false, dest >= 8, false, src >= 8));
    }
    bytes.extend_from_slice(opcode);
    bytes.push(modrm(0b11, dest, src));
    bytes
}
