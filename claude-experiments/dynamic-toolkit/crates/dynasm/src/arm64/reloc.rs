use super::inst::Arm64Inst;
use crate::{Arch, CodeBuffer};

/// Marker type for the ARM64 architecture.
pub struct Arm64;

/// ARM64 relocation kinds.
#[derive(Debug, Copy, Clone)]
pub enum Arm64RelocKind {
    /// B/BL — 26-bit signed offset in instruction words (bits [25:0]).
    Branch26,
    /// B.cond / CBZ / CBNZ — 19-bit signed offset in instruction words (bits [23:5]).
    Cond19,
    /// ADR — 21-bit signed byte offset split across immhi[23:5] and immlo[30:29].
    Adr21,
}

impl Arch for Arm64 {
    type Inst = Arm64Inst;
    type RelocKind = Arm64RelocKind;

    fn emit(buf: &mut CodeBuffer<Self>, inst: Self::Inst) -> usize {
        let offset = buf.current_offset();
        let word = inst.encode();
        buf.push_bytes(&word.to_le_bytes());
        offset
    }

    fn patch(code: &mut [u8], reloc_offset: usize, kind: Self::RelocKind, target_offset: usize) {
        let delta = (target_offset as i64) - (reloc_offset as i64);
        // ARM64 branch offsets are in instruction words (4 bytes)
        let delta_insns = delta / 4;

        // Read the existing instruction word
        let bytes = &code[reloc_offset..reloc_offset + 4];
        let mut word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);

        match kind {
            Arm64RelocKind::Branch26 => {
                let imm26 = (delta_insns as u32) & 0x03FF_FFFF;
                word = (word & !0x03FF_FFFF) | imm26;
            }
            Arm64RelocKind::Cond19 => {
                let imm19 = ((delta_insns as u32) & 0x7FFFF) << 5;
                word = (word & !(0x7FFFF << 5)) | imm19;
            }
            Arm64RelocKind::Adr21 => {
                // ADR uses byte offset, not instruction offset
                let immlo = ((delta as u32) & 0x3) << 29;
                let immhi = (((delta as u32) >> 2) & 0x7FFFF) << 5;
                word = (word & !(0x3 << 29) & !(0x7FFFF << 5)) | immlo | immhi;
            }
        }

        code[reloc_offset..reloc_offset + 4].copy_from_slice(&word.to_le_bytes());
    }
}
