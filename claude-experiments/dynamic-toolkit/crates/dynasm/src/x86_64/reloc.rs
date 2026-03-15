use super::inst::X64Inst;
use crate::{Arch, CodeBuffer};

/// Marker type for the x86-64 architecture.
pub struct X64;

/// x86-64 relocation kinds.
#[derive(Debug, Copy, Clone)]
pub enum X64RelocKind {
    /// 4-byte relative displacement (for JMP/Jcc/CALL rel32).
    /// The displacement is relative to the end of the instruction
    /// (i.e., reloc_offset + 4).
    Rel32,
}

impl Arch for X64 {
    type Inst = X64Inst;
    type RelocKind = X64RelocKind;

    fn emit(buf: &mut CodeBuffer<Self>, inst: Self::Inst) -> usize {
        let offset = buf.current_offset();
        let encoded = inst.encode();
        buf.push_bytes(&encoded);
        offset
    }

    fn patch(code: &mut [u8], reloc_offset: usize, kind: Self::RelocKind, target_offset: usize) {
        match kind {
            X64RelocKind::Rel32 => {
                // displacement = target - (reloc_offset + 4)
                let disp = (target_offset as i64) - ((reloc_offset as i64) + 4);
                let disp32 = disp as i32;
                code[reloc_offset..reloc_offset + 4].copy_from_slice(&disp32.to_le_bytes());
            }
        }
    }
}
