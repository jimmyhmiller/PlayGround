#![allow(clippy::identity_op)]
#![allow(clippy::unusual_byte_groupings)]

pub mod arm64;
pub mod buffer;
pub mod x86_64;

#[cfg(test)]
mod tests;

pub use buffer::{CodeBuffer, ExecutableBuffer, Label, Relocation};

/// Architecture trait — connects instruction type, relocation kind, and
/// the emit/patch logic that is architecture-specific.
pub trait Arch {
    type Inst;
    type RelocKind: Copy + core::fmt::Debug;

    /// Emit a single instruction into `buf`, returning the offset where it was written.
    fn emit(buf: &mut CodeBuffer<Self>, inst: Self::Inst) -> usize
    where
        Self: Sized;

    /// Patch a relocation in the code buffer.
    /// `code` is the full mutable code slice, `reloc_offset` is where the
    /// relocation field lives, `kind` describes how to patch, and
    /// `target_offset` is the resolved label position.
    fn patch(code: &mut [u8], reloc_offset: usize, kind: Self::RelocKind, target_offset: usize);
}
