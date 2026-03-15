use crate::Arch;

/// An opaque label handle. Created via `CodeBuffer::create_label()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Label(pub(crate) usize);

/// A pending relocation that needs to be patched when labels are resolved.
#[derive(Debug)]
pub struct Relocation<A: Arch> {
    /// Offset in the code buffer where the relocation field starts.
    pub offset: usize,
    /// Which label this relocation refers to.
    pub label: Label,
    /// Architecture-specific relocation kind.
    pub kind: A::RelocKind,
}

/// Accumulates machine code bytes, manages labels, and resolves relocations.
pub struct CodeBuffer<A: Arch> {
    code: Vec<u8>,
    labels: Vec<Option<usize>>,
    relocs: Vec<Relocation<A>>,
}

impl<A: Arch> CodeBuffer<A> {
    pub fn new() -> Self {
        Self {
            code: Vec::new(),
            labels: Vec::new(),
            relocs: Vec::new(),
        }
    }

    /// Create a new unbound label.
    pub fn create_label(&mut self) -> Label {
        let id = self.labels.len();
        self.labels.push(None);
        Label(id)
    }

    /// Bind a label to the current code offset.
    pub fn bind_label(&mut self, label: Label) {
        assert!(self.labels[label.0].is_none(), "label already bound");
        self.labels[label.0] = Some(self.code.len());
    }

    /// Emit an instruction. Returns the offset where it was placed.
    pub fn emit(&mut self, inst: A::Inst) -> usize {
        A::emit(self, inst)
    }

    /// Push raw bytes into the code buffer. Returns the starting offset.
    pub fn push_bytes(&mut self, bytes: &[u8]) -> usize {
        let offset = self.code.len();
        self.code.extend_from_slice(bytes);
        offset
    }

    /// Record a relocation to be patched during `finalize()`.
    pub fn add_reloc(&mut self, offset: usize, label: Label, kind: A::RelocKind) {
        self.relocs.push(Relocation {
            offset,
            label,
            kind,
        });
    }

    /// Current length of emitted code.
    pub fn current_offset(&self) -> usize {
        self.code.len()
    }

    /// Resolve all relocations. Panics if any label is unbound.
    pub fn finalize(&mut self) {
        for reloc in &self.relocs {
            let target = self.labels[reloc.label.0].expect("unbound label during finalize");
            A::patch(&mut self.code, reloc.offset, reloc.kind, target);
        }
        self.relocs.clear();
    }

    /// Borrow the emitted code bytes.
    pub fn code(&self) -> &[u8] {
        &self.code
    }

    /// Patch bytes at the given offset in the code buffer.
    /// Used for fixing up instruction fields (e.g., frame sizes) after codegen.
    pub fn patch_bytes(&mut self, offset: usize, bytes: &[u8]) {
        self.code[offset..offset + bytes.len()].copy_from_slice(bytes);
    }

    /// Consume the buffer and return the code bytes.
    pub fn into_code(mut self) -> Vec<u8> {
        self.finalize();
        self.code
    }
}

impl<A: Arch> Default for CodeBuffer<A> {
    fn default() -> Self {
        Self::new()
    }
}
