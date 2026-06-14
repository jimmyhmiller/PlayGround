/// Describes the shape of a heap object so the runtime can compute
/// field offsets, allocation sizes, and GC scan boundaries.
///
/// `header_size` is stored directly (from `ObjHeader::SIZE`) so that
/// all offset methods can be `const fn` without needing generic bounds.
///
/// # Memory layout
///
/// ```text
/// ┌───────────────────┐  offset 0
/// │   header          │  header_size bytes
/// ├───────────────────┤
/// │   field[0] (u64)  │  value_field_count × 8 bytes (GC-traced)
/// │   field[1] (u64)  │
/// │   ...             │
/// ├───────────────────┤
/// │   raw bytes       │  raw_byte_count bytes, padded to 8
/// ├───────────────────┤
/// │   varlen_len (u64)│  only if varlen != None
/// │   varlen[0..n]    │  n elements (Values or bytes)
/// └───────────────────┘
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TypeInfo {
    /// Numeric type identifier. Stored in the object header for fast type dispatch.
    /// Set by the runtime when registering types (e.g., 0=String, 1=Closure, etc.)
    pub type_id: u16,

    /// Size of the object header in bytes (from `ObjHeader::SIZE`).
    pub header_size: u16,

    /// Number of GC-traced Value slots (each 8 bytes).
    pub value_field_count: u16,

    /// Number of untraced raw bytes (after value fields).
    pub raw_byte_count: u16,

    /// Whether this object has a variable-length tail, and what kind.
    pub varlen: VarLenKind,

    /// Log2 of allocation alignment (minimum 3, i.e. 8-byte aligned).
    pub align_log2: u8,
}

/// Whether a heap object has a variable-length tail section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarLenKind {
    /// Fixed-size object — no variable-length tail.
    None,
    /// Variable-length array of GC-traced Values (each 8 bytes).
    Values,
    /// Variable-length byte array (e.g. string contents).
    Bytes,
}

/// Round `n` up to the next multiple of 8.
const fn align8(n: usize) -> usize {
    (n + 7) & !7
}

impl TypeInfo {
    /// Start building a TypeInfo for objects using a header of the given size.
    ///
    /// ```rust,ignore
    /// use ai_lang::gc::{TypeInfo, Full, Compact, ObjHeader};
    ///
    /// // A cons cell: 2 value fields, Full header
    /// const CONS: TypeInfo = TypeInfo::for_header(Full::SIZE).with_fields(2);
    ///
    /// // A string: no fixed fields, variable-length bytes, Compact header
    /// const STR: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_varlen_bytes(0);
    /// ```
    pub const fn for_header(header_size: usize) -> Self {
        Self {
            type_id: 0, // set by runtime when registering types
            header_size: header_size as u16,
            value_field_count: 0,
            raw_byte_count: 0,
            varlen: VarLenKind::None,
            align_log2: 3, // 8-byte alignment minimum
        }
    }

    /// Set the type_id for this TypeInfo.
    pub const fn with_type_id(mut self, id: u16) -> Self {
        self.type_id = id;
        self
    }

    /// Set the number of GC-traced Value fields (each 8 bytes).
    pub const fn with_fields(mut self, count: u16) -> Self {
        self.value_field_count = count;
        self
    }

    /// Set the number of untraced raw bytes.
    pub const fn with_raw_bytes(mut self, count: u16) -> Self {
        self.raw_byte_count = count;
        self
    }

    /// Add a variable-length Values tail (GC-traced).
    /// `fixed_fields` sets the number of fixed Value fields before the varlen section.
    pub const fn with_varlen_values(mut self, fixed_fields: u16) -> Self {
        self.value_field_count = fixed_fields;
        self.varlen = VarLenKind::Values;
        self
    }

    /// Add a variable-length byte tail.
    /// `fixed_fields` sets the number of fixed Value fields before the varlen section.
    pub const fn with_varlen_bytes(mut self, fixed_fields: u16) -> Self {
        self.value_field_count = fixed_fields;
        self.varlen = VarLenKind::Bytes;
        self
    }

    /// Set the alignment (as log2). Minimum is 3 (8-byte aligned).
    pub const fn with_align_log2(mut self, log2: u8) -> Self {
        assert!(log2 >= 3, "alignment must be at least 8 bytes (log2 >= 3)");
        self.align_log2 = log2;
        self
    }

    /// Byte offset of value field `index` from the start of the object.
    pub const fn value_field_offset(&self, index: u16) -> usize {
        self.header_size as usize + (index as usize) * 8
    }

    /// Byte offset of the raw data section from the start of the object.
    pub const fn raw_data_offset(&self) -> usize {
        self.header_size as usize + (self.value_field_count as usize) * 8
    }

    /// Byte offset of the varlen count word from the start of the object.
    /// Only meaningful when `varlen != VarLenKind::None`.
    pub const fn varlen_count_offset(&self) -> usize {
        align8(self.raw_data_offset() + self.raw_byte_count as usize)
    }

    /// Byte offset of varlen element `index` from the start of the object.
    /// For `Values`, each element is 8 bytes. For `Bytes`, each is 1 byte.
    pub const fn varlen_element_offset(&self, index: usize) -> usize {
        let base = self.varlen_count_offset() + 8; // skip the count word
        match self.varlen {
            VarLenKind::None => panic!("no varlen section"),
            VarLenKind::Values => base + index * 8,
            VarLenKind::Bytes => base + index,
        }
    }

    /// Total allocation size in bytes for an object with `varlen_len` variable-length elements.
    /// Result is aligned to the object's alignment requirement.
    pub const fn allocation_size(&self, varlen_len: usize) -> usize {
        let size = match self.varlen {
            VarLenKind::None => align8(self.raw_data_offset() + self.raw_byte_count as usize),
            VarLenKind::Values => {
                // count word + n×8 bytes
                self.varlen_count_offset() + 8 + varlen_len * 8
            }
            VarLenKind::Bytes => {
                // count word + n bytes, aligned up
                align8(self.varlen_count_offset() + 8 + varlen_len)
            }
        };
        let align = 1usize << self.align_log2;
        (size + align - 1) & !(align - 1)
    }
}
