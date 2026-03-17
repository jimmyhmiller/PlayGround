#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Type {
    I8,
    I32,
    I64,
    F64,
    Ptr,
    /// GC-managed pointer. Same size/ABI as Ptr, but the lowering must
    /// emit stack maps at safepoints so the collector can trace and update it.
    GcPtr,
    /// Opaque handle to a captured delimited frame slice.
    /// This is GC-managed runtime state, but it is not a raw pointer type.
    FrameSlice,
}

impl Type {
    pub fn is_int(self) -> bool {
        matches!(self, Type::I8 | Type::I32 | Type::I64)
    }

    pub fn is_float(self) -> bool {
        matches!(self, Type::F64)
    }

    /// True for any pointer type (raw or GC-managed).
    pub fn is_ptr(self) -> bool {
        matches!(self, Type::Ptr | Type::GcPtr)
    }

    /// True if this type is GC-traced and must appear in stack maps at safepoints.
    pub fn is_gc(self) -> bool {
        matches!(self, Type::GcPtr | Type::FrameSlice)
    }

    pub fn size_bytes(self) -> u32 {
        match self {
            Type::I8 => 1,
            Type::I32 => 4,
            Type::I64 | Type::F64 | Type::Ptr | Type::GcPtr | Type::FrameSlice => 8,
        }
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::I8 => write!(f, "i8"),
            Type::I32 => write!(f, "i32"),
            Type::I64 => write!(f, "i64"),
            Type::F64 => write!(f, "f64"),
            Type::Ptr => write!(f, "ptr"),
            Type::GcPtr => write!(f, "gcptr"),
            Type::FrameSlice => write!(f, "frameslice"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signature {
    pub params: Vec<Type>,
    pub ret: Option<Type>,
}
