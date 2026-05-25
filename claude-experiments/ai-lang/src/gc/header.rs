use crate::gc::type_info::TypeInfo;

/// Defines the header prepended to every heap object.
///
/// Choose compact (8 bytes), full (16 bytes), or define your own.
/// The header lives at the start of every heap allocation and lets
/// the runtime (GC, debugger, etc.) identify the object's shape.
///
/// # Safety
///
/// Implementations must be `#[repr(C)]` and have a size equal to `Self::SIZE`.
pub trait ObjHeader: Copy + 'static {
    /// Size of this header in bytes.
    const SIZE: usize;

    /// Byte offset of the `type_id` field within this header.
    /// Used by the heap walker to recover the type from any object.
    const TYPE_ID_OFFSET: usize;

    /// Initialize a header for a newly allocated object.
    fn new(type_id: u16) -> Self;

    /// Get the type ID (index into the runtime's TypeInfo table).
    fn type_id(&self) -> u16;
}

// Compact header: type_id + padding (8 bytes total).
// The remaining 6 bytes after type_id are available for future use
// (e.g., GC mark bits, hash code, etc.)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Compact {
    type_id: u16,
    _pad: u16,
    _pad2: u32,
}

impl ObjHeader for Compact {
    const SIZE: usize = core::mem::size_of::<Compact>();
    const TYPE_ID_OFFSET: usize = core::mem::offset_of!(Compact, type_id);

    #[inline(always)]
    fn new(type_id: u16) -> Self {
        Compact {
            type_id,
            _pad: 0,
            _pad2: 0,
        }
    }

    #[inline(always)]
    fn type_id(&self) -> u16 {
        self.type_id
    }
}

// Full header: GC word + type_id + padding (16 bytes total).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Full {
    gc_word: u64,
    type_id: u16,
    _pad: u16,
    _pad2: u32,
}

impl ObjHeader for Full {
    const SIZE: usize = core::mem::size_of::<Full>();
    const TYPE_ID_OFFSET: usize = core::mem::offset_of!(Full, type_id);

    #[inline(always)]
    fn new(type_id: u16) -> Self {
        Full {
            gc_word: 0,
            type_id,
            _pad: 0,
            _pad2: 0,
        }
    }

    #[inline(always)]
    fn type_id(&self) -> u16 {
        self.type_id
    }
}

impl Full {
    #[inline(always)]
    pub fn gc_word(&self) -> u64 {
        self.gc_word
    }

    #[inline(always)]
    pub fn set_gc_word(&mut self, val: u64) {
        self.gc_word = val;
    }
}

// `TypeInfo` is referenced here so the type_info module is reachable.
#[allow(dead_code)]
fn _link_type_info(_t: &TypeInfo) {}
