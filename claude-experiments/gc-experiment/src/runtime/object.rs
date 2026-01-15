use std::ffi::c_void;

/// Type tag for objects - identifies what kind of object this is
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectType {
    /// A user-defined struct
    Struct = 0,
    /// An array of GC pointers
    Array = 1,
}

/// Metadata about an object's type - how to scan it for GC roots.
/// Generated as constants by the compiler for each struct/array type.
#[repr(C)]
pub struct ObjectMeta {
    /// The type tag
    pub object_type: ObjectType,
    /// For structs: number of pointer fields that need scanning
    /// For arrays: unused (length is in the object itself)
    pub num_pointer_fields: u32,
    /// For structs: offsets of pointer fields (in bytes from start of payload)
    /// Points to an array of u32 offsets
    pub pointer_field_offsets: *const u32,
    /// Type name for debugging
    pub type_name: *const u8,
}

impl ObjectMeta {
    pub fn type_name_str(&self) -> &str {
        if self.type_name.is_null() {
            "<unknown>"
        } else {
            unsafe {
                let mut len = 0;
                while *self.type_name.add(len) != 0 {
                    len += 1;
                }
                std::str::from_utf8_unchecked(std::slice::from_raw_parts(self.type_name, len))
            }
        }
    }
}

/// Header for all GC-managed objects.
/// Every object starts with this header, followed by its payload.
#[repr(C)]
pub struct ObjectHeader {
    /// Pointer to type metadata (how to scan this object)
    pub meta: *const ObjectMeta,
    /// GC mark bit and other flags
    pub gc_flags: u32,
    /// For arrays: the length. For structs: unused (could store size or other data)
    pub aux: u32,
}

impl ObjectHeader {
    pub const MARK_BIT: u32 = 1;

    pub fn is_marked(&self) -> bool {
        self.gc_flags & Self::MARK_BIT != 0
    }

    pub fn set_marked(&mut self, marked: bool) {
        if marked {
            self.gc_flags |= Self::MARK_BIT;
        } else {
            self.gc_flags &= !Self::MARK_BIT;
        }
    }

    pub fn object_type(&self) -> ObjectType {
        if self.meta.is_null() {
            ObjectType::Struct // default
        } else {
            unsafe { (*self.meta).object_type }
        }
    }

    /// Get the payload pointer (immediately after the header)
    pub fn payload(&self) -> *mut c_void {
        unsafe {
            let header_end = (self as *const ObjectHeader).add(1);
            header_end as *mut c_void
        }
    }

    /// Iterate over all pointer fields in this object (for GC tracing)
    pub fn pointer_fields(&self) -> PointerFieldIter {
        if self.meta.is_null() {
            return PointerFieldIter {
                payload: std::ptr::null(),
                offsets: std::ptr::null(),
                remaining: 0,
            };
        }

        let meta = unsafe { &*self.meta };
        match meta.object_type {
            ObjectType::Struct => PointerFieldIter {
                payload: self.payload() as *const u8,
                offsets: meta.pointer_field_offsets,
                remaining: meta.num_pointer_fields as usize,
            },
            ObjectType::Array => {
                // For arrays, aux is the length, and each element is a pointer
                PointerFieldIter {
                    payload: self.payload() as *const u8,
                    offsets: std::ptr::null(), // special case: sequential
                    remaining: self.aux as usize,
                }
            }
        }
    }
}

/// Iterator over pointer fields in an object
pub struct PointerFieldIter {
    payload: *const u8,
    offsets: *const u32, // null for arrays (sequential pointers)
    remaining: usize,
}

impl Iterator for PointerFieldIter {
    type Item = *mut *mut c_void;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let ptr = if self.offsets.is_null() {
            // Array case: sequential pointers
            let index = self.remaining;
            self.remaining -= 1;
            // Return in reverse order so we count down
            let actual_index = unsafe {
                let total = self.remaining + 1;
                total - index
            };
            unsafe {
                self.payload.add(actual_index * std::mem::size_of::<*mut c_void>())
                    as *mut *mut c_void
            }
        } else {
            // Struct case: use offsets
            let offset = unsafe { *self.offsets } as usize;
            self.offsets = unsafe { self.offsets.add(1) };
            self.remaining -= 1;
            unsafe { self.payload.add(offset) as *mut *mut c_void }
        };

        Some(ptr)
    }
}

/// Convenience type alias - a GC object is a pointer to its header
pub type Object = *mut ObjectHeader;

/// Object header offsets for codegen
pub mod offsets {
    pub const META: u64 = 0;
    pub const GC_FLAGS: u64 = 8;
    pub const AUX: u64 = 12;
    pub const PAYLOAD: u64 = 16;
}
