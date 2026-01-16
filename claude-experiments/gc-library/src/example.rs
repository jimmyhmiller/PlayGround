//! Example/reference implementation of the GC traits.
//!
//! This module provides a simple implementation of all GC traits for testing
//! and as a reference for implementors. It uses:
//!
//! - 3-bit tags in the low bits of pointers
//! - 8-byte headers with mark bit, opaque bit, and size field
//! - Simple type discriminants (Int, HeapObject, Null)
//!
//! # Usage
//!
//! ```rust,ignore
//! use gc_library::example::{ExampleRuntime, ExampleTaggedPtr, ExampleObject};
//! use gc_library::gc::MarkAndSweep;
//!
//! // Create a mark-and-sweep GC for the example runtime
//! let gc: MarkAndSweep<ExampleRuntime> = MarkAndSweep::new(options);
//! ```

use crate::traits::{
    ForwardingSupport, GcObject, GcTypes, HeaderOps, ObjectKind, RootProvider, TaggedPointer,
};

// =============================================================================
// Type Tags
// =============================================================================

/// Example type tags using 3 bits.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum ExampleTypeTag {
    /// Immediate integer (tagged, not heap allocated)
    Int = 0,
    /// Heap-allocated object
    HeapObject = 1,
    /// Null/nil value
    Null = 7,
}

impl ExampleTypeTag {
    fn from_bits(bits: usize) -> Self {
        match bits & 0b111 {
            0 => ExampleTypeTag::Int,
            1 => ExampleTypeTag::HeapObject,
            7 => ExampleTypeTag::Null,
            _ => ExampleTypeTag::Null, // Default unknown tags to Null
        }
    }
}

impl ObjectKind for ExampleTypeTag {
    fn is_heap_type(self) -> bool {
        matches!(self, ExampleTypeTag::HeapObject)
    }
}

// =============================================================================
// Tagged Pointer
// =============================================================================

/// Example tagged pointer using 3-bit tags.
///
/// Layout: `[pointer (61 bits)][tag (3 bits)]`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ExampleTaggedPtr(usize);

impl ExampleTaggedPtr {
    /// Create a tagged integer.
    pub fn from_int(value: i64) -> Self {
        // Shift left by 3 to make room for tag, then add Int tag
        ExampleTaggedPtr(((value as usize) << 3) | ExampleTypeTag::Int as usize)
    }

    /// Create a null value.
    pub fn null() -> Self {
        ExampleTaggedPtr(ExampleTypeTag::Null as usize)
    }

    /// Extract integer value (if this is an Int).
    pub fn as_int(self) -> Option<i64> {
        if self.get_kind() == ExampleTypeTag::Int {
            Some((self.0 as i64) >> 3)
        } else {
            None
        }
    }
}

impl TaggedPointer for ExampleTaggedPtr {
    type Kind = ExampleTypeTag;

    fn tag(raw_ptr: *const u8, kind: ExampleTypeTag) -> Self {
        debug_assert!(
            (raw_ptr as usize) & 0b111 == 0,
            "pointer must be 8-byte aligned"
        );
        ExampleTaggedPtr((raw_ptr as usize) | kind as usize)
    }

    fn untag(self) -> *const u8 {
        (self.0 & !0b111) as *const u8
    }

    fn get_kind(self) -> ExampleTypeTag {
        ExampleTypeTag::from_bits(self.0)
    }

    fn is_heap_pointer(self) -> bool {
        self.get_kind().is_heap_type()
    }

    fn as_usize(self) -> usize {
        self.0
    }

    fn from_usize(value: usize) -> Self {
        ExampleTaggedPtr(value)
    }
}

// =============================================================================
// Object Header
// =============================================================================

/// Example header format (8 bytes).
///
/// Layout:
/// ```text
/// Bits 0:     Marked bit
/// Bits 1:     Opaque bit (no pointer fields)
/// Bits 2:     Forwarding bit
/// Bits 3-15:  Reserved
/// Bits 16-31: Size in words (u16)
/// Bits 32-63: Type data (user-defined)
/// ```
#[derive(Debug, Copy, Clone)]
pub struct ExampleHeader {
    pub marked: bool,
    pub opaque: bool,
    pub forwarding: bool,
    pub size_words: u16,
    pub type_data: u32,
}

impl ExampleHeader {
    const MARKED_BIT: usize = 1 << 0;
    const OPAQUE_BIT: usize = 1 << 1;
    const FORWARDING_BIT: usize = 1 << 2;
    const SIZE_SHIFT: usize = 16;
    const SIZE_MASK: usize = 0xFFFF;
    const TYPE_DATA_SHIFT: usize = 32;

    pub fn from_usize(value: usize) -> Self {
        ExampleHeader {
            marked: (value & Self::MARKED_BIT) != 0,
            opaque: (value & Self::OPAQUE_BIT) != 0,
            forwarding: (value & Self::FORWARDING_BIT) != 0,
            size_words: ((value >> Self::SIZE_SHIFT) & Self::SIZE_MASK) as u16,
            type_data: (value >> Self::TYPE_DATA_SHIFT) as u32,
        }
    }

    pub fn to_usize(self) -> usize {
        let mut value = 0usize;
        if self.marked {
            value |= Self::MARKED_BIT;
        }
        if self.opaque {
            value |= Self::OPAQUE_BIT;
        }
        if self.forwarding {
            value |= Self::FORWARDING_BIT;
        }
        value |= (self.size_words as usize) << Self::SIZE_SHIFT;
        value |= (self.type_data as usize) << Self::TYPE_DATA_SHIFT;
        value
    }

    pub fn new(size_words: u16, opaque: bool) -> Self {
        ExampleHeader {
            marked: false,
            opaque,
            forwarding: false,
            size_words,
            type_data: 0,
        }
    }
}

impl HeaderOps for ExampleHeader {
    const MAX_INLINE_SIZE: usize = 0xFFFF;

    fn set_marked_bit(header: usize) -> usize {
        header | Self::MARKED_BIT
    }

    fn clear_marked_bit(header: usize) -> usize {
        header & !Self::MARKED_BIT
    }

    fn is_marked_bit_set(header: usize) -> bool {
        (header & Self::MARKED_BIT) != 0
    }

    fn set_forwarding_bit(value: usize) -> usize {
        value | Self::FORWARDING_BIT
    }

    fn clear_forwarding_bit(value: usize) -> usize {
        value & !Self::FORWARDING_BIT
    }

    fn is_forwarding_bit_set(value: usize) -> bool {
        (value & Self::FORWARDING_BIT) != 0
    }

    fn is_large_object(header: usize) -> bool {
        let size = (header >> Self::SIZE_SHIFT) & Self::SIZE_MASK;
        size == Self::MAX_INLINE_SIZE
    }
}

// =============================================================================
// Heap Object
// =============================================================================

/// Example heap object handle.
pub struct ExampleObject {
    ptr: *const u8,
}

impl ExampleObject {
    fn header_ptr(&self) -> *mut usize {
        self.ptr as *mut usize
    }

    fn read_header(&self) -> ExampleHeader {
        let raw = unsafe { *self.header_ptr() };
        ExampleHeader::from_usize(raw)
    }

    fn write_header_raw(&self, header: ExampleHeader) {
        unsafe {
            *self.header_ptr() = header.to_usize();
        }
    }

    fn fields_ptr(&self) -> *mut usize {
        unsafe { self.header_ptr().add(1) }
    }
}

impl GcObject for ExampleObject {
    type TaggedValue = ExampleTaggedPtr;

    fn from_tagged(tagged: ExampleTaggedPtr) -> Self {
        debug_assert!(
            tagged.is_heap_pointer(),
            "expected heap pointer, got {:?}",
            tagged.get_kind()
        );
        ExampleObject {
            ptr: tagged.untag(),
        }
    }

    fn from_untagged(ptr: *const u8) -> Self {
        ExampleObject { ptr }
    }

    fn get_pointer(&self) -> *const u8 {
        self.ptr
    }

    fn tagged_pointer(&self) -> ExampleTaggedPtr {
        ExampleTaggedPtr::tag(self.ptr, ExampleTypeTag::HeapObject)
    }

    fn mark(&self) {
        let header = self.read_header();
        let mut new_header = header;
        new_header.marked = true;
        self.write_header_raw(new_header);
    }

    fn unmark(&self) {
        let header = self.read_header();
        let mut new_header = header;
        new_header.marked = false;
        self.write_header_raw(new_header);
    }

    fn marked(&self) -> bool {
        self.read_header().marked
    }

    fn get_fields(&self) -> &[usize] {
        let header = self.read_header();
        if header.opaque {
            return &[];
        }
        let count = header.size_words as usize;
        unsafe { std::slice::from_raw_parts(self.fields_ptr(), count) }
    }

    fn get_fields_mut(&mut self) -> &mut [usize] {
        let header = self.read_header();
        if header.opaque {
            return &mut [];
        }
        let count = header.size_words as usize;
        unsafe { std::slice::from_raw_parts_mut(self.fields_ptr(), count) }
    }

    fn is_opaque(&self) -> bool {
        self.read_header().opaque
    }

    fn is_zero_size(&self) -> bool {
        self.read_header().size_words == 0
    }

    fn get_object_kind(&self) -> Option<ExampleTypeTag> {
        Some(ExampleTypeTag::HeapObject)
    }

    fn full_size(&self) -> usize {
        self.header_size() + (self.read_header().size_words as usize * 8)
    }

    fn header_size(&self) -> usize {
        8 // Always 8 bytes for this simple implementation
    }

    fn get_full_object_data(&self) -> &[u8] {
        let size = self.full_size();
        unsafe { std::slice::from_raw_parts(self.ptr, size) }
    }

    fn write_header(&mut self, field_size_bytes: usize) {
        let size_words = field_size_bytes / 8;
        debug_assert!(size_words <= 0xFFFF, "object too large for inline size");
        let header = ExampleHeader::new(size_words as u16, false);
        self.write_header_raw(header);
    }
}

impl ForwardingSupport for ExampleObject {
    fn is_forwarded(&self) -> bool {
        let raw = unsafe { *self.header_ptr() };
        ExampleHeader::is_forwarding_bit_set(raw)
    }

    fn get_forwarding_pointer(&self) -> ExampleTaggedPtr {
        debug_assert!(self.is_forwarded(), "object is not forwarded");
        let raw = unsafe { *self.header_ptr() };
        // Clear the forwarding bit to get the clean tagged pointer
        ExampleTaggedPtr::from_usize(ExampleHeader::clear_forwarding_bit(raw))
    }

    fn set_forwarding_pointer(&mut self, new_location: ExampleTaggedPtr) {
        let forwarding_value = ExampleHeader::set_forwarding_bit(new_location.as_usize());
        unsafe {
            *self.header_ptr() = forwarding_value;
        }
    }
}

// =============================================================================
// Runtime Type Bundle
// =============================================================================

/// Example runtime type bundle.
///
/// Use this as the type parameter for GC implementations:
/// ```rust,ignore
/// let gc: MarkAndSweep<ExampleRuntime> = MarkAndSweep::new(options);
/// ```
pub struct ExampleRuntime;

impl GcTypes for ExampleRuntime {
    type TaggedValue = ExampleTaggedPtr;
    type ObjectHandle = ExampleObject;
    type ObjectKind = ExampleTypeTag;
}

// =============================================================================
// Simple Root Provider
// =============================================================================

/// A simple root provider that holds a list of root slot addresses.
pub struct SimpleRoots {
    /// List of addresses where roots are stored
    slots: Vec<*mut usize>,
}

impl SimpleRoots {
    pub fn new() -> Self {
        SimpleRoots { slots: Vec::new() }
    }

    /// Add a root slot to be scanned.
    ///
    /// # Safety
    /// The slot must remain valid for the lifetime of this RootProvider.
    pub unsafe fn add_slot(&mut self, slot: *mut usize) {
        self.slots.push(slot);
    }

    /// Clear all registered slots.
    pub fn clear(&mut self) {
        self.slots.clear();
    }
}

impl Default for SimpleRoots {
    fn default() -> Self {
        Self::new()
    }
}

impl RootProvider<ExampleTaggedPtr> for SimpleRoots {
    fn enumerate_roots(&self, callback: &mut dyn FnMut(usize, ExampleTaggedPtr)) {
        for &slot_ptr in &self.slots {
            let value = unsafe { *slot_ptr };
            let tagged = ExampleTaggedPtr::from_usize(value);
            if tagged.is_heap_pointer() {
                callback(slot_ptr as usize, tagged);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tagged_int() {
        let tagged = ExampleTaggedPtr::from_int(42);
        assert_eq!(tagged.get_kind(), ExampleTypeTag::Int);
        assert!(!tagged.is_heap_pointer());
        assert_eq!(tagged.as_int(), Some(42));
    }

    #[test]
    fn test_tagged_null() {
        let tagged = ExampleTaggedPtr::null();
        assert_eq!(tagged.get_kind(), ExampleTypeTag::Null);
        assert!(!tagged.is_heap_pointer());
    }

    #[test]
    fn test_header_roundtrip() {
        let header = ExampleHeader {
            marked: true,
            opaque: false,
            forwarding: false,
            size_words: 42,
            type_data: 0x12345678,
        };
        let raw = header.to_usize();
        let restored = ExampleHeader::from_usize(raw);
        assert_eq!(header.marked, restored.marked);
        assert_eq!(header.opaque, restored.opaque);
        assert_eq!(header.size_words, restored.size_words);
        assert_eq!(header.type_data, restored.type_data);
    }

    #[test]
    fn test_header_ops() {
        let header = ExampleHeader::new(10, false).to_usize();
        assert!(!ExampleHeader::is_marked_bit_set(header));

        let marked = ExampleHeader::set_marked_bit(header);
        assert!(ExampleHeader::is_marked_bit_set(marked));

        let unmarked = ExampleHeader::clear_marked_bit(marked);
        assert!(!ExampleHeader::is_marked_bit_set(unmarked));
    }
}
