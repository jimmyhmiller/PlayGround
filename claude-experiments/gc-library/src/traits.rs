//! Core traits for the generic GC library.
//!
//! This module defines the trait bounds required for a language runtime
//! to use the GC library. The design follows these principles:
//!
//! 1. **Minimal Surface Area**: Only methods actually used by GC are required
//! 2. **Ergonomic**: A single `GcTypes` trait bundles associated types
//! 3. **Composable**: Separate traits for tagging, objects, and roots
//! 4. **Safe Defaults**: Where possible, provide default implementations
//!
//! # Implementor Invariants
//!
//! Users implementing these traits MUST ensure:
//!
//! 1. **Pointer Alignment**: All untagged pointers are 8-byte aligned
//! 2. **Tag Consistency**: `get_kind(tag(ptr, kind))` returns `kind`
//! 3. **Heap Pointer Completeness**: `is_heap_pointer()` returns true for ALL GC-managed values
//! 4. **Field Bounds**: `get_fields()` returns correctly-sized slice
//! 5. **Size Accuracy**: `full_size()` returns exact allocation size
//! 6. **Forwarding Safety**: Forwarding bit must not conflict with valid tags
//! 7. **Root Completeness**: `RootProvider` must enumerate ALL roots

use core::fmt::Debug;

// =============================================================================
// Core Type System Trait
// =============================================================================

/// The main trait that users implement to integrate their runtime with the GC.
///
/// This bundles all associated types together for ergonomic generic bounds.
///
/// # Example
/// ```rust,ignore
/// struct MyRuntime;
///
/// impl GcTypes for MyRuntime {
///     type TaggedValue = MyTaggedPointer;
///     type ObjectHandle = MyHeapObject;
///     type ObjectKind = MyTypeTag;
/// }
/// ```
pub trait GcTypes: Sized + 'static {
    /// The representation of a tagged pointer value (typically wraps `usize`)
    type TaggedValue: TaggedPointer<Kind = Self::ObjectKind>;

    /// A handle to a heap object for field access and marking
    type ObjectHandle: GcObject<TaggedValue = Self::TaggedValue>;

    /// The type discriminant stored in the tag bits
    type ObjectKind: ObjectKind;
}

// =============================================================================
// Tagged Pointer Operations
// =============================================================================

/// Operations for tagged pointer values.
///
/// Tagged pointers encode type information in the low bits of a pointer.
/// The GC needs to:
/// - Check if a value is a heap pointer that needs tracing
/// - Extract the raw pointer for heap access
/// - Re-tag pointers after relocation
///
/// # Invariants
/// - `untag(tag(ptr, kind))` must equal `ptr`
/// - `get_kind(tag(ptr, kind))` must equal `kind`
/// - Tagged values must be 8-byte aligned after untagging
///
/// # Example Implementation
/// ```rust,ignore
/// #[derive(Copy, Clone, PartialEq, Eq, Debug)]
/// struct TaggedPtr(usize);
///
/// impl TaggedPointer for TaggedPtr {
///     type Kind = TypeTag;
///
///     fn tag(raw_ptr: *const u8, kind: TypeTag) -> Self {
///         TaggedPtr((raw_ptr as usize) << 3 | kind as usize)
///     }
///
///     fn untag(self) -> *const u8 {
///         (self.0 >> 3) as *const u8
///     }
///
///     fn get_kind(self) -> TypeTag {
///         TypeTag::from_bits(self.0 & 0b111)
///     }
///
///     fn is_heap_pointer(self) -> bool {
///         matches!(self.get_kind(), TypeTag::HeapObject | TypeTag::Closure)
///     }
///
///     fn as_usize(self) -> usize { self.0 }
///     fn from_usize(value: usize) -> Self { TaggedPtr(value) }
/// }
/// ```
pub trait TaggedPointer: Copy + Clone + Eq + Debug + 'static {
    /// The type kind/discriminant
    type Kind: ObjectKind;

    /// Create a tagged pointer from a raw pointer and type kind.
    ///
    /// The raw pointer must be 8-byte aligned.
    fn tag(raw_ptr: *const u8, kind: Self::Kind) -> Self;

    /// Extract the raw pointer by removing the tag bits.
    ///
    /// The returned pointer will be 8-byte aligned.
    fn untag(self) -> *const u8;

    /// Extract the type kind from the tag bits.
    fn get_kind(self) -> Self::Kind;

    /// Check if this value represents a heap pointer that needs GC tracing.
    ///
    /// This is the critical method for determining what to trace.
    /// Typically returns true for: closures, heap objects, boxed floats
    /// Typically returns false for: immediate integers, booleans, null
    ///
    /// # Safety
    /// Missing any heap pointers will cause use-after-free bugs.
    fn is_heap_pointer(self) -> bool;

    /// Get the raw usize representation (for storage and manipulation).
    fn as_usize(self) -> usize;

    /// Create from raw usize.
    ///
    /// # Safety
    /// Caller must ensure the value represents a valid tagged pointer.
    fn from_usize(value: usize) -> Self;
}

/// Type kind/discriminant for tagged values.
///
/// This represents the different types that can be stored in tagged pointers.
pub trait ObjectKind: Copy + Clone + Eq + Debug + 'static {
    /// Check if this kind represents a heap-allocated value.
    ///
    /// Used as a fast path - if false, the GC can skip tracing this value entirely.
    fn is_heap_type(self) -> bool;
}

// =============================================================================
// Heap Object Operations
// =============================================================================

/// Handle for accessing and manipulating heap objects.
///
/// The GC needs to:
/// - Mark objects as visited
/// - Read/write object fields (for tracing and relocation)
/// - Copy object data (for compacting/generational GC)
/// - Query object size
///
/// # Safety
/// Implementations must ensure that field access is bounds-checked
/// and that marking operations are safe for concurrent access patterns.
///
/// # Example Implementation
/// ```rust,ignore
/// struct HeapObject {
///     ptr: *const u8,
/// }
///
/// impl GcObject for HeapObject {
///     type TaggedValue = TaggedPtr;
///
///     fn from_tagged(tagged: TaggedPtr) -> Self {
///         HeapObject { ptr: tagged.untag() }
///     }
///
///     fn from_untagged(ptr: *const u8) -> Self {
///         HeapObject { ptr }
///     }
///
///     fn get_pointer(&self) -> *const u8 { self.ptr }
///
///     // ... other methods
/// }
/// ```
pub trait GcObject: Sized {
    /// The tagged value type this object works with
    type TaggedValue: TaggedPointer;

    // --- Construction ---

    /// Create an object handle from a tagged pointer.
    ///
    /// # Panics
    /// May panic if the tagged value is not a valid heap pointer.
    fn from_tagged(tagged: Self::TaggedValue) -> Self;

    /// Create an object handle from an untagged raw pointer.
    ///
    /// Used when allocating new objects or iterating the heap.
    fn from_untagged(ptr: *const u8) -> Self;

    // --- Pointer Access ---

    /// Get the untagged raw pointer to this object's header.
    fn get_pointer(&self) -> *const u8;

    /// Get the tagged pointer representation of this object.
    fn tagged_pointer(&self) -> Self::TaggedValue;

    /// Get the untagged address as usize.
    fn untagged(&self) -> usize {
        self.get_pointer() as usize
    }

    // --- GC Marking ---

    /// Mark this object as visited during GC.
    ///
    /// For concurrent collectors, this may need to use atomic operations.
    fn mark(&self);

    /// Clear the mark from this object.
    fn unmark(&self);

    /// Check if this object is marked.
    fn marked(&self) -> bool;

    // --- Field Access ---

    /// Get an immutable slice of all pointer-sized fields.
    ///
    /// The slice should contain the raw field values (which may be tagged pointers).
    fn get_fields(&self) -> &[usize];

    /// Get a mutable slice of all fields (for updating pointers during compaction).
    fn get_fields_mut(&mut self) -> &mut [usize];

    // --- Object Classification ---

    /// Check if this object is opaque (no traceable fields).
    ///
    /// Opaque objects contain raw data (strings, byte arrays) that
    /// should not be scanned for heap pointers.
    fn is_opaque(&self) -> bool;

    /// Check if this is a zero-size object (header only).
    fn is_zero_size(&self) -> bool;

    /// Get the type kind of this object (if available from header).
    fn get_object_kind(&self) -> Option<<Self::TaggedValue as TaggedPointer>::Kind>;

    // --- Size Information ---

    /// Total size of the object including header, in bytes.
    ///
    /// This must be accurate for heap iteration to work correctly.
    fn full_size(&self) -> usize;

    /// Size of just the header, in bytes (typically 8 or 16).
    fn header_size(&self) -> usize;

    // --- Copying (for compacting/generational GC) ---

    /// Get the complete object data (header + fields) as a byte slice.
    ///
    /// Used by copying collectors to move objects between spaces.
    fn get_full_object_data(&self) -> &[u8];

    /// Write a header for a new object with the given field size in bytes.
    fn write_header(&mut self, field_size_bytes: usize);
}

/// Extension trait for objects that support forwarding pointers (compacting GC).
///
/// Forwarding pointers are used during compaction to redirect references
/// from old object locations to new locations.
///
/// # Implementation Notes
/// - The forwarding bit must not conflict with valid type tags
/// - Typically uses a reserved bit in the header (e.g., bit 3 if tags use bits 0-2)
/// - The forwarding pointer replaces the header temporarily during GC
pub trait ForwardingSupport: GcObject {
    /// Check if this object has been forwarded (header contains forwarding pointer).
    fn is_forwarded(&self) -> bool;

    /// Get the forwarding pointer (new location after compaction).
    ///
    /// # Panics
    /// May panic if `is_forwarded()` returns false.
    fn get_forwarding_pointer(&self) -> Self::TaggedValue;

    /// Set the forwarding pointer in this object's header.
    ///
    /// This overwrites the header with the new location.
    fn set_forwarding_pointer(&mut self, new_location: Self::TaggedValue);
}

// =============================================================================
// Header Operations (Low-level bit manipulation)
// =============================================================================

/// Low-level header bit operations.
///
/// This trait provides static methods for manipulating header values
/// without needing an object handle. Useful for direct memory manipulation
/// during GC phases.
///
/// # Note
/// This trait is optional - implementations can choose to expose these
/// operations through `GcObject` instead. It's provided for runtimes
/// that need fine-grained control over header bit manipulation.
pub trait HeaderOps {
    /// Maximum field count that fits in the inline size field.
    ///
    /// Objects larger than this need extended headers (e.g., 16 bytes instead of 8).
    const MAX_INLINE_SIZE: usize;

    /// Set the marked bit in a raw header value.
    fn set_marked_bit(header: usize) -> usize;

    /// Clear the marked bit in a raw header value.
    fn clear_marked_bit(header: usize) -> usize;

    /// Check if the marked bit is set.
    fn is_marked_bit_set(header: usize) -> bool;

    /// Set the forwarding bit (marks header as containing forwarding pointer).
    fn set_forwarding_bit(value: usize) -> usize;

    /// Clear the forwarding bit.
    fn clear_forwarding_bit(value: usize) -> usize;

    /// Check if the forwarding bit is set.
    fn is_forwarding_bit_set(value: usize) -> bool;

    /// Check if this is a large object (size stored in extended header).
    fn is_large_object(header: usize) -> bool;
}

// =============================================================================
// Root Scanning
// =============================================================================

/// Trait for providing GC roots.
///
/// Roots are the entry points for GC tracing - typically stack slots,
/// global variables, and other runtime-managed locations that may
/// contain heap pointers.
///
/// The GC library does not prescribe a specific root scanning mechanism.
/// Implementations can use:
/// - Frame pointer chain walking
/// - Conservative stack scanning
/// - Precise stack maps with register saves
/// - Global root registries
///
/// # Important
/// The callback receives `(slot_address, tagged_value)`. The slot address
/// is required so that compacting GC can update the root in-place after
/// relocating the target object.
///
/// # Example Implementation
/// ```rust,ignore
/// struct StackRoots {
///     slots: Vec<*mut usize>,
/// }
///
/// impl<T: TaggedPointer> RootProvider<T> for StackRoots {
///     fn enumerate_roots(&self, callback: &mut dyn FnMut(usize, T)) {
///         for &slot_ptr in &self.slots {
///             let value = unsafe { *slot_ptr };
///             let tagged = T::from_usize(value);
///             if tagged.is_heap_pointer() {
///                 callback(slot_ptr as usize, tagged);
///             }
///         }
///     }
/// }
/// ```
pub trait RootProvider<T: TaggedPointer> {
    /// Enumerate all roots and invoke the callback for each one.
    ///
    /// # Parameters
    /// - `callback`: Called with `(slot_address, tagged_value)` for each root
    ///
    /// # Callback Parameters
    /// - `slot_addr`: Address of the memory location containing the root.
    ///   This allows the GC to update the root after object relocation.
    /// - `value`: The tagged pointer value at that slot.
    fn enumerate_roots(&self, callback: &mut dyn FnMut(usize, T));
}

// =============================================================================
// Size Abstraction
// =============================================================================

/// Word-size abstraction for memory calculations.
///
/// Provides consistent conversion between words (8 bytes on 64-bit) and bytes.
/// This helps avoid off-by-one errors in size calculations.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Word(usize);

impl Word {
    /// Number of bytes per word (8 on 64-bit systems).
    pub const BYTES_PER_WORD: usize = 8;

    /// Create a Word from a word count.
    pub fn from_words(words: usize) -> Self {
        Word(words)
    }

    /// Create a Word from a byte count.
    ///
    /// # Panics
    /// Debug-asserts that `bytes` is a multiple of `BYTES_PER_WORD`.
    pub fn from_bytes(bytes: usize) -> Self {
        debug_assert!(
            bytes % Self::BYTES_PER_WORD == 0,
            "byte count {} is not word-aligned",
            bytes
        );
        Word(bytes / Self::BYTES_PER_WORD)
    }

    /// Convert to word count.
    pub fn to_words(self) -> usize {
        self.0
    }

    /// Convert to byte count.
    pub fn to_bytes(self) -> usize {
        self.0 * Self::BYTES_PER_WORD
    }
}

// =============================================================================
// Convenience Type Alias
// =============================================================================

/// Convenience bound for GC implementations that require forwarding support.
///
/// This is used by compacting and generational collectors.
pub trait GcRuntime: GcTypes
where
    Self::ObjectHandle: ForwardingSupport,
{
}

impl<T> GcRuntime for T
where
    T: GcTypes,
    T::ObjectHandle: ForwardingSupport,
{
}
