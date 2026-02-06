//! C Foreign Function Interface for gc-library.
//!
//! This module provides a C-compatible API for using the garbage collector
//! from non-Rust languages. It uses the [`crate::example::ExampleRuntime`]
//! as the default runtime implementation.
//!
//! # Usage from C
//!
//! ```c
//! #include "gc_library.h"
//!
//! // Root enumeration callback
//! void enumerate_roots(void* context, GcRootCallback callback) {
//!     // Call callback(slot_addr, value) for each root
//!     for (int i = 0; i < root_count; i++) {
//!         callback(&roots[i], roots[i]);
//!     }
//! }
//!
//! int main() {
//!     // Create a mark-and-sweep GC
//!     GcHandle* gc = gc_lib_create_mark_sweep();
//!
//!     // Allocate an object with 2 pointer fields
//!     uint8_t* obj = gc_lib_allocate(gc, 2);
//!
//!     // Run GC (with root enumeration)
//!     gc_lib_collect(gc, enumerate_roots, NULL);
//!
//!     // Clean up
//!     gc_lib_destroy(gc);
//! }
//! ```
//!
//! # Thread Safety
//!
//! The basic GC handles are NOT thread-safe. For multi-threaded use,
//! use the `gc_lib_create_*_threadsafe` variants.

use std::ffi::c_void;

use crate::example::{ExampleObject, ExampleRuntime, ExampleTaggedPtr, ExampleTypeTag};
use crate::gc::generational::GenerationalGC;
use crate::gc::mark_and_sweep::MarkAndSweep;
use crate::gc::mutex_allocator::MutexAllocator;
use crate::gc::{AllocateAction, Allocator, AllocatorOptions, LibcMemoryProvider};
use crate::traits::{GcObject, RootProvider, TaggedPointer};

// =============================================================================
// Opaque Handle Types
// =============================================================================

/// Opaque handle to a GC instance.
///
/// This is an enum internally to support different GC algorithms,
/// but is opaque to C code.
pub enum GcHandle {
    MarkSweep(MarkAndSweep<ExampleRuntime, LibcMemoryProvider>),
    Generational(GenerationalGC<ExampleRuntime, LibcMemoryProvider>),
    MarkSweepThreadSafe(MutexAllocator<MarkAndSweep<ExampleRuntime, LibcMemoryProvider>, ExampleRuntime, LibcMemoryProvider>),
    GenerationalThreadSafe(MutexAllocator<GenerationalGC<ExampleRuntime, LibcMemoryProvider>, ExampleRuntime, LibcMemoryProvider>),
}

/// Callback type for root enumeration.
///
/// Called by the user's root enumerator for each root.
/// - `slot_addr`: Address of the memory location containing the root
/// - `value`: The tagged pointer value at that slot
pub type GcRootCallback = extern "C" fn(slot_addr: usize, value: usize);

/// User-provided function to enumerate roots.
///
/// - `context`: User-provided context pointer
/// - `callback`: Function to call for each root
pub type GcRootEnumerator = extern "C" fn(context: *mut c_void, callback: GcRootCallback);

// =============================================================================
// Root Provider Wrapper
// =============================================================================

/// Wrapper to convert C-style root enumeration to Rust trait.
struct CRootProvider {
    enumerator: GcRootEnumerator,
    context: *mut c_void,
    /// Storage for collected roots (slot_addr, tagged_value)
    roots: std::cell::RefCell<Vec<(usize, ExampleTaggedPtr)>>,
}

impl CRootProvider {
    fn new(enumerator: GcRootEnumerator, context: *mut c_void) -> Self {
        Self {
            enumerator,
            context,
            roots: std::cell::RefCell::new(Vec::new()),
        }
    }

    /// Collect all roots from the C callback into our internal storage.
    fn collect_roots(&self) {
        // Use a static mutable to temporarily store the roots vector pointer
        // This is safe because we're single-threaded and control the scope
        static mut ROOTS_PTR: *mut Vec<(usize, ExampleTaggedPtr)> = std::ptr::null_mut();

        extern "C" fn trampoline(slot_addr: usize, value: usize) {
            let tagged = ExampleTaggedPtr::from_usize(value);
            if tagged.is_heap_pointer() {
                unsafe {
                    if !ROOTS_PTR.is_null() {
                        (*ROOTS_PTR).push((slot_addr, tagged));
                    }
                }
            }
        }

        let mut roots = self.roots.borrow_mut();
        roots.clear();

        unsafe {
            ROOTS_PTR = roots.as_mut() as *mut Vec<(usize, ExampleTaggedPtr)>;
        }

        // Call user's enumerator
        (self.enumerator)(self.context, trampoline);

        unsafe {
            ROOTS_PTR = std::ptr::null_mut();
        }
    }
}

impl RootProvider<ExampleTaggedPtr> for CRootProvider {
    fn enumerate_roots(&self, callback: &mut dyn FnMut(usize, ExampleTaggedPtr)) {
        // First collect all roots from the C callback
        self.collect_roots();

        // Then iterate over them
        let roots = self.roots.borrow();
        for &(slot_addr, tagged) in roots.iter() {
            callback(slot_addr, tagged);
        }
    }
}

// =============================================================================
// GC Creation Functions
// =============================================================================

/// Create a new mark-and-sweep garbage collector.
///
/// # Returns
/// Pointer to a new GC handle, or NULL on failure.
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_create_mark_sweep() -> *mut GcHandle {
    let gc = MarkAndSweep::new(AllocatorOptions::new(), LibcMemoryProvider::new());
    Box::into_raw(Box::new(GcHandle::MarkSweep(gc)))
}

/// Create a new mark-and-sweep garbage collector with custom options.
///
/// # Parameters
/// - `gc_enabled`: Whether GC is enabled (false = never collect)
/// - `print_stats`: Whether to print GC timing statistics
///
/// # Returns
/// Pointer to a new GC handle, or NULL on failure.
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_create_mark_sweep_with_options(
    gc_enabled: bool,
    print_stats: bool,
) -> *mut GcHandle {
    let options = AllocatorOptions {
        gc: gc_enabled,
        print_stats,
        gc_always: false,
    };
    let gc = MarkAndSweep::new(options, LibcMemoryProvider::new());
    Box::into_raw(Box::new(GcHandle::MarkSweep(gc)))
}

/// Create a new generational garbage collector.
///
/// # Returns
/// Pointer to a new GC handle, or NULL on failure.
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_create_generational() -> *mut GcHandle {
    let gc = GenerationalGC::new(AllocatorOptions::new(), LibcMemoryProvider::new());
    Box::into_raw(Box::new(GcHandle::Generational(gc)))
}

/// Create a new generational garbage collector with custom options.
///
/// # Parameters
/// - `gc_enabled`: Whether GC is enabled (false = never collect)
/// - `print_stats`: Whether to print GC timing statistics
///
/// # Returns
/// Pointer to a new GC handle, or NULL on failure.
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_create_generational_with_options(
    gc_enabled: bool,
    print_stats: bool,
) -> *mut GcHandle {
    let options = AllocatorOptions {
        gc: gc_enabled,
        print_stats,
        gc_always: false,
    };
    let gc = GenerationalGC::new(options, LibcMemoryProvider::new());
    Box::into_raw(Box::new(GcHandle::Generational(gc)))
}

/// Create a thread-safe mark-and-sweep garbage collector.
///
/// # Returns
/// Pointer to a new GC handle, or NULL on failure.
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_create_mark_sweep_threadsafe() -> *mut GcHandle {
    let gc = MutexAllocator::new(AllocatorOptions::new(), LibcMemoryProvider::new());
    Box::into_raw(Box::new(GcHandle::MarkSweepThreadSafe(gc)))
}

/// Create a thread-safe generational garbage collector.
///
/// # Returns
/// Pointer to a new GC handle, or NULL on failure.
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_create_generational_threadsafe() -> *mut GcHandle {
    let gc = MutexAllocator::new(AllocatorOptions::new(), LibcMemoryProvider::new());
    Box::into_raw(Box::new(GcHandle::GenerationalThreadSafe(gc)))
}

/// Destroy a GC instance and free its resources.
///
/// # Safety
/// - `gc` must be a valid pointer returned by a `gc_lib_create_*` function
/// - `gc` must not be used after this call
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_destroy(gc: *mut GcHandle) {
    if gc.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(gc));
    }
}

// =============================================================================
// Allocation Functions
// =============================================================================

/// Allocate a new object with the given number of pointer fields.
///
/// The object will have:
/// - An 8-byte header (managed by the GC)
/// - `field_count * 8` bytes of payload for pointer fields
///
/// # Parameters
/// - `gc`: GC handle
/// - `field_count`: Number of pointer-sized fields
///
/// # Returns
/// Pointer to the allocated object (points to header), or NULL if GC is needed.
///
/// # Note
/// If this returns NULL, call `gc_lib_collect()` then retry allocation.
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_try_allocate(gc: *mut GcHandle, field_count: usize) -> *const u8 {
    if gc.is_null() {
        return std::ptr::null();
    }

    let gc = unsafe { &mut *gc };

    let result = match gc {
        GcHandle::MarkSweep(ms) => ms.try_allocate(field_count, ExampleTypeTag::HeapObject),
        GcHandle::Generational(gengc) => {
            gengc.try_allocate(field_count, ExampleTypeTag::HeapObject)
        }
        GcHandle::MarkSweepThreadSafe(ms) => {
            ms.try_allocate(field_count, ExampleTypeTag::HeapObject)
        }
        GcHandle::GenerationalThreadSafe(gengc) => {
            gengc.try_allocate(field_count, ExampleTypeTag::HeapObject)
        }
    };

    match result {
        Ok(AllocateAction::Allocated(ptr)) => ptr,
        Ok(AllocateAction::Gc) => std::ptr::null(),
        Err(_) => std::ptr::null(),
    }
}

/// Allocate a new object, running GC if needed.
///
/// This is a convenience function that handles the GC loop internally.
///
/// # Parameters
/// - `gc`: GC handle
/// - `field_count`: Number of pointer-sized fields
/// - `root_enumerator`: Function that enumerates all GC roots
/// - `context`: User context passed to root_enumerator
///
/// # Returns
/// Pointer to the allocated object, or NULL on fatal error.
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_allocate(
    gc: *mut GcHandle,
    field_count: usize,
    root_enumerator: GcRootEnumerator,
    context: *mut c_void,
) -> *const u8 {
    if gc.is_null() {
        return std::ptr::null();
    }

    let gc_ref = unsafe { &mut *gc };
    let roots = CRootProvider::new(root_enumerator, context);

    // Allocation loop with GC retry
    loop {
        let result = match gc_ref {
            GcHandle::MarkSweep(ms) => ms.try_allocate(field_count, ExampleTypeTag::HeapObject),
            GcHandle::Generational(gengc) => {
                gengc.try_allocate(field_count, ExampleTypeTag::HeapObject)
            }
            GcHandle::MarkSweepThreadSafe(ms) => {
                ms.try_allocate(field_count, ExampleTypeTag::HeapObject)
            }
            GcHandle::GenerationalThreadSafe(gengc) => {
                gengc.try_allocate(field_count, ExampleTypeTag::HeapObject)
            }
        };

        match result {
            Ok(AllocateAction::Allocated(ptr)) => return ptr,
            Ok(AllocateAction::Gc) => {
                // Run GC and retry
                match gc_ref {
                    GcHandle::MarkSweep(ms) => ms.gc(&roots),
                    GcHandle::Generational(gengc) => gengc.gc(&roots),
                    GcHandle::MarkSweepThreadSafe(ms) => ms.gc(&roots),
                    GcHandle::GenerationalThreadSafe(gengc) => gengc.gc(&roots),
                }
            }
            Err(_) => return std::ptr::null(),
        }
    }
}

/// Initialize an allocated object's fields to zero.
///
/// # Parameters
/// - `obj`: Pointer to the object (as returned by allocate functions)
/// - `field_count`: Number of fields to zero
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_zero_fields(obj: *mut u8, field_count: usize) {
    if obj.is_null() {
        return;
    }

    // Fields start after the 8-byte header
    let fields_ptr = unsafe { obj.add(8) as *mut usize };
    for i in 0..field_count {
        unsafe {
            *fields_ptr.add(i) = 0;
        }
    }
}

// =============================================================================
// Object Field Access
// =============================================================================

/// Read a field from an object.
///
/// # Parameters
/// - `obj`: Pointer to the object
/// - `field_index`: Index of the field to read (0-based)
///
/// # Returns
/// The field value (a tagged pointer or other value).
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_read_field(obj: *const u8, field_index: usize) -> usize {
    if obj.is_null() {
        return 0;
    }

    let heap_obj = ExampleObject::from_untagged(obj);
    let fields = heap_obj.get_fields();

    if field_index < fields.len() {
        fields[field_index]
    } else {
        0
    }
}

/// Write a field in an object.
///
/// # Parameters
/// - `obj`: Pointer to the object
/// - `field_index`: Index of the field to write (0-based)
/// - `value`: The value to write (a tagged pointer or other value)
///
/// # Note
/// For generational GC, you should call `gc_lib_write_barrier` after this.
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_write_field(obj: *mut u8, field_index: usize, value: usize) {
    if obj.is_null() {
        return;
    }

    let mut heap_obj = ExampleObject::from_untagged(obj);
    let fields = heap_obj.get_fields_mut();

    if field_index < fields.len() {
        fields[field_index] = value;
    }
}

// =============================================================================
// Garbage Collection
// =============================================================================

/// Run garbage collection.
///
/// # Parameters
/// - `gc`: GC handle
/// - `root_enumerator`: Function that enumerates all GC roots
/// - `context`: User context passed to root_enumerator
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_collect(
    gc: *mut GcHandle,
    root_enumerator: GcRootEnumerator,
    context: *mut c_void,
) {
    if gc.is_null() {
        return;
    }

    let gc = unsafe { &mut *gc };
    let roots = CRootProvider::new(root_enumerator, context);

    match gc {
        GcHandle::MarkSweep(ms) => ms.gc(&roots),
        GcHandle::Generational(gengc) => gengc.gc(&roots),
        GcHandle::MarkSweepThreadSafe(ms) => ms.gc(&roots),
        GcHandle::GenerationalThreadSafe(gengc) => gengc.gc(&roots),
    }
}

/// Grow the heap to accommodate more objects.
///
/// Call this after GC if allocation still fails.
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_grow(gc: *mut GcHandle) {
    if gc.is_null() {
        return;
    }

    let gc = unsafe { &mut *gc };

    match gc {
        GcHandle::MarkSweep(ms) => ms.grow(),
        GcHandle::Generational(gengc) => gengc.grow(),
        GcHandle::MarkSweepThreadSafe(ms) => ms.grow(),
        GcHandle::GenerationalThreadSafe(gengc) => gengc.grow(),
    }
}

// =============================================================================
// Write Barrier (for Generational GC)
// =============================================================================

/// Write barrier for generational GC.
///
/// Call this after writing a pointer into a heap object's field.
/// This is only needed for generational GC; it's a no-op for mark-sweep.
///
/// # Parameters
/// - `gc`: GC handle
/// - `object_ptr`: Tagged pointer to the object being written to
/// - `new_value`: The new value being written (tagged pointer)
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_write_barrier(gc: *mut GcHandle, object_ptr: usize, new_value: usize) {
    if gc.is_null() {
        return;
    }

    let gc = unsafe { &mut *gc };

    match gc {
        GcHandle::MarkSweep(_) => {} // No-op for mark-sweep
        GcHandle::Generational(gengc) => gengc.write_barrier(object_ptr, new_value),
        GcHandle::MarkSweepThreadSafe(_) => {} // No-op for mark-sweep
        GcHandle::GenerationalThreadSafe(gengc) => gengc.write_barrier(object_ptr, new_value),
    }
}

/// Mark a card unconditionally (for generated code write barriers).
///
/// # Parameters
/// - `gc`: GC handle
/// - `object_ptr`: Address of the object being written to
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_mark_card(gc: *mut GcHandle, object_ptr: usize) {
    if gc.is_null() {
        return;
    }

    let gc = unsafe { &mut *gc };

    match gc {
        GcHandle::MarkSweep(_) => {}
        GcHandle::Generational(gengc) => gengc.mark_card_unconditional(object_ptr),
        GcHandle::MarkSweepThreadSafe(_) => {}
        GcHandle::GenerationalThreadSafe(gengc) => gengc.mark_card_unconditional(object_ptr),
    }
}

/// Get young generation bounds for fast write barrier checks.
///
/// # Parameters
/// - `gc`: GC handle
/// - `start_out`: Output pointer for young gen start address
/// - `end_out`: Output pointer for young gen end address
///
/// # Note
/// For non-generational GCs, both values will be 0.
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_get_young_gen_bounds(
    gc: *mut GcHandle,
    start_out: *mut usize,
    end_out: *mut usize,
) {
    if gc.is_null() {
        return;
    }

    let gc = unsafe { &*gc };

    let (start, end) = match gc {
        GcHandle::MarkSweep(_) => (0, 0),
        GcHandle::Generational(gengc) => gengc.get_young_gen_bounds(),
        GcHandle::MarkSweepThreadSafe(_) => (0, 0),
        GcHandle::GenerationalThreadSafe(gengc) => gengc.get_young_gen_bounds(),
    };

    if !start_out.is_null() {
        unsafe { *start_out = start };
    }
    if !end_out.is_null() {
        unsafe { *end_out = end };
    }
}

/// Get the card table biased pointer for generated code write barriers.
///
/// # Parameters
/// - `gc`: GC handle
///
/// # Returns
/// Biased pointer for card marking, or NULL for non-generational GCs.
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_get_card_table_ptr(gc: *mut GcHandle) -> *mut u8 {
    if gc.is_null() {
        return std::ptr::null_mut();
    }

    let gc = unsafe { &*gc };

    match gc {
        GcHandle::MarkSweep(_) => std::ptr::null_mut(),
        GcHandle::Generational(gengc) => gengc.get_card_table_biased_ptr(),
        GcHandle::MarkSweepThreadSafe(_) => std::ptr::null_mut(),
        GcHandle::GenerationalThreadSafe(gengc) => gengc.get_card_table_biased_ptr(),
    }
}

// =============================================================================
// Tagged Pointer Utilities
// =============================================================================

/// Create a tagged heap pointer from a raw object pointer.
///
/// # Parameters
/// - `raw_ptr`: Raw pointer to an object (as returned by allocate)
///
/// # Returns
/// Tagged pointer value that can be stored in object fields.
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_tag_heap_ptr(raw_ptr: *const u8) -> usize {
    ExampleTaggedPtr::tag(raw_ptr, ExampleTypeTag::HeapObject).as_usize()
}

/// Create a tagged integer value.
///
/// # Parameters
/// - `value`: Integer value to tag
///
/// # Returns
/// Tagged integer value.
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_tag_int(value: i64) -> usize {
    ExampleTaggedPtr::from_int(value).as_usize()
}

/// Create a null tagged value.
///
/// # Returns
/// Tagged null value.
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_tag_null() -> usize {
    ExampleTaggedPtr::null().as_usize()
}

/// Extract the raw pointer from a tagged heap pointer.
///
/// # Parameters
/// - `tagged`: Tagged pointer value
///
/// # Returns
/// Raw pointer, or NULL if not a heap pointer.
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_untag_ptr(tagged: usize) -> *const u8 {
    let ptr = ExampleTaggedPtr::from_usize(tagged);
    if ptr.is_heap_pointer() {
        ptr.untag()
    } else {
        std::ptr::null()
    }
}

/// Extract an integer from a tagged integer value.
///
/// # Parameters
/// - `tagged`: Tagged pointer value
/// - `value_out`: Output pointer for the integer value
///
/// # Returns
/// true if the value was an integer, false otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_untag_int(tagged: usize, value_out: *mut i64) -> bool {
    let ptr = ExampleTaggedPtr::from_usize(tagged);
    if let Some(value) = ptr.as_int() {
        if !value_out.is_null() {
            unsafe { *value_out = value };
        }
        true
    } else {
        false
    }
}

/// Check if a tagged value is a heap pointer.
///
/// # Parameters
/// - `tagged`: Tagged pointer value
///
/// # Returns
/// true if heap pointer, false otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_is_heap_ptr(tagged: usize) -> bool {
    ExampleTaggedPtr::from_usize(tagged).is_heap_pointer()
}

/// Check if a tagged value is null.
///
/// # Parameters
/// - `tagged`: Tagged pointer value
///
/// # Returns
/// true if null, false otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_is_null(tagged: usize) -> bool {
    ExampleTaggedPtr::from_usize(tagged).get_kind() == ExampleTypeTag::Null
}

/// Check if a tagged value is an integer.
///
/// # Parameters
/// - `tagged`: Tagged pointer value
///
/// # Returns
/// true if integer, false otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_is_int(tagged: usize) -> bool {
    ExampleTaggedPtr::from_usize(tagged).get_kind() == ExampleTypeTag::Int
}

// =============================================================================
// Constants (for C header generation)
// =============================================================================

/// Header size in bytes.
pub const GC_LIB_HEADER_SIZE: usize = 8;

/// Bytes per word.
pub const GC_LIB_WORD_SIZE: usize = 8;

/// Tag bits used (3 bits).
pub const GC_LIB_TAG_BITS: usize = 3;

// Export constants as C functions for easy access
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_header_size() -> usize {
    GC_LIB_HEADER_SIZE
}

#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_word_size() -> usize {
    GC_LIB_WORD_SIZE
}

#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_tag_bits() -> usize {
    GC_LIB_TAG_BITS
}

#[cfg(test)]
mod tests {
    use super::*;

    extern "C" fn empty_root_enumerator(_context: *mut c_void, _callback: GcRootCallback) {
        // No roots
    }

    #[test]
    fn test_create_destroy_mark_sweep() {
        let gc = gc_lib_create_mark_sweep();
        assert!(!gc.is_null());
        gc_lib_destroy(gc);
    }

    #[test]
    fn test_create_destroy_generational() {
        let gc = gc_lib_create_generational();
        assert!(!gc.is_null());
        gc_lib_destroy(gc);
    }

    #[test]
    fn test_allocate_and_access() {
        let gc = gc_lib_create_mark_sweep();

        // Allocate object with 2 fields
        let obj = gc_lib_allocate(gc, 2, empty_root_enumerator, std::ptr::null_mut());
        assert!(!obj.is_null());

        // Zero fields
        gc_lib_zero_fields(obj as *mut u8, 2);

        // Read initial values (should be 0)
        let field0 = gc_lib_read_field(obj, 0);
        let field1 = gc_lib_read_field(obj, 1);
        assert_eq!(field0, 0);
        assert_eq!(field1, 0);

        // Write tagged integer
        let int_val = gc_lib_tag_int(42);
        gc_lib_write_field(obj as *mut u8, 0, int_val);

        // Read back
        let read_val = gc_lib_read_field(obj, 0);
        assert!(gc_lib_is_int(read_val));

        let mut extracted: i64 = 0;
        assert!(gc_lib_untag_int(read_val, &mut extracted));
        assert_eq!(extracted, 42);

        gc_lib_destroy(gc);
    }

    #[test]
    fn test_tagged_pointer_utilities() {
        // Test null
        let null_val = gc_lib_tag_null();
        assert!(gc_lib_is_null(null_val));
        assert!(!gc_lib_is_heap_ptr(null_val));
        assert!(!gc_lib_is_int(null_val));

        // Test integer
        let int_val = gc_lib_tag_int(123);
        assert!(!gc_lib_is_null(int_val));
        assert!(!gc_lib_is_heap_ptr(int_val));
        assert!(gc_lib_is_int(int_val));

        let mut extracted: i64 = 0;
        assert!(gc_lib_untag_int(int_val, &mut extracted));
        assert_eq!(extracted, 123);
    }

    #[test]
    fn test_young_gen_bounds() {
        let gc = gc_lib_create_generational();

        let mut start: usize = 0;
        let mut end: usize = 0;
        gc_lib_get_young_gen_bounds(gc, &mut start, &mut end);

        // For generational GC, bounds should be non-zero
        assert!(start > 0);
        assert!(end > start);

        gc_lib_destroy(gc);
    }

    #[test]
    fn test_mark_sweep_young_gen_bounds() {
        let gc = gc_lib_create_mark_sweep();

        let mut start: usize = 0;
        let mut end: usize = 0;
        gc_lib_get_young_gen_bounds(gc, &mut start, &mut end);

        // For mark-sweep, bounds should be zero
        assert_eq!(start, 0);
        assert_eq!(end, 0);

        gc_lib_destroy(gc);
    }
}
