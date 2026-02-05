//! Garbage collection implementations.
//!
//! This module provides several GC algorithms that work with any runtime
//! implementing the traits in [`crate::traits`].
//!
//! # Available Collectors
//!
//! - [`mark_and_sweep::MarkAndSweep`] - Simple mark-and-sweep collector
//! - [`compacting::CompactingHeap`] - Copying/compacting collector (Cheney's algorithm)
//! - [`generational::GenerationalGC`] - Generational collector with write barriers
//! - [`mutex_allocator::MutexAllocator`] - Thread-safe wrapper for any allocator

use std::{error::Error, thread::ThreadId};

use crate::traits::{GcTypes, RootProvider, TaggedPointer};

pub mod compacting;
pub mod generational;
pub mod mark_and_sweep;
pub mod mutex_allocator;
pub mod usdt_probes;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod adversarial_tests;

// =============================================================================
// Platform Utilities
// =============================================================================

/// Get the system page size.
///
/// This is cached after the first call for efficiency.
pub fn get_page_size() -> usize {
    use std::sync::OnceLock;
    static PAGE_SIZE: OnceLock<usize> = OnceLock::new();

    *PAGE_SIZE.get_or_init(|| {
        #[cfg(target_os = "macos")]
        {
            unsafe { libc::vm_page_size as usize }
        }
        #[cfg(target_os = "linux")]
        {
            unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            4096 // Default fallback
        }
    })
}

// =============================================================================
// Allocator Configuration
// =============================================================================

/// Options for configuring allocator behavior.
#[derive(Debug, Clone, Copy)]
pub struct AllocatorOptions {
    /// Whether GC is enabled. If false, memory is never collected.
    pub gc: bool,
    /// Whether to print GC timing statistics.
    pub print_stats: bool,
    /// Whether to GC on every allocation (for debugging).
    pub gc_always: bool,
}

impl Default for AllocatorOptions {
    fn default() -> Self {
        AllocatorOptions {
            gc: true,
            print_stats: false,
            gc_always: false,
        }
    }
}

impl AllocatorOptions {
    /// Create options with GC enabled and no debugging.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create options with GC disabled (useful for benchmarking).
    pub fn no_gc() -> Self {
        AllocatorOptions {
            gc: false,
            ..Default::default()
        }
    }
}

// =============================================================================
// Allocation Result
// =============================================================================

/// Result of an allocation attempt.
pub enum AllocateAction {
    /// Allocation succeeded, returns pointer to the new object.
    Allocated(*const u8),
    /// Allocation failed due to space pressure, GC is needed.
    Gc,
}

// =============================================================================
// Allocator Trait
// =============================================================================

/// Core trait for GC allocators.
///
/// Implementations provide memory allocation and garbage collection for a
/// specific runtime type system (defined by `T: GcTypes`).
///
/// # Type Parameters
///
/// - `T`: The runtime's type system, implementing [`GcTypes`]
///
/// # Example
///
/// ```rust,ignore
/// use gc_library::gc::{Allocator, AllocatorOptions, AllocateAction};
/// use gc_library::example::ExampleRuntime;
///
/// fn allocate_object<A: Allocator<ExampleRuntime>>(
///     alloc: &mut A,
///     size_words: usize,
/// ) -> *const u8 {
///     loop {
///         match alloc.try_allocate(size_words, ExampleTypeTag::HeapObject).unwrap() {
///             AllocateAction::Allocated(ptr) => return ptr,
///             AllocateAction::Gc => {
///                 // Collect roots and run GC
///                 alloc.gc(&mut roots);
///             }
///         }
///     }
/// }
/// ```
pub trait Allocator<T: GcTypes>: Sized {
    /// Create a new allocator with the given options.
    fn new(options: AllocatorOptions) -> Self;

    /// Attempt to allocate space for an object.
    ///
    /// # Parameters
    /// - `words`: Size of the object in words (not including header)
    /// - `kind`: The type of object being allocated
    ///
    /// # Returns
    /// - `AllocateAction::Allocated(ptr)` - Allocation succeeded
    /// - `AllocateAction::Gc` - Need to run GC before retrying
    fn try_allocate(
        &mut self,
        words: usize,
        kind: T::ObjectKind,
    ) -> Result<AllocateAction, Box<dyn Error>>;

    /// Allocate with zeroed memory.
    ///
    /// Used for arrays that don't initialize all fields immediately.
    /// Default implementation just calls `try_allocate`.
    fn try_allocate_zeroed(
        &mut self,
        words: usize,
        kind: T::ObjectKind,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        self.try_allocate(words, kind)
    }

    /// Allocate a long-lived object for runtime infrastructure.
    ///
    /// For generational GC, this allocates directly in the old generation.
    /// For other GCs, this is the same as `try_allocate`.
    ///
    /// # Returns
    /// Tagged pointer to the allocated object, or error if allocation failed.
    fn allocate_for_runtime(
        &mut self,
        words: usize,
        kind: T::ObjectKind,
    ) -> Result<T::TaggedValue, Box<dyn Error>> {
        match self.try_allocate(words, kind)? {
            AllocateAction::Allocated(ptr) => Ok(T::TaggedValue::tag(ptr, kind)),
            AllocateAction::Gc => Err("Need GC to allocate runtime object".into()),
        }
    }

    /// Run garbage collection.
    ///
    /// # Parameters
    /// - `roots`: Provider that enumerates all GC roots
    fn gc(&mut self, roots: &dyn RootProvider<T::TaggedValue>);

    /// Grow the heap to accommodate more objects.
    ///
    /// Called when allocation fails even after GC.
    fn grow(&mut self);

    /// Get a pointer used for thread pause synchronization.
    ///
    /// Returns 0 for allocators that don't support this.
    #[allow(unused)]
    fn get_pause_pointer(&self) -> usize {
        0
    }

    /// Register a thread with the allocator.
    ///
    /// Called when a new thread starts that may allocate.
    fn register_thread(&mut self, _thread_id: ThreadId) {}

    /// Unregister a thread from the allocator.
    ///
    /// Called when a thread exits.
    fn remove_thread(&mut self, _thread_id: ThreadId) {}

    /// Register a parked thread's stack pointer.
    ///
    /// Used for stop-the-world GC to know where to scan.
    fn register_parked_thread(&mut self, _thread_id: ThreadId, _stack_pointer: usize) {}

    /// Get the current allocator options.
    fn get_allocation_options(&self) -> AllocatorOptions;

    /// Write barrier for generational GC.
    ///
    /// Called after writing a pointer into a heap object's field.
    /// For generational GC, records old-to-young pointers in a remembered set.
    ///
    /// Default implementation is a no-op (for non-generational GCs).
    fn write_barrier(&mut self, _object_ptr: usize, _new_value: usize) {
        // Default: no-op
    }

    /// Get the card table biased pointer for generated code write barriers.
    ///
    /// For generational GC, returns a biased pointer for fast card marking.
    /// For non-generational GCs, returns null.
    fn get_card_table_biased_ptr(&self) -> *mut u8 {
        std::ptr::null_mut()
    }

    /// Mark a card unconditionally for an object.
    ///
    /// Used by generated code write barriers when the written value is unknown.
    /// Default implementation is a no-op (for non-generational GCs).
    fn mark_card_unconditional(&mut self, _object_ptr: usize) {
        // Default: no-op
    }

    /// Get young generation bounds (start, end) for fast write barrier checks.
    ///
    /// Objects in young generation don't need write barriers.
    /// For non-generational GCs, returns (0, 0).
    fn get_young_gen_bounds(&self) -> (usize, usize) {
        (0, 0)
    }
}
