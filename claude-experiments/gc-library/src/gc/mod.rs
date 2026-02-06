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
//! - [`mutex_allocator::MutexAllocator`] - Thread-safe wrapper for any allocator (requires `std`)

use crate::traits::{GcTypes, RootProvider, TaggedPointer};

pub mod compacting;
pub mod generational;
pub mod mark_and_sweep;

#[cfg(feature = "std")]
pub mod mutex_allocator;

pub mod usdt_probes;

#[cfg(all(test, feature = "std"))]
mod tests;

#[cfg(all(test, feature = "std"))]
mod adversarial_tests;

// =============================================================================
// Memory Provider Trait
// =============================================================================

/// Trait for providing memory regions to the GC.
///
/// Implementations handle platform-specific memory allocation (mmap, VirtualAlloc, etc.)
/// or can provide a simple fixed buffer for no_std environments.
///
/// # Example (no_std with fixed buffer)
/// ```rust,ignore
/// struct FixedMemory {
///     buffer: &'static mut [u8],
///     used: usize,
/// }
///
/// impl MemoryProvider for FixedMemory {
///     fn allocate_region(&mut self, size: usize) -> Option<*mut u8> {
///         if self.used + size <= self.buffer.len() {
///             let ptr = self.buffer[self.used..].as_mut_ptr();
///             self.used += size;
///             Some(ptr)
///         } else {
///             None
///         }
///     }
///     fn page_size(&self) -> usize { 4096 }
/// }
/// ```
pub trait MemoryProvider {
    /// Allocate a region of at least `size` bytes.
    ///
    /// Returns a pointer to the start of the region, or None if allocation failed.
    /// The memory should be readable and writable.
    fn allocate_region(&mut self, size: usize) -> Option<*mut u8>;

    /// Commit/make accessible a range of previously allocated memory.
    ///
    /// Some systems (like mmap with PROT_NONE) reserve address space without
    /// making it accessible. This method makes the memory usable.
    ///
    /// Default implementation assumes memory is already accessible.
    fn commit(&mut self, _ptr: *mut u8, _size: usize) -> bool {
        true
    }

    /// Return the page size for this memory provider.
    ///
    /// Used for alignment and growth calculations.
    fn page_size(&self) -> usize;
}

// =============================================================================
// Libc Memory Provider (std feature only)
// =============================================================================

#[cfg(feature = "std")]
mod libc_provider {
    use super::MemoryProvider;

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

    /// Memory provider using libc mmap/mprotect.
    ///
    /// This is the default provider when the `std` feature is enabled.
    #[derive(Clone)]
    pub struct LibcMemoryProvider {
        page_size: usize,
    }

    impl LibcMemoryProvider {
        /// Create a new libc-based memory provider.
        pub fn new() -> Self {
            Self {
                page_size: get_page_size(),
            }
        }
    }

    impl Default for LibcMemoryProvider {
        fn default() -> Self {
            Self::new()
        }
    }

    impl MemoryProvider for LibcMemoryProvider {
        fn allocate_region(&mut self, size: usize) -> Option<*mut u8> {
            let ptr = unsafe {
                libc::mmap(
                    core::ptr::null_mut(),
                    size,
                    libc::PROT_NONE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            if ptr == libc::MAP_FAILED {
                None
            } else {
                Some(ptr as *mut u8)
            }
        }

        fn commit(&mut self, ptr: *mut u8, size: usize) -> bool {
            unsafe {
                libc::mprotect(ptr as *mut _, size, libc::PROT_READ | libc::PROT_WRITE) == 0
            }
        }

        fn page_size(&self) -> usize {
            self.page_size
        }
    }
}

#[cfg(feature = "std")]
pub use libc_provider::{get_page_size, LibcMemoryProvider};

// =============================================================================
// Allocator Configuration
// =============================================================================

/// Options for configuring allocator behavior.
#[derive(Debug, Clone, Copy)]
pub struct AllocatorOptions {
    /// Whether GC is enabled. If false, memory is never collected.
    pub gc: bool,
    /// Whether to print GC timing statistics (requires `std`).
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
// Allocation Error
// =============================================================================

/// Error type for allocation failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocError {
    /// Out of memory - no space available even after GC
    OutOfMemory,
    /// Memory provider failed to allocate/commit
    ProviderFailed,
    /// Object too large for this allocator
    ObjectTooLarge,
}

impl core::fmt::Display for AllocError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AllocError::OutOfMemory => write!(f, "out of memory"),
            AllocError::ProviderFailed => write!(f, "memory provider failed"),
            AllocError::ObjectTooLarge => write!(f, "object too large"),
        }
    }
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
/// - `M`: The memory provider, implementing [`MemoryProvider`]
pub trait Allocator<T: GcTypes, M: MemoryProvider>: Sized {
    /// Create a new allocator with the given options and memory provider.
    fn new(options: AllocatorOptions, memory: M) -> Self;

    /// Attempt to allocate space for an object.
    ///
    /// # Parameters
    /// - `words`: Size of the object in words (not including header)
    /// - `kind`: The type of object being allocated
    ///
    /// # Returns
    /// - `Ok(AllocateAction::Allocated(ptr))` - Allocation succeeded
    /// - `Ok(AllocateAction::Gc)` - Need to run GC before retrying
    /// - `Err(AllocError)` - Fatal allocation error
    fn try_allocate(
        &mut self,
        words: usize,
        kind: T::ObjectKind,
    ) -> Result<AllocateAction, AllocError>;

    /// Allocate with zeroed memory.
    ///
    /// Used for arrays that don't initialize all fields immediately.
    /// Default implementation just calls `try_allocate`.
    fn try_allocate_zeroed(
        &mut self,
        words: usize,
        kind: T::ObjectKind,
    ) -> Result<AllocateAction, AllocError> {
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
    ) -> Result<T::TaggedValue, AllocError> {
        match self.try_allocate(words, kind)? {
            AllocateAction::Allocated(ptr) => Ok(T::TaggedValue::tag(ptr, kind)),
            AllocateAction::Gc => Err(AllocError::OutOfMemory),
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
        core::ptr::null_mut()
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

// =============================================================================
// Thread-related methods (std only)
// =============================================================================

#[cfg(feature = "std")]
use std::thread::ThreadId;

/// Extension trait for thread-aware allocators (requires `std`).
#[cfg(feature = "std")]
pub trait ThreadAwareAllocator<T: GcTypes, M: MemoryProvider>: Allocator<T, M> {
    /// Get a pointer used for thread pause synchronization.
    ///
    /// Returns 0 for allocators that don't support this.
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
}
