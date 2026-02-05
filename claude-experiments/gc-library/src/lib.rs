//! Generic garbage collection library.
//!
//! This library provides several GC algorithms that can be used with any
//! language runtime implementing the traits in [`traits`].
//!
//! # Getting Started
//!
//! 1. Implement the traits in [`traits`] for your runtime's object model
//! 2. Choose a GC algorithm from [`gc`]
//! 3. Create roots using [`traits::RootProvider`]
//!
//! # Available Collectors
//!
//! - [`gc::mark_and_sweep::MarkAndSweep`] - Simple non-moving collector
//! - [`gc::compacting::CompactingHeap`] - Semi-space copying collector
//! - [`gc::generational::GenerationalGC`] - Generational collector with write barriers
//! - [`gc::mutex_allocator::MutexAllocator`] - Thread-safe wrapper
//!
//! # C Interface
//!
//! For C/C++ integration, use the functions in [`ffi`]. These provide a stable
//! C ABI using opaque handles and callback-based root enumeration.
//!
//! # Example (Rust)
//!
//! ```rust,ignore
//! use gc_library::traits::{GcTypes, GcObject, TaggedPointer};
//! use gc_library::gc::{Allocator, AllocatorOptions};
//! use gc_library::gc::mark_and_sweep::MarkAndSweep;
//! use gc_library::example::{ExampleRuntime, ExampleTypeTag, SimpleRoots};
//!
//! // Create a mark-and-sweep GC
//! let mut gc: MarkAndSweep<ExampleRuntime> = MarkAndSweep::new(AllocatorOptions::new());
//!
//! // Allocate objects
//! let obj = gc.try_allocate(2, ExampleTypeTag::HeapObject).unwrap();
//!
//! // Run GC with roots
//! let roots = SimpleRoots::new();
//! gc.gc(&roots);
//! ```
//!
//! # Example (C)
//!
//! ```c
//! #include "gc_library.h"
//!
//! void enumerate_roots(void* ctx, GcRootCallback cb) {
//!     // Call cb(slot_addr, value) for each root
//! }
//!
//! int main() {
//!     GcHandle* gc = gc_lib_create_mark_sweep();
//!     uint8_t* obj = gc_lib_allocate(gc, 2, enumerate_roots, NULL);
//!     gc_lib_collect(gc, enumerate_roots, NULL);
//!     gc_lib_destroy(gc);
//! }
//! ```

pub mod example;
pub mod ffi;
pub mod gc;
pub mod traits;

// Re-export commonly used items
pub use gc::{AllocateAction, Allocator, AllocatorOptions, get_page_size};
pub use traits::{
    ForwardingSupport, GcObject, GcRuntime, GcTypes, HeaderOps, ObjectKind, RootProvider,
    TaggedPointer, Word,
};

// Re-export FFI types for C interop
pub use ffi::{GcHandle, GcRootCallback, GcRootEnumerator};
