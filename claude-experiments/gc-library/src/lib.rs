//! Generic garbage collection library.
//!
//! This library provides several GC algorithms that can be used with any
//! language runtime implementing the traits in [`traits`].
//!
//! # no_std Support
//!
//! This crate supports `no_std` environments with the `alloc` crate.
//! Disable the default `std` feature and provide a [`MemoryProvider`] implementation.
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
//! - [`gc::mutex_allocator::MutexAllocator`] - Thread-safe wrapper (requires `std` feature)
//!
//! # C Interface
//!
//! For C/C++ integration, use the functions in [`ffi`] (requires `std` feature).

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

// For no_std staticlib builds: provide minimal panic handler
#[cfg(not(feature = "std"))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

// For no_std staticlib builds: use a minimal allocator
// Users can override this with their own #[global_allocator]
#[cfg(not(feature = "std"))]
mod no_std_alloc {
    use core::alloc::{GlobalAlloc, Layout};

    struct Abort;

    unsafe impl GlobalAlloc for Abort {
        unsafe fn alloc(&self, _: Layout) -> *mut u8 {
            // This will be overridden by the actual runtime's allocator
            core::ptr::null_mut()
        }
        unsafe fn dealloc(&self, _: *mut u8, _: Layout) {}
    }

    #[global_allocator]
    static ALLOCATOR: Abort = Abort;
}

pub mod gc;
pub mod traits;

#[cfg(feature = "std")]
pub mod example;

#[cfg(feature = "std")]
pub mod ffi;

// Re-export commonly used items
pub use gc::{AllocateAction, Allocator, AllocatorOptions};
pub use traits::{
    ForwardingSupport, GcObject, GcRuntime, GcTypes, HeaderOps, ObjectKind, RootProvider,
    TaggedPointer, Word,
};

// Re-export memory provider trait
pub use gc::MemoryProvider;

#[cfg(feature = "std")]
pub use gc::LibcMemoryProvider;

#[cfg(feature = "std")]
pub use gc::get_page_size;

// Re-export FFI types for C interop
#[cfg(feature = "std")]
pub use ffi::{GcHandle, GcRootCallback, GcRootEnumerator};
