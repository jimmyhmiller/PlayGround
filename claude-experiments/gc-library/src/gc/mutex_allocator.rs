//! Thread-safe allocator wrapper.
//!
//! Wraps any `Allocator` implementation with a mutex for thread safety.

use std::{
    error::Error,
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use crate::traits::{GcTypes, RootProvider};

use super::{AllocateAction, Allocator, AllocatorOptions};

/// Thread-safe wrapper for any allocator.
///
/// Uses a mutex to synchronize allocation and GC operations.
///
/// # Type Parameters
/// - `Alloc`: The underlying allocator type
/// - `T`: The runtime's type system implementing [`GcTypes`]
pub struct MutexAllocator<Alloc, T>
where
    T: GcTypes,
    Alloc: Allocator<T>,
{
    alloc: Alloc,
    mutex: Mutex<()>,
    options: AllocatorOptions,
    registered_threads: AtomicUsize,
    _phantom: std::marker::PhantomData<T>,
}

impl<Alloc, T> MutexAllocator<Alloc, T>
where
    T: GcTypes,
    Alloc: Allocator<T>,
{
    /// Execute a function with exclusive access to the underlying allocator.
    #[cfg(feature = "thread-safe")]
    pub fn with_locked_alloc<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Alloc) -> R,
    {
        let _lock = self.mutex.lock().unwrap();
        f(&mut self.alloc)
    }
}

impl<Alloc, T> Allocator<T> for MutexAllocator<Alloc, T>
where
    T: GcTypes,
    Alloc: Allocator<T>,
{
    fn new(options: AllocatorOptions) -> Self {
        MutexAllocator {
            alloc: Alloc::new(options),
            mutex: Mutex::new(()),
            options,
            registered_threads: AtomicUsize::new(0),
            _phantom: std::marker::PhantomData,
        }
    }

    fn try_allocate(
        &mut self,
        bytes: usize,
        kind: T::ObjectKind,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        if self.registered_threads.load(Ordering::Acquire) == 0 {
            return self.alloc.try_allocate(bytes, kind);
        }

        let lock = self.mutex.lock().unwrap();
        let result = self.alloc.try_allocate(bytes, kind);
        drop(lock);
        result
    }

    fn gc(&mut self, roots: &dyn RootProvider<T::TaggedValue>) {
        if self.registered_threads.load(Ordering::Acquire) == 0 {
            return self.alloc.gc(roots);
        }
        let lock = self.mutex.lock().unwrap();
        self.alloc.gc(roots);
        drop(lock)
    }

    fn grow(&mut self) {
        if self.registered_threads.load(Ordering::Acquire) == 0 {
            return self.alloc.grow();
        }
        let lock = self.mutex.lock().unwrap();
        self.alloc.grow();
        drop(lock)
    }

    fn get_allocation_options(&self) -> AllocatorOptions {
        self.options
    }

    fn register_thread(&mut self, _thread_id: std::thread::ThreadId) {
        self.registered_threads.fetch_add(1, Ordering::AcqRel);
    }

    fn remove_thread(&mut self, _thread_id: std::thread::ThreadId) {
        let previous = self.registered_threads.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(
            previous > 0,
            "remove_thread called with no registered threads"
        );
    }

    fn write_barrier(&mut self, object_ptr: usize, new_value: usize) {
        // No lock needed - write barrier is called after the write has happened,
        // and the remembered set is only read during GC (which holds the lock).
        self.alloc.write_barrier(object_ptr, new_value);
    }

    fn get_card_table_biased_ptr(&self) -> *mut u8 {
        // Delegate to inner allocator to get the actual card table pointer
        self.alloc.get_card_table_biased_ptr()
    }

    fn mark_card_unconditional(&mut self, object_ptr: usize) {
        // Delegate to inner allocator to mark the card
        // No lock needed - card marking is atomic and only read during GC (which holds the lock)
        self.alloc.mark_card_unconditional(object_ptr);
    }
}
