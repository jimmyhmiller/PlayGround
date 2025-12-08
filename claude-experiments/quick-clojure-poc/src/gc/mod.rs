// GC module - Garbage Collection infrastructure
//
// This module provides pluggable garbage collection with multiple algorithms:
// - Mark-and-sweep (default)
// - Compacting (Cheney's two-space copying)
// - Generational (young/old generation)

use std::{error::Error, thread::ThreadId};

pub mod compacting;
pub mod generational;
pub mod mark_and_sweep;
pub mod mutex_allocator;
pub mod space;
pub mod stack_walker;
pub mod types;

pub use types::{BuiltInTypes, Header, HeapObject, Word};

/// Stack map entry details for a function
#[derive(Debug, Clone)]
pub struct StackMapDetails {
    pub function_name: Option<String>,
    pub number_of_locals: usize,
    pub current_stack_size: usize,
    pub max_stack_size: usize,
}

/// Stack size constant (128 MB)
pub const STACK_SIZE: usize = 1024 * 1024 * 128;

/// Stack map for precise GC root scanning
#[derive(Debug, Clone)]
pub struct StackMap {
    details: Vec<(usize, StackMapDetails)>,
}

impl Default for StackMap {
    fn default() -> Self {
        Self::new()
    }
}

impl StackMap {
    pub fn new() -> Self {
        Self { details: vec![] }
    }

    /// Find stack data for a given instruction pointer
    /// Looks for return address minus 4 (ARM64 BL instruction width)
    pub fn find_stack_data(&self, pointer: usize) -> Option<&StackMapDetails> {
        for (key, value) in self.details.iter() {
            if *key == pointer.saturating_sub(4) {
                return Some(value);
            }
        }
        None
    }

    /// Extend the stack map with new entries
    pub fn extend(&mut self, translated_stack_map: Vec<(usize, StackMapDetails)>) {
        self.details.extend(translated_stack_map);
    }

    /// Get number of entries in the stack map
    pub fn entry_count(&self) -> usize {
        self.details.len()
    }
}

/// Options for configuring the allocator
#[derive(Debug, Clone, Copy)]
pub struct AllocatorOptions {
    pub gc: bool,
    pub print_stats: bool,
    pub gc_always: bool,
}

impl Default for AllocatorOptions {
    fn default() -> Self {
        Self {
            gc: true,
            print_stats: false,
            gc_always: false,
        }
    }
}

/// Result of an allocation attempt
pub enum AllocateAction {
    /// Allocation succeeded, returns pointer to allocated memory
    Allocated(*const u8),
    /// Allocation failed, GC needed
    Gc,
}

/// Trait for garbage collector implementations
pub trait Allocator {
    /// Create a new allocator with the given options
    fn new(options: AllocatorOptions) -> Self;

    /// Try to allocate memory for an object
    fn try_allocate(
        &mut self,
        words: usize,
        kind: BuiltInTypes,
    ) -> Result<AllocateAction, Box<dyn Error>>;

    /// Run garbage collection
    fn gc(&mut self, stack_map: &StackMap, stack_pointers: &[(usize, usize)]);

    /// Grow the heap
    fn grow(&mut self);

    /// Add a root for write barrier (generational GC)
    fn gc_add_root(&mut self, old: usize);

    /// Register a temporary root
    fn register_temporary_root(&mut self, root: usize) -> usize;

    /// Unregister a temporary root
    fn unregister_temporary_root(&mut self, id: usize) -> usize;

    /// Add a namespace root
    fn add_namespace_root(&mut self, namespace_id: usize, root: usize);

    /// Remove a namespace root
    fn remove_namespace_root(&mut self, namespace_id: usize, root: usize) -> bool;

    /// Get namespace relocations (for compacting/generational GC)
    fn get_namespace_relocations(&mut self) -> Vec<(usize, Vec<(usize, usize)>)>;

    /// Get pause pointer (for thread synchronization)
    #[allow(unused)]
    fn get_pause_pointer(&self) -> usize {
        0
    }

    /// Register a thread for multi-threaded allocation
    fn register_thread(&mut self, _thread_id: ThreadId) {}

    /// Remove a thread registration
    fn remove_thread(&mut self, _thread_id: ThreadId) {}

    /// Register a parked thread
    fn register_parked_thread(&mut self, _thread_id: ThreadId, _stack_pointer: usize) {}

    /// Get allocation options
    fn get_allocation_options(&self) -> AllocatorOptions;
}
