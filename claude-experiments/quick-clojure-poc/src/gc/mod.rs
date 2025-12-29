// GC module - Garbage Collection infrastructure
//
// This module provides pluggable garbage collection with multiple algorithms:
// - Mark-and-sweep (default)
// - Compacting (Cheney's two-space copying)
// - Generational (young/old generation)

use std::error::Error;

#[cfg(feature = "compacting")]
pub mod compacting;
#[cfg(feature = "generational")]
pub mod generational;
pub mod mark_and_sweep;
#[cfg(feature = "thread-safe")]
pub mod mutex_allocator;
pub mod space;
pub mod stack_walker;
pub mod types;

pub use types::{BuiltInTypes, HeapObject, Word};

/// Stack map entry details for a function
#[derive(Debug, Clone)]
pub struct StackMapDetails {
    pub function_name: Option<String>,
    pub number_of_locals: usize,
    pub current_stack_size: usize,
    pub max_stack_size: usize,
}

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

    /// Find stack data for a return address
    pub fn find_stack_data(&self, pointer: usize) -> Option<&StackMapDetails> {
        // First, try to find exact match
        for (key, value) in self.details.iter() {
            if *key == pointer {
                return Some(value);
            }
        }

        #[cfg(feature = "debug-gc")]
        {
            eprintln!("[GC DEBUG] find_stack_data: looking for {:#x} - NOT FOUND", pointer);
            eprintln!("[GC DEBUG] Stack map has {} entries", self.details.len());
            // Find closest entries to see the range
            if !self.details.is_empty() {
                let min_key = self.details.iter().map(|(k, _)| *k).min().unwrap_or(0);
                let max_key = self.details.iter().map(|(k, _)| *k).max().unwrap_or(0);
                eprintln!("[GC DEBUG] Stack map range: {:#x} to {:#x}", min_key, max_key);
                if pointer < min_key {
                    eprintln!("[GC DEBUG] Pointer is {} bytes BELOW min", min_key - pointer);
                } else if pointer > max_key {
                    eprintln!("[GC DEBUG] Pointer is {} bytes ABOVE max", pointer - max_key);
                }
            }
        }
        None
    }

    /// Extend the stack map with new entries
    pub fn extend(&mut self, translated_stack_map: Vec<(usize, StackMapDetails)>) {
        #[cfg(feature = "debug-gc")]
        {
            if !translated_stack_map.is_empty() {
                let min = translated_stack_map.iter().map(|(k, _)| *k).min().unwrap();
                let max = translated_stack_map.iter().map(|(k, _)| *k).max().unwrap();
                eprintln!("[GC DEBUG] Adding {} entries to stack map: {:#x} to {:#x}",
                    translated_stack_map.len(), min, max);
            }
        }
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
    #[allow(dead_code)] // Used by feature-gated GC implementations
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
    /// stack_info is a slice of (stack_base, frame_pointer, gc_return_addr) tuples
    fn gc(&mut self, stack_map: &StackMap, stack_info: &[(usize, usize, usize)]);

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
}

// ========== Heap Inspection Types ==========

/// Information about a heap object for inspection
#[derive(Debug, Clone)]
pub struct ObjectInfo {
    /// Untagged address of the object
    pub address: usize,
    /// Type ID from header
    pub type_id: u8,
    /// Human-readable type name
    pub type_name: String,
    /// Type-specific data (string length, deftype ID, etc.)
    pub type_data: u32,
    /// Total size in bytes including header
    pub size_bytes: usize,
    /// Number of pointer fields
    pub field_count: usize,
    /// Whether object contains raw bytes vs pointers
    pub is_opaque: bool,
    /// Preview of the object's value (for strings, var names, etc.)
    pub value_preview: Option<String>,
}

/// Detailed heap statistics
#[derive(Debug, Clone)]
pub struct DetailedHeapStats {
    /// GC algorithm name
    pub gc_algorithm: &'static str,
    /// Total heap capacity in bytes
    pub total_bytes: usize,
    /// Bytes used by live objects
    pub used_bytes: usize,
    /// Total number of live objects
    pub object_count: usize,
    /// Per-type breakdown: (type_id, type_name, count, total_bytes)
    pub objects_by_type: Vec<(u8, String, usize, usize)>,
    /// Number of free list entries (mark-and-sweep only)
    pub free_list_entries: Option<usize>,
    /// Total free bytes (mark-and-sweep only)
    pub free_bytes: Option<usize>,
    /// Largest contiguous free block (mark-and-sweep only)
    pub largest_free_block: Option<usize>,
}

/// A reference from one object to another
#[derive(Debug, Clone)]
pub struct ObjectReference {
    /// Source object address (untagged)
    pub from_address: usize,
    /// Which field holds the reference
    pub field_index: usize,
    /// The tagged pointer value
    pub tagged_value: usize,
}

/// Convert type_id to human-readable name
pub fn type_id_to_name(type_id: u8) -> &'static str {
    match type_id {
        0 => "Nil",
        1 => "Bool",
        2 => "Int",
        3 => "Float",
        4 => "String",
        5 => "Keyword",
        6 => "Symbol",
        7 => "List",
        8 => "Vector",
        9 => "Map",
        10 => "Set",
        11 => "Function",
        12 => "Closure",
        13 => "Namespace",
        14 => "Var",
        15 => "Array",
        16 => "MultiArityFn",
        17 => "DefType",
        18 => "DynamicArray",
        _ => "Unknown",
    }
}

/// Trait for heap inspection capabilities
///
/// This trait provides read-only access to heap state for debugging
/// and introspection without modifying GC algorithms.
pub trait HeapInspector {
    /// Iterate over all live objects in the heap
    fn iter_objects(&self) -> Box<dyn Iterator<Item = HeapObject> + '_>;

    /// Get detailed statistics about heap state
    fn detailed_stats(&self) -> DetailedHeapStats;

    /// Check if an address is within the managed heap
    fn contains_address(&self, addr: usize) -> bool;
}
