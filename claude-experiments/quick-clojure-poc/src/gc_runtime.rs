// GC Runtime - High-level runtime using pluggable GC allocators
//
// This module provides the high-level runtime interface for managing
// heap-allocated objects (namespaces, vars, functions, deftypes) using
// the pluggable GC system from the gc module.

use std::collections::{HashMap, HashSet};

use crate::gc::{
    Allocator, AllocateAction, AllocatorOptions, StackMap, StackMapDetails,
    BuiltInTypes, HeapObject, Word, HeapInspector, DetailedHeapStats, ObjectInfo,
    ObjectReference, type_id_to_name,
};

// Re-export for compatibility
pub use crate::gc::{BuiltInTypes as GcBuiltInTypes, HeapObject as GcHeapObject};

// Select allocator based on feature flags
cfg_if::cfg_if! {
    if #[cfg(feature = "compacting")] {
        use crate::gc::compacting::CompactingHeap as AllocImpl;
    } else if #[cfg(feature = "generational")] {
        use crate::gc::generational::GenerationalGC as AllocImpl;
    } else {
        use crate::gc::mark_and_sweep::MarkAndSweep as AllocImpl;
    }
}

#[cfg(feature = "thread-safe")]
type Alloc = crate::gc::mutex_allocator::MutexAllocator<AllocImpl>;

#[cfg(not(feature = "thread-safe"))]
type Alloc = AllocImpl;

const TYPE_ID_STRING: u8 = 2;
const TYPE_ID_NAMESPACE: u8 = 10;
const TYPE_ID_VAR: u8 = 11;
const TYPE_ID_FUNCTION: u8 = 12;
const TYPE_ID_DEFTYPE: u8 = 13;
const TYPE_ID_MULTI_ARITY_FN: u8 = 14;
const TYPE_ID_CONS: u8 = 15;

// ========== Built-in Type IDs for Protocol Dispatch ==========
// These are used by the protocol vtable to dispatch on type.
// Tagged primitives use their tag-derived ID, heap objects use these.

pub const BUILTIN_TYPE_NIL: usize = 0;
pub const BUILTIN_TYPE_BOOL: usize = 1;
pub const BUILTIN_TYPE_INT: usize = 2;
pub const BUILTIN_TYPE_FLOAT: usize = 3;
pub const BUILTIN_TYPE_STRING: usize = 4;
pub const BUILTIN_TYPE_KEYWORD: usize = 5;
pub const BUILTIN_TYPE_SYMBOL: usize = 6;
pub const BUILTIN_TYPE_LIST: usize = 7;
pub const BUILTIN_TYPE_VECTOR: usize = 8;
pub const BUILTIN_TYPE_MAP: usize = 9;
pub const BUILTIN_TYPE_SET: usize = 10;
pub const BUILTIN_TYPE_FUNCTION: usize = 11;
pub const BUILTIN_TYPE_CLOSURE: usize = 12;
pub const BUILTIN_TYPE_NAMESPACE: usize = 13;
pub const BUILTIN_TYPE_VAR: usize = 14;

/// Offset added to deftype IDs to avoid collision with built-in types
pub const DEFTYPE_ID_OFFSET: usize = 100;

/// Definition of a deftype (name and field names)
#[derive(Debug, Clone)]
pub struct TypeDef {
    pub name: String,
    pub fields: Vec<String>,
}

impl TypeDef {
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    pub fn field_index(&self, field_name: &str) -> Option<usize> {
        self.fields.iter().position(|f| f == field_name)
    }
}

// ========== Protocol Definitions ==========

/// Definition of a protocol (name and method signatures)
#[derive(Debug, Clone)]
pub struct ProtocolDef {
    pub name: String,
    pub methods: Vec<ProtocolMethod>,
}

/// Method signature in a protocol
#[derive(Debug, Clone)]
pub struct ProtocolMethod {
    pub name: String,
    /// Supported arities (number of args including 'this')
    pub arities: Vec<usize>,
}

impl ProtocolDef {
    pub fn method_count(&self) -> usize {
        self.methods.len()
    }

    pub fn method_index(&self, method_name: &str) -> Option<usize> {
        self.methods.iter().position(|m| m.name == method_name)
    }
}

/// Exception handler saved state
/// When a try block is entered, we save the current SP/FP/LR so throw can restore them
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ExceptionHandler {
    pub handler_address: usize,   // Label address to jump to (catch block)
    pub stack_pointer: usize,     // Saved SP
    pub frame_pointer: usize,     // Saved FP (x29)
    pub link_register: usize,     // Saved LR (x30)
    pub result_local: isize,      // Where to store exception (FP-relative offset, negative)
}

/// Closure heap object layout constants (single-arity functions)
#[allow(dead_code)]
pub mod closure_layout {
    pub const HEADER_SIZE: usize = 8;
    pub const FIELD_0_NAME_PTR: usize = 8;
    pub const FIELD_1_CODE_PTR: usize = 16;
    pub const FIELD_2_CLOSURE_COUNT: usize = 24;
    pub const FIELD_3_FIRST_VALUE: usize = 32;
    pub const VALUES_OFFSET: usize = FIELD_3_FIRST_VALUE;
    pub const VALUE_SIZE: usize = 8;

    pub const fn value_offset(index: usize) -> usize {
        VALUES_OFFSET + (index * VALUE_SIZE)
    }

    pub const FIELD_NAME_PTR: usize = 0;
    pub const FIELD_CODE_PTR: usize = 1;
    pub const FIELD_CLOSURE_COUNT: usize = 2;
    pub const FIELD_FIRST_VALUE: usize = 3;
}

/// Multi-arity function heap object layout constants
/// Layout:
///   [header(8)]
///   [name_ptr(8)]           - field 0: Optional function name (0 if anonymous)
///   [arity_count(8)]        - field 1: Number of arities
///   [variadic_min(8)]       - field 2: Min args for variadic arity (i64::MAX if none)
///   [closure_count(8)]      - field 3: Number of closure values
///   [arity_table...]        - fields 4+: Array of (param_count, code_ptr) pairs
///   [closure_values...]     - After arity table: captured closure values
#[allow(dead_code)]
pub mod multi_arity_layout {
    pub const HEADER_SIZE: usize = 8;

    // Field indices (word offsets from start of object data, after header)
    pub const FIELD_NAME_PTR: usize = 0;
    pub const FIELD_ARITY_COUNT: usize = 1;
    pub const FIELD_VARIADIC_MIN: usize = 2;
    pub const FIELD_CLOSURE_COUNT: usize = 3;
    pub const ARITY_TABLE_START: usize = 4;

    // Each arity entry is 2 words: (param_count, code_ptr)
    pub const ARITY_ENTRY_SIZE: usize = 2;

    /// Get the field index of an arity entry's param_count
    pub const fn arity_param_count_field(arity_index: usize) -> usize {
        ARITY_TABLE_START + arity_index * ARITY_ENTRY_SIZE
    }

    /// Get the field index of an arity entry's code_ptr
    pub const fn arity_code_ptr_field(arity_index: usize) -> usize {
        ARITY_TABLE_START + arity_index * ARITY_ENTRY_SIZE + 1
    }

    /// Get the field index of the first closure value
    pub const fn closure_values_start(arity_count: usize) -> usize {
        ARITY_TABLE_START + arity_count * ARITY_ENTRY_SIZE
    }

    /// Get the field index of a specific closure value
    pub const fn closure_value_field(arity_count: usize, closure_index: usize) -> usize {
        closure_values_start(arity_count) + closure_index
    }

    /// Calculate total object size in words (not including header)
    pub const fn total_size_words(arity_count: usize, closure_count: usize) -> usize {
        4 + arity_count * ARITY_ENTRY_SIZE + closure_count
    }

    /// Sentinel value for "no variadic arity"
    pub const NO_VARIADIC: usize = usize::MAX;
}

/// Cons cell layout for lists (used for variadic args)
/// Layout:
///   [header(8)]
///   [head(8)]     - field 0: First element (tagged value)
///   [tail(8)]     - field 1: Rest of list (tagged cons or nil)
#[allow(dead_code)]
pub mod cons_layout {
    pub const FIELD_HEAD: usize = 0;
    pub const FIELD_TAIL: usize = 1;
    pub const SIZE_WORDS: usize = 2;
}

/// GC Runtime with pluggable allocator
pub struct GCRuntime {
    /// The underlying allocator
    allocator: Alloc,

    /// Stack map for precise GC root scanning
    stack_map: StackMap,

    /// Stack base (top of JIT stack) for GC root scanning
    stack_base: usize,

    /// Namespace roots: namespace_name -> tagged pointer
    namespace_roots: HashMap<String, usize>,

    /// Namespace name to numeric ID (for allocator)
    namespace_name_to_id: HashMap<String, usize>,
    next_namespace_id: usize,

    /// Thread-local binding stacks for dynamic vars
    dynamic_bindings: HashMap<usize, Vec<usize>>,

    /// Set of vars that are marked as dynamic
    dynamic_vars: HashSet<usize>,

    /// Type registry for deftype
    type_registry: Vec<TypeDef>,

    /// Type name to ID mapping
    type_name_to_id: HashMap<String, usize>,

    /// GC options
    options: AllocatorOptions,

    /// Stack of exception handlers for try/catch
    exception_handlers: Vec<ExceptionHandler>,

    // ========== Protocol System ==========

    /// Protocol registry: indexed by protocol_id
    protocol_registry: Vec<ProtocolDef>,

    /// Protocol name to ID mapping (fully qualified name -> protocol_id)
    protocol_name_to_id: HashMap<String, usize>,

    /// Global vtable for protocol dispatch
    /// Key: (type_id, protocol_id, method_index) -> function pointer (tagged closure)
    protocol_vtable: HashMap<(usize, usize, usize), usize>,

    /// Reverse lookup: method_name -> (protocol_id, method_index)
    /// Used for dispatch when we only know the method name
    method_to_protocol: HashMap<String, (usize, usize)>,
}

impl Default for GCRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl GCRuntime {
    pub fn new() -> Self {
        Self::with_options(AllocatorOptions::default())
    }

    pub fn with_options(options: AllocatorOptions) -> Self {
        GCRuntime {
            allocator: Alloc::new(options),
            stack_map: StackMap::new(),
            stack_base: 0,  // Set by set_stack_base() before running JIT code
            namespace_roots: HashMap::new(),
            namespace_name_to_id: HashMap::new(),
            next_namespace_id: 0,
            dynamic_bindings: HashMap::new(),
            dynamic_vars: HashSet::new(),
            type_registry: Vec::new(),
            type_name_to_id: HashMap::new(),
            options,
            exception_handlers: Vec::new(),
            // Protocol system
            protocol_registry: Vec::new(),
            protocol_name_to_id: HashMap::new(),
            protocol_vtable: HashMap::new(),
            method_to_protocol: HashMap::new(),
        }
    }

    /// Set the stack base (top of JIT stack) for GC root scanning
    pub fn set_stack_base(&mut self, stack_base: usize) {
        self.stack_base = stack_base;
    }

    /// Get the stack base
    pub fn get_stack_base(&self) -> usize {
        self.stack_base
    }

    /// Enable or disable gc_always mode (GC before every allocation)
    pub fn set_gc_always(&mut self, enabled: bool) {
        self.options.gc_always = enabled;
    }

    /// Check if gc_always mode is enabled
    pub fn gc_always(&self) -> bool {
        self.options.gc_always
    }

    /// Run GC if gc_always is enabled (called before allocations)
    pub fn maybe_gc_before_alloc(&mut self, stack_pointer: usize) {
        if self.options.gc_always && self.stack_base != 0 {
            self.gc(stack_pointer);
        }
    }

    /// Run GC with just the current stack pointer
    /// Uses the stored stack_base
    pub fn gc(&mut self, stack_pointer: usize) {
        if self.stack_base == 0 {
            // Stack base not set, can't run GC
            return;
        }
        self.allocator.gc(&self.stack_map, &[(self.stack_base, stack_pointer)]);

        // Handle relocations
        let relocations = self.allocator.get_namespace_relocations();
        for (_ns_id, updates) in relocations {
            for (old_ptr, new_ptr) in updates {
                for (_name, ptr) in self.namespace_roots.iter_mut() {
                    if *ptr == old_ptr {
                        *ptr = new_ptr;
                    }
                }
            }
        }
    }

    /// Add stack map entry
    pub fn add_stack_map_entry(&mut self, code_addr: usize, details: StackMapDetails) {
        self.stack_map.extend(vec![(code_addr, details)]);
    }

    /// Get the stack map (for external use)
    pub fn stack_map(&self) -> &StackMap {
        &self.stack_map
    }

    /// Allocate raw memory from the heap
    fn allocate_raw(&mut self, size_words: usize, type_id: u8) -> Result<usize, String> {
        // Try to allocate
        match self.allocator.try_allocate(size_words, BuiltInTypes::HeapObject) {
            Ok(AllocateAction::Allocated(ptr)) => {
                let mut heap_obj = HeapObject::from_untagged(ptr);
                heap_obj.write_header(Word::from_word(size_words));
                heap_obj.write_type_id(type_id as usize);
                Ok(ptr as usize)
            }
            Ok(AllocateAction::Gc) => {
                // Need GC - for now just grow and retry
                // TODO: Actually run GC with stack map when we have stack pointers
                self.allocator.grow();
                self.allocate_raw(size_words, type_id)
            }
            Err(e) => Err(e.to_string()),
        }
    }

    /// Allocate a string on the heap
    pub fn allocate_string(&mut self, s: &str) -> Result<usize, String> {
        let bytes = s.as_bytes();
        let words = bytes.len().div_ceil(8);

        let ptr = self.allocate_raw(words, TYPE_ID_STRING)?;

        let mut heap_obj = HeapObject::from_untagged(ptr as *const u8);

        // Update header with byte length and mark as opaque
        let mut header = heap_obj.get_header();
        header.type_data = bytes.len() as u32;
        header.opaque = true;
        heap_obj.write_header_direct(header);

        // Write string data
        unsafe {
            let data_ptr = (ptr + 8) as *mut u8;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), data_ptr, bytes.len());
        }

        Ok(self.tag_string(ptr))
    }

    /// Tag a pointer as a string
    fn tag_string(&self, ptr: usize) -> usize {
        (ptr << 3) | 0b010
    }

    /// Tag a pointer as a heap object
    fn tag_heap_object(&self, ptr: usize) -> usize {
        (ptr << 3) | 0b110
    }

    /// Tag a pointer as a closure
    fn tag_closure(&self, ptr: usize) -> usize {
        (ptr << 3) | 0b101
    }

    /// Untag a string pointer
    fn untag_string(&self, tagged: usize) -> usize {
        tagged >> 3
    }

    /// Untag a heap object pointer
    fn untag_heap_object(&self, tagged: usize) -> usize {
        tagged >> 3
    }

    /// Untag a closure pointer
    fn untag_closure(&self, tagged: usize) -> usize {
        tagged >> 3
    }

    /// Allocate a namespace object on the heap
    pub fn allocate_namespace(&mut self, name: &str) -> Result<usize, String> {
        let name_ptr = self.allocate_string(name)?;
        let size_words = 1;
        let ns_ptr = self.allocate_raw(size_words, TYPE_ID_NAMESPACE)?;

        let heap_obj = HeapObject::from_untagged(ns_ptr as *const u8);
        heap_obj.write_field(0, name_ptr as usize);

        Ok(self.tag_heap_object(ns_ptr))
    }

    /// Add or update a binding in a namespace (may reallocate!)
    pub fn namespace_add_binding(
        &mut self,
        ns_ptr: usize,
        symbol_name: &str,
        value: usize,
    ) -> Result<usize, String> {
        let ns_untagged = self.untag_heap_object(ns_ptr);
        let heap_obj = HeapObject::from_untagged(ns_untagged as *const u8);
        let header = heap_obj.get_header();
        let current_size = header.size as usize;

        // Check if binding already exists
        if current_size > 0 {
            let num_bindings = (current_size - 1) / 2;
            for i in 0..num_bindings {
                let name_ptr = heap_obj.get_field(1 + i * 2);
                let stored_name = self.read_string(name_ptr);
                if stored_name == symbol_name {
                    heap_obj.write_field(1 + i * 2 + 1, value as usize);
                    return Ok(ns_ptr);
                }
            }
        }

        // Allocate symbol name string
        let symbol_ptr = self.allocate_string(symbol_name)?;

        // Reallocate with +2 words
        let new_size = current_size + 2;
        let new_ns_ptr = self.allocate_raw(new_size, TYPE_ID_NAMESPACE)?;

        // Copy existing fields
        let new_heap_obj = HeapObject::from_untagged(new_ns_ptr as *const u8);
        for i in 0..current_size {
            let field = heap_obj.get_field(i);
            new_heap_obj.write_field(i, field);
        }

        // Add new binding
        new_heap_obj.write_field(current_size, symbol_ptr);
        new_heap_obj.write_field(current_size + 1, value);

        let tagged_new_ptr = self.tag_heap_object(new_ns_ptr);

        // Update GC root since namespace was reallocated
        let name = self.namespace_name(tagged_new_ptr);
        self.update_namespace_root(&name, tagged_new_ptr);

        Ok(tagged_new_ptr)
    }

    /// Look up a binding in a namespace
    pub fn namespace_lookup(&self, ns_ptr: usize, symbol_name: &str) -> Option<usize> {
        let ns_untagged = self.untag_heap_object(ns_ptr);
        let heap_obj = HeapObject::from_untagged(ns_untagged as *const u8);
        let header = heap_obj.get_header();
        let size = header.size as usize;

        if size == 0 {
            return None;
        }

        let num_bindings = (size - 1) / 2;

        for i in 0..num_bindings {
            let name_ptr = heap_obj.get_field(1 + i * 2);
            let stored_name = self.read_string(name_ptr);
            if stored_name == symbol_name {
                return Some(heap_obj.get_field(1 + i * 2 + 1));
            }
        }
        None
    }

    /// Get namespace name
    pub fn namespace_name(&self, ns_ptr: usize) -> String {
        let ns_untagged = self.untag_heap_object(ns_ptr);
        let heap_obj = HeapObject::from_untagged(ns_untagged as *const u8);
        let name_ptr = heap_obj.get_field(0);
        self.read_string(name_ptr)
    }

    /// Register namespace as GC root
    pub fn add_namespace_root(&mut self, name: String, ns_ptr: usize) {
        let ns_id = self.next_namespace_id;
        self.next_namespace_id += 1;

        self.namespace_name_to_id.insert(name.clone(), ns_id);
        self.namespace_roots.insert(name, ns_ptr);
        self.allocator.add_namespace_root(ns_id, ns_ptr);
    }

    /// Update namespace root (after reallocation)
    pub fn update_namespace_root(&mut self, name: &str, new_ptr: usize) {
        if let Some(&ns_id) = self.namespace_name_to_id.get(name) {
            if let Some(old_ptr) = self.namespace_roots.get(name).copied() {
                self.allocator.remove_namespace_root(ns_id, old_ptr);
                self.allocator.add_namespace_root(ns_id, new_ptr);
            }
        }
        self.namespace_roots.insert(name.to_string(), new_ptr);
    }

    /// Get all namespace pointers (for syncing compiler registry after GC)
    pub fn get_namespace_pointers(&self) -> &HashMap<String, usize> {
        &self.namespace_roots
    }

    /// Read a string from a tagged pointer
    pub fn read_string(&self, tagged_ptr: usize) -> String {
        let ptr = self.untag_string(tagged_ptr);
        let heap_obj = HeapObject::from_untagged(ptr as *const u8);
        let header = heap_obj.get_header();
        let byte_len = header.type_data as usize;

        unsafe {
            let data_ptr = (ptr + 8) as *const u8;
            let bytes = std::slice::from_raw_parts(data_ptr, byte_len);
            String::from_utf8_unchecked(bytes.to_vec())
        }
    }

    /// Run garbage collection
    pub fn run_gc(&mut self) -> Result<(), String> {
        // For now, just return - full GC requires stack pointers
        // TODO: Implement proper GC trigger with stack walking
        Ok(())
    }

    /// Run GC with stack pointers
    pub fn run_gc_with_stack(&mut self, stack_base: usize, stack_pointer: usize) {
        self.allocator.gc(&self.stack_map, &[(stack_base, stack_pointer)]);

        // Handle relocations
        let relocations = self.allocator.get_namespace_relocations();
        for (ns_id, updates) in relocations {
            for (old_ptr, new_ptr) in updates {
                // Find and update namespace_roots
                for (name, ptr) in self.namespace_roots.iter_mut() {
                    if *ptr == old_ptr {
                        *ptr = new_ptr;
                        // Also need to find ns_id by name for update
                        if let Some(&id) = self.namespace_name_to_id.get(name) {
                            if id == ns_id {
                                // Already updated the root
                            }
                        }
                    }
                }
            }
        }
    }

    /// Get heap statistics
    pub fn heap_stats(&self) -> HeapStats {
        HeapStats {
            gc_algorithm: if cfg!(feature = "compacting") {
                "compacting"
            } else if cfg!(feature = "generational") {
                "generational"
            } else {
                "mark-and-sweep"
            }.to_string(),
            namespace_count: self.namespace_roots.len(),
            type_count: self.type_registry.len(),
            stack_map_entries: self.stack_map.entry_count(),
        }
    }

    /// List all namespaces
    pub fn list_namespaces(&self) -> Vec<(String, usize, usize)> {
        let mut namespaces = Vec::new();
        for (name, &ptr) in &self.namespace_roots {
            let ns_untagged = self.untag_heap_object(ptr);
            let heap_obj = HeapObject::from_untagged(ns_untagged as *const u8);
            let header = heap_obj.get_header();
            let num_bindings = if header.size > 0 {
                (header.size as usize - 1) / 2
            } else {
                0
            };
            namespaces.push((name.clone(), ptr, num_bindings));
        }
        namespaces.sort_by(|a, b| a.0.cmp(&b.0));
        namespaces
    }

    /// Get bindings in a namespace
    pub fn namespace_bindings(&self, ns_ptr: usize) -> Vec<(String, usize)> {
        let ns_untagged = self.untag_heap_object(ns_ptr);
        let heap_obj = HeapObject::from_untagged(ns_untagged as *const u8);
        let header = heap_obj.get_header();
        let size = header.size as usize;

        if size == 0 {
            return Vec::new();
        }

        let num_bindings = (size - 1) / 2;
        let mut bindings = Vec::new();

        for i in 0..num_bindings {
            let name_ptr = heap_obj.get_field(1 + i * 2);
            let value = heap_obj.get_field(1 + i * 2 + 1);
            let name = self.read_string(name_ptr);
            bindings.push((name, value));
        }

        bindings
    }

    // ========== Var Methods ==========

    /// Allocate a var object
    pub fn allocate_var(
        &mut self,
        ns_ptr: usize,
        symbol_name: &str,
        initial_value: usize,
    ) -> Result<usize, String> {
        let symbol_ptr = self.allocate_string(symbol_name)?;

        let var_ptr = self.allocate_raw(3, TYPE_ID_VAR)?;
        let heap_obj = HeapObject::from_untagged(var_ptr as *const u8);
        heap_obj.write_field(0, ns_ptr);
        heap_obj.write_field(1, symbol_ptr);
        heap_obj.write_field(2, initial_value);

        Ok(self.tag_heap_object(var_ptr))
    }

    /// Get the current value from a var
    pub fn var_get_value(&self, var_ptr: usize) -> usize {
        let untagged = self.untag_heap_object(var_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_field(2)
    }

    /// Mark a var as dynamic
    pub fn mark_var_dynamic(&mut self, var_ptr: usize) {
        self.dynamic_vars.insert(var_ptr);
    }

    /// Check if a var is dynamic
    pub fn is_var_dynamic(&self, var_ptr: usize) -> bool {
        self.dynamic_vars.contains(&var_ptr)
    }

    /// Push a thread-local binding for a dynamic var
    pub fn push_binding(&mut self, var_ptr: usize, value: usize) -> Result<(), String> {
        if !self.is_var_dynamic(var_ptr) {
            let (ns_name, symbol_name) = self.var_info(var_ptr);
            return Err(format!(
                "Can't dynamically bind non-dynamic var: {}/{}",
                ns_name, symbol_name
            ));
        }

        self.dynamic_bindings
            .entry(var_ptr)
            .or_default()
            .push(value);

        Ok(())
    }

    /// Pop a thread-local binding for a dynamic var
    pub fn pop_binding(&mut self, var_ptr: usize) -> Result<(), String> {
        if let Some(stack) = self.dynamic_bindings.get_mut(&var_ptr) {
            if stack.is_empty() {
                return Err(format!("No bindings to pop for var: {}", var_ptr));
            }
            stack.pop();
            Ok(())
        } else {
            Err(format!("No binding stack for var: {}", var_ptr))
        }
    }

    /// Set the value of a thread-local binding (for set!)
    pub fn set_binding(&mut self, var_ptr: usize, value: usize) -> Result<(), String> {
        if let Some(stack) = self.dynamic_bindings.get_mut(&var_ptr) {
            if let Some(last) = stack.last_mut() {
                *last = value;
                return Ok(());
            }
        }

        let (ns_name, symbol_name) = self.var_info(var_ptr);
        Err(format!(
            "Can't change/establish root binding of: {}/{} with set",
            ns_name, symbol_name
        ))
    }

    /// Get the current value of a var, checking dynamic bindings first
    pub fn var_get_value_dynamic(&self, var_ptr: usize) -> usize {
        if let Some(stack) = self.dynamic_bindings.get(&var_ptr) {
            if let Some(&value) = stack.last() {
                return value;
            }
        }
        self.var_get_value(var_ptr)
    }

    /// Get var namespace and symbol name
    pub fn var_info(&self, var_ptr: usize) -> (String, String) {
        let untagged = self.untag_heap_object(var_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);

        let ns_ptr = heap_obj.get_field(0);
        let ns_name = self.namespace_name(ns_ptr);

        let symbol_ptr = heap_obj.get_field(1);
        let symbol_name = self.read_string(symbol_ptr);

        (ns_name, symbol_name)
    }

    // ========== Function Methods ==========

    /// Allocate a function object on the heap
    pub fn allocate_function(
        &mut self,
        name: Option<String>,
        code_ptr: usize,
        closure_values: Vec<usize>,
    ) -> Result<usize, String> {
        let name_ptr = if let Some(n) = name {
            self.allocate_string(&n)?
        } else {
            0
        };

        let size_words = 3 + closure_values.len();
        let fn_ptr = self.allocate_raw(size_words, TYPE_ID_FUNCTION)?;
        let heap_obj = HeapObject::from_untagged(fn_ptr as *const u8);

        heap_obj.write_field(closure_layout::FIELD_NAME_PTR, name_ptr);
        heap_obj.write_field(closure_layout::FIELD_CODE_PTR, code_ptr);
        heap_obj.write_field(closure_layout::FIELD_CLOSURE_COUNT, closure_values.len());

        for (i, value) in closure_values.iter().enumerate() {
            heap_obj.write_field(closure_layout::FIELD_FIRST_VALUE + i, *value);
        }

        Ok(self.tag_closure(fn_ptr))
    }

    /// Get function code pointer from closure
    pub fn function_code_ptr(&self, fn_ptr: usize) -> usize {
        let untagged = self.untag_closure(fn_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_field(closure_layout::FIELD_CODE_PTR)
    }

    /// Get closure count from closure
    pub fn function_closure_count(&self, fn_ptr: usize) -> usize {
        let untagged = self.untag_closure(fn_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_field(closure_layout::FIELD_CLOSURE_COUNT)
    }

    /// Get closure value by index from closure
    pub fn function_get_closure(&self, fn_ptr: usize, index: usize) -> usize {
        let untagged = self.untag_closure(fn_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        let closure_count = heap_obj.get_field(closure_layout::FIELD_CLOSURE_COUNT);
        if index >= closure_count {
            panic!("Closure index out of bounds: {} >= {}", index, closure_count);
        }
        heap_obj.get_field(closure_layout::FIELD_FIRST_VALUE + index)
    }

    /// Get function name
    #[allow(dead_code)]
    pub fn function_name(&self, fn_ptr: usize) -> String {
        let untagged = self.untag_heap_object(fn_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        let name_ptr = heap_obj.get_field(0);
        if name_ptr == 0 {
            "anonymous".to_string()
        } else {
            self.read_string(name_ptr)
        }
    }

    // ========== Multi-Arity Function Methods ==========

    /// Allocate a multi-arity function object on the heap
    ///
    /// arities: Vec of (param_count, code_ptr) pairs for each arity
    /// variadic_min: If Some(n), the function has a variadic arity accepting n+ args
    /// closure_values: Captured closure values (shared across all arities)
    pub fn allocate_multi_arity_function(
        &mut self,
        name: Option<String>,
        arities: Vec<(usize, usize)>,
        variadic_min: Option<usize>,
        closure_values: Vec<usize>,
    ) -> Result<usize, String> {
        let name_ptr = if let Some(n) = name {
            self.allocate_string(&n)?
        } else {
            0
        };

        let arity_count = arities.len();
        let size_words = multi_arity_layout::total_size_words(arity_count, closure_values.len());
        let fn_ptr = self.allocate_raw(size_words, TYPE_ID_MULTI_ARITY_FN)?;
        let heap_obj = HeapObject::from_untagged(fn_ptr as *const u8);

        // Write header fields
        heap_obj.write_field(multi_arity_layout::FIELD_NAME_PTR, name_ptr);
        heap_obj.write_field(multi_arity_layout::FIELD_ARITY_COUNT, arity_count);
        heap_obj.write_field(
            multi_arity_layout::FIELD_VARIADIC_MIN,
            variadic_min.unwrap_or(multi_arity_layout::NO_VARIADIC),
        );
        heap_obj.write_field(multi_arity_layout::FIELD_CLOSURE_COUNT, closure_values.len());

        // Write arity table
        for (i, (param_count, code_ptr)) in arities.iter().enumerate() {
            heap_obj.write_field(multi_arity_layout::arity_param_count_field(i), *param_count);
            heap_obj.write_field(multi_arity_layout::arity_code_ptr_field(i), *code_ptr);
        }

        // Write closure values
        for (i, value) in closure_values.iter().enumerate() {
            heap_obj.write_field(
                multi_arity_layout::closure_value_field(arity_count, i),
                *value,
            );
        }

        Ok(self.tag_closure(fn_ptr))
    }

    /// Check if a function is multi-arity (vs single-arity)
    pub fn is_multi_arity_function(&self, fn_ptr: usize) -> bool {
        let untagged = self.untag_closure(fn_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_header().type_id == TYPE_ID_MULTI_ARITY_FN
    }

    /// Look up the code pointer for a given argument count in a multi-arity function
    /// Returns (code_ptr, is_variadic) if found
    pub fn multi_arity_lookup(&self, fn_ptr: usize, arg_count: usize) -> Option<(usize, bool)> {
        let untagged = self.untag_closure(fn_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);

        let arity_count = heap_obj.get_field(multi_arity_layout::FIELD_ARITY_COUNT);
        let variadic_min = heap_obj.get_field(multi_arity_layout::FIELD_VARIADIC_MIN);

        // First, try to find an exact match
        for i in 0..arity_count {
            let param_count = heap_obj.get_field(multi_arity_layout::arity_param_count_field(i));
            if param_count == arg_count {
                let code_ptr = heap_obj.get_field(multi_arity_layout::arity_code_ptr_field(i));
                return Some((code_ptr, false));
            }
        }

        // If no exact match and we have a variadic arity, check if args >= variadic_min
        if variadic_min != multi_arity_layout::NO_VARIADIC && arg_count >= variadic_min {
            // Find the variadic arity (the one with param_count == variadic_min)
            for i in 0..arity_count {
                let param_count = heap_obj.get_field(multi_arity_layout::arity_param_count_field(i));
                if param_count == variadic_min {
                    let code_ptr = heap_obj.get_field(multi_arity_layout::arity_code_ptr_field(i));
                    return Some((code_ptr, true));
                }
            }
        }

        None
    }

    /// Get closure count from a multi-arity function
    pub fn multi_arity_closure_count(&self, fn_ptr: usize) -> usize {
        let untagged = self.untag_closure(fn_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_field(multi_arity_layout::FIELD_CLOSURE_COUNT)
    }

    /// Get arity count from a multi-arity function
    pub fn multi_arity_arity_count(&self, fn_ptr: usize) -> usize {
        let untagged = self.untag_closure(fn_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_field(multi_arity_layout::FIELD_ARITY_COUNT)
    }

    /// Get closure value by index from a multi-arity function
    pub fn multi_arity_get_closure(&self, fn_ptr: usize, index: usize) -> usize {
        let untagged = self.untag_closure(fn_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        let arity_count = heap_obj.get_field(multi_arity_layout::FIELD_ARITY_COUNT);
        let closure_count = heap_obj.get_field(multi_arity_layout::FIELD_CLOSURE_COUNT);
        if index >= closure_count {
            panic!("Multi-arity closure index out of bounds: {} >= {}", index, closure_count);
        }
        heap_obj.get_field(multi_arity_layout::closure_value_field(arity_count, index))
    }

    // ========== Cons Cell Methods (for variadic args) ==========

    /// Allocate a cons cell (head, tail)
    pub fn allocate_cons(&mut self, head: usize, tail: usize) -> Result<usize, String> {
        let cons_ptr = self.allocate_raw(cons_layout::SIZE_WORDS, TYPE_ID_CONS)?;
        let heap_obj = HeapObject::from_untagged(cons_ptr as *const u8);

        heap_obj.write_field(cons_layout::FIELD_HEAD, head);
        heap_obj.write_field(cons_layout::FIELD_TAIL, tail);

        Ok(self.tag_heap_object(cons_ptr))
    }

    /// Get the head of a cons cell
    pub fn cons_head(&self, cons_ptr: usize) -> usize {
        let untagged = self.untag_heap_object(cons_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_field(cons_layout::FIELD_HEAD)
    }

    /// Get the tail of a cons cell
    pub fn cons_tail(&self, cons_ptr: usize) -> usize {
        let untagged = self.untag_heap_object(cons_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_field(cons_layout::FIELD_TAIL)
    }

    /// Check if a value is a cons cell
    pub fn is_cons(&self, value: usize) -> bool {
        // Check if it's a heap object
        if (value & 0b111) != 0b110 {
            return false;
        }
        let untagged = self.untag_heap_object(value);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_header().type_id == TYPE_ID_CONS
    }

    /// Build a list from a slice of values (right-to-left cons)
    /// Returns nil (tagged) for empty slice
    pub fn build_list(&mut self, values: &[usize]) -> Result<usize, String> {
        let nil = 7usize; // Tagged nil value
        let mut list = nil;
        for value in values.iter().rev() {
            list = self.allocate_cons(*value, list)?;
        }
        Ok(list)
    }

    // ========== DefType Methods ==========

    /// Register a new deftype and return its type_id
    pub fn register_type(&mut self, name: String, fields: Vec<String>) -> usize {
        if let Some(&type_id) = self.type_name_to_id.get(&name) {
            return type_id;
        }

        let type_id = self.type_registry.len();
        self.type_registry.push(TypeDef {
            name: name.clone(),
            fields,
        });
        self.type_name_to_id.insert(name, type_id);
        type_id
    }

    /// Get type definition by ID
    pub fn get_type_def(&self, type_id: usize) -> Option<&TypeDef> {
        self.type_registry.get(type_id)
    }

    /// Get type ID by name
    pub fn get_type_id(&self, name: &str) -> Option<usize> {
        self.type_name_to_id.get(name).copied()
    }

    /// Get field index for a type by field name
    pub fn get_type_field_index(&self, type_id: usize, field_name: &str) -> Option<usize> {
        self.type_registry.get(type_id)?.field_index(field_name)
    }

    /// Allocate a deftype instance on the heap
    pub fn allocate_type_instance(
        &mut self,
        type_id: usize,
        field_values: Vec<usize>,
    ) -> Result<usize, String> {
        let type_def = self.type_registry.get(type_id)
            .ok_or_else(|| format!("Unknown type_id: {}", type_id))?;

        if field_values.len() != type_def.fields.len() {
            return Err(format!(
                "Type {} expects {} fields, got {}",
                type_def.name, type_def.fields.len(), field_values.len()
            ));
        }

        let size_words = field_values.len();
        let obj_ptr = self.allocate_raw(size_words, TYPE_ID_DEFTYPE)?;
        let mut heap_obj = HeapObject::from_untagged(obj_ptr as *const u8);

        // Store type_id in header's type_data field
        let mut header = heap_obj.get_header();
        header.type_data = type_id as u32;
        heap_obj.write_header_direct(header);

        for (i, value) in field_values.iter().enumerate() {
            heap_obj.write_field(i, *value);
        }

        Ok(self.tag_heap_object(obj_ptr))
    }

    /// Read a field from a deftype instance
    pub fn read_type_field(&self, obj_ptr: usize, field_index: usize) -> usize {
        let untagged = self.untag_heap_object(obj_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_field(field_index)
    }

    /// Get the type_id from a deftype instance
    pub fn get_instance_type_id(&self, obj_ptr: usize) -> usize {
        let untagged = self.untag_heap_object(obj_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_header().type_data as usize
    }

    /// Load a field from a deftype instance by field name
    pub fn load_type_field_by_name(&self, obj_ptr: usize, field_name: &str) -> Result<usize, String> {
        let type_id = self.get_instance_type_id(obj_ptr);

        let type_def = self.type_registry.get(type_id)
            .ok_or_else(|| format!("Unknown type_id {} in object", type_id))?;

        let field_index = type_def.fields.iter()
            .position(|f| f == field_name)
            .ok_or_else(|| format!("Field '{}' not found in type '{}'", field_name, type_def.name))?;

        Ok(self.read_type_field(obj_ptr, field_index))
    }

    // ========== Heap Inspection Methods ==========

    /// Get detailed heap statistics
    pub fn detailed_heap_stats(&self) -> DetailedHeapStats {
        self.allocator.detailed_stats()
    }

    /// List all live objects in the heap
    pub fn list_objects(&self) -> Vec<ObjectInfo> {
        self.allocator.iter_objects()
            .map(|obj| self.object_to_info(&obj))
            .collect()
    }

    /// List objects filtered by type name
    pub fn list_objects_by_type(&self, type_name: &str) -> Vec<ObjectInfo> {
        self.allocator.iter_objects()
            .filter(|obj| {
                let type_id = obj.get_type_id() as u8;
                type_id_to_name(type_id) == type_name
            })
            .map(|obj| self.object_to_info(&obj))
            .collect()
    }

    /// Convert HeapObject to ObjectInfo
    fn object_to_info(&self, obj: &HeapObject) -> ObjectInfo {
        let header = obj.get_header();
        let type_id = header.type_id;
        let address = obj.get_pointer() as usize;

        // Compute tagged pointer based on type
        let tagged_ptr = match type_id {
            TYPE_ID_STRING => self.tag_string(address),
            TYPE_ID_FUNCTION => self.tag_closure(address),
            _ => self.tag_heap_object(address),
        };

        ObjectInfo {
            address,
            tagged_ptr,
            type_id,
            type_name: type_id_to_name(type_id),
            type_data: header.type_data,
            size_bytes: obj.full_size(),
            field_count: header.size as usize,
            is_opaque: header.opaque,
        }
    }

    /// Inspect a specific object by its tagged pointer
    pub fn inspect_object(&self, tagged_ptr: usize) -> Option<ObjectInfo> {
        if !BuiltInTypes::is_heap_pointer(tagged_ptr) {
            return None;
        }

        let untagged = BuiltInTypes::untag(tagged_ptr);
        if !self.allocator.contains_address(untagged) {
            return None;
        }

        let obj = HeapObject::from_untagged(untagged as *const u8);
        Some(self.object_to_info(&obj))
    }

    /// Get all fields of an object as (index, tagged_value, description)
    pub fn object_fields(&self, tagged_ptr: usize) -> Vec<(usize, usize, String)> {
        if !BuiltInTypes::is_heap_pointer(tagged_ptr) {
            return Vec::new();
        }

        let untagged = BuiltInTypes::untag(tagged_ptr);
        let obj = HeapObject::from_untagged(untagged as *const u8);
        let header = obj.get_header();

        if header.opaque {
            // Opaque objects (strings) don't have pointer fields
            return Vec::new();
        }

        let mut fields = Vec::new();
        for i in 0..header.size as usize {
            let value = obj.get_field(i);
            let desc = self.format_value(value);
            fields.push((i, value, desc));
        }
        fields
    }

    /// Format a tagged value for display
    pub fn format_value(&self, value: usize) -> String {
        let kind = BuiltInTypes::get_kind(value);
        match kind {
            BuiltInTypes::Int => format!("{}", (value as i64) >> 3),
            BuiltInTypes::Bool => {
                if value == 11 { "true".to_string() }
                else if value == 3 { "false".to_string() }
                else { format!("bool?{}", value) }
            }
            BuiltInTypes::Null => "nil".to_string(),
            BuiltInTypes::String => {
                let s = self.read_string(value);
                if s.len() > 32 {
                    format!("\"{}...\"", &s[..32])
                } else {
                    format!("\"{}\"", s)
                }
            }
            BuiltInTypes::Closure => format!("#<closure@{:x}>", value >> 3),
            BuiltInTypes::HeapObject => {
                let untagged = value >> 3;
                let obj = HeapObject::from_untagged(untagged as *const u8);
                let type_id = obj.get_type_id() as u8;
                match type_id {
                    TYPE_ID_NAMESPACE => {
                        let name = self.namespace_name(value);
                        format!("#<ns:{}>", name)
                    }
                    TYPE_ID_VAR => {
                        let (ns, sym) = self.var_info(value);
                        format!("#'{}/{}", ns, sym)
                    }
                    TYPE_ID_DEFTYPE => {
                        let type_data = obj.get_header().type_data as usize;
                        if let Some(def) = self.get_type_def(type_data) {
                            format!("#<{}@{:x}>", def.name, untagged)
                        } else {
                            format!("#<deftype@{:x}>", untagged)
                        }
                    }
                    TYPE_ID_CONS => {
                        // Format cons cells as a list
                        self.format_list(value)
                    }
                    _ => format!("#<object@{:x}>", untagged),
                }
            }
            BuiltInTypes::Function => format!("#<fn@{:x}>", value >> 3),
            BuiltInTypes::Float => {
                let bits = (value >> 3) as u64;
                format!("{}", f64::from_bits(bits))
            }
        }
    }

    /// Format a cons list as (a b c)
    fn format_list(&self, value: usize) -> String {
        let mut items = Vec::new();
        let mut current = value;
        let mut max_depth = 100; // Prevent infinite loops on circular lists

        while max_depth > 0 {
            max_depth -= 1;

            // Check if current is nil (end of list)
            if current == 7 {
                break;
            }

            // Check if current is a cons cell
            if BuiltInTypes::get_kind(current) != BuiltInTypes::HeapObject {
                // Improper list - add dot notation
                items.push(format!(". {}", self.format_value(current)));
                break;
            }

            let untagged = current >> 3;
            let obj = HeapObject::from_untagged(untagged as *const u8);
            let type_id = obj.get_type_id() as u8;

            if type_id != TYPE_ID_CONS {
                // Improper list - add dot notation
                items.push(format!(". {}", self.format_value(current)));
                break;
            }

            // Get head and tail
            let head = self.cons_head(current);
            let tail = self.cons_tail(current);

            items.push(self.format_value(head));
            current = tail;
        }

        format!("({})", items.join(" "))
    }

    /// Find all objects that reference a given target object
    pub fn find_references_to(&self, target_ptr: usize) -> Vec<ObjectReference> {
        if !BuiltInTypes::is_heap_pointer(target_ptr) {
            return Vec::new();
        }

        let target_untagged = BuiltInTypes::untag(target_ptr);
        let mut refs = Vec::new();

        for obj in self.allocator.iter_objects() {
            let header = obj.get_header();
            if header.opaque {
                continue;
            }

            let from_address = obj.get_pointer() as usize;

            for i in 0..header.size as usize {
                let field_value = obj.get_field(i);
                if BuiltInTypes::is_heap_pointer(field_value) {
                    let field_untagged = BuiltInTypes::untag(field_value);
                    if field_untagged == target_untagged {
                        refs.push(ObjectReference {
                            from_address,
                            to_address: target_untagged,
                            field_index: i,
                            tagged_value: field_value,
                        });
                    }
                }
            }
        }

        refs
    }

    /// Find all objects that a given source object references
    pub fn find_references_from(&self, source_ptr: usize) -> Vec<ObjectReference> {
        if !BuiltInTypes::is_heap_pointer(source_ptr) {
            return Vec::new();
        }

        let source_untagged = BuiltInTypes::untag(source_ptr);
        let obj = HeapObject::from_untagged(source_untagged as *const u8);
        let header = obj.get_header();

        if header.opaque {
            return Vec::new();
        }

        let mut refs = Vec::new();
        for i in 0..header.size as usize {
            let field_value = obj.get_field(i);
            if BuiltInTypes::is_heap_pointer(field_value) {
                refs.push(ObjectReference {
                    from_address: source_untagged,
                    to_address: BuiltInTypes::untag(field_value),
                    field_index: i,
                    tagged_value: field_value,
                });
            }
        }

        refs
    }

    /// List all GC roots (namespace bindings that are heap pointers)
    pub fn list_gc_roots(&self) -> Vec<(String, String, usize)> {
        let mut roots = Vec::new();

        for (ns_name, &ns_ptr) in &self.namespace_roots {
            // The namespace itself is a root
            roots.push((ns_name.clone(), "<namespace>".to_string(), ns_ptr));

            // Each binding in the namespace is also reachable
            for (var_name, value) in self.namespace_bindings(ns_ptr) {
                if BuiltInTypes::is_heap_pointer(value) {
                    roots.push((ns_name.clone(), var_name, value));
                }
            }
        }

        roots.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));
        roots
    }

    // ========== Exception Handling Methods ==========

    /// Push an exception handler onto the stack
    pub fn push_exception_handler(&mut self, handler: ExceptionHandler) {
        self.exception_handlers.push(handler);
    }

    /// Pop an exception handler from the stack
    pub fn pop_exception_handler(&mut self) -> Option<ExceptionHandler> {
        self.exception_handlers.pop()
    }

    /// Get the current number of exception handlers
    pub fn exception_handler_count(&self) -> usize {
        self.exception_handlers.len()
    }

    // ========== Protocol Methods ==========

    /// Register a new protocol and return its protocol_id
    pub fn register_protocol(&mut self, name: String, methods: Vec<ProtocolMethod>) -> usize {
        // Check if already registered
        if let Some(&protocol_id) = self.protocol_name_to_id.get(&name) {
            return protocol_id;
        }

        let protocol_id = self.protocol_registry.len();

        // Register method -> (protocol_id, method_index) mappings
        for (method_index, method) in methods.iter().enumerate() {
            self.method_to_protocol.insert(
                method.name.clone(),
                (protocol_id, method_index),
            );
        }

        self.protocol_registry.push(ProtocolDef {
            name: name.clone(),
            methods,
        });
        self.protocol_name_to_id.insert(name, protocol_id);
        protocol_id
    }

    /// Get protocol definition by ID
    pub fn get_protocol_def(&self, protocol_id: usize) -> Option<&ProtocolDef> {
        self.protocol_registry.get(protocol_id)
    }

    /// Get protocol ID by name
    pub fn get_protocol_id(&self, name: &str) -> Option<usize> {
        self.protocol_name_to_id.get(name).copied()
    }

    /// Get method index within a protocol by method name
    pub fn get_protocol_method_index(&self, protocol_id: usize, method_name: &str) -> Option<usize> {
        self.protocol_registry.get(protocol_id)?.method_index(method_name)
    }

    /// Register a protocol method implementation in the vtable
    pub fn register_protocol_method_impl(
        &mut self,
        type_id: usize,
        protocol_id: usize,
        method_index: usize,
        fn_ptr: usize,
    ) {
        self.protocol_vtable.insert((type_id, protocol_id, method_index), fn_ptr);
    }

    /// Look up protocol method implementation by type_id and method_name
    /// Returns the function pointer (tagged closure) if found
    pub fn lookup_protocol_method(&self, type_id: usize, method_name: &str) -> Option<usize> {
        // Look up which protocol this method belongs to
        let &(protocol_id, method_index) = self.method_to_protocol.get(method_name)?;

        // Look up the implementation in the vtable
        self.protocol_vtable.get(&(type_id, protocol_id, method_index)).copied()
    }

    /// Look up protocol method by protocol_id and method_index (direct vtable lookup)
    pub fn lookup_protocol_method_direct(
        &self,
        type_id: usize,
        protocol_id: usize,
        method_index: usize,
    ) -> Option<usize> {
        self.protocol_vtable.get(&(type_id, protocol_id, method_index)).copied()
    }

    /// Get the protocol type ID for a tagged value
    /// This is used for protocol dispatch - extracts the type_id used in the vtable
    pub fn get_type_id_for_value(&self, value: usize) -> usize {
        let tag = value & 0b111;

        match tag {
            0b000 => BUILTIN_TYPE_INT,      // Integer
            0b001 => BUILTIN_TYPE_FLOAT,    // Float
            0b010 => BUILTIN_TYPE_STRING,   // String (could be keyword/symbol too)
            0b011 => BUILTIN_TYPE_BOOL,     // Bool
            0b100 => BUILTIN_TYPE_FUNCTION, // Function
            0b101 => BUILTIN_TYPE_CLOSURE,  // Closure
            0b110 => {
                // HeapObject - need to check type_id in header
                let untagged = value >> 3;
                let heap_obj = HeapObject::from_untagged(untagged as *const u8);
                let header = heap_obj.get_header();

                match header.type_id {
                    TYPE_ID_NAMESPACE => BUILTIN_TYPE_NAMESPACE,
                    TYPE_ID_VAR => BUILTIN_TYPE_VAR,
                    TYPE_ID_DEFTYPE => {
                        // For deftypes, use type_data + offset to get unique type_id
                        header.type_data as usize + DEFTYPE_ID_OFFSET
                    }
                    // TODO: Add vector, map, list, set when implemented as heap objects
                    _ => 0, // Unknown type
                }
            }
            0b111 => BUILTIN_TYPE_NIL,      // Nil
            _ => unreachable!(),
        }
    }

    /// Check if a type implements a protocol
    pub fn type_satisfies_protocol(&self, type_id: usize, protocol_id: usize) -> bool {
        if let Some(protocol) = self.protocol_registry.get(protocol_id) {
            // Check if at least one method is implemented
            for method_index in 0..protocol.methods.len() {
                if self.protocol_vtable.contains_key(&(type_id, protocol_id, method_index)) {
                    return true;
                }
            }
        }
        false
    }

    /// Get the name of a built-in type ID (for error messages)
    pub fn builtin_type_name(type_id: usize) -> &'static str {
        match type_id {
            BUILTIN_TYPE_NIL => "nil",
            BUILTIN_TYPE_BOOL => "Boolean",
            BUILTIN_TYPE_INT => "Long",
            BUILTIN_TYPE_FLOAT => "Double",
            BUILTIN_TYPE_STRING => "String",
            BUILTIN_TYPE_KEYWORD => "Keyword",
            BUILTIN_TYPE_SYMBOL => "Symbol",
            BUILTIN_TYPE_LIST => "PersistentList",
            BUILTIN_TYPE_VECTOR => "PersistentVector",
            BUILTIN_TYPE_MAP => "PersistentHashMap",
            BUILTIN_TYPE_SET => "PersistentHashSet",
            BUILTIN_TYPE_FUNCTION => "Function",
            BUILTIN_TYPE_CLOSURE => "Closure",
            BUILTIN_TYPE_NAMESPACE => "Namespace",
            BUILTIN_TYPE_VAR => "Var",
            _ if type_id >= DEFTYPE_ID_OFFSET => "deftype",
            _ => "unknown",
        }
    }

    /// List all registered protocols
    pub fn list_protocols(&self) -> Vec<(usize, String, usize)> {
        self.protocol_registry
            .iter()
            .enumerate()
            .map(|(id, p)| (id, p.name.clone(), p.methods.len()))
            .collect()
    }
}

#[derive(Debug)]
pub struct HeapStats {
    pub gc_algorithm: String,
    pub namespace_count: usize,
    pub type_count: usize,
    pub stack_map_entries: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_string() {
        let mut runtime = GCRuntime::new();
        let str_ptr = runtime.allocate_string("hello").unwrap();
        let str_val = runtime.read_string(str_ptr);
        assert_eq!(str_val, "hello");
    }

    #[test]
    fn test_allocate_namespace() {
        let mut runtime = GCRuntime::new();
        let ns_ptr = runtime.allocate_namespace("user").unwrap();
        let ns_name = runtime.namespace_name(ns_ptr);
        assert_eq!(ns_name, "user");
    }

    #[test]
    fn test_namespace_binding() {
        let mut runtime = GCRuntime::new();
        let ns_ptr = runtime.allocate_namespace("user").unwrap();

        // Add a binding (42 tagged as int)
        let value = 42 << 3;
        let new_ns_ptr = runtime.namespace_add_binding(ns_ptr, "x", value).unwrap();

        // Look it up
        let result = runtime.namespace_lookup(new_ns_ptr, "x");
        assert_eq!(result, Some(value));
    }
}
