// GC Runtime - High-level runtime using pluggable GC allocators
//
// This module provides the high-level runtime interface for managing
// heap-allocated objects (namespaces, vars, functions, deftypes) using
// the pluggable GC system from the gc module.

use std::collections::{HashMap, HashSet};

use crate::gc::{
    Allocator, AllocateAction, AllocatorOptions, StackMap, StackMapDetails,
    BuiltInTypes, HeapObject, Word,
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

/// Closure heap object layout constants
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

/// GC Runtime with pluggable allocator
pub struct GCRuntime {
    /// The underlying allocator
    allocator: Alloc,

    /// Stack map for precise GC root scanning
    stack_map: StackMap,

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
            namespace_roots: HashMap::new(),
            namespace_name_to_id: HashMap::new(),
            next_namespace_id: 0,
            dynamic_bindings: HashMap::new(),
            dynamic_vars: HashSet::new(),
            type_registry: Vec::new(),
            type_name_to_id: HashMap::new(),
            options,
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

        Ok(self.tag_heap_object(new_ns_ptr))
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
}

#[derive(Debug)]
pub struct HeapStats {
    pub gc_algorithm: String,
    pub namespace_count: usize,
    pub type_count: usize,
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
