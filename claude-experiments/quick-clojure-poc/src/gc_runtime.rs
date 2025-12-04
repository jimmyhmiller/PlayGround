// Simplified GC Runtime for Namespaces
//
// This is a minimal GC implementation that manages heap-allocated namespaces.
// For now, we use a simple bump allocator with mark-and-sweep GC.
//
// Namespace heap layout:
// Header (8 bytes) | name_ptr (8 bytes) | [symbol_name_ptr, value] pairs...

use std::collections::{HashMap, HashSet};

const TYPE_ID_STRING: u8 = 2;
const TYPE_ID_NAMESPACE: u8 = 10;
const TYPE_ID_VAR: u8 = 11;
const TYPE_ID_FUNCTION: u8 = 12;

/// Header for heap-allocated objects
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Header {
    pub type_id: u8,
    pub type_data: u32,  // For strings: byte length
    pub size: u8,        // Size in 8-byte words (excluding header)
    pub opaque: bool,    // True if object contains no pointers
    pub marked: bool,    // GC mark bit
}

impl Header {
    fn to_usize(self) -> usize {
        let mut data: usize = 0;
        data |= (self.type_id as usize) << 56;
        data |= (self.type_data as usize) << 24;
        data |= (self.size as usize) << 16;
        if self.opaque {
            data |= 1 << 1;
        }
        if self.marked {
            data |= 1 << 0;
        }
        data
    }

    fn from_usize(data: usize) -> Self {
        Header {
            type_id: (data >> 56) as u8,
            type_data: (data >> 24) as u32,
            size: (data >> 16) as u8,
            opaque: (data & 0b10) != 0,
            marked: (data & 0b01) != 0,
        }
    }
}

/// Wrapper for heap objects
pub struct HeapObject {
    pointer: usize,
}

impl HeapObject {
    pub fn from_untagged(pointer: usize) -> Self {
        assert!(pointer % 8 == 0, "Heap objects must be 8-byte aligned");
        HeapObject { pointer }
    }

    pub fn read_header(&self) -> Header {
        unsafe {
            let header_ptr = self.pointer as *const usize;
            let header_value = *header_ptr;
            Header::from_usize(header_value)
        }
    }

    pub fn write_header_direct(&mut self, header: Header) {
        unsafe {
            let header_ptr = self.pointer as *mut usize;
            *header_ptr = header.to_usize();
        }
    }

    pub fn get_field(&self, index: usize) -> usize {
        unsafe {
            let ptr = self.pointer as *const usize;
            let field_ptr = ptr.add(1 + index);  // Skip header
            *field_ptr
        }
    }

    pub fn write_field(&mut self, index: usize, value: usize) {
        unsafe {
            let ptr = self.pointer as *mut usize;
            let field_ptr = ptr.add(1 + index);  // Skip header
            *field_ptr = value;
        }
    }

    pub fn write_bytes(&mut self, bytes: &[u8]) {
        unsafe {
            let ptr = self.pointer as *mut u8;
            let data_ptr = ptr.add(8);  // Skip header
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), data_ptr, bytes.len());
        }
    }

    pub fn read_string(&self, byte_len: usize) -> String {
        unsafe {
            let ptr = self.pointer as *const u8;
            let data_ptr = ptr.add(8);  // Skip header
            let bytes = std::slice::from_raw_parts(data_ptr, byte_len);
            String::from_utf8_unchecked(bytes.to_vec())
        }
    }

    pub fn mark(&mut self) {
        let mut header = self.read_header();
        header.marked = true;
        self.write_header_direct(header);
    }

    pub fn is_marked(&self) -> bool {
        self.read_header().marked
    }

    pub fn unmark(&mut self) {
        let mut header = self.read_header();
        header.marked = false;
        self.write_header_direct(header);
    }

    pub fn full_size(&self) -> usize {
        let header = self.read_header();
        8 + (header.size as usize * 8)
    }
}

/// Built-in type tagging
#[derive(Debug, Copy, Clone)]
pub enum BuiltInTypes {
    Int,
    String,
    HeapObject,
}

impl BuiltInTypes {
    pub fn tag(&self) -> usize {
        match self {
            BuiltInTypes::Int => 0b000,
            BuiltInTypes::String => 0b010,
            BuiltInTypes::HeapObject => 0b110,
        }
    }

    pub fn tagged(&self, value: usize) -> usize {
        (value << 3) | self.tag()
    }

    pub fn untag(&self, value: usize) -> usize {
        value >> 3
    }
}

/// Simple bump allocator with mark-and-sweep GC
pub struct GCRuntime {
    heap: Vec<u8>,
    next_alloc: usize,
    heap_size: usize,

    /// Namespace roots: namespace_name -> tagged pointer
    namespace_roots: HashMap<String, usize>,

    /// All allocated objects (for GC marking)
    objects: Vec<usize>,

    /// Thread-local binding stacks for dynamic vars
    /// Map from var_ptr to stack of values
    /// When a var is accessed, we check this stack first before reading the root value
    dynamic_bindings: HashMap<usize, Vec<usize>>,

    /// Set of vars that are marked as dynamic
    /// Only vars in this set can have thread-local bindings
    dynamic_vars: std::collections::HashSet<usize>,
}

impl Default for GCRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl GCRuntime {
    pub fn new() -> Self {
        let heap_size = 1024 * 1024;  // 1MB heap
        let mut heap = vec![0u8; heap_size];
        let heap_ptr = heap.as_mut_ptr() as usize;

        // Align to 8 bytes
        let aligned_start = (heap_ptr + 7) & !7;
        let offset = aligned_start - heap_ptr;

        GCRuntime {
            heap,
            next_alloc: offset,
            heap_size,
            namespace_roots: HashMap::new(),
            objects: Vec::new(),
            dynamic_bindings: HashMap::new(),
            dynamic_vars: HashSet::new(),
        }
    }

    /// Allocate a raw heap object
    fn allocate_raw(&mut self, size_words: usize, type_id: u8) -> Result<usize, String> {
        let total_size = 8 + (size_words * 8);  // Header + fields

        if self.next_alloc + total_size > self.heap_size {
            // Try GC first
            self.gc()?;

            // Check again after GC
            if self.next_alloc + total_size > self.heap_size {
                return Err("Out of memory".to_string());
            }
        }

        let heap_ptr = self.heap.as_ptr() as usize;
        let object_ptr = heap_ptr + self.next_alloc;

        // Initialize header
        let mut heap_obj = HeapObject::from_untagged(object_ptr);
        heap_obj.write_header_direct(Header {
            type_id,
            type_data: 0,
            size: size_words as u8,
            opaque: false,
            marked: false,
        });

        self.next_alloc += total_size;
        self.objects.push(object_ptr);

        Ok(object_ptr)
    }

    /// Allocate a string on the heap
    pub fn allocate_string(&mut self, s: &str) -> Result<usize, String> {
        let bytes = s.as_bytes();
        let words = bytes.len().div_ceil(8);  // Round up to 8-byte words

        let ptr = self.allocate_raw(words, TYPE_ID_STRING)?;

        let mut heap_obj = HeapObject::from_untagged(ptr);

        // Update header with actual byte length
        let mut header = heap_obj.read_header();
        header.type_data = bytes.len() as u32;
        header.opaque = true;  // Strings have no pointers
        heap_obj.write_header_direct(header);

        // Write string data
        heap_obj.write_bytes(bytes);

        Ok(BuiltInTypes::String.tagged(ptr))
    }

    /// Allocate a namespace object on the heap
    pub fn allocate_namespace(&mut self, name: &str) -> Result<usize, String> {
        // 1. Allocate name string
        let name_ptr = self.allocate_string(name)?;

        // 2. Allocate namespace: 1 word for name, start with 0 bindings
        let size_words = 1;
        let ns_ptr = self.allocate_raw(size_words, TYPE_ID_NAMESPACE)?;

        // 3. Write namespace data
        let mut heap_obj = HeapObject::from_untagged(ns_ptr);
        heap_obj.write_field(0, name_ptr);  // Store name pointer

        // 4. Tag and return
        Ok(BuiltInTypes::HeapObject.tagged(ns_ptr))
    }

    /// Add or update a binding in a namespace (may reallocate!)
    pub fn namespace_add_binding(
        &mut self,
        ns_ptr: usize,
        symbol_name: &str,
        value: usize,
    ) -> Result<usize, String> {
        let ns_untagged = BuiltInTypes::HeapObject.untag(ns_ptr);
        let heap_obj = HeapObject::from_untagged(ns_untagged);
        let header = heap_obj.read_header();
        let current_size = header.size as usize;

        // Check if binding already exists - if so, update it in place
        if current_size > 0 {
            let num_bindings = (current_size - 1) / 2;
            for i in 0..num_bindings {
                let name_ptr = heap_obj.get_field(1 + i * 2);
                let stored_name = self.read_string(name_ptr);
                if stored_name == symbol_name {
                    // Found existing binding - update value in place
                    let mut mutable_obj = HeapObject::from_untagged(ns_untagged);
                    mutable_obj.write_field(1 + i * 2 + 1, value);
                    return Ok(ns_ptr);  // Return same pointer, no reallocation needed
                }
            }
        }

        // Binding doesn't exist - add new one
        // Allocate symbol name string
        let symbol_ptr = self.allocate_string(symbol_name)?;

        // Reallocate with +2 words (name + value)
        let new_size = current_size + 2;
        let new_ns_ptr = self.allocate_raw(new_size, TYPE_ID_NAMESPACE)?;

        // Copy existing fields
        let mut new_heap_obj = HeapObject::from_untagged(new_ns_ptr);
        for i in 0..current_size {
            let field = heap_obj.get_field(i);
            new_heap_obj.write_field(i, field);
        }

        // Add new binding
        new_heap_obj.write_field(current_size, symbol_ptr);
        new_heap_obj.write_field(current_size + 1, value);

        Ok(BuiltInTypes::HeapObject.tagged(new_ns_ptr))
    }

    /// Look up a binding in a namespace
    pub fn namespace_lookup(&self, ns_ptr: usize, symbol_name: &str) -> Option<usize> {
        let ns_untagged = BuiltInTypes::HeapObject.untag(ns_ptr);
        let heap_obj = HeapObject::from_untagged(ns_untagged);
        let header = heap_obj.read_header();
        let size = header.size as usize;

        if size == 0 {
            return None;
        }

        let num_bindings = (size - 1) / 2;

        // Linear search through bindings
        for i in 0..num_bindings {
            let name_ptr = heap_obj.get_field(1 + i * 2);
            let stored_name = self.read_string(name_ptr);
            if stored_name == symbol_name {
                return Some(heap_obj.get_field(1 + i * 2 + 1));
            }
        }
        None
    }

    /// Get namespace name (for display)
    pub fn namespace_name(&self, ns_ptr: usize) -> String {
        let ns_untagged = BuiltInTypes::HeapObject.untag(ns_ptr);
        let heap_obj = HeapObject::from_untagged(ns_untagged);
        let name_ptr = heap_obj.get_field(0);
        self.read_string(name_ptr)
    }

    /// Register namespace as GC root
    pub fn add_namespace_root(&mut self, name: String, ns_ptr: usize) {
        self.namespace_roots.insert(name, ns_ptr);
    }

    /// Read a string from a tagged pointer
    pub fn read_string(&self, tagged_ptr: usize) -> String {
        let ptr = BuiltInTypes::String.untag(tagged_ptr);
        let heap_obj = HeapObject::from_untagged(ptr);
        let header = heap_obj.read_header();
        let byte_len = header.type_data as usize;
        heap_obj.read_string(byte_len)
    }

    /// Mark phase of GC
    fn mark(&mut self) {
        // Mark all namespace roots - collect them first to avoid borrow checker issues
        let root_ptrs: Vec<usize> = self.namespace_roots.values().copied().collect();
        for ns_ptr in root_ptrs {
            self.mark_object(ns_ptr);
        }
    }

    /// Mark a single object and its references
    fn mark_object(&mut self, tagged_ptr: usize) {
        let tag = tagged_ptr & 0b111;

        // Only mark heap objects (strings and namespaces)
        if tag != BuiltInTypes::String.tag() && tag != BuiltInTypes::HeapObject.tag() {
            return;
        }

        let ptr = tagged_ptr >> 3;
        let mut heap_obj = HeapObject::from_untagged(ptr);

        // Already marked?
        if heap_obj.is_marked() {
            return;
        }

        heap_obj.mark();

        let header = heap_obj.read_header();

        // If opaque (like strings), no pointers to follow
        if header.opaque {
            return;
        }

        // Mark all fields (they might be pointers)
        let size = header.size as usize;
        for i in 0..size {
            let field = heap_obj.get_field(i);
            if field != 0 {
                self.mark_object(field);
            }
        }
    }

    /// Sweep phase of GC
    fn sweep(&mut self) {
        // Unmark all objects (for now, we don't actually free memory)
        // In a real implementation, we'd compact or free unmarked objects
        for &obj_ptr in &self.objects {
            let mut heap_obj = HeapObject::from_untagged(obj_ptr);
            heap_obj.unmark();
        }
    }

    /// Run garbage collection
    fn gc(&mut self) -> Result<(), String> {
        self.mark();
        self.sweep();
        Ok(())
    }

    /// Public API to trigger GC manually
    pub fn run_gc(&mut self) -> Result<(), String> {
        self.gc()
    }

    /// Get heap statistics
    pub fn heap_stats(&self) -> HeapStats {
        let mut stats = HeapStats {
            heap_size: self.heap_size,
            used_bytes: self.next_alloc,
            free_bytes: self.heap_size - self.next_alloc,
            object_count: self.objects.len(),
            namespace_count: self.namespace_roots.len(),
            objects: Vec::new(),
        };

        // Collect info about each object
        for &obj_ptr in &self.objects {
            let heap_obj = HeapObject::from_untagged(obj_ptr);
            let header = heap_obj.read_header();

            let obj_type = match header.type_id {
                TYPE_ID_STRING => "String",
                TYPE_ID_NAMESPACE => "Namespace",
                TYPE_ID_VAR => "Var",
                _ => "Unknown",
            };

            let size = heap_obj.full_size();

            let name = if header.type_id == TYPE_ID_NAMESPACE {
                let name_ptr = heap_obj.get_field(0);
                Some(self.read_string(name_ptr))
            } else if header.type_id == TYPE_ID_STRING {
                let byte_len = header.type_data as usize;
                Some(heap_obj.read_string(byte_len))
            } else if header.type_id == TYPE_ID_VAR {
                // Show var as #'ns/name
                let (ns_name, symbol_name) = self.var_info(BuiltInTypes::HeapObject.tagged(obj_ptr));
                Some(format!("#'{}/{}", ns_name, symbol_name))
            } else {
                None
            };

            stats.objects.push(ObjectInfo {
                address: obj_ptr,
                obj_type: obj_type.to_string(),
                size_bytes: size,
                marked: header.marked,
                name,
            });
        }

        stats
    }

    /// List all namespaces
    pub fn list_namespaces(&self) -> Vec<(String, usize, usize)> {
        let mut namespaces = Vec::new();
        for (name, &ptr) in &self.namespace_roots {
            let ns_untagged = BuiltInTypes::HeapObject.untag(ptr);
            let heap_obj = HeapObject::from_untagged(ns_untagged);
            let header = heap_obj.read_header();
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
        let ns_untagged = BuiltInTypes::HeapObject.untag(ns_ptr);
        let heap_obj = HeapObject::from_untagged(ns_untagged);
        let header = heap_obj.read_header();
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

    /// Allocate a var object
    pub fn allocate_var(
        &mut self,
        ns_ptr: usize,
        symbol_name: &str,
        initial_value: usize,
    ) -> Result<usize, String> {
        let symbol_ptr = self.allocate_string(symbol_name)?;

        let var_ptr = self.allocate_raw(3, TYPE_ID_VAR)?;
        let mut heap_obj = HeapObject::from_untagged(var_ptr);
        heap_obj.write_field(0, ns_ptr);        // namespace
        heap_obj.write_field(1, symbol_ptr);    // symbol name
        heap_obj.write_field(2, initial_value); // current value

        Ok(BuiltInTypes::HeapObject.tagged(var_ptr))
    }

    /// Get the current value from a var
    pub fn var_get_value(&self, var_ptr: usize) -> usize {
        let untagged = BuiltInTypes::HeapObject.untag(var_ptr);
        let heap_obj = HeapObject::from_untagged(untagged);
        heap_obj.get_field(2)
    }

    /// Mark a var as dynamic (allows thread-local bindings)
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
    /// Errors if no binding exists (can't set root with set!)
    pub fn set_binding(&mut self, var_ptr: usize, value: usize) -> Result<(), String> {
        if let Some(stack) = self.dynamic_bindings.get_mut(&var_ptr)
            && let Some(last) = stack.last_mut() {
                *last = value;
                return Ok(());
            }

        // No thread-local binding exists
        let (ns_name, symbol_name) = self.var_info(var_ptr);
        Err(format!(
            "Can't change/establish root binding of: {}/{} with set",
            ns_name, symbol_name
        ))
    }

    /// Get the current value of a var, checking dynamic bindings first
    pub fn var_get_value_dynamic(&self, var_ptr: usize) -> usize {
        // Check dynamic bindings first
        if let Some(stack) = self.dynamic_bindings.get(&var_ptr)
            && let Some(&value) = stack.last() {
                return value;
            }

        // Fall back to root value
        self.var_get_value(var_ptr)
    }

    /// Get var namespace and symbol name (for printing)
    pub fn var_info(&self, var_ptr: usize) -> (String, String) {
        let untagged = BuiltInTypes::HeapObject.untag(var_ptr);
        let heap_obj = HeapObject::from_untagged(untagged);

        let ns_ptr = heap_obj.get_field(0);
        let ns_name = self.namespace_name(ns_ptr);

        let symbol_ptr = heap_obj.get_field(1);
        let symbol_name = self.read_string(symbol_ptr);

        (ns_name, symbol_name)
    }

    /// Allocate a function object on the heap
    /// Layout: Header | name_ptr | code_ptr | closure_count | [closure_value...]
    ///
    /// For Phase 1: Simple implementation without multi-arity dispatch
    /// code_ptr points to the compiled ARM64 code entry point
    /// closure values are captured variables from the enclosing scope
    pub fn allocate_function(
        &mut self,
        name: Option<String>,
        code_ptr: usize,
        closure_values: Vec<usize>,
    ) -> Result<usize, String> {
        // Allocate name string if present
        let name_ptr = if let Some(n) = name {
            self.allocate_string(&n)?
        } else {
            0 // null pointer for anonymous functions
        };

        // Calculate size: name + code_ptr + closure_count + closure_values
        let size_words = 3 + closure_values.len();

        let fn_ptr = self.allocate_raw(size_words, TYPE_ID_FUNCTION)?;
        let mut heap_obj = HeapObject::from_untagged(fn_ptr);

        heap_obj.write_field(0, name_ptr);
        heap_obj.write_field(1, code_ptr);
        heap_obj.write_field(2, closure_values.len());

        // Write closure values
        for (i, value) in closure_values.iter().enumerate() {
            heap_obj.write_field(3 + i, *value);
        }

        Ok(BuiltInTypes::HeapObject.tagged(fn_ptr))
    }

    /// Get function code pointer
    pub fn function_code_ptr(&self, fn_ptr: usize) -> usize {
        let untagged = BuiltInTypes::HeapObject.untag(fn_ptr);
        let heap_obj = HeapObject::from_untagged(untagged);
        heap_obj.get_field(1)
    }

    /// Get closure value by index
    pub fn function_get_closure(&self, fn_ptr: usize, index: usize) -> usize {
        let untagged = BuiltInTypes::HeapObject.untag(fn_ptr);
        let heap_obj = HeapObject::from_untagged(untagged);
        let closure_count = heap_obj.get_field(2);
        if index >= closure_count {
            panic!("Closure index out of bounds: {} >= {}", index, closure_count);
        }
        heap_obj.get_field(3 + index)
    }

    /// Get function name (for debugging/printing)
    pub fn function_name(&self, fn_ptr: usize) -> String {
        let untagged = BuiltInTypes::HeapObject.untag(fn_ptr);
        let heap_obj = HeapObject::from_untagged(untagged);
        let name_ptr = heap_obj.get_field(0);
        if name_ptr == 0 {
            "anonymous".to_string()
        } else {
            self.read_string(name_ptr)
        }
    }
}

#[derive(Debug)]
pub struct HeapStats {
    pub heap_size: usize,
    pub used_bytes: usize,
    pub free_bytes: usize,
    pub object_count: usize,
    pub namespace_count: usize,
    pub objects: Vec<ObjectInfo>,
}

#[derive(Debug)]
pub struct ObjectInfo {
    pub address: usize,
    pub obj_type: String,
    pub size_bytes: usize,
    pub marked: bool,
    pub name: Option<String>,
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

        // Add a binding
        let value = BuiltInTypes::Int.tagged(42);
        let new_ns_ptr = runtime.namespace_add_binding(ns_ptr, "x", value).unwrap();

        // Look it up
        let result = runtime.namespace_lookup(new_ns_ptr, "x");
        assert_eq!(result, Some(value));
    }
}
