// GC Runtime - High-level runtime using pluggable GC allocators
//
// This module provides the high-level runtime interface for managing
// heap-allocated objects (namespaces, vars, functions, deftypes) using
// the pluggable GC system from the gc module.

use std::collections::{HashMap, HashSet};

use crate::gc::{
    AllocateAction, Allocator, AllocatorOptions, BuiltInTypes, DetailedHeapStats, HeapInspector,
    HeapObject, ObjectInfo, ObjectReference, StackMap, StackMapDetails, Word, type_id_to_name,
};

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

// ========== Type IDs ==========
// Used both in heap object headers (cast to u8) and for protocol dispatch (as usize).
// Tagged primitives (int, bool, nil) don't have heap headers but use these IDs for dispatch.

pub const TYPE_NIL: usize = 0;
pub const TYPE_BOOL: usize = 1;
pub const TYPE_INT: usize = 2;
pub const TYPE_FLOAT: usize = 3;
pub const TYPE_STRING: usize = 4;
pub const TYPE_KEYWORD: usize = 5;
pub const TYPE_SYMBOL: usize = 6;
pub const TYPE_LIST: usize = 7; // PersistentList / Cons
pub const TYPE_VECTOR: usize = 8;
pub const TYPE_MAP: usize = 9;
pub const TYPE_SET: usize = 10;
pub const TYPE_FUNCTION: usize = 11;
pub const TYPE_CLOSURE: usize = 12;
pub const TYPE_NAMESPACE: usize = 13;
pub const TYPE_VAR: usize = 14;
pub const TYPE_ARRAY: usize = 15;
pub const TYPE_MULTI_ARITY_FN: usize = 16;
pub const TYPE_DEFTYPE: usize = 17; // Base for deftypes, actual ID = TYPE_DEFTYPE + type_data
pub const TYPE_DYNAMIC_ARRAY: usize = 18; // Dynamic array: header.size = capacity, header.type_data = used count

// ========== Reader Types ==========
// Opaque types produced by the Rust reader, used directly by macros.
// These types have Rust-implemented operations exposed as builtins.
pub const TYPE_READER_LIST: usize = 30;
pub const TYPE_READER_VECTOR: usize = 31;
pub const TYPE_READER_MAP: usize = 32;
pub const TYPE_READER_SYMBOL: usize = 33;

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
    pub handler_address: usize, // Label address to jump to (catch block)
    pub stack_pointer: usize,   // Saved SP
    pub frame_pointer: usize,   // Saved FP (x29)
    pub link_register: usize,   // Saved LR (x30)
    pub result_local: isize,    // Where to store exception (FP-relative offset, negative)
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
    pub const FIELD_VARIADIC_INDEX: usize = 3; // Index of variadic arity in table, or NO_VARIADIC
    pub const FIELD_CLOSURE_COUNT: usize = 4;
    pub const ARITY_TABLE_START: usize = 5;

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
        5 + arity_count * ARITY_ENTRY_SIZE + closure_count
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

/// Raw mutable array layout constants
/// Layout:
///   [header(8)]
///   [length(8)]         - field 0: Array length (tagged integer for GC safety)
///   [element0(8)]       - field 1: First element (tagged value)
///   [element1(8)]       - field 2: Second element (tagged value)
///   ...
#[allow(dead_code)]
pub mod array_layout {
    pub const FIELD_LENGTH: usize = 0;
    pub const FIELD_FIRST_ELEMENT: usize = 1;

    /// Total size in words: 1 (length) + element_count
    pub const fn total_size_words(element_count: usize) -> usize {
        1 + element_count
    }

    /// Get field index for an element
    pub const fn element_field(index: usize) -> usize {
        FIELD_FIRST_ELEMENT + index
    }
}

// ========== Reader Type Layouts ==========
// These types are produced by the Rust reader and consumed by macros.
// They are simple array-backed structures, not persistent data structures.

/// ReaderList layout: simple array of elements
/// Layout:
///   [header(8)]
///   [count(8)]        - field 0: element count (raw usize, not tagged)
///   [element0(8)]     - field 1+: elements (tagged values)
#[allow(dead_code)]
pub mod reader_list_layout {
    pub const FIELD_COUNT: usize = 0;
    pub const FIELD_FIRST_ELEMENT: usize = 1;

    pub const fn total_size_words(element_count: usize) -> usize {
        1 + element_count
    }

    pub const fn element_field(index: usize) -> usize {
        FIELD_FIRST_ELEMENT + index
    }
}

/// ReaderVector layout: simple array of elements (same as list but different type ID)
/// Layout:
///   [header(8)]
///   [count(8)]        - field 0: element count (raw usize, not tagged)
///   [element0(8)]     - field 1+: elements (tagged values)
#[allow(dead_code)]
pub mod reader_vector_layout {
    pub const FIELD_COUNT: usize = 0;
    pub const FIELD_FIRST_ELEMENT: usize = 1;

    pub const fn total_size_words(element_count: usize) -> usize {
        1 + element_count
    }

    pub const fn element_field(index: usize) -> usize {
        FIELD_FIRST_ELEMENT + index
    }
}

/// ReaderMap layout: array of key-value pairs
/// Layout:
///   [header(8)]
///   [count(8)]        - field 0: number of entries (raw usize, not tagged)
///   [key0(8)]         - field 1+: alternating key, value pairs (tagged values)
///   [val0(8)]
///   [key1(8)]
///   [val1(8)]
///   ...
#[allow(dead_code)]
pub mod reader_map_layout {
    pub const FIELD_COUNT: usize = 0;
    pub const FIELD_FIRST_ENTRY: usize = 1;

    pub const fn total_size_words(entry_count: usize) -> usize {
        1 + entry_count * 2
    }

    pub const fn key_field(index: usize) -> usize {
        FIELD_FIRST_ENTRY + index * 2
    }

    pub const fn value_field(index: usize) -> usize {
        FIELD_FIRST_ENTRY + index * 2 + 1
    }
}

/// ReaderSymbol layout: namespace, name, and metadata as tagged pointers
/// Layout:
///   [header(8)]
///   [namespace(8)]    - field 0: namespace string (tagged) or nil
///   [name(8)]         - field 1: name string (tagged)
///   [metadata(8)]     - field 2: metadata map (tagged ReaderMap) or nil
#[allow(dead_code)]
pub mod reader_symbol_layout {
    pub const FIELD_NAMESPACE: usize = 0;
    pub const FIELD_NAME: usize = 1;
    pub const FIELD_METADATA: usize = 2;
    pub const SIZE_WORDS: usize = 3;
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

    /// Mapping from protocol vtable key to namespace root ID (for GC tracking)
    /// This allows protocol method closures to be treated as GC roots
    protocol_vtable_root_ids: HashMap<(usize, usize, usize), usize>,

    /// Marker protocol satisfaction: Set of (type_id, protocol_id) for marker protocols
    /// Used for protocols that have no methods (like IList, ISeq marker interfaces)
    marker_protocol_satisfies: HashSet<(usize, usize)>,

    // ========== Keyword Interning ==========
    /// Keyword constant storage: index -> keyword text (without colon)
    keyword_constants: Vec<String>,

    /// Cache of allocated keyword heap pointers: index -> Some(tagged_ptr) if allocated
    keyword_heap_ptrs: Vec<Option<usize>>,

    /// Mapping from keyword index to namespace root ID (for GC tracking)
    keyword_root_ids: HashMap<usize, usize>,

    // ========== String Constants (for compile-time string literals) ==========
    /// String constants that need to survive GC (allocated at compile time)
    /// Maps string pointer to its namespace root ID
    string_constant_root_ids: HashMap<usize, usize>,

    // ========== Symbol Interning (for runtime var lookup) ==========
    /// Symbol table: symbol_id -> symbol string
    symbol_table: Vec<String>,

    /// Reverse lookup: symbol string -> symbol_id
    symbol_name_to_id: HashMap<String, u32>,

    // ========== Macro Support ==========
    /// Cached :macro keyword pointer for fast macro checking
    macro_keyword_ptr: Option<usize>,
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
            stack_base: 0, // Set by set_stack_base() before running JIT code
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
            protocol_vtable_root_ids: HashMap::new(),
            marker_protocol_satisfies: HashSet::new(),
            // Keyword interning
            keyword_constants: Vec::new(),
            keyword_heap_ptrs: Vec::new(),
            keyword_root_ids: HashMap::new(),
            // String constants (compile-time literals)
            string_constant_root_ids: HashMap::new(),
            // Symbol interning (for runtime var lookup)
            symbol_table: Vec::new(),
            symbol_name_to_id: HashMap::new(),
            // Macro support
            macro_keyword_ptr: None,
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

    /// Add an object to the GC root set (write barrier for generational GC)
    /// This should be called before writing a pointer to a mutable field
    /// to ensure the old-generation object is scanned for young-generation references
    pub fn gc_add_root(&mut self, ptr: usize) {
        self.allocator.gc_add_root(ptr);
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
        self.allocator
            .gc(&self.stack_map, &[(self.stack_base, stack_pointer)]);

        // Handle relocations (for compacting/generational GC)
        let relocations = self.allocator.get_namespace_relocations();
        for (_ns_id, updates) in relocations {
            for (old_ptr, new_ptr) in updates {
                // Update namespace roots
                for (_name, ptr) in self.namespace_roots.iter_mut() {
                    if *ptr == old_ptr {
                        *ptr = new_ptr;
                    }
                }

                // Update protocol_vtable entries if they were relocated
                for (_key, ptr) in self.protocol_vtable.iter_mut() {
                    if *ptr == old_ptr {
                        *ptr = new_ptr;
                    }
                }

                // Update keyword_heap_ptrs if they were relocated
                for ptr_opt in self.keyword_heap_ptrs.iter_mut() {
                    if let Some(ptr) = ptr_opt
                        && *ptr == old_ptr {
                            *ptr = new_ptr;
                        }
                }

                // Update dynamic_vars set if a var was relocated
                if self.dynamic_vars.remove(&old_ptr) {
                    self.dynamic_vars.insert(new_ptr);
                }

                // Update dynamic_bindings keys if a var was relocated
                if let Some(stack) = self.dynamic_bindings.remove(&old_ptr) {
                    self.dynamic_bindings.insert(new_ptr, stack);
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
        match self
            .allocator
            .try_allocate(size_words, BuiltInTypes::HeapObject)
        {
            Ok(AllocateAction::Allocated(ptr)) => {
                let mut heap_obj = HeapObject::from_untagged(ptr);
                heap_obj.write_header(Word::from_word(size_words));
                heap_obj.write_type_id(type_id as usize);

                // Initialize all fields to null to prevent GC from seeing garbage
                // as heap pointers. This is critical for gc-always mode.
                let null_value = BuiltInTypes::null_value() as usize;
                for i in 0..size_words {
                    heap_obj.write_field(i, null_value);
                }

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

    /// Allocate a dynamic array with given capacity
    /// header.size = capacity (words), header.type_data = used count (words)
    fn allocate_dynamic_array(&mut self, capacity: usize) -> Result<usize, String> {
        let ptr = self.allocate_raw(capacity, TYPE_DYNAMIC_ARRAY as u8)?;
        let heap_obj = HeapObject::from_untagged(ptr as *const u8);
        // Set used count to 0
        heap_obj.set_type_data(0);
        Ok(ptr)
    }

    /// Allocate a string on the heap
    pub fn allocate_string(&mut self, s: &str) -> Result<usize, String> {
        let bytes = s.as_bytes();
        let words = bytes.len().div_ceil(8);

        let ptr = self.allocate_raw(words, TYPE_STRING as u8)?;

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

    /// Allocate a string constant that will survive GC
    /// This is used for compile-time string literals that need to be rooted
    pub fn allocate_string_constant(&mut self, s: &str) -> Result<usize, String> {
        // Allocate the string normally
        let str_ptr = self.allocate_string(s)?;

        // Register as GC root so string constants are never collected
        let root_id = self.next_namespace_id;
        self.next_namespace_id += 1;
        self.string_constant_root_ids.insert(str_ptr, root_id);
        self.allocator.add_namespace_root(root_id, str_ptr);

        Ok(str_ptr)
    }

    /// Allocate a float on the heap
    /// Floats are heap-allocated because embedding 64-bit float bits in a tagged
    /// pointer would lose precision (we only have 61 bits after the 3-bit tag)
    pub fn allocate_float(&mut self, value: f64) -> Result<usize, String> {
        // Allocate 1 word for the float value (8 bytes)
        let ptr = self.allocate_raw(1, TYPE_FLOAT as u8)?;

        let mut heap_obj = HeapObject::from_untagged(ptr as *const u8);

        // Mark as opaque so GC doesn't scan float bits as pointers
        let mut header = heap_obj.get_header();
        header.opaque = true;
        heap_obj.write_header_direct(header);

        // Write the float bits directly to the heap
        unsafe {
            let data_ptr = (ptr + 8) as *mut u64; // Skip header (8 bytes)
            *data_ptr = value.to_bits();
        }

        Ok(self.tag_float(ptr))
    }

    /// Read a float value from a tagged float pointer
    pub fn read_float(&self, tagged: usize) -> f64 {
        let ptr = self.untag_float(tagged);
        unsafe {
            let data_ptr = (ptr + 8) as *const u64; // Skip header
            f64::from_bits(*data_ptr)
        }
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

    /// Tag a pointer as a float
    fn tag_float(&self, ptr: usize) -> usize {
        (ptr << 3) | 0b001
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

    /// Untag a float pointer
    pub fn untag_float(&self, tagged: usize) -> usize {
        tagged >> 3
    }

    // ========== Keyword Interning ==========

    /// Add a keyword to the constant table (at compile time)
    /// Returns the index into keyword_constants
    pub fn add_keyword(&mut self, text: String) -> usize {
        // Check if keyword already exists
        if let Some(index) = self.keyword_constants.iter().position(|k| k == &text) {
            return index;
        }
        // Add new keyword
        let index = self.keyword_constants.len();
        self.keyword_constants.push(text);
        self.keyword_heap_ptrs.push(None);
        index
    }

    /// Allocate a keyword on the heap
    /// Layout: [header(8)][hash(8)][text bytes padded to word boundary]
    pub fn allocate_keyword(&mut self, text: &str) -> Result<usize, String> {
        let bytes = text.as_bytes();
        // Words needed: 1 for hash + ceil(bytes.len() / 8) for text
        let text_words = bytes.len().div_ceil(8);
        let total_words = 1 + text_words;

        let ptr = self.allocate_raw(total_words, TYPE_KEYWORD as u8)?;

        let mut heap_obj = HeapObject::from_untagged(ptr as *const u8);

        // Update header with text length and mark as opaque
        let mut header = heap_obj.get_header();
        header.type_data = bytes.len() as u32;
        header.opaque = true;
        heap_obj.write_header_direct(header);

        // Compute hash using DefaultHasher for stable hashing
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        bytes.hash(&mut hasher);
        let hash = hasher.finish();

        // Write hash as first 8 bytes after header
        unsafe {
            let hash_ptr = (ptr + 8) as *mut u64;
            *hash_ptr = hash;

            // Write text bytes after hash
            let text_ptr = (ptr + 16) as *mut u8;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), text_ptr, bytes.len());
        }

        Ok(self.tag_heap_object(ptr))
    }

    /// Intern a keyword - ensures same text always returns same heap pointer
    /// This is called at runtime when a keyword constant is first used
    pub fn intern_keyword(&mut self, index: usize) -> Result<usize, String> {
        // Check if already allocated
        if let Some(ptr) = self.keyword_heap_ptrs.get(index).and_then(|p| *p) {
            return Ok(ptr);
        }

        // Get the text from constants
        let text = self
            .keyword_constants
            .get(index)
            .ok_or_else(|| format!("Invalid keyword index: {}", index))?
            .clone();

        // Allocate the keyword
        let ptr = self.allocate_keyword(&text)?;

        // Cache the pointer
        if let Some(slot) = self.keyword_heap_ptrs.get_mut(index) {
            *slot = Some(ptr);
        }

        // Register as GC root so keywords are never collected
        // Use add_namespace_root which works for all GC implementations
        let root_id = self.next_namespace_id;
        self.next_namespace_id += 1;
        self.keyword_root_ids.insert(index, root_id);
        self.allocator.add_namespace_root(root_id, ptr);

        Ok(ptr)
    }

    /// Get keyword text from a tagged keyword pointer
    pub fn get_keyword_text(&self, tagged: usize) -> Result<&str, String> {
        let ptr = self.untag_heap_object(tagged);
        let heap_obj = HeapObject::from_untagged(ptr as *const u8);

        // Verify it's a keyword
        let header = heap_obj.get_header();
        if header.type_id != TYPE_KEYWORD as u8 {
            return Err(format!("Not a keyword: type_id={}", header.type_id));
        }

        let text_len = header.type_data as usize;
        unsafe {
            // Skip header (8 bytes) and hash (8 bytes)
            let text_ptr = (ptr + 16) as *const u8;
            let bytes = std::slice::from_raw_parts(text_ptr, text_len);
            std::str::from_utf8(bytes).map_err(|e| format!("Invalid UTF-8 in keyword: {}", e))
        }
    }

    /// Check if a tagged value is a keyword
    pub fn is_keyword(&self, tagged: usize) -> bool {
        // Check tag bits first
        let tag = tagged & 0b111;
        if tag != 0b110 {
            // heap object tag
            return false;
        }

        let ptr = self.untag_heap_object(tagged);
        let heap_obj = HeapObject::from_untagged(ptr as *const u8);
        heap_obj.get_header().type_id == TYPE_KEYWORD as u8
    }

    /// Get the keyword constant text by index (for trampolines)
    pub fn get_keyword_constant(&self, index: usize) -> Option<&str> {
        self.keyword_constants.get(index).map(|s| s.as_str())
    }

    // ========== Symbol Interning Methods ==========

    /// Intern a symbol name at compile time.
    /// Returns a stable symbol_id that can be embedded in generated code.
    pub fn intern_symbol(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.symbol_name_to_id.get(name) {
            return id;
        }
        let id = self.symbol_table.len() as u32;
        self.symbol_table.push(name.to_string());
        self.symbol_name_to_id.insert(name.to_string(), id);
        id
    }

    /// Get symbol string by ID (for runtime lookup).
    pub fn get_symbol(&self, id: u32) -> Option<&str> {
        self.symbol_table.get(id as usize).map(|s| s.as_str())
    }

    /// Allocate a namespace object on the heap
    pub fn allocate_namespace(&mut self, name: &str) -> Result<usize, String> {
        let name_ptr = self.allocate_string(name)?;

        // Allocate dynamic bindings array with initial capacity of 16 words (8 bindings)
        let bindings_ptr = self.allocate_dynamic_array(16)?;
        let tagged_bindings = self.tag_heap_object(bindings_ptr);

        // Namespace is fixed size: [name_ptr, bindings_array_ptr]
        let ns_ptr = self.allocate_raw(2, TYPE_NAMESPACE as u8)?;
        let heap_obj = HeapObject::from_untagged(ns_ptr as *const u8);
        heap_obj.write_field(0, name_ptr);
        heap_obj.write_field(1, tagged_bindings);

        Ok(self.tag_heap_object(ns_ptr))
    }

    /// Refer all bindings from source namespace into target namespace
    /// This is used to implement (ns foo) which implicitly refers clojure.core
    /// Returns the target namespace pointer (unchanged since namespaces are now fixed-size)
    pub fn refer_all(
        &mut self,
        target_ns_ptr: usize,
        source_ns_ptr: usize,
    ) -> Result<usize, String> {
        // Get all bindings from source namespace first
        // We collect them to avoid issues with GC during iteration
        let bindings = self.namespace_all_bindings(source_ns_ptr);

        // Add each binding to the target namespace
        for (name, var_ptr) in bindings {
            self.namespace_add_binding(target_ns_ptr, &name, var_ptr)?;
        }

        Ok(target_ns_ptr)
    }

    /// Add or update a binding in a namespace
    /// The namespace object itself is fixed-size; only the bindings array may be reallocated
    /// Uses dynamic array with capacity (header.size) and used count (header.type_data)
    pub fn namespace_add_binding(
        &mut self,
        ns_ptr: usize,
        symbol_name: &str,
        value: usize,
    ) -> Result<usize, String> {
        self.namespace_add_binding_impl(ns_ptr, symbol_name, value, None)
    }

    /// Add a binding using a pre-allocated symbol string pointer
    /// This avoids duplicate string allocation when adding vars
    pub fn namespace_add_binding_with_symbol_ptr(
        &mut self,
        ns_ptr: usize,
        symbol_name: &str,
        value: usize,
        symbol_ptr: usize,
    ) -> Result<usize, String> {
        self.namespace_add_binding_impl(ns_ptr, symbol_name, value, Some(symbol_ptr))
    }

    /// Core implementation for adding namespace bindings
    fn namespace_add_binding_impl(
        &mut self,
        ns_ptr: usize,
        symbol_name: &str,
        value: usize,
        existing_symbol_ptr: Option<usize>,
    ) -> Result<usize, String> {
        // Save namespace name for GC safety
        let ns_name = self.namespace_name(ns_ptr);

        // Get bindings array from namespace field 1
        let ns_untagged = self.untag_heap_object(ns_ptr);
        let ns_heap_obj = HeapObject::from_untagged(ns_untagged as *const u8);
        let bindings_tagged = ns_heap_obj.get_field(1);
        let bindings_untagged = self.untag_heap_object(bindings_tagged);
        let bindings_obj = HeapObject::from_untagged(bindings_untagged as *const u8);
        let bindings_header = bindings_obj.get_header();
        let _capacity = bindings_header.size as usize;
        let used = bindings_obj.get_type_data();

        // Check if binding already exists - update in place if so
        let num_bindings = used / 2;
        for i in 0..num_bindings {
            let name_ptr = bindings_obj.get_field(i * 2);
            let stored_name = self.read_string(name_ptr);
            if stored_name == symbol_name {
                bindings_obj.write_field(i * 2 + 1, value);
                return Ok(ns_ptr);
            }
        }

        // Use existing symbol pointer if provided, otherwise allocate a new string
        let symbol_ptr = if let Some(ptr) = existing_symbol_ptr {
            ptr
        } else {
            // Allocate symbol name string - this might trigger GC!
            self.allocate_string(symbol_name)?
        };

        // Re-fetch namespace after allocation (GC may have moved things)
        let current_ns_ptr = self
            .get_namespace_by_name(&ns_name)
            .ok_or_else(|| format!("Namespace {} disappeared during allocation", ns_name))?;
        let current_ns_untagged = self.untag_heap_object(current_ns_ptr);
        let current_ns_obj = HeapObject::from_untagged(current_ns_untagged as *const u8);

        // Re-fetch bindings array (may have been relocated by GC)
        let bindings_tagged = current_ns_obj.get_field(1);
        let bindings_untagged = self.untag_heap_object(bindings_tagged);
        let bindings_obj = HeapObject::from_untagged(bindings_untagged as *const u8);
        let bindings_header = bindings_obj.get_header();
        let capacity = bindings_header.size as usize;
        let used = bindings_obj.get_type_data();

        // Check if we have room in the current array
        if used + 2 <= capacity {
            // Add binding in place
            bindings_obj.write_field(used, symbol_ptr);
            bindings_obj.write_field(used + 1, value);
            bindings_obj.set_type_data((used + 2) as u32);
            return Ok(ns_ptr);
        }

        // Need to grow: double capacity (minimum 16)
        let new_capacity = if capacity == 0 { 16 } else { capacity * 2 };
        let new_bindings_ptr = self.allocate_dynamic_array(new_capacity)?;

        // Re-fetch namespace again after allocation
        let current_ns_ptr = self
            .get_namespace_by_name(&ns_name)
            .ok_or_else(|| format!("Namespace {} disappeared during allocation", ns_name))?;
        let current_ns_untagged = self.untag_heap_object(current_ns_ptr);
        let current_ns_obj = HeapObject::from_untagged(current_ns_untagged as *const u8);

        // Get old bindings array (may have been relocated by GC)
        let old_bindings_tagged = current_ns_obj.get_field(1);
        let old_bindings_untagged = self.untag_heap_object(old_bindings_tagged);
        let old_bindings_obj = HeapObject::from_untagged(old_bindings_untagged as *const u8);
        let old_used = old_bindings_obj.get_type_data();

        // Copy existing bindings to new array
        let new_bindings_obj = HeapObject::from_untagged(new_bindings_ptr as *const u8);
        for i in 0..old_used {
            let field = old_bindings_obj.get_field(i);
            new_bindings_obj.write_field(i, field);
        }

        // Add new binding
        new_bindings_obj.write_field(old_used, symbol_ptr);
        new_bindings_obj.write_field(old_used + 1, value);
        new_bindings_obj.set_type_data((old_used + 2) as u32);

        // Update namespace to point to new bindings array
        let tagged_new_bindings = self.tag_heap_object(new_bindings_ptr);
        current_ns_obj.write_field(1, tagged_new_bindings);

        // Namespace itself didn't move, return original pointer
        Ok(ns_ptr)
    }

    /// Debug: print bindings in a namespace
    pub fn debug_namespace_bindings(&self, ns_ptr: usize, limit: usize) {
        let ns_untagged = self.untag_heap_object(ns_ptr);
        let ns_obj = HeapObject::from_untagged(ns_untagged as *const u8);

        // Get bindings array from field 1
        let bindings_tagged = ns_obj.get_field(1);
        eprintln!("DEBUG: bindings_tagged = 0x{:x}", bindings_tagged);
        let bindings_untagged = self.untag_heap_object(bindings_tagged);
        eprintln!("DEBUG: bindings_untagged = 0x{:x}", bindings_untagged);
        let bindings_obj = HeapObject::from_untagged(bindings_untagged as *const u8);
        // For dynamic arrays, used count is in type_data
        let used = bindings_obj.get_type_data();
        let header = bindings_obj.get_header();

        let num_bindings = used / 2;
        eprintln!("DEBUG: used={}, header.size={}, num_bindings={}", used, header.size, num_bindings);

        for i in 0..num_bindings.min(limit) {
            let name_ptr = bindings_obj.get_field(i * 2);
            eprintln!("DEBUG: binding[{}] name_ptr = 0x{:x}", i, name_ptr);
            let stored_name = self.read_string(name_ptr);
            eprintln!("DEBUG: binding[{}] = '{}'", i, stored_name);
        }
    }

    /// Look up a binding in a namespace
    pub fn namespace_lookup(&self, ns_ptr: usize, symbol_name: &str) -> Option<usize> {
        let ns_untagged = self.untag_heap_object(ns_ptr);
        let ns_obj = HeapObject::from_untagged(ns_untagged as *const u8);

        // Get namespace name for debug
        let ns_name_ptr = ns_obj.get_field(0);
        let _ns_name = self.read_string(ns_name_ptr);

        // Get bindings array from field 1
        let bindings_tagged = ns_obj.get_field(1);
        let bindings_untagged = self.untag_heap_object(bindings_tagged);
        let bindings_obj = HeapObject::from_untagged(bindings_untagged as *const u8);
        // For dynamic arrays, used count is in type_data
        let used = bindings_obj.get_type_data();
        let num_bindings = used / 2;

        for i in 0..num_bindings {
            let name_ptr = bindings_obj.get_field(i * 2);
            let stored_name = self.read_string(name_ptr);
            if stored_name == symbol_name {
                return Some(bindings_obj.get_field(i * 2 + 1));
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

    /// Get all bindings from a namespace as (name, var_ptr) pairs
    pub fn namespace_all_bindings(&self, ns_ptr: usize) -> Vec<(String, usize)> {
        let ns_untagged = self.untag_heap_object(ns_ptr);
        let ns_obj = HeapObject::from_untagged(ns_untagged as *const u8);

        // Get bindings array from field 1
        let bindings_tagged = ns_obj.get_field(1);
        let bindings_untagged = self.untag_heap_object(bindings_tagged);
        let bindings_obj = HeapObject::from_untagged(bindings_untagged as *const u8);
        // For dynamic arrays, used count is in type_data
        let used = bindings_obj.get_type_data();

        let num_bindings = used / 2;
        let mut bindings = Vec::with_capacity(num_bindings);

        for i in 0..num_bindings {
            let name_ptr = bindings_obj.get_field(i * 2);
            let var_ptr = bindings_obj.get_field(i * 2 + 1);
            let name = self.read_string(name_ptr);
            bindings.push((name, var_ptr));
        }

        bindings
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
        if let Some(&ns_id) = self.namespace_name_to_id.get(name)
            && let Some(old_ptr) = self.namespace_roots.get(name).copied() {
                self.allocator.remove_namespace_root(ns_id, old_ptr);
                self.allocator.add_namespace_root(ns_id, new_ptr);
            }
        self.namespace_roots.insert(name.to_string(), new_ptr);
    }

    /// Get all namespace pointers (for syncing compiler registry after GC)
    pub fn get_namespace_pointers(&self) -> &HashMap<String, usize> {
        &self.namespace_roots
    }

    /// Get namespace pointer by name (for runtime var lookup)
    pub fn get_namespace_by_name(&self, name: &str) -> Option<usize> {
        self.namespace_roots.get(name).copied()
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

    /// Run garbage collection from REPL (no JIT code on stack)
    pub fn run_gc(&mut self) -> Result<(), String> {
        // When called from REPL, we're not in the middle of executing JIT code,
        // so there are no live Clojure values on the Rust stack to scan.
        // Just trace from namespace roots.
        self.allocator.gc(&self.stack_map, &[]);

        // Handle relocations (for compacting GC)
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
        Ok(())
    }

    /// Run GC with stack pointers
    pub fn run_gc_with_stack(&mut self, stack_base: usize, stack_pointer: usize) {
        self.allocator
            .gc(&self.stack_map, &[(stack_base, stack_pointer)]);

        // Handle relocations
        let relocations = self.allocator.get_namespace_relocations();
        for (ns_id, updates) in relocations {
            for (old_ptr, new_ptr) in updates {
                // Find and update namespace_roots
                for (name, ptr) in self.namespace_roots.iter_mut() {
                    if *ptr == old_ptr {
                        *ptr = new_ptr;
                        // Also need to find ns_id by name for update
                        if let Some(&id) = self.namespace_name_to_id.get(name)
                            && id == ns_id {
                                // Already updated the root
                            }
                    }
                }

                // Update protocol_vtable entries if they were relocated
                for (_key, ptr) in self.protocol_vtable.iter_mut() {
                    if *ptr == old_ptr {
                        *ptr = new_ptr;
                    }
                }

                // Update keyword_heap_ptrs if they were relocated
                for ptr_opt in self.keyword_heap_ptrs.iter_mut() {
                    if let Some(ptr) = ptr_opt
                        && *ptr == old_ptr {
                            *ptr = new_ptr;
                        }
                }

                // Update dynamic_vars set if a var was relocated
                if self.dynamic_vars.remove(&old_ptr) {
                    self.dynamic_vars.insert(new_ptr);
                }

                // Update dynamic_bindings keys if a var was relocated
                if let Some(stack) = self.dynamic_bindings.remove(&old_ptr) {
                    self.dynamic_bindings.insert(new_ptr, stack);
                }
            }
        }
    }

    /// Register a temporary GC root.
    /// Returns a root ID that must be passed to unregister_temporary_root when done.
    /// Use this to protect heap objects during operations that may trigger GC.
    pub fn register_temporary_root(&mut self, root: usize) -> usize {
        self.allocator.register_temporary_root(root)
    }

    /// Unregister a temporary GC root.
    /// Returns the value that was registered.
    pub fn unregister_temporary_root(&mut self, id: usize) -> usize {
        self.allocator.unregister_temporary_root(id)
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
            }
            .to_string(),
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

    /// Allocate a var object on the heap
    /// Returns (tagged_var_ptr, symbol_ptr) - symbol_ptr can be reused for namespace binding
    ///
    /// Var layout (4 fields):
    ///   field 0: namespace_ptr (tagged pointer to namespace)
    ///   field 1: symbol_ptr (tagged string pointer with symbol name)
    ///   field 2: current_value (the var's root binding)
    ///   field 3: metadata_ptr (tagged map or nil)
    pub fn allocate_var(
        &mut self,
        ns_ptr: usize,
        symbol_name: &str,
        initial_value: usize,
    ) -> Result<(usize, usize), String> {
        let symbol_ptr = self.allocate_string(symbol_name)?;

        let var_ptr = self.allocate_raw(4, TYPE_VAR as u8)?;
        let heap_obj = HeapObject::from_untagged(var_ptr as *const u8);
        heap_obj.write_field(0, ns_ptr);
        heap_obj.write_field(1, symbol_ptr);
        heap_obj.write_field(2, initial_value);
        heap_obj.write_field(3, 7); // nil - no metadata initially

        let tagged_var_ptr = self.tag_heap_object(var_ptr);

        // Return the symbol pointer so it can be reused for namespace binding
        Ok((tagged_var_ptr, symbol_ptr))
    }

    /// Bootstrap all builtin functions as Vars in clojure.core.
    ///
    /// This creates proper function objects for each builtin (+, -, *, /, etc.)
    /// and binds them as Vars in the clojure.core namespace. This enables:
    /// - `(def my-add +)` to work (passing builtins as values)
    /// - `(map + [1 2] [3 4])` to work (builtins as first-class functions)
    ///
    /// Returns the updated core_ns_ptr after adding all bindings.
    pub fn bootstrap_builtins(&mut self, mut core_ns_ptr: usize) -> Result<usize, String> {
        use crate::trampoline::generate_builtin_wrappers;

        let wrappers = generate_builtin_wrappers();

        for (name, code_ptr) in &wrappers {
            // Create a tagged function pointer (no closures, just raw code)
            // Function tag is 0b100 (4)
            let fn_tagged = (code_ptr << 3) | 0b100;

            // Create a var for this builtin in clojure.core
            let (var_ptr, symbol_ptr) = self.allocate_var(core_ns_ptr, name, fn_tagged)?;

            // Add the var to the namespace bindings, reusing the symbol string from the var
            core_ns_ptr =
                self.namespace_add_binding_with_symbol_ptr(core_ns_ptr, name, var_ptr, symbol_ptr)?;
        }

        Ok(core_ns_ptr)
    }

    /// Get the current value from a var
    pub fn var_get_value(&self, var_ptr: usize) -> usize {
        let untagged = self.untag_heap_object(var_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_field(2)
    }

    /// Set var value (field 2)
    pub fn var_set_value(&self, var_ptr: usize, value: usize) {
        let untagged = self.untag_heap_object(var_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.write_field(2, value);
    }

    /// Get var metadata (field 3)
    pub fn var_get_meta(&self, var_ptr: usize) -> usize {
        let untagged = self.untag_heap_object(var_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_field(3)
    }

    /// Set var metadata (field 3)
    pub fn var_set_meta(&self, var_ptr: usize, meta: usize) {
        let untagged = self.untag_heap_object(var_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.write_field(3, meta);
    }

    /// Alter var metadata - assoc a key-value pair into the metadata map
    /// Uses ReaderMap for storage since it's fully implemented
    /// If metadata is nil, creates a new map with just the key-value pair
    pub fn var_alter_meta(&mut self, var_ptr: usize, key: usize, value: usize) -> Result<(), String> {
        let current_meta = self.var_get_meta(var_ptr);
        let new_meta = if current_meta == 7 {
            // nil -> create new ReaderMap with single entry
            self.allocate_reader_map(&[(key, value)])?
        } else {
            // assoc into existing map
            self.reader_map_assoc(current_meta, key, value)?
        };
        self.var_set_meta(var_ptr, new_meta);
        Ok(())
    }

    /// Check if var is a macro (has :macro true in metadata)
    pub fn is_var_macro(&self, var_ptr: usize) -> bool {
        let meta = self.var_get_meta(var_ptr);
        if meta == 7 {
            return false; // nil metadata
        }
        // Look up :macro key in the map using the cached keyword pointer
        if let Some(&macro_kw) = self.macro_keyword_ptr.as_ref() {
            let not_found = 0usize; // Use 0 as sentinel (not a valid tagged value)
            let val = self.reader_map_lookup(meta, macro_kw, not_found);
            // Check if value is true (tagged bool: 0b1011)
            val == 0b1011
        } else {
            false
        }
    }

    /// Mark var as macro (set :macro true in metadata)
    pub fn set_var_macro(&mut self, var_ptr: usize) -> Result<(), String> {
        // Ensure :macro keyword is interned
        let macro_kw = self.ensure_macro_keyword()?;
        let true_val = 0b1011; // tagged true
        self.var_alter_meta(var_ptr, macro_kw, true_val)
    }

    /// Ensure :macro keyword is interned and cached
    fn ensure_macro_keyword(&mut self) -> Result<usize, String> {
        if let Some(kw) = self.macro_keyword_ptr {
            return Ok(kw);
        }
        let index = self.add_keyword("macro".to_string());
        let kw = self.intern_keyword(index)?;
        self.macro_keyword_ptr = Some(kw);
        Ok(kw)
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
        if let Some(stack) = self.dynamic_bindings.get_mut(&var_ptr)
            && let Some(last) = stack.last_mut() {
                *last = value;
                return Ok(());
            }

        let (ns_name, symbol_name) = self.var_info(var_ptr);
        Err(format!(
            "Can't change/establish root binding of: {}/{} with set",
            ns_name, symbol_name
        ))
    }

    /// Get the current value of a var, checking dynamic bindings first
    pub fn var_get_value_dynamic(&self, var_ptr: usize) -> usize {
        if let Some(stack) = self.dynamic_bindings.get(&var_ptr)
            && let Some(&value) = stack.last() {
                return value;
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

    /// Get the symbol string pointer from a var (field 1)
    pub fn var_symbol_ptr(&self, var_ptr: usize) -> usize {
        let untagged = self.untag_heap_object(var_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_field(1)
    }

    /// Check if a tagged value is a Var
    pub fn is_var(&self, value: usize) -> bool {
        // Vars use heap object tag (0b101)
        if value & 0b111 != 0b101 {
            return false;
        }
        let untagged = self.untag_heap_object(value);
        if !self.allocator.contains_address(untagged) {
            return false;
        }
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_type_id() == TYPE_VAR
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
        let fn_ptr = self.allocate_raw(size_words, TYPE_FUNCTION as u8)?;
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
            panic!(
                "Closure index out of bounds: {} >= {}",
                index, closure_count
            );
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
        variadic_index: Option<usize>,
        closure_values: Vec<usize>,
    ) -> Result<usize, String> {
        let name_ptr = if let Some(n) = name {
            self.allocate_string(&n)?
        } else {
            0
        };

        let arity_count = arities.len();
        let size_words = multi_arity_layout::total_size_words(arity_count, closure_values.len());
        let fn_ptr = self.allocate_raw(size_words, TYPE_MULTI_ARITY_FN as u8)?;
        let heap_obj = HeapObject::from_untagged(fn_ptr as *const u8);

        // Write header fields
        // All integer fields must be tagged to avoid GC treating them as heap pointers
        heap_obj.write_field(multi_arity_layout::FIELD_NAME_PTR, name_ptr);
        heap_obj.write_field(
            multi_arity_layout::FIELD_ARITY_COUNT,
            BuiltInTypes::Int.tag(arity_count as isize) as usize,
        );
        heap_obj.write_field(
            multi_arity_layout::FIELD_VARIADIC_MIN,
            BuiltInTypes::Int.tag(variadic_min.unwrap_or(multi_arity_layout::NO_VARIADIC) as isize)
                as usize,
        );
        heap_obj.write_field(
            multi_arity_layout::FIELD_VARIADIC_INDEX,
            BuiltInTypes::Int.tag(variadic_index.unwrap_or(multi_arity_layout::NO_VARIADIC) as isize)
                as usize,
        );
        heap_obj.write_field(
            multi_arity_layout::FIELD_CLOSURE_COUNT,
            BuiltInTypes::Int.tag(closure_values.len() as isize) as usize,
        );

        // Write arity table
        // Note: param_count must be tagged as Int to avoid GC treating it as a heap pointer
        // code_ptr is a raw address (not a tagged value) but has high bits set so it won't
        // match heap pointer tag patterns in practice
        for (i, (param_count, code_ptr)) in arities.iter().enumerate() {
            let tagged_param_count = BuiltInTypes::Int.tag(*param_count as isize) as usize;
            heap_obj.write_field(
                multi_arity_layout::arity_param_count_field(i),
                tagged_param_count,
            );
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
        heap_obj.get_header().type_id == TYPE_MULTI_ARITY_FN as u8
    }

    /// Look up the code pointer for a given argument count in a multi-arity function
    /// Returns (code_ptr, is_variadic) if found
    pub fn multi_arity_lookup(&self, fn_ptr: usize, arg_count: usize) -> Option<(usize, bool)> {
        let untagged = self.untag_closure(fn_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);

        // Untag integer fields (they were tagged to avoid GC issues)
        let arity_count = heap_obj.get_field(multi_arity_layout::FIELD_ARITY_COUNT) >> 3;
        let variadic_min = heap_obj.get_field(multi_arity_layout::FIELD_VARIADIC_MIN) >> 3;
        let variadic_index = heap_obj.get_field(multi_arity_layout::FIELD_VARIADIC_INDEX) >> 3;

        let no_variadic_untagged = multi_arity_layout::NO_VARIADIC >> 3;
        let has_variadic = variadic_index != no_variadic_untagged;

        // First, try to find an exact match among FIXED arities (skip the variadic arity)
        for i in 0..arity_count {
            // Skip the variadic arity - we only want exact matches for fixed arities
            if has_variadic && i == variadic_index {
                continue;
            }
            let param_count =
                heap_obj.get_field(multi_arity_layout::arity_param_count_field(i)) >> 3;
            if param_count == arg_count {
                let code_ptr = heap_obj.get_field(multi_arity_layout::arity_code_ptr_field(i));
                return Some((code_ptr, false));
            }
        }

        // If no exact fixed match and we have a variadic arity, check if args >= variadic_min
        if has_variadic && arg_count >= variadic_min {
            // Use the variadic index directly instead of searching
            let code_ptr = heap_obj.get_field(multi_arity_layout::arity_code_ptr_field(variadic_index));
            return Some((code_ptr, true));
        }

        None
    }

    /// Get closure count from a multi-arity function
    pub fn multi_arity_closure_count(&self, fn_ptr: usize) -> usize {
        let untagged = self.untag_closure(fn_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_field(multi_arity_layout::FIELD_CLOSURE_COUNT) >> 3
    }

    /// Get arity count from a multi-arity function
    pub fn multi_arity_arity_count(&self, fn_ptr: usize) -> usize {
        let untagged = self.untag_closure(fn_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_field(multi_arity_layout::FIELD_ARITY_COUNT) >> 3
    }

    /// Get variadic_min from a multi-arity function (the fixed param count before &rest)
    pub fn multi_arity_variadic_min(&self, fn_ptr: usize) -> usize {
        let untagged = self.untag_closure(fn_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_field(multi_arity_layout::FIELD_VARIADIC_MIN) >> 3
    }

    /// Get closure value by index from a multi-arity function
    pub fn multi_arity_get_closure(&self, fn_ptr: usize, index: usize) -> usize {
        let untagged = self.untag_closure(fn_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        let arity_count = heap_obj.get_field(multi_arity_layout::FIELD_ARITY_COUNT) >> 3;
        let closure_count = heap_obj.get_field(multi_arity_layout::FIELD_CLOSURE_COUNT) >> 3;
        if index >= closure_count {
            panic!(
                "Multi-arity closure index out of bounds: {} >= {}",
                index, closure_count
            );
        }
        heap_obj.get_field(multi_arity_layout::closure_value_field(arity_count, index))
    }

    // ========== Cons Cell Methods (for variadic args) ==========

    /// Allocate a cons cell (head, tail)
    pub fn allocate_cons(&mut self, head: usize, tail: usize) -> Result<usize, String> {
        let cons_ptr = self.allocate_raw(cons_layout::SIZE_WORDS, TYPE_LIST as u8)?;
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
        heap_obj.get_header().type_id == TYPE_LIST as u8
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

    // ========== Raw Mutable Array Methods ==========

    /// Allocate a raw mutable array of the given length
    /// All elements are initialized to nil
    pub fn allocate_array(&mut self, length: usize) -> Result<usize, String> {
        let size_words = array_layout::total_size_words(length);
        let arr_ptr = self.allocate_raw(size_words, TYPE_ARRAY as u8)?;
        let heap_obj = HeapObject::from_untagged(arr_ptr as *const u8);

        // Write length as tagged integer (so GC doesn't try to trace it as a pointer)
        heap_obj.write_field(array_layout::FIELD_LENGTH, length << 3);

        // Initialize all elements to nil (tagged value 7)
        let nil_value = 7usize;
        for i in 0..length {
            heap_obj.write_field(array_layout::element_field(i), nil_value);
        }

        Ok(self.tag_heap_object(arr_ptr))
    }

    /// Get the length of an array
    pub fn array_length(&self, arr_ptr: usize) -> usize {
        let untagged = self.untag_heap_object(arr_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        // Length is stored as tagged integer
        heap_obj.get_field(array_layout::FIELD_LENGTH) >> 3
    }

    /// Get element at index (bounds checked)
    pub fn array_get(&self, arr_ptr: usize, index: usize) -> Result<usize, String> {
        let length = self.array_length(arr_ptr);
        if index >= length {
            return Err(format!(
                "Array index {} out of bounds for length {}",
                index, length
            ));
        }
        let untagged = self.untag_heap_object(arr_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        Ok(heap_obj.get_field(array_layout::element_field(index)))
    }

    /// Set element at index (bounds checked)
    pub fn array_set(&self, arr_ptr: usize, index: usize, value: usize) -> Result<usize, String> {
        let length = self.array_length(arr_ptr);
        if index >= length {
            return Err(format!(
                "Array index {} out of bounds for length {}",
                index, length
            ));
        }
        let untagged = self.untag_heap_object(arr_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.write_field(array_layout::element_field(index), value);
        Ok(value)
    }

    /// Check if a value is an array
    pub fn is_array(&self, value: usize) -> bool {
        // Check if it's a heap object
        if (value & 0b111) != 0b110 {
            return false;
        }
        let untagged = self.untag_heap_object(value);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_header().type_id == TYPE_ARRAY as u8
    }

    /// Check if a value is a map (PersistentHashMap)
    pub fn is_map(&self, value: usize) -> bool {
        // Check if it's a heap object
        if (value & 0b111) != 0b110 {
            return false;
        }
        let untagged = self.untag_heap_object(value);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_header().type_id == TYPE_MAP as u8
    }

    /// Check if a value is a vector
    pub fn is_vector(&self, value: usize) -> bool {
        // Check if it's a heap object
        if (value & 0b111) != 0b110 {
            return false;
        }
        let untagged = self.untag_heap_object(value);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_header().type_id == TYPE_VECTOR as u8
    }

    // ========== Reader Type Methods ==========
    // These types are produced by the Rust reader and used directly by macros.

    /// Simple equality check for tagged values
    /// For primitives: uses identity
    /// For strings/keywords: compares content
    /// For other heap objects: uses identity (for now)
    fn values_equal(&self, a: usize, b: usize) -> bool {
        if a == b {
            return true;
        }

        // Get tags
        let tag_a = a & 0b111;
        let tag_b = b & 0b111;

        if tag_a != tag_b {
            return false;
        }

        // For strings, compare content
        if tag_a == 0b010 {
            // String tag
            let str_a = self.read_string(a);
            let str_b = self.read_string(b);
            return str_a == str_b;
        }

        // For heap objects, check type-specific equality
        if tag_a == 0b110 {
            // HeapObject tag
            let untagged_a = self.untag_heap_object(a);
            let untagged_b = self.untag_heap_object(b);
            let obj_a = HeapObject::from_untagged(untagged_a as *const u8);
            let obj_b = HeapObject::from_untagged(untagged_b as *const u8);

            let type_a = obj_a.get_header().type_id;
            let type_b = obj_b.get_header().type_id;

            if type_a != type_b {
                return false;
            }

            // For keywords, compare the text
            if type_a == TYPE_KEYWORD as u8 {
                if let (Ok(text_a), Ok(text_b)) =
                    (self.get_keyword_text(a), self.get_keyword_text(b))
                {
                    return text_a == text_b;
                }
                return false;
            }

            // For ReaderSymbol, compare namespace and name
            if type_a == TYPE_READER_SYMBOL as u8 {
                let ns_a = self.reader_symbol_namespace(a);
                let ns_b = self.reader_symbol_namespace(b);
                let name_a = self.reader_symbol_name(a);
                let name_b = self.reader_symbol_name(b);
                return ns_a == ns_b && name_a == name_b;
            }

            // Default: identity
            return false;
        }

        false
    }

    /// Allocate a ReaderList with the given elements
    pub fn allocate_reader_list(&mut self, elements: &[usize]) -> Result<usize, String> {
        let size_words = reader_list_layout::total_size_words(elements.len());
        let ptr = self.allocate_raw(size_words, TYPE_READER_LIST as u8)?;
        let heap_obj = HeapObject::from_untagged(ptr as *const u8);

        // Write count (raw usize, not tagged)
        heap_obj.write_field(reader_list_layout::FIELD_COUNT, elements.len());

        // Write elements
        for (i, &elem) in elements.iter().enumerate() {
            heap_obj.write_field(reader_list_layout::element_field(i), elem);
        }

        Ok(self.tag_heap_object(ptr))
    }

    /// Get the count of a ReaderList
    pub fn reader_list_count(&self, list_ptr: usize) -> usize {
        let untagged = self.untag_heap_object(list_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_field(reader_list_layout::FIELD_COUNT)
    }

    /// Get the first element of a ReaderList (returns nil if empty)
    pub fn reader_list_first(&self, list_ptr: usize) -> usize {
        let count = self.reader_list_count(list_ptr);
        if count == 0 {
            return 7; // nil
        }
        let untagged = self.untag_heap_object(list_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_field(reader_list_layout::element_field(0))
    }

    /// Get the rest of a ReaderList (returns a new ReaderList without first element)
    pub fn reader_list_rest(&mut self, list_ptr: usize) -> Result<usize, String> {
        let count = self.reader_list_count(list_ptr);
        if count <= 1 {
            return self.allocate_reader_list(&[]); // empty list
        }

        let untagged = self.untag_heap_object(list_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);

        // Collect rest elements
        let mut rest_elements = Vec::with_capacity(count - 1);
        for i in 1..count {
            rest_elements.push(heap_obj.get_field(reader_list_layout::element_field(i)));
        }

        self.allocate_reader_list(&rest_elements)
    }

    /// Get element at index from a ReaderList
    pub fn reader_list_nth(&self, list_ptr: usize, index: usize) -> Result<usize, String> {
        let count = self.reader_list_count(list_ptr);
        if index >= count {
            return Err(format!(
                "ReaderList index {} out of bounds for count {}",
                index, count
            ));
        }
        let untagged = self.untag_heap_object(list_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        Ok(heap_obj.get_field(reader_list_layout::element_field(index)))
    }

    /// Conj onto a ReaderList (prepends element, like Clojure's cons)
    pub fn reader_list_conj(&mut self, list_ptr: usize, elem: usize) -> Result<usize, String> {
        let count = self.reader_list_count(list_ptr);
        let untagged = self.untag_heap_object(list_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);

        // Collect all elements with new one at front
        let mut new_elements = Vec::with_capacity(count + 1);
        new_elements.push(elem);
        for i in 0..count {
            new_elements.push(heap_obj.get_field(reader_list_layout::element_field(i)));
        }

        self.allocate_reader_list(&new_elements)
    }

    /// Create a ReaderList from a Vec of tagged pointers
    pub fn reader_list_from_vec(&mut self, elements: &[usize]) -> Result<usize, String> {
        self.allocate_reader_list(elements)
    }

    /// Check if a value is a ReaderList
    pub fn is_reader_list(&self, value: usize) -> bool {
        if (value & 0b111) != 0b110 {
            return false;
        }
        let untagged = self.untag_heap_object(value);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_header().type_id == TYPE_READER_LIST as u8
    }

    /// Allocate a ReaderVector with the given elements
    pub fn allocate_reader_vector(&mut self, elements: &[usize]) -> Result<usize, String> {
        let size_words = reader_vector_layout::total_size_words(elements.len());
        let ptr = self.allocate_raw(size_words, TYPE_READER_VECTOR as u8)?;
        let heap_obj = HeapObject::from_untagged(ptr as *const u8);

        // Write count (raw usize, not tagged)
        heap_obj.write_field(reader_vector_layout::FIELD_COUNT, elements.len());

        // Write elements
        for (i, &elem) in elements.iter().enumerate() {
            heap_obj.write_field(reader_vector_layout::element_field(i), elem);
        }

        Ok(self.tag_heap_object(ptr))
    }

    /// Get the count of a ReaderVector
    pub fn reader_vector_count(&self, vec_ptr: usize) -> usize {
        let untagged = self.untag_heap_object(vec_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_field(reader_vector_layout::FIELD_COUNT)
    }

    /// Get element at index from a ReaderVector
    pub fn reader_vector_nth(&self, vec_ptr: usize, index: usize) -> Result<usize, String> {
        let count = self.reader_vector_count(vec_ptr);
        if index >= count {
            return Err(format!(
                "ReaderVector index {} out of bounds for count {}",
                index, count
            ));
        }
        let untagged = self.untag_heap_object(vec_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        Ok(heap_obj.get_field(reader_vector_layout::element_field(index)))
    }

    /// Get element at index from a ReaderVector with default
    pub fn reader_vector_nth_or(&self, vec_ptr: usize, index: usize, not_found: usize) -> usize {
        let count = self.reader_vector_count(vec_ptr);
        if index >= count {
            return not_found;
        }
        let untagged = self.untag_heap_object(vec_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_field(reader_vector_layout::element_field(index))
    }

    /// Conj onto a ReaderVector (appends element, like Clojure's conj)
    pub fn reader_vector_conj(&mut self, vec_ptr: usize, elem: usize) -> Result<usize, String> {
        let count = self.reader_vector_count(vec_ptr);
        let untagged = self.untag_heap_object(vec_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);

        // Collect all elements with new one at end
        let mut new_elements = Vec::with_capacity(count + 1);
        for i in 0..count {
            new_elements.push(heap_obj.get_field(reader_vector_layout::element_field(i)));
        }
        new_elements.push(elem);

        self.allocate_reader_vector(&new_elements)
    }

    /// Check if a value is a ReaderVector
    pub fn is_reader_vector(&self, value: usize) -> bool {
        if (value & 0b111) != 0b110 {
            return false;
        }
        let untagged = self.untag_heap_object(value);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_header().type_id == TYPE_READER_VECTOR as u8
    }

    /// Allocate a ReaderMap with the given key-value pairs
    pub fn allocate_reader_map(&mut self, entries: &[(usize, usize)]) -> Result<usize, String> {
        let size_words = reader_map_layout::total_size_words(entries.len());
        let ptr = self.allocate_raw(size_words, TYPE_READER_MAP as u8)?;
        let heap_obj = HeapObject::from_untagged(ptr as *const u8);

        // Write count (raw usize, not tagged)
        heap_obj.write_field(reader_map_layout::FIELD_COUNT, entries.len());

        // Write key-value pairs
        for (i, &(key, value)) in entries.iter().enumerate() {
            heap_obj.write_field(reader_map_layout::key_field(i), key);
            heap_obj.write_field(reader_map_layout::value_field(i), value);
        }

        Ok(self.tag_heap_object(ptr))
    }

    /// Get the count (number of entries) in a ReaderMap
    pub fn reader_map_count(&self, map_ptr: usize) -> usize {
        let untagged = self.untag_heap_object(map_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_field(reader_map_layout::FIELD_COUNT)
    }

    /// Lookup a key in a ReaderMap (returns not_found if not present)
    pub fn reader_map_lookup(&self, map_ptr: usize, key: usize, not_found: usize) -> usize {
        let count = self.reader_map_count(map_ptr);
        let untagged = self.untag_heap_object(map_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);

        // Linear search for key (simple for reader data structures)
        for i in 0..count {
            let k = heap_obj.get_field(reader_map_layout::key_field(i));
            if self.values_equal(k, key) {
                return heap_obj.get_field(reader_map_layout::value_field(i));
            }
        }

        not_found
    }

    /// Assoc a key-value pair into a ReaderMap (returns new map)
    pub fn reader_map_assoc(
        &mut self,
        map_ptr: usize,
        key: usize,
        value: usize,
    ) -> Result<usize, String> {
        let count = self.reader_map_count(map_ptr);
        let untagged = self.untag_heap_object(map_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);

        // Check if key exists and collect entries
        let mut entries = Vec::with_capacity(count + 1);
        let mut found = false;

        for i in 0..count {
            let k = heap_obj.get_field(reader_map_layout::key_field(i));
            let v = heap_obj.get_field(reader_map_layout::value_field(i));
            if self.values_equal(k, key) {
                entries.push((k, value)); // Replace with new value
                found = true;
            } else {
                entries.push((k, v));
            }
        }

        if !found {
            entries.push((key, value));
        }

        self.allocate_reader_map(&entries)
    }

    /// Get all keys from a ReaderMap as a ReaderList
    pub fn reader_map_keys(&mut self, map_ptr: usize) -> Result<usize, String> {
        let count = self.reader_map_count(map_ptr);
        let untagged = self.untag_heap_object(map_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);

        let mut keys = Vec::with_capacity(count);
        for i in 0..count {
            keys.push(heap_obj.get_field(reader_map_layout::key_field(i)));
        }

        self.allocate_reader_list(&keys)
    }

    /// Get all values from a ReaderMap as a ReaderList
    pub fn reader_map_vals(&mut self, map_ptr: usize) -> Result<usize, String> {
        let count = self.reader_map_count(map_ptr);
        let untagged = self.untag_heap_object(map_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);

        let mut vals = Vec::with_capacity(count);
        for i in 0..count {
            vals.push(heap_obj.get_field(reader_map_layout::value_field(i)));
        }

        self.allocate_reader_list(&vals)
    }

    /// Get a key-value pair from a ReaderMap by index
    /// Returns (key_ptr, value_ptr)
    pub fn reader_map_entry(&self, map_ptr: usize, index: usize) -> (usize, usize) {
        let untagged = self.untag_heap_object(map_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        let key = heap_obj.get_field(reader_map_layout::key_field(index));
        let value = heap_obj.get_field(reader_map_layout::value_field(index));
        (key, value)
    }

    /// Check if a value is a ReaderMap
    pub fn is_reader_map(&self, value: usize) -> bool {
        if (value & 0b111) != 0b110 {
            return false;
        }
        let untagged = self.untag_heap_object(value);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_header().type_id == TYPE_READER_MAP as u8
    }

    /// Allocate a ReaderSymbol with namespace and name (no metadata)
    pub fn allocate_reader_symbol(
        &mut self,
        namespace: Option<&str>,
        name: &str,
    ) -> Result<usize, String> {
        self.allocate_reader_symbol_with_meta(namespace, name, 7) // nil metadata
    }

    /// Allocate a ReaderSymbol with namespace, name, and metadata
    pub fn allocate_reader_symbol_with_meta(
        &mut self,
        namespace: Option<&str>,
        name: &str,
        metadata: usize, // tagged pointer to ReaderMap or nil
    ) -> Result<usize, String> {
        // Allocate string for name
        let name_ptr = self.allocate_string(name)?;

        // Allocate string for namespace if present, otherwise nil
        let ns_ptr = match namespace {
            Some(ns) => self.allocate_string(ns)?,
            None => 7, // nil
        };

        let ptr = self.allocate_raw(reader_symbol_layout::SIZE_WORDS, TYPE_READER_SYMBOL as u8)?;
        let heap_obj = HeapObject::from_untagged(ptr as *const u8);

        heap_obj.write_field(reader_symbol_layout::FIELD_NAMESPACE, ns_ptr);
        heap_obj.write_field(reader_symbol_layout::FIELD_NAME, name_ptr);
        heap_obj.write_field(reader_symbol_layout::FIELD_METADATA, metadata);

        Ok(self.tag_heap_object(ptr))
    }

    /// Get the name of a ReaderSymbol as a string
    pub fn reader_symbol_name(&self, sym_ptr: usize) -> String {
        let untagged = self.untag_heap_object(sym_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        let name_ptr = heap_obj.get_field(reader_symbol_layout::FIELD_NAME);
        self.read_string(name_ptr)
    }

    /// Get the namespace of a ReaderSymbol as an Option<String>
    pub fn reader_symbol_namespace(&self, sym_ptr: usize) -> Option<String> {
        let untagged = self.untag_heap_object(sym_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        let ns_ptr = heap_obj.get_field(reader_symbol_layout::FIELD_NAMESPACE);
        if ns_ptr == 7 {
            // nil
            None
        } else {
            Some(self.read_string(ns_ptr))
        }
    }

    /// Get the name pointer (tagged) of a ReaderSymbol
    pub fn reader_symbol_name_ptr(&self, sym_ptr: usize) -> usize {
        let untagged = self.untag_heap_object(sym_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_field(reader_symbol_layout::FIELD_NAME)
    }

    /// Get the namespace pointer (tagged, or nil) of a ReaderSymbol
    pub fn reader_symbol_namespace_ptr(&self, sym_ptr: usize) -> usize {
        let untagged = self.untag_heap_object(sym_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_field(reader_symbol_layout::FIELD_NAMESPACE)
    }

    /// Get the metadata pointer (tagged ReaderMap, or nil) of a ReaderSymbol
    pub fn reader_symbol_metadata(&self, sym_ptr: usize) -> usize {
        let untagged = self.untag_heap_object(sym_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_field(reader_symbol_layout::FIELD_METADATA)
    }

    /// Check if a value is a ReaderSymbol
    pub fn is_reader_symbol(&self, value: usize) -> bool {
        if (value & 0b111) != 0b110 {
            return false;
        }
        let untagged = self.untag_heap_object(value);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.get_header().type_id == TYPE_READER_SYMBOL as u8
    }

    // ========== End Reader Type Methods ==========

    /// Clone an array (allocates a new array with same contents)
    pub fn array_clone(&mut self, arr_ptr: usize) -> Result<usize, String> {
        let length = self.array_length(arr_ptr);
        let new_arr = self.allocate_array(length)?;

        // Copy each element
        for i in 0..length {
            let value = self.array_get(arr_ptr, i)?;
            self.array_set(new_arr, i, value)?;
        }

        Ok(new_arr)
    }

    // ========== IndexedSeq Support (for variadic args) ==========

    /// Create an IndexedSeq wrapping an array of values
    /// Used by trampoline_collect_rest_args for variadic function arguments
    /// Returns nil if values is empty, otherwise an IndexedSeq
    pub fn allocate_indexed_seq(&mut self, values: &[usize]) -> Result<usize, String> {
        if values.is_empty() {
            return Ok(7); // nil
        }

        // Allocate and fill array with values
        let arr = self.allocate_array(values.len())?;
        for (i, &value) in values.iter().enumerate() {
            self.array_set(arr, i, value)?;
        }

        // Look up IndexedSeq type ID
        let indexed_seq_type_id = self
            .get_type_id("clojure.core/IndexedSeq")
            .ok_or_else(|| "IndexedSeq type not registered".to_string())?;

        // Create IndexedSeq instance with fields: [arr, i=0, meta=nil]
        let field_values = vec![arr, 0usize, 7usize]; // arr, i=0 (tagged int), meta=nil
        self.allocate_type_instance(indexed_seq_type_id, field_values)
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
        let type_def = self
            .type_registry
            .get(type_id)
            .ok_or_else(|| format!("Unknown type_id: {}", type_id))?;

        if field_values.len() != type_def.fields.len() {
            return Err(format!(
                "Type {} expects {} fields, got {}",
                type_def.name,
                type_def.fields.len(),
                field_values.len()
            ));
        }

        let size_words = field_values.len();
        let obj_ptr = self.allocate_raw(size_words, TYPE_DEFTYPE as u8)?;
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

    /// Allocate a deftype instance on the heap without writing fields.
    /// Returns UNTAGGED pointer. Caller is responsible for:
    /// 1. Writing field values using HeapStore at offsets 1, 2, 3, ... (after header)
    /// 2. Tagging the pointer with HeapObject tag (0b110)
    ///
    /// This is used by the refactored MakeType compilation which emits
    /// HeapStore instructions for each field instead of passing an array.
    pub fn allocate_type_object_raw(
        &mut self,
        type_id: usize,
        field_count: usize,
    ) -> Result<usize, String> {
        // Validate type exists
        let type_def = self
            .type_registry
            .get(type_id)
            .ok_or_else(|| format!("Unknown type_id: {}", type_id))?;

        if field_count != type_def.fields.len() {
            return Err(format!(
                "Type {} expects {} fields, got {}",
                type_def.name,
                type_def.fields.len(),
                field_count
            ));
        }

        // Allocate raw heap space
        let obj_ptr = self.allocate_raw(field_count, TYPE_DEFTYPE as u8)?;
        let mut heap_obj = HeapObject::from_untagged(obj_ptr as *const u8);

        // Store type_id in header's type_data field
        let mut header = heap_obj.get_header();
        header.type_data = type_id as u32;
        heap_obj.write_header_direct(header);

        // Return UNTAGGED pointer - caller will write fields and tag
        Ok(obj_ptr)
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
    pub fn load_type_field_by_name(
        &self,
        obj_ptr: usize,
        field_name: &str,
    ) -> Result<usize, String> {
        let type_id = self.get_instance_type_id(obj_ptr);

        let type_def = self
            .type_registry
            .get(type_id)
            .ok_or_else(|| format!("Unknown type_id {} in object", type_id))?;

        let field_index = type_def
            .fields
            .iter()
            .position(|f| f == field_name)
            .ok_or_else(|| {
                format!(
                    "Field '{}' not found in type '{}'",
                    field_name, type_def.name
                )
            })?;

        Ok(self.read_type_field(obj_ptr, field_index))
    }

    /// Store a value to a field in a deftype instance by field name
    /// Used by trampoline_store_type_field for mutable field assignment
    pub fn store_type_field_by_name(
        &mut self,
        obj_ptr: usize,
        field_name: &str,
        value: usize,
    ) -> Result<usize, String> {
        let type_id = self.get_instance_type_id(obj_ptr);

        let type_def = self
            .type_registry
            .get(type_id)
            .ok_or_else(|| format!("Unknown type_id {} in object", type_id))?;

        let field_index = type_def
            .fields
            .iter()
            .position(|f| f == field_name)
            .ok_or_else(|| {
                format!(
                    "Field '{}' not found in type '{}'",
                    field_name, type_def.name
                )
            })?;

        self.write_type_field(obj_ptr, field_index, value);
        Ok(value)
    }

    /// Write a value to a field in a deftype instance
    fn write_type_field(&mut self, obj_ptr: usize, field_index: usize, value: usize) {
        // Use the proper untagging and HeapObject API for consistency
        let untagged = self.untag_heap_object(obj_ptr);
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        heap_obj.write_field(field_index, value);
    }

    // ========== Heap Inspection Methods ==========

    /// Get detailed heap statistics
    pub fn detailed_heap_stats(&self) -> DetailedHeapStats {
        self.allocator.detailed_stats()
    }

    /// List all live objects in the heap
    pub fn list_objects(&self) -> Vec<ObjectInfo> {
        self.allocator
            .iter_objects()
            .map(|obj| self.object_to_info(&obj))
            .collect()
    }

    /// List objects filtered by type name
    pub fn list_objects_by_type(&self, type_name: &str) -> Vec<ObjectInfo> {
        self.allocator
            .iter_objects()
            .filter(|obj| {
                let header = obj.get_header();
                let type_id = header.type_id;
                // For DefType, look up actual type name from registry
                let resolved_name = if type_id == TYPE_DEFTYPE as u8 {
                    let deftype_id = header.type_data as usize;
                    self.type_registry
                        .get(deftype_id)
                        .map(|td| td.name.as_str())
                        .unwrap_or("DefType")
                } else {
                    type_id_to_name(type_id)
                };
                resolved_name == type_name
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
            t if t == TYPE_STRING as u8 => self.tag_string(address),
            t if t == TYPE_FUNCTION as u8 => self.tag_closure(address),
            _ => self.tag_heap_object(address),
        };

        // Get type name - for DefType, look up actual type name from registry
        let type_name = if type_id == TYPE_DEFTYPE as u8 {
            let deftype_id = header.type_data as usize;
            self.type_registry
                .get(deftype_id)
                .map(|td| td.name.clone())
                .unwrap_or_else(|| format!("DefType#{}", deftype_id))
        } else {
            type_id_to_name(type_id).to_string()
        };

        // Generate value preview based on type
        let value_preview = self.generate_value_preview(tagged_ptr, type_id);

        ObjectInfo {
            address,
            tagged_ptr,
            type_id,
            type_name,
            type_data: header.type_data,
            size_bytes: obj.full_size(),
            field_count: header.size as usize,
            is_opaque: header.opaque,
            value_preview,
        }
    }

    /// Generate a preview string for an object's value
    fn generate_value_preview(&self, tagged_ptr: usize, type_id: u8) -> Option<String> {
        const MAX_PREVIEW_LEN: usize = 40;

        match type_id as usize {
            TYPE_STRING => {
                let s = self.read_string(tagged_ptr);
                if s.len() > MAX_PREVIEW_LEN {
                    Some(format!("\"{}...\"", &s[..MAX_PREVIEW_LEN]))
                } else {
                    Some(format!("\"{}\"", s))
                }
            }
            TYPE_KEYWORD => {
                let s = self.read_string(tagged_ptr);
                Some(format!(":{}", s))
            }
            TYPE_SYMBOL => {
                let s = self.read_string(tagged_ptr);
                Some(s)
            }
            TYPE_NAMESPACE => {
                let name = self.namespace_name(tagged_ptr);
                Some(format!("#<ns:{}>", name))
            }
            TYPE_VAR => {
                let (ns, sym) = self.var_info(tagged_ptr);
                Some(format!("#'{}/{}", ns, sym))
            }
            TYPE_FUNCTION | TYPE_CLOSURE => {
                let name = self.function_name(tagged_ptr);
                Some(format!("#<fn:{}>", name))
            }
            _ => None,
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
            // Untag by shifting right 3 bits using arithmetic shift (sign-preserving)
            BuiltInTypes::Int => format!("{}", (value as isize) >> 3),
            BuiltInTypes::Bool => {
                if value == 11 {
                    "true".to_string()
                } else if value == 3 {
                    "false".to_string()
                } else {
                    format!("bool?{}", value)
                }
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
                let type_id = obj.get_type_id();
                match type_id {
                    TYPE_NAMESPACE => {
                        let name = self.namespace_name(value);
                        format!("#<ns:{}>", name)
                    }
                    TYPE_VAR => {
                        let (ns, sym) = self.var_info(value);
                        format!("#'{}/{}", ns, sym)
                    }
                    TYPE_DEFTYPE => {
                        let type_data = obj.get_header().type_data as usize;
                        if let Some(def) = self.get_type_def(type_data) {
                            // Special handling for Cons cells - print as list
                            if def.name == "clojure.core/Cons" {
                                self.format_cons(value)
                            } else {
                                format!("#<{}@{:x}>", def.name, untagged)
                            }
                        } else {
                            format!("#<deftype@{:x}>", untagged)
                        }
                    }
                    TYPE_LIST => {
                        // Format cons cells as a list
                        self.format_list(value)
                    }
                    TYPE_KEYWORD => {
                        // Format keyword with colon prefix
                        match self.get_keyword_text(value) {
                            Ok(text) => format!(":{}", text),
                            Err(_) => format!("#<keyword@{:x}>", untagged),
                        }
                    }
                    TYPE_ARRAY => {
                        let len = self.array_length(value);
                        if len == 0 {
                            "#<array[]>".to_string()
                        } else if len <= 5 {
                            let elements: Vec<String> = (0..len)
                                .map(|i| self.format_value(self.array_get(value, i).unwrap_or(7)))
                                .collect();
                            format!("#<array[{}]>", elements.join(" "))
                        } else {
                            format!("#<array[{} elements]>", len)
                        }
                    }
                    TYPE_READER_LIST => {
                        let count = self.reader_list_count(value);
                        if count == 0 {
                            "()".to_string()
                        } else {
                            let elements: Vec<String> = (0..count)
                                .map(|i| {
                                    self.format_value(
                                        self.reader_list_nth(value, i).unwrap_or(7),
                                    )
                                })
                                .collect();
                            format!("({})", elements.join(" "))
                        }
                    }
                    TYPE_READER_VECTOR => {
                        let count = self.reader_vector_count(value);
                        if count == 0 {
                            "[]".to_string()
                        } else {
                            let elements: Vec<String> = (0..count)
                                .map(|i| {
                                    self.format_value(
                                        self.reader_vector_nth(value, i).unwrap_or(7),
                                    )
                                })
                                .collect();
                            format!("[{}]", elements.join(" "))
                        }
                    }
                    TYPE_READER_MAP => {
                        let count = self.reader_map_count(value);
                        if count == 0 {
                            "{}".to_string()
                        } else {
                            let untagged = self.untag_heap_object(value);
                            let heap_obj = HeapObject::from_untagged(untagged as *const u8);
                            let pairs: Vec<String> = (0..count)
                                .map(|i| {
                                    let k = heap_obj.get_field(reader_map_layout::key_field(i));
                                    let v = heap_obj.get_field(reader_map_layout::value_field(i));
                                    format!("{} {}", self.format_value(k), self.format_value(v))
                                })
                                .collect();
                            format!("{{{}}}", pairs.join(", "))
                        }
                    }
                    TYPE_READER_SYMBOL => {
                        let ns = self.reader_symbol_namespace(value);
                        let name = self.reader_symbol_name(value);
                        match ns {
                            Some(ns_str) => format!("{}/{}", ns_str, name),
                            None => name.to_string(),
                        }
                    }
                    _ => format!("#<object@{:x}>", untagged),
                }
            }
            BuiltInTypes::Function => format!("#<fn@{:x}>", value >> 3),
            BuiltInTypes::Float => {
                // Floats are heap-allocated
                let float_val = self.read_float(value);
                format!("{}", float_val)
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
            let type_id = obj.get_type_id();

            if type_id != TYPE_LIST {
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

    /// Format a Cons cell (deftype) as a list
    fn format_cons(&self, value: usize) -> String {
        let mut items = Vec::new();
        let mut current = value;
        let mut max_depth = 100;

        while max_depth > 0 {
            max_depth -= 1;

            // Check if nil
            if current == 7 {
                break;
            }

            // Check if it's a heap object
            if BuiltInTypes::get_kind(current) != BuiltInTypes::HeapObject {
                items.push(format!(". {}", self.format_value(current)));
                break;
            }

            let untagged = current >> 3;
            let obj = HeapObject::from_untagged(untagged as *const u8);
            let type_id = obj.get_type_id();

            // Check if it's a deftype (Cons)
            if type_id != TYPE_DEFTYPE {
                // Check for EmptyList (TYPE_LIST with count 0)
                if type_id == TYPE_LIST {
                    break;
                }
                items.push(format!(". {}", self.format_value(current)));
                break;
            }

            // Check if it's a Cons type by looking at the type_data
            let type_data = obj.get_header().type_data as usize;
            if let Some(def) = self.get_type_def(type_data) {
                if def.name != "clojure.core/Cons" {
                    items.push(format!(". {}", self.format_value(current)));
                    break;
                }
            } else {
                items.push(format!(". {}", self.format_value(current)));
                break;
            }

            // Cons has fields: [meta, first, rest, __hash]
            // first is at field index 1, rest is at field index 2
            let first = obj.get_field(1);
            let rest = obj.get_field(2);

            items.push(self.format_value(first));
            current = rest;
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
            self.method_to_protocol
                .insert(method.name.clone(), (protocol_id, method_index));
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

    /// Check if a type satisfies a protocol (has at least one method implemented or is registered for marker protocol)
    pub fn type_satisfies_protocol(&self, type_id: usize, protocol_id: usize) -> bool {
        // First check if this is a marker protocol satisfaction
        if self.marker_protocol_satisfies.contains(&(type_id, protocol_id)) {
            return true;
        }

        // Get the protocol definition to find how many methods it has
        if let Some(protocol_def) = self.protocol_registry.get(protocol_id) {
            // Check if any method is implemented for this type
            for method_index in 0..protocol_def.methods.len() {
                if self.protocol_vtable.contains_key(&(type_id, protocol_id, method_index)) {
                    return true;
                }
            }
        }
        false
    }

    /// Check if a value satisfies a protocol
    pub fn value_satisfies_protocol(&self, value: usize, protocol_id: usize) -> bool {
        let type_id = self.get_type_id_for_value(value);
        self.type_satisfies_protocol(type_id, protocol_id)
    }

    /// Get method index within a protocol by method name
    pub fn get_protocol_method_index(
        &self,
        protocol_id: usize,
        method_name: &str,
    ) -> Option<usize> {
        self.protocol_registry
            .get(protocol_id)?
            .method_index(method_name)
    }

    /// Register a protocol method implementation in the vtable
    pub fn register_protocol_method_impl(
        &mut self,
        type_id: usize,
        protocol_id: usize,
        method_index: usize,
        fn_ptr: usize,
    ) {
        let key = (type_id, protocol_id, method_index);

        // If we're replacing an existing implementation, remove the old root
        if let Some(&old_root_id) = self.protocol_vtable_root_ids.get(&key)
            && let Some(&old_ptr) = self.protocol_vtable.get(&key) {
                self.allocator.remove_namespace_root(old_root_id, old_ptr);
            }

        // Add the new implementation
        self.protocol_vtable.insert(key, fn_ptr);

        // Register the fn_ptr as a GC root so it's not collected
        // Use namespace_id starting from a high value to avoid collision with real namespaces
        let root_id = self.next_namespace_id;
        self.next_namespace_id += 1;
        self.protocol_vtable_root_ids.insert(key, root_id);
        self.allocator.add_namespace_root(root_id, fn_ptr);
    }

    /// Register that a type satisfies a marker protocol (protocol with no methods)
    pub fn register_marker_protocol_impl(&mut self, type_id: usize, protocol_id: usize) {
        self.marker_protocol_satisfies.insert((type_id, protocol_id));
    }

    /// Look up protocol method implementation by type_id and method_name
    /// Returns the function pointer (tagged closure) if found
    pub fn lookup_protocol_method(&self, type_id: usize, method_name: &str) -> Option<usize> {
        // Look up which protocol this method belongs to
        let &(protocol_id, method_index) = self.method_to_protocol.get(method_name)?;

        // Look up the implementation in the vtable
        self.protocol_vtable
            .get(&(type_id, protocol_id, method_index))
            .copied()
    }

    /// Look up protocol method by protocol_id and method_index (direct vtable lookup)
    pub fn lookup_protocol_method_direct(
        &self,
        type_id: usize,
        protocol_id: usize,
        method_index: usize,
    ) -> Option<usize> {
        self.protocol_vtable
            .get(&(type_id, protocol_id, method_index))
            .copied()
    }

    /// Get the protocol type ID for a tagged value
    /// This is used for protocol dispatch - extracts the type_id used in the vtable
    pub fn get_type_id_for_value(&self, value: usize) -> usize {
        let tag = value & 0b111;

        match tag {
            0b000 => TYPE_INT,      // Integer
            0b001 => TYPE_FLOAT,    // Float
            0b010 => TYPE_STRING,   // String (could be keyword/symbol too)
            0b011 => TYPE_BOOL,     // Bool
            0b100 => TYPE_FUNCTION, // Function
            0b101 => TYPE_CLOSURE,  // Closure
            0b110 => {
                // HeapObject - read type_id directly from header
                let untagged = value >> 3;
                let heap_obj = HeapObject::from_untagged(untagged as *const u8);
                let header = heap_obj.get_header();

                if header.type_id as usize == TYPE_DEFTYPE {
                    // For deftypes, use type_data + offset to get unique type_id
                    header.type_data as usize + DEFTYPE_ID_OFFSET
                } else {
                    header.type_id as usize
                }
            }
            0b111 => TYPE_NIL, // Nil
            _ => unreachable!(),
        }
    }

    /// Get the name of a built-in type ID (for error messages)
    pub fn builtin_type_name(type_id: usize) -> &'static str {
        match type_id {
            TYPE_NIL => "nil",
            TYPE_BOOL => "Boolean",
            TYPE_INT => "Long",
            TYPE_FLOAT => "Double",
            TYPE_STRING => "String",
            TYPE_KEYWORD => "Keyword",
            TYPE_SYMBOL => "Symbol",
            TYPE_LIST => "PersistentList",
            TYPE_VECTOR => "PersistentVector",
            TYPE_MAP => "PersistentHashMap",
            TYPE_SET => "PersistentHashSet",
            TYPE_FUNCTION => "Function",
            TYPE_CLOSURE => "Closure",
            TYPE_NAMESPACE => "Namespace",
            TYPE_VAR => "Var",
            TYPE_ARRAY => "Array",
            TYPE_READER_LIST => "ReaderList",
            TYPE_READER_VECTOR => "ReaderVector",
            TYPE_READER_MAP => "ReaderMap",
            TYPE_READER_SYMBOL => "ReaderSymbol",
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

    // ========== Primitive Dispatch Functions ==========
    // These work before protocols exist by hardcoding behavior for built-in types,
    // then falling back to protocol dispatch for user-defined types (deftypes).

    /// Check if value is seq-able (list-like: can call first/rest on it)
    pub fn prim_is_seq(&self, value: usize) -> bool {
        let type_id = self.get_type_id_for_value(value);
        match type_id {
            TYPE_READER_LIST | TYPE_READER_VECTOR => true,
            TYPE_NIL => false, // nil is not a seq but is seq-able (returns nil)
            _ if type_id >= DEFTYPE_ID_OFFSET => {
                // Check if implements ISeq protocol (protocol ID 0 by convention)
                // For now, assume any deftype with -first method is seq-able
                self.lookup_protocol_method(type_id, "-first").is_some()
            }
            _ => false,
        }
    }

    /// Check if value is seqable (can be converted to a seq, including nil)
    pub fn prim_is_seqable(&self, value: usize) -> bool {
        let type_id = self.get_type_id_for_value(value);
        match type_id {
            TYPE_NIL => true, // nil is seqable (returns nil)
            TYPE_READER_LIST | TYPE_READER_VECTOR => true,
            _ if type_id >= DEFTYPE_ID_OFFSET => {
                self.lookup_protocol_method(type_id, "-first").is_some()
                    || self.lookup_protocol_method(type_id, "-seq").is_some()
            }
            _ => false,
        }
    }

    /// Get first element of a seq. Returns nil (7) for empty/nil.
    pub fn prim_first(&mut self, value: usize) -> Result<usize, String> {
        // nil returns nil
        if value == 7 {
            return Ok(7);
        }

        let type_id = self.get_type_id_for_value(value);
        match type_id {
            TYPE_NIL => Ok(7),
            TYPE_READER_LIST => {
                if self.reader_list_count(value) == 0 {
                    Ok(7) // empty list returns nil
                } else {
                    Ok(self.reader_list_first(value))
                }
            }
            TYPE_READER_VECTOR => {
                if self.reader_vector_count(value) == 0 {
                    Ok(7)
                } else {
                    self.reader_vector_nth(value, 0)
                }
            }
            _ if type_id >= DEFTYPE_ID_OFFSET => {
                // Fast path for known types
                let deftype_idx = type_id - DEFTYPE_ID_OFFSET;
                let type_name = self.get_type_def(deftype_idx).map(|td| td.name.as_str());

                match type_name {
                    Some("EmptyList") => Ok(7), // empty list returns nil
                    Some("PList") => {
                        // PList fields: [meta, first, rest, count, __hash]
                        // first is at field index 1
                        Ok(self.read_type_field(value, 1))
                    }
                    Some("Cons") => {
                        // Cons fields: [meta, first, rest, __hash]
                        // first is at field index 1
                        Ok(self.read_type_field(value, 1))
                    }
                    _ => {
                        // Fall back to protocol dispatch
                        crate::trampoline::invoke_protocol_method(self, value, "-first", &[])
                    }
                }
            }
            _ => Err(format!(
                "prim_first: unsupported type {}",
                Self::builtin_type_name(type_id)
            )),
        }
    }

    /// Get rest of a seq. Returns nil (7) for empty/nil.
    pub fn prim_rest(&mut self, value: usize) -> Result<usize, String> {
        // nil returns nil (or empty list in Clojure, but nil for simplicity)
        if value == 7 {
            return Ok(7);
        }

        let type_id = self.get_type_id_for_value(value);
        match type_id {
            TYPE_NIL => Ok(7),
            TYPE_READER_LIST => {
                if self.reader_list_count(value) == 0 {
                    Ok(7)
                } else {
                    self.reader_list_rest(value)
                }
            }
            TYPE_READER_VECTOR => {
                let count = self.reader_vector_count(value);
                if count <= 1 {
                    Ok(7) // Return nil for empty or single-element vector
                } else {
                    // For vectors, we'd need to return a subvec or seq
                    // For now, this is a limitation - vectors don't efficiently support rest
                    Err("prim_rest: ReaderVector rest not efficiently supported".to_string())
                }
            }
            _ if type_id >= DEFTYPE_ID_OFFSET => {
                // Fast path for known types
                let deftype_idx = type_id - DEFTYPE_ID_OFFSET;
                let type_name = self.get_type_def(deftype_idx).map(|td| td.name.as_str());

                match type_name {
                    Some("EmptyList") => {
                        // EmptyList.-rest returns itself (EMPTY-LIST)
                        Ok(value)
                    }
                    Some("PList") => {
                        // PList fields: [meta, first, rest, count, __hash]
                        // rest is at field index 2
                        let rest = self.read_type_field(value, 2);
                        // If rest is nil, return EMPTY-LIST (which we need to look up)
                        // For now, return nil if rest is nil
                        if rest == 7 {
                            // Need to return EMPTY-LIST, but we don't have easy access
                            // For safety, return nil - caller should handle
                            Ok(7)
                        } else {
                            Ok(rest)
                        }
                    }
                    Some("Cons") => {
                        // Cons fields: [meta, first, rest, __hash]
                        // rest is at field index 2
                        // Cons.-rest: (if (nil? (.-rest this)) EMPTY-LIST (.-rest this))
                        let rest = self.read_type_field(value, 2);
                        if rest == 7 {
                            // Would return EMPTY-LIST in Clojure
                            // For now, return nil - caller should check
                            Ok(7)
                        } else {
                            Ok(rest)
                        }
                    }
                    _ => {
                        // Fall back to protocol dispatch
                        crate::trampoline::invoke_protocol_method(self, value, "-rest", &[])
                    }
                }
            }
            _ => Err(format!(
                "prim_rest: unsupported type {}",
                Self::builtin_type_name(type_id)
            )),
        }
    }

    /// Get count of a collection
    /// Like Clojure's count, this will walk seqs that don't implement ICounted
    pub fn prim_count(&mut self, value: usize) -> Result<usize, String> {
        if value == 7 {
            return Ok(0); // nil has count 0
        }

        let type_id = self.get_type_id_for_value(value);
        match type_id {
            TYPE_NIL => Ok(0),
            TYPE_READER_LIST => Ok(self.reader_list_count(value)),
            TYPE_READER_VECTOR => Ok(self.reader_vector_count(value)),
            TYPE_READER_MAP => Ok(self.reader_map_count(value)),
            _ if type_id >= DEFTYPE_ID_OFFSET => {
                // Check for known types with hardcoded fast paths
                let deftype_idx = type_id - DEFTYPE_ID_OFFSET;
                let type_name = self.get_type_def(deftype_idx).map(|td| td.name.as_str());

                match type_name {
                    Some("EmptyList") => Ok(0),
                    Some("PList") => {
                        // PList has fields: [meta, first, rest, count, __hash]
                        // count is at field index 3
                        let count_tagged = self.read_type_field(value, 3);
                        Ok(count_tagged >> 3)
                    }
                    Some("Cons") => {
                        // Cons doesn't have count, walk the seq in Rust without protocol dispatch
                        // Cons fields: [meta, first, rest, __hash]
                        // rest is at field index 2
                        let mut count = 0usize;
                        let mut current = value;
                        loop {
                            if current == 7 {
                                break;
                            }
                            let cur_type_id = self.get_type_id_for_value(current);
                            if cur_type_id == TYPE_NIL {
                                break;
                            }
                            if cur_type_id < DEFTYPE_ID_OFFSET {
                                // Not a deftype - shouldn't happen but safety check
                                break;
                            }
                            let cur_deftype_idx = cur_type_id - DEFTYPE_ID_OFFSET;
                            let cur_type_name = self.get_type_def(cur_deftype_idx).map(|td| td.name.as_str());

                            match cur_type_name {
                                Some("EmptyList") => break,
                                Some("PList") => {
                                    // PList has count, add it and done
                                    let c = self.read_type_field(current, 3);
                                    count += c >> 3;
                                    break;
                                }
                                Some("Cons") => {
                                    // Cons: increment and get rest (field 2)
                                    count += 1;
                                    current = self.read_type_field(current, 2);
                                }
                                _ => {
                                    // Unknown type - fall back to protocol dispatch for -next
                                    count += 1;
                                    current = crate::trampoline::invoke_protocol_method(
                                        self, current, "-next", &[],
                                    )?;
                                }
                            }
                        }
                        Ok(count)
                    }
                    _ => {
                        // Unknown deftype - try protocol dispatch
                        if self.lookup_protocol_method(type_id, "-count").is_some() {
                            let result =
                                crate::trampoline::invoke_protocol_method(self, value, "-count", &[])?;
                            Ok((result >> 3) as usize)
                        } else if self.prim_is_seq(value) || self.prim_is_seqable(value) {
                            // Fall back to walking the seq
                            let mut count = 0usize;
                            let mut current = value;
                            loop {
                                if current == 7 {
                                    break;
                                }
                                let current_type = self.get_type_id_for_value(current);
                                if current_type == TYPE_NIL {
                                    break;
                                }
                                if current_type >= DEFTYPE_ID_OFFSET
                                    && self.lookup_protocol_method(current_type, "-count").is_some()
                                {
                                    let result = crate::trampoline::invoke_protocol_method(
                                        self, current, "-count", &[],
                                    )?;
                                    count += (result >> 3) as usize;
                                    break;
                                }
                                count += 1;
                                current = crate::trampoline::invoke_protocol_method(
                                    self, current, "-next", &[],
                                )?;
                            }
                            Ok(count)
                        } else {
                            Err(format!(
                                "prim_count: type {} is not countable or seqable",
                                Self::builtin_type_name(type_id)
                            ))
                        }
                    }
                }
            }
            _ => Err(format!(
                "prim_count: unsupported type {}",
                Self::builtin_type_name(type_id)
            )),
        }
    }

    /// Get nth element of a collection
    pub fn prim_nth(&mut self, value: usize, index: usize) -> Result<usize, String> {
        if value == 7 {
            return Err("prim_nth: cannot index nil".to_string());
        }

        let type_id = self.get_type_id_for_value(value);
        match type_id {
            TYPE_READER_LIST => self.reader_list_nth(value, index),
            TYPE_READER_VECTOR => self.reader_vector_nth(value, index),
            _ if type_id >= DEFTYPE_ID_OFFSET => {
                // Protocol dispatch to -nth
                // -nth takes (coll, n) where n is a tagged integer
                let tagged_index = index << 3;
                crate::trampoline::invoke_protocol_method(self, value, "-nth", &[tagged_index])
            }
            _ => Err(format!(
                "prim_nth: unsupported type {}",
                Self::builtin_type_name(type_id)
            )),
        }
    }

    /// Check if value is a symbol
    pub fn prim_is_symbol(&self, value: usize) -> bool {
        let type_id = self.get_type_id_for_value(value);
        match type_id {
            TYPE_READER_SYMBOL | TYPE_SYMBOL => true,
            _ if type_id >= DEFTYPE_ID_OFFSET => {
                // Check if implements INamed or has -name method
                self.lookup_protocol_method(type_id, "-name").is_some()
            }
            _ => false,
        }
    }

    /// Get the name of a symbol
    pub fn prim_symbol_name(&mut self, value: usize) -> Result<String, String> {
        let type_id = self.get_type_id_for_value(value);
        match type_id {
            TYPE_READER_SYMBOL => Ok(self.reader_symbol_name(value)),
            TYPE_SYMBOL => {
                // TYPE_SYMBOL is just a heap-allocated string containing the full symbol name
                // e.g. "foo" or "ns/foo"
                let full_name = self.read_string(value);
                // Extract just the name part (after /)
                if let Some(pos) = full_name.rfind('/') {
                    Ok(full_name[pos + 1..].to_string())
                } else {
                    Ok(full_name)
                }
            }
            _ if type_id >= DEFTYPE_ID_OFFSET => {
                // Protocol dispatch to -name
                // -name returns a string, which we need to read
                let result = crate::trampoline::invoke_protocol_method(self, value, "-name", &[])?;
                Ok(self.read_string(result))
            }
            _ => Err(format!(
                "prim_symbol_name: unsupported type {}",
                Self::builtin_type_name(type_id)
            )),
        }
    }

    /// Get the namespace of a symbol (None if unqualified)
    pub fn prim_symbol_namespace(&mut self, value: usize) -> Result<Option<String>, String> {
        let type_id = self.get_type_id_for_value(value);
        match type_id {
            TYPE_READER_SYMBOL => Ok(self.reader_symbol_namespace(value)),
            TYPE_SYMBOL => {
                // TYPE_SYMBOL is just a heap-allocated string containing the full symbol name
                // e.g. "foo" or "ns/foo"
                let full_name = self.read_string(value);
                // Extract namespace part (before /)
                if let Some(pos) = full_name.rfind('/') {
                    Ok(Some(full_name[..pos].to_string()))
                } else {
                    Ok(None)
                }
            }
            _ if type_id >= DEFTYPE_ID_OFFSET => {
                // Protocol dispatch to -namespace
                // -namespace returns a string or nil
                let result =
                    crate::trampoline::invoke_protocol_method(self, value, "-namespace", &[])?;
                if result == 7 {
                    // nil
                    Ok(None)
                } else {
                    Ok(Some(self.read_string(result)))
                }
            }
            _ => Err(format!(
                "prim_symbol_namespace: unsupported type {}",
                Self::builtin_type_name(type_id)
            )),
        }
    }

    /// Check if value is map-like
    pub fn prim_is_map(&self, value: usize) -> bool {
        let type_id = self.get_type_id_for_value(value);
        match type_id {
            TYPE_READER_MAP | TYPE_MAP => true,
            _ if type_id >= DEFTYPE_ID_OFFSET => {
                // Check if implements ILookup
                self.lookup_protocol_method(type_id, "-lookup").is_some()
            }
            _ => false,
        }
    }

    /// Lookup a key in a map-like value
    pub fn prim_get(&mut self, map: usize, key: usize, not_found: usize) -> Result<usize, String> {
        if map == 7 {
            return Ok(not_found); // nil returns not-found
        }

        let type_id = self.get_type_id_for_value(map);
        match type_id {
            TYPE_NIL => Ok(not_found),
            TYPE_READER_MAP => Ok(self.reader_map_lookup(map, key, not_found)),
            _ if type_id >= DEFTYPE_ID_OFFSET => {
                // Protocol dispatch to -lookup
                // -lookup takes (coll, key, not-found)
                crate::trampoline::invoke_protocol_method(self, map, "-lookup", &[key, not_found])
            }
            _ => Err(format!(
                "prim_get: unsupported type {}",
                Self::builtin_type_name(type_id)
            )),
        }
    }

    /// Check if value is a vector
    pub fn prim_is_vector(&self, value: usize) -> bool {
        let type_id = self.get_type_id_for_value(value);
        match type_id {
            TYPE_READER_VECTOR | TYPE_VECTOR => true,
            // For user-defined types, we'd need a marker protocol
            _ => false,
        }
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

        // Register as GC root so it survives allocations
        runtime.add_namespace_root("user".to_string(), ns_ptr);

        // Add a binding (42 tagged as int)
        let value = 42 << 3;
        let new_ns_ptr = runtime.namespace_add_binding(ns_ptr, "x", value).unwrap();

        // Look it up
        let result = runtime.namespace_lookup(new_ns_ptr, "x");
        assert_eq!(result, Some(value));
    }

    #[test]
    fn test_string_is_opaque() {
        use crate::gc::types::HeapObject;

        let mut runtime = GCRuntime::new();
        let str_ptr = runtime.allocate_string("hello").unwrap();

        // Untag to get raw pointer
        let untagged = str_ptr >> 3;
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);

        // Verify it's marked as opaque
        assert!(heap_obj.is_opaque_object(), "String should be opaque");

        // Verify get_fields returns empty (GC won't scan contents)
        assert!(
            heap_obj.get_fields().is_empty(),
            "Opaque object should have no fields"
        );
    }

    #[test]
    fn test_string_with_pointer_like_content() {
        // Test that strings containing bytes that look like pointers don't confuse the GC
        use crate::gc::types::HeapObject;

        let mut runtime = GCRuntime::new();

        // Create a string with content that could be misinterpreted as a pointer
        // (8 bytes that when interpreted as usize would look like an address)
        let str_ptr = runtime
            .allocate_string("\x00\x00\x00\x10\x00\x00\x00\x00")
            .unwrap();

        let untagged = str_ptr >> 3;
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);

        // Verify GC won't try to follow the "pointer"
        assert!(heap_obj.is_opaque_object());
        let refs: Vec<_> = heap_obj.get_heap_references().collect();
        assert!(
            refs.is_empty(),
            "Opaque objects should not report heap references"
        );
    }

    #[test]
    fn test_string_survives_gc() {
        let mut runtime = GCRuntime::new();

        // Allocate a namespace to serve as root
        let ns_ptr = runtime.allocate_namespace("test").unwrap();
        runtime.add_namespace_root("test".to_string(), ns_ptr);

        // Allocate a string and bind it to the namespace
        let str_ptr = runtime.allocate_string("test string").unwrap();
        let ns_ptr = runtime.namespace_add_binding(ns_ptr, "s", str_ptr).unwrap();
        runtime.add_namespace_root("test".to_string(), ns_ptr);

        // Run GC
        runtime.run_gc().unwrap();

        // String should still be readable
        let result = runtime.namespace_lookup(ns_ptr, "s").unwrap();
        let s = runtime.read_string(result);
        assert_eq!(s, "test string");
    }

    #[test]
    fn test_format_value_string() {
        let mut runtime = GCRuntime::new();
        let str_ptr = runtime.allocate_string("hello").unwrap();

        let formatted = runtime.format_value(str_ptr);
        assert_eq!(formatted, "\"hello\"");
    }

    #[test]
    fn test_format_value_empty_string() {
        let mut runtime = GCRuntime::new();
        let str_ptr = runtime.allocate_string("").unwrap();

        let formatted = runtime.format_value(str_ptr);
        assert_eq!(formatted, "\"\"");
    }

    #[test]
    fn test_string_tag() {
        let mut runtime = GCRuntime::new();
        let str_ptr = runtime.allocate_string("test").unwrap();

        // Verify the tag is correct (0b010 for String)
        assert_eq!(str_ptr & 0b111, 0b010);
    }
}
