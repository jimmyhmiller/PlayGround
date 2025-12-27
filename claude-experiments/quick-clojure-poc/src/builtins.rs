/// Builtin functions for quick-clojure-poc runtime
///
/// These builtins replace what were previously specialized VM instructions.
/// Following beagle's pattern, builtins are regular functions that can be called
/// using the standard Call instruction.
///
/// All builtin functions follow the standard calling convention:
/// - Arguments in x0, x1, x2, etc (untagged values)
/// - Return value in x0 (tagged value)
/// - May call other runtime functions
/// - Safe points for GC
use crate::gc_runtime::GCRuntime;
use crate::trampoline;
use std::cell::UnsafeCell;
use std::sync::Arc;

/// Runtime pointer used by builtins to access the runtime
/// SAFETY: Set once during initialization, immutable thereafter
static mut RUNTIME_PTR: Option<Arc<UnsafeCell<GCRuntime>>> = None;

/// Initialize the builtin runtime pointer
/// SAFETY: Must be called once before any builtins are used
pub unsafe fn initialize_builtins(runtime: Arc<UnsafeCell<GCRuntime>>) {
    RUNTIME_PTR = Some(runtime);
}

/// Get the runtime reference
/// SAFETY: Must be called after initialize_builtins
#[inline]
#[allow(static_mut_refs)]
unsafe fn get_runtime() -> &'static mut GCRuntime {
    &mut *RUNTIME_PTR.as_ref().unwrap().get()
}

// ============================================================================
// Variable access builtins
// ============================================================================

/// builtin_load_var_by_symbol(tagged_ns_symbol_id, tagged_name_symbol_id) -> tagged_value
///
/// Looks up a var by namespace and name symbols, returns its value.
/// Panics if the namespace or var doesn't exist.
/// Arguments are tagged integers that need to be untagged before use.
#[unsafe(no_mangle)]
pub extern "C" fn builtin_load_var_by_symbol(
    tagged_ns_symbol_id: usize,
    tagged_name_symbol_id: usize,
) -> usize {
    unsafe {
        // Untag the symbol IDs (shift right by 3)
        let ns_symbol_id = (tagged_ns_symbol_id >> 3) as u32;
        let name_symbol_id = (tagged_name_symbol_id >> 3) as u32;
        trampoline::trampoline_load_var_by_symbol(ns_symbol_id, name_symbol_id)
    }
}

/// builtin_load_var_by_symbol_dynamic(tagged_ns_symbol_id, tagged_name_symbol_id) -> tagged_value
///
/// Like builtin_load_var_by_symbol but checks dynamic bindings first.
/// Arguments are tagged integers that need to be untagged before use.
#[unsafe(no_mangle)]
pub extern "C" fn builtin_load_var_by_symbol_dynamic(
    tagged_ns_symbol_id: usize,
    tagged_name_symbol_id: usize,
) -> usize {
    unsafe {
        // Untag the symbol IDs (shift right by 3)
        let ns_symbol_id = (tagged_ns_symbol_id >> 3) as u32;
        let name_symbol_id = (tagged_name_symbol_id >> 3) as u32;
        trampoline::trampoline_load_var_by_symbol_dynamic(ns_symbol_id, name_symbol_id)
    }
}

/// builtin_store_var_by_symbol(tagged_ns_symbol_id, tagged_name_symbol_id, value) -> tagged_value (nil)
///
/// Stores a value in a var, creating the var if it doesn't exist.
/// Returns nil (tagged value 7).
/// Symbol ID arguments are tagged integers that need to be untagged before use.
#[unsafe(no_mangle)]
pub extern "C" fn builtin_store_var_by_symbol(
    tagged_ns_symbol_id: usize,
    tagged_name_symbol_id: usize,
    value: usize,
) -> usize {
    unsafe {
        // Untag the symbol IDs (shift right by 3)
        let ns_symbol_id = (tagged_ns_symbol_id >> 3) as u32;
        let name_symbol_id = (tagged_name_symbol_id >> 3) as u32;
        trampoline::trampoline_store_var_by_symbol(ns_symbol_id, name_symbol_id, value)
    }
}

/// builtin_ensure_var_by_symbol(tagged_ns_symbol_id, tagged_name_symbol_id) -> tagged_value (nil)
///
/// Ensures a var exists, creating it with an unbound placeholder if needed.
/// Returns nil (tagged value 7).
/// Arguments are tagged integers that need to be untagged before use.
#[unsafe(no_mangle)]
pub extern "C" fn builtin_ensure_var_by_symbol(
    tagged_ns_symbol_id: usize,
    tagged_name_symbol_id: usize,
) -> usize {
    unsafe {
        // Untag the symbol IDs (shift right by 3)
        let ns_symbol_id = (tagged_ns_symbol_id >> 3) as u32;
        let name_symbol_id = (tagged_name_symbol_id >> 3) as u32;

        let rt = get_runtime();
        // Clone strings to avoid borrow issues
        let ns_name = rt
            .get_symbol(ns_symbol_id)
            .expect("invalid ns_symbol_id")
            .to_string();
        let var_name = rt
            .get_symbol(name_symbol_id)
            .expect("invalid name_symbol_id")
            .to_string();
        let ns_ptr = rt
            .get_namespace_by_name(&ns_name)
            .unwrap_or_else(|| panic!("Namespace not found: {}", ns_name));

        // Check if var already exists
        if rt.namespace_lookup(ns_ptr, &var_name).is_none() {
            // Create var with unbound placeholder (we'll use nil for now)
            let (var_ptr, symbol_ptr) = rt.allocate_var(ns_ptr, &var_name, 7).unwrap(); // 7 = nil
            rt.namespace_add_binding_with_symbol_ptr(ns_ptr, &var_name, var_ptr, symbol_ptr)
                .unwrap();
        }

        7 // nil
    }
}

// ============================================================================
// Keyword interning
// ============================================================================

/// builtin_load_keyword(tagged_keyword_index) -> tagged_keyword_ptr
///
/// Loads/interns a keyword from the keyword constant pool.
/// Argument is a tagged integer that needs to be untagged before use.
#[unsafe(no_mangle)]
pub extern "C" fn builtin_load_keyword(tagged_keyword_index: usize) -> usize {
    unsafe {
        // Untag the keyword index (shift right by 3)
        let keyword_index = tagged_keyword_index >> 3;
        trampoline::trampoline_intern_keyword(keyword_index)
    }
}

// ============================================================================
// I/O operations
// ============================================================================

/// builtin_println_value(value) -> tagged_value (nil)
///
/// Prints a single value followed by a newline.
/// Returns nil (tagged value 7).
#[unsafe(no_mangle)]
pub extern "C" fn builtin_println_value(value: usize) -> usize {
    unsafe { trampoline::trampoline_println_value(value) }
}

/// builtin_print_value(value) -> tagged_value (nil)
///
/// Prints a single value without a newline.
/// Returns nil (tagged value 7).
#[unsafe(no_mangle)]
pub extern "C" fn builtin_print_value(value: usize) -> usize {
    unsafe { trampoline::trampoline_print_value(value) }
}

/// builtin_newline() -> tagged_value (nil)
///
/// Prints a newline.
/// Returns nil (tagged value 7).
#[unsafe(no_mangle)]
pub extern "C" fn builtin_newline() -> usize {
    unsafe { trampoline::trampoline_newline() }
}

/// builtin_print_space() -> tagged_value (nil)
///
/// Prints a space.
/// Returns nil (tagged value 7).
#[unsafe(no_mangle)]
pub extern "C" fn builtin_print_space() -> usize {
    unsafe { trampoline::trampoline_print_space() }
}

// ============================================================================
// Garbage collection
// ============================================================================

/// builtin_gc(stack_pointer) -> tagged_value (nil)
///
/// Forces a garbage collection.
/// Returns nil (tagged value 7).
#[unsafe(no_mangle)]
pub extern "C" fn builtin_gc(stack_pointer: usize) -> usize {
    unsafe { trampoline::trampoline_gc(stack_pointer) }
}

// ============================================================================
// Type predicates
// ============================================================================

/// builtin_is_map(value) -> tagged_boolean
///
/// Returns true if value is a map (PersistentHashMap), false otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn builtin_is_map(value: usize) -> usize {
    unsafe {
        let rt = get_runtime();
        if rt.is_map(value) {
            0b00001_111 // true
        } else {
            0b00000_111 // false
        }
    }
}

/// builtin_is_vector(value) -> tagged_boolean
///
/// Returns true if value is a vector, false otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn builtin_is_vector(value: usize) -> usize {
    unsafe {
        let rt = get_runtime();
        if rt.is_vector(value) {
            0b00001_111 // true
        } else {
            0b00000_111 // false
        }
    }
}

// ============================================================================
// Reader Type Constants
// ============================================================================
// These builtins return the type IDs for reader types, for use in extend-type.

use crate::gc_runtime::{TYPE_READER_LIST, TYPE_READER_MAP, TYPE_READER_SYMBOL, TYPE_READER_VECTOR};

/// __ReaderList() -> tagged_type_id
///
/// Returns the type ID for ReaderList (for use in extend-type).
#[unsafe(no_mangle)]
pub extern "C" fn builtin__reader_list_type() -> usize {
    TYPE_READER_LIST << 3 // Tag as integer
}

/// __ReaderVector() -> tagged_type_id
///
/// Returns the type ID for ReaderVector (for use in extend-type).
#[unsafe(no_mangle)]
pub extern "C" fn builtin__reader_vector_type() -> usize {
    TYPE_READER_VECTOR << 3 // Tag as integer
}

/// __ReaderMap() -> tagged_type_id
///
/// Returns the type ID for ReaderMap (for use in extend-type).
#[unsafe(no_mangle)]
pub extern "C" fn builtin__reader_map_type() -> usize {
    TYPE_READER_MAP << 3 // Tag as integer
}

/// __ReaderSymbol() -> tagged_type_id
///
/// Returns the type ID for ReaderSymbol (for use in extend-type).
#[unsafe(no_mangle)]
pub extern "C" fn builtin__reader_symbol_type() -> usize {
    TYPE_READER_SYMBOL << 3 // Tag as integer
}

// ============================================================================
// Reader Type Operations - ReaderList
// ============================================================================

/// __reader_list_first(list) -> tagged_value
///
/// Returns the first element of a ReaderList, or nil if empty.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__reader_list_first(list: usize) -> usize {
    unsafe {
        let rt = get_runtime();
        rt.reader_list_first(list)
    }
}

/// __reader_list_rest(list) -> tagged_reader_list
///
/// Returns a new ReaderList without the first element.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__reader_list_rest(list: usize) -> usize {
    unsafe {
        let rt = get_runtime();
        rt.reader_list_rest(list).unwrap_or(7) // Return nil on error
    }
}

/// __reader_list_count(list) -> tagged_integer
///
/// Returns the count of elements in a ReaderList.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__reader_list_count(list: usize) -> usize {
    unsafe {
        let rt = get_runtime();
        let count = rt.reader_list_count(list);
        count << 3 // Tag as integer
    }
}

/// __reader_list_conj(list, elem) -> tagged_reader_list
///
/// Returns a new ReaderList with elem prepended.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__reader_list_conj(list: usize, elem: usize) -> usize {
    unsafe {
        let rt = get_runtime();
        rt.reader_list_conj(list, elem).unwrap_or(7)
    }
}

/// __reader_list_nth(list, index) -> tagged_value
///
/// Returns the element at index in a ReaderList.
/// Index should be a tagged integer.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__reader_list_nth(list: usize, tagged_index: usize) -> usize {
    unsafe {
        let rt = get_runtime();
        let index = tagged_index >> 3;
        rt.reader_list_nth(list, index).unwrap_or(7)
    }
}

// ============================================================================
// Reader Type Operations - ReaderVector
// ============================================================================

/// __reader_vector_count(vec) -> tagged_integer
///
/// Returns the count of elements in a ReaderVector.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__reader_vector_count(vec: usize) -> usize {
    unsafe {
        let rt = get_runtime();
        let count = rt.reader_vector_count(vec);
        count << 3 // Tag as integer
    }
}

/// __reader_vector_nth(vec, index) -> tagged_value
///
/// Returns the element at index in a ReaderVector.
/// Index should be a tagged integer.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__reader_vector_nth(vec: usize, tagged_index: usize) -> usize {
    unsafe {
        let rt = get_runtime();
        let index = tagged_index >> 3;
        rt.reader_vector_nth(vec, index).unwrap_or(7)
    }
}

/// __reader_vector_nth_or(vec, index, not_found) -> tagged_value
///
/// Returns the element at index in a ReaderVector, or not_found if out of bounds.
/// Index should be a tagged integer.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__reader_vector_nth_or(
    vec: usize,
    tagged_index: usize,
    not_found: usize,
) -> usize {
    unsafe {
        let rt = get_runtime();
        let index = tagged_index >> 3;
        rt.reader_vector_nth_or(vec, index, not_found)
    }
}

/// __reader_vector_conj(vec, elem) -> tagged_reader_vector
///
/// Returns a new ReaderVector with elem appended.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__reader_vector_conj(vec: usize, elem: usize) -> usize {
    unsafe {
        let rt = get_runtime();
        rt.reader_vector_conj(vec, elem).unwrap_or(7)
    }
}

// ============================================================================
// Reader Type Operations - ReaderMap
// ============================================================================

/// __reader_map_count(map) -> tagged_integer
///
/// Returns the count of entries in a ReaderMap.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__reader_map_count(map: usize) -> usize {
    unsafe {
        let rt = get_runtime();
        let count = rt.reader_map_count(map);
        count << 3 // Tag as integer
    }
}

/// __reader_map_lookup(map, key, not_found) -> tagged_value
///
/// Looks up key in a ReaderMap, returns not_found if not present.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__reader_map_lookup(map: usize, key: usize, not_found: usize) -> usize {
    unsafe {
        let rt = get_runtime();
        rt.reader_map_lookup(map, key, not_found)
    }
}

/// __reader_map_assoc(map, key, value) -> tagged_reader_map
///
/// Returns a new ReaderMap with key associated to value.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__reader_map_assoc(map: usize, key: usize, value: usize) -> usize {
    unsafe {
        let rt = get_runtime();
        rt.reader_map_assoc(map, key, value).unwrap_or(7)
    }
}

/// __reader_map_keys(map) -> tagged_reader_list
///
/// Returns a ReaderList of all keys in the ReaderMap.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__reader_map_keys(map: usize) -> usize {
    unsafe {
        let rt = get_runtime();
        rt.reader_map_keys(map).unwrap_or(7)
    }
}

/// __reader_map_vals(map) -> tagged_reader_list
///
/// Returns a ReaderList of all values in the ReaderMap.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__reader_map_vals(map: usize) -> usize {
    unsafe {
        let rt = get_runtime();
        rt.reader_map_vals(map).unwrap_or(7)
    }
}

// ============================================================================
// Reader Type Operations - ReaderSymbol
// ============================================================================

/// __reader_symbol_name(sym) -> tagged_string
///
/// Returns the name of a ReaderSymbol as a string.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__reader_symbol_name(sym: usize) -> usize {
    unsafe {
        let rt = get_runtime();
        rt.reader_symbol_name_ptr(sym)
    }
}

/// __reader_symbol_namespace(sym) -> tagged_string_or_nil
///
/// Returns the namespace of a ReaderSymbol as a string, or nil if none.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__reader_symbol_namespace(sym: usize) -> usize {
    unsafe {
        let rt = get_runtime();
        rt.reader_symbol_namespace_ptr(sym)
    }
}

// ============================================================================
// Reader Type Predicates
// ============================================================================

/// __reader_list?(value) -> tagged_boolean
///
/// Returns true if value is a ReaderList, false otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__is_reader_list(value: usize) -> usize {
    unsafe {
        let rt = get_runtime();
        if rt.is_reader_list(value) {
            0b00001_111 // true
        } else {
            0b00000_111 // false
        }
    }
}

/// __reader_vector?(value) -> tagged_boolean
///
/// Returns true if value is a ReaderVector, false otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__is_reader_vector(value: usize) -> usize {
    unsafe {
        let rt = get_runtime();
        if rt.is_reader_vector(value) {
            0b00001_111 // true
        } else {
            0b00000_111 // false
        }
    }
}

/// __reader_map?(value) -> tagged_boolean
///
/// Returns true if value is a ReaderMap, false otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__is_reader_map(value: usize) -> usize {
    unsafe {
        let rt = get_runtime();
        if rt.is_reader_map(value) {
            0b00001_111 // true
        } else {
            0b00000_111 // false
        }
    }
}

/// __reader_symbol?(value) -> tagged_boolean
///
/// Returns true if value is a ReaderSymbol, false otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__is_reader_symbol(value: usize) -> usize {
    unsafe {
        let rt = get_runtime();
        if rt.is_reader_symbol(value) {
            0b00001_111 // true
        } else {
            0b00000_111 // false
        }
    }
}

// ============================================================================
// Reader Type Constructors
// These allow Clojure code to construct reader types for macro expansion
// ============================================================================

/// __make_reader_symbol(name_str) -> tagged_reader_symbol
/// __make_reader_symbol(ns_str, name_str) -> tagged_reader_symbol
///
/// Creates a new ReaderSymbol from a name (and optional namespace).
/// Arguments should be tagged strings, or nil for the namespace.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__make_reader_symbol(ns_or_name: usize, name_or_sentinel: usize) -> usize {
    unsafe {
        let rt = get_runtime();

        // Check if this is a 1-arg or 2-arg call
        // If second arg is a special sentinel (we use 0 for "not provided"), it's 1-arg
        // But in practice, we need a way to distinguish. Let's use:
        // - 1-arg: ns_or_name is the name, name_or_sentinel is 0 (will be provided by wrapper)
        // - 2-arg: ns_or_name is the ns (or nil for no namespace), name_or_sentinel is the name

        if name_or_sentinel == 0 {
            // 1-arg form: just name
            let name = rt.read_string(ns_or_name);
            rt.allocate_reader_symbol(None, &name).unwrap_or(7)
        } else {
            // 2-arg form: namespace (or nil) + name
            let namespace = if ns_or_name == 7 {
                // nil = no namespace
                None
            } else {
                Some(rt.read_string(ns_or_name))
            };
            let name = rt.read_string(name_or_sentinel);
            rt.allocate_reader_symbol(namespace.as_deref(), &name).unwrap_or(7)
        }
    }
}

/// __make_reader_symbol_1(name_str) -> tagged_reader_symbol
///
/// Creates a new unqualified ReaderSymbol from a name string.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__make_reader_symbol_1(name: usize) -> usize {
    unsafe {
        let rt = get_runtime();
        let name_str = rt.read_string(name);
        rt.allocate_reader_symbol(None, &name_str).unwrap_or(7)
    }
}

/// __make_reader_symbol_2(ns_or_nil, name_str) -> tagged_reader_symbol
///
/// Creates a new ReaderSymbol with optional namespace.
/// First arg is namespace string or nil, second is name string.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__make_reader_symbol_2(ns_or_nil: usize, name: usize) -> usize {
    unsafe {
        let rt = get_runtime();
        let namespace = if ns_or_nil == 7 {
            None
        } else {
            Some(rt.read_string(ns_or_nil))
        };
        let name_str = rt.read_string(name);
        rt.allocate_reader_symbol(namespace.as_deref(), &name_str).unwrap_or(7)
    }
}

/// __make_reader_list() -> tagged_reader_list
///
/// Creates an empty ReaderList.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__make_reader_list_0() -> usize {
    unsafe {
        let rt = get_runtime();
        rt.allocate_reader_list(&[]).unwrap_or(7)
    }
}

/// __reader_cons(elem, list) -> tagged_reader_list
///
/// Prepends an element to a ReaderList, returning a new ReaderList.
/// Similar to cons but for reader types.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__reader_cons(elem: usize, list: usize) -> usize {
    unsafe {
        let rt = get_runtime();
        // Get elements from existing list
        let count = if list == 7 { 0 } else { rt.reader_list_count(list) };
        let mut elements = Vec::with_capacity(count + 1);
        elements.push(elem);

        // Copy existing elements
        let mut current = list;
        for _ in 0..count {
            elements.push(rt.reader_list_first(current));
            current = rt.reader_list_rest(current).unwrap_or(7);
        }

        rt.allocate_reader_list(&elements).unwrap_or(7)
    }
}

/// __reader_list_1(a) -> tagged_reader_list
/// Creates a ReaderList with one element.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__reader_list_1(a: usize) -> usize {
    unsafe {
        let rt = get_runtime();
        rt.allocate_reader_list(&[a]).unwrap_or(7)
    }
}

/// __reader_list_2(a, b) -> tagged_reader_list
/// Creates a ReaderList with two elements.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__reader_list_2(a: usize, b: usize) -> usize {
    unsafe {
        let rt = get_runtime();
        rt.allocate_reader_list(&[a, b]).unwrap_or(7)
    }
}

/// __reader_list_3(a, b, c) -> tagged_reader_list
/// Creates a ReaderList with three elements.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__reader_list_3(a: usize, b: usize, c: usize) -> usize {
    unsafe {
        let rt = get_runtime();
        rt.allocate_reader_list(&[a, b, c]).unwrap_or(7)
    }
}

/// __reader_list_4(a, b, c, d) -> tagged_reader_list
/// Creates a ReaderList with four elements.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__reader_list_4(a: usize, b: usize, c: usize, d: usize) -> usize {
    unsafe {
        let rt = get_runtime();
        rt.allocate_reader_list(&[a, b, c, d]).unwrap_or(7)
    }
}

/// __reader_list_5(a, b, c, d, e) -> tagged_reader_list
/// Creates a ReaderList with five elements.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__reader_list_5(a: usize, b: usize, c: usize, d: usize, e: usize) -> usize {
    unsafe {
        let rt = get_runtime();
        rt.allocate_reader_list(&[a, b, c, d, e]).unwrap_or(7)
    }
}

/// __make_reader_list_from_vec(vec) -> tagged_reader_list
///
/// Creates a ReaderList from a ReaderVector's elements.
/// This is useful for (apply list [...]) style operations.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__make_reader_list_from_vec(vec: usize) -> usize {
    unsafe {
        let rt = get_runtime();
        let count = rt.reader_vector_count(vec);
        let mut elements = Vec::with_capacity(count);
        for i in 0..count {
            elements.push(rt.reader_vector_nth(vec, i).unwrap_or(7));
        }
        rt.allocate_reader_list(&elements).unwrap_or(7)
    }
}

/// __make_reader_vector() -> tagged_reader_vector
///
/// Creates an empty ReaderVector.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__make_reader_vector_0() -> usize {
    unsafe {
        let rt = get_runtime();
        rt.allocate_reader_vector(&[]).unwrap_or(7)
    }
}

/// __make_reader_vector_from_list(list) -> tagged_reader_vector
///
/// Creates a ReaderVector from a ReaderList's elements.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__make_reader_vector_from_list(list: usize) -> usize {
    unsafe {
        let rt = get_runtime();
        let count = rt.reader_list_count(list);
        let mut elements = Vec::with_capacity(count);
        let mut current = list;
        for _ in 0..count {
            elements.push(rt.reader_list_first(current));
            current = rt.reader_list_rest(current).unwrap_or(7);
        }
        rt.allocate_reader_vector(&elements).unwrap_or(7)
    }
}

/// __make_reader_map() -> tagged_reader_map
///
/// Creates an empty ReaderMap.
#[unsafe(no_mangle)]
pub extern "C" fn builtin__make_reader_map_0() -> usize {
    unsafe {
        let rt = get_runtime();
        rt.allocate_reader_map(&[]).unwrap_or(7)
    }
}

// ============================================================================
// Builtin registration
// ============================================================================

/// Builtin function descriptor
pub struct BuiltinDescriptor {
    pub name: &'static str,
    pub function_ptr: usize,
    pub arity: usize, // Fixed arity (for now, no variadic builtins)
}

/// Get all builtin function descriptors
pub fn get_builtin_descriptors() -> Vec<BuiltinDescriptor> {
    vec![
        BuiltinDescriptor {
            name: "runtime.builtin/load-var-by-symbol",
            function_ptr: builtin_load_var_by_symbol as usize,
            arity: 2,
        },
        BuiltinDescriptor {
            name: "runtime.builtin/load-var-by-symbol-dynamic",
            function_ptr: builtin_load_var_by_symbol_dynamic as usize,
            arity: 2,
        },
        BuiltinDescriptor {
            name: "runtime.builtin/store-var-by-symbol",
            function_ptr: builtin_store_var_by_symbol as usize,
            arity: 3,
        },
        BuiltinDescriptor {
            name: "runtime.builtin/ensure-var-by-symbol",
            function_ptr: builtin_ensure_var_by_symbol as usize,
            arity: 2,
        },
        BuiltinDescriptor {
            name: "runtime.builtin/load-keyword",
            function_ptr: builtin_load_keyword as usize,
            arity: 1,
        },
        BuiltinDescriptor {
            name: "runtime.builtin/_println",
            function_ptr: builtin_println_value as usize,
            arity: 1, // single value
        },
        BuiltinDescriptor {
            name: "runtime.builtin/_print",
            function_ptr: builtin_print_value as usize,
            arity: 1, // single value
        },
        BuiltinDescriptor {
            name: "runtime.builtin/_newline",
            function_ptr: builtin_newline as usize,
            arity: 0,
        },
        BuiltinDescriptor {
            name: "runtime.builtin/_print-space",
            function_ptr: builtin_print_space as usize,
            arity: 0,
        },
        BuiltinDescriptor {
            name: "runtime.builtin/gc",
            function_ptr: builtin_gc as usize,
            arity: 1, // stack_pointer
        },
        BuiltinDescriptor {
            name: "runtime.builtin/map?",
            function_ptr: builtin_is_map as usize,
            arity: 1,
        },
        BuiltinDescriptor {
            name: "runtime.builtin/vector?",
            function_ptr: builtin_is_vector as usize,
            arity: 1,
        },
        // Reader Type Constants (for use in extend-type)
        BuiltinDescriptor {
            name: "__ReaderList",
            function_ptr: builtin__reader_list_type as usize,
            arity: 0,
        },
        BuiltinDescriptor {
            name: "__ReaderVector",
            function_ptr: builtin__reader_vector_type as usize,
            arity: 0,
        },
        BuiltinDescriptor {
            name: "__ReaderMap",
            function_ptr: builtin__reader_map_type as usize,
            arity: 0,
        },
        BuiltinDescriptor {
            name: "__ReaderSymbol",
            function_ptr: builtin__reader_symbol_type as usize,
            arity: 0,
        },
        // Reader List operations
        BuiltinDescriptor {
            name: "__reader_list_first",
            function_ptr: builtin__reader_list_first as usize,
            arity: 1,
        },
        BuiltinDescriptor {
            name: "__reader_list_rest",
            function_ptr: builtin__reader_list_rest as usize,
            arity: 1,
        },
        BuiltinDescriptor {
            name: "__reader_list_count",
            function_ptr: builtin__reader_list_count as usize,
            arity: 1,
        },
        BuiltinDescriptor {
            name: "__reader_list_conj",
            function_ptr: builtin__reader_list_conj as usize,
            arity: 2,
        },
        BuiltinDescriptor {
            name: "__reader_list_nth",
            function_ptr: builtin__reader_list_nth as usize,
            arity: 2,
        },
        // Reader Vector operations
        BuiltinDescriptor {
            name: "__reader_vector_count",
            function_ptr: builtin__reader_vector_count as usize,
            arity: 1,
        },
        BuiltinDescriptor {
            name: "__reader_vector_nth",
            function_ptr: builtin__reader_vector_nth as usize,
            arity: 2,
        },
        BuiltinDescriptor {
            name: "__reader_vector_nth_or",
            function_ptr: builtin__reader_vector_nth_or as usize,
            arity: 3,
        },
        BuiltinDescriptor {
            name: "__reader_vector_conj",
            function_ptr: builtin__reader_vector_conj as usize,
            arity: 2,
        },
        // Reader Map operations
        BuiltinDescriptor {
            name: "__reader_map_count",
            function_ptr: builtin__reader_map_count as usize,
            arity: 1,
        },
        BuiltinDescriptor {
            name: "__reader_map_lookup",
            function_ptr: builtin__reader_map_lookup as usize,
            arity: 3,
        },
        BuiltinDescriptor {
            name: "__reader_map_assoc",
            function_ptr: builtin__reader_map_assoc as usize,
            arity: 3,
        },
        BuiltinDescriptor {
            name: "__reader_map_keys",
            function_ptr: builtin__reader_map_keys as usize,
            arity: 1,
        },
        BuiltinDescriptor {
            name: "__reader_map_vals",
            function_ptr: builtin__reader_map_vals as usize,
            arity: 1,
        },
        // Reader Symbol operations
        BuiltinDescriptor {
            name: "__reader_symbol_name",
            function_ptr: builtin__reader_symbol_name as usize,
            arity: 1,
        },
        BuiltinDescriptor {
            name: "__reader_symbol_namespace",
            function_ptr: builtin__reader_symbol_namespace as usize,
            arity: 1,
        },
        // Reader Type predicates
        BuiltinDescriptor {
            name: "__reader_list?",
            function_ptr: builtin__is_reader_list as usize,
            arity: 1,
        },
        BuiltinDescriptor {
            name: "__reader_vector?",
            function_ptr: builtin__is_reader_vector as usize,
            arity: 1,
        },
        BuiltinDescriptor {
            name: "__reader_map?",
            function_ptr: builtin__is_reader_map as usize,
            arity: 1,
        },
        BuiltinDescriptor {
            name: "__reader_symbol?",
            function_ptr: builtin__is_reader_symbol as usize,
            arity: 1,
        },
        // Reader Type Constructors
        BuiltinDescriptor {
            name: "__make_reader_symbol_1",
            function_ptr: builtin__make_reader_symbol_1 as usize,
            arity: 1,
        },
        BuiltinDescriptor {
            name: "__make_reader_symbol_2",
            function_ptr: builtin__make_reader_symbol_2 as usize,
            arity: 2,
        },
        BuiltinDescriptor {
            name: "__make_reader_list_0",
            function_ptr: builtin__make_reader_list_0 as usize,
            arity: 0,
        },
        BuiltinDescriptor {
            name: "__reader_cons",
            function_ptr: builtin__reader_cons as usize,
            arity: 2,
        },
        BuiltinDescriptor {
            name: "__reader_list_1",
            function_ptr: builtin__reader_list_1 as usize,
            arity: 1,
        },
        BuiltinDescriptor {
            name: "__reader_list_2",
            function_ptr: builtin__reader_list_2 as usize,
            arity: 2,
        },
        BuiltinDescriptor {
            name: "__reader_list_3",
            function_ptr: builtin__reader_list_3 as usize,
            arity: 3,
        },
        BuiltinDescriptor {
            name: "__reader_list_4",
            function_ptr: builtin__reader_list_4 as usize,
            arity: 4,
        },
        BuiltinDescriptor {
            name: "__reader_list_5",
            function_ptr: builtin__reader_list_5 as usize,
            arity: 5,
        },
        BuiltinDescriptor {
            name: "__make_reader_list_from_vec",
            function_ptr: builtin__make_reader_list_from_vec as usize,
            arity: 1,
        },
        BuiltinDescriptor {
            name: "__make_reader_vector_0",
            function_ptr: builtin__make_reader_vector_0 as usize,
            arity: 0,
        },
        BuiltinDescriptor {
            name: "__make_reader_vector_from_list",
            function_ptr: builtin__make_reader_vector_from_list as usize,
            arity: 1,
        },
        BuiltinDescriptor {
            name: "__make_reader_map_0",
            function_ptr: builtin__make_reader_map_0 as usize,
            arity: 0,
        },
    ]
}

/// Register all builtin functions in the runtime
///
/// Note: Builtins are NOT stored in any namespace. They are called directly
/// by function pointer at compile time. This function is kept for compatibility
/// but does nothing - builtins are accessed via get_builtin_descriptors().
pub fn register_builtins(_runtime: &mut GCRuntime) {
    // Builtins are now accessed directly by function pointer,
    // not through namespace/var lookup. No registration needed.
}
