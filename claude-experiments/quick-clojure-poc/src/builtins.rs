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
use std::sync::Arc;
use std::cell::UnsafeCell;

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
pub extern "C" fn builtin_load_var_by_symbol(tagged_ns_symbol_id: usize, tagged_name_symbol_id: usize) -> usize {
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
pub extern "C" fn builtin_load_var_by_symbol_dynamic(tagged_ns_symbol_id: usize, tagged_name_symbol_id: usize) -> usize {
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
pub extern "C" fn builtin_store_var_by_symbol(tagged_ns_symbol_id: usize, tagged_name_symbol_id: usize, value: usize) -> usize {
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
pub extern "C" fn builtin_ensure_var_by_symbol(tagged_ns_symbol_id: usize, tagged_name_symbol_id: usize) -> usize {
    unsafe {
        // Untag the symbol IDs (shift right by 3)
        let ns_symbol_id = (tagged_ns_symbol_id >> 3) as u32;
        let name_symbol_id = (tagged_name_symbol_id >> 3) as u32;

        let rt = get_runtime();
        // Clone strings to avoid borrow issues
        let ns_name = rt.get_symbol(ns_symbol_id).expect("invalid ns_symbol_id").to_string();
        let var_name = rt.get_symbol(name_symbol_id).expect("invalid name_symbol_id").to_string();
        let ns_ptr = rt.get_namespace_by_name(&ns_name)
            .unwrap_or_else(|| panic!("Namespace not found: {}", ns_name));

        // Check if var already exists
        if rt.namespace_lookup(ns_ptr, &var_name).is_none() {
            // Create var with unbound placeholder (we'll use nil for now)
            let (var_ptr, _) = rt.allocate_var(ns_ptr, &var_name, 7).unwrap(); // 7 = nil
            rt.namespace_add_binding(ns_ptr, &var_name, var_ptr).unwrap();
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
    unsafe {
        trampoline::trampoline_println_value(value)
    }
}

/// builtin_print_value(value) -> tagged_value (nil)
///
/// Prints a single value without a newline.
/// Returns nil (tagged value 7).
#[unsafe(no_mangle)]
pub extern "C" fn builtin_print_value(value: usize) -> usize {
    unsafe {
        trampoline::trampoline_print_value(value)
    }
}

/// builtin_newline() -> tagged_value (nil)
///
/// Prints a newline.
/// Returns nil (tagged value 7).
#[unsafe(no_mangle)]
pub extern "C" fn builtin_newline() -> usize {
    unsafe {
        trampoline::trampoline_newline()
    }
}

/// builtin_print_space() -> tagged_value (nil)
///
/// Prints a space.
/// Returns nil (tagged value 7).
#[unsafe(no_mangle)]
pub extern "C" fn builtin_print_space() -> usize {
    unsafe {
        trampoline::trampoline_print_space()
    }
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
    unsafe {
        trampoline::trampoline_gc(stack_pointer)
    }
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
            0b00001_111  // true
        } else {
            0b00000_111  // false
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
            0b00001_111  // true
        } else {
            0b00000_111  // false
        }
    }
}

// ============================================================================
// Builtin registration
// ============================================================================

/// Builtin function descriptor
pub struct BuiltinDescriptor {
    pub name: &'static str,
    pub function_ptr: usize,
    pub arity: usize,  // Fixed arity (for now, no variadic builtins)
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
            arity: 1,  // single value
        },
        BuiltinDescriptor {
            name: "runtime.builtin/_print",
            function_ptr: builtin_print_value as usize,
            arity: 1,  // single value
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
            arity: 1,  // stack_pointer
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
