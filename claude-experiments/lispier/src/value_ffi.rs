//! FFI interface for Value manipulation from compiled code
//!
//! This module provides C-compatible functions for manipulating Values,
//! allowing macros written in MLIR Lisp to construct and transform ASTs.

use std::ffi::{c_char, CStr};

use crate::value::{Symbol, Value};

// ============================================================================
// Value Construction
// ============================================================================

/// Create a new empty list
#[unsafe(no_mangle)]
pub extern "C" fn value_list_new() -> *mut Value {
    Box::into_raw(Box::new(Value::List(Vec::new())))
}

/// Create a new empty vector
#[unsafe(no_mangle)]
pub extern "C" fn value_vector_new() -> *mut Value {
    Box::into_raw(Box::new(Value::Vector(Vec::new())))
}

/// Create a symbol from a C string
///
/// # Safety
/// The name pointer must be a valid null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn value_symbol_new(name: *const c_char) -> *mut Value {
    let name_str = unsafe { CStr::from_ptr(name).to_string_lossy().into_owned() };
    Box::into_raw(Box::new(Value::Symbol(Symbol::new(name_str))))
}

/// Create a string value from a C string
///
/// # Safety
/// The s pointer must be a valid null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn value_string_new(s: *const c_char) -> *mut Value {
    let string = unsafe { CStr::from_ptr(s).to_string_lossy().into_owned() };
    Box::into_raw(Box::new(Value::String(string)))
}

/// Create a number value
#[unsafe(no_mangle)]
pub extern "C" fn value_number_new(n: f64) -> *mut Value {
    Box::into_raw(Box::new(Value::Number(n)))
}

/// Create a keyword from a C string
///
/// # Safety
/// The name pointer must be a valid null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn value_keyword_new(name: *const c_char) -> *mut Value {
    let name_str = unsafe { CStr::from_ptr(name).to_string_lossy().into_owned() };
    Box::into_raw(Box::new(Value::Keyword(name_str)))
}

/// Create a boolean value
#[unsafe(no_mangle)]
pub extern "C" fn value_bool_new(b: bool) -> *mut Value {
    Box::into_raw(Box::new(Value::Boolean(b)))
}

/// Create a nil value
#[unsafe(no_mangle)]
pub extern "C" fn value_nil_new() -> *mut Value {
    Box::into_raw(Box::new(Value::Nil))
}

// ============================================================================
// List Operations
// ============================================================================

/// Push a value onto a list
///
/// # Safety
/// - list must be a valid pointer to a Value::List
/// - item must be a valid pointer to a Value
/// - The item is cloned, caller retains ownership of item
#[unsafe(no_mangle)]
pub unsafe extern "C" fn value_list_push(list: *mut Value, item: *const Value) {
    let list_ref = unsafe { &mut *list };
    let item_ref = unsafe { &*item };

    if let Value::List(items) = list_ref {
        items.push(item_ref.clone());
    }
}

/// Get the first element of a list (returns null if empty or not a list)
///
/// # Safety
/// - list must be a valid pointer to a Value
/// - Returned pointer is a new allocation owned by the caller
#[unsafe(no_mangle)]
pub unsafe extern "C" fn value_list_first(list: *const Value) -> *mut Value {
    let list_ref = unsafe { &*list };

    if let Value::List(items) = list_ref {
        if let Some(first) = items.first() {
            return Box::into_raw(Box::new(first.clone()));
        }
    }
    std::ptr::null_mut()
}

/// Get the rest of a list (tail, everything after first element)
///
/// # Safety
/// - list must be a valid pointer to a Value
/// - Returned pointer is a new allocation owned by the caller
#[unsafe(no_mangle)]
pub unsafe extern "C" fn value_list_rest(list: *const Value) -> *mut Value {
    let list_ref = unsafe { &*list };

    if let Value::List(items) = list_ref {
        if items.len() > 1 {
            let rest: Vec<Value> = items[1..].to_vec();
            return Box::into_raw(Box::new(Value::List(rest)));
        } else {
            // Empty or single element - return empty list
            return Box::into_raw(Box::new(Value::List(Vec::new())));
        }
    }
    std::ptr::null_mut()
}

/// Get the length of a list
///
/// # Safety
/// - list must be a valid pointer to a Value
#[unsafe(no_mangle)]
pub unsafe extern "C" fn value_list_len(list: *const Value) -> usize {
    let list_ref = unsafe { &*list };

    if let Value::List(items) = list_ref {
        items.len()
    } else {
        0
    }
}

/// Get an element at a specific index
///
/// # Safety
/// - list must be a valid pointer to a Value::List
/// - Returns null if index is out of bounds
/// - Returned pointer is a new allocation owned by the caller
#[unsafe(no_mangle)]
pub unsafe extern "C" fn value_list_get(list: *const Value, index: usize) -> *mut Value {
    let list_ref = unsafe { &*list };

    if let Value::List(items) = list_ref {
        if let Some(item) = items.get(index) {
            return Box::into_raw(Box::new(item.clone()));
        }
    }
    std::ptr::null_mut()
}

// ============================================================================
// Type Checking
// ============================================================================

/// Check if value is a list
///
/// # Safety
/// - value must be a valid pointer to a Value
#[unsafe(no_mangle)]
pub unsafe extern "C" fn value_is_list(value: *const Value) -> bool {
    let value_ref = unsafe { &*value };
    matches!(value_ref, Value::List(_))
}

/// Check if value is a vector
///
/// # Safety
/// - value must be a valid pointer to a Value
#[unsafe(no_mangle)]
pub unsafe extern "C" fn value_is_vector(value: *const Value) -> bool {
    let value_ref = unsafe { &*value };
    matches!(value_ref, Value::Vector(_))
}

/// Check if value is a symbol
///
/// # Safety
/// - value must be a valid pointer to a Value
#[unsafe(no_mangle)]
pub unsafe extern "C" fn value_is_symbol(value: *const Value) -> bool {
    let value_ref = unsafe { &*value };
    matches!(value_ref, Value::Symbol(_))
}

/// Check if value is a string
///
/// # Safety
/// - value must be a valid pointer to a Value
#[unsafe(no_mangle)]
pub unsafe extern "C" fn value_is_string(value: *const Value) -> bool {
    let value_ref = unsafe { &*value };
    matches!(value_ref, Value::String(_))
}

/// Check if value is a number
///
/// # Safety
/// - value must be a valid pointer to a Value
#[unsafe(no_mangle)]
pub unsafe extern "C" fn value_is_number(value: *const Value) -> bool {
    let value_ref = unsafe { &*value };
    matches!(value_ref, Value::Number(_))
}

/// Check if value is a keyword
///
/// # Safety
/// - value must be a valid pointer to a Value
#[unsafe(no_mangle)]
pub unsafe extern "C" fn value_is_keyword(value: *const Value) -> bool {
    let value_ref = unsafe { &*value };
    matches!(value_ref, Value::Keyword(_))
}

/// Check if value is nil
///
/// # Safety
/// - value must be a valid pointer to a Value
#[unsafe(no_mangle)]
pub unsafe extern "C" fn value_is_nil(value: *const Value) -> bool {
    let value_ref = unsafe { &*value };
    matches!(value_ref, Value::Nil)
}

// ============================================================================
// Value Access
// ============================================================================

/// Get the name of a symbol (returns null if not a symbol)
///
/// # Safety
/// - value must be a valid pointer to a Value
/// - Returned string is valid until value is freed
#[unsafe(no_mangle)]
pub unsafe extern "C" fn value_symbol_name(value: *const Value) -> *const c_char {
    let value_ref = unsafe { &*value };

    if let Value::Symbol(sym) = value_ref {
        // Create a CString and leak it - caller must not free
        let c_string = std::ffi::CString::new(sym.name.as_str()).unwrap();
        c_string.into_raw()
    } else {
        std::ptr::null()
    }
}

/// Get the number value (returns 0.0 if not a number)
///
/// # Safety
/// - value must be a valid pointer to a Value
#[unsafe(no_mangle)]
pub unsafe extern "C" fn value_number_get(value: *const Value) -> f64 {
    let value_ref = unsafe { &*value };

    if let Value::Number(n) = value_ref {
        *n
    } else {
        0.0
    }
}

// ============================================================================
// Common Symbol Constructors (for macros)
// ============================================================================

/// Create the symbol "arith.addi"
#[unsafe(no_mangle)]
pub extern "C" fn value_symbol_arith_addi() -> *mut Value {
    Box::into_raw(Box::new(Value::Symbol(Symbol::new("arith.addi"))))
}

/// Create the symbol "arith.subi"
#[unsafe(no_mangle)]
pub extern "C" fn value_symbol_arith_subi() -> *mut Value {
    Box::into_raw(Box::new(Value::Symbol(Symbol::new("arith.subi"))))
}

/// Create the symbol "arith.muli"
#[unsafe(no_mangle)]
pub extern "C" fn value_symbol_arith_muli() -> *mut Value {
    Box::into_raw(Box::new(Value::Symbol(Symbol::new("arith.muli"))))
}

/// Create the symbol "arith.divsi"
#[unsafe(no_mangle)]
pub extern "C" fn value_symbol_arith_divsi() -> *mut Value {
    Box::into_raw(Box::new(Value::Symbol(Symbol::new("arith.divsi"))))
}

/// Create the symbol "func.return"
#[unsafe(no_mangle)]
pub extern "C" fn value_symbol_func_return() -> *mut Value {
    Box::into_raw(Box::new(Value::Symbol(Symbol::new("func.return"))))
}

/// Create the symbol "def"
#[unsafe(no_mangle)]
pub extern "C" fn value_symbol_def() -> *mut Value {
    Box::into_raw(Box::new(Value::Symbol(Symbol::new("def"))))
}

/// Create the symbol "do"
#[unsafe(no_mangle)]
pub extern "C" fn value_symbol_do() -> *mut Value {
    Box::into_raw(Box::new(Value::Symbol(Symbol::new("do"))))
}

/// Create the symbol ":"
#[unsafe(no_mangle)]
pub extern "C" fn value_symbol_colon() -> *mut Value {
    Box::into_raw(Box::new(Value::Symbol(Symbol::new(":"))))
}

// ============================================================================
// Gensym (Unique Symbol Generation)
// ============================================================================

use std::sync::atomic::{AtomicU64, Ordering};

/// Global counter for generating unique symbols
static GENSYM_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Create a unique symbol with the default prefix "__G__"
#[unsafe(no_mangle)]
pub extern "C" fn value_gensym() -> *mut Value {
    let n = GENSYM_COUNTER.fetch_add(1, Ordering::SeqCst);
    let name = format!("__G__{}", n);
    Box::into_raw(Box::new(Value::Symbol(Symbol::new(name))))
}

/// Create a unique symbol with a custom prefix
///
/// # Safety
/// The prefix pointer must be a valid null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn value_gensym_prefix(prefix: *const c_char) -> *mut Value {
    let prefix_str = unsafe { CStr::from_ptr(prefix).to_string_lossy() };
    let n = GENSYM_COUNTER.fetch_add(1, Ordering::SeqCst);
    let name = format!("{}__{}", prefix_str, n);
    Box::into_raw(Box::new(Value::Symbol(Symbol::new(name))))
}

// ============================================================================
// Memory Management
// ============================================================================

/// Free a value
///
/// # Safety
/// - value must be a valid pointer allocated by one of the value_*_new functions
/// - The pointer must not be used after this call
#[unsafe(no_mangle)]
pub unsafe extern "C" fn value_free(value: *mut Value) {
    if !value.is_null() {
        unsafe {
            let _ = Box::from_raw(value);
        }
    }
}

/// Clone a value
///
/// # Safety
/// - value must be a valid pointer to a Value
/// - Returned pointer is a new allocation owned by the caller
#[unsafe(no_mangle)]
pub unsafe extern "C" fn value_clone(value: *const Value) -> *mut Value {
    let value_ref = unsafe { &*value };
    Box::into_raw(Box::new(value_ref.clone()))
}

// ============================================================================
// FFI Registration
// ============================================================================

/// Information about an FFI function
pub struct FfiFunction {
    pub name: &'static str,
    pub ptr: *mut (),
}

/// Get all Value FFI functions for registration with the JIT
pub fn get_value_ffi_functions() -> Vec<FfiFunction> {
    vec![
        FfiFunction {
            name: "value_list_new",
            ptr: value_list_new as *mut (),
        },
        FfiFunction {
            name: "value_vector_new",
            ptr: value_vector_new as *mut (),
        },
        FfiFunction {
            name: "value_symbol_new",
            ptr: value_symbol_new as *mut (),
        },
        FfiFunction {
            name: "value_string_new",
            ptr: value_string_new as *mut (),
        },
        FfiFunction {
            name: "value_number_new",
            ptr: value_number_new as *mut (),
        },
        FfiFunction {
            name: "value_keyword_new",
            ptr: value_keyword_new as *mut (),
        },
        FfiFunction {
            name: "value_bool_new",
            ptr: value_bool_new as *mut (),
        },
        FfiFunction {
            name: "value_nil_new",
            ptr: value_nil_new as *mut (),
        },
        FfiFunction {
            name: "value_list_push",
            ptr: value_list_push as *mut (),
        },
        FfiFunction {
            name: "value_list_first",
            ptr: value_list_first as *mut (),
        },
        FfiFunction {
            name: "value_list_rest",
            ptr: value_list_rest as *mut (),
        },
        FfiFunction {
            name: "value_list_len",
            ptr: value_list_len as *mut (),
        },
        FfiFunction {
            name: "value_list_get",
            ptr: value_list_get as *mut (),
        },
        FfiFunction {
            name: "value_is_list",
            ptr: value_is_list as *mut (),
        },
        FfiFunction {
            name: "value_is_vector",
            ptr: value_is_vector as *mut (),
        },
        FfiFunction {
            name: "value_is_symbol",
            ptr: value_is_symbol as *mut (),
        },
        FfiFunction {
            name: "value_is_string",
            ptr: value_is_string as *mut (),
        },
        FfiFunction {
            name: "value_is_number",
            ptr: value_is_number as *mut (),
        },
        FfiFunction {
            name: "value_is_keyword",
            ptr: value_is_keyword as *mut (),
        },
        FfiFunction {
            name: "value_is_nil",
            ptr: value_is_nil as *mut (),
        },
        FfiFunction {
            name: "value_symbol_name",
            ptr: value_symbol_name as *mut (),
        },
        FfiFunction {
            name: "value_number_get",
            ptr: value_number_get as *mut (),
        },
        FfiFunction {
            name: "value_free",
            ptr: value_free as *mut (),
        },
        FfiFunction {
            name: "value_clone",
            ptr: value_clone as *mut (),
        },
        // Common symbol constructors
        FfiFunction {
            name: "value_symbol_arith_addi",
            ptr: value_symbol_arith_addi as *mut (),
        },
        FfiFunction {
            name: "value_symbol_arith_subi",
            ptr: value_symbol_arith_subi as *mut (),
        },
        FfiFunction {
            name: "value_symbol_arith_muli",
            ptr: value_symbol_arith_muli as *mut (),
        },
        FfiFunction {
            name: "value_symbol_arith_divsi",
            ptr: value_symbol_arith_divsi as *mut (),
        },
        FfiFunction {
            name: "value_symbol_func_return",
            ptr: value_symbol_func_return as *mut (),
        },
        FfiFunction {
            name: "value_symbol_def",
            ptr: value_symbol_def as *mut (),
        },
        FfiFunction {
            name: "value_symbol_do",
            ptr: value_symbol_do as *mut (),
        },
        FfiFunction {
            name: "value_symbol_colon",
            ptr: value_symbol_colon as *mut (),
        },
        // Gensym functions
        FfiFunction {
            name: "value_gensym",
            ptr: value_gensym as *mut (),
        },
        FfiFunction {
            name: "value_gensym_prefix",
            ptr: value_gensym_prefix as *mut (),
        },
        // libc functions for memory allocation
        FfiFunction {
            name: "malloc",
            ptr: libc::malloc as *mut (),
        },
        FfiFunction {
            name: "free",
            ptr: libc::free as *mut (),
        },
        FfiFunction {
            name: "calloc",
            ptr: libc::calloc as *mut (),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_operations() {
        unsafe {
            let list = value_list_new();
            assert!(value_is_list(list));
            assert_eq!(value_list_len(list), 0);

            let sym = value_symbol_new(b"foo\0".as_ptr() as *const c_char);
            value_list_push(list, sym);
            assert_eq!(value_list_len(list), 1);

            let first = value_list_first(list);
            assert!(!first.is_null());
            assert!(value_is_symbol(first));

            value_free(first);
            value_free(sym);
            value_free(list);
        }
    }

    #[test]
    fn test_value_types() {
        unsafe {
            let num = value_number_new(42.0);
            assert!(value_is_number(num));
            assert_eq!(value_number_get(num), 42.0);

            let b = value_bool_new(true);
            assert!(!value_is_number(b));

            let nil = value_nil_new();
            assert!(value_is_nil(nil));

            value_free(num);
            value_free(b);
            value_free(nil);
        }
    }
}
