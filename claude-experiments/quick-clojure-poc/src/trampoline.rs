/// Trampoline for executing JIT code safely
///
/// This provides:
/// 1. A separate stack for JIT code
/// 2. Saves/restores callee-saved registers (x19-x28)
/// 3. Proper function calling convention
/// 4. Runtime function call trampolines for dynamic bindings
///
/// Based on Beagle's trampoline implementation

use std::alloc::{dealloc, Layout};
use std::arch::asm;
use std::collections::HashMap;
use crate::gc_runtime::{GCRuntime, ExceptionHandler};
use std::cell::UnsafeCell;
use std::sync::Arc;

// Global runtime reference (set during initialization)
// SAFETY: Must be initialized before any JIT code runs
static mut RUNTIME: Option<Arc<UnsafeCell<GCRuntime>>> = None;

/// Get the current frame pointer (x29) for GC stack walking
/// This is used by allocation trampolines when gc_always is enabled
#[inline(always)]
fn get_frame_pointer() -> usize {
    let fp: usize;
    unsafe {
        asm!("mov {}, x29", out(reg) fp, options(nomem, nostack, preserves_flags));
    }
    fp
}

/// Set the global runtime reference for trampolines
///
/// SAFETY: Must be called exactly once before any JIT code runs
pub fn set_runtime(runtime: Arc<UnsafeCell<GCRuntime>>) {
    unsafe { RUNTIME = Some(runtime); }
}

// ========== Builtin Wrapper Code Generation ==========

/// Generate ARM64 wrapper functions for all builtins.
/// Returns a map of builtin name -> code pointer.
///
/// Each wrapper is a proper function that can be called via BLR:
/// - Binary ops: x0 = arg0 (tagged), x1 = arg1 (tagged) -> x0 = result (tagged)
/// - Unary ops: x0 = arg (tagged) -> x0 = result (tagged)
///
/// The wrappers untag inputs, perform the operation, and retag the result.
pub fn generate_builtin_wrappers() -> HashMap<&'static str, usize> {
    let mut wrappers = HashMap::new();

    // Allocate a page for all builtin code
    // Each builtin is ~20 bytes max, we have ~15 builtins, so 4KB is plenty
    let page_size = 4096;
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            page_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };

    if ptr == libc::MAP_FAILED {
        panic!("Failed to allocate builtin wrapper code memory");
    }

    let base = ptr as *mut u32;
    let mut offset = 0;

    // Helper to emit a builtin and advance offset
    let mut emit_builtin = |name: &'static str, code: &[u32]| {
        unsafe {
            for (i, &instr) in code.iter().enumerate() {
                *base.add(offset + i) = instr;
            }
        }
        let code_ptr = unsafe { base.add(offset) } as usize;
        wrappers.insert(name, code_ptr);
        offset += code.len();
    };

    // +: (x0, x1) -> x0 + x1  (tagged integers)
    // Untag both, add, retag
    emit_builtin("+", &[
        0xD343FC00,  // lsr x0, x0, #3  (untag arg0)
        0xD343FC21,  // lsr x1, x1, #3  (untag arg1)
        0x8B010000,  // add x0, x0, x1
        0xD37DF000,  // lsl x0, x0, #3  (retag)
        0xD65F03C0,  // ret
    ]);

    // -: (x0, x1) -> x0 - x1
    emit_builtin("-", &[
        0xD343FC00,  // lsr x0, x0, #3
        0xD343FC21,  // lsr x1, x1, #3
        0xCB010000,  // sub x0, x0, x1
        0xD37DF000,  // lsl x0, x0, #3
        0xD65F03C0,  // ret
    ]);

    // *: (x0, x1) -> x0 * x1
    emit_builtin("*", &[
        0xD343FC00,  // lsr x0, x0, #3
        0xD343FC21,  // lsr x1, x1, #3
        0x9B017C00,  // mul x0, x0, x1
        0xD37DF000,  // lsl x0, x0, #3
        0xD65F03C0,  // ret
    ]);

    // /: (x0, x1) -> x0 / x1  (signed division)
    emit_builtin("/", &[
        0xD343FC00,  // lsr x0, x0, #3
        0xD343FC21,  // lsr x1, x1, #3
        0x9AC10C00,  // sdiv x0, x0, x1
        0xD37DF000,  // lsl x0, x0, #3
        0xD65F03C0,  // ret
    ]);

    // <: (x0, x1) -> true if x0 < x1
    // Compare untagged values, return tagged boolean (true=11, false=3)
    emit_builtin("<", &[
        0xD343FC00,  // lsr x0, x0, #3  (untag for proper signed compare)
        0xD343FC21,  // lsr x1, x1, #3
        0xEB01001F,  // cmp x0, x1
        0x9A9FB7E0,  // cset x0, lt (set x0 = 1 if less than, else 0)
        0xD37DF000,  // lsl x0, x0, #3  (0 -> 0, 1 -> 8)
        0x91000C00,  // add x0, x0, #3  (0 -> 3=false, 8 -> 11=true)
        0xD65F03C0,  // ret
    ]);

    // >: (x0, x1) -> true if x0 > x1
    emit_builtin(">", &[
        0xD343FC00,  // lsr x0, x0, #3
        0xD343FC21,  // lsr x1, x1, #3
        0xEB01001F,  // cmp x0, x1
        0x9A9FC7E0,  // cset x0, gt (set x0 = 1 if greater than, else 0)
        0xD37DF000,  // lsl x0, x0, #3
        0x91000C00,  // add x0, x0, #3
        0xD65F03C0,  // ret
    ]);

    // =: (x0, x1) -> true if x0 == x1
    // Compare tagged values directly (identity comparison)
    emit_builtin("=", &[
        0xEB01001F,  // cmp x0, x1  (compare tagged values directly)
        0x9A9F17E0,  // cset x0, eq (set x0 = 1 if equal, else 0)
        0xD37DF000,  // lsl x0, x0, #3
        0x91000C00,  // add x0, x0, #3
        0xD65F03C0,  // ret
    ]);

    // bit-and: (x0, x1) -> x0 & x1
    emit_builtin("bit-and", &[
        0xD343FC00,  // lsr x0, x0, #3
        0xD343FC21,  // lsr x1, x1, #3
        0x8A010000,  // and x0, x0, x1
        0xD37DF000,  // lsl x0, x0, #3
        0xD65F03C0,  // ret
    ]);

    // bit-or: (x0, x1) -> x0 | x1
    emit_builtin("bit-or", &[
        0xD343FC00,  // lsr x0, x0, #3
        0xD343FC21,  // lsr x1, x1, #3
        0xAA010000,  // orr x0, x0, x1
        0xD37DF000,  // lsl x0, x0, #3
        0xD65F03C0,  // ret
    ]);

    // bit-xor: (x0, x1) -> x0 ^ x1
    emit_builtin("bit-xor", &[
        0xD343FC00,  // lsr x0, x0, #3
        0xD343FC21,  // lsr x1, x1, #3
        0xCA010000,  // eor x0, x0, x1
        0xD37DF000,  // lsl x0, x0, #3
        0xD65F03C0,  // ret
    ]);

    // bit-not: (x0) -> ~x0
    emit_builtin("bit-not", &[
        0xD343FC00,  // lsr x0, x0, #3
        0xAA2003E0,  // mvn x0, x0  (orn x0, xzr, x0)
        0xD37DF000,  // lsl x0, x0, #3
        0xD65F03C0,  // ret
    ]);

    // bit-shift-left: (x0, x1) -> x0 << x1
    emit_builtin("bit-shift-left", &[
        0xD343FC00,  // lsr x0, x0, #3  (untag value)
        0xD343FC21,  // lsr x1, x1, #3  (untag shift amount)
        0x9AC12000,  // lsl x0, x0, x1  (lslv)
        0xD37DF000,  // lsl x0, x0, #3  (retag)
        0xD65F03C0,  // ret
    ]);

    // bit-shift-right: (x0, x1) -> x0 >> x1 (arithmetic/signed)
    emit_builtin("bit-shift-right", &[
        0xD343FC00,  // lsr x0, x0, #3
        0xD343FC21,  // lsr x1, x1, #3
        0x9AC12800,  // asr x0, x0, x1  (asrv)
        0xD37DF000,  // lsl x0, x0, #3
        0xD65F03C0,  // ret
    ]);

    // unsigned-bit-shift-right: (x0, x1) -> x0 >>> x1 (logical/unsigned)
    emit_builtin("unsigned-bit-shift-right", &[
        0xD343FC00,  // lsr x0, x0, #3
        0xD343FC21,  // lsr x1, x1, #3
        0x9AC12400,  // lsr x0, x0, x1  (lsrv)
        0xD37DF000,  // lsl x0, x0, #3
        0xD65F03C0,  // ret
    ]);

    // Helper to emit a wrapper that calls a trampoline
    // Args are already in x0, x1, x2 - just need to call the trampoline
    let mut emit_trampoline_wrapper = |name: &'static str, trampoline_addr: usize| {
        let addr = trampoline_addr;
        let mut code = Vec::new();

        // stp x29, x30, [sp, #-16]!   ; save fp/lr
        code.push(0xA9BF7BFD);
        // mov x29, sp                  ; set frame pointer
        code.push(0x910003FD);

        // Load 64-bit address into x9 using movz/movk sequence
        // movz x9, #(addr[15:0])
        code.push(0xD2800009 | (((addr & 0xFFFF) as u32) << 5));
        // movk x9, #(addr[31:16]), lsl 16
        code.push(0xF2A00009 | ((((addr >> 16) & 0xFFFF) as u32) << 5));
        // movk x9, #(addr[47:32]), lsl 32
        code.push(0xF2C00009 | ((((addr >> 32) & 0xFFFF) as u32) << 5));
        // movk x9, #(addr[63:48]), lsl 48
        code.push(0xF2E00009 | ((((addr >> 48) & 0xFFFF) as u32) << 5));

        // blr x9                       ; call trampoline
        code.push(0xD63F0120);
        // ldp x29, x30, [sp], #16      ; restore fp/lr
        code.push(0xA8C17BFD);
        // ret
        code.push(0xD65F03C0);

        unsafe {
            for (i, &instr) in code.iter().enumerate() {
                *base.add(offset + i) = instr;
            }
        }
        let code_ptr = unsafe { base.add(offset) } as usize;
        wrappers.insert(name, code_ptr);
        offset += code.len();
    };

    // make-array: (x0 = length) -> array
    emit_trampoline_wrapper("make-array", trampoline_make_array as usize);

    // aget: (x0 = array, x1 = index) -> value
    emit_trampoline_wrapper("aget", trampoline_aget as usize);

    // aset!: (x0 = array, x1 = index, x2 = value) -> value
    emit_trampoline_wrapper("aset!", trampoline_aset as usize);

    // alength: (x0 = array) -> length
    emit_trampoline_wrapper("alength", trampoline_alength as usize);

    // Make the page executable
    unsafe {
        if libc::mprotect(ptr, page_size, libc::PROT_READ | libc::PROT_EXEC) != 0 {
            libc::munmap(ptr, page_size);
            panic!("Failed to make builtin wrappers executable");
        }

        // Clear instruction cache on ARM64
        #[cfg(target_os = "macos")]
        {
            unsafe extern "C" {
                fn sys_icache_invalidate(start: *const libc::c_void, size: libc::size_t);
            }
            sys_icache_invalidate(ptr, page_size);
        }
    }

    wrappers
}

/// Trampoline: Get var value checking dynamic bindings
///
/// ARM64 Calling Convention:
/// - Args: x0 = var_ptr (tagged)
/// - Returns: x0 = value (tagged)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_var_get_value_dynamic(var_ptr: usize) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &*(*runtime_ptr).as_ref().unwrap().get();
        rt.var_get_value_dynamic(var_ptr)
    }
}

/// Trampoline: Push dynamic binding
///
/// ARM64 Calling Convention:
/// - Args: x0 = var_ptr (tagged), x1 = value (tagged)
/// - Returns: x0 = 0 on success, 1 on error
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_push_binding(var_ptr: usize, value: usize) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();
        match rt.push_binding(var_ptr, value) {
            Ok(()) => 0,
            Err(msg) => {
                eprintln!("IllegalStateException: {}", msg);
                1
            }
        }
    }
}

/// Trampoline: Pop dynamic binding
///
/// ARM64 Calling Convention:
/// - Args: x0 = var_ptr (tagged)
/// - Returns: x0 = 0 on success, 1 on error
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_pop_binding(var_ptr: usize) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();
        match rt.pop_binding(var_ptr) {
            Ok(()) => 0,
            Err(msg) => {
                eprintln!("IllegalStateException: {}", msg);
                1
            }
        }
    }
}

/// Trampoline: Set dynamic binding (for set!)
///
/// ARM64 Calling Convention:
/// - Args: x0 = var_ptr (tagged), x1 = value (tagged)
/// - Returns: x0 = 0 on success, 1 on error
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_set_binding(var_ptr: usize, value: usize) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();
        match rt.set_binding(var_ptr, value) {
            Ok(()) => 0,
            Err(msg) => {
                eprintln!("IllegalStateException: {}", msg);
                1
            }
        }
    }
}

/// Trampoline: Force garbage collection
///
/// ARM64 Calling Convention:
/// - Args: x0 = stack_pointer (current frame pointer / x29)
/// - Returns: x0 = nil (0b111)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_gc(stack_pointer: usize) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();
        rt.gc(stack_pointer);
        7 // nil
    }
}

/// Trampoline: Add object to GC write barrier (for mutable field writes)
///
/// This is critical for generational GC correctness. When storing a pointer
/// to a mutable field in an old-generation object, we must track that object
/// so the GC can find cross-generational references during minor collection.
///
/// ARM64 Calling Convention:
/// - Args: x0 = object_ptr (tagged pointer to deftype instance)
/// - Returns: x0 = nil (0b111)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_gc_add_root(object_ptr: usize) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();
        rt.gc_add_root(object_ptr);
        7 // nil
    }
}

/// Trampoline: Store to mutable field in deftype
///
/// ARM64 Calling Convention:
/// - Args: x0 = object_ptr (tagged), x1 = field_name_ptr (pointer to bytes),
///         x2 = field_name_len, x3 = value (tagged)
/// - Returns: x0 = value (the stored value)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_store_type_field(
    object_ptr: usize,
    field_name_ptr: *const u8,
    field_name_len: usize,
    value: usize,
) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();

        // Convert field name pointer to string slice
        let field_name_bytes = std::slice::from_raw_parts(field_name_ptr, field_name_len);
        let field_name = match std::str::from_utf8(field_name_bytes) {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Error: Invalid UTF-8 in field name");
                return 7; // nil
            }
        };

        match rt.store_type_field_by_name(object_ptr, field_name, value) {
            Ok(v) => v,
            Err(msg) => {
                eprintln!("Error storing field: {}", msg);
                7 // nil
            }
        }
    }
}

/// Trampoline: Print values followed by newline
///
/// This implements the println builtin. It takes a count and a pointer to
/// an array of tagged values on the stack.
///
/// ARM64 Calling Convention:
/// - Args: x0 = count (number of values), x1 = values_ptr (pointer to array of tagged values)
/// - Returns: x0 = nil (0b111)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_println(count: usize, values_ptr: *const usize) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &*(*runtime_ptr).as_ref().unwrap().get();

        // Collect values into a slice
        let values = if count > 0 && !values_ptr.is_null() {
            std::slice::from_raw_parts(values_ptr, count)
        } else {
            &[]
        };

        // Print each value, space-separated
        for (i, &val) in values.iter().enumerate() {
            if i > 0 {
                print!(" ");
            }
            // Use runtime's value_to_string for proper formatting
            let s = rt.format_value(val);
            // format_value wraps strings in quotes, but println should print raw strings
            // Check if it's a string and unwrap
            let kind = crate::gc::types::BuiltInTypes::get_kind(val);
            if kind == crate::gc::types::BuiltInTypes::String && s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
                print!("{}", &s[1..s.len()-1]);
            } else {
                print!("{}", s);
            }
        }
        println!();

        7 // nil
    }
}

/// Trampoline: Allocate function object
///
/// ARM64 Calling Convention:
/// - Args: x0 = name_ptr (0 for anonymous), x1 = code_ptr, x2 = closure_count, x3 = values_ptr
/// - Returns: x0 = function pointer (tagged)
///
/// Note: values_ptr points to an array of closure_count tagged values on the stack
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_allocate_function(
    name_ptr: usize,
    code_ptr: usize,
    closure_count: usize,
    values_ptr: *const usize,
) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();

        // GC before allocation if gc_always is enabled
        rt.maybe_gc_before_alloc(get_frame_pointer());

        // Get function name if provided
        let name = if name_ptr != 0 {
            Some(rt.read_string(name_ptr >> 3)) // Untag string pointer
        } else {
            None
        };

        // Read closure values from the pointer
        let closure_values = if closure_count > 0 {
            let values_slice = std::slice::from_raw_parts(values_ptr, closure_count);
            values_slice.to_vec()
        } else {
            vec![]
        };

        match rt.allocate_function(name, code_ptr, closure_values) {
            Ok(fn_ptr) => fn_ptr,
            Err(msg) => {
                eprintln!("Error allocating function: {}", msg);
                7 // Return nil on error
            }
        }
    }
}

/// Trampoline: Get function code pointer
///
/// ARM64 Calling Convention:
/// - Args: x0 = function pointer (tagged)
/// - Returns: x0 = code pointer (untagged)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_function_code_ptr(fn_ptr: usize) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &*(*runtime_ptr).as_ref().unwrap().get();
        rt.function_code_ptr(fn_ptr)
    }
}

/// Trampoline: Get closure value from function
///
/// ARM64 Calling Convention:
/// - Args: x0 = function pointer (tagged), x1 = index
/// - Returns: x0 = closure value (tagged)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_function_get_closure(fn_ptr: usize, index: usize) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &*(*runtime_ptr).as_ref().unwrap().get();
        rt.function_get_closure(fn_ptr, index)
    }
}

/// Trampoline: Get closure count from function
///
/// ARM64 Calling Convention:
/// - Args: x0 = function pointer (tagged)
/// - Returns: x0 = closure_count (untagged integer)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_function_closure_count(fn_ptr: usize) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &*(*runtime_ptr).as_ref().unwrap().get();
        rt.function_closure_count(fn_ptr)
    }
}

/// Trampoline: Allocate deftype instance
///
/// ARM64 Calling Convention:
/// - Args: x0 = type_id, x1 = field_count, x2 = values_ptr
/// - Returns: x0 = instance pointer (tagged HeapObject)
///
/// Note: values_ptr points to an array of field_count tagged values on the stack
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_allocate_type(
    type_id: usize,
    field_count: usize,
    values_ptr: *const usize,
) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();

        // GC before allocation if gc_always is enabled
        rt.maybe_gc_before_alloc(get_frame_pointer());

        // Read field values from the pointer
        let field_values = if field_count > 0 {
            let values_slice = std::slice::from_raw_parts(values_ptr, field_count);
            values_slice.to_vec()
        } else {
            vec![]
        };

        match rt.allocate_type_instance(type_id, field_values) {
            Ok(obj_ptr) => obj_ptr,
            Err(msg) => {
                eprintln!("Error allocating type instance: {}", msg);
                7 // Return nil on error
            }
        }
    }
}

/// Trampoline: Load field from deftype instance by field index
///
/// ARM64 Calling Convention:
/// - Args: x0 = instance pointer (tagged), x1 = field_index
/// - Returns: x0 = field value (tagged)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_load_type_field(
    obj_ptr: usize,
    field_index: usize,
) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &*(*runtime_ptr).as_ref().unwrap().get();
        rt.read_type_field(obj_ptr, field_index)
    }
}

/// Trampoline: Allocate a float on the heap
///
/// ARM64 Calling Convention:
/// - Args: x0 = f64 bits (as u64)
/// - Returns: x0 = tagged float pointer
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_allocate_float(float_bits: u64) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();

        // GC before allocation if gc_always is enabled
        rt.maybe_gc_before_alloc(get_frame_pointer());

        let value = f64::from_bits(float_bits);
        match rt.allocate_float(value) {
            Ok(ptr) => ptr,
            Err(msg) => {
                eprintln!("Error allocating float: {}", msg);
                7 // Return nil on error
            }
        }
    }
}

/// Trampoline: Load field from deftype instance by field name (runtime lookup)
///
/// ARM64 Calling Convention:
/// - Args: x0 = instance pointer (tagged), x1 = field_name_ptr, x2 = field_name_len
/// - Returns: x0 = field value (tagged), or nil (7) on error
///
/// This performs runtime field lookup:
/// 1. Extracts type_id from object header
/// 2. Looks up type definition in registry
/// 3. Finds field index by name
/// 4. Returns field value
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_load_type_field_by_name(
    obj_ptr: usize,
    field_name_ptr: *const u8,
    field_name_len: usize,
) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &*(*runtime_ptr).as_ref().unwrap().get();

        // Convert field name pointer to string slice
        let field_name_bytes = std::slice::from_raw_parts(field_name_ptr, field_name_len);
        let field_name = match std::str::from_utf8(field_name_bytes) {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Error: Invalid UTF-8 in field name");
                return 7; // nil
            }
        };

        match rt.load_type_field_by_name(obj_ptr, field_name) {
            Ok(value) => value,
            Err(msg) => {
                eprintln!("Error loading field: {}", msg);
                7 // nil
            }
        }
    }
}

// ========== Multi-Arity Function Trampolines ==========

/// Trampoline: Allocate a multi-arity function object
///
/// ARM64 Calling Convention:
/// - Args: x0 = name_ptr (0 for anonymous)
///         x1 = arity_count
///         x2 = arities_ptr (pointer to (param_count, code_ptr) pairs on stack)
///         x3 = variadic_min (usize::MAX if no variadic)
///         x4 = closure_count
///         x5 = closures_ptr (pointer to closure values on stack)
/// - Returns: x0 = tagged closure pointer
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_allocate_multi_arity_fn(
    _name_ptr: usize,
    arity_count: usize,
    arities_ptr: *const usize,
    variadic_min: usize,
    closure_count: usize,
    closures_ptr: *const usize,
) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();

        // GC before allocation if gc_always is enabled
        rt.maybe_gc_before_alloc(get_frame_pointer());

        // Read arities from pointer (each arity is 2 words: param_count, code_ptr)
        let arities: Vec<(usize, usize)> = if arity_count > 0 {
            let arities_slice = std::slice::from_raw_parts(arities_ptr, arity_count * 2);
            arities_slice.chunks(2)
                .map(|chunk| (chunk[0], chunk[1]))
                .collect()
        } else {
            vec![]
        };

        // Read closure values from pointer
        let closure_values: Vec<usize> = if closure_count > 0 {
            let closures_slice = std::slice::from_raw_parts(closures_ptr, closure_count);
            closures_slice.to_vec()
        } else {
            vec![]
        };

        // Convert variadic_min sentinel to Option
        let variadic_min_opt = if variadic_min == usize::MAX {
            None
        } else {
            Some(variadic_min)
        };

        // TODO: Handle name_ptr if non-zero
        match rt.allocate_multi_arity_function(None, arities, variadic_min_opt, closure_values) {
            Ok(fn_ptr) => fn_ptr,
            Err(msg) => {
                eprintln!("Error allocating multi-arity function: {}", msg);
                7 // Return nil on error
            }
        }
    }
}

/// Trampoline: Collect rest arguments into a list
///
/// ARM64 Calling Convention:
/// - Args: x0 = pointer to args array on stack (excess args after fixed params)
///         x1 = count of excess arguments
/// - Returns: x0 = tagged list (cons cells) or nil
///
/// Builds a list from right to left using cons cells.
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_collect_rest_args(
    args_ptr: *const usize,
    count: usize,
) -> usize {
    if count == 0 {
        return 7; // nil
    }

    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();

        // GC before allocation if gc_always is enabled
        rt.maybe_gc_before_alloc(get_frame_pointer());

        // Read args from pointer
        let args_slice = std::slice::from_raw_parts(args_ptr, count);

        // Build list from the values
        match rt.build_list(args_slice) {
            Ok(list) => list,
            Err(msg) => {
                eprintln!("Error building rest args list: {}", msg);
                7 // nil on error
            }
        }
    }
}

/// Trampoline: Look up code pointer for multi-arity function dispatch
///
/// ARM64 Calling Convention:
/// - Args: x0 = tagged closure pointer (multi-arity function)
///         x1 = argument count
/// - Returns: x0 = code pointer to call (or 0 if no matching arity)
///
/// This is called at runtime to determine which arity implementation to invoke.
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_multi_arity_lookup(
    fn_ptr: usize,
    arg_count: usize,
) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &*(*runtime_ptr).as_ref().unwrap().get();

        // Check if this is actually a multi-arity function
        if !rt.is_multi_arity_function(fn_ptr) {
            // Not a multi-arity function - return 0 to indicate error
            eprintln!("Error: trampoline_multi_arity_lookup called on non-multi-arity function");
            return 0;
        }

        match rt.multi_arity_lookup(fn_ptr, arg_count) {
            Some((code_ptr, _is_variadic)) => code_ptr,
            None => {
                eprintln!("Error: No matching arity for {} args", arg_count);
                0 // Return 0 to indicate no matching arity
            }
        }
    }
}

// ========== Exception Handling Trampolines ==========

/// Trampoline: Push exception handler
///
/// ARM64 Calling Convention:
/// - Args: x0 = handler_address, x1 = result_local, x2 = link_register, x3 = stack_pointer, x4 = frame_pointer
/// - Returns: x0 = nil (7)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_push_exception_handler(
    handler_address: usize,
    result_local: isize,  // Negative FP-relative offset
    link_register: usize,
    stack_pointer: usize,
    frame_pointer: usize,
) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();

        let handler = ExceptionHandler {
            handler_address,
            stack_pointer,
            frame_pointer,
            link_register,
            result_local,
        };

        rt.push_exception_handler(handler);
        7 // nil
    }
}

/// Trampoline: Pop exception handler (normal exit from try)
///
/// ARM64 Calling Convention:
/// - Args: none
/// - Returns: x0 = nil (7)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_pop_exception_handler() -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();
        rt.pop_exception_handler();
        7 // nil
    }
}

/// Trampoline: Throw exception - never returns
///
/// ARM64 Calling Convention:
/// - Args: x0 = stack_pointer (for potential stack trace), x1 = exception_value
/// - Never returns (longjmp-like behavior)
///
/// This function:
/// 1. Pops the exception handler
/// 2. Stores exception value at result_local (FP-relative offset)
/// 3. Restores SP, FP, LR from handler
/// 4. Jumps to handler_address (catch block)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_throw(_stack_pointer: usize, exception_value: usize) -> ! {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();

        if let Some(handler) = rt.pop_exception_handler() {
            // Store exception at result_local (FP-relative)
            // result_local is a negative offset from FP (since locals are below FP)
            // Use signed arithmetic to compute the address correctly
            let result_ptr = ((handler.frame_pointer as isize) + handler.result_local) as *mut usize;
            *result_ptr = exception_value;

            // Restore SP, FP, LR and jump to handler
            asm!(
                "mov sp, {sp}",
                "mov x29, {fp}",
                "mov x30, {lr}",
                "br {addr}",
                sp = in(reg) handler.stack_pointer,
                fp = in(reg) handler.frame_pointer,
                lr = in(reg) handler.link_register,
                addr = in(reg) handler.handler_address,
                options(noreturn)
            );
        } else {
            // No handler - format the exception value and abort
            let formatted = rt.format_value(exception_value);
            eprintln!("Uncaught exception: {}", formatted);
            std::process::abort();
        }
    }
}

// ========== Assertion Trampolines ==========

/// Trampoline: Pre-condition assertion failed
///
/// ARM64 Calling Convention:
/// - Args: x0 = stack_pointer (for exception), x1 = condition index (0-based)
/// - Never returns (throws AssertionError)
///
/// Creates an AssertionError with message "Assert failed: :pre condition {index}"
/// and throws it via the exception mechanism.
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_pre_condition_failed(_stack_pointer: usize, condition_index: usize) -> ! {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();

        // Create error message
        let msg = format!("Assert failed: :pre condition {}", condition_index);
        let msg_ptr = match rt.allocate_string(&msg) {
            Ok(ptr) => ptr,
            Err(_) => {
                eprintln!("AssertionError: {}", msg);
                std::process::abort();
            }
        };

        // For now, throw the message string as the exception
        // In a full implementation, we'd create an AssertionError object
        trampoline_throw(0, msg_ptr);
    }
}

/// Trampoline: Post-condition assertion failed
///
/// ARM64 Calling Convention:
/// - Args: x0 = stack_pointer (for exception), x1 = condition index (0-based)
/// - Never returns (throws AssertionError)
///
/// Creates an AssertionError with message "Assert failed: :post condition {index}"
/// and throws it via the exception mechanism.
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_post_condition_failed(_stack_pointer: usize, condition_index: usize) -> ! {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();

        // Create error message
        let msg = format!("Assert failed: :post condition {}", condition_index);
        let msg_ptr = match rt.allocate_string(&msg) {
            Ok(ptr) => ptr,
            Err(_) => {
                eprintln!("AssertionError: {}", msg);
                std::process::abort();
            }
        };

        // For now, throw the message string as the exception
        // In a full implementation, we'd create an AssertionError object
        trampoline_throw(0, msg_ptr);
    }
}

// ========== Protocol System Trampolines ==========

/// Trampoline: Register a protocol method implementation in the vtable
///
/// ARM64 Calling Convention:
/// - Args: x0 = type_id, x1 = protocol_id, x2 = method_index, x3 = fn_ptr (tagged)
/// - Returns: x0 = nil (7)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_register_protocol_method(
    type_id: usize,
    protocol_id: usize,
    method_index: usize,
    fn_ptr: usize,
) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();
        rt.register_protocol_method_impl(type_id, protocol_id, method_index, fn_ptr);
        7 // nil
    }
}

/// Trampoline: Look up a protocol method and return the fn_ptr
///
/// ARM64 Calling Convention:
/// - Args: x0 = target (first arg, used for type dispatch)
///         x1 = method_name_ptr, x2 = method_name_len
/// - Returns: x0 = fn_ptr (tagged function/closure pointer)
///
/// This trampoline:
/// 1. Gets the type_id from the target value
/// 2. Looks up the method implementation in the vtable
/// 3. Returns the fn_ptr (or throws IllegalArgumentException if not found)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_protocol_lookup(
    target: usize,
    method_name_ptr: *const u8,
    method_name_len: usize,
) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();

        // Get method name
        let method_name = std::str::from_utf8_unchecked(
            std::slice::from_raw_parts(method_name_ptr, method_name_len)
        );

        // Get type_id from target
        let type_id = rt.get_type_id_for_value(target);

        // Look up method implementation
        match rt.lookup_protocol_method(type_id, method_name) {
            Some(fn_ptr) => fn_ptr,
            None => {
                // Throw IllegalArgumentException like Clojure
                let type_name = crate::gc_runtime::GCRuntime::builtin_type_name(type_id);
                let error_msg = format!(
                    "IllegalArgumentException: No implementation of method: :{} found for class: {}",
                    method_name, type_name
                );
                let error_str = rt.allocate_string(&error_msg).unwrap_or(7);
                trampoline_throw(0, error_str);
            }
        }
    }
}

/// Trampoline: Intern a keyword constant
///
/// ARM64 Calling Convention:
/// - Args: x0 = keyword_index (index into keyword_constants table)
/// - Returns: x0 = tagged keyword pointer
///
/// This is called the first time a keyword literal is used. After that,
/// the keyword is cached in keyword_heap_ptrs and subsequent calls return
/// the cached pointer (ensuring identity-based equality works).
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_intern_keyword(keyword_index: usize) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();

        match rt.intern_keyword(keyword_index) {
            Ok(ptr) => ptr,
            Err(msg) => {
                eprintln!("Failed to intern keyword: {}", msg);
                // Return nil on error
                7  // nil tagged value
            }
        }
    }
}

// ========== Raw Mutable Array Trampolines ==========

/// Trampoline: Allocate a new raw mutable array
///
/// ARM64 Calling Convention:
/// - Args: x0 = length (tagged integer)
/// - Returns: x0 = tagged array pointer
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_make_array(length: usize) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();

        // GC before allocation if gc_always is enabled
        rt.maybe_gc_before_alloc(get_frame_pointer());

        // Untag the length
        let len = length >> 3;

        match rt.allocate_array(len) {
            Ok(ptr) => ptr,
            Err(msg) => {
                eprintln!("Error allocating array: {}", msg);
                7 // Return nil on error
            }
        }
    }
}

/// Trampoline: Get array element at index
///
/// ARM64 Calling Convention:
/// - Args: x0 = array (tagged), x1 = index (tagged integer)
/// - Returns: x0 = element value (tagged)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_aget(arr_ptr: usize, index: usize) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &*(*runtime_ptr).as_ref().unwrap().get();

        // Untag index
        let idx = index >> 3;

        match rt.array_get(arr_ptr, idx) {
            Ok(value) => value,
            Err(msg) => {
                eprintln!("IndexOutOfBoundsException: {}", msg);
                7 // Return nil on error
            }
        }
    }
}

/// Trampoline: Set array element at index
///
/// ARM64 Calling Convention:
/// - Args: x0 = array (tagged), x1 = index (tagged integer), x2 = value (tagged)
/// - Returns: x0 = value (tagged)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_aset(arr_ptr: usize, index: usize, value: usize) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &*(*runtime_ptr).as_ref().unwrap().get();

        // Untag index
        let idx = index >> 3;

        match rt.array_set(arr_ptr, idx, value) {
            Ok(val) => val,
            Err(msg) => {
                eprintln!("IndexOutOfBoundsException: {}", msg);
                7 // Return nil on error
            }
        }
    }
}

/// Trampoline: Get array length
///
/// ARM64 Calling Convention:
/// - Args: x0 = array (tagged)
/// - Returns: x0 = length (tagged integer)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_alength(arr_ptr: usize) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &*(*runtime_ptr).as_ref().unwrap().get();

        let len = rt.array_length(arr_ptr);
        // Tag as integer
        len << 3
    }
}

/// Trampoline that sets up a safe environment for JIT code execution
///
/// The trampoline:
/// - Saves all callee-saved registers (x19-x28)
/// - Switches to a dedicated stack
/// - Calls the JIT function
/// - Restores registers and stack
pub struct Trampoline {
    code: Vec<u32>,
    code_ptr: *mut u8,
    stack_ptr: *mut u8,
    stack_size: usize,
}

impl Trampoline {
    pub fn new(stack_size: usize) -> Self {
        let mut trampoline = Trampoline {
            code: Vec::new(),
            code_ptr: std::ptr::null_mut(),
            stack_ptr: std::ptr::null_mut(),
            stack_size,
        };

        trampoline.generate_trampoline();

        trampoline.allocate_code();

        trampoline.allocate_stack();

        trampoline
    }

    fn generate_trampoline(&mut self) {
        // Trampoline: fn(stack_ptr: u64, jit_fn: u64) -> u64
        // x0 = JIT stack pointer (top of allocated region), x1 = JIT function pointer
        // Must save ALL callee-saved registers (x19-x28, x29, x30)
        // ARM64 ABI requires these to be preserved across function calls
        //
        // This trampoline switches to a dedicated JIT stack so GC can scan it.

        // Save frame pointer and link register on ORIGINAL stack
        // stp x29, x30, [sp, #-16]!
        self.code.push(0xa9bf7bfd);

        // Save callee-saved registers x19-x28 on ORIGINAL stack (5 pairs = 80 bytes)
        // stp x27, x28, [sp, #-16]!
        self.code.push(0xa9bf77fc);
        // stp x25, x26, [sp, #-16]!
        self.code.push(0xa9bf6ffa);
        // stp x23, x24, [sp, #-16]!
        self.code.push(0xa9bf67f8);
        // stp x21, x22, [sp, #-16]!
        self.code.push(0xa9bf5ff6);
        // stp x19, x20, [sp, #-16]!
        self.code.push(0xa9bf57f4);

        // Save original SP to x20 (which is now safely saved to the stack)
        // mov x20, sp  (alias for: add x20, sp, #0)
        self.code.push(0x910003f4);

        // Switch to JIT stack (x0 contains stack_ptr - top of JIT stack)
        // mov sp, x0  (alias for: add sp, x0, #0)
        self.code.push(0x9100001f);

        // Set up frame pointer on JIT stack
        // mov x29, sp
        self.code.push(0x910003fd);

        // Call the JIT function
        // blr x1
        self.code.push(0xd63f0020);

        // Switch back to original stack
        // mov sp, x20  (alias for: add sp, x20, #0)
        self.code.push(0x9100029f);

        // Restore callee-saved registers x19-x28 from ORIGINAL stack
        // ldp x19, x20, [sp], #16
        self.code.push(0xa8c157f4);
        // ldp x21, x22, [sp], #16
        self.code.push(0xa8c15ff6);
        // ldp x23, x24, [sp], #16
        self.code.push(0xa8c167f8);
        // ldp x25, x26, [sp], #16
        self.code.push(0xa8c16ffa);
        // ldp x27, x28, [sp], #16
        self.code.push(0xa8c177fc);

        // Restore frame pointer and link register
        // ldp x29, x30, [sp], #16
        self.code.push(0xa8c17bfd);

        // Return
        // ret
        self.code.push(0xd65f03c0);
    }

    fn allocate_code(&mut self) {
        unsafe {
            let code_size = self.code.len() * 4;

            if code_size == 0 {
                panic!("Code size is zero!");
            }

            // Allocate executable memory
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                code_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );

            if ptr == libc::MAP_FAILED {
                panic!("Failed to allocate trampoline code memory");
            }

            // Copy code
            let code_bytes = std::slice::from_raw_parts(
                self.code.as_ptr() as *const u8,
                code_size,
            );
            std::ptr::copy_nonoverlapping(code_bytes.as_ptr(), ptr as *mut u8, code_size);

            // Make executable
            if libc::mprotect(ptr, code_size, libc::PROT_READ | libc::PROT_EXEC) != 0 {
                libc::munmap(ptr, code_size);
                panic!("Failed to make trampoline executable");
            }

            // Clear instruction cache on ARM64
            #[cfg(target_os = "macos")]
            {
                unsafe extern "C" {
                    fn sys_icache_invalidate(start: *const libc::c_void, size: libc::size_t);
                }
                sys_icache_invalidate(ptr, code_size);
            }

            self.code_ptr = ptr as *mut u8;
        }
    }

    fn allocate_stack(&mut self) {
        let layout = Layout::from_size_align(self.stack_size, 16).unwrap();
        unsafe {
            let stack_base = std::alloc::alloc(layout);
            if stack_base.is_null() {
                panic!("Failed to allocate JIT stack");
            }
            // Stack grows downward, so stack_ptr is at the TOP of the allocated region
            self.stack_ptr = stack_base.add(self.stack_size);
        }
    }

    /// Allocate executable memory and copy code into it
    /// Returns the code pointer (executable memory address)
    pub fn execute_code(code: &[u32]) -> usize {
        unsafe {
            let code_size = code.len() * 4;

            if code_size == 0 {
                panic!("Code size is zero!");
            }

            // Allocate as READ+WRITE first
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                code_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );

            if ptr == libc::MAP_FAILED {
                panic!("Failed to allocate code memory");
            }

            // Copy code
            let code_bytes = std::slice::from_raw_parts(
                code.as_ptr() as *const u8,
                code_size,
            );
            std::ptr::copy_nonoverlapping(code_bytes.as_ptr(), ptr as *mut u8, code_size);

            // Make executable (can't be writeable and executable at same time on macOS)
            if libc::mprotect(ptr, code_size, libc::PROT_READ | libc::PROT_EXEC) != 0 {
                libc::munmap(ptr, code_size);
                panic!("Failed to make code executable");
            }

            // Clear instruction cache on ARM64
            #[cfg(target_os = "macos")]
            {
                unsafe extern "C" {
                    fn sys_icache_invalidate(start: *const libc::c_void, size: libc::size_t);
                }
                sys_icache_invalidate(ptr, code_size);
            }

            ptr as usize
        }
    }

    /// Execute JIT code through the trampoline
    ///
    /// # Safety
    /// The jit_fn must be valid ARM64 code
    pub unsafe fn execute(&self, jit_fn: *const u8) -> i64 {
        unsafe {
            // Set stack_base in runtime BEFORE executing JIT code
            // This allows GC to scan the JIT stack for roots
            let runtime_ptr = std::ptr::addr_of!(RUNTIME);
            if let Some(runtime) = &*runtime_ptr {
                let rt = &mut *runtime.get();
                // stack_ptr is the TOP of the allocated region (highest address)
                // GC will scan from current frame pointer up to this address
                rt.set_stack_base(self.stack_ptr as usize);
            }

            let trampoline_fn: extern "C" fn(u64, u64) -> i64 =
                std::mem::transmute(self.code_ptr);
            trampoline_fn(self.stack_ptr as u64, jit_fn as u64)
        }
    }

}

impl Drop for Trampoline {
    fn drop(&mut self) {
        unsafe {
            if !self.code_ptr.is_null() {
                let code_size = self.code.len() * 4;
                libc::munmap(self.code_ptr as *mut libc::c_void, code_size);
            }
            if !self.stack_ptr.is_null() {
                let layout = Layout::from_size_align(self.stack_size, 16).unwrap();
                let stack_base = self.stack_ptr.sub(self.stack_size);
                dealloc(stack_base, layout);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trampoline_simple() {
        // Simple function that returns 42
        let mut code: Vec<u32> = Vec::new();

        // mov x0, #42
        code.push(0xD2800540);
        // ret
        code.push(0xD65F03C0);

        unsafe {
            let code_size = code.len() * 4;

            // Allocate as READ+WRITE first
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                code_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );

            if ptr == libc::MAP_FAILED {
                panic!("Failed to allocate code memory");
            }

            // Copy code
            let code_bytes = std::slice::from_raw_parts(
                code.as_ptr() as *const u8,
                code_size,
            );
            std::ptr::copy_nonoverlapping(code_bytes.as_ptr(), ptr as *mut u8, code_size);

            // Make executable (can't be writeable and executable at same time on macOS)
            if libc::mprotect(ptr, code_size, libc::PROT_READ | libc::PROT_EXEC) != 0 {
                libc::munmap(ptr, code_size);
                panic!("Failed to make code executable");
            }

            // Clear instruction cache on ARM64
            #[cfg(target_os = "macos")]
            {
                unsafe extern "C" {
                    fn sys_icache_invalidate(start: *const libc::c_void, size: libc::size_t);
                }
                sys_icache_invalidate(ptr, code_size);
            }

            let trampoline = Trampoline::new(64 * 1024); // 64KB stack
            let result = trampoline.execute(ptr as *const u8);

            assert_eq!(result, 42);

            libc::munmap(ptr, code_size);
        }
    }
}
