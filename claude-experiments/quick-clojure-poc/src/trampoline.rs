use crate::arm_instructions as arm;
use crate::gc_runtime::{ExceptionHandler, GCRuntime};
/// Trampoline for executing JIT code safely
///
/// This provides:
/// 1. A separate stack for JIT code
/// 2. Saves/restores callee-saved registers (x19-x28)
/// 3. Proper function calling convention
/// 4. Runtime function call trampolines for dynamic bindings
///
/// Based on Beagle's trampoline implementation
use std::alloc::{Layout, dealloc};
use std::arch::asm;
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::sync::Arc;

/// Call a closure function with arg_count passed in x9 (for variadic dispatch)
///
/// This uses inline assembly to properly set up x9 before calling the closure.
/// The closure calling convention is:
/// - x0 = closure pointer (tagged)
/// - x1-x7 = user arguments (up to 7)
/// - x9 = total argument count (for variadic rest-arg collection)
///
/// # Safety
/// - code_ptr must be a valid function pointer
/// - closure must be a valid tagged closure pointer
/// - args must have at least 7 elements (padded with nil)
#[cfg(target_arch = "aarch64")]
unsafe fn call_closure_with_arg_count(
    code_ptr: usize,
    closure: usize,
    args: &[usize; 8],
    arg_count: usize,
) -> usize {
    let result: usize;
    // Use a trampoline approach: save inputs to callee-saved regs first,
    // then set up the call
    unsafe { asm!(
        // Save frame pointer and link register (callee-saved)
        "stp x29, x30, [sp, #-16]!",
        "mov x29, sp",

        // Save inputs to scratch registers (x10-x13)
        // x10 = code_ptr, x11 = closure, x12 = args_ptr, x13 = arg_count
        "mov x10, {code_ptr}",
        "mov x11, {closure}",
        "mov x12, {args_ptr}",
        "mov x13, {arg_count}",

        // Now set up the actual call registers
        // x0 = closure
        "mov x0, x11",

        // Load args from array (x12)
        "ldr x1, [x12, #0]",
        "ldr x2, [x12, #8]",
        "ldr x3, [x12, #16]",
        "ldr x4, [x12, #24]",
        "ldr x5, [x12, #32]",
        "ldr x6, [x12, #40]",
        "ldr x7, [x12, #48]",

        // Set x9 = arg_count
        "mov x9, x13",

        // Call the function (code_ptr is in x10)
        "blr x10",

        // Restore frame pointer and link register
        "ldp x29, x30, [sp], #16",

        // Result is in x0
        code_ptr = in(reg) code_ptr,
        closure = in(reg) closure,
        args_ptr = in(reg) args.as_ptr(),
        arg_count = in(reg) arg_count,
        lateout("x0") result,
        // Clobbers - all caller-saved registers that the function might use
        out("x1") _,
        out("x2") _,
        out("x3") _,
        out("x4") _,
        out("x5") _,
        out("x6") _,
        out("x7") _,
        out("x9") _,
        // Also clobber the scratch regs we used
        out("x10") _,
        out("x11") _,
        out("x12") _,
        out("x13") _,
        clobber_abi("C"),
    ); }
    result
}

#[cfg(not(target_arch = "aarch64"))]
unsafe fn call_closure_with_arg_count(
    _code_ptr: usize,
    _closure: usize,
    _args: &[usize; 8],
    _arg_count: usize,
) -> usize {
    panic!("call_closure_with_arg_count only supported on aarch64");
}

// Global runtime reference (set during initialization)
// SAFETY: Must be initialized before any JIT code runs
static mut RUNTIME: Option<Arc<UnsafeCell<GCRuntime>>> = None;

// Static storage for the original stack pointer during JIT execution.
// We can't use a register (x18 is not callee-saved, x19-x28 are used by JIT code).
// This is safe because we only have single-threaded JIT execution.
// SAFETY: Only accessed from a single thread during JIT execution.
static mut TRAMPOLINE_SAVED_SP: usize = 0;

/// Save the original stack pointer before switching to JIT stack
#[inline(never)]
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_save_original_sp(sp: usize) {
    unsafe {
        TRAMPOLINE_SAVED_SP = sp;
    }
}

/// Restore the original stack pointer after JIT execution
#[inline(never)]
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_get_original_sp() -> usize {
    unsafe { TRAMPOLINE_SAVED_SP }
}

/// Set the global runtime reference for trampolines
///
/// SAFETY: Must be called exactly once before any JIT code runs
pub fn set_runtime(runtime: Arc<UnsafeCell<GCRuntime>>) {
    unsafe {
        RUNTIME = Some(runtime);
    }
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
    emit_builtin(
        "+",
        &[
            0xD343FC00, // lsr x0, x0, #3  (untag arg0)
            0xD343FC21, // lsr x1, x1, #3  (untag arg1)
            0x8B010000, // add x0, x0, x1
            0xD37DF000, // lsl x0, x0, #3  (retag)
            0xD65F03C0, // ret
        ],
    );

    // -: (x0, x1) -> x0 - x1
    emit_builtin(
        "-",
        &[
            0xD343FC00, // lsr x0, x0, #3
            0xD343FC21, // lsr x1, x1, #3
            0xCB010000, // sub x0, x0, x1
            0xD37DF000, // lsl x0, x0, #3
            0xD65F03C0, // ret
        ],
    );

    // *: (x0, x1) -> x0 * x1
    emit_builtin(
        "*",
        &[
            0xD343FC00, // lsr x0, x0, #3
            0xD343FC21, // lsr x1, x1, #3
            0x9B017C00, // mul x0, x0, x1
            0xD37DF000, // lsl x0, x0, #3
            0xD65F03C0, // ret
        ],
    );

    // /: (x0, x1) -> x0 / x1  (signed division)
    emit_builtin(
        "/",
        &[
            0xD343FC00, // lsr x0, x0, #3
            0xD343FC21, // lsr x1, x1, #3
            0x9AC10C00, // sdiv x0, x0, x1
            0xD37DF000, // lsl x0, x0, #3
            0xD65F03C0, // ret
        ],
    );

    // <: (x0, x1) -> true if x0 < x1
    // Compare untagged values, return tagged boolean (true=11, false=3)
    emit_builtin(
        "<",
        &[
            0xD343FC00, // lsr x0, x0, #3  (untag for proper signed compare)
            0xD343FC21, // lsr x1, x1, #3
            0xEB01001F, // cmp x0, x1
            0x9A9FB7E0, // cset x0, lt (set x0 = 1 if less than, else 0)
            0xD37DF000, // lsl x0, x0, #3  (0 -> 0, 1 -> 8)
            0x91000C00, // add x0, x0, #3  (0 -> 3=false, 8 -> 11=true)
            0xD65F03C0, // ret
        ],
    );

    // >: (x0, x1) -> true if x0 > x1
    emit_builtin(
        ">",
        &[
            0xD343FC00, // lsr x0, x0, #3
            0xD343FC21, // lsr x1, x1, #3
            0xEB01001F, // cmp x0, x1
            0x9A9FC7E0, // cset x0, gt (set x0 = 1 if greater than, else 0)
            0xD37DF000, // lsl x0, x0, #3
            0x91000C00, // add x0, x0, #3
            0xD65F03C0, // ret
        ],
    );

    // =: (x0, x1) -> true if x0 == x1
    // Compare tagged values directly (identity comparison)
    emit_builtin(
        "=",
        &[
            0xEB01001F, // cmp x0, x1  (compare tagged values directly)
            0x9A9F17E0, // cset x0, eq (set x0 = 1 if equal, else 0)
            0xD37DF000, // lsl x0, x0, #3
            0x91000C00, // add x0, x0, #3
            0xD65F03C0, // ret
        ],
    );

    // bit-and: (x0, x1) -> x0 & x1
    emit_builtin(
        "bit-and",
        &[
            0xD343FC00, // lsr x0, x0, #3
            0xD343FC21, // lsr x1, x1, #3
            0x8A010000, // and x0, x0, x1
            0xD37DF000, // lsl x0, x0, #3
            0xD65F03C0, // ret
        ],
    );

    // bit-or: (x0, x1) -> x0 | x1
    emit_builtin(
        "bit-or",
        &[
            0xD343FC00, // lsr x0, x0, #3
            0xD343FC21, // lsr x1, x1, #3
            0xAA010000, // orr x0, x0, x1
            0xD37DF000, // lsl x0, x0, #3
            0xD65F03C0, // ret
        ],
    );

    // bit-xor: (x0, x1) -> x0 ^ x1
    emit_builtin(
        "bit-xor",
        &[
            0xD343FC00, // lsr x0, x0, #3
            0xD343FC21, // lsr x1, x1, #3
            0xCA010000, // eor x0, x0, x1
            0xD37DF000, // lsl x0, x0, #3
            0xD65F03C0, // ret
        ],
    );

    // bit-not: (x0) -> ~x0
    emit_builtin(
        "bit-not",
        &[
            0xD343FC00, // lsr x0, x0, #3
            0xAA2003E0, // mvn x0, x0  (orn x0, xzr, x0)
            0xD37DF000, // lsl x0, x0, #3
            0xD65F03C0, // ret
        ],
    );

    // bit-shift-left: (x0, x1) -> x0 << x1
    emit_builtin(
        "bit-shift-left",
        &[
            0xD343FC00, // lsr x0, x0, #3  (untag value)
            0xD343FC21, // lsr x1, x1, #3  (untag shift amount)
            0x9AC12000, // lsl x0, x0, x1  (lslv)
            0xD37DF000, // lsl x0, x0, #3  (retag)
            0xD65F03C0, // ret
        ],
    );

    // bit-shift-right: (x0, x1) -> x0 >> x1 (arithmetic/signed)
    emit_builtin(
        "bit-shift-right",
        &[
            0xD343FC00, // lsr x0, x0, #3
            0xD343FC21, // lsr x1, x1, #3
            0x9AC12800, // asr x0, x0, x1  (asrv)
            0xD37DF000, // lsl x0, x0, #3
            0xD65F03C0, // ret
        ],
    );

    // unsigned-bit-shift-right: (x0, x1) -> x0 >>> x1 (logical/unsigned)
    emit_builtin(
        "unsigned-bit-shift-right",
        &[
            0xD343FC00, // lsr x0, x0, #3
            0xD343FC21, // lsr x1, x1, #3
            0x9AC12400, // lsr x0, x0, x1  (lsrv)
            0xD37DF000, // lsl x0, x0, #3
            0xD65F03C0, // ret
        ],
    );

    // Helper to emit a wrapper that calls a trampoline
    // Args are already in x0, x1, x2 - just need to call the trampoline
    // If with_stack_pointer is true, pass SP as first arg and shift original args
    let mut emit_trampoline_wrapper =
        |name: &'static str, trampoline_addr: usize, with_stack_pointer: bool| {
            let addr = trampoline_addr;
            let mut code = Vec::new();

            if with_stack_pointer {
                // Shift args: x2 <- x1, x1 <- x0 (do in reverse order to avoid clobbering)
                // mov x2, x1
                code.push(0xAA0103E2);
                // mov x1, x0
                code.push(0xAA0003E1);
                // add x0, sp, #0   ; pass current stack pointer as first argument
                // This is the correct encoding for "mov x0, sp" on ARM64
                // SP is used for GC to scan all values on the stack, including those
                // pushed via STP by ExternalCallWithSaves before calling allocating operations.
                code.push(0x910003E0);
            }

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

    // make-array: (x0 = length) -> array  [ALLOCATES - needs JIT frame pointer]
    emit_trampoline_wrapper("make-array", trampoline_make_array as usize, true);

    // aget: (x0 = array, x1 = index) -> value
    emit_trampoline_wrapper("aget", trampoline_aget as usize, false);

    // aset!: (x0 = array, x1 = index, x2 = value) -> value
    emit_trampoline_wrapper("aset!", trampoline_aset as usize, false);
    // aset: alias for aset! (ClojureScript uses aset without bang)
    emit_trampoline_wrapper("aset", trampoline_aset as usize, false);

    // alength: (x0 = array) -> length
    emit_trampoline_wrapper("alength", trampoline_alength as usize, false);

    // aclone: (x0 = array) -> new array  [ALLOCATES - needs JIT frame pointer]
    emit_trampoline_wrapper("aclone", trampoline_aclone as usize, true);

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

// ========== Runtime Symbol-Based Var Access ==========
// These trampolines enable forward references by looking up vars at runtime.

/// Trampoline: Load var value by symbol lookup at runtime
///
/// ARM64 Calling Convention:
/// - Args: x0 = ns_symbol_id (u32), x1 = name_symbol_id (u32)
/// - Returns: x0 = value (tagged)
/// - Panics if var not found (throws "Unable to resolve symbol" error)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_load_var_by_symbol(ns_symbol_id: u32, name_symbol_id: u32) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &*(*runtime_ptr).as_ref().unwrap().get();

        // Get symbol strings
        let ns_name = rt.get_symbol(ns_symbol_id).expect("invalid ns_symbol_id");
        let var_name = rt
            .get_symbol(name_symbol_id)
            .expect("invalid name_symbol_id");

        // Look up namespace
        let ns_ptr = rt
            .get_namespace_by_name(ns_name)
            .unwrap_or_else(|| panic!("Namespace not found: {}", ns_name));

        let var_ptr = rt
            .namespace_lookup(ns_ptr, var_name)
            .unwrap_or_else(|| panic!("Unable to resolve symbol: {}/{}", ns_name, var_name));

        // Get and return value
        rt.var_get_value(var_ptr)
    }
}

/// Trampoline: Load var value by symbol with dynamic binding check
///
/// ARM64 Calling Convention:
/// - Args: x0 = ns_symbol_id (u32), x1 = name_symbol_id (u32)
/// - Returns: x0 = value (tagged)
/// - Panics if var not found
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_load_var_by_symbol_dynamic(
    ns_symbol_id: u32,
    name_symbol_id: u32,
) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &*(*runtime_ptr).as_ref().unwrap().get();

        // Get symbol strings
        let ns_name = rt.get_symbol(ns_symbol_id).expect("invalid ns_symbol_id");
        let var_name = rt
            .get_symbol(name_symbol_id)
            .expect("invalid name_symbol_id");

        // Look up namespace
        let ns_ptr = rt
            .get_namespace_by_name(ns_name)
            .unwrap_or_else(|| panic!("Namespace not found: {}", ns_name));

        // Look up var in namespace
        let var_ptr = rt
            .namespace_lookup(ns_ptr, var_name)
            .unwrap_or_else(|| panic!("Unable to resolve symbol: {}/{}", ns_name, var_name));

        // Get value (checking dynamic bindings)
        rt.var_get_value_dynamic(var_ptr)
    }
}

/// Trampoline: Store value to var by symbol (creates var if needed)
///
/// ARM64 Calling Convention:
/// - Args: x0 = ns_symbol_id (u32), x1 = name_symbol_id (u32), x2 = value (tagged)
/// - Returns: x0 = value (tagged) - def returns the value
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_store_var_by_symbol(
    ns_symbol_id: u32,
    name_symbol_id: u32,
    value: usize,
) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();

        // Get symbol strings (clone to avoid borrow issues)
        let ns_name = rt
            .get_symbol(ns_symbol_id)
            .expect("invalid ns_symbol_id")
            .to_string();
        let var_name = rt
            .get_symbol(name_symbol_id)
            .expect("invalid name_symbol_id")
            .to_string();

        // Look up namespace
        let ns_ptr = rt
            .get_namespace_by_name(&ns_name)
            .unwrap_or_else(|| panic!("Namespace not found: {}", ns_name));

        // Look up or create var
        let var_ptr = if let Some(existing) = rt.namespace_lookup(ns_ptr, &var_name) {
            existing
        } else {
            // Create new var
            let (new_var_ptr, symbol_ptr) = rt
                .allocate_var(ns_ptr, &var_name, value)
                .expect("Failed to allocate var");
            rt.namespace_add_binding_with_symbol_ptr(ns_ptr, &var_name, new_var_ptr, symbol_ptr)
                .expect("Failed to add namespace binding");
            new_var_ptr
        };

        // Store value
        rt.var_set_value(var_ptr, value);
        value
    }
}

/// Trampoline: Ensure var exists (creates with nil if needed)
///
/// Used at start of def to enable recursive references within the definition.
///
/// ARM64 Calling Convention:
/// - Args: x0 = ns_symbol_id (u32), x1 = name_symbol_id (u32)
/// - Returns: x0 = nil (7)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_ensure_var_by_symbol(ns_symbol_id: u32, name_symbol_id: u32) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();

        // Get symbol strings (clone to avoid borrow issues)
        let ns_name = rt
            .get_symbol(ns_symbol_id)
            .expect("invalid ns_symbol_id")
            .to_string();
        let var_name = rt
            .get_symbol(name_symbol_id)
            .expect("invalid name_symbol_id")
            .to_string();

        // Look up namespace
        let ns_ptr = rt
            .get_namespace_by_name(&ns_name)
            .unwrap_or_else(|| panic!("Namespace not found: {}", ns_name));

        // Create var if it doesn't exist
        if rt.namespace_lookup(ns_ptr, &var_name).is_none() {
            let nil_value = 7usize; // nil tagged value
            let (new_var_ptr, symbol_ptr) = rt
                .allocate_var(ns_ptr, &var_name, nil_value)
                .expect("Failed to allocate var");
            rt.namespace_add_binding_with_symbol_ptr(ns_ptr, &var_name, new_var_ptr, symbol_ptr)
                .expect("Failed to add namespace binding");
        }

        7 // nil
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
///   x2 = field_name_len, x3 = value (tagged)
/// - Returns: x0 = value (the stored value)
///
/// # Safety
/// Caller must ensure `field_name_ptr` points to valid memory of at least `field_name_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn trampoline_store_type_field(
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
///
/// # Safety
/// Caller must ensure `values_ptr` points to valid memory of at least `count` usize values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn trampoline_println(count: usize, values_ptr: *const usize) -> usize {
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
            if kind == crate::gc::types::BuiltInTypes::String
                && s.starts_with('"')
                && s.ends_with('"')
                && s.len() >= 2
            {
                print!("{}", &s[1..s.len() - 1]);
            } else {
                print!("{}", s);
            }
        }
        println!();

        7 // nil
    }
}

/// Trampoline: Print values with register-based argument passing
///
/// REFACTORED: This version takes values directly in registers instead of on the stack.
/// This allows the compiler to emit ExternalCall instead of the complex Println instruction.
///
/// ARM64 Calling Convention:
/// - Args: x0 = count (0-7), x1-x7 = values (unused args can be anything)
/// - Returns: x0 = nil (0b111)
///
/// For more than 7 values, fall back to multiple calls or the legacy stack-based approach.
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_println_regs(
    count: usize,
    v0: usize,
    v1: usize,
    v2: usize,
    v3: usize,
    v4: usize,
    v5: usize,
    v6: usize,
) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &*(*runtime_ptr).as_ref().unwrap().get();

        let values = [v0, v1, v2, v3, v4, v5, v6];

        // Print each value, space-separated
        for i in 0..count.min(7) {
            if i > 0 {
                print!(" ");
            }
            let val = values[i];
            let s = rt.format_value(val);
            // format_value wraps strings in quotes, but println should print raw strings
            let kind = crate::gc::types::BuiltInTypes::get_kind(val);
            if kind == crate::gc::types::BuiltInTypes::String
                && s.starts_with('"')
                && s.ends_with('"')
                && s.len() >= 2
            {
                print!("{}", &s[1..s.len() - 1]);
            } else {
                print!("{}", s);
            }
        }
        println!();

        7 // nil
    }
}

/// Trampoline: Print single value with newline
///
/// This implements the _println builtin (like Beagle's pattern).
/// Takes a single tagged value and prints it followed by a newline.
///
/// ARM64 Calling Convention:
/// - Args: x0 = tagged value to print
/// - Returns: x0 = nil (0b111)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_println_value(value: usize) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &*(*runtime_ptr).as_ref().unwrap().get();

        // Use runtime's value_to_string for proper formatting
        let s = rt.format_value(value);
        // format_value wraps strings in quotes, but println should print raw strings
        // Check if it's a string and unwrap
        let kind = crate::gc::types::BuiltInTypes::get_kind(value);
        if kind == crate::gc::types::BuiltInTypes::String
            && s.starts_with('"')
            && s.ends_with('"')
            && s.len() >= 2
        {
            println!("{}", &s[1..s.len() - 1]);
        } else {
            println!("{}", s);
        }

        7 // nil
    }
}

/// Trampoline: Print single value without newline
///
/// This implements the _print builtin (like Beagle's pattern).
/// Takes a single tagged value and prints it.
///
/// ARM64 Calling Convention:
/// - Args: x0 = tagged value to print
/// - Returns: x0 = nil (0b111)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_print_value(value: usize) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &*(*runtime_ptr).as_ref().unwrap().get();

        // Use runtime's value_to_string for proper formatting
        let s = rt.format_value(value);
        // format_value wraps strings in quotes, but print should print raw strings
        // Check if it's a string and unwrap
        let kind = crate::gc::types::BuiltInTypes::get_kind(value);
        if kind == crate::gc::types::BuiltInTypes::String
            && s.starts_with('"')
            && s.ends_with('"')
            && s.len() >= 2
        {
            print!("{}", &s[1..s.len() - 1]);
        } else {
            print!("{}", s);
        }

        // Flush stdout to ensure output appears immediately
        use std::io::Write;
        let _ = std::io::stdout().flush();

        7 // nil
    }
}

/// Trampoline: Print newline only
///
/// This implements _newline builtin for when println is called with no args.
///
/// ARM64 Calling Convention:
/// - Args: none
/// - Returns: x0 = nil (0b111)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_newline() -> usize {
    println!();
    7 // nil
}

/// Trampoline: Print space
///
/// This implements _print_space builtin for separating values.
///
/// ARM64 Calling Convention:
/// - Args: none
/// - Returns: x0 = nil (0b111)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_print_space() -> usize {
    print!(" ");
    7 // nil
}

/// Trampoline: Allocate function object
///
/// ARM64 Calling Convention:
/// - Args: x0 = stack_pointer (JIT frame pointer for GC), x1 = name_ptr (0 for anonymous),
///   x2 = code_ptr, x3 = closure_count, x4 = values_ptr
/// - Returns: x0 = function pointer (tagged)
///
/// Note: values_ptr points to an array of closure_count tagged values on the stack
///
/// # Safety
/// Caller must ensure `values_ptr` points to valid memory of at least `closure_count` usize values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn trampoline_allocate_function(
    stack_pointer: usize,
    name_ptr: usize,
    code_ptr: usize,
    closure_count: usize,
    values_ptr: *const usize,
) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();

        // GC before allocation if gc_always is enabled
        rt.maybe_gc_before_alloc(stack_pointer);

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
/// - Args: x0 = stack_pointer (JIT frame pointer for GC), x1 = type_id, x2 = field_count, x3 = values_ptr
/// - Returns: x0 = instance pointer (tagged HeapObject)
///
/// Note: values_ptr points to an array of field_count tagged values on the stack
///
/// # Safety
/// Caller must ensure `values_ptr` points to valid memory of at least `field_count` usize values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn trampoline_allocate_type(
    stack_pointer: usize,
    type_id: usize,
    field_count: usize,
    values_ptr: *const usize,
) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();

        // GC before allocation if gc_always is enabled
        rt.maybe_gc_before_alloc(stack_pointer);

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

/// Trampoline: Allocate deftype instance WITHOUT writing fields
///
/// ARM64 Calling Convention:
/// - Args: x0 = stack_pointer (JIT frame pointer for GC), x1 = type_id, x2 = field_count
/// - Returns: x0 = UNTAGGED pointer to allocated object
///
/// This is used by the refactored MakeType compilation. The caller is responsible for:
/// 1. Writing field values using HeapStore instructions at offsets 1, 2, 3, ... (after 8-byte header)
/// 2. Tagging the result pointer with HeapObject tag (0b110) using Tag instruction
///
/// The trampoline allocates space, writes the header with type_id, and initializes
/// all fields to nil. The JIT code then overwrites fields with actual values.
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_allocate_type_object_raw(
    stack_pointer: usize,
    type_id: usize,
    field_count: usize,
) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();

        // GC before allocation if gc_always is enabled
        rt.maybe_gc_before_alloc(stack_pointer);

        match rt.allocate_type_object_raw(type_id, field_count) {
            Ok(obj_ptr) => obj_ptr,
            Err(msg) => {
                eprintln!("Error allocating type object: {}", msg);
                0 // Return null pointer on error (not nil - this is untagged!)
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
pub extern "C" fn trampoline_load_type_field(obj_ptr: usize, field_index: usize) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &*(*runtime_ptr).as_ref().unwrap().get();
        rt.read_type_field(obj_ptr, field_index)
    }
}

/// Trampoline: Allocate a float on the heap
///
/// ARM64 Calling Convention:
/// - Args: x0 = stack_pointer (JIT frame pointer for GC), x1 = f64 bits (as u64)
/// - Returns: x0 = tagged float pointer
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_allocate_float(stack_pointer: usize, float_bits: u64) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();

        // GC before allocation if gc_always is enabled
        rt.maybe_gc_before_alloc(stack_pointer);

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
///
/// # Safety
/// Caller must ensure `field_name_ptr` points to valid memory of at least `field_name_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn trampoline_load_type_field_by_name(
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

/// Trampoline: Load field by symbol ID (REFACTORED version)
///
/// This avoids the stack-based string passing of trampoline_load_type_field_by_name.
/// The field name is pre-interned as a symbol at compile time, and its ID is passed directly.
///
/// ARM64 Calling Convention:
/// - Args: x0 = instance pointer (tagged), x1 = field_name_symbol_id (untagged symbol index)
/// - Returns: x0 = field value (tagged)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_load_type_field_by_symbol(
    obj_ptr: usize,
    field_symbol_id: usize,
) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &*(*runtime_ptr).as_ref().unwrap().get();

        // Get field name from symbol table
        let field_name = match rt.get_symbol(field_symbol_id as u32) {
            Some(s) => s,
            None => {
                eprintln!(
                    "Error: Unknown symbol ID {} for field access",
                    field_symbol_id
                );
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

/// Trampoline: Store field by symbol ID (REFACTORED version)
///
/// ARM64 Calling Convention:
/// - Args: x0 = instance pointer (tagged), x1 = field_name_symbol_id, x2 = value
/// - Returns: x0 = value (the stored value)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_store_type_field_by_symbol(
    obj_ptr: usize,
    field_symbol_id: usize,
    value: usize,
) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();

        // Get field name from symbol table
        let field_name = match rt.get_symbol(field_symbol_id as u32) {
            Some(s) => s.to_string(), // Clone because we need mutable borrow below
            None => {
                eprintln!(
                    "Error: Unknown symbol ID {} for field store",
                    field_symbol_id
                );
                return 7; // nil
            }
        };

        match rt.store_type_field_by_name(obj_ptr, &field_name, value) {
            Ok(v) => v,
            Err(msg) => {
                eprintln!("Error storing field: {}", msg);
                7 // nil
            }
        }
    }
}

// ========== Multi-Arity Function Trampolines ==========

/// Trampoline: Allocate a multi-arity function object
///
/// ARM64 Calling Convention:
/// - Args: x0 = stack_pointer (JIT frame pointer for GC)
///   x1 = name_ptr (0 for anonymous)
///   x2 = arity_count
///   x3 = arities_ptr (pointer to (param_count, code_ptr) pairs on stack)
///   x4 = variadic_min (usize::MAX if no variadic)
///   x5 = variadic_index (usize::MAX if no variadic)
///   x6 = closure_count
///   x7 = closures_ptr (pointer to closure values on stack)
/// - Returns: x0 = tagged closure pointer
/// # Safety
/// Caller must ensure that `arities_ptr` points to valid memory of at least
/// `arity_count * 2` usize values, and `closures_ptr` points to valid memory
/// of at least `closure_count` usize values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn trampoline_allocate_multi_arity_fn(
    stack_pointer: usize,
    _name_ptr: usize,
    arity_count: usize,
    arities_ptr: *const usize,
    variadic_min: usize,
    variadic_index: usize,
    closure_count: usize,
    closures_ptr: *const usize,
) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();

        // GC before allocation if gc_always is enabled
        rt.maybe_gc_before_alloc(stack_pointer);

        // Read arities from pointer (each arity is 2 words: param_count, code_ptr)
        let arities: Vec<(usize, usize)> = if arity_count > 0 {
            let arities_slice = std::slice::from_raw_parts(arities_ptr, arity_count * 2);
            arities_slice
                .chunks(2)
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

        // Convert variadic_min and variadic_index sentinels to Option
        let variadic_min_opt = if variadic_min == usize::MAX {
            None
        } else {
            Some(variadic_min)
        };

        let variadic_index_opt = if variadic_index == usize::MAX {
            None
        } else {
            Some(variadic_index)
        };

        // TODO: Handle name_ptr if non-zero
        match rt.allocate_multi_arity_function(None, arities, variadic_min_opt, variadic_index_opt, closure_values) {
            Ok(fn_ptr) => fn_ptr,
            Err(msg) => {
                eprintln!("Error allocating multi-arity function: {}", msg);
                7 // Return nil on error
            }
        }
    }
}

/// Trampoline: Collect rest arguments into an IndexedSeq
///
/// ARM64 Calling Convention:
/// - Args: x0 = stack_pointer (JIT frame pointer for GC)
///   x1 = pointer to args array on stack (excess args after fixed params)
///   x2 = count of excess arguments
/// - Returns: x0 = tagged IndexedSeq wrapping an Array, or nil if empty
///
/// Creates an IndexedSeq (like ClojureScript) wrapping a mutable array.
/// # Safety
/// Caller must ensure that `args_ptr` points to valid memory of at least
/// `count` usize values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn trampoline_collect_rest_args(
    stack_pointer: usize,
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
        rt.maybe_gc_before_alloc(stack_pointer);

        // Read args from pointer
        let args_slice = std::slice::from_raw_parts(args_ptr, count);

        // Create IndexedSeq wrapping an array of the values
        match rt.allocate_indexed_seq(args_slice) {
            Ok(indexed_seq) => indexed_seq,
            Err(msg) => {
                eprintln!("Error building rest args IndexedSeq: {}", msg);
                7 // nil on error
            }
        }
    }
}

/// Trampoline: Look up code pointer for multi-arity function dispatch
///
/// ARM64 Calling Convention:
/// - Args: x0 = tagged closure pointer (multi-arity function)
///   x1 = argument count
/// - Returns: x0 = code pointer to call (or 0 if no matching arity)
///
/// This is called at runtime to determine which arity implementation to invoke.
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_multi_arity_lookup(fn_ptr: usize, arg_count: usize) -> usize {
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
    result_local: isize, // Negative FP-relative offset
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
            let result_ptr =
                ((handler.frame_pointer as isize) + handler.result_local) as *mut usize;
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

// ========== Debug Trampolines ==========

/// Trampoline: Debug marker to trace execution
///
/// ARM64 Calling Convention:
/// - Args: x0 = marker value
/// - Returns: x0 = same marker value (pass through)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_debug_marker(marker: usize) -> usize {
    marker
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
pub extern "C" fn trampoline_pre_condition_failed(
    _stack_pointer: usize,
    condition_index: usize,
) -> ! {
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
pub extern "C" fn trampoline_post_condition_failed(
    _stack_pointer: usize,
    condition_index: usize,
) -> ! {
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

/// Trampoline: Register marker protocol satisfaction
///
/// Called when extend-type adds a marker protocol (no methods).
///
/// ARM64 Calling Convention:
/// - Args: x0 = type_id, x1 = protocol_id
/// - Returns: x0 = nil (7)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_register_marker_protocol(type_id: usize, protocol_id: usize) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();
        rt.register_marker_protocol_impl(type_id, protocol_id);
        7 // nil
    }
}

/// Trampoline: Look up a protocol method and return the fn_ptr
///
/// ARM64 Calling Convention:
/// - Args: x0 = target (first arg, used for type dispatch)
///   x1 = method_name_ptr, x2 = method_name_len
/// - Returns: x0 = fn_ptr (tagged function/closure pointer)
///
/// This trampoline:
/// 1. Gets the type_id from the target value
/// 2. Looks up the method implementation in the vtable
/// 3. Returns the fn_ptr (or throws IllegalArgumentException if not found)
/// # Safety
/// Caller must ensure that `method_name_ptr` points to valid UTF-8 memory
/// of at least `method_name_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn trampoline_protocol_lookup(
    target: usize,
    method_name_ptr: *const u8,
    method_name_len: usize,
) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();

        // Get method name
        let method_name = std::str::from_utf8_unchecked(std::slice::from_raw_parts(
            method_name_ptr,
            method_name_len,
        ));

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
                7 // nil tagged value
            }
        }
    }
}

// ========== Type Checking Trampolines ==========

/// Trampoline: Check if a value is an instance of a deftype
///
/// ARM64 Calling Convention:
/// - Args: x0 = expected_type_id (full type ID including DEFTYPE_ID_OFFSET)
///   x1 = value to check (tagged)
/// - Returns: x0 = tagged boolean (true if instance, false otherwise)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_instance_check(expected_type_id: usize, value: usize) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &*(*runtime_ptr).as_ref().unwrap().get();

        let actual_type_id = rt.get_type_id_for_value(value);

        // Return tagged boolean
        // true = (1 << 3) | 0b011 = 11 = 0b01011
        // false = (0 << 3) | 0b011 = 3 = 0b00011
        if actual_type_id == expected_type_id {
            11 // true
        } else {
            3 // false
        }
    }
}

// ========== Raw Mutable Array Trampolines ==========

/// Trampoline: Allocate a new raw mutable array
///
/// ARM64 Calling Convention:
/// - Args: x0 = stack_pointer (JIT frame pointer for GC), x1 = length (tagged integer)
/// - Returns: x0 = tagged array pointer
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_make_array(stack_pointer: usize, length: usize) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();

        // GC before allocation if gc_always is enabled
        rt.maybe_gc_before_alloc(stack_pointer);

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

/// Trampoline: Clone an array
///
/// ARM64 Calling Convention:
/// - Args: x0 = stack_pointer (JIT frame pointer for GC), x1 = array (tagged)
/// - Returns: x0 = new array (tagged)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_aclone(stack_pointer: usize, arr_ptr: usize) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();

        // GC before allocation if gc_always is enabled
        rt.maybe_gc_before_alloc(stack_pointer);

        match rt.array_clone(arr_ptr) {
            Ok(ptr) => ptr,
            Err(msg) => {
                eprintln!("Error cloning array: {}", msg);
                7 // Return nil on error
            }
        }
    }
}

/// Hash a value - works for keywords, strings, and other primitive types
/// - x0 = value (tagged)
/// - Returns: x0 = hash value (tagged integer)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_hash_value(value: usize) -> usize {
    use crate::gc::types::{BuiltInTypes, HeapObject};
    use crate::gc_runtime::TYPE_KEYWORD;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &*(*runtime_ptr).as_ref().unwrap().get();

        let type_id = rt.get_type_id_for_value(value);

        // Compute hash and mask to 60 bits to avoid overflow in construct_int
        // (which reserves 3 bits for tagging)
        let hash = match type_id {
            id if id == BuiltInTypes::Int as usize => {
                // Integer - use the untagged value as the hash (already fits)
                BuiltInTypes::untag(value)
            }
            id if id == TYPE_KEYWORD => {
                // Keyword - hash based on its pointer value (keywords are interned)
                // Use a simple multiplicative hash that's deterministic and fast
                // Mask to 60 bits (isize::MAX >> 3) to fit in tagged integer
                // The pointer value is already unique per keyword due to interning
                let ptr = value >> 3; // Remove tag bits
                // Use a multiplicative hash constant (FNV-1a inspired)
                let h = ptr.wrapping_mul(0x517cc1b727220a95);
                h & 0x0fff_ffff_ffff_ffff
            }
            id if id == BuiltInTypes::String as usize => {
                // String - hash the string content
                let untagged = (value & !0b111) as *const u8;
                let obj = HeapObject::from_untagged(untagged);
                let s = obj.get_str_unchecked();
                let mut hasher = DefaultHasher::new();
                s.hash(&mut hasher);
                (hasher.finish() as usize) & 0x0fff_ffff_ffff_ffff
            }
            _ => {
                // For other types, use pointer identity
                let mut hasher = DefaultHasher::new();
                value.hash(&mut hasher);
                (hasher.finish() as usize) & 0x0fff_ffff_ffff_ffff
            }
        };

        // Return as tagged integer
        BuiltInTypes::construct_int(hash as isize) as usize
    }
}

/// Check if a value is a keyword
/// - x0 = value (tagged)
/// - Returns: x0 = true (0b01011) or false (0b00011)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_is_keyword(value: usize) -> usize {
    use crate::gc_runtime::TYPE_KEYWORD;

    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &*(*runtime_ptr).as_ref().unwrap().get();

        let type_id = rt.get_type_id_for_value(value);
        if type_id == TYPE_KEYWORD {
            11 // true
        } else {
            3 // false
        }
    }
}

/// Check if a value is a cons cell (used for list operations)
/// - x0 = value (tagged)
/// - Returns: x0 = true (0b01011) or false (0b00011)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_is_cons(value: usize) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &*(*runtime_ptr).as_ref().unwrap().get();

        if rt.is_cons(value) {
            11 // true
        } else {
            3 // false
        }
    }
}

/// Get the first element of a cons cell
/// - x0 = cons cell (tagged)
/// - Returns: x0 = head element (tagged)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_cons_first(value: usize) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &*(*runtime_ptr).as_ref().unwrap().get();

        if rt.is_cons(value) {
            rt.cons_head(value)
        } else {
            7 // nil
        }
    }
}

/// Get the rest of a cons cell (the tail)
/// - x0 = cons cell (tagged)
/// - Returns: x0 = tail (tagged, may be nil or another cons)
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_cons_rest(value: usize) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &*(*runtime_ptr).as_ref().unwrap().get();

        if rt.is_cons(value) {
            rt.cons_tail(value)
        } else {
            7 // nil
        }
    }
}

/// Invoke an object as a function via IFn protocol dispatch
///
/// Called when trying to invoke a non-function/closure value (e.g., keyword, map).
/// Looks up -invoke method for obj's type and calls it with appropriate args.
///
/// Args (in ARM64 calling convention):
/// - x0 = obj (the IFn implementor, e.g., keyword)
/// - x1 = arg_count (number of user arguments)
/// - x2 = arg0 (first user arg)
/// - x3 = arg1, x4 = arg2, ... x8 = arg6 (up to 7 user args)
///
/// Returns: result of -invoke call
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_ifn_invoke(
    obj: usize,
    arg_count: usize,
    arg0: usize,
    arg1: usize,
    arg2: usize,
    arg3: usize,
    arg4: usize,
    arg5: usize,
    arg6: usize,
) -> usize {
    use crate::gc::types::HeapObject;
    use crate::gc_runtime::closure_layout;

    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();

        // Get type_id from obj
        let type_id = rt.get_type_id_for_value(obj);

        // Look up -invoke method for this type
        let fn_ptr = match rt.lookup_protocol_method(type_id, "-invoke") {
            Some(ptr) => ptr,
            None => {
                let type_name = crate::gc_runtime::GCRuntime::builtin_type_name(type_id);
                let error_msg = format!(
                    "IllegalArgumentException: {} cannot be cast to clojure.lang.IFn",
                    type_name
                );
                let error_str = rt.allocate_string(&error_msg).unwrap_or(7);
                trampoline_throw(0, error_str);
            }
        };

        // Now we have fn_ptr (tagged function or closure)
        // We need to call it with (obj, arg0, arg1, ...) based on arg_count
        // The arity of -invoke is arg_count + 1 (including 'this')

        let tag = fn_ptr & 0b111;

        if tag == 0b100 {
            // Raw function - untag to get code pointer
            let code_ptr = fn_ptr >> 3;

            // Call with the appropriate number of args
            // -invoke(this, a, b, ...) where 'this' = obj
            type Fn1 = extern "C" fn(usize) -> usize;
            type Fn2 = extern "C" fn(usize, usize) -> usize;
            type Fn3 = extern "C" fn(usize, usize, usize) -> usize;
            type Fn4 = extern "C" fn(usize, usize, usize, usize) -> usize;
            type Fn5 = extern "C" fn(usize, usize, usize, usize, usize) -> usize;
            type Fn6 = extern "C" fn(usize, usize, usize, usize, usize, usize) -> usize;
            type Fn7 = extern "C" fn(usize, usize, usize, usize, usize, usize, usize) -> usize;
            type Fn8 =
                extern "C" fn(usize, usize, usize, usize, usize, usize, usize, usize) -> usize;

            match arg_count {
                0 => std::mem::transmute::<usize, Fn1>(code_ptr)(obj),
                1 => std::mem::transmute::<usize, Fn2>(code_ptr)(obj, arg0),
                2 => std::mem::transmute::<usize, Fn3>(code_ptr)(obj, arg0, arg1),
                3 => std::mem::transmute::<usize, Fn4>(code_ptr)(obj, arg0, arg1, arg2),
                4 => std::mem::transmute::<usize, Fn5>(code_ptr)(obj, arg0, arg1, arg2, arg3),
                5 => std::mem::transmute::<usize, Fn6>(code_ptr)(obj, arg0, arg1, arg2, arg3, arg4),
                6 => std::mem::transmute::<usize, Fn7>(code_ptr)(
                    obj, arg0, arg1, arg2, arg3, arg4, arg5,
                ),
                7 => std::mem::transmute::<usize, Fn8>(code_ptr)(
                    obj, arg0, arg1, arg2, arg3, arg4, arg5, arg6,
                ),
                _ => {
                    let error_msg = format!("Too many arguments to IFn: {}", arg_count);
                    let error_str = rt.allocate_string(&error_msg).unwrap_or(7);
                    trampoline_throw(0, error_str);
                }
            }
        } else if tag == 0b101 {
            // Closure - could be single-arity or multi-arity
            let untagged = fn_ptr >> 3;
            let heap_obj = HeapObject::from_untagged(untagged as *const u8);
            let closure_type_id = heap_obj.get_header().type_id;

            // Check if multi-arity
            use crate::gc_runtime::TYPE_MULTI_ARITY_FN;
            if closure_type_id == TYPE_MULTI_ARITY_FN as u8 {
                // For multi-arity, need to look up the right arity
                // arg_count is the number of user args, -invoke takes arg_count + 1 (including 'this')
                match rt.multi_arity_lookup(fn_ptr, arg_count + 1) {
                    Some((code_ptr, _is_variadic)) => {
                        // Call with closure calling convention
                        type ClosureFn2 = extern "C" fn(usize, usize) -> usize;
                        type ClosureFn3 = extern "C" fn(usize, usize, usize) -> usize;
                        type ClosureFn4 = extern "C" fn(usize, usize, usize, usize) -> usize;

                        return match arg_count {
                            0 => std::mem::transmute::<usize, ClosureFn2>(code_ptr)(fn_ptr, obj),
                            1 => std::mem::transmute::<usize, ClosureFn3>(code_ptr)(
                                fn_ptr, obj, arg0,
                            ),
                            2 => std::mem::transmute::<usize, ClosureFn4>(code_ptr)(
                                fn_ptr, obj, arg0, arg1,
                            ),
                            _ => {
                                let error_msg = format!("Too many arguments to IFn: {}", arg_count);
                                let error_str = rt.allocate_string(&error_msg).unwrap_or(7);
                                trampoline_throw(0, error_str);
                            }
                        };
                    }
                    None => {
                        let error_msg =
                            format!("No matching arity for {} args in -invoke", arg_count + 1);
                        let error_str = rt.allocate_string(&error_msg).unwrap_or(7);
                        trampoline_throw(0, error_str);
                    }
                }
            }

            // Single-arity closure
            let code_ptr = heap_obj.get_field(closure_layout::FIELD_CODE_PTR);

            // For closures, x0 = closure, x1-x7 = user args, x9 = arg count
            // But -invoke's args are (this, a, b, ...) so we need:
            // closure call args: (closure, this, a, b, ...)
            // arg count for -invoke is arg_count + 1
            type ClosureFn2 = extern "C" fn(usize, usize) -> usize;
            type ClosureFn3 = extern "C" fn(usize, usize, usize) -> usize;
            type ClosureFn4 = extern "C" fn(usize, usize, usize, usize) -> usize;
            type ClosureFn5 = extern "C" fn(usize, usize, usize, usize, usize) -> usize;
            type ClosureFn6 = extern "C" fn(usize, usize, usize, usize, usize, usize) -> usize;
            type ClosureFn7 =
                extern "C" fn(usize, usize, usize, usize, usize, usize, usize) -> usize;
            type ClosureFn8 =
                extern "C" fn(usize, usize, usize, usize, usize, usize, usize, usize) -> usize;
            type ClosureFn9 = extern "C" fn(
                usize,
                usize,
                usize,
                usize,
                usize,
                usize,
                usize,
                usize,
                usize,
            ) -> usize;

            match arg_count {
                0 => std::mem::transmute::<usize, ClosureFn2>(code_ptr)(fn_ptr, obj),
                1 => std::mem::transmute::<usize, ClosureFn3>(code_ptr)(fn_ptr, obj, arg0),
                2 => std::mem::transmute::<usize, ClosureFn4>(code_ptr)(fn_ptr, obj, arg0, arg1),
                3 => std::mem::transmute::<usize, ClosureFn5>(code_ptr)(
                    fn_ptr, obj, arg0, arg1, arg2,
                ),
                4 => std::mem::transmute::<usize, ClosureFn6>(code_ptr)(
                    fn_ptr, obj, arg0, arg1, arg2, arg3,
                ),
                5 => std::mem::transmute::<usize, ClosureFn7>(code_ptr)(
                    fn_ptr, obj, arg0, arg1, arg2, arg3, arg4,
                ),
                6 => std::mem::transmute::<usize, ClosureFn8>(code_ptr)(
                    fn_ptr, obj, arg0, arg1, arg2, arg3, arg4, arg5,
                ),
                7 => std::mem::transmute::<usize, ClosureFn9>(code_ptr)(
                    fn_ptr, obj, arg0, arg1, arg2, arg3, arg4, arg5, arg6,
                ),
                _ => {
                    let error_msg = format!("Too many arguments to IFn: {}", arg_count);
                    let error_str = rt.allocate_string(&error_msg).unwrap_or(7);
                    trampoline_throw(0, error_str);
                }
            }
        } else {
            // Not a callable type
            let error_msg = format!("Cannot invoke object with tag {}", tag);
            let error_str = rt.allocate_string(&error_msg).unwrap_or(7);
            trampoline_throw(0, error_str);
        }
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

        // Register numbers for ARM64
        const SP: u8 = 31;
        const FP: u8 = 29;
        const LR: u8 = 30;

        // Save frame pointer and link register on ORIGINAL stack
        self.code.push(arm::stp_pre(FP, LR, SP, -16));

        // Save callee-saved registers x19-x28 on ORIGINAL stack (5 pairs = 80 bytes)
        self.code.push(arm::stp_pre(27, 28, SP, -16));
        self.code.push(arm::stp_pre(25, 26, SP, -16));
        self.code.push(arm::stp_pre(23, 24, SP, -16));
        self.code.push(arm::stp_pre(21, 22, SP, -16));
        self.code.push(arm::stp_pre(19, 20, SP, -16));

        // At this point:
        // x0 = JIT stack top (where we will switch to)
        // x1 = JIT function pointer
        // SP = original stack (with callee-saved regs pushed)
        //
        // Strategy (following beagle): Push original SP onto JIT stack, call JIT,
        // pop original SP after JIT returns (ARM64 ABI guarantees callee restores SP)

        // Save JIT function pointer to x10 (caller-saved scratch register)
        self.code.push(arm::mov(10, 1));

        // Save original SP to x9
        self.code.push(arm::mov_sp(9, SP));

        // Switch to JIT stack: sp = x0 (JIT stack top)
        self.code.push(arm::mov_sp(SP, 0));

        // Push original SP onto JIT stack (16-byte aligned)
        self.code.push(arm::str_pre(9, SP, -16));

        // Call the JIT function
        self.code.push(arm::blr(10));

        // JIT returned with result in x0
        // ARM64 ABI guarantees callee restores SP to where it was at call time
        // So SP now points to where we pushed original SP

        // Pop original SP from JIT stack (post-index: load then increment)
        self.code.push(arm::ldr_post(9, SP, 16));

        // Restore original SP
        self.code.push(arm::mov_sp(SP, 9));

        // Restore callee-saved registers x19-x28 from ORIGINAL stack
        self.code.push(arm::ldp_post(19, 20, SP, 16));
        self.code.push(arm::ldp_post(21, 22, SP, 16));
        self.code.push(arm::ldp_post(23, 24, SP, 16));
        self.code.push(arm::ldp_post(25, 26, SP, 16));
        self.code.push(arm::ldp_post(27, 28, SP, 16));

        // Restore frame pointer and link register
        self.code.push(arm::ldp_post(FP, LR, SP, 16));

        // Return
        self.code.push(arm::ret());
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
            let code_bytes = std::slice::from_raw_parts(self.code.as_ptr() as *const u8, code_size);
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
            let code_bytes = std::slice::from_raw_parts(code.as_ptr() as *const u8, code_size);
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

            let trampoline_fn: extern "C" fn(u64, u64) -> i64 = std::mem::transmute(self.code_ptr);
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

// ========== Macro Invocation ==========

/// Invoke a compiled function with the given arguments at compile time.
/// This is used for macro expansion - the macro function is called with
/// the unevaluated reader forms as arguments.
///
/// Arguments:
/// - rt: The GC runtime
/// - fn_tagged: Tagged function pointer (tag 0b100) or closure (tag 0b101)
/// - args: Slice of tagged reader form arguments
///
/// Returns: The result of the macro function (a tagged reader form)
pub fn invoke_macro(
    rt: &mut GCRuntime,
    fn_tagged: usize,
    args: &[usize],
) -> Result<usize, String> {
    use crate::gc::types::HeapObject;
    use crate::gc_runtime::closure_layout;

    let tag = fn_tagged & 0b111;
    let arg_count = args.len();

    // Pad args to 8 elements for uniform dispatch
    let mut padded_args = [7usize; 8]; // 7 = nil
    for (i, &arg) in args.iter().enumerate().take(8) {
        padded_args[i] = arg;
    }

    if tag == 0b100 {
        // Raw function pointer
        let code_ptr = fn_tagged >> 3;

        // Define function types for different arities
        type Fn0 = extern "C" fn() -> usize;
        type Fn1 = extern "C" fn(usize) -> usize;
        type Fn2 = extern "C" fn(usize, usize) -> usize;
        type Fn3 = extern "C" fn(usize, usize, usize) -> usize;
        type Fn4 = extern "C" fn(usize, usize, usize, usize) -> usize;
        type Fn5 = extern "C" fn(usize, usize, usize, usize, usize) -> usize;
        type Fn6 = extern "C" fn(usize, usize, usize, usize, usize, usize) -> usize;
        type Fn7 = extern "C" fn(usize, usize, usize, usize, usize, usize, usize) -> usize;
        type Fn8 = extern "C" fn(usize, usize, usize, usize, usize, usize, usize, usize) -> usize;

        let result = unsafe {
            match arg_count {
                0 => std::mem::transmute::<usize, Fn0>(code_ptr)(),
                1 => std::mem::transmute::<usize, Fn1>(code_ptr)(padded_args[0]),
                2 => std::mem::transmute::<usize, Fn2>(code_ptr)(padded_args[0], padded_args[1]),
                3 => std::mem::transmute::<usize, Fn3>(code_ptr)(
                    padded_args[0],
                    padded_args[1],
                    padded_args[2],
                ),
                4 => std::mem::transmute::<usize, Fn4>(code_ptr)(
                    padded_args[0],
                    padded_args[1],
                    padded_args[2],
                    padded_args[3],
                ),
                5 => std::mem::transmute::<usize, Fn5>(code_ptr)(
                    padded_args[0],
                    padded_args[1],
                    padded_args[2],
                    padded_args[3],
                    padded_args[4],
                ),
                6 => std::mem::transmute::<usize, Fn6>(code_ptr)(
                    padded_args[0],
                    padded_args[1],
                    padded_args[2],
                    padded_args[3],
                    padded_args[4],
                    padded_args[5],
                ),
                7 => std::mem::transmute::<usize, Fn7>(code_ptr)(
                    padded_args[0],
                    padded_args[1],
                    padded_args[2],
                    padded_args[3],
                    padded_args[4],
                    padded_args[5],
                    padded_args[6],
                ),
                8 => std::mem::transmute::<usize, Fn8>(code_ptr)(
                    padded_args[0],
                    padded_args[1],
                    padded_args[2],
                    padded_args[3],
                    padded_args[4],
                    padded_args[5],
                    padded_args[6],
                    padded_args[7],
                ),
                _ => return Err(format!("Macro has too many arguments: {}", arg_count)),
            }
        };
        Ok(result)
    } else if tag == 0b101 {
        // Closure - need to pass closure as first arg
        let untagged = fn_tagged >> 3;
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        let closure_type_id = heap_obj.get_header().type_id;

        use crate::gc_runtime::TYPE_MULTI_ARITY_FN;
        if closure_type_id == TYPE_MULTI_ARITY_FN as u8 {
            // Multi-arity closure - look up the right arity
            match rt.multi_arity_lookup(fn_tagged, arg_count) {
                Some((code_ptr, is_variadic)) => {
                    // For variadic functions, use apply_fn which properly handles x9
                    if is_variadic {
                        return apply_fn(rt, fn_tagged, args);
                    }

                    // Non-variadic multi-arity - standard call
                    // Closure calling convention: x0 = closure, x1-x7 = args
                    type ClosureFn1 = extern "C" fn(usize) -> usize;
                    type ClosureFn2 = extern "C" fn(usize, usize) -> usize;
                    type ClosureFn3 = extern "C" fn(usize, usize, usize) -> usize;
                    type ClosureFn4 = extern "C" fn(usize, usize, usize, usize) -> usize;
                    type ClosureFn5 = extern "C" fn(usize, usize, usize, usize, usize) -> usize;
                    type ClosureFn6 =
                        extern "C" fn(usize, usize, usize, usize, usize, usize) -> usize;
                    type ClosureFn7 =
                        extern "C" fn(usize, usize, usize, usize, usize, usize, usize) -> usize;
                    type ClosureFn8 = extern "C" fn(
                        usize,
                        usize,
                        usize,
                        usize,
                        usize,
                        usize,
                        usize,
                        usize,
                    ) -> usize;

                    let result = unsafe {
                        match arg_count {
                            0 => std::mem::transmute::<usize, ClosureFn1>(code_ptr)(fn_tagged),
                            1 => std::mem::transmute::<usize, ClosureFn2>(code_ptr)(
                                fn_tagged,
                                padded_args[0],
                            ),
                            2 => std::mem::transmute::<usize, ClosureFn3>(code_ptr)(
                                fn_tagged,
                                padded_args[0],
                                padded_args[1],
                            ),
                            3 => std::mem::transmute::<usize, ClosureFn4>(code_ptr)(
                                fn_tagged,
                                padded_args[0],
                                padded_args[1],
                                padded_args[2],
                            ),
                            4 => std::mem::transmute::<usize, ClosureFn5>(code_ptr)(
                                fn_tagged,
                                padded_args[0],
                                padded_args[1],
                                padded_args[2],
                                padded_args[3],
                            ),
                            5 => std::mem::transmute::<usize, ClosureFn6>(code_ptr)(
                                fn_tagged,
                                padded_args[0],
                                padded_args[1],
                                padded_args[2],
                                padded_args[3],
                                padded_args[4],
                            ),
                            6 => std::mem::transmute::<usize, ClosureFn7>(code_ptr)(
                                fn_tagged,
                                padded_args[0],
                                padded_args[1],
                                padded_args[2],
                                padded_args[3],
                                padded_args[4],
                                padded_args[5],
                            ),
                            7 => std::mem::transmute::<usize, ClosureFn8>(code_ptr)(
                                fn_tagged,
                                padded_args[0],
                                padded_args[1],
                                padded_args[2],
                                padded_args[3],
                                padded_args[4],
                                padded_args[5],
                                padded_args[6],
                            ),
                            _ => return Err(format!("Macro has too many arguments: {}", arg_count)),
                        }
                    };
                    Ok(result)
                }
                None => Err(format!(
                    "Wrong number of arguments ({}) to macro",
                    arg_count
                )),
            }
        } else {
            // Single-arity closure
            let code_ptr = heap_obj.get_field(closure_layout::FIELD_1_CODE_PTR / 8);

            type ClosureFn1 = extern "C" fn(usize) -> usize;
            type ClosureFn2 = extern "C" fn(usize, usize) -> usize;
            type ClosureFn3 = extern "C" fn(usize, usize, usize) -> usize;
            type ClosureFn4 = extern "C" fn(usize, usize, usize, usize) -> usize;
            type ClosureFn5 = extern "C" fn(usize, usize, usize, usize, usize) -> usize;
            type ClosureFn6 = extern "C" fn(usize, usize, usize, usize, usize, usize) -> usize;
            type ClosureFn7 =
                extern "C" fn(usize, usize, usize, usize, usize, usize, usize) -> usize;
            type ClosureFn8 =
                extern "C" fn(usize, usize, usize, usize, usize, usize, usize, usize) -> usize;

            let result = unsafe {
                match arg_count {
                    0 => std::mem::transmute::<usize, ClosureFn1>(code_ptr)(fn_tagged),
                    1 => std::mem::transmute::<usize, ClosureFn2>(code_ptr)(
                        fn_tagged,
                        padded_args[0],
                    ),
                    2 => std::mem::transmute::<usize, ClosureFn3>(code_ptr)(
                        fn_tagged,
                        padded_args[0],
                        padded_args[1],
                    ),
                    3 => std::mem::transmute::<usize, ClosureFn4>(code_ptr)(
                        fn_tagged,
                        padded_args[0],
                        padded_args[1],
                        padded_args[2],
                    ),
                    4 => std::mem::transmute::<usize, ClosureFn5>(code_ptr)(
                        fn_tagged,
                        padded_args[0],
                        padded_args[1],
                        padded_args[2],
                        padded_args[3],
                    ),
                    5 => std::mem::transmute::<usize, ClosureFn6>(code_ptr)(
                        fn_tagged,
                        padded_args[0],
                        padded_args[1],
                        padded_args[2],
                        padded_args[3],
                        padded_args[4],
                    ),
                    6 => std::mem::transmute::<usize, ClosureFn7>(code_ptr)(
                        fn_tagged,
                        padded_args[0],
                        padded_args[1],
                        padded_args[2],
                        padded_args[3],
                        padded_args[4],
                        padded_args[5],
                    ),
                    7 => std::mem::transmute::<usize, ClosureFn8>(code_ptr)(
                        fn_tagged,
                        padded_args[0],
                        padded_args[1],
                        padded_args[2],
                        padded_args[3],
                        padded_args[4],
                        padded_args[5],
                        padded_args[6],
                    ),
                    _ => return Err(format!("Macro has too many arguments: {}", arg_count)),
                }
            };
            Ok(result)
        }
    } else {
        Err(format!(
            "Cannot invoke non-function as macro (tag: {})",
            tag
        ))
    }
}

// ========== Protocol Method Invocation (for primitive dispatch) ==========

/// Invoke a protocol method on a target value.
/// This is used by primitive dispatch functions in gc_runtime.rs when
/// the type is a user-defined type (deftype) that implements a protocol.
///
/// Arguments:
/// - rt: The GC runtime
/// - target: The tagged value to invoke the method on (becomes first arg)
/// - method_name: The protocol method name (e.g., "-first", "-count")
/// - extra_args: Additional arguments (for methods that take more than just 'this')
///
/// Returns: The result of the method call
pub fn invoke_protocol_method(
    rt: &mut GCRuntime,
    target: usize,
    method_name: &str,
    extra_args: &[usize],
) -> Result<usize, String> {
    use crate::gc::types::HeapObject;
    use crate::gc_runtime::closure_layout;

    // Look up the method
    let type_id = rt.get_type_id_for_value(target);
    let fn_ptr = rt
        .lookup_protocol_method(type_id, method_name)
        .ok_or_else(|| {
            format!(
                "No implementation of {} for type {}",
                method_name,
                GCRuntime::builtin_type_name(type_id)
            )
        })?;

    let tag = fn_ptr & 0b111;
    let total_args = 1 + extra_args.len(); // 'this' + extra args

    // Build args array: [target, extra_args...]
    let mut args = vec![target];
    args.extend_from_slice(extra_args);

    if tag == 0b100 {
        // Raw function pointer
        let code_ptr = fn_ptr >> 3;

        type Fn1 = extern "C" fn(usize) -> usize;
        type Fn2 = extern "C" fn(usize, usize) -> usize;
        type Fn3 = extern "C" fn(usize, usize, usize) -> usize;

        let result = unsafe {
            match total_args {
                1 => std::mem::transmute::<usize, Fn1>(code_ptr)(args[0]),
                2 => std::mem::transmute::<usize, Fn2>(code_ptr)(args[0], args[1]),
                3 => std::mem::transmute::<usize, Fn3>(code_ptr)(args[0], args[1], args[2]),
                _ => {
                    return Err(format!(
                        "Protocol method {} has too many arguments: {}",
                        method_name, total_args
                    ))
                }
            }
        };
        Ok(result)
    } else if tag == 0b101 {
        // Closure
        let untagged = fn_ptr >> 3;
        let heap_obj = HeapObject::from_untagged(untagged as *const u8);
        let code_ptr = heap_obj.get_field(closure_layout::FIELD_1_CODE_PTR / 8);

        // Closure calling convention: closure in first position, then args
        type ClosureFn2 = extern "C" fn(usize, usize) -> usize;
        type ClosureFn3 = extern "C" fn(usize, usize, usize) -> usize;
        type ClosureFn4 = extern "C" fn(usize, usize, usize, usize) -> usize;

        let result = unsafe {
            match total_args {
                1 => std::mem::transmute::<usize, ClosureFn2>(code_ptr)(fn_ptr, args[0]),
                2 => std::mem::transmute::<usize, ClosureFn3>(code_ptr)(fn_ptr, args[0], args[1]),
                3 => std::mem::transmute::<usize, ClosureFn4>(code_ptr)(
                    fn_ptr, args[0], args[1], args[2],
                ),
                _ => {
                    return Err(format!(
                        "Protocol method {} has too many arguments: {}",
                        method_name, total_args
                    ))
                }
            }
        };
        Ok(result)
    } else {
        Err(format!(
            "Protocol method {} is not a function (tag: {})",
            method_name, tag
        ))
    }
}

// ========== Apply Trampoline ==========

/// Trampoline: Apply a function to a list of arguments
///
/// ARM64 Calling Convention:
/// - Args: x0 = stack_pointer (JIT frame pointer for GC)
///   x1 = fn_value (tagged function/closure)
///   x2 = args_seq (tagged seq/list of arguments)
/// - Returns: x0 = result of applying the function
///
/// This handles all function types (raw functions, closures, multi-arity, IFn)
/// and properly sets up x9 for variadic functions.
#[unsafe(no_mangle)]
pub extern "C" fn trampoline_apply(
    stack_pointer: usize,
    fn_value: usize,
    args_seq: usize,
) -> usize {
    unsafe {
        let runtime_ptr = std::ptr::addr_of!(RUNTIME);
        let rt = &mut *(*runtime_ptr).as_ref().unwrap().get();

        // GC before operation if needed
        rt.maybe_gc_before_alloc(stack_pointer);

        // Convert args_seq to a Vec<usize>
        let args = match seq_to_vec(rt, args_seq) {
            Ok(v) => v,
            Err(msg) => {
                eprintln!("apply: error converting args to vec: {}", msg);
                return 7; // nil
            }
        };

        // Call the function with the args
        match apply_fn(rt, fn_value, &args) {
            Ok(result) => result,
            Err(msg) => {
                eprintln!("apply: error: {}", msg);
                7 // nil
            }
        }
    }
}

/// Convert a seq/list to a Vec of tagged values
fn seq_to_vec(rt: &mut GCRuntime, seq: usize) -> Result<Vec<usize>, String> {
    use crate::gc_runtime::{TYPE_LIST, TYPE_READER_LIST, TYPE_READER_VECTOR, TYPE_VECTOR};

    if seq == 7 {
        // nil = empty list
        return Ok(vec![]);
    }

    let tag = seq & 0b111;
    if tag != 0b110 {
        return Err(format!("apply: args must be a seq, got tag {}", tag));
    }

    let type_id = rt.get_type_id_for_value(seq);
    let mut result = Vec::new();

    match type_id {
        TYPE_READER_LIST | TYPE_LIST => {
            // Cons-based list - use first/rest
            let mut current = seq;
            loop {
                if current == 7 {
                    break;
                }
                let curr_type = rt.get_type_id_for_value(current);
                if curr_type == TYPE_READER_LIST || curr_type == TYPE_LIST
                {
                    let first = rt.prim_first(current)?;
                    result.push(first);
                    current = rt.prim_rest(current)?;
                } else if current == 7 {
                    break;
                } else {
                    return Err(format!("apply: unexpected type in list: {}", curr_type));
                }
            }
        }
        TYPE_READER_VECTOR | TYPE_VECTOR => {
            // Indexed collection - use count and nth
            let count = rt.prim_count(seq)?;
            for i in 0..count {
                let elem = rt.prim_nth(seq, i)?;
                result.push(elem);
            }
        }
        _ => {
            return Err(format!("apply: unsupported seq type {}", type_id));
        }
    }

    Ok(result)
}

/// Apply a function to a slice of arguments
fn apply_fn(rt: &mut GCRuntime, fn_value: usize, args: &[usize]) -> Result<usize, String> {
    use crate::gc_runtime::{closure_layout, TYPE_FUNCTION, TYPE_MULTI_ARITY_FN};
    use crate::gc::types::HeapObject;

    let tag = fn_value & 0b111;

    // Pad args to 8 elements
    let mut padded_args = [7usize; 8];
    for (i, &arg) in args.iter().take(8).enumerate() {
        padded_args[i] = arg;
    }
    let arg_count = args.len();

    match tag {
        0b100 => {
            // Raw function pointer
            type RawFn0 = extern "C" fn() -> usize;
            type RawFn1 = extern "C" fn(usize) -> usize;
            type RawFn2 = extern "C" fn(usize, usize) -> usize;
            type RawFn3 = extern "C" fn(usize, usize, usize) -> usize;
            type RawFn4 = extern "C" fn(usize, usize, usize, usize) -> usize;
            type RawFn5 = extern "C" fn(usize, usize, usize, usize, usize) -> usize;
            type RawFn6 = extern "C" fn(usize, usize, usize, usize, usize, usize) -> usize;
            type RawFn7 = extern "C" fn(usize, usize, usize, usize, usize, usize, usize) -> usize;
            type RawFn8 = extern "C" fn(usize, usize, usize, usize, usize, usize, usize, usize) -> usize;

            let code_ptr = fn_value >> 3;
            let result = unsafe {
                match arg_count {
                    0 => std::mem::transmute::<usize, RawFn0>(code_ptr)(),
                    1 => std::mem::transmute::<usize, RawFn1>(code_ptr)(padded_args[0]),
                    2 => std::mem::transmute::<usize, RawFn2>(code_ptr)(padded_args[0], padded_args[1]),
                    3 => std::mem::transmute::<usize, RawFn3>(code_ptr)(padded_args[0], padded_args[1], padded_args[2]),
                    4 => std::mem::transmute::<usize, RawFn4>(code_ptr)(padded_args[0], padded_args[1], padded_args[2], padded_args[3]),
                    5 => std::mem::transmute::<usize, RawFn5>(code_ptr)(padded_args[0], padded_args[1], padded_args[2], padded_args[3], padded_args[4]),
                    6 => std::mem::transmute::<usize, RawFn6>(code_ptr)(padded_args[0], padded_args[1], padded_args[2], padded_args[3], padded_args[4], padded_args[5]),
                    7 => std::mem::transmute::<usize, RawFn7>(code_ptr)(padded_args[0], padded_args[1], padded_args[2], padded_args[3], padded_args[4], padded_args[5], padded_args[6]),
                    _ => std::mem::transmute::<usize, RawFn8>(code_ptr)(padded_args[0], padded_args[1], padded_args[2], padded_args[3], padded_args[4], padded_args[5], padded_args[6], padded_args[7]),
                }
            };
            Ok(result)
        }
        0b101 => {
            // Closure - check if single-arity or multi-arity
            let untagged = fn_value >> 3;
            let heap_obj = HeapObject::from_untagged(untagged as *const u8);
            let type_id = heap_obj.get_header().type_id;

            if type_id == TYPE_MULTI_ARITY_FN as u8 {
                // Multi-arity function - look up arity and call with x9 set
                match rt.multi_arity_lookup(fn_value, arg_count) {
                    Some((code_ptr, is_variadic)) => {
                        if is_variadic {
                            // Use the inline asm helper to set x9
                            let result = unsafe {
                                call_closure_with_arg_count(code_ptr, fn_value, &padded_args, arg_count)
                            };
                            Ok(result)
                        } else {
                            // Non-variadic - standard call
                            type ClosureFn1 = extern "C" fn(usize) -> usize;
                            type ClosureFn2 = extern "C" fn(usize, usize) -> usize;
                            type ClosureFn3 = extern "C" fn(usize, usize, usize) -> usize;
                            type ClosureFn4 = extern "C" fn(usize, usize, usize, usize) -> usize;
                            type ClosureFn5 = extern "C" fn(usize, usize, usize, usize, usize) -> usize;
                            type ClosureFn6 = extern "C" fn(usize, usize, usize, usize, usize, usize) -> usize;
                            type ClosureFn7 = extern "C" fn(usize, usize, usize, usize, usize, usize, usize) -> usize;
                            type ClosureFn8 = extern "C" fn(usize, usize, usize, usize, usize, usize, usize, usize) -> usize;

                            let result = unsafe {
                                match arg_count {
                                    0 => std::mem::transmute::<usize, ClosureFn1>(code_ptr)(fn_value),
                                    1 => std::mem::transmute::<usize, ClosureFn2>(code_ptr)(fn_value, padded_args[0]),
                                    2 => std::mem::transmute::<usize, ClosureFn3>(code_ptr)(fn_value, padded_args[0], padded_args[1]),
                                    3 => std::mem::transmute::<usize, ClosureFn4>(code_ptr)(fn_value, padded_args[0], padded_args[1], padded_args[2]),
                                    4 => std::mem::transmute::<usize, ClosureFn5>(code_ptr)(fn_value, padded_args[0], padded_args[1], padded_args[2], padded_args[3]),
                                    5 => std::mem::transmute::<usize, ClosureFn6>(code_ptr)(fn_value, padded_args[0], padded_args[1], padded_args[2], padded_args[3], padded_args[4]),
                                    6 => std::mem::transmute::<usize, ClosureFn7>(code_ptr)(fn_value, padded_args[0], padded_args[1], padded_args[2], padded_args[3], padded_args[4], padded_args[5]),
                                    _ => std::mem::transmute::<usize, ClosureFn8>(code_ptr)(fn_value, padded_args[0], padded_args[1], padded_args[2], padded_args[3], padded_args[4], padded_args[5], padded_args[6]),
                                }
                            };
                            Ok(result)
                        }
                    }
                    None => Err(format!("apply: wrong number of args ({}) for function", arg_count)),
                }
            } else if type_id == TYPE_FUNCTION as u8 {
                // Single-arity closure
                let code_ptr = heap_obj.get_field(closure_layout::FIELD_1_CODE_PTR / 8);

                type ClosureFn1 = extern "C" fn(usize) -> usize;
                type ClosureFn2 = extern "C" fn(usize, usize) -> usize;
                type ClosureFn3 = extern "C" fn(usize, usize, usize) -> usize;
                type ClosureFn4 = extern "C" fn(usize, usize, usize, usize) -> usize;
                type ClosureFn5 = extern "C" fn(usize, usize, usize, usize, usize) -> usize;
                type ClosureFn6 = extern "C" fn(usize, usize, usize, usize, usize, usize) -> usize;
                type ClosureFn7 = extern "C" fn(usize, usize, usize, usize, usize, usize, usize) -> usize;
                type ClosureFn8 = extern "C" fn(usize, usize, usize, usize, usize, usize, usize, usize) -> usize;

                let result = unsafe {
                    match arg_count {
                        0 => std::mem::transmute::<usize, ClosureFn1>(code_ptr)(fn_value),
                        1 => std::mem::transmute::<usize, ClosureFn2>(code_ptr)(fn_value, padded_args[0]),
                        2 => std::mem::transmute::<usize, ClosureFn3>(code_ptr)(fn_value, padded_args[0], padded_args[1]),
                        3 => std::mem::transmute::<usize, ClosureFn4>(code_ptr)(fn_value, padded_args[0], padded_args[1], padded_args[2]),
                        4 => std::mem::transmute::<usize, ClosureFn5>(code_ptr)(fn_value, padded_args[0], padded_args[1], padded_args[2], padded_args[3]),
                        5 => std::mem::transmute::<usize, ClosureFn6>(code_ptr)(fn_value, padded_args[0], padded_args[1], padded_args[2], padded_args[3], padded_args[4]),
                        6 => std::mem::transmute::<usize, ClosureFn7>(code_ptr)(fn_value, padded_args[0], padded_args[1], padded_args[2], padded_args[3], padded_args[4], padded_args[5]),
                        _ => std::mem::transmute::<usize, ClosureFn8>(code_ptr)(fn_value, padded_args[0], padded_args[1], padded_args[2], padded_args[3], padded_args[4], padded_args[5], padded_args[6]),
                    }
                };
                Ok(result)
            } else {
                Err(format!("apply: unknown closure type_id {}", type_id))
            }
        }
        _ => {
            // Try IFn invoke
            Err(format!("apply: cannot apply value with tag {}", tag))
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
            let code_bytes = std::slice::from_raw_parts(code.as_ptr() as *const u8, code_size);
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
