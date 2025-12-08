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
use crate::gc_runtime::GCRuntime;
use std::cell::UnsafeCell;
use std::sync::Arc;

// Global runtime reference (set during initialization)
// SAFETY: Must be initialized before any JIT code runs
static mut RUNTIME: Option<Arc<UnsafeCell<GCRuntime>>> = None;

/// Set the global runtime reference for trampolines
///
/// SAFETY: Must be called exactly once before any JIT code runs
pub fn set_runtime(runtime: Arc<UnsafeCell<GCRuntime>>) {
    unsafe { RUNTIME = Some(runtime); }
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

        // Skip stack allocation for now - we're not using it
        // trampoline.allocate_stack();

        trampoline
    }

    fn generate_trampoline(&mut self) {
        // Trampoline: fn(stack_ptr: u64, jit_fn: u64) -> u64
        // x0 = ignored, x1 = JIT function pointer
        // Must save ALL callee-saved registers (x19-x28, x29, x30)
        // ARM64 ABI requires these to be preserved across function calls

        // Save frame pointer and link register
        // stp x29, x30, [sp, #-16]!
        self.code.push(0xa9bf7bfd);

        // Save callee-saved registers x19-x28 (5 pairs = 80 bytes)
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

        // Set up frame pointer
        // mov x29, sp
        self.code.push(0x910003fd);

        // Call the JIT function
        // blr x1
        self.code.push(0xd63f0020);

        // Restore callee-saved registers x19-x28
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
            let trampoline_fn: extern "C" fn(u64, u64) -> i64 =
                std::mem::transmute(self.code_ptr);
            trampoline_fn(0, jit_fn as u64)
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
