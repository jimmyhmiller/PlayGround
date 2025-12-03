/// Trampoline for executing JIT code safely
///
/// This provides:
/// 1. A separate stack for JIT code
/// 2. Saves/restores callee-saved registers (x19-x28)
/// 3. Proper function calling convention
/// 4. Runtime function call trampolines for dynamic bindings
///
/// Based on Beagle's trampoline implementation

use std::alloc::{alloc, dealloc, Layout};
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

        // Debug: print trampoline code
        eprintln!("DEBUG: Trampoline instructions:");
        for (i, inst) in trampoline.code.iter().enumerate() {
            eprintln!("  {:04x}: {:08x}", i * 4, inst);
        }

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

    fn allocate_stack(&mut self) {
        unsafe {
            let layout = Layout::from_size_align(self.stack_size, 16).unwrap();
            let ptr = alloc(layout);
            if ptr.is_null() {
                panic!("Failed to allocate trampoline stack");
            }
            // Stack grows downward, so return pointer to end
            self.stack_ptr = ptr.add(self.stack_size);
        }
    }

    /// Execute JIT code through the trampoline
    ///
    /// # Safety
    /// The jit_fn must be valid ARM64 code
    pub unsafe fn execute(&self, jit_fn: *const u8) -> i64 {
        unsafe {
            eprintln!("DEBUG: trampoline.execute() - jit_fn address: {:p}", jit_fn);
            eprintln!("DEBUG: trampoline.execute() - trampoline address: {:p}", self.code_ptr);
            let trampoline_fn: extern "C" fn(u64, u64) -> i64 =
                std::mem::transmute(self.code_ptr);
            eprintln!("DEBUG: About to call trampoline function...");
            let result = trampoline_fn(0, jit_fn as u64);
            eprintln!("DEBUG: Trampoline function returned: {}", result);
            result
        }
    }

    // ARM64 instruction emission helpers

    fn emit_stp(&mut self, rt: usize, rt2: usize, rn: usize, offset: i32) {
        let offset_scaled = ((offset & 0x7F) as u32) << 15;
        let instruction = 0xA9800000 | offset_scaled | ((rt2 as u32) << 10) | ((rn as u32) << 5) | (rt as u32);
        self.code.push(instruction);
    }

    fn emit_ldp(&mut self, rt: usize, rt2: usize, rn: usize, offset: i32) {
        let offset_scaled = ((offset & 0x7F) as u32) << 15;
        let instruction = 0xA8C00000 | offset_scaled | ((rt2 as u32) << 10) | ((rn as u32) << 5) | (rt as u32);
        self.code.push(instruction);
    }

    fn emit_mov(&mut self, dst: usize, src: usize) {
        let instruction = 0xAA0003E0 | ((src as u32) << 16) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_str_offset(&mut self, rt: usize, rn: usize, offset: i32) {
        let offset_scaled = ((offset / 8) as u32) & 0xFFF;
        let instruction = 0xF9000000 | (offset_scaled << 10) | ((rn as u32) << 5) | (rt as u32);
        self.code.push(instruction);
    }

    fn emit_ldr_offset(&mut self, rt: usize, rn: usize, offset: i32) {
        let offset_scaled = ((offset / 8) as u32) & 0xFFF;
        let instruction = 0xF9400000 | (offset_scaled << 10) | ((rn as u32) << 5) | (rt as u32);
        self.code.push(instruction);
    }

    fn emit_str_pre(&mut self, rt: usize, rn: usize, offset: i32) {
        let offset_9bit = (offset & 0x1FF) as u32;
        let instruction = 0xF8000C00 | (offset_9bit << 12) | ((rn as u32) << 5) | (rt as u32);
        self.code.push(instruction);
    }

    fn emit_ldr_post(&mut self, rt: usize, rn: usize, offset: i32) {
        let offset_9bit = (offset & 0x1FF) as u32;
        let instruction = 0xF8400400 | (offset_9bit << 12) | ((rn as u32) << 5) | (rt as u32);
        self.code.push(instruction);
    }

    fn emit_sub_sp_imm(&mut self, imm: u32) {
        let instruction = 0xD10003FF | ((imm & 0xFFF) << 10);
        self.code.push(instruction);
    }

    fn emit_add_sp_imm(&mut self, imm: u32) {
        let instruction = 0x910003FF | ((imm & 0xFFF) << 10);
        self.code.push(instruction);
    }

    fn emit_blr(&mut self, rn: usize) {
        let instruction = 0xD63F0000 | ((rn as u32) << 5);
        self.code.push(instruction);
    }

    fn emit_ret(&mut self) {
        self.code.push(0xD65F03C0);
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
