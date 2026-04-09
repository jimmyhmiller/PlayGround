//! ARM64 native runtime: mmaps executable memory and runs JIT-compiled code.

use tensor_lang_graph::TensorRuntime;

use crate::arm::ArmCode;

pub struct ArmRuntime {
    /// Executable code region (mmap'd with RX permissions).
    code_ptr: *mut u8,
    code_len: usize,
}

// ArmRuntime manages executable memory, safe to send across threads.
unsafe impl Send for ArmRuntime {}

impl ArmRuntime {
    /// Create a new runtime from compiled ARM code.
    pub fn new(arm_code: &ArmCode) -> Self {
        let code_len = arm_code.code.len();
        let page_size = 16384; // macOS ARM64 uses 16KB pages
        let alloc_len = (code_len + page_size - 1) & !(page_size - 1);

        unsafe {
            // On macOS ARM64 with hardened runtime, we need MAP_JIT
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                alloc_len,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_JIT,
                -1,
                0,
            ) as *mut u8;
            assert!(
                ptr != libc::MAP_FAILED as *mut u8,
                "mmap failed: {}",
                std::io::Error::last_os_error()
            );

            // Enable writing (macOS JIT pages start write-protected)
            libc::pthread_jit_write_protect_np(0);

            // Copy code
            std::ptr::copy_nonoverlapping(arm_code.code.as_ptr(), ptr, code_len);

            // Switch back to execute mode
            libc::pthread_jit_write_protect_np(1);

            // Clear instruction cache
            sys_icache_invalidate(ptr as *mut _, alloc_len);

            ArmRuntime {
                code_ptr: ptr,
                code_len: alloc_len,
            }
        }
    }

    /// Run the compiled code with the given inputs.
    pub fn run(&mut self, inputs: &[&[f32]], output_size: usize) -> Vec<f32> {
        self.run_with_dim_params(&[], inputs, output_size)
    }

    /// Run with symbolic dimension parameters and inputs.
    pub fn run_with_dim_params(
        &mut self,
        dim_params: &[i32],
        inputs: &[&[f32]],
        output_size: usize,
    ) -> Vec<f32> {
        // Allocate a flat memory buffer (same model as WASM backend)
        let mut total_input_bytes = 0usize;
        for arr in inputs {
            total_input_bytes = align16(total_input_bytes) + arr.len() * 4;
        }
        let total_bytes = align16(total_input_bytes + 16) + 2 * 1024 * 1024 * 1024;
        let mut memory = vec![0u8; total_bytes];

        // Copy inputs and record their byte offsets
        let mut offset = 16usize;
        let mut input_offsets: Vec<i64> = Vec::new();
        for arr in inputs {
            offset = align16(offset);
            let byte_offset = offset;
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(arr.as_ptr() as *const u8, arr.len() * 4)
            };
            memory[byte_offset..byte_offset + bytes.len()].copy_from_slice(bytes);
            input_offsets.push(byte_offset as i64);
            offset += arr.len() * 4;
        }

        let heap_ptr = align16(offset) as u64;

        // Build params array: [dim_params..., input_offsets...]
        let mut params: Vec<i64> = Vec::new();
        for &dp in dim_params {
            params.push(dp as i64);
        }
        for &off in &input_offsets {
            params.push(off);
        }

        let memory_base = memory.as_mut_ptr();
        let params_ptr = params.as_ptr();

        type JitFn = unsafe extern "C" fn(*mut u8, u64, *const i64) -> u64;
        let func: JitFn = unsafe { std::mem::transmute(self.code_ptr) };

        // Ensure the compiler doesn't optimize away the memory buffer
        let result_offset = unsafe {
            let r = func(memory_base, heap_ptr, params_ptr);
            std::hint::black_box(&memory);
            r
        };

        // Read output
        let result_ptr = result_offset as usize;
        let mut output = Vec::with_capacity(output_size);
        for i in 0..output_size {
            let pos = result_ptr + i * 4;
            let bytes = [memory[pos], memory[pos + 1], memory[pos + 2], memory[pos + 3]];
            output.push(f32::from_le_bytes(bytes));
        }

        output
    }
}

impl Drop for ArmRuntime {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.code_ptr as *mut _, self.code_len);
        }
    }
}

impl TensorRuntime for ArmRuntime {
    fn backend_name(&self) -> &str { "arm" }

    fn run(&mut self, inputs: &[&[f32]], output_size: usize) -> Vec<f32> {
        self.run_with_dim_params(&[], inputs, output_size)
    }

    fn run_with_dim_params(
        &mut self,
        dim_param_values: &[u32],
        inputs: &[&[f32]],
        output_size: usize,
    ) -> Vec<f32> {
        let i32_params: Vec<i32> = dim_param_values.iter().map(|&v| v as i32).collect();
        ArmRuntime::run_with_dim_params(self, &i32_params, inputs, output_size)
    }
}

fn align16(n: usize) -> usize {
    (n + 15) & !15
}

unsafe extern "C" {
    fn sys_icache_invalidate(start: *mut libc::c_void, size: usize);
}
