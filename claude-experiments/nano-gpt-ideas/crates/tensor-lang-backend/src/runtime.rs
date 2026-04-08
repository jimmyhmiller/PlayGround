//! In-process WASM execution via wasmtime.
//!
//! Runs WASM modules emitted by `wasm::WasmBackend` directly from Rust,
//! no Node.js required.

use tensor_lang_graph::TensorRuntime;
use wasmtime::*;

pub struct WasmRuntime {
    store: Store<()>,
    instance: Instance,
    memory: Memory,
}

/// Error type for WASM runtime operations.
#[derive(Debug)]
pub enum RuntimeError {
    Wasmtime(wasmtime::Error),
    NoMemory,
    NoExecute,
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeError::Wasmtime(e) => write!(f, "wasmtime error: {e}"),
            RuntimeError::NoMemory => write!(f, "no memory export"),
            RuntimeError::NoExecute => write!(f, "no execute export"),
        }
    }
}

impl From<wasmtime::Error> for RuntimeError {
    fn from(e: wasmtime::Error) -> Self {
        RuntimeError::Wasmtime(e)
    }
}

impl TensorRuntime for WasmRuntime {
    fn backend_name(&self) -> &str { "wasm" }

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
        WasmRuntime::run_with_dim_params(self, &i32_params, inputs, output_size)
    }
}

fn align16(n: usize) -> usize {
    (n + 15) & !15
}

impl WasmRuntime {
    /// Create a new runtime from WASM bytes.
    pub fn new(wasm_bytes: &[u8]) -> Result<Self, RuntimeError> {
        let engine = Engine::default();
        let module = Module::new(&engine, wasm_bytes)?;
        let mut store = Store::new(&engine, ());
        let instance = Instance::new(&mut store, &module, &[])?;

        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or(RuntimeError::NoMemory)?;

        // Verify execute exists
        instance
            .get_func(&mut store, "execute")
            .ok_or(RuntimeError::NoExecute)?;

        Ok(WasmRuntime {
            store,
            instance,
            memory,
        })
    }

    /// Run the execute function with the given inputs.
    /// Returns the flat f32 output buffer.
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
        // Layout inputs into linear memory starting at offset 16
        let mut offset = 16usize;
        let mut input_ptrs: Vec<i32> = Vec::new();

        // Calculate total memory needed
        let mut total_needed = 16usize;
        for arr in inputs {
            total_needed = align16(total_needed) + arr.len() * 4;
        }
        // Add generous space for intermediates
        total_needed = align16(total_needed) + 64 * 1024 * 1024; // 64MB for intermediates

        // Grow memory if needed
        let current_bytes = self.memory.data_size(&self.store);
        if total_needed > current_bytes {
            let needed_pages = ((total_needed - current_bytes) + 65535) / 65536;
            let _ = self.memory.grow(&mut self.store, needed_pages as u64);
        }

        // Copy inputs
        for arr in inputs {
            offset = align16(offset);
            let byte_offset = offset;
            let data = self.memory.data_mut(&mut self.store);
            for (i, &v) in arr.iter().enumerate() {
                let bytes = v.to_le_bytes();
                let pos = byte_offset + i * 4;
                data[pos..pos + 4].copy_from_slice(&bytes);
            }
            input_ptrs.push(byte_offset as i32);
            offset += arr.len() * 4;
        }

        // Set heap pointer at byte 0
        offset = align16(offset);
        {
            let data = self.memory.data_mut(&mut self.store);
            let heap_bytes = (offset as i32).to_le_bytes();
            data[0..4].copy_from_slice(&heap_bytes);
        }

        // Build params: dim_params followed by input_ptrs
        let mut params: Vec<Val> = Vec::new();
        for &dp in dim_params {
            params.push(Val::I32(dp));
        }
        for &ptr in &input_ptrs {
            params.push(Val::I32(ptr));
        }

        // Call execute
        let execute_fn = self.instance
            .get_func(&mut self.store, "execute")
            .expect("no execute export");

        let mut results = [Val::I32(0)];
        execute_fn.call(&mut self.store, &params, &mut results)
            .expect("execute() failed");

        let result_ptr = match results[0] {
            Val::I32(v) => v as usize,
            _ => panic!("execute() returned non-i32"),
        };

        // Read output
        let data = self.memory.data(&self.store);
        let mut output = Vec::with_capacity(output_size);
        for i in 0..output_size {
            let pos = result_ptr + i * 4;
            let bytes = [data[pos], data[pos + 1], data[pos + 2], data[pos + 3]];
            output.push(f32::from_le_bytes(bytes));
        }

        output
    }
}
