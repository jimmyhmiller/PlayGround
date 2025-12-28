//! GPT-2 checkpoint loading and validation utilities
//!
//! This module provides functions to load llm.c format checkpoints
//! and debug states for GPT-2 inference validation.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// GPT-2 model configuration
#[derive(Debug, Clone)]
pub struct GPT2Config {
    pub max_seq_len: usize,   // 1024
    pub vocab_size: usize,    // 50257
    pub padded_vocab_size: usize, // 50304
    pub num_layers: usize,    // 12
    pub num_heads: usize,     // 12
    pub channels: usize,      // 768
}

impl Default for GPT2Config {
    fn default() -> Self {
        // GPT-2 Small configuration
        Self {
            max_seq_len: 1024,
            vocab_size: 50257,
            padded_vocab_size: 50304,
            num_layers: 12,
            num_heads: 12,
            channels: 768,
        }
    }
}

/// Parameter tensor offsets and sizes
#[derive(Debug, Clone)]
pub struct ParameterSpec {
    pub name: &'static str,
    pub offset: usize,  // offset in floats from start of params
    pub size: usize,    // size in floats
}

impl GPT2Config {
    /// Calculate all parameter specifications based on model config
    pub fn param_specs(&self) -> Vec<ParameterSpec> {
        let c = self.channels;
        let l = self.num_layers;
        let v = self.padded_vocab_size;
        let max_t = self.max_seq_len;

        let mut specs = Vec::new();
        let mut offset = 0usize;

        // wte: (V, C) - token embeddings
        let wte_size = v * c;
        specs.push(ParameterSpec { name: "wte", offset, size: wte_size });
        offset += wte_size;

        // wpe: (maxT, C) - position embeddings
        let wpe_size = max_t * c;
        specs.push(ParameterSpec { name: "wpe", offset, size: wpe_size });
        offset += wpe_size;

        // Per-layer weights (all stored contiguously by parameter type, not by layer)
        // ln1w: (L, C)
        let ln1w_size = l * c;
        specs.push(ParameterSpec { name: "ln1w", offset, size: ln1w_size });
        offset += ln1w_size;

        // ln1b: (L, C)
        let ln1b_size = l * c;
        specs.push(ParameterSpec { name: "ln1b", offset, size: ln1b_size });
        offset += ln1b_size;

        // qkvw: (L, 3*C, C)
        let qkvw_size = l * 3 * c * c;
        specs.push(ParameterSpec { name: "qkvw", offset, size: qkvw_size });
        offset += qkvw_size;

        // qkvb: (L, 3*C)
        let qkvb_size = l * 3 * c;
        specs.push(ParameterSpec { name: "qkvb", offset, size: qkvb_size });
        offset += qkvb_size;

        // attprojw: (L, C, C)
        let attprojw_size = l * c * c;
        specs.push(ParameterSpec { name: "attprojw", offset, size: attprojw_size });
        offset += attprojw_size;

        // attprojb: (L, C)
        let attprojb_size = l * c;
        specs.push(ParameterSpec { name: "attprojb", offset, size: attprojb_size });
        offset += attprojb_size;

        // ln2w: (L, C)
        let ln2w_size = l * c;
        specs.push(ParameterSpec { name: "ln2w", offset, size: ln2w_size });
        offset += ln2w_size;

        // ln2b: (L, C)
        let ln2b_size = l * c;
        specs.push(ParameterSpec { name: "ln2b", offset, size: ln2b_size });
        offset += ln2b_size;

        // fcw: (L, 4*C, C)
        let fcw_size = l * 4 * c * c;
        specs.push(ParameterSpec { name: "fcw", offset, size: fcw_size });
        offset += fcw_size;

        // fcb: (L, 4*C)
        let fcb_size = l * 4 * c;
        specs.push(ParameterSpec { name: "fcb", offset, size: fcb_size });
        offset += fcb_size;

        // fcprojw: (L, C, 4*C)
        let fcprojw_size = l * c * 4 * c;
        specs.push(ParameterSpec { name: "fcprojw", offset, size: fcprojw_size });
        offset += fcprojw_size;

        // fcprojb: (L, C)
        let fcprojb_size = l * c;
        specs.push(ParameterSpec { name: "fcprojb", offset, size: fcprojb_size });
        offset += fcprojb_size;

        // lnfw: (C,) - final layer norm weight
        let lnfw_size = c;
        specs.push(ParameterSpec { name: "lnfw", offset, size: lnfw_size });
        offset += lnfw_size;

        // lnfb: (C,) - final layer norm bias
        let lnfb_size = c;
        specs.push(ParameterSpec { name: "lnfb", offset, size: lnfb_size });
        // offset += lnfb_size; // not needed for last one

        specs
    }

    /// Calculate total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.param_specs().iter().map(|s| s.size).sum()
    }
}

/// Loaded GPT-2 checkpoint
pub struct GPT2Checkpoint {
    pub config: GPT2Config,
    pub params: Vec<f32>,  // All parameters as contiguous float array
}

impl GPT2Checkpoint {
    /// Load checkpoint from llm.c binary format
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let mut file = File::open(path.as_ref())
            .map_err(|e| format!("Failed to open checkpoint: {}", e))?;

        // Read header (256 i32 values)
        let mut header = [0i32; 256];
        let header_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                header.as_mut_ptr() as *mut u8,
                256 * 4,
            )
        };
        file.read_exact(header_bytes)
            .map_err(|e| format!("Failed to read header: {}", e))?;

        // Verify magic number
        const CHECKPOINT_MAGIC: i32 = 20240326;
        if header[0] != CHECKPOINT_MAGIC {
            return Err(format!(
                "Invalid checkpoint magic: expected {}, got {}",
                CHECKPOINT_MAGIC, header[0]
            ));
        }

        // Verify version
        if header[1] != 3 {
            return Err(format!(
                "Unsupported checkpoint version: expected 3, got {}",
                header[1]
            ));
        }

        // Extract config from header
        let config = GPT2Config {
            max_seq_len: header[2] as usize,
            vocab_size: header[3] as usize,
            num_layers: header[4] as usize,
            num_heads: header[5] as usize,
            channels: header[6] as usize,
            padded_vocab_size: header[7] as usize,
        };

        eprintln!("Loading GPT-2 checkpoint:");
        eprintln!("  max_seq_len: {}", config.max_seq_len);
        eprintln!("  vocab_size: {}", config.vocab_size);
        eprintln!("  padded_vocab_size: {}", config.padded_vocab_size);
        eprintln!("  num_layers: {}", config.num_layers);
        eprintln!("  num_heads: {}", config.num_heads);
        eprintln!("  channels: {}", config.channels);

        // Calculate expected number of parameters
        let num_params = config.num_parameters();
        eprintln!("  num_parameters: {} ({:.1}M)", num_params, num_params as f64 / 1_000_000.0);

        // Allocate and read parameters
        let mut params = vec![0.0f32; num_params];
        let params_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                params.as_mut_ptr() as *mut u8,
                num_params * 4,
            )
        };
        file.read_exact(params_bytes)
            .map_err(|e| format!("Failed to read parameters: {}", e))?;

        Ok(Self { config, params })
    }

    /// Get a parameter tensor by name
    pub fn get_param(&self, name: &str) -> Option<&[f32]> {
        for spec in self.config.param_specs() {
            if spec.name == name {
                return Some(&self.params[spec.offset..spec.offset + spec.size]);
            }
        }
        None
    }

    /// Get pointer to start of all parameters (for FFI)
    pub fn params_ptr(&self) -> *const f32 {
        self.params.as_ptr()
    }
}

/// Debug state for validation
pub struct GPT2DebugState {
    pub batch_size: usize,
    pub seq_len: usize,
    pub x: Vec<i32>,           // Input tokens (B, T)
    pub y: Vec<i32>,           // Target tokens (B, T)
    pub expected_logits: Vec<f32>,  // Expected output (B, T, V)
    pub expected_loss: f32,
}

impl GPT2DebugState {
    /// Load debug state from llm.c binary format
    pub fn load<P: AsRef<Path>>(path: P, vocab_size: usize) -> Result<Self, String> {
        let mut file = File::open(path.as_ref())
            .map_err(|e| format!("Failed to open debug state: {}", e))?;

        // Read header (256 i32 values)
        let mut header = [0i32; 256];
        let header_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                header.as_mut_ptr() as *mut u8,
                256 * 4,
            )
        };
        file.read_exact(header_bytes)
            .map_err(|e| format!("Failed to read debug state header: {}", e))?;

        // Verify magic number
        const DEBUG_MAGIC: i32 = 20240327;
        if header[0] != DEBUG_MAGIC {
            return Err(format!(
                "Invalid debug state magic: expected {}, got {}",
                DEBUG_MAGIC, header[0]
            ));
        }

        // Verify version
        if header[1] != 2 {
            return Err(format!(
                "Unsupported debug state version: expected 2, got {}",
                header[1]
            ));
        }

        let batch_size = header[2] as usize;
        let seq_len = header[3] as usize;

        eprintln!("Loading GPT-2 debug state:");
        eprintln!("  batch_size: {}", batch_size);
        eprintln!("  seq_len: {}", seq_len);

        // Read input tokens x: (B, T) as i32
        let bt = batch_size * seq_len;
        let mut x = vec![0i32; bt];
        let x_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                x.as_mut_ptr() as *mut u8,
                bt * 4,
            )
        };
        file.read_exact(x_bytes)
            .map_err(|e| format!("Failed to read x tokens: {}", e))?;

        // Read target tokens y: (B, T) as i32
        let mut y = vec![0i32; bt];
        let y_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                y.as_mut_ptr() as *mut u8,
                bt * 4,
            )
        };
        file.read_exact(y_bytes)
            .map_err(|e| format!("Failed to read y tokens: {}", e))?;

        // Read expected logits: (B, T, V) as f32
        let btv = batch_size * seq_len * vocab_size;
        let mut expected_logits = vec![0.0f32; btv];
        let logits_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                expected_logits.as_mut_ptr() as *mut u8,
                btv * 4,
            )
        };
        file.read_exact(logits_bytes)
            .map_err(|e| format!("Failed to read expected logits: {}", e))?;

        // Read expected loss: single f32
        let mut expected_loss = 0.0f32;
        let loss_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                &mut expected_loss as *mut f32 as *mut u8,
                4,
            )
        };
        file.read_exact(loss_bytes)
            .map_err(|e| format!("Failed to read expected loss: {}", e))?;

        eprintln!("  expected_loss: {}", expected_loss);
        eprintln!("  First few input tokens: {:?}", &x[..8.min(x.len())]);
        eprintln!("  First few logits: {:?}", &expected_logits[..8.min(expected_logits.len())]);

        Ok(Self {
            batch_size,
            seq_len,
            x,
            y,
            expected_logits,
            expected_loss,
        })
    }
}

// ============================================================================
// FFI Interface for MLIR
// ============================================================================

use std::sync::atomic::{AtomicPtr, AtomicI64, AtomicU32};

/// Cached raw pointers for FFI (set when loading)
static PARAMS_PTR: AtomicPtr<f32> = AtomicPtr::new(std::ptr::null_mut());
static DEBUG_X_PTR: AtomicPtr<i32> = AtomicPtr::new(std::ptr::null_mut());
static DEBUG_LOGITS_PTR: AtomicPtr<f32> = AtomicPtr::new(std::ptr::null_mut());

/// Cached config values for FFI
static CONFIG_MAX_SEQ_LEN: AtomicI64 = AtomicI64::new(-1);
static CONFIG_VOCAB_SIZE: AtomicI64 = AtomicI64::new(-1);
static CONFIG_PADDED_VOCAB_SIZE: AtomicI64 = AtomicI64::new(-1);
static CONFIG_NUM_LAYERS: AtomicI64 = AtomicI64::new(-1);
static CONFIG_NUM_HEADS: AtomicI64 = AtomicI64::new(-1);
static CONFIG_CHANNELS: AtomicI64 = AtomicI64::new(-1);

static DEBUG_BATCH_SIZE: AtomicI64 = AtomicI64::new(-1);
static DEBUG_SEQ_LEN: AtomicI64 = AtomicI64::new(-1);
static DEBUG_EXPECTED_LOSS: AtomicU32 = AtomicU32::new(0);

use std::sync::atomic::Ordering;

/// Load checkpoint file (call once at startup)
#[unsafe(no_mangle)]
pub extern "C" fn gpt2_load_checkpoint(path: *const std::ffi::c_char) -> i32 {
    let path_str = unsafe {
        std::ffi::CStr::from_ptr(path).to_string_lossy().into_owned()
    };

    match GPT2Checkpoint::load(&path_str) {
        Ok(checkpoint) => {
            // Cache config values in atomics for lock-free access
            CONFIG_MAX_SEQ_LEN.store(checkpoint.config.max_seq_len as i64, Ordering::SeqCst);
            CONFIG_VOCAB_SIZE.store(checkpoint.config.vocab_size as i64, Ordering::SeqCst);
            CONFIG_PADDED_VOCAB_SIZE.store(checkpoint.config.padded_vocab_size as i64, Ordering::SeqCst);
            CONFIG_NUM_LAYERS.store(checkpoint.config.num_layers as i64, Ordering::SeqCst);
            CONFIG_NUM_HEADS.store(checkpoint.config.num_heads as i64, Ordering::SeqCst);
            CONFIG_CHANNELS.store(checkpoint.config.channels as i64, Ordering::SeqCst);

            // Leak the checkpoint to get a stable pointer (lives for program lifetime)
            let boxed = Box::new(checkpoint);
            let leaked: &'static GPT2Checkpoint = Box::leak(boxed);
            PARAMS_PTR.store(leaked.params.as_ptr() as *mut f32, Ordering::SeqCst);

            0 // success
        }
        Err(e) => {
            eprintln!("Failed to load checkpoint: {}", e);
            -1 // failure
        }
    }
}

/// Load debug state file (call once at startup)
#[unsafe(no_mangle)]
pub extern "C" fn gpt2_load_debug_state(path: *const std::ffi::c_char) -> i32 {
    let path_str = unsafe {
        std::ffi::CStr::from_ptr(path).to_string_lossy().into_owned()
    };

    // Get vocab size from cached atomic
    let vocab_size = CONFIG_PADDED_VOCAB_SIZE.load(Ordering::SeqCst);
    if vocab_size < 0 {
        eprintln!("Must load checkpoint before debug state");
        return -1;
    }

    match GPT2DebugState::load(&path_str, vocab_size as usize) {
        Ok(debug_state) => {
            // Cache values in atomics
            DEBUG_BATCH_SIZE.store(debug_state.batch_size as i64, Ordering::SeqCst);
            DEBUG_SEQ_LEN.store(debug_state.seq_len as i64, Ordering::SeqCst);
            DEBUG_EXPECTED_LOSS.store(debug_state.expected_loss.to_bits(), Ordering::SeqCst);

            // Leak to get stable pointers
            let boxed = Box::new(debug_state);
            let leaked: &'static GPT2DebugState = Box::leak(boxed);
            DEBUG_X_PTR.store(leaked.x.as_ptr() as *mut i32, Ordering::SeqCst);
            DEBUG_LOGITS_PTR.store(leaked.expected_logits.as_ptr() as *mut f32, Ordering::SeqCst);

            0 // success
        }
        Err(e) => {
            eprintln!("Failed to load debug state: {}", e);
            -1 // failure
        }
    }
}

/// Get pointer to all parameters as contiguous f32 array
#[unsafe(no_mangle)]
pub extern "C" fn gpt2_get_params_ptr() -> *const f32 {
    PARAMS_PTR.load(Ordering::SeqCst)
}

/// Get model config value by index:
/// 0: max_seq_len, 1: vocab_size, 2: padded_vocab_size
/// 3: num_layers, 4: num_heads, 5: channels
#[unsafe(no_mangle)]
pub extern "C" fn gpt2_get_config(index: i32) -> i64 {
    match index {
        0 => CONFIG_MAX_SEQ_LEN.load(Ordering::SeqCst),
        1 => CONFIG_VOCAB_SIZE.load(Ordering::SeqCst),
        2 => CONFIG_PADDED_VOCAB_SIZE.load(Ordering::SeqCst),
        3 => CONFIG_NUM_LAYERS.load(Ordering::SeqCst),
        4 => CONFIG_NUM_HEADS.load(Ordering::SeqCst),
        5 => CONFIG_CHANNELS.load(Ordering::SeqCst),
        _ => -1,
    }
}

/// Get debug state batch size
#[unsafe(no_mangle)]
pub extern "C" fn gpt2_debug_batch_size() -> i64 {
    DEBUG_BATCH_SIZE.load(Ordering::SeqCst)
}

/// Get debug state sequence length
#[unsafe(no_mangle)]
pub extern "C" fn gpt2_debug_seq_len() -> i64 {
    DEBUG_SEQ_LEN.load(Ordering::SeqCst)
}

/// Get pointer to debug input tokens (x)
#[unsafe(no_mangle)]
pub extern "C" fn gpt2_debug_x_ptr() -> *const i32 {
    DEBUG_X_PTR.load(Ordering::SeqCst)
}

/// Get pointer to expected logits
#[unsafe(no_mangle)]
pub extern "C" fn gpt2_debug_expected_logits_ptr() -> *const f32 {
    DEBUG_LOGITS_PTR.load(Ordering::SeqCst)
}

/// Get expected loss
#[unsafe(no_mangle)]
pub extern "C" fn gpt2_debug_expected_loss() -> f32 {
    f32::from_bits(DEBUG_EXPECTED_LOSS.load(Ordering::SeqCst))
}

/// Compare output logits with expected and return max absolute difference
#[unsafe(no_mangle)]
pub extern "C" fn gpt2_compare_logits(output_ptr: *const f32, size: i64) -> f32 {
    let size = size as usize;
    let expected_ptr = DEBUG_LOGITS_PTR.load(Ordering::SeqCst);

    if output_ptr.is_null() || expected_ptr.is_null() {
        return -1.0;
    }

    let output = unsafe { std::slice::from_raw_parts(output_ptr, size) };
    let expected = unsafe { std::slice::from_raw_parts(expected_ptr, size) };

    let mut max_diff = 0.0f32;
    for (i, (&out, &exp)) in output.iter().zip(expected.iter()).enumerate() {
        let diff = (out - exp).abs();
        if diff > max_diff {
            max_diff = diff;
            if i < 10 {
                eprintln!("  [{}] out={:.6}, exp={:.6}, diff={:.6}", i, out, exp, diff);
            }
        }
    }
    max_diff
}

// ============================================================================
// High-level forward pass helpers
// ============================================================================

use std::sync::atomic::AtomicPtr as AtomicPtrType;

static ENCODED_PTR: AtomicPtrType<f32> = AtomicPtrType::new(std::ptr::null_mut());

/// Run encoder: encoded[b,t,c] = wte[token[b,t], c] + wpe[t, c]
/// Returns pointer to encoded tensor (B, T, C) as contiguous f32 array
#[unsafe(no_mangle)]
pub extern "C" fn gpt2_encoder_forward() -> *const f32 {
    let params_ptr = PARAMS_PTR.load(Ordering::SeqCst);
    let tokens_ptr = DEBUG_X_PTR.load(Ordering::SeqCst);

    if params_ptr.is_null() || tokens_ptr.is_null() {
        eprintln!("gpt2_encoder_forward: params or tokens not loaded");
        return std::ptr::null();
    }

    let b = DEBUG_BATCH_SIZE.load(Ordering::SeqCst) as usize;
    let t = DEBUG_SEQ_LEN.load(Ordering::SeqCst) as usize;
    let c = CONFIG_CHANNELS.load(Ordering::SeqCst) as usize;
    let v = CONFIG_PADDED_VOCAB_SIZE.load(Ordering::SeqCst) as usize;

    // Allocate encoded tensor
    let mut encoded = vec![0.0f32; b * t * c];

    // Get wte and wpe pointers
    let params = unsafe { std::slice::from_raw_parts(params_ptr, v * c + 1024 * c) };
    let wte = &params[0..v * c];
    let wpe = &params[v * c..v * c + 1024 * c];
    let tokens = unsafe { std::slice::from_raw_parts(tokens_ptr, b * t) };

    // Compute encoded = wte[token] + wpe[position]
    for batch in 0..b {
        for pos in 0..t {
            let token = tokens[batch * t + pos] as usize;
            for ch in 0..c {
                let wte_val = wte[token * c + ch];
                let wpe_val = wpe[pos * c + ch];
                encoded[(batch * t + pos) * c + ch] = wte_val + wpe_val;
            }
        }
    }

    // Store in leaked box for stable pointer
    let boxed = Box::new(encoded);
    let leaked: &'static mut Vec<f32> = Box::leak(boxed);
    let ptr = leaked.as_ptr() as *mut f32;
    ENCODED_PTR.store(ptr, Ordering::SeqCst);

    ptr
}

/// Get the encoded tensor pointer (from last encoder_forward call)
#[unsafe(no_mangle)]
pub extern "C" fn gpt2_get_encoded_ptr() -> *const f32 {
    ENCODED_PTR.load(Ordering::SeqCst)
}

/// Load f32 from pointer + offset (for dynamic pointer arithmetic)
#[unsafe(no_mangle)]
pub extern "C" fn gpt2_load_f32(ptr: *const f32, offset: i64) -> f32 {
    if ptr.is_null() {
        return 0.0;
    }
    unsafe { *ptr.offset(offset as isize) }
}

/// Load i32 from pointer + offset
#[unsafe(no_mangle)]
pub extern "C" fn gpt2_load_i32(ptr: *const i32, offset: i64) -> i32 {
    if ptr.is_null() {
        return 0;
    }
    unsafe { *ptr.offset(offset as isize) }
}

/// Print 4 f32 values with a label (for debugging encoder output)
#[unsafe(no_mangle)]
pub extern "C" fn gpt2_print_f32_4(v0: f32, v1: f32, v2: f32, v3: f32) {
    println!("Encoder output: [{}, {}, {}, {}, ...]", v0, v1, v2, v3);
}

/// Print encoder output values (for debugging)
#[unsafe(no_mangle)]
pub extern "C" fn gpt2_compare_encoder(output_ptr: *const f32, size: i64) -> f32 {
    if output_ptr.is_null() {
        return -1.0;
    }

    // Print the first 8 output values
    let size = size as usize;
    let output = unsafe { std::slice::from_raw_parts(output_ptr, size.min(8)) };
    println!("MLIR encoder output (first 8): {:?}", output);

    0.0
}

/// Static pointer to store test data
static TEST_DATA_PTR: AtomicPtr<f32> = AtomicPtr::new(std::ptr::null_mut());

/// Static pointer to store full checkpoint params
static RUST_PARAMS_PTR: AtomicPtr<f32> = AtomicPtr::new(std::ptr::null_mut());

/// Static pointer to store debug tokens
static RUST_DEBUG_X_PTR: AtomicPtr<i32> = AtomicPtr::new(std::ptr::null_mut());

/// Static debug seq_len
static RUST_DEBUG_SEQ_LEN: AtomicI64 = AtomicI64::new(0);

/// Read file into malloc'd buffer, store in global, and return pointer
#[unsafe(no_mangle)]
pub extern "C" fn read_checkpoint_data() -> *mut f32 {
    let path = "/home/jimmyhmiller/llm.c/gpt2_124M.bin";
    eprintln!("[read_checkpoint_data] Opening {}", path);

    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[read_checkpoint_data] Failed to open: {}", e);
            return std::ptr::null_mut();
        }
    };

    // Skip header (1024 bytes)
    use std::io::{Read, Seek, SeekFrom};
    if let Err(e) = file.seek(SeekFrom::Start(1024)) {
        eprintln!("[read_checkpoint_data] Failed to seek: {}", e);
        return std::ptr::null_mut();
    }

    // Read first 16 floats
    let mut buffer = vec![0f32; 16];
    let buf_bytes = unsafe {
        std::slice::from_raw_parts_mut(buffer.as_mut_ptr() as *mut u8, 16 * 4)
    };
    if let Err(e) = file.read_exact(buf_bytes) {
        eprintln!("[read_checkpoint_data] Failed to read: {}", e);
        return std::ptr::null_mut();
    }

    eprintln!("[read_checkpoint_data] Read values: {:?}", &buffer[..4]);

    // Leak the buffer to get a stable pointer
    let boxed = buffer.into_boxed_slice();
    let ptr = Box::into_raw(boxed) as *mut f32;

    // Store in global
    TEST_DATA_PTR.store(ptr, Ordering::SeqCst);

    eprintln!("[read_checkpoint_data] Returning ptr: {:p}", ptr);
    ptr
}

/// Get the test data pointer from global
#[unsafe(no_mangle)]
pub extern "C" fn get_test_data_ptr() -> *mut f32 {
    let ptr = TEST_DATA_PTR.load(Ordering::SeqCst);
    eprintln!("[get_test_data_ptr] Returning ptr: {:p}", ptr);
    // Also verify we can read the data
    if !ptr.is_null() {
        let vals = unsafe { std::slice::from_raw_parts(ptr, 4) };
        eprintln!("[get_test_data_ptr] Values at ptr: {:?}", vals);
    }
    ptr
}

/// Load a f32 value from a pointer (Rust function to verify memory access works)
#[unsafe(no_mangle)]
pub extern "C" fn rust_load_f32(ptr: *const f32, idx: i64) -> f32 {
    if ptr.is_null() {
        eprintln!("[rust_load_f32] NULL ptr!");
        return 0.0;
    }
    let val = unsafe { *ptr.offset(idx as isize) };
    eprintln!("[rust_load_f32] ptr={:p}, idx={}, val={}", ptr, idx, val);
    val
}

/// Load a f32 value and return as f64 (to test ABI)
#[unsafe(no_mangle)]
pub extern "C" fn rust_load_f32_as_f64(ptr: *const f32, idx: i64) -> f64 {
    if ptr.is_null() {
        eprintln!("[rust_load_f32_as_f64] NULL ptr!");
        return 0.0;
    }
    let val = unsafe { *ptr.offset(idx as isize) };
    eprintln!("[rust_load_f32_as_f64] ptr={:p}, idx={}, val={}", ptr, idx, val);
    val as f64
}

/// Load full checkpoint using Rust's file I/O (bypasses libc fread issue)
/// Returns pointer to params, stores in global
#[unsafe(no_mangle)]
pub extern "C" fn rust_load_checkpoint(path: *const i8) -> *mut f32 {
    use std::io::{Read, Seek, SeekFrom};

    let path_str = unsafe {
        std::ffi::CStr::from_ptr(path).to_string_lossy().into_owned()
    };
    eprintln!("[rust_load_checkpoint] Opening {}", path_str);

    let mut file = match std::fs::File::open(&path_str) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[rust_load_checkpoint] Failed to open: {}", e);
            return std::ptr::null_mut();
        }
    };

    // Skip header (1024 bytes)
    if let Err(e) = file.seek(SeekFrom::Start(1024)) {
        eprintln!("[rust_load_checkpoint] Failed to seek: {}", e);
        return std::ptr::null_mut();
    }

    // Read all parameters: 124,439,808 f32 values
    let num_params: usize = 124_439_808;
    let mut buffer = vec![0f32; num_params];
    let buf_bytes = unsafe {
        std::slice::from_raw_parts_mut(buffer.as_mut_ptr() as *mut u8, num_params * 4)
    };
    if let Err(e) = file.read_exact(buf_bytes) {
        eprintln!("[rust_load_checkpoint] Failed to read: {}", e);
        return std::ptr::null_mut();
    }

    eprintln!("[rust_load_checkpoint] Read {} params, first 4: {:?}", num_params, &buffer[..4]);

    // Leak the buffer to get a stable pointer
    let boxed = buffer.into_boxed_slice();
    let ptr = Box::into_raw(boxed) as *mut f32;

    // Store in global
    RUST_PARAMS_PTR.store(ptr, Ordering::SeqCst);

    eprintln!("[rust_load_checkpoint] Stored ptr: {:p}", ptr);
    ptr
}

/// Load debug state using Rust's file I/O
#[unsafe(no_mangle)]
pub extern "C" fn rust_load_debug_state(path: *const i8) -> i32 {
    use std::io::{Read, Seek, SeekFrom};

    let path_str = unsafe {
        std::ffi::CStr::from_ptr(path).to_string_lossy().into_owned()
    };
    eprintln!("[rust_load_debug_state] Opening {}", path_str);

    let mut file = match std::fs::File::open(&path_str) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[rust_load_debug_state] Failed to open: {}", e);
            return -1;
        }
    };

    // Read header (256 i32 values)
    let mut header = [0i32; 256];
    let header_bytes = unsafe {
        std::slice::from_raw_parts_mut(header.as_mut_ptr() as *mut u8, 256 * 4)
    };
    if let Err(e) = file.read_exact(header_bytes) {
        eprintln!("[rust_load_debug_state] Failed to read header: {}", e);
        return -1;
    }

    let batch_size = header[2] as usize;
    let seq_len = header[3] as usize;
    eprintln!("[rust_load_debug_state] batch_size={}, seq_len={}", batch_size, seq_len);

    // Read input tokens: batch_size * seq_len i32 values
    let num_tokens = batch_size * seq_len;
    let mut tokens = vec![0i32; num_tokens];
    let tokens_bytes = unsafe {
        std::slice::from_raw_parts_mut(tokens.as_mut_ptr() as *mut u8, num_tokens * 4)
    };
    if let Err(e) = file.read_exact(tokens_bytes) {
        eprintln!("[rust_load_debug_state] Failed to read tokens: {}", e);
        return -1;
    }

    eprintln!("[rust_load_debug_state] First tokens: {:?}", &tokens[..8.min(tokens.len())]);

    // Leak and store
    let boxed_tokens = tokens.into_boxed_slice();
    let tokens_ptr = Box::into_raw(boxed_tokens) as *mut i32;
    RUST_DEBUG_X_PTR.store(tokens_ptr, Ordering::SeqCst);
    RUST_DEBUG_SEQ_LEN.store(seq_len as i64, Ordering::SeqCst);

    0
}

/// Get params pointer from Rust global
#[unsafe(no_mangle)]
pub extern "C" fn rust_get_params_ptr() -> *mut f32 {
    RUST_PARAMS_PTR.load(Ordering::SeqCst)
}

/// Get debug tokens pointer from Rust global
#[unsafe(no_mangle)]
pub extern "C" fn rust_get_debug_x_ptr() -> *mut i32 {
    RUST_DEBUG_X_PTR.load(Ordering::SeqCst)
}

/// Get debug seq_len from Rust global
#[unsafe(no_mangle)]
pub extern "C" fn rust_get_debug_seq_len() -> i64 {
    RUST_DEBUG_SEQ_LEN.load(Ordering::SeqCst)
}

/// Wrapper around libc fread for debugging
#[unsafe(no_mangle)]
pub extern "C" fn my_fread(ptr: *mut u8, size: usize, count: usize, file: *mut libc::FILE) -> usize {
    eprintln!("[my_fread] ptr={:p}, size={}, count={}, file={:p}", ptr, size, count, file);
    let result = unsafe { libc::fread(ptr as *mut libc::c_void, size, count, file) };
    eprintln!("[my_fread] returned {}", result);
    // Print first 4 bytes to verify data was read
    if result > 0 && !ptr.is_null() {
        let bytes = unsafe { std::slice::from_raw_parts(ptr, 16.min(result * size)) };
        eprintln!("[my_fread] first bytes: {:?}", bytes);
        // Interpret as f32
        let floats: &[f32] = unsafe { std::slice::from_raw_parts(ptr as *const f32, 4.min(result)) };
        eprintln!("[my_fread] as f32: {:?}", floats);
    }
    result
}

/// Wrapper around libc fopen for debugging
#[unsafe(no_mangle)]
pub extern "C" fn my_fopen(path: *const i8, mode: *const i8) -> *mut libc::FILE {
    let result = unsafe { libc::fopen(path, mode) };
    let path_str = unsafe { std::ffi::CStr::from_ptr(path).to_string_lossy() };
    let mode_str = unsafe { std::ffi::CStr::from_ptr(mode).to_string_lossy() };
    eprintln!("[my_fopen] path='{}', mode='{}' -> {:p}", path_str, mode_str, result);
    result
}

/// Get FFI function list for JIT registration
pub fn get_gpt2_ffi_functions() -> Vec<(&'static str, *mut ())> {
    vec![
        ("gpt2_load_checkpoint", gpt2_load_checkpoint as *mut ()),
        ("gpt2_load_debug_state", gpt2_load_debug_state as *mut ()),
        ("gpt2_get_params_ptr", gpt2_get_params_ptr as *mut ()),
        ("gpt2_get_config", gpt2_get_config as *mut ()),
        ("gpt2_debug_batch_size", gpt2_debug_batch_size as *mut ()),
        ("gpt2_debug_seq_len", gpt2_debug_seq_len as *mut ()),
        ("gpt2_debug_x_ptr", gpt2_debug_x_ptr as *mut ()),
        ("gpt2_debug_expected_logits_ptr", gpt2_debug_expected_logits_ptr as *mut ()),
        ("gpt2_debug_expected_loss", gpt2_debug_expected_loss as *mut ()),
        ("gpt2_compare_logits", gpt2_compare_logits as *mut ()),
        ("gpt2_encoder_forward", gpt2_encoder_forward as *mut ()),
        ("gpt2_get_encoded_ptr", gpt2_get_encoded_ptr as *mut ()),
        ("gpt2_load_f32", gpt2_load_f32 as *mut ()),
        ("gpt2_load_i32", gpt2_load_i32 as *mut ()),
        ("gpt2_print_f32_4", gpt2_print_f32_4 as *mut ()),
        ("gpt2_compare_encoder", gpt2_compare_encoder as *mut ()),
        ("my_fread", my_fread as *mut ()),
        ("my_fopen", my_fopen as *mut ()),
        ("read_checkpoint_data", read_checkpoint_data as *mut ()),
        ("get_test_data_ptr", get_test_data_ptr as *mut ()),
        ("rust_load_f32", rust_load_f32 as *mut ()),
        ("rust_load_f32_as_f64", rust_load_f32_as_f64 as *mut ()),
        ("rust_load_checkpoint", rust_load_checkpoint as *mut ()),
        ("rust_load_debug_state", rust_load_debug_state as *mut ()),
        ("rust_get_params_ptr", rust_get_params_ptr as *mut ()),
        ("rust_get_debug_x_ptr", rust_get_debug_x_ptr as *mut ()),
        ("rust_get_debug_seq_len", rust_get_debug_seq_len as *mut ()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_param_sizes() {
        let config = GPT2Config::default();
        let specs = config.param_specs();

        // Verify we have 16 parameter tensors
        assert_eq!(specs.len(), 16);

        // Verify total matches expected ~124M parameters
        let total: usize = specs.iter().map(|s| s.size).sum();
        assert!(total > 120_000_000 && total < 130_000_000,
            "Expected ~124M params, got {}", total);
    }

    #[test]
    #[ignore]  // Run with: cargo test --ignored -- --nocapture
    fn test_load_checkpoint() {
        // This test requires the llm.c checkpoint file to exist
        let checkpoint_path = std::env::var("HOME")
            .map(|h| format!("{}/llm.c/gpt2_124M.bin", h))
            .unwrap_or("/home/jimmyhmiller/llm.c/gpt2_124M.bin".to_string());

        let checkpoint = GPT2Checkpoint::load(&checkpoint_path).expect("Failed to load checkpoint");

        println!("Loaded GPT-2 checkpoint:");
        println!("  Config: {:?}", checkpoint.config);
        println!("  Total params: {}", checkpoint.params.len());

        // Verify some basic properties
        assert_eq!(checkpoint.config.num_layers, 12);
        assert_eq!(checkpoint.config.channels, 768);
        assert_eq!(checkpoint.config.num_heads, 12);

        // Verify wte is accessible and has expected shape
        let wte = checkpoint.get_param("wte").expect("wte not found");
        assert_eq!(wte.len(), checkpoint.config.padded_vocab_size * checkpoint.config.channels);
        println!("  wte first 5 values: {:?}", &wte[..5]);
    }

    #[test]
    #[ignore]  // Run with: cargo test --ignored -- --nocapture
    fn test_load_debug_state() {
        // Load checkpoint first
        let checkpoint_path = std::env::var("HOME")
            .map(|h| format!("{}/llm.c/gpt2_124M.bin", h))
            .unwrap_or("/home/jimmyhmiller/llm.c/gpt2_124M.bin".to_string());
        let checkpoint = GPT2Checkpoint::load(&checkpoint_path).expect("Failed to load checkpoint");

        // Load debug state
        let debug_path = std::env::var("HOME")
            .map(|h| format!("{}/llm.c/gpt2_124M_debug_state.bin", h))
            .unwrap_or("/home/jimmyhmiller/llm.c/gpt2_124M_debug_state.bin".to_string());
        let debug_state = GPT2DebugState::load(&debug_path, checkpoint.config.padded_vocab_size)
            .expect("Failed to load debug state");

        println!("Loaded GPT-2 debug state:");
        println!("  Batch size: {}", debug_state.batch_size);
        println!("  Seq len: {}", debug_state.seq_len);
        println!("  Expected loss: {}", debug_state.expected_loss);
        println!("  First 8 input tokens: {:?}", &debug_state.x[..8]);
        println!("  First 8 logits: {:?}", &debug_state.expected_logits[..8]);

        // Verify basic properties
        assert_eq!(debug_state.batch_size, 4);
        assert_eq!(debug_state.seq_len, 64);
        assert!(debug_state.expected_loss > 0.0);
    }
}
