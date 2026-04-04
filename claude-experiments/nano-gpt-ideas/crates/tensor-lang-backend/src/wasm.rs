//! Direct WASM binary backend: emits a `.wasm` module from the fused loop IR.
//!
//! Memory layout:
//! - Linear memory starts with a "heap pointer" at byte 0..4 (i32).
//! - The host writes input buffers into memory starting at offset 16 (aligned).
//! - The `execute` function returns the byte offset of the output buffer.
//! - Intermediate buffers are bump-allocated from the heap pointer.
//!
//! Function signature (exported as "execute"):
//!   (dim_param_0: i32, ..., input_ptr_0: i32, ...) -> i32
//!
//! Each input_ptr is a byte offset into linear memory where the host placed the f32 data.
//! The return value is the byte offset of the output f32 array.

use std::collections::HashMap;

use tensor_lang_graph::{Dim, Graph, Op};
use crate::loop_ir::{self, Stmt, Inst, Index, ReduceOp};

// ============================================================================
// WASM binary encoding primitives
// ============================================================================

/// WASM value types
const I32: u8 = 0x7F;
const F32: u8 = 0x7D;

/// WASM section IDs
const SEC_TYPE: u8 = 1;
const SEC_FUNCTION: u8 = 3;
const SEC_MEMORY: u8 = 5;
const SEC_EXPORT: u8 = 7;
const SEC_CODE: u8 = 10;

/// WASM opcodes
#[allow(dead_code)]
mod op {
    // Control
    pub const UNREACHABLE: u8 = 0x00;
    pub const BLOCK: u8 = 0x02;
    pub const LOOP: u8 = 0x03;
    pub const END: u8 = 0x0B;
    pub const BR: u8 = 0x0C;
    pub const BR_IF: u8 = 0x0D;
    pub const RETURN: u8 = 0x0F;
    pub const CALL: u8 = 0x10;

    // Variables
    pub const LOCAL_GET: u8 = 0x20;
    pub const LOCAL_SET: u8 = 0x21;
    pub const LOCAL_TEE: u8 = 0x22;

    // Memory
    pub const I32_LOAD: u8 = 0x28;
    pub const F32_LOAD: u8 = 0x2A;
    pub const I32_STORE: u8 = 0x36;
    pub const F32_STORE: u8 = 0x38;

    // Constants
    pub const I32_CONST: u8 = 0x41;
    pub const F32_CONST: u8 = 0x43;

    // i32 ops
    pub const I32_ADD: u8 = 0x6A;
    pub const I32_SUB: u8 = 0x6B;
    pub const I32_MUL: u8 = 0x6C;
    pub const I32_DIV_S: u8 = 0x6D;
    pub const I32_REM_S: u8 = 0x6F;
    pub const I32_AND: u8 = 0x71;
    pub const I32_OR: u8 = 0x72;
    pub const I32_SHL: u8 = 0x74;
    pub const I32_SHR_U: u8 = 0x76;
    pub const I32_GE_S: u8 = 0x4E;
    pub const I32_LT_S: u8 = 0x48;
    pub const I32_GT_S: u8 = 0x4A;
    pub const I32_LE_S: u8 = 0x4C;
    pub const I32_EQ: u8 = 0x46;

    // f32 ops
    pub const F32_ADD: u8 = 0x92;
    pub const F32_SUB: u8 = 0x93;
    pub const F32_MUL: u8 = 0x94;
    pub const F32_DIV: u8 = 0x95;
    pub const F32_SQRT: u8 = 0x91;
    pub const F32_NEG: u8 = 0x8C;
    pub const F32_ABS: u8 = 0x8B;
    pub const F32_MIN: u8 = 0x96;
    pub const F32_MAX: u8 = 0x97;
    pub const F32_FLOOR: u8 = 0x8E;
    pub const F32_LT: u8 = 0x5D;
    pub const F32_GT: u8 = 0x5E;

    // Conversions
    pub const F32_CONVERT_I32_S: u8 = 0xB2;
    pub const I32_TRUNC_F32_S: u8 = 0xA8;
    pub const F32_REINTERPRET_I32: u8 = 0xBE;
    pub const I32_REINTERPRET_F32: u8 = 0xBC;

    // Select
    pub const SELECT: u8 = 0x1B;
}

/// Encode an unsigned LEB128 integer.
fn leb128_u(mut val: u32, out: &mut Vec<u8>) {
    loop {
        let mut byte = (val & 0x7F) as u8;
        val >>= 7;
        if val != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if val == 0 {
            break;
        }
    }
}

/// Encode a signed LEB128 integer.
fn leb128_s(mut val: i32, out: &mut Vec<u8>) {
    loop {
        let byte = (val & 0x7F) as u8;
        val >>= 7;
        if (val == 0 && (byte & 0x40) == 0) || (val == -1 && (byte & 0x40) != 0) {
            out.push(byte);
            break;
        }
        out.push(byte | 0x80);
    }
}

/// Encode a section: section_id + size + contents.
fn encode_section(section_id: u8, contents: &[u8], out: &mut Vec<u8>) {
    out.push(section_id);
    leb128_u(contents.len() as u32, out);
    out.extend_from_slice(contents);
}

/// Encode a vector (count + items).
#[allow(dead_code)]
fn encode_vec(items: &[Vec<u8>], out: &mut Vec<u8>) {
    leb128_u(items.len() as u32, out);
    for item in items {
        out.extend_from_slice(item);
    }
}

// ============================================================================
// WASM function body builder
// ============================================================================

struct FuncBuilder {
    /// Locals declared in this function (type, count pairs).
    local_types: Vec<u8>,
    /// Number of parameters (not included in locals).
    n_params: u32,
    /// Total locals allocated so far.
    n_locals: u32,
    /// The instruction bytes.
    code: Vec<u8>,
}

impl FuncBuilder {
    fn new(n_params: u32) -> Self {
        FuncBuilder {
            local_types: Vec::new(),
            n_params,
            n_locals: 0,
            code: Vec::new(),
        }
    }

    /// Allocate a new local variable, returns its index.
    fn alloc_local(&mut self, ty: u8) -> u32 {
        let idx = self.n_params + self.n_locals;
        self.local_types.push(ty);
        self.n_locals += 1;
        idx
    }

    /// Emit a single opcode.
    fn emit(&mut self, opcode: u8) {
        self.code.push(opcode);
    }

    /// Emit opcode + unsigned LEB128 immediate.
    fn emit_u(&mut self, opcode: u8, val: u32) {
        self.code.push(opcode);
        leb128_u(val, &mut self.code);
    }

    /// Emit opcode + signed LEB128 immediate.
    fn emit_s(&mut self, opcode: u8, val: i32) {
        self.code.push(opcode);
        leb128_s(val, &mut self.code);
    }

    /// Emit i32.const.
    fn i32_const(&mut self, val: i32) {
        self.emit_s(op::I32_CONST, val);
    }

    /// Emit f32.const.
    fn f32_const(&mut self, val: f32) {
        self.code.push(op::F32_CONST);
        self.code.extend_from_slice(&val.to_le_bytes());
    }

    /// Emit local.get.
    fn local_get(&mut self, idx: u32) {
        self.emit_u(op::LOCAL_GET, idx);
    }

    /// Emit local.set.
    fn local_set(&mut self, idx: u32) {
        self.emit_u(op::LOCAL_SET, idx);
    }

    /// Emit local.tee.
    fn local_tee(&mut self, idx: u32) {
        self.emit_u(op::LOCAL_TEE, idx);
    }

    /// Emit f32.load with alignment 2 (4-byte aligned) and given offset.
    fn f32_load(&mut self, memarg_offset: u32) {
        self.code.push(op::F32_LOAD);
        leb128_u(2, &mut self.code); // align
        leb128_u(memarg_offset, &mut self.code); // offset
    }

    /// Emit f32.store with alignment 2 (4-byte aligned) and given offset.
    fn f32_store(&mut self, memarg_offset: u32) {
        self.code.push(op::F32_STORE);
        leb128_u(2, &mut self.code); // align
        leb128_u(memarg_offset, &mut self.code); // offset
    }

    /// Emit i32.load with alignment 2 and given offset.
    fn i32_load(&mut self, memarg_offset: u32) {
        self.code.push(op::I32_LOAD);
        leb128_u(2, &mut self.code);
        leb128_u(memarg_offset, &mut self.code);
    }

    /// Emit i32.store with alignment 2 and given offset.
    #[allow(dead_code)]
    fn i32_store(&mut self, memarg_offset: u32) {
        self.code.push(op::I32_STORE);
        leb128_u(2, &mut self.code);
        leb128_u(memarg_offset, &mut self.code);
    }

    /// Emit call to function index.
    fn call(&mut self, func_idx: u32) {
        self.emit_u(op::CALL, func_idx);
    }

    /// Emit block with void result type.
    fn block(&mut self) {
        self.code.push(op::BLOCK);
        self.code.push(0x40); // void
    }

    /// Emit loop with void result type.
    fn loop_(&mut self) {
        self.code.push(op::LOOP);
        self.code.push(0x40); // void
    }

    /// Emit end.
    fn end(&mut self) {
        self.code.push(op::END);
    }

    /// Emit br (break to label).
    fn br(&mut self, label: u32) {
        self.emit_u(op::BR, label);
    }

    /// Emit br_if (conditional break to label).
    fn br_if(&mut self, label: u32) {
        self.emit_u(op::BR_IF, label);
    }

    /// Encode the function body (locals + code + end) into bytes.
    fn encode(&self) -> Vec<u8> {
        let mut body = Vec::new();

        // Encode locals as (count=1, type) pairs
        leb128_u(self.n_locals as u32, &mut body);
        for &ty in &self.local_types {
            leb128_u(1, &mut body); // count
            body.push(ty);
        }

        body.extend_from_slice(&self.code);
        body.push(op::END); // function end

        // Prefix with size
        let mut result = Vec::new();
        leb128_u(body.len() as u32, &mut result);
        result.extend_from_slice(&body);
        result
    }
}

// ============================================================================
// WASM module builder
// ============================================================================

struct WasmModule {
    /// Type section entries: each is a functype (params, results).
    types: Vec<(Vec<u8>, Vec<u8>)>,
    /// Function section: type index for each function.
    functions: Vec<u32>,
    /// Export section: (name, kind=0x00 for func, index).
    exports: Vec<(String, u8, u32)>,
    /// Code section: encoded function bodies.
    code_bodies: Vec<Vec<u8>>,
    /// Initial memory pages.
    initial_pages: u32,
}

impl WasmModule {
    fn new(initial_pages: u32) -> Self {
        WasmModule {
            types: Vec::new(),
            functions: Vec::new(),
            exports: Vec::new(),
            code_bodies: Vec::new(),
            initial_pages,
        }
    }

    /// Add a function type, returns the type index.
    fn add_type(&mut self, params: Vec<u8>, results: Vec<u8>) -> u32 {
        // Check for existing identical type
        for (i, (p, r)) in self.types.iter().enumerate() {
            if p == &params && r == &results {
                return i as u32;
            }
        }
        let idx = self.types.len() as u32;
        self.types.push((params, results));
        idx
    }

    /// Add a function, returns the function index.
    fn add_function(&mut self, type_idx: u32, body: Vec<u8>) -> u32 {
        let idx = self.functions.len() as u32;
        self.functions.push(type_idx);
        self.code_bodies.push(body);
        idx
    }

    /// Add an export.
    fn add_export(&mut self, name: &str, kind: u8, idx: u32) {
        self.exports.push((name.to_string(), kind, idx));
    }

    /// Encode the full WASM module.
    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();

        // Magic + version
        out.extend_from_slice(b"\0asm");
        out.extend_from_slice(&1u32.to_le_bytes());

        // Type section
        {
            let mut sec = Vec::new();
            leb128_u(self.types.len() as u32, &mut sec);
            for (params, results) in &self.types {
                sec.push(0x60); // functype
                leb128_u(params.len() as u32, &mut sec);
                sec.extend_from_slice(params);
                leb128_u(results.len() as u32, &mut sec);
                sec.extend_from_slice(results);
            }
            encode_section(SEC_TYPE, &sec, &mut out);
        }

        // Function section
        {
            let mut sec = Vec::new();
            leb128_u(self.functions.len() as u32, &mut sec);
            for &type_idx in &self.functions {
                leb128_u(type_idx, &mut sec);
            }
            encode_section(SEC_FUNCTION, &sec, &mut out);
        }

        // Memory section: 1 memory, min pages, no max
        {
            let mut sec = Vec::new();
            leb128_u(1, &mut sec); // 1 memory
            sec.push(0x00); // no maximum
            leb128_u(self.initial_pages, &mut sec);
            encode_section(SEC_MEMORY, &sec, &mut out);
        }

        // Export section
        {
            let mut sec = Vec::new();
            // Export memory as "memory"
            let n_exports = self.exports.len() + 1;
            leb128_u(n_exports as u32, &mut sec);
            // Memory export
            let name = b"memory";
            leb128_u(name.len() as u32, &mut sec);
            sec.extend_from_slice(name);
            sec.push(0x02); // memory
            leb128_u(0, &mut sec); // memory index 0

            for (name, kind, idx) in &self.exports {
                leb128_u(name.len() as u32, &mut sec);
                sec.extend_from_slice(name.as_bytes());
                sec.push(*kind);
                leb128_u(*idx, &mut sec);
            }
            encode_section(SEC_EXPORT, &sec, &mut out);
        }

        // Code section
        {
            let mut sec = Vec::new();
            leb128_u(self.code_bodies.len() as u32, &mut sec);
            for body in &self.code_bodies {
                sec.extend_from_slice(body);
            }
            encode_section(SEC_CODE, &sec, &mut out);
        }

        out
    }
}

// ============================================================================
// Dim evaluation: emit i32 computation for a Dim expression
// ============================================================================

/// Map from dim param name to the WASM local index holding it.
type DimLocals = HashMap<String, u32>;

fn emit_dim(fb: &mut FuncBuilder, dim: &Dim, dim_locals: &DimLocals) {
    match dim {
        Dim::Lit(n) => fb.i32_const(*n as i32),
        Dim::Param(name) => {
            let local = dim_locals[name];
            fb.local_get(local);
        }
        Dim::Add(a, b) => {
            emit_dim(fb, a, dim_locals);
            emit_dim(fb, b, dim_locals);
            fb.emit(op::I32_ADD);
        }
        Dim::Sub(a, b) => {
            emit_dim(fb, a, dim_locals);
            emit_dim(fb, b, dim_locals);
            fb.emit(op::I32_SUB);
        }
        Dim::Mul(a, b) => {
            emit_dim(fb, a, dim_locals);
            emit_dim(fb, b, dim_locals);
            fb.emit(op::I32_MUL);
        }
        Dim::Div(a, b) => {
            emit_dim(fb, a, dim_locals);
            emit_dim(fb, b, dim_locals);
            fb.emit(op::I32_DIV_S);
        }
    }
}

// ============================================================================
// Helper math functions: exp2 and log2 implemented as WASM functions
// ============================================================================

/// Build a WASM function for exp2(x: f32) -> f32.
///
/// Uses the identity: 2^x = 2^floor(x) * 2^frac(x)
/// where 2^floor(x) is computed via IEEE 754 bit manipulation
/// and 2^frac(x) is approximated with a polynomial.
fn build_exp2_func(fb: &mut FuncBuilder) {
    // param 0: x (f32)
    let x = 0u32;
    let x_floor = fb.alloc_local(F32);
    let n = fb.alloc_local(I32);
    let frac = fb.alloc_local(F32);
    let _result = fb.alloc_local(F32);
    let int_pow = fb.alloc_local(F32);

    // Clamp x to [-126, 127] to avoid overflow
    // x = max(-126.0, min(127.0, x))
    fb.f32_const(-126.0);
    fb.f32_const(127.0);
    fb.local_get(x);
    fb.emit(op::F32_MIN);
    fb.emit(op::F32_MAX);
    fb.local_set(x);  // Note: x is param 0, but we can set params too

    // x_floor = floor(x)
    fb.local_get(x);
    fb.emit(op::F32_FLOOR);
    fb.local_set(x_floor);

    // n = i32(x_floor)
    fb.local_get(x_floor);
    fb.emit(op::I32_TRUNC_F32_S);
    fb.local_set(n);

    // frac = x - x_floor  (in [0, 1))
    fb.local_get(x);
    fb.local_get(x_floor);
    fb.emit(op::F32_SUB);
    fb.local_set(frac);

    // Compute 2^n via IEEE 754: reinterpret((n + 127) << 23) as f32
    fb.local_get(n);
    fb.i32_const(127);
    fb.emit(op::I32_ADD);
    fb.i32_const(23);
    fb.emit(op::I32_SHL);
    fb.emit(op::F32_REINTERPRET_I32);
    fb.local_set(int_pow);

    // Polynomial approximation for 2^frac on [0,1):
    // Using minimax: p(f) = 1 + f*(0.6931472 + f*(0.2402265 + f*(0.0554993 + f*0.009618)))
    // Horner form:
    fb.f32_const(0.009618f32);
    fb.local_get(frac);
    fb.emit(op::F32_MUL);
    fb.f32_const(0.0554993f32);
    fb.emit(op::F32_ADD);
    fb.local_get(frac);
    fb.emit(op::F32_MUL);
    fb.f32_const(0.2402265f32);
    fb.emit(op::F32_ADD);
    fb.local_get(frac);
    fb.emit(op::F32_MUL);
    fb.f32_const(0.6931472f32);
    fb.emit(op::F32_ADD);
    fb.local_get(frac);
    fb.emit(op::F32_MUL);
    fb.f32_const(1.0f32);
    fb.emit(op::F32_ADD);

    // result = int_pow * poly
    fb.local_get(int_pow);
    fb.emit(op::F32_MUL);

    // Return is implicit (value left on stack)
}

/// Build a WASM function for log2(x: f32) -> f32.
///
/// Uses IEEE 754 decomposition: x = 2^e * m where m in [1, 2)
/// log2(x) = e + log2(m)
/// log2(m) approximated with a polynomial on [1, 2).
fn build_log2_func(fb: &mut FuncBuilder) {
    // param 0: x (f32)
    let x = 0u32;
    let bits = fb.alloc_local(I32);
    let e = fb.alloc_local(I32);
    let m = fb.alloc_local(F32);
    let p = fb.alloc_local(F32); // m - 1

    // bits = reinterpret x as i32
    fb.local_get(x);
    fb.emit(op::I32_REINTERPRET_F32);
    fb.local_set(bits);

    // e = ((bits >> 23) & 0xFF) - 127
    fb.local_get(bits);
    fb.i32_const(23);
    fb.emit(op::I32_SHR_U);
    fb.i32_const(0xFF);
    fb.emit(op::I32_AND);
    fb.i32_const(127);
    fb.emit(op::I32_SUB);
    fb.local_set(e);

    // m = reinterpret((bits & 0x7FFFFF) | 0x3F800000) as f32
    // This sets exponent to 0 (bias 127), keeping mantissa
    fb.local_get(bits);
    fb.i32_const(0x007FFFFF);
    fb.emit(op::I32_AND);
    fb.i32_const(0x3F800000u32 as i32);
    fb.emit(op::I32_OR);
    fb.emit(op::F32_REINTERPRET_I32);
    fb.local_set(m);

    // p = m - 1.0  (p in [0, 1))
    fb.local_get(m);
    fb.f32_const(1.0);
    fb.emit(op::F32_SUB);
    fb.local_set(p);

    // Polynomial approximation of log2(1+p) for p in [0, 1):
    // log2(1+p) ~ p * (1.4426950 + p * (-0.7213475 + p * (0.4808983 + p * (-0.3606738 + p * 0.2885390))))
    // Horner form:
    fb.f32_const(0.2885390f32);
    fb.local_get(p);
    fb.emit(op::F32_MUL);
    fb.f32_const(-0.3606738f32);
    fb.emit(op::F32_ADD);
    fb.local_get(p);
    fb.emit(op::F32_MUL);
    fb.f32_const(0.4808983f32);
    fb.emit(op::F32_ADD);
    fb.local_get(p);
    fb.emit(op::F32_MUL);
    fb.f32_const(-0.7213475f32);
    fb.emit(op::F32_ADD);
    fb.local_get(p);
    fb.emit(op::F32_MUL);
    fb.f32_const(1.4426950f32);
    fb.emit(op::F32_ADD);
    fb.local_get(p);
    fb.emit(op::F32_MUL);

    // result = e + poly
    fb.local_get(e);
    fb.emit(op::F32_CONVERT_I32_S);
    fb.emit(op::F32_ADD);
}

// ============================================================================
// Code generation context
// ============================================================================

/// Tracks locals for dimension variables (d0, d1, ...) and temporary values (t0, t1, ...).
struct CodegenCtx {
    /// Local indices for dim variables: d0, d1, ... (i32)
    dim_vars: HashMap<usize, u32>,
    /// Local indices for instruction results: t0, t1, ... (f32)
    inst_vars: HashMap<usize, u32>,
    /// Local indices for buffer pointers (i32)
    buf_ptrs: HashMap<usize, u32>,
    /// Dim param locals
    dim_locals: DimLocals,
    /// Function indices for exp2, log2
    exp2_fn: u32,
    log2_fn: u32,
}

/// Sentinel buffer ID for kernel output (matches assemblyscript.rs convention).
#[allow(dead_code)]
const KERNEL_OUT_BUF: usize = usize::MAX;

impl CodegenCtx {
    fn get_or_alloc_dim(&mut self, fb: &mut FuncBuilder, dim: usize) -> u32 {
        if let Some(&local) = self.dim_vars.get(&dim) {
            local
        } else {
            let local = fb.alloc_local(I32);
            self.dim_vars.insert(dim, local);
            local
        }
    }

    fn get_or_alloc_inst(&mut self, fb: &mut FuncBuilder, inst: usize) -> u32 {
        if let Some(&local) = self.inst_vars.get(&inst) {
            local
        } else {
            let local = fb.alloc_local(F32);
            self.inst_vars.insert(inst, local);
            local
        }
    }
}

// ============================================================================
// Emit helpers
// ============================================================================

/// Emit instructions to compute an index expression, leaving an i32 on the stack.
/// The result is a *byte offset* (multiplied by 4 for f32).
fn emit_index_offset(fb: &mut FuncBuilder, base_ptr_local: u32, index: &Index, ctx: &CodegenCtx) {
    // First compute the element index
    match index {
        Index::Flat => {
            // base_ptr + oi * 4
            fb.local_get(base_ptr_local);
            let oi = ctx.dim_vars.get(&usize::MAX).copied()
                .expect("flat index requires oi local");
            fb.local_get(oi);
            fb.i32_const(4);
            fb.emit(op::I32_MUL);
            fb.emit(op::I32_ADD);
        }
        Index::Strided { parts, offset } => {
            // base_ptr + (sum of d{dim} * stride + offset) * 4
            fb.local_get(base_ptr_local);

            // Compute element index
            let mut has_terms = false;
            for (dim, stride) in parts {
                if stride.is_zero() {
                    continue;
                }
                let dim_local = ctx.dim_vars[dim];
                fb.local_get(dim_local);
                if !stride.is_one() {
                    emit_dim(fb, stride, &ctx.dim_locals);
                    fb.emit(op::I32_MUL);
                }
                if has_terms {
                    fb.emit(op::I32_ADD);
                }
                has_terms = true;
            }

            if !offset.is_zero() {
                emit_dim(fb, offset, &ctx.dim_locals);
                if has_terms {
                    fb.emit(op::I32_ADD);
                } else {
                    has_terms = true;
                }
            }

            if !has_terms {
                fb.i32_const(0);
            }

            // Multiply by 4 (f32 size) and add to base
            fb.i32_const(4);
            fb.emit(op::I32_MUL);
            fb.emit(op::I32_ADD);
        }
    }
}

/// Emit a single loop IR instruction. The result f32 is stored in the inst's local.
fn emit_inst_code(
    fb: &mut FuncBuilder,
    j: usize,
    inst: &Inst,
    ctx: &mut CodegenCtx,
) {
    let result_local = ctx.get_or_alloc_inst(fb, j);

    match inst {
        Inst::Load { buf, index } => {
            let buf_ptr = ctx.buf_ptrs[buf];
            emit_index_offset(fb, buf_ptr, index, ctx);
            fb.f32_load(0);
            fb.local_set(result_local);
        }
        Inst::Const(v) => {
            fb.f32_const(*v as f32);
            fb.local_set(result_local);
        }
        Inst::DimVar(d) => {
            fb.local_get(ctx.dim_vars[d]);
            fb.emit(op::F32_CONVERT_I32_S);
            fb.local_set(result_local);
        }
        Inst::Neg(a) => {
            fb.local_get(ctx.inst_vars[a]);
            fb.emit(op::F32_NEG);
            fb.local_set(result_local);
        }
        Inst::Recip(a) => {
            fb.f32_const(1.0);
            fb.local_get(ctx.inst_vars[a]);
            fb.emit(op::F32_DIV);
            fb.local_set(result_local);
        }
        Inst::Exp2(a) => {
            fb.local_get(ctx.inst_vars[a]);
            fb.call(ctx.exp2_fn);
            fb.local_set(result_local);
        }
        Inst::Log2(a) => {
            fb.local_get(ctx.inst_vars[a]);
            fb.call(ctx.log2_fn);
            fb.local_set(result_local);
        }
        Inst::Sqrt(a) => {
            fb.local_get(ctx.inst_vars[a]);
            fb.emit(op::F32_SQRT);
            fb.local_set(result_local);
        }
        Inst::Add(a, b) => {
            fb.local_get(ctx.inst_vars[a]);
            fb.local_get(ctx.inst_vars[b]);
            fb.emit(op::F32_ADD);
            fb.local_set(result_local);
        }
        Inst::Mul(a, b) => {
            fb.local_get(ctx.inst_vars[a]);
            fb.local_get(ctx.inst_vars[b]);
            fb.emit(op::F32_MUL);
            fb.local_set(result_local);
        }
        Inst::Max(a, b) => {
            // f32.max has NaN propagation issues; use select-based max
            fb.local_get(ctx.inst_vars[a]);
            fb.local_get(ctx.inst_vars[b]);
            fb.local_get(ctx.inst_vars[a]);
            fb.local_get(ctx.inst_vars[b]);
            fb.emit(op::F32_GT);
            fb.emit(op::SELECT);
            fb.local_set(result_local);
        }
        Inst::CmpLt(a, b) => {
            // result = a < b ? 1.0 : 0.0
            fb.f32_const(1.0);
            fb.f32_const(0.0);
            fb.local_get(ctx.inst_vars[a]);
            fb.local_get(ctx.inst_vars[b]);
            fb.emit(op::F32_LT);
            fb.emit(op::SELECT);
            fb.local_set(result_local);
        }
    }
}

/// Emit a standard (non-tiled) loop.
fn emit_loop_code(
    fb: &mut FuncBuilder,
    out_buf_ptr: u32,
    shape: &[Dim],
    reduce: Option<&loop_ir::ReduceDesc>,
    body: &[Inst],
    result: usize,
    ctx: &mut CodegenCtx,
) {
    let ndim = shape.len();
    let out_strides = Dim::strides(shape);

    if let Some(reduce) = reduce {
        // Reduce loop: for (oi = 0; oi < out_size; oi++) { ... }
        let oi_local = ctx.get_or_alloc_dim(fb, usize::MAX);
        let acc_local = fb.alloc_local(F32);
        let out_size_local = fb.alloc_local(I32);

        // Compute output size
        emit_dim(fb, &Dim::product(shape), &ctx.dim_locals);
        fb.local_set(out_size_local);

        // oi = 0
        fb.i32_const(0);
        fb.local_set(oi_local);

        // block { loop {
        fb.block();
        fb.loop_();

        // br_if $block (oi >= out_size)
        fb.local_get(oi_local);
        fb.local_get(out_size_local);
        fb.emit(op::I32_GE_S);
        fb.br_if(1);

        // Decompose oi into dim vars
        for d in 0..ndim {
            let dim_local = ctx.get_or_alloc_dim(fb, d);
            if d < ndim - 1 {
                fb.local_get(oi_local);
                emit_dim(fb, &out_strides[d], &ctx.dim_locals);
                fb.emit(op::I32_DIV_S);
                emit_dim(fb, &shape[d], &ctx.dim_locals);
                fb.emit(op::I32_REM_S);
            } else {
                fb.local_get(oi_local);
                emit_dim(fb, &shape[d], &ctx.dim_locals);
                fb.emit(op::I32_REM_S);
            }
            fb.local_set(dim_local);
        }

        // acc = init
        let init_val = match reduce.op {
            ReduceOp::Sum => 0.0f32,
            ReduceOp::Max => f32::NEG_INFINITY,
        };
        fb.f32_const(init_val);
        fb.local_set(acc_local);

        // Inner reduce loop
        let reduce_dim = ndim; // virtual dim
        let reduce_dim_local = ctx.get_or_alloc_dim(fb, reduce_dim);
        let reduce_size_local = fb.alloc_local(I32);
        emit_dim(fb, &reduce.size, &ctx.dim_locals);
        fb.local_set(reduce_size_local);

        fb.i32_const(0);
        fb.local_set(reduce_dim_local);

        fb.block();
        fb.loop_();

        // br_if (reduce_dim >= reduce_size)
        fb.local_get(reduce_dim_local);
        fb.local_get(reduce_size_local);
        fb.emit(op::I32_GE_S);
        fb.br_if(1);

        // Emit body instructions
        for (j, inst) in body.iter().enumerate() {
            emit_inst_code(fb, j, inst, ctx);
        }

        // Accumulate
        let result_val = ctx.inst_vars[&result];
        match reduce.op {
            ReduceOp::Sum => {
                fb.local_get(acc_local);
                fb.local_get(result_val);
                fb.emit(op::F32_ADD);
                fb.local_set(acc_local);
            }
            ReduceOp::Max => {
                // acc = result > acc ? result : acc
                fb.local_get(result_val);
                fb.local_get(acc_local);
                fb.local_get(result_val);
                fb.local_get(acc_local);
                fb.emit(op::F32_GT);
                fb.emit(op::SELECT);
                fb.local_set(acc_local);
            }
        }

        // reduce_dim++
        fb.local_get(reduce_dim_local);
        fb.i32_const(1);
        fb.emit(op::I32_ADD);
        fb.local_set(reduce_dim_local);
        fb.br(0);

        fb.end(); // loop
        fb.end(); // block

        // Store: out[oi] = acc
        fb.local_get(out_buf_ptr);
        fb.local_get(oi_local);
        fb.i32_const(4);
        fb.emit(op::I32_MUL);
        fb.emit(op::I32_ADD);
        fb.local_get(acc_local);
        fb.f32_store(0);

        // oi++
        fb.local_get(oi_local);
        fb.i32_const(1);
        fb.emit(op::I32_ADD);
        fb.local_set(oi_local);
        fb.br(0);

        fb.end(); // loop
        fb.end(); // block
    } else {
        // Simple elementwise loop
        let oi_local = ctx.get_or_alloc_dim(fb, usize::MAX);
        let out_size_local = fb.alloc_local(I32);

        emit_dim(fb, &Dim::product(shape), &ctx.dim_locals);
        fb.local_set(out_size_local);

        fb.i32_const(0);
        fb.local_set(oi_local);

        fb.block();
        fb.loop_();

        fb.local_get(oi_local);
        fb.local_get(out_size_local);
        fb.emit(op::I32_GE_S);
        fb.br_if(1);

        // Decompose oi into dim vars
        for d in 0..ndim {
            let dim_local = ctx.get_or_alloc_dim(fb, d);
            if d < ndim - 1 {
                fb.local_get(oi_local);
                emit_dim(fb, &out_strides[d], &ctx.dim_locals);
                fb.emit(op::I32_DIV_S);
                emit_dim(fb, &shape[d], &ctx.dim_locals);
                fb.emit(op::I32_REM_S);
            } else {
                fb.local_get(oi_local);
                emit_dim(fb, &shape[d], &ctx.dim_locals);
                fb.emit(op::I32_REM_S);
            }
            fb.local_set(dim_local);
        }

        // Emit body
        for (j, inst) in body.iter().enumerate() {
            emit_inst_code(fb, j, inst, ctx);
        }

        // Store result
        fb.local_get(out_buf_ptr);
        fb.local_get(oi_local);
        fb.i32_const(4);
        fb.emit(op::I32_MUL);
        fb.emit(op::I32_ADD);
        fb.local_get(ctx.inst_vars[&result]);
        fb.f32_store(0);

        // oi++
        fb.local_get(oi_local);
        fb.i32_const(1);
        fb.emit(op::I32_ADD);
        fb.local_set(oi_local);
        fb.br(0);

        fb.end(); // loop
        fb.end(); // block
    }
}

/// Emit a tiled loop (for matmul-like reduces with large dimensions).
fn emit_tiled_loop_code(
    fb: &mut FuncBuilder,
    out_buf_ptr: u32,
    shape: &[Dim],
    reduce: &loop_ir::ReduceDesc,
    body: &[Inst],
    result: usize,
    tile_cfg: &loop_ir::TileConfig,
    ctx: &mut CodegenCtx,
) {
    let ndim = shape.len();
    let tiles = &tile_cfg.tiles;
    let tk = tiles[ndim].as_usize().expect("tile sizes must be concrete");
    let reduce_dim = ndim;

    let batch_dims = ndim.saturating_sub(2);
    let m_dim = ndim - 2;
    let n_dim = ndim - 1;
    let tm = tiles[m_dim].as_usize().expect("tile sizes must be concrete");
    let tn = tiles[n_dim].as_usize().expect("tile sizes must be concrete");
    let unroll: usize = 32_usize.min(tn);

    let init_val = match reduce.op {
        ReduceOp::Sum => 0.0f32,
        ReduceOp::Max => f32::NEG_INFINITY,
    };

    let batch_strides = Dim::strides(shape);

    // Alloc some working locals
    let row_base_local = fb.alloc_local(I32);
    let m_end_local = fb.alloc_local(I32);
    let n_tile_local = fb.alloc_local(I32);
    let k_end_local = fb.alloc_local(I32);
    let ni_base_local = fb.alloc_local(I32);

    // Pre-alloc dim vars
    for d in 0..=ndim {
        ctx.get_or_alloc_dim(fb, d);
    }

    // Compute n_dependence for hoisting
    let depends_on_n = compute_n_dependence(body, n_dim);

    // Accumulator locals for unrolled section
    let mut acc_locals: Vec<u32> = Vec::new();
    for _ in 0..unroll {
        acc_locals.push(fb.alloc_local(F32));
    }
    let acc_r_local = fb.alloc_local(F32);

    // --- Batch loops ---
    let mut batch_locals = Vec::new();
    for d in 0..batch_dims {
        let dim_local = ctx.dim_vars[&d];
        let size_local = fb.alloc_local(I32);
        emit_dim(fb, &shape[d], &ctx.dim_locals);
        fb.local_set(size_local);
        batch_locals.push((dim_local, size_local));

        fb.i32_const(0);
        fb.local_set(dim_local);
        fb.block();
        fb.loop_();
        fb.local_get(dim_local);
        fb.local_get(size_local);
        fb.emit(op::I32_GE_S);
        fb.br_if(1);
    }

    // --- M block loop ---
    let m_blk_local = fb.alloc_local(I32);
    let m_blocks_local = fb.alloc_local(I32);
    let mi_local = fb.alloc_local(I32);
    let m_size_local = fb.alloc_local(I32);
    let n_size_local = fb.alloc_local(I32);
    let k_size_local = fb.alloc_local(I32);

    emit_dim(fb, &shape[m_dim], &ctx.dim_locals);
    fb.local_set(m_size_local);
    emit_dim(fb, &shape[n_dim], &ctx.dim_locals);
    fb.local_set(n_size_local);
    emit_dim(fb, &reduce.size, &ctx.dim_locals);
    fb.local_set(k_size_local);

    // m_blocks = (M + tm - 1) / tm
    fb.local_get(m_size_local);
    fb.i32_const((tm - 1) as i32);
    fb.emit(op::I32_ADD);
    fb.i32_const(tm as i32);
    fb.emit(op::I32_DIV_S);
    fb.local_set(m_blocks_local);

    fb.i32_const(0);
    fb.local_set(m_blk_local);
    fb.block();
    fb.loop_();
    fb.local_get(m_blk_local);
    fb.local_get(m_blocks_local);
    fb.emit(op::I32_GE_S);
    fb.br_if(1);

    // m_end = min(tm, M - m_blk * tm)
    fb.local_get(m_size_local);
    fb.local_get(m_blk_local);
    fb.i32_const(tm as i32);
    fb.emit(op::I32_MUL);
    fb.emit(op::I32_SUB);
    fb.local_tee(m_end_local);
    fb.i32_const(tm as i32);
    fb.local_get(m_end_local);
    fb.i32_const(tm as i32);
    fb.emit(op::I32_LT_S);
    fb.emit(op::SELECT);
    fb.local_set(m_end_local);

    // --- mi loop ---
    fb.i32_const(0);
    fb.local_set(mi_local);
    fb.block();
    fb.loop_();
    fb.local_get(mi_local);
    fb.local_get(m_end_local);
    fb.emit(op::I32_GE_S);
    fb.br_if(1);

    // d{m_dim} = m_blk * tm + mi
    let m_dim_local = ctx.dim_vars[&m_dim];
    fb.local_get(m_blk_local);
    fb.i32_const(tm as i32);
    fb.emit(op::I32_MUL);
    fb.local_get(mi_local);
    fb.emit(op::I32_ADD);
    fb.local_set(m_dim_local);

    // row_base = sum(d{batch} * stride) + d{m_dim} * n_size
    let mut first_term = true;
    for d in 0..batch_dims {
        fb.local_get(ctx.dim_vars[&d]);
        emit_dim(fb, &batch_strides[d], &ctx.dim_locals);
        fb.emit(op::I32_MUL);
        if !first_term {
            fb.emit(op::I32_ADD);
        }
        first_term = false;
    }
    fb.local_get(m_dim_local);
    fb.local_get(n_size_local);
    fb.emit(op::I32_MUL);
    if !first_term {
        fb.emit(op::I32_ADD);
    }
    fb.local_set(row_base_local);

    // --- N block loop ---
    let n_blk_local = fb.alloc_local(I32);
    let n_blocks_local = fb.alloc_local(I32);

    // n_blocks = (N + tn - 1) / tn
    fb.local_get(n_size_local);
    fb.i32_const((tn - 1) as i32);
    fb.emit(op::I32_ADD);
    fb.i32_const(tn as i32);
    fb.emit(op::I32_DIV_S);
    fb.local_set(n_blocks_local);

    fb.i32_const(0);
    fb.local_set(n_blk_local);
    fb.block();
    fb.loop_();
    fb.local_get(n_blk_local);
    fb.local_get(n_blocks_local);
    fb.emit(op::I32_GE_S);
    fb.br_if(1);

    // n_tile = min(tn, N - n_blk * tn)
    fb.local_get(n_size_local);
    fb.local_get(n_blk_local);
    fb.i32_const(tn as i32);
    fb.emit(op::I32_MUL);
    fb.emit(op::I32_SUB);
    fb.local_tee(n_tile_local);
    fb.i32_const(tn as i32);
    fb.local_get(n_tile_local);
    fb.i32_const(tn as i32);
    fb.emit(op::I32_LT_S);
    fb.emit(op::SELECT);
    fb.local_set(n_tile_local);

    // --- Unrolled ni_grp loop ---
    let ni_grp_local = fb.alloc_local(I32);
    let ni_grp_count_local = fb.alloc_local(I32);

    // ni_grp_count = n_tile / unroll
    fb.local_get(n_tile_local);
    fb.i32_const(unroll as i32);
    fb.emit(op::I32_DIV_S);
    fb.local_set(ni_grp_count_local);

    fb.i32_const(0);
    fb.local_set(ni_grp_local);
    fb.block();
    fb.loop_();
    fb.local_get(ni_grp_local);
    fb.local_get(ni_grp_count_local);
    fb.emit(op::I32_GE_S);
    fb.br_if(1);

    // ni_base = n_blk * tn + ni_grp * unroll
    fb.local_get(n_blk_local);
    fb.i32_const(tn as i32);
    fb.emit(op::I32_MUL);
    fb.local_get(ni_grp_local);
    fb.i32_const(unroll as i32);
    fb.emit(op::I32_MUL);
    fb.emit(op::I32_ADD);
    fb.local_set(ni_base_local);

    // Init accumulators
    for u in 0..unroll {
        fb.f32_const(init_val);
        fb.local_set(acc_locals[u]);
    }

    // --- K block loop ---
    let k_blk_local = fb.alloc_local(I32);
    let k_blocks_local = fb.alloc_local(I32);
    let ki_local = fb.alloc_local(I32);

    // k_blocks = (K + tk - 1) / tk
    fb.local_get(k_size_local);
    fb.i32_const((tk - 1) as i32);
    fb.emit(op::I32_ADD);
    fb.i32_const(tk as i32);
    fb.emit(op::I32_DIV_S);
    fb.local_set(k_blocks_local);

    fb.i32_const(0);
    fb.local_set(k_blk_local);
    fb.block();
    fb.loop_();
    fb.local_get(k_blk_local);
    fb.local_get(k_blocks_local);
    fb.emit(op::I32_GE_S);
    fb.br_if(1);

    // k_end = min(tk, K - k_blk * tk)
    fb.local_get(k_size_local);
    fb.local_get(k_blk_local);
    fb.i32_const(tk as i32);
    fb.emit(op::I32_MUL);
    fb.emit(op::I32_SUB);
    fb.local_tee(k_end_local);
    fb.i32_const(tk as i32);
    fb.local_get(k_end_local);
    fb.i32_const(tk as i32);
    fb.emit(op::I32_LT_S);
    fb.emit(op::SELECT);
    fb.local_set(k_end_local);

    // --- ki loop ---
    fb.i32_const(0);
    fb.local_set(ki_local);
    fb.block();
    fb.loop_();
    fb.local_get(ki_local);
    fb.local_get(k_end_local);
    fb.emit(op::I32_GE_S);
    fb.br_if(1);

    // d{reduce_dim} = k_blk * tk + ki
    let reduce_dim_local = ctx.dim_vars[&reduce_dim];
    fb.local_get(k_blk_local);
    fb.i32_const(tk as i32);
    fb.emit(op::I32_MUL);
    fb.local_get(ki_local);
    fb.emit(op::I32_ADD);
    fb.local_set(reduce_dim_local);

    // Emit n-invariant body instructions once
    for (j, inst) in body.iter().enumerate() {
        if !depends_on_n[j] {
            emit_inst_code(fb, j, inst, ctx);
        }
    }

    // Unrolled: emit n-dependent instructions for each unroll position
    let n_dim_local = ctx.dim_vars[&n_dim];
    for u in 0..unroll {
        // d{n_dim} = ni_base + u
        fb.local_get(ni_base_local);
        fb.i32_const(u as i32);
        fb.emit(op::I32_ADD);
        fb.local_set(n_dim_local);

        for (j, inst) in body.iter().enumerate() {
            if depends_on_n[j] {
                emit_inst_code(fb, j, inst, ctx);
            }
        }

        // Accumulate
        let result_val = ctx.inst_vars[&result];
        match reduce.op {
            ReduceOp::Sum => {
                fb.local_get(acc_locals[u]);
                fb.local_get(result_val);
                fb.emit(op::F32_ADD);
                fb.local_set(acc_locals[u]);
            }
            ReduceOp::Max => {
                fb.local_get(result_val);
                fb.local_get(acc_locals[u]);
                fb.local_get(result_val);
                fb.local_get(acc_locals[u]);
                fb.emit(op::F32_GT);
                fb.emit(op::SELECT);
                fb.local_set(acc_locals[u]);
            }
        }
    }

    // ki++
    fb.local_get(ki_local);
    fb.i32_const(1);
    fb.emit(op::I32_ADD);
    fb.local_set(ki_local);
    fb.br(0);
    fb.end(); // ki loop
    fb.end(); // ki block

    // k_blk++
    fb.local_get(k_blk_local);
    fb.i32_const(1);
    fb.emit(op::I32_ADD);
    fb.local_set(k_blk_local);
    fb.br(0);
    fb.end(); // k_blk loop
    fb.end(); // k_blk block

    // Write back accumulators: out[row_base + ni_base + u]
    for u in 0..unroll {
        fb.local_get(out_buf_ptr);
        fb.local_get(row_base_local);
        fb.local_get(ni_base_local);
        fb.emit(op::I32_ADD);
        fb.i32_const(u as i32);
        fb.emit(op::I32_ADD);
        fb.i32_const(4);
        fb.emit(op::I32_MUL);
        fb.emit(op::I32_ADD);
        fb.local_get(acc_locals[u]);
        fb.f32_store(0);
    }

    // ni_grp++
    fb.local_get(ni_grp_local);
    fb.i32_const(1);
    fb.emit(op::I32_ADD);
    fb.local_set(ni_grp_local);
    fb.br(0);
    fb.end(); // ni_grp loop
    fb.end(); // ni_grp block

    // --- Remainder loop for leftover ni elements ---
    let ni_local = fb.alloc_local(I32);
    let ni_start_local = fb.alloc_local(I32);

    // ni_start = (n_tile / unroll) * unroll
    fb.local_get(n_tile_local);
    fb.i32_const(unroll as i32);
    fb.emit(op::I32_DIV_S);
    fb.i32_const(unroll as i32);
    fb.emit(op::I32_MUL);
    fb.local_set(ni_start_local);

    fb.local_get(ni_start_local);
    fb.local_set(ni_local);
    fb.block();
    fb.loop_();
    fb.local_get(ni_local);
    fb.local_get(n_tile_local);
    fb.emit(op::I32_GE_S);
    fb.br_if(1);

    // d{n_dim} = n_blk * tn + ni
    fb.local_get(n_blk_local);
    fb.i32_const(tn as i32);
    fb.emit(op::I32_MUL);
    fb.local_get(ni_local);
    fb.emit(op::I32_ADD);
    fb.local_set(n_dim_local);

    // acc_r = init
    fb.f32_const(init_val);
    fb.local_set(acc_r_local);

    // K block loop for remainder
    let k_blk2_local = fb.alloc_local(I32);
    let ki2_local = fb.alloc_local(I32);
    let k_end2_local = fb.alloc_local(I32);

    fb.i32_const(0);
    fb.local_set(k_blk2_local);
    fb.block();
    fb.loop_();
    fb.local_get(k_blk2_local);
    fb.local_get(k_blocks_local);
    fb.emit(op::I32_GE_S);
    fb.br_if(1);

    // k_end
    fb.local_get(k_size_local);
    fb.local_get(k_blk2_local);
    fb.i32_const(tk as i32);
    fb.emit(op::I32_MUL);
    fb.emit(op::I32_SUB);
    fb.local_tee(k_end2_local);
    fb.i32_const(tk as i32);
    fb.local_get(k_end2_local);
    fb.i32_const(tk as i32);
    fb.emit(op::I32_LT_S);
    fb.emit(op::SELECT);
    fb.local_set(k_end2_local);

    fb.i32_const(0);
    fb.local_set(ki2_local);
    fb.block();
    fb.loop_();
    fb.local_get(ki2_local);
    fb.local_get(k_end2_local);
    fb.emit(op::I32_GE_S);
    fb.br_if(1);

    // d{reduce_dim} = k_blk * tk + ki
    fb.local_get(k_blk2_local);
    fb.i32_const(tk as i32);
    fb.emit(op::I32_MUL);
    fb.local_get(ki2_local);
    fb.emit(op::I32_ADD);
    fb.local_set(reduce_dim_local);

    // Emit full body
    for (j, inst) in body.iter().enumerate() {
        emit_inst_code(fb, j, inst, ctx);
    }

    // Accumulate
    let result_val = ctx.inst_vars[&result];
    match reduce.op {
        ReduceOp::Sum => {
            fb.local_get(acc_r_local);
            fb.local_get(result_val);
            fb.emit(op::F32_ADD);
            fb.local_set(acc_r_local);
        }
        ReduceOp::Max => {
            fb.local_get(result_val);
            fb.local_get(acc_r_local);
            fb.local_get(result_val);
            fb.local_get(acc_r_local);
            fb.emit(op::F32_GT);
            fb.emit(op::SELECT);
            fb.local_set(acc_r_local);
        }
    }

    fb.local_get(ki2_local);
    fb.i32_const(1);
    fb.emit(op::I32_ADD);
    fb.local_set(ki2_local);
    fb.br(0);
    fb.end(); // ki2 loop
    fb.end(); // ki2 block

    fb.local_get(k_blk2_local);
    fb.i32_const(1);
    fb.emit(op::I32_ADD);
    fb.local_set(k_blk2_local);
    fb.br(0);
    fb.end(); // k_blk2 loop
    fb.end(); // k_blk2 block

    // Store remainder: out[row_base + n_blk*tn + ni]
    fb.local_get(out_buf_ptr);
    fb.local_get(row_base_local);
    fb.local_get(n_blk_local);
    fb.i32_const(tn as i32);
    fb.emit(op::I32_MUL);
    fb.emit(op::I32_ADD);
    fb.local_get(ni_local);
    fb.emit(op::I32_ADD);
    fb.i32_const(4);
    fb.emit(op::I32_MUL);
    fb.emit(op::I32_ADD);
    fb.local_get(acc_r_local);
    fb.f32_store(0);

    // ni++
    fb.local_get(ni_local);
    fb.i32_const(1);
    fb.emit(op::I32_ADD);
    fb.local_set(ni_local);
    fb.br(0);
    fb.end(); // ni remainder loop
    fb.end(); // ni remainder block

    // Close n_blk loop
    fb.local_get(n_blk_local);
    fb.i32_const(1);
    fb.emit(op::I32_ADD);
    fb.local_set(n_blk_local);
    fb.br(0);
    fb.end(); // n_blk loop
    fb.end(); // n_blk block

    // Close mi loop
    fb.local_get(mi_local);
    fb.i32_const(1);
    fb.emit(op::I32_ADD);
    fb.local_set(mi_local);
    fb.br(0);
    fb.end(); // mi loop
    fb.end(); // mi block

    // Close m_blk loop
    fb.local_get(m_blk_local);
    fb.i32_const(1);
    fb.emit(op::I32_ADD);
    fb.local_set(m_blk_local);
    fb.br(0);
    fb.end(); // m_blk loop
    fb.end(); // m_blk block

    // Close batch loops
    for d in (0..batch_dims).rev() {
        let (dim_local, _) = batch_locals[d];
        fb.local_get(dim_local);
        fb.i32_const(1);
        fb.emit(op::I32_ADD);
        fb.local_set(dim_local);
        fb.br(0);
        fb.end(); // loop
        fb.end(); // block
    }
}

/// Emit pad statement code.
fn emit_pad_code(
    fb: &mut FuncBuilder,
    buf_ptr: u32,
    input_buf_ptr: u32,
    output_shape: &[Dim],
    input_shape: &[Dim],
    padding: &[(usize, usize)],
    ctx: &CodegenCtx,
) {
    let ndim = output_shape.len();
    let out_size_local = fb.alloc_local(I32);
    let in_size_local = fb.alloc_local(I32);
    let i_local = fb.alloc_local(I32);
    let ai_local = fb.alloc_local(I32);

    // Zero-fill output
    emit_dim(fb, &Dim::product(output_shape), &ctx.dim_locals);
    fb.local_set(out_size_local);

    fb.i32_const(0);
    fb.local_set(i_local);
    fb.block();
    fb.loop_();
    fb.local_get(i_local);
    fb.local_get(out_size_local);
    fb.emit(op::I32_GE_S);
    fb.br_if(1);

    // out[i] = 0.0
    fb.local_get(buf_ptr);
    fb.local_get(i_local);
    fb.i32_const(4);
    fb.emit(op::I32_MUL);
    fb.emit(op::I32_ADD);
    fb.f32_const(0.0);
    fb.f32_store(0);

    fb.local_get(i_local);
    fb.i32_const(1);
    fb.emit(op::I32_ADD);
    fb.local_set(i_local);
    fb.br(0);
    fb.end();
    fb.end();

    // Copy input with padding offsets
    let in_strides = Dim::strides(input_shape);
    let out_strides = Dim::strides(output_shape);

    emit_dim(fb, &Dim::product(input_shape), &ctx.dim_locals);
    fb.local_set(in_size_local);

    // Alloc dim locals for decomposition
    let mut pad_dims: Vec<u32> = Vec::new();
    for _ in 0..ndim {
        pad_dims.push(fb.alloc_local(I32));
    }
    let out_idx_local = fb.alloc_local(I32);

    fb.i32_const(0);
    fb.local_set(ai_local);
    fb.block();
    fb.loop_();
    fb.local_get(ai_local);
    fb.local_get(in_size_local);
    fb.emit(op::I32_GE_S);
    fb.br_if(1);

    // Decompose ai into dims
    for d in 0..ndim {
        if d < ndim - 1 {
            fb.local_get(ai_local);
            emit_dim(fb, &in_strides[d], &ctx.dim_locals);
            fb.emit(op::I32_DIV_S);
            emit_dim(fb, &input_shape[d], &ctx.dim_locals);
            fb.emit(op::I32_REM_S);
        } else {
            fb.local_get(ai_local);
            emit_dim(fb, &input_shape[d], &ctx.dim_locals);
            fb.emit(op::I32_REM_S);
        }
        fb.local_set(pad_dims[d]);
    }

    // Compute output index: sum of (d + lo) * out_stride
    let mut first = true;
    for d in 0..ndim {
        let (lo, _) = padding[d];
        fb.local_get(pad_dims[d]);
        fb.i32_const(lo as i32);
        fb.emit(op::I32_ADD);
        emit_dim(fb, &out_strides[d], &ctx.dim_locals);
        fb.emit(op::I32_MUL);
        if !first {
            fb.emit(op::I32_ADD);
        }
        first = false;
    }
    fb.local_set(out_idx_local);

    // out[out_idx] = in[ai]
    fb.local_get(buf_ptr);
    fb.local_get(out_idx_local);
    fb.i32_const(4);
    fb.emit(op::I32_MUL);
    fb.emit(op::I32_ADD);
    // Load from input
    fb.local_get(input_buf_ptr);
    fb.local_get(ai_local);
    fb.i32_const(4);
    fb.emit(op::I32_MUL);
    fb.emit(op::I32_ADD);
    fb.f32_load(0);
    fb.f32_store(0);

    fb.local_get(ai_local);
    fb.i32_const(1);
    fb.emit(op::I32_ADD);
    fb.local_set(ai_local);
    fb.br(0);
    fb.end();
    fb.end();
}

/// Determine which body instructions depend on the N dimension.
fn compute_n_dependence(body: &[Inst], n_dim: usize) -> Vec<bool> {
    let mut depends_on_n = vec![false; body.len()];
    for (j, inst) in body.iter().enumerate() {
        depends_on_n[j] = match inst {
            Inst::Load { index, .. } => match index {
                Index::Strided { parts, .. } => parts.iter().any(|(dim, _)| *dim == n_dim),
                Index::Flat => true,
            },
            Inst::DimVar(d) => *d == n_dim,
            Inst::Const(_) => false,
            Inst::Neg(a) | Inst::Recip(a) | Inst::Exp2(a) | Inst::Log2(a) | Inst::Sqrt(a) => {
                depends_on_n[*a]
            }
            Inst::Add(a, b) | Inst::Mul(a, b) | Inst::Max(a, b) | Inst::CmpLt(a, b) => {
                depends_on_n[*a] || depends_on_n[*b]
            }
        };
    }
    depends_on_n
}

// ============================================================================
// Public API
// ============================================================================

pub struct WasmBackend;

/// Collect all Param names from a Dim expression.
fn collect_params(dim: &Dim, params: &mut Vec<String>) {
    match dim {
        Dim::Param(name) => {
            if !params.contains(name) {
                params.push(name.clone());
            }
        }
        Dim::Add(a, b) | Dim::Mul(a, b) | Dim::Div(a, b) | Dim::Sub(a, b) => {
            collect_params(a, params);
            collect_params(b, params);
        }
        Dim::Lit(_) => {}
    }
}

impl WasmBackend {
    /// Emit a complete WASM binary from a graph using the fused loop IR.
    pub fn emit_fused(&self, graph: &Graph) -> Vec<u8> {
        self.emit_fused_inner(graph, None)
    }

    /// Emit a WASM binary with multiple outputs concatenated.
    pub fn emit_fused_multi_output(
        &self,
        graph: &Graph,
        outputs: &[tensor_lang_graph::NodeId],
    ) -> Vec<u8> {
        self.emit_fused_inner(graph, Some(outputs))
    }

    fn emit_fused_inner(
        &self,
        graph: &Graph,
        multi_outputs: Option<&[tensor_lang_graph::NodeId]>,
    ) -> Vec<u8> {
        let mut stmts = if let Some(outputs) = multi_outputs {
            loop_ir::lower_with_outputs(graph, outputs)
        } else {
            loop_ir::lower(graph)
        };
        loop_ir::tile_reduce_loops(&mut stmts);

        // Collect symbolic dim params
        let mut dim_params: Vec<String> = Vec::new();
        for node in &graph.nodes {
            for d in &node.shape {
                collect_params(d, &mut dim_params);
            }
        }
        dim_params.sort();
        dim_params.dedup();

        // Count inputs
        let inputs: Vec<(usize, String)> = graph
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| {
                if let Op::Input { name } = &n.op {
                    Some((i, name.clone()))
                } else {
                    None
                }
            })
            .collect();

        let n_dim_params = dim_params.len();
        let n_inputs = inputs.len();
        let n_params = (n_dim_params + n_inputs) as u32;

        // Build module
        let mut module = WasmModule::new(256); // 256 pages = 16MB initial

        // Add exp2 and log2 helper functions
        let f32_f32_type = module.add_type(vec![F32], vec![F32]);
        let exp2_fb = {
            let mut fb = FuncBuilder::new(1); // 1 param: x
            build_exp2_func(&mut fb);
            fb.encode()
        };
        let exp2_fn = module.add_function(f32_f32_type, exp2_fb);

        let log2_fb = {
            let mut fb = FuncBuilder::new(1);
            build_log2_func(&mut fb);
            fb.encode()
        };
        let log2_fn = module.add_function(f32_f32_type, log2_fb);

        // Build the execute function
        // Params: dim_params (i32...) + input_ptrs (i32...)
        let mut exec_params = Vec::new();
        for _ in 0..n_params {
            exec_params.push(I32);
        }
        let exec_type = module.add_type(exec_params, vec![I32]); // returns i32 (output ptr)
        let mut fb = FuncBuilder::new(n_params);

        // Map dim params to their parameter locals
        let mut dim_locals: DimLocals = HashMap::new();
        for (i, name) in dim_params.iter().enumerate() {
            dim_locals.insert(name.clone(), i as u32);
        }

        // Map input pointers to their parameter locals
        let mut input_ptr_locals: HashMap<usize, u32> = HashMap::new();
        for (i, (node_id, _)) in inputs.iter().enumerate() {
            input_ptr_locals.insert(*node_id, (n_dim_params + i) as u32);
        }

        // Heap pointer local — we bump-allocate from here
        let heap_ptr = fb.alloc_local(I32);

        // Initialize heap pointer: read from memory[0]
        // The host writes the initial heap pointer at byte 0.
        fb.i32_const(0);
        fb.i32_load(0);
        fb.local_set(heap_ptr);

        // Buffer pointer locals
        let mut buf_ptr_locals: HashMap<usize, u32> = HashMap::new();

        // Helper: bump-allocate n_bytes from heap
        fn bump_alloc(fb: &mut FuncBuilder, heap_ptr: u32, size_dim: &Dim, dim_locals: &DimLocals, result_local: u32) {
            // Align heap_ptr to 16 bytes first
            fb.local_get(heap_ptr);
            fb.i32_const(15);
            fb.emit(op::I32_ADD);
            fb.i32_const(-16); // ~15
            fb.emit(op::I32_AND);
            fb.local_set(heap_ptr);

            // result = heap_ptr
            fb.local_get(heap_ptr);
            fb.local_set(result_local);

            // heap_ptr += size * 4
            fb.local_get(heap_ptr);
            emit_dim(fb, size_dim, dim_locals);
            fb.i32_const(4);
            fb.emit(op::I32_MUL);
            fb.emit(op::I32_ADD);
            fb.local_set(heap_ptr);
        }

        // Process each statement
        for stmt in &stmts {
            match stmt {
                Stmt::Alloc { buf, size } => {
                    if let Op::Input { .. } = &graph.nodes[*buf].op {
                        // Input: use the parameter pointer directly
                        let param_local = input_ptr_locals[buf];
                        buf_ptr_locals.insert(*buf, param_local);
                    } else {
                        // Allocate from heap
                        let buf_local = fb.alloc_local(I32);
                        bump_alloc(&mut fb, heap_ptr, size, &dim_locals, buf_local);
                        buf_ptr_locals.insert(*buf, buf_local);
                    }
                }
                Stmt::Fill { buf, value } => {
                    let buf_ptr = buf_ptr_locals[buf];
                    fb.local_get(buf_ptr);
                    fb.f32_const(*value as f32);
                    fb.f32_store(0);
                }
                Stmt::FillArange { buf, size } => {
                    let buf_ptr = buf_ptr_locals[buf];
                    let i_local = fb.alloc_local(I32);
                    let size_local = fb.alloc_local(I32);
                    emit_dim(&mut fb, size, &dim_locals);
                    fb.local_set(size_local);

                    fb.i32_const(0);
                    fb.local_set(i_local);
                    fb.block();
                    fb.loop_();
                    fb.local_get(i_local);
                    fb.local_get(size_local);
                    fb.emit(op::I32_GE_S);
                    fb.br_if(1);

                    // buf[i] = f32(i)
                    fb.local_get(buf_ptr);
                    fb.local_get(i_local);
                    fb.i32_const(4);
                    fb.emit(op::I32_MUL);
                    fb.emit(op::I32_ADD);
                    fb.local_get(i_local);
                    fb.emit(op::F32_CONVERT_I32_S);
                    fb.f32_store(0);

                    fb.local_get(i_local);
                    fb.i32_const(1);
                    fb.emit(op::I32_ADD);
                    fb.local_set(i_local);
                    fb.br(0);
                    fb.end();
                    fb.end();
                }
                Stmt::Loop { buf, shape, reduce, body, result, tile } => {
                    let out_ptr = buf_ptr_locals[buf];

                    // Build codegen context for this loop
                    let mut ctx = CodegenCtx {
                        dim_vars: HashMap::new(),
                        inst_vars: HashMap::new(),
                        buf_ptrs: buf_ptr_locals.clone(),
                        dim_locals: dim_locals.clone(),
                        exp2_fn,
                        log2_fn,
                    };

                    if let (Some(reduce_desc), Some(tile_cfg)) = (reduce, tile) {
                        emit_tiled_loop_code(
                            &mut fb, out_ptr, shape, reduce_desc, body, *result, tile_cfg, &mut ctx,
                        );
                    } else {
                        emit_loop_code(
                            &mut fb, out_ptr, shape, reduce.as_ref(), body, *result, &mut ctx,
                        );
                    }
                }
                Stmt::Pad { buf, input_buf, output_shape, input_shape, padding } => {
                    let buf_ptr = buf_ptr_locals[buf];
                    let input_ptr = buf_ptr_locals[input_buf];
                    let ctx = CodegenCtx {
                        dim_vars: HashMap::new(),
                        inst_vars: HashMap::new(),
                        buf_ptrs: buf_ptr_locals.clone(),
                        dim_locals: dim_locals.clone(),
                        exp2_fn,
                        log2_fn,
                    };
                    emit_pad_code(&mut fb, buf_ptr, input_ptr, output_shape, input_shape, padding, &ctx);
                }
            }
        }

        // Return the output pointer
        if let Some(outputs) = multi_outputs {
            // Multi-output: allocate a result buffer and copy each output
            let result_ptr = fb.alloc_local(I32);
            let total_size = fb.alloc_local(I32);
            let off = fb.alloc_local(I32);
            let copy_i = fb.alloc_local(I32);

            // Compute total size
            fb.i32_const(0);
            for id in outputs {
                emit_dim(&mut fb, &Dim::product(&graph.nodes[id.0].shape), &dim_locals);
                fb.emit(op::I32_ADD);
            }
            fb.local_set(total_size);

            // Allocate result
            // total_size_local used directly below
            fb.local_get(heap_ptr);
            fb.i32_const(15);
            fb.emit(op::I32_ADD);
            fb.i32_const(-16);
            fb.emit(op::I32_AND);
            fb.local_tee(result_ptr);
            fb.local_get(total_size);
            fb.i32_const(4);
            fb.emit(op::I32_MUL);
            fb.emit(op::I32_ADD);
            fb.local_set(heap_ptr);

            // Copy each output
            fb.i32_const(0);
            fb.local_set(off);

            for id in outputs {
                let src_ptr = buf_ptr_locals[&id.0];
                let size_local = fb.alloc_local(I32);
                emit_dim(&mut fb, &Dim::product(&graph.nodes[id.0].shape), &dim_locals);
                fb.local_set(size_local);

                fb.i32_const(0);
                fb.local_set(copy_i);
                fb.block();
                fb.loop_();
                fb.local_get(copy_i);
                fb.local_get(size_local);
                fb.emit(op::I32_GE_S);
                fb.br_if(1);

                // result[off + i] = src[i]
                fb.local_get(result_ptr);
                fb.local_get(off);
                fb.local_get(copy_i);
                fb.emit(op::I32_ADD);
                fb.i32_const(4);
                fb.emit(op::I32_MUL);
                fb.emit(op::I32_ADD);

                fb.local_get(src_ptr);
                fb.local_get(copy_i);
                fb.i32_const(4);
                fb.emit(op::I32_MUL);
                fb.emit(op::I32_ADD);
                fb.f32_load(0);
                fb.f32_store(0);

                fb.local_get(copy_i);
                fb.i32_const(1);
                fb.emit(op::I32_ADD);
                fb.local_set(copy_i);
                fb.br(0);
                fb.end();
                fb.end();

                fb.local_get(off);
                fb.local_get(size_local);
                fb.emit(op::I32_ADD);
                fb.local_set(off);
            }

            // Store total_size at result_ptr - 4 for host to read
            // Actually, just return result_ptr; host knows sizes from graph shapes
            fb.local_get(result_ptr);
        } else {
            let last = graph.nodes.len() - 1;
            let out_ptr = buf_ptr_locals[&last];
            fb.local_get(out_ptr);
        }

        // Return is implicit (value on stack)
        let exec_body = fb.encode();
        let exec_fn = module.add_function(exec_type, exec_body);
        module.add_export("execute", 0x00, exec_fn);

        // Also export a "get_heap_ptr" so the host knows how much memory was used
        // (useful for grow decisions)

        module.encode()
    }
}
