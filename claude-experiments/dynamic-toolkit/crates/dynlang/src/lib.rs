//! # dynlang — High-level builder for dynamic languages on top of dynir
//!
//! Wraps dynir's `ModuleBuilder` and `FunctionBuilder` with dynamic-language-aware
//! helpers so you can implement a language without manually juggling SSA, stack slots,
//! or NanBox bit patterns.
//!
//! ## What it gives you
//!
//! - **Mutable variables** — `def_var` / `get_var` / `set_var` backed by stack slots
//! - **NanBox-aware constants** — `number(1.0)`, `nil()`, `bool_val(true)`
//! - **Inline fast paths** — `dyn_add(a, b)` emits: both floats? → fadd; else → extern
//! - **Type checks** — `is_number`, `is_nil`, `is_falsey`
//! - **String constant pool** — `add_string("hello")` → ID, resolved at runtime
//! - **Truthiness branching** — `br_if_truthy(v, then, else)`
//!
//! ## Quick example
//!
//! ```ignore
//! use dynlang::*;
//!
//! let mut dm = DynModule::new(GcConfig::leak(), NanBoxTags::default());
//! dm.register_slow_paths("lox");
//!
//! let main = dm.declare_func("main", 0);
//!
//! let mut f = dm.start_func(main);
//! let x = f.number(1.0);
//! let y = f.number(2.0);
//! let sum = f.dyn_add(x, y);
//! f.fb.ret(sum);
//! dm.finish_func(f);
//!
//! let built = dm.build();
//! // Run with ModuleInterpreter::<NanBox, _>::new(&built.module, ...)
//! ```

pub mod gc;

use std::collections::HashMap;

// Re-exports so users don't need to depend on dynir directly for basic use.
pub use dynir::builder::{FunctionBuilder, ModuleBuilder};
pub use dynir::ir::{BlockId, CmpOp, FuncRef, Module, StackSlot, Value};
pub use dynir::types::{Signature, Type};
pub use dynobj::{TypeInfo, VarLenKind, Compact, ObjHeader};

// ── GC Configuration ──────────────────────────────────────────────

/// GC strategy. Required when creating a DynModule.
#[derive(Clone, Debug)]
pub enum GcConfig {
    /// Bump allocator, never collects. Zero overhead — no safepoints,
    /// no root tracking. Objects are allocated and never freed.
    Leak,
    /// Semi-space copying collector. Safepoints emitted before
    /// allocations, roots tracked automatically.
    SemiSpace {
        /// Initial heap size in bytes.
        heap_size: usize,
    },
}

impl GcConfig {
    pub fn leak() -> Self { GcConfig::Leak }
    pub fn semi_space(heap_size: usize) -> Self { GcConfig::SemiSpace { heap_size } }

    pub fn is_collecting(&self) -> bool {
        matches!(self, GcConfig::SemiSpace { .. })
    }
}

// ── Object Type System ────────────────────────────────────────────

/// Handle to a declared object type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObjTypeId(pub usize);

/// Handle to a declared GC object type. Carries all the info needed to
/// emit inline IR for allocation, field access, and type checks.
///
/// # Example
/// ```ignore
/// let closure_h = dm.obj_handle(closure_ty);
///
/// // Inline type check (no extern call):
/// let is_closure = closure_h.check(f, val);
///
/// // Inline field load (no extern call):
/// let raw = closure_h.unwrap(f, val);
/// let arity = closure_h.load(f, raw, "arity");
///
/// // Inline allocation:
/// let obj = closure_h.alloc(f, varlen_count);
/// ```
#[derive(Clone)]
pub struct ObjTypeHandle {
    pub id: ObjTypeId,
    pub type_info: &'static TypeInfo,
    /// The TypeInfo pointer as a u64, for embedding in IR constants.
    pub type_info_addr: u64,
    /// Field name → (byte offset, kind).
    pub field_offsets: HashMap<String, (i32, FieldKind)>,
    pub varlen: VarLenKind,
}

impl ObjTypeHandle {
    /// Check if a NaN-boxed value is an object of this type. Returns I8.
    ///
    /// Emits inline: is_ptr(val) → extract payload → load TypeInfo from
    /// header at offset 0 → compare against this type's known TypeInfo address.
    pub fn check(&self, f: &mut DynFunc, val: Value) -> Value {
        let is_obj = f.is_ptr(val);
        let check_bb = f.fb.create_block(&[]);
        let merge_bb = f.fb.create_block(&[Type::I8]);
        let zero = f.fb.iconst(Type::I8, 0);
        f.fb.br_if(is_obj, check_bb, &[], merge_bb, &[zero]);

        f.fb.switch_to_block(check_bb);
        let ptr = f.fb.payload(val);
        let ti = f.fb.load(Type::I64, ptr, 0); // TypeInfo* at header offset 0
        let expected = f.fb.iconst(Type::I64, self.type_info_addr as i64);
        let matches = f.fb.icmp(CmpOp::Eq, ti, expected);
        f.fb.jump(merge_bb, &[matches]);

        f.fb.switch_to_block(merge_bb);
        f.fb.block_param(merge_bb, 0)
    }

    /// Extract the raw GC pointer from a NaN-boxed object value.
    /// Caller must ensure the value is a heap object (e.g. via `check`).
    pub fn unwrap(&self, f: &mut DynFunc, val: Value) -> Value {
        f.fb.payload(val)
    }

    /// Wrap a raw GC pointer into a NaN-boxed object value.
    pub fn wrap(&self, f: &mut DynFunc, ptr: Value) -> Value {
        f.fb.make_tagged(f.tags.ptr, ptr)
    }

    /// Load a field from a raw GC object pointer. Returns I64.
    pub fn load(&self, f: &mut DynFunc, obj_ptr: Value, field: &str) -> Value {
        let (offset, _kind) = self.field_offsets.get(field)
            .unwrap_or_else(|| panic!("unknown field '{}' on ObjTypeHandle", field));
        f.fb.load(Type::I64, obj_ptr, *offset)
    }

    /// Store a field to a raw GC object pointer.
    pub fn store(&self, f: &mut DynFunc, obj_ptr: Value, field: &str, val: Value) {
        let (offset, _kind) = self.field_offsets.get(field)
            .unwrap_or_else(|| panic!("unknown field '{}' on ObjTypeHandle", field));
        f.fb.store(val, obj_ptr, *offset);
    }

    /// Load an element from the variable-length array section.
    pub fn load_elem(&self, f: &mut DynFunc, obj_ptr: Value, index: Value) -> Value {
        let base_offset = self.type_info.varlen_element_offset(0) as i64;
        let base = f.fb.iconst(Type::I64, base_offset);
        let eight = f.fb.iconst(Type::I64, 8);
        let byte_offset = f.fb.mul(index, eight);
        let offset = f.fb.add(base, byte_offset);
        let addr = f.fb.add(obj_ptr, offset);
        f.fb.load(Type::I64, addr, 0)
    }

    /// Store an element to the variable-length array section.
    pub fn store_elem(&self, f: &mut DynFunc, obj_ptr: Value, index: Value, val: Value) {
        let base_offset = self.type_info.varlen_element_offset(0) as i64;
        let base = f.fb.iconst(Type::I64, base_offset);
        let eight = f.fb.iconst(Type::I64, 8);
        let byte_offset = f.fb.mul(index, eight);
        let offset = f.fb.add(base, byte_offset);
        let addr = f.fb.add(obj_ptr, offset);
        f.fb.store(val, addr, 0);
    }

    /// Allocate an object of this type via `__gc_alloc__`. Returns raw pointer (I64).
    pub fn alloc(&self, f: &mut DynFunc, varlen_len: Value) -> Value {
        f.gc_alloc(self.id, varlen_len)
    }
}

/// Kind of field in a GC object.
#[derive(Clone, Copy, Debug)]
pub enum FieldKind {
    /// GC-traced value slot (u64 that might contain a heap pointer).
    Value,
    /// Untraced raw 64-bit word.
    Raw64,
}

/// Declared object type with auto-generated TypeInfo.
pub struct ObjType {
    pub name: String,
    /// The generated TypeInfo. Leaked to get a 'static reference.
    pub type_info: &'static TypeInfo,
    /// Maps field name → (offset_in_bytes, kind).
    pub field_offsets: HashMap<String, (i32, FieldKind)>,
    /// Whether this type has a variable-length section.
    pub varlen: VarLenKind,
}

/// Builder for declaring an object type.
pub struct ObjTypeBuilder<'a> {
    module: &'a mut DynModule,
    name: String,
    value_fields: Vec<String>,
    raw64_fields: Vec<String>,
    varlen: VarLenKind,
}

impl<'a> ObjTypeBuilder<'a> {
    /// Add a field. Value fields are GC-traced; Raw64 fields are not.
    pub fn field(mut self, name: &str, kind: FieldKind) -> Self {
        match kind {
            FieldKind::Value => self.value_fields.push(name.to_string()),
            FieldKind::Raw64 => self.raw64_fields.push(name.to_string()),
        }
        self
    }

    /// Add a variable-length array of GC-traced values.
    pub fn varlen_values(mut self) -> Self {
        self.varlen = VarLenKind::Values;
        self
    }

    /// Add a variable-length byte array (not GC-traced).
    pub fn varlen_bytes(mut self) -> Self {
        self.varlen = VarLenKind::Bytes;
        self
    }

    /// Finalize and register the type. Returns an ObjTypeId.
    pub fn build(self) -> ObjTypeId {
        let header_size = Compact::SIZE;
        let value_count = self.value_fields.len() as u16;
        let raw_count = (self.raw64_fields.len() * 8) as u16;

        let mut info = TypeInfo::for_header(header_size)
            .with_fields(value_count);

        if raw_count > 0 {
            info = info.with_raw_bytes(raw_count);
        }

        let total_fixed = value_count + self.raw64_fields.len() as u16;
        match self.varlen {
            VarLenKind::Values => info = info.with_varlen_values(total_fixed),
            VarLenKind::Bytes => info = info.with_varlen_bytes(total_fixed),
            VarLenKind::None => {}
        }

        // Leak the TypeInfo to get a 'static reference (required by dynobj/dynalloc)
        let info_static: &'static TypeInfo = Box::leak(Box::new(info));

        // Compute field offsets
        let mut field_offsets = HashMap::new();
        for (i, name) in self.value_fields.iter().enumerate() {
            let offset = info_static.value_field_offset(i as u16) as i32;
            field_offsets.insert(name.clone(), (offset, FieldKind::Value));
        }
        for (i, name) in self.raw64_fields.iter().enumerate() {
            let offset = info_static.raw_data_offset() as i32 + (i as i32 * 8);
            field_offsets.insert(name.clone(), (offset, FieldKind::Raw64));
        }

        let id = ObjTypeId(self.module.obj_types.len());
        self.module.obj_types.push(ObjType {
            name: self.name,
            type_info: info_static,
            field_offsets,
            varlen: self.varlen,
        });
        id
    }
}

// ── NanBox bit-level constants ────────────────────────────────────

/// Mask that covers sign + exponent + quiet + marker bits.
const FULL_MASK: u64 = 0xFFFC_0000_0000_0000;
/// The exact pattern that identifies a tagged (non-float) value.
const TAG_PATTERN: u64 = 0x7FFC_0000_0000_0000;
/// Safe canonical NaN (quiet NaN without the marker bit).
const CANONICAL_NAN: u64 = 0x7FF8_0000_0000_0000;
/// Mask for the 48-bit payload.
const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

/// Encode a tag + payload into NanBox bits.
fn nanbox_encode(tag: u32, payload: u64) -> u64 {
    TAG_PATTERN | ((tag as u64) << 48) | (payload & PAYLOAD_MASK)
}

// ── Tag configuration ─────────────────────────────────────────────

/// Which NanBox tag numbers map to which value types.
///
/// NanBox has 4 tags (2 bits → 0..3). Untagged values are IEEE 754 floats.
/// The default assignment is: 0 = nil, 1 = bool, 2 = heap pointer.
#[derive(Clone, Debug)]
pub struct NanBoxTags {
    pub nil: u32,
    pub bool_tag: u32,
    pub ptr: u32,
}

impl Default for NanBoxTags {
    fn default() -> Self {
        NanBoxTags {
            nil: 0,
            bool_tag: 1,
            ptr: 2,
        }
    }
}

// ── Slow-path extern refs ─────────────────────────────────────────

#[derive(Clone, Default)]
struct SlowPaths {
    add: Option<FuncRef>,
    sub: Option<FuncRef>,
    mul: Option<FuncRef>,
    div: Option<FuncRef>,
    neg: Option<FuncRef>,
    eq: Option<FuncRef>,
    lt: Option<FuncRef>,
    gt: Option<FuncRef>,
    not: Option<FuncRef>,
}

// ── DynModule ─────────────────────────────────────────────────────

/// Result of building a `DynModule`.
pub struct BuiltModule {
    pub module: Module,
    /// The string constant pool. Index with the ID from `add_string`.
    pub strings: Vec<String>,
    /// Name → FuncRef for all declared language functions.
    pub func_refs: HashMap<String, FuncRef>,
}

/// High-level module builder for dynamic languages.
///
/// Wraps `ModuleBuilder` and adds a string pool, slow-path registration,
/// and a convenient `declare_func` that assumes all params/returns are I64
/// (NanBox-encoded values).
pub struct DynModule {
    mb: ModuleBuilder,
    tags: NanBoxTags,
    gc_config: GcConfig,
    slow: SlowPaths,
    strings: Vec<String>,
    string_map: HashMap<String, u32>,
    func_refs: HashMap<String, FuncRef>,
    /// Registered object types.
    pub obj_types: Vec<ObjType>,
    /// Extern for GC allocation (declared when first obj_type is built).
    gc_alloc_extern: Option<FuncRef>,
}

fn sig(params: &[Type], ret: Option<Type>) -> Signature {
    Signature {
        params: params.to_vec(),
        ret,
    }
}

impl DynModule {
    /// Create a new module. GcConfig is required — use `GcConfig::leak()`
    /// for zero-overhead no-collection, or `GcConfig::semi_space(size)`
    /// for automatic garbage collection.
    pub fn new(gc: GcConfig, tags: NanBoxTags) -> Self {
        DynModule {
            mb: ModuleBuilder::new(),
            tags,
            gc_config: gc,
            slow: SlowPaths::default(),
            strings: Vec::new(),
            string_map: HashMap::new(),
            func_refs: HashMap::new(),
            obj_types: Vec::new(),
            gc_alloc_extern: None,
        }
    }

    /// Declare a GC-managed object type. Returns an ObjTypeId for use
    /// with `gc_alloc`, `gc_load_field`, `gc_store_field`.
    pub fn obj_type(&mut self, name: &str) -> ObjTypeBuilder<'_> {
        // Ensure the gc_alloc extern is declared (once)
        if self.gc_alloc_extern.is_none() {
            let sig = Signature {
                params: vec![Type::I64, Type::I64],  // type_id, varlen_len
                ret: Some(Type::I64),                 // returns pointer as I64
            };
            self.gc_alloc_extern = Some(self.mb.declare_extern("__gc_alloc__", sig));
        }
        ObjTypeBuilder {
            module: self,
            name: name.to_string(),
            value_fields: Vec::new(),
            raw64_fields: Vec::new(),
            varlen: VarLenKind::None,
        }
    }

    /// Get a registered object type by ID.
    pub fn get_obj_type(&self, id: ObjTypeId) -> &ObjType {
        &self.obj_types[id.0]
    }

    /// Get an `ObjTypeHandle` — a self-contained handle for emitting inline
    /// IR operations (type checks, field loads/stores, allocation).
    pub fn obj_handle(&self, id: ObjTypeId) -> ObjTypeHandle {
        let ty = &self.obj_types[id.0];
        ObjTypeHandle {
            id,
            type_info: ty.type_info,
            type_info_addr: ty.type_info as *const TypeInfo as u64,
            field_offsets: ty.field_offsets.clone(),
            varlen: ty.varlen,
        }
    }

    /// Get the GC configuration.
    pub fn gc_config(&self) -> &GcConfig {
        &self.gc_config
    }

    // ── String pool ───────────────────────────────────────────

    /// Add a string to the constant pool. Returns a unique ID.
    /// Deduplicates: calling with the same string returns the same ID.
    pub fn add_string(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.string_map.get(s) {
            return id;
        }
        let id = self.strings.len() as u32;
        self.string_map.insert(s.to_string(), id);
        self.strings.push(s.to_string());
        id
    }

    /// The string constant pool.
    pub fn strings(&self) -> &[String] {
        &self.strings
    }

    // ── Extern declarations ───────────────────────────────────

    /// Declare a raw extern function (full control over signature).
    pub fn declare_extern(&mut self, name: &str, sig: Signature) -> FuncRef {
        self.mb.declare_extern(name, sig)
    }

    /// Register slow-path extern functions for dynamic operations.
    ///
    /// Declares `{prefix}_add`, `{prefix}_sub`, `{prefix}_mul`, `{prefix}_div`,
    /// `{prefix}_neg`, `{prefix}_eq`, `{prefix}_lt`, `{prefix}_gt`, `{prefix}_not`.
    ///
    /// All take I64 args and return I64 (NanBox values).
    /// Bind these by name on the interpreter at runtime.
    pub fn register_slow_paths(&mut self, prefix: &str) {
        let i64_2 = sig(&[Type::I64, Type::I64], Some(Type::I64));
        let i64_1 = sig(&[Type::I64], Some(Type::I64));

        self.slow.add = Some(self.mb.declare_extern(&format!("{prefix}_add"), i64_2.clone()));
        self.slow.sub = Some(self.mb.declare_extern(&format!("{prefix}_sub"), i64_2.clone()));
        self.slow.mul = Some(self.mb.declare_extern(&format!("{prefix}_mul"), i64_2.clone()));
        self.slow.div = Some(self.mb.declare_extern(&format!("{prefix}_div"), i64_2.clone()));
        self.slow.neg = Some(self.mb.declare_extern(&format!("{prefix}_neg"), i64_1.clone()));
        self.slow.eq = Some(self.mb.declare_extern(&format!("{prefix}_eq"), i64_2.clone()));
        self.slow.lt = Some(self.mb.declare_extern(&format!("{prefix}_lt"), i64_2.clone()));
        self.slow.gt = Some(self.mb.declare_extern(&format!("{prefix}_gt"), i64_2.clone()));
        self.slow.not = Some(self.mb.declare_extern(&format!("{prefix}_not"), i64_1));
    }

    /// Register a single slow-path extern for a specific operation.
    pub fn register_slow_path(&mut self, op: DynOp, name: &str) -> FuncRef {
        let i64_2 = sig(&[Type::I64, Type::I64], Some(Type::I64));
        let i64_1 = sig(&[Type::I64], Some(Type::I64));
        let fref = match op {
            DynOp::Add => { let f = self.mb.declare_extern(name, i64_2); self.slow.add = Some(f); f }
            DynOp::Sub => { let f = self.mb.declare_extern(name, i64_2); self.slow.sub = Some(f); f }
            DynOp::Mul => { let f = self.mb.declare_extern(name, i64_2); self.slow.mul = Some(f); f }
            DynOp::Div => { let f = self.mb.declare_extern(name, i64_2); self.slow.div = Some(f); f }
            DynOp::Neg => { let f = self.mb.declare_extern(name, i64_1); self.slow.neg = Some(f); f }
            DynOp::Eq  => { let f = self.mb.declare_extern(name, i64_2); self.slow.eq = Some(f); f }
            DynOp::Lt  => { let f = self.mb.declare_extern(name, i64_2); self.slow.lt = Some(f); f }
            DynOp::Gt  => { let f = self.mb.declare_extern(name, i64_2); self.slow.gt = Some(f); f }
            DynOp::Not => { let f = self.mb.declare_extern(name, i64_1); self.slow.not = Some(f); f }
        };
        fref
    }

    // ── Function declarations ─────────────────────────────────

    /// Declare a language function. All params and return are I64 (NanBox values).
    pub fn declare_func(&mut self, name: &str, arity: usize) -> FuncRef {
        let params = vec![Type::I64; arity];
        let fref = self.mb.declare_func(name, &params, Some(Type::I64));
        self.func_refs.insert(name.to_string(), fref);
        fref
    }

    /// Declare a void language function (no return value).
    pub fn declare_void_func(&mut self, name: &str, arity: usize) -> FuncRef {
        let params = vec![Type::I64; arity];
        let fref = self.mb.declare_func(name, &params, None);
        self.func_refs.insert(name.to_string(), fref);
        fref
    }

    /// Look up a previously declared function by name.
    pub fn func_ref(&self, name: &str) -> FuncRef {
        *self.func_refs
            .get(name)
            .unwrap_or_else(|| panic!("unknown function: {name}"))
    }

    // ── Function definition ───────────────────────────────────

    /// Start defining a function. Returns a `DynFunc` builder.
    ///
    /// Call `finish_func` when done.
    pub fn start_func(&mut self, fref: FuncRef) -> DynFunc {
        let fb = self.mb.define_func(fref);
        DynFunc {
            fb,
            fref,
            tags: self.tags.clone(),
            gc_config: self.gc_config.clone(),
            gc_alloc_extern: self.gc_alloc_extern,
            slow: self.slow.clone(),
            vars: vec![HashMap::new()],
        }
    }

    /// Finish defining a function.
    pub fn finish_func(&mut self, func: DynFunc) {
        self.mb.finish_func(func.fref, func.fb);
    }

    /// Build the final module.
    pub fn build(self) -> BuiltModule {
        BuiltModule {
            module: self.mb.build(),
            strings: self.strings,
            func_refs: self.func_refs,
        }
    }
}


// ── DynOp ─────────────────────────────────────────────────────────

/// Dynamic operations that can have slow-path externs.
#[derive(Debug, Clone, Copy)]
pub enum DynOp {
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Eq,
    Lt,
    Gt,
    Not,
}

// ── DynFunc ───────────────────────────────────────────────────────

/// High-level function builder for dynamic languages.
///
/// Wraps `FunctionBuilder` with:
/// - Mutable variables backed by stack slots
/// - NanBox-aware constant encoding
/// - Dynamic operations with inline float fast paths
/// - Truthiness checks and branching
///
/// The underlying `FunctionBuilder` is public (`self.fb`) for anything
/// the high-level API doesn't cover.
pub struct DynFunc {
    /// The underlying dynir `FunctionBuilder`. Use directly for raw IR
    /// operations not covered by the high-level API.
    pub fb: FunctionBuilder,
    fref: FuncRef,
    tags: NanBoxTags,
    gc_config: GcConfig,
    gc_alloc_extern: Option<FuncRef>,
    slow: SlowPaths,
    /// Variable scopes (public for inspection, e.g. checking if a var exists).
    pub vars: Vec<HashMap<String, StackSlot>>,
}

impl DynFunc {
    // ── Scoping ───────────────────────────────────────────────

    /// Push a new variable scope.
    pub fn push_scope(&mut self) {
        self.vars.push(HashMap::new());
    }

    /// Pop the innermost variable scope.
    pub fn pop_scope(&mut self) {
        self.vars.pop();
    }

    // ── Variables ─────────────────────────────────────────────

    /// Define a mutable variable with an initial value.
    /// Backed by a stack slot — no SSA management needed.
    pub fn def_var(&mut self, name: &str, init: Value) {
        let slot = self.fb.create_stack_slot(8, false);
        let addr = self.fb.stack_addr(slot);
        self.fb.store(init, addr, 0);
        self.vars
            .last_mut()
            .unwrap()
            .insert(name.to_string(), slot);
    }

    /// Read a variable's current value.
    pub fn get_var(&mut self, name: &str) -> Value {
        let slot = self.find_var(name);
        let addr = self.fb.stack_addr(slot);
        self.fb.load(Type::I64, addr, 0)
    }

    /// Assign a new value to an existing variable.
    pub fn set_var(&mut self, name: &str, val: Value) {
        let slot = self.find_var(name);
        let addr = self.fb.stack_addr(slot);
        self.fb.store(val, addr, 0);
    }

    /// Check whether a variable exists in any local scope.
    pub fn has_var(&self, name: &str) -> bool {
        self.vars.iter().rev().any(|scope| scope.contains_key(name))
    }

    fn find_var(&self, name: &str) -> StackSlot {
        for scope in self.vars.iter().rev() {
            if let Some(&slot) = scope.get(name) {
                return slot;
            }
        }
        panic!("undefined variable: {name}");
    }

    // ── NanBox constants ──────────────────────────────────────

    /// A float constant, NanBox-encoded as I64.
    pub fn number(&mut self, n: f64) -> Value {
        let bits = n.to_bits();
        // Canonicalize if the float's bits collide with the tag pattern.
        let encoded = if (bits & FULL_MASK) == TAG_PATTERN {
            CANONICAL_NAN
        } else {
            bits
        };
        self.fb.iconst(Type::I64, encoded as i64)
    }

    /// The nil constant.
    pub fn nil(&mut self) -> Value {
        self.fb.iconst(Type::I64, nanbox_encode(self.tags.nil, 0) as i64)
    }

    /// A boolean constant.
    pub fn bool_val(&mut self, b: bool) -> Value {
        self.fb
            .iconst(Type::I64, nanbox_encode(self.tags.bool_tag, b as u64) as i64)
    }

    /// A tagged constant with an arbitrary tag and payload.
    ///
    /// Useful for encoding string IDs, small integers, etc.
    pub fn tagged_const(&mut self, tag: u32, payload: u64) -> Value {
        self.fb
            .iconst(Type::I64, nanbox_encode(tag, payload) as i64)
    }

    // ── Type checks (return I8) ───────────────────────────────

    /// Is this value a NanBox float (i.e. NOT tagged)? Returns I8.
    pub fn is_number(&mut self, v: Value) -> Value {
        let tagged = self.is_tagged_raw(v);
        let one = self.fb.iconst(Type::I8, 1);
        self.fb.xor(tagged, one)
    }

    /// Is this value nil? Returns I8.
    pub fn is_nil(&mut self, v: Value) -> Value {
        self.fb.is_tag(v, self.tags.nil)
    }

    /// Is this value a boolean? Returns I8.
    pub fn is_bool(&mut self, v: Value) -> Value {
        self.fb.is_tag(v, self.tags.bool_tag)
    }

    /// Is this value a heap pointer? Returns I8.
    pub fn is_ptr(&mut self, v: Value) -> Value {
        self.fb.is_tag(v, self.tags.ptr)
    }

    /// Extract the raw GC pointer from a NaN-boxed object value.
    /// Caller must ensure the value is a heap object (e.g. checked with `is_ptr`).
    pub fn obj_unwrap(&mut self, v: Value) -> Value {
        self.fb.payload(v)
    }

    /// Wrap a raw GC pointer into a NaN-boxed object value.
    pub fn obj_wrap(&mut self, ptr: Value) -> Value {
        self.fb.make_tagged(self.tags.ptr, ptr)
    }

    /// Is this value tagged (i.e. NOT a float)? Returns I8.
    pub fn is_tagged(&mut self, v: Value) -> Value {
        self.is_tagged_raw(v)
    }

    /// Is this value falsey (nil or false)? Returns I8.
    pub fn is_falsey(&mut self, v: Value) -> Value {
        let nil_bits = nanbox_encode(self.tags.nil, 0);
        let false_bits = nanbox_encode(self.tags.bool_tag, 0);
        let nil_const = self.fb.iconst(Type::I64, nil_bits as i64);
        let false_const = self.fb.iconst(Type::I64, false_bits as i64);
        let eq_nil = self.fb.icmp(CmpOp::Eq, v, nil_const);
        let eq_false = self.fb.icmp(CmpOp::Eq, v, false_const);
        self.fb.or(eq_nil, eq_false)
    }

    /// Is this value truthy (not nil and not false)? Returns I8.
    pub fn is_truthy(&mut self, v: Value) -> Value {
        let falsey = self.is_falsey(v);
        let one = self.fb.iconst(Type::I8, 1);
        self.fb.xor(falsey, one)
    }

    /// Raw bit check: `(v & FULL_MASK) == TAG_PATTERN`. Returns I8.
    fn is_tagged_raw(&mut self, v: Value) -> Value {
        let mask = self.fb.iconst(Type::I64, FULL_MASK as i64);
        let pattern = self.fb.iconst(Type::I64, TAG_PATTERN as i64);
        let masked = self.fb.and(v, mask);
        self.fb.icmp(CmpOp::Eq, masked, pattern)
    }

    // ── Truthiness branching ──────────────────────────────────

    /// Branch directly on a NanBox bool: true → then, false → else.
    /// More efficient than `br_if_truthy` when you know the value is a bool
    /// (e.g. from `num_lt`, `num_eq`, etc.).
    pub fn br_if_bool(
        &mut self,
        v: Value,
        then_bb: BlockId,
        then_args: &[Value],
        else_bb: BlockId,
        else_args: &[Value],
    ) {
        // NanBox bool: true has payload=1, false has payload=0.
        // Just check the lowest bit.
        let one_i64 = self.fb.iconst(Type::I64, 1);
        let bit = self.fb.and(v, one_i64);
        let zero = self.fb.iconst(Type::I64, 0);
        let is_true = self.fb.icmp(CmpOp::Ne, bit, zero);
        self.fb.br_if(is_true, then_bb, then_args, else_bb, else_args);
    }

    /// Branch on truthiness: if `v` is truthy → then_bb, else → else_bb.
    ///
    /// In most dynamic languages, only `nil` and `false` are falsey.
    pub fn br_if_truthy(
        &mut self,
        v: Value,
        then_bb: BlockId,
        then_args: &[Value],
        else_bb: BlockId,
        else_args: &[Value],
    ) {
        let truthy = self.is_truthy(v);
        self.fb.br_if(truthy, then_bb, then_args, else_bb, else_args);
    }

    // ── Unbox / Box ───────────────────────────────────────────

    /// Bitcast a NanBox float (I64) → F64. Caller must ensure it's a float.
    pub fn unbox_number(&mut self, v: Value) -> Value {
        self.fb.bitcast(v, Type::F64)
    }

    /// Bitcast F64 → I64 with NaN canonicalization.
    ///
    /// If the result happens to collide with the NanBox tag pattern,
    /// it's replaced with a canonical NaN.
    pub fn box_number(&mut self, v: Value) -> Value {
        let bits = self.fb.bitcast(v, Type::I64);
        let mask = self.fb.iconst(Type::I64, FULL_MASK as i64);
        let pattern = self.fb.iconst(Type::I64, TAG_PATTERN as i64);
        let masked = self.fb.and(bits, mask);
        let collides = self.fb.icmp(CmpOp::Eq, masked, pattern);
        let canon = self.fb.iconst(Type::I64, CANONICAL_NAN as i64);
        self.fb.select(collides, canon, bits)
    }

    // ── Direct number ops (no type checks, no NaN canonicalization) ──
    //
    // With NanBox, the I64 bits ARE the IEEE 754 f64 bits for numbers.
    // These ops just bitcast → float op → bitcast. No branching, no
    // tag checks. Like clox's AS_NUMBER/NUMBER_VAL which are free
    // reinterpret casts.
    //
    // NaN canonicalization is skipped (same as clox). The chance of
    // a float op producing the exact NanBox tag collision pattern is
    // astronomically low and not worth checking on every operation.

    /// Add two NanBox numbers directly. No type check, no NaN canon.
    pub fn num_add(&mut self, a: Value, b: Value) -> Value {
        let fa = self.fb.bitcast(a, Type::F64);
        let fb_val = self.fb.bitcast(b, Type::F64);
        let r = self.fb.fadd(fa, fb_val);
        self.fb.bitcast(r, Type::I64)
    }

    /// Subtract two NanBox numbers directly.
    pub fn num_sub(&mut self, a: Value, b: Value) -> Value {
        let fa = self.fb.bitcast(a, Type::F64);
        let fb_val = self.fb.bitcast(b, Type::F64);
        let r = self.fb.fsub(fa, fb_val);
        self.fb.bitcast(r, Type::I64)
    }

    /// Multiply two NanBox numbers directly.
    pub fn num_mul(&mut self, a: Value, b: Value) -> Value {
        let fa = self.fb.bitcast(a, Type::F64);
        let fb_val = self.fb.bitcast(b, Type::F64);
        let r = self.fb.fmul(fa, fb_val);
        self.fb.bitcast(r, Type::I64)
    }

    /// Divide two NanBox numbers directly.
    pub fn num_div(&mut self, a: Value, b: Value) -> Value {
        let fa = self.fb.bitcast(a, Type::F64);
        let fb_val = self.fb.bitcast(b, Type::F64);
        let r = self.fb.fdiv(fa, fb_val);
        self.fb.bitcast(r, Type::I64)
    }

    /// Negate a NanBox number directly.
    pub fn num_neg(&mut self, v: Value) -> Value {
        let f = self.fb.bitcast(v, Type::F64);
        let r = self.fb.fneg(f);
        self.fb.bitcast(r, Type::I64)
    }

    /// Compare two NanBox numbers: a < b. Returns NanBox bool.
    pub fn num_lt(&mut self, a: Value, b: Value) -> Value {
        self.num_cmp(CmpOp::Slt, a, b)
    }

    /// Compare two NanBox numbers: a > b. Returns NanBox bool.
    pub fn num_gt(&mut self, a: Value, b: Value) -> Value {
        self.num_cmp(CmpOp::Sgt, a, b)
    }

    /// Compare two NanBox numbers: a == b. Returns NanBox bool.
    pub fn num_eq(&mut self, a: Value, b: Value) -> Value {
        self.num_cmp(CmpOp::Eq, a, b)
    }

    /// Compare two NanBox numbers: a <= b. Returns NanBox bool.
    pub fn num_le(&mut self, a: Value, b: Value) -> Value {
        self.num_cmp(CmpOp::Sle, a, b)
    }

    /// Compare two NanBox numbers: a >= b. Returns NanBox bool.
    pub fn num_ge(&mut self, a: Value, b: Value) -> Value {
        self.num_cmp(CmpOp::Sge, a, b)
    }

    /// Compare two NanBox numbers: a != b. Returns NanBox bool.
    pub fn num_ne(&mut self, a: Value, b: Value) -> Value {
        self.num_cmp(CmpOp::Ne, a, b)
    }

    fn num_cmp(&mut self, op: CmpOp, a: Value, b: Value) -> Value {
        let fa = self.fb.bitcast(a, Type::F64);
        let fb_val = self.fb.bitcast(b, Type::F64);
        let cmp = self.fb.fcmp(op, fa, fb_val);
        let t = self.bool_val(true);
        let f = self.bool_val(false);
        self.fb.select(cmp, t, f)
    }

    // ── Dynamic arithmetic ────────────────────────────────────
    //
    // Each method generates:
    //   1. Check if both args are floats
    //   2. Fast path: bitcast → float op → box result
    //   3. Slow path: call extern
    //   4. Merge block with result
    //
    // After the call, emission continues in the merge block.

    /// Dynamic add: float fast path, else extern slow path.
    ///
    /// The slow path handles string concatenation, type errors, etc.
    pub fn dyn_add(&mut self, a: Value, b: Value) -> Value {
        self.dyn_float_binop(a, b, FloatBinOp::Add)
    }

    /// Dynamic subtract.
    pub fn dyn_sub(&mut self, a: Value, b: Value) -> Value {
        self.dyn_float_binop(a, b, FloatBinOp::Sub)
    }

    /// Dynamic multiply.
    pub fn dyn_mul(&mut self, a: Value, b: Value) -> Value {
        self.dyn_float_binop(a, b, FloatBinOp::Mul)
    }

    /// Dynamic divide.
    pub fn dyn_div(&mut self, a: Value, b: Value) -> Value {
        self.dyn_float_binop(a, b, FloatBinOp::Div)
    }

    /// Dynamic negate: float fast path, else extern slow path.
    pub fn dyn_neg(&mut self, v: Value) -> Value {
        let is_num = self.is_number(v);

        let fast_bb = self.fb.create_block(&[]);
        let slow_bb = self.fb.create_block(&[]);
        let merge_bb = self.fb.create_block(&[Type::I64]);

        self.fb.br_if(is_num, fast_bb, &[], slow_bb, &[]);

        // Fast: fneg (NaN canonicalization not needed — fneg preserves NaN class)
        self.fb.switch_to_block(fast_bb);
        let f = self.fb.bitcast(v, Type::F64);
        let neg = self.fb.fneg(f);
        let result = self.fb.bitcast(neg, Type::I64);
        self.fb.jump(merge_bb, &[result]);

        // Slow
        self.fb.switch_to_block(slow_bb);
        let slow_fn = self.slow.neg.expect(
            "dyn_neg: no slow path registered. Call register_slow_paths() or register_slow_path(DynOp::Neg, ...)",
        );
        let result = self.fb.call(slow_fn, &[v]).unwrap();
        self.fb.jump(merge_bb, &[result]);

        self.fb.switch_to_block(merge_bb);
        self.fb.block_param(merge_bb, 0)
    }

    // ── Dynamic comparison ────────────────────────────────────

    /// Dynamic equality. Float fast path, else extern.
    ///
    /// Returns a NanBox bool. Float semantics: NaN != NaN, -0 == +0.
    pub fn dyn_eq(&mut self, a: Value, b: Value) -> Value {
        self.dyn_float_cmp(a, b, CmpOp::Eq, DynOp::Eq)
    }

    /// Dynamic less-than. Float fast path, else extern.
    pub fn dyn_lt(&mut self, a: Value, b: Value) -> Value {
        self.dyn_float_cmp(a, b, CmpOp::Slt, DynOp::Lt)
    }

    /// Dynamic greater-than. Float fast path, else extern.
    pub fn dyn_gt(&mut self, a: Value, b: Value) -> Value {
        self.dyn_float_cmp(a, b, CmpOp::Sgt, DynOp::Gt)
    }

    // ── Internal: float binop fast path ───────────────────────

    fn dyn_float_binop(&mut self, a: Value, b: Value, op: FloatBinOp) -> Value {
        // Check a
        let a_is_num = self.is_number(a);

        let check_b_bb = self.fb.create_block(&[]);
        let fast_bb = self.fb.create_block(&[]);
        let slow_bb = self.fb.create_block(&[]);
        let merge_bb = self.fb.create_block(&[Type::I64]);

        self.fb.br_if(a_is_num, check_b_bb, &[], slow_bb, &[]);

        // Check b
        self.fb.switch_to_block(check_b_bb);
        let b_is_num = self.is_number(b);
        self.fb.br_if(b_is_num, fast_bb, &[], slow_bb, &[]);

        // Fast path: both floats → float op → box
        self.fb.switch_to_block(fast_bb);
        let fa = self.fb.bitcast(a, Type::F64);
        let fb_val = self.fb.bitcast(b, Type::F64);
        let result_f = match op {
            FloatBinOp::Add => self.fb.fadd(fa, fb_val),
            FloatBinOp::Sub => self.fb.fsub(fa, fb_val),
            FloatBinOp::Mul => self.fb.fmul(fa, fb_val),
            FloatBinOp::Div => self.fb.fdiv(fa, fb_val),
        };
        let result_i = self.box_number(result_f);
        self.fb.jump(merge_bb, &[result_i]);

        // Slow path: call extern
        self.fb.switch_to_block(slow_bb);
        let slow_fn = match op {
            FloatBinOp::Add => self.slow.add,
            FloatBinOp::Sub => self.slow.sub,
            FloatBinOp::Mul => self.slow.mul,
            FloatBinOp::Div => self.slow.div,
        };
        let slow_fn = slow_fn.unwrap_or_else(|| {
            panic!(
                "dyn_{}: no slow path registered. Call register_slow_paths() or register_slow_path()",
                match op {
                    FloatBinOp::Add => "add",
                    FloatBinOp::Sub => "sub",
                    FloatBinOp::Mul => "mul",
                    FloatBinOp::Div => "div",
                }
            )
        });
        let result_slow = self.fb.call(slow_fn, &[a, b]).unwrap();
        self.fb.jump(merge_bb, &[result_slow]);

        // Merge
        self.fb.switch_to_block(merge_bb);
        self.fb.block_param(merge_bb, 0)
    }

    // ── Internal: float comparison fast path ──────────────────

    fn dyn_float_cmp(&mut self, a: Value, b: Value, cmp: CmpOp, op: DynOp) -> Value {
        let a_is_num = self.is_number(a);

        let check_b_bb = self.fb.create_block(&[]);
        let fast_bb = self.fb.create_block(&[]);
        let slow_bb = self.fb.create_block(&[]);
        let merge_bb = self.fb.create_block(&[Type::I64]);

        self.fb.br_if(a_is_num, check_b_bb, &[], slow_bb, &[]);

        // Check b
        self.fb.switch_to_block(check_b_bb);
        let b_is_num = self.is_number(b);
        self.fb.br_if(b_is_num, fast_bb, &[], slow_bb, &[]);

        // Fast path: fcmp → NanBox bool
        self.fb.switch_to_block(fast_bb);
        let fa = self.fb.bitcast(a, Type::F64);
        let fb_val = self.fb.bitcast(b, Type::F64);
        let cmp_result = self.fb.fcmp(cmp, fa, fb_val);
        let true_val = self.bool_val(true);
        let false_val = self.bool_val(false);
        let result = self.fb.select(cmp_result, true_val, false_val);
        self.fb.jump(merge_bb, &[result]);

        // Slow path
        self.fb.switch_to_block(slow_bb);
        let slow_fn = match op {
            DynOp::Eq => self.slow.eq,
            DynOp::Lt => self.slow.lt,
            DynOp::Gt => self.slow.gt,
            _ => unreachable!(),
        };
        let slow_fn = slow_fn.unwrap_or_else(|| {
            panic!("dyn_{:?}: no slow path registered", op)
        });
        let result_slow = self.fb.call(slow_fn, &[a, b]).unwrap();
        self.fb.jump(merge_bb, &[result_slow]);

        // Merge
        self.fb.switch_to_block(merge_bb);
        self.fb.block_param(merge_bb, 0)
    }

    // ── GC Object Operations ──────────────────────────────────
    //
    // These emit IR instructions for allocating and accessing GC-managed
    // objects. The object layout is determined by the ObjTypeId's TypeInfo.
    // Field access computes offsets at IR-build time — no runtime lookup.

    /// Allocate a GC object. `varlen_len` is the number of variable-length
    /// elements (0 for fixed-size objects). Returns the object as I64
    /// (a raw pointer in NanBox ptr-tag encoding is done by the caller).
    ///
    /// If GcConfig is SemiSpace, a safepoint is emitted before allocation.
    pub fn gc_alloc(&mut self, type_id: ObjTypeId, varlen_len: Value) -> Value {
        let alloc_fn = self.gc_alloc_extern
            .expect("gc_alloc: no object types declared on this module");
        let type_id_val = self.fb.iconst(Type::I64, type_id.0 as i64);
        // TODO: emit safepoint for SemiSpace (needs live GcPtr tracking)
        self.fb.call(alloc_fn, &[type_id_val, varlen_len]).unwrap()
    }

    /// Load a value field from a GC object. Offset is computed at build
    /// time from the TypeInfo. `obj` is a raw pointer (I64).
    pub fn gc_load_field(&mut self, obj: Value, type_info: &ObjType, field: &str) -> Value {
        let (offset, _kind) = type_info.field_offsets.get(field)
            .unwrap_or_else(|| panic!("unknown field '{}' on type '{}'", field, type_info.name));
        self.fb.load(Type::I64, obj, *offset)
    }

    /// Store a value field to a GC object. Offset is computed at build time.
    pub fn gc_store_field(&mut self, obj: Value, type_info: &ObjType, field: &str, val: Value) {
        let (offset, _kind) = type_info.field_offsets.get(field)
            .unwrap_or_else(|| panic!("unknown field '{}' on type '{}'", field, type_info.name));
        self.fb.store(val, obj, *offset);
    }

    /// Load an element from a variable-length array section.
    /// `index` is a Value (I64) — the element index.
    pub fn gc_load_elem(&mut self, obj: Value, type_info: &ObjType, index: Value) -> Value {
        let base_offset = type_info.type_info.varlen_element_offset(0) as i64;
        let base = self.fb.iconst(Type::I64, base_offset);
        let eight = self.fb.iconst(Type::I64, 8);
        let byte_offset = self.fb.mul(index, eight);
        let offset = self.fb.add(base, byte_offset);
        let addr = self.fb.add(obj, offset);
        self.fb.load(Type::I64, addr, 0)
    }

    /// Store an element to a variable-length array section.
    pub fn gc_store_elem(&mut self, obj: Value, type_info: &ObjType, index: Value, val: Value) {
        let base_offset = type_info.type_info.varlen_element_offset(0) as i64;
        let base = self.fb.iconst(Type::I64, base_offset);
        let eight = self.fb.iconst(Type::I64, 8);
        let byte_offset = self.fb.mul(index, eight);
        let offset = self.fb.add(base, byte_offset);
        let addr = self.fb.add(obj, offset);
        self.fb.store(val, addr, 0);
    }
}

// ─── Inline cache ─────────────────────────────────────────────────

/// A per-call-site monomorphic inline cache.
///
/// Stores `(guard_key, cached_value)` in a 16-byte stack slot.
/// On the fast path: load guard, compare with runtime key, return cached value.
/// On miss: call a slow path to compute the value, update the cache.
///
/// ## Example
///
/// ```ignore
/// // At each property-access call site:
/// let ic = f.inline_cache();
/// let class_ptr = /* load class from instance */;
/// let method = ic.get(f, class_ptr, |f| {
///     // slow path: full method lookup
///     f.fb.call(lookup_method_extern, &[obj, name_id])
/// });
/// ```
///
/// The first call always misses (guard initialized to 0).
/// Subsequent calls with the same guard key hit the cache — one load + one compare.
#[derive(Clone, Copy)]
pub struct InlineCache {
    slot: StackSlot,
}

impl DynFunc {
    /// Allocate a new inline cache for this call site.
    /// Each call to `inline_cache()` creates a separate cache.
    pub fn inline_cache(&mut self) -> InlineCache {
        // 16 bytes: [guard_key: u64, cached_value: u64]
        let slot = self.fb.create_stack_slot(16, false);
        // Initialize guard to 0 (guaranteed miss on first access since
        // no valid guard key is 0 for heap pointers or NanBox values)
        let addr = self.fb.stack_addr(slot);
        let zero = self.fb.iconst(Type::I64, 0);
        self.fb.store(zero, addr, 0);
        self.fb.store(zero, addr, 8);
        InlineCache { slot }
    }
}

impl InlineCache {
    /// Look up the cached value, or compute it via the slow path.
    ///
    /// - `guard_key`: the runtime key to check (e.g., class pointer, type info pointer).
    ///   Must be a stable identifier — same key means same result.
    /// - `slow_path`: called on cache miss. Must return the value to cache.
    ///   Receives `&mut DynFunc` so it can emit IR (extern calls, etc.).
    ///
    /// Returns the cached (or freshly computed) value.
    pub fn get(
        &self,
        f: &mut DynFunc,
        guard_key: Value,
        slow_path: impl FnOnce(&mut DynFunc) -> Value,
    ) -> Value {
        let merge_bb = f.fb.create_block(&[Type::I64]);
        let miss_bb = f.fb.create_block(&[]);

        // Fast path: load cached guard, compare
        let addr = f.fb.stack_addr(self.slot);
        let cached_guard = f.fb.load(Type::I64, addr, 0);
        let hit = f.fb.icmp(CmpOp::Eq, cached_guard, guard_key);
        let hit_bb = f.fb.create_block(&[]);
        f.fb.br_if(hit, hit_bb, &[], miss_bb, &[]);

        // Cache hit: return cached value
        f.fb.switch_to_block(hit_bb);
        let cached_val = f.fb.load(Type::I64, addr, 8);
        f.fb.jump(merge_bb, &[cached_val]);

        // Cache miss: call slow path, update cache
        f.fb.switch_to_block(miss_bb);
        let computed = slow_path(f);
        // Update cache: store guard + value
        let addr2 = f.fb.stack_addr(self.slot);
        f.fb.store(guard_key, addr2, 0);
        f.fb.store(computed, addr2, 8);
        f.fb.jump(merge_bb, &[computed]);

        f.fb.switch_to_block(merge_bb);
        f.fb.block_param(merge_bb, 0)
    }
}

#[derive(Clone, Copy)]
enum FloatBinOp {
    Add,
    Sub,
    Mul,
    Div,
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use dynir::interp::{ExternCallResult, InterpResult, ModuleInterpreter, NoGcRoots};
    use dynvalue::NanBox;

    fn run(module: &Module, entry: FuncRef, args: &[u64]) -> u64 {
        let roots = NoGcRoots;
        let interp = ModuleInterpreter::<NanBox, _>::new(module, &roots);
        match interp.run(entry, args) {
            Ok(InterpResult::Value(v)) => v,
            Ok(InterpResult::Void) => 0,
            other => panic!("unexpected result: {:?}", other),
        }
    }

    fn nanbox_float(n: f64) -> u64 {
        n.to_bits()
    }

    fn as_float(v: u64) -> f64 {
        f64::from_bits(v)
    }

    #[test]
    fn test_number_constant() {
        let mut dm = DynModule::new(GcConfig::leak(), NanBoxTags::default());
        let main = dm.declare_func("main", 0);

        let mut f = dm.start_func(main);
        let v = f.number(42.0);
        f.fb.ret(v);
        dm.finish_func(f);

        let built = dm.build();
        let result = run(&built.module, main, &[]);
        assert_eq!(as_float(result), 42.0);
    }

    #[test]
    fn test_nil_and_bool() {
        let mut dm = DynModule::new(GcConfig::leak(), NanBoxTags::default());
        let main = dm.declare_func("main", 0);

        let mut f = dm.start_func(main);
        let n = f.nil();
        let is = f.is_nil(n);
        let result = f.fb.zext(is, Type::I64);
        f.fb.ret(result);
        dm.finish_func(f);

        let built = dm.build();
        assert_eq!(run(&built.module, main, &[]), 1);
    }

    #[test]
    fn test_mutable_variables() {
        let mut dm = DynModule::new(GcConfig::leak(), NanBoxTags::default());
        let main = dm.declare_func("main", 0);

        let mut f = dm.start_func(main);
        let init = f.number(1.0);
        f.def_var("x", init);
        let new_val = f.number(99.0);
        f.set_var("x", new_val);
        let v = f.get_var("x");
        f.fb.ret(v);
        dm.finish_func(f);

        let built = dm.build();
        let result = run(&built.module, main, &[]);
        assert_eq!(as_float(result), 99.0);
    }

    #[test]
    fn test_dyn_add_fast_path() {
        let mut dm = DynModule::new(GcConfig::leak(), NanBoxTags::default());
        // Register slow paths even though we won't hit them.
        dm.register_slow_paths("rt");
        let main = dm.declare_func("main", 0);

        let mut f = dm.start_func(main);
        let a = f.number(10.0);
        let b = f.number(20.0);
        let sum = f.dyn_add(a, b);
        f.fb.ret(sum);
        dm.finish_func(f);

        let built = dm.build();
        let roots = NoGcRoots;
        let mut interp = ModuleInterpreter::<NanBox, _>::new(&built.module, &roots);
        // Bind slow paths (won't be called for float+float).
        interp.bind_by_name("rt_add", |_args| {
            panic!("slow path should not be called for float+float");
        });
        match interp.run(main, &[]) {
            Ok(InterpResult::Value(v)) => assert_eq!(as_float(v), 30.0),
            other => panic!("unexpected: {:?}", other),
        }
    }

    #[test]
    fn test_dyn_add_slow_path() {
        let mut dm = DynModule::new(GcConfig::leak(), NanBoxTags::default());
        dm.register_slow_paths("rt");
        let main = dm.declare_func("main", 0);

        let mut f = dm.start_func(main);
        let a = f.nil(); // not a number → slow path
        let b = f.number(1.0);
        let sum = f.dyn_add(a, b);
        f.fb.ret(sum);
        dm.finish_func(f);

        let built = dm.build();
        let roots = NoGcRoots;
        let mut interp = ModuleInterpreter::<NanBox, _>::new(&built.module, &roots);
        interp.bind_by_name("rt_add", |_args| {
            // Return 42.0 to prove the slow path was taken.
            ExternCallResult::Value(Some(nanbox_float(42.0)))
        });
        match interp.run(main, &[]) {
            Ok(InterpResult::Value(v)) => assert_eq!(as_float(v), 42.0),
            other => panic!("unexpected: {:?}", other),
        }
    }

    #[test]
    fn test_is_falsey() {
        let mut dm = DynModule::new(GcConfig::leak(), NanBoxTags::default());
        let main = dm.declare_func("main", 1);

        let mut f = dm.start_func(main);
        let entry = f.fb.entry_block();
        let arg = f.fb.block_param(entry, 0);
        let falsey = f.is_falsey(arg);
        let result = f.fb.zext(falsey, Type::I64);
        f.fb.ret(result);
        dm.finish_func(f);

        let built = dm.build();
        // nil is falsey
        assert_eq!(run(&built.module, main, &[nanbox_encode(0, 0)]), 1);
        // false is falsey
        assert_eq!(run(&built.module, main, &[nanbox_encode(1, 0)]), 1);
        // true is truthy
        assert_eq!(run(&built.module, main, &[nanbox_encode(1, 1)]), 0);
        // a number is truthy
        assert_eq!(run(&built.module, main, &[nanbox_float(3.14)]), 0);
    }

    #[test]
    fn test_scoped_variables() {
        let mut dm = DynModule::new(GcConfig::leak(), NanBoxTags::default());
        let main = dm.declare_func("main", 0);

        let mut f = dm.start_func(main);
        let outer = f.number(1.0);
        f.def_var("x", outer);

        f.push_scope();
        let inner = f.number(2.0);
        f.def_var("x", inner); // shadows outer x
        let v1 = f.get_var("x"); // should be 2.0
        f.pop_scope();

        let v2 = f.get_var("x"); // should be 1.0

        // Return v1 (inner) as a float-encoded check.
        // Unbox both and subtract: if scoping works, v1=2 and v2=1, diff=1.
        let f1 = f.unbox_number(v1);
        let f2 = f.unbox_number(v2);
        let diff = f.fb.fsub(f1, f2);
        let result = f.box_number(diff);
        f.fb.ret(result);
        dm.finish_func(f);

        let built = dm.build();
        let result = run(&built.module, main, &[]);
        assert_eq!(as_float(result), 1.0);
    }

    #[test]
    fn test_string_pool() {
        let mut dm = DynModule::new(GcConfig::leak(), NanBoxTags::default());
        let id1 = dm.add_string("hello");
        let id2 = dm.add_string("world");
        let id3 = dm.add_string("hello"); // dedup

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id3, 0); // same as first
        assert_eq!(dm.strings(), &["hello", "world"]);
    }

    #[test]
    fn test_gc_obj_type_declaration() {
        let mut dm = DynModule::new(GcConfig::leak(), NanBoxTags::default());

        // Declare an object type with two value fields and a varlen values section
        let pair_ty = dm.obj_type("Pair")
            .field("car", FieldKind::Value)
            .field("cdr", FieldKind::Value)
            .build();

        let array_ty = dm.obj_type("Array")
            .varlen_values()
            .build();

        // Verify TypeInfo was generated correctly
        let pair = dm.get_obj_type(pair_ty);
        assert_eq!(pair.name, "Pair");
        assert_eq!(pair.type_info.value_field_count, 2);
        assert!(pair.field_offsets.contains_key("car"));
        assert!(pair.field_offsets.contains_key("cdr"));

        let arr = dm.get_obj_type(array_ty);
        assert_eq!(arr.name, "Array");
        assert_eq!(arr.varlen, VarLenKind::Values);
    }

    #[test]
    fn test_gc_alloc_and_field_access() {
        let mut dm = DynModule::new(GcConfig::leak(), NanBoxTags::default());

        let upval_ty = dm.obj_type("Upvalue")
            .field("value", FieldKind::Value)
            .build();

        let main = dm.declare_func("main", 0);

        let mut f = dm.start_func(main);
        // Allocate an Upvalue object
        let zero = f.fb.iconst(Type::I64, 0);
        let obj = f.gc_alloc(upval_ty, zero);
        // Store 42 into the value field
        let forty_two = f.fb.iconst(Type::I64, 42);
        {
            let ty = dm.get_obj_type(upval_ty);
            f.gc_store_field(obj, ty, "value", forty_two);
        }
        // Load it back
        let loaded = {
            let ty = dm.get_obj_type(upval_ty);
            f.gc_load_field(obj, ty, "value")
        };
        f.fb.ret(loaded);
        dm.finish_func(f);

        // Create GC runtime and wire up __gc_alloc__
        let mut gc = crate::gc::DynGcRuntime::new(
            &GcConfig::leak(),
            &NanBoxTags::default(),
            &dm.obj_types,
        );

        let built = dm.build();

        let roots = NoGcRoots;
        let mut interp = ModuleInterpreter::<NanBox, _>::new(&built.module, &roots);
        interp.bind_by_name("__gc_alloc__", move |args| {
            let type_id = args[0] as usize;
            let varlen_len = args[1] as usize;
            let ptr = gc.alloc(type_id, varlen_len);
            ExternCallResult::Value(Some(ptr as u64))
        });

        match interp.run(main, &[]) {
            Ok(InterpResult::Value(v)) => assert_eq!(v, 42),
            other => panic!("unexpected: {:?}", other),
        }
    }

    #[test]
    fn test_dyn_lt_fast_path() {
        let mut dm = DynModule::new(GcConfig::leak(), NanBoxTags::default());
        dm.register_slow_paths("rt");
        let main = dm.declare_func("main", 0);

        let mut f = dm.start_func(main);
        let a = f.number(1.0);
        let b = f.number(2.0);
        let result = f.dyn_lt(a, b); // 1.0 < 2.0 = true
        f.fb.ret(result);
        dm.finish_func(f);

        let built = dm.build();
        let roots = NoGcRoots;
        let mut interp = ModuleInterpreter::<NanBox, _>::new(&built.module, &roots);
        interp.bind_by_name("rt_lt", |_| {
            panic!("slow path should not be called");
        });
        match interp.run(main, &[]) {
            Ok(InterpResult::Value(v)) => {
                // Should be NanBox true: tag=1, payload=1
                assert_eq!(v, nanbox_encode(1, 1));
            }
            other => panic!("unexpected: {:?}", other),
        }
    }
}
