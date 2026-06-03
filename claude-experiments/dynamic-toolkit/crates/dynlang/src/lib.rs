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
//! - **GC-aware object refs** — use `ObjRef` helpers so raw object pointers stay rooted across allocations
//!
//! ## Quick example
//!
//! ```ignore
//! use dynlang::*;
//!
//! let mut dm = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
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

pub mod closure;
pub mod gc;
pub mod host;
pub mod ic;
pub mod inline_body;
pub mod slow_paths;
pub mod stdlib;

use std::collections::HashMap;

// Re-exports so users don't need to depend on dynir directly for basic use.
pub use dynir::builder::{FunctionBuilder, ModuleBuilder};
pub use dynir::ir::{BlockId, CmpOp, FuncRef, Module, StackSlot, Value};
pub use dynir::types::{Signature, Type};
pub use dynobj::{Compact, ObjHeader, TypeInfo, VarLenKind};
pub use dynruntime::GcPolicy;

// ── GC Configuration ──────────────────────────────────────────────

/// GC strategy. Required when creating a DynModule.
///
/// Generational, thread-aware, concurrent-capable `dynalloc::Heap`.
/// Supports precise stack-map roots for the JIT and card-table write
/// barriers for old→young pointers.
///
/// **Pair with `CallMode::ControlAware { safepoint_handler }`** and
/// emit `Inst::Safepoint` at every allocation site (`Module::
/// validate_safepoints` enforces this). Anything else is a memory-
/// safety bug waiting to happen — a moving collection running
/// without a stack map for the active frame relocates objects out
/// from under live values that the GC can't see.
#[derive(Clone, Debug)]
pub enum GcConfig {
    Generational {
        /// Total heap size per space, in bytes.
        heap_size: usize,
        /// Optional nursery size for a young-generation split. `None`
        /// disables the generational split (single tenured space).
        nursery_size: Option<usize>,
    },
}

impl GcConfig {
    pub fn generational(heap_size: usize) -> Self {
        GcConfig::Generational {
            heap_size,
            nursery_size: None,
        }
    }
    pub fn generational_with_nursery(heap_size: usize, nursery_size: usize) -> Self {
        GcConfig::Generational {
            heap_size,
            nursery_size: Some(nursery_size),
        }
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
/// // Typed raw object ref:
/// let raw = closure_h.unwrap_ref(f, val);
/// let arity = closure_h.load_ref(f, raw, "arity");
///
/// // Rooted allocation:
/// let obj = closure_h.alloc_ref(f, varlen_count);
/// ```
#[derive(Clone)]
pub struct ObjTypeHandle {
    pub id: ObjTypeId,
    pub type_info: &'static TypeInfo,
    /// Field name → (byte offset, kind).
    pub field_offsets: HashMap<String, (i32, FieldKind)>,
    pub varlen: VarLenKind,
}

/// Typed handle to a raw GC-managed object pointer in IR.
///
/// This wraps a `dynir::Value` whose type is `Type::GcPtr`, making it harder
/// to accidentally pass plain integer values through rooted allocation paths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObjRef {
    value: Value,
}

impl ObjRef {
    pub fn value(self) -> Value {
        self.value
    }
}

/// Handle to a GC-rooted slot minted by [`DynFunc::with_rooted`]. Use
/// `get(f)` to reload the current (post-forwarding) value from the slot.
#[derive(Clone, Debug)]
pub struct RootedSlot {
    name: String,
}

impl RootedSlot {
    /// Reload the slot's current value. Call this *after* every safepoint
    /// or `gc_alloc` in the rooted-scope body to pick up any forwarding
    /// the GC performed.
    pub fn get(&self, f: &mut DynFunc) -> Value {
        f.get_var(&self.name)
    }

    /// The underlying slot name. Useful when interoperating with
    /// `def_var`/`get_var`/`set_var` directly.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl ObjTypeHandle {
    /// Check if a NaN-boxed value is an object of this type. Returns I8.
    ///
    /// Emits inline: is_ptr(val) → extract payload → load the u16 type_id
    /// from the `Compact` header (at offset 0, zero-padded in the rest of
    /// the word) → compare against this type's known type_id.
    pub fn check(&self, f: &mut DynFunc, val: Value) -> Value {
        let is_obj = f.is_ptr(val);
        let check_bb = f.fb.create_block(&[]);
        let merge_bb = f.fb.create_block(&[Type::I8]);
        let zero = f.fb.iconst(Type::I8, 0);
        f.fb.br_if(is_obj, check_bb, &[], merge_bb, &[zero]);

        f.fb.switch_to_block(check_bb);
        let ptr = f.fb.payload(val);
        // Compact header: u16 type_id at offset 0, 6 bytes of zeroed
        // padding follow — so a full I64 load gives `type_id as u64`.
        let ti = f.fb.load(Type::I64, ptr, 0);
        let expected = f.fb.iconst(Type::I64, self.type_info.type_id as i64);
        let matches = f.fb.icmp(CmpOp::Eq, ti, expected);
        f.fb.jump(merge_bb, &[matches]);

        f.fb.switch_to_block(merge_bb);
        f.fb.block_param(merge_bb, 0)
    }

    /// Extract the raw GC pointer from a NaN-boxed object value.
    /// Caller must ensure the value is a heap object (e.g. via `check`).
    pub fn unwrap(&self, f: &mut DynFunc, val: Value) -> Value {
        f.obj_unwrap(val)
    }

    /// Extract the raw GC pointer from a boxed object value as a typed `ObjRef`.
    pub fn unwrap_ref(&self, f: &mut DynFunc, val: Value) -> ObjRef {
        f.obj_unwrap_ref(val)
    }

    /// Wrap a raw GC pointer into a NaN-boxed object value.
    pub fn wrap(&self, f: &mut DynFunc, ptr: Value) -> Value {
        f.obj_wrap(ptr)
    }

    /// Wrap a typed raw GC pointer into a NaN-boxed object value.
    pub fn wrap_ref(&self, f: &mut DynFunc, ptr: ObjRef) -> Value {
        f.obj_wrap_ref(ptr)
    }

    /// Load a field from a raw GC object pointer. Returns I64.
    pub fn load(&self, f: &mut DynFunc, obj_ptr: Value, field: &str) -> Value {
        let (offset, _kind) = self
            .field_offsets
            .get(field)
            .unwrap_or_else(|| panic!("unknown field '{}' on ObjTypeHandle", field));
        f.fb.load(Type::I64, obj_ptr, *offset)
    }

    /// Load a field from a typed raw GC object pointer. Returns I64.
    pub fn load_ref(&self, f: &mut DynFunc, obj_ptr: ObjRef, field: &str) -> Value {
        self.load(f, obj_ptr.value(), field)
    }

    /// Store a field to a raw GC object pointer.
    pub fn store(&self, f: &mut DynFunc, obj_ptr: Value, field: &str, val: Value) {
        let (offset, _kind) = self
            .field_offsets
            .get(field)
            .unwrap_or_else(|| panic!("unknown field '{}' on ObjTypeHandle", field));
        f.fb.store(val, obj_ptr, *offset);
    }

    /// Store a field to a typed raw GC object pointer.
    pub fn store_ref(&self, f: &mut DynFunc, obj_ptr: ObjRef, field: &str, val: Value) {
        self.store(f, obj_ptr.value(), field, val)
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

    pub fn load_elem_ref(&self, f: &mut DynFunc, obj_ptr: ObjRef, index: Value) -> Value {
        self.load_elem(f, obj_ptr.value(), index)
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

    pub fn store_elem_ref(&self, f: &mut DynFunc, obj_ptr: ObjRef, index: Value, val: Value) {
        self.store_elem(f, obj_ptr.value(), index, val)
    }

    /// Allocate an object of this type via `__gc_alloc__`. Returns raw pointer (`GcPtr`).
    pub fn alloc(&self, f: &mut DynFunc, varlen_len: Value) -> Value {
        f.gc_alloc(self.id, varlen_len)
    }

    pub fn alloc_ref(&self, f: &mut DynFunc, varlen_len: Value) -> ObjRef {
        f.gc_alloc_ref(self.id, varlen_len)
    }

    pub fn alloc_ref_with_roots(
        &self,
        f: &mut DynFunc,
        varlen_len: Value,
        live_roots: &[ObjRef],
    ) -> ObjRef {
        f.gc_alloc_ref_with_roots(self.id, varlen_len, live_roots)
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

impl ObjType {
    /// Byte offset of the named `Value` field. Panics with a clear
    /// message if the field doesn't exist or isn't a Value field.
    /// Use this once at startup to populate a layout cache; not meant
    /// for hot-path lookups (HashMap hashing on every call).
    pub fn value_field_offset_named(&self, name: &str) -> usize {
        match self.field_offsets.get(name) {
            Some((off, FieldKind::Value)) => *off as usize,
            Some((_, k)) => panic!(
                "ObjType {:?}: field {:?} is {:?}, not Value",
                self.name, name, k
            ),
            None => panic!("ObjType {:?}: no field named {:?}", self.name, name),
        }
    }

    /// Byte offset of the named `Raw64` field. Panics with a clear
    /// message if the field doesn't exist or isn't a Raw64 field.
    pub fn raw64_field_offset_named(&self, name: &str) -> usize {
        match self.field_offsets.get(name) {
            Some((off, FieldKind::Raw64)) => *off as usize,
            Some((_, k)) => panic!(
                "ObjType {:?}: field {:?} is {:?}, not Raw64",
                self.name, name, k
            ),
            None => panic!("ObjType {:?}: no field named {:?}", self.name, name),
        }
    }
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

        let mut info = TypeInfo::for_header(header_size).with_fields(value_count);

        if raw_count > 0 {
            info = info.with_raw_bytes(raw_count);
        }

        let total_fixed = value_count + self.raw64_fields.len() as u16;
        match self.varlen {
            VarLenKind::Values => info = info.with_varlen_values(total_fixed),
            VarLenKind::Bytes => info = info.with_varlen_bytes(total_fixed),
            VarLenKind::None => {}
        }

        // Set the type_id to the sequential index
        let id = ObjTypeId(self.module.obj_types.len());
        info = info.with_type_id(id.0 as u16);

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
    /// FuncRefs that the toolkit considers GC-allocators — every direct
    /// `Call` to one of these must be preceded by `Inst::Safepoint`.
    /// Currently this is just `__gc_alloc__` (declared on first
    /// `obj_type` call); frontends can extend this list with their own
    /// allocator externs and pass the combined vec to
    /// `Module::validate_safepoints` at JIT-compile time.
    pub allocator_frefs: Vec<FuncRef>,
}

/// High-level module builder for dynamic languages.
///
/// Wraps `ModuleBuilder` and adds a string pool, slow-path registration,
/// and a convenient `declare_func` that assumes all params/returns are I64
/// (NanBox-encoded values).
pub struct DynModule {
    /// The underlying IR builder. Public so frontends that emit IR
    /// outside the methods on `DynModule` itself can declare/define
    /// functions on the same builder the toolkit registers
    /// `__gc_alloc__` and other auto-externs into. Reach in only when
    /// you need a `ModuleBuilder` API not surfaced through `DynModule`;
    /// otherwise prefer the higher-level wrappers on this type.
    pub module_builder: ModuleBuilder,
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
    /// Externs the toolkit auto-binds at JIT time. Populated by
    /// `register_slow_paths_with_defaults`. Keyed by extern name.
    /// `DynGcRuntime::compile_jit` consults this after the reserved names
    /// (`__gc_alloc__`, `__dynlang_prop_slow__`) and before the
    /// embedder-supplied resolver.
    pub auto_externs: HashMap<String, *const u8>,
}

fn sig(params: &[Type], ret: Option<Type>) -> Signature {
    Signature {
        params: params.to_vec(),
        ret,
    }
}

impl DynModule {
    /// Create a new module. GcConfig is required — use `GcConfig::generational(64 * 1024)`
    /// for zero-overhead no-collection, or `GcConfig::generational(size)`
    /// for automatic garbage collection.
    pub fn new(gc: GcConfig, tags: NanBoxTags) -> Self {
        DynModule {
            module_builder: ModuleBuilder::new(),
            tags,
            gc_config: gc,
            slow: SlowPaths::default(),
            strings: Vec::new(),
            string_map: HashMap::new(),
            func_refs: HashMap::new(),
            obj_types: Vec::new(),
            gc_alloc_extern: None,
            auto_externs: HashMap::new(),
        }
    }

    /// Declare a GC-managed object type. Returns an ObjTypeId for use
    /// with `gc_alloc`, `gc_load_field`, `gc_store_field`.
    pub fn obj_type(&mut self, name: &str) -> ObjTypeBuilder<'_> {
        // Ensure the gc_alloc extern is declared (once)
        if self.gc_alloc_extern.is_none() {
            let sig = Signature {
                params: vec![Type::I64, Type::I64], // type_id, varlen_len
                ret: Some(Type::GcPtr),             // returns raw GC pointer
            };
            self.gc_alloc_extern = Some(self.module_builder.declare_extern("__gc_alloc__", sig));
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
            field_offsets: ty.field_offsets.clone(),
            varlen: ty.varlen,
        }
    }

    /// Get the GC configuration.
    pub fn gc_config(&self) -> &GcConfig {
        &self.gc_config
    }

    /// The GC allocator extern's FuncRef, if any `obj_type` has been
    /// declared on this module (the extern is registered lazily). Used
    /// by primitives that emit GC-allocation IR without going through
    /// `DynFunc` (e.g. `ClosureKit::make`).
    pub fn gc_alloc_extern(&self) -> Option<FuncRef> {
        self.gc_alloc_extern
    }

    /// Get the NanBox tag scheme.
    pub fn tags(&self) -> &NanBoxTags {
        &self.tags
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
        self.module_builder.declare_extern(name, sig)
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

        self.slow.add = Some(
            self.module_builder
                .declare_extern(&format!("{prefix}_add"), i64_2.clone()),
        );
        self.slow.sub = Some(
            self.module_builder
                .declare_extern(&format!("{prefix}_sub"), i64_2.clone()),
        );
        self.slow.mul = Some(
            self.module_builder
                .declare_extern(&format!("{prefix}_mul"), i64_2.clone()),
        );
        self.slow.div = Some(
            self.module_builder
                .declare_extern(&format!("{prefix}_div"), i64_2.clone()),
        );
        self.slow.neg = Some(
            self.module_builder
                .declare_extern(&format!("{prefix}_neg"), i64_1.clone()),
        );
        self.slow.eq = Some(
            self.module_builder
                .declare_extern(&format!("{prefix}_eq"), i64_2.clone()),
        );
        self.slow.lt = Some(
            self.module_builder
                .declare_extern(&format!("{prefix}_lt"), i64_2.clone()),
        );
        self.slow.gt = Some(
            self.module_builder
                .declare_extern(&format!("{prefix}_gt"), i64_2.clone()),
        );
        self.slow.not = Some(
            self.module_builder
                .declare_extern(&format!("{prefix}_not"), i64_1),
        );
    }

    /// Like [`register_slow_paths`](Self::register_slow_paths), but also
    /// pre-binds each declared extern to a default panic stub from
    /// [`crate::slow_paths`]. The runtime auto-binds these at JIT time, so
    /// the embedder gets sane behavior (clear panic on slow-path entry)
    /// without writing 9 stub thunks. Call
    /// [`override_extern`](Self::override_extern) to replace any one with
    /// a real implementation.
    pub fn register_slow_paths_with_defaults(&mut self, prefix: &str) {
        self.register_slow_paths(prefix);
        self.auto_externs
            .insert(format!("{prefix}_add"), slow_paths::panic_add as *const u8);
        self.auto_externs
            .insert(format!("{prefix}_sub"), slow_paths::panic_sub as *const u8);
        self.auto_externs
            .insert(format!("{prefix}_mul"), slow_paths::panic_mul as *const u8);
        self.auto_externs
            .insert(format!("{prefix}_div"), slow_paths::panic_div as *const u8);
        self.auto_externs
            .insert(format!("{prefix}_eq"), slow_paths::panic_eq as *const u8);
        self.auto_externs
            .insert(format!("{prefix}_lt"), slow_paths::panic_lt as *const u8);
        self.auto_externs
            .insert(format!("{prefix}_gt"), slow_paths::panic_gt as *const u8);
        self.auto_externs
            .insert(format!("{prefix}_neg"), slow_paths::panic_neg as *const u8);
        self.auto_externs
            .insert(format!("{prefix}_not"), slow_paths::panic_not as *const u8);
    }

    /// Override (or set) the auto-bound implementation for an extern.
    /// Use this to replace a default panic stub installed by
    /// `register_slow_paths_with_defaults` with a real implementation.
    pub fn override_extern(&mut self, name: &str, ptr: *const u8) {
        self.auto_externs.insert(name.to_string(), ptr);
    }

    /// Register a single slow-path extern for a specific operation.
    pub fn register_slow_path(&mut self, op: DynOp, name: &str) -> FuncRef {
        let i64_2 = sig(&[Type::I64, Type::I64], Some(Type::I64));
        let i64_1 = sig(&[Type::I64], Some(Type::I64));
        let fref = match op {
            DynOp::Add => {
                let f = self.module_builder.declare_extern(name, i64_2);
                self.slow.add = Some(f);
                f
            }
            DynOp::Sub => {
                let f = self.module_builder.declare_extern(name, i64_2);
                self.slow.sub = Some(f);
                f
            }
            DynOp::Mul => {
                let f = self.module_builder.declare_extern(name, i64_2);
                self.slow.mul = Some(f);
                f
            }
            DynOp::Div => {
                let f = self.module_builder.declare_extern(name, i64_2);
                self.slow.div = Some(f);
                f
            }
            DynOp::Neg => {
                let f = self.module_builder.declare_extern(name, i64_1);
                self.slow.neg = Some(f);
                f
            }
            DynOp::Eq => {
                let f = self.module_builder.declare_extern(name, i64_2);
                self.slow.eq = Some(f);
                f
            }
            DynOp::Lt => {
                let f = self.module_builder.declare_extern(name, i64_2);
                self.slow.lt = Some(f);
                f
            }
            DynOp::Gt => {
                let f = self.module_builder.declare_extern(name, i64_2);
                self.slow.gt = Some(f);
                f
            }
            DynOp::Not => {
                let f = self.module_builder.declare_extern(name, i64_1);
                self.slow.not = Some(f);
                f
            }
        };
        fref
    }

    // ── Function declarations ─────────────────────────────────

    /// Declare a language function. All params and return are I64 (NanBox values).
    pub fn declare_func(&mut self, name: &str, arity: usize) -> FuncRef {
        let params = vec![Type::I64; arity];
        let fref = self
            .module_builder
            .declare_func(name, &params, Some(Type::I64));
        self.func_refs.insert(name.to_string(), fref);
        fref
    }

    /// Declare a void language function (no return value).
    pub fn declare_void_func(&mut self, name: &str, arity: usize) -> FuncRef {
        let params = vec![Type::I64; arity];
        let fref = self.module_builder.declare_func(name, &params, None);
        self.func_refs.insert(name.to_string(), fref);
        fref
    }

    /// Look up a previously declared function by name.
    pub fn func_ref(&self, name: &str) -> FuncRef {
        *self
            .func_refs
            .get(name)
            .unwrap_or_else(|| panic!("unknown function: {name}"))
    }

    // ── Function definition ───────────────────────────────────

    /// Start defining a function. Returns a `DynFunc` builder.
    ///
    /// Call `finish_func` when done.
    pub fn start_func(&mut self, fref: FuncRef) -> DynFunc {
        let fb = self.module_builder.define_func(fref);
        DynFunc {
            fb,
            fref,
            tags: self.tags.clone(),
            gc_config: self.gc_config.clone(),
            gc_alloc_extern: self.gc_alloc_extern,
            slow: self.slow.clone(),
            vars: vec![HashMap::new()],
            next_fresh_slot: 0,
        }
    }

    /// Finish defining a function.
    pub fn finish_func(&mut self, func: DynFunc) {
        self.module_builder.finish_func(func.fref, func.fb);
    }

    /// Build the final module.
    pub fn build(self) -> BuiltModule {
        let allocator_frefs: Vec<FuncRef> = self.gc_alloc_extern.into_iter().collect();
        BuiltModule {
            module: self.module_builder.build(),
            strings: self.strings,
            func_refs: self.func_refs,
            allocator_frefs,
        }
    }

    /// Snapshot the currently-defined IR module without consuming `self`.
    ///
    /// The `DynModule` remains live and can be extended with additional
    /// `declare_func` / `start_func` / `finish_func` calls; later snapshots
    /// will include the new functions. Strings, func_refs, obj_types,
    /// and auto_externs are all keyed by stable IDs that survive across
    /// snapshots.
    pub fn snapshot(&self) -> Module {
        self.module_builder.snapshot()
    }

    /// Number of functions declared so far (extern + internal).
    pub fn func_count(&self) -> usize {
        self.module_builder.func_count()
    }

    /// Number of internal functions defined so far.
    pub fn internal_func_count(&self) -> usize {
        self.module_builder.internal_func_count()
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

// ── TypeHint ──────────────────────────────────────────────────────

/// Static knowledge a frontend has about a NanBox value at a use site.
/// Drives [`DynFunc::add`] etc. to pick between the `num_*` fast path
/// (no tag check) and the conservative `dyn_*` form (tag check + slow
/// extern). `Unknown` is always sound.
///
/// Non-exhaustive: future variants (`Bool`, `Object(ObjTypeId)`,
/// `IntInRange { lo, hi }`) can be added without breaking matchers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum TypeHint {
    /// No information; emit the conservative `dyn_*` form.
    #[default]
    Unknown,
    /// Operand is statically known to be a NanBox-encoded number (float).
    Number,
    /// Operand is statically known to be a NanBox-encoded boolean.
    Bool,
}

impl TypeHint {
    pub fn is_number(&self) -> bool {
        matches!(self, TypeHint::Number)
    }
    pub fn is_bool(&self) -> bool {
        matches!(self, TypeHint::Bool)
    }
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
    /// Monotonic counter for `fresh_slot_name`. Per-DynFunc; never resets,
    /// so even nested inlining never mints a colliding slot name.
    next_fresh_slot: u32,
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
    ///
    /// Dynlang values are NaN-boxed I64s that may carry heap pointers,
    /// so the slot is created as a GC root. The GC's PtrPolicy filters
    /// non-pointer payloads, and under a moving collector the slot is
    /// updated in place so reloads see the forwarded pointer.
    pub fn def_var(&mut self, name: &str, init: Value) {
        let slot = self.fb.create_stack_slot(8, true);
        let addr = self.fb.stack_addr(slot);
        self.fb.store(init, addr, 0);
        self.vars.last_mut().unwrap().insert(name.to_string(), slot);
    }

    /// Read a variable's current value.
    pub fn get_var(&mut self, name: &str) -> Value {
        let slot = self.find_var(name);
        let addr = self.fb.stack_addr(slot);
        self.fb.load(Type::I64, addr, 0)
    }

    // ── GC-rooted temporaries ────────────────────────────────

    /// Mint a unique slot name. The counter is per-`DynFunc` and monotonic,
    /// so nested inlining can mint as many fresh names as it wants without
    /// colliding with anything in the caller's scope.
    pub fn fresh_slot_name(&mut self) -> String {
        let id = self.next_fresh_slot;
        self.next_fresh_slot += 1;
        format!("__rooted_{}__", id)
    }

    /// Pin `values` into uniquely-named GC-rooted slots so they survive any
    /// safepoint or `gc_alloc` inside `body`. The closure receives
    /// `RootedSlot` handles whose `get(f)` reloads the (possibly forwarded)
    /// current value from the slot — call it *after* each allocation site
    /// in the body, before reading the value again.
    ///
    /// The caller is still responsible for emitting `safepoint` and
    /// `gc_alloc` inside the closure; this helper only owns the rooting.
    ///
    /// # Example
    /// ```ignore
    /// let new_obj = f.with_rooted(&[v1, v2], |f, slots| {
    ///     f.fb.safepoint(&[]);
    ///     let raw = f.gc_alloc(ty, len);
    ///     // Reload after alloc — pointers may have been forwarded.
    ///     let v1_now = slots[0].get(f);
    ///     let v2_now = slots[1].get(f);
    ///     f.fb.store(v1_now, raw, off1);
    ///     f.fb.store(v2_now, raw, off2);
    ///     raw
    /// });
    /// ```
    pub fn with_rooted<R>(
        &mut self,
        values: &[Value],
        body: impl FnOnce(&mut Self, &[RootedSlot]) -> R,
    ) -> R {
        let slots: Vec<RootedSlot> = values
            .iter()
            .map(|v| {
                let name = self.fresh_slot_name();
                self.def_var(&name, *v);
                RootedSlot { name }
            })
            .collect();
        body(self, &slots)
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
        self.fb
            .iconst(Type::I64, nanbox_encode(self.tags.nil, 0) as i64)
    }

    /// A boolean constant.
    pub fn bool_val(&mut self, b: bool) -> Value {
        self.fb.iconst(
            Type::I64,
            nanbox_encode(self.tags.bool_tag, b as u64) as i64,
        )
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
    /// The result is typed as `GcPtr`, so it can participate in safepoints.
    /// Caller must ensure the value is a heap object (e.g. checked with `is_ptr`).
    pub fn obj_unwrap(&mut self, v: Value) -> Value {
        let payload = self.fb.payload(v);
        self.fb.bitcast(payload, Type::GcPtr)
    }

    /// Wrap a raw GC pointer into a NaN-boxed object value.
    pub fn obj_wrap(&mut self, ptr: Value) -> Value {
        let payload = self.fb.bitcast(ptr, Type::I64);
        self.fb.make_tagged(self.tags.ptr, payload)
    }

    pub fn obj_unwrap_ref(&mut self, v: Value) -> ObjRef {
        ObjRef {
            value: self.obj_unwrap(v),
        }
    }

    pub fn obj_wrap_ref(&mut self, ptr: ObjRef) -> Value {
        self.obj_wrap(ptr.value())
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
        self.fb
            .br_if(is_true, then_bb, then_args, else_bb, else_args);
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
        self.fb
            .br_if(truthy, then_bb, then_args, else_bb, else_args);
    }

    // ── Unbox / Box ───────────────────────────────────────────

    /// Bitcast a NanBox float (I64) → F64. Caller must ensure it's a float.
    pub fn unbox_number(&mut self, v: Value) -> Value {
        self.fb.bitcast(v, Type::F64)
    }

    /// Decode a NanBox-encoded float and truncate to a signed I64. Used
    /// for array indices and other "this float is really an int" sites.
    /// Caller must ensure the value is a number; out-of-range floats
    /// follow the IR's `float_to_int` semantics (target-dependent).
    pub fn nanbox_to_int(&mut self, v: Value) -> Value {
        let as_f64 = self.unbox_number(v);
        self.fb.float_to_int(as_f64)
    }

    /// Bit-equal two NanBox values: true iff their underlying I64 bits
    /// are equal. Correct for nil / boolean / pointer-identity / canonical
    /// integer comparisons. **Not IEEE equality** — `NaN == NaN` here, so
    /// non-canonical floats with the same bits compare equal regardless
    /// of whether IEEE would agree. Returns a NanBox bool.
    pub fn bit_eq(&mut self, a: Value, b: Value) -> Value {
        let raw = self.fb.icmp(CmpOp::Eq, a, b);
        let t = self.bool_val(true);
        let fal = self.bool_val(false);
        self.fb.select(raw, t, fal)
    }

    /// Logical not of a NanBox value, using the configured truthiness
    /// policy ([`is_falsey`](Self::is_falsey)). Returns a NanBox bool.
    pub fn bool_not(&mut self, v: Value) -> Value {
        let falsey = self.is_falsey(v);
        let t = self.bool_val(true);
        let fal = self.bool_val(false);
        self.fb.select(falsey, t, fal)
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

    // ── Type-hinted binop dispatch ────────────────────────────
    //
    // Pick between `num_*` (skip the tag-check branch) and `dyn_*`
    // (full fast/slow dispatch) based on caller-supplied type hints.
    // For comparisons that don't have a `dyn_*` primitive (`le`, `ge`),
    // synthesize via `!gt` / `!lt`.

    /// Hinted addition. Emits `num_add` when both operands are `Number`,
    /// else `dyn_add`.
    pub fn add(&mut self, l: Value, lh: TypeHint, r: Value, rh: TypeHint) -> Value {
        if lh.is_number() && rh.is_number() {
            self.num_add(l, r)
        } else {
            self.dyn_add(l, r)
        }
    }
    pub fn sub(&mut self, l: Value, lh: TypeHint, r: Value, rh: TypeHint) -> Value {
        if lh.is_number() && rh.is_number() {
            self.num_sub(l, r)
        } else {
            self.dyn_sub(l, r)
        }
    }
    pub fn mul(&mut self, l: Value, lh: TypeHint, r: Value, rh: TypeHint) -> Value {
        if lh.is_number() && rh.is_number() {
            self.num_mul(l, r)
        } else {
            self.dyn_mul(l, r)
        }
    }
    pub fn div(&mut self, l: Value, lh: TypeHint, r: Value, rh: TypeHint) -> Value {
        if lh.is_number() && rh.is_number() {
            self.num_div(l, r)
        } else {
            self.dyn_div(l, r)
        }
    }
    pub fn lt(&mut self, l: Value, lh: TypeHint, r: Value, rh: TypeHint) -> Value {
        if lh.is_number() && rh.is_number() {
            self.num_lt(l, r)
        } else {
            self.dyn_lt(l, r)
        }
    }
    pub fn gt(&mut self, l: Value, lh: TypeHint, r: Value, rh: TypeHint) -> Value {
        if lh.is_number() && rh.is_number() {
            self.num_gt(l, r)
        } else {
            self.dyn_gt(l, r)
        }
    }
    /// Less-than-or-equal. No `dyn_le` primitive exists — synthesize as
    /// `!gt` when operands aren't both numbers.
    pub fn le(&mut self, l: Value, lh: TypeHint, r: Value, rh: TypeHint) -> Value {
        if lh.is_number() && rh.is_number() {
            self.num_le(l, r)
        } else {
            let gt = self.dyn_gt(l, r);
            self.bool_not(gt)
        }
    }
    /// Greater-than-or-equal. No `dyn_ge` primitive exists — synthesize as
    /// `!lt` when operands aren't both numbers.
    pub fn ge(&mut self, l: Value, lh: TypeHint, r: Value, rh: TypeHint) -> Value {
        if lh.is_number() && rh.is_number() {
            self.num_ge(l, r)
        } else {
            let lt = self.dyn_lt(l, r);
            self.bool_not(lt)
        }
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
        let slow_fn = slow_fn.unwrap_or_else(|| panic!("dyn_{:?}: no slow path registered", op));
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
    /// elements (0 for fixed-size objects). Returns the raw object pointer as `GcPtr`.
    ///
    /// Use [`obj_wrap`](Self::obj_wrap) if you need the boxed NanBox form.
    pub fn gc_alloc(&mut self, type_id: ObjTypeId, varlen_len: Value) -> Value {
        self.gc_alloc_with_roots(type_id, varlen_len, &[])
    }

    /// Allocate a GC object and return a typed object reference.
    pub fn gc_alloc_ref(&mut self, type_id: ObjTypeId, varlen_len: Value) -> ObjRef {
        ObjRef {
            value: self.gc_alloc(type_id, varlen_len),
        }
    }

    /// Allocate a GC object, first emitting a safepoint for the provided live roots.
    ///
    /// When using a moving collector, pass every raw `GcPtr` object reference that must
    /// remain valid across the allocation. This keeps the high-level dynlang builder on
    /// the same GC contract as raw dynir.
    pub fn gc_alloc_with_roots(
        &mut self,
        type_id: ObjTypeId,
        varlen_len: Value,
        live_roots: &[Value],
    ) -> Value {
        let alloc_fn = self
            .gc_alloc_extern
            .expect("gc_alloc: no object types declared on this module");
        let type_id_val = self.fb.iconst(Type::I64, type_id.0 as i64);
        // Always emit a safepoint immediately before the alloc call, so
        // `Module::validate_safepoints` accepts this site. (Frontends
        // that have caller-side roots already emitted a safepoint before
        // entering this method; the validator only requires *some*
        // safepoint immediately preceding the call, and an empty live
        // set is harmless — the precise dynlower stack-map collector
        // sweeps all spilled values regardless of `live`.)
        self.fb.safepoint(live_roots);
        self.fb.call(alloc_fn, &[type_id_val, varlen_len]).unwrap()
    }

    pub fn gc_alloc_ref_with_roots(
        &mut self,
        type_id: ObjTypeId,
        varlen_len: Value,
        live_roots: &[ObjRef],
    ) -> ObjRef {
        let roots: Vec<Value> = live_roots.iter().map(|root| root.value()).collect();
        ObjRef {
            value: self.gc_alloc_with_roots(type_id, varlen_len, &roots),
        }
    }

    /// Load a value field from a GC object. Offset is computed at build
    /// time from the TypeInfo. `obj` is a raw pointer (I64).
    pub fn gc_load_field(&mut self, obj: Value, type_info: &ObjType, field: &str) -> Value {
        let (offset, _kind) = type_info
            .field_offsets
            .get(field)
            .unwrap_or_else(|| panic!("unknown field '{}' on type '{}'", field, type_info.name));
        self.fb.load(Type::I64, obj, *offset)
    }

    /// Store a value field to a GC object. Offset is computed at build time.
    pub fn gc_store_field(&mut self, obj: Value, type_info: &ObjType, field: &str, val: Value) {
        let (offset, _kind) = type_info
            .field_offsets
            .get(field)
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
/// Stores `(guard_key, cached_value)` in separate stack slots.
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
    guard_slot: StackSlot,
    value_slot: StackSlot,
}

impl DynFunc {
    /// Allocate a new inline cache for this call site.
    /// Each call to `inline_cache()` creates a separate cache.
    pub fn inline_cache(&mut self) -> InlineCache {
        // The guard is a raw stable key. The cached value may be a heap
        // NanBox, so it must be a GC root independently of the guard.
        let guard_slot = self.fb.create_stack_slot(8, false);
        let value_slot = self.fb.create_stack_slot(8, true);
        let guard_addr = self.fb.stack_addr(guard_slot);
        let value_addr = self.fb.stack_addr(value_slot);
        let zero = self.fb.iconst(Type::I64, 0);
        self.fb.store(zero, guard_addr, 0);
        self.fb.store(zero, value_addr, 0);
        InlineCache {
            guard_slot,
            value_slot,
        }
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
        let guard_addr = f.fb.stack_addr(self.guard_slot);
        let cached_guard = f.fb.load(Type::I64, guard_addr, 0);
        let hit = f.fb.icmp(CmpOp::Eq, cached_guard, guard_key);
        let hit_bb = f.fb.create_block(&[]);
        f.fb.br_if(hit, hit_bb, &[], miss_bb, &[]);

        // Cache hit: return cached value
        f.fb.switch_to_block(hit_bb);
        let value_addr = f.fb.stack_addr(self.value_slot);
        let cached_val = f.fb.load(Type::I64, value_addr, 0);
        f.fb.jump(merge_bb, &[cached_val]);

        // Cache miss: call slow path, update cache
        f.fb.switch_to_block(miss_bb);
        let computed = slow_path(f);
        // Update cache: store guard + value
        let guard_addr = f.fb.stack_addr(self.guard_slot);
        let value_addr = f.fb.stack_addr(self.value_slot);
        f.fb.store(guard_key, guard_addr, 0);
        f.fb.store(computed, value_addr, 0);
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
    use dynalloc::LowBitPtrPolicy;
    use dynalloc::{alloc_obj, PtrPolicy, SemiSpace};
    use dynir::dynexec::ContinuationTypes;
    use dynir::gc_runtime::{GcInterpCtx, GcInterpPolicy};
    use dynir::interp::{ExternCallResult, InterpResult, ModuleInterpreter};
    use dynobj::{Compact, TypeInfo};
    use dynvalue::NanBox;
    type TestRoots = GcInterpCtx<Compact, LowBitPtrPolicy<3>>;

    struct TestNanBoxPolicy;

    impl PtrPolicy for TestNanBoxPolicy {
        fn try_decode_ptr(bits: u64) -> Option<*mut u8> {
            if bits == 0 {
                None
            } else if bits & 0b111 == 0 {
                Some(bits as *mut u8)
            } else {
                None
            }
        }

        fn encode_ptr(ptr: *mut u8) -> u64 {
            debug_assert_eq!((ptr as u64) & 0b111, 0);
            ptr as u64
        }
    }

    fn make_gc_ctx(
        obj_types: &[ObjType],
        heap_size: usize,
    ) -> GcInterpCtx<Compact, TestNanBoxPolicy> {
        let mut type_table: Vec<TypeInfo> = obj_types.iter().map(|obj| *obj.type_info).collect();
        let cont_types = ContinuationTypes::register_into::<Compact>(&mut type_table);
        let heap = SemiSpace::new::<Compact>(heap_size);
        GcInterpCtx::new(heap, type_table, cont_types)
    }

    fn run(module: &Module, entry: FuncRef, args: &[u64]) -> u64 {
        let roots: TestRoots = GcInterpCtx::new_unallocating();
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
        let mut dm = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
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
        let mut dm = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
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
        let mut dm = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
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
        let mut dm = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
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
        let roots: TestRoots = GcInterpCtx::new_unallocating();
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
        let mut dm = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
        dm.register_slow_paths("rt");
        let main = dm.declare_func("main", 0);

        let mut f = dm.start_func(main);
        let a = f.nil(); // not a number → slow path
        let b = f.number(1.0);
        let sum = f.dyn_add(a, b);
        f.fb.ret(sum);
        dm.finish_func(f);

        let built = dm.build();
        let roots: TestRoots = GcInterpCtx::new_unallocating();
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
        let mut dm = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
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
        let mut dm = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
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
        let mut dm = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
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
        let mut dm = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());

        // Declare an object type with two value fields and a varlen values section
        let pair_ty = dm
            .obj_type("Pair")
            .field("car", FieldKind::Value)
            .field("cdr", FieldKind::Value)
            .build();

        let array_ty = dm.obj_type("Array").varlen_values().build();

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
        let mut dm = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());

        let upval_ty = dm
            .obj_type("Upvalue")
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
        let gc = crate::gc::DynGcRuntime::new(
            &GcConfig::generational(64 * 1024),
            &NanBoxTags::default(),
            &dm.obj_types,
        );

        let built = dm.build();

        let roots: TestRoots = GcInterpCtx::new_unallocating();
        let mut interp = ModuleInterpreter::<NanBox, _>::new(&built.module, &roots);
        interp.bind_by_name(crate::gc::GC_ALLOC_EXTERN, gc.interp_gc_alloc());

        match interp.run(main, &[]) {
            Ok(InterpResult::Value(v)) => assert_eq!(v, 42),
            other => panic!("unexpected: {:?}", other),
        }
    }

    #[test]
    fn test_gc_alloc_with_roots_survives_moving_gc() {
        let mut dm = DynModule::new(GcConfig::generational(8192), NanBoxTags::default());

        let pair_ty = dm.obj_type("Pair").field("value", FieldKind::Value).build();

        let main = dm.declare_func("main", 0);

        let mut f = dm.start_func(main);
        let zero = f.fb.iconst(Type::I64, 0);
        let obj1 = f.gc_alloc(pair_ty, zero);
        let val = f.fb.iconst(Type::I64, 42);
        {
            let ty = dm.get_obj_type(pair_ty);
            f.gc_store_field(obj1, ty, "value", val);
        }

        // This allocation triggers a safepoint first, so obj1 must survive a moving GC.
        let _obj2 = f.gc_alloc_with_roots(pair_ty, zero, &[obj1]);

        let loaded = {
            let ty = dm.get_obj_type(pair_ty);
            f.gc_load_field(obj1, ty, "value")
        };
        f.fb.ret(loaded);
        dm.finish_func(f);

        let ctx = make_gc_ctx(&dm.obj_types, 256);
        ctx.set_gc_policy(GcInterpPolicy::EVERY_ALLOC);
        let built = dm.build();

        let mut interp = ModuleInterpreter::<NanBox, _>::new(&built.module, &ctx);
        interp.bind_by_name("__gc_alloc__", |args| {
            let type_id = args[0] as usize;
            let varlen_len = args[1] as usize;
            let info = &ctx.type_table()[type_id];
            let ptr = unsafe { alloc_obj::<Compact>(&ctx, info, varlen_len) };
            assert!(!ptr.is_null(), "gc alloc failed");
            ExternCallResult::Value(Some(ptr as u64))
        });

        match interp.run(main, &[]) {
            Ok(InterpResult::Value(v)) => assert_eq!(v, 42),
            other => panic!("unexpected: {:?}", other),
        }
        assert!(
            ctx.collection_count() >= 1,
            "expected moving GC to run during rooted allocation path"
        );
    }

    #[test]
    fn test_boxed_object_round_trip_after_rooted_gc_alloc() {
        let mut dm = DynModule::new(GcConfig::generational(8192), NanBoxTags::default());

        let pair_ty = dm.obj_type("Pair").field("value", FieldKind::Value).build();

        let main = dm.declare_func("main", 0);

        let mut f = dm.start_func(main);
        let zero = f.fb.iconst(Type::I64, 0);
        let obj = f.gc_alloc(pair_ty, zero);
        let boxed = f.obj_wrap(obj);
        let raw = f.obj_unwrap(boxed);
        let val = f.fb.iconst(Type::I64, 7);
        {
            let ty = dm.get_obj_type(pair_ty);
            f.gc_store_field(raw, ty, "value", val);
        }

        let _moved = f.gc_alloc_with_roots(pair_ty, zero, &[raw]);

        let loaded = {
            let ty = dm.get_obj_type(pair_ty);
            f.gc_load_field(raw, ty, "value")
        };
        f.fb.ret(loaded);
        dm.finish_func(f);

        let ctx = make_gc_ctx(&dm.obj_types, 256);
        ctx.set_gc_policy(GcInterpPolicy::EVERY_ALLOC);
        let built = dm.build();

        let mut interp = ModuleInterpreter::<NanBox, _>::new(&built.module, &ctx);
        interp.bind_by_name("__gc_alloc__", |args| {
            let type_id = args[0] as usize;
            let varlen_len = args[1] as usize;
            let info = &ctx.type_table()[type_id];
            let ptr = unsafe { alloc_obj::<Compact>(&ctx, info, varlen_len) };
            assert!(!ptr.is_null(), "gc alloc failed");
            ExternCallResult::Value(Some(ptr as u64))
        });

        match interp.run(main, &[]) {
            Ok(InterpResult::Value(v)) => assert_eq!(v, 7),
            other => panic!("unexpected: {:?}", other),
        }
        assert!(
            ctx.collection_count() >= 1,
            "expected moving GC to run during boxed round-trip path"
        );
    }

    #[test]
    fn test_obj_ref_api_survives_rooted_allocation() {
        let mut dm = DynModule::new(GcConfig::generational(8192), NanBoxTags::default());

        let pair_ty = dm.obj_type("Pair").field("value", FieldKind::Value).build();
        let pair = dm.obj_handle(pair_ty);

        let main = dm.declare_func("main", 0);

        let mut f = dm.start_func(main);
        let zero = f.fb.iconst(Type::I64, 0);
        let obj1 = pair.alloc_ref(&mut f, zero);
        let forty_two = f.fb.iconst(Type::I64, 42);
        pair.store_ref(&mut f, obj1, "value", forty_two);

        let _obj2 = pair.alloc_ref_with_roots(&mut f, zero, &[obj1]);
        let loaded = pair.load_ref(&mut f, obj1, "value");
        f.fb.ret(loaded);
        dm.finish_func(f);

        let ctx = make_gc_ctx(&dm.obj_types, 256);
        ctx.set_gc_policy(GcInterpPolicy::EVERY_ALLOC);
        let built = dm.build();

        let mut interp = ModuleInterpreter::<NanBox, _>::new(&built.module, &ctx);
        interp.bind_by_name("__gc_alloc__", |args| {
            let type_id = args[0] as usize;
            let varlen_len = args[1] as usize;
            let info = &ctx.type_table()[type_id];
            let ptr = unsafe { alloc_obj::<Compact>(&ctx, info, varlen_len) };
            assert!(!ptr.is_null(), "gc alloc failed");
            ExternCallResult::Value(Some(ptr as u64))
        });

        match interp.run(main, &[]) {
            Ok(InterpResult::Value(v)) => assert_eq!(v, 42),
            other => panic!("unexpected: {:?}", other),
        }
        assert!(
            ctx.collection_count() >= 1,
            "expected moving GC to run during ObjRef allocation path"
        );
    }

    #[test]
    fn test_dyn_lt_fast_path() {
        let mut dm = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
        dm.register_slow_paths("rt");
        let main = dm.declare_func("main", 0);

        let mut f = dm.start_func(main);
        let a = f.number(1.0);
        let b = f.number(2.0);
        let result = f.dyn_lt(a, b); // 1.0 < 2.0 = true
        f.fb.ret(result);
        dm.finish_func(f);

        let built = dm.build();
        let roots: TestRoots = GcInterpCtx::new_unallocating();
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

    #[test]
    fn fresh_slot_name_is_unique_and_with_rooted_reloads() {
        let mut dm = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
        dm.register_slow_paths("rt");
        let main = dm.declare_func("main", 0);

        let mut f = dm.start_func(main);
        // Counter mints distinct names monotonically.
        let n1 = f.fresh_slot_name();
        let n2 = f.fresh_slot_name();
        assert_ne!(n1, n2);

        // with_rooted closure receives slots whose `get` reloads from the
        // backing stack slot. Reading should produce the same value bits
        // (no GC ran between def and read).
        let v1 = f.number(1.5);
        let v2 = f.number(2.5);
        let result = f.with_rooted(&[v1, v2], |f, slots| {
            assert_eq!(slots.len(), 2);
            assert_ne!(slots[0].name(), slots[1].name());
            let r1 = slots[0].get(f);
            let r2 = slots[1].get(f);
            f.dyn_add(r1, r2)
        });
        f.fb.ret(result);
        dm.finish_func(f);

        let built = dm.build();
        let roots: TestRoots = GcInterpCtx::new_unallocating();
        let mut interp = ModuleInterpreter::<NanBox, _>::new(&built.module, &roots);
        interp.bind_by_name("rt_add", |_| {
            panic!("slow path should not be called");
        });
        match interp.run(main, &[]) {
            Ok(InterpResult::Value(v)) => assert_eq!(f64::from_bits(v), 4.0),
            other => panic!("unexpected: {:?}", other),
        }
    }

    // ── DynFunc helpers (doc 10) ─────────────────────────────────

    /// Returns a closure that takes a `DynFunc` argument constructor for
    /// `bool_not(input)` and runs it through the interp; result is the
    /// raw NanBox `u64` returned.
    fn run_unary<B>(build: B) -> u64
    where
        B: FnOnce(&mut DynFunc) -> Value,
    {
        let mut dm = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
        let main = dm.declare_func("main", 0);
        let mut f = dm.start_func(main);
        let v = build(&mut f);
        f.fb.ret(v);
        dm.finish_func(f);
        let built = dm.build();
        run(&built.module, main, &[])
    }

    #[test]
    fn bool_not_inverts_falsey_and_truthy() {
        let true_v = run_unary(|f| {
            let v = f.bool_val(true);
            f.bool_not(v)
        });
        assert_eq!(true_v, nanbox_encode(1, 0)); // false

        let false_v = run_unary(|f| {
            let v = f.bool_val(false);
            f.bool_not(v)
        });
        assert_eq!(false_v, nanbox_encode(1, 1)); // true

        let nil_v = run_unary(|f| {
            let v = f.nil();
            f.bool_not(v)
        });
        assert_eq!(nil_v, nanbox_encode(1, 1)); // !nil = true (nil is falsey)

        let num_v = run_unary(|f| {
            let v = f.number(0.0);
            f.bool_not(v)
        });
        // 0.0 is *truthy* in dynlang (only nil and false are falsey),
        // so !0.0 = false.
        assert_eq!(num_v, nanbox_encode(1, 0));
    }

    #[test]
    fn bit_eq_returns_nanbox_bool() {
        // Equal floats → true.
        let eq_floats = run_unary(|f| {
            let a = f.number(3.0);
            let b = f.number(3.0);
            f.bit_eq(a, b)
        });
        assert_eq!(eq_floats, nanbox_encode(1, 1));

        // Unequal floats → false.
        let neq_floats = run_unary(|f| {
            let a = f.number(3.0);
            let b = f.number(4.0);
            f.bit_eq(a, b)
        });
        assert_eq!(neq_floats, nanbox_encode(1, 0));

        // nil == nil → true (bit-equal).
        let eq_nil = run_unary(|f| {
            let a = f.nil();
            let b = f.nil();
            f.bit_eq(a, b)
        });
        assert_eq!(eq_nil, nanbox_encode(1, 1));

        // nil vs false: different bit patterns (different tag), so false.
        let nil_vs_false = run_unary(|f| {
            let a = f.nil();
            let b = f.bool_val(false);
            f.bit_eq(a, b)
        });
        assert_eq!(nil_vs_false, nanbox_encode(1, 0));
    }

    #[test]
    fn nanbox_to_int_truncates() {
        // 3.7 → 3 (truncate, not round).
        let r = run_unary(|f| {
            let v = f.number(3.7);
            f.nanbox_to_int(v)
        });
        assert_eq!(r as i64, 3);

        // Negative.
        let r = run_unary(|f| {
            let v = f.number(-2.9);
            f.nanbox_to_int(v)
        });
        assert_eq!(r as i64, -2);

        // Integer-valued float round-trips.
        let r = run_unary(|f| {
            let v = f.number(42.0);
            f.nanbox_to_int(v)
        });
        assert_eq!(r as i64, 42);
    }

    // ── Typed binops (doc 08) ────────────────────────────────────

    /// Build + run a binop with the given type hints, asserting the
    /// rt_<op> slow path is never called. Returns the raw NanBox bits.
    fn run_binop<B>(slow_name: &str, build: B) -> u64
    where
        B: FnOnce(&mut DynFunc, Value, Value) -> Value,
    {
        let mut dm = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
        dm.register_slow_paths("rt");
        let main = dm.declare_func("main", 0);
        let mut f = dm.start_func(main);
        let a = f.number(10.0);
        let b = f.number(3.0);
        let r = build(&mut f, a, b);
        f.fb.ret(r);
        dm.finish_func(f);
        let built = dm.build();
        let roots: TestRoots = GcInterpCtx::new_unallocating();
        let mut interp = ModuleInterpreter::<NanBox, _>::new(&built.module, &roots);
        let slow_owned = slow_name.to_string();
        interp.bind_by_name(slow_name, move |_| {
            panic!("slow path `{}` should not be called", slow_owned);
        });
        match interp.run(main, &[]) {
            Ok(InterpResult::Value(v)) => v,
            other => panic!("unexpected: {:?}", other),
        }
    }

    #[test]
    fn add_number_number_takes_fast_path() {
        // Number + Number → num_add; the rt_add panic stub must not fire.
        let r = run_binop("rt_add", |f, a, b| {
            f.add(a, TypeHint::Number, b, TypeHint::Number)
        });
        assert_eq!(as_float(r), 13.0);
    }

    #[test]
    fn sub_mul_div_number_number_take_fast_path() {
        let r = run_binop("rt_sub", |f, a, b| {
            f.sub(a, TypeHint::Number, b, TypeHint::Number)
        });
        assert_eq!(as_float(r), 7.0);

        let r = run_binop("rt_mul", |f, a, b| {
            f.mul(a, TypeHint::Number, b, TypeHint::Number)
        });
        assert_eq!(as_float(r), 30.0);

        let r = run_binop("rt_div", |f, a, b| {
            f.div(a, TypeHint::Number, b, TypeHint::Number)
        });
        assert!((as_float(r) - 10.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn cmp_number_number_take_fast_path() {
        // 10 < 3 → false; 10 > 3 → true; 10 <= 3 → false; 10 >= 3 → true.
        let lt = run_binop("rt_lt", |f, a, b| {
            f.lt(a, TypeHint::Number, b, TypeHint::Number)
        });
        assert_eq!(lt, nanbox_encode(1, 0));

        let gt = run_binop("rt_gt", |f, a, b| {
            f.gt(a, TypeHint::Number, b, TypeHint::Number)
        });
        assert_eq!(gt, nanbox_encode(1, 1));

        let le = run_binop("rt_lt", |f, a, b| {
            f.le(a, TypeHint::Number, b, TypeHint::Number)
        });
        assert_eq!(le, nanbox_encode(1, 0));

        let ge = run_binop("rt_gt", |f, a, b| {
            f.ge(a, TypeHint::Number, b, TypeHint::Number)
        });
        assert_eq!(ge, nanbox_encode(1, 1));
    }

    #[test]
    fn add_unknown_unknown_takes_dyn_path_with_inline_fast() {
        // (Unknown, Unknown) routes through dyn_add. dyn_add still has
        // an inline float fast path that fires when both operands are
        // floats at runtime, so the rt_add stub doesn't need to be
        // bound — but the IR shape is the conservative one.
        let mut dm = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
        dm.register_slow_paths("rt");
        let main = dm.declare_func("main", 0);
        let mut f = dm.start_func(main);
        let a = f.number(10.0);
        let b = f.number(3.0);
        // No type hints → conservative dyn_add.
        let r = f.add(a, TypeHint::Unknown, b, TypeHint::Unknown);
        f.fb.ret(r);
        dm.finish_func(f);
        let built = dm.build();
        let roots: TestRoots = GcInterpCtx::new_unallocating();
        let mut interp = ModuleInterpreter::<NanBox, _>::new(&built.module, &roots);
        // Inline fast path fires at runtime since both args are floats —
        // slow stub still shouldn't be hit.
        interp.bind_by_name("rt_add", |_| {
            panic!("rt_add hit even though operands are float at runtime");
        });
        match interp.run(main, &[]) {
            Ok(InterpResult::Value(v)) => assert_eq!(as_float(v), 13.0),
            other => panic!("unexpected: {:?}", other),
        }
    }

    #[test]
    fn type_hint_default_is_unknown() {
        // Sanity check the Default impl — easy to break with a
        // non_exhaustive enum.
        assert_eq!(TypeHint::default(), TypeHint::Unknown);
        assert!(!TypeHint::Unknown.is_number());
        assert!(TypeHint::Number.is_number());
        assert!(!TypeHint::Bool.is_number());
        assert!(TypeHint::Bool.is_bool());
    }
}
