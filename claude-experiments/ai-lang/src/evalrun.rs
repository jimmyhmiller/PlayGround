//! `evalrun` — JIT-and-invoke a def from a content-addressed codebase and
//! render its **real value of any type** back as structured JSON.
//!
//! This is the executable counterpart to the structural-edit server's
//! typecheck-only path: it lets a caller (the `ai-lang-edit` `eval` op)
//! actually *run* a def and observe the real value, closing the gap between
//! "edits and typechecks" and "confirms behavior".
//!
//! The JIT-and-invoke skeleton is a faithful extraction of `cmd_run` in
//! `bin/ai-lang.rs`: walk the transitive closure of the root hash, reconstruct
//! a [`ResolvedModule`], build a `CompiledModule`, spin up a `Jit`, install the
//! runtime / knowledge-base / at-binding, and call the entry symbol.
//!
//! ## Honesty contract
//!
//! The raw `i64` a def returns is EITHER a plain scalar (Int / Float-bits /
//! Bool) OR a heap pointer — and which one is decided **solely by the def's
//! declared return [`Type`]**. We never reinterpret a pointer as an int or
//! vice versa. The renderer walks the declared return type (and recursively
//! field / variant / element types), reading the heap through the runtime's
//! shape metadata for *layout* (offsets, tag offset, varlen sections) and the
//! codebase `Def`s for *names* (struct field names, enum variant names). If a
//! type genuinely can't be rendered we return a structured [`EvalError::Unsupported`]
//! naming the type — never a wrong value.
//!
//! ## Supported return types
//!
//! - `Int`    → JSON number (the raw i64).
//! - `Float`  → JSON number via `f64::from_bits` (the kernel carries Float as
//!              the i64 bit-pattern of the f64).
//! - `Bool`   → JSON bool (0/1).
//! - `String` → JSON string (heap varlen-bytes: header + 8-byte count + bytes).
//! - `TypeRef(h)` to a struct → JSON object `{field_name: value, ...}`.
//! - `TypeRef(h)` to an enum   → `{VariantName: payload}` (or `"VariantName"`
//!              for a nullary variant), reading the tag at the variant shape's
//!              `tag_offset`.
//! - `Apply(Array, [T])`       → JSON array (varlen-Values pointer slots).
//! - A list-shaped enum (an enum with a nullary "nil" variant and a single
//!   recursive "cons" variant) renders naturally as the nested enum it is;
//!   we do NOT flatten — nesting is honest and unambiguous.
//!
//! ## Supported argument types
//!
//! Arguments are now *general*, not Int-only. Each JSON arg is converted to
//! wire bytes **according to the corresponding parameter's declared [`Type`]**,
//! then [`crate::wire::decode_value`] allocates a real heap object of that type
//! in the eval runtime's heap and hands back a pointer. We never coerce or
//! fabricate: a JSON shape that doesn't match the declared type yields a
//! structured error, never garbage. Supported:
//!
//! - `Int`    ← JSON number (or integral float).
//! - `Float`  ← JSON number, carried as the i64 bit-pattern (the kernel ABI).
//! - `Bool`   ← JSON `true`/`false` (Int 0/1).
//! - struct (`TypeRef` → `Def::Struct`) ← JSON object; each field is built
//!   recursively by its declared field type, in declared field order.
//! - enum  (`TypeRef` → `Def::Enum`)    ← JSON object `{"Variant": payload}` for
//!   a variant with a payload, or the bare string `"Variant"` for a nullary
//!   variant. This also covers a cons-cell **list** (it's just a recursive
//!   enum: `{"Cons": {...}}` / `"Nil"`).
//! - `Apply(TypeRef, [..])` (a generic struct/enum instantiation) ← handled by
//!   the same struct/enum path; the head's declared type drives the layout.
//!
//! - **`String` / `Bytes`** ← JSON string. These are runtime-reserved varlen
//!   shapes with no wire value kind, so they do NOT go through the wire format.
//!   Instead we allocate the heap object directly in the eval runtime's heap via
//!   [`crate::runtime::ai_str_new`] (header + count + memcpy'd UTF-8 bytes) and
//!   pass the pointer as the arg. `Bytes` shares the String shape, same call.
//! - **`Array<T>`** ← JSON array. Also a runtime-reserved varlen-Values shape;
//!   built directly via [`crate::runtime::ai_array_new`] + per-element
//!   [`crate::runtime::ai_array_set`]. Each element is built recursively *by the
//!   element type* (so struct/enum/String/nested-Array elements all work). Array
//!   slots are uniform boxed pointers: a SCALAR element (`Int`/`Float`/`Bool`)
//!   is boxed via [`crate::runtime::ai_gc_box_int`] — a type-agnostic 8-byte box
//!   — and re-interpreted per the element type on the way out, so `Array<Int>`,
//!   `Array<Float>`, and `Array<Bool>` ALL round-trip. Pointer-shaped elements
//!   are stored as-is. (A `List<T>` can ALSO be passed — it's a cons-cell enum.)
//! - **structs / enums** are built field-by-field directly in the heap, so ANY
//!   field/payload type composes: a struct with a `String` field, an enum
//!   carrying an `Array`, a nested struct, etc. all work.
//!
//! ### Argument types that remain Unsupported (and why)
//!
//! - **`fn(..)` / closures** — a closure can't be constructed from JSON data.
//! - **bare `TypeVar` / `SelfRef`** — no concrete shape to build against.
//!
//! ### Arity
//!
//! The explicit fn-pointer call arms cover **0..=12** arguments (raised from
//! the old 0..=4). Beyond 12, a structured [`EvalError::Arity`] is returned.
//!
//! ## Cost note
//!
//! Each [`eval`] builds a fresh `inkwell::Context` + `CompiledModule` + `Jit`.
//! LLVM contexts are not trivially reusable across calls, so this is the
//! correct (if not the cheapest) thing to do. An `eval` is a deliberate
//! "run it and tell me what it does" action, not a hot loop.

use std::collections::{HashMap, HashSet};

use crate::ast::{Def, Type};
use crate::codebase::Codebase;
use crate::codegen::{CompiledModule, Jit, ShapeMeta, def_symbol, init_native_target};
use crate::gc::{Full, ObjHeader};
use crate::hash::Hash;
use crate::jsonl::Json;
use crate::knowledge::{self, KnowledgeBase};
use crate::net::{
    build_at_runtime_binding, clear_at_conn_cache, clear_current_at_binding,
    clear_current_knowledge_base, clear_current_runtime, install_current_at_binding,
    install_current_knowledge_base, install_current_runtime,
};
use crate::parser::parse_module;
use crate::resolve::{AtBinding, ExternSig, ResolvedDef, ResolvedModule, resolve_module};
use crate::handle::Alloc;
use crate::runtime::{Runtime, Thread, ai_str_new};
use crate::stdlib::SOURCE as STDLIB;
use inkwell::context::Context;

/// A structured failure from [`eval`]. Each variant maps to a stable `kind`
/// string the edit-server surfaces verbatim, so an agent can branch on it.
#[derive(Debug)]
pub enum EvalError {
    /// The def isn't a function (it's a struct/enum), so it can't be invoked.
    NotCallable(String),
    /// The number of supplied args doesn't match the def's parameter count.
    BadParams(String),
    /// Something on the *arg* side or the *return-render* side isn't supported
    /// yet (a non-Int parameter, or a return type we honestly can't render).
    /// We refuse to fabricate a value.
    Unsupported(String),
    /// More args than the explicit fn-pointer arities we handle (0..=12).
    Arity(String),
    /// The JIT couldn't produce the entry symbol with the expected signature.
    Jit(String),
    /// A codebase / load error while walking the closure.
    Codebase(String),
}

impl EvalError {
    /// The stable, machine-branchable `kind` for this error.
    pub fn kind(&self) -> &'static str {
        match self {
            EvalError::NotCallable(_) => "NotCallable",
            EvalError::BadParams(_) => "BadParams",
            EvalError::Unsupported(_) => "Unsupported",
            EvalError::Arity(_) => "Arity",
            EvalError::Jit(_) => "Jit",
            EvalError::Codebase(_) => "Codebase",
        }
    }

    pub fn message(&self) -> &str {
        match self {
            EvalError::NotCallable(m)
            | EvalError::BadParams(m)
            | EvalError::Unsupported(m)
            | EvalError::Arity(m)
            | EvalError::Jit(m)
            | EvalError::Codebase(m) => m,
        }
    }
}

/// The result of a successful [`eval`]: the rendered value plus a readable
/// rendering of the declared return type.
#[derive(Debug)]
pub struct Evaluated {
    pub value: Json,
    pub type_str: String,
}

/// `true` iff `t` is exactly the `Int` builtin.
/// A scalar that, when stored in a boxed slot (an `Array<T>` element or a
/// generic `TypeVar` field), is carried as an 8-byte value inside a `BoxedInt`
/// heap box. `Int`, `Float` (the f64 bit-pattern), and `Bool` (0/1) are all
/// 8-byte scalars and share the SAME box/unbox machinery (`ai_gc_box_int` /
/// `ai_gc_unbox_int`) — the box is type-agnostic, only the interpretation of
/// the unboxed i64 differs. This is why an `Array<Float>` or `Array<Bool>` is
/// just as buildable/renderable as an `Array<Int>`: same bytes, different lens.
fn is_boxed_scalar(t: &Type) -> bool {
    matches!(t, Type::Builtin(b) if b == "Int" || b == "Float" || b == "Bool")
}

/// Interpret an already-unboxed 8-byte scalar `bits` as JSON per its type.
fn render_scalar_bits(t: &Type, bits: i64) -> Option<Json> {
    match t {
        Type::Builtin(b) => match b.as_str() {
            "Int" => Some(Json::Int(bits)),
            "Float" => Some(Json::Float(f64::from_bits(bits as u64))),
            "Bool" => Some(Json::Bool(bits != 0)),
            _ => None,
        },
        _ => None,
    }
}

/// Re-resolve the stdlib to recover its extern signatures. Externs aren't
/// content-addressed, so the codebase doesn't store them; the runtime supplies
/// the implementations, and the resolver/codegen need the signatures at call
/// sites. This mirrors `stdlib_externs()` in `bin/ai-lang.rs`.
fn stdlib_externs() -> HashMap<String, ExternSig> {
    let m = parse_module(STDLIB).expect("parse stdlib");
    let r = resolve_module(&m).expect("resolve stdlib");
    r.externs
}

/// Rebuild the resolver's `AtBinding` from named `Result`/`Failure`/`Node`
/// defs in the codebase, or `None` if the program doesn't use `at()`. Mirrors
/// `at_binding_from_codebase()` in `bin/ai-lang.rs`.
fn at_binding_from_codebase(cb: &Codebase) -> Option<AtBinding> {
    let result_hash = cb.get_name("Result")?;
    let failure_hash = cb.get_name("Failure")?;
    let node_hash = cb.get_name("Node")?;
    let result_def = cb.load_def(&result_hash).ok()?;
    let failure_def = cb.load_def(&failure_hash).ok()?;
    let result_variants = match result_def {
        Def::Enum { variants, .. } => variants,
        _ => return None,
    };
    let failure_variants = match failure_def {
        Def::Enum { variants, .. } => variants,
        _ => return None,
    };
    let find = |vs: &[(String, Option<Type>)], n: &str| -> Option<u32> {
        vs.iter().position(|(name, _)| name == n).map(|i| i as u32)
    };
    // DecodeError is optional (only `decode::<T>` needs it).
    let (decode_error_hash, decode_tm_idx, decode_mf_idx) = match cb.get_name("DecodeError") {
        Some(h) => match cb.load_def(&h) {
            Ok(Def::Enum { variants, .. }) => (
                Some(h),
                find(&variants, "TypeMismatch").unwrap_or(0),
                find(&variants, "Malformed").unwrap_or(0),
            ),
            _ => (None, 0, 0),
        },
        None => (None, 0, 0),
    };
    Some(AtBinding {
        result_hash,
        failure_hash,
        node_hash,
        ok_variant_index: find(&result_variants, "Ok")?,
        err_variant_index: find(&result_variants, "Err")?,
        unreachable_variant_index: find(&failure_variants, "Unreachable")?,
        crashed_variant_index: find(&failure_variants, "Crashed")?,
        code_missing_variant_index: find(&failure_variants, "CodeMissing")?,
        cancelled_variant_index: find(&failure_variants, "Cancelled")?,
        decode_error_hash,
        decode_type_mismatch_index: decode_tm_idx,
        decode_malformed_index: decode_mf_idx,
    })
}

/// Reconstruct a [`ResolvedModule`] by walking the transitive closure of
/// `root_hash` (exactly the `cmd_run` walk). Hashes whose `load_def` returns
/// `MissingDef` are inline lambdas (stored inline, not as `.def` files) and are
/// skipped. Each def is given a name from the current names table, or a
/// synthetic `def_<8hex>` so codegen can still dispatch.
fn build_resolved_module(cb: &Codebase, root_hash: Hash) -> Result<ResolvedModule, EvalError> {
    let mut wanted: Vec<Hash> = Vec::new();
    let mut seen: HashSet<Hash> = HashSet::new();
    let mut frontier: Vec<Hash> = vec![root_hash];
    seen.insert(root_hash);
    while let Some(h) = frontier.pop() {
        let def = match cb.load_def(&h) {
            Ok(d) => d,
            Err(crate::codebase::CodebaseError::MissingDef(_)) => continue,
            Err(e) => return Err(EvalError::Codebase(format!("load_def {}: {:?}", h, e))),
        };
        wanted.push(h);
        let mut deps: Vec<Hash> = Vec::new();
        let mut local_seen: HashSet<Hash> = HashSet::new();
        knowledge::walk_def_deps(&def, &mut deps, &mut local_seen);
        for d in deps {
            if seen.insert(d) {
                frontier.push(d);
            }
        }
    }

    // hash → name reverse map (shortest-then-lexicographically-smallest, the
    // same deterministic tiebreak `cmd_run` uses).
    let mut name_for: HashMap<Hash, String> = HashMap::new();
    for (n, h) in cb.names().iter() {
        match name_for.get(h) {
            None => {
                name_for.insert(*h, n.clone());
            }
            Some(existing) if (n.len(), n.as_str()) < (existing.len(), existing.as_str()) => {
                name_for.insert(*h, n.clone());
            }
            _ => {}
        }
    }

    wanted.sort_by(|a, b| a.to_hex().cmp(&b.to_hex()));
    let mut defs: Vec<ResolvedDef> = Vec::with_capacity(wanted.len());
    for h in wanted {
        let def = cb
            .load_def(&h)
            .map_err(|e| EvalError::Codebase(format!("load_def {}: {:?}", h, e)))?;
        let name = name_for
            .get(&h)
            .cloned()
            .unwrap_or_else(|| format!("def_{}", &h.to_hex()[..8]));
        defs.push(ResolvedDef { name, hash: h, def });
    }

    Ok(ResolvedModule {
        defs,
        at_binding: at_binding_from_codebase(cb),
        externs: stdlib_externs(),
    })
}

// =============================================================================
// Value rendering
//
// The renderer is driven by the *declared* type. For each type it knows
// whether the i64 in hand is a scalar or a heap pointer, so it never has to
// guess from raw bytes. Layout comes from `rt.shape_meta` (keyed by shape
// hash); names come from the codebase `Def`s.
// =============================================================================

/// A returned scalar slot is either a raw i64 or a heap pointer. The caller's
/// declared type tells us which interpretation is honest.
struct RenderCtx<'a> {
    rt: &'a Runtime,
    cb: &'a Codebase,
}

impl<'a> RenderCtx<'a> {
    /// Render the raw i64 `bits` (a top-level return value, or a slot already
    /// loaded from the heap) as JSON, interpreting it per `ty`.
    ///
    /// # Safety
    /// If `ty` is a pointer type, `bits` must be a live heap pointer in
    /// `self.rt`'s heap (it is, for a genuine JIT return / field load).
    unsafe fn render(&self, ty: &Type, bits: i64) -> Result<Json, EvalError> {
        match ty {
            Type::Builtin(b) => match b.as_str() {
                "Int" => Ok(Json::Int(bits)),
                "Float" => Ok(Json::Float(f64::from_bits(bits as u64))),
                "Bool" => Ok(Json::Bool(bits != 0)),
                "String" => Ok(Json::Str(unsafe { self.read_string(bits as *const u8) })),
                "Bytes" => Ok(Json::Str(unsafe { self.read_string(bits as *const u8) })),
                other => Err(EvalError::Unsupported(format!(
                    "cannot render builtin type `{}`",
                    other
                ))),
            },
            Type::TypeRef(h) => unsafe { self.render_ref(*h, bits) },
            Type::Apply(head, args) => unsafe { self.render_apply(head, args, bits) },
            Type::TypeVar(_) => Err(EvalError::Unsupported(
                "result has a generic (TypeVar) type; its runtime value is a boxed pointer with \
                 no static element type to render against. Provide a concrete return type."
                    .to_string(),
            )),
            Type::SelfRef(_) => Err(EvalError::Unsupported(
                "result type is an unresolved SelfRef (recursive type placeholder)".to_string(),
            )),
            Type::FnType { .. } => Err(EvalError::Unsupported(
                "result is a function/closure value; rendering closures is not supported"
                    .to_string(),
            )),
        }
    }

    /// Render `Apply(head, args)`. The only structural case we special-case is
    /// `Array<T>` (a varlen-Values heap object of boxed elements). For a
    /// user generic like `List<Int>` the head is a `TypeRef`; we render the
    /// concrete struct/enum and recurse, substituting the type arguments for
    /// `TypeVar(i)` slots encountered inside fields/variants.
    unsafe fn render_apply(
        &self,
        head: &Type,
        args: &[Type],
        bits: i64,
    ) -> Result<Json, EvalError> {
        if let Type::Builtin(b) = head {
            if b == "Array" {
                let elem_ty = args.first().ok_or_else(|| {
                    EvalError::Unsupported("Array<...> with no element type".to_string())
                })?;
                return unsafe { self.render_array(elem_ty, bits as *const u8) };
            }
        }
        if let Type::TypeRef(h) = head {
            // A concrete instantiation of a user struct/enum. Render the named
            // type, substituting `args` for the type parameters.
            return unsafe { self.render_ref_with_subst(*h, bits, args) };
        }
        Err(EvalError::Unsupported(format!(
            "cannot render applied type with head `{}`",
            render_ret_type(self.cb, head)
        )))
    }

    /// Render a `TypeRef(h)` value (no type-argument substitution).
    unsafe fn render_ref(&self, h: Hash, bits: i64) -> Result<Json, EvalError> {
        unsafe { self.render_ref_with_subst(h, bits, &[]) }
    }

    /// Render a `TypeRef(h)` value, substituting `type_args` for any `TypeVar`
    /// references that appear in the struct's field types or the enum's
    /// variant payload types.
    unsafe fn render_ref_with_subst(
        &self,
        h: Hash,
        bits: i64,
        type_args: &[Type],
    ) -> Result<Json, EvalError> {
        let def = self
            .cb
            .load_def(&h)
            .map_err(|e| EvalError::Codebase(format!("load_def {}: {:?}", h, e)))?;
        match def {
            Def::Struct { fields, .. } => {
                unsafe { self.render_struct(h, bits as *const u8, &fields, type_args) }
            }
            Def::Enum { variants, .. } => {
                unsafe { self.render_enum(h, bits as *const u8, &variants, type_args) }
            }
            Def::Fn { .. } => Err(EvalError::Unsupported(format!(
                "TypeRef {} resolves to a function, not a struct/enum",
                h
            ))),
            Def::State { .. } => Err(EvalError::Unsupported(format!(
                "TypeRef {} resolves to a state binding, not a struct/enum",
                h
            ))),
        }
    }

    /// Render a struct: `{field_name: rendered_value, ...}`. Field offsets and
    /// pointer-ness come from the runtime `Struct` shape (keyed by the struct's
    /// own hash); field *names and types* come from the `Def`.
    unsafe fn render_struct(
        &self,
        struct_hash: Hash,
        ptr: *const u8,
        fields: &[(String, Type)],
        type_args: &[Type],
    ) -> Result<Json, EvalError> {
        let meta = self.rt.shape_meta.get(&struct_hash).ok_or_else(|| {
            EvalError::Unsupported(format!("no runtime shape for struct {}", struct_hash))
        })?;
        let field_metas = match meta {
            ShapeMeta::Struct { fields, .. } => fields.clone(),
            _ => {
                return Err(EvalError::Unsupported(format!(
                    "shape for {} is not a Struct shape",
                    struct_hash
                )));
            }
        };
        if field_metas.len() != fields.len() {
            return Err(EvalError::Unsupported(format!(
                "struct {}: shape has {} fields but def has {}",
                struct_hash,
                field_metas.len(),
                fields.len()
            )));
        }
        let mut out: Vec<(String, Json)> = Vec::with_capacity(fields.len());
        for (i, (name, fty)) in fields.iter().enumerate() {
            let fm = &field_metas[i];
            let slot = unsafe { ptr.add(fm.offset as usize) };
            let bits = unsafe { *(slot as *const i64) };
            let concrete = subst(fty, type_args);
            let rendered = unsafe { self.render_slot(&concrete, fm.is_pointer, bits)? };
            out.push((name.clone(), rendered));
        }
        Ok(Json::obj(out))
    }

    /// Render an enum: read the tag, pick the variant, render its payload (if
    /// any) as `{VariantName: payload}`, or the bare `"VariantName"` string for
    /// a nullary variant.
    unsafe fn render_enum(
        &self,
        enum_hash: Hash,
        ptr: *const u8,
        variants: &[(String, Option<Type>)],
        type_args: &[Type],
    ) -> Result<Json, EvalError> {
        // The tag lives at the variant shape's `tag_offset`. Every variant of
        // a given enum shares the same tag offset, so we find any variant shape
        // for this enum to learn it, read the tag, then re-select by index.
        let tag_offset = self
            .rt
            .shape_meta
            .values()
            .find_map(|m| match m {
                ShapeMeta::EnumVariant {
                    enum_ref,
                    tag_offset,
                    ..
                } if *enum_ref == enum_hash => Some(*tag_offset),
                _ => None,
            })
            .ok_or_else(|| {
                EvalError::Unsupported(format!("no runtime variant shape for enum {}", enum_hash))
            })?;
        let tag = unsafe { *(ptr.add(tag_offset as usize) as *const u32) } as usize;
        let (vname, vpayload) = variants.get(tag).ok_or_else(|| {
            EvalError::Unsupported(format!(
                "enum {}: tag {} out of range ({} variants)",
                enum_hash,
                tag,
                variants.len()
            ))
        })?;
        match vpayload {
            None => Ok(Json::Str(vname.clone())),
            Some(pty) => {
                // Find this variant's shape to get the payload FieldMeta.
                let payload_meta = self
                    .rt
                    .shape_meta
                    .values()
                    .find_map(|m| match m {
                        ShapeMeta::EnumVariant {
                            enum_ref,
                            variant_index,
                            payload,
                            ..
                        } if *enum_ref == enum_hash && *variant_index as usize == tag => {
                            Some(payload.clone())
                        }
                        _ => None,
                    })
                    .flatten()
                    .ok_or_else(|| {
                        EvalError::Unsupported(format!(
                            "enum {} variant {} has a payload type but no payload shape",
                            enum_hash, vname
                        ))
                    })?;
                let slot = unsafe { ptr.add(payload_meta.offset as usize) };
                let bits = unsafe { *(slot as *const i64) };
                let concrete = subst(pty, type_args);
                let rendered =
                    unsafe { self.render_slot(&concrete, payload_meta.is_pointer, bits)? };
                Ok(Json::obj([(vname.clone(), rendered)]))
            }
        }
    }

    /// Render a value loaded from a struct field / enum payload slot. We have
    /// both the declared type AND the shape's `is_pointer` flag; cross-check
    /// them so a layout/type disagreement surfaces as an error rather than a
    /// mis-read. (A `TypeVar`-typed slot is pointer-shaped and holds a
    /// `BoxedInt`; if the substituted concrete type is `Int` we unbox it.)
    unsafe fn render_slot(
        &self,
        ty: &Type,
        is_pointer: bool,
        bits: i64,
    ) -> Result<Json, EvalError> {
        // A concrete scalar (`Int`/`Float`/`Bool`) stored in a boxed slot is a
        // *boxed* pointer at the ABI level: the shape says pointer, the type
        // says scalar. All three share the same 8-byte box; unbox the bits and
        // interpret them per the declared type. (Float = f64 bit-pattern,
        // Bool = 0/1.)
        if is_boxed_scalar(ty) && is_pointer {
            let unboxed = unsafe { crate::runtime::ai_gc_unbox_int(bits as *const u8) };
            return render_scalar_bits(ty, unboxed)
                .ok_or_else(|| EvalError::Unsupported("unreachable: boxed scalar".to_string()));
        }
        unsafe { self.render(ty, bits) }
    }

    /// Render an `Array<T>`: a varlen-Values heap object of boxed pointer
    /// slots. Each element is rendered per `elem_ty`; an `Int` element is a
    /// `BoxedInt` (unbox), other element types are direct heap pointers.
    unsafe fn render_array(&self, elem_ty: &Type, ptr: *const u8) -> Result<Json, EvalError> {
        if ptr.is_null() {
            return Ok(Json::Array(Vec::new()));
        }
        let len = unsafe { crate::runtime::ai_array_len(ptr) };
        let mut out: Vec<Json> = Vec::with_capacity(len.max(0) as usize);
        for i in 0..len {
            let elem_ptr = unsafe { crate::runtime::ai_array_get(ptr, i) };
            // Array slots always hold pointers; an Int element is boxed.
            let rendered = unsafe { self.render_slot(elem_ty, true, elem_ptr as i64)? };
            out.push(rendered);
        }
        Ok(Json::Array(out))
    }

    /// Read a heap `String`/`Bytes` (header + 8-byte count + raw bytes) into a
    /// Rust `String` (lossy UTF-8 — honest about non-UTF-8 input rather than
    /// erroring).
    unsafe fn read_string(&self, ptr: *const u8) -> String {
        if ptr.is_null() {
            return String::new();
        }
        unsafe {
            let count_off = <Full as ObjHeader>::SIZE;
            let len = *(ptr.add(count_off) as *const u64) as usize;
            let payload = ptr.add(count_off + 8);
            let bytes = core::slice::from_raw_parts(payload, len);
            String::from_utf8_lossy(bytes).into_owned()
        }
    }
}

/// Substitute `type_args` for `TypeVar(i)` references in `ty` (one level; the
/// substituted args may themselves be concrete or further `Apply`s). Used so a
/// `List<Int>`'s `Cons(ListCell<T>)` renders `T` as `Int`.
fn subst(ty: &Type, type_args: &[Type]) -> Type {
    match ty {
        Type::TypeVar(i) => type_args
            .get(*i as usize)
            .cloned()
            .unwrap_or_else(|| ty.clone()),
        Type::Apply(head, args) => Type::Apply(
            Box::new(subst(head, type_args)),
            args.iter().map(|a| subst(a, type_args)).collect(),
        ),
        // Builtin / TypeRef / SelfRef / FnType: no top-level TypeVar to swap.
        // (We don't descend into FnType param/ret — closures aren't rendered.)
        other => other.clone(),
    }
}

/// Render a return [`Type`] to a short readable string for the response.
fn render_ret_type(cb: &Codebase, ty: &Type) -> String {
    match ty {
        Type::Builtin(b) => b.clone(),
        Type::TypeRef(h) => name_of(cb, h).unwrap_or_else(|| format!("#{}", &h.to_hex()[..8])),
        Type::TypeVar(i) => format!("T{}", i),
        Type::SelfRef(i) => format!("Self{}", i),
        Type::Apply(head, args) => {
            let inner: Vec<String> = args.iter().map(|a| render_ret_type(cb, a)).collect();
            format!("{}<{}>", render_ret_type(cb, head), inner.join(", "))
        }
        Type::FnType { params, ret } => {
            let ps: Vec<String> = params.iter().map(|p| render_ret_type(cb, p)).collect();
            format!("fn({}) -> {}", ps.join(", "), render_ret_type(cb, ret))
        }
    }
}

/// Best-effort reverse name lookup for a type hash.
fn name_of(cb: &Codebase, h: &Hash) -> Option<String> {
    cb.names()
        .iter()
        .find(|(_, hh)| *hh == h)
        .map(|(n, _)| n.clone())
}

// =============================================================================
// Argument building
//
// Each JSON argument is built DIRECTLY as a real heap object (or raw scalar) of
// its declared parameter type in the eval runtime's heap — see `build_arg`.
// We deliberately do NOT route through the wire format (`wire::encode_value` /
// `decode_value`): the wire format has no value kind for `String`/`Bytes`/
// `Array`, so a struct or enum *containing* one of those couldn't be built that
// way. Direct construction (`ai_str_new`, `ai_array_new`, `ai_gc_alloc_closure`
// + per-field recursion) composes uniformly: any field/element/payload type is
// built by its own declared type, so `Array<Float>`, a struct with a `String`
// field, an enum carrying an `Array`, etc. all work. A shape/type mismatch is a
// structured error; we never emit garbage.
// =============================================================================

/// A short human name for a JSON value's shape, for error messages.
fn json_shape(j: &Json) -> &'static str {
    match j {
        Json::Null => "null",
        Json::Bool(_) => "a boolean",
        Json::Int(_) => "an integer",
        Json::Float(_) => "a float",
        Json::Str(_) => "a string",
        Json::Array(_) => "an array",
        Json::Object(_) => "an object",
    }
}

/// Build a single call argument as a raw `i64` (scalar or heap pointer) in the
/// eval runtime's heap, validating it against the declared parameter `ty`.
///
/// This is the single entry point the call loop uses. Every type is built
/// DIRECTLY (no wire format), recursively by its declared type:
///
/// - scalars (`Int`/`Float`/`Bool`) return a raw 8-byte value (Float = f64
///   bit-pattern, Bool = 0/1).
/// - `String` / `Bytes` / `Array<T>` are runtime-reserved varlen shapes,
///   allocated via the runtime constructors (`ai_str_new`, `ai_array_new` +
///   `ai_array_set`) and returned as heap pointers.
/// - user struct / enum (and generic instantiations) are allocated via
///   `ai_gc_alloc_closure` and each field/payload built by recursing through
///   `build_arg` on its declared type (see `build_named_arg`), so any
///   composition (a String field, an Array payload, ...) works.
///
/// All are REAL heap objects / honest scalars of the correct shape — never
/// fabricated values. A JSON shape that doesn't match the declared type is a
/// structured error.
///
/// # Safety
/// `rt` must be installed (so its `Thread` has `string_ti`/`array_ti`/
/// `boxed_int_ti` initialised) and freshly allocated objects must stay alive
/// across the subsequent call (no GC between build and call — the eval-heap
/// assumption documented at the call site).
unsafe fn build_arg(
    cb: &Codebase,
    rt: &Runtime,
    ty: &Type,
    json: &Json,
) -> Result<i64, EvalError> {
    let thread = rt.thread_ptr();
    match ty {
        // --- Scalars: a raw 8-byte value (Int / Float-bits / Bool). ---
        Type::Builtin(b) if b == "Int" => {
            let n = json.as_i64().ok_or_else(|| {
                EvalError::BadParams(format!(
                    "expected an integer for an `Int` parameter, got {}",
                    json_shape(json)
                ))
            })?;
            Ok(n)
        }
        Type::Builtin(b) if b == "Float" => {
            let f = match json {
                Json::Float(f) => *f,
                Json::Int(i) => *i as f64,
                _ => {
                    return Err(EvalError::BadParams(format!(
                        "expected a number for a `Float` parameter, got {}",
                        json_shape(json)
                    )));
                }
            };
            Ok(f.to_bits() as i64)
        }
        Type::Builtin(b) if b == "Bool" => {
            let v = json.as_bool().ok_or_else(|| {
                EvalError::BadParams(format!(
                    "expected a boolean for a `Bool` parameter, got {}",
                    json_shape(json)
                ))
            })?;
            Ok(if v { 1 } else { 0 })
        }
        // --- Runtime-reserved varlen shapes: built directly in the heap. ---
        Type::Builtin(b) if b == "String" || b == "Bytes" => {
            let s = match json {
                Json::Str(s) => s,
                _ => {
                    return Err(EvalError::BadParams(format!(
                        "expected a string for a `{}` parameter, got {}",
                        b,
                        json_shape(json)
                    )));
                }
            };
            let bytes = s.as_bytes();
            let ptr = unsafe { ai_str_new(thread, bytes.as_ptr(), bytes.len() as i64) };
            Ok(ptr as i64)
        }
        Type::Apply(head, args) if matches!(head.as_ref(), Type::Builtin(b) if b == "Array") => {
            let elem_ty = args.first().ok_or_else(|| {
                EvalError::Unsupported("`Array<...>` argument with no element type".to_string())
            })?;
            unsafe { build_array_arg(cb, rt, elem_ty, json) }
        }
        // --- User struct / enum (incl. generic instantiations): built field-
        //     by-field directly in the heap so ANY field type composes
        //     (a struct with a String field, an enum carrying an Array, etc.).
        //     This is why we do NOT route these through the wire format — the
        //     wire format has no String/Array value kind, so a struct
        //     *containing* one couldn't be built that way.
        Type::TypeRef(h) => unsafe { build_named_arg(cb, rt, *h, &[], json) },
        Type::Apply(head, type_args) => {
            if let Type::TypeRef(h) = head.as_ref() {
                unsafe { build_named_arg(cb, rt, *h, type_args, json) }
            } else {
                Err(EvalError::Unsupported(format!(
                    "cannot build an argument of applied type with head `{}`",
                    render_ret_type(cb, head)
                )))
            }
        }
        Type::TypeVar(_) => Err(EvalError::Unsupported(
            "parameter has a generic (TypeVar) type with no concrete shape to build against"
                .to_string(),
        )),
        Type::SelfRef(_) => Err(EvalError::Unsupported(
            "parameter has an unresolved SelfRef (recursive type placeholder)".to_string(),
        )),
        Type::FnType { .. } => Err(EvalError::Unsupported(
            "function/closure arguments can't be built from JSON data".to_string(),
        )),
        Type::Builtin(other) => Err(EvalError::Unsupported(format!(
            "cannot build an argument of builtin type `{}`",
            other
        ))),
    }
}

/// Build a user `struct`/`enum` argument (named by content hash `h`, with
/// `type_args` substituted for its type parameters) directly in `rt`'s heap.
///
/// We mirror `wire::decode_struct`/`decode_enum`'s allocation (`ai_gc_alloc_closure`
/// + the shape's field offsets) but build each field/payload by recursing
/// through [`build_arg`] on its DECLARED type — so a `String`/`Array`/nested
/// struct field is constructed by its own direct-heap path. Scalar fields are
/// stored raw at their slot; pointer fields store the built pointer; a boxed
/// scalar in a generic (`TypeVar`) field is boxed via `ai_gc_box_int`.
unsafe fn build_named_arg(
    cb: &Codebase,
    rt: &Runtime,
    h: Hash,
    type_args: &[Type],
    json: &Json,
) -> Result<i64, EvalError> {
    let thread = rt.thread_ptr();
    let def = cb
        .load_def(&h)
        .map_err(|e| EvalError::Codebase(format!("load_def {}: {:?}", h, e)))?;
    match def {
        Def::Struct { fields, .. } => {
            let obj = match json {
                Json::Object(m) => m,
                _ => {
                    return Err(EvalError::BadParams(format!(
                        "expected a JSON object for struct `{}`, got {}",
                        name_of(cb, &h).unwrap_or_else(|| h.to_hex()[..8].to_string()),
                        json_shape(json)
                    )));
                }
            };
            for key in obj.keys() {
                if !fields.iter().any(|(fname, _)| fname == key) {
                    return Err(EvalError::BadParams(format!(
                        "struct `{}` has no field `{}`",
                        name_of(cb, &h).unwrap_or_else(|| h.to_hex()[..8].to_string()),
                        key
                    )));
                }
            }
            // Pull the runtime shape (field offsets + pointer-ness).
            let ti = rt.type_info_for(&h).ok_or_else(|| {
                EvalError::Unsupported(format!("no runtime shape for struct {}", h))
            })?;
            let metas = match rt.shape_meta.get(&h) {
                Some(ShapeMeta::Struct { fields: fm, .. }) => fm.clone(),
                _ => {
                    return Err(EvalError::Unsupported(format!(
                        "runtime shape for {} is not a struct",
                        h
                    )));
                }
            };
            if metas.len() != fields.len() {
                return Err(EvalError::Unsupported(format!(
                    "struct `{}` shape/def field-count disagree ({} vs {})",
                    h,
                    metas.len(),
                    fields.len()
                )));
            }
            let ptr = unsafe { crate::runtime::ai_gc_alloc_closure(thread, ti) };
            // Root the struct across the field builds (each allocates and may
            // relocate it); re-read the live pointer before each store. The
            // scope auto-resets on every exit path (incl. the `?`s below).
            let dyna = unsafe { &*(*thread).dyna_thread };
            let scope = dyna.scratch_scope();
            let ptr_slot = scope.push(ptr);
            for (idx, (fname, fty)) in fields.iter().enumerate() {
                let fval = obj.get(fname).ok_or_else(|| {
                    EvalError::BadParams(format!(
                        "struct `{}` is missing field `{}`",
                        name_of(cb, &h).unwrap_or_else(|| h.to_hex()[..8].to_string()),
                        fname
                    ))
                })?;
                let concrete = subst(fty, type_args);
                let cur = scope.get(ptr_slot);
                unsafe { store_built_field(cb, rt, cur, &metas[idx], &concrete, fval)? };
            }
            Ok(scope.get(ptr_slot) as i64)
        }
        Def::Enum { variants, .. } => {
            // `{"Variant": payload}` (one key) or bare `"Variant"` (nullary).
            let (vname, payload_json): (&str, Option<&Json>) = match json {
                Json::Str(s) => (s.as_str(), None),
                Json::Object(m) if m.len() == 1 => {
                    let (k, v) = m.iter().next().unwrap();
                    (k.as_str(), Some(v))
                }
                _ => {
                    return Err(EvalError::BadParams(format!(
                        "expected `\"Variant\"` or `{{\"Variant\": payload}}` for enum `{}`, got {}",
                        name_of(cb, &h).unwrap_or_else(|| h.to_hex()[..8].to_string()),
                        json_shape(json)
                    )));
                }
            };
            let variant_index = variants
                .iter()
                .position(|(n, _)| n == vname)
                .ok_or_else(|| {
                    EvalError::BadParams(format!(
                        "enum `{}` has no variant `{}`",
                        name_of(cb, &h).unwrap_or_else(|| h.to_hex()[..8].to_string()),
                        vname
                    ))
                })?;
            let payload_ty = &variants[variant_index].1;
            let variant_hash = crate::codegen::derive_variant_hash(&h, vname);
            let ti = rt.type_info_for(&variant_hash).ok_or_else(|| {
                EvalError::Unsupported(format!("no runtime shape for variant {}::{}", h, vname))
            })?;
            let (tag_offset, payload_meta) = match rt.shape_meta.get(&variant_hash) {
                Some(ShapeMeta::EnumVariant {
                    tag_offset,
                    payload,
                    ..
                }) => (*tag_offset, payload.clone()),
                _ => {
                    return Err(EvalError::Unsupported(format!(
                        "runtime shape for variant {}::{} is not an EnumVariant",
                        h, vname
                    )));
                }
            };
            let ptr = unsafe { crate::runtime::ai_gc_alloc_closure(thread, ti) };
            unsafe {
                *(ptr.add(tag_offset as usize) as *mut u32) = variant_index as u32;
            }
            match (payload_ty, payload_meta, payload_json) {
                (None, None, None) => Ok(ptr as i64),
                (Some(pty), Some(meta), Some(pj)) => {
                    let concrete = subst(pty, type_args);
                    // Root the variant object across the payload build (which
                    // allocates and may relocate it); re-read before returning.
                    let dyna = unsafe { &*(*thread).dyna_thread };
                    let scope = dyna.scratch_scope();
                    let ptr_slot = scope.push(ptr);
                    let cur = scope.get(ptr_slot);
                    unsafe { store_built_field(cb, rt, cur, &meta, &concrete, pj)? };
                    Ok(scope.get(ptr_slot) as i64)
                }
                (Some(_), _, None) => Err(EvalError::BadParams(format!(
                    "enum variant `{}::{}` carries a payload; supply `{{\"{}\": payload}}`",
                    name_of(cb, &h).unwrap_or_else(|| h.to_hex()[..8].to_string()),
                    vname,
                    vname
                ))),
                (None, _, Some(_)) => Err(EvalError::BadParams(format!(
                    "enum variant `{}::{}` is nullary; supply the bare string `\"{}\"`",
                    name_of(cb, &h).unwrap_or_else(|| h.to_hex()[..8].to_string()),
                    vname,
                    vname
                ))),
                _ => Err(EvalError::Unsupported(format!(
                    "variant `{}::{}` shape/def payload disagree",
                    h, vname
                ))),
            }
        }
        Def::Fn { .. } => Err(EvalError::Unsupported(format!(
            "`{}` is a function, not a constructible value",
            name_of(cb, &h).unwrap_or_else(|| h.to_hex()[..8].to_string())
        ))),
        Def::State { .. } => Err(EvalError::Unsupported(format!(
            "`{}` is a state binding, not a constructible value",
            name_of(cb, &h).unwrap_or_else(|| h.to_hex()[..8].to_string())
        ))),
    }
}

/// Build `fval` per its declared `fty` and store it into the heap object `ptr`
/// at the field's `meta` slot. A pointer-shaped slot stores the built pointer;
/// a boxed scalar in a generic (pointer) slot is boxed; a raw scalar slot
/// stores the raw 8 bytes.
unsafe fn store_built_field(
    cb: &Codebase,
    rt: &Runtime,
    ptr: *mut u8,
    meta: &crate::codegen::FieldMeta,
    fty: &Type,
    fval: &Json,
) -> Result<(), EvalError> {
    let thread = rt.thread_ptr();
    // Root the destination object across the field build (and the box below):
    // both allocate, and a collection there would relocate `ptr`, leaving the
    // store writing through a stale pointer. Re-read after each child alloc.
    let dyna = unsafe { &*(*thread).dyna_thread };
    let scope = dyna.scratch_scope();
    let ptr_slot = scope.push(ptr);
    let raw = unsafe { build_arg(cb, rt, fty, fval)? };
    if meta.is_pointer {
        // A scalar declared type in a pointer slot means a generic (TypeVar)
        // field carrying a boxed scalar; box it. Otherwise `raw` is already a
        // heap pointer (String/Array/struct/enum).
        let p: *mut u8 = if is_boxed_scalar(fty) {
            unsafe { crate::runtime::ai_gc_box_int(thread, raw) }
        } else {
            raw as *mut u8
        };
        // Re-read the relocated object AFTER the box alloc; the store itself
        // does not allocate, so `p` stays valid.
        let ptr = scope.get(ptr_slot);
        unsafe {
            *(ptr.add(meta.offset as usize) as *mut *mut u8) = p;
        }
    } else {
        let ptr = scope.get(ptr_slot);
        unsafe {
            *(ptr.add(meta.offset as usize) as *mut i64) = raw;
        }
    }
    Ok(())
}

/// Build an `Array<elem_ty>` arg directly in `rt`'s heap from a JSON array.
///
/// `ai_array_new(thread, n)` allocates `n` null GC slots; each element is built
/// recursively by its declared `elem_ty` (reusing [`build_arg`], so struct /
/// enum / String / nested-Array elements all work) and stored via `ai_array_set`.
///
/// Array slots are **uniform boxed pointers**. The renderer (`render_array`)
/// treats every slot as a pointer and unboxes an `Int` element via
/// `ai_gc_unbox_int`. To round-trip correctly we must therefore BOX an `Int`
/// element on the way in via `ai_gc_box_int` (its symmetric inverse), and store
/// the box pointer. Pointer-shaped element types (struct/enum/String/Array) are
/// stored as-is. `Float`/`Bool` elements would also need boxing but the renderer
/// can't distinguish them from `Int` on the way out, so we refuse them honestly
/// rather than mis-render.
///
/// # Safety
/// Same contract as [`build_arg`]: `rt` installed, no GC between build and call.
unsafe fn build_array_arg(
    cb: &Codebase,
    rt: &Runtime,
    elem_ty: &Type,
    json: &Json,
) -> Result<i64, EvalError> {
    let thread = rt.thread_ptr();
    let items = match json {
        Json::Array(v) => v,
        _ => {
            return Err(EvalError::BadParams(format!(
                "expected a JSON array for an `Array<...>` parameter, got {}",
                json_shape(json)
            )));
        }
    };
    let n = items.len() as i64;
    // Borrow-checked allocation: `arr` stays rooted across every element
    // build (each allocates and may move `arr`), and the borrow checker —
    // not hand-written scratch bookkeeping — guarantees we re-read it.
    unsafe {
        Alloc::enter(thread, |a, scope| {
            let arr = a.array_new(n).root(scope);
            for (i, item) in items.iter().enumerate() {
                // Build the element by its declared type. A scalar element
                // (`Int`/`Float`/`Bool`) comes back as a raw 8-byte value;
                // box it so the slot holds a pointer the renderer can unbox.
                // Pointer-shaped elements (struct/enum/String/Array) are
                // stored as-is.
                let raw = build_arg(cb, rt, elem_ty, item)?;
                let it = a.scope();
                let elem = if is_boxed_scalar(elem_ty) {
                    a.box_int(raw).root(&it)
                } else {
                    a.adopt(raw as *mut u8).root(&it)
                };
                a.array_set(arr.get(a), i as i64, elem.get(a));
            }
            Ok(arr.get(a).ptr() as i64)
        })
    }
}

// =============================================================================
// eval
// =============================================================================

/// JIT and invoke the def at `root_hash` with JSON `args`, then render the
/// result per the declared return type.
///
/// `params` / `ret` are the def's cached signature (from the typecache). Each
/// JSON arg is built into a real heap object of its declared parameter type in
/// the eval runtime's heap (see [`json_arg_to_wire`]); the return side is
/// fully general. Both sides refuse to fabricate values: a mismatch or an
/// unsupported type yields a structured [`EvalError`].
pub fn eval(
    cb: &Codebase,
    root_hash: Hash,
    params: &[Type],
    ret: &Type,
    args: &[Json],
) -> Result<Evaluated, EvalError> {
    // --- Arity gates (independent of the runtime). ---
    if params.len() != args.len() {
        return Err(EvalError::BadParams(format!(
            "arity mismatch: def takes {} arg(s), got {}",
            params.len(),
            args.len()
        )));
    }
    if args.len() > 12 {
        return Err(EvalError::Arity(format!(
            "eval handles 0..=12 arguments, got {}",
            args.len()
        )));
    }

    init_native_target().map_err(|e| EvalError::Jit(format!("init native target: {}", e)))?;

    let rm = build_resolved_module(cb, root_hash)?;

    // Build a fresh LLVM context + module + JIT for this single invocation.
    let ctx = Context::create();
    let cm = CompiledModule::build(&ctx, &rm)
        .map_err(|e| EvalError::Jit(format!("build module: {}", e)))?;
    let rt = Runtime::new_with_metadata(
        cm.closure_type_infos.clone(),
        cm.shape_registry.clone(),
        cm.shape_meta.clone(),
        cm.shape_by_type_id.clone(),
    );
    let jit = Jit::new(cm, &rt).map_err(|e| EvalError::Jit(format!("jit: {}", e)))?;

    install_current_runtime(&rt);
    let kb = KnowledgeBase::build(&rm);
    install_current_knowledge_base(&kb);
    let _rb_storage;
    if let Some(rb) = rm
        .at_binding
        .as_ref()
        .and_then(|ab| build_at_runtime_binding(&rt, ab))
    {
        _rb_storage = rb;
        install_current_at_binding(&_rb_storage);
    }

    let symbol = def_symbol(&root_hash);
    let thread = rt.thread_ptr();

    // --- Build the call arguments INTO THIS runtime's heap. ---
    //
    // Ordering is load-bearing: each complex arg is decoded by allocating a
    // fresh heap object via `decode_value` against `rt` (the same runtime we
    // then call the function in). We do this AFTER `rt`/`jit` exist and BEFORE
    // the call, and keep the resulting pointers alive (in `av`) across the
    // call. Assumption: no GC runs between decode and call — this is a fresh
    // eval with a fresh heap and we don't trigger a collection in between, so
    // the freshly-allocated arg objects stay valid. (If decode itself fails on
    // an unsupported/mismatched arg we bail out before any call, after tearing
    // down the installed globals below.)
    let decode_result: Result<Vec<i64>, EvalError> = (|| {
        let mut av: Vec<i64> = Vec::with_capacity(args.len());
        for (i, (pty, json)) in params.iter().zip(args.iter()).enumerate() {
            // `build_arg` either allocates a heap object directly (String /
            // Bytes / Array — runtime-reserved shapes) or routes through the
            // honest wire path (`Int`/`Float`/`Bool`/struct/enum). All against
            // THIS `rt`'s heap, so the resulting pointers are valid for the
            // call below. (`decode_value` returns the scalar for Int kinds
            // without allocating.)
            let v = unsafe {
                build_arg(cb, &rt, pty, json).map_err(|e| match e {
                    // Preserve the kind, but prefix which arg failed.
                    EvalError::BadParams(m) => EvalError::BadParams(format!("arg {}: {}", i, m)),
                    EvalError::Unsupported(m) => EvalError::Unsupported(format!("arg {}: {}", i, m)),
                    other => other,
                })?
            };
            av.push(v);
        }
        Ok(av)
    })();

    let av = match decode_result {
        Ok(av) => av,
        Err(e) => {
            // Tear down installed globals before returning the arg error so a
            // long-lived server's next eval starts clean.
            clear_at_conn_cache();
            clear_current_runtime();
            clear_current_knowledge_base();
            clear_current_at_binding();
            return Err(e);
        }
    };

    // Call at the matching arity. The kernel's uniform ABI threads the
    // `*mut Thread` first; the raw i64 result is interpreted per `ret` below.
    let call_result: Result<i64, EvalError> = (|| {
        let result = unsafe {
            match av.len() {
                0 => {
                    let f = jit
                        .engine
                        .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(&symbol)
                        .map_err(|_| {
                            EvalError::Jit(format!("`{}` is not a 0-arg function", symbol))
                        })?;
                    f.call(thread)
                }
                1 => {
                    let f = jit
                        .engine
                        .get_function::<unsafe extern "C" fn(*mut Thread, i64) -> i64>(&symbol)
                        .map_err(|_| {
                            EvalError::Jit(format!("`{}` is not a 1-arg function", symbol))
                        })?;
                    f.call(thread, av[0])
                }
                2 => {
                    let f = jit
                        .engine
                        .get_function::<unsafe extern "C" fn(*mut Thread, i64, i64) -> i64>(&symbol)
                        .map_err(|_| {
                            EvalError::Jit(format!("`{}` is not a 2-arg function", symbol))
                        })?;
                    f.call(thread, av[0], av[1])
                }
                3 => {
                    let f = jit
                        .engine
                        .get_function::<unsafe extern "C" fn(*mut Thread, i64, i64, i64) -> i64>(
                            &symbol,
                        )
                        .map_err(|_| {
                            EvalError::Jit(format!("`{}` is not a 3-arg function", symbol))
                        })?;
                    f.call(thread, av[0], av[1], av[2])
                }
                4 => {
                    let f = jit
                        .engine
                        .get_function::<unsafe extern "C" fn(
                            *mut Thread,
                            i64,
                            i64,
                            i64,
                            i64,
                        ) -> i64>(&symbol)
                        .map_err(|_| {
                            EvalError::Jit(format!("`{}` is not a 4-arg function", symbol))
                        })?;
                    f.call(thread, av[0], av[1], av[2], av[3])
                }
                5 => {
                    let f = jit
                        .engine
                        .get_function::<unsafe extern "C" fn(
                            *mut Thread,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                        ) -> i64>(&symbol)
                        .map_err(|_| {
                            EvalError::Jit(format!("`{}` is not a 5-arg function", symbol))
                        })?;
                    f.call(thread, av[0], av[1], av[2], av[3], av[4])
                }
                6 => {
                    let f = jit
                        .engine
                        .get_function::<unsafe extern "C" fn(
                            *mut Thread,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                        ) -> i64>(&symbol)
                        .map_err(|_| {
                            EvalError::Jit(format!("`{}` is not a 6-arg function", symbol))
                        })?;
                    f.call(thread, av[0], av[1], av[2], av[3], av[4], av[5])
                }
                7 => {
                    let f = jit
                        .engine
                        .get_function::<unsafe extern "C" fn(
                            *mut Thread,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                        ) -> i64>(&symbol)
                        .map_err(|_| {
                            EvalError::Jit(format!("`{}` is not a 7-arg function", symbol))
                        })?;
                    f.call(thread, av[0], av[1], av[2], av[3], av[4], av[5], av[6])
                }
                8 => {
                    let f = jit
                        .engine
                        .get_function::<unsafe extern "C" fn(
                            *mut Thread,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                        ) -> i64>(&symbol)
                        .map_err(|_| {
                            EvalError::Jit(format!("`{}` is not an 8-arg function", symbol))
                        })?;
                    f.call(thread, av[0], av[1], av[2], av[3], av[4], av[5], av[6], av[7])
                }
                9 => {
                    let f = jit
                        .engine
                        .get_function::<unsafe extern "C" fn(
                            *mut Thread,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                        ) -> i64>(&symbol)
                        .map_err(|_| {
                            EvalError::Jit(format!("`{}` is not a 9-arg function", symbol))
                        })?;
                    f.call(
                        thread, av[0], av[1], av[2], av[3], av[4], av[5], av[6], av[7], av[8],
                    )
                }
                10 => {
                    let f = jit
                        .engine
                        .get_function::<unsafe extern "C" fn(
                            *mut Thread,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                        ) -> i64>(&symbol)
                        .map_err(|_| {
                            EvalError::Jit(format!("`{}` is not a 10-arg function", symbol))
                        })?;
                    f.call(
                        thread, av[0], av[1], av[2], av[3], av[4], av[5], av[6], av[7], av[8],
                        av[9],
                    )
                }
                11 => {
                    let f = jit
                        .engine
                        .get_function::<unsafe extern "C" fn(
                            *mut Thread,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                        ) -> i64>(&symbol)
                        .map_err(|_| {
                            EvalError::Jit(format!("`{}` is not an 11-arg function", symbol))
                        })?;
                    f.call(
                        thread, av[0], av[1], av[2], av[3], av[4], av[5], av[6], av[7], av[8],
                        av[9], av[10],
                    )
                }
                12 => {
                    let f = jit
                        .engine
                        .get_function::<unsafe extern "C" fn(
                            *mut Thread,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                            i64,
                        ) -> i64>(&symbol)
                        .map_err(|_| {
                            EvalError::Jit(format!("`{}` is not a 12-arg function", symbol))
                        })?;
                    f.call(
                        thread, av[0], av[1], av[2], av[3], av[4], av[5], av[6], av[7], av[8],
                        av[9], av[10], av[11],
                    )
                }
                n => return Err(EvalError::Arity(format!("unhandled arity {}", n))),
            }
        };
        Ok(result)
    })();

    // Render the result BEFORE tearing down the runtime (the heap is still
    // live and `rt` is still installed). Any rendering error is captured here.
    let rendered: Result<Evaluated, EvalError> = call_result.and_then(|bits| {
        let rc = RenderCtx { rt: &rt, cb };
        let value = unsafe { rc.render(ret, bits)? };
        Ok(Evaluated {
            value,
            type_str: render_ret_type(cb, ret),
        })
    });

    // Tear down installed globals so repeated evals in one long-lived process
    // (an edit-server session) don't corrupt each other's state.
    clear_at_conn_cache();
    clear_current_runtime();
    clear_current_knowledge_base();
    clear_current_at_binding();

    rendered
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edit;
    use std::sync::Once;

    static INIT: Once = Once::new();
    fn init() {
        INIT.call_once(|| {
            init_native_target().expect("init native target");
        });
    }

    /// Build a throwaway on-disk codebase, add `src`, return it.
    fn cb_with(src: &str) -> Codebase {
        let dir = std::env::temp_dir().join(format!(
            "ai-lang-evalrun-test-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let mut cb = Codebase::open(&dir).expect("open cb");
        edit::add(&mut cb, src).expect("add src");
        cb
    }

    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    /// Resolve a name + run eval with integer args, returning the rendered
    /// value. (Ints are the common case; this wraps `run_json`.)
    fn run(cb: &Codebase, name: &str, args: &[i64]) -> Result<Evaluated, EvalError> {
        let jargs: Vec<Json> = args.iter().map(|n| Json::Int(*n)).collect();
        run_json(cb, name, &jargs)
    }

    /// Resolve a name + run eval with arbitrary JSON args.
    fn run_json(cb: &Codebase, name: &str, args: &[Json]) -> Result<Evaluated, EvalError> {
        let hash = cb.get_name(name).expect("name exists");
        let scheme = cb.types().get(&hash).expect("scheme");
        let (params, ret) = match scheme {
            TypeScheme::Fn { params, ret, .. } => (params.clone(), ret.clone()),
            _ => panic!("not a fn"),
        };
        eval(cb, hash, &params, &ret, args)
    }

    use crate::typecheck::TypeScheme;

    #[test]
    fn eval_int() {
        init();
        let cb = cb_with("def f(x: Int) -> Int = x * 2");
        let r = run(&cb, "f", &[21]).unwrap();
        assert_eq!(r.value, Json::Int(42));
        assert_eq!(r.type_str, "Int");
    }

    #[test]
    fn eval_float() {
        init();
        // 7.0 / 2.0 = 3.5 — proves bit-pattern handling, not raw-i64 garbage.
        let cb = cb_with("def g() -> Float = 7.0 / 2.0");
        let r = run(&cb, "g", &[]).unwrap();
        assert_eq!(r.value, Json::Float(3.5));
        assert_eq!(r.type_str, "Float");
    }

    #[test]
    fn eval_bool_return_renders_json_bool() {
        // Bool is i64 0/1 at runtime and now has full codegen support (literal
        // lowering + Bool as a scalar-int ABI param/return type), so a
        // `-> Bool` def runs and the renderer maps it to a JSON bool.
        init();
        let cb = cb_with("def t() -> Bool = true");
        let r = run(&cb, "t", &[]).unwrap();
        assert_eq!(r.value, Json::Bool(true));
        assert_eq!(r.type_str, "Bool");

        let cb2 = cb_with("def f() -> Bool = false");
        let r2 = run(&cb2, "f", &[]).unwrap();
        assert_eq!(r2.value, Json::Bool(false));
        assert_eq!(r2.type_str, "Bool");
    }

    #[test]
    fn eval_string() {
        init();
        let cb = cb_with("def hi() -> String = \"hello\"");
        let r = run(&cb, "hi", &[]).unwrap();
        assert_eq!(r.value, Json::Str("hello".to_string()));
        assert_eq!(r.type_str, "String");
    }

    #[test]
    fn eval_struct() {
        init();
        let cb = cb_with(
            "struct Point { x: Int, y: Int }\n\
             def mk() -> Point = Point { x: 3, y: 4 }",
        );
        let r = run(&cb, "mk", &[]).unwrap();
        assert_eq!(
            r.value,
            Json::obj([
                ("x".to_string(), Json::Int(3)),
                ("y".to_string(), Json::Int(4)),
            ])
        );
        assert_eq!(r.type_str, "Point");
    }

    #[test]
    fn eval_enum_with_payload() {
        init();
        let cb = cb_with(
            "enum Opt { Some(Int), None }\n\
             def s() -> Opt = Opt::Some(5)\n\
             def n() -> Opt = Opt::None",
        );
        let s = run(&cb, "s", &[]).unwrap();
        assert_eq!(s.value, Json::obj([("Some".to_string(), Json::Int(5))]));
        let n = run(&cb, "n", &[]).unwrap();
        assert_eq!(n.value, Json::Str("None".to_string()));
    }

    #[test]
    fn eval_struct_with_string_field() {
        init();
        let cb = cb_with(
            "struct Named { id: Int, label: String }\n\
             def mk() -> Named = Named { id: 7, label: \"widget\" }",
        );
        let r = run(&cb, "mk", &[]).unwrap();
        assert_eq!(
            r.value,
            Json::obj([
                ("id".to_string(), Json::Int(7)),
                ("label".to_string(), Json::Str("widget".to_string())),
            ])
        );
    }

    #[test]
    fn eval_array_of_ints() {
        init();
        // `Array<Int>` renders as a JSON array; Int elements are boxed in the
        // varlen-Values slots and unboxed honestly by the renderer. (We fill
        // sequentially rather than recursively because `edit::add` currently
        // can't commit a self-recursive helper — unrelated to eval.)
        let cb = cb_with(
            "def mk() -> Array<Int> = { \
                let a = array_new(3); \
                let _0 = array_set(a, 0, 10); \
                let _1 = array_set(a, 1, 20); \
                let _2 = array_set(a, 2, 30); \
                a }",
        );
        let r = run(&cb, "mk", &[]).unwrap();
        assert_eq!(
            r.value,
            Json::Array(vec![Json::Int(10), Json::Int(20), Json::Int(30)])
        );
        assert_eq!(r.type_str, "Array<Int>");
    }

    #[test]
    fn repeated_eval_no_corruption() {
        init();
        let cb = cb_with(
            "struct P { x: Int, y: Int }\n\
             def mk(a: Int, b: Int) -> P = P { x: a, y: b }\n\
             def dbl(x: Int) -> Int = x * 2",
        );
        // Interleave heap-returning and scalar-returning evals repeatedly.
        for i in 0..6 {
            let p = run(&cb, "mk", &[i, i + 1]).unwrap();
            assert_eq!(
                p.value,
                Json::obj([
                    ("x".to_string(), Json::Int(i)),
                    ("y".to_string(), Json::Int(i + 1)),
                ])
            );
            let d = run(&cb, "dbl", &[i]).unwrap();
            assert_eq!(d.value, Json::Int(i * 2));
        }
    }

    #[test]
    fn arity_mismatch_is_bad_params() {
        init();
        let cb = cb_with("def f(x: Int) -> Int = x");
        let e = run(&cb, "f", &[]).unwrap_err();
        assert_eq!(e.kind(), "BadParams");
    }

    // --- Complex argument tests (arg side is now general). ---

    #[test]
    fn struct_arg() {
        init();
        let cb = cb_with(
            "struct Point { x: Int, y: Int }\n\
             def sumxy(p: Point) -> Int = p.x + p.y",
        );
        let arg = Json::obj([
            ("x".to_string(), Json::Int(3)),
            ("y".to_string(), Json::Int(4)),
        ]);
        let r = run_json(&cb, "sumxy", &[arg]).unwrap();
        assert_eq!(r.value, Json::Int(7));
    }

    #[test]
    fn enum_arg_payload_and_nullary() {
        init();
        let cb = cb_with(
            "enum Opt { Some(Int), None }\n\
             def unwrap_or(o: Opt, d: Int) -> Int = match o { Opt::Some(x) => x, Opt::None => d }",
        );
        // {"Some": 5} -> 5
        let some = Json::obj([("Some".to_string(), Json::Int(5))]);
        let r = run_json(&cb, "unwrap_or", &[some, Json::Int(99)]).unwrap();
        assert_eq!(r.value, Json::Int(5));
        // "None" (nullary) -> default 99
        let r2 = run_json(&cb, "unwrap_or", &[Json::Str("None".to_string()), Json::Int(99)])
            .unwrap();
        assert_eq!(r2.value, Json::Int(99));
    }

    #[test]
    fn many_int_args_six() {
        init();
        // Proves arity raised past 4.
        let cb = cb_with(
            "def add6(a: Int, b: Int, c: Int, d: Int, e: Int, f: Int) -> Int = \
             a + b + c + d + e + f",
        );
        let r = run(&cb, "add6", &[1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(r.value, Json::Int(21));
    }

    #[test]
    fn many_int_args_twelve() {
        init();
        let cb = cb_with(
            "def add12(a: Int, b: Int, c: Int, d: Int, e: Int, f: Int, g: Int, h: Int, \
             i: Int, j: Int, k: Int, l: Int) -> Int = \
             a + b + c + d + e + f + g + h + i + j + k + l",
        );
        let r = run(&cb, "add12", &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).unwrap();
        assert_eq!(r.value, Json::Int(12));
    }

    #[test]
    fn float_arg_roundtrips() {
        init();
        // Operators are shared with Int; `x + x` on a Float param proves the
        // f64 bit-pattern crosses the ABI correctly (1.25 + 1.25 = 2.5), not
        // raw-i64 garbage.
        let cb = cb_with("def dbl(x: Float) -> Float = x + x");
        let r = run_json(&cb, "dbl", &[Json::Float(1.25)]).unwrap();
        assert_eq!(r.value, Json::Float(2.5));
    }

    #[test]
    fn bool_arg_flows_through_and_renders() {
        init();
        // Bool is a first-class scalar now (i64 0/1 ABI), so a `Bool` param
        // flows through to a `Bool` return and renders as a JSON bool.
        let cb = cb_with("def id(b: Bool) -> Bool = b");
        let r = run_json(&cb, "id", &[Json::Bool(true)]).unwrap();
        assert_eq!(r.value, Json::Bool(true));
        let r2 = run_json(&cb, "id", &[Json::Bool(false)]).unwrap();
        assert_eq!(r2.value, Json::Bool(false));
        // A non-bool for a Bool param is a structured error, not a coercion.
        let e = run_json(&cb, "id", &[Json::Int(1)]).unwrap_err();
        assert_eq!(e.kind(), "BadParams");
    }

    #[test]
    fn struct_arg_wrong_shape_is_bad_params() {
        init();
        let cb = cb_with(
            "struct Q { v: Int }\n\
             def takes(q: Q) -> Int = q.v",
        );
        // Passing a bare integer where a struct object is required must be a
        // structured error, never a coerced/garbage pointer.
        let e = run_json(&cb, "takes", &[Json::Int(1)]).unwrap_err();
        assert_eq!(e.kind(), "BadParams");
        // Missing field is also a structured error.
        let e2 = run_json(&cb, "takes", &[Json::obj([("w".to_string(), Json::Int(1))])])
            .unwrap_err();
        assert_eq!(e2.kind(), "BadParams");
    }

    #[test]
    fn string_arg_builds_real_heap_string() {
        init();
        // String args are now built directly via `ai_str_new`: a real heap
        // varlen-bytes object. `string_len("hello") + 3 = 8` proves the bytes +
        // length crossed the ABI correctly (not a garbage pointer).
        let cb = cb_with("def strlen_plus(s: String, k: Int) -> Int = string_len(s) + k");
        let r = run_json(
            &cb,
            "strlen_plus",
            &[Json::Str("hello".to_string()), Json::Int(3)],
        )
        .unwrap();
        assert_eq!(r.value, Json::Int(8));
    }

    #[test]
    fn array_int_arg_sums_correctly() {
        init();
        // `Array<Int>` args are built directly via `ai_array_new` + per-element
        // `ai_gc_box_int` + `ai_array_set`. Summing the array proves the boxed
        // Int elements are CORRECT (round-trips with the renderer's unbox).
        let cb = cb_with(
            "def sum(a: Array<Int>) -> Int = { \
                let n = array_len(a); \
                let s0 = 0; \
                let s1 = if n > 0 { s0 + array_get(a, 0) } else { s0 }; \
                let s2 = if n > 1 { s1 + array_get(a, 1) } else { s1 }; \
                let s3 = if n > 2 { s2 + array_get(a, 2) } else { s2 }; \
                s3 }",
        );
        let arr = Json::Array(vec![Json::Int(10), Json::Int(20), Json::Int(30)]);
        let r = run_json(&cb, "sum", &[arr]).unwrap();
        assert_eq!(r.value, Json::Int(60));
    }

    #[test]
    fn array_int_arg_roundtrips_through_identity() {
        init();
        // Pass an Array<Int> in, return it out: the renderer must unbox the same
        // boxed Ints we put in. Tightest possible round-trip.
        let cb = cb_with("def id(a: Array<Int>) -> Array<Int> = a");
        let arr = Json::Array(vec![Json::Int(1), Json::Int(2), Json::Int(3)]);
        let r = run_json(&cb, "id", &[arr]).unwrap();
        assert_eq!(
            r.value,
            Json::Array(vec![Json::Int(1), Json::Int(2), Json::Int(3)])
        );
    }

    #[test]
    fn array_float_arg_round_trips() {
        init();
        // `Array<Float>` works: a boxed slot holds a type-agnostic 8-byte value,
        // and the renderer interprets it per the declared element type (here
        // f64 bit-pattern). Pass in floats, get the same floats back.
        let cb = cb_with("def fid(a: Array<Float>) -> Array<Float> = a");
        let r = run_json(
            &cb,
            "fid",
            &[Json::Array(vec![Json::Float(1.5), Json::Float(2.5)])],
        )
        .unwrap();
        assert_eq!(r.value, Json::Array(vec![Json::Float(1.5), Json::Float(2.5)]));
    }

    #[test]
    fn array_bool_arg_round_trips() {
        init();
        let cb = cb_with("def bid(a: Array<Bool>) -> Array<Bool> = a");
        let r = run_json(
            &cb,
            "bid",
            &[Json::Array(vec![Json::Bool(true), Json::Bool(false)])],
        )
        .unwrap();
        assert_eq!(r.value, Json::Array(vec![Json::Bool(true), Json::Bool(false)]));
    }

    #[test]
    fn string_field_in_struct_arg_works() {
        init();
        // The deepest former limitation: a String INSIDE a struct field. The
        // direct-construction arg builder composes, so this now works.
        let cb = cb_with(
            "struct Person { name: String, age: Int }\n\
             def name_len(p: Person) -> Int = string_len(p.name)",
        );
        let mut obj = std::collections::BTreeMap::new();
        obj.insert("name".to_string(), Json::Str("Alice".to_string()));
        obj.insert("age".to_string(), Json::Int(30));
        let r = run_json(&cb, "name_len", &[Json::Object(obj)]).unwrap();
        assert_eq!(r.value, Json::Int(5));
    }

    #[test]
    fn string_payload_in_enum_arg_works() {
        init();
        let cb = cb_with(
            "enum Msg { Text(String), Empty }\n\
             def msg_len(m: Msg) -> Int = match m { Msg::Text(s) => string_len(s), Msg::Empty => 0 }",
        );
        let mut obj = std::collections::BTreeMap::new();
        obj.insert("Text".to_string(), Json::Str("hello world".to_string()));
        let r = run_json(&cb, "msg_len", &[Json::Object(obj)]).unwrap();
        assert_eq!(r.value, Json::Int(11));
        let r2 = run_json(&cb, "msg_len", &[Json::Str("Empty".to_string())]).unwrap();
        assert_eq!(r2.value, Json::Int(0));
    }
}
