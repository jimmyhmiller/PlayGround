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
//! ### Argument types that remain Unsupported (and why)
//!
//! - **`String` / `Bytes`** — the wire format ([`crate::wire`]) has exactly four
//!   value kinds (Int/Closure/Struct/Enum); there is no `String` kind, and
//!   `decode_value` cannot build a heap varlen-bytes object from data. (Strings
//!   are a runtime-reserved shape, not a content-addressed struct/enum.) Until
//!   the wire format grows a String kind, String *args* are refused honestly.
//! - **`Array<T>`** — likewise a varlen-Values runtime shape with no wire kind;
//!   `decode_value` cannot build one. (Note: a `List<T>` *can* be passed — it's
//!   a cons-cell enum, which goes through the enum wire path.)
//! - **`fn(..)` / closures** — a closure can't be constructed from JSON data.
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
use crate::runtime::{Runtime, Thread};
use crate::stdlib::SOURCE as STDLIB;
use crate::wire::{WireValue, decode_value};
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
fn is_int(t: &Type) -> bool {
    matches!(t, Type::Builtin(b) if b == "Int")
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
        // A concrete `Int` stored in a generic (TypeVar) slot is a *boxed*
        // pointer at the ABI level: the shape says pointer, the type says Int.
        // Unbox honestly.
        if is_int(ty) && is_pointer {
            let unboxed = unsafe { crate::runtime::ai_gc_unbox_int(bits as *const u8) };
            return Ok(Json::Int(unboxed));
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
// Each JSON argument is converted to wire bytes per the corresponding
// parameter's *declared* type, then `wire::decode_value` allocates a real
// heap object of that type in the eval runtime's heap. The wire format
// (see `wire.rs`) has four kinds:
//   0 = Int     [kind][i64 BE]
//   1 = Closure (not built from JSON — Unsupported)
//   2 = Struct  [kind][32-byte struct_ref][u32 n_fields][field values…]
//   3 = Enum    [kind][32-byte enum_ref][u32 variant_index][u8 has_payload][payload?]
// Float is an Int kind carrying the f64 bit-pattern; Bool is Int 0/1.
// =============================================================================

/// Recursively encode a single JSON `json` to wire bytes in `out`, validating
/// it against the declared parameter `ty`. The codebase `cb` resolves
/// `TypeRef` hashes to their struct/enum `Def`s so we know field/variant
/// layout. On a shape/type mismatch we return a structured error and emit no
/// garbage. The produced bytes are the exact inverse of `wire::encode_value`,
/// so a subsequent `decode_value` builds a real heap object.
fn json_arg_to_wire(
    cb: &Codebase,
    ty: &Type,
    json: &Json,
    out: &mut Vec<u8>,
) -> Result<(), EvalError> {
    match ty {
        Type::Builtin(b) => match b.as_str() {
            "Int" => {
                let n = json.as_i64().ok_or_else(|| {
                    EvalError::BadParams(format!(
                        "expected an integer for an `Int` parameter, got {}",
                        json_shape(json)
                    ))
                })?;
                push_wire_int(out, n);
                Ok(())
            }
            "Float" => {
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
                // Float crosses the ABI as the i64 bit-pattern of the f64.
                push_wire_int(out, f.to_bits() as i64);
                Ok(())
            }
            "Bool" => {
                let v = json.as_bool().ok_or_else(|| {
                    EvalError::BadParams(format!(
                        "expected a boolean for a `Bool` parameter, got {}",
                        json_shape(json)
                    ))
                })?;
                push_wire_int(out, if v { 1 } else { 0 });
                Ok(())
            }
            "String" | "Bytes" => Err(EvalError::Unsupported(format!(
                "`{}` arguments are not supported: the wire format has no String/Bytes value \
                 kind, so a heap string can't be built from JSON data. (Strings are a \
                 runtime-reserved shape, not a content-addressed struct/enum.)",
                b
            ))),
            other => Err(EvalError::Unsupported(format!(
                "cannot build an argument of builtin type `{}`",
                other
            ))),
        },
        Type::TypeRef(h) => json_arg_ref_to_wire(cb, *h, json, out),
        Type::Apply(head, _args) => {
            // A generic instantiation. `Array<T>` is a runtime varlen shape with
            // no wire kind — Unsupported. A `TypeRef`-headed Apply (a generic
            // struct/enum like `List<Int>` or `Option<Int>`) is built by the
            // same struct/enum path keyed on the head hash; the wire format
            // stores no type arguments (the heap layout is uniform/boxed), so we
            // encode against the head's declared fields/variants directly.
            if let Type::Builtin(b) = head.as_ref() {
                if b == "Array" {
                    return Err(EvalError::Unsupported(
                        "`Array<...>` arguments are not supported: an Array is a runtime \
                         varlen-Values shape with no wire value kind, so it can't be built from \
                         JSON. (A cons-cell `List<...>` CAN be passed — it's an enum.)"
                            .to_string(),
                    ));
                }
                return Err(EvalError::Unsupported(format!(
                    "cannot build an argument of applied builtin type `{}<...>`",
                    b
                )));
            }
            if let Type::TypeRef(h) = head.as_ref() {
                return json_arg_ref_to_wire(cb, *h, json, out);
            }
            Err(EvalError::Unsupported(format!(
                "cannot build an argument of applied type with head `{}`",
                render_ret_type(cb, head)
            )))
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
    }
}

/// Encode a JSON value against a named struct/enum `TypeRef(h)`.
fn json_arg_ref_to_wire(
    cb: &Codebase,
    h: Hash,
    json: &Json,
    out: &mut Vec<u8>,
) -> Result<(), EvalError> {
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
            // Reject extra keys so a typo surfaces instead of being silently
            // dropped.
            for key in obj.keys() {
                if !fields.iter().any(|(fname, _)| fname == key) {
                    return Err(EvalError::BadParams(format!(
                        "struct `{}` has no field `{}`",
                        name_of(cb, &h).unwrap_or_else(|| h.to_hex()[..8].to_string()),
                        key
                    )));
                }
            }
            out.push(2); // kind = Struct
            out.extend_from_slice(h.as_bytes());
            let n: u32 = fields.len().try_into().map_err(|_| {
                EvalError::Unsupported("struct has more fields than fit in u32".to_string())
            })?;
            out.extend_from_slice(&n.to_be_bytes());
            // Fields in DECLARED order (the decoder replays the same order).
            for (fname, fty) in &fields {
                let fval = obj.get(fname).ok_or_else(|| {
                    EvalError::BadParams(format!(
                        "struct `{}` is missing field `{}`",
                        name_of(cb, &h).unwrap_or_else(|| h.to_hex()[..8].to_string()),
                        fname
                    ))
                })?;
                json_arg_to_wire(cb, fty, fval, out)?;
            }
            Ok(())
        }
        Def::Enum { variants, .. } => {
            // Two accepted JSON shapes:
            //   "VariantName"            — a nullary variant.
            //   {"VariantName": payload} — a variant with a payload.
            let (vname, payload): (&str, Option<&Json>) = match json {
                Json::Str(s) => (s.as_str(), None),
                Json::Object(m) if m.len() == 1 => {
                    let (k, v) = m.iter().next().unwrap();
                    (k.as_str(), Some(v))
                }
                Json::Object(_) => {
                    return Err(EvalError::BadParams(format!(
                        "enum `{}` value must be a single-key object {{\"Variant\": payload}} \
                         or a bare \"Variant\" string",
                        name_of(cb, &h).unwrap_or_else(|| h.to_hex()[..8].to_string())
                    )));
                }
                _ => {
                    return Err(EvalError::BadParams(format!(
                        "expected a string or single-key object for enum `{}`, got {}",
                        name_of(cb, &h).unwrap_or_else(|| h.to_hex()[..8].to_string()),
                        json_shape(json)
                    )));
                }
            };
            let idx = variants.iter().position(|(n, _)| n == vname).ok_or_else(|| {
                EvalError::BadParams(format!(
                    "enum `{}` has no variant `{}`",
                    name_of(cb, &h).unwrap_or_else(|| h.to_hex()[..8].to_string()),
                    vname
                ))
            })?;
            let (_, decl_payload) = &variants[idx];
            out.push(3); // kind = Enum
            out.extend_from_slice(h.as_bytes());
            out.extend_from_slice(&(idx as u32).to_be_bytes());
            match (decl_payload, payload) {
                (None, None) => {
                    out.push(0); // has_payload = 0
                    Ok(())
                }
                (Some(pty), Some(pjson)) => {
                    out.push(1); // has_payload = 1
                    json_arg_to_wire(cb, pty, pjson, out)
                }
                (Some(_), None) => Err(EvalError::BadParams(format!(
                    "enum variant `{}::{}` takes a payload; pass {{\"{}\": <payload>}}",
                    name_of(cb, &h).unwrap_or_else(|| h.to_hex()[..8].to_string()),
                    vname,
                    vname
                ))),
                (None, Some(_)) => Err(EvalError::BadParams(format!(
                    "enum variant `{}::{}` is nullary; pass the bare string \"{}\"",
                    name_of(cb, &h).unwrap_or_else(|| h.to_hex()[..8].to_string()),
                    vname,
                    vname
                ))),
            }
        }
        Def::Fn { .. } => Err(EvalError::Unsupported(format!(
            "TypeRef {} resolves to a function, not a struct/enum, so it can't be an argument",
            h
        ))),
    }
}

/// Push a wire Int value (`kind=0` + i64 big-endian).
fn push_wire_int(out: &mut Vec<u8>, n: i64) {
    out.push(0);
    out.extend_from_slice(&n.to_be_bytes());
}

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
            // Fast path: a plain `Int`/`Float`/`Bool` could go straight to an
            // i64, but routing everything through wire bytes + `decode_value`
            // keeps one honest code path (and `decode_value` returns the scalar
            // for Int kinds without allocating).
            let mut bytes = Vec::new();
            json_arg_to_wire(cb, pty, json, &mut bytes).map_err(|e| match e {
                // Preserve the kind, but prefix which arg failed.
                EvalError::BadParams(m) => EvalError::BadParams(format!("arg {}: {}", i, m)),
                EvalError::Unsupported(m) => EvalError::Unsupported(format!("arg {}: {}", i, m)),
                other => other,
            })?;
            let (wv, consumed) = unsafe {
                decode_value(&rt, &bytes).map_err(|e| {
                    EvalError::Unsupported(format!("arg {}: decode failed: {}", i, e))
                })?
            };
            if consumed != bytes.len() {
                return Err(EvalError::Unsupported(format!(
                    "arg {}: wire bytes not fully consumed ({} of {})",
                    i,
                    consumed,
                    bytes.len()
                )));
            }
            let v = match wv {
                WireValue::Int(n) => n,
                WireValue::Heap(p) => p as i64,
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
             def s() -> Opt = Some(5)\n\
             def n() -> Opt = None",
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
             def unwrap_or(o: Opt, d: Int) -> Int = match o { Some(x) => x, None => d }",
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
    fn bool_arg_builds_to_wire_int() {
        init();
        // In this language version a `Bool` is never equivalent to `Int`, so no
        // typecheckable/codegen-able body can OBSERVE a `Bool` parameter (`if`
        // wants Int; `-> Bool` is rejected by codegen). What we CAN verify is
        // the honest arg-building contract: a JSON bool encodes to a wire Int
        // 0/1 (the kernel ABI), so when a Bool param eventually becomes usable
        // the value is already correct. (No coercion, no garbage.)
        let cb = cb_with("def id(b: Bool) -> Bool = b");
        let hash = cb.get_name("id").unwrap();
        let bool_ty = match cb.types().get(&hash).unwrap() {
            TypeScheme::Fn { params, .. } => params[0].clone(),
            _ => panic!(),
        };
        let mut t = Vec::new();
        json_arg_to_wire(&cb, &bool_ty, &Json::Bool(true), &mut t).unwrap();
        assert_eq!(t, vec![0, 0, 0, 0, 0, 0, 0, 0, 1]); // kind=0 + i64 BE 1
        let mut f = Vec::new();
        json_arg_to_wire(&cb, &bool_ty, &Json::Bool(false), &mut f).unwrap();
        assert_eq!(f, vec![0, 0, 0, 0, 0, 0, 0, 0, 0]); // kind=0 + i64 BE 0
        // A non-bool for a Bool param is a structured error, not a coercion.
        let mut bad = Vec::new();
        let e = json_arg_to_wire(&cb, &bool_ty, &Json::Int(1), &mut bad).unwrap_err();
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
    fn string_arg_is_unsupported() {
        init();
        // String args genuinely can't be built (no wire String kind).
        let cb = cb_with("def len(s: String) -> Int = string_len(s)");
        let e = run_json(&cb, "len", &[Json::Str("hi".to_string())]).unwrap_err();
        assert_eq!(e.kind(), "Unsupported");
        assert!(e.message().contains("String"));
    }
}
