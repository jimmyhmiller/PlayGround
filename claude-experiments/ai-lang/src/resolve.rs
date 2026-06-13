//! Resolver: surface AST → canonical AST.
//!
//! Responsibilities:
//!
//! - **Locals → de Bruijn.** Lexical variable lookups become `LocalVar(idx)`,
//!   where `idx` counts outwards from the innermost binder (parameter 0 of
//!   the current function is the outermost binder for that function).
//! - **Top-level names → hashes.** Each `def` is processed via a Tarjan-SCC
//!   pass over the call graph, supporting full mutual recursion. References
//!   to in-module fns/states become `SelfRef`/`StateSelfRef` during hashing
//!   and `TopRef`/`StateRef` in the stored form.
//! - **Operators → builtins.** `+`, `*`, `==`, etc. lower to `BuiltinRef`
//!   under stable string ids (`core/i64.add`, `core/i64.eq`, …). Dispatch is
//!   type-directed by the typechecker.
//! - **Named types → builtins / user types.** `Int`/`Bool`/`String`/`Float`/
//!   `Bytes` map to `Type::Builtin`. User-defined structs/enums become
//!   `Type::TypeRef(hash)`. Type-parameter references become `Type::TypeVar`.
//! - **Generics.** Type parameters on defs/structs/enums are resolved to
//!   `TypeVar` de Bruijn indices within their scope. Generic instantiations
//!   (`List<Int>`) produce `Type::Apply`. Turbofish (`f::<T>(…)`) passes
//!   explicit type arguments.
//!
//! Remaining v1 restrictions:
//!
//! - Mutually recursive **struct types** are not yet supported (forward
//!   references to later structs error cleanly). Fn defs and state bindings
//!   ARE mutually recursive via Tarjan SCC.

use crate::ast::{Def, Expr, MatchArm, Pattern, Type};
use crate::codec::encode_def;
use crate::hash::Hash;
use crate::lexer::Span;
use crate::surface::{
    BinOp, Module, SurfaceDef, SurfaceDefKind, SurfaceExpr, SurfacePattern, SurfaceStmt,
    SurfaceType, UnaryOp,
};

use std::collections::HashMap;

// =============================================================================
// Errors
// =============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveError {
    UnknownName { name: String, span: Span },
    ForwardOrCyclicRef { name: String, span: Span },
    UnknownType { name: String, span: Span },
    DuplicateDef {
        name: String,
        first_span: Span,
        second_span: Span,
    },
    /// Struct literal references a type name that isn't a struct in scope.
    UnknownStruct { name: String, span: Span },
    /// Struct literal mentions a field name not declared on the struct.
    UnknownField {
        struct_name: String,
        field: String,
        span: Span,
    },
    /// Struct literal is missing a field that the struct declares.
    MissingField {
        struct_name: String,
        field: String,
        span: Span,
    },
    /// Struct literal sets the same field twice.
    DuplicateFieldInLiteral {
        struct_name: String,
        field: String,
        span: Span,
    },
    /// Field access on a base whose type isn't a struct.
    FieldOnNonStruct { ty: Type, span: Span },
    /// Pattern references a constructor name that isn't an enum variant.
    UnknownVariant { name: String, span: Span },
    /// Variant called with payload but declared as nullary, or vice-versa.
    VariantArityMismatch {
        variant: String,
        expected_payload: bool,
        span: Span,
    },
    /// Match arm patterns disagree on which enum they're matching.
    MatchEnumMismatch {
        expected: Hash,
        found: Hash,
        span: Span,
    },
    /// `at(...)` was used but the module is missing the user-defined
    /// `Result` / `Failure` / `Node` types it lowers to.
    AtRequiresBinding { missing: String, span: Span },
    /// The user-defined `Result` / `Failure` / `Node` is in scope but
    /// doesn't have the shape `at(...)` expects.
    AtBindingShape { what: String, span: Span },
    /// The `?` operator was applied to a value that isn't a `Result<T, E>`
    /// (a 2-variant enum with `Ok` and `Err` variants).
    TryOnNonResult { ty: Type, span: Span },
}

impl core::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ResolveError::UnknownName { name, .. } => {
                write!(f, "unknown name `{}`", name)
            }
            ResolveError::ForwardOrCyclicRef { name, .. } => write!(
                f,
                "`{}` is referenced before it is defined; \
                 forward references are not supported for this kind of definition",
                name
            ),
            ResolveError::UnknownType { name, .. } => {
                write!(f, "unknown type `{}`", name)
            }
            ResolveError::DuplicateDef { name, .. } => {
                write!(f, "duplicate definition `{}`", name)
            }
            ResolveError::UnknownStruct { name, .. } => {
                write!(f, "unknown struct `{}`", name)
            }
            ResolveError::UnknownField { struct_name, field, .. } => {
                write!(f, "struct `{}` has no field `{}`", struct_name, field)
            }
            ResolveError::MissingField { struct_name, field, .. } => {
                write!(f, "struct `{}` literal is missing field `{}`", struct_name, field)
            }
            ResolveError::DuplicateFieldInLiteral { struct_name, field, .. } => {
                write!(f, "struct `{}` literal sets field `{}` twice", struct_name, field)
            }
            ResolveError::FieldOnNonStruct { ty, .. } => {
                write!(f, "field access on non-struct value of type {:?}", ty)
            }
            ResolveError::UnknownVariant { name, .. } => {
                write!(f, "unknown variant `{}`", name)
            }
            ResolveError::VariantArityMismatch { variant, expected_payload, .. } => {
                if *expected_payload {
                    write!(f, "variant `{}` declared with a payload but used as nullary", variant)
                } else {
                    write!(f, "variant `{}` declared nullary but used with a payload", variant)
                }
            }
            ResolveError::MatchEnumMismatch { expected, found, .. } => {
                write!(f, "match arms must all be of the same enum (saw {} and {})", expected, found)
            }
            ResolveError::AtRequiresBinding { missing, .. } => write!(
                f,
                "`at(...)` requires the user-defined type `{}` to be in scope",
                missing
            ),
            ResolveError::TryOnNonResult { ty, .. } => write!(
                f,
                "the `?` operator requires a Result<T, E> (an enum with `Ok` and `Err` \
                 variants), but got {:?}",
                ty
            ),
            ResolveError::AtBindingShape { what, .. } => {
                write!(f, "`at(...)` binding has the wrong shape: {}", what)
            }
        }
    }
}

impl std::error::Error for ResolveError {}

// =============================================================================
// Output
// =============================================================================

/// A resolved + hashed top-level definition.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedDef {
    pub name: String,
    pub hash: Hash,
    pub def: Def,
}

/// The result of resolving a module: defs in source order, each with its
/// canonical form and content hash.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedModule {
    pub defs: Vec<ResolvedDef>,
    /// Set when the module uses `at(...)` — captures the hashes and
    /// variant indices of the user's `Result` / `Failure` / `Node`
    /// defs so the runtime can construct the Result enum from Rust.
    pub at_binding: Option<AtBinding>,
    /// Surface-level `extern fn` declarations, keyed by surface name.
    /// Each maps to its declared signature. Externs are NOT
    /// content-addressed (no canonical AST entry); they exist so
    /// call-sites can lower to `BuiltinRef("ext/<name>")` and the
    /// typechecker / codegen can recover the signature at the call
    /// site. The host runtime supplies the actual implementations
    /// via the global extern registry.
    pub externs: HashMap<String, ExternSig>,
}

/// The declared signature of an `extern fn`. Mirrors the canonical
/// types of its parameters + return, with no body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternSig {
    pub params: Vec<Type>,
    pub ret: Type,
    /// `None` for a host (Rust) extern resolved from the process extern
    /// registry; `Some(lib)` for a real C symbol resolved from the
    /// shared library `lib` via dlopen/dlsym (no `Thread*` arg, plain C
    /// ABI). See [`crate::surface::SurfaceDefKind::Extern`].
    pub library: Option<String>,
    /// `true` if the C function is variadic (declared with a trailing
    /// `...`). `params` then holds only the fixed leading parameters;
    /// call sites may pass additional trailing C-scalar args.
    pub variadic: bool,
}

/// Per-module metadata for the `at(...)` builtin lowering.
///
/// `at(node, thunk)` returns `Result<Int, Failure>` (without generics
/// yet, this is the user-defined `Result` enum with `Ok(Int)` and
/// `Err(Failure)`). The runtime needs to know the variant indices and
/// the heap-object shapes to allocate when it constructs the return
/// value — those depend on how the user wrote the enums in this
/// module, so the resolver records them here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtBinding {
    pub result_hash: Hash,
    pub failure_hash: Hash,
    pub node_hash: Hash,
    pub ok_variant_index: u32,
    pub err_variant_index: u32,
    pub unreachable_variant_index: u32,
    pub crashed_variant_index: u32,
    pub code_missing_variant_index: u32,
    pub cancelled_variant_index: u32,
    /// `Failure::TimedOut(Node)` is OPTIONAL: declared, it receives
    /// deadline expirations distinctly; absent, timeouts surface as
    /// `Unreachable`. Optional so pre-deadline 4-variant `Failure`
    /// enums keep working (their content hash is unchanged).
    pub timed_out_variant_index: Option<u32>,
    /// `enum DecodeError { TypeMismatch, Malformed }` — used by the
    /// checked `decode::<T>` to build its `Err`. Optional: `at()`-only
    /// modules need not declare it.
    pub decode_error_hash: Option<Hash>,
    pub decode_type_mismatch_index: u32,
    pub decode_malformed_index: u32,
}

impl ResolvedModule {
    /// Look up a resolved def by its surface name.
    pub fn get(&self, name: &str) -> Option<&ResolvedDef> {
        self.defs.iter().find(|d| d.name == name)
    }
}

// =============================================================================
// Resolution entry point
// =============================================================================

/// Prefix used to encode the `Result` enum's content hash into the
/// `at(...)` builtin name. The hashed result enum varies per module
/// (users author their own), so we stamp it into the `BuiltinRef` so
/// the typechecker and codegen can recover the return type without
/// threading the binding through every visitor.
pub const AT_BUILTIN_PREFIX: &str = "core/net.at#";

/// Parse a `core/net.at#<result_hex>[#<failure_hex>]` builtin name.
/// Returns the `Result` hash plus, for the generic form, the
/// `Failure` hash too. `None` if the name doesn't have that shape.
/// Map a surface builtin name to its `core/*` name when it may be used as
/// a first-class VALUE (a bare reference, not a call). These mirror the
/// entries handled in call position inside `resolve_expr`. Both concrete
/// builtins (string/bytes/ptr/...) AND generic ones (`array_*`, `atom_swap`,
/// whose signatures carry a `TypeVar`) qualify: a generic value reference
/// resolves to a polymorphic `fn(...)` type and is eta-expanded into an
/// adapter closure that composes through the uniform closure ABI, with the
/// concrete (un)boxing settled at the instantiation boundary.
///
/// Call-site-special builtins (`at`) and ones whose value form is pointless
/// (`panic` returns `Never`, `gc_collect` is a nullary side-effect) are
/// absent, so a bare reference to them stays an `UnknownName`. The actual
/// signature is read from [`crate::typecheck::builtin_signature`], keeping
/// types single-sourced.
fn value_position_builtin(name: &str) -> Option<&'static str> {
    Some(match name {
        "string_len" => "core/string.len",
        "string_eq" => "core/string.eq",
        "string_concat" => "core/string.concat",
        "bytes_new" => "core/bytes.new",
        "bytes_len" => "core/bytes.len",

        "bytes_slice" => "core/bytes.slice",
        "bytes_concat" => "core/bytes.concat",
        "bytes_from_string" => "core/bytes.from_string",
        "string_from_bytes" => "core/string.from_bytes",
        "int_to_float" => "core/f64.of_int",
        "float_to_int" => "core/f64.to_int",
        "float_sqrt" => "core/f64.sqrt",
        "ptr_null" => "core/ptr.null",
        "ptr_is_null" => "core/ptr.is_null",
        "ptr_add" => "core/ptr.add",
        "ptr_read_u8" => "core/ptr.read_u8",
        "ptr_write_u8" => "core/ptr.write_u8",
        "ptr_read_i64" => "core/ptr.read_i64",
        "ptr_write_i64" => "core/ptr.write_i64",
        "ptr_read_ptr" => "core/ptr.read_ptr",
        "ptr_write_ptr" => "core/ptr.write_ptr",
        "ptr_to_int" => "core/ptr.to_int",
        "int_to_ptr" => "core/ptr.from_int",
        "bit_and" => "core/i64.and",
        "bit_or" => "core/i64.or",
        "bit_xor" => "core/i64.xor",
        "bit_shl" => "core/i64.shl",
        "bit_shr" => "core/i64.shr",
        // Generic builtins (signatures carry `TypeVar(0)`).
        "array_new" => "core/array.new",
        "array_len" => "core/array.len",

        "atom_new" => "core/atom.new",
        "atom_load" => "core/atom.load",
        "atom_swap" => "core/atom.swap",
        "thread_spawn" => "core/thread.spawn",
        "thread_spawn_shared" => "core/thread.spawn_shared",
        "thread_join" => "core/thread.join",
        _ => return None,
    })
}

/// Prefix for the asynchronous twin of `at(...)`: `at_async(node, thunk)`
/// ships the thunk from a background thread and returns a
/// `ThreadHandle<Result<T, Failure>>` immediately; the existing `join`
/// awaits it. Same embedded hashes as the `at` name.
pub const AT_ASYNC_BUILTIN_PREFIX: &str = "core/net.at_async#";

/// `true` for any at-family builtin name (`core/net.at#...` or
/// `core/net.at_async#...`). The hash-rename, dependency-index, and
/// slice walkers treat both identically: the names embed the same
/// `Result`/`Failure` hashes.
pub fn is_at_family_builtin(name: &str) -> bool {
    name.starts_with(AT_BUILTIN_PREFIX) || name.starts_with(AT_ASYNC_BUILTIN_PREFIX)
}

/// Parse an at-family builtin name (`core/net.at#...` or
/// `core/net.at_async#...`), returning the embedded hashes.
pub fn parse_at_builtin_name(name: &str) -> Option<(Hash, Option<Hash>)> {
    let body = name
        .strip_prefix(AT_BUILTIN_PREFIX)
        .or_else(|| name.strip_prefix(AT_ASYNC_BUILTIN_PREFIX))?;
    let parts: Vec<&str> = body.split('#').collect();
    let parse_hex = |s: &str| -> Option<Hash> {
        if s.len() != Hash::SIZE * 2 {
            return None;
        }
        let mut out = [0u8; Hash::SIZE];
        for (i, b) in out.iter_mut().enumerate() {
            let chunk = &s[i * 2..i * 2 + 2];
            *b = u8::from_str_radix(chunk, 16).ok()?;
        }
        Some(Hash(out))
    };
    match parts.as_slice() {
        [r] => Some((parse_hex(r)?, None)),
        [r, f] => Some((parse_hex(r)?, Some(parse_hex(f)?))),
        _ => None,
    }
}

/// Parse a `core/wire.decode#<expected>#<result>#<okint>#<decode_error>`
/// builtin name, returning `(result_enum_hash, decode_error_hash)`. Lets the
/// typechecker rebuild the call's `Result<T, DecodeError>` type (with `T` left
/// as a wildcard — the runtime does the real shape check). Returns `None` for
/// any name that is not a decode builtin or is malformed.
pub fn parse_decode_builtin_name(name: &str) -> Option<(Hash, Hash)> {
    let body = name.strip_prefix("core/wire.decode#")?;
    let parts: Vec<&str> = body.split('#').collect();
    let parse_hex = |s: &str| -> Option<Hash> {
        if s.len() != Hash::SIZE * 2 {
            return None;
        }
        let mut out = [0u8; Hash::SIZE];
        for (i, b) in out.iter_mut().enumerate() {
            *b = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).ok()?;
        }
        Some(Hash(out))
    };
    match parts.as_slice() {
        [_expected, result, _okint, decode_error] => {
            Some((parse_hex(result)?, parse_hex(decode_error)?))
        }
        _ => None,
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }
    s
}

/// Externally-supplied top-level bindings for `resolve_module_with_env`.
///
/// Built from an existing codebase (its namespace + type cache) so an
/// edited definition can reference any already-stored top-level def by
/// name. Each map is keyed by surface name. Empty by default, in which
/// case `resolve_module_with_env` behaves identically to `resolve_module`.
#[derive(Debug, Clone, Default)]
pub struct ExternalEnv {
    /// surface name -> (content hash, param types, return type)
    pub fns: HashMap<String, (Hash, Vec<Type>, Type)>,
    /// surface name -> (content hash, type-param names, fields)
    pub structs: HashMap<String, (Hash, Vec<String>, Vec<(String, Type)>)>,
    /// surface name -> (content hash, type-param names, variants)
    pub enums: HashMap<String, (Hash, Vec<String>, Vec<(String, Option<Type>)>)>,
}

impl ExternalEnv {
    pub fn new() -> Self {
        ExternalEnv::default()
    }
}

/// Resolve a module with no external environment (the original entry
/// point). Equivalent to `resolve_module_with_env(m, &ExternalEnv::new())`.
pub fn resolve_module(m: &Module) -> Result<ResolvedModule, ResolveError> {
    resolve_module_with_env(m, &ExternalEnv::new())
}

pub fn resolve_module_with_env(
    m: &Module,
    env: &ExternalEnv,
) -> Result<ResolvedModule, ResolveError> {
    // Detect duplicate names.
    let mut seen: HashMap<&str, Span> = HashMap::new();
    for d in &m.defs {
        if let Some(&first) = seen.get(d.name.as_str()) {
            return Err(ResolveError::DuplicateDef {
                name: d.name.clone(),
                first_span: first,
                second_span: d.span,
            });
        }
        seen.insert(&d.name, d.span);
    }

    // ---- Pass 1: resolve struct defs first ----
    //
    // Struct field types can reference Int / fn / other (previously
    // declared) structs. Fn defs can reference both. Doing structs
    // first guarantees fn bodies see all struct types in scope.
    //
    // No mutual recursion across structs in v1 (forward refs to later
    // structs error cleanly).
    let mut structs: HashMap<String, StructInfo> = HashMap::new();
    let mut enums: HashMap<String, EnumInfo> = HashMap::new();
    let mut variants: HashMap<String, VariantBinding> = HashMap::new();
    let mut top: HashMap<String, TopBinding> = HashMap::new();
    let mut out: Vec<ResolvedDef> = Vec::with_capacity(m.defs.len());

    // ---- Pass 0 (edit layer): seed externally-provided top-level bindings ----
    //
    // When re-resolving a single edited definition against an existing
    // codebase (see `edit::update`), the new source references OTHER
    // already-stored top-level defs by name. Those names are not present
    // in `m`, so we pre-populate the resolver's scopes from `env` (built
    // from the codebase namespace + type cache). References then resolve to
    // `TopRef(hash)` / `TypeRef` exactly as if the referenced defs had been
    // earlier in source order. A name that ALSO appears in `m` is shadowed
    // by the in-module pass (which overwrites the seeded entry).
    for (name, (hash, params, ret)) in &env.fns {
        structs.remove(name);
        enums.remove(name);
        top.insert(
            name.clone(),
            TopBinding {
                hash: *hash,
                ty: Type::FnType {
                    params: params.clone(),
                    ret: Box::new(ret.clone()),
                },
                kind: TopKind::Def,
            },
        );
    }
    for (name, (hash, type_params, fields)) in &env.structs {
        structs.insert(
            name.clone(),
            StructInfo {
                hash: *hash,
                type_params: type_params.clone(),
                fields: fields.clone(),
            },
        );
        top.insert(
            name.clone(),
            TopBinding {
                hash: *hash,
                ty: Type::TypeRef(*hash),
                kind: TopKind::Def,
            },
        );
    }
    for (name, (hash, type_params, vs)) in &env.enums {
        enums.insert(
            name.clone(),
            EnumInfo {
                hash: *hash,
                type_params: type_params.clone(),
                variants: vs.clone(),
            },
        );
        top.insert(
            name.clone(),
            TopBinding {
                hash: *hash,
                ty: Type::TypeRef(*hash),
                kind: TopKind::Def,
            },
        );
        for (vname, _payload) in vs.iter() {
            variants.insert(
                vname.clone(),
                VariantBinding {
                    enum_surface_name: name.clone(),
                },
            );
        }
    }


    // ---- Pass 1 (combined): struct + enum defs with SCC ----
    //
    // Both kinds are processed together so types in a cycle (a struct
    // whose field is itself, or two structs that reference each
    // other, etc.) can hash correctly. The pipeline mirrors the fn
    // SCC pass (see Pass 2 below):
    //
    //   Phase A. Register every type's name in `pending_types` with a
    //            per-module idx. Insert placeholder StructInfo /
    //            EnumInfo entries so name resolution succeeds in
    //            phase B.
    //   Phase B. Resolve each type's body (fields / variants). Names
    //            of in-pending types become `Type::SelfRef(idx)` via
    //            the `pending_types` parameter to `resolve_type_inner`.
    //   Phase C. Build a type→type reference graph from each body's
    //            `Type::SelfRef`s.
    //   Phase D. Tarjan SCC.
    //   Phase E. Per SCC (in dependency-first order): build canonical
    //            body (SelfRef-local for in-SCC, TopRef for resolved
    //            deps), hash, build stored body (SelfRef-local →
    //            TopRef with the just-computed hashes), update
    //            structs/enums maps, add to `top` and `out`.

    struct PendingType {
        sd_idx: usize,
        type_params: Vec<String>,
        is_enum: bool,
    }

    // Phase A: register every type name.
    let mut pending_types: HashMap<String, u32> = HashMap::new();
    let mut pending: Vec<PendingType> = Vec::new();
    for (sd_idx, sd) in m.defs.iter().enumerate() {
        let (type_params, is_enum) = match &sd.kind {
            SurfaceDefKind::Struct { type_params, .. } => (type_params.clone(), false),
            SurfaceDefKind::Enum { type_params, .. } => (type_params.clone(), true),
            _ => continue,
        };
        if pending_types.contains_key(&sd.name) {
            return Err(ResolveError::DuplicateDef {
                name: sd.name.clone(),
                first_span: m.defs[pending[pending_types[&sd.name] as usize].sd_idx]
                    .span,
                second_span: sd.span,
            });
        }
        let idx = pending.len() as u32;
        pending_types.insert(sd.name.clone(), idx);
        pending.push(PendingType {
            sd_idx,
            type_params,
            is_enum,
        });
        // Placeholder StructInfo / EnumInfo so phase B's
        // `resolve_type_inner` can fall through `structs` / `enums`
        // lookups without finding stale entries — pending takes
        // priority in `resolve_type_inner`.
        if is_enum {
            enums.insert(
                sd.name.clone(),
                EnumInfo {
                    hash: Hash([0; 32]),
                    type_params: pending[idx as usize].type_params.clone(),
                    variants: Vec::new(),
                },
            );
        } else {
            structs.insert(
                sd.name.clone(),
                StructInfo {
                    hash: Hash([0; 32]),
                    type_params: pending[idx as usize].type_params.clone(),
                    fields: Vec::new(),
                },
            );
        }
    }

    // Phase B: resolve each type's body.
    let mut pending_bodies: Vec<PendingTypeBody> = Vec::with_capacity(pending.len());
    for p in &pending {
        let sd = &m.defs[p.sd_idx];
        let body = match &sd.kind {
            SurfaceDefKind::Struct { fields, .. } => {
                let fields_canon: Result<Vec<(String, Type)>, _> = fields
                    .iter()
                    .map(|(n, t)| {
                        Ok((
                            n.clone(),
                            resolve_type_inner(
                                t,
                                &structs,
                                &enums,
                                &p.type_params,
                                Some(&pending_types),
                            )?,
                        ))
                    })
                    .collect();
                PendingTypeBody::Struct(fields_canon?)
            }
            SurfaceDefKind::Enum { variants: vs, .. } => {
                let mut variants_canon = Vec::with_capacity(vs.len());
                for (vname, payload_ty) in vs {
                    let payload_canon = match payload_ty {
                        None => None,
                        Some(t) => Some(resolve_type_inner(
                            t,
                            &structs,
                            &enums,
                            &p.type_params,
                            Some(&pending_types),
                        )?),
                    };
                    variants_canon.push((vname.clone(), payload_canon));
                }
                PendingTypeBody::Enum(variants_canon)
            }
            _ => unreachable!("pending only contains struct/enum"),
        };
        pending_bodies.push(body);
    }

    // Phase C: build adjacency graph from each body's SelfRefs.
    let n_types = pending.len();
    let mut adj_types: Vec<Vec<u32>> = vec![Vec::new(); n_types];
    for (i, body) in pending_bodies.iter().enumerate() {
        let mut refs = std::collections::BTreeSet::new();
        match body {
            PendingTypeBody::Struct(fields) => {
                for (_, t) in fields {
                    collect_type_self_refs(t, &mut refs);
                }
            }
            PendingTypeBody::Enum(vs) => {
                for (_, p) in vs {
                    if let Some(t) = p {
                        collect_type_self_refs(t, &mut refs);
                    }
                }
            }
        }
        for r in refs {
            adj_types[i].push(r);
        }
    }

    // Phase D: Tarjan SCC (re-uses the same algorithm as the fn SCC).
    let type_sccs = tarjan_scc(&adj_types);

    // Phase E: process each SCC.
    let mut type_real_hash: Vec<Option<Hash>> = vec![None; n_types];
    for scc in type_sccs {
        // Sort by source position for canonical SCC-local indices.
        let mut scc_sorted = scc.clone();
        scc_sorted.sort_by_key(|&g| pending[g as usize].sd_idx);
        let local_of: HashMap<u32, u32> = scc_sorted
            .iter()
            .enumerate()
            .map(|(local, &global)| (global, local as u32))
            .collect();

        // Phase E.1: build canonical body for hashing. SelfRef(global)
        // → SelfRef(local) for in-SCC members; → TopRef(real_hash)
        // for deps already resolved in earlier SCCs.
        let canonical_bodies: Vec<PendingTypeBody> = scc_sorted
            .iter()
            .map(|&global_idx| {
                rewrite_type_body_for_canonical(
                    &pending_bodies[global_idx as usize],
                    &local_of,
                    &type_real_hash,
                )
            })
            .collect();

        // Phase E.2: compute the hash of each member.
        let mut scc_hashes: Vec<Hash> = Vec::with_capacity(scc_sorted.len());
        for (local, &global_idx) in scc_sorted.iter().enumerate() {
            let p = &pending[global_idx as usize];
            let def = match &canonical_bodies[local] {
                PendingTypeBody::Struct(fields) => Def::Struct {
                    type_params: p.type_params.len() as u32,
                    fields: fields.clone(),
                },
                PendingTypeBody::Enum(vs) => Def::Enum {
                    type_params: p.type_params.len() as u32,
                    variants: vs.clone(),
                },
            };
            scc_hashes.push(Hash::of_bytes(&encode_def(&def)));
        }

        // Phase E.3: build stored Def + update maps + emit.
        for (local, &global_idx) in scc_sorted.iter().enumerate() {
            let stored_body = rewrite_type_body_selfref_to_topref(
                &canonical_bodies[local],
                &scc_hashes,
            );
            let p = &pending[global_idx as usize];
            let sd = &m.defs[p.sd_idx];
            let hash = scc_hashes[local];
            type_real_hash[global_idx as usize] = Some(hash);

            let def = match &stored_body {
                PendingTypeBody::Struct(fields) => {
                    structs.insert(
                        sd.name.clone(),
                        StructInfo {
                            hash,
                            type_params: p.type_params.clone(),
                            fields: fields.clone(),
                        },
                    );
                    Def::Struct {
                        type_params: p.type_params.len() as u32,
                        fields: fields.clone(),
                    }
                }
                PendingTypeBody::Enum(vs) => {
                    enums.insert(
                        sd.name.clone(),
                        EnumInfo {
                            hash,
                            type_params: p.type_params.clone(),
                            variants: vs.clone(),
                        },
                    );
                    // Register each variant for constructor lookup.
                    for (vname, _payload_ty) in vs.iter() {
                        if let Some(existing) = variants.get(vname) {
                            return Err(ResolveError::DuplicateDef {
                                name: vname.clone(),
                                first_span: m
                                    .defs
                                    .iter()
                                    .find(|d| d.name == existing.enum_surface_name)
                                    .map(|d| d.span)
                                    .unwrap_or(sd.span),
                                second_span: sd.span,
                            });
                        }
                        variants.insert(
                            vname.clone(),
                            VariantBinding {
                                enum_surface_name: sd.name.clone(),
                            },
                        );
                    }
                    Def::Enum {
                        type_params: p.type_params.len() as u32,
                        variants: vs.clone(),
                    }
                }
            };
            top.insert(
                sd.name.clone(),
                TopBinding {
                    hash,
                    ty: Type::TypeRef(hash),
                    kind: TopKind::Def,
                },
            );
            out.push(ResolvedDef {
                name: sd.name.clone(),
                hash,
                def,
            });
        }
    }

    // ---- Pass 1c: extern fn declarations ----
    //
    // Externs aren't content-addressed (no Def is emitted), but their
    // signature must be in scope when fn bodies resolve. We register
    // each extern's `BuiltinRef("ext/<name>")` placeholder in `top`
    // so a call-site Var lookup finds it as a callable. The signature
    // travels to typecheck / codegen via `ResolvedModule.externs`.
    let mut externs: HashMap<String, ExternSig> = HashMap::new();
    for sd in &m.defs {
        if let SurfaceDefKind::Extern { params, ret, library, variadic } = &sd.kind {
            if externs.contains_key(&sd.name) {
                return Err(ResolveError::DuplicateDef {
                    name: sd.name.clone(),
                    first_span: m
                        .defs
                        .iter()
                        .find(|d| d.name == sd.name)
                        .map(|d| d.span)
                        .unwrap_or(sd.span),
                    second_span: sd.span,
                });
            }
            let params_canon: Result<Vec<Type>, _> = params
                .iter()
                .map(|(_, t)| resolve_type(t, &structs, &enums, &[]))
                .collect();
            let params_canon = params_canon?;
            let ret_canon = resolve_type(ret, &structs, &enums, &[])?;
            externs.insert(
                sd.name.clone(),
                ExternSig {
                    params: params_canon.clone(),
                    ret: ret_canon.clone(),
                    library: library.clone(),
                    variadic: *variadic,
                },
            );
            // Stash the surface name in `top` so the Var lookup in
            // body resolution finds it as a callable. We use a
            // sentinel hash that nothing else will produce; the
            // resolver's Var branch detects this and rewrites to
            // BuiltinRef instead of TopRef.
            top.insert(
                sd.name.clone(),
                TopBinding {
                    hash: Hash([0; 32]),
                    ty: Type::FnType {
                        params: params_canon,
                        ret: Box::new(ret_canon),
                    },
                    kind: TopKind::Extern,
                },
            );
        }
    }

    // ---- Pass 2: resolve fn defs (with SCC-based recursion support) ----
    //
    // The pipeline:
    //   2a. Resolve each fn def's signature (params + ret). Register
    //       a placeholder `TopBinding { kind: PendingFn { idx } }` so
    //       Var lookups for in-module fn names emit `SelfRef(idx)`.
    //   2b. Resolve each fn body. References to other in-module fns
    //       are SelfRef-by-global-idx; references to anything else
    //       (struct, enum, extern) work as before.
    //   2c. Build the call graph from each body's `SelfRef`s.
    //   2d. Tarjan SCC → list of SCCs in reverse-DFS (dependencies-first)
    //       order, which is exactly the order we process them in.
    //   2e. For each SCC:
    //       - Sort SCC members by source position for stable canonical
    //         SelfRef indices.
    //       - Build each member's *canonical* body: SelfRefs to other
    //         SCC members get rewritten to SelfRef(local_idx); SelfRefs
    //         to already-resolved fns (earlier SCCs) get rewritten to
    //         TopRef(real_hash). The canonical body is what we HASH.
    //       - Compute each member's content hash.
    //       - Build the *stored* body: SelfRef(local_idx) → TopRef
    //         (using the just-computed hashes). The stored body is what
    //         typecheck + codegen consume; SelfRef only appears during
    //         hashing.
    let mut at_binding: Option<AtBinding> = None;

    // A pending hashable code item. Fns and `state` bindings share one
    // index space so a fn that references a state (and a state init that
    // references a fn) hash together in one SCC, in dependency order.
    enum PendingKind {
        Fn {
            type_params: Vec<String>,
            params_canon: Vec<Type>,
            ret_canon: Type,
            is_local: bool,
        },
        State {
            ty: Type,
        },
    }
    struct Pending {
        sd_idx: usize,
        kind: PendingKind,
    }

    // 2a. Register every fn + state under a placeholder binding so body
    //     resolution can emit SelfRef / StateSelfRef for forward refs.
    let mut pendings: Vec<Pending> = Vec::new();
    for (sd_idx, sd) in m.defs.iter().enumerate() {
        match &sd.kind {
            SurfaceDefKind::Fn {
                type_params,
                params,
                ret,
                is_local,
                ..
            } => {
                let params_canon: Result<Vec<Type>, _> = params
                    .iter()
                    .map(|(_, t)| resolve_type(t, &structs, &enums, type_params))
                    .collect();
                let params_canon = params_canon?;
                let ret_canon = resolve_type(ret, &structs, &enums, type_params)?;
                let idx = pendings.len() as u32;
                top.insert(
                    sd.name.clone(),
                    TopBinding {
                        hash: Hash([0; 32]), // placeholder
                        ty: Type::FnType {
                            params: params_canon.clone(),
                            ret: Box::new(ret_canon.clone()),
                        },
                        kind: TopKind::PendingFn { idx },
                    },
                );
                pendings.push(Pending {
                    sd_idx,
                    kind: PendingKind::Fn {
                        type_params: type_params.clone(),
                        params_canon,
                        ret_canon,
                        is_local: *is_local,
                    },
                });
            }
            SurfaceDefKind::State { ty, .. } => {
                let ty_canon = resolve_type(ty, &structs, &enums, &[])?;
                let idx = pendings.len() as u32;
                top.insert(
                    sd.name.clone(),
                    TopBinding {
                        hash: Hash([0; 32]), // placeholder
                        ty: ty_canon.clone(),
                        kind: TopKind::PendingState { idx },
                    },
                );
                pendings.push(Pending {
                    sd_idx,
                    kind: PendingKind::State { ty: ty_canon },
                });
            }
            _ => {}
        }
    }

    // 2b. Resolve each item's body. A fn body resolves with its params
    //     in scope; a state init resolves with an empty environment
    //     (state initializers take no parameters).
    let mut bodies: Vec<Expr> = Vec::with_capacity(pendings.len());
    for p in &pendings {
        let sd = &m.defs[p.sd_idx];
        let body_canon = match (&p.kind, &sd.kind) {
            (
                PendingKind::Fn {
                    type_params,
                    params_canon,
                    ..
                },
                SurfaceDefKind::Fn {
                    params,
                    body: src_body,
                    ..
                },
            ) => {
                let mut env: Vec<(String, Type)> = params
                    .iter()
                    .zip(params_canon.iter())
                    .map(|((n, _), t)| (n.clone(), t.clone()))
                    .collect();
                resolve_expr(
                    src_body,
                    &mut env,
                    &top,
                    &structs,
                    &enums,
                    &variants,
                    &mut at_binding,
                    type_params,
                    &sd.name,
                )?
            }
            (PendingKind::State { .. }, SurfaceDefKind::State { init, .. }) => {
                let mut env: Vec<(String, Type)> = Vec::new();
                resolve_expr(
                    init,
                    &mut env,
                    &top,
                    &structs,
                    &enums,
                    &variants,
                    &mut at_binding,
                    &[],
                    &sd.name,
                )?
            }
            _ => unreachable!("pending kind must match its surface def kind"),
        };
        bodies.push(body_canon);
    }

    // 2b'. Type-directed array_new specialization: where the context
    // (def return type, struct field, call argument) pins an array's
    // element type to a scalar, `core/array.new` becomes the unboxed
    // `core/array.new_prim`. Runs before SCC hashing so the choice is
    // part of each def's content hash.
    {
        let fields_by_struct: HashMap<Hash, Vec<Type>> = structs
            .values()
            .map(|si| {
                (
                    si.hash,
                    si.fields.iter().map(|(_, t)| t.clone()).collect(),
                )
            })
            .collect();
        let params_by_self: Vec<Vec<Type>> = pendings
            .iter()
            .map(|p| match &p.kind {
                PendingKind::Fn { params_canon, .. } => params_canon.clone(),
                PendingKind::State { .. } => Vec::new(),
            })
            .collect();
        let cx = ArraySpecCtx {
            fields_by_struct: &fields_by_struct,
            params_by_self: &params_by_self,
        };
        for (i, p) in pendings.iter().enumerate() {
            let expected = match &p.kind {
                PendingKind::Fn { ret_canon, .. } => ret_canon.clone(),
                PendingKind::State { ty } => ty.clone(),
            };
            bodies[i] = specialize_array_new(&bodies[i], Some(&expected), &cx);
        }
    }

    // 2c.
    let n = pendings.len();
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (i, body) in bodies.iter().enumerate() {
        let mut refs = std::collections::BTreeSet::new();
        collect_self_refs(body, &mut refs);
        for r in refs {
            adj[i].push(r);
        }
    }

    // 2d.
    let sccs = tarjan_scc(&adj);

    // 2e.
    let mut real_hash: Vec<Option<Hash>> = vec![None; n];
    for scc in sccs {
        // Stable canonical order within the SCC: by source position.
        let mut scc_sorted = scc.clone();
        scc_sorted.sort_by_key(|&g| pendings[g as usize].sd_idx);
        let local_of: HashMap<u32, u32> = scc_sorted
            .iter()
            .enumerate()
            .map(|(local, &global)| (global, local as u32))
            .collect();

        // Phase 1: build canonical (SelfRef + TopRef) body per member.
        let mut canonical_bodies: Vec<Expr> = Vec::with_capacity(scc_sorted.len());
        for &global_idx in &scc_sorted {
            let canon =
                rewrite_for_canonical(&bodies[global_idx as usize], &local_of, &real_hash);
            canonical_bodies.push(canon);
        }

        // Helper: build the canonical `Def` for a member from its body.
        let def_of = |p: &Pending, body: Expr| -> Def {
            match &p.kind {
                PendingKind::Fn {
                    type_params,
                    params_canon,
                    ret_canon,
                    is_local,
                } => Def::Fn {
                    is_local: *is_local,
                    type_params: type_params.len() as u32,
                    params: params_canon.clone(),
                    ret: ret_canon.clone(),
                    body,
                },
                PendingKind::State { ty } => Def::State {
                    ty: ty.clone(),
                    init: body,
                },
            }
        };

        // Phase 2: compute the content hash of each member.
        let mut scc_hashes: Vec<Hash> = Vec::with_capacity(scc_sorted.len());
        for (local, &global_idx) in scc_sorted.iter().enumerate() {
            let p = &pendings[global_idx as usize];
            let def = def_of(p, canonical_bodies[local].clone());
            scc_hashes.push(Hash::of_bytes(&encode_def(&def)));
        }

        // Phase 3: build the stored Def per member.
        for (local, &global_idx) in scc_sorted.iter().enumerate() {
            let stored_body =
                rewrite_selfref_to_topref(&canonical_bodies[local], &scc_hashes);
            let p = &pendings[global_idx as usize];
            let sd = &m.defs[p.sd_idx];
            let def = def_of(p, stored_body);
            let hash = scc_hashes[local];
            real_hash[global_idx as usize] = Some(hash);
            let (ty, kind) = match &p.kind {
                PendingKind::Fn {
                    params_canon,
                    ret_canon,
                    ..
                } => (
                    Type::FnType {
                        params: params_canon.clone(),
                        ret: Box::new(ret_canon.clone()),
                    },
                    TopKind::Def,
                ),
                PendingKind::State { ty } => (ty.clone(), TopKind::State),
            };
            top.insert(
                sd.name.clone(),
                TopBinding {
                    hash,
                    ty,
                    kind,
                },
            );
            out.push(ResolvedDef {
                name: sd.name.clone(),
                hash,
                def,
            });
        }
    }

    Ok(ResolvedModule {
        defs: out,
        at_binding,
        externs,
    })
}

// =============================================================================
// Local-name side-car capture
// =============================================================================

/// The author's original local names for every `fn` def in `m`, keyed by
/// the def's content hash, in **binder-push order**: fn parameters first,
/// then each `let` / lambda-parameter / match-binding name in the exact
/// pre-order the resolver pushes onto `env` (which is the same order the
/// `printer` pushes onto its binder stack). The `printer` replays this list
/// 1:1 against its binder stack to recover readable names.
///
/// This is a SIDE-CAR. It is NOT part of any canonical `Def` and never
/// enters a content hash. Renaming a local changes only the strings in this
/// vector; the def's hash is unaffected (the canonical AST is name-erased).
///
/// We resolve `m` to recover each fn's content hash, then re-derive the
/// names from the surface body. Struct/enum defs get no entry (no locals).
/// Returns an empty map if `m` has no fn defs.
pub fn local_names_for_module(
    m: &Module,
) -> Result<HashMap<Hash, Vec<String>>, ResolveError> {
    local_names_for_module_with_env(m, &ExternalEnv::new())
}

/// As [`local_names_for_module`] but resolving against an external env
/// (mirrors `resolve_module_with_env`), so an edited def that references
/// already-stored defs still hashes correctly.
pub fn local_names_for_module_with_env(
    m: &Module,
    env: &ExternalEnv,
) -> Result<HashMap<Hash, Vec<String>>, ResolveError> {
    let rm = resolve_module_with_env(m, env)?;

    // Surface metadata needed to traverse bodies in canonical order:
    //   - which constructor names are NULLARY variants (those are NOT
    //     match-bindings — the printer pushes no binder for them);
    //   - each struct's declared field order (the canonical StructNew
    //     reorders surface fields to declaration order, and the printer
    //     prints/visits them in that order).
    let mut nullary_variants: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    let mut struct_field_order: HashMap<String, Vec<String>> = HashMap::new();
    for sd in &m.defs {
        match &sd.kind {
            SurfaceDefKind::Enum { variants, .. } => {
                for (vname, payload) in variants {
                    if payload.is_none() {
                        nullary_variants.insert(vname.clone());
                    }
                }
            }
            SurfaceDefKind::Struct { fields, .. } => {
                struct_field_order.insert(
                    sd.name.clone(),
                    fields.iter().map(|(n, _)| n.clone()).collect(),
                );
            }
            _ => {}
        }
    }
    let meta = NameMeta {
        nullary_variants,
        struct_field_order,
    };

    // Map fn surface name -> hash from the resolved module.
    let mut name_to_hash: HashMap<&str, Hash> = HashMap::new();
    for rd in &rm.defs {
        name_to_hash.insert(rd.name.as_str(), rd.hash);
    }

    let mut out: HashMap<Hash, Vec<String>> = HashMap::new();
    for sd in &m.defs {
        if let SurfaceDefKind::Fn { params, body, .. } = &sd.kind {
            let Some(&hash) = name_to_hash.get(sd.name.as_str()) else {
                // Resolved set didn't include this fn (shouldn't happen,
                // but skip rather than mis-key).
                continue;
            };
            let mut names: Vec<String> = Vec::new();
            // Parameters are the outermost binders, pushed in order.
            for (pname, _) in params {
                names.push(pname.clone());
            }
            collect_body_local_names(body, &meta, &mut names);
            out.insert(hash, names);
        }
    }
    Ok(out)
}

/// Surface metadata for the canonical-order body traversal.
struct NameMeta {
    nullary_variants: std::collections::HashSet<String>,
    struct_field_order: HashMap<String, Vec<String>>,
}

/// Append the binder names introduced inside `e`, in the exact pre-order
/// the resolver pushes them onto `env` (and the `printer` pushes onto its
/// binder stack). Free-variable occurrences introduce nothing.
fn collect_body_local_names(e: &SurfaceExpr, meta: &NameMeta, out: &mut Vec<String>) {
    match e {
        SurfaceExpr::IntLit { .. }
        | SurfaceExpr::FloatLit { .. }
        | SurfaceExpr::BoolLit { .. }
        | SurfaceExpr::StringLit { .. }
        | SurfaceExpr::Var { .. }
        | SurfaceExpr::VariantRef { .. } => {}

        SurfaceExpr::Call { callee, args, .. } => {
            collect_body_local_names(callee, meta, out);
            for a in args {
                collect_body_local_names(a, meta, out);
            }
        }
        SurfaceExpr::BinOp { left, right, .. } => {
            collect_body_local_names(left, meta, out);
            collect_body_local_names(right, meta, out);
        }
        SurfaceExpr::UnaryOp { operand, .. } => {
            collect_body_local_names(operand, meta, out);
        }
        SurfaceExpr::Block { stmts, tail, .. } => {
            // `let v = e;` visits `e` (binder not yet in scope) then pushes
            // the name; `defer e;` visits `e` and pushes nothing. The tail
            // is visited last with every let-binder in scope.
            for s in stmts {
                match s {
                    SurfaceStmt::Let { name, value, .. } => {
                        collect_body_local_names(value, meta, out);
                        out.push(name.clone());
                    }
                    SurfaceStmt::Defer { expr, .. } => {
                        collect_body_local_names(expr, meta, out);
                    }
                }
            }
            collect_body_local_names(tail, meta, out);
        }
        SurfaceExpr::Lambda { params, body, .. } => {
            for (pname, _) in params {
                out.push(pname.clone());
            }
            collect_body_local_names(body, meta, out);
        }
        SurfaceExpr::StructLit { type_name, fields, .. } => {
            // Canonical StructNew reorders fields to DECLARATION order; the
            // printer visits them in that order, so we must too. If the
            // struct's field order is unknown (e.g. an external struct),
            // fall back to surface order — binder order only matters when a
            // field value introduces a binder, which is rare.
            match meta.struct_field_order.get(type_name) {
                Some(decl_order) => {
                    for fname in decl_order {
                        if let Some((_, fexpr)) =
                            fields.iter().find(|(n, _)| n == fname)
                        {
                            collect_body_local_names(fexpr, meta, out);
                        }
                    }
                }
                None => {
                    for (_, fexpr) in fields {
                        collect_body_local_names(fexpr, meta, out);
                    }
                }
            }
        }
        SurfaceExpr::FieldAccess { base, .. } => {
            collect_body_local_names(base, meta, out);
        }
        SurfaceExpr::Try { expr, .. } => {
            collect_body_local_names(expr, meta, out);
        }
        SurfaceExpr::Match { scrutinee, arms, .. } => {
            collect_body_local_names(scrutinee, meta, out);
            for arm in arms {
                collect_pattern_binder_names(&arm.pattern, meta, out);
                collect_body_local_names(&arm.body, meta, out);
            }
        }
        SurfaceExpr::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            collect_body_local_names(cond, meta, out);
            collect_body_local_names(then_branch, meta, out);
            collect_body_local_names(else_branch, meta, out);
        }
    }
}

/// Append the binder names a match pattern introduces, in pattern-traversal
/// order, skipping nullary-variant idents (which are constructor patterns,
/// not bindings) — mirrors [`walk_pattern_bindings`] / the printer's
/// `print_pattern`.
fn collect_pattern_binder_names(
    p: &SurfacePattern,
    meta: &NameMeta,
    out: &mut Vec<String>,
) {
    match p {
        SurfacePattern::Wildcard { .. } => {}
        SurfacePattern::Ident { name, .. } => {
            // A bare ident that names a nullary variant is a constructor
            // pattern (no binding). Anything else binds.
            if meta.nullary_variants.contains(name) {
                return;
            }
            out.push(name.clone());
        }
        // Bare ctor patterns are rejected by resolve_pattern; no binders.
        SurfacePattern::Ctor { .. } => {}
        SurfacePattern::QualifiedCtor { payload, .. } => {
            if let Some(sub) = payload {
                collect_pattern_binder_names(sub, meta, out);
            }
        }
    }
}

/// Walk `e` and collect every `Expr::SelfRef(idx)` index into `out`.
fn collect_self_refs(e: &Expr, out: &mut std::collections::BTreeSet<u32>) {
    match e {
        Expr::SelfRef(i) | Expr::StateSelfRef(i) => {
            out.insert(*i);
        }
        Expr::Call(callee, args) => {
            collect_self_refs(callee, out);
            for a in args {
                collect_self_refs(a, out);
            }
        }
        Expr::Lambda { body, .. } => collect_self_refs(body, out),
        Expr::Let { value, body } => {
            collect_self_refs(value, out);
            collect_self_refs(body, out);
        }
        Expr::StructNew { fields, .. } => {
            for f in fields {
                collect_self_refs(f, out);
            }
        }
        Expr::Field { base, .. } => collect_self_refs(base, out),
        Expr::EnumNew { payload, .. } => {
            if let Some(p) = payload {
                collect_self_refs(p, out);
            }
        }
        Expr::Match { scrutinee, arms } => {
            collect_self_refs(scrutinee, out);
            for arm in arms {
                collect_self_refs(&arm.body, out);
            }
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } => {
            collect_self_refs(cond, out);
            collect_self_refs(then_branch, out);
            collect_self_refs(else_branch, out);
        }
        Expr::Try { expr, .. } => collect_self_refs(expr, out),
        Expr::Defer { cleanup, body } => {
            collect_self_refs(cleanup, out);
            collect_self_refs(body, out);
        }
        Expr::IntLit(_)
        | Expr::FloatLit(_)
        | Expr::BoolLit(_)
        | Expr::StringLit(_)
        | Expr::LocalVar(_)
        | Expr::TopRef(_)
        | Expr::StateRef(_)
        | Expr::BuiltinRef(_) => {}
    }
}

/// Rewrite each `SelfRef(global_idx)` in `e`:
///   - if `global_idx` is in this SCC (per `local_of`): stays as
///     `SelfRef(local_idx)` — load-bearing for canonical hashing.
///   - else: becomes `TopRef(real_hash[global_idx])`, which must be
///     populated because earlier (dependency) SCCs were already
///     processed.
/// Whether a type is `Array<scalar>` — the contexts where `array_new`
/// specializes to the UNBOXED `core/array.new_prim` (raw i64/f64 slots,
/// no per-element box allocation).
fn is_prim_array_type(t: &Type) -> bool {
    match t {
        Type::Apply(head, args) => {
            matches!(head.as_ref(), Type::Builtin(n) if n == "Array")
                && matches!(
                    args.first(),
                    Some(Type::Builtin(n)) if n == "Int" || n == "Float" || n == "Bool"
                )
        }
        _ => false,
    }
}

/// Context for [`specialize_array_new`]: where expected types come from.
struct ArraySpecCtx<'a> {
    /// Struct hash → declared field types, for struct-literal positions.
    fields_by_struct: &'a HashMap<Hash, Vec<Type>>,
    /// Module-local def (global pending index) → declared param types,
    /// for call-argument positions. Bodies at this stage reference
    /// module-local defs via `SelfRef(global)`.
    params_by_self: &'a [Vec<Type>],
}

/// Type-directed specialization of `core/array.new`: rewrite it to
/// `core/array.new_prim` wherever the CONTEXT pins the element type to a
/// scalar (Int/Float/Bool). Same precedent as the `at#`/`decode#` baking —
/// the choice is part of the definition, so it lands in the content hash.
///
/// `expected` flows down through tail positions (let bodies, if branches,
/// match arms, defer bodies) and into typed holes (struct-literal fields,
/// call arguments of known defs). Creation sites with no contextual type
/// (e.g. an unannotated intermediate) keep the boxed representation —
/// that's a performance default, never a correctness issue: every array
/// accessor handles both representations at runtime.
fn specialize_array_new(e: &Expr, expected: Option<&Type>, cx: &ArraySpecCtx) -> Expr {
    let down = |sub: &Expr| specialize_array_new(sub, None, cx);
    match e {
        Expr::Call(callee, args) => {
            if let Expr::BuiltinRef(name) = callee.as_ref() {
                if name == "core/array.new"
                    && expected.map(is_prim_array_type).unwrap_or(false)
                {
                    return Expr::Call(
                        Box::new(Expr::BuiltinRef("core/array.new_prim".to_owned())),
                        args.iter().map(down).collect(),
                    );
                }
            }
            // Known callee: push its declared param types into the args.
            let param_tys: Option<&Vec<Type>> = match callee.as_ref() {
                Expr::SelfRef(g) => cx.params_by_self.get(*g as usize),
                _ => None,
            };
            let new_args = args
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    let exp = param_tys.and_then(|ps| ps.get(i));
                    specialize_array_new(a, exp, cx)
                })
                .collect();
            Expr::Call(Box::new(down(callee)), new_args)
        }
        Expr::Let { value, body } => Expr::Let {
            value: Box::new(down(value)),
            body: Box::new(specialize_array_new(body, expected, cx)),
        },
        Expr::Defer { cleanup, body } => Expr::Defer {
            cleanup: Box::new(down(cleanup)),
            body: Box::new(specialize_array_new(body, expected, cx)),
        },
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } => Expr::If {
            cond: Box::new(down(cond)),
            then_branch: Box::new(specialize_array_new(then_branch, expected, cx)),
            else_branch: Box::new(specialize_array_new(else_branch, expected, cx)),
        },
        Expr::Match { scrutinee, arms } => Expr::Match {
            scrutinee: Box::new(down(scrutinee)),
            arms: arms
                .iter()
                .map(|arm| MatchArm {
                    pattern: arm.pattern.clone(),
                    body: specialize_array_new(&arm.body, expected, cx),
                })
                .collect(),
        },
        Expr::StructNew { struct_ref, fields } => {
            let field_tys = cx.fields_by_struct.get(struct_ref);
            Expr::StructNew {
                struct_ref: *struct_ref,
                fields: fields
                    .iter()
                    .enumerate()
                    .map(|(i, f)| {
                        let exp = field_tys.and_then(|ts| ts.get(i));
                        specialize_array_new(f, exp, cx)
                    })
                    .collect(),
            }
        }
        Expr::EnumNew {
            enum_ref,
            variant_index,
            payload,
        } => Expr::EnumNew {
            enum_ref: *enum_ref,
            variant_index: *variant_index,
            payload: payload.as_ref().map(|p| Box::new(down(p))),
        },
        Expr::Lambda { params, body } => Expr::Lambda {
            params: params.clone(),
            body: Box::new(down(body)),
        },
        Expr::Field {
            base,
            struct_ref,
            index,
        } => Expr::Field {
            base: Box::new(down(base)),
            struct_ref: *struct_ref,
            index: *index,
        },
        Expr::Try {
            expr,
            enum_ref,
            ok_index,
            err_index,
        } => Expr::Try {
            expr: Box::new(down(expr)),
            enum_ref: *enum_ref,
            ok_index: *ok_index,
            err_index: *err_index,
        },
        Expr::IntLit(_)
        | Expr::FloatLit(_)
        | Expr::BoolLit(_)
        | Expr::StringLit(_)
        | Expr::LocalVar(_)
        | Expr::TopRef(_)
        | Expr::SelfRef(_)
        | Expr::StateRef(_)
        | Expr::StateSelfRef(_)
        | Expr::BuiltinRef(_) => e.clone(),
    }
}

fn rewrite_for_canonical(
    e: &Expr,
    local_of: &HashMap<u32, u32>,
    real_hash: &[Option<Hash>],
) -> Expr {
    match e {
        Expr::SelfRef(global) => {
            if let Some(&local) = local_of.get(global) {
                Expr::SelfRef(local)
            } else {
                let h = real_hash[*global as usize]
                    .expect("dependency SCC must be processed before referent");
                Expr::TopRef(h)
            }
        }
        Expr::StateSelfRef(global) => {
            if let Some(&local) = local_of.get(global) {
                Expr::StateSelfRef(local)
            } else {
                let h = real_hash[*global as usize]
                    .expect("dependency SCC must be processed before referent");
                Expr::StateRef(h)
            }
        }
        Expr::Call(callee, args) => Expr::Call(
            Box::new(rewrite_for_canonical(callee, local_of, real_hash)),
            args.iter()
                .map(|a| rewrite_for_canonical(a, local_of, real_hash))
                .collect(),
        ),
        Expr::Lambda { params, body } => Expr::Lambda {
            params: params.clone(),
            body: Box::new(rewrite_for_canonical(body, local_of, real_hash)),
        },
        Expr::Let { value, body } => Expr::Let {
            value: Box::new(rewrite_for_canonical(value, local_of, real_hash)),
            body: Box::new(rewrite_for_canonical(body, local_of, real_hash)),
        },
        Expr::StructNew { struct_ref, fields } => Expr::StructNew {
            struct_ref: *struct_ref,
            fields: fields
                .iter()
                .map(|f| rewrite_for_canonical(f, local_of, real_hash))
                .collect(),
        },
        Expr::Field {
            base,
            struct_ref,
            index,
        } => Expr::Field {
            base: Box::new(rewrite_for_canonical(base, local_of, real_hash)),
            struct_ref: *struct_ref,
            index: *index,
        },
        Expr::EnumNew {
            enum_ref,
            variant_index,
            payload,
        } => Expr::EnumNew {
            enum_ref: *enum_ref,
            variant_index: *variant_index,
            payload: payload
                .as_ref()
                .map(|p| Box::new(rewrite_for_canonical(p, local_of, real_hash))),
        },
        Expr::Match { scrutinee, arms } => Expr::Match {
            scrutinee: Box::new(rewrite_for_canonical(scrutinee, local_of, real_hash)),
            arms: arms
                .iter()
                .map(|arm| MatchArm {
                    pattern: arm.pattern.clone(),
                    body: rewrite_for_canonical(&arm.body, local_of, real_hash),
                })
                .collect(),
        },
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } => Expr::If {
            cond: Box::new(rewrite_for_canonical(cond, local_of, real_hash)),
            then_branch: Box::new(rewrite_for_canonical(then_branch, local_of, real_hash)),
            else_branch: Box::new(rewrite_for_canonical(else_branch, local_of, real_hash)),
        },
        Expr::Try {
            expr,
            enum_ref,
            ok_index,
            err_index,
        } => Expr::Try {
            expr: Box::new(rewrite_for_canonical(expr, local_of, real_hash)),
            enum_ref: *enum_ref,
            ok_index: *ok_index,
            err_index: *err_index,
        },
        Expr::Defer { cleanup, body } => Expr::Defer {
            cleanup: Box::new(rewrite_for_canonical(cleanup, local_of, real_hash)),
            body: Box::new(rewrite_for_canonical(body, local_of, real_hash)),
        },
        Expr::IntLit(_)
        | Expr::FloatLit(_)
        | Expr::BoolLit(_)
        | Expr::StringLit(_)
        | Expr::LocalVar(_)
        | Expr::TopRef(_)
        | Expr::StateRef(_)
        | Expr::BuiltinRef(_) => e.clone(),
    }
}

/// Rewrite any remaining `SelfRef(local_idx)` in `e` to
/// `TopRef(scc_hashes[local_idx])` (and `StateSelfRef` to `StateRef`).
/// Used to produce the STORED form of an SCC member's body (the form
/// codegen + typecheck consume).
fn rewrite_selfref_to_topref(e: &Expr, scc_hashes: &[Hash]) -> Expr {
    match e {
        Expr::SelfRef(local) => Expr::TopRef(scc_hashes[*local as usize]),
        Expr::StateSelfRef(local) => Expr::StateRef(scc_hashes[*local as usize]),
        Expr::Call(callee, args) => Expr::Call(
            Box::new(rewrite_selfref_to_topref(callee, scc_hashes)),
            args.iter()
                .map(|a| rewrite_selfref_to_topref(a, scc_hashes))
                .collect(),
        ),
        Expr::Lambda { params, body } => Expr::Lambda {
            params: params.clone(),
            body: Box::new(rewrite_selfref_to_topref(body, scc_hashes)),
        },
        Expr::Let { value, body } => Expr::Let {
            value: Box::new(rewrite_selfref_to_topref(value, scc_hashes)),
            body: Box::new(rewrite_selfref_to_topref(body, scc_hashes)),
        },
        Expr::StructNew { struct_ref, fields } => Expr::StructNew {
            struct_ref: *struct_ref,
            fields: fields
                .iter()
                .map(|f| rewrite_selfref_to_topref(f, scc_hashes))
                .collect(),
        },
        Expr::Field {
            base,
            struct_ref,
            index,
        } => Expr::Field {
            base: Box::new(rewrite_selfref_to_topref(base, scc_hashes)),
            struct_ref: *struct_ref,
            index: *index,
        },
        Expr::EnumNew {
            enum_ref,
            variant_index,
            payload,
        } => Expr::EnumNew {
            enum_ref: *enum_ref,
            variant_index: *variant_index,
            payload: payload
                .as_ref()
                .map(|p| Box::new(rewrite_selfref_to_topref(p, scc_hashes))),
        },
        Expr::Match { scrutinee, arms } => Expr::Match {
            scrutinee: Box::new(rewrite_selfref_to_topref(scrutinee, scc_hashes)),
            arms: arms
                .iter()
                .map(|arm| MatchArm {
                    pattern: arm.pattern.clone(),
                    body: rewrite_selfref_to_topref(&arm.body, scc_hashes),
                })
                .collect(),
        },
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } => Expr::If {
            cond: Box::new(rewrite_selfref_to_topref(cond, scc_hashes)),
            then_branch: Box::new(rewrite_selfref_to_topref(then_branch, scc_hashes)),
            else_branch: Box::new(rewrite_selfref_to_topref(else_branch, scc_hashes)),
        },
        Expr::Try {
            expr,
            enum_ref,
            ok_index,
            err_index,
        } => Expr::Try {
            expr: Box::new(rewrite_selfref_to_topref(expr, scc_hashes)),
            enum_ref: *enum_ref,
            ok_index: *ok_index,
            err_index: *err_index,
        },
        Expr::Defer { cleanup, body } => Expr::Defer {
            cleanup: Box::new(rewrite_selfref_to_topref(cleanup, scc_hashes)),
            body: Box::new(rewrite_selfref_to_topref(body, scc_hashes)),
        },
        Expr::IntLit(_)
        | Expr::FloatLit(_)
        | Expr::BoolLit(_)
        | Expr::StringLit(_)
        | Expr::LocalVar(_)
        | Expr::TopRef(_)
        | Expr::StateRef(_)
        | Expr::BuiltinRef(_) => e.clone(),
    }
}

/// Body of a struct or enum during the pass-1 SCC pipeline. Carries
/// the SelfRef-laden fields/variants between phase B (resolution)
/// and phase E (hashing + storage).
enum PendingTypeBody {
    Struct(Vec<(String, Type)>),
    Enum(Vec<(String, Option<Type>)>),
}

/// Walk `t` and collect every `Type::SelfRef(idx)` index into `out`.
fn collect_type_self_refs(t: &Type, out: &mut std::collections::BTreeSet<u32>) {
    match t {
        Type::SelfRef(i) => {
            out.insert(*i);
        }
        Type::Apply(head, args) => {
            collect_type_self_refs(head, out);
            for a in args {
                collect_type_self_refs(a, out);
            }
        }
        Type::FnType { params, ret } => {
            for p in params {
                collect_type_self_refs(p, out);
            }
            collect_type_self_refs(ret, out);
        }
        Type::Builtin(_) | Type::TypeRef(_) | Type::TypeVar(_) => {}
    }
}

/// Rewrite `Type::SelfRef(global_idx)` in `t`:
///   - if `global_idx` is in this SCC (per `local_of`): becomes
///     `Type::SelfRef(local_idx)` — the canonical form used for hashing.
///   - else: becomes `Type::TypeRef(real_hash[global_idx])`.
fn rewrite_type_for_canonical(
    t: &Type,
    local_of: &HashMap<u32, u32>,
    real_hash: &[Option<Hash>],
) -> Type {
    match t {
        Type::SelfRef(global) => {
            if let Some(&local) = local_of.get(global) {
                Type::SelfRef(local)
            } else {
                let h = real_hash[*global as usize].expect(
                    "dependency type SCC must be processed before referent",
                );
                Type::TypeRef(h)
            }
        }
        Type::Apply(head, args) => Type::Apply(
            Box::new(rewrite_type_for_canonical(head, local_of, real_hash)),
            args.iter()
                .map(|a| rewrite_type_for_canonical(a, local_of, real_hash))
                .collect(),
        ),
        Type::FnType { params, ret } => Type::FnType {
            params: params
                .iter()
                .map(|p| rewrite_type_for_canonical(p, local_of, real_hash))
                .collect(),
            ret: Box::new(rewrite_type_for_canonical(ret, local_of, real_hash)),
        },
        Type::Builtin(_) | Type::TypeRef(_) | Type::TypeVar(_) => t.clone(),
    }
}

/// Rewrite any remaining `Type::SelfRef(local)` in `t` to
/// `Type::TypeRef(scc_hashes[local])`. The stored form contains no
/// SelfRefs — typecheck + codegen never need to handle them.
fn rewrite_type_selfref_to_topref(t: &Type, scc_hashes: &[Hash]) -> Type {
    match t {
        Type::SelfRef(local) => Type::TypeRef(scc_hashes[*local as usize]),
        Type::Apply(head, args) => Type::Apply(
            Box::new(rewrite_type_selfref_to_topref(head, scc_hashes)),
            args.iter()
                .map(|a| rewrite_type_selfref_to_topref(a, scc_hashes))
                .collect(),
        ),
        Type::FnType { params, ret } => Type::FnType {
            params: params
                .iter()
                .map(|p| rewrite_type_selfref_to_topref(p, scc_hashes))
                .collect(),
            ret: Box::new(rewrite_type_selfref_to_topref(ret, scc_hashes)),
        },
        Type::Builtin(_) | Type::TypeRef(_) | Type::TypeVar(_) => t.clone(),
    }
}

/// Apply `rewrite_type_for_canonical` to every Type in a
/// `PendingTypeBody`. Returns a new body with the same shape.
#[allow(non_local_definitions)] // PendingTypeBody is a local enum in resolve_module
fn rewrite_type_body_for_canonical(
    body: &PendingTypeBody,
    local_of: &HashMap<u32, u32>,
    real_hash: &[Option<Hash>],
) -> PendingTypeBody {
    use PendingTypeBody::*;
    match body {
        Struct(fields) => Struct(
            fields
                .iter()
                .map(|(n, t)| {
                    (
                        n.clone(),
                        rewrite_type_for_canonical(t, local_of, real_hash),
                    )
                })
                .collect(),
        ),
        Enum(vs) => Enum(
            vs.iter()
                .map(|(n, p)| {
                    (
                        n.clone(),
                        p.as_ref()
                            .map(|t| rewrite_type_for_canonical(t, local_of, real_hash)),
                    )
                })
                .collect(),
        ),
    }
}

fn rewrite_type_body_selfref_to_topref(
    body: &PendingTypeBody,
    scc_hashes: &[Hash],
) -> PendingTypeBody {
    use PendingTypeBody::*;
    match body {
        Struct(fields) => Struct(
            fields
                .iter()
                .map(|(n, t)| (n.clone(), rewrite_type_selfref_to_topref(t, scc_hashes)))
                .collect(),
        ),
        Enum(vs) => Enum(
            vs.iter()
                .map(|(n, p)| {
                    (
                        n.clone(),
                        p.as_ref()
                            .map(|t| rewrite_type_selfref_to_topref(t, scc_hashes)),
                    )
                })
                .collect(),
        ),
    }
}

/// Tarjan's strongly-connected-components algorithm on a directed
/// adjacency list. Returns SCCs in **reverse DFS order**, which for
/// our purposes is **dependencies-first**: a node's deps appear in
/// earlier SCCs than the node itself. (Tarjan emits sinks first.)
pub fn tarjan_scc(adj: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let n = adj.len();
    let mut index_of: Vec<i32> = vec![-1; n];
    let mut lowlink: Vec<i32> = vec![0; n];
    let mut on_stack: Vec<bool> = vec![false; n];
    let mut stack: Vec<u32> = Vec::new();
    let mut index = 0i32;
    let mut sccs: Vec<Vec<u32>> = Vec::new();

    // Iterative DFS to avoid stack overflow on deeply mutually-rec code.
    // Each frame: (node, edge_cursor).
    fn strongconnect(
        v: u32,
        adj: &[Vec<u32>],
        index: &mut i32,
        index_of: &mut [i32],
        lowlink: &mut [i32],
        on_stack: &mut [bool],
        stack: &mut Vec<u32>,
        sccs: &mut Vec<Vec<u32>>,
    ) {
        // Use an explicit work stack mimicking recursion.
        // Each frame stores the node and the next adjacency index to process.
        let mut frames: Vec<(u32, usize)> = vec![(v, 0)];
        index_of[v as usize] = *index;
        lowlink[v as usize] = *index;
        *index += 1;
        stack.push(v);
        on_stack[v as usize] = true;
        while let Some(&(u, k)) = frames.last() {
            if k < adj[u as usize].len() {
                let w = adj[u as usize][k];
                // Bump cursor before recurse.
                frames.last_mut().unwrap().1 += 1;
                if index_of[w as usize] < 0 {
                    // recurse on w
                    index_of[w as usize] = *index;
                    lowlink[w as usize] = *index;
                    *index += 1;
                    stack.push(w);
                    on_stack[w as usize] = true;
                    frames.push((w, 0));
                } else if on_stack[w as usize] {
                    lowlink[u as usize] = lowlink[u as usize].min(index_of[w as usize]);
                }
            } else {
                // All successors visited — pop, propagate lowlink to parent.
                if lowlink[u as usize] == index_of[u as usize] {
                    let mut group = Vec::new();
                    loop {
                        let w = stack.pop().expect("non-empty stack");
                        on_stack[w as usize] = false;
                        group.push(w);
                        if w == u {
                            break;
                        }
                    }
                    sccs.push(group);
                }
                frames.pop();
                if let Some(&(parent, _)) = frames.last() {
                    lowlink[parent as usize] =
                        lowlink[parent as usize].min(lowlink[u as usize]);
                }
            }
        }
    }

    for v in 0..n as u32 {
        if index_of[v as usize] < 0 {
            strongconnect(
                v,
                adj,
                &mut index,
                &mut index_of,
                &mut lowlink,
                &mut on_stack,
                &mut stack,
                &mut sccs,
            );
        }
    }
    sccs
}

/// Build an `AtBinding` from the user's `Result` / `Failure` / `Node`
/// defs in scope, or report what's missing / mis-shaped. Called the
/// first time a function body references `at(...)`; the resulting
/// binding is cached on the resolved module.
fn build_at_binding(
    structs: &HashMap<String, StructInfo>,
    enums: &HashMap<String, EnumInfo>,
    span: Span,
) -> Result<AtBinding, ResolveError> {
    let node = structs.get("Node").ok_or(ResolveError::AtRequiresBinding {
        missing: "struct Node".to_owned(),
        span,
    })?;
    let result = enums.get("Result").ok_or(ResolveError::AtRequiresBinding {
        missing: "enum Result".to_owned(),
        span,
    })?;
    let failure = enums.get("Failure").ok_or(ResolveError::AtRequiresBinding {
        missing: "enum Failure".to_owned(),
        span,
    })?;

    // `at(...)` requires the generic form: `enum Result<T, E> { Ok(T), Err(E) }`.
    //
    // The wire protocol ships every closure return as a heap pointer
    // (uniform closure ABI), so the Ok payload slot must be a pointer
    // slot, which means a TypeVar. A monomorphic `Result { Ok(Int), ... }`
    // would put a raw i64 in the payload slot and `build_ok` couldn't
    // write a heap pointer into it. We used to accept both shapes
    // back when at() returned Int only; that path is gone.
    let ok_idx = find_variant(result, "Ok").ok_or(ResolveError::AtBindingShape {
        what: "enum Result must declare variant `Ok`".to_owned(),
        span,
    })?;
    let err_idx = find_variant(result, "Err").ok_or(ResolveError::AtBindingShape {
        what: "enum Result must declare variant `Err`".to_owned(),
        span,
    })?;
    if result.type_params.len() != 2 {
        return Err(ResolveError::AtBindingShape {
            what: "Result must be generic: `enum Result<T, E> { Ok(T), Err(E) }`"
                .to_owned(),
            span,
        });
    }
    match &result.variants[ok_idx as usize].1 {
        Some(Type::TypeVar(0)) => {}
        _ => {
            return Err(ResolveError::AtBindingShape {
                what: "Result::Ok must carry the first type param (T) of Result<T, E>"
                    .to_owned(),
                span,
            });
        }
    }
    match &result.variants[err_idx as usize].1 {
        Some(Type::TypeVar(1)) => {}
        _ => {
            return Err(ResolveError::AtBindingShape {
                what: "Result::Err must carry the second type param (E) of Result<T, E>"
                    .to_owned(),
                span,
            });
        }
    }

    // Validate Failure shape: must have Unreachable/Crashed/CodeMissing/Cancelled,
    // each with a Node payload.
    let variant_idx = |name: &str| -> Result<u32, ResolveError> {
        let idx = find_variant(failure, name).ok_or_else(|| ResolveError::AtBindingShape {
            what: format!("enum Failure must declare variant `{}`", name),
            span,
        })?;
        match &failure.variants[idx as usize].1 {
            Some(Type::TypeRef(h)) if *h == node.hash => Ok(idx),
            _ => Err(ResolveError::AtBindingShape {
                what: format!(
                    "Failure::{} must carry a Node payload ({}(Node))",
                    name, name
                ),
                span,
            }),
        }
    };
    let unreachable_idx = variant_idx("Unreachable")?;
    let crashed_idx = variant_idx("Crashed")?;
    let code_missing_idx = variant_idx("CodeMissing")?;
    let cancelled_idx = variant_idx("Cancelled")?;
    // Optional: when declared, it must carry a Node like the others.
    let timed_out_idx = match find_variant(failure, "TimedOut") {
        Some(_) => Some(variant_idx("TimedOut")?),
        None => None,
    };

    // DecodeError is optional (only `decode::<T>` needs it). Discover it
    // opportunistically so the runtime can build `Err(TypeMismatch)` /
    // `Err(Malformed)` without a separate binding path.
    let (decode_error_hash, decode_tm_idx, decode_mf_idx) = match enums.get("DecodeError") {
        Some(de) => (
            Some(de.hash),
            find_variant(de, "TypeMismatch").unwrap_or(0),
            find_variant(de, "Malformed").unwrap_or(0),
        ),
        None => (None, 0, 0),
    };

    Ok(AtBinding {
        result_hash: result.hash,
        failure_hash: failure.hash,
        node_hash: node.hash,
        ok_variant_index: ok_idx,
        err_variant_index: err_idx,
        unreachable_variant_index: unreachable_idx,
        crashed_variant_index: crashed_idx,
        code_missing_variant_index: code_missing_idx,
        cancelled_variant_index: cancelled_idx,
        timed_out_variant_index: timed_out_idx,
        decode_error_hash,
        decode_type_mismatch_index: decode_tm_idx,
        decode_malformed_index: decode_mf_idx,
    })
}

fn find_variant(info: &EnumInfo, name: &str) -> Option<u32> {
    info.variants
        .iter()
        .position(|(n, _)| n == name)
        .map(|i| i as u32)
}

/// Resolve a qualified variant `Enum::Variant` to its enum content hash,
/// variant index, and declared payload type. Looking up by enum name (not by
/// the flat variant map) makes it robust to two enums sharing a variant name —
/// which is exactly why qualification is required.
fn lookup_qualified_variant(
    enums: &HashMap<String, EnumInfo>,
    enum_name: &str,
    variant_name: &str,
    span: Span,
) -> Result<(Hash, u32, Option<Type>), ResolveError> {
    let info = enums.get(enum_name).ok_or_else(|| ResolveError::UnknownType {
        name: enum_name.to_owned(),
        span,
    })?;
    match info.variants.iter().position(|(n, _)| n == variant_name) {
        Some(idx) => Ok((info.hash, idx as u32, info.variants[idx].1.clone())),
        None => Err(ResolveError::UnknownVariant {
            name: format!("{}::{}", enum_name, variant_name),
            span,
        }),
    }
}

/// Build the "bare variant must be qualified" error, suggesting the qualified
/// spelling using one enum that declares the variant.
fn bare_variant_error(
    variants: &HashMap<String, VariantBinding>,
    name: &str,
    span: Span,
) -> ResolveError {
    let hint = variants
        .get(name)
        .map(|v| format!("{}::{}", v.enum_surface_name, name))
        .unwrap_or_else(|| format!("Enum::{}", name));
    ResolveError::UnknownName {
        name: format!(
            "`{}` is an enum variant and must be qualified — write `{}`",
            name, hint
        ),
        span,
    }
}

#[derive(Clone)]
struct EnumInfo {
    hash: Hash,
    /// Names of declared type params, in declaration order. Empty for
    /// non-generic enums. Used to validate `Foo<...>` applications.
    type_params: Vec<String>,
    variants: Vec<(String, Option<Type>)>,
}

/// Records that a name is an enum variant. Now that variant construction must
/// be qualified, the resolver looks variants up by enum name (via the `enums`
/// map); this table only answers "is this bare name a variant?" and, if so,
/// which enum to suggest in the must-qualify error.
#[derive(Clone)]
struct VariantBinding {
    /// The surface name of the enclosing enum (for the must-qualify hint).
    enum_surface_name: String,
}

/// Top-level binding metadata: hash for `TopRef`, plus the canonical
/// type so expression-position references can be type-inferred (used
/// for field-access on call results, etc.).
#[derive(Clone)]
struct TopBinding {
    hash: Hash,
    ty: Type,
    kind: TopKind,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TopKind {
    /// A regular content-addressed def: hash is its real content hash.
    Def,
    /// A surface-level `extern fn` declaration: hash is meaningless
    /// (sentinel zero). Var references that resolve to an extern get
    /// rewritten to `BuiltinRef("ext/<name>")` by the Var-handling
    /// branch; call-sites still work because the runtime resolves the
    /// builtin by name at codegen time.
    Extern,
    /// An in-flight fn def in the current module. Used during pass 2
    /// resolution to support self- and mutual recursion: references
    /// to a `PendingFn` emit `Expr::SelfRef(idx)` instead of TopRef,
    /// and SCC analysis later substitutes the right hashes.
    /// `idx` is a per-module index over fn defs in source order.
    PendingFn { idx: u32 },
    /// A fully-resolved node `state` binding: `hash` is its content hash.
    /// Var references emit `Expr::StateRef(hash)` (load the live cell).
    State,
    /// An in-flight `state` binding in the current module (pass-2 SCC).
    /// References emit `Expr::StateSelfRef(idx)`; the SCC pass rewrites
    /// to `StateRef` after hashing. `idx` is the shared pending index
    /// (states and fns share one index space so they can hash in one SCC).
    PendingState { idx: u32 },
}

/// Resolver-side info about a struct in scope: its content hash and
/// field-name → (index, type) for fast lookup during field access /
/// struct-literal resolution.
#[derive(Clone)]
struct StructInfo {
    hash: Hash,
    /// Names of declared type params, in declaration order. Empty for
    /// non-generic structs.
    type_params: Vec<String>,
    fields: Vec<(String, Type)>,
}

impl StructInfo {
    fn find_field(&self, name: &str) -> Option<(u32, &Type)> {
        self.fields
            .iter()
            .enumerate()
            .find_map(|(i, (n, t))| (n == name).then_some((i as u32, t)))
    }
}

// =============================================================================
// Internals
// =============================================================================

fn resolve_struct_fields(
    fields: &[(String, SurfaceType)],
    structs: &HashMap<String, StructInfo>,
    enums: &HashMap<String, EnumInfo>,
    type_params: &[String],
) -> Result<Vec<(String, Type)>, ResolveError> {
    fields
        .iter()
        .map(|(n, t)| Ok((n.clone(), resolve_type(t, structs, enums, type_params)?)))
        .collect()
}

fn resolve_fn_def(
    sd: &SurfaceDef,
    top: &HashMap<String, TopBinding>,
    structs: &HashMap<String, StructInfo>,
    enums: &HashMap<String, EnumInfo>,
    variants: &HashMap<String, VariantBinding>,
    at_binding: &mut Option<AtBinding>,
) -> Result<Def, ResolveError> {
    let SurfaceDefKind::Fn {
        is_local,
        type_params,
        params,
        ret,
        body,
    } = &sd.kind
    else {
        unreachable!("only called for fn defs");
    };

    let params_canon: Result<Vec<Type>, _> = params
        .iter()
        .map(|(_, t)| resolve_type(t, structs, enums, type_params))
        .collect();
    let params_canon = params_canon?;
    let ret_canon = resolve_type(ret, structs, enums, type_params)?;

    let mut env: Vec<(String, Type)> = params
        .iter()
        .zip(params_canon.iter())
        .map(|((n, _), t)| (n.clone(), t.clone()))
        .collect();
    let body_canon = resolve_expr(
        body,
        &mut env,
        top,
        structs,
        enums,
        variants,
        at_binding,
        type_params,
        &sd.name,
    )?;

    Ok(Def::Fn {
        is_local: *is_local,
        type_params: type_params.len() as u32,
        params: params_canon,
        ret: ret_canon,
        body: body_canon,
    })
}

fn resolve_type(
    t: &SurfaceType,
    structs: &HashMap<String, StructInfo>,
    enums: &HashMap<String, EnumInfo>,
    type_params: &[String],
) -> Result<Type, ResolveError> {
    resolve_type_inner(t, structs, enums, type_params, None)
}

/// Like `resolve_type` but accepts an optional `pending_types` map
/// listing names of type defs whose hashes aren't yet known (because
/// the resolver hasn't finished processing them — used during the
/// combined struct+enum SCC pass). Hits on `pending_types` emit
/// `Type::SelfRef(idx)`; after SCC hashing these are rewritten to
/// `TypeRef(real_hash)` in stored bodies.
fn resolve_type_inner(
    t: &SurfaceType,
    structs: &HashMap<String, StructInfo>,
    enums: &HashMap<String, EnumInfo>,
    type_params: &[String],
    pending: Option<&HashMap<String, u32>>,
) -> Result<Type, ResolveError> {
    let resolve_name_head = |name: &str| -> Option<Type> {
        if let Some(p) = pending {
            if let Some(&idx) = p.get(name) {
                return Some(Type::SelfRef(idx));
            }
        }
        if let Some(info) = structs.get(name) {
            Some(Type::TypeRef(info.hash))
        } else if let Some(info) = enums.get(name) {
            Some(Type::TypeRef(info.hash))
        } else {
            None
        }
    };
    match t {
        SurfaceType::Named { name, span } => match name.as_str() {
            "Int" | "Bool" | "String" | "Float" | "Bytes" | "Ptr" => {
                Ok(Type::Builtin(name.clone()))
            }
            _ => {
                if let Some(idx) = type_params.iter().position(|p| p == name) {
                    return Ok(Type::TypeVar(idx as u32));
                }
                resolve_name_head(name).ok_or(ResolveError::UnknownType {
                    name: name.clone(),
                    span: *span,
                })
            }
        },
        SurfaceType::Applied {
            name,
            name_span,
            args,
            ..
        } => {
            // `Array<T>`, `Atom<T>`, and `ThreadHandle<T>` are builtin type
            // constructors (runtime-managed shapes), not user structs/enums.
            let head = if name == "Array" || name == "Atom" || name == "ThreadHandle" {
                Type::Builtin(name.clone())
            } else {
                resolve_name_head(name).ok_or(ResolveError::UnknownType {
                    name: name.clone(),
                    span: *name_span,
                })?
            };
            let args_canon: Result<Vec<Type>, _> = args
                .iter()
                .map(|a| resolve_type_inner(a, structs, enums, type_params, pending))
                .collect();
            Ok(Type::Apply(Box::new(head), args_canon?))
        }
        SurfaceType::FnType { params, ret, .. } => {
            let params_c: Result<Vec<Type>, _> = params
                .iter()
                .map(|p| resolve_type_inner(p, structs, enums, type_params, pending))
                .collect();
            let ret_c = resolve_type_inner(ret, structs, enums, type_params, pending)?;
            Ok(Type::FnType {
                params: params_c?,
                ret: Box::new(ret_c),
            })
        }
    }
}

fn resolve_expr(
    e: &SurfaceExpr,
    env: &mut Vec<(String, Type)>,
    top: &HashMap<String, TopBinding>,
    structs: &HashMap<String, StructInfo>,
    enums: &HashMap<String, EnumInfo>,
    variants: &HashMap<String, VariantBinding>,
    at_binding: &mut Option<AtBinding>,
    type_params: &[String],
    self_name: &str,
) -> Result<Expr, ResolveError> {
    Ok(resolve_expr_typed(
        e,
        env,
        top,
        structs,
        enums,
        variants,
        at_binding,
        type_params,
        self_name,
    )?
    .0)
}

/// Compute the runtime *identity hash* a decoded value of type `t` must
/// have, for the checked `decode::<T>` to accept it. This must match what
/// the runtime reads back from a decoded value's shape (see
/// `identity_hash_of` in `net.rs`): structs/enums by their content hash,
/// scalars/strings/arrays by their canonical shape hash. Closures and
/// type variables are rejected (a closure is identified by its code hash,
/// not a type, so it can't be checked structurally).
fn decode_expected_hash(t: &Type, span: Span) -> Result<Hash, ResolveError> {
    let unsupported = || ResolveError::UnknownName {
        name: "decode::<T>: T must be a concrete data type (Int / Bool / Float / \
               String / Bytes / Array / a struct / an enum). Closures and type \
               variables cannot be decoded by type."
            .to_owned(),
        span,
    };
    match t {
        Type::Builtin(n) if n == "Int" || n == "Bool" || n == "Float" => {
            Ok(crate::runtime::boxed_int_shape_hash())
        }
        Type::Builtin(n) if n == "String" || n == "Bytes" => {
            Ok(crate::runtime::string_shape_hash())
        }
        Type::Builtin(n) if n == "Array" => Ok(crate::runtime::array_shape_hash()),
        Type::TypeRef(h) => Ok(*h),
        Type::Apply(head, _) => match head.as_ref() {
            Type::Builtin(n) if n == "Array" => Ok(crate::runtime::array_shape_hash()),
            Type::TypeRef(h) => Ok(*h),
            _ => Err(unsupported()),
        },
        _ => Err(unsupported()),
    }
}

/// The checked indexing surface: which accessor, and the builtin trio it
/// expands over.
struct CheckedAccessor {
    /// `core/array.get` / `core/bytes.get` / ... — the trusted accessor.
    access: &'static str,
    /// `core/array.len` / `core/bytes.len` — total, used by the bounds
    /// check and the error payload.
    len: &'static str,
    /// Whether this is a SET (3 args, Ok payload is the set's Int).
    is_set: bool,
    /// Containers whose element type names the Ok payload ("Array");
    /// None means the element is always Int (Bytes).
    container: Option<&'static str>,
}

fn checked_accessor(name: &str) -> Option<CheckedAccessor> {
    Some(match name {
        "array_get" => CheckedAccessor {
            access: "core/array.get",
            len: "core/array.len",
            is_set: false,
            container: Some("Array"),
        },
        "array_set" => CheckedAccessor {
            access: "core/array.set",
            len: "core/array.len",
            is_set: true,
            container: Some("Array"),
        },
        "bytes_get" => CheckedAccessor {
            access: "core/bytes.get",
            len: "core/bytes.len",
            is_set: false,
            container: None,
        },
        "bytes_set" => CheckedAccessor {
            access: "core/bytes.set",
            len: "core/bytes.len",
            is_set: true,
            container: None,
        },
        _ => return None,
    })
}

/// Number of locals a canonical pattern binds (mirrors codegen's view).
fn pattern_binder_count(p: &crate::ast::Pattern) -> u32 {
    use crate::ast::Pattern;
    match p {
        Pattern::Wildcard => 0,
        Pattern::Var => 1,
        Pattern::Enum { payload, .. } => {
            payload.as_deref().map(pattern_binder_count).unwrap_or(0)
        }
    }
}

/// Shift every free de Bruijn local in `e` (index >= `cutoff`) up by
/// `by`. Used when a resolved expression is re-placed under freshly
/// introduced `let` binders by the checked-accessor expansion.
fn shift_free_locals(e: &Expr, by: u32, cutoff: u32) -> Expr {
    match e {
        Expr::LocalVar(i) => {
            if *i >= cutoff {
                Expr::LocalVar(i + by)
            } else {
                Expr::LocalVar(*i)
            }
        }
        Expr::Lambda { params, body } => Expr::Lambda {
            params: params.clone(),
            body: Box::new(shift_free_locals(body, by, cutoff + params.len() as u32)),
        },
        Expr::Let { value, body } => Expr::Let {
            value: Box::new(shift_free_locals(value, by, cutoff)),
            body: Box::new(shift_free_locals(body, by, cutoff + 1)),
        },
        Expr::Defer { cleanup, body } => Expr::Defer {
            cleanup: Box::new(shift_free_locals(cleanup, by, cutoff)),
            body: Box::new(shift_free_locals(body, by, cutoff)),
        },
        Expr::Call(callee, args) => Expr::Call(
            Box::new(shift_free_locals(callee, by, cutoff)),
            args.iter().map(|a| shift_free_locals(a, by, cutoff)).collect(),
        ),
        Expr::StructNew { struct_ref, fields } => Expr::StructNew {
            struct_ref: *struct_ref,
            fields: fields.iter().map(|f| shift_free_locals(f, by, cutoff)).collect(),
        },
        Expr::Field {
            base,
            struct_ref,
            index,
        } => Expr::Field {
            base: Box::new(shift_free_locals(base, by, cutoff)),
            struct_ref: *struct_ref,
            index: *index,
        },
        Expr::EnumNew {
            enum_ref,
            variant_index,
            payload,
        } => Expr::EnumNew {
            enum_ref: *enum_ref,
            variant_index: *variant_index,
            payload: payload
                .as_ref()
                .map(|p| Box::new(shift_free_locals(p, by, cutoff))),
        },
        Expr::Match { scrutinee, arms } => Expr::Match {
            scrutinee: Box::new(shift_free_locals(scrutinee, by, cutoff)),
            arms: arms
                .iter()
                .map(|arm| crate::ast::MatchArm {
                    pattern: arm.pattern.clone(),
                    body: shift_free_locals(
                        &arm.body,
                        by,
                        cutoff + pattern_binder_count(&arm.pattern),
                    ),
                })
                .collect(),
        },
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } => Expr::If {
            cond: Box::new(shift_free_locals(cond, by, cutoff)),
            then_branch: Box::new(shift_free_locals(then_branch, by, cutoff)),
            else_branch: Box::new(shift_free_locals(else_branch, by, cutoff)),
        },
        Expr::Try {
            expr,
            enum_ref,
            ok_index,
            err_index,
        } => Expr::Try {
            expr: Box::new(shift_free_locals(expr, by, cutoff)),
            enum_ref: *enum_ref,
            ok_index: *ok_index,
            err_index: *err_index,
        },
        Expr::IntLit(_)
        | Expr::FloatLit(_)
        | Expr::BoolLit(_)
        | Expr::StringLit(_)
        | Expr::TopRef(_)
        | Expr::SelfRef(_)
        | Expr::StateRef(_)
        | Expr::StateSelfRef(_)
        | Expr::BuiltinRef(_) => e.clone(),
    }
}

/// Whether a resolved operand is cheap, pure, and total — safe to
/// duplicate into both the bounds check and the access instead of
/// let-binding it (locals, literals, field chains over those). Keeps the
/// hot path of a checked access free of extra root-slot traffic.
fn duplicable_operand(e: &Expr) -> bool {
    match e {
        Expr::LocalVar(_)
        | Expr::IntLit(_)
        | Expr::FloatLit(_)
        | Expr::BoolLit(_) => true,
        Expr::Field { base, .. } => duplicable_operand(base),
        _ => false,
    }
}

/// Resolve a CHECKED accessor call — the public `array_get` /
/// `array_set` / `bytes_get` / `bytes_set` — into ordinary canonical
/// AST. Out-of-bounds is a VALUE, not a crash:
///
///   array_get(a, i)            : Result<T, IndexError>
///   array_get(a, i)?           : T        (fused: no Ok is ever built)
///
/// expands (with duplicable operands inlined, others let-bound) to
///
///   if i < 0      { Err(OutOfBounds(OobInfo { index: i, len: len(a) })) }
///   else if i >= len(a) { ...same Err... }
///   else          { Ok(<trusted access>) }
///
/// and, under `?` (`fused = true`), the Ok wrapper disappears — the else
/// arm IS the trusted access, and each Err arm is wrapped in `Try` so it
/// early-returns through the enclosing function exactly like any other
/// `?`. The raw `*_trusted` accessors (which ABORT on a violation — a
/// contract for callers that have proven their indices) remain available
/// for stdlib internals and hot kernels.
#[allow(clippy::too_many_arguments)]
fn resolve_checked_accessor(
    acc: &CheckedAccessor,
    args: &[SurfaceExpr],
    fused: bool,
    span: Span,
    env: &mut Vec<(String, Type)>,
    top: &HashMap<String, TopBinding>,
    structs: &HashMap<String, StructInfo>,
    enums: &HashMap<String, EnumInfo>,
    variants: &HashMap<String, VariantBinding>,
    at_binding: &mut Option<AtBinding>,
    type_params: &[String],
    self_name: &str,
) -> Result<(Expr, Type), ResolveError> {
    let want = if acc.is_set { 3 } else { 2 };
    if args.len() != want {
        return Err(ResolveError::UnknownName {
            name: format!(
                "checked accessor expects {} argument(s), got {}",
                want,
                args.len()
            ),
            span,
        });
    }
    // The error-shape types come from the stdlib by name.
    let missing = |what: &str| ResolveError::UnknownName {
        name: format!(
            "checked array/bytes accessors need the stdlib type `{}` \
             (declare it or include the stdlib)",
            what
        ),
        span,
    };
    let result_info = enums.get("Result").ok_or_else(|| missing("Result"))?.clone();
    let ixerr_info = enums
        .get("IndexError")
        .ok_or_else(|| missing("IndexError"))?
        .clone();
    let oob_info = structs
        .get("OobInfo")
        .ok_or_else(|| missing("OobInfo"))?
        .clone();
    let ok_index = find_variant(&result_info, "Ok").ok_or_else(|| missing("Result::Ok"))?;
    let err_index = find_variant(&result_info, "Err").ok_or_else(|| missing("Result::Err"))?;
    let oob_index =
        find_variant(&ixerr_info, "OutOfBounds").ok_or_else(|| missing("IndexError::OutOfBounds"))?;
    let uninit_index = find_variant(&ixerr_info, "Uninitialized")
        .ok_or_else(|| missing("IndexError::Uninitialized"))?;

    // Resolve the operands in the CURRENT env.
    let mut resolved: Vec<(Expr, Type)> = Vec::with_capacity(args.len());
    for a in args {
        resolved.push(resolve_expr_typed(
            a, env, top, structs, enums, variants, at_binding, type_params, self_name,
        )?);
    }
    // The container's statically-known element type, when there is one.
    let container_elem: Option<Type> = match acc.container {
        Some(container) => match &resolved[0].1 {
            Type::Apply(head, elt)
                if matches!(head.as_ref(), Type::Builtin(n) if n == container) =>
            {
                elt.first().cloned()
            }
            _ => None,
        },
        None => None,
    };
    // The Ok payload type: the container's element for arrays, Int for
    // bytes and for every set.
    let elem_ty = if acc.is_set {
        Type::Builtin("Int".to_owned())
    } else {
        container_elem
            .clone()
            .unwrap_or(Type::Builtin("Int".to_owned()))
    };
    // A statically-scalar array element pins the access to the SCALAR
    // builtin variant by NAME (`core/array.get_scalar` / `set_scalar`)
    // instead of leaving codegen to re-infer it per site. Same lowering,
    // but the name carries a guarantee the operand-rooting analysis can
    // use: a scalar GET's every path is allocation-free or aborts, so it
    // can never make an already-evaluated pointer operand go stale.
    let elem_scalar = matches!(
        &container_elem,
        Some(Type::Builtin(n)) if n == "Int" || n == "Float" || n == "Bool"
    );
    let access_name: String = if elem_scalar && acc.container == Some("Array") {
        if acc.is_set {
            "core/array.set_scalar".to_owned()
        } else {
            "core/array.get_scalar".to_owned()
        }
    } else {
        acc.access.to_owned()
    };

    // Plan the binders: non-duplicable operands are let-bound (outermost
    // first, in argument order — preserving evaluation order); duplicable
    // ones are inlined at every use, shifted past the introduced lets.
    let n_lets: u32 = resolved
        .iter()
        .filter(|(e, _)| !duplicable_operand(e))
        .count() as u32;
    let mut let_seen = 0u32;
    let operand_refs: Vec<Expr> = resolved
        .iter()
        .map(|(e, _)| {
            if duplicable_operand(e) {
                shift_free_locals(e, n_lets, 0)
            } else {
                // The let_seen-th let (0-based, outermost first) sits at
                // de Bruijn distance n_lets - 1 - let_seen from the core.
                let idx = n_lets - 1 - let_seen;
                let_seen += 1;
                Expr::LocalVar(idx)
            }
        })
        .collect();

    // The container length is LET-BOUND once per site (immediately inside
    // the operand lets) and referenced by the bounds check AND every error
    // arm. Before this, `len()` was re-called at each of those positions —
    // up to 4 full inline len-protocol derivations (null check + shape
    // discrimination + count load) per access site, which both re-derived
    // on the hot path and bloated the cold arms. The extra binder means
    // every operand ref used INSIDE the len-let shifts by 1; the len-let's
    // own value uses the unshifted refs.
    let len_value = Expr::Call(
        Box::new(Expr::BuiltinRef(acc.len.to_owned())),
        vec![operand_refs[0].clone()],
    );
    let inner_refs: Vec<Expr> = operand_refs
        .iter()
        .map(|e| shift_free_locals(e, 1, 0))
        .collect();
    let a_ref = &inner_refs[0];
    let i_ref = &inner_refs[1];
    let len_ref = || Expr::LocalVar(0);
    let err_expr = |variant: u32| -> Expr {
        Expr::EnumNew {
            enum_ref: result_info.hash,
            variant_index: err_index,
            payload: Some(Box::new(Expr::EnumNew {
                enum_ref: ixerr_info.hash,
                variant_index: variant,
                payload: Some(Box::new(Expr::StructNew {
                    struct_ref: oob_info.hash,
                    fields: vec![i_ref.clone(), len_ref()],
                })),
            })),
        }
    };
    // Under `?`, each Err arm early-returns through the enclosing fn.
    let err_arm = |variant: u32| -> Expr {
        if fused {
            Expr::Try {
                expr: Box::new(err_expr(variant)),
                enum_ref: result_info.hash,
                ok_index,
                err_index,
            }
        } else {
            err_expr(variant)
        }
    };
    let access_call = Expr::Call(
        Box::new(Expr::BuiltinRef(access_name)),
        inner_refs.clone(),
    );
    let mut ok_arm = if fused {
        access_call
    } else {
        Expr::EnumNew {
            enum_ref: result_info.hash,
            variant_index: ok_index,
            payload: Some(Box::new(access_call)),
        }
    };
    // Array GET: a slot that was never written has no value to return —
    // prim arrays are zero-filled (always initialized), boxed arrays
    // report `Err(Uninitialized)`. SETs initialize; bytes are zero-filled.
    if !acc.is_set && acc.container.is_some() {
        ok_arm = Expr::If {
            cond: Box::new(Expr::Call(
                Box::new(Expr::BuiltinRef("core/array.is_init".to_owned())),
                vec![a_ref.clone(), i_ref.clone()],
            )),
            then_branch: Box::new(ok_arm),
            else_branch: Box::new(err_arm(uninit_index)),
        };
    }
    let ult = |l: Expr, r: Expr| {
        Expr::Call(Box::new(Expr::BuiltinRef("core/i64.ult".to_owned())), vec![l, r])
    };
    // ONE unsigned compare is the whole range test: a negative index
    // wraps to a huge unsigned value and fails it, and `len >= 0` always
    // holds, so `(i as u64) < (len as u64)` ⟺ `0 <= i < len`. The
    // negative-index and too-big arms constructed the IDENTICAL
    // `Err(OutOfBounds(OobInfo { i, len }))` value, so folding them is
    // exactly semantics-preserving — and the unsigned predicate matches
    // the access protocol's internal bounds check, so GVN folds those
    // branches too. Ok-first: the happy path is the THEN branch all the
    // way down, which is also what codegen's `infer_type` reads for an
    // If — so the expansion's type (and box/unbox decisions downstream)
    // come from the access, not from the error arms.
    let mut core = Expr::If {
        cond: Box::new(ult(i_ref.clone(), len_ref())),
        then_branch: Box::new(ok_arm),
        else_branch: Box::new(err_arm(oob_index)),
    };
    // The len binder sits innermost, directly around the core.
    core = Expr::Let {
        value: Box::new(len_value),
        body: Box::new(core),
    };
    // Wrap the lets, innermost-last argument outward, shifting each
    // value past the lets already inside it.
    let let_values: Vec<&Expr> = resolved
        .iter()
        .filter(|(e, _)| !duplicable_operand(e))
        .map(|(e, _)| e)
        .collect();
    for (depth, v) in let_values.iter().enumerate().rev() {
        core = Expr::Let {
            value: Box::new(shift_free_locals(v, depth as u32, 0)),
            body: Box::new(core),
        };
    }

    let ty = if fused {
        elem_ty
    } else {
        Type::Apply(
            Box::new(Type::TypeRef(result_info.hash)),
            vec![elem_ty, Type::TypeRef(ixerr_info.hash)],
        )
    };
    Ok((core, ty))
}

/// Resolve `decode::<T>(bytes) -> Result<T, Int>`. `T` is named explicitly
/// (turbofish) because it cannot be inferred from a `Bytes`-only call. We
/// bake `T`'s expected identity hash into the builtin name; the runtime
/// (`ai_wire_decode_checked`) decodes, verifies the shape matches, and
/// builds the user's `Result` (`Ok(value)` / `Err(code)` where code 1 =
/// type mismatch, 2 = malformed). The `Result` enum is reused from the
/// `at()` binding.
#[allow(clippy::too_many_arguments)]
fn resolve_checked_decode(
    args: &[SurfaceExpr],
    type_args: &[SurfaceType],
    span: Span,
    env: &mut Vec<(String, Type)>,
    top: &HashMap<String, TopBinding>,
    structs: &HashMap<String, StructInfo>,
    enums: &HashMap<String, EnumInfo>,
    variants: &HashMap<String, VariantBinding>,
    at_binding: &mut Option<AtBinding>,
    type_params: &[String],
    self_name: &str,
) -> Result<(Expr, Type), ResolveError> {
    if args.len() != 1 {
        return Err(ResolveError::UnknownName {
            name: format!("decode expects 1 argument (Bytes), got {}", args.len()),
            span,
        });
    }
    if type_args.len() != 1 {
        return Err(ResolveError::UnknownName {
            name: format!(
                "decode::<T> expects exactly one explicit type argument, got {}",
                type_args.len()
            ),
            span,
        });
    }
    let t = resolve_type(&type_args[0], structs, enums, type_params)?;
    // A type variable (a generic `decode::<T>`, e.g. inside `at_via<T>`)
    // has no concrete identity hash; bake the all-zero "no-check" sentinel
    // so the runtime trusts the bytes (same model as `at`). Concrete types
    // keep their checked identity hash.
    let expected = match &t {
        Type::TypeVar(_) => Hash([0u8; 32]),
        _ => decode_expected_hash(&t, span)?,
    };
    // Ensure the Result/Failure binding exists so the runtime can build
    // the Result and the runtime-side binding gets installed at startup.
    let binding = match at_binding.as_ref() {
        Some(b) => b.clone(),
        None => {
            let b = build_at_binding(structs, enums, span)?;
            *at_binding = Some(b.clone());
            b
        }
    };
    let bytes_canon = resolve_expr(
        &args[0], env, top, structs, enums, variants, at_binding, type_params, self_name,
    )?;
    let decode_error_hash = binding.decode_error_hash.ok_or(ResolveError::AtRequiresBinding {
        missing: "enum DecodeError { TypeMismatch, Malformed }".to_owned(),
        span,
    })?;
    // Name layout:
    //   core/wire.decode#<expected>#<result_hash>#<okint>#<decode_error_hash>
    // expected: what the runtime checks against. result_hash + okint +
    // decode_error_hash: let codegen's infer_type rebuild
    // `Result<T, DecodeError>` (okint=1 when T is a boxed scalar, so the
    // Ok-arm binding unboxes; the Err payload is the DecodeError enum).
    let ok_scalar = matches!(&t, Type::Builtin(n) if n == "Int" || n == "Bool" || n == "Float");
    let builtin_name = format!(
        "core/wire.decode#{}#{}#{}#{}",
        hex_encode(expected.as_bytes()),
        hex_encode(binding.result_hash.as_bytes()),
        if ok_scalar { 1 } else { 0 },
        hex_encode(decode_error_hash.as_bytes()),
    );
    let ret_ty = Type::Apply(
        Box::new(Type::TypeRef(binding.result_hash)),
        vec![t, Type::TypeRef(decode_error_hash)],
    );
    Ok((
        Expr::Call(Box::new(Expr::BuiltinRef(builtin_name)), vec![bytes_canon]),
        ret_ty,
    ))
}

/// Resolve an expression, returning both the canonical form and its
/// inferred type. Used for field-access resolution (which needs to know
/// the base's struct type to compute the field index).
fn resolve_expr_typed(
    e: &SurfaceExpr,
    env: &mut Vec<(String, Type)>,
    top: &HashMap<String, TopBinding>,
    structs: &HashMap<String, StructInfo>,
    enums: &HashMap<String, EnumInfo>,
    variants: &HashMap<String, VariantBinding>,
    at_binding: &mut Option<AtBinding>,
    type_params: &[String],
    self_name: &str,
) -> Result<(Expr, Type), ResolveError> {
    match e {
        SurfaceExpr::IntLit { value, .. } => {
            Ok((Expr::IntLit(*value), Type::Builtin("Int".to_owned())))
        }
        SurfaceExpr::FloatLit { value, .. } => {
            Ok((Expr::FloatLit(*value), Type::Builtin("Float".to_owned())))
        }
        SurfaceExpr::BoolLit { value, .. } => {
            Ok((Expr::BoolLit(*value), Type::Builtin("Bool".to_owned())))
        }
        SurfaceExpr::StringLit { value, .. } => Ok((
            Expr::StringLit(value.clone()),
            Type::Builtin("String".to_owned()),
        )),

        SurfaceExpr::Var { name, span } => {
            if let Some((idx, ty)) = lookup_local(env, name) {
                return Ok((Expr::LocalVar(idx), ty));
            }
            // A bare name that is an enum variant must be qualified:
            // `Option::None`, not `None`. (Variant construction is always
            // qualified in this language.)
            if variants.contains_key(name) {
                return Err(bare_variant_error(variants, name, *span));
            }
            if let Some(b) = top.get(name) {
                match b.kind {
                    TopKind::Extern => {
                        return Err(ResolveError::UnknownName {
                            name: format!(
                                "extern `{}` must appear in call position",
                                name
                            ),
                            span: *span,
                        });
                    }
                    TopKind::PendingFn { idx } => {
                        // Recursive or forward reference to an
                        // in-flight fn def. The hash isn't known
                        // yet; emit `SelfRef(idx)` and let the SCC
                        // pass rewrite to TopRef after hashing.
                        return Ok((Expr::SelfRef(idx), b.ty.clone()));
                    }
                    TopKind::PendingState { idx } => {
                        // Reference to an in-flight `state` binding.
                        // Hash not known yet; emit `StateSelfRef(idx)`
                        // and let the SCC pass rewrite to StateRef.
                        return Ok((Expr::StateSelfRef(idx), b.ty.clone()));
                    }
                    TopKind::State => {
                        // Reference to an already-hashed node `state`:
                        // load the live cell on the executing node.
                        return Ok((Expr::StateRef(b.hash), b.ty.clone()));
                    }
                    TopKind::Def => {
                        return Ok((Expr::TopRef(b.hash), b.ty.clone()));
                    }
                }
            }
            if name == self_name {
                return Err(ResolveError::ForwardOrCyclicRef {
                    name: name.clone(),
                    span: *span,
                });
            }
            // A bare reference to a concrete core builtin (e.g. passing
            // `string_len` to a higher-order fn). These names are mapped
            // to `core/*` BuiltinRefs in call position above; here we
            // mirror that for VALUE position. The type is the builtin's
            // `fn(...) -> ...` signature. Only concrete builtins qualify —
            // generic ones (`array_get`, `atom_swap`) and call-site-special
            // ones (`at`) deliberately stay unresolved as bare values.
            if let Some(core) = value_position_builtin(name) {
                if let Some((params, ret)) =
                    crate::typecheck::builtin_signature(core)
                {
                    return Ok((
                        Expr::BuiltinRef(core.to_owned()),
                        Type::FnType {
                            params,
                            ret: Box::new(ret),
                        },
                    ));
                }
            }
            // The CHECKED accessors expand at their call sites (they are
            // not single builtins), so they have no value form — point
            // the user at a lambda or the trusted tier.
            if checked_accessor(name).is_some() {
                return Err(ResolveError::UnknownName {
                    name: format!(
                        "`{}` is a checked operation and can't be used as a bare \
                         value; wrap it in a lambda (e.g. `|a, i| {}(a, i)`)",
                        name, name
                    ),
                    span: *span,
                });
            }
            Err(ResolveError::UnknownName {
                name: name.clone(),
                span: *span,
            })
        }

        // `Enum::Variant` in value position — a nullary variant constructor.
        // (Payload variants appear as the callee of a `Call`, handled below.)
        SurfaceExpr::VariantRef { enum_name, variant_name, span } => {
            let (enum_ref, index, payload_ty) =
                lookup_qualified_variant(enums, enum_name, variant_name, *span)?;
            if payload_ty.is_some() {
                return Err(ResolveError::VariantArityMismatch {
                    variant: format!("{}::{}", enum_name, variant_name),
                    expected_payload: true,
                    span: *span,
                });
            }
            Ok((
                Expr::EnumNew {
                    enum_ref,
                    variant_index: index,
                    payload: None,
                },
                Type::TypeRef(enum_ref),
            ))
        }

        SurfaceExpr::Call { callee, args, type_args, span } => {
            // Generic checked decode: `decode::<T>(bytes) -> Result<T, Int>`.
            // T is named explicitly (turbofish) because it can't be
            // inferred from a Bytes-only argument. We bake T's expected
            // runtime identity-hash + an is-Int flag into the builtin name
            // (the same trick `at()` uses) so codegen can emit the check.
            if let SurfaceExpr::Var { name, .. } = callee.as_ref() {
                if name == "decode" {
                    return resolve_checked_decode(
                        args, type_args, *span, env, top, structs, enums, variants,
                        at_binding, type_params, self_name,
                    );
                }
            }
            // CHECKED indexing — the public array/bytes accessors. An
            // out-of-bounds index is a Result::Err VALUE (errors are
            // values); the raw `*_trusted` forms abort instead and stay
            // available for callers with proven indices.
            if let SurfaceExpr::Var { name, .. } = callee.as_ref() {
                if let Some(acc) = checked_accessor(name) {
                    return resolve_checked_accessor(
                        &acc, args, false, *span, env, top, structs, enums, variants,
                        at_binding, type_params, self_name,
                    );
                }
            }

            // Built-in string ops mapped to `core/string.*` so users
            // can write `string_len(s)`, `string_eq(a, b)`,
            // `string_concat(a, b)` without an FFI registry.
            if let SurfaceExpr::Var { name, .. } = callee.as_ref() {
                let (builtin, ret_ty) = match name.as_str() {
                    "string_len" => (
                        Some("core/string.len"),
                        Some(Type::Builtin("Int".to_owned())),
                    ),
                    "string_eq" => (
                        Some("core/string.eq"),
                        Some(Type::Builtin("Int".to_owned())),
                    ),
                    "string_concat" => (
                        Some("core/string.concat"),
                        Some(Type::Builtin("String".to_owned())),
                    ),
                    // Bytes ops. `Bytes` is a mutable, indexable byte
                    // buffer; see `core/bytes.*` in the typechecker.
                    "bytes_new" => (
                        Some("core/bytes.new"),
                        Some(Type::Builtin("Bytes".to_owned())),
                    ),
                    "bytes_len" => (
                        Some("core/bytes.len"),
                        Some(Type::Builtin("Int".to_owned())),
                    ),
                    "bytes_slice" => (
                        Some("core/bytes.slice"),
                        Some(Type::Builtin("Bytes".to_owned())),
                    ),
                    "bytes_concat" => (
                        Some("core/bytes.concat"),
                        Some(Type::Builtin("Bytes".to_owned())),
                    ),
                    "bytes_from_string" => (
                        Some("core/bytes.from_string"),
                        Some(Type::Builtin("Bytes".to_owned())),
                    ),
                    "string_from_bytes" => (
                        Some("core/string.from_bytes"),
                        Some(Type::Builtin("String".to_owned())),
                    ),
                    // Int <-> Float conversions.
                    "int_to_float" => (
                        Some("core/f64.of_int"),
                        Some(Type::Builtin("Float".to_owned())),
                    ),
                    "float_to_int" => (
                        Some("core/f64.to_int"),
                        Some(Type::Builtin("Int".to_owned())),
                    ),
                    "float_sqrt" => (
                        Some("core/f64.sqrt"),
                        Some(Type::Builtin("Float".to_owned())),
                    ),
                    // Raw-pointer / memory intrinsics. `Ptr` is an
                    // i64-represented raw address (non-GC). These are the
                    // irreducible primitives the C FFI is built on:
                    // allocate via libc `malloc` through an extern, then
                    // read/write the bytes with these.
                    "ptr_null" => (
                        Some("core/ptr.null"),
                        Some(Type::Builtin("Ptr".to_owned())),
                    ),
                    "ptr_is_null" => (
                        Some("core/ptr.is_null"),
                        Some(Type::Builtin("Int".to_owned())),
                    ),
                    "ptr_add" => (
                        Some("core/ptr.add"),
                        Some(Type::Builtin("Ptr".to_owned())),
                    ),
                    "ptr_read_u8" => (
                        Some("core/ptr.read_u8"),
                        Some(Type::Builtin("Int".to_owned())),
                    ),
                    "ptr_write_u8" => (
                        Some("core/ptr.write_u8"),
                        Some(Type::Builtin("Int".to_owned())),
                    ),
                    "ptr_read_i64" => (
                        Some("core/ptr.read_i64"),
                        Some(Type::Builtin("Int".to_owned())),
                    ),
                    "ptr_write_i64" => (
                        Some("core/ptr.write_i64"),
                        Some(Type::Builtin("Int".to_owned())),
                    ),
                    "ptr_read_ptr" => (
                        Some("core/ptr.read_ptr"),
                        Some(Type::Builtin("Ptr".to_owned())),
                    ),
                    "ptr_write_ptr" => (
                        Some("core/ptr.write_ptr"),
                        Some(Type::Builtin("Int".to_owned())),
                    ),
                    // Explicit reinterpret casts between a `Ptr` and its
                    // raw address as an `Int`. These are the DELIBERATE
                    // boundary that turns a non-shippable local address
                    // into plain data (and back) — the foundation of the
                    // wire-portable `RemotePtr`. `int_to_ptr` is unsafe by
                    // nature (it fabricates a pointer from an integer); use
                    // it only on an address you know is valid on THIS node.
                    "ptr_to_int" => (
                        Some("core/ptr.to_int"),
                        Some(Type::Builtin("Int".to_owned())),
                    ),
                    "int_to_ptr" => (
                        Some("core/ptr.from_int"),
                        Some(Type::Builtin("Ptr".to_owned())),
                    ),
                    // Bitwise ops on Int (base64 / CRC32 / zip / general
                    // bit manipulation). `bit_shr` is a logical shift.
                    "bit_and" => (Some("core/i64.and"), Some(Type::Builtin("Int".to_owned()))),
                    "bit_or" => (Some("core/i64.or"), Some(Type::Builtin("Int".to_owned()))),
                    "bit_xor" => (Some("core/i64.xor"), Some(Type::Builtin("Int".to_owned()))),
                    "bit_shl" => (Some("core/i64.shl"), Some(Type::Builtin("Int".to_owned()))),
                    "bit_shr" => (Some("core/i64.shr"), Some(Type::Builtin("Int".to_owned()))),
                    // Structural hash / equality of ANY value. The whole
                    // point of the language: every value has a canonical
                    // form, so every value is hashable. Generic in the arg
                    // type; result is always Int (`value_eq` returns 0/1).
                    // These are what let `HashMap<K, V>` key on any type.
                    "value_hash" => (Some("core/hash.value"), Some(Type::Builtin("Int".to_owned()))),
                    "value_eq" => (Some("core/value.eq"), Some(Type::Builtin("Int".to_owned()))),
                    _ => (None, None),
                };
                if let (Some(b), Some(rt)) = (builtin, ret_ty) {
                    let args_r: Result<Vec<Expr>, _> = args
                        .iter()
                        .map(|a| {
                            resolve_expr(
                                a, env, top, structs, enums, variants, at_binding,
                                type_params, self_name,
                            )
                        })
                        .collect();
                    return Ok((
                        Expr::Call(
                            Box::new(Expr::BuiltinRef(b.to_owned())),
                            args_r?,
                        ),
                        rt,
                    ));
                }
            }

            // Array builtins. `array_get`'s result type is the array's
            // element type T (pulled from its `Array<T>` instantiation)
            // so downstream `.field`/match on the result resolve. The
            // others have fixed result types. Int box/unbox of elements
            // is handled in codegen.
            if let SurfaceExpr::Var { name, .. } = callee.as_ref() {
                // Value-boundary builtins (the wire codec exposed to
                // ai-lang so a node loop can live in the language).
                let wire_builtin: Option<(&str, Type)> = match name.as_str() {
                    "wire_encode" => {
                        Some(("core/wire.encode", Type::Builtin("Bytes".to_owned())))
                    }
                    "wire_decode_int" => {
                        Some(("core/wire.decode_int", Type::Builtin("Int".to_owned())))
                    }
                    // Decode a shipped `fn(Int)->Int` updater (for a remote
                    // atom: the node `swap`s its own atom with it).
                    "wire_decode_fn1" => Some((
                        "core/wire.decode_fn1",
                        Type::FnType {
                            params: vec![Type::Builtin("Int".to_owned())],
                            ret: Box::new(Type::Builtin("Int".to_owned())),
                        },
                    )),
                    "wire_invoke" => {
                        Some(("core/wire.invoke", Type::Builtin("Bytes".to_owned())))
                    }
                    _ => None,
                };
                if let Some((builtin, ret_ty)) = wire_builtin {
                    let args_canon: Vec<Expr> = args
                        .iter()
                        .map(|a| {
                            resolve_expr_typed(
                                a, env, top, structs, enums, variants, at_binding,
                                type_params, self_name,
                            )
                            .map(|(e, _)| e)
                        })
                        .collect::<Result<_, _>>()?;
                    return Ok((
                        Expr::Call(
                            Box::new(Expr::BuiltinRef(builtin.to_owned())),
                            args_canon,
                        ),
                        ret_ty,
                    ));
                }
                let array_builtin = match name.as_str() {
                    "array_new" => Some("core/array.new"),
                    "array_len" => Some("core/array.len"),

                    // The atom primitives over the dedicated `Atom` cell:
                    //   atom_new(init) -> Atom<T>
                    //   atom_load(a) -> T
                    //   atom_swap(a, f) -> T   (lock-free CAS retry loop)
                    "atom_new" => Some("core/atom.new"),
                    "atom_load" => Some("core/atom.load"),
                    "atom_swap" => Some("core/atom.swap"),
                    // Thread primitives over the `ThreadHandle<T>` shape:
                    //   thread_spawn(thunk: fn() -> T) -> ThreadHandle<T>
                    //   thread_join(h: ThreadHandle<T>) -> T
                    "thread_spawn" => Some("core/thread.spawn"),
                    "thread_spawn_shared" => Some("core/thread.spawn_shared"),
                    "thread_join" => Some("core/thread.join"),
                    _ => None,
                };
                if let Some(builtin) = array_builtin {
                    let args_typed: Vec<(Expr, Type)> = args
                        .iter()
                        .map(|a| {
                            resolve_expr_typed(
                                a, env, top, structs, enums, variants, at_binding,
                                type_params, self_name,
                            )
                        })
                        .collect::<Result<_, _>>()?;
                    // Extract `T` from an `Apply(Builtin(container), [T])`
                    // argument type (used to type the element-returning
                    // builtins). Falls back to Int when unpinned.
                    let elem_of = |container: &str, args_typed: &[(Expr, Type)]| match args_typed
                        .first()
                        .map(|(_, t)| t)
                    {
                        Some(Type::Apply(head, elt))
                            if matches!(head.as_ref(), Type::Builtin(n) if n == container) =>
                        {
                            elt.first().cloned().unwrap_or(Type::Builtin("Int".to_owned()))
                        }
                        _ => Type::Builtin("Int".to_owned()),
                    };
                    let ret_ty = match name.as_str() {
                        "array_len" | "array_set" => Type::Builtin("Int".to_owned()),
                        // atom_load / atom_swap: value type T from arg 0's Atom<T>.
                        "atom_load" | "atom_swap" => elem_of("Atom", &args_typed),
                        // atom_new: Atom<typeof init>.
                        "atom_new" => Type::Apply(
                            Box::new(Type::Builtin("Atom".to_owned())),
                            vec![args_typed
                                .first()
                                .map(|(_, t)| t.clone())
                                .unwrap_or(Type::TypeVar(0))],
                        ),
                        // thread_join: value type T from arg 0's ThreadHandle<T>.
                        "thread_join" => elem_of("ThreadHandle", &args_typed),
                        // thread_spawn / thread_spawn_shared: ThreadHandle<T>
                        // where T is the thunk's return type (arg 0 = fn()->T).
                        "thread_spawn" | "thread_spawn_shared" => {
                            let t = match args_typed.first().map(|(_, t)| t) {
                                Some(Type::FnType { params, ret }) if params.is_empty() => {
                                    (**ret).clone()
                                }
                                _ => Type::Builtin("Int".to_owned()),
                            };
                            Type::Apply(
                                Box::new(Type::Builtin("ThreadHandle".to_owned())),
                                vec![t],
                            )
                        }
                        "array_new" => Type::Apply(
                            Box::new(Type::Builtin("Array".to_owned())),
                            vec![Type::TypeVar(0)],
                        ),
                        // array_get: element type T from arg 0's Array<T>.
                        _ => elem_of("Array", &args_typed),
                    };
                    let args_canon: Vec<Expr> =
                        args_typed.into_iter().map(|(e, _)| e).collect();
                    return Ok((
                        Expr::Call(
                            Box::new(Expr::BuiltinRef(builtin.to_owned())),
                            args_canon,
                        ),
                        ret_ty,
                    ));
                }
            }

            // `gc_collect()` — language-level hook for forcing a
            // stop-the-world collection. Lowers to a call to the
            // `core/gc.collect` builtin. Returns Int (always 0); the
            // side effect is the GC. Used by tests and stress
            // harnesses to verify root tracking.
            if let SurfaceExpr::Var { name, .. } = callee.as_ref() {
                if name == "gc_collect" {
                    if !args.is_empty() {
                        return Err(ResolveError::UnknownName {
                            name: format!(
                                "gc_collect takes no args, got {}",
                                args.len()
                            ),
                            span: *span,
                        });
                    }
                    return Ok((
                        Expr::Call(
                            Box::new(Expr::BuiltinRef("core/gc.collect".to_owned())),
                            vec![],
                        ),
                        Type::Builtin("Int".to_owned()),
                    ));
                }
            }

            // Language-level `at(node, thunk)` lowers to a call on a
            // `core/net.at#<hex>` builtin where `<hex>` is the user's
            // `Result` enum's content hash. The runtime constructs a
            // Result value (Ok or Err) and returns its heap pointer.
            //
            // v1 signature: at(node: Node, thunk: fn() -> Int) -> Result
            if let SurfaceExpr::Var { name, .. } = callee.as_ref() {
                if name == "at" || name == "at_async" {
                    let is_async = name == "at_async";
                    if args.len() != 2 {
                        return Err(ResolveError::UnknownName {
                            name: format!("{} expects 2 args, got {}", name, args.len()),
                            span: *span,
                        });
                    }
                    // Build / reuse the at_binding once per module.
                    let binding = match at_binding.as_ref() {
                        Some(b) => b.clone(),
                        None => {
                            let b = build_at_binding(structs, enums, *span)?;
                            *at_binding = Some(b.clone());
                            b
                        }
                    };
                    let node_canon =
                        resolve_expr(&args[0], env, top, structs, enums, variants, at_binding, type_params, self_name)?;
                    let (thunk_canon, thunk_ty) =
                        resolve_expr_typed(&args[1], env, top, structs, enums, variants, at_binding, type_params, self_name)?;
                    // Extract the thunk's return type so the resolver's
                    // tracked `ret_ty` reflects the real instantiation
                    // (downstream pattern bindings substitute against
                    // it).  Fall back to Int if the thunk wasn't a
                    // recognizable `fn() -> T` — the typechecker will
                    // catch the misuse with a clearer error.
                    let thunk_ret_ty = match &thunk_ty {
                        Type::FnType { params, ret } if params.is_empty() => {
                            (**ret).clone()
                        }
                        _ => Type::Builtin("Int".to_owned()),
                    };
                    // `build_at_binding` already enforced the generic
                    // form `Result<T, E>`, so always embed both hashes
                    // in the builtin name.
                    let builtin_name = format!(
                        "{}{}#{}",
                        if is_async {
                            AT_ASYNC_BUILTIN_PREFIX
                        } else {
                            AT_BUILTIN_PREFIX
                        },
                        hex_encode(binding.result_hash.as_bytes()),
                        hex_encode(binding.failure_hash.as_bytes())
                    );
                    let result_ty = Type::Apply(
                        Box::new(Type::TypeRef(binding.result_hash)),
                        vec![thunk_ret_ty, Type::TypeRef(binding.failure_hash)],
                    );
                    // `at` yields the Result directly; `at_async` yields a
                    // handle the existing `join` awaits:
                    // `ThreadHandle<Result<T, Failure>>`.
                    let ret_ty = if is_async {
                        Type::Apply(
                            Box::new(Type::Builtin("ThreadHandle".to_owned())),
                            vec![result_ty],
                        )
                    } else {
                        result_ty
                    };
                    return Ok((
                        Expr::Call(
                            Box::new(Expr::BuiltinRef(builtin_name)),
                            vec![node_canon, thunk_canon],
                        ),
                        ret_ty,
                    ));
                }
            }

            // `extern fn` call: the callee is a bare `Var` whose name
            // is registered as an extern. Rewrite to a builtin call
            // on `ext/<name>` — typecheck + codegen pick the
            // signature up from the resolved module's extern table.
            if let SurfaceExpr::Var { name, .. } = callee.as_ref() {
                if let Some(binding) = top.get(name) {
                    if binding.kind == TopKind::Extern {
                        let ret_ty = match &binding.ty {
                            Type::FnType { ret, .. } => (**ret).clone(),
                            _ => Type::Builtin("Int".to_owned()),
                        };
                        let args_r: Result<Vec<Expr>, _> = args
                            .iter()
                            .map(|a| {
                                resolve_expr(
                                    a, env, top, structs, enums, variants,
                                    at_binding, type_params, self_name,
                                )
                            })
                            .collect();
                        return Ok((
                            Expr::Call(
                                Box::new(Expr::BuiltinRef(format!("ext/{}", name))),
                                args_r?,
                            ),
                            ret_ty,
                        ));
                    }
                }
            }

            // A bare `Var` callee that names a variant is an unqualified
            // variant construction (`Some(x)`); reject it — variants must be
            // qualified (`Option::Some(x)`).
            if let SurfaceExpr::Var { name, span: var_span } = callee.as_ref() {
                if variants.contains_key(name) {
                    return Err(bare_variant_error(variants, name, *var_span));
                }
            }

            // Qualified variant-constructor calls: `Option::Some(x)`,
            // `Val::VInt(n)`. Rewrite to `EnumNew { ..., payload: Some(arg) }`.
            if let SurfaceExpr::VariantRef {
                enum_name,
                variant_name,
                span: var_span,
            } = callee.as_ref()
            {
                let (enum_ref, index, payload_decl) =
                    lookup_qualified_variant(enums, enum_name, variant_name, *var_span)?;
                if payload_decl.is_none() {
                    return Err(ResolveError::VariantArityMismatch {
                        variant: format!("{}::{}", enum_name, variant_name),
                        expected_payload: false,
                        span: *var_span,
                    });
                }
                if args.len() != 1 {
                    return Err(ResolveError::VariantArityMismatch {
                        variant: format!("{}::{}", enum_name, variant_name),
                        expected_payload: true,
                        span: *span,
                    });
                }
                let (payload, payload_ty) = resolve_expr_typed(
                    &args[0], env, top, structs, enums, variants, at_binding,
                    type_params, self_name,
                )?;
                // Bottom-up infer the enum's type-arg substitution by
                // unifying the variant's declared payload type against the
                // actual arg's resolved type. Non-generic enums: no-op.
                let n_params = enums
                    .get(enum_name)
                    .map(|e| e.type_params.len())
                    .unwrap_or(0);
                let result_ty = if n_params > 0 {
                    let mut subst: Vec<Option<Type>> = vec![None; n_params];
                    if let Some(decl) = payload_decl.as_ref() {
                        unify_resolver(decl, &payload_ty, &mut subst);
                    }
                    if subst.iter().any(|s| s.is_some()) {
                        build_instantiated(enum_ref, n_params, &subst)
                    } else {
                        Type::TypeRef(enum_ref)
                    }
                } else {
                    Type::TypeRef(enum_ref)
                };
                return Ok((
                    Expr::EnumNew {
                        enum_ref,
                        variant_index: index,
                        payload: Some(Box::new(payload)),
                    },
                    result_ty,
                ));
            }

            let (callee_r, callee_ty) =
                resolve_expr_typed(callee, env, top, structs, enums, variants, at_binding, type_params, self_name)?;
            // Resolve args TYPED so a generic callee's return type can be
            // instantiated from the actual arg types — e.g.
            // `opt_unwrap_or(opt, 0.0)` resolves to `Float`, which a
            // surrounding `+` needs to pick the f64 (not i64) builtin.
            let mut arg_exprs: Vec<Expr> = Vec::with_capacity(args.len());
            let mut arg_types: Vec<Type> = Vec::with_capacity(args.len());
            for a in args {
                let (ae, at) = resolve_expr_typed(
                    a, env, top, structs, enums, variants, at_binding, type_params, self_name,
                )?;
                arg_exprs.push(ae);
                arg_types.push(at);
            }
            let ret_ty = match &callee_ty {
                Type::FnType { params, ret } => {
                    let n = max_typevar_plus1(params, ret);
                    if n > 0 {
                        let mut subst: Vec<Option<Type>> = vec![None; n];
                        for (p, a) in params.iter().zip(arg_types.iter()) {
                            unify_resolver(p, a, &mut subst);
                        }
                        let concrete: Vec<Type> = subst
                            .into_iter()
                            .enumerate()
                            .map(|(i, o)| o.unwrap_or(Type::TypeVar(i as u32)))
                            .collect();
                        substitute_type_resolver(ret, &concrete)
                    } else {
                        (**ret).clone()
                    }
                }
                _ => Type::Builtin("Int".to_owned()),
            };
            Ok((Expr::Call(Box::new(callee_r), arg_exprs), ret_ty))
        }

        SurfaceExpr::BinOp {
            op, left, right, ..
        } => {
            let (l, lt) = resolve_expr_typed(left, env, top, structs, enums, variants, at_binding, type_params, self_name)?;
            let (r, rt) = resolve_expr_typed(right, env, top, structs, enums, variants, at_binding, type_params, self_name)?;
            // Type-directed dispatch: if either operand is Float, use the
            // f64 builtin variant (the typechecker enforces both are
            // Float). Otherwise fall back to the i64 variants. Arithmetic
            // ops preserve the operand type; comparisons return Int.
            let is_float = is_float_ty(&lt) || is_float_ty(&rt);
            let (builtin, result_ty) = binop_builtin_typed(*op, is_float);
            Ok((
                Expr::Call(
                    Box::new(Expr::BuiltinRef(builtin.to_owned())),
                    vec![l, r],
                ),
                result_ty,
            ))
        }

        SurfaceExpr::UnaryOp { op, operand, .. } => {
            let o = resolve_expr(operand, env, top, structs, enums, variants, at_binding, type_params, self_name)?;
            Ok((
                Expr::Call(
                    Box::new(Expr::BuiltinRef(unop_builtin(*op).to_owned())),
                    vec![o],
                ),
                Type::Builtin("Int".to_owned()),
            ))
        }

        SurfaceExpr::Lambda { params, body, .. } => {
            let params_canon: Result<Vec<Type>, _> = params
                .iter()
                .map(|(_, t)| resolve_type(t, structs, enums, type_params))
                .collect();
            let params_canon = params_canon?;
            let pushed = params.len();
            for ((name, _), ty) in params.iter().zip(params_canon.iter()) {
                env.push((name.clone(), ty.clone()));
            }
            let (body_canon, body_ty) =
                resolve_expr_typed(body, env, top, structs, enums, variants, at_binding, type_params, self_name)?;
            for _ in 0..pushed {
                env.pop();
            }
            let lambda_ty = Type::FnType {
                params: params_canon.clone(),
                ret: Box::new(body_ty),
            };
            Ok((
                Expr::Lambda {
                    params: params_canon,
                    body: Box::new(body_canon),
                },
                lambda_ty,
            ))
        }

        SurfaceExpr::Block { stmts, tail, .. } => {
            // `{ let x = e1; defer c; let y = e2; tail }` lowers to
            // `Let { e1, Defer { c, Let { e2, tail } } }`. A `let` extends
            // the env by one binder; a `defer` adds none (its cleanup is
            // resolved in the env at the defer point, and the Defer node
            // wraps the rest of the block as its body).
            //
            // `lowered` records each statement as `(is_defer, canon_expr)`
            // in source order; we wrap the tail right-to-left at the end.
            let mut lowered: Vec<(bool, Expr)> = Vec::with_capacity(stmts.len());
            let mut pushed = 0usize;
            for s in stmts {
                match s {
                    SurfaceStmt::Let { name, value, .. } => {
                        let (v_canon, v_ty) = resolve_expr_typed(
                            value, env, top, structs, enums, variants, at_binding, type_params,
                            self_name,
                        )?;
                        env.push((name.clone(), v_ty));
                        pushed += 1;
                        lowered.push((false, v_canon));
                    }
                    SurfaceStmt::Defer { expr, .. } => {
                        // Cleanup is resolved in the current env; no binder.
                        let (c_canon, _c_ty) = resolve_expr_typed(
                            expr, env, top, structs, enums, variants, at_binding, type_params,
                            self_name,
                        )?;
                        lowered.push((true, c_canon));
                    }
                }
            }
            let (mut body_canon, body_ty) =
                resolve_expr_typed(tail, env, top, structs, enums, variants, at_binding, type_params, self_name)?;
            for _ in 0..pushed {
                env.pop();
            }
            for (is_defer, canon) in lowered.into_iter().rev() {
                body_canon = if is_defer {
                    Expr::Defer {
                        cleanup: Box::new(canon),
                        body: Box::new(body_canon),
                    }
                } else {
                    Expr::Let {
                        value: Box::new(canon),
                        body: Box::new(body_canon),
                    }
                };
            }
            Ok((body_canon, body_ty))
        }

        SurfaceExpr::StructLit {
            type_name,
            type_name_span,
            fields,
            span: _,
        } => {
            let info = structs.get(type_name).ok_or_else(|| {
                ResolveError::UnknownStruct {
                    name: type_name.clone(),
                    span: *type_name_span,
                }
            })?;

            // Reorder surface fields to declaration order; error on
            // duplicates, missing, or unknown field names.
            let mut by_name: HashMap<&str, (&SurfaceExpr, Span)> = HashMap::new();
            for (n, e) in fields {
                if by_name.contains_key(n.as_str()) {
                    return Err(ResolveError::DuplicateFieldInLiteral {
                        struct_name: type_name.clone(),
                        field: n.clone(),
                        span: e.span(),
                    });
                }
                by_name.insert(n.as_str(), (e, e.span()));
            }
            for (n, _) in fields {
                if info.find_field(n).is_none() {
                    return Err(ResolveError::UnknownField {
                        struct_name: type_name.clone(),
                        field: n.clone(),
                        span: by_name[n.as_str()].1,
                    });
                }
            }

            let n_params = info.type_params.len();
            let mut subst: Vec<Option<Type>> = vec![None; n_params];
            let mut fields_canon = Vec::with_capacity(info.fields.len());
            let info = info.clone();
            for (dname, dty) in &info.fields {
                let (v_expr, _) = by_name.get(dname.as_str()).ok_or_else(|| {
                    ResolveError::MissingField {
                        struct_name: type_name.clone(),
                        field: dname.clone(),
                        span: *type_name_span,
                    }
                })?;
                let (e_canon, e_ty) = resolve_expr_typed(
                    v_expr, env, top, structs, enums, variants, at_binding,
                    type_params, self_name,
                )?;
                if n_params > 0 {
                    unify_resolver(dty, &e_ty, &mut subst);
                }
                fields_canon.push(e_canon);
            }

            let result_ty = if n_params > 0 && subst.iter().any(|s| s.is_some()) {
                build_instantiated(info.hash, n_params, &subst)
            } else {
                Type::TypeRef(info.hash)
            };

            Ok((
                Expr::StructNew {
                    struct_ref: info.hash,
                    fields: fields_canon,
                },
                result_ty,
            ))
        }

        SurfaceExpr::Match {
            scrutinee,
            arms,
            span,
        } => {
            let (scrut_canon, scrut_ty) =
                resolve_expr_typed(scrutinee, env, top, structs, enums, variants, at_binding, type_params, self_name)?;
            // If the scrutinee is `Apply(Result, [T_actual, E_actual])`
            // (an instantiated generic enum), extract the instantiation
            // args so we can substitute them into pattern bindings —
            // otherwise `Ok(p)` would bind p to `TypeVar(0)` and
            // downstream field access on p fails to find the struct.
            let scrut_instantiation: Vec<Type> = match &scrut_ty {
                Type::Apply(_, args) => args.clone(),
                _ => Vec::new(),
            };

            // The result type is taken from the FIRST arm; we don't
            // enforce arm-type equality here (typechecker's job).
            let mut canonical_arms: Vec<MatchArm> = Vec::with_capacity(arms.len());
            let mut shared_enum: Option<Hash> = None;
            let mut result_ty: Option<Type> = None;

            for arm in arms {
                let (pattern_canon, _bind_count, pat_enum) = resolve_pattern(
                    &arm.pattern,
                    variants,
                    enums,
                )?;
                if let Some(eh) = pat_enum {
                    if let Some(existing) = shared_enum {
                        if existing != eh {
                            return Err(ResolveError::MatchEnumMismatch {
                                expected: existing,
                                found: eh,
                                span: arm.pattern.span(),
                            });
                        }
                    } else {
                        shared_enum = Some(eh);
                    }
                }

                // Bind pattern variables into env so the arm body can
                // refer to them by name. Each (name, type) pair is
                // pushed in pattern-traversal order. Pattern bindings
                // are substituted against the scrutinee's instantiation
                // so e.g. `Ok(p)` against `Result<Pair, Failure>`
                // binds `p: Pair` rather than `p: TypeVar(0)`.
                let bindings =
                    collect_pattern_bindings(&arm.pattern, variants, enums, &scrut_instantiation);
                for (name, ty) in &bindings {
                    env.push((name.clone(), ty.clone()));
                }
                let bind_count = bindings.len() as u32;
                let _ = bind_count;
                let body_canon = resolve_expr(
                    &arm.body,
                    env,
                    top,
                    structs,
                    enums,
                    variants,
                    at_binding,
                    type_params,
                    self_name,
                )?;
                let arm_ty = match &body_canon {
                    Expr::IntLit(_) => Type::Builtin("Int".to_owned()),
                    _ => Type::Builtin("Int".to_owned()),
                };
                if result_ty.is_none() {
                    result_ty = Some(arm_ty);
                }
                for _ in 0..bind_count {
                    env.pop();
                }
                canonical_arms.push(MatchArm {
                    pattern: pattern_canon,
                    body: body_canon,
                });
            }

            let _ = span; // span retained for future error messages
            Ok((
                Expr::Match {
                    scrutinee: Box::new(scrut_canon),
                    arms: canonical_arms,
                },
                result_ty.unwrap_or(Type::Builtin("Int".to_owned())),
            ))
        }

        SurfaceExpr::If {
            cond,
            then_branch,
            else_branch,
            span: _,
        } => {
            let cond_canon = resolve_expr(
                cond, env, top, structs, enums, variants, at_binding, type_params,
                self_name,
            )?;
            let (then_canon, then_ty) = resolve_expr_typed(
                then_branch, env, top, structs, enums, variants, at_binding,
                type_params, self_name,
            )?;
            let (else_canon, _else_ty) = resolve_expr_typed(
                else_branch, env, top, structs, enums, variants, at_binding,
                type_params, self_name,
            )?;
            // Branch-type agreement is enforced by the typechecker;
            // resolver just propagates the then-branch type.
            Ok((
                Expr::If {
                    cond: Box::new(cond_canon),
                    then_branch: Box::new(then_canon),
                    else_branch: Box::new(else_canon),
                },
                then_ty,
            ))
        }

        SurfaceExpr::FieldAccess {
            base,
            field_name,
            field_span,
            span: _,
        } => {
            let (base_canon, base_ty) =
                resolve_expr_typed(base, env, top, structs, enums, variants, at_binding, type_params, self_name)?;
            // Base must be a struct type (possibly instantiated as
            // `Apply(TypeRef(h), [args])`) to project a named field.
            // Pull out the struct hash and (if Apply) the instantiation
            // args so the field's declared type can be substituted.
            let (struct_ref, instantiation): (Hash, Vec<Type>) = match &base_ty {
                Type::TypeRef(h) => (*h, Vec::new()),
                Type::Apply(head, args) => match head.as_ref() {
                    Type::TypeRef(h) => (*h, args.clone()),
                    _ => {
                        return Err(ResolveError::FieldOnNonStruct {
                            ty: base_ty,
                            span: *field_span,
                        });
                    }
                },
                _ => {
                    return Err(ResolveError::FieldOnNonStruct {
                        ty: base_ty,
                        span: *field_span,
                    });
                }
            };
            // Find the struct in our registry to map field_name → index.
            let info = structs
                .values()
                .find(|i| i.hash == struct_ref)
                .ok_or_else(|| ResolveError::UnknownStruct {
                    name: format!("<unnamed struct {}>", struct_ref),
                    span: *field_span,
                })?
                .clone();
            let (index, field_ty) = info.find_field(field_name).ok_or_else(|| {
                ResolveError::UnknownField {
                    struct_name: info_name(&info, structs),
                    field: field_name.clone(),
                    span: *field_span,
                }
            })?;
            // Substitute the base's instantiation into the field type.
            // This makes `(cell : ListCell<Int>).head` return type Int,
            // not TypeVar(0). Downstream consumers (Cons constructor
            // type inference, future typecheck) get the right shape.
            let ty = if instantiation.is_empty() {
                field_ty.clone()
            } else {
                substitute_type_resolver(field_ty, &instantiation)
            };
            Ok((
                Expr::Field {
                    base: Box::new(base_canon),
                    struct_ref,
                    index,
                },
                ty,
            ))
        }

        SurfaceExpr::Try { expr, span } => {
            // FUSED checked indexing: `array_get(a, i)?` skips the Result
            // entirely on the happy path — the bounds check branches
            // straight to the trusted access, and only the Err arm builds
            // (and early-returns) a Result value.
            if let SurfaceExpr::Call { callee, args, .. } = expr.as_ref() {
                if let SurfaceExpr::Var { name, .. } = callee.as_ref() {
                    if let Some(acc) = checked_accessor(name) {
                        return resolve_checked_accessor(
                            &acc, args, true, *span, env, top, structs, enums, variants,
                            at_binding, type_params, self_name,
                        );
                    }
                }
            }
            let (inner_canon, inner_ty) = resolve_expr_typed(
                expr, env, top, structs, enums, variants, at_binding, type_params, self_name,
            )?;
            // The operand must be a `Result<T, E>`: either a bare
            // `TypeRef(h)` (monomorphic) or `Apply(TypeRef(h), [T, E])`.
            let (enum_ref, instantiation): (Hash, Vec<Type>) = match &inner_ty {
                Type::TypeRef(h) => (*h, Vec::new()),
                Type::Apply(head, args) => match head.as_ref() {
                    Type::TypeRef(h) => (*h, args.clone()),
                    _ => {
                        return Err(ResolveError::TryOnNonResult {
                            ty: inner_ty,
                            span: *span,
                        });
                    }
                },
                _ => {
                    return Err(ResolveError::TryOnNonResult {
                        ty: inner_ty,
                        span: *span,
                    });
                }
            };
            // The enum must declare `Ok` and `Err` variants (the
            // `Result<T, E>` shape). We identify it structurally rather
            // than by a hardcoded hash so any conforming user enum works.
            let info = enums
                .values()
                .find(|i| i.hash == enum_ref)
                .ok_or_else(|| ResolveError::TryOnNonResult {
                    ty: inner_ty.clone(),
                    span: *span,
                })?
                .clone();
            let ok_index = find_variant(&info, "Ok").ok_or_else(|| {
                ResolveError::TryOnNonResult {
                    ty: inner_ty.clone(),
                    span: *span,
                }
            })?;
            let err_index = find_variant(&info, "Err").ok_or_else(|| {
                ResolveError::TryOnNonResult {
                    ty: inner_ty.clone(),
                    span: *span,
                }
            })?;
            // The value of `expr?` is the `Ok` payload type `T`, with the
            // operand's instantiation substituted in (so `Result<Int, E>?`
            // resolves to `Int`, not `TypeVar(0)`).
            let ok_payload = info.variants[ok_index as usize]
                .1
                .clone()
                .ok_or_else(|| ResolveError::TryOnNonResult {
                    ty: inner_ty.clone(),
                    span: *span,
                })?;
            let ty = if instantiation.is_empty() {
                ok_payload
            } else {
                substitute_type_resolver(&ok_payload, &instantiation)
            };
            Ok((
                Expr::Try {
                    expr: Box::new(inner_canon),
                    enum_ref,
                    ok_index,
                    err_index,
                },
                ty,
            ))
        }
    }
}

/// Resolve a surface pattern to a canonical pattern. Returns the
/// pattern, the number of bindings it introduces (Var counts as 1,
/// Wildcard as 0, Ctor recurses), and the enum hash (if this is a
/// variant pattern — for cross-arm consistency checks).
fn resolve_pattern(
    p: &SurfacePattern,
    variants: &HashMap<String, VariantBinding>,
    enums: &HashMap<String, EnumInfo>,
) -> Result<(Pattern, u32, Option<Hash>), ResolveError> {
    match p {
        SurfacePattern::Wildcard { .. } => Ok((Pattern::Wildcard, 0, None)),
        SurfacePattern::Ident { name, span } => {
            // A bare ident that names a variant must be qualified — silently
            // treating `None` as a catch-all binding would be a footgun.
            if variants.contains_key(name) {
                return Err(bare_variant_error(variants, name, *span));
            }
            // Otherwise it's a binding.
            Ok((Pattern::Var, 1, None))
        }
        // `Some(x)` in pattern position — bare ctor must be qualified.
        SurfacePattern::Ctor { name, span, .. } => {
            Err(bare_variant_error(variants, name, *span))
        }
        SurfacePattern::QualifiedCtor {
            enum_name,
            variant_name,
            payload,
            span,
        } => {
            let (enum_ref, index, payload_decl) =
                lookup_qualified_variant(enums, enum_name, variant_name, *span)?;
            match (payload, payload_decl) {
                (None, None) => Ok((
                    Pattern::Enum {
                        enum_ref,
                        variant_index: index,
                        payload: None,
                    },
                    0,
                    Some(enum_ref),
                )),
                (Some(sub), Some(_)) => {
                    let (sub_canon, sub_bindings, _sub_enum) =
                        resolve_pattern(sub, variants, enums)?;
                    Ok((
                        Pattern::Enum {
                            enum_ref,
                            variant_index: index,
                            payload: Some(Box::new(sub_canon)),
                        },
                        sub_bindings,
                        Some(enum_ref),
                    ))
                }
                (Some(_), None) => Err(ResolveError::VariantArityMismatch {
                    variant: format!("{}::{}", enum_name, variant_name),
                    expected_payload: false,
                    span: *span,
                }),
                (None, Some(_)) => Err(ResolveError::VariantArityMismatch {
                    variant: format!("{}::{}", enum_name, variant_name),
                    expected_payload: true,
                    span: *span,
                }),
            }
        }
    }
}

/// Collect (binding_name, canonical_type) pairs in the order they
/// should be pushed onto env for the match-arm body.
///
/// - `Wildcard` introduces nothing.
/// - `Ident name` at top level: if `name` is a registered variant, no
///   binding (it's a constructor pattern); otherwise it's a catch-all
///   binding (typed Int as a placeholder — typechecker will refine).
/// - `Ctor name(sub)`: sub-pattern's bindings, typed using the
///   variant's declared payload type when known.
fn collect_pattern_bindings(
    p: &SurfacePattern,
    variants: &HashMap<String, VariantBinding>,
    enums: &HashMap<String, EnumInfo>,
    instantiation: &[Type],
) -> Vec<(String, Type)> {
    let mut out: Vec<(String, Type)> = Vec::new();
    walk_pattern_bindings(p, variants, enums, instantiation, None, &mut out);
    out
}

fn walk_pattern_bindings(
    p: &SurfacePattern,
    variants: &HashMap<String, VariantBinding>,
    enums: &HashMap<String, EnumInfo>,
    instantiation: &[Type],
    payload_ty_hint: Option<&Type>,
    out: &mut Vec<(String, Type)>,
) {
    match p {
        SurfacePattern::Wildcard { .. } => {}
        SurfacePattern::Ident { name, .. } => {
            // Bare variant idents are rejected in resolve_pattern, so a bare
            // ident reaching here is always a binding.
            let raw = payload_ty_hint
                .cloned()
                .unwrap_or(Type::Builtin("Int".to_owned()));
            let ty = substitute_typevars(&raw, instantiation);
            out.push((name.clone(), ty));
        }
        // Bare ctor patterns are rejected in resolve_pattern; nothing to bind.
        SurfacePattern::Ctor { .. } => {}
        SurfacePattern::QualifiedCtor {
            enum_name,
            variant_name,
            payload,
            ..
        } => {
            if let Some(sub) = payload {
                let sub_payload_ty = enums.get(enum_name).and_then(|e| {
                    e.variants
                        .iter()
                        .find(|(n, _)| n == variant_name)
                        .and_then(|(_, ty)| ty.clone())
                });
                walk_pattern_bindings(
                    sub,
                    variants,
                    enums,
                    instantiation,
                    sub_payload_ty.as_ref(),
                    out,
                );
            }
        }
    }
}

/// Replace `Type::TypeVar(i)` references with `instantiation[i]` where
/// applicable. Used to propagate the scrutinee's instantiation into
/// match-arm pattern bindings: matching `Ok(p)` against
/// `Apply(Result, [Pair, Failure])` should bind `p: Pair`, not
/// `p: TypeVar(0)`.
fn substitute_typevars(ty: &Type, instantiation: &[Type]) -> Type {
    match ty {
        Type::TypeVar(i) => instantiation
            .get(*i as usize)
            .cloned()
            .unwrap_or_else(|| ty.clone()),
        Type::Apply(head, args) => Type::Apply(
            Box::new(substitute_typevars(head, instantiation)),
            args.iter().map(|a| substitute_typevars(a, instantiation)).collect(),
        ),
        Type::FnType { params, ret } => Type::FnType {
            params: params.iter().map(|p| substitute_typevars(p, instantiation)).collect(),
            ret: Box::new(substitute_typevars(ret, instantiation)),
        },
        _ => ty.clone(),
    }
}

fn info_name(info: &StructInfo, structs: &HashMap<String, StructInfo>) -> String {
    structs
        .iter()
        .find(|(_, i)| i.hash == info.hash)
        .map(|(n, _)| n.clone())
        .unwrap_or_else(|| format!("<struct {}>", info.hash))
}

/// Look up `name` in the lexical environment. Returns the de Bruijn
/// index (0 = innermost) and the binding's canonical type.
fn lookup_local(env: &[(String, Type)], name: &str) -> Option<(u32, Type)> {
    let n = env.len();
    for (i, (binder, ty)) in env.iter().enumerate().rev() {
        if binder == name {
            return Some(((n - 1 - i) as u32, ty.clone()));
        }
    }
    None
}

fn binop_builtin(op: BinOp) -> &'static str {
    // v1: only Int and Bool exist. Operators are statically resolved to the
    // i64 / bool variants. A real typechecker will replace this with
    // type-directed dispatch.
    match op {
        BinOp::Add => "core/i64.add",
        BinOp::Sub => "core/i64.sub",
        BinOp::Mul => "core/i64.mul",
        BinOp::Div => "core/i64.div",
        BinOp::Rem => "core/i64.rem",
        BinOp::Eq => "core/i64.eq",
        BinOp::NotEq => "core/i64.ne",
        BinOp::Lt => "core/i64.lt",
        BinOp::LtEq => "core/i64.le",
        BinOp::Gt => "core/i64.gt",
        BinOp::GtEq => "core/i64.ge",
        BinOp::And => "core/bool.and",
        BinOp::Or => "core/bool.or",
    }
}

fn unop_builtin(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg => "core/i64.neg",
        UnaryOp::Not => "core/bool.not",
    }
}

fn is_float_ty(t: &Type) -> bool {
    matches!(t, Type::Builtin(n) if n == "Float")
}

/// Pick the builtin + result type for a binary op given whether the
/// operands are Float. Float arithmetic returns Float; comparisons
/// return Int (Bool widened). `And`/`Or` stay Int-only.
fn binop_builtin_typed(op: BinOp, is_float: bool) -> (&'static str, Type) {
    let int_t = || Type::Builtin("Int".to_owned());
    let float_t = || Type::Builtin("Float".to_owned());
    if is_float {
        match op {
            BinOp::Add => ("core/f64.add", float_t()),
            BinOp::Sub => ("core/f64.sub", float_t()),
            BinOp::Mul => ("core/f64.mul", float_t()),
            BinOp::Div => ("core/f64.div", float_t()),
            BinOp::Rem => ("core/f64.rem", float_t()),
            BinOp::Eq => ("core/f64.eq", int_t()),
            BinOp::NotEq => ("core/f64.ne", int_t()),
            BinOp::Lt => ("core/f64.lt", int_t()),
            BinOp::LtEq => ("core/f64.le", int_t()),
            BinOp::Gt => ("core/f64.gt", int_t()),
            BinOp::GtEq => ("core/f64.ge", int_t()),
            // Logical and/or aren't meaningful on Float; fall back to
            // the int builtins (typecheck will reject Float operands).
            BinOp::And => ("core/bool.and", int_t()),
            BinOp::Or => ("core/bool.or", int_t()),
        }
    } else {
        let b = binop_builtin(op);
        // Comparisons and logicals are already Int; arithmetic on Int
        // is Int — so the result is Int in every non-float case.
        (b, int_t())
    }
}

/// Best-effort unification used by the resolver to recover the type-arg
/// substitution at variant / struct construction sites. Where `pattern`
/// has a `TypeVar(i)` and `actual` is concrete, record `subst[i] = actual`.
/// Shape mismatches are silent (caller gets a partial substitution).
///
/// Returns nothing — this is bottom-up best-effort, not a strict check.
/// Number of distinct `TypeVar(i)` slots a signature scopes (max index
/// + 1), so callers can size a substitution vector.
fn max_typevar_plus1(params: &[Type], ret: &Type) -> usize {
    fn go(t: &Type, m: &mut usize) {
        match t {
            Type::TypeVar(i) => *m = (*m).max(*i as usize + 1),
            Type::Apply(h, args) => {
                go(h, m);
                for a in args {
                    go(a, m);
                }
            }
            Type::FnType { params, ret } => {
                for p in params {
                    go(p, m);
                }
                go(ret, m);
            }
            _ => {}
        }
    }
    let mut m = 0;
    for p in params {
        go(p, &mut m);
    }
    go(ret, &mut m);
    m
}

fn unify_resolver(pattern: &Type, actual: &Type, subst: &mut [Option<Type>]) {
    match (pattern, actual) {
        // A TypeVar on the actual side is "instantiation unknown" — it
        // never constrains (matches typecheck's `unify`). Without this,
        // an early placeholder (e.g. `smap_new(): StringMap<TypeVar(0)>`)
        // would pin the slot and block the real binding from a later
        // concrete arg.
        (Type::TypeVar(_), Type::TypeVar(_)) => {}
        (Type::TypeVar(i), other) => {
            let idx = *i as usize;
            // Bind when unset, or refine a prior TypeVar placeholder to
            // a concrete type.
            if idx < subst.len()
                && (subst[idx].is_none() || matches!(subst[idx], Some(Type::TypeVar(_))))
            {
                subst[idx] = Some(other.clone());
            }
        }
        (Type::Apply(ph, pa), Type::Apply(ah, aa)) => {
            unify_resolver(ph, ah, subst);
            if pa.len() == aa.len() {
                for (p, a) in pa.iter().zip(aa.iter()) {
                    unify_resolver(p, a, subst);
                }
            }
        }
        (
            Type::FnType { params: pp, ret: pr },
            Type::FnType { params: ap, ret: ar },
        ) if pp.len() == ap.len() => {
            for (p, a) in pp.iter().zip(ap.iter()) {
                unify_resolver(p, a, subst);
            }
            unify_resolver(pr, ar, subst);
        }
        _ => {}
    }
}

/// Apply a positional type-arg substitution throughout `ty`. Used at
/// field-access sites to lower `ListCell<Int>.head : TypeVar(0)` to
/// `Int`, which then feeds the next layer of constructor inference.
fn substitute_type_resolver(ty: &Type, subst: &[Type]) -> Type {
    match ty {
        Type::TypeVar(i) => subst
            .get(*i as usize)
            .cloned()
            .unwrap_or_else(|| Type::TypeVar(*i)),
        Type::Builtin(_) | Type::TypeRef(_) | Type::SelfRef(_) => ty.clone(),
        Type::FnType { params, ret } => Type::FnType {
            params: params.iter().map(|p| substitute_type_resolver(p, subst)).collect(),
            ret: Box::new(substitute_type_resolver(ret, subst)),
        },
        Type::Apply(head, args) => Type::Apply(
            Box::new(substitute_type_resolver(head, subst)),
            args.iter().map(|a| substitute_type_resolver(a, subst)).collect(),
        ),
    }
}

/// Build an instantiated `Apply` from a partial substitution. Unbound
/// TypeVars round-trip as themselves — usually that means the caller
/// learned nothing at all (no Apply needed), so the caller decides
/// whether to wrap or fall back to bare `TypeRef`.
fn build_instantiated(head_hash: Hash, n_params: usize, subst: &[Option<Type>]) -> Type {
    let args: Vec<Type> = (0..n_params)
        .map(|i| {
            subst
                .get(i)
                .and_then(|o| o.clone())
                .unwrap_or(Type::TypeVar(i as u32))
        })
        .collect();
    Type::Apply(Box::new(Type::TypeRef(head_hash)), args)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{parse_def, parse_module};

    fn resolve_one(src: &str) -> Def {
        let sd = parse_def(src).unwrap();
        let m = Module { defs: vec![sd] };
        let r = resolve_module(&m).unwrap();
        r.defs.into_iter().next().unwrap().def
    }

    #[test]
    fn param_becomes_de_bruijn_zero() {
        let d = resolve_one("def f(x: Int) -> Int = x");
        let Def::Fn { body, .. } = d else { panic!("expected Fn") };
        assert_eq!(body, Expr::LocalVar(0));
    }

    fn resolve_mod_src(src: &str) -> Result<(), ResolveError> {
        let m = parse_module(src).unwrap();
        resolve_module(&m).map(|_| ())
    }

    #[test]
    fn qualified_variant_construction_and_match_resolve() {
        // Qualified construction (nullary + payload) and qualified patterns
        // are the legal form and must resolve cleanly.
        resolve_mod_src(
            "enum Color { Red, Pair(Int) }\n\
             def mk() -> Color = Color::Pair(7)\n\
             def nul() -> Color = Color::Red\n\
             def rd(c: Color) -> Int = match c { Color::Red => 0, Color::Pair(n) => n }",
        )
        .expect("qualified variants must resolve");
    }

    #[test]
    fn bare_variant_construction_is_rejected() {
        let err = resolve_mod_src(
            "enum Color { Red, Pair(Int) }\n\
             def mk() -> Color = Pair(7)",
        )
        .unwrap_err();
        match err {
            ResolveError::UnknownName { name, .. } => {
                assert!(name.contains("must be qualified"), "got: {name}");
                assert!(name.contains("Color::Pair"), "should suggest qualified form: {name}");
            }
            other => panic!("expected must-qualify UnknownName, got {other:?}"),
        }
    }

    #[test]
    fn bare_nullary_variant_construction_is_rejected() {
        let err = resolve_mod_src(
            "enum Color { Red, Pair(Int) }\n\
             def mk() -> Color = Red",
        )
        .unwrap_err();
        assert!(matches!(err, ResolveError::UnknownName { ref name, .. } if name.contains("Color::Red")));
    }

    #[test]
    fn bare_variant_pattern_is_rejected() {
        // A bare ctor pattern must be qualified — and must NOT silently become
        // a binding, which would be a correctness footgun.
        let err = resolve_mod_src(
            "enum Color { Red, Pair(Int) }\n\
             def rd(c: Color) -> Int = match c { Pair(n) => n, Color::Red => 0 }",
        )
        .unwrap_err();
        assert!(matches!(err, ResolveError::UnknownName { ref name, .. } if name.contains("must be qualified")));
    }

    #[test]
    fn two_params_get_descending_indices() {
        // params [x, y]; in the body `x` is the outer binder (index 1),
        // `y` is the innermost (index 0).
        let d = resolve_one("def f(x: Int, y: Int) -> Int = x");
        let Def::Fn { body, .. } = &d else { panic!("expected Fn") };
        assert_eq!(*body, Expr::LocalVar(1));
        let d2 = resolve_one("def f(x: Int, y: Int) -> Int = y");
        let Def::Fn { body, .. } = &d2 else { panic!("expected Fn") };
        assert_eq!(*body, Expr::LocalVar(0));
    }

    #[test]
    fn binop_lowers_to_builtin_call() {
        let d = resolve_one("def f(x: Int) -> Int = x * 2");
        let Def::Fn { body, .. } = d else { panic!("expected Fn") };
        match body {
            Expr::Call(callee, args) => {
                assert_eq!(*callee, Expr::BuiltinRef("core/i64.mul".to_owned()));
                assert_eq!(args.len(), 2);
                assert_eq!(args[0], Expr::LocalVar(0));
                assert_eq!(args[1], Expr::IntLit(2));
            }
            other => panic!("expected Call, got {:?}", other),
        }
    }

    #[test]
    fn unknown_name_is_an_error() {
        let m = parse_module("def f(x: Int) -> Int = nope").unwrap();
        let err = resolve_module(&m).unwrap_err();
        assert!(matches!(err, ResolveError::UnknownName { ref name, .. } if name == "nope"));
    }

    #[test]
    fn unknown_type_is_an_error() {
        let m = parse_module("def f(x: Widget) -> Widget = x").unwrap();
        let err = resolve_module(&m).unwrap_err();
        assert!(matches!(err, ResolveError::UnknownType { ref name, .. } if name == "Widget"));
    }

    #[test]
    fn self_reference_resolves() {
        // Self-recursion is now supported via the SCC pass: a singleton
        // SCC produces a SelfRef(0) in the canonical body and rewrites
        // to TopRef(self_hash) in the stored body.
        let m = parse_module("def f(x: Int) -> Int = f(x)").unwrap();
        let r = resolve_module(&m).expect("self-rec should resolve");
        let f = r.get("f").expect("f present");
        match &f.def {
            Def::Fn { body, .. } => match body {
                // The stored body's call refers to f by its own hash.
                Expr::Call(callee, _) => match callee.as_ref() {
                    Expr::TopRef(h) => assert_eq!(*h, f.hash),
                    other => panic!("expected TopRef(self), got {:?}", other),
                },
                other => panic!("expected Call body, got {:?}", other),
            },
            _ => panic!("expected Fn def"),
        }
    }

    #[test]
    fn forward_reference_resolves() {
        // f → g where g is defined later in source. The SCC pass
        // handles the singleton SCC for g first, then f (which
        // depends on g) gets a real TopRef(g_hash).
        let m = parse_module(
            "def f(x: Int) -> Int = g(x)
             def g(x: Int) -> Int = x",
        )
        .unwrap();
        let r = resolve_module(&m).expect("forward ref should resolve");
        let f = r.get("f").expect("f");
        let g = r.get("g").expect("g");
        match &f.def {
            Def::Fn { body, .. } => match body {
                Expr::Call(callee, _) => match callee.as_ref() {
                    Expr::TopRef(h) => assert_eq!(*h, g.hash),
                    other => panic!("expected TopRef(g), got {:?}", other),
                },
                other => panic!("expected Call, got {:?}", other),
            },
            _ => panic!("expected Fn"),
        }
    }

    #[test]
    fn duplicate_def_is_an_error() {
        let m = parse_module(
            "def f(x: Int) -> Int = x
             def f(x: Int) -> Int = x",
        )
        .unwrap();
        let err = resolve_module(&m).unwrap_err();
        assert!(matches!(err, ResolveError::DuplicateDef { ref name, .. } if name == "f"));
    }

    // ---- Recursive types ----

    /// Self-referential struct: a `Node` whose `next` field is another
    /// `Node` (wrapped in `Option` for terminus). Tests singleton SCC
    /// at the type level — same code path as fn self-recursion.
    #[test]
    fn self_referential_struct_resolves() {
        // We don't have Option<T> here so use a self-referencing enum
        // payload for the terminus.
        let src = "
            enum LinkedList { Cons(Cell), Nil }
            struct Cell { head: Int, tail: LinkedList }
        ";
        let m = parse_module(src).unwrap();
        let r = resolve_module(&m).expect("recursive types should resolve");
        let cell = r.get("Cell").expect("Cell present");
        let list = r.get("LinkedList").expect("LinkedList present");
        // Cell.tail should reference LinkedList via TopRef(list.hash).
        match &cell.def {
            Def::Struct { fields, .. } => {
                let tail = &fields[1];
                assert_eq!(tail.0, "tail");
                match &tail.1 {
                    Type::TypeRef(h) => assert_eq!(*h, list.hash),
                    other => panic!("expected TopRef(LinkedList), got {:?}", other),
                }
            }
            _ => panic!("expected Cell struct"),
        }
        // LinkedList's Cons variant should reference Cell via TopRef(cell.hash).
        match &list.def {
            Def::Enum { variants, .. } => {
                let cons = &variants[0];
                assert_eq!(cons.0, "Cons");
                match &cons.1 {
                    Some(Type::TypeRef(h)) => assert_eq!(*h, cell.hash),
                    other => panic!("expected TopRef(Cell), got {:?}", other),
                }
            }
            _ => panic!("expected LinkedList enum"),
        }
    }

    /// Forward reference: a struct's field references a type defined
    /// later in source. Singleton SCC, processed in dependency order.
    #[test]
    fn forward_referential_struct_resolves() {
        let src = "
            struct Wrapper { inner: Inner }
            struct Inner { value: Int }
        ";
        let m = parse_module(src).unwrap();
        let r = resolve_module(&m).expect("forward ref should resolve");
        let wrapper = r.get("Wrapper").expect("Wrapper");
        let inner = r.get("Inner").expect("Inner");
        match &wrapper.def {
            Def::Struct { fields, .. } => match &fields[0].1 {
                Type::TypeRef(h) => assert_eq!(*h, inner.hash),
                other => panic!("expected TopRef(Inner), got {:?}", other),
            },
            _ => panic!("expected Wrapper struct"),
        }
    }

    /// Mutually recursive: A → B → A. Multi-member SCC at type level.
    #[test]
    fn mutually_recursive_types_resolve() {
        let src = "
            enum A { AVal(Int), AB(B) }
            enum B { BA(A), BVal(Int) }
        ";
        let m = parse_module(src).unwrap();
        let r = resolve_module(&m).expect("mutual rec types should resolve");
        let a = r.get("A").expect("A");
        let b = r.get("B").expect("B");
        // A's AB variant references B; B's BA variant references A.
        match &a.def {
            Def::Enum { variants, .. } => {
                let ab = &variants[1];
                assert_eq!(ab.0, "AB");
                match &ab.1 {
                    Some(Type::TypeRef(h)) => assert_eq!(*h, b.hash),
                    other => panic!("expected TopRef(B), got {:?}", other),
                }
            }
            _ => panic!("expected A enum"),
        }
        match &b.def {
            Def::Enum { variants, .. } => {
                let ba = &variants[0];
                assert_eq!(ba.0, "BA");
                match &ba.1 {
                    Some(Type::TypeRef(h)) => assert_eq!(*h, a.hash),
                    other => panic!("expected TopRef(A), got {:?}", other),
                }
            }
            _ => panic!("expected B enum"),
        }
    }

    /// Renaming a self-reference name in source doesn't change hashes —
    /// SCC-local indices are what get encoded.
    #[test]
    fn renaming_self_ref_preserves_hash() {
        let src_a = "enum L1 { CCons(Int, L1), CNil }";
        // The grammar only supports 0-or-1 payloads per variant; we
        // can't express Cons(Int, L1) directly without a struct.
        // Use a self-referencing single-payload form instead.
        let src_a = "enum L1 { Cycle(L1), Done }";
        let src_b = "enum L2 { Cycle(L2), Done }";
        let m_a = parse_module(src_a).unwrap();
        let m_b = parse_module(src_b).unwrap();
        let r_a = resolve_module(&m_a).unwrap();
        let r_b = resolve_module(&m_b).unwrap();
        let h_a = r_a.get("L1").unwrap().hash;
        let h_b = r_b.get("L2").unwrap().hash;
        // Different variant constructor names (Cycle is the same but
        // the enum's surface name differs) — but our enum codec
        // doesn't include the type's own surface name, only variant
        // names + payload types. The variants ARE the same; the
        // payload SelfRef encodes positionally. Hashes should match.
        assert_eq!(h_a, h_b, "hash should be invariant under type rename");
    }

    #[test]
    fn caller_references_callee_by_hash() {
        let m = parse_module(
            "def double(x: Int) -> Int = x * 2
             def quadruple(x: Int) -> Int = double(double(x))",
        )
        .unwrap();
        let r = resolve_module(&m).unwrap();
        let double_hash = r.get("double").unwrap().hash;
        let quad = r.get("quadruple").unwrap();

        let Def::Fn { body, .. } = &quad.def else { panic!("expected Fn") };
        // outer: Call(TopRef(double), [Call(TopRef(double), [LocalVar(0)])])
        let Expr::Call(outer_callee, outer_args) = body else {
            panic!("expected Call at root: {:?}", body);
        };
        assert_eq!(**outer_callee, Expr::TopRef(double_hash));
        let Expr::Call(inner_callee, inner_args) = &outer_args[0] else {
            panic!("expected inner Call: {:?}", outer_args[0]);
        };
        assert_eq!(**inner_callee, Expr::TopRef(double_hash));
        assert_eq!(inner_args[0], Expr::LocalVar(0));
    }

    // ---- The central property, end-to-end through parser + resolver + hasher ----

    #[test]
    fn renaming_param_keeps_hash_stable() {
        let a = resolve_one("def double(x: Int) -> Int = x * 2");
        let b = resolve_one("def double(y: Int) -> Int = y * 2");
        let ah = Hash::of_bytes(&encode_def(&a));
        let bh = Hash::of_bytes(&encode_def(&b));
        assert_eq!(ah, bh, "renaming the parameter must not change the hash");
    }

    #[test]
    fn renaming_def_keeps_its_hash_stable() {
        let a = resolve_one("def double(x: Int) -> Int = x * 2");
        let b = resolve_one("def dbl(x: Int) -> Int = x * 2");
        let ah = Hash::of_bytes(&encode_def(&a));
        let bh = Hash::of_bytes(&encode_def(&b));
        assert_eq!(ah, bh, "the def's surface name is not part of its identity");
    }

    #[test]
    fn renaming_callee_keeps_callers_hash_stable() {
        // Two modules, same code modulo renaming `double` to `dbl`.
        // `quadruple` references its callee by hash, not name, so its hash
        // is the same in both modules.
        let m1 = parse_module(
            "def double(x: Int) -> Int = x * 2
             def quadruple(x: Int) -> Int = double(double(x))",
        )
        .unwrap();
        let m2 = parse_module(
            "def dbl(x: Int) -> Int = x * 2
             def quadruple(x: Int) -> Int = dbl(dbl(x))",
        )
        .unwrap();
        let r1 = resolve_module(&m1).unwrap();
        let r2 = resolve_module(&m2).unwrap();

        // double and dbl are the same def (same body, same sig).
        assert_eq!(r1.get("double").unwrap().hash, r2.get("dbl").unwrap().hash);
        // quadruple is the same def in both — its TopRef pins by hash.
        assert_eq!(
            r1.get("quadruple").unwrap().hash,
            r2.get("quadruple").unwrap().hash
        );
    }

    // ---- Lambda / FnType ----

    #[test]
    fn lambda_resolves_with_explicit_arity() {
        let m = parse_module("def make() -> fn(Int) -> Int = |x: Int| x").unwrap();
        let r = resolve_module(&m).unwrap();
        let Def::Fn { body, ret, .. } = &r.defs[0].def else { panic!("expected Fn") };
        // body should be Lambda { params: [Int], body: LocalVar(0) }
        match body {
            Expr::Lambda { params, body } => {
                assert_eq!(params.len(), 1);
                assert!(matches!(&params[0], Type::Builtin(n) if n == "Int"));
                assert_eq!(**body, Expr::LocalVar(0));
            }
            other => panic!("expected Lambda, got {:?}", other),
        }
        // ret should be FnType { params: [Int], ret: Int }
        match ret {
            Type::FnType { params, ret } => {
                assert_eq!(params.len(), 1);
                assert!(matches!(&params[0], Type::Builtin(n) if n == "Int"));
                assert!(matches!(ret.as_ref(), Type::Builtin(n) if n == "Int"));
            }
            other => panic!("expected FnType, got {:?}", other),
        }
    }

    #[test]
    fn lambda_with_capture_uses_outer_de_bruijn() {
        let m = parse_module("def f(n: Int) -> fn(Int) -> Int = |x: Int| x + n").unwrap();
        let r = resolve_module(&m).unwrap();
        let Def::Fn { body, .. } = &r.defs[0].def else { panic!("expected Fn") };
        let Expr::Lambda { params, body: lbody } = body else {
            panic!("expected Lambda");
        };
        assert_eq!(params.len(), 1);
        // body should be Call(BuiltinRef("core/i64.add"), [LocalVar(0), LocalVar(1)])
        let Expr::Call(callee, args) = lbody.as_ref() else {
            panic!("expected Call in lambda body");
        };
        assert_eq!(**callee, Expr::BuiltinRef("core/i64.add".to_owned()));
        assert_eq!(args[0], Expr::LocalVar(0)); // x (innermost)
        assert_eq!(args[1], Expr::LocalVar(1)); // n (captured)
    }

    #[test]
    fn zero_arg_lambda_resolves() {
        let m = parse_module("def make() -> fn() -> Int = || 42").unwrap();
        let r = resolve_module(&m).unwrap();
        let Def::Fn { body, .. } = &r.defs[0].def else { panic!("expected Fn") };
        match body {
            Expr::Lambda { params, body } => {
                assert!(params.is_empty());
                assert_eq!(**body, Expr::IntLit(42));
            }
            other => panic!("expected Lambda, got {:?}", other),
        }
    }

    #[test]
    fn lambdas_with_different_param_types_hash_differently() {
        // `|x: Int| x` and `|x: fn()->Int| x` are different defs even
        // though their bodies look identical — different argument types
        // mean different identity.
        let m1 = parse_module("def f() -> fn(Int) -> Int = |x: Int| x").unwrap();
        let m2 = parse_module("def f() -> fn(fn() -> Int) -> fn() -> Int = |x: fn() -> Int| x")
            .unwrap();
        let r1 = resolve_module(&m1).unwrap();
        let r2 = resolve_module(&m2).unwrap();
        assert_ne!(r1.defs[0].hash, r2.defs[0].hash);
    }

    #[test]
    fn lambdas_with_same_canonical_body_hash_equal() {
        // Identical lambdas modulo parameter names hash equally
        // (parameter names aren't in the canonical AST).
        let m1 = parse_module("def f() -> fn(Int) -> Int = |x: Int| x * 2").unwrap();
        let m2 = parse_module("def f() -> fn(Int) -> Int = |y: Int| y * 2").unwrap();
        let r1 = resolve_module(&m1).unwrap();
        let r2 = resolve_module(&m2).unwrap();
        assert_eq!(r1.defs[0].hash, r2.defs[0].hash);
    }

    #[test]
    fn changing_body_changes_hash() {
        let a = resolve_one("def f(x: Int) -> Int = x * 2");
        let b = resolve_one("def f(x: Int) -> Int = x * 3");
        let ah = Hash::of_bytes(&encode_def(&a));
        let bh = Hash::of_bytes(&encode_def(&b));
        assert_ne!(ah, bh);
    }
}
