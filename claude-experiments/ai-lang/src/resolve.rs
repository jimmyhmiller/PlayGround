//! Resolver: surface AST → canonical AST.
//!
//! Responsibilities:
//!
//! - **Locals → de Bruijn.** Lexical variable lookups become `LocalVar(idx)`,
//!   where `idx` counts outwards from the innermost binder (parameter 0 of
//!   the current function is the outermost binder for that function).
//! - **Top-level names → hashes.** Each `def` is processed in source order,
//!   producing a `Hash`. References to a previously-resolved `def` become
//!   `TopRef(hash)`.
//! - **Operators → builtins.** `+`, `*`, `==`, etc. lower to `BuiltinRef`
//!   under stable string ids (`core/i64.add`, `core/bool.and`, …). For v1
//!   we only have `Int` and `Bool`, so the dispatch is static; a real
//!   typechecker will replace this with type-directed overload resolution.
//! - **Named types → builtins.** `Int`/`Bool`/`String`/`Float`/`Bytes` map
//!   to `Type::Builtin`. Anything else is an error in v1 (no user types yet).
//!
//! Out of scope for v1 (each errors cleanly, never silently):
//!
//! - Mutually recursive top-level groups — must be processable in source
//!   order. A `def` referring to a later `def` errors with `UnknownName`,
//!   not a stub `SelfRef`. Component hashing (Tarjan SCC over the call
//!   graph) is a follow-up.
//! - Closures, lambdas, `let`-expressions, `if`/`match` — the parser does
//!   not yet produce these, so the resolver does not yet handle them.

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
                 forward references and mutual recursion are not yet supported",
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedDef {
    pub name: String,
    pub hash: Hash,
    pub def: Def,
}

/// The result of resolving a module: defs in source order, each with its
/// canonical form and content hash.
#[derive(Debug, Clone, PartialEq, Eq)]
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
pub fn parse_at_builtin_name(name: &str) -> Option<(Hash, Option<Hash>)> {
    let body = name.strip_prefix(AT_BUILTIN_PREFIX)?;
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

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }
    s
}

pub fn resolve_module(m: &Module) -> Result<ResolvedModule, ResolveError> {
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
                    for (i, (vname, payload_ty)) in vs.iter().enumerate() {
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
                                enum_ref: hash,
                                index: i as u32,
                                payload: payload_ty.clone(),
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
        if let SurfaceDefKind::Extern { params, ret } = &sd.kind {
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

    struct FnPending {
        sd_idx: usize,
        type_params: Vec<String>,
        params_canon: Vec<Type>,
        ret_canon: Type,
        is_local: bool,
    }

    // 2a.
    let mut pendings: Vec<FnPending> = Vec::new();
    for (sd_idx, sd) in m.defs.iter().enumerate() {
        if let SurfaceDefKind::Fn {
            type_params,
            params,
            ret,
            is_local,
            ..
        } = &sd.kind
        {
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
            pendings.push(FnPending {
                sd_idx,
                type_params: type_params.clone(),
                params_canon,
                ret_canon,
                is_local: *is_local,
            });
        }
    }

    // 2b.
    let mut bodies: Vec<Expr> = Vec::with_capacity(pendings.len());
    for p in &pendings {
        let sd = &m.defs[p.sd_idx];
        let SurfaceDefKind::Fn {
            params, body: src_body, ..
        } = &sd.kind
        else {
            unreachable!("only Fn defs are in pendings")
        };
        let mut env: Vec<(String, Type)> = params
            .iter()
            .zip(p.params_canon.iter())
            .map(|((n, _), t)| (n.clone(), t.clone()))
            .collect();
        let body_canon = resolve_expr(
            src_body,
            &mut env,
            &top,
            &structs,
            &enums,
            &variants,
            &mut at_binding,
            &p.type_params,
            &sd.name,
        )?;
        bodies.push(body_canon);
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

        // Phase 2: compute the content hash of each member.
        let mut scc_hashes: Vec<Hash> = Vec::with_capacity(scc_sorted.len());
        for (local, &global_idx) in scc_sorted.iter().enumerate() {
            let p = &pendings[global_idx as usize];
            let def = Def::Fn {
                is_local: p.is_local,
                type_params: p.type_params.len() as u32,
                params: p.params_canon.clone(),
                ret: p.ret_canon.clone(),
                body: canonical_bodies[local].clone(),
            };
            scc_hashes.push(Hash::of_bytes(&encode_def(&def)));
        }

        // Phase 3: build the stored Def per member.
        for (local, &global_idx) in scc_sorted.iter().enumerate() {
            let stored_body =
                rewrite_selfref_to_topref(&canonical_bodies[local], &scc_hashes);
            let p = &pendings[global_idx as usize];
            let sd = &m.defs[p.sd_idx];
            let def = Def::Fn {
                is_local: p.is_local,
                type_params: p.type_params.len() as u32,
                params: p.params_canon.clone(),
                ret: p.ret_canon.clone(),
                body: stored_body,
            };
            let hash = scc_hashes[local];
            real_hash[global_idx as usize] = Some(hash);
            let ty = Type::FnType {
                params: p.params_canon.clone(),
                ret: Box::new(p.ret_canon.clone()),
            };
            top.insert(
                sd.name.clone(),
                TopBinding {
                    hash,
                    ty,
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

    Ok(ResolvedModule {
        defs: out,
        at_binding,
        externs,
    })
}

/// Walk `e` and collect every `Expr::SelfRef(idx)` index into `out`.
fn collect_self_refs(e: &Expr, out: &mut std::collections::BTreeSet<u32>) {
    match e {
        Expr::SelfRef(i) => {
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
        Expr::IntLit(_)
        | Expr::BoolLit(_)
        | Expr::StringLit(_)
        | Expr::LocalVar(_)
        | Expr::TopRef(_)
        | Expr::BuiltinRef(_) => {}
    }
}

/// Rewrite each `SelfRef(global_idx)` in `e`:
///   - if `global_idx` is in this SCC (per `local_of`): stays as
///     `SelfRef(local_idx)` — load-bearing for canonical hashing.
///   - else: becomes `TopRef(real_hash[global_idx])`, which must be
///     populated because earlier (dependency) SCCs were already
///     processed.
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
        Expr::IntLit(_)
        | Expr::BoolLit(_)
        | Expr::StringLit(_)
        | Expr::LocalVar(_)
        | Expr::TopRef(_)
        | Expr::BuiltinRef(_) => e.clone(),
    }
}

/// Rewrite any remaining `SelfRef(local_idx)` in `e` to
/// `TopRef(scc_hashes[local_idx])`. Used to produce the STORED form
/// of an SCC member's body (the form codegen + typecheck consume).
fn rewrite_selfref_to_topref(e: &Expr, scc_hashes: &[Hash]) -> Expr {
    match e {
        Expr::SelfRef(local) => Expr::TopRef(scc_hashes[*local as usize]),
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
        Expr::IntLit(_)
        | Expr::BoolLit(_)
        | Expr::StringLit(_)
        | Expr::LocalVar(_)
        | Expr::TopRef(_)
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
fn tarjan_scc(adj: &[Vec<u32>]) -> Vec<Vec<u32>> {
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

    // Validate Result shape. Two flavours are accepted:
    //   - Monomorphic: `enum Result { Ok(Int), Err(Failure) }`
    //   - Generic   : `enum Result<T, E> { Ok(T), Err(E) }`
    // The generic form is preferred per the proposal; the monomorphic
    // form is retained for back-compat with existing tests/examples.
    let ok_idx = find_variant(result, "Ok").ok_or(ResolveError::AtBindingShape {
        what: "enum Result must declare variant `Ok`".to_owned(),
        span,
    })?;
    let err_idx = find_variant(result, "Err").ok_or(ResolveError::AtBindingShape {
        what: "enum Result must declare variant `Err`".to_owned(),
        span,
    })?;
    match (result.type_params.len(), &result.variants[ok_idx as usize].1) {
        // Generic: Ok(TypeVar(0))
        (2, Some(Type::TypeVar(0))) => {}
        // Monomorphic: Ok(Int)
        (0, Some(Type::Builtin(t))) if t == "Int" => {}
        _ => {
            return Err(ResolveError::AtBindingShape {
                what: "Result::Ok must carry Int (monomorphic) or T (generic Result<T, E>)"
                    .to_owned(),
                span,
            });
        }
    }
    match (result.type_params.len(), &result.variants[err_idx as usize].1) {
        // Generic: Err(TypeVar(1))
        (2, Some(Type::TypeVar(1))) => {}
        // Monomorphic: Err(Failure)
        (0, Some(Type::TypeRef(h))) if *h == failure.hash => {}
        _ => {
            return Err(ResolveError::AtBindingShape {
                what: "Result::Err must carry Failure (monomorphic) or E (generic Result<T, E>)"
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
    })
}

fn find_variant(info: &EnumInfo, name: &str) -> Option<u32> {
    info.variants
        .iter()
        .position(|(n, _)| n == name)
        .map(|i| i as u32)
}

#[derive(Clone)]
struct EnumInfo {
    hash: Hash,
    /// Names of declared type params, in declaration order. Empty for
    /// non-generic enums. Used to validate `Foo<...>` applications.
    type_params: Vec<String>,
    variants: Vec<(String, Option<Type>)>,
}

#[derive(Clone)]
struct VariantBinding {
    /// The surface name of the enclosing enum (for error messages).
    enum_surface_name: String,
    /// Content hash of the enum this variant belongs to.
    enum_ref: Hash,
    /// Position of this variant within the enum's `variants` list.
    index: u32,
    /// Payload type, if any.
    payload: Option<Type>,
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
            "Int" | "Bool" | "String" | "Float" | "Bytes" => {
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
            let head = resolve_name_head(name).ok_or(ResolveError::UnknownType {
                name: name.clone(),
                span: *name_span,
            })?;
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
            // Nullary variant in expression position: `None`, `Empty`, etc.
            if let Some(v) = variants.get(name) {
                if v.payload.is_some() {
                    return Err(ResolveError::VariantArityMismatch {
                        variant: name.clone(),
                        expected_payload: true,
                        span: *span,
                    });
                }
                return Ok((
                    Expr::EnumNew {
                        enum_ref: v.enum_ref,
                        variant_index: v.index,
                        payload: None,
                    },
                    Type::TypeRef(v.enum_ref),
                ));
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
            Err(ResolveError::UnknownName {
                name: name.clone(),
                span: *span,
            })
        }

        SurfaceExpr::Call { callee, args, span } => {
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
                if name == "at" {
                    if args.len() != 2 {
                        return Err(ResolveError::UnknownName {
                            name: format!("at expects 2 args, got {}", args.len()),
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
                    let thunk_canon =
                        resolve_expr(&args[1], env, top, structs, enums, variants, at_binding, type_params, self_name)?;
                    // If the user's Result is generic (`Result<T, E>`),
                    // embed the Failure hash in the builtin name so the
                    // typechecker / codegen can reconstruct the
                    // Apply'd return type `Result<Int, Failure>`.
                    let is_generic_result = enums
                        .get("Result")
                        .map(|e| e.type_params.len() == 2)
                        .unwrap_or(false);
                    let builtin_name = if is_generic_result {
                        format!(
                            "{}{}#{}",
                            AT_BUILTIN_PREFIX,
                            hex_encode(binding.result_hash.as_bytes()),
                            hex_encode(binding.failure_hash.as_bytes())
                        )
                    } else {
                        format!(
                            "{}{}",
                            AT_BUILTIN_PREFIX,
                            hex_encode(binding.result_hash.as_bytes())
                        )
                    };
                    let ret_ty = if is_generic_result {
                        Type::Apply(
                            Box::new(Type::TypeRef(binding.result_hash)),
                            vec![
                                Type::Builtin("Int".to_owned()),
                                Type::TypeRef(binding.failure_hash),
                            ],
                        )
                    } else {
                        Type::TypeRef(binding.result_hash)
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

            // Detect variant-constructor calls: `Some(x)`, `Circle(r)`.
            // The callee is a bare `Var` whose name is a registered
            // variant. Rewrite to `EnumNew { ..., payload: Some(arg) }`.
            if let SurfaceExpr::Var {
                name,
                span: var_span,
            } = callee.as_ref()
            {
                if let Some(v) = variants.get(name) {
                    if v.payload.is_none() {
                        return Err(ResolveError::VariantArityMismatch {
                            variant: name.clone(),
                            expected_payload: false,
                            span: *var_span,
                        });
                    }
                    if args.len() != 1 {
                        return Err(ResolveError::VariantArityMismatch {
                            variant: name.clone(),
                            expected_payload: true,
                            span: *span,
                        });
                    }
                    let v = v.clone();
                    let (payload, payload_ty) = resolve_expr_typed(
                        &args[0], env, top, structs, enums, variants, at_binding,
                        type_params, self_name,
                    )?;
                    // Bottom-up infer the enum's type-arg substitution by
                    // unifying the variant's declared payload type
                    // against the actual arg's resolved type. If the
                    // enum is non-generic this is a no-op and the
                    // result type stays bare TypeRef.
                    let n_params = enums
                        .get(&v.enum_surface_name)
                        .map(|e| e.type_params.len())
                        .unwrap_or(0);
                    let result_ty = if n_params > 0 {
                        let mut subst: Vec<Option<Type>> = vec![None; n_params];
                        if let Some(decl) = v.payload.as_ref() {
                            unify_resolver(decl, &payload_ty, &mut subst);
                        }
                        if subst.iter().any(|s| s.is_some()) {
                            build_instantiated(v.enum_ref, n_params, &subst)
                        } else {
                            Type::TypeRef(v.enum_ref)
                        }
                    } else {
                        Type::TypeRef(v.enum_ref)
                    };
                    return Ok((
                        Expr::EnumNew {
                            enum_ref: v.enum_ref,
                            variant_index: v.index,
                            payload: Some(Box::new(payload)),
                        },
                        result_ty,
                    ));
                }
            }

            let (callee_r, callee_ty) =
                resolve_expr_typed(callee, env, top, structs, enums, variants, at_binding, type_params, self_name)?;
            let args_r: Result<Vec<Expr>, _> = args
                .iter()
                .map(|a| resolve_expr(a, env, top, structs, enums, variants, at_binding, type_params, self_name))
                .collect();
            let ret_ty = match &callee_ty {
                Type::FnType { ret, .. } => (**ret).clone(),
                _ => Type::Builtin("Int".to_owned()),
            };
            Ok((Expr::Call(Box::new(callee_r), args_r?), ret_ty))
        }

        SurfaceExpr::BinOp {
            op, left, right, ..
        } => {
            let l = resolve_expr(left, env, top, structs, enums, variants, at_binding, type_params, self_name)?;
            let r = resolve_expr(right, env, top, structs, enums, variants, at_binding, type_params, self_name)?;
            // All v1 binops return Int (arithmetic) or Int (comparison widened to i64).
            Ok((
                Expr::Call(
                    Box::new(Expr::BuiltinRef(binop_builtin(*op).to_owned())),
                    vec![l, r],
                ),
                Type::Builtin("Int".to_owned()),
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
            // `{ let x = e1; let y = e2; tail }` lowers to
            // `Let { e1, Let { e2, tail } }`. Each let extends env by 1.
            let pushed = stmts.len();
            let mut values_canon: Vec<Expr> = Vec::with_capacity(pushed);
            for s in stmts {
                let SurfaceStmt::Let { name, value, .. } = s;
                let (v_canon, v_ty) =
                    resolve_expr_typed(value, env, top, structs, enums, variants, at_binding, type_params, self_name)?;
                values_canon.push(v_canon);
                env.push((name.clone(), v_ty));
            }
            let (mut body_canon, body_ty) =
                resolve_expr_typed(tail, env, top, structs, enums, variants, at_binding, type_params, self_name)?;
            for _ in 0..pushed {
                env.pop();
            }
            for v in values_canon.into_iter().rev() {
                body_canon = Expr::Let {
                    value: Box::new(v),
                    body: Box::new(body_canon),
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
            let (scrut_canon, _scrut_ty) =
                resolve_expr_typed(scrutinee, env, top, structs, enums, variants, at_binding, type_params, self_name)?;

            // The result type is taken from the FIRST arm; we don't
            // enforce arm-type equality here (typechecker's job).
            let mut canonical_arms: Vec<MatchArm> = Vec::with_capacity(arms.len());
            let mut shared_enum: Option<Hash> = None;
            let mut result_ty: Option<Type> = None;

            for arm in arms {
                let (pattern_canon, bind_count, pat_enum) = resolve_pattern(
                    &arm.pattern,
                    variants,
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
                // pushed in pattern-traversal order.
                let bindings = collect_pattern_bindings(&arm.pattern, variants);
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
    }
}

/// Resolve a surface pattern to a canonical pattern. Returns the
/// pattern, the number of bindings it introduces (Var counts as 1,
/// Wildcard as 0, Ctor recurses), and the enum hash (if this is a
/// variant pattern — for cross-arm consistency checks).
fn resolve_pattern(
    p: &SurfacePattern,
    variants: &HashMap<String, VariantBinding>,
) -> Result<(Pattern, u32, Option<Hash>), ResolveError> {
    match p {
        SurfacePattern::Wildcard { .. } => Ok((Pattern::Wildcard, 0, None)),
        SurfacePattern::Ident { name, span } => {
            // Could be a nullary variant pattern (`None`, `Empty`),
            // or a binding pattern.
            if let Some(v) = variants.get(name) {
                if v.payload.is_some() {
                    return Err(ResolveError::VariantArityMismatch {
                        variant: name.clone(),
                        expected_payload: true,
                        span: *span,
                    });
                }
                return Ok((
                    Pattern::Enum {
                        enum_ref: v.enum_ref,
                        variant_index: v.index,
                        payload: None,
                    },
                    0,
                    Some(v.enum_ref),
                ));
            }
            // Otherwise it's a binding.
            Ok((Pattern::Var, 1, None))
        }
        SurfacePattern::Ctor { name, payload, span } => {
            let v = variants.get(name).ok_or_else(|| ResolveError::UnknownVariant {
                name: name.clone(),
                span: *span,
            })?;
            if v.payload.is_none() {
                return Err(ResolveError::VariantArityMismatch {
                    variant: name.clone(),
                    expected_payload: false,
                    span: *span,
                });
            }
            let (sub_canon, sub_bindings, _sub_enum) = resolve_pattern(payload, variants)?;
            Ok((
                Pattern::Enum {
                    enum_ref: v.enum_ref,
                    variant_index: v.index,
                    payload: Some(Box::new(sub_canon)),
                },
                sub_bindings,
                Some(v.enum_ref),
            ))
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
) -> Vec<(String, Type)> {
    let mut out: Vec<(String, Type)> = Vec::new();
    walk_pattern_bindings(p, variants, None, &mut out);
    out
}

fn walk_pattern_bindings(
    p: &SurfacePattern,
    variants: &HashMap<String, VariantBinding>,
    payload_ty_hint: Option<&Type>,
    out: &mut Vec<(String, Type)>,
) {
    match p {
        SurfacePattern::Wildcard { .. } => {}
        SurfacePattern::Ident { name, .. } => {
            if let Some(v) = variants.get(name) {
                if v.payload.is_none() {
                    return;
                }
            }
            let ty = payload_ty_hint
                .cloned()
                .unwrap_or(Type::Builtin("Int".to_owned()));
            out.push((name.clone(), ty));
        }
        SurfacePattern::Ctor { name, payload, .. } => {
            let sub_payload_ty = variants
                .get(name)
                .and_then(|v| v.payload.clone());
            walk_pattern_bindings(payload, variants, sub_payload_ty.as_ref(), out);
        }
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

/// Best-effort unification used by the resolver to recover the type-arg
/// substitution at variant / struct construction sites. Where `pattern`
/// has a `TypeVar(i)` and `actual` is concrete, record `subst[i] = actual`.
/// Shape mismatches are silent (caller gets a partial substitution).
///
/// Returns nothing — this is bottom-up best-effort, not a strict check.
fn unify_resolver(pattern: &Type, actual: &Type, subst: &mut [Option<Type>]) {
    match (pattern, actual) {
        (Type::TypeVar(i), other) => {
            let idx = *i as usize;
            if idx < subst.len() && subst[idx].is_none() {
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
