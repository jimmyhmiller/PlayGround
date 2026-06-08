//! The edit algebra (Phase 1 subset): namespace-only rename, reverse-index
//! find-usages, and forward/reverse dependency queries.
//!
//! This module lives strictly above the hashing line: it never touches
//! `hash.rs`, `codec.rs`, or the `ast.rs` identity model. Every operation here
//! is either a pure read off the [`DependencyIndex`] or a mutation of the
//! mutable *name* layer only. No definition is ever recompiled and no def hash
//! is ever changed.
//!
//! ## The headline property
//!
//! [`rename`] is O(1) and *unbreakable*: callers reference definitions by
//! content hash (`Expr::TopRef(Hash)`), not by name. Renaming therefore only
//! moves a name alias from one hash to another; every caller's stored AST, and
//! thus every caller's own hash, is untouched. This is the property a text
//! language cannot have, and the test suite below proves it directly.
//!
//! ## No silent failures
//!
//! Every operation that cannot proceed returns a typed [`EditError`] (modelled
//! on `depindex::DepIndexError`). `rename` is a hard error if `from` does not
//! exist or if `to` already exists; it never silently overwrites. A hash-prefix
//! target that matches zero or many definitions is a hard error, never a
//! silently-wrong empty result.

use crate::ast::{Def, Expr, MatchArm, Type};
use crate::codebase::Codebase;
use crate::codec::encode_def;
use crate::depindex::DependencyIndex;
use crate::hash::Hash;
use crate::resolve::{parse_at_builtin_name, ExternalEnv, AT_BUILTIN_PREFIX};
use crate::typecheck::{typecheck_cone, typecheck_def, typecheck_module, TypeScheme};
use std::collections::HashMap;

// =============================================================================
// Errors
// =============================================================================

/// Errors from the edit algebra. Modelled on `depindex::DepIndexError`: a small
/// typed enum with a `Display` impl, `std::error::Error`, and `From` for the
/// wrapped codebase error.
#[derive(Debug)]
pub enum EditError {
    /// A name was looked up that is not in the namespace.
    NameNotFound(String),
    /// `rename`/`set_name` would overwrite an existing name. Refused.
    NameExists(String),
    /// A hex hash prefix matched no definition in the store.
    HashNotFound(String),
    /// A hex hash prefix matched more than one definition. Carries the prefix
    /// and the first few full hashes it matched, so the caller can disambiguate.
    AmbiguousHashPrefix { prefix: String, matches: Vec<Hash> },
    /// A target string is neither a known name nor a usable hex hash prefix
    /// (e.g. contains non-hex characters and is not a name).
    BadTarget(String),
    /// `update` source failed to parse.
    ParseError(String),
    /// `update` source parsed but did not contain exactly one definition.
    /// `update` edits a single named def; ambiguous input is refused.
    NotSingleDef { found: usize },
    /// The single def in the `update` source had a name different from the
    /// name being updated. Refused (use `add` + `rename` for that).
    NameMismatch { expected: String, found: String },
    /// `update` source failed to resolve against the codebase namespace.
    ResolveError(String),
    /// A definition in the cone (or the edited def) could not be typechecked
    /// even mechanically (a kernel invariant violation, not a user type error
    /// — user type errors become `todos`, never this).
    TypeError(String),
    /// Wrapped codebase error (load/store/name persistence).
    Codebase(crate::codebase::CodebaseError),
    /// A Phase 3 refactor was asked to do something it cannot perform exactly
    /// (e.g. inline a def referenced in value position, extract an expression
    /// capturing a non-parameter binder, reorder with a non-permutation). This
    /// is always a clear, typed HARD ERROR — never a silent no-op or a wrong
    /// partial result.
    Unsupported(String),
}

impl From<crate::resolve::ResolveError> for EditError {
    fn from(e: crate::resolve::ResolveError) -> Self {
        EditError::ResolveError(e.to_string())
    }
}

impl From<crate::typecheck::TypeError> for EditError {
    fn from(e: crate::typecheck::TypeError) -> Self {
        EditError::TypeError(e.to_string())
    }
}

impl From<crate::codebase::CodebaseError> for EditError {
    fn from(e: crate::codebase::CodebaseError) -> Self {
        EditError::Codebase(e)
    }
}

impl From<crate::depindex::DepIndexError> for EditError {
    fn from(e: crate::depindex::DepIndexError) -> Self {
        // A failed rebuild is, at root, a codebase / store problem; surface it
        // through Display so callers see the underlying cause without a new
        // variant the CLI would have to special-case.
        match e {
            crate::depindex::DepIndexError::Codebase(c) => EditError::Codebase(c),
            other => EditError::BadTarget(format!("dependency index rebuild failed: {}", other)),
        }
    }
}

impl std::fmt::Display for EditError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EditError::NameNotFound(n) => write!(f, "no such name: {}", n),
            EditError::NameExists(n) => {
                write!(f, "name already exists (refusing to overwrite): {}", n)
            }
            EditError::HashNotFound(p) => write!(f, "no definition matches hash prefix: {}", p),
            EditError::AmbiguousHashPrefix { prefix, matches } => write!(
                f,
                "hash prefix {} is ambiguous, matches {} definitions (e.g. {})",
                prefix,
                matches.len(),
                matches
                    .iter()
                    .take(3)
                    .map(|h| h.to_hex())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            EditError::BadTarget(t) => {
                write!(f, "target is neither a known name nor a hex hash prefix: {}", t)
            }
            EditError::ParseError(m) => write!(f, "parse error: {}", m),
            EditError::NotSingleDef { found } => write!(
                f,
                "update source must contain exactly one definition, found {}",
                found
            ),
            EditError::NameMismatch { expected, found } => write!(
                f,
                "update source defines `{}` but the name being updated is `{}`",
                found, expected
            ),
            EditError::ResolveError(m) => write!(f, "resolve error: {}", m),
            EditError::TypeError(m) => write!(f, "type error: {}", m),
            EditError::Codebase(e) => write!(f, "codebase error: {}", e),
            EditError::Unsupported(m) => write!(f, "unsupported refactor: {}", m),
        }
    }
}

impl std::error::Error for EditError {}

// =============================================================================
// Result structs (the structured changelog / query results)
// =============================================================================

/// A reference to a definition: its content hash plus the name it currently
/// resolves through (if any). `name` is `None` when the hash has no alias in the
/// current namespace (e.g. an intermediate def reachable only by hash).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefRef {
    pub name: Option<String>,
    pub hash: Hash,
}

impl DefRef {
    fn new(name: Option<String>, hash: Hash) -> Self {
        DefRef { name, hash }
    }
}

/// A single definition's hash transition, with the name it resolves through
/// (if any). Used for both the directly-edited def (`updated`) and the
/// dependents that were automatically re-pointed (`propagated`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Change {
    /// The name the def resolves through, if any.
    pub name: Option<String>,
    /// The hash before the edit.
    pub old: Hash,
    /// The hash after the edit.
    pub new: Hash,
}

/// The structured changelog returned by edit operations.
#[derive(Debug, Clone, Default)]
pub struct EditResult {
    /// Each rename performed: (from_name, to_name, the unchanged target hash).
    pub renamed: Vec<(String, String, Hash)>,
    /// The definition(s) directly changed by `update` (old -> new hash).
    pub updated: Vec<Change>,
    /// Dependents automatically re-pointed up the cone (old -> new hash).
    pub propagated: Vec<Change>,
    /// Dependents that no longer typecheck after a type-changing `update`.
    pub todos: Vec<crate::typecheck::Todo>,
    /// `true` if the update was a no-op (new source hashed to the old hash).
    pub no_op: bool,
}

impl EditResult {
    fn new() -> Self {
        EditResult::default()
    }
}

/// Direction of a dependency query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// What the target references (forward edges).
    Forward,
    /// What references the target (reverse edges).
    Reverse,
}

// =============================================================================
// Index loading
// =============================================================================

/// Load the persisted dependency index, falling back to a full rebuild from the
/// store if the cache is absent or unreadable. The persisted index is only ever
/// a cache of what `rebuild_from_codebase` produces, so this is always safe.
pub fn load_or_rebuild_index(cb: &Codebase) -> Result<DependencyIndex, EditError> {
    match DependencyIndex::load(cb.root()) {
        Ok(idx) => Ok(idx),
        Err(_) => DependencyIndex::rebuild_from_codebase(cb).map_err(EditError::from),
    }
}

// =============================================================================
// rename (namespace-only, O(1), unbreakable)
// =============================================================================

/// Rename `from` to `to` at the namespace layer only.
///
/// This looks up `from`'s hash, points `to` at that same hash, and removes the
/// `from` alias. It recompiles nothing and changes no def hash, so it cannot
/// break any caller (callers hold the hash, not the name).
///
/// Hard errors:
///   - [`EditError::NameNotFound`] if `from` is not a known name.
///   - [`EditError::NameExists`] if `to` already exists (we refuse to silently
///     overwrite an existing binding).
///
/// A no-op rename (`from == to`) is accepted and reported as a single renamed
/// entry, since the postcondition (`to` resolves to the hash, no other change)
/// already holds.
pub fn rename(cb: &mut Codebase, from: &str, to: &str) -> Result<EditResult, EditError> {
    let hash = cb
        .get_name(from)
        .ok_or_else(|| EditError::NameNotFound(from.to_owned()))?;

    if from == to {
        // Already in the desired state; nothing to move.
        let mut res = EditResult::new();
        res.renamed.push((from.to_owned(), to.to_owned(), hash));
        return Ok(res);
    }

    // Refuse to clobber an existing target name.
    if cb.get_name(to).is_some() {
        return Err(EditError::NameExists(to.to_owned()));
    }

    // Point the new name at the (unchanged) hash, then drop the old alias.
    cb.set_name(to.to_owned(), hash)?;
    cb.remove_name(from)?;

    let mut res = EditResult::new();
    res.renamed.push((from.to_owned(), to.to_owned(), hash));
    Ok(res)
}

// =============================================================================
// Target resolution (name | hex hash prefix)
// =============================================================================

/// Resolve a target string to a single definition hash. The target is either:
///   - an exact name in the current namespace, OR
///   - a hex hash prefix (>= 1 hex char) that uniquely identifies a stored def.
///
/// A name takes precedence over a hash-prefix interpretation (a name is the
/// agent's explicit handle). If the string is not a name, it is treated as a
/// hex prefix and matched against every `defs/<hex>.def` file.
pub fn resolve_target(cb: &Codebase, target: &str) -> Result<Hash, EditError> {
    if let Some(h) = cb.get_name(target) {
        return Ok(h);
    }
    resolve_hash_prefix(cb, target)
}

/// Match a hex hash prefix against the def store. Errors if zero or multiple
/// defs match. Never returns a wrong/empty silent result.
fn resolve_hash_prefix(cb: &Codebase, prefix: &str) -> Result<Hash, EditError> {
    let lower = prefix.to_ascii_lowercase();
    if lower.is_empty() || !lower.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(EditError::BadTarget(prefix.to_owned()));
    }

    let defs_dir = cb.root().join("defs");
    let mut matches: Vec<Hash> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&defs_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("def") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if stem.starts_with(&lower) {
                if let Some(h) = hash_from_hex(stem) {
                    matches.push(h);
                }
            }
        }
    }

    match matches.len() {
        0 => Err(EditError::HashNotFound(prefix.to_owned())),
        1 => Ok(matches[0]),
        _ => Err(EditError::AmbiguousHashPrefix {
            prefix: prefix.to_owned(),
            matches,
        }),
    }
}

// =============================================================================
// find_usages and deps (pure reads off the DependencyIndex)
// =============================================================================

/// Find every definition that references `target` (a name or hex hash prefix).
/// This is a reverse-index lookup, O(dependents). Each dependent hash is mapped
/// back to its name(s) via the codebase namespace.
///
/// Results are sorted (by name then hash) for deterministic output.
pub fn find_usages(
    cb: &Codebase,
    index: &DependencyIndex,
    target: &str,
) -> Result<Vec<DefRef>, EditError> {
    let hash = resolve_target(cb, target)?;
    let dependents = index.dependents(&hash);
    Ok(to_defrefs(cb, dependents.into_iter()))
}

/// Forward (`Forward`) or reverse (`Reverse`) dependency query for `target`,
/// either direct (`transitive == false`) or to a fixpoint (`transitive ==
/// true`). Results are sorted for deterministic output.
pub fn deps(
    cb: &Codebase,
    index: &DependencyIndex,
    target: &str,
    direction: Direction,
    transitive: bool,
) -> Result<Vec<DefRef>, EditError> {
    let hash = resolve_target(cb, target)?;
    let set = match (direction, transitive) {
        (Direction::Forward, false) => index.dependencies(&hash),
        (Direction::Forward, true) => index.transitive_dependencies(&hash),
        (Direction::Reverse, false) => index.dependents(&hash),
        (Direction::Reverse, true) => index.transitive_dependents(&hash),
    };
    Ok(to_defrefs(cb, set.into_iter()))
}

/// Map a set of hashes to sorted `DefRef`s, recovering each hash's name (if any)
/// from the codebase. Sorted by (name-or-empty, hash-hex) for determinism.
fn to_defrefs(cb: &Codebase, hashes: impl Iterator<Item = Hash>) -> Vec<DefRef> {
    let mut out: Vec<DefRef> = hashes
        .map(|h| DefRef::new(name_for_hash(cb, &h), h))
        .collect();
    out.sort_by(|a, b| {
        let an = a.name.clone().unwrap_or_default();
        let bn = b.name.clone().unwrap_or_default();
        an.cmp(&bn).then_with(|| a.hash.to_hex().cmp(&b.hash.to_hex()))
    });
    out
}

/// Recover a single deterministic name for a hash from the namespace: the
/// lexicographically smallest alias, or `None` if the hash has no name.
fn name_for_hash(cb: &Codebase, hash: &Hash) -> Option<String> {
    let mut best: Option<&String> = None;
    for (name, h) in cb.names() {
        if h == hash {
            best = match best {
                Some(b) if b <= name => Some(b),
                _ => Some(name),
            };
        }
    }
    best.cloned()
}

// =============================================================================
// Hex helpers (local; no shared private fn exported from depindex)
// =============================================================================

/// Parse a 64-char lowercase hex string into a `Hash`. Returns `None` on any
/// length or character error.
fn hash_from_hex(hex: &str) -> Option<Hash> {
    if hex.len() != 64 {
        return None;
    }
    let bytes = hex.as_bytes();
    let mut out = [0u8; 32];
    for i in 0..32 {
        let hi = hex_val(bytes[2 * i])?;
        let lo = hex_val(bytes[2 * i + 1])?;
        out[i] = (hi << 4) | lo;
    }
    Some(Hash(out))
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

// =============================================================================
// add: ingest new source into the codebase (resolved against the current
// namespace, typechecked, stored, named). The engine path behind the server's
// `add` op and a direct analog of the CLI `add` (minus the stdlib concat: the
// server adds against the already-stored namespace via the ExternalEnv).
// =============================================================================

/// Parse + resolve + typecheck `source` against the codebase's current
/// namespace, store every resolved def under its content hash, and point each
/// surface name at that hash. Returns a [`DefRef`] per added top-level def.
///
/// Resolution uses the same [`ExternalEnv`] as [`update`], so the new source may
/// reference any already-stored top-level def by name. A parse / resolve /
/// typecheck failure is a hard, typed error and stores nothing extra beyond the
/// content-addressed (harmless) def bytes the resolver may have produced.
pub fn add(cb: &mut Codebase, source: &str) -> Result<Vec<DefRef>, EditError> {
    let module = crate::parser::parse_module(source)
        .map_err(|e| EditError::ParseError(format!("{:?}", e)))?;
    if module.defs.is_empty() {
        return Err(EditError::NotSingleDef { found: 0 });
    }
    let env = build_external_env(cb);
    let rm = crate::resolve::resolve_module_with_env(&module, &env)?;

    // Typecheck the whole new module against the codebase's cached schemes,
    // seeded from the codebase's types. We use `typecheck_module` (not a manual
    // per-def loop) precisely because it runs a PROVISIONAL pre-pass: it inserts
    // each new def's declared signature into the cache BEFORE checking any body,
    // so a self-recursive body (`fact` calling `TopRef(fact_hash)`) or a
    // mutually-recursive group (`even`/`odd`) — and recursive type defs — all
    // resolve their own/peer references. A manual one-at-a-time loop fails these
    // with "unknown top-level ref". A user type error is still a hard, typed
    // error here (not a silent partial add).
    let mut work_cache = cb.types().clone();
    typecheck_module(&rm, &mut work_cache)?;

    // Commit: store defs + names + types. `store_resolved_module` writes the
    // resolver-canonical hashes (honouring recursive-type SelfRef forms).
    cb.store_resolved_module(&rm)?;
    cb.store_typecache(&work_cache)?;

    // Capture the author's surface local/param names into the side-car keyed by
    // content hash (outside the hashed bytes, so identity stays rename-invariant)
    // so `view`/print_def can replay readable names instead of `p0, p1, ...`.
    if let Ok(names) = crate::resolve::local_names_for_module_with_env(&module, &env) {
        let _ = cb.store_local_names_batch(&names);
    }

    // Update + persist the dependency index incrementally for the new defs.
    let mut idx = load_or_rebuild_index(cb)?;
    for rd in &rm.defs {
        idx.add_def(rd.hash, &rd.def);
    }
    let _ = idx.save(cb.root());

    Ok(rm
        .defs
        .iter()
        .map(|rd| DefRef::new(Some(rd.name.clone()), rd.hash))
        .collect())
}

// =============================================================================
// update (the heart): change a def's content and propagate the new hash up
// its entire dependency cone, Unison-style.
// =============================================================================

/// Build an [`ExternalEnv`] from the codebase's current namespace + type
/// cache, so an edited definition can reference any already-stored top-level
/// def by name. Only named, typechecked defs are seeded (that is exactly the
/// set the author can mention by name in the edited source).
fn build_external_env(cb: &Codebase) -> ExternalEnv {
    let mut env = ExternalEnv::new();
    for (name, hash) in cb.names() {
        match cb.types().get(hash) {
            Some(TypeScheme::Fn { params, ret, .. }) => {
                env.fns
                    .insert(name.clone(), (*hash, params.clone(), ret.clone()));
            }
            Some(TypeScheme::Struct { fields, .. }) => {
                env.structs
                    .insert(name.clone(), (*hash, Vec::new(), fields.clone()));
            }
            Some(TypeScheme::Enum { variants, .. }) => {
                env.enums
                    .insert(name.clone(), (*hash, Vec::new(), variants.clone()));
            }
            Some(TypeScheme::State { .. }) => {
                // Node `state` bindings are not yet seeded into the edit
                // layer's external env; an edited def referencing a state
                // gets a clean resolver "unknown name" error rather than a
                // silently wrong result. (Edit-layer state support is a
                // follow-up; node-state semantics themselves don't need it.)
            }
            None => {
                // No cached type: an edited def referencing this name will get
                // a clean resolver "unknown name" error rather than a silently
                // wrong result. Every stored def should be typed, so this is
                // an edge case, handled loudly downstream.
            }
        }
    }
    env
}

/// Rewrite every content-hash reference inside a `Def` according to `remap`
/// (old hash -> new hash). Pure canonical-AST transform — no resolver, no
/// typecheck. Covers `TopRef`, `TypeRef`, `struct_ref`/`enum_ref` on
/// constructors / `Field` / `Match` patterns / `Try`, and the hashes embedded
/// in `core/net.at#<hex>` builtin names. References not in `remap` are left as-is.
fn remap_def(def: &Def, remap: &HashMap<Hash, Hash>) -> Def {
    match def {
        Def::Fn {
            is_local,
            type_params,
            params,
            ret,
            body,
        } => Def::Fn {
            is_local: *is_local,
            type_params: *type_params,
            params: params.iter().map(|t| remap_type(t, remap)).collect(),
            ret: remap_type(ret, remap),
            body: remap_expr(body, remap),
        },
        Def::Struct {
            type_params,
            fields,
        } => Def::Struct {
            type_params: *type_params,
            fields: fields
                .iter()
                .map(|(n, t)| (n.clone(), remap_type(t, remap)))
                .collect(),
        },
        Def::State { ty, init } => Def::State {
            ty: remap_type(ty, remap),
            init: remap_expr(init, remap),
        },
        Def::Enum {
            type_params,
            variants,
        } => Def::Enum {
            type_params: *type_params,
            variants: variants
                .iter()
                .map(|(n, p)| (n.clone(), p.as_ref().map(|t| remap_type(t, remap))))
                .collect(),
        },
    }
}

fn remap_hash(h: &Hash, remap: &HashMap<Hash, Hash>) -> Hash {
    remap.get(h).copied().unwrap_or(*h)
}

fn remap_type(t: &Type, remap: &HashMap<Hash, Hash>) -> Type {
    match t {
        Type::TypeRef(h) => Type::TypeRef(remap_hash(h, remap)),
        Type::Apply(head, args) => Type::Apply(
            Box::new(remap_type(head, remap)),
            args.iter().map(|a| remap_type(a, remap)).collect(),
        ),
        Type::FnType { params, ret } => Type::FnType {
            params: params.iter().map(|p| remap_type(p, remap)).collect(),
            ret: Box::new(remap_type(ret, remap)),
        },
        Type::Builtin(_) | Type::TypeVar(_) | Type::SelfRef(_) => t.clone(),
    }
}

fn remap_at_builtin_name(name: &str, remap: &HashMap<Hash, Hash>) -> String {
    match parse_at_builtin_name(name) {
        Some((primary, secondary)) => {
            let p = remap_hash(&primary, remap);
            match secondary {
                Some(s) => format!(
                    "{}{}#{}",
                    AT_BUILTIN_PREFIX,
                    p.to_hex(),
                    remap_hash(&s, remap).to_hex()
                ),
                None => format!("{}{}", AT_BUILTIN_PREFIX, p.to_hex()),
            }
        }
        // A prefix-bearing name that doesn't parse is a hard error — never a
        // silent passthrough (mirrors depindex's invariant).
        None => panic!("edit: malformed hash-bearing builtin name: {:?}", name),
    }
}

fn remap_expr(e: &Expr, remap: &HashMap<Hash, Hash>) -> Expr {
    match e {
        Expr::TopRef(h) => Expr::TopRef(remap_hash(h, remap)),
        Expr::StateRef(h) => Expr::StateRef(remap_hash(h, remap)),
        Expr::StateSelfRef(_) => e.clone(),
        Expr::BuiltinRef(name) if name.starts_with(AT_BUILTIN_PREFIX) => {
            Expr::BuiltinRef(remap_at_builtin_name(name, remap))
        }
        Expr::Call(callee, args) => Expr::Call(
            Box::new(remap_expr(callee, remap)),
            args.iter().map(|a| remap_expr(a, remap)).collect(),
        ),
        Expr::Lambda { params, body } => Expr::Lambda {
            params: params.iter().map(|p| remap_type(p, remap)).collect(),
            body: Box::new(remap_expr(body, remap)),
        },
        Expr::Let { value, body } => Expr::Let {
            value: Box::new(remap_expr(value, remap)),
            body: Box::new(remap_expr(body, remap)),
        },
        Expr::Defer { cleanup, body } => Expr::Defer {
            cleanup: Box::new(remap_expr(cleanup, remap)),
            body: Box::new(remap_expr(body, remap)),
        },
        Expr::StructNew { struct_ref, fields } => Expr::StructNew {
            struct_ref: remap_hash(struct_ref, remap),
            fields: fields.iter().map(|f| remap_expr(f, remap)).collect(),
        },
        Expr::Field {
            base,
            struct_ref,
            index,
        } => Expr::Field {
            base: Box::new(remap_expr(base, remap)),
            struct_ref: remap_hash(struct_ref, remap),
            index: *index,
        },
        Expr::EnumNew {
            enum_ref,
            variant_index,
            payload,
        } => Expr::EnumNew {
            enum_ref: remap_hash(enum_ref, remap),
            variant_index: *variant_index,
            payload: payload.as_ref().map(|p| Box::new(remap_expr(p, remap))),
        },
        Expr::Match { scrutinee, arms } => Expr::Match {
            scrutinee: Box::new(remap_expr(scrutinee, remap)),
            arms: arms
                .iter()
                .map(|arm| MatchArm {
                    pattern: remap_pattern(&arm.pattern, remap),
                    body: remap_expr(&arm.body, remap),
                })
                .collect(),
        },
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } => Expr::If {
            cond: Box::new(remap_expr(cond, remap)),
            then_branch: Box::new(remap_expr(then_branch, remap)),
            else_branch: Box::new(remap_expr(else_branch, remap)),
        },
        Expr::Try {
            expr,
            enum_ref,
            ok_index,
            err_index,
        } => Expr::Try {
            expr: Box::new(remap_expr(expr, remap)),
            enum_ref: remap_hash(enum_ref, remap),
            ok_index: *ok_index,
            err_index: *err_index,
        },
        Expr::IntLit(_)
        | Expr::FloatLit(_)
        | Expr::BoolLit(_)
        | Expr::StringLit(_)
        | Expr::LocalVar(_)
        | Expr::SelfRef(_)
        | Expr::BuiltinRef(_) => e.clone(),
    }
}

fn remap_pattern(p: &crate::ast::Pattern, remap: &HashMap<Hash, Hash>) -> crate::ast::Pattern {
    use crate::ast::Pattern;
    match p {
        Pattern::Wildcard | Pattern::Var => p.clone(),
        Pattern::Enum {
            enum_ref,
            variant_index,
            payload,
        } => Pattern::Enum {
            enum_ref: remap_hash(enum_ref, remap),
            variant_index: *variant_index,
            payload: payload
                .as_ref()
                .map(|sub| Box::new(remap_pattern(sub, remap))),
        },
    }
}

/// Topologically sort the cone so a def is processed only after every def it
/// depends on (that is also in the cone) has been processed. Kahn's algorithm
/// over forward edges restricted to the cone; deterministic via a BTreeSet.
fn topo_sort_cone(index: &DependencyIndex, cone: &std::collections::BTreeSet<Hash>) -> Vec<Hash> {
    let mut indeg: HashMap<Hash, usize> = cone.iter().map(|h| (*h, 0usize)).collect();
    let mut succs: HashMap<Hash, Vec<Hash>> = HashMap::new();
    for h in cone {
        for dep in index.dependencies(h) {
            if cone.contains(&dep) {
                // h depends on dep => dep must come first; edge dep -> h.
                *indeg.get_mut(h).unwrap() += 1;
                succs.entry(dep).or_default().push(*h);
            }
        }
    }
    let mut ready: std::collections::BTreeSet<Hash> = indeg
        .iter()
        .filter(|(_, d)| **d == 0)
        .map(|(h, _)| *h)
        .collect();
    let mut order: Vec<Hash> = Vec::with_capacity(cone.len());
    while let Some(h) = ready.iter().next().copied() {
        ready.remove(&h);
        order.push(h);
        if let Some(ss) = succs.get(&h) {
            for s in ss {
                let d = indeg.get_mut(s).unwrap();
                *d -= 1;
                if *d == 0 {
                    ready.insert(*s);
                }
            }
        }
    }
    // A dependency cone of immutable content-addressed defs is acyclic.
    debug_assert_eq!(order.len(), cone.len(), "cone must be acyclic");
    order
}

/// Whether two schemes have the same *signature* (what callers depend on),
/// ignoring the derived `wire` flag. A scheme-kind change is a signature change.
fn schemes_signature_eq(a: &TypeScheme, b: &TypeScheme) -> bool {
    match (a, b) {
        (
            TypeScheme::Fn { params: pa, ret: ra, .. },
            TypeScheme::Fn { params: pb, ret: rb, .. },
        ) => pa == pb && ra == rb,
        (TypeScheme::Struct { fields: fa, .. }, TypeScheme::Struct { fields: fb, .. }) => fa == fb,
        (TypeScheme::Enum { variants: va, .. }, TypeScheme::Enum { variants: vb, .. }) => va == vb,
        _ => false,
    }
}

/// Update the definition named `name` to `new_source`. The name `name` is
/// re-pointed to the new hash. Dependents are NOT touched — they continue to
/// reference the old hash (which still exists).
///
/// - **Same type:** the edit is mechanical. `todos` lists transitive dependents
///   of the old hash as a worklist — these defs still reference the old version
///   and may optionally be propagated to the new one via [`propagate`].
/// - **Type changed:** the edit succeeds. `todos` is the worklist of dependents
///   that still reference the old hash and would need source changes to work
///   with the new type. Use [`update_propagate`] to opt into automatic cone
///   rewiring (same-type only).
///
/// Transactional: the new def is stored and the name is moved atomically. Old
/// defs are never deleted. Names are the commit point.
pub fn update(cb: &mut Codebase, name: &str, new_source: &str) -> Result<EditResult, EditError> {
    update_impl(cb, name, new_source, false, false)
}

/// Like [`update`] but computes the full impact without committing (no names
/// are moved). New `.def` files may still be written (content-addressed,
/// harmless); the namespace is untouched.
pub fn update_dry_run(
    cb: &mut Codebase,
    name: &str,
    new_source: &str,
) -> Result<EditResult, EditError> {
    update_impl(cb, name, new_source, true, false)
}

/// Like [`update`] but also rewrites every transitive dependent to reference
/// the new hash. Only valid when the type signature is unchanged (same-type).
/// Type-changing edits must use [`update`] and then manually address the
/// worklist of dependents.
pub fn update_propagate(
    cb: &mut Codebase,
    name: &str,
    new_source: &str,
) -> Result<EditResult, EditError> {
    update_impl(cb, name, new_source, false, true)
}

/// Like [`update_dry_run`] but with propagation. Preview without commit.
pub fn update_dry_run_propagate(
    cb: &mut Codebase,
    name: &str,
    new_source: &str,
) -> Result<EditResult, EditError> {
    update_impl(cb, name, new_source, true, true)
}

/// Take the current hash of `name` and rewrite every transitive dependent to
/// reference it. Only valid when the dependents would typecheck against the
/// current hash (same-type guarantee). Useful after a cascade of edits where
/// you've updated both a def and its callers and now want to rewire the cone.
pub fn propagate(cb: &mut Codebase, name: &str) -> Result<EditResult, EditError> {
    propagate_impl(cb, name, false)
}

/// Like [`propagate`] but preview without commit.
pub fn propagate_dry_run(cb: &mut Codebase, name: &str) -> Result<EditResult, EditError> {
    propagate_impl(cb, name, true)
}

fn update_impl(
    cb: &mut Codebase,
    name: &str,
    new_source: &str,
    dry_run: bool,
    propagate: bool,
) -> Result<EditResult, EditError> {
    // 1. Look up the old hash.
    let old_hash = cb
        .get_name(name)
        .ok_or_else(|| EditError::NameNotFound(name.to_owned()))?;

    // 2. Parse + resolve the new source against the codebase namespace.
    let module = crate::parser::parse_module(new_source)
        .map_err(|e| EditError::ParseError(format!("{:?}", e)))?;
    if module.defs.len() != 1 {
        return Err(EditError::NotSingleDef {
            found: module.defs.len(),
        });
    }
    let surface_name = module.defs[0].name.clone();
    if surface_name != name {
        return Err(EditError::NameMismatch {
            expected: name.to_owned(),
            found: surface_name,
        });
    }
    let env = build_external_env(cb);
    let rm = crate::resolve::resolve_module_with_env(&module, &env)?;
    let new_rd = rm
        .get(name)
        .ok_or_else(|| EditError::ResolveError(format!("resolved module lost def `{}`", name)))?;
    let new_hash = new_rd.hash;
    let new_def = new_rd.def.clone();

    // Store the new def eagerly (idempotent, content-addressed, harmless).
    cb.store_def_at(&new_hash, &new_def)?;

    // Capture the edited def's surface local/param names into the side-car so
    // `view` shows readable names for the new hash too (outside the hash).
    if let Ok(names) = crate::resolve::local_names_for_module_with_env(&module, &env) {
        let _ = cb.store_local_names_batch(&names);
    }

    // No-op: identical source hashes to the same def.
    if new_hash == old_hash {
        let mut res = EditResult::default();
        res.no_op = true;
        return Ok(res);
    }

    // 3. Compare the TYPE of the new def vs the old.
    let mut work_cache = cb.types().clone();
    // Register stdlib externs so typechecking rewritten dependents
    // (which reference `ext/println` etc.) succeeds.
    if let Ok(std_m) = crate::parser::parse_module(crate::stdlib::SOURCE) {
        if let Ok(std_r) = crate::resolve::resolve_module(&std_m) {
            work_cache.register_externs(&std_r.externs);
        }
    }
    let new_scheme = typecheck_def(&new_def, &work_cache)?;
    let old_scheme = cb.types().get(&old_hash).cloned();
    let type_changed = match &old_scheme {
        Some(old) => !schemes_signature_eq(old, &new_scheme),
        // No cached old type for a stored def — treat as type-changed so we
        // never silently assume compatibility.
        None => true,
    };
    work_cache.insert(new_hash, new_scheme.clone());

    let index = load_or_rebuild_index(cb)?;

    if !propagate {
        // ---- Non-propagating: only the edited def's name moves. -------
        // Build a worklist of transitive dependents that still reference the
        // old hash. These are fine — they compile against the old def which
        // still exists. The caller can choose to propagate later.
        let cone = index.transitive_dependents(&old_hash);
        let mut todos: Vec<crate::typecheck::Todo> = Vec::new();
        if !cone.is_empty() {
            let mut sorted: Vec<Hash> = cone.into_iter().collect();
            sorted.sort_by_key(|h| h.to_hex());
            for h in sorted {
                let nm = name_for_hash(cb, &h);
                todos.push(crate::typecheck::Todo {
                    hash: h,
                    name: nm,
                    message: format!("still references old version of `{}`", name),
                });
            }
        }

        let updated = vec![Change {
            name: Some(name.to_owned()),
            old: old_hash,
            new: new_hash,
        }];

        let mut result = EditResult {
            renamed: Vec::new(),
            updated,
            propagated: Vec::new(),
            todos,
            no_op: false,
        };

        if dry_run {
            return Ok(result);
        }

        // COMMIT: only the edited def's name moves.
        cb.set_name(name.to_owned(), new_hash)?;
        cb.store_typecache(&work_cache)?;
        let mut idx = index;
        idx.add_def(new_hash, &new_def);
        let _ = idx.save(cb.root());
        result.updated[0].name = name_for_hash(cb, &new_hash);
        return Ok(result);
    }

    // ---- Propagating: require same-type, rewrite cone. ------------
    if type_changed {
        return Err(EditError::TypeError(format!(
            "cannot propagate a type-changing update to `{}`; \
             use `update` without `--propagate` to create the new def, \
             then update callers manually before calling `propagate`",
            name
        )));
    }

    // 4. Compute the cone in topological order; build old->new remap.
    let cone = index.transitive_dependents(&old_hash);
    let topo = topo_sort_cone(&index, &cone);
    let mut remap: HashMap<Hash, Hash> = HashMap::new();
    remap.insert(old_hash, new_hash);

    // 5. Rewrite each dependent in topo order, store, record remap.
    let mut propagated: Vec<Change> = Vec::new();
    for d in &topo {
        let old_def = cb.load_def(d)?;
        let rewritten = remap_def(&old_def, &remap);
        let new_d_hash = Hash::of_bytes(&encode_def(&rewritten));
        cb.store_def_at(&new_d_hash, &rewritten)?;
        remap.insert(*d, new_d_hash);
        propagated.push(Change {
            name: name_for_hash(cb, d),
            old: *d,
            new: new_d_hash,
        });
    }

    // 6. Typecheck the cone (only the rewritten dependents), in topo order.
    let changed_new: Vec<Hash> = topo.iter().map(|d| remap[d]).collect();
    let todos = {
        let inverse: HashMap<Hash, Hash> = remap.iter().map(|(o, n)| (*n, *o)).collect();
        typecheck_cone(cb, &changed_new, &mut work_cache, |h| {
            inverse.get(h).and_then(|old| name_for_hash(cb, old))
        })?
    };

    if !todos.is_empty() {
        // Same-type propagation should never produce type errors.
        let details: Vec<String> = todos.iter().map(|t| format!("{}: {}", t.name.as_deref().unwrap_or("?"), t.message)).collect();
        return Err(EditError::TypeError(format!(
            "propagation unexpectedly broke {} dependent(s): {}",
            todos.len(),
            details.join("; ")
        )));
    }

    let updated = vec![Change {
        name: Some(name.to_owned()),
        old: old_hash,
        new: new_hash,
    }];

    let mut result = EditResult {
        renamed: Vec::new(),
        updated,
        propagated,
        todos: Vec::new(),
        no_op: false,
    };

    if dry_run {
        return Ok(result);
    }

    // 7. COMMIT: move every name whose hash is a remap key to its new hash.
    let mut moves: Vec<(String, Hash)> = Vec::new();
    for (old_h, new_h) in &remap {
        for (nm, h) in cb.names() {
            if h == old_h {
                moves.push((nm.clone(), *new_h));
            }
        }
    }
    for (nm, new_h) in &moves {
        cb.set_name(nm.clone(), *new_h)?;
    }

    // Persist new type schemes.
    cb.store_typecache(&work_cache)?;

    // Update + save the dependency index for all new hashes.
    let mut idx = index;
    for new_h in remap.values() {
        if let Ok(def) = cb.load_def(new_h) {
            idx.add_def(*new_h, &def);
        }
    }
    let _ = idx.save(cb.root());

    // Recompute post-commit names on the result.
    result.updated[0].name = name_for_hash(cb, &new_hash);
    for ch in &mut result.propagated {
        ch.name = name_for_hash(cb, &ch.new);
    }

    Ok(result)
}

/// Propagate: (not yet implemented as a standalone command).
/// Use `update --propagate` to rewire the cone at edit time.
/// Standalone propagation requires patch tracking (old→new hash mapping)
/// which will be added in a follow-up.
fn propagate_impl(
    _cb: &mut Codebase,
    name: &str,
    _dry_run: bool,
) -> Result<EditResult, EditError> {
    Err(EditError::Unsupported(format!(
        "standalone `propagate` is not yet implemented. \
         Use `update {} --propagate` to rewire the cone at edit time, \
         or update callers manually with `update`.",
        name
    )))
}

// =============================================================================
// Phase 3 refactors: move, inline, reorder-params (change_signature), extract.
//
// Each is an exact, deterministic transform on the canonical (nameless,
// de Bruijn) AST. After producing the new canonical Def(s) for the directly
// edited definition, every one of these drives the SAME cone-propagation +
// transactional-commit pipeline as `update` (extracted into `commit_remap`),
// so dependents re-point automatically and we inherit the identical
// todos / immutability / index-update behavior.
//
// The hard part is de Bruijn correctness across binders; `shift_expr` and
// `subst_params` below are the capture-correct primitives, exercised directly
// by the tests.
// =============================================================================

/// Phase 3 errors. Kept separate from the variant list so the existing
/// Phase 1/2 `EditError` is untouched; these are added variants.
impl EditError {
    fn unsupported(msg: impl Into<String>) -> Self {
        EditError::Unsupported(msg.into())
    }
}
fn pattern_binders(p: &crate::ast::Pattern) -> u32 {
    use crate::ast::Pattern;
    match p {
        Pattern::Wildcard => 0,
        Pattern::Var => 1,
        Pattern::Enum { payload, .. } => payload.as_ref().map_or(0, |sub| pattern_binders(sub)),
    }
}

/// Shift every FREE `LocalVar(i)` in `e` (one with `i >= cutoff`) by `+delta`.
/// `cutoff` rises by the binders descended through, so locals bound inside `e`
/// are untouched. Standard de Bruijn lifting; used to move an inlinee's argument
/// under the extra binders it ends up beneath after substitution.
fn shift_expr(e: &Expr, delta: i64, cutoff: u32) -> Expr {
    match e {
        Expr::LocalVar(i) => {
            if *i >= cutoff {
                let shifted = *i as i64 + delta;
                debug_assert!(shifted >= 0, "edit: de Bruijn shift underflow");
                Expr::LocalVar(shifted as u32)
            } else {
                Expr::LocalVar(*i)
            }
        }
        Expr::Call(callee, args) => Expr::Call(
            Box::new(shift_expr(callee, delta, cutoff)),
            args.iter().map(|a| shift_expr(a, delta, cutoff)).collect(),
        ),
        Expr::Lambda { params, body } => Expr::Lambda {
            params: params.clone(),
            body: Box::new(shift_expr(body, delta, cutoff + params.len() as u32)),
        },
        Expr::Let { value, body } => Expr::Let {
            value: Box::new(shift_expr(value, delta, cutoff)),
            body: Box::new(shift_expr(body, delta, cutoff + 1)),
        },
        Expr::Defer { cleanup, body } => Expr::Defer {
            cleanup: Box::new(shift_expr(cleanup, delta, cutoff)),
            body: Box::new(shift_expr(body, delta, cutoff)),
        },
        Expr::StructNew { struct_ref, fields } => Expr::StructNew {
            struct_ref: *struct_ref,
            fields: fields.iter().map(|f| shift_expr(f, delta, cutoff)).collect(),
        },
        Expr::Field { base, struct_ref, index } => Expr::Field {
            base: Box::new(shift_expr(base, delta, cutoff)),
            struct_ref: *struct_ref,
            index: *index,
        },
        Expr::EnumNew { enum_ref, variant_index, payload } => Expr::EnumNew {
            enum_ref: *enum_ref,
            variant_index: *variant_index,
            payload: payload.as_ref().map(|p| Box::new(shift_expr(p, delta, cutoff))),
        },
        Expr::Match { scrutinee, arms } => Expr::Match {
            scrutinee: Box::new(shift_expr(scrutinee, delta, cutoff)),
            arms: arms
                .iter()
                .map(|arm| MatchArm {
                    pattern: arm.pattern.clone(),
                    body: shift_expr(&arm.body, delta, cutoff + pattern_binders(&arm.pattern)),
                })
                .collect(),
        },
        Expr::If { cond, then_branch, else_branch } => Expr::If {
            cond: Box::new(shift_expr(cond, delta, cutoff)),
            then_branch: Box::new(shift_expr(then_branch, delta, cutoff)),
            else_branch: Box::new(shift_expr(else_branch, delta, cutoff)),
        },
        Expr::Try { expr, enum_ref, ok_index, err_index } => Expr::Try {
            expr: Box::new(shift_expr(expr, delta, cutoff)),
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

/// Beta-reduce: produce the body of an `arity`-param top-level fn with each
/// parameter replaced by the corresponding argument, capture-correctly.
///
/// In the fn body (top level), source-order parameter `i` is `LocalVar(arity-1-i)`
/// (0 = innermost binder). Under `depth` further binders it appears as
/// `LocalVar(depth + (arity-1-i))`. We detect those, recover `i`, and splice in
/// `args[i]` lifted by `depth` (the argument is written in the call site's env, so
/// it must rise to clear the `depth` binders it now sits beneath). A
/// `LocalVar(j)` with `j < depth` is bound inside the body — left as-is. A
/// `LocalVar(j)` with `j-depth >= arity` would be free above the parameters,
/// impossible in a well-formed top-level fn body — hard error, never silent.
fn subst_params(body: &Expr, args: &[Expr], depth: u32) -> Result<Expr, EditError> {
    let arity = args.len() as u32;
    Ok(match body {
        Expr::LocalVar(i) => {
            if *i < depth {
                Expr::LocalVar(*i)
            } else {
                let from_top = *i - depth;
                if from_top >= arity {
                    return Err(EditError::unsupported(format!(
                        "inline: body references a free local above its parameters \
                         (LocalVar({}) at depth {}, arity {}); cannot beta-reduce safely",
                        i, depth, arity
                    )));
                }
                let param_i = (arity - 1 - from_top) as usize;
                shift_expr(&args[param_i], depth as i64, 0)
            }
        }
        Expr::Call(callee, cargs) => Expr::Call(
            Box::new(subst_params(callee, args, depth)?),
            cargs
                .iter()
                .map(|a| subst_params(a, args, depth))
                .collect::<Result<_, _>>()?,
        ),
        Expr::Lambda { params, body } => Expr::Lambda {
            params: params.clone(),
            body: Box::new(subst_params(body, args, depth + params.len() as u32)?),
        },
        Expr::Let { value, body } => Expr::Let {
            value: Box::new(subst_params(value, args, depth)?),
            body: Box::new(subst_params(body, args, depth + 1)?),
        },
        Expr::Defer { cleanup, body } => Expr::Defer {
            cleanup: Box::new(subst_params(cleanup, args, depth)?),
            body: Box::new(subst_params(body, args, depth)?),
        },
        Expr::StructNew { struct_ref, fields } => Expr::StructNew {
            struct_ref: *struct_ref,
            fields: fields
                .iter()
                .map(|f| subst_params(f, args, depth))
                .collect::<Result<_, _>>()?,
        },
        Expr::Field { base, struct_ref, index } => Expr::Field {
            base: Box::new(subst_params(base, args, depth)?),
            struct_ref: *struct_ref,
            index: *index,
        },
        Expr::EnumNew { enum_ref, variant_index, payload } => Expr::EnumNew {
            enum_ref: *enum_ref,
            variant_index: *variant_index,
            payload: match payload {
                Some(p) => Some(Box::new(subst_params(p, args, depth)?)),
                None => None,
            },
        },
        Expr::Match { scrutinee, arms } => Expr::Match {
            scrutinee: Box::new(subst_params(scrutinee, args, depth)?),
            arms: arms
                .iter()
                .map(|arm| {
                    Ok(MatchArm {
                        pattern: arm.pattern.clone(),
                        body: subst_params(&arm.body, args, depth + pattern_binders(&arm.pattern))?,
                    })
                })
                .collect::<Result<_, EditError>>()?,
        },
        Expr::If { cond, then_branch, else_branch } => Expr::If {
            cond: Box::new(subst_params(cond, args, depth)?),
            then_branch: Box::new(subst_params(then_branch, args, depth)?),
            else_branch: Box::new(subst_params(else_branch, args, depth)?),
        },
        Expr::Try { expr, enum_ref, ok_index, err_index } => Expr::Try {
            expr: Box::new(subst_params(expr, args, depth)?),
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
        | Expr::BuiltinRef(_) => body.clone(),
    })
}

/// Collect every FREE `LocalVar` of `e`, normalized to the environment OUTSIDE
/// `e` (the `cutoff` binders descended through are subtracted off). 0 = innermost
/// binder, so descending a binder raises `cutoff` by one (or by the pattern's
/// binder count). Output is ascending and deduplicated. Used by `extract` to find
/// the variables a lifted sub-expression captures.
fn free_locals(e: &Expr, cutoff: u32, out: &mut std::collections::BTreeSet<u32>) {
    match e {
        Expr::LocalVar(i) => {
            if *i >= cutoff {
                out.insert(*i - cutoff);
            }
        }
        Expr::Call(callee, args) => {
            free_locals(callee, cutoff, out);
            for a in args {
                free_locals(a, cutoff, out);
            }
        }
        Expr::Lambda { params, body } => free_locals(body, cutoff + params.len() as u32, out),
        Expr::Let { value, body } => {
            free_locals(value, cutoff, out);
            free_locals(body, cutoff + 1, out);
        }
        Expr::Defer { cleanup, body } => {
            free_locals(cleanup, cutoff, out);
            free_locals(body, cutoff, out);
        }
        Expr::StructNew { fields, .. } => {
            for f in fields {
                free_locals(f, cutoff, out);
            }
        }
        Expr::Field { base, .. } => free_locals(base, cutoff, out),
        Expr::EnumNew { payload, .. } => {
            if let Some(p) = payload {
                free_locals(p, cutoff, out);
            }
        }
        Expr::Match { scrutinee, arms } => {
            free_locals(scrutinee, cutoff, out);
            for arm in arms {
                free_locals(&arm.body, cutoff + pattern_binders(&arm.pattern), out);
            }
        }
        Expr::If { cond, then_branch, else_branch } => {
            free_locals(cond, cutoff, out);
            free_locals(then_branch, cutoff, out);
            free_locals(else_branch, cutoff, out);
        }
        Expr::Try { expr, .. } => free_locals(expr, cutoff, out),
        Expr::IntLit(_)
        | Expr::FloatLit(_)
        | Expr::BoolLit(_)
        | Expr::StringLit(_)
        | Expr::TopRef(_)
        | Expr::SelfRef(_)
        | Expr::StateRef(_)
        | Expr::StateSelfRef(_)
        | Expr::BuiltinRef(_) => {}
    }
}

/// Renumber the FREE `LocalVar`s of `e`: `map(free_index)` gives the new free
/// index, where the free index is measured relative to the environment OUTSIDE
/// the `cutoff` binders descended through. A `None` is a hard error — the caller
/// must supply an image for every free index that can occur. Locals bound inside
/// `e` (index `< cutoff` at that point) are left untouched. This is the
/// capture-correct renumbering used by `reorder_params` (permute params) and
/// `extract` (move captures into the helper's param slots).
fn remap_locals(
    e: &Expr,
    cutoff: u32,
    map: &dyn Fn(u32) -> Option<u32>,
) -> Result<Expr, EditError> {
    Ok(match e {
        Expr::LocalVar(i) => {
            if *i < cutoff {
                Expr::LocalVar(*i)
            } else {
                let free = *i - cutoff;
                match map(free) {
                    Some(new_free) => Expr::LocalVar(new_free + cutoff),
                    None => {
                        return Err(EditError::unsupported(format!(
                            "local renumbering has no image for free LocalVar({})",
                            free
                        )))
                    }
                }
            }
        }
        Expr::Call(callee, args) => Expr::Call(
            Box::new(remap_locals(callee, cutoff, map)?),
            args.iter()
                .map(|a| remap_locals(a, cutoff, map))
                .collect::<Result<_, _>>()?,
        ),
        Expr::Lambda { params, body } => Expr::Lambda {
            params: params.clone(),
            body: Box::new(remap_locals(body, cutoff + params.len() as u32, map)?),
        },
        Expr::Let { value, body } => Expr::Let {
            value: Box::new(remap_locals(value, cutoff, map)?),
            body: Box::new(remap_locals(body, cutoff + 1, map)?),
        },
        Expr::Defer { cleanup, body } => Expr::Defer {
            cleanup: Box::new(remap_locals(cleanup, cutoff, map)?),
            body: Box::new(remap_locals(body, cutoff, map)?),
        },
        Expr::StructNew { struct_ref, fields } => Expr::StructNew {
            struct_ref: *struct_ref,
            fields: fields
                .iter()
                .map(|f| remap_locals(f, cutoff, map))
                .collect::<Result<_, _>>()?,
        },
        Expr::Field { base, struct_ref, index } => Expr::Field {
            base: Box::new(remap_locals(base, cutoff, map)?),
            struct_ref: *struct_ref,
            index: *index,
        },
        Expr::EnumNew { enum_ref, variant_index, payload } => Expr::EnumNew {
            enum_ref: *enum_ref,
            variant_index: *variant_index,
            payload: match payload {
                Some(p) => Some(Box::new(remap_locals(p, cutoff, map)?)),
                None => None,
            },
        },
        Expr::Match { scrutinee, arms } => Expr::Match {
            scrutinee: Box::new(remap_locals(scrutinee, cutoff, map)?),
            arms: arms
                .iter()
                .map(|arm| {
                    Ok(MatchArm {
                        pattern: arm.pattern.clone(),
                        body: remap_locals(&arm.body, cutoff + pattern_binders(&arm.pattern), map)?,
                    })
                })
                .collect::<Result<_, EditError>>()?,
        },
        Expr::If { cond, then_branch, else_branch } => Expr::If {
            cond: Box::new(remap_locals(cond, cutoff, map)?),
            then_branch: Box::new(remap_locals(then_branch, cutoff, map)?),
            else_branch: Box::new(remap_locals(else_branch, cutoff, map)?),
        },
        Expr::Try { expr, enum_ref, ok_index, err_index } => Expr::Try {
            expr: Box::new(remap_locals(expr, cutoff, map)?),
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
    })
}

// -----------------------------------------------------------------------------
// Shared commit: drive Phase 2's cone propagation from an already-computed
// `edited_hash -> new_hash` change, plus an explicitly rewritten new def.
// -----------------------------------------------------------------------------

/// Given that the definition currently named `name` (hash `old_hash`) has been
/// transformed into `new_def` (already stored at `new_hash`), propagate that
/// change up the dependency cone exactly like [`update`] does: re-point every
/// dependent, typecheck the cone, return todos, and commit names + types +
/// index transactionally (unless `dry_run`).
///
/// This is the *same* machinery `update_impl` uses, factored out so all the
/// Phase 3 refactors inherit identical transactional / todos / immutability
/// behavior rather than reimplementing propagation.
fn commit_remap(
    cb: &mut Codebase,
    name: &str,
    old_hash: Hash,
    new_hash: Hash,
    new_def: &Def,
    dry_run: bool,
) -> Result<EditResult, EditError> {
    if new_hash == old_hash {
        let mut res = EditResult::default();
        res.no_op = true;
        return Ok(res);
    }

    // Type comparison (same-type fast path vs type-changed worklist).
    let mut work_cache = cb.types().clone();
    let new_scheme = typecheck_def(new_def, &work_cache)?;
    let old_scheme = cb.types().get(&old_hash).cloned();
    let type_changed = match &old_scheme {
        Some(old) => !schemes_signature_eq(old, &new_scheme),
        None => true,
    };
    work_cache.insert(new_hash, new_scheme.clone());

    let index = load_or_rebuild_index(cb)?;
    let cone = index.transitive_dependents(&old_hash);
    let topo = topo_sort_cone(&index, &cone);
    let mut remap: HashMap<Hash, Hash> = HashMap::new();
    remap.insert(old_hash, new_hash);

    let mut propagated: Vec<Change> = Vec::new();
    for d in &topo {
        let old_def = cb.load_def(d)?;
        let rewritten = remap_def(&old_def, &remap);
        let new_d_hash = Hash::of_bytes(&encode_def(&rewritten));
        cb.store_def_at(&new_d_hash, &rewritten)?;
        remap.insert(*d, new_d_hash);
        propagated.push(Change {
            name: name_for_hash(cb, d),
            old: *d,
            new: new_d_hash,
        });
    }

    let changed_new: Vec<Hash> = topo.iter().map(|d| remap[d]).collect();
    let todos = {
        let inverse: HashMap<Hash, Hash> = remap.iter().map(|(o, n)| (*n, *o)).collect();
        typecheck_cone(cb, &changed_new, &mut work_cache, |h| {
            inverse.get(h).and_then(|old| name_for_hash(cb, old))
        })?
    };

    if !type_changed && !todos.is_empty() {
        return Err(EditError::TypeError(format!(
            "same-type refactor unexpectedly broke {} dependent(s); kernel invariant violated",
            todos.len()
        )));
    }

    let updated = vec![Change {
        name: Some(name.to_owned()),
        old: old_hash,
        new: new_hash,
    }];

    let mut result = EditResult {
        renamed: Vec::new(),
        updated,
        propagated,
        todos,
        no_op: false,
    };

    if dry_run {
        return Ok(result);
    }

    // COMMIT.
    let mut moves: Vec<(String, Hash)> = Vec::new();
    for (old_h, new_h) in &remap {
        for (nm, h) in cb.names() {
            if h == old_h {
                moves.push((nm.clone(), *new_h));
            }
        }
    }
    for (nm, new_h) in &moves {
        cb.set_name(nm.clone(), *new_h)?;
    }
    cb.store_typecache(&work_cache)?;
    let mut idx = index;
    for new_h in remap.values() {
        if let Ok(def) = cb.load_def(new_h) {
            idx.add_def(*new_h, &def);
        }
    }
    let _ = idx.save(cb.root());

    result.updated[0].name = name_for_hash(cb, &new_hash);
    for ch in &mut result.propagated {
        ch.name = name_for_hash(cb, &ch.new);
    }
    Ok(result)
}

// -----------------------------------------------------------------------------
// move_def — namespace-path move (flat names this phase)
// -----------------------------------------------------------------------------

/// Move the definition named `from` to `to`.
///
/// Per the design (section 5.1), `move` is `rename` across the hierarchical
/// namespace path. The namespace is still the flat `names.txt` map in this
/// phase, so a dotted `to` (e.g. `math.float.sqrt`) is treated as an opaque
/// string key — exactly like `rename`. Hierarchical namespaces (a real tree,
/// with child namespaces and conflict surfacing) arrive in Phase 4; until then
/// `move_def` is a thin, documented wrapper over [`rename`] so callers already
/// have the verb and its hard-error contract:
///   - [`EditError::NameNotFound`] if `from` does not exist.
///   - [`EditError::NameExists`] if `to` already exists (never silent clobber).
pub fn move_def(cb: &mut Codebase, from: &str, to: &str) -> Result<EditResult, EditError> {
    rename(cb, from, to)
}

// -----------------------------------------------------------------------------
// inline — beta-reduce calls to a def into each of its callers
// -----------------------------------------------------------------------------

/// Inline the definition named `name` into every caller.
///
/// For each dependent that references `name`'s hash, every *call*
/// `Call(TopRef(name_hash), args)` is replaced by `name`'s body with the
/// arguments beta-reduced into the parameter positions, using capture-correct
/// de Bruijn substitution ([`subst_params`] + [`shift_expr`]). The arity of
/// each call must match the inlinee's parameter count.
///
/// Restrictions (each a HARD ERROR, never a silent skip):
///   - `name` must be a `Def::Fn`. Inlining a struct/enum is meaningless.
///   - `name`'s body must not reference its own hash (self-recursion): inlining
///     would not terminate. Refused with [`EditError::Unsupported`].
///   - Every reference to `name` in a caller must be in *call position*. A bare
///     `TopRef(name_hash)` used as a first-class value (passed as a closure,
///     stored, etc.) cannot be replaced by a body without eta-expansion, which
///     would change types/identity; refused with [`EditError::Unsupported`].
///
/// After rewriting all callers, the change is propagated up the cone via
/// [`commit_remap`] for each caller (their hashes update, dependents re-point).
///
/// `name`'s own def + name are left in place (the def is immutable on disk; we
/// do not remove the name, so existing tooling and any not-yet-inlined external
/// references keep resolving). Removing it is a separate, explicit `delete`.
pub fn inline(cb: &mut Codebase, name: &str) -> Result<EditResult, EditError> {
    inline_impl(cb, name, false)
}

/// Dry-run [`inline`]: compute the full impact without moving any names.
pub fn inline_dry_run(cb: &mut Codebase, name: &str) -> Result<EditResult, EditError> {
    inline_impl(cb, name, true)
}

fn inline_impl(cb: &mut Codebase, name: &str, dry_run: bool) -> Result<EditResult, EditError> {
    let target_hash = cb
        .get_name(name)
        .ok_or_else(|| EditError::NameNotFound(name.to_owned()))?;

    let target_def = cb.load_def(&target_hash)?;
    let (arity, body) = match &target_def {
        Def::Fn { params, body, .. } => (params.len(), body.clone()),
        _ => {
            return Err(EditError::unsupported(format!(
                "inline: `{}` is not a function; only `def` functions can be inlined",
                name
            )))
        }
    };

    // Refuse self-recursive inlinees (non-terminating).
    if crate::depindex::def_dependencies(&target_def).contains(&target_hash) {
        return Err(EditError::unsupported(format!(
            "inline: `{}` is self-recursive; inlining would not terminate",
            name
        )));
    }

    // Find all callers (direct dependents).
    let index = load_or_rebuild_index(cb)?;
    let callers = index.dependents(&target_hash);

    // Rewrite each caller's body, beta-reducing calls and refusing value-position
    // references. Each caller is committed independently through `commit_remap`,
    // which re-points that caller's own dependents up the cone.
    let mut combined = EditResult::default();
    let mut any = false;

    // Process deterministically.
    let mut caller_list: Vec<Hash> = callers.into_iter().collect();
    caller_list.sort_by_key(|h| h.to_hex());

    for caller in caller_list {
        // The caller hash may already have been re-pointed by a prior caller's
        // cone commit (e.g. if callers are themselves nested). Recover its
        // current hash via its name(s); if it has no name and isn't loadable,
        // skip-as-error.
        let current = current_hash_for(cb, caller);
        let caller_def = cb.load_def(&current)?;
        let caller_name = name_for_hash(cb, &current);

        let rewritten_body = inline_in_expr(def_body(&caller_def)?, &target_hash, arity, &body)?;
        let new_caller_def = with_body(&caller_def, rewritten_body);
        let new_hash = Hash::of_bytes(&encode_def(&new_caller_def));
        cb.store_def_at(&new_hash, &new_caller_def)?;

        // Drive the shared pipeline. A caller with no name still commits its
        // cone; we use a synthetic display name only for the changelog.
        let display = caller_name.clone().unwrap_or_else(|| format!("def_{}", &current.to_hex()[..8]));
        let res = commit_remap(cb, &display, current, new_hash, &new_caller_def, dry_run)?;
        if !res.no_op {
            any = true;
            combined.updated.extend(res.updated);
            combined.propagated.extend(res.propagated);
            combined.todos.extend(res.todos);
        }
    }

    if !any {
        combined.no_op = true;
    }
    Ok(combined)
}

/// The caller's body, or a hard error if the caller isn't a fn (a struct/enum
/// cannot "call" anything, so it cannot reference a fn in call position; this
/// only triggers if a type somehow depends on a fn hash, which is impossible —
/// kept as a loud guard).
fn def_body(def: &Def) -> Result<&Expr, EditError> {
    match def {
        Def::Fn { body, .. } => Ok(body),
        _ => Err(EditError::unsupported(
            "inline: a non-fn dependent references the inlinee; cannot rewrite".to_string(),
        )),
    }
}

/// Replace a fn def's body, preserving signature.
fn with_body(def: &Def, new_body: Expr) -> Def {
    match def {
        Def::Fn { is_local, type_params, params, ret, .. } => Def::Fn {
            is_local: *is_local,
            type_params: *type_params,
            params: params.clone(),
            ret: ret.clone(),
            body: new_body,
        },
        // def_body already guarantees Fn; unreachable, but keep total + loud.
        other => other.clone(),
    }
}

/// The current hash a (possibly-renamed-by-prior-commit) hash now resolves to.
/// If `h` is still a stored def with a name, that name's hash is authoritative;
/// otherwise `h` itself.
fn current_hash_for(cb: &Codebase, h: Hash) -> Hash {
    if let Some(nm) = name_for_hash(cb, &h) {
        if let Some(cur) = cb.get_name(&nm) {
            return cur;
        }
    }
    h
}

/// Rewrite `e`, beta-reducing every `Call(TopRef(target), args)` into the
/// inlinee's `body` and refusing any value-position `TopRef(target)`.
fn inline_in_expr(
    e: &Expr,
    target: &Hash,
    arity: usize,
    body: &Expr,
) -> Result<Expr, EditError> {
    match e {
        // Call in call position: the callee is exactly the inlinee.
        Expr::Call(callee, args) if matches!(&**callee, Expr::TopRef(h) if h == target) => {
            if args.len() != arity {
                return Err(EditError::unsupported(format!(
                    "inline: call to the inlinee has {} args but it takes {} params",
                    args.len(),
                    arity
                )));
            }
            // First inline inside the arguments (they may also call the inlinee),
            // then beta-reduce them into the body.
            let inlined_args: Vec<Expr> = args
                .iter()
                .map(|a| inline_in_expr(a, target, arity, body))
                .collect::<Result<_, _>>()?;
            subst_params(body, &inlined_args, 0)
        }
        // Any other use of the inlinee's hash as a value is unsupported.
        Expr::TopRef(h) if h == target => Err(EditError::unsupported(
            "inline: the def is referenced in value position (as a closure/value), \
             not called; cannot inline without eta-expansion".to_string(),
        )),
        Expr::Call(callee, args) => Ok(Expr::Call(
            Box::new(inline_in_expr(callee, target, arity, body)?),
            args.iter()
                .map(|a| inline_in_expr(a, target, arity, body))
                .collect::<Result<_, _>>()?,
        )),
        Expr::Lambda { params, body: b } => Ok(Expr::Lambda {
            params: params.clone(),
            body: Box::new(inline_in_expr(b, target, arity, body)?),
        }),
        Expr::Let { value, body: b } => Ok(Expr::Let {
            value: Box::new(inline_in_expr(value, target, arity, body)?),
            body: Box::new(inline_in_expr(b, target, arity, body)?),
        }),
        Expr::Defer { cleanup, body: b } => Ok(Expr::Defer {
            cleanup: Box::new(inline_in_expr(cleanup, target, arity, body)?),
            body: Box::new(inline_in_expr(b, target, arity, body)?),
        }),
        Expr::StructNew { struct_ref, fields } => Ok(Expr::StructNew {
            struct_ref: *struct_ref,
            fields: fields
                .iter()
                .map(|f| inline_in_expr(f, target, arity, body))
                .collect::<Result<_, _>>()?,
        }),
        Expr::Field { base, struct_ref, index } => Ok(Expr::Field {
            base: Box::new(inline_in_expr(base, target, arity, body)?),
            struct_ref: *struct_ref,
            index: *index,
        }),
        Expr::EnumNew { enum_ref, variant_index, payload } => Ok(Expr::EnumNew {
            enum_ref: *enum_ref,
            variant_index: *variant_index,
            payload: match payload {
                Some(p) => Some(Box::new(inline_in_expr(p, target, arity, body)?)),
                None => None,
            },
        }),
        Expr::Match { scrutinee, arms } => Ok(Expr::Match {
            scrutinee: Box::new(inline_in_expr(scrutinee, target, arity, body)?),
            arms: arms
                .iter()
                .map(|arm| {
                    Ok(MatchArm {
                        pattern: arm.pattern.clone(),
                        body: inline_in_expr(&arm.body, target, arity, body)?,
                    })
                })
                .collect::<Result<_, EditError>>()?,
        }),
        Expr::If { cond, then_branch, else_branch } => Ok(Expr::If {
            cond: Box::new(inline_in_expr(cond, target, arity, body)?),
            then_branch: Box::new(inline_in_expr(then_branch, target, arity, body)?),
            else_branch: Box::new(inline_in_expr(else_branch, target, arity, body)?),
        }),
        Expr::Try { expr, enum_ref, ok_index, err_index } => Ok(Expr::Try {
            expr: Box::new(inline_in_expr(expr, target, arity, body)?),
            enum_ref: *enum_ref,
            ok_index: *ok_index,
            err_index: *err_index,
        }),
        // A `TopRef` to some OTHER def (not the inlinee) is left untouched.
        Expr::IntLit(_)
        | Expr::FloatLit(_)
        | Expr::BoolLit(_)
        | Expr::StringLit(_)
        | Expr::LocalVar(_)
        | Expr::TopRef(_)
        | Expr::SelfRef(_)
        | Expr::StateRef(_)
        | Expr::StateSelfRef(_)
        | Expr::BuiltinRef(_) => Ok(e.clone()),
    }
}

// -----------------------------------------------------------------------------
// reorder_params — change_signature: permute a fn's parameters
// -----------------------------------------------------------------------------

/// Reorder the parameters of the fn named `name` according to `perm`, then
/// rewrite every call site in the dependency cone so behavior is preserved.
///
/// `perm` is a permutation of `0..arity`: the NEW parameter list is
/// `[old_params[perm[0]], old_params[perm[1]], ...]`. Equivalently, new
/// position `j` holds what used to be old position `perm[j]`.
///
/// What changes, exactly:
///   - The fn's `params` vec is permuted.
///   - Inside the body, every reference to old parameter `i` (de Bruijn
///     `LocalVar(arity-1-i)` at top level) is renumbered to its NEW position so
///     the body still reads the same values. (Param names aren't in the hash,
///     so this is the *only* body change; a pure rename would be a hash no-op.)
///   - Every `Call(TopRef(name_hash), args)` in the cone has its argument list
///     permuted the same way, so each call passes the same value to the same
///     (renumbered) parameter — the result is unchanged.
///
/// Hard errors (never silent):
///   - `name` is not a fn, or `perm` is not a permutation of `0..arity`.
///   - The fn is self-recursive: its own body contains `Call(SelfRef/TopRef
///     name, args)` whose args we would also have to permute. We DO handle the
///     self-call case by permuting those args too (they live in the same body),
///     so this is supported; documented here for clarity.
pub fn reorder_params(
    cb: &mut Codebase,
    name: &str,
    perm: &[usize],
) -> Result<EditResult, EditError> {
    reorder_params_impl(cb, name, perm, false)
}

/// Dry-run [`reorder_params`].
pub fn reorder_params_dry_run(
    cb: &mut Codebase,
    name: &str,
    perm: &[usize],
) -> Result<EditResult, EditError> {
    reorder_params_impl(cb, name, perm, true)
}

fn reorder_params_impl(
    cb: &mut Codebase,
    name: &str,
    perm: &[usize],
    dry_run: bool,
) -> Result<EditResult, EditError> {
    let old_hash = cb
        .get_name(name)
        .ok_or_else(|| EditError::NameNotFound(name.to_owned()))?;
    let def = cb.load_def(&old_hash)?;
    let (is_local, type_params, params, ret, body) = match &def {
        Def::Fn { is_local, type_params, params, ret, body } => {
            (*is_local, *type_params, params.clone(), ret.clone(), body.clone())
        }
        _ => {
            return Err(EditError::unsupported(format!(
                "reorder-params: `{}` is not a function",
                name
            )))
        }
    };
    let arity = params.len();
    validate_perm(perm, arity)?;

    // New params: new position j holds old position perm[j].
    let new_params: Vec<Type> = perm.iter().map(|&old_i| params[old_i].clone()).collect();

    // Body renumber. `new_pos[old_i]` is old param `old_i`'s new position.
    // de Bruijn: source-order param `i` is the FREE index `arity-1-i`; after
    // reordering it lives at new position `new_pos[i]`, i.e. free index
    // `arity-1-new_pos[i]`. `remap_locals` descends binders for us (cutoff),
    // so this map only ever sees free indices in `0..arity` (the params); a
    // free index `>= arity` would be malformed and is rejected (None).
    let mut new_pos = vec![0usize; arity];
    for (j, &old_i) in perm.iter().enumerate() {
        new_pos[old_i] = j;
    }
    let arity_u = arity as u32;
    let remap = move |free: u32| -> Option<u32> {
        if free < arity_u {
            let old_i = (arity_u - 1 - free) as usize;
            Some(arity_u - 1 - new_pos[old_i] as u32)
        } else {
            None
        }
    };
    let new_body = remap_locals(&body, 0, &remap)?;

    // If the fn calls itself (self-reference by its own hash in the body), the
    // self-call's args must also be permuted so the recursion stays correct.
    let new_body = permute_self_call_args(&new_body, &old_hash, perm)?;

    let new_def = Def::Fn {
        is_local,
        type_params,
        params: new_params,
        ret,
        body: new_body,
    };
    let new_hash = Hash::of_bytes(&encode_def(&new_def));
    cb.store_def_at(&new_hash, &new_def)?;

    // Propagate up the cone, but with a twist: a plain `remap_def` would only
    // re-point `TopRef(old)->TopRef(new)`, leaving call *argument order*
    // unchanged — which would silently break behavior. So we drive
    // `commit_remap` (for the same-type/typecheck/commit machinery) but first
    // rewrite each dependent's call sites to permute arguments. We do that by
    // pre-rewriting the cone here, then letting commit handle re-pointing.
    //
    // Implementation: commit_remap re-points hashes mechanically. To also
    // permute args we wrap: we rewrite the *whole cone* ourselves, storing
    // permuted+repointed defs, and pass dry_run through. Because permuting args
    // is part of the content change, we cannot reuse commit_remap's mechanical
    // remap_def for the cone. Instead we run a dedicated cone pass here.
    reorder_commit(cb, name, old_hash, new_hash, &new_def, perm, dry_run)
}

/// Like [`commit_remap`] but, for every dependent in the cone, ALSO permutes
/// the argument order of each `Call(TopRef(old_callee), args)` whose callee is
/// the def being reordered. This is what keeps call sites behavior-preserving.
fn reorder_commit(
    cb: &mut Codebase,
    name: &str,
    old_hash: Hash,
    new_hash: Hash,
    new_def: &Def,
    perm: &[usize],
    dry_run: bool,
) -> Result<EditResult, EditError> {
    if new_hash == old_hash {
        let mut res = EditResult::default();
        res.no_op = true;
        return Ok(res);
    }

    let mut work_cache = cb.types().clone();
    let new_scheme = typecheck_def(new_def, &work_cache)?;
    let old_scheme = cb.types().get(&old_hash).cloned();
    let type_changed = match &old_scheme {
        Some(old) => !schemes_signature_eq(old, &new_scheme),
        None => true,
    };
    work_cache.insert(new_hash, new_scheme.clone());

    let index = load_or_rebuild_index(cb)?;
    let cone = index.transitive_dependents(&old_hash);
    let topo = topo_sort_cone(&index, &cone);

    // The remap re-points ALL changed hashes (callee + transitively-changed
    // dependents). We build it incrementally as we rewrite each dependent.
    let mut remap: HashMap<Hash, Hash> = HashMap::new();
    remap.insert(old_hash, new_hash);

    let mut propagated: Vec<Change> = Vec::new();
    for d in &topo {
        let old_def = cb.load_def(d)?;
        // 1. Permute args of calls to the ORIGINAL callee (by its old hash).
        let arg_permuted = permute_calls_to(&old_def, &old_hash, perm)?;
        // 2. Mechanically re-point every changed hash (callee + earlier cone).
        let rewritten = remap_def(&arg_permuted, &remap);
        let new_d_hash = Hash::of_bytes(&encode_def(&rewritten));
        cb.store_def_at(&new_d_hash, &rewritten)?;
        remap.insert(*d, new_d_hash);
        propagated.push(Change {
            name: name_for_hash(cb, d),
            old: *d,
            new: new_d_hash,
        });
    }

    let changed_new: Vec<Hash> = topo.iter().map(|d| remap[d]).collect();
    let todos = {
        let inverse: HashMap<Hash, Hash> = remap.iter().map(|(o, n)| (*n, *o)).collect();
        typecheck_cone(cb, &changed_new, &mut work_cache, |h| {
            inverse.get(h).and_then(|old| name_for_hash(cb, old))
        })?
    };

    if !type_changed && !todos.is_empty() {
        return Err(EditError::TypeError(format!(
            "reorder-params unexpectedly broke {} dependent(s); kernel invariant violated",
            todos.len()
        )));
    }

    let updated = vec![Change {
        name: Some(name.to_owned()),
        old: old_hash,
        new: new_hash,
    }];
    let mut result = EditResult {
        renamed: Vec::new(),
        updated,
        propagated,
        todos,
        no_op: false,
    };
    if dry_run {
        return Ok(result);
    }

    let mut moves: Vec<(String, Hash)> = Vec::new();
    for (old_h, new_h) in &remap {
        for (nm, h) in cb.names() {
            if h == old_h {
                moves.push((nm.clone(), *new_h));
            }
        }
    }
    for (nm, new_h) in &moves {
        cb.set_name(nm.clone(), *new_h)?;
    }
    cb.store_typecache(&work_cache)?;
    let mut idx = index;
    for new_h in remap.values() {
        if let Ok(def) = cb.load_def(new_h) {
            idx.add_def(*new_h, &def);
        }
    }
    let _ = idx.save(cb.root());
    result.updated[0].name = name_for_hash(cb, &new_hash);
    for ch in &mut result.propagated {
        ch.name = name_for_hash(cb, &ch.new);
    }
    Ok(result)
}

/// `perm` must be a permutation of `0..arity`.
fn validate_perm(perm: &[usize], arity: usize) -> Result<(), EditError> {
    if perm.len() != arity {
        return Err(EditError::unsupported(format!(
            "reorder-params: permutation has {} entries but the fn has {} params",
            perm.len(),
            arity
        )));
    }
    let mut seen = vec![false; arity];
    for &p in perm {
        if p >= arity || seen[p] {
            return Err(EditError::unsupported(format!(
                "reorder-params: `{:?}` is not a permutation of 0..{}",
                perm, arity
            )));
        }
        seen[p] = true;
    }
    Ok(())
}

/// In a `Def`, permute the argument order of every `Call(TopRef(callee), args)`.
fn permute_calls_to(def: &Def, callee: &Hash, perm: &[usize]) -> Result<Def, EditError> {
    match def {
        Def::Fn { is_local, type_params, params, ret, body } => Ok(Def::Fn {
            is_local: *is_local,
            type_params: *type_params,
            params: params.clone(),
            ret: ret.clone(),
            body: permute_calls_to_expr(body, callee, perm)?,
        }),
        // Structs/enums hold no call sites.
        other => Ok(other.clone()),
    }
}

/// Permute args of self-calls (`Call(TopRef(self_hash), args)`) inside a body —
/// used during the reorder of the def itself.
fn permute_self_call_args(body: &Expr, self_hash: &Hash, perm: &[usize]) -> Result<Expr, EditError> {
    permute_calls_to_expr(body, self_hash, perm)
}

fn permute_calls_to_expr(e: &Expr, callee: &Hash, perm: &[usize]) -> Result<Expr, EditError> {
    match e {
        Expr::Call(c, args) if matches!(&**c, Expr::TopRef(h) if h == callee) => {
            if args.len() != perm.len() {
                return Err(EditError::unsupported(format!(
                    "reorder-params: a call to the target passes {} args but \
                     the permutation is for {} params",
                    args.len(),
                    perm.len()
                )));
            }
            // Recurse into args first, then permute their order.
            let inner: Vec<Expr> = args
                .iter()
                .map(|a| permute_calls_to_expr(a, callee, perm))
                .collect::<Result<_, _>>()?;
            let permuted: Vec<Expr> = perm.iter().map(|&old_i| inner[old_i].clone()).collect();
            Ok(Expr::Call(c.clone(), permuted))
        }
        Expr::Call(c, args) => Ok(Expr::Call(
            Box::new(permute_calls_to_expr(c, callee, perm)?),
            args.iter()
                .map(|a| permute_calls_to_expr(a, callee, perm))
                .collect::<Result<_, _>>()?,
        )),
        Expr::Lambda { params, body } => Ok(Expr::Lambda {
            params: params.clone(),
            body: Box::new(permute_calls_to_expr(body, callee, perm)?),
        }),
        Expr::Let { value, body } => Ok(Expr::Let {
            value: Box::new(permute_calls_to_expr(value, callee, perm)?),
            body: Box::new(permute_calls_to_expr(body, callee, perm)?),
        }),
        Expr::Defer { cleanup, body } => Ok(Expr::Defer {
            cleanup: Box::new(permute_calls_to_expr(cleanup, callee, perm)?),
            body: Box::new(permute_calls_to_expr(body, callee, perm)?),
        }),
        Expr::StructNew { struct_ref, fields } => Ok(Expr::StructNew {
            struct_ref: *struct_ref,
            fields: fields
                .iter()
                .map(|f| permute_calls_to_expr(f, callee, perm))
                .collect::<Result<_, _>>()?,
        }),
        Expr::Field { base, struct_ref, index } => Ok(Expr::Field {
            base: Box::new(permute_calls_to_expr(base, callee, perm)?),
            struct_ref: *struct_ref,
            index: *index,
        }),
        Expr::EnumNew { enum_ref, variant_index, payload } => Ok(Expr::EnumNew {
            enum_ref: *enum_ref,
            variant_index: *variant_index,
            payload: match payload {
                Some(p) => Some(Box::new(permute_calls_to_expr(p, callee, perm)?)),
                None => None,
            },
        }),
        Expr::Match { scrutinee, arms } => Ok(Expr::Match {
            scrutinee: Box::new(permute_calls_to_expr(scrutinee, callee, perm)?),
            arms: arms
                .iter()
                .map(|arm| {
                    Ok(MatchArm {
                        pattern: arm.pattern.clone(),
                        body: permute_calls_to_expr(&arm.body, callee, perm)?,
                    })
                })
                .collect::<Result<_, EditError>>()?,
        }),
        Expr::If { cond, then_branch, else_branch } => Ok(Expr::If {
            cond: Box::new(permute_calls_to_expr(cond, callee, perm)?),
            then_branch: Box::new(permute_calls_to_expr(then_branch, callee, perm)?),
            else_branch: Box::new(permute_calls_to_expr(else_branch, callee, perm)?),
        }),
        Expr::Try { expr, enum_ref, ok_index, err_index } => Ok(Expr::Try {
            expr: Box::new(permute_calls_to_expr(expr, callee, perm)?),
            enum_ref: *enum_ref,
            ok_index: *ok_index,
            err_index: *err_index,
        }),
        Expr::IntLit(_)
        | Expr::FloatLit(_)
        | Expr::BoolLit(_)
        | Expr::StringLit(_)
        | Expr::LocalVar(_)
        | Expr::TopRef(_)
        | Expr::SelfRef(_)
        | Expr::StateRef(_)
        | Expr::StateSelfRef(_)
        | Expr::BuiltinRef(_) => Ok(e.clone()),
    }
}

// -----------------------------------------------------------------------------
// extract — lift a sub-expression into a new top-level def + call it
// -----------------------------------------------------------------------------

/// Which sub-expression of a def's body to extract. A concrete, testable
/// selector over the canonical AST (no fragile surface spans).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractSelector {
    /// The body must be a top-level `let value in body`; extract `value`.
    LetValue,
    /// Extract the whole body.
    WholeBody,
}

impl ExtractSelector {
    /// Parse a CLI selector string. Hard error on anything else.
    pub fn parse(s: &str) -> Result<Self, EditError> {
        match s {
            "let-value" => Ok(ExtractSelector::LetValue),
            "body" => Ok(ExtractSelector::WholeBody),
            other => Err(EditError::unsupported(format!(
                "extract: unknown selector `{}` (expected `let-value` or `body`)",
                other
            ))),
        }
    }
}

/// Extract a sub-expression of `name`'s body into a new top-level fn `new_name`,
/// replacing the original site with a call to `new_name` passing the captured
/// locals as arguments. Then propagate `name`'s change up the cone.
///
/// de Bruijn handling: the selected sub-expression's FREE locals (variables
/// bound outside it, i.e. `name`'s parameters and any enclosing binders) become
/// the new fn's parameters, in ascending de Bruijn order. The extracted body is
/// renumbered so those captures land in the new fn's parameter slots; the call
/// at the original site passes each captured local (by its index in the
/// *original* environment) as the corresponding argument.
///
/// Restrictions (hard errors):
///   - `name` must be a fn.
///   - For `LetValue`, the body must be a top-level `Let`.
///   - The captured locals must all be *parameters* of `name` (free index <
///     param count) — i.e. the selected expression sits directly under the
///     fn's params, not under an intervening `let`/`lambda`/`match` binder.
///     Capturing an enclosing non-param binder would require materializing a
///     type for it that we cannot recover from the canonical AST alone, so it
///     is refused rather than guessed. (`let-value` always satisfies this,
///     since the value is evaluated in the fn's param environment.)
pub fn extract(
    cb: &mut Codebase,
    name: &str,
    selector: ExtractSelector,
    new_name: &str,
) -> Result<EditResult, EditError> {
    extract_impl(cb, name, selector, new_name, false)
}

/// Dry-run [`extract`].
pub fn extract_dry_run(
    cb: &mut Codebase,
    name: &str,
    selector: ExtractSelector,
    new_name: &str,
) -> Result<EditResult, EditError> {
    extract_impl(cb, name, selector, new_name, true)
}

fn extract_impl(
    cb: &mut Codebase,
    name: &str,
    selector: ExtractSelector,
    new_name: &str,
    dry_run: bool,
) -> Result<EditResult, EditError> {
    if cb.get_name(new_name).is_some() {
        return Err(EditError::NameExists(new_name.to_owned()));
    }
    let old_hash = cb
        .get_name(name)
        .ok_or_else(|| EditError::NameNotFound(name.to_owned()))?;
    let def = cb.load_def(&old_hash)?;
    let (is_local, type_params, params, ret, body) = match &def {
        Def::Fn { is_local, type_params, params, ret, body } => {
            (*is_local, *type_params, params.clone(), ret.clone(), body.clone())
        }
        _ => {
            return Err(EditError::unsupported(format!(
                "extract: `{}` is not a function",
                name
            )))
        }
    };
    let param_count = params.len() as u32;

    // Select the sub-expression and the residual builder.
    let (sub_expr, rebuild): (Expr, Box<dyn Fn(Expr) -> Expr>) = match selector {
        ExtractSelector::LetValue => match &body {
            Expr::Let { value, body: let_body } => {
                let lb = (**let_body).clone();
                (
                    (**value).clone(),
                    Box::new(move |call: Expr| Expr::Let {
                        value: Box::new(call),
                        body: Box::new(lb.clone()),
                    }),
                )
            }
            _ => {
                return Err(EditError::unsupported(format!(
                    "extract let-value: `{}`'s body is not a top-level `let`",
                    name
                )))
            }
        },
        ExtractSelector::WholeBody => (body.clone(), Box::new(|call: Expr| call)),
    };

    // Discover the captured locals of the selected expression. The selected
    // `sub_expr` is evaluated directly in `name`'s parameter environment (for
    // `let-value`, the let value sees exactly the params), so its FREE locals
    // (cutoff 0) are de Bruijn references to `name`'s parameters: indices in
    // `0..param_count`. `free_locals` returns them ascending and handles any
    // binders introduced inside `sub_expr` via the cutoff.
    let mut captures: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    free_locals(&sub_expr, 0, &mut captures);
    let captures: Vec<u32> = captures.into_iter().collect();

    // Each capture must be a parameter of `name` (de Bruijn index < param_count).
    // A capture `>= param_count` would be a free local above the params, which a
    // well-formed fn body cannot have — refuse loudly rather than guess a type.
    for &c in &captures {
        if c >= param_count {
            return Err(EditError::unsupported(format!(
                "extract: the selected expression captures a non-parameter local \
                 (free LocalVar({}) >= param count {}); cannot recover its type to \
                 build a parameter for the new def",
                c, param_count
            )));
        }
    }

    // The helper takes the captured params, ordered to match the call args built
    // below (ascending capture de Bruijn order). de Bruijn: capture `c` is
    // `name`'s source param `param_count-1-c`.
    let new_fn_params: Vec<Type> = captures
        .iter()
        .map(|&c| params[(param_count - 1 - c) as usize].clone())
        .collect();
    let new_arity = captures.len() as u32;

    // Renumber the extracted body into the helper's environment. The helper's
    // source-order param `k` receives `call_args[k]` (the k-th capture), and
    // source param `k` is de Bruijn `new_arity-1-k` in the helper body. So map
    // captured free index `captures[k]` -> `new_arity-1-k`. `remap_locals`
    // descends binders via the cutoff, so internal binders are never remapped.
    let cap_to_slot: HashMap<u32, u32> = captures
        .iter()
        .enumerate()
        .map(|(k, &c)| (c, new_arity - 1 - k as u32))
        .collect();
    let new_body = remap_locals(&sub_expr, 0, &|free: u32| cap_to_slot.get(&free).copied())?;

    // The new fn's return type: for `WholeBody` it's `name`'s ret; for
    // `LetValue` we cannot in general name the let-bound value's type without
    // running inference. We resolve it by typechecking the new def candidate
    // against the cache below; but the canonical Def needs a declared ret.
    // Strategy: build the new def with a provisional ret, typecheck the BODY's
    // inferred type via the typechecker, and use that. To stay within the
    // existing kernel (which has no standalone "infer expr type"), we require
    // the selector to make the type recoverable:
    //   - WholeBody: ret = name's ret.
    //   - LetValue: infer by typechecking; if the kernel can't, hard error.
    let new_ret: Type = match selector {
        ExtractSelector::WholeBody => ret.clone(),
        ExtractSelector::LetValue => infer_let_value_type(&def, &params, cb.types())?,
    };

    let new_def = Def::Fn {
        is_local,
        type_params,
        params: new_fn_params,
        ret: new_ret,
        body: new_body,
    };
    let new_def_hash = Hash::of_bytes(&encode_def(&new_def));
    cb.store_def_at(&new_def_hash, &new_def)?;

    // Build the call that replaces the original site: pass each captured local
    // (by its index in the ORIGINAL environment) in capture order. At the site,
    // the captured local `c` is `LocalVar(c)` (cutoff 0 = the fn's param env;
    // for LetValue the value is in that same env).
    let call_args: Vec<Expr> = captures.iter().map(|&c| Expr::LocalVar(c)).collect();
    let call = Expr::Call(Box::new(Expr::TopRef(new_def_hash)), call_args);
    let new_outer_body = rebuild(call);

    let new_outer_def = Def::Fn {
        is_local,
        type_params,
        params,
        ret,
        body: new_outer_body,
    };
    let new_outer_hash = Hash::of_bytes(&encode_def(&new_outer_def));
    cb.store_def_at(&new_outer_hash, &new_outer_def)?;

    // Typecheck the new helper and register its scheme into the codebase type
    // cache BEFORE propagation. `commit_remap` typechecks the cone, and the
    // rewritten outer def now calls `new_def` by hash — so the helper's scheme
    // must already be visible or that cone typecheck fails with an unknown
    // TopRef. The helper has no dependents and is content-addressed, so seeding
    // its scheme + index entry early is harmless (names are still the commit
    // point; we add the new NAME only in the non-dry-run branch below).
    let new_scheme = {
        let work = cb.types().clone();
        typecheck_def(&new_def, &work)?
    };
    {
        let mut cache: crate::typecheck::TypeCache = cb.types().clone();
        cache.insert(new_def_hash, new_scheme.clone());
        cb.store_typecache(&cache)?;
    }

    // Propagate the OUTER def's change up the cone via the shared pipeline.
    let mut result = commit_remap(cb, name, old_hash, new_outer_hash, &new_outer_def, dry_run)?;

    if !dry_run {
        // Name the new helper + record it in the index (scheme already cached).
        cb.set_name(new_name.to_owned(), new_def_hash)?;
        let mut idx = load_or_rebuild_index(cb)?;
        idx.add_def(new_def_hash, &new_def);
        let _ = idx.save(cb.root());
    }

    // Report the new def as an addition (old hash None -> new hash).
    result.updated.push(Change {
        name: Some(new_name.to_owned()),
        old: new_def_hash, // no prior hash; use new as both for "added" rows
        new: new_def_hash,
    });
    Ok(result)
}

/// Infer the type of a top-level `let value in body`'s VALUE, which becomes the
/// extracted helper's declared return type. The let value is evaluated in `name`'s
/// parameter environment, so we hand `infer_expr_type` exactly the fn's params as
/// the lexical environment (source order, matching `Def::Fn` param ordering) and
/// the codebase's real type cache (the value may reference other defs/types).
fn infer_let_value_type(
    def: &Def,
    params: &[Type],
    cache: &crate::typecheck::TypeCache,
) -> Result<Type, EditError> {
    let value = match def {
        Def::Fn { body: Expr::Let { value, .. }, .. } => (**value).clone(),
        _ => {
            return Err(EditError::unsupported(
                "extract let-value: body is not a top-level let".to_string(),
            ))
        }
    };
    crate::typecheck::infer_expr_type(&value, params, cache).map_err(|e| {
        EditError::TypeError(format!(
            "extract let-value: could not infer the let value's type: {}",
            e
        ))
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_module;
    use crate::resolve::resolve_module;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Unique temp dir under the OS temp dir (mirrors depindex's helper).
    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let pid = std::process::id();
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let dir = std::env::temp_dir().join(format!("ai_lang_edit_{}_{}_{}", pid, nanos, n));
            std::fs::create_dir_all(&dir).unwrap();
            TempDir(dir)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn build_codebase(tmp: &TempDir, source: &str) -> Codebase {
        let module = parse_module(source).expect("parse");
        let rm = resolve_module(&module).expect("resolve");
        let mut cb = Codebase::open(tmp.path()).expect("open");
        cb.store_resolved_module(&rm).expect("store");
        cb
    }

    fn hash_of(cb: &Codebase, name: &str) -> Hash {
        cb.get_name(name)
            .unwrap_or_else(|| panic!("no name {:?} in codebase", name))
    }

    // ---- rename: the headline property ----

    #[test]
    fn rename_is_namespace_only_and_does_not_break_callers() {
        let tmp = TempDir::new();
        let src = "
            def leaf(x: Int) -> Int = x
            def caller(y: Int) -> Int = leaf(y)
        ";
        let mut cb = build_codebase(&tmp, src);

        let leaf_hash = hash_of(&cb, "leaf");
        let caller_hash_before = hash_of(&cb, "caller");

        let res = rename(&mut cb, "leaf", "leaf2").expect("rename");
        assert_eq!(res.renamed, vec![("leaf".to_owned(), "leaf2".to_owned(), leaf_hash)]);

        // New name resolves to the OLD hash; old name is gone.
        assert_eq!(cb.get_name("leaf2"), Some(leaf_hash));
        assert_eq!(cb.get_name("leaf"), None);

        // The caller's hash is UNCHANGED — it references leaf by hash, not name.
        let caller_hash_after = hash_of(&cb, "caller");
        assert_eq!(
            caller_hash_after, caller_hash_before,
            "renaming a callee must not change the caller's hash"
        );

        // The caller still loads and still references leaf's (unchanged) hash.
        let index = DependencyIndex::rebuild_from_codebase(&cb).expect("rebuild");
        assert!(
            index.dependencies(&caller_hash_after).contains(&leaf_hash),
            "caller must still depend on the renamed callee's hash"
        );
    }

    #[test]
    fn rename_missing_name_errors() {
        let tmp = TempDir::new();
        let mut cb = build_codebase(&tmp, "def f(x: Int) -> Int = x");
        match rename(&mut cb, "nope", "whatever") {
            Err(EditError::NameNotFound(n)) => assert_eq!(n, "nope"),
            other => panic!("expected NameNotFound, got {:?}", other),
        }
    }

    #[test]
    fn rename_to_existing_name_errors() {
        let tmp = TempDir::new();
        let src = "
            def a(x: Int) -> Int = x
            def b(x: Int) -> Int = x
        ";
        let mut cb = build_codebase(&tmp, src);
        match rename(&mut cb, "a", "b") {
            Err(EditError::NameExists(n)) => assert_eq!(n, "b"),
            other => panic!("expected NameExists, got {:?}", other),
        }
        // Both names must be untouched after the refused rename.
        assert!(cb.get_name("a").is_some());
        assert!(cb.get_name("b").is_some());
    }

    // ---- find_usages ----

    #[test]
    fn find_usages_returns_the_caller_of_a_callee() {
        let tmp = TempDir::new();
        let src = "
            def leaf(x: Int) -> Int = x
            def caller(y: Int) -> Int = leaf(y)
        ";
        let cb = build_codebase(&tmp, src);
        let index = DependencyIndex::rebuild_from_codebase(&cb).expect("rebuild");

        let users = find_usages(&cb, &index, "leaf").expect("find_usages");
        assert_eq!(users.len(), 1);
        assert_eq!(users[0].name.as_deref(), Some("caller"));
        assert_eq!(users[0].hash, hash_of(&cb, "caller"));

        // Leaf-only def has no users.
        let none = find_usages(&cb, &index, "caller").expect("find_usages");
        assert!(none.is_empty());
    }

    #[test]
    fn find_usages_by_hash_prefix() {
        let tmp = TempDir::new();
        let src = "
            def leaf(x: Int) -> Int = x
            def caller(y: Int) -> Int = leaf(y)
        ";
        let cb = build_codebase(&tmp, src);
        let index = DependencyIndex::rebuild_from_codebase(&cb).expect("rebuild");

        let leaf = hash_of(&cb, "leaf");
        let prefix = &leaf.to_hex()[..12];
        let users = find_usages(&cb, &index, prefix).expect("find_usages by prefix");
        assert_eq!(users.len(), 1);
        assert_eq!(users[0].name.as_deref(), Some("caller"));
    }

    // ---- deps ----

    #[test]
    fn deps_forward_reverse_direct_and_transitive() {
        let tmp = TempDir::new();
        // c <- b <- a   (a calls b, b calls c)
        let src = "
            def c(x: Int) -> Int = x
            def b(x: Int) -> Int = c(x)
            def a(x: Int) -> Int = b(x)
        ";
        let cb = build_codebase(&tmp, src);
        let index = DependencyIndex::rebuild_from_codebase(&cb).expect("rebuild");

        let b = hash_of(&cb, "b");
        let c = hash_of(&cb, "c");
        let a = hash_of(&cb, "a");

        // Forward direct: a -> b.
        let fwd = deps(&cb, &index, "a", Direction::Forward, false).expect("deps");
        assert_eq!(fwd.iter().map(|d| d.hash).collect::<Vec<_>>(), vec![b]);

        // Forward transitive: a -> {b, c}.
        let fwd_t = deps(&cb, &index, "a", Direction::Forward, true).expect("deps");
        let fwd_hashes: Vec<Hash> = fwd_t.iter().map(|d| d.hash).collect();
        assert!(fwd_hashes.contains(&b) && fwd_hashes.contains(&c));
        assert_eq!(fwd_hashes.len(), 2);

        // Reverse direct: c <- b.
        let rev = deps(&cb, &index, "c", Direction::Reverse, false).expect("deps");
        assert_eq!(rev.iter().map(|d| d.hash).collect::<Vec<_>>(), vec![b]);

        // Reverse transitive (the cone): c <- {b, a}.
        let rev_t = deps(&cb, &index, "c", Direction::Reverse, true).expect("deps");
        let rev_hashes: Vec<Hash> = rev_t.iter().map(|d| d.hash).collect();
        assert!(rev_hashes.contains(&b) && rev_hashes.contains(&a));
        assert_eq!(rev_hashes.len(), 2);
    }

    #[test]
    fn deps_unknown_target_errors() {
        let tmp = TempDir::new();
        let cb = build_codebase(&tmp, "def f(x: Int) -> Int = x");
        let index = DependencyIndex::rebuild_from_codebase(&cb).expect("rebuild");
        // "zzz" is not a name and not valid hex -> BadTarget.
        match deps(&cb, &index, "zzz", Direction::Forward, false) {
            Err(EditError::BadTarget(_)) => {}
            other => panic!("expected BadTarget, got {:?}", other),
        }
        // A valid-hex prefix that matches nothing -> HashNotFound.
        match deps(&cb, &index, "deadbeef", Direction::Forward, false) {
            Err(EditError::HashNotFound(_)) => {}
            other => panic!("expected HashNotFound, got {:?}", other),
        }
    }

    #[test]
    fn load_or_rebuild_falls_back_when_no_cache() {
        let tmp = TempDir::new();
        let src = "
            def leaf(x: Int) -> Int = x
            def caller(y: Int) -> Int = leaf(y)
        ";
        let cb = build_codebase(&tmp, src);
        // No index/ dir was written, so load() fails and we rebuild.
        let index = load_or_rebuild_index(&cb).expect("load_or_rebuild");
        let leaf = hash_of(&cb, "leaf");
        let caller = hash_of(&cb, "caller");
        assert!(index.dependents(&leaf).contains(&caller));
    }

    // =========================================================================
    // update (Phase 2) tests
    // =========================================================================

    /// Build a codebase AND typecheck it (so the type cache is populated,
    /// which `update` relies on for the same-type vs type-changed decision and
    /// for seeding the external env).
    fn build_typed_codebase(tmp: &TempDir, source: &str) -> Codebase {
        use crate::typecheck::typecheck_module;
        let module = parse_module(source).expect("parse");
        let rm = resolve_module(&module).expect("resolve");
        let mut cb = Codebase::open(tmp.path()).expect("open");
        cb.store_resolved_module(&rm).expect("store");
        let mut cache = cb.types().clone();
        cache.register_externs(&rm.externs);
        typecheck_module(&rm, &mut cache).expect("typecheck");
        cb.store_typecache(&cache).expect("store types");
        cb
    }

    #[test]
    fn headline_update_repoints_dependent_to_new_callee() {
        let tmp = TempDir::new();
        let src = "
            def double(x: Int) -> Int = x * 2
            def quadruple(x: Int) -> Int = double(double(x))
        ";
        let mut cb = build_typed_codebase(&tmp, src);

        let old_double = hash_of(&cb, "double");
        let old_quad = hash_of(&cb, "quadruple");

        // Propagating update: same-type, rewrites the cone.
        let res = update_propagate(&mut cb, "double", "def double(x: Int) -> Int = x * 3")
            .expect("update_propagate");
        assert!(!res.no_op);
        assert!(res.todos.is_empty(), "same-type propagation must have no todos");

        // double now points at a NEW hash.
        let new_double = hash_of(&cb, "double");
        assert_ne!(new_double, old_double, "double must get a new hash");
        assert_eq!(res.updated.len(), 1);
        assert_eq!(res.updated[0].old, old_double);
        assert_eq!(res.updated[0].new, new_double);

        // quadruple was re-pointed: its NAME now points at a NEW hash.
        let new_quad = hash_of(&cb, "quadruple");
        assert_ne!(new_quad, old_quad, "quadruple must be re-pointed to a new hash");
        assert!(
            res.propagated.iter().any(|c| c.old == old_quad && c.new == new_quad),
            "propagated must record quadruple's old->new transition"
        );

        // The new quadruple's body references the NEW double's hash.
        let new_quad_def = cb.load_def(&new_quad).expect("load new quad");
        let deps = crate::depindex::def_dependencies(&new_quad_def);
        assert!(
            deps.contains(&new_double),
            "new quadruple must depend on the NEW double hash"
        );
        assert!(
            !deps.contains(&old_double),
            "new quadruple must NOT still depend on the OLD double hash"
        );

        // Immutability: the OLD .def files still exist on disk.
        assert!(cb.def_path(&old_double).exists(), "old double .def preserved");
        assert!(cb.def_path(&old_quad).exists(), "old quadruple .def preserved");
        assert!(cb.def_path(&new_double).exists());
        assert!(cb.def_path(&new_quad).exists());
    }

    #[test]
    fn same_type_update_has_zero_todos_and_repoints_whole_cone() {
        let tmp = TempDir::new();
        // a -> b -> c ; propagate c, both b and a re-point (cone depth >= 3).
        let src = "
            def c(x: Int) -> Int = x + 1
            def b(x: Int) -> Int = c(x)
            def a(x: Int) -> Int = b(x)
        ";
        let mut cb = build_typed_codebase(&tmp, src);
        let old_a = hash_of(&cb, "a");
        let old_b = hash_of(&cb, "b");
        let old_c = hash_of(&cb, "c");

        let res = update_propagate(&mut cb, "c", "def c(x: Int) -> Int = x + 2").expect("update");
        assert!(res.todos.is_empty(), "same-type cone must produce no todos");

        let new_a = hash_of(&cb, "a");
        let new_b = hash_of(&cb, "b");
        let new_c = hash_of(&cb, "c");
        assert_ne!(new_c, old_c);
        assert_ne!(new_b, old_b, "b must re-point");
        assert_ne!(new_a, old_a, "a must re-point transitively");

        // Wiring: new b -> new c, new a -> new b.
        let b_def = cb.load_def(&new_b).unwrap();
        assert!(crate::depindex::def_dependencies(&b_def).contains(&new_c));
        let a_def = cb.load_def(&new_a).unwrap();
        assert!(crate::depindex::def_dependencies(&a_def).contains(&new_b));
    }

    #[test]
    fn type_changed_update_yields_todos_and_moves_edited_name() {
        let tmp = TempDir::new();
        let src = "
            def base(x: Int) -> Int = x * 2
            def user(x: Int) -> Int = base(x) + 1
        ";
        let mut cb = build_typed_codebase(&tmp, src);
        let old_base = hash_of(&cb, "base");
        let old_user = hash_of(&cb, "user");

        // Change base's arity. Without --propagate, only base's name moves.
        let res = update(&mut cb, "base", "def base(x: Int, y: Int) -> Int = x * y")
            .expect("update");

        // The edited def's name MOVED to the new hash.
        let new_base = hash_of(&cb, "base");
        assert_ne!(new_base, old_base);
        assert_eq!(res.updated[0].old, old_base);
        assert_eq!(res.updated[0].new, new_base);

        // Dependents were NOT propagated.
        assert!(res.propagated.is_empty(), "no propagation without --propagate");

        // user is a dependent of old base -> listed as a todo (worklist).
        assert!(!res.todos.is_empty(), "type-changed update must yield todos");
        assert!(
            res.todos.iter().any(|t| t.name.as_deref() == Some("user")),
            "todos must name the dependent `user`, got {:?}",
            res.todos
        );

        // user's name still points at the OLD hash (not broken).
        let current_user = hash_of(&cb, "user");
        assert_eq!(current_user, old_user, "dependent's name does NOT move");

        // Old defs preserved (immutability).
        assert!(cb.def_path(&old_base).exists());
        assert!(cb.def_path(&old_user).exists());
    }

    #[test]
    fn noop_update_changes_nothing() {
        let tmp = TempDir::new();
        let src = "def f(x: Int) -> Int = x * 2";
        let mut cb = build_typed_codebase(&tmp, src);
        let before = hash_of(&cb, "f");

        let res = update(&mut cb, "f", "def f(x: Int) -> Int = x * 2").expect("update");
        assert!(res.no_op, "identical source must be a no-op");
        assert!(res.updated.is_empty());
        assert!(res.propagated.is_empty());
        assert!(res.todos.is_empty());
        assert_eq!(hash_of(&cb, "f"), before);
    }

    #[test]
    fn update_missing_name_errors() {
        let tmp = TempDir::new();
        let mut cb = build_typed_codebase(&tmp, "def f(x: Int) -> Int = x");
        match update(&mut cb, "nope", "def nope(x: Int) -> Int = x") {
            Err(EditError::NameNotFound(n)) => assert_eq!(n, "nope"),
            other => panic!("expected NameNotFound, got {:?}", other),
        }
    }

    #[test]
    fn update_name_mismatch_errors() {
        let tmp = TempDir::new();
        let mut cb = build_typed_codebase(&tmp, "def f(x: Int) -> Int = x");
        match update(&mut cb, "f", "def g(x: Int) -> Int = x") {
            Err(EditError::NameMismatch { expected, found }) => {
                assert_eq!(expected, "f");
                assert_eq!(found, "g");
            }
            other => panic!("expected NameMismatch, got {:?}", other),
        }
    }

    #[test]
    fn dry_run_computes_cone_without_moving_names() {
        let tmp = TempDir::new();
        let src = "
            def leaf(x: Int) -> Int = x * 2
            def caller(x: Int) -> Int = leaf(x)
        ";
        let mut cb = build_typed_codebase(&tmp, src);
        let old_leaf = hash_of(&cb, "leaf");
        let old_caller = hash_of(&cb, "caller");

        let res = update_dry_run(&mut cb, "leaf", "def leaf(x: Int) -> Int = x * 3")
            .expect("dry run");

        assert_eq!(res.updated[0].old, old_leaf);
        assert!(!res.todos.is_empty(), "dry run must report dependents as worklist");
        assert!(
            res.todos.iter().any(|t| t.hash == old_caller),
            "todos must include caller"
        );

        // NO names moved.
        assert_eq!(hash_of(&cb, "leaf"), old_leaf, "dry run must not move leaf");
        assert_eq!(hash_of(&cb, "caller"), old_caller, "dry run must not move caller");
    }

    // =========================================================================
    // Phase 3 refactor tests (move / inline / reorder-params / extract).
    //
    // Behavior preservation is proved by JIT-running a zero-arg entry def from
    // the codebase BEFORE and AFTER the refactor and asserting the same result.
    // =========================================================================

    /// JIT-run a zero-arg `() -> Int` named def from the codebase by
    /// reconstructing a ResolvedModule from its transitive hash closure
    /// (the same shape `ai-lang run` builds), compiling, and invoking.
    fn jit_run_named_int(cb: &Codebase, entry: &str) -> i64 {
        use crate::codegen::{CompiledModule, Jit, def_symbol, init_native_target};
        use crate::resolve::ResolvedModule;
        use crate::runtime::{Runtime, Thread};
        use inkwell::context::Context;
        use std::collections::BTreeSet;

        init_native_target().expect("init native target");
        let root = cb.get_name(entry).expect("entry name");

        // Transitive closure of the root over forward dependencies.
        let mut closure: BTreeSet<Hash> = BTreeSet::new();
        let mut stack = vec![root];
        while let Some(h) = stack.pop() {
            if !closure.insert(h) {
                continue;
            }
            let def = cb.load_def(&h).expect("load def in closure");
            for dep in crate::depindex::def_dependencies(&def) {
                if !closure.contains(&dep) {
                    stack.push(dep);
                }
            }
        }

        let mut name_for: HashMap<Hash, String> = HashMap::new();
        for (n, h) in cb.names() {
            name_for.entry(*h).or_insert_with(|| n.clone());
        }
        let mut defs: Vec<crate::resolve::ResolvedDef> = closure
            .iter()
            .map(|h| {
                let def = cb.load_def(h).expect("load");
                let name = name_for
                    .get(h)
                    .cloned()
                    .unwrap_or_else(|| format!("def_{}", &h.to_hex()[..8]));
                crate::resolve::ResolvedDef { name, hash: *h, def }
            })
            .collect();
        defs.sort_by(|a, b| a.hash.to_hex().cmp(&b.hash.to_hex()));

        let rm = ResolvedModule {
            defs,
            at_binding: None,
            externs: HashMap::new(),
        };
        let ctx = Context::create();
        let cm = CompiledModule::build(&ctx, &rm).expect("build module");
        let rt = Runtime::new_with_metadata(
            cm.closure_type_infos.clone(),
            cm.shape_registry.clone(),
            cm.shape_meta.clone(),
            cm.shape_by_type_id.clone(),
        );
        let jit = Jit::new(cm, &rt).expect("jit");
        let f = unsafe {
            jit.engine
                .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(&def_symbol(&root))
                .expect("get entry")
        };
        unsafe { f.call(rt.thread_ptr()) }
    }

    // ---- move ----

    #[test]
    fn move_is_namespace_only_callers_unchanged() {
        let tmp = TempDir::new();
        let src = "
            def leaf(x: Int) -> Int = x
            def caller(y: Int) -> Int = leaf(y)
        ";
        let mut cb = build_typed_codebase(&tmp, src);
        let leaf_hash = hash_of(&cb, "leaf");
        let caller_before = hash_of(&cb, "caller");

        let res = move_def(&mut cb, "leaf", "math.float.leaf").expect("move");
        assert_eq!(res.renamed.len(), 1);

        // New (dotted) name resolves to the OLD hash; old name gone.
        assert_eq!(cb.get_name("math.float.leaf"), Some(leaf_hash));
        assert_eq!(cb.get_name("leaf"), None);
        // Caller's hash unchanged (refs are by hash).
        assert_eq!(hash_of(&cb, "caller"), caller_before);
    }

    #[test]
    fn move_missing_source_and_existing_target_error() {
        let tmp = TempDir::new();
        let src = "
            def a(x: Int) -> Int = x
            def b(x: Int) -> Int = x
        ";
        let mut cb = build_typed_codebase(&tmp, src);
        match move_def(&mut cb, "nope", "x") {
            Err(EditError::NameNotFound(n)) => assert_eq!(n, "nope"),
            other => panic!("expected NameNotFound, got {:?}", other),
        }
        match move_def(&mut cb, "a", "b") {
            Err(EditError::NameExists(n)) => assert_eq!(n, "b"),
            other => panic!("expected NameExists, got {:?}", other),
        }
    }

    // ---- inline ----

    #[test]
    fn inline_beta_reduces_and_preserves_behavior() {
        let tmp = TempDir::new();
        let src = "
            def inc(x: Int) -> Int = x + 1
            def f(y: Int) -> Int = inc(y) + inc(y)
            def main() -> Int = f(3)
        ";
        let mut cb = build_typed_codebase(&tmp, src);
        let inc_hash = hash_of(&cb, "inc");
        let before = jit_run_named_int(&cb, "main");
        assert_eq!(before, 8, "f(3) = (3+1)+(3+1) = 8");

        let res = inline(&mut cb, "inc").expect("inline");
        assert!(!res.no_op, "inline must change f");

        // f's new body no longer references inc's hash.
        let new_f = hash_of(&cb, "f");
        let f_def = cb.load_def(&new_f).expect("load f");
        assert!(
            !crate::depindex::def_dependencies(&f_def).contains(&inc_hash),
            "inlined f must not reference inc's hash"
        );

        // Behavior preserved: main() still = 8.
        let after = jit_run_named_int(&cb, "main");
        assert_eq!(after, 8, "inlining inc must preserve f(3) = 8");

        // Old inc def still on disk (immutability); name still resolves.
        assert!(cb.def_path(&inc_hash).exists());
        assert!(cb.get_name("inc").is_some());
    }

    #[test]
    fn inline_two_param_de_bruijn_correct() {
        let tmp = TempDir::new();
        // sub(a,b) = a - b ; inlined into g where the call args are NOT in
        // declaration order, exercising de Bruijn substitution.
        let src = "
            def sub(a: Int, b: Int) -> Int = a - b
            def g(p: Int, q: Int) -> Int = sub(q, p)
            def main() -> Int = g(10, 3)
        ";
        let mut cb = build_typed_codebase(&tmp, src);
        let sub_hash = hash_of(&cb, "sub");
        // g(10,3) = sub(3,10) = 3 - 10 = -7
        assert_eq!(jit_run_named_int(&cb, "main"), -7);

        inline(&mut cb, "sub").expect("inline");
        let new_g = hash_of(&cb, "g");
        let g_def = cb.load_def(&new_g).expect("load g");
        assert!(!crate::depindex::def_dependencies(&g_def).contains(&sub_hash));

        // Behavior preserved: still -7 (proves args mapped to right params).
        assert_eq!(jit_run_named_int(&cb, "main"), -7);
    }

    #[test]
    fn inline_self_recursive_hard_errors() {
        let tmp = TempDir::new();
        let src = "
            def loopy(n: Int) -> Int = if n == 0 { 0 } else { loopy(n - 1) }
            def main() -> Int = loopy(0)
        ";
        let mut cb = build_typed_codebase(&tmp, src);
        match inline(&mut cb, "loopy") {
            Err(EditError::Unsupported(m)) => assert!(m.contains("self-recursive")),
            other => panic!("expected Unsupported(self-recursive), got {:?}", other),
        }
    }

    #[test]
    fn inline_value_position_hard_errors() {
        let tmp = TempDir::new();
        // `apply` takes a fn value; `use_it` passes `target` as a VALUE, not
        // a call. Inlining `target` cannot be done without eta-expansion.
        let src = "
            def target(x: Int) -> Int = x + 1
            def apply(f: fn(Int) -> Int, v: Int) -> Int = f(v)
            def use_it(v: Int) -> Int = apply(target, v)
        ";
        let mut cb = build_typed_codebase(&tmp, src);
        match inline(&mut cb, "target") {
            Err(EditError::Unsupported(m)) => assert!(m.contains("value position")),
            other => panic!("expected Unsupported(value position), got {:?}", other),
        }
    }

    // ---- reorder-params (change_signature) ----

    #[test]
    fn reorder_params_preserves_behavior_via_callsite_rewrite() {
        let tmp = TempDir::new();
        let src = "
            def sub(a: Int, b: Int) -> Int = a - b
            def g() -> Int = sub(10, 3)
        ";
        let mut cb = build_typed_codebase(&tmp, src);
        let old_sub = hash_of(&cb, "sub");
        let old_g = hash_of(&cb, "g");
        assert_eq!(jit_run_named_int(&cb, "g"), 7, "sub(10,3) = 7");

        // Reorder sub's params to (b, a): new position 0 = old param 1, etc.
        let res = reorder_params(&mut cb, "sub", &[1, 0]).expect("reorder");
        assert!(res.todos.is_empty(), "reorder is type-preserving here");

        // sub got a new hash; g was re-pointed.
        let new_sub = hash_of(&cb, "sub");
        let new_g = hash_of(&cb, "g");
        assert_ne!(new_sub, old_sub);
        assert_ne!(new_g, old_g, "caller must be re-pointed");

        // Behavior preserved: g() STILL = 7. This proves the call site was
        // rewritten sub(10,3) -> sub(3,10) AND the body re-indexed so the new
        // param 0 (was b) and param 1 (was a) still compute a - b.
        assert_eq!(jit_run_named_int(&cb, "g"), 7, "reorder must preserve g() = 7");

        // Old defs still on disk.
        assert!(cb.def_path(&old_sub).exists());
        assert!(cb.def_path(&old_g).exists());
    }

    #[test]
    fn reorder_params_self_recursive_preserved() {
        let tmp = TempDir::new();
        // A self-recursive fn whose self-call args must ALSO be permuted.
        // diff(a,b) recurses once swapping; verify behavior after reorder.
        let src = "
            def pick(a: Int, b: Int) -> Int = if a == 0 { b } else { pick(0, a) }
            def main() -> Int = pick(5, 9)
        ";
        let mut cb = build_typed_codebase(&tmp, src);
        // pick(5,9): a!=0 -> pick(0,5) -> a==0 -> b=5. So main = 5.
        assert_eq!(jit_run_named_int(&cb, "main"), 5);

        reorder_params(&mut cb, "pick", &[1, 0]).expect("reorder");
        // After swapping params (a,b)->(b,a), every call site (incl self) is
        // permuted, so behavior is identical: main still = 5.
        assert_eq!(jit_run_named_int(&cb, "main"), 5);
    }

    #[test]
    fn reorder_params_bad_permutation_hard_errors() {
        let tmp = TempDir::new();
        let src = "def sub(a: Int, b: Int) -> Int = a - b";
        let mut cb = build_typed_codebase(&tmp, src);
        match reorder_params(&mut cb, "sub", &[0, 0]) {
            Err(EditError::Unsupported(_)) => {}
            other => panic!("expected Unsupported, got {:?}", other),
        }
        match reorder_params(&mut cb, "sub", &[0]) {
            Err(EditError::Unsupported(_)) => {}
            other => panic!("expected Unsupported, got {:?}", other),
        }
    }

    // ---- extract ----

    #[test]
    fn extract_let_value_preserves_behavior() {
        let tmp = TempDir::new();
        // f's body is a top-level let; extract the value into a new def.
        let src = "
            def f(x: Int, y: Int) -> Int = { let s = x + y; s * 2 }
            def main() -> Int = f(3, 4)
        ";
        let mut cb = build_typed_codebase(&tmp, src);
        let before = jit_run_named_int(&cb, "main");
        assert_eq!(before, 14, "f(3,4) = (3+4)*2 = 14");
        let res = extract(&mut cb, "f", ExtractSelector::LetValue, "f_sum").expect("extract");
        assert!(res.updated.iter().any(|c| c.name.as_deref() == Some("f_sum")));

        // The new def exists and resolves.
        let sum_hash = cb.get_name("f_sum").expect("f_sum named");
        // f now calls f_sum.
        let new_f = hash_of(&cb, "f");
        let f_def = cb.load_def(&new_f).expect("load f");
        assert!(
            crate::depindex::def_dependencies(&f_def).contains(&sum_hash),
            "extracted f must call f_sum"
        );

        // Behavior preserved.
        assert_eq!(jit_run_named_int(&cb, "main"), 14, "extract must preserve f(3,4) = 14");
    }

    #[test]
    fn extract_existing_target_name_errors() {
        let tmp = TempDir::new();
        let src = "
            def f(x: Int) -> Int = { let s = x + 1; s }
            def taken(z: Int) -> Int = z
        ";
        let mut cb = build_typed_codebase(&tmp, src);
        match extract(&mut cb, "f", ExtractSelector::LetValue, "taken") {
            Err(EditError::NameExists(n)) => assert_eq!(n, "taken"),
            other => panic!("expected NameExists, got {:?}", other),
        }
    }

    #[test]
    fn extract_let_value_on_non_let_body_errors() {
        let tmp = TempDir::new();
        let src = "def f(x: Int) -> Int = x + 1";
        let mut cb = build_typed_codebase(&tmp, src);
        match extract(&mut cb, "f", ExtractSelector::LetValue, "g") {
            Err(EditError::Unsupported(m)) => assert!(m.contains("not a top-level `let`")),
            other => panic!("expected Unsupported, got {:?}", other),
        }
    }
}
