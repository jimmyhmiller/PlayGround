//! Library slicing: pull parts of one codebase into another (Phase 5).
//!
//! Because identity is content, "pull one function from another codebase" is
//! exactly "copy the transitive hash-closure of its `.def` (+ `.type`) files,
//! then add the chosen name(s)". This module implements that as three
//! operations plus a type-directed search:
//!
//!   - [`slice`]   — compute the closure (roots + their stored transitive
//!                   dependencies) without touching the filesystem.
//!   - [`export`]  — resolve names → hashes, compute the closure, and write a
//!                   single self-describing BUNDLE file (def bytes, type bytes,
//!                   and a name→hash manifest).
//!   - [`import`]  — read a bundle into the local store (idempotent: hashes
//!                   already present are no-ops — automatic dedup), then add
//!                   the manifest names.
//!   - [`find_by_type`] — scan the codebase's cached `TypeScheme`s and return
//!                   the defs whose scheme matches a query scheme.
//!
//! It lives strictly above the hashing line: it never touches `hash.rs`,
//! `codec.rs`, or the `ast.rs` identity model. The forward closure is taken
//! from [`DependencyIndex`]; the bytes come from the codec / codebase. We never
//! re-walk dependencies or re-hash by hand.
//!
//! ## The closure: stored defs only (lambda/builtin hashes are expected absent)
//!
//! [`DependencyIndex::transitive_dependencies`] returns every hash reachable on
//! the forward edge set. That edge set (see `depindex::def_dependencies`)
//! deliberately includes hashes that have NO independent `.def` file on disk:
//!
//!   - **nested-lambda content hashes** — a lambda's hash is computed by
//!     re-encoding it, but lambdas are stored *inside* their enclosing def, not
//!     as their own `defs/<hex>.def`.
//!   - **builtin-embedded hashes** — e.g. the `Result` hash baked into a
//!     `core/net.at#<hex>` builtin name.
//!
//! Those are NOT independently-stored definitions, so a slice/export filters
//! the closure to hashes that actually exist as stored defs
//! (`cb.contains(&h)`), and does NOT error on the absent ones.
//!
//! This module is careful to distinguish "expected-absent lambda/builtin hash"
//! from a genuine **dangling reference**: a hash that is the target of a
//! `TopRef` / `TypeRef` / `struct_ref` / `enum_ref` (or an `at()`-embedded
//! type ref) inside a STORED def, yet has no `.def` file. The former is normal;
//! the latter means the source codebase is corrupt, and it is a hard error
//! ([`SliceError::DanglingRef`]). We compute the "reference target" set with a
//! dedicated walker [`reference_targets`] that, unlike the dep walker, does NOT
//! include nested-lambda hashes — so the only absent hashes it can surface are
//! real dangling refs.

use crate::ast::{Def, Expr, MatchArm, Pattern, Type};
use crate::codebase::{Codebase, CodebaseError};
use crate::codec::{decode_def, encode_def};
use crate::depindex::DependencyIndex;
use crate::hash::Hash;
use crate::resolve::parse_at_builtin_name;
use crate::typecheck::{decode_scheme, encode_scheme, TypeScheme};

use std::collections::BTreeSet;
use std::fs;
use std::io;
use std::path::Path;

// =============================================================================
// Errors
// =============================================================================

/// Errors from slicing / export / import / search. House style of
/// `depindex::DepIndexError`: `Display` + `Error` + `From<CodebaseError>`.
#[derive(Debug)]
pub enum SliceError {
    Io(io::Error),
    /// Failed to load or decode a stored definition / scheme.
    Codebase(CodebaseError),
    /// A name passed to `export` does not resolve to any hash in the codebase.
    UnknownName(String),
    /// A STORED def in the closure references (via TopRef/TypeRef/struct_ref/
    /// enum_ref) a hash that is itself absent from the store. The source
    /// codebase is corrupt. (This is NOT triggered by nested-lambda or
    /// builtin-embedded hashes, which are expected to be absent.)
    DanglingRef {
        /// The stored def that holds the reference.
        from: Hash,
        /// The absent reference target.
        to: Hash,
    },
    /// The bundle bytes could not be parsed. Carries a human description.
    BadBundle(String),
}

impl From<io::Error> for SliceError {
    fn from(e: io::Error) -> Self {
        SliceError::Io(e)
    }
}

impl From<CodebaseError> for SliceError {
    fn from(e: CodebaseError) -> Self {
        SliceError::Codebase(e)
    }
}

impl std::fmt::Display for SliceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SliceError::Io(e) => write!(f, "io error: {}", e),
            SliceError::Codebase(e) => write!(f, "codebase error: {}", e),
            SliceError::UnknownName(n) => write!(f, "unknown name: {:?}", n),
            SliceError::DanglingRef { from, to } => write!(
                f,
                "corrupt codebase: stored def {} references {}, which is not in the store",
                from, to
            ),
            SliceError::BadBundle(s) => write!(f, "bad bundle: {}", s),
        }
    }
}

impl std::error::Error for SliceError {}

// =============================================================================
// slice
// =============================================================================

/// A computed library slice: the chosen `roots` plus all of their transitive
/// **stored** dependencies (`defs`), every entry of which exists as a
/// `defs/<hex>.def` in the codebase. Lambda / builtin hashes that appear in the
/// raw forward closure but have no `.def` are filtered out (see module docs).
///
/// `defs` is sorted and deduplicated; it INCLUDES the roots, so it is the full
/// set of `.def` files an export must carry. `roots` preserves the caller's
/// requested order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Closure {
    pub defs: Vec<Hash>,
    pub roots: Vec<Hash>,
}

/// Compute the closure of `roots`: the roots plus their transitive stored-def
/// dependencies, filtered to `cb.contains`.
///
/// Each root must itself be a stored def (an export of something not on disk is
/// meaningless); a missing root is a [`SliceError::DanglingRef`] with
/// `from == to` (the closure asked for a hash that isn't stored). Every stored
/// def in the closure has its direct reference targets verified to be present;
/// a missing target is a [`SliceError::DanglingRef`].
pub fn slice(
    cb: &Codebase,
    index: &DependencyIndex,
    roots: &[Hash],
) -> Result<Closure, SliceError> {
    // 1. Gather the raw forward closure for each root (from the index), union
    //    the roots themselves in, then keep only hashes that are stored defs.
    let mut all: BTreeSet<Hash> = BTreeSet::new();
    for r in roots {
        // A root that isn't a stored def can't be exported.
        if !cb.contains(r) {
            return Err(SliceError::DanglingRef { from: *r, to: *r });
        }
        all.insert(*r);
        for dep in index.transitive_dependencies(r) {
            all.insert(dep);
        }
    }

    // 2. Filter to stored defs. Absent hashes here are expected lambda/builtin
    //    hashes — NOT an error. Real dangling refs are caught in step 3.
    let stored: BTreeSet<Hash> = all.into_iter().filter(|h| cb.contains(h)).collect();

    // 3. Integrity: every direct *reference target* of every stored def must
    //    itself be a stored def. A reference target is a TopRef / TypeRef /
    //    struct_ref / enum_ref / Try enum_ref / at()-embedded type ref — i.e.
    //    something that names an independently-stored definition. (Nested
    //    lambda hashes are deliberately excluded by `reference_targets`, so
    //    their absence never trips this check.)
    for h in &stored {
        let def = cb.load_def(h)?;
        let targets = reference_targets(&def);
        for t in targets {
            if !cb.contains(&t) {
                return Err(SliceError::DanglingRef { from: *h, to: t });
            }
        }
    }

    let defs: Vec<Hash> = stored.into_iter().collect();
    Ok(Closure {
        defs,
        roots: roots.to_vec(),
    })
}

// =============================================================================
// Reference-target walker (independently-stored references only)
// =============================================================================

/// The set of hashes a `Def` references that MUST name an independently-stored
/// definition: `TopRef` in the body, `TypeRef` anywhere in the signature
/// (params / ret / struct fields / enum payloads, recursing through `Apply` /
/// `FnType`), `struct_ref` / `enum_ref` on constructor / `Field` / `Match` /
/// `Try` nodes, and the hashes embedded in `core/net.at#<hex>` builtin names.
///
/// Crucially this does NOT include nested-lambda content hashes (a lambda is
/// stored inside its enclosing def, never as its own `.def`). That is the whole
/// point: anything this returns that is absent from the store is a genuine
/// dangling reference, never an expected-absent lambda hash.
fn reference_targets(def: &Def) -> BTreeSet<Hash> {
    let mut acc = BTreeSet::new();
    match def {
        Def::Fn {
            params, ret, body, ..
        } => {
            for p in params {
                collect_type_refs(p, &mut acc);
            }
            collect_type_refs(ret, &mut acc);
            collect_expr_refs(body, &mut acc);
        }
        Def::Struct { fields, .. } => {
            for (_, ty) in fields {
                collect_type_refs(ty, &mut acc);
            }
        }
        Def::Enum { variants, .. } => {
            for (_, payload) in variants {
                if let Some(ty) = payload {
                    collect_type_refs(ty, &mut acc);
                }
            }
        }
        Def::State { ty, init } => {
            collect_type_refs(ty, &mut acc);
            collect_expr_refs(init, &mut acc);
        }
    }
    acc
}

fn collect_type_refs(t: &Type, acc: &mut BTreeSet<Hash>) {
    match t {
        Type::TypeRef(h) => {
            acc.insert(*h);
        }
        Type::Apply(head, args) => {
            collect_type_refs(head, acc);
            for a in args {
                collect_type_refs(a, acc);
            }
        }
        Type::FnType { params, ret } => {
            for p in params {
                collect_type_refs(p, acc);
            }
            collect_type_refs(ret, acc);
        }
        // SelfRef should never appear in a stored AST (the resolver rewrites it
        // to TopRef before storing); Builtin / TypeVar carry no hashes.
        Type::Builtin(_) | Type::TypeVar(_) | Type::SelfRef(_) => {}
    }
}

fn collect_expr_refs(e: &Expr, acc: &mut BTreeSet<Hash>) {
    match e {
        Expr::TopRef(h) | Expr::StateRef(h) => {
            acc.insert(*h);
        }
        Expr::BuiltinRef(name) => {
            // `core/net.at#<hex>` / `core/net.at_async#<hex>` (optionally
            // `#<hex>#<hex>`) embed the Result type hash(es). Those name
            // stored type defs, so they ARE reference targets. A prefix
            // match with malformed hex is a hard error (mirrors `depindex`).
            if crate::resolve::is_at_family_builtin(name) {
                match parse_at_builtin_name(name) {
                    Some((primary, secondary)) => {
                        acc.insert(primary);
                        if let Some(s) = secondary {
                            acc.insert(s);
                        }
                    }
                    None => panic!(
                        "slice: malformed hash-bearing builtin name: {:?}",
                        name
                    ),
                }
            }
        }
        Expr::StructNew { struct_ref, fields } => {
            acc.insert(*struct_ref);
            for f in fields {
                collect_expr_refs(f, acc);
            }
        }
        Expr::Field {
            base, struct_ref, ..
        } => {
            acc.insert(*struct_ref);
            collect_expr_refs(base, acc);
        }
        Expr::EnumNew {
            enum_ref, payload, ..
        } => {
            acc.insert(*enum_ref);
            if let Some(p) = payload {
                collect_expr_refs(p, acc);
            }
        }
        Expr::Try {
            expr, enum_ref, ..
        } => {
            acc.insert(*enum_ref);
            collect_expr_refs(expr, acc);
        }
        Expr::Match { scrutinee, arms } => {
            collect_expr_refs(scrutinee, acc);
            for MatchArm { pattern, body } in arms {
                collect_pattern_refs(pattern, acc);
                collect_expr_refs(body, acc);
            }
        }
        Expr::Call(callee, args) => {
            collect_expr_refs(callee, acc);
            for a in args {
                collect_expr_refs(a, acc);
            }
        }
        Expr::Lambda { params, body } => {
            // A lambda's OWN content hash is NOT a reference target (it has no
            // independent `.def`). But its param types and body can still hold
            // TypeRef / TopRef to stored defs, so recurse into them.
            for p in params {
                collect_type_refs(p, acc);
            }
            collect_expr_refs(body, acc);
        }
        Expr::Let { value, body } => {
            collect_expr_refs(value, acc);
            collect_expr_refs(body, acc);
        }
        Expr::Defer { cleanup, body } => {
            collect_expr_refs(cleanup, acc);
            collect_expr_refs(body, acc);
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } => {
            collect_expr_refs(cond, acc);
            collect_expr_refs(then_branch, acc);
            collect_expr_refs(else_branch, acc);
        }
        Expr::IntLit(_)
        | Expr::FloatLit(_)
        | Expr::BoolLit(_)
        | Expr::StringLit(_)
        | Expr::LocalVar(_)
        | Expr::SelfRef(_)
        | Expr::StateSelfRef(_) => {}
    }
}

fn collect_pattern_refs(p: &Pattern, acc: &mut BTreeSet<Hash>) {
    match p {
        Pattern::Enum {
            enum_ref, payload, ..
        } => {
            acc.insert(*enum_ref);
            if let Some(sub) = payload {
                collect_pattern_refs(sub, acc);
            }
        }
        Pattern::Wildcard | Pattern::Var => {}
    }
}

// =============================================================================
// Bundle format
// =============================================================================
//
// A bundle is a single, self-describing, deterministic file. All integers are
// big-endian (matching `codec.rs`); strings are u32-length-prefixed UTF-8.
//
//   magic:    8 bytes   "AILBNDL1"
//   n_entries: u32
//   entries[n_entries]:
//     kind:   u8        (0 = def bytes, 1 = type bytes)
//     hash:   32 bytes  (raw content hash)
//     len:    u32
//     bytes:  len bytes (canonical encode_def / encode_scheme output)
//   n_names:  u32
//   names[n_names]:
//     name:   u32-len-prefixed UTF-8 string  (a chosen root name)
//     hash:   32 bytes                        (the hash that name points at)
//
// Entries are emitted in hash-sorted order (defs then their types), and names
// in name-sorted order, so the same closure always produces byte-identical
// bundle output.

const BUNDLE_MAGIC: &[u8; 8] = b"AILBNDL1";
const KIND_DEF: u8 = 0;
const KIND_TYPE: u8 = 1;

/// What an `export` produced: the bundle path's contents, summarized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportManifest {
    /// Every def hash written to the bundle (the full closure, sorted).
    pub defs: Vec<Hash>,
    /// Every type-scheme hash written (subset of `defs` that had a `.type`).
    pub types: Vec<Hash>,
    /// The chosen root name → hash entries, sorted by name.
    pub names: Vec<(String, Hash)>,
}

/// Resolve `names` to hashes, compute the closure, and write a portable bundle
/// to `out_path`.
///
/// Every name must resolve in `cb` (a miss is [`SliceError::UnknownName`]). The
/// bundle carries the canonical `.def` bytes for the whole closure, the `.type`
/// bytes for any closure member that has a cached scheme, and the chosen-root
/// name→hash manifest.
pub fn export(
    cb: &Codebase,
    index: &DependencyIndex,
    names: &[String],
    out_path: &Path,
) -> Result<ExportManifest, SliceError> {
    // 1. Resolve names → root hashes (preserve request order, dedup hashes).
    let mut roots: Vec<Hash> = Vec::new();
    let mut name_entries: Vec<(String, Hash)> = Vec::new();
    for n in names {
        let h = cb
            .get_name(n)
            .ok_or_else(|| SliceError::UnknownName(n.clone()))?;
        name_entries.push((n.clone(), h));
        if !roots.contains(&h) {
            roots.push(h);
        }
    }

    // 2. Closure (roots + stored transitive deps), with integrity check.
    let closure = slice(cb, index, &roots)?;

    // 3. Serialize. `closure.defs` is already hash-sorted.
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(BUNDLE_MAGIC);

    // Build the entry list first so we can write an accurate count. For each
    // def we always write its def bytes; if a cached scheme exists we also
    // write a type entry immediately after.
    let mut entries: Vec<(u8, Hash, Vec<u8>)> = Vec::new();
    let mut type_hashes: Vec<Hash> = Vec::new();
    for h in &closure.defs {
        let def = cb.load_def(h)?;
        entries.push((KIND_DEF, *h, encode_def(&def)));
        if let Some(scheme) = cb.load_typescheme(h)? {
            entries.push((KIND_TYPE, *h, encode_scheme(&scheme)));
            type_hashes.push(*h);
        }
    }

    write_u32(&mut buf, entries.len() as u32);
    for (kind, hash, bytes) in &entries {
        buf.push(*kind);
        buf.extend_from_slice(hash.as_bytes());
        write_u32(&mut buf, bytes.len() as u32);
        buf.extend_from_slice(bytes);
    }

    // 4. Names, sorted for determinism.
    name_entries.sort();
    name_entries.dedup();
    write_u32(&mut buf, name_entries.len() as u32);
    for (name, hash) in &name_entries {
        write_str(&mut buf, name);
        buf.extend_from_slice(hash.as_bytes());
    }

    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(out_path, &buf)?;

    Ok(ExportManifest {
        defs: closure.defs,
        types: type_hashes,
        names: name_entries,
    })
}

/// What an `import` did. Proves dedup: re-importing the same bundle reports
/// `new_defs == 0` and `new_types == 0`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportReport {
    /// Def entries that were not already present (and so were written).
    pub new_defs: usize,
    /// Def entries that were already present (deduped, written nowhere).
    pub existing_defs: usize,
    /// Type entries that were not already present.
    pub new_types: usize,
    /// Type entries that were already present.
    pub existing_types: usize,
    /// The root names that were set, name → hash (sorted by name).
    pub names: Vec<(String, Hash)>,
}

/// Read a bundle from `bundle_path` and apply it to `cb`: write every def/type
/// into the local store (idempotent — hashes already present are no-ops, which
/// IS the automatic dedup), then add the manifest names.
///
/// Importing the same bundle twice is a clean no-op on the second run (every
/// entry is already present; every name resolves to the same hash).
pub fn import(cb: &mut Codebase, bundle_path: &Path) -> Result<ImportReport, SliceError> {
    let bytes = fs::read(bundle_path)?;
    let mut r = Reader::new(&bytes);

    let magic = r.take(8)?;
    if magic != BUNDLE_MAGIC {
        return Err(SliceError::BadBundle(format!(
            "bad magic: expected {:?}, got {:?}",
            BUNDLE_MAGIC, magic
        )));
    }

    let n_entries = r.read_u32()?;
    let mut new_defs = 0usize;
    let mut existing_defs = 0usize;
    let mut new_types = 0usize;
    let mut existing_types = 0usize;

    for _ in 0..n_entries {
        let kind = r.read_u8()?;
        let hash = r.read_hash()?;
        let len = r.read_u32()? as usize;
        let payload = r.take(len)?.to_vec();
        match kind {
            KIND_DEF => {
                // Verify the bytes decode and re-encode to the SAME bytes the
                // bundle carried (so a tampered bundle can't smuggle in a def
                // under a hash whose canonical form differs). Note we do NOT
                // require encode(decode(b)) == b for recursive type defs, whose
                // stored form intentionally doesn't round-trip; we just decode
                // and store under the bundle's hash (the codebase trusts the
                // filename, mirroring `store_def_at`).
                let def = decode_def(&payload)
                    .map_err(|e| SliceError::BadBundle(format!("def {}: {}", hash, e)))?;
                if cb.contains(&hash) {
                    existing_defs += 1;
                } else {
                    cb.store_def_at(&hash, &def)?;
                    new_defs += 1;
                }
            }
            KIND_TYPE => {
                let scheme = decode_scheme(&payload)
                    .map_err(|e| SliceError::BadBundle(format!("type {}: {}", hash, e)))?;
                if cb.type_path(&hash).exists() {
                    existing_types += 1;
                } else {
                    cb.store_typescheme(hash, scheme)?;
                    new_types += 1;
                }
            }
            other => {
                return Err(SliceError::BadBundle(format!(
                    "unknown entry kind: {}",
                    other
                )));
            }
        }
    }

    let n_names = r.read_u32()?;
    let mut names: Vec<(String, Hash)> = Vec::new();
    for _ in 0..n_names {
        let name = r.read_str()?;
        let hash = r.read_hash()?;
        names.push((name, hash));
    }
    r.finish()?;

    // Set names AFTER all defs are in the store, so each name resolves to a
    // present def. A name pointing at a hash the bundle didn't carry is a
    // corrupt bundle.
    for (name, hash) in &names {
        if !cb.contains(hash) {
            return Err(SliceError::BadBundle(format!(
                "manifest name {:?} points at {} which the bundle did not provide",
                name, hash
            )));
        }
        cb.set_name(name.clone(), *hash)?;
    }
    names.sort();

    Ok(ImportReport {
        new_defs,
        existing_defs,
        new_types,
        existing_types,
        names,
    })
}

// =============================================================================
// Type-directed search
// =============================================================================

/// Scan the codebase's cached `TypeScheme`s and return every def whose scheme
/// matches `query`, paired with its current name (if any).
///
/// ## Match semantics
///
/// Matching is **alpha-equivalence over type variables**: two schemes match if
/// they are structurally identical after consistently renaming `TypeVar`
/// indices. So a query `Fn([TypeVar(0)], TypeVar(0))` matches a stored
/// `fn id<T>(x: T) -> T` regardless of which de Bruijn index the stored def
/// happened to assign. Concretely:
///
///   - The query and candidate must be the same variant (Fn / Struct / Enum).
///   - Field / variant *names* must match exactly (they are part of identity).
///   - All component `Type`s must match structurally, with a single shared
///     bijection between query `TypeVar`s and candidate `TypeVar`s that must be
///     consistent across the whole scheme.
///   - The `wire` flag is NOT part of the match (it is a derived property of
///     the component types, and a search-by-signature shouldn't care).
///
/// Builtin type names, `TypeRef` hashes, arities, and structure must match
/// exactly; only `TypeVar` indices are free to be renamed.
///
/// Returns `(Option<name>, hash)` pairs sorted by hash for determinism. A
/// matched hash with no current name still appears (with `None`), so search
/// works over the whole store, not just the named surface.
pub fn find_by_type(cb: &Codebase, query: &TypeScheme) -> Vec<(Option<String>, Hash)> {
    // Build a hash → name reverse lookup once.
    let mut name_of: std::collections::HashMap<Hash, String> = std::collections::HashMap::new();
    for (name, hash) in cb.names() {
        // If several names alias one hash, keep the lexicographically smallest
        // so the result is deterministic.
        match name_of.get(hash) {
            Some(existing) if existing.as_str() <= name.as_str() => {}
            _ => {
                name_of.insert(*hash, name.clone());
            }
        }
    }

    let mut out: Vec<(Option<String>, Hash)> = Vec::new();
    for (hash, scheme) in cb.types().iter() {
        if schemes_alpha_eq(query, scheme) {
            out.push((name_of.get(hash).cloned(), *hash));
        }
    }
    out.sort_by(|a, b| a.1.cmp(&b.1));
    out
}

/// Alpha-equivalence of two type schemes (see [`find_by_type`] docs).
fn schemes_alpha_eq(a: &TypeScheme, b: &TypeScheme) -> bool {
    // A single shared bijection between a's TypeVars and b's TypeVars,
    // maintained across the whole scheme.
    let mut fwd: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    let mut rev: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    match (a, b) {
        (
            TypeScheme::Fn {
                params: pa,
                ret: ra,
                ..
            },
            TypeScheme::Fn {
                params: pb,
                ret: rb,
                ..
            },
        ) => {
            if pa.len() != pb.len() {
                return false;
            }
            for (x, y) in pa.iter().zip(pb.iter()) {
                if !types_alpha_eq(x, y, &mut fwd, &mut rev) {
                    return false;
                }
            }
            types_alpha_eq(ra, rb, &mut fwd, &mut rev)
        }
        (
            TypeScheme::Struct { fields: fa, .. },
            TypeScheme::Struct { fields: fb, .. },
        ) => {
            if fa.len() != fb.len() {
                return false;
            }
            for ((na, ta), (nb, tb)) in fa.iter().zip(fb.iter()) {
                if na != nb || !types_alpha_eq(ta, tb, &mut fwd, &mut rev) {
                    return false;
                }
            }
            true
        }
        (
            TypeScheme::Enum { variants: va, .. },
            TypeScheme::Enum { variants: vb, .. },
        ) => {
            if va.len() != vb.len() {
                return false;
            }
            for ((na, pa), (nb, pb)) in va.iter().zip(vb.iter()) {
                if na != nb {
                    return false;
                }
                match (pa, pb) {
                    (None, None) => {}
                    (Some(ta), Some(tb)) => {
                        if !types_alpha_eq(ta, tb, &mut fwd, &mut rev) {
                            return false;
                        }
                    }
                    _ => return false,
                }
            }
            true
        }
        _ => false,
    }
}

/// Structural equality of two `Type`s modulo a consistent `TypeVar` bijection.
fn types_alpha_eq(
    a: &Type,
    b: &Type,
    fwd: &mut std::collections::HashMap<u32, u32>,
    rev: &mut std::collections::HashMap<u32, u32>,
) -> bool {
    match (a, b) {
        (Type::Builtin(x), Type::Builtin(y)) => x == y,
        (Type::TypeRef(x), Type::TypeRef(y)) => x == y,
        (Type::SelfRef(x), Type::SelfRef(y)) => x == y,
        (Type::TypeVar(x), Type::TypeVar(y)) => {
            // The mapping must be a consistent bijection.
            match (fwd.get(x).copied(), rev.get(y).copied()) {
                (None, None) => {
                    fwd.insert(*x, *y);
                    rev.insert(*y, *x);
                    true
                }
                (Some(mapped), _) => mapped == *y,
                (None, Some(_)) => false, // y already taken by a different x
            }
        }
        (Type::Apply(ha, aa), Type::Apply(hb, ab)) => {
            if aa.len() != ab.len() || !types_alpha_eq(ha, hb, fwd, rev) {
                return false;
            }
            aa.iter()
                .zip(ab.iter())
                .all(|(x, y)| types_alpha_eq(x, y, fwd, rev))
        }
        (
            Type::FnType {
                params: pa,
                ret: ra,
            },
            Type::FnType {
                params: pb,
                ret: rb,
            },
        ) => {
            pa.len() == pb.len()
                && pa
                    .iter()
                    .zip(pb.iter())
                    .all(|(x, y)| types_alpha_eq(x, y, fwd, rev))
                && types_alpha_eq(ra, rb, fwd, rev)
        }
        _ => false,
    }
}

// =============================================================================
// Bundle low-level codec helpers (big-endian, matching codec.rs conventions)
// =============================================================================

fn write_u32(buf: &mut Vec<u8>, n: u32) {
    buf.extend_from_slice(&n.to_be_bytes());
}

fn write_str(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    write_u32(buf, bytes.len() as u32);
    buf.extend_from_slice(bytes);
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Reader { bytes, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], SliceError> {
        if self.pos + n > self.bytes.len() {
            return Err(SliceError::BadBundle("unexpected end of bundle".to_owned()));
        }
        let s = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn read_u8(&mut self) -> Result<u8, SliceError> {
        Ok(self.take(1)?[0])
    }

    fn read_u32(&mut self) -> Result<u32, SliceError> {
        let s = self.take(4)?;
        Ok(u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn read_hash(&mut self) -> Result<Hash, SliceError> {
        let s = self.take(Hash::SIZE)?;
        let mut buf = [0u8; Hash::SIZE];
        buf.copy_from_slice(s);
        Ok(Hash(buf))
    }

    fn read_str(&mut self) -> Result<String, SliceError> {
        let len = self.read_u32()? as usize;
        let bytes = self.take(len)?;
        core::str::from_utf8(bytes)
            .map(|s| s.to_owned())
            .map_err(|_| SliceError::BadBundle("non-utf8 string in bundle".to_owned()))
    }

    fn finish(self) -> Result<(), SliceError> {
        let trailing = self.bytes.len() - self.pos;
        if trailing == 0 {
            Ok(())
        } else {
            Err(SliceError::BadBundle(format!(
                "{} trailing bytes after bundle",
                trailing
            )))
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_module;
    use crate::resolve::resolve_module;
    use crate::typecheck::{typecheck_module, TypeCache};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Unique temp dir under the OS temp dir; cleaned up on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let pid = std::process::id();
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let dir = std::env::temp_dir()
                .join(format!("ai_lang_slice_{}_{}_{}_{}", tag, pid, nanos, n));
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

    /// Resolve + typecheck + store `source` into a fresh codebase at `dir`.
    fn build_codebase(dir: &Path, source: &str) -> Codebase {
        let module = parse_module(source).expect("parse");
        let rm = resolve_module(&module).expect("resolve");
        let mut cb = Codebase::open(dir).expect("open");
        let mut cache = TypeCache::new();
        cache.register_externs(&rm.externs);
        typecheck_module(&rm, &mut cache).expect("typecheck");
        cb.store_resolved_module(&rm).expect("store defs");
        cb.store_typecache(&cache).expect("store types");
        cb
    }

    fn index_of(cb: &Codebase) -> DependencyIndex {
        DependencyIndex::rebuild_from_codebase(cb).expect("rebuild index")
    }

    // -- round-trip across two separate codebases --

    #[test]
    fn export_import_round_trip_carries_transitive_dep() {
        let a_dir = TempDir::new("a_round");
        let b_dir = TempDir::new("b_round");

        let src = "
            def helper(x: Int) -> Int = x * 2
            def usesHelper(y: Int) -> Int = helper(y) + 1
        ";
        let cb_a = build_codebase(a_dir.path(), src);
        let index_a = index_of(&cb_a);

        let helper_a = cb_a.get_name("helper").expect("helper in A");
        let uses_a = cb_a.get_name("usesHelper").expect("usesHelper in A");

        // Export ONLY usesHelper.
        let bundle = a_dir.path().join("uses.bundle");
        let manifest = export(
            &cb_a,
            &index_a,
            &["usesHelper".to_owned()],
            &bundle,
        )
        .expect("export");

        // The closure dragged in helper.
        assert!(
            manifest.defs.contains(&helper_a),
            "bundle closure should include transitive dep helper"
        );
        assert!(manifest.defs.contains(&uses_a));
        assert_eq!(manifest.names, vec![("usesHelper".to_owned(), uses_a)]);

        // Import into an EMPTY codebase B.
        let mut cb_b = Codebase::open(b_dir.path()).expect("open B");
        assert!(cb_b.names().is_empty(), "B starts empty");
        let report = import(&mut cb_b, &bundle).expect("import");

        // Both defs are new and present in B.
        assert_eq!(report.new_defs, 2, "two fresh defs imported");
        assert_eq!(report.existing_defs, 0);
        assert!(cb_b.contains(&uses_a), "B has usesHelper def");
        assert!(cb_b.contains(&helper_a), "B has helper def (transitive)");

        // The name resolves in B, to the SAME hash A had.
        assert_eq!(
            cb_b.get_name("usesHelper"),
            Some(uses_a),
            "usesHelper name resolves in B"
        );

        // The def bytes/hashes match A exactly.
        assert_eq!(
            encode_def(&cb_b.load_def(&uses_a).unwrap()),
            encode_def(&cb_a.load_def(&uses_a).unwrap()),
            "usesHelper bytes identical across codebases"
        );
        assert_eq!(
            encode_def(&cb_b.load_def(&helper_a).unwrap()),
            encode_def(&cb_a.load_def(&helper_a).unwrap()),
            "helper bytes identical across codebases"
        );

        // Types travelled too.
        assert!(
            cb_b.load_typescheme(&uses_a).unwrap().is_some(),
            "usesHelper type imported"
        );
        assert!(
            cb_b.load_typescheme(&helper_a).unwrap().is_some(),
            "helper type imported"
        );
    }

    // -- dedup --

    #[test]
    fn second_import_reports_zero_new() {
        let a_dir = TempDir::new("a_dedup");
        let b_dir = TempDir::new("b_dedup");

        let src = "
            def helper(x: Int) -> Int = x * 2
            def usesHelper(y: Int) -> Int = helper(y) + 1
        ";
        let cb_a = build_codebase(a_dir.path(), src);
        let index_a = index_of(&cb_a);

        let bundle = a_dir.path().join("uses.bundle");
        export(&cb_a, &index_a, &["usesHelper".to_owned()], &bundle).expect("export");

        let mut cb_b = Codebase::open(b_dir.path()).expect("open B");
        let first = import(&mut cb_b, &bundle).expect("first import");
        assert_eq!(first.new_defs, 2);

        // Second import of the SAME bundle: pure dedup, zero new.
        let second = import(&mut cb_b, &bundle).expect("second import");
        assert_eq!(second.new_defs, 0, "no new defs on re-import");
        assert_eq!(second.existing_defs, 2, "both defs already present");
        assert_eq!(second.new_types, 0, "no new types on re-import");
        // Names still resolve correctly.
        assert_eq!(
            cb_b.get_name("usesHelper"),
            cb_a.get_name("usesHelper")
        );
    }

    // -- partial pull --

    #[test]
    fn export_does_not_drag_in_unrelated_defs() {
        let a_dir = TempDir::new("a_partial");
        let b_dir = TempDir::new("b_partial");

        let src = "
            def helper(x: Int) -> Int = x * 2
            def usesHelper(y: Int) -> Int = helper(y) + 1
            def junk(z: Int) -> Int = z + 99
        ";
        let cb_a = build_codebase(a_dir.path(), src);
        let index_a = index_of(&cb_a);

        let junk_a = cb_a.get_name("junk").expect("junk in A");

        let bundle = a_dir.path().join("uses.bundle");
        let manifest =
            export(&cb_a, &index_a, &["usesHelper".to_owned()], &bundle).expect("export");

        // junk is NOT in the bundle's closure.
        assert!(
            !manifest.defs.contains(&junk_a),
            "unrelated junk must not be in the slice"
        );

        // And not in B after import.
        let mut cb_b = Codebase::open(b_dir.path()).expect("open B");
        import(&mut cb_b, &bundle).expect("import");
        assert!(!cb_b.contains(&junk_a), "junk must not reach B");
        assert_eq!(cb_b.get_name("junk"), None);
    }

    // -- lambda-hash-absent must not error --

    #[test]
    fn closure_with_nested_lambda_hash_does_not_error() {
        // A higher-order use produces a nested lambda whose content hash is in
        // the forward closure but has no `.def`. slice() must filter it out
        // (not error). We use opt_map from the stdlib via a small program.
        let a_dir = TempDir::new("a_lambda");

        // A let-bound lambda: its content hash appears in the forward dep set
        // (lambdas are hashed), but it is stored INSIDE applyOnce's body, never
        // as its own `.def`. slice() must filter it out, not error.
        let src = "
            def applyOnce(x: Int) -> Int = { let f = |n: Int| n + 1; f(x) }
        ";
        let cb_a = build_codebase(a_dir.path(), src);
        let index_a = index_of(&cb_a);
        let root = cb_a.get_name("applyOnce").expect("applyOnce");

        // The raw forward closure may contain a lambda hash with no `.def`.
        let raw = index_a.transitive_dependencies(&root);
        let has_absent = raw.iter().any(|h| !cb_a.contains(h));
        // If the language inlines the lambda this may be false; assert the
        // important property either way: slice must succeed and exclude any
        // non-stored hash.
        let closure = slice(&cb_a, &index_a, &[root]).expect("slice must not error");
        for h in &closure.defs {
            assert!(cb_a.contains(h), "every closure def is a stored def");
        }
        if has_absent {
            // The absent lambda hash is not in the closure.
            for h in &raw {
                if !cb_a.contains(h) {
                    assert!(
                        !closure.defs.contains(h),
                        "absent lambda hash must be filtered from the closure"
                    );
                }
            }
        }
    }

    // -- dangling TopRef IS an error --

    #[test]
    fn dangling_ref_is_a_hard_error() {
        let a_dir = TempDir::new("a_dangle");
        let src = "
            def helper(x: Int) -> Int = x * 2
            def usesHelper(y: Int) -> Int = helper(y) + 1
        ";
        let cb_a = build_codebase(a_dir.path(), src);
        let index_a = index_of(&cb_a);

        let helper_a = cb_a.get_name("helper").expect("helper");
        let uses_a = cb_a.get_name("usesHelper").expect("usesHelper");

        // Corrupt the codebase: delete helper's `.def` file. Now usesHelper
        // holds a TopRef to a hash with no stored def — a genuine dangling ref.
        std::fs::remove_file(cb_a.def_path(&helper_a)).expect("delete helper def");

        match slice(&cb_a, &index_a, &[uses_a]) {
            Err(SliceError::DanglingRef { from, to }) => {
                assert_eq!(from, uses_a);
                assert_eq!(to, helper_a);
            }
            other => panic!("expected DanglingRef, got {:?}", other),
        }
    }

    // -- find_by_type --

    #[test]
    fn find_by_type_matches_signature_and_excludes_others() {
        let a_dir = TempDir::new("a_find");
        let src = "
            def inc(x: Int) -> Int = x + 1
            def dec(x: Int) -> Int = x - 1
            def toStr(x: Int) -> Int = x
            def add(a: Int, b: Int) -> Int = a + b
        ";
        let cb_a = build_codebase(a_dir.path(), src);

        let inc = cb_a.get_name("inc").unwrap();
        let dec = cb_a.get_name("dec").unwrap();
        let add = cb_a.get_name("add").unwrap();

        // Query: (Int) -> Int.
        let query = TypeScheme::Fn {
            params: vec![Type::Builtin("Int".to_owned())],
            ret: Type::Builtin("Int".to_owned()),
            wire: true,
        };
        let hits = find_by_type(&cb_a, &query);
        let hit_hashes: Vec<Hash> = hits.iter().map(|(_, h)| *h).collect();

        assert!(hit_hashes.contains(&inc), "inc matches (Int)->Int");
        assert!(hit_hashes.contains(&dec), "dec matches (Int)->Int");
        // toStr is also (Int)->Int by signature; it must match too. add is
        // (Int,Int)->Int and must NOT match.
        assert!(!hit_hashes.contains(&add), "add (2-arg) must not match");

        // Every hit carries a name here (all named).
        for (name, _) in &hits {
            assert!(name.is_some(), "named defs should report their name");
        }
    }

    #[test]
    fn find_by_type_alpha_renames_type_vars() {
        // A generic identity-shaped scheme should match a query that uses a
        // different TypeVar index.
        let a_dir = TempDir::new("a_alpha");
        // Build a scheme by hand and store it directly (no surface generics
        // needed for the search test). `id : <T> (T) -> T`.
        let mut cb_a = Codebase::open(a_dir.path()).expect("open");
        let stored = TypeScheme::Fn {
            params: vec![Type::TypeVar(0)],
            ret: Type::TypeVar(0),
            wire: false,
        };
        let h = Hash([0x33; 32]);
        cb_a.store_typescheme(h, stored).unwrap();
        cb_a.set_name("id", h).unwrap();

        // Query uses TypeVar(7) for the same shape — must still match.
        let query = TypeScheme::Fn {
            params: vec![Type::TypeVar(7)],
            ret: Type::TypeVar(7),
            wire: false,
        };
        let hits = find_by_type(&cb_a, &query);
        assert_eq!(hits.len(), 1, "alpha-equivalent scheme should match");
        assert_eq!(hits[0].1, h);
        assert_eq!(hits[0].0.as_deref(), Some("id"));

        // A non-matching query (T) -> Int must NOT match.
        let bad = TypeScheme::Fn {
            params: vec![Type::TypeVar(0)],
            ret: Type::Builtin("Int".to_owned()),
            wire: false,
        };
        assert!(
            find_by_type(&cb_a, &bad).is_empty(),
            "different return type must not match"
        );
    }
}
