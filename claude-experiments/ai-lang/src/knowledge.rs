//! Client-side knowledge base: hash → canonical bytes.
//!
//! When a client ships a closure to a server that doesn't have the code,
//! the server replies with `NeedCode(hashes)`. The client must look up
//! each hash and return the canonical bytes.
//!
//! ## What gets stored
//!
//! - Every `Def` in the `ResolvedModule`, encoded via `encode_def`,
//!   stored under `def.hash`.
//! - Every unique `Expr::Lambda { ... }` reachable from any def body,
//!   encoded via `encode_expr(&Expr::Lambda { ... })`, stored under the
//!   lambda's content hash.
//!
//! ## Dependency walk
//!
//! `collect_transitive_deps` performs a BFS over hashes. For each hash:
//! - Look up the bytes.
//! - Decode (as either `Def` or `Expr::Lambda`).
//! - Walk the structure, collecting `TopRef` hashes, `StructNew`
//!   `struct_ref`s, `EnumNew` `enum_ref`s, `Field` `struct_ref`s,
//!   pattern `enum_ref`s, and `Lambda` content hashes computed by
//!   re-encoding nested lambdas.
//!
//! The result is ordered "deps first, root last" so the receiver can
//! install in order: each item's `TopRef`s / type references resolve
//! against items installed earlier in the list.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::ast::{Def, Expr, Pattern, Type};
use crate::codec::{decode_def, decode_expr, encode_def, encode_expr};
use crate::hash::Hash;
use crate::net::ItemKind;
use crate::resolve::{ExternSig, ResolvedModule};

#[derive(Debug)]
pub enum KbError {
    MissingHash(Hash),
    DecodeFailed(String),
    /// The bytes stored under a hash don't re-hash to that hash. Indicates
    /// the KB was built incorrectly — this is a bug, not a network error.
    HashMismatch { expected: Hash, computed: Hash },
}

impl core::fmt::Display for KbError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            KbError::MissingHash(h) => write!(f, "hash {} not in knowledge base", h),
            KbError::DecodeFailed(msg) => write!(f, "decode failed: {}", msg),
            KbError::HashMismatch { expected, computed } => write!(
                f,
                "hash mismatch in knowledge base: expected {} but bytes hash to {}",
                expected, computed
            ),
        }
    }
}

impl std::error::Error for KbError {}

/// Maps a content hash to `(item_kind, canonical_bytes)`.
pub struct KnowledgeBase {
    items: HashMap<Hash, (ItemKind, Vec<u8>)>,
    /// Extern signatures by surface name (e.g. `curl_easy_init`).
    /// Externs aren't content-addressed; this lets the shipping side
    /// attach the "requires symbol X from library Y" requirement to any
    /// shipped code that calls an extern.
    externs: HashMap<String, ExternSig>,
}

impl KnowledgeBase {
    pub fn new() -> Self {
        KnowledgeBase {
            items: HashMap::new(),
            externs: HashMap::new(),
        }
    }

    /// The extern signature registered under `name`, if any.
    pub fn extern_sig(&self, name: &str) -> Option<&ExternSig> {
        self.externs.get(name)
    }

    /// Build a KB from a resolved module. Stores every def + every
    /// nested unique lambda under its resolver-computed hash.
    ///
    /// Note on hash consistency: for recursive-type defs (members of a
    /// type-SCC), the resolver hashed the CANONICAL form (with
    /// `Type::SelfRef(i)` placeholders) but the STORED form replaces
    /// those with `Type::TypeRef(real_hash)`. Re-encoding the stored
    /// form therefore produces different bytes than the resolver
    /// hashed. We store the stored form anyway: the server decodes it
    /// into a `Def` that is semantically equivalent (the TopRefs point
    /// to the right hashes in the dep batch) and installs under the
    /// provided key. End-to-end correctness is preserved; only "anyone
    /// can re-hash the bytes and get the key back" is sacrificed for
    /// types in a cycle.
    pub fn build(rm: &ResolvedModule) -> Self {
        let mut kb = KnowledgeBase::new();
        for rd in &rm.defs {
            let bytes = encode_def(&rd.def);
            kb.items.insert(rd.hash, (ItemKind::Def, bytes));

            if let Def::Fn { body, .. } = &rd.def {
                walk_collect_lambdas(body, &mut kb);
            }
        }
        // Remember every extern signature so shipped code that calls one
        // can carry the requirement to the receiving node.
        for (name, sig) in &rm.externs {
            kb.externs.insert(name.clone(), sig.clone());
        }
        kb
    }

    pub fn lookup(&self, hash: &Hash) -> Option<&(ItemKind, Vec<u8>)> {
        self.items.get(hash)
    }

    /// For the code items identified by `hashes`, collect the extern
    /// requirements (name + signature) their bodies reference. The
    /// shipping side attaches these as `ItemKind::Extern` items so the
    /// receiver can declare and resolve each symbol. A referenced extern
    /// with no registered signature is silently skipped (the receiver
    /// will then fail clearly at "extern not declared in module").
    pub fn extern_requirements(&self, hashes: &[Hash]) -> Vec<(String, ExternSig)> {
        let mut names: HashSet<String> = HashSet::new();
        for h in hashes {
            let Some((kind, bytes)) = self.items.get(h) else {
                continue;
            };
            match kind {
                ItemKind::Def => {
                    if let Ok(Def::Fn { body, .. }) = decode_def(bytes) {
                        collect_ext_names(&body, &mut names);
                    }
                }
                ItemKind::Lambda => {
                    if let Ok(Expr::Lambda { body, .. }) = decode_expr(bytes) {
                        collect_ext_names(&body, &mut names);
                    }
                }
                ItemKind::Extern => {}
            }
        }
        let mut out: Vec<(String, ExternSig)> = Vec::new();
        for name in names {
            if let Some(sig) = self.externs.get(&name) {
                out.push((name, sig.clone()));
            }
        }
        out
    }

    /// Iterate all (hash, kind, bytes) tuples. Used by demos / tests
    /// that want to inspect the KB contents.
    pub fn iter(&self) -> impl Iterator<Item = (&Hash, &ItemKind, &Vec<u8>)> {
        self.items.iter().map(|(h, (k, b))| (h, k, b))
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Insert an item directly. Useful for tests; production code should
    /// build via [`KnowledgeBase::build`].
    pub fn insert(&mut self, hash: Hash, kind: ItemKind, bytes: Vec<u8>) {
        self.items.insert(hash, (kind, bytes));
    }

    /// DFS-walk the dependency graph rooted at `root_hashes`. Returns
    /// hashes in post-order: leaf deps first, root last, deduped.
    ///
    /// Cycles are valid (recursive type defs like `enum List<T> { Cons(ListCell<T>), Nil }`
    /// and mutually recursive fn SCCs). When DFS hits a hash already
    /// on the active stack, we treat it as "will be emitted by its
    /// own frame" and don't recurse in. Codegen's pre-declaration
    /// phase handles in-SCC references regardless of emit order.
    ///
    /// Errors only if a reachable hash isn't in the KB.
    pub fn collect_transitive_deps(
        &self,
        root_hashes: &[Hash],
    ) -> Result<Vec<Hash>, KbError> {
        let mut order: Vec<Hash> = Vec::new();
        let mut visited: HashSet<Hash> = HashSet::new();
        let mut on_stack: HashSet<Hash> = HashSet::new();
        for &root in root_hashes {
            if visited.contains(&root) {
                continue;
            }
            let mut work: VecDeque<(Hash, Vec<Hash>, usize)> = VecDeque::new();
            let root_deps = self.deps_of(&root)?;
            work.push_back((root, root_deps, 0));
            on_stack.insert(root);
            while let Some((h, deps, idx)) = work.pop_back() {
                if idx < deps.len() {
                    let next = deps[idx];
                    work.push_back((h, deps.clone(), idx + 1));
                    if visited.contains(&next) || on_stack.contains(&next) {
                        // Either already emitted, or pending on the
                        // stack as part of an SCC — skip either way.
                        continue;
                    }
                    on_stack.insert(next);
                    let next_deps = self.deps_of(&next)?;
                    work.push_back((next, next_deps, 0));
                } else {
                    on_stack.remove(&h);
                    if visited.insert(h) {
                        order.push(h);
                    }
                }
            }
        }
        Ok(order)
    }

    /// Compute the direct dependency hashes of the item stored under
    /// `hash`. Returns an error if `hash` is absent.
    fn deps_of(&self, hash: &Hash) -> Result<Vec<Hash>, KbError> {
        let (kind, bytes) = self
            .items
            .get(hash)
            .ok_or(KbError::MissingHash(*hash))?;
        let mut out: Vec<Hash> = Vec::new();
        let mut seen: HashSet<Hash> = HashSet::new();
        match kind {
            ItemKind::Def => {
                let d = decode_def(bytes).map_err(|e| KbError::DecodeFailed(format!("{}", e)))?;
                walk_def_deps(&d, &mut out, &mut seen);
            }
            ItemKind::Lambda => {
                let e = decode_expr(bytes).map_err(|e| KbError::DecodeFailed(format!("{}", e)))?;
                // The bytes encode an Expr::Lambda { params, body }.
                match e {
                    Expr::Lambda { params, body } => {
                        for t in &params {
                            walk_type_deps(t, &mut out, &mut seen);
                        }
                        walk_expr_deps(&body, &mut out, &mut seen);
                    }
                    other => {
                        return Err(KbError::DecodeFailed(format!(
                            "KB entry tagged Lambda but decoded as {:?}",
                            other
                        )));
                    }
                }
            }
            // Externs live in `self.externs`, never in `self.items`, so a
            // hash keyed to one is never passed here. C-FFI extern
            // signatures use only builtin scalar types (no hash deps)
            // regardless.
            ItemKind::Extern => {}
        }
        Ok(out)
    }
}

impl Default for KnowledgeBase {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------
// Lambda collection (during KB::build)
// ---------------------------------------------------------------------

fn walk_collect_lambdas(e: &Expr, kb: &mut KnowledgeBase) {
    match e {
        Expr::Lambda { params, body } => {
            let lambda_expr = Expr::Lambda {
                params: params.clone(),
                body: body.clone(),
            };
            let bytes = encode_expr(&lambda_expr);
            let h = Hash::of_bytes(&bytes);
            kb.items
                .entry(h)
                .or_insert_with(|| (ItemKind::Lambda, bytes));
            // Walk the body for any nested lambdas — v1 codegen rejects
            // these, but the KB should still collect them in case a
            // future expansion allows them, and to avoid a footgun.
            walk_collect_lambdas(body, kb);
        }
        Expr::Call(callee, args) => {
            walk_collect_lambdas(callee, kb);
            for a in args {
                walk_collect_lambdas(a, kb);
            }
        }
        Expr::Let { value, body } => {
            walk_collect_lambdas(value, kb);
            walk_collect_lambdas(body, kb);
        }
        Expr::StructNew { fields, .. } => {
            for f in fields {
                walk_collect_lambdas(f, kb);
            }
        }
        Expr::Field { base, .. } => {
            walk_collect_lambdas(base, kb);
        }
        Expr::EnumNew { payload, .. } => {
            if let Some(p) = payload {
                walk_collect_lambdas(p, kb);
            }
        }
        Expr::Match { scrutinee, arms } => {
            walk_collect_lambdas(scrutinee, kb);
            for arm in arms {
                walk_collect_lambdas(&arm.body, kb);
            }
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } => {
            walk_collect_lambdas(cond, kb);
            walk_collect_lambdas(then_branch, kb);
            walk_collect_lambdas(else_branch, kb);
        }
        Expr::Try { expr, .. } => walk_collect_lambdas(expr, kb),
        Expr::Defer { cleanup, body } => {
            walk_collect_lambdas(cleanup, kb);
            walk_collect_lambdas(body, kb);
        }
        Expr::IntLit(_)
        | Expr::FloatLit(_)
        | Expr::BoolLit(_)
        | Expr::StringLit(_)
        | Expr::LocalVar(_)
        | Expr::TopRef(_)
        | Expr::SelfRef(_)
        | Expr::StateRef(_)
        | Expr::StateSelfRef(_)
        | Expr::BuiltinRef(_) => {}
    }
}

/// Collect the bare names of every extern (`BuiltinRef("ext/<name>")`)
/// referenced by `e`. Used to attach extern requirements to shipped code.
fn collect_ext_names(e: &Expr, out: &mut HashSet<String>) {
    match e {
        Expr::BuiltinRef(name) => {
            if let Some(ext) = name.strip_prefix("ext/") {
                out.insert(ext.to_owned());
            }
        }
        Expr::Lambda { body, .. } => collect_ext_names(body, out),
        Expr::Call(callee, args) => {
            collect_ext_names(callee, out);
            for a in args {
                collect_ext_names(a, out);
            }
        }
        Expr::Let { value, body } => {
            collect_ext_names(value, out);
            collect_ext_names(body, out);
        }
        Expr::Defer { cleanup, body } => {
            collect_ext_names(cleanup, out);
            collect_ext_names(body, out);
        }
        Expr::StructNew { fields, .. } => {
            for f in fields {
                collect_ext_names(f, out);
            }
        }
        Expr::Field { base, .. } => collect_ext_names(base, out),
        Expr::EnumNew { payload, .. } => {
            if let Some(p) = payload {
                collect_ext_names(p, out);
            }
        }
        Expr::Match { scrutinee, arms } => {
            collect_ext_names(scrutinee, out);
            for arm in arms {
                collect_ext_names(&arm.body, out);
            }
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } => {
            collect_ext_names(cond, out);
            collect_ext_names(then_branch, out);
            collect_ext_names(else_branch, out);
        }
        Expr::Try { expr, .. } => collect_ext_names(expr, out),
        Expr::IntLit(_)
        | Expr::FloatLit(_)
        | Expr::BoolLit(_)
        | Expr::StringLit(_)
        | Expr::LocalVar(_)
        | Expr::TopRef(_)
        | Expr::SelfRef(_)
        | Expr::StateRef(_)
        | Expr::StateSelfRef(_) => {}
    }
}

// ---------------------------------------------------------------------
// Dependency walking (during collect_transitive_deps)
// ---------------------------------------------------------------------

fn record(h: Hash, out: &mut Vec<Hash>, seen: &mut HashSet<Hash>) {
    if seen.insert(h) {
        out.push(h);
    }
}

// ---------------------------------------------------------------------
// Stateful-hash analysis (at() cache soundness)
// ---------------------------------------------------------------------

/// Compute the set of def + lambda content hashes that, when invoked,
/// transitively touch a node `state` cell (contain an `Expr::StateRef`
/// directly, or call a `TopRef` to another stateful hash).
///
/// The `at()` server memoizes Call replies by payload bytes, which is
/// sound ONLY for pure thunks. A thunk whose body reaches a `StateRef`
/// reads/writes node state and must NOT be cached (else a repeated
/// identical call would skip its mutation). `serve_one` looks up the
/// shipped closure's lambda hash in this set and bypasses the cache when
/// present. Computed over the whole module so transitive `TopRef` chains
/// resolve; a fixpoint handles recursion cycles.
pub fn stateful_hashes(
    defs: &[crate::resolve::ResolvedDef],
    extra_lambdas: &[(Hash, Expr)],
) -> HashSet<Hash> {
    // (hash, body-to-walk) for every def AND every nested lambda.
    let mut entries: Vec<(Hash, Expr)> = Vec::new();
    for rd in defs {
        match &rd.def {
            Def::Fn { body, .. } => {
                entries.push((rd.hash, body.clone()));
                collect_lambda_entries(body, &mut entries);
            }
            Def::State { init, .. } => {
                entries.push((rd.hash, init.clone()));
                collect_lambda_entries(init, &mut entries);
            }
            Def::Struct { .. } | Def::Enum { .. } => {}
        }
    }
    // Standalone lambdas (the code-fetch path ships the entry closure as
    // an `ItemKind::Lambda`, not embedded in any installed def body).
    for (h, e) in extra_lambdas {
        entries.push((*h, e.clone()));
        if let Expr::Lambda { body, .. } = e {
            collect_lambda_entries(body, &mut entries);
        }
    }
    let mut set: HashSet<Hash> = HashSet::new();
    loop {
        let mut changed = false;
        for (h, body) in &entries {
            if !set.contains(h) && expr_is_stateful(body, &set) {
                set.insert(*h);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    set
}

/// Gather `(hash, Expr::Lambda)` for every lambda nested anywhere in `e`.
/// The hash matches the one the lifter/knowledge base assigns
/// (`Hash::of_bytes(encode_expr(&Expr::Lambda { .. }))`), so it equals the
/// `code_hash` a shipped closure carries.
fn collect_lambda_entries(e: &Expr, out: &mut Vec<(Hash, Expr)>) {
    if let Expr::Lambda { params, body } = e {
        let lam = Expr::Lambda {
            params: params.clone(),
            body: body.clone(),
        };
        out.push((Hash::of_bytes(&encode_expr(&lam)), lam));
    }
    walk_children(e, &mut |c| collect_lambda_entries(c, out));
}

/// `true` if `e` reaches a `StateRef` directly or via a `TopRef` already
/// known to be stateful (per `set`). Recurses into all children including
/// lambda bodies.
fn expr_is_stateful(e: &Expr, set: &HashSet<Hash>) -> bool {
    match e {
        Expr::StateRef(_) | Expr::StateSelfRef(_) => true,
        Expr::TopRef(h) => set.contains(h),
        _ => {
            let mut found = false;
            walk_children(e, &mut |c| {
                if expr_is_stateful(c, set) {
                    found = true;
                }
            });
            found
        }
    }
}

/// Hashes of every def/lambda whose result is NOT safe to memoize across an
/// `at` call — a strict superset of [`stateful_hashes`]. A thunk is
/// cache-unsafe if it reads/writes node `state`, performs any
/// non-deterministic / external / shared-mutable effect (IO, Net, Atom,
/// FFI), or transitively calls something that does. Local-only mutation of
/// an owned value (`Mut`) and `Panic` remain cacheable.
///
/// Generalizes the State-only `stateful_hashes`: the leaf classifier is
/// [`crate::effects::builtin_effect_sig`] (the single source of effect
/// truth); transitivity / lambda-hashing / the fixpoint are reused.
pub fn non_cacheable_hashes(
    defs: &[crate::resolve::ResolvedDef],
    extra_lambdas: &[(Hash, Expr)],
) -> HashSet<Hash> {
    let mut entries: Vec<(Hash, Expr)> = Vec::new();
    for rd in defs {
        match &rd.def {
            Def::Fn { body, .. } => {
                entries.push((rd.hash, body.clone()));
                collect_lambda_entries(body, &mut entries);
            }
            Def::State { init, .. } => {
                entries.push((rd.hash, init.clone()));
                collect_lambda_entries(init, &mut entries);
            }
            Def::Struct { .. } | Def::Enum { .. } => {}
        }
    }
    for (h, e) in extra_lambdas {
        entries.push((*h, e.clone()));
        if let Expr::Lambda { body, .. } = e {
            collect_lambda_entries(body, &mut entries);
        }
    }
    let mut set: HashSet<Hash> = HashSet::new();
    loop {
        let mut changed = false;
        for (h, body) in &entries {
            if !set.contains(h) && expr_is_non_cacheable(body, &set) {
                set.insert(*h);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    set
}

/// Leaf predicate for [`non_cacheable_hashes`].
fn expr_is_non_cacheable(e: &Expr, set: &HashSet<Hash>) -> bool {
    use crate::effects::EffectSet;
    match e {
        Expr::StateRef(_) | Expr::StateSelfRef(_) => true,
        Expr::TopRef(h) => set.contains(h),
        Expr::BuiltinRef(name) => {
            let c = crate::effects::builtin_effect_sig(name).concrete;
            !c.without(EffectSet::MUT.union(EffectSet::PANIC)).is_empty()
        }
        _ => {
            let mut found = false;
            walk_children(e, &mut |c| {
                if expr_is_non_cacheable(c, set) {
                    found = true;
                }
            });
            found
        }
    }
}

/// Apply `f` to each immediate child expression of `e`.
fn walk_children(e: &Expr, f: &mut dyn FnMut(&Expr)) {
    match e {
        Expr::Call(callee, args) => {
            f(callee);
            for a in args {
                f(a);
            }
        }
        Expr::Lambda { body, .. } => f(body),
        Expr::Let { value, body } => {
            f(value);
            f(body);
        }
        Expr::Defer { cleanup, body } => {
            f(cleanup);
            f(body);
        }
        Expr::StructNew { fields, .. } => {
            for fd in fields {
                f(fd);
            }
        }
        Expr::Field { base, .. } => f(base),
        Expr::EnumNew { payload, .. } => {
            if let Some(p) = payload {
                f(p);
            }
        }
        Expr::Match { scrutinee, arms } => {
            f(scrutinee);
            for arm in arms {
                f(&arm.body);
            }
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } => {
            f(cond);
            f(then_branch);
            f(else_branch);
        }
        Expr::Try { expr, .. } => f(expr),
        Expr::IntLit(_)
        | Expr::FloatLit(_)
        | Expr::BoolLit(_)
        | Expr::StringLit(_)
        | Expr::LocalVar(_)
        | Expr::TopRef(_)
        | Expr::SelfRef(_)
        | Expr::StateRef(_)
        | Expr::StateSelfRef(_)
        | Expr::BuiltinRef(_) => {}
    }
}

/// Walk a `Def` collecting every content hash it references —
/// TopRef call targets, TypeRef bases of struct/enum/Apply types,
/// `struct_ref`/`enum_ref` constructor + accessor + pattern hashes.
/// Each hash is appended to `out` at most once (deduped via `seen`).
///
/// Used by the runner to compute the transitive closure of a root
/// def from a codebase, so we only load defs that are actually
/// reachable from the entry point (avoids loading stale named
/// versions that happen to share a name with the root's deps).
pub fn walk_def_deps(d: &Def, out: &mut Vec<Hash>, seen: &mut HashSet<Hash>) {
    match d {
        Def::Fn {
            params, ret, body, ..
        } => {
            for p in params {
                walk_type_deps(p, out, seen);
            }
            walk_type_deps(ret, out, seen);
            walk_expr_deps(body, out, seen);
        }
        Def::Struct { fields, .. } => {
            for (_, t) in fields {
                walk_type_deps(t, out, seen);
            }
        }
        Def::Enum { variants, .. } => {
            for (_, payload) in variants {
                if let Some(t) = payload {
                    walk_type_deps(t, out, seen);
                }
            }
        }
        Def::State { ty, init } => {
            walk_type_deps(ty, out, seen);
            walk_expr_deps(init, out, seen);
        }
    }
}

fn walk_type_deps(t: &Type, out: &mut Vec<Hash>, seen: &mut HashSet<Hash>) {
    match t {
        Type::Builtin(_) | Type::TypeVar(_) | Type::SelfRef(_) => {}
        Type::TypeRef(h) => record(*h, out, seen),
        Type::Apply(head, args) => {
            walk_type_deps(head, out, seen);
            for a in args {
                walk_type_deps(a, out, seen);
            }
        }
        Type::FnType { params, ret } => {
            for p in params {
                walk_type_deps(p, out, seen);
            }
            walk_type_deps(ret, out, seen);
        }
    }
}

fn walk_expr_deps(e: &Expr, out: &mut Vec<Hash>, seen: &mut HashSet<Hash>) {
    match e {
        Expr::IntLit(_)
        | Expr::FloatLit(_)
        | Expr::BoolLit(_)
        | Expr::StringLit(_)
        | Expr::LocalVar(_)
        | Expr::SelfRef(_)
        | Expr::StateSelfRef(_)
        | Expr::BuiltinRef(_) => {}
        // A node `state` reference is a dependency: shipping a handler
        // that reads/swaps a state must also ship the state definition.
        Expr::TopRef(h) | Expr::StateRef(h) => record(*h, out, seen),
        Expr::Call(callee, args) => {
            walk_expr_deps(callee, out, seen);
            for a in args {
                walk_expr_deps(a, out, seen);
            }
        }
        Expr::Lambda { params, body } => {
            // Nested lambda: its hash is a dependency.
            for t in params {
                walk_type_deps(t, out, seen);
            }
            let lambda_expr = Expr::Lambda {
                params: params.clone(),
                body: body.clone(),
            };
            let bytes = encode_expr(&lambda_expr);
            let h = Hash::of_bytes(&bytes);
            record(h, out, seen);
            // ALSO walk the body so any further nested deps surface.
            walk_expr_deps(body, out, seen);
        }
        Expr::Let { value, body } => {
            walk_expr_deps(value, out, seen);
            walk_expr_deps(body, out, seen);
        }
        Expr::StructNew { struct_ref, fields } => {
            record(*struct_ref, out, seen);
            for f in fields {
                walk_expr_deps(f, out, seen);
            }
        }
        Expr::Field {
            base, struct_ref, ..
        } => {
            record(*struct_ref, out, seen);
            walk_expr_deps(base, out, seen);
        }
        Expr::EnumNew {
            enum_ref, payload, ..
        } => {
            record(*enum_ref, out, seen);
            if let Some(p) = payload {
                walk_expr_deps(p, out, seen);
            }
        }
        Expr::Match { scrutinee, arms } => {
            walk_expr_deps(scrutinee, out, seen);
            for arm in arms {
                walk_pattern_deps(&arm.pattern, out, seen);
                walk_expr_deps(&arm.body, out, seen);
            }
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } => {
            walk_expr_deps(cond, out, seen);
            walk_expr_deps(then_branch, out, seen);
            walk_expr_deps(else_branch, out, seen);
        }
        Expr::Try { expr, enum_ref, .. } => {
            record(*enum_ref, out, seen);
            walk_expr_deps(expr, out, seen);
        }
        Expr::Defer { cleanup, body } => {
            walk_expr_deps(cleanup, out, seen);
            walk_expr_deps(body, out, seen);
        }
    }
}

fn walk_pattern_deps(p: &Pattern, out: &mut Vec<Hash>, seen: &mut HashSet<Hash>) {
    match p {
        Pattern::Wildcard | Pattern::Var => {}
        Pattern::Enum {
            enum_ref, payload, ..
        } => {
            record(*enum_ref, out, seen);
            if let Some(sub) = payload {
                walk_pattern_deps(sub, out, seen);
            }
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

    fn build(src: &str) -> ResolvedModule {
        let m = parse_module(src).unwrap();
        resolve_module(&m).unwrap()
    }

    #[test]
    fn kb_holds_every_def_under_its_hash() {
        let rm = build(
            "
            def x(n: Int) -> Int = n + 1
            def y(n: Int) -> Int = x(n) * 2
        ",
        );
        let kb = KnowledgeBase::build(&rm);
        for rd in &rm.defs {
            let entry = kb.lookup(&rd.hash).expect("def missing from KB");
            assert_eq!(entry.0, ItemKind::Def);
            // re-decoding should give the same def.
            let decoded = decode_def(&entry.1).unwrap();
            assert_eq!(decoded, rd.def);
        }
    }

    #[test]
    fn kb_holds_nested_lambdas() {
        let rm = build(
            "
            def make_adder(n: Int) -> fn(Int) -> Int = |x: Int| x + n
        ",
        );
        let kb = KnowledgeBase::build(&rm);
        // KB should have 1 def + 1 lambda.
        let mut def_count = 0;
        let mut lambda_count = 0;
        for (_, k, _) in kb.iter() {
            match k {
                ItemKind::Def => def_count += 1,
                ItemKind::Lambda => lambda_count += 1,
                ItemKind::Extern => {}
            }
        }
        assert_eq!(def_count, 1);
        assert_eq!(lambda_count, 1);
    }

    #[test]
    fn collect_transitive_deps_orders_deps_first() {
        let rm = build(
            "
            def x(n: Int) -> Int = n + 1
            def y(n: Int) -> Int = x(n) * 2
        ",
        );
        let x_hash = rm.get("x").unwrap().hash;
        let y_hash = rm.get("y").unwrap().hash;
        let kb = KnowledgeBase::build(&rm);
        let order = kb.collect_transitive_deps(&[y_hash]).unwrap();
        // x must come before y.
        let x_idx = order.iter().position(|h| h == &x_hash).unwrap();
        let y_idx = order.iter().position(|h| h == &y_hash).unwrap();
        assert!(x_idx < y_idx, "x must precede y in topological order");
    }

    #[test]
    fn collect_transitive_deps_includes_struct_and_lambda_refs() {
        let rm = build(
            "
            struct Node { a: Int, b: Int }
            def make(n: Int) -> Node = Node { a: n, b: n + 1 }
            def call_with(node: Node) -> Int = {
                let f = |x: Int| x + node.a;
                f(10)
            }
        ",
        );
        let node_hash = rm.get("Node").unwrap().hash;
        let make_hash = rm.get("make").unwrap().hash;
        let call_hash = rm.get("call_with").unwrap().hash;
        let kb = KnowledgeBase::build(&rm);
        let order = kb.collect_transitive_deps(&[call_hash]).unwrap();
        // Node must appear before call_with.
        let n_idx = order.iter().position(|h| h == &node_hash).unwrap();
        let c_idx = order.iter().position(|h| h == &call_hash).unwrap();
        assert!(n_idx < c_idx, "Node must precede call_with");
        // make is NOT a dep of call_with (call_with doesn't call make);
        // it should NOT appear unless we asked for it.
        assert!(!order.contains(&make_hash));
        // The lambda inside call_with must be in the dep list.
        let lambda_count = order
            .iter()
            .filter(|h| {
                matches!(
                    kb.lookup(h).map(|(k, _)| *k),
                    Some(ItemKind::Lambda)
                )
            })
            .count();
        assert_eq!(lambda_count, 1, "expected exactly 1 lambda in deps");
    }

    #[test]
    fn missing_hash_errors_clearly() {
        let kb = KnowledgeBase::new();
        let bogus = Hash([0; 32]);
        match kb.collect_transitive_deps(&[bogus]) {
            Err(KbError::MissingHash(h)) => assert_eq!(h, bogus),
            other => panic!("expected MissingHash, got {:?}", other),
        }
    }
}
