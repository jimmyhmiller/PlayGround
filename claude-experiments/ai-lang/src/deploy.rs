//! Type-safe live deploys onto a running node.
//!
//! A deploy installs a new immutable version into a running
//! [`IncrementalJit`] node, verifies it node-side — interface
//! compatibility for every rebound entry point AND shape continuity for
//! every node-resident `state` cell — and only then atomically flips
//! named bindings to the new hashes. Old versions stay resident, so
//! rollback is a flip back to a still-JIT'd hash.
//!
//! Everything leans on content addressing:
//!
//! - Installs are **additive and inert**: code only becomes reachable
//!   when a binding points at it, so every check here runs before
//!   anything is visible, and a refused deploy changes nothing.
//! - A `state` def's hash covers its `(type, initializer)` pair
//!   (`resolve` hashes the canonical `Def::State`), so *same hash
//!   implies same shape* by construction. Deploy-level **names** exist
//!   only to pair the new version's states with the node's live cells;
//!   runtime identity stays purely hash-based.
//!
//! State continuity, decided per deploy-level name:
//!
//! | live cell           | incoming state        | action |
//! |---------------------|-----------------------|--------|
//! | hash already live   | same hash             | `Keep` (the node's cell wins; init never re-runs) |
//! | same name, same type| new hash              | `Carryover` — alias the new hash to the live cell |
//! | same name, new type | new hash              | `Migrate` — requires a def `Fn(OldT) -> NewT`, typechecked **against the live cell's type**, run once at deploy time |
//! | live name absent    | —                     | refused unless `allow_state_drop` |
//!
//! The typecheck story: shipped items are decoded and the whole union
//! (everything installed so far + the new items) goes through
//! [`typecheck_module`] on the node before any mutation. This is the only
//! type validation installed code gets — `IncrementalJit::install`
//! deliberately does not typecheck — so the deploy path is where "I know
//! this change is type safe" is enforced.
//!
//! Known limitation (matches the rest of the install path): item hashes
//! are trusted, not re-derived from the bytes. Until the
//! re-canonicalisation pass exists, the guarantee is "type safe assuming
//! honest hashes", which authenticated deployers provide by convention.

use std::collections::{HashMap, HashSet};

use crate::ast::{Def, Type};
use crate::codec::{decode_def, encode_def, encode_extern};
use crate::codegen::{def_symbol, IncrementalJit};
use crate::edit::schemes_signature_eq;
use crate::hash::Hash;
use crate::net::ItemKind;
use crate::resolve::{ExternSig, ResolvedDef, ResolvedModule};
use crate::runtime::{
    ai_gc_box_int, ai_gc_unbox_int, ai_state_get, ai_state_set, Runtime, Thread,
};
use crate::typecheck::{typecheck_module, TypeCache, TypeScheme};

// =============================================================================
// Node-side deploy metadata
// =============================================================================

/// What the node knows about one live `state` cell: which hash currently
/// owns it and its declared type. The type is the node-side truth that
/// migrations are checked against — never the deployer's claim.
#[derive(Debug, Clone)]
pub struct LiveStateMeta {
    pub hash: Hash,
    pub ty: Type,
}

/// A named, typed, rebindable entry point. The `interface` is pinned when
/// the binding is first created; every later rebind must match it
/// (signature equality, the same predicate `update --propagate` uses).
#[derive(Debug, Clone)]
pub struct Binding {
    pub interface: TypeScheme,
    pub current: Hash,
    /// Previous roots, oldest first. Rollback pops the most recent.
    /// Every entry is still resident in the JIT, so a rollback is a
    /// pointer flip — no recompilation, no refetch.
    pub history: Vec<Hash>,
}

/// The node's mutable deploy root: name -> binding, name -> live state.
/// This is deliberately the *only* mutable thing in the deploy model —
/// code and state cells are immutable-by-hash underneath it.
#[derive(Debug, Default)]
pub struct DeployManager {
    live_states: HashMap<String, LiveStateMeta>,
    bindings: HashMap<String, Binding>,
    /// Schemes of everything typechecked so far, grown per deploy.
    /// Content-addressed memoisation: deploy typecheck cost is
    /// proportional to NEW code only.
    cache: TypeCache,
}

impl DeployManager {
    /// The hash a binding currently dispatches to.
    pub fn resolve(&self, name: &str) -> Option<Hash> {
        self.bindings.get(name).map(|b| b.current)
    }

    pub fn binding(&self, name: &str) -> Option<&Binding> {
        self.bindings.get(name)
    }

    pub fn live_state(&self, name: &str) -> Option<&LiveStateMeta> {
        self.live_states.get(name)
    }

    /// Flip a binding back to its previous root. The old code is still
    /// resident (installs are additive), so this is instant.
    ///
    /// State is NOT rolled back: a migrated cell keeps its migrated
    /// value, and the pre-migration cell still holds the value it had at
    /// migration time (the migration read it without destroying it), so
    /// rolled-back code finds its own cell exactly as it left it.
    pub fn rollback(&mut self, name: &str) -> Result<Hash, DeployError> {
        let b = self
            .bindings
            .get_mut(name)
            .ok_or_else(|| DeployError::UnknownBinding { name: name.to_string() })?;
        let prev = b
            .history
            .pop()
            .ok_or_else(|| DeployError::RollbackNoHistory { name: name.to_string() })?;
        b.current = prev;
        Ok(prev)
    }
}

// =============================================================================
// Request / report / errors
// =============================================================================

/// One deploy: the items of the new version plus the deploy-level
/// metadata that content addressing alone can't carry (names, migration
/// choices, rebind targets).
#[derive(Debug, Clone)]
pub struct DeployRequest {
    /// The new version's items, same wire shape the code-fetch handshake
    /// ships. Items already installed on the node are skipped.
    pub items: Vec<(ItemKind, Hash, Vec<u8>)>,
    /// Every `state` in the new version: deploy-level name -> state hash.
    pub state_names: HashMap<String, Hash>,
    /// State name -> hash of a migration def `Fn(OldT) -> NewT`, for
    /// states whose shape changed. The def must be in `items` (or
    /// already installed).
    pub migrations: HashMap<String, Hash>,
    /// Binding name -> new root def hash. A name not yet bound is
    /// created with its interface pinned to the root's scheme.
    pub rebinds: Vec<(String, Hash)>,
    /// Permit live states whose name is absent from the new version.
    /// Their cells stay resident (hash-keyed; a later deploy of the same
    /// def revives the same cell) but the node stops tracking them.
    pub allow_state_drop: bool,
}

impl DeployRequest {
    /// Build a request that ships a whole resolved module: every def
    /// (lambdas embedded in bodies travel inside them) plus every extern
    /// requirement, with the state name table derived from the module.
    pub fn from_module(rm: &ResolvedModule, rebinds: Vec<(String, Hash)>) -> Self {
        let mut items: Vec<(ItemKind, Hash, Vec<u8>)> = Vec::new();
        let mut state_names = HashMap::new();
        for rd in &rm.defs {
            items.push((ItemKind::Def, rd.hash, encode_def(&rd.def)));
            if matches!(rd.def, Def::State { .. }) {
                state_names.insert(rd.name.clone(), rd.hash);
            }
        }
        for (name, sig) in &rm.externs {
            let bytes = encode_extern(
                name,
                &sig.params,
                &sig.ret,
                sig.library.as_deref(),
                sig.variadic,
            );
            // Externs are not content-addressed; the hash slot is unused
            // by the installer. Ship a zero hash.
            items.push((ItemKind::Extern, Hash([0u8; 32]), bytes));
        }
        DeployRequest {
            items,
            state_names,
            migrations: HashMap::new(),
            rebinds,
            allow_state_drop: false,
        }
    }
}

/// What happened to one named state.
#[derive(Debug, Clone, PartialEq)]
pub enum StateAction {
    /// The incoming hash's cell is already live on this node (same def,
    /// or a revival of a previously-deployed def). The node's cell wins;
    /// the initializer never re-runs.
    Keep,
    /// No live cell under this name; the initializer runs at install.
    Fresh,
    /// Same name + same declared type under a new hash (the initializer
    /// changed): the new hash is aliased to the live cell. Data
    /// preserved, initializer skipped.
    Carryover { from: Hash },
    /// Same name, different declared type: the typechecked migration
    /// `via` ran over the old cell's value to produce the new cell.
    Migrate { from: Hash, via: Hash },
}

#[derive(Debug)]
pub struct DeployReport {
    /// Per-state outcome, sorted by name.
    pub states: Vec<(String, StateAction)>,
    /// Live states the new version no longer declares (only with
    /// `allow_state_drop`).
    pub dropped_states: Vec<String>,
    /// (binding, new root, previous root) per rebind.
    pub rebound: Vec<(String, Hash, Option<Hash>)>,
}

#[derive(Debug)]
pub enum DeployError {
    /// A shipped item failed to decode.
    Decode(String),
    /// The union of installed + shipped code does not typecheck.
    Typecheck(String),
    /// `state_names` references a hash that is neither shipped nor installed.
    UnknownStateHash { name: String, hash: Hash },
    /// `state_names` references a def that is not a `state`.
    NotAState { name: String, hash: Hash },
    /// A live state's declared type changed and no migration was supplied.
    ShapeChangedNoMigration {
        name: String,
        old: Type,
        new: Type,
    },
    /// The named migration def is neither shipped nor installed.
    MigrationUnknown { name: String, hash: Hash },
    /// The migration def doesn't have type `Fn(OldT) -> NewT` for the
    /// LIVE cell's `OldT` and the incoming state's `NewT`.
    MigrationWrongType {
        name: String,
        expected: String,
        found: String,
    },
    /// A live state's name is absent from the new version and
    /// `allow_state_drop` was not set.
    StateDropped { name: String },
    /// A rebind target is neither shipped nor installed.
    UnknownRoot { name: String, hash: Hash },
    /// A rebind target is not a function.
    RootNotFn { name: String, hash: Hash },
    /// A rebind target's signature doesn't match the binding's pinned
    /// interface.
    InterfaceMismatch {
        binding: String,
        interface: String,
        found: String,
    },
    /// Internal inconsistency: tracked live state has no cell.
    LiveStateMissingCell { name: String },
    /// `IncrementalJit::install` failed (the deploy was reverted).
    Install(String),
    RollbackNoHistory { name: String },
    UnknownBinding { name: String },
    /// A binding invoke the node can't perform (arity/type mismatch
    /// against the pinned interface, or a v1 transport limit).
    InvokeUnsupported { name: String, why: String },
}

impl std::fmt::Display for DeployError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeployError::Decode(m) => write!(f, "deploy: decode failed: {}", m),
            DeployError::Typecheck(m) => write!(f, "deploy: typecheck failed: {}", m),
            DeployError::UnknownStateHash { name, hash } => write!(
                f,
                "deploy: state `{}` references hash {} which is neither shipped nor installed",
                name, hash
            ),
            DeployError::NotAState { name, hash } => {
                write!(f, "deploy: `{}` ({}) is not a `state` def", name, hash)
            }
            DeployError::ShapeChangedNoMigration { name, old, new } => write!(
                f,
                "deploy: state `{}` changed shape ({:?} -> {:?}) and no migration was \
                 supplied; provide a migration fn({:?}) -> {:?} or deploy with a fresh name",
                name, old, new, old, new
            ),
            DeployError::MigrationUnknown { name, hash } => write!(
                f,
                "deploy: migration for state `{}` ({}) is neither shipped nor installed",
                name, hash
            ),
            DeployError::MigrationWrongType {
                name,
                expected,
                found,
            } => write!(
                f,
                "deploy: migration for state `{}` has the wrong type: expected {}, found {}",
                name, expected, found
            ),
            DeployError::StateDropped { name } => write!(
                f,
                "deploy: live state `{}` is absent from the new version; refuse to orphan \
                 its data (pass allow_state_drop to override)",
                name
            ),
            DeployError::UnknownRoot { name, hash } => write!(
                f,
                "deploy: rebind `{}` targets hash {} which is neither shipped nor installed",
                name, hash
            ),
            DeployError::RootNotFn { name, hash } => {
                write!(f, "deploy: rebind `{}` target {} is not a function", name, hash)
            }
            DeployError::InterfaceMismatch {
                binding,
                interface,
                found,
            } => write!(
                f,
                "deploy: rebind `{}` breaks its pinned interface: bound as {}, new root is {}",
                binding, interface, found
            ),
            DeployError::LiveStateMissingCell { name } => write!(
                f,
                "deploy: internal error: tracked live state `{}` has no cell on this node",
                name
            ),
            DeployError::Install(m) => {
                write!(f, "deploy: install failed (deploy reverted): {}", m)
            }
            DeployError::RollbackNoHistory { name } => {
                write!(f, "rollback: binding `{}` has no previous version", name)
            }
            DeployError::UnknownBinding { name } => {
                write!(f, "rollback: no binding named `{}`", name)
            }
            DeployError::InvokeUnsupported { name, why } => {
                write!(f, "invoke `{}`: {}", name, why)
            }
        }
    }
}

impl std::error::Error for DeployError {}

// =============================================================================
// The deploy
// =============================================================================

/// Internal: one state's verified plan.
struct PlannedState {
    name: String,
    hash: Hash,
    ty: Type,
    action: StateAction,
}

fn is_int(t: &Type) -> bool {
    matches!(t, Type::Builtin(n) if n == "Int")
}

fn scheme_desc(s: &TypeScheme) -> String {
    match s {
        TypeScheme::Fn { params, ret, .. } => format!("fn({:?}) -> {:?}", params, ret),
        other => format!("{:?}", other),
    }
}

/// Apply one deploy to a running node: decode, typecheck the union,
/// verify state continuity and binding interfaces, and only then install
/// + flip. Every refusal happens before any mutation; a refused deploy
/// leaves the node exactly as it was.
pub fn apply_deploy<'ctx>(
    mgr: &mut DeployManager,
    runtime: &mut Runtime,
    jit: &mut IncrementalJit<'ctx>,
    req: DeployRequest,
) -> Result<DeployReport, DeployError> {
    let DeployRequest {
        items,
        state_names,
        migrations,
        rebinds,
        allow_state_drop,
    } = req;

    // ---- 1. Decode (no mutation) -------------------------------------
    let mut new_defs: Vec<ResolvedDef> = Vec::new();
    let mut new_externs: HashMap<String, ExternSig> = HashMap::new();
    for (kind, hash, bytes) in &items {
        match kind {
            ItemKind::Def => {
                if jit.is_installed_def(hash) {
                    continue;
                }
                let def = decode_def(bytes)
                    .map_err(|e| DeployError::Decode(format!("def {}: {}", hash, e)))?;
                new_defs.push(ResolvedDef {
                    name: format!("deploy_{}", &hash.to_hex()[..8]),
                    hash: *hash,
                    def,
                });
            }
            ItemKind::Extern => {
                let (name, params, ret, library, variadic) =
                    crate::codec::decode_extern(bytes)
                        .map_err(|e| DeployError::Decode(format!("extern: {}", e)))?;
                new_externs.insert(
                    name,
                    ExternSig {
                        params,
                        ret,
                        library,
                        variadic,
                    },
                );
            }
            // Standalone lambdas carry no top-level type of their own;
            // they typecheck inside the def bodies that reference them.
            ItemKind::Lambda => {}
        }
    }

    // ---- 2. Typecheck the union (no mutation) ------------------------
    // This is the node's only type validation of shipped code: the
    // install path codegens without typechecking. Run against a CLONE of
    // the manager's scheme cache so a refused deploy can't poison it
    // with provisional schemes of bad defs.
    let mut union = ResolvedModule {
        defs: jit.installed_defs().to_vec(),
        at_binding: None,
        externs: jit.installed_externs().clone(),
    };
    union.defs.extend(new_defs.iter().cloned());
    union.externs.extend(new_externs);

    let mut cache = mgr.cache.clone();
    typecheck_module(&union, &mut cache)
        .map_err(|e| DeployError::Typecheck(format!("{:?}", e)))?;

    let by_hash: HashMap<Hash, &Def> =
        union.defs.iter().map(|rd| (rd.hash, &rd.def)).collect();

    // ---- 3. Plan state continuity (no mutation) ----------------------
    // Snapshot which hashes already have live cells: an incoming hash
    // that is already live is ALWAYS `Keep` — the node's cell wins (the
    // language's core idempotence invariant). This also covers redeploys
    // of an older version after a migration: the old cell is revived
    // as-is rather than a migration being silently dropped.
    let live_hashes: HashSet<[u8; 32]> = {
        let heap = runtime.heap.clone();
        let slots = heap.state_slots.lock().expect("state_slots poisoned");
        slots.keys().copied().collect()
    };

    let mut names_sorted: Vec<&String> = state_names.keys().collect();
    names_sorted.sort();

    let mut plan: Vec<PlannedState> = Vec::new();
    let mut skip_init: HashSet<Hash> = HashSet::new();
    for name in names_sorted {
        let hash = state_names[name];
        let def = by_hash
            .get(&hash)
            .ok_or_else(|| DeployError::UnknownStateHash { name: name.clone(), hash })?;
        let new_ty = match def {
            Def::State { ty, .. } => ty.clone(),
            _ => return Err(DeployError::NotAState { name: name.clone(), hash }),
        };

        let action = if live_hashes.contains(hash.as_bytes()) {
            StateAction::Keep
        } else {
            match mgr.live_states.get(name) {
                None => StateAction::Fresh,
                Some(m) if m.ty == new_ty => {
                    skip_init.insert(hash);
                    StateAction::Carryover { from: m.hash }
                }
                Some(m) => {
                    let via = *migrations.get(name).ok_or_else(|| {
                        DeployError::ShapeChangedNoMigration {
                            name: name.clone(),
                            old: m.ty.clone(),
                            new: new_ty.clone(),
                        }
                    })?;
                    let scheme = cache.get(&via).ok_or_else(|| {
                        DeployError::MigrationUnknown { name: name.clone(), hash: via }
                    })?;
                    // The migration is checked against the LIVE cell's
                    // type — node-side truth — not the deployer's claim.
                    let ok = matches!(
                        scheme,
                        TypeScheme::Fn { params, ret, .. }
                            if params.len() == 1 && params[0] == m.ty && *ret == new_ty
                    );
                    if !ok {
                        return Err(DeployError::MigrationWrongType {
                            name: name.clone(),
                            expected: format!("fn({:?}) -> {:?}", m.ty, new_ty),
                            found: scheme_desc(scheme),
                        });
                    }
                    skip_init.insert(hash);
                    StateAction::Migrate { from: m.hash, via }
                }
            }
        };
        plan.push(PlannedState {
            name: name.clone(),
            hash,
            ty: new_ty,
            action,
        });
    }

    let mut dropped_states: Vec<String> = Vec::new();
    for name in mgr.live_states.keys() {
        if !state_names.contains_key(name) {
            if !allow_state_drop {
                return Err(DeployError::StateDropped { name: name.clone() });
            }
            dropped_states.push(name.clone());
        }
    }
    dropped_states.sort();

    // ---- 4. Check binding interfaces (no mutation) --------------------
    let mut rebinds_checked: Vec<(String, Hash, TypeScheme)> = Vec::new();
    for (bname, root) in &rebinds {
        let scheme = cache
            .get(root)
            .ok_or_else(|| DeployError::UnknownRoot { name: bname.clone(), hash: *root })?;
        if !matches!(scheme, TypeScheme::Fn { .. }) {
            return Err(DeployError::RootNotFn { name: bname.clone(), hash: *root });
        }
        if let Some(b) = mgr.bindings.get(bname) {
            if !schemes_signature_eq(&b.interface, scheme) {
                return Err(DeployError::InterfaceMismatch {
                    binding: bname.clone(),
                    interface: scheme_desc(&b.interface),
                    found: scheme_desc(scheme),
                });
            }
        }
        rebinds_checked.push((bname.clone(), *root, scheme.clone()));
    }

    // ---- 5. Mutation begins: alias carryover cells --------------------
    // Aliasing BEFORE install makes the new hash's installer thunk see
    // `ai_state_present == 1` and no-op, which is exactly the carryover
    // semantics: the live cell survives, the initializer never runs.
    let heap = runtime.heap.clone();
    let mut added_aliases: Vec<[u8; 32]> = Vec::new();
    {
        let mut slots = heap.state_slots.lock().expect("state_slots poisoned");
        for p in &plan {
            if let StateAction::Carryover { from } = &p.action {
                if slots.contains_key(p.hash.as_bytes()) {
                    continue;
                }
                let idx = match slots.get(from.as_bytes()) {
                    Some(&i) => i,
                    None => {
                        return Err(DeployError::LiveStateMissingCell { name: p.name.clone() })
                    }
                };
                slots.insert(*p.hash.as_bytes(), idx);
                added_aliases.push(*p.hash.as_bytes());
            }
        }
    }

    // ---- 6. Install (additive; new code unreachable until the flip) ---
    if let Err(e) = jit.install_with(runtime, items, &skip_init) {
        // Revert the aliases so a failed install leaves no trace.
        let mut slots = heap.state_slots.lock().expect("state_slots poisoned");
        for k in &added_aliases {
            slots.remove(k);
        }
        return Err(DeployError::Install(e.to_string()));
    }

    // ---- 7. Run migrations -------------------------------------------
    // The migration def is now compiled. Its scheme was verified above:
    // `fn(OldT) -> NewT` against the live cell's recorded type. Values
    // cross the boundary with the same convention the state installer
    // uses: cells always hold heap pointers, with bare-Int states boxed.
    for p in &plan {
        if let StateAction::Migrate { from, via } = &p.action {
            let old_ty = &mgr.live_states[&p.name].ty;
            let thread = runtime.thread_ptr();
            unsafe {
                let old_val = ai_state_get(thread, from.0.as_ptr());
                let arg: u64 = if is_int(old_ty) {
                    ai_gc_unbox_int(old_val) as u64
                } else {
                    old_val as u64
                };
                let addr = jit
                    .engine
                    .get_function_address(&def_symbol(via))
                    .map_err(|_| {
                        DeployError::Install(format!(
                            "migration fn {} has no compiled symbol",
                            via
                        ))
                    })?;
                let mig: unsafe extern "C" fn(*mut Thread, u64) -> u64 =
                    core::mem::transmute(addr);
                // No panic channel: a contract violation in the migration
                // fn aborts the process before this returns; modeled
                // failures should be designed into the migration's types.
                let res = mig(thread, arg);
                let val: *mut u8 = if is_int(&p.ty) {
                    ai_gc_box_int(thread, res as i64)
                } else {
                    res as *mut u8
                };
                // Not present (its installer was suppressed), so this
                // installs the migrated value as the new hash's cell.
                ai_state_set(thread, p.hash.0.as_ptr(), val);
            }
        }
    }

    // ---- 8. Flip bindings + commit metadata ----------------------------
    let mut rebound: Vec<(String, Hash, Option<Hash>)> = Vec::new();
    for (bname, root, scheme) in rebinds_checked {
        match mgr.bindings.entry(bname.clone()) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                let b = e.get_mut();
                let prev = b.current;
                if prev != root {
                    b.history.push(prev);
                    b.current = root;
                }
                rebound.push((bname, root, Some(prev)));
            }
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert(Binding {
                    interface: scheme,
                    current: root,
                    history: Vec::new(),
                });
                rebound.push((bname, root, None));
            }
        }
    }

    let mut new_live: HashMap<String, LiveStateMeta> = HashMap::new();
    let mut report_states: Vec<(String, StateAction)> = Vec::new();
    for p in plan {
        new_live.insert(
            p.name.clone(),
            LiveStateMeta {
                hash: p.hash,
                ty: p.ty,
            },
        );
        report_states.push((p.name, p.action));
    }
    mgr.live_states = new_live;
    mgr.cache = cache;

    Ok(DeployReport {
        states: report_states,
        dropped_states,
        rebound,
    })
}

// =============================================================================
// Wire protocol: deploy / rollback / invoke over the ail frame transport
// =============================================================================
//
// The client side is **runtime-less**: building, sending, and decoding
// these frames needs no JIT, heap, or thread context — just bytes. The
// CLI deploy/rollback/invoke commands are thin wrappers over the
// `*_on_channel` functions below.

use crate::net::{
    decode_items, encode_items, Channel, NetError, KIND_CALL, KIND_DEPLOY, KIND_DEPLOY_ERR,
    KIND_DEPLOY_OK, KIND_INVOKE, KIND_INVOKE_RESULT, KIND_ROLLBACK,
};

fn write_str(out: &mut Vec<u8>, s: &str) {
    let n: u32 = s.len().try_into().expect("string too long for frame");
    out.extend_from_slice(&n.to_be_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn read_str(body: &[u8], pos: &mut usize) -> Result<String, NetError> {
    if *pos + 4 > body.len() {
        return Err(NetError::ProtocolViolation("string truncated before length"));
    }
    let n = u32::from_be_bytes([body[*pos], body[*pos + 1], body[*pos + 2], body[*pos + 3]])
        as usize;
    *pos += 4;
    if *pos + n > body.len() {
        return Err(NetError::ProtocolViolation("string truncated before bytes"));
    }
    let s = std::str::from_utf8(&body[*pos..*pos + n])
        .map_err(|_| NetError::ProtocolViolation("string is not utf-8"))?
        .to_string();
    *pos += n;
    Ok(s)
}

fn read_hash(body: &[u8], pos: &mut usize) -> Result<Hash, NetError> {
    if *pos + 32 > body.len() {
        return Err(NetError::ProtocolViolation("hash truncated"));
    }
    let mut buf = [0u8; 32];
    buf.copy_from_slice(&body[*pos..*pos + 32]);
    *pos += 32;
    Ok(Hash(buf))
}

fn read_u32(body: &[u8], pos: &mut usize) -> Result<u32, NetError> {
    if *pos + 4 > body.len() {
        return Err(NetError::ProtocolViolation("u32 truncated"));
    }
    let v = u32::from_be_bytes([body[*pos], body[*pos + 1], body[*pos + 2], body[*pos + 3]]);
    *pos += 4;
    Ok(v)
}

/// Append a name->hash table, sorted by name so encodings are
/// deterministic regardless of HashMap iteration order.
fn write_name_hash_table(out: &mut Vec<u8>, table: &HashMap<String, Hash>) {
    let mut entries: Vec<(&String, &Hash)> = table.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    let n: u32 = entries.len().try_into().expect("table too large");
    out.extend_from_slice(&n.to_be_bytes());
    for (name, h) in entries {
        write_str(out, name);
        out.extend_from_slice(h.as_bytes());
    }
}

fn read_name_hash_table(
    body: &[u8],
    pos: &mut usize,
) -> Result<HashMap<String, Hash>, NetError> {
    let n = read_u32(body, pos)? as usize;
    let mut out = HashMap::with_capacity(n);
    for _ in 0..n {
        let name = read_str(body, pos)?;
        let h = read_hash(body, pos)?;
        out.insert(name, h);
    }
    Ok(out)
}

/// Encode a [`DeployRequest`] as a `KIND_DEPLOY` frame body.
pub fn encode_deploy_request(req: &DeployRequest) -> Vec<u8> {
    let mut out = Vec::with_capacity(256);
    out.push(KIND_DEPLOY);
    encode_items(&req.items, &mut out);
    write_name_hash_table(&mut out, &req.state_names);
    write_name_hash_table(&mut out, &req.migrations);
    let n: u32 = req.rebinds.len().try_into().expect("too many rebinds");
    out.extend_from_slice(&n.to_be_bytes());
    for (name, h) in &req.rebinds {
        write_str(&mut out, name);
        out.extend_from_slice(h.as_bytes());
    }
    out.push(if req.allow_state_drop { 1 } else { 0 });
    out
}

/// Decode a `KIND_DEPLOY` frame body.
pub fn decode_deploy_request(body: &[u8]) -> Result<DeployRequest, NetError> {
    if body.is_empty() || body[0] != KIND_DEPLOY {
        return Err(NetError::ProtocolViolation("not a Deploy frame"));
    }
    let mut pos = 1usize;
    let items = decode_items(body, &mut pos)?;
    let state_names = read_name_hash_table(body, &mut pos)?;
    let migrations = read_name_hash_table(body, &mut pos)?;
    let n = read_u32(body, &mut pos)? as usize;
    let mut rebinds = Vec::with_capacity(n);
    for _ in 0..n {
        let name = read_str(body, &mut pos)?;
        let h = read_hash(body, &mut pos)?;
        rebinds.push((name, h));
    }
    if pos + 1 != body.len() {
        return Err(NetError::ProtocolViolation("Deploy frame has wrong tail"));
    }
    let allow_state_drop = body[pos] != 0;
    Ok(DeployRequest {
        items,
        state_names,
        migrations,
        rebinds,
        allow_state_drop,
    })
}

const ACTION_KEEP: u8 = 0;
const ACTION_FRESH: u8 = 1;
const ACTION_CARRYOVER: u8 = 2;
const ACTION_MIGRATE: u8 = 3;

/// Encode a [`DeployReport`] as a `KIND_DEPLOY_OK` frame body.
pub fn encode_deploy_report(report: &DeployReport) -> Vec<u8> {
    let mut out = Vec::with_capacity(128);
    out.push(KIND_DEPLOY_OK);
    let n: u32 = report.states.len() as u32;
    out.extend_from_slice(&n.to_be_bytes());
    for (name, action) in &report.states {
        write_str(&mut out, name);
        match action {
            StateAction::Keep => out.push(ACTION_KEEP),
            StateAction::Fresh => out.push(ACTION_FRESH),
            StateAction::Carryover { from } => {
                out.push(ACTION_CARRYOVER);
                out.extend_from_slice(from.as_bytes());
            }
            StateAction::Migrate { from, via } => {
                out.push(ACTION_MIGRATE);
                out.extend_from_slice(from.as_bytes());
                out.extend_from_slice(via.as_bytes());
            }
        }
    }
    let n: u32 = report.dropped_states.len() as u32;
    out.extend_from_slice(&n.to_be_bytes());
    for name in &report.dropped_states {
        write_str(&mut out, name);
    }
    let n: u32 = report.rebound.len() as u32;
    out.extend_from_slice(&n.to_be_bytes());
    for (name, new, prev) in &report.rebound {
        write_str(&mut out, name);
        out.extend_from_slice(new.as_bytes());
        match prev {
            Some(p) => {
                out.push(1);
                out.extend_from_slice(p.as_bytes());
            }
            None => out.push(0),
        }
    }
    out
}

/// Decode a `KIND_DEPLOY_OK` frame body.
pub fn decode_deploy_report(body: &[u8]) -> Result<DeployReport, NetError> {
    if body.is_empty() || body[0] != KIND_DEPLOY_OK {
        return Err(NetError::ProtocolViolation("not a DeployOk frame"));
    }
    let mut pos = 1usize;
    let n = read_u32(body, &mut pos)? as usize;
    let mut states = Vec::with_capacity(n);
    for _ in 0..n {
        let name = read_str(body, &mut pos)?;
        if pos >= body.len() {
            return Err(NetError::ProtocolViolation("state action truncated"));
        }
        let tag = body[pos];
        pos += 1;
        let action = match tag {
            ACTION_KEEP => StateAction::Keep,
            ACTION_FRESH => StateAction::Fresh,
            ACTION_CARRYOVER => StateAction::Carryover { from: read_hash(body, &mut pos)? },
            ACTION_MIGRATE => StateAction::Migrate {
                from: read_hash(body, &mut pos)?,
                via: read_hash(body, &mut pos)?,
            },
            other => {
                return Err(NetError::ProtocolViolation_owned(format!(
                    "unknown state action tag {}",
                    other
                )))
            }
        };
        states.push((name, action));
    }
    let n = read_u32(body, &mut pos)? as usize;
    let mut dropped_states = Vec::with_capacity(n);
    for _ in 0..n {
        dropped_states.push(read_str(body, &mut pos)?);
    }
    let n = read_u32(body, &mut pos)? as usize;
    let mut rebound = Vec::with_capacity(n);
    for _ in 0..n {
        let name = read_str(body, &mut pos)?;
        let new = read_hash(body, &mut pos)?;
        if pos >= body.len() {
            return Err(NetError::ProtocolViolation("rebound entry truncated"));
        }
        let has_prev = body[pos] != 0;
        pos += 1;
        let prev = if has_prev { Some(read_hash(body, &mut pos)?) } else { None };
        rebound.push((name, new, prev));
    }
    if pos != body.len() {
        return Err(NetError::ProtocolViolation("DeployOk frame has trailing bytes"));
    }
    Ok(DeployReport {
        states,
        dropped_states,
        rebound,
    })
}

fn encode_deploy_err(msg: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + msg.len());
    out.push(KIND_DEPLOY_ERR);
    out.extend_from_slice(msg.as_bytes());
    out
}

/// Parse a deploy/rollback reply: `DeployOk` -> report, `DeployErr` ->
/// [`NetError::DeployRefused`] with the node's message.
fn parse_deploy_reply(body: &[u8]) -> Result<DeployReport, NetError> {
    match body.first() {
        Some(&KIND_DEPLOY_OK) => decode_deploy_report(body),
        Some(&KIND_DEPLOY_ERR) => Err(NetError::DeployRefused(
            String::from_utf8_lossy(&body[1..]).into_owned(),
        )),
        Some(&other) => Err(NetError::BadKind(other)),
        None => Err(NetError::ProtocolViolation("empty deploy reply")),
    }
}

// =============================================================================
// Node-side dispatcher
// =============================================================================

/// Serve one request on a deploy-capable node: `Deploy`, `Rollback`,
/// `Invoke`, or a plain `Call` (the existing `at()` path with the
/// code-fetch handshake). Refusals are *replies* — the connection stays
/// usable and the node is untouched.
///
/// # Safety
/// Same contract as [`crate::net::serve_with_install`]: invokes JIT'd
/// code; the runtime and JIT must outlive the call; no concurrent
/// mutation of the runtime (single-threaded node v1).
pub unsafe fn serve_deploy_turn<'ctx>(
    rt: &mut Runtime,
    jit: &mut IncrementalJit<'ctx>,
    mgr: &mut DeployManager,
    channel: &mut dyn Channel,
) -> Result<(), NetError> {
    let body = {
        let _blocked = crate::net::BlockedRegion::enter(rt);
        channel.read_frame()
    }?;
    if body.is_empty() {
        return Err(NetError::ProtocolViolation("empty frame body"));
    }
    match body[0] {
        KIND_DEPLOY => {
            let req = decode_deploy_request(&body)?;
            match apply_deploy(mgr, rt, jit, req) {
                Ok(report) => channel.write_frame(&encode_deploy_report(&report)),
                Err(e) => channel.write_frame(&encode_deploy_err(&e.to_string())),
            }
        }
        KIND_ROLLBACK => {
            let mut pos = 1usize;
            let name = read_str(&body, &mut pos)?;
            let before = mgr.resolve(&name);
            match mgr.rollback(&name) {
                Ok(now) => {
                    let report = DeployReport {
                        states: vec![],
                        dropped_states: vec![],
                        rebound: vec![(name, now, before)],
                    };
                    channel.write_frame(&encode_deploy_report(&report))
                }
                Err(e) => channel.write_frame(&encode_deploy_err(&e.to_string())),
            }
        }
        KIND_INVOKE => {
            let mut pos = 1usize;
            let name = read_str(&body, &mut pos)?;
            let argc = read_u32(&body, &mut pos)? as usize;
            let mut args = Vec::with_capacity(argc);
            for _ in 0..argc {
                if pos + 8 > body.len() {
                    return Err(NetError::ProtocolViolation("invoke args truncated"));
                }
                let mut buf = [0u8; 8];
                buf.copy_from_slice(&body[pos..pos + 8]);
                pos += 8;
                args.push(i64::from_be_bytes(buf));
            }
            match unsafe { invoke_binding(mgr, rt, jit, &name, &args) } {
                Ok(v) => {
                    let mut reply = Vec::with_capacity(9);
                    reply.push(KIND_INVOKE_RESULT);
                    reply.extend_from_slice(&v.to_be_bytes());
                    channel.write_frame(&reply)
                }
                Err(e) => channel.write_frame(&encode_deploy_err(&e.to_string())),
            }
        }
        KIND_CALL => unsafe { crate::net::handle_call_frame(rt, jit, channel, &body) },
        // A token-less node acks auth frames so clients configured with
        // a token still work against it. Token-PROTECTED nodes verify
        // the first frame before entering this dispatcher (see
        // `net::server_expect_auth`); later auth frames are no-ops.
        crate::net::KIND_AUTH => channel.write_frame(&[crate::net::KIND_AUTH]),
        other => Err(NetError::BadKind(other)),
    }
}

/// Invoke a binding by name with Int args. The binding's pinned
/// interface is the gate: arity must match and (v1) every param and the
/// return type must be `Int` — richer values go through the `at()`
/// path, which ships real wire values.
///
/// # Safety
/// Invokes JIT'd code; same contract as [`serve_deploy_turn`].
pub unsafe fn invoke_binding<'ctx>(
    mgr: &DeployManager,
    rt: &Runtime,
    jit: &IncrementalJit<'ctx>,
    name: &str,
    args: &[i64],
) -> Result<i64, DeployError> {
    let b = mgr
        .binding(name)
        .ok_or_else(|| DeployError::UnknownBinding { name: name.to_string() })?;
    let (params, ret) = match &b.interface {
        TypeScheme::Fn { params, ret, .. } => (params, ret),
        // Bindings are only ever created over Fn schemes (apply_deploy
        // checks), so this is an internal inconsistency, not user error.
        other => {
            return Err(DeployError::InvokeUnsupported {
                name: name.to_string(),
                why: format!("binding is not a function: {:?}", other),
            })
        }
    };
    if params.len() != args.len() {
        return Err(DeployError::InvokeUnsupported {
            name: name.to_string(),
            why: format!("arity mismatch: binding takes {}, got {}", params.len(), args.len()),
        });
    }
    if !params.iter().all(is_int) || !is_int(ret) {
        return Err(DeployError::InvokeUnsupported {
            name: name.to_string(),
            why: format!(
                "invoke v1 carries Int args/returns only; interface is {}",
                scheme_desc(&b.interface)
            ),
        });
    }
    if args.len() > 4 {
        return Err(DeployError::InvokeUnsupported {
            name: name.to_string(),
            why: format!("invoke v1 supports at most 4 args, got {}", args.len()),
        });
    }
    let addr = jit
        .engine
        .get_function_address(&def_symbol(&b.current))
        .map_err(|_| DeployError::InvokeUnsupported {
            name: name.to_string(),
            why: format!("bound hash {} has no compiled symbol", b.current),
        })?;
    let thread = crate::net::serve_thread_ptr(rt);
    let v = unsafe {
        match args.len() {
            0 => core::mem::transmute::<usize, unsafe extern "C" fn(*mut Thread) -> i64>(addr)(
                thread,
            ),
            1 => core::mem::transmute::<usize, unsafe extern "C" fn(*mut Thread, i64) -> i64>(
                addr,
            )(thread, args[0]),
            2 => core::mem::transmute::<
                usize,
                unsafe extern "C" fn(*mut Thread, i64, i64) -> i64,
            >(addr)(thread, args[0], args[1]),
            3 => core::mem::transmute::<
                usize,
                unsafe extern "C" fn(*mut Thread, i64, i64, i64) -> i64,
            >(addr)(thread, args[0], args[1], args[2]),
            4 => core::mem::transmute::<
                usize,
                unsafe extern "C" fn(*mut Thread, i64, i64, i64, i64) -> i64,
            >(addr)(thread, args[0], args[1], args[2], args[3]),
            _ => unreachable!("arity capped above"),
        }
    };
    // No panic channel: a contract violation in the binding aborts the
    // process before this returns; modeled failures are Result values
    // in the binding's own signature.
    Ok(v)
}

// =============================================================================
// Client side (runtime-less)
// =============================================================================

/// Ship a deploy to a node and wait for its verdict.
pub fn deploy_on_channel(
    channel: &mut dyn Channel,
    req: &DeployRequest,
) -> Result<DeployReport, NetError> {
    channel.write_frame(&encode_deploy_request(req))?;
    let reply = channel.read_frame()?;
    parse_deploy_reply(&reply)
}

/// Flip a binding on a node back to its previous version. The returned
/// report's `rebound` entry is `(name, now-current, what-it-was)`.
pub fn rollback_on_channel(
    channel: &mut dyn Channel,
    name: &str,
) -> Result<DeployReport, NetError> {
    let mut out = Vec::with_capacity(1 + 4 + name.len());
    out.push(KIND_ROLLBACK);
    write_str(&mut out, name);
    channel.write_frame(&out)?;
    let reply = channel.read_frame()?;
    parse_deploy_reply(&reply)
}

/// Call a deployed binding by name with Int args.
pub fn invoke_on_channel(
    channel: &mut dyn Channel,
    name: &str,
    args: &[i64],
) -> Result<i64, NetError> {
    let mut out = Vec::with_capacity(1 + 4 + name.len() + 4 + args.len() * 8);
    out.push(KIND_INVOKE);
    write_str(&mut out, name);
    let n: u32 = args.len() as u32;
    out.extend_from_slice(&n.to_be_bytes());
    for a in args {
        out.extend_from_slice(&a.to_be_bytes());
    }
    channel.write_frame(&out)?;
    let reply = channel.read_frame()?;
    match reply.first() {
        Some(&KIND_INVOKE_RESULT) => {
            if reply.len() != 9 {
                return Err(NetError::ProtocolViolation("bad InvokeResult length"));
            }
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&reply[1..9]);
            Ok(i64::from_be_bytes(buf))
        }
        Some(&KIND_DEPLOY_ERR) => Err(NetError::DeployRefused(
            String::from_utf8_lossy(&reply[1..]).into_owned(),
        )),
        Some(&other) => Err(NetError::BadKind(other)),
        None => Err(NetError::ProtocolViolation("empty invoke reply")),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::{init_native_target, CompiledModule};
    use inkwell::context::Context;

    static INIT: std::sync::Once = std::sync::Once::new();
    fn init() {
        INIT.call_once(|| {
            init_native_target().expect("init native target");
        });
    }

    /// An empty node, the same way `ai-lang serve` boots one.
    fn fresh_node<'ctx>(ctx: &'ctx Context) -> (Runtime, IncrementalJit<'ctx>) {
        let m = crate::parser::parse_module("").expect("parse empty");
        let r = crate::resolve::resolve_module(&m).expect("resolve empty");
        let cm = CompiledModule::build(ctx, &r).expect("build empty");
        let rt = Runtime::new_with_metadata(
            cm.closure_type_infos.clone(),
            cm.shape_registry.clone(),
            cm.shape_meta.clone(),
            cm.shape_by_type_id.clone(),
        );
        let jit = IncrementalJit::new(cm, &rt).expect("incremental jit");
        (rt, jit)
    }

    fn resolve_src(src: &str) -> (ResolvedModule, HashMap<String, Hash>) {
        let m = crate::parser::parse_module(src).expect("parse");
        let r = crate::resolve::resolve_module(&m).expect("resolve");
        let names = r.defs.iter().map(|d| (d.name.clone(), d.hash)).collect();
        (r, names)
    }

    /// Invoke a `fn(Int) -> Int` def by hash on the node.
    unsafe fn call1(jit: &IncrementalJit, rt: &Runtime, h: &Hash, arg: i64) -> i64 {
        let addr = jit
            .engine
            .get_function_address(&def_symbol(h))
            .expect("deployed fn must have a compiled symbol");
        let f: unsafe extern "C" fn(*mut Thread, i64) -> i64 =
            unsafe { core::mem::transmute(addr) };
        unsafe { f(rt.thread_ptr(), arg) }
    }

    /// Invoke whatever a binding currently dispatches to.
    unsafe fn call_svc(
        mgr: &DeployManager,
        jit: &IncrementalJit,
        rt: &Runtime,
        arg: i64,
    ) -> i64 {
        let h = mgr.resolve("svc").expect("binding `svc` resolves");
        unsafe { call1(jit, rt, &h, arg) }
    }

    const V1: &str = "
        state counter: Atom<Int> = atom_new(0)
        def bump(d: Int) -> Int = atom_swap(counter, |n: Int| n + d)
    ";

    fn action_for<'r>(report: &'r DeployReport, name: &str) -> &'r StateAction {
        &report
            .states
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("state `{}` missing from report", name))
            .1
    }

    /// First deploy onto an empty node: fresh state, binding created with
    /// the root's interface, dispatch through the binding works.
    #[test]
    fn deploy_fresh_bind_invoke() {
        init();
        let ctx = Context::create();
        let (mut rt, mut jit) = fresh_node(&ctx);
        let mut mgr = DeployManager::default();

        let (rm, names) = resolve_src(V1);
        let req = DeployRequest::from_module(&rm, vec![("svc".to_string(), names["bump"])]);
        let report = apply_deploy(&mut mgr, &mut rt, &mut jit, req).expect("v1 deploys");

        assert_eq!(*action_for(&report, "counter"), StateAction::Fresh);
        assert_eq!(report.rebound, vec![("svc".to_string(), names["bump"], None)]);
        assert_eq!(mgr.resolve("svc"), Some(names["bump"]));
        unsafe {
            assert_eq!(call_svc(&mgr, &jit, &rt, 5), 5);
            assert_eq!(call_svc(&mgr, &jit, &rt, 10), 15, "state cell is live");
        }
    }

    /// The headline property: an initializer edit (same declared type)
    /// carries the live cell over — data survives the deploy, the new
    /// initializer never runs — and rollback is an instant flip back to
    /// the old hash, sharing the same cell.
    #[test]
    fn carryover_preserves_state_and_rollback_is_instant() {
        init();
        let ctx = Context::create();
        let (mut rt, mut jit) = fresh_node(&ctx);
        let mut mgr = DeployManager::default();

        let (rm1, names1) = resolve_src(V1);
        let req = DeployRequest::from_module(&rm1, vec![("svc".to_string(), names1["bump"])]);
        apply_deploy(&mut mgr, &mut rt, &mut jit, req).expect("v1 deploys");
        unsafe {
            assert_eq!(call_svc(&mgr, &jit, &rt, 5), 5);
            assert_eq!(call_svc(&mgr, &jit, &rt, 10), 15);
        }

        // v2: initializer changed (0 -> 100), type unchanged; body now
        // adds twice. If the initializer ran, the counter would read 100.
        let v2 = "
            state counter: Atom<Int> = atom_new(100)
            def bump(d: Int) -> Int = atom_swap(counter, |n: Int| n + d + d)
        ";
        let (rm2, names2) = resolve_src(v2);
        assert_ne!(names1["counter"], names2["counter"], "init change = new hash");
        let req = DeployRequest::from_module(&rm2, vec![("svc".to_string(), names2["bump"])]);
        let report = apply_deploy(&mut mgr, &mut rt, &mut jit, req).expect("v2 deploys");

        assert_eq!(
            *action_for(&report, "counter"),
            StateAction::Carryover { from: names1["counter"] }
        );
        unsafe {
            // 15 (carried over) + 1 + 1 — NOT 100 + 2.
            assert_eq!(call_svc(&mgr, &jit, &rt, 1), 17, "cell carried over, init skipped");
        }

        // Rollback: flip back to the v1 hash. Old code still resident;
        // both versions' state hashes alias the same live cell.
        let prev = mgr.rollback("svc").expect("one deploy of history");
        assert_eq!(prev, names1["bump"]);
        unsafe {
            assert_eq!(call_svc(&mgr, &jit, &rt, 2), 19, "v1 adds once, same shared cell");
        }
    }

    /// Shape changes are refused without a migration, refused with a
    /// wrong-typed migration (checked against the LIVE cell's type), and
    /// applied with a correctly-typed one. Refusals leave the node
    /// untouched. After migrating, rollback finds the old cell exactly
    /// as the migration read it.
    #[test]
    fn shape_change_requires_typechecked_migration() {
        init();
        let ctx = Context::create();
        let (mut rt, mut jit) = fresh_node(&ctx);
        let mut mgr = DeployManager::default();

        let (rm1, names1) = resolve_src(V1);
        let req = DeployRequest::from_module(&rm1, vec![("svc".to_string(), names1["bump"])]);
        apply_deploy(&mut mgr, &mut rt, &mut jit, req).expect("v1 deploys");
        unsafe {
            assert_eq!(call_svc(&mgr, &jit, &rt, 5), 5);
        }

        // Shape change: Atom<Int> -> bare Int (the cell becomes an
        // immutable snapshot). The migration must read the old atom; if
        // the new initializer ran instead, the cell would read -999.
        let v2 = "
            state counter: Int = 0 - 999
            def bump(d: Int) -> Int = counter + d
            def mig(old: Atom<Int>) -> Int = atom_load(old)
            def mig_bad(old: Int) -> Int = old
        ";
        let (rm2, names2) = resolve_src(v2);

        // (a) No migration supplied -> refused, node untouched.
        let req = DeployRequest::from_module(&rm2, vec![("svc".to_string(), names2["bump"])]);
        match apply_deploy(&mut mgr, &mut rt, &mut jit, req) {
            Err(DeployError::ShapeChangedNoMigration { name, .. }) => {
                assert_eq!(name, "counter")
            }
            other => panic!("expected shape-change refusal, got {:?}", other.map(|_| ())),
        }
        assert_eq!(mgr.resolve("svc"), Some(names1["bump"]), "binding unchanged");
        unsafe {
            assert_eq!(call_svc(&mgr, &jit, &rt, 3), 8, "old version fully live after refusal");
        }

        // (b) Wrong-typed migration (takes Int, live cell is Atom<Int>)
        //     -> refused. The check runs against the node's recorded type.
        let mut req =
            DeployRequest::from_module(&rm2, vec![("svc".to_string(), names2["bump"])]);
        req.migrations.insert("counter".to_string(), names2["mig_bad"]);
        match apply_deploy(&mut mgr, &mut rt, &mut jit, req) {
            Err(DeployError::MigrationWrongType { name, .. }) => assert_eq!(name, "counter"),
            other => panic!("expected wrong-type refusal, got {:?}", other.map(|_| ())),
        }

        // (c) Correct migration -> deploy lands; the new version sees the
        //     migrated value (the atom held 8, so the Int cell is 8).
        let mut req =
            DeployRequest::from_module(&rm2, vec![("svc".to_string(), names2["bump"])]);
        req.migrations.insert("counter".to_string(), names2["mig"]);
        let report = apply_deploy(&mut mgr, &mut rt, &mut jit, req).expect("migrating deploy");
        assert_eq!(
            *action_for(&report, "counter"),
            StateAction::Migrate { from: names1["counter"], via: names2["mig"] }
        );
        unsafe {
            assert_eq!(call_svc(&mgr, &jit, &rt, 7), 15, "migrated cell holds 8, not -999");
        }

        // Rollback after a migration: the old cell still holds the value
        // the migration read (8); rolled-back code finds it untouched.
        let prev = mgr.rollback("svc").expect("history");
        assert_eq!(prev, names1["bump"]);
        unsafe {
            assert_eq!(call_svc(&mgr, &jit, &rt, 2), 10, "pre-migration cell intact");
        }
    }

    /// A rebind whose new root has a different signature is refused —
    /// the binding's interface is pinned at creation.
    #[test]
    fn interface_mismatch_is_refused() {
        init();
        let ctx = Context::create();
        let (mut rt, mut jit) = fresh_node(&ctx);
        let mut mgr = DeployManager::default();

        let (rm1, names1) = resolve_src(V1);
        let req = DeployRequest::from_module(&rm1, vec![("svc".to_string(), names1["bump"])]);
        apply_deploy(&mut mgr, &mut rt, &mut jit, req).expect("v1 deploys");

        // Same state def (same hash -> Keep); incompatible root.
        let v3 = "
            state counter: Atom<Int> = atom_new(0)
            def bump2(a: Int, b: Int) -> Int = a + b
        ";
        let (rm3, names3) = resolve_src(v3);
        assert_eq!(names1["counter"], names3["counter"], "identical state def, same hash");
        let req = DeployRequest::from_module(&rm3, vec![("svc".to_string(), names3["bump2"])]);
        match apply_deploy(&mut mgr, &mut rt, &mut jit, req) {
            Err(DeployError::InterfaceMismatch { binding, .. }) => assert_eq!(binding, "svc"),
            other => panic!("expected interface refusal, got {:?}", other.map(|_| ())),
        }
        assert_eq!(mgr.resolve("svc"), Some(names1["bump"]), "binding unchanged");
    }

    /// Dropping a live state requires explicit consent; the orphaned cell
    /// stays resident (hash-keyed), so redeploying the old version later
    /// revives the same cell with its data (`Keep`).
    #[test]
    fn state_drop_is_guarded_and_revival_keeps_data() {
        init();
        let ctx = Context::create();
        let (mut rt, mut jit) = fresh_node(&ctx);
        let mut mgr = DeployManager::default();

        let (rm1, names1) = resolve_src(V1);
        let req = DeployRequest::from_module(&rm1, vec![("svc".to_string(), names1["bump"])]);
        apply_deploy(&mut mgr, &mut rt, &mut jit, req).expect("v1 deploys");
        unsafe {
            assert_eq!(call_svc(&mgr, &jit, &rt, 5), 5);
            assert_eq!(call_svc(&mgr, &jit, &rt, 10), 15);
        }

        // v4 has no `counter`: refused without the flag.
        let v4 = "def bump(d: Int) -> Int = d";
        let (rm4, names4) = resolve_src(v4);
        let req = DeployRequest::from_module(&rm4, vec![("svc".to_string(), names4["bump"])]);
        match apply_deploy(&mut mgr, &mut rt, &mut jit, req) {
            Err(DeployError::StateDropped { name }) => assert_eq!(name, "counter"),
            other => panic!("expected drop refusal, got {:?}", other.map(|_| ())),
        }

        // With consent the deploy lands and the node stops tracking it.
        let mut req = DeployRequest::from_module(&rm4, vec![("svc".to_string(), names4["bump"])]);
        req.allow_state_drop = true;
        let report = apply_deploy(&mut mgr, &mut rt, &mut jit, req).expect("drop deploy");
        assert_eq!(report.dropped_states, vec!["counter".to_string()]);
        assert!(mgr.live_state("counter").is_none());
        unsafe {
            assert_eq!(call_svc(&mgr, &jit, &rt, 4), 4, "stateless v4 live");
        }

        // Redeploy v1: its state hash still has a live cell -> Keep, and
        // the data is exactly where the drop left it.
        let req = DeployRequest::from_module(&rm1, vec![("svc".to_string(), names1["bump"])]);
        let report = apply_deploy(&mut mgr, &mut rt, &mut jit, req).expect("revival deploy");
        assert_eq!(*action_for(&report, "counter"), StateAction::Keep);
        unsafe {
            assert_eq!(call_svc(&mgr, &jit, &rt, 4), 19, "revived cell kept its value");
        }
    }

    /// The whole feature over the wire: a node serving on a channel, a
    /// RUNTIME-LESS client (pure bytes — exactly what the CLI is) doing
    /// deploy -> invoke -> redeploy (carryover) -> refused deploy ->
    /// rollback, observing live behavior flips with state preserved.
    #[test]
    fn wire_deploy_invoke_rollback_end_to_end() {
        init();
        let (mut client, mut server_chan) = crate::net::InProcessChannel::pair();

        let server = std::thread::spawn(move || {
            init();
            let ctx = Context::create();
            let (mut rt, mut jit) = fresh_node(&ctx);
            let mut mgr = DeployManager::default();
            loop {
                match unsafe {
                    serve_deploy_turn(&mut rt, &mut jit, &mut mgr, &mut server_chan)
                } {
                    Ok(()) => continue,
                    // Client dropped its end — clean shutdown.
                    Err(NetError::Io(_)) | Err(NetError::ConnectionClosed) => break,
                    Err(e) => panic!("server error: {}", e),
                }
            }
        });

        // ---- v1: deploy, bind, invoke ----
        let (rm1, names1) = resolve_src(V1);
        let req = DeployRequest::from_module(&rm1, vec![("svc".to_string(), names1["bump"])]);
        let report = deploy_on_channel(&mut client, &req).expect("v1 deploys over the wire");
        assert_eq!(*action_for(&report, "counter"), StateAction::Fresh);
        assert_eq!(report.rebound, vec![("svc".to_string(), names1["bump"], None)]);

        assert_eq!(invoke_on_channel(&mut client, "svc", &[5]).unwrap(), 5);
        assert_eq!(invoke_on_channel(&mut client, "svc", &[10]).unwrap(), 15);

        // ---- v2: carryover deploy flips behavior, keeps state ----
        let v2 = "
            state counter: Atom<Int> = atom_new(100)
            def bump(d: Int) -> Int = atom_swap(counter, |n: Int| n + d + d)
        ";
        let (rm2, names2) = resolve_src(v2);
        let req = DeployRequest::from_module(&rm2, vec![("svc".to_string(), names2["bump"])]);
        let report = deploy_on_channel(&mut client, &req).expect("v2 deploys over the wire");
        assert_eq!(
            *action_for(&report, "counter"),
            StateAction::Carryover { from: names1["counter"] }
        );
        assert_eq!(
            invoke_on_channel(&mut client, "svc", &[1]).unwrap(),
            17,
            "new behavior over carried-over state"
        );

        // ---- refused deploy: node untouched, connection stays usable ----
        let v3 = "
            state counter: Int = 0
            def bump(d: Int) -> Int = counter + d
        ";
        let (rm3, names3) = resolve_src(v3);
        let req = DeployRequest::from_module(&rm3, vec![("svc".to_string(), names3["bump"])]);
        match deploy_on_channel(&mut client, &req) {
            Err(NetError::DeployRefused(msg)) => {
                assert!(msg.contains("changed shape"), "got: {}", msg)
            }
            other => panic!("expected refusal, got {:?}", other.map(|_| ())),
        }
        assert_eq!(
            invoke_on_channel(&mut client, "svc", &[1]).unwrap(),
            19,
            "v2 still live after refusal"
        );

        // ---- bad invokes are refusals, not crashes ----
        match invoke_on_channel(&mut client, "nope", &[1]) {
            Err(NetError::DeployRefused(_)) => {}
            other => panic!("expected unknown-binding refusal, got {:?}", other),
        }
        match invoke_on_channel(&mut client, "svc", &[1, 2]) {
            Err(NetError::DeployRefused(msg)) => {
                assert!(msg.contains("arity"), "got: {}", msg)
            }
            other => panic!("expected arity refusal, got {:?}", other),
        }

        // ---- rollback: instant flip back to v1 over the wire ----
        let report = rollback_on_channel(&mut client, "svc").expect("rollback");
        assert_eq!(
            report.rebound,
            vec![("svc".to_string(), names1["bump"], Some(names2["bump"]))]
        );
        assert_eq!(
            invoke_on_channel(&mut client, "svc", &[2]).unwrap(),
            21,
            "v1 behavior again (adds once), same shared cell"
        );

        drop(client);
        server.join().expect("server thread");
    }

    /// Shipped code that doesn't typecheck is refused before any
    /// mutation — the node-side union typecheck is the gate.
    #[test]
    fn ill_typed_deploy_is_refused() {
        init();
        let ctx = Context::create();
        let (mut rt, mut jit) = fresh_node(&ctx);
        let mut mgr = DeployManager::default();

        // Resolves fine (names + hashes exist) but the body's type is
        // wrong: declared `-> Int`, returns `Atom<Int>`.
        let bad = "def bad(d: Int) -> Int = atom_new(d)";
        let (rm, names) = resolve_src(bad);
        let req = DeployRequest::from_module(&rm, vec![("svc".to_string(), names["bad"])]);
        match apply_deploy(&mut mgr, &mut rt, &mut jit, req) {
            Err(DeployError::Typecheck(_)) => {}
            other => panic!("expected typecheck refusal, got {:?}", other.map(|_| ())),
        }
        assert_eq!(mgr.resolve("svc"), None, "no binding was created");
    }
}
