//! User-defined types and protocols.
//!
//! Three process-global registries live here:
//!
//! 1. `USER_TYPES` — every `deftype`/`defrecord` allocates a fresh
//!    `UserTypeId` (a `u32`). Stores the type's namespaced name and its
//!    declared field order. Field-index resolution at analyze time reads
//!    from here.
//!
//! 2. `PROTOCOLS` — every `defprotocol` allocates a `ProtocolId` and one
//!    `ProtoMethodId` per declared method. Method ids are unique across
//!    all protocols (flat `u32` namespace).
//!
//! 3. `DISPATCH` — the `(LogicalTypeId, ProtoMethodId) → FnHandle`
//!    table populated by `extend-type` / `extend-protocol` / inline
//!    deftype impls. `cljvm_rt_protocol_dispatch` reads it on every
//!    protocol-method call.
//!
//! ## LogicalTypeId space
//!
//! A `u32` partitioned as follows so the dispatch table can key on a
//! single integer regardless of whether the receiver is a built-in or
//! a user type:
//!
//! * `0` .. `USER_TYPE_BASE` — built-in heap `ObjTypeId` values, plus
//!   reserved synthetic ids for primitives (`nil`, bool, double, fn).
//! * `USER_TYPE_BASE` ..     — user-allocated deftype ids (offset by
//!   `USER_TYPE_BASE` so the two spaces never collide).
//!
//! `effective_type_id(bits)` (in `runtime.rs`) maps a NanBox value to
//! its LogicalTypeId.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, RwLock};

use crate::lang::symbol::Symbol;

/// Disjoint union of built-in ObjTypeIds (low range) and user-allocated
/// deftype ids (high range, offset by `USER_TYPE_BASE`). Synthetic ids
/// for tag-encoded primitives live in `BUILTIN_*` constants.
pub type LogicalTypeId = u32;

/// Boundary between built-in (`ObjTypeId` + synthetic primitive ids) and
/// user-allocated (`deftype`) type ids. Chosen well above any realistic
/// built-in ObjTypeId count.
pub const USER_TYPE_BASE: u32 = 0x0001_0000;

// Synthetic LogicalTypeIds for tag-encoded NanBox primitives. These
// don't correspond to any ObjTypeId — they exist so the dispatch table
// can hold entries like `(BUILTIN_DOUBLE, ISeq.first.method_id) → fn`.
// Picked above ObjTypeId range but below USER_TYPE_BASE.
pub const BUILTIN_NIL: LogicalTypeId = 0x0000_FF00;
pub const BUILTIN_BOOL: LogicalTypeId = 0x0000_FF01;
pub const BUILTIN_DOUBLE: LogicalTypeId = 0x0000_FF02;
pub const BUILTIN_FN: LogicalTypeId = 0x0000_FF03;
/// Wildcard / `Object` fallback. Methods installed under this id apply
/// when no exact `(type_id, method_id)` entry exists.
pub const BUILTIN_OBJECT: LogicalTypeId = 0x0000_FF04;
/// Boxed `clojure.lang.Long` — Clojure's integer type, distinct from
/// `double`. A boxed Long is a TAG_PTR heap cell, but for protocol
/// dispatch / `extend-type` it gets this synthetic logical id (rather
/// than its raw `ObjTypeId`) so number protocols can target it the same
/// way `BUILTIN_DOUBLE` targets floats.
pub const BUILTIN_LONG: LogicalTypeId = 0x0000_FF05;

#[inline]
pub fn user_type_logical(user_type_id: u32) -> LogicalTypeId {
    USER_TYPE_BASE
        .checked_add(user_type_id)
        .expect("clojure-jvm: user_types: LogicalTypeId overflow — too many user types")
}

/// Base for host-class LogicalTypeIds (e.g. `java.util.Date`).
///
/// A host class named in `(extend-type java.util.Date P …)` has no value
/// representation in this runtime — no NanBox value's `effective_type_id`
/// ever returns a host-class id, because we have no host (JVM) instances.
/// So an impl registered against a host class loads correctly but only
/// dispatches if/when a runtime value reports its id. Today that means
/// such impls register but never fire, which is the honest state of the
/// runtime. `java.lang.Object` is special-cased to `BUILTIN_OBJECT` (the
/// catch-all bucket) in `resolve_extend_type_target`, so it is *not* an
/// id from this band.
///
/// The band sits below the `BUILTIN_*` synthetic ids (`0xFF00`) and above
/// any realistic heap `ObjTypeId` count.
pub const HOST_CLASS_BASE: LogicalTypeId = 0x0000_F000;
const HOST_CLASS_LIMIT: LogicalTypeId = BUILTIN_NIL; // 0xFF00

static HOST_CLASS_BY_NAME: LazyLock<RwLock<HashMap<String, LogicalTypeId>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));
static NEXT_HOST_CLASS_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(HOST_CLASS_BASE);

/// Intern a host-class name (e.g. `"java.util.Date"`) to a stable
/// `LogicalTypeId`, allocating one on first sight. Idempotent per name.
pub fn host_class_logical(name: &str) -> LogicalTypeId {
    if let Some(id) = HOST_CLASS_BY_NAME.read().unwrap().get(name).copied() {
        return id;
    }
    let mut map = HOST_CLASS_BY_NAME.write().unwrap();
    // Re-check under the write lock — another thread may have interned it
    // between the read above and acquiring the write lock.
    if let Some(id) = map.get(name).copied() {
        return id;
    }
    let id = NEXT_HOST_CLASS_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    assert!(
        id < HOST_CLASS_LIMIT,
        "clojure-jvm: host-class LogicalTypeId space exhausted at {id:#x}"
    );
    map.insert(name.to_string(), id);
    id
}

/// A user-allocated `deftype`/`defrecord` shape.
#[derive(Debug, Clone)]
pub struct UserTypeInfo {
    pub id: u32,
    /// Namespaced symbol — e.g. `cljs.core/PersistentVector`. Used for
    /// `print-method` and error messages.
    pub name: Arc<Symbol>,
    /// Field names in declaration order. Index into this vec is the
    /// slot offset baked into codegen for `(.-field-name inst)`.
    pub fields: Vec<Arc<Symbol>>,
}

/// One method declared on a protocol.
#[derive(Debug, Clone)]
pub struct ProtoMethod {
    pub id: u32,
    pub name: Arc<Symbol>,
    /// Declared arities. Empty means "no arities recorded yet" — accepted
    /// during `defprotocol` parsing; dispatch uses arity-tagged ids if
    /// multiple arities are supplied. v1 supports single arity only.
    pub arities: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct ProtocolInfo {
    pub id: u32,
    pub name: Arc<Symbol>,
    pub methods: Vec<ProtoMethod>,
}

// ── Registries ─────────────────────────────────────────────────────────

static USER_TYPES: RwLock<Vec<UserTypeInfo>> = RwLock::new(Vec::new());
static PROTOCOLS: RwLock<Vec<ProtocolInfo>> = RwLock::new(Vec::new());
/// Reverse map: protocol-name → ProtocolId. Lets `(extend-type Foo P …)`
/// look up `P` by symbol.
static PROTOCOL_BY_NAME: LazyLock<RwLock<HashMap<Arc<Symbol>, u32>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));
/// Reverse map: user-type-name → UserTypeId. Lets `(extend-type Foo …)`
/// when `Foo` is a user deftype look up the id.
static USER_TYPE_BY_NAME: LazyLock<RwLock<HashMap<Arc<Symbol>, u32>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// `(LogicalTypeId, ProtoMethodId) → FnHandle (NanBox u64 bits)`.
/// FnHandle is the raw NanBox u64 of the implementation fn — either a
/// TAG_FN direct handle or a TAG_PTR pointer to a `clojure.lang.Closure`
/// heap cell. The runtime dispatches by invoking it like any other fn.
static DISPATCH: LazyLock<RwLock<HashMap<(LogicalTypeId, u32), u64>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// Process-global allocator state. Counter values, not registry indices,
/// to keep ProtoMethodId allocation independent of ProtocolId ordering.
static NEXT_PROTO_METHOD_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1); // 0 reserved as "invalid"

// ── User-type API ──────────────────────────────────────────────────────

/// Allocate a fresh UserTypeId for a `deftype`/`defrecord`. Duplicate
/// names overwrite the prior entry's name→id mapping but allocate a
/// fresh id — Clojure's `deftype` semantics (a redefinition is a new
/// class, existing instances of the old class keep working).
pub fn register_user_type(name: Arc<Symbol>, fields: Vec<Arc<Symbol>>) -> u32 {
    let mut types = USER_TYPES.write().unwrap();
    let id = types.len() as u32;
    let info = UserTypeInfo {
        id,
        name: name.clone(),
        fields,
    };
    types.push(info);
    USER_TYPE_BY_NAME.write().unwrap().insert(name, id);
    id
}

pub fn user_type_info(id: u32) -> Option<UserTypeInfo> {
    USER_TYPES.read().unwrap().get(id as usize).cloned()
}

pub fn user_type_id_by_name(name: &Arc<Symbol>) -> Option<u32> {
    USER_TYPE_BY_NAME.read().unwrap().get(name).copied()
}

/// Look up `(field_name)` → field index for a user type, or `None`.
pub fn user_type_field_index(id: u32, field_name: &Symbol) -> Option<usize> {
    let types = USER_TYPES.read().unwrap();
    types
        .get(id as usize)
        .and_then(|t| t.fields.iter().position(|f| *f.as_ref() == *field_name))
}

// ── Protocol API ───────────────────────────────────────────────────────

pub fn register_protocol(
    name: Arc<Symbol>,
    method_specs: Vec<(Arc<Symbol>, Vec<usize>)>,
) -> (u32, Vec<u32>) {
    let mut protos = PROTOCOLS.write().unwrap();
    let id = protos.len() as u32;
    let mut method_ids = Vec::with_capacity(method_specs.len());
    let methods: Vec<ProtoMethod> = method_specs
        .into_iter()
        .map(|(mname, arities)| {
            let mid = NEXT_PROTO_METHOD_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            method_ids.push(mid);
            ProtoMethod {
                id: mid,
                name: mname,
                arities,
            }
        })
        .collect();
    let info = ProtocolInfo {
        id,
        name: name.clone(),
        methods,
    };
    protos.push(info);
    PROTOCOL_BY_NAME.write().unwrap().insert(name, id);
    (id, method_ids)
}

pub fn protocol_info(id: u32) -> Option<ProtocolInfo> {
    PROTOCOLS.read().unwrap().get(id as usize).cloned()
}

pub fn protocol_id_by_name(name: &Arc<Symbol>) -> Option<u32> {
    let map = PROTOCOL_BY_NAME.read().unwrap();
    // Exact match first (same ns + name).
    if let Some(id) = map.get(name).copied() {
        return Some(id);
    }
    // Namespace-tolerant fallback: a protocol defined in clojure.core as
    // `Inst` is referenced elsewhere as `clojure.core/Inst`, and vice
    // versa. Match on the bare name component, ignoring the namespace.
    // Protocol bare names are effectively unique in practice (a redefine
    // overwrites), so this can't pick the wrong one.
    let bare = name.get_name();
    map.iter()
        .find(|(k, _)| k.get_name() == bare)
        .map(|(_, id)| *id)
}

/// Find a protocol method's id by its (unqualified) method name. The
/// flat-namespace assumption: in real Clojure, multiple protocols may
/// declare the same method name, in which case dispatch is by which
/// protocol the var resolves to. We track method_id per protocol; this
/// helper takes the protocol explicitly.
pub fn protocol_method_id(proto_id: u32, method_name: &Symbol) -> Option<u32> {
    let protos = PROTOCOLS.read().unwrap();
    protos.get(proto_id as usize).and_then(|p| {
        p.methods
            .iter()
            .find(|m| *m.name.as_ref() == *method_name)
            .map(|m| m.id)
    })
}

// ── Dispatch table API ─────────────────────────────────────────────────

/// Install an impl. Overwrites any existing entry for `(type_id,
/// method_id)`. Matches Clojure's semantics: later `extend-type` wins.
pub fn install_impl(type_id: LogicalTypeId, method_id: u32, fn_bits: u64) {
    DISPATCH
        .write()
        .unwrap()
        .insert((type_id, method_id), fn_bits);
}

/// Look up an impl. Returns `None` if no entry exists for the exact
/// `(type_id, method_id)`. Callers should fall back to
/// `BUILTIN_OBJECT` on miss (the runtime dispatch entry point does
/// this).
pub fn lookup_impl(type_id: LogicalTypeId, method_id: u32) -> Option<u64> {
    DISPATCH.read().unwrap().get(&(type_id, method_id)).copied()
}

/// Test-only: clear all registries. Used by unit tests so they don't
/// leak state across runs. Not exposed outside this crate.
#[cfg(test)]
pub(crate) fn _reset_for_tests() {
    USER_TYPES.write().unwrap().clear();
    PROTOCOLS.write().unwrap().clear();
    DISPATCH.write().unwrap().clear();
    USER_TYPE_BY_NAME.write().unwrap().clear();
    PROTOCOL_BY_NAME.write().unwrap().clear();
    NEXT_PROTO_METHOD_ID.store(1, std::sync::atomic::Ordering::SeqCst);
}

/// Process-wide lock serializing every test that touches the global
/// user-type / protocol registries (here AND in `compiler.rs`). Without a
/// single shared lock, parallel tests interleave registrations and one
/// test's `_reset_for_tests()` wipes another's protocols mid-run (observed
/// as "protocol not registered" panics under `cargo test` default
/// parallelism). The registries are intentionally process-global (protocols
/// behave like JVM classes — visible to JIT code on any thread), so the
/// production design can't isolate them per-thread; tests serialize instead.
#[cfg(test)]
pub(crate) static REGISTRY_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Acquire [`REGISTRY_TEST_LOCK`] and reset the registries to a clean slate.
/// Every test that registers or queries user types / protocols must hold the
/// returned guard for its whole body. Recovers from poisoning so one failing
/// test doesn't wedge the rest.
#[cfg(test)]
pub(crate) fn registry_test_guard() -> std::sync::MutexGuard<'static, ()> {
    let g = REGISTRY_TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    _reset_for_tests();
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(name: &str) -> Arc<Symbol> {
        Symbol::intern_ns_name(None, name)
    }

    fn sym_ns(ns: &str, name: &str) -> Arc<Symbol> {
        Symbol::intern_ns_name(Some(ns), name)
    }

    // These tests share global state with the registry-touching tests in
    // `compiler.rs`; both serialize via the one crate-level lock so neither
    // resets the registry out from under the other.
    fn guard() -> std::sync::MutexGuard<'static, ()> {
        super::registry_test_guard()
    }

    #[test]
    fn user_type_register_and_field_index() {
        let _g = guard();
        let id = register_user_type(
            sym_ns("cljs.core", "PersistentVector"),
            vec![sym("count"), sym("shift"), sym("root"), sym("tail")],
        );
        assert_eq!(id, 0);
        let info = user_type_info(id).expect("registered");
        assert_eq!(info.fields.len(), 4);
        assert_eq!(user_type_field_index(id, &sym("shift")), Some(1));
        assert_eq!(user_type_field_index(id, &sym("missing")), None);
        assert_eq!(
            user_type_id_by_name(&sym_ns("cljs.core", "PersistentVector")),
            Some(0)
        );
    }

    #[test]
    fn user_type_ids_increment() {
        let _g = guard();
        let a = register_user_type(sym("A"), vec![]);
        let b = register_user_type(sym("B"), vec![]);
        let c = register_user_type(sym("C"), vec![]);
        assert_eq!((a, b, c), (0, 1, 2));
    }

    #[test]
    fn protocol_register_assigns_unique_method_ids() {
        let _g = guard();
        let (p1, m1) = register_protocol(
            sym_ns("cljs.core", "IFn"),
            vec![(sym("-invoke"), vec![1, 2])],
        );
        let (p2, m2) = register_protocol(
            sym_ns("cljs.core", "ISeq"),
            vec![(sym("-first"), vec![1]), (sym("-rest"), vec![1])],
        );
        assert_eq!(p1, 0);
        assert_eq!(p2, 1);
        assert_eq!(m1.len(), 1);
        assert_eq!(m2.len(), 2);
        // Method ids are globally unique (flat namespace across protocols).
        let all: std::collections::HashSet<u32> = m1.iter().chain(m2.iter()).copied().collect();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn protocol_method_id_lookup() {
        let _g = guard();
        let (pid, mids) = register_protocol(
            sym_ns("cljs.core", "ISeq"),
            vec![(sym("-first"), vec![1]), (sym("-rest"), vec![1])],
        );
        assert_eq!(protocol_method_id(pid, &sym("-first")), Some(mids[0]));
        assert_eq!(protocol_method_id(pid, &sym("-rest")), Some(mids[1]));
        assert_eq!(protocol_method_id(pid, &sym("missing")), None);
    }

    #[test]
    fn dispatch_install_and_lookup() {
        let _g = guard();
        let (_pid, mids) = register_protocol(sym("P"), vec![(sym("m"), vec![1])]);
        let uid = register_user_type(sym("Foo"), vec![]);
        let tid = user_type_logical(uid);
        // Initially absent.
        assert_eq!(lookup_impl(tid, mids[0]), None);
        // Install + read back.
        install_impl(tid, mids[0], 0xDEADBEEF);
        assert_eq!(lookup_impl(tid, mids[0]), Some(0xDEADBEEF));
        // Object fallback is a separate slot, not aliased.
        assert_eq!(lookup_impl(BUILTIN_OBJECT, mids[0]), None);
        install_impl(BUILTIN_OBJECT, mids[0], 0xCAFEBABE);
        assert_eq!(lookup_impl(BUILTIN_OBJECT, mids[0]), Some(0xCAFEBABE));
        // Exact slot still wins.
        assert_eq!(lookup_impl(tid, mids[0]), Some(0xDEADBEEF));
    }

    #[test]
    fn user_type_logical_does_not_collide_with_builtins() {
        let uid = 0u32;
        let logical = user_type_logical(uid);
        assert!(logical >= USER_TYPE_BASE);
        assert!(logical > BUILTIN_OBJECT);
        assert!(logical > BUILTIN_FN);
    }

    #[test]
    #[should_panic(expected = "LogicalTypeId overflow")]
    fn user_type_logical_overflow_panics() {
        // u32::MAX - USER_TYPE_BASE + 1 would overflow.
        let _ = user_type_logical(u32::MAX);
    }
}
