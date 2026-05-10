//! Heap-object allocators and accessors for the namespace system.
//!
//! Layouts (mirror `types.rs` declaration order):
//!
//! ```text
//!   Namespace { name: Value, mappings: Value, aliases: Value,
//!               meta: Value, version: Raw64 }
//!   Var       { ns: Value, sym: Value, root: Value, meta: Value,
//!               flags: Raw64 }
//!   Registry  { namespaces: Value }
//!   Map       { count: Raw64, varlen_values [k0, v0, k1, v1, ...] }
//!   Fn        { func_ref: Raw64, arity: Raw64,
//!               varlen_values (captured env) }
//! ```
//!
//! All `_alloc` helpers root the result in the supplied `RootScope`
//! and root any pointer-typed inputs internally for the duration of
//! the allocation.

use dynobj::roots::{Rooted, RootScope};
use dynobj::{Compact, ObjHeader};

use crate::host::with_host;
use crate::value::{self as v, NanBoxTag};

// ── Field-offset constants ──────────────────────────────────────────
//
// Mirrors declaration order in `types.rs`. If you reorder fields
// there, update these. Header is `Compact::SIZE` (= 16).

const HDR: usize = Compact::SIZE;

// Map: count: Raw64 (8B) → then varlen.
const MAP_COUNT_OFFSET: usize = HDR;
const MAP_VARLEN_COUNT_OFFSET: usize = HDR + 8;
const MAP_VARLEN_ELEM_BASE: usize = HDR + 16;

// Namespace: name(V) + mappings(V) + aliases(V) + meta(V) + version(R64).
const NS_NAME_OFFSET: usize = HDR;
const NS_MAPPINGS_OFFSET: usize = HDR + 8;
const NS_ALIASES_OFFSET: usize = HDR + 16;
const NS_META_OFFSET: usize = HDR + 24;
const NS_VERSION_OFFSET: usize = HDR + 32;

// Var: ns(V) + sym(V) + root(V) + meta(V) + flags(R64).
const VAR_NS_OFFSET: usize = HDR;
const VAR_SYM_OFFSET: usize = HDR + 8;
const VAR_ROOT_OFFSET: usize = HDR + 16;
const VAR_META_OFFSET: usize = HDR + 24;
const VAR_FLAGS_OFFSET: usize = HDR + 32;

// Registry: namespaces(V).
const REG_NAMESPACES_OFFSET: usize = HDR;

// Fn: func_ref(R64) + arity(R64) → then varlen env.
const FN_FUNCREF_OFFSET: usize = HDR;
const FN_ARITY_OFFSET: usize = HDR + 8;

// ── Node (vector backing) ───────────────────────────────────────────
//
// Layout: pure varlen-values (no fixed fields).
//   varlen_count at HDR
//   first element at HDR + 8

const NODE_VARLEN_COUNT_OFFSET: usize = HDR;
const NODE_ELEM_BASE: usize = HDR + 8;

pub fn alloc_node_values<'scope>(
    scope: &'scope RootScope<'_>,
    items: &[u64],
) -> Rooted<'scope, NanBoxTag> {
    with_host(|h| {
        let gc = unsafe { &*h.gc };
        let type_id = h.types.node.0;
        let raw = gc.alloc(type_id, items.len());
        assert!(!raw.is_null(), "alloc_node_values: GC alloc returned null");
        unsafe {
            (raw.add(NODE_VARLEN_COUNT_OFFSET) as *mut u64).write(items.len() as u64);
            for (i, x) in items.iter().enumerate() {
                (raw.add(NODE_ELEM_BASE + i * 8) as *mut u64).write(*x);
            }
        }
        scope.root::<NanBoxTag>(gc.tag_ptr(raw))
    })
}

pub fn node_count(node: u64) -> usize {
    let p = v::as_ptr(node);
    unsafe { (p.add(NODE_VARLEN_COUNT_OFFSET) as *const u64).read() as usize }
}

pub fn node_get(node: u64, i: usize) -> u64 {
    let p = v::as_ptr(node);
    debug_assert!(i < node_count(node), "node_get: out of bounds");
    unsafe { (p.add(NODE_ELEM_BASE + i * 8) as *const u64).read() }
}

// ── Map allocators / accessors ──────────────────────────────────────

/// Allocate an empty Map.
pub fn alloc_map_empty<'scope>(scope: &'scope RootScope<'_>) -> Rooted<'scope, NanBoxTag> {
    alloc_map_pairs(scope, &[])
}

/// Allocate a Map with the given (key, value) pairs. Keys/values must
/// already be rooted by the caller (we do not push them ourselves; the
/// caller owns lifetime). The freshly-allocated Map is rooted in
/// `scope` and returned.
pub fn alloc_map_pairs<'scope>(
    scope: &'scope RootScope<'_>,
    pairs: &[(u64, u64)],
) -> Rooted<'scope, NanBoxTag> {
    with_host(|h| {
        let gc = unsafe { &*h.gc };
        let type_id = h.types.map.0;
        let varlen_len = pairs.len() * 2;
        let raw = gc.alloc(type_id, varlen_len);
        assert!(!raw.is_null(), "alloc_map_pairs: GC alloc returned null");
        unsafe {
            // count field (entry count, NOT slot count)
            (raw.add(MAP_COUNT_OFFSET) as *mut u64).write(pairs.len() as u64);
            // varlen count word (slot count)
            (raw.add(MAP_VARLEN_COUNT_OFFSET) as *mut u64).write(varlen_len as u64);
            // entries
            for (i, (k, vv)) in pairs.iter().enumerate() {
                let kp = raw.add(MAP_VARLEN_ELEM_BASE + i * 16) as *mut u64;
                let vp = raw.add(MAP_VARLEN_ELEM_BASE + i * 16 + 8) as *mut u64;
                kp.write(*k);
                vp.write(*vv);
            }
        }
        scope.root::<NanBoxTag>(gc.tag_ptr(raw))
    })
}

/// Read the entry count of a Map. Caller must verify v is a Map.
pub fn map_count(map: u64) -> u64 {
    debug_assert!(v::is_ptr(map));
    let p = v::as_ptr(map);
    unsafe { (p.add(MAP_COUNT_OFFSET) as *const u64).read() }
}

/// Read the i-th key/value pair from a Map.
pub fn map_entry(map: u64, i: usize) -> (u64, u64) {
    debug_assert!(v::is_ptr(map));
    let p = v::as_ptr(map);
    unsafe {
        let k = (p.add(MAP_VARLEN_ELEM_BASE + i * 16) as *const u64).read();
        let v = (p.add(MAP_VARLEN_ELEM_BASE + i * 16 + 8) as *const u64).read();
        (k, v)
    }
}

/// Linear-scan lookup. Returns NIL when not found. Equality on
/// keys: bit-wise (works for sym-ids, ints, nil, bool — since they
/// have unique bit patterns).
pub fn map_get(map: u64, key: u64) -> u64 {
    let n = map_count(map) as usize;
    for i in 0..n {
        let (k, val) = map_entry(map, i);
        if k == key {
            return val;
        }
    }
    v::NIL
}

/// Return a fresh Map with `(key, val)` added (or replacing the
/// existing entry for `key`). Roots inputs internally for the
/// duration of the allocation.
pub fn map_assoc<'scope>(
    scope: &'scope RootScope<'_>,
    map: u64,
    key: u64,
    val: u64,
) -> Rooted<'scope, NanBoxTag> {
    let map_r = scope.root::<NanBoxTag>(map);
    let key_r = scope.root::<NanBoxTag>(key);
    let val_r = scope.root::<NanBoxTag>(val);

    // Decide whether `key` already exists.
    let n = map_count(map_r.get()) as usize;
    let mut found_at: Option<usize> = None;
    for i in 0..n {
        let (k, _) = map_entry(map_r.get(), i);
        if k == key_r.get() {
            found_at = Some(i);
            break;
        }
    }

    let new_pairs: Vec<(u64, u64)> = if let Some(idx) = found_at {
        (0..n)
            .map(|i| {
                if i == idx {
                    (key_r.get(), val_r.get())
                } else {
                    map_entry(map_r.get(), i)
                }
            })
            .collect()
    } else {
        let mut out: Vec<(u64, u64)> = (0..n).map(|i| map_entry(map_r.get(), i)).collect();
        out.push((key_r.get(), val_r.get()));
        out
    };
    alloc_map_pairs(scope, &new_pairs)
}

// ── Namespace ──────────────────────────────────────────────────────

pub fn alloc_namespace<'scope>(
    scope: &'scope RootScope<'_>,
    name_sym: u64,
) -> Rooted<'scope, NanBoxTag> {
    let name_r = scope.root::<NanBoxTag>(name_sym);
    let mappings = alloc_map_empty(scope);
    let aliases = alloc_map_empty(scope);
    let meta = alloc_map_empty(scope);
    with_host(|h| {
        let gc = unsafe { &*h.gc };
        let type_id = h.types.namespace.0;
        let raw = gc.alloc(type_id, 0);
        assert!(!raw.is_null(), "alloc_namespace: GC alloc returned null");
        unsafe {
            (raw.add(NS_NAME_OFFSET) as *mut u64).write(name_r.get());
            (raw.add(NS_MAPPINGS_OFFSET) as *mut u64).write(mappings.get());
            (raw.add(NS_ALIASES_OFFSET) as *mut u64).write(aliases.get());
            (raw.add(NS_META_OFFSET) as *mut u64).write(meta.get());
            (raw.add(NS_VERSION_OFFSET) as *mut u64).write(0);
        }
        scope.root::<NanBoxTag>(gc.tag_ptr(raw))
    })
}

pub fn ns_name(ns: u64) -> u64 {
    let p = v::as_ptr(ns);
    unsafe { (p.add(NS_NAME_OFFSET) as *const u64).read() }
}

pub fn ns_mappings(ns: u64) -> u64 {
    let p = v::as_ptr(ns);
    unsafe { (p.add(NS_MAPPINGS_OFFSET) as *const u64).read() }
}

pub fn ns_set_mappings(ns: u64, new_map: u64) {
    let p = v::as_ptr(ns);
    unsafe { (p.add(NS_MAPPINGS_OFFSET) as *mut u64).write(new_map) }
}

pub fn ns_bump_version(ns: u64) {
    let p = v::as_ptr(ns);
    unsafe {
        let cur = (p.add(NS_VERSION_OFFSET) as *const u64).read();
        (p.add(NS_VERSION_OFFSET) as *mut u64).write(cur + 1);
    }
}

/// Look up a Var in a namespace by symbol-id key. Returns NIL if not
/// present.
pub fn ns_lookup(ns: u64, sym: u64) -> u64 {
    map_get(ns_mappings(ns), sym)
}

/// Find-or-create a Var bound to (sym → root) in the namespace.
/// Returns the Var's tagged pointer, rooted in `scope`. The
/// namespace's mappings field is updated to the new Map containing
/// the fresh / updated entry.
pub fn ns_intern<'scope>(
    scope: &'scope RootScope<'_>,
    ns: u64,
    sym: u64,
    root: u64,
) -> Rooted<'scope, NanBoxTag> {
    let ns_r = scope.root::<NanBoxTag>(ns);
    let sym_r = scope.root::<NanBoxTag>(sym);

    // If a Var for `sym` already exists, update its root in place
    // (preserving identity for any cached resolutions).
    let existing = ns_lookup(ns_r.get(), sym_r.get());
    if !v::is_nil(existing) {
        var_set_root(existing, root);
        return scope.root::<NanBoxTag>(existing);
    }

    // Else allocate a fresh Var and add it to the mappings.
    let var = alloc_var(scope, ns_r.get(), sym_r.get(), root);
    let new_map = map_assoc(scope, ns_mappings(ns_r.get()), sym_r.get(), var.get());
    ns_set_mappings(ns_r.get(), new_map.get());
    ns_bump_version(ns_r.get());
    var
}

// ── Var ─────────────────────────────────────────────────────────────

pub fn alloc_var<'scope>(
    scope: &'scope RootScope<'_>,
    ns: u64,
    sym: u64,
    root: u64,
) -> Rooted<'scope, NanBoxTag> {
    let ns_r = scope.root::<NanBoxTag>(ns);
    let sym_r = scope.root::<NanBoxTag>(sym);
    let root_r = scope.root::<NanBoxTag>(root);
    with_host(|h| {
        let gc = unsafe { &*h.gc };
        let type_id = h.types.var.0;
        let raw = gc.alloc(type_id, 0);
        assert!(!raw.is_null(), "alloc_var: GC alloc returned null");
        unsafe {
            (raw.add(VAR_NS_OFFSET) as *mut u64).write(ns_r.get());
            (raw.add(VAR_SYM_OFFSET) as *mut u64).write(sym_r.get());
            (raw.add(VAR_ROOT_OFFSET) as *mut u64).write(root_r.get());
            (raw.add(VAR_META_OFFSET) as *mut u64).write(v::NIL);
            (raw.add(VAR_FLAGS_OFFSET) as *mut u64).write(0);
        }
        scope.root::<NanBoxTag>(gc.tag_ptr(raw))
    })
}

pub fn var_root(var: u64) -> u64 {
    let p = v::as_ptr(var);
    unsafe { (p.add(VAR_ROOT_OFFSET) as *const u64).read() }
}

pub fn var_set_root(var: u64, root: u64) {
    let p = v::as_ptr(var);
    unsafe { (p.add(VAR_ROOT_OFFSET) as *mut u64).write(root) }
}

pub fn var_sym(var: u64) -> u64 {
    let p = v::as_ptr(var);
    unsafe { (p.add(VAR_SYM_OFFSET) as *const u64).read() }
}

// Var flag bits (mirror `types::var::FLAG_*`).
pub const FLAG_DYNAMIC: u64 = 1 << 0;
pub const FLAG_MACRO: u64 = 1 << 1;
pub const FLAG_PRIVATE: u64 = 1 << 2;
pub const FLAG_BOUND: u64 = 1 << 3;

pub fn var_flags(var: u64) -> u64 {
    let p = v::as_ptr(var);
    unsafe { (p.add(VAR_FLAGS_OFFSET) as *const u64).read() }
}

pub fn var_set_flag(var: u64, flag: u64) {
    let p = v::as_ptr(var);
    unsafe {
        let cur = (p.add(VAR_FLAGS_OFFSET) as *const u64).read();
        (p.add(VAR_FLAGS_OFFSET) as *mut u64).write(cur | flag);
    }
}

pub fn var_is_macro(var: u64) -> bool {
    var_flags(var) & FLAG_MACRO != 0
}

// ── Registry ────────────────────────────────────────────────────────

pub fn alloc_registry<'scope>(scope: &'scope RootScope<'_>) -> Rooted<'scope, NanBoxTag> {
    let nss = alloc_map_empty(scope);
    with_host(|h| {
        let gc = unsafe { &*h.gc };
        let type_id = h.types.registry.0;
        let raw = gc.alloc(type_id, 0);
        assert!(!raw.is_null(), "alloc_registry: GC alloc returned null");
        unsafe {
            (raw.add(REG_NAMESPACES_OFFSET) as *mut u64).write(nss.get());
        }
        scope.root::<NanBoxTag>(gc.tag_ptr(raw))
    })
}

pub fn registry_namespaces(reg: u64) -> u64 {
    let p = v::as_ptr(reg);
    unsafe { (p.add(REG_NAMESPACES_OFFSET) as *const u64).read() }
}

pub fn registry_set_namespaces(reg: u64, m: u64) {
    let p = v::as_ptr(reg);
    unsafe { (p.add(REG_NAMESPACES_OFFSET) as *mut u64).write(m) }
}

pub fn registry_create_ns<'scope>(
    scope: &'scope RootScope<'_>,
    reg: u64,
    name_sym: u64,
) -> Rooted<'scope, NanBoxTag> {
    let reg_r = scope.root::<NanBoxTag>(reg);
    let sym_r = scope.root::<NanBoxTag>(name_sym);
    let existing = map_get(registry_namespaces(reg_r.get()), sym_r.get());
    if !v::is_nil(existing) {
        return scope.root::<NanBoxTag>(existing);
    }
    let ns = alloc_namespace(scope, sym_r.get());
    let new_map = map_assoc(scope, registry_namespaces(reg_r.get()), sym_r.get(), ns.get());
    registry_set_namespaces(reg_r.get(), new_map.get());
    ns
}

pub fn registry_find_ns(reg: u64, name_sym: u64) -> u64 {
    map_get(registry_namespaces(reg), name_sym)
}

// ── Fn ──────────────────────────────────────────────────────────────

/// Allocate an Fn heap object holding a FuncRef and an empty captured
/// env. (Closures with non-empty env arrive when first-class fn values
/// land.)
pub fn alloc_fn<'scope>(
    scope: &'scope RootScope<'_>,
    func_ref: u32,
    arity: usize,
) -> Rooted<'scope, NanBoxTag> {
    with_host(|h| {
        let gc = unsafe { &*h.gc };
        let type_id = h.types.fn_obj.0;
        let raw = gc.alloc(type_id, 0);
        assert!(!raw.is_null(), "alloc_fn: GC alloc returned null");
        unsafe {
            (raw.add(FN_FUNCREF_OFFSET) as *mut u64).write(func_ref as u64);
            (raw.add(FN_ARITY_OFFSET) as *mut u64).write(arity as u64);
            // varlen count = 0
            // The varlen count word lives after FN_ARITY_OFFSET + 8.
            // For our checkpoint there's no captured env, so write 0.
            (raw.add(FN_ARITY_OFFSET + 8) as *mut u64).write(0);
        }
        scope.root::<NanBoxTag>(gc.tag_ptr(raw))
    })
}

pub fn fn_func_ref(fn_obj: u64) -> u32 {
    let p = v::as_ptr(fn_obj);
    unsafe { (p.add(FN_FUNCREF_OFFSET) as *const u64).read() as u32 }
}
