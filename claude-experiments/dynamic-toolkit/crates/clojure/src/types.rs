//! Heap object type declarations.
//!
//! Every Clojure value that lives on the heap is an `ObjType` declared
//! here. The toolkit's `dynobj::field` reads/writes Value fields via
//! `AtomicU64::load(Relaxed)` / `store(Relaxed)` already, so a plain
//! `FieldKind::Value` is the right shape for both immutable identity
//! fields (`Var.ns`, `Var.sym`) and atomic-mutated fields
//! (`Var.root`, `Namespace.mappings`). The atomicity comes from the
//! storage representation; the access mode is the access ordering.
//!
//! CAS on a Value field is provided by an extern (see `externs.rs`).
//! It is only used at compile-time-ish operations (intern,
//! alter-var-root, with-redefs, registry insertion), never on the hot
//! path of a Var deref.
//!
//! ## Type IDs
//!
//! Type IDs are assigned in declaration order by `dynlang`. The
//! `Types` struct returned from [`declare_types`] hands back the
//! [`ObjTypeId`]s so other modules can refer to the types without
//! recomputing the order.

use dynlang::{DynModule, FieldKind, ObjTypeId};

/// Handles to every Clojure heap type, returned by [`declare_types`].
///
/// Hold this alongside your [`DynModule`] for the lifetime of the
/// language session. Field access in IR uses these IDs.
pub struct Types {
    pub symbol: ObjTypeId,
    pub keyword: ObjTypeId,
    pub string: ObjTypeId,
    pub list: ObjTypeId,
    pub vector: ObjTypeId,
    /// Native mutable array — the primitive on which user-space
    /// persistent collections (PersistentVector, PersistentHashMap,
    /// etc.) are built in `core.clj`. Layout is pure
    /// `varlen_values`. Also used internally as the storage for the
    /// reader's transient Vector type.
    pub array: ObjTypeId,
    pub map: ObjTypeId,
    pub set: ObjTypeId,
    pub fn_obj: ObjTypeId,
    pub var: ObjTypeId,
    pub namespace: ObjTypeId,
    pub registry: ObjTypeId,
}

/// Declare all Clojure heap types on `dm`, returning their IDs.
///
/// Call this exactly once, before any function declarations. The
/// returned [`Types`] struct must outlive the [`DynModule`].
pub fn declare_types(dm: &mut DynModule) -> Types {
    // ── Symbol: immutable. ────────────────────────────────────────
    // `ns` is an Optional<String> represented as a String pointer or
    // nil. `name` is always a String pointer. Hash is cached for
    // map lookups (Clojure's PersistentHashMap relies on it heavily).
    let symbol = dm
        .obj_type("Symbol")
        .field("ns", FieldKind::Value)
        .field("name", FieldKind::Value)
        .field("hash", FieldKind::Raw64)
        .build();

    // ── Keyword: immutable, interned. ─────────────────────────────
    // Backed by a Symbol (so `:foo` and `'foo` share the textual
    // representation but have distinct identity tags). All keywords
    // are interned via the registry-side keyword table.
    let keyword = dm
        .obj_type("Keyword")
        .field("sym", FieldKind::Value)
        .field("hash", FieldKind::Raw64)
        .build();

    // ── String: immutable bytes. ──────────────────────────────────
    let string = dm
        .obj_type("String")
        .field("hash", FieldKind::Raw64)
        .varlen_bytes()
        .build();

    // ── List (cons cell): immutable. ──────────────────────────────
    // `count` is precomputed so `count` on a list is O(1).
    let list = dm
        .obj_type("List")
        .field("first", FieldKind::Value)
        .field("rest", FieldKind::Value)
        .field("count", FieldKind::Raw64)
        .build();

    // ── Vector node (HAMT/RRB): immutable per-node. ───────────────
    // Stored as a tree of Value-array nodes; the count and shift live
    // in the root. The varlen-values storage holds either child
    // pointers (interior) or values (leaf).
    let vector = dm
        .obj_type("Vector")
        .field("count", FieldKind::Raw64)
        .field("shift", FieldKind::Raw64)
        .field("root", FieldKind::Value)
        .field("tail", FieldKind::Value)
        .build();

    // ── Array: native mutable array. Pure varlen_values.
    //
    // Shared between user-callable `(make-array N)` / `aget` / `aset!`
    // and the reader's transient Vector storage — same layout, same
    // type-id. core.clj treats it as the primitive backing store for
    // its PersistentVector / PersistentHashMap implementations.
    let array = dm
        .obj_type("Array")
        .varlen_values()
        .build();

    // ── Map: v1 is a flat array of [k0, v0, k1, v1, …] entries.
    // `count` stores the entry count (= varlen_count / 2). HAMT
    // replaces this layout once user-facing {} literals demand it.
    let map = dm
        .obj_type("Map")
        .field("count", FieldKind::Raw64)
        .varlen_values()
        .build();

    let set = dm
        .obj_type("Set")
        .field("backing", FieldKind::Value)
        .build();

    // ── Function / closure. ───────────────────────────────────────
    // `func_ref` stores the FuncRef as a Raw64 (it's a u32, but kept
    // 64-bit aligned for simplicity). The varlen-values section
    // captures the closure's environment.
    let fn_obj = dm
        .obj_type("Fn")
        .field("func_ref", FieldKind::Raw64)
        .field("arity", FieldKind::Raw64)
        .varlen_values()
        .build();

    // ── Var. ──────────────────────────────────────────────────────
    // `ns` and `sym` are immutable identity. `root` and `meta` are
    // CAS-mutated (alter-var-root, alter-meta!). `flags` packs
    // boolean attributes (:dynamic, :macro, :private, :bound) into
    // a Raw64 — these are effectively set-once at `def` time, so
    // non-atomic Raw64 access is acceptable.
    let var = dm
        .obj_type("Var")
        .field("ns", FieldKind::Value)
        .field("sym", FieldKind::Value)
        .field("root", FieldKind::Value)
        .field("meta", FieldKind::Value)
        .field("flags", FieldKind::Raw64)
        .build();

    // ── Namespace. ────────────────────────────────────────────────
    // `name` immutable. `mappings`, `aliases`, `meta` are all CAS-
    // updated pointer-to-persistent-map. `version` bumps on every
    // mappings/aliases mutation; the Property IC uses it to
    // invalidate cached var resolutions.
    let namespace = dm
        .obj_type("Namespace")
        .field("name", FieldKind::Value)
        .field("mappings", FieldKind::Value)
        .field("aliases", FieldKind::Value)
        .field("meta", FieldKind::Value)
        .field("version", FieldKind::Raw64)
        .build();

    // ── Registry: singleton, holds the global namespace table. ────
    // CAS-updated on `create-ns` / `remove-ns`.
    let registry = dm
        .obj_type("Registry")
        .field("namespaces", FieldKind::Value)
        .build();

    Types {
        symbol,
        keyword,
        string,
        list,
        vector,
        array,
        map,
        set,
        fn_obj,
        var,
        namespace,
        registry,
    }
}

// ── Field index constants ───────────────────────────────────────────
//
// These mirror the declaration order above. They're used by the
// CAS extern (which takes a field index, not a name) and by any
// IR-side code that needs to bypass the field-name hashmap.
//
// If you reorder the fields in `declare_types`, update these too —
// no static check ties them together yet.

pub mod var {
    pub const FIELD_NS: u16 = 0;
    pub const FIELD_SYM: u16 = 1;
    pub const FIELD_ROOT: u16 = 2;
    pub const FIELD_META: u16 = 3;

    /// Bit positions in the `flags` Raw64 field.
    pub const FLAG_DYNAMIC: u64 = 1 << 0;
    pub const FLAG_MACRO: u64 = 1 << 1;
    pub const FLAG_PRIVATE: u64 = 1 << 2;
    pub const FLAG_BOUND: u64 = 1 << 3;
}

pub mod namespace {
    pub const FIELD_NAME: u16 = 0;
    pub const FIELD_MAPPINGS: u16 = 1;
    pub const FIELD_ALIASES: u16 = 2;
    pub const FIELD_META: u16 = 3;
}

pub mod registry {
    pub const FIELD_NAMESPACES: u16 = 0;
}
