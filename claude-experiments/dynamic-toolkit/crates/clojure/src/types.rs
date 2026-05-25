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
    /// Uniform heap-object backing for every user-declared
    /// `(deftype* Name [fields])`. Layout:
    ///   `type_name: Value` (a symbol identifying the user type)
    ///   `varlen_values` (the user's fields, in declaration order)
    /// We don't dynamically register one `ObjType` per user type:
    /// that would require runtime extension of the GC's type table,
    /// which the toolkit doesn't currently support. Tagging
    /// instances with a `type_name` symbol gets the same observable
    /// behavior — `instance?` is a sym-id compare, `.-field` is a
    /// `varlen_values[i]` load — at the cost of slightly looser
    /// type identity in the heap walker.
    pub record: ObjTypeId,
    /// Mutable single-cell reference type. `(atom v)` allocates one
    /// of these; `@a` / `(deref a)` reads the cell; `(reset! a v')`
    /// stores. Concurrent mutation uses Relaxed atomic load/store
    /// today (single-threaded mutator); a future revision can
    /// graduate to a CAS-based path for proper `swap!`.
    pub atom: ObjTypeId,
}

/// Declare all Clojure heap types on `dyn_module`, returning their IDs.
///
/// Call this exactly once, before any function declarations. The
/// returned [`Types`] struct must outlive the [`DynModule`].
pub fn declare_types(dyn_module: &mut DynModule) -> Types {
    // ── Symbol: immutable. ────────────────────────────────────────
    // `ns` is an Optional<String> represented as a String pointer or
    // nil. `name` is always a String pointer. Hash is cached for
    // map lookups (Clojure's PersistentHashMap relies on it heavily).
    let symbol = dyn_module
        .obj_type("Symbol")
        .field("ns", FieldKind::Value)
        .field("name", FieldKind::Value)
        .field("hash", FieldKind::Raw64)
        .build();

    // ── Keyword: immutable, interned. ─────────────────────────────
    // Backed by a Symbol (so `:foo` and `'foo` share the textual
    // representation but have distinct identity tags). All keywords
    // are interned via the registry-side keyword table.
    let keyword = dyn_module
        .obj_type("Keyword")
        .field("sym", FieldKind::Value)
        .field("hash", FieldKind::Raw64)
        .build();

    // ── String: immutable bytes. ──────────────────────────────────
    let string = dyn_module
        .obj_type("String")
        .field("hash", FieldKind::Raw64)
        .varlen_bytes()
        .build();

    // ── List (cons cell): immutable. ──────────────────────────────
    // `count` is precomputed so `count` on a list is O(1).
    let list = dyn_module
        .obj_type("List")
        .field("first", FieldKind::Value)
        .field("rest", FieldKind::Value)
        .field("count", FieldKind::Raw64)
        .build();

    // ── Vector node (HAMT/RRB): immutable per-node. ───────────────
    // Stored as a tree of Value-array nodes; the count and shift live
    // in the root. The varlen-values storage holds either child
    // pointers (interior) or values (leaf).
    let vector = dyn_module
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
    let array = dyn_module
        .obj_type("Array")
        .varlen_values()
        .build();

    // ── Map: v1 is a flat array of [k0, v0, k1, v1, …] entries.
    // `count` stores the entry count (= varlen_count / 2). HAMT
    // replaces this layout once user-facing {} literals demand it.
    let map = dyn_module
        .obj_type("Map")
        .field("count", FieldKind::Raw64)
        .varlen_values()
        .build();

    let set = dyn_module
        .obj_type("Set")
        .field("backing", FieldKind::Value)
        .build();

    // ── Function / closure. ───────────────────────────────────────
    // `func_ref` stores the FuncRef as a Raw64 (it's a u32, but kept
    // 64-bit aligned for simplicity). The varlen-values section
    // captures the closure's environment.
    let fn_obj = dyn_module
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
    let var = dyn_module
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
    let namespace = dyn_module
        .obj_type("Namespace")
        .field("name", FieldKind::Value)
        .field("mappings", FieldKind::Value)
        .field("aliases", FieldKind::Value)
        .field("meta", FieldKind::Value)
        .field("version", FieldKind::Raw64)
        .build();

    // ── Registry: singleton, holds the global namespace table. ────
    // CAS-updated on `create-ns` / `remove-ns`.
    let registry = dyn_module
        .obj_type("Registry")
        .field("namespaces", FieldKind::Value)
        .build();

    // ── Record: backing storage for all user `deftype*` instances.
    let record = dyn_module
        .obj_type("Record")
        .field("type_name", FieldKind::Value)
        .varlen_values()
        .build();

    // ── Atom.
    let atom = dyn_module
        .obj_type("Atom")
        .field("val", FieldKind::Value)
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
        record,
        atom,
    }
}

/// All heap-object field offsets, resolved from the dynlang `ObjType`
/// declarations once at engine init. This is the single source of
/// truth — every `unsafe { p.add(N) }` site reads from here instead
/// of duplicating its own `const FOO_OFFSET: usize = ...` constant.
///
/// `Copy` so it can be passed by value (it's just `usize`s).
#[derive(Clone, Copy)]
pub struct Layouts {
    // ── String (varlen_bytes) ────────────────────────────────────
    pub string_hash: usize,
    pub string_varlen_count: usize,
    pub string_bytes: usize,

    // ── Keyword ──────────────────────────────────────────────────
    pub keyword_sym: usize,
    pub keyword_hash: usize,

    // ── List (cons cell) ─────────────────────────────────────────
    pub list_first: usize,
    pub list_rest: usize,
    pub list_count: usize,

    // ── Vector (HAMT/RRB-shaped declaration) ─────────────────────
    pub vector_root: usize,
    pub vector_tail: usize,
    pub vector_count: usize,
    pub vector_shift: usize,

    // ── Set ──────────────────────────────────────────────────────
    pub set_backing: usize,

    // ── Atom ─────────────────────────────────────────────────────
    pub atom_val: usize,

    // ── Map (varlen_values) ──────────────────────────────────────
    pub map_count: usize,
    pub map_varlen_count: usize,
    pub map_varlen_elem_base: usize,

    // ── Array (pure varlen_values) ───────────────────────────────
    pub array_varlen_count: usize,
    pub array_elem_base: usize,

    // ── Record (varlen_values) ───────────────────────────────────
    pub record_type_name: usize,
    pub record_varlen_count: usize,
    pub record_fields_base: usize,

    // ── Fn (varlen_values capture) ───────────────────────────────
    pub fn_func_ref: usize,
    pub fn_arity: usize,
    pub fn_varlen_count: usize,
    pub fn_captures_base: usize,

    // ── Var ──────────────────────────────────────────────────────
    pub var_ns: usize,
    pub var_sym: usize,
    pub var_root: usize,
    pub var_meta: usize,
    pub var_flags: usize,

    // ── Namespace ────────────────────────────────────────────────
    pub ns_name: usize,
    pub ns_mappings: usize,
    pub ns_aliases: usize,
    pub ns_meta: usize,
    pub ns_version: usize,

    // ── Registry ─────────────────────────────────────────────────
    pub registry_namespaces: usize,
}

impl Layouts {
    /// Resolve every offset from the dynlang `ObjType` registry. Call
    /// once at engine init (after `declare_types`). Panics with a
    /// clear message if any field is missing or has the wrong kind —
    /// surfacing a `declare_types` typo immediately rather than at the
    /// first runtime field access.
    pub fn from_module(dyn_module: &DynModule, t: &Types) -> Self {
        let str_t = dyn_module.get_obj_type(t.string);
        let kw_t = dyn_module.get_obj_type(t.keyword);
        let list_t = dyn_module.get_obj_type(t.list);
        let vec_t = dyn_module.get_obj_type(t.vector);
        let set_t = dyn_module.get_obj_type(t.set);
        let atom_t = dyn_module.get_obj_type(t.atom);
        let map_t = dyn_module.get_obj_type(t.map);
        let array_t = dyn_module.get_obj_type(t.array);
        let rec_t = dyn_module.get_obj_type(t.record);
        let fn_t = dyn_module.get_obj_type(t.fn_obj);
        let var_t = dyn_module.get_obj_type(t.var);
        let ns_t = dyn_module.get_obj_type(t.namespace);
        let reg_t = dyn_module.get_obj_type(t.registry);

        Layouts {
            // String: hash (Raw64) then varlen_bytes. The first byte
            // of the bytes section is varlen_element_offset(0).
            string_hash: str_t.raw64_field_offset_named("hash"),
            string_varlen_count: str_t.type_info.varlen_count_offset(),
            string_bytes: str_t.type_info.varlen_element_offset(0),

            keyword_sym: kw_t.value_field_offset_named("sym"),
            keyword_hash: kw_t.raw64_field_offset_named("hash"),

            list_first: list_t.value_field_offset_named("first"),
            list_rest: list_t.value_field_offset_named("rest"),
            list_count: list_t.raw64_field_offset_named("count"),

            // Builder reorders: value fields first, then raw64. So
            // root/tail (Value) come before count/shift (Raw64).
            vector_root: vec_t.value_field_offset_named("root"),
            vector_tail: vec_t.value_field_offset_named("tail"),
            vector_count: vec_t.raw64_field_offset_named("count"),
            vector_shift: vec_t.raw64_field_offset_named("shift"),

            set_backing: set_t.value_field_offset_named("backing"),

            atom_val: atom_t.value_field_offset_named("val"),

            map_count: map_t.raw64_field_offset_named("count"),
            map_varlen_count: map_t.type_info.varlen_count_offset(),
            map_varlen_elem_base: map_t.type_info.varlen_element_offset(0),

            array_varlen_count: array_t.type_info.varlen_count_offset(),
            array_elem_base: array_t.type_info.varlen_element_offset(0),

            record_type_name: rec_t.value_field_offset_named("type_name"),
            record_varlen_count: rec_t.type_info.varlen_count_offset(),
            record_fields_base: rec_t.type_info.varlen_element_offset(0),

            fn_func_ref: fn_t.raw64_field_offset_named("func_ref"),
            fn_arity: fn_t.raw64_field_offset_named("arity"),
            fn_varlen_count: fn_t.type_info.varlen_count_offset(),
            fn_captures_base: fn_t.type_info.varlen_element_offset(0),

            var_ns: var_t.value_field_offset_named("ns"),
            var_sym: var_t.value_field_offset_named("sym"),
            var_root: var_t.value_field_offset_named("root"),
            var_meta: var_t.value_field_offset_named("meta"),
            var_flags: var_t.raw64_field_offset_named("flags"),

            ns_name: ns_t.value_field_offset_named("name"),
            ns_mappings: ns_t.value_field_offset_named("mappings"),
            ns_aliases: ns_t.value_field_offset_named("aliases"),
            ns_meta: ns_t.value_field_offset_named("meta"),
            ns_version: ns_t.raw64_field_offset_named("version"),

            registry_namespaces: reg_t.value_field_offset_named("namespaces"),
        }
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
