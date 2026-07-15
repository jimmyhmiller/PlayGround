//! The ONE collection representation — Clojure's model. `[a b]`, `{k v}`, and
//! `#{e}` are *data*: the reader builds the real runtime persistent collection
//! directly (holding the raw element forms), macros receive it as data, `quote`
//! keeps it literal, and only *expression position* lowers it to element-wise
//! evaluation. There is no reader-only "code record" for collections.
//!
//! Two runtime layouts coexist by bootstrap phase (the same seam
//! `binding_items` always had to know about):
//!   * while `clojure.core` itself is loading: the core prelude's `PVec`
//!     (record: cnt shift root tail — raw arrays all the way down) and the
//!     list-backed `Map`/`Set` records;
//!   * once the cljs-ported types exist (detected by `clojure.core/-EMPTY-PV`):
//!     `PersistentVector` (VectorNode trie), `PersistentArrayMap`, and
//!     `PersistentHashSet` — exactly the objects their constructors build.
//!
//! The readers here (`vector_items`/`map_entries`/`set_items`) accept every
//! layout, so the expander is representation-blind.
//!
//! A literal map/set of ANY size builds a PersistentArrayMap (linear lookup,
//! still correct); the first `assoc` past the 8-entry threshold promotes it to
//! the HAMT, as cljs does. Duplicate literal keys are not detected (real
//! Clojure throws at read time); the first occurrence wins on lookup.

use microlang::runtime::ObjView;
use microlang::value::Sym;
use microlang::{Obj, Repr, Runtime, Val, ValueModel};

fn alloc<M: ValueModel>(rt: &mut Runtime<M>, o: Obj) -> u64 {
    let id = rt.alloc(o);
    <M::R as Repr>::enc_ref(id)
}

fn record<M: ValueModel>(rt: &mut Runtime<M>, ty: &str, fields: Vec<u64>) -> u64 {
    let type_id: Sym = rt.intern(ty);
    let id = rt.alloc_record(type_id, &fields);
    <M::R as Repr>::enc_ref(id)
}

fn array<M: ValueModel>(rt: &mut Runtime<M>, elems: &[u64]) -> u64 {
    let id = rt.alloc_vector(elems);
    <M::R as Repr>::enc_ref(id)
}

/// Are the cljs-ported persistent types loaded yet? Before that boundary the
/// runtime collection types are core's `PVec` / list-backed `Map`/`Set`.
fn user_phase<M: ValueModel>(rt: &mut Runtime<M>) -> bool {
    let s = rt.intern("clojure.core/-EMPTY-PV");
    rt.global(s).is_some()
}

// ─────────────────────────────────────────────────────────────────────────
// Builders — construct the phase-appropriate runtime collection in Rust
// (the reader runs before any in-language constructor can).
// ─────────────────────────────────────────────────────────────────────────

/// Build the current phase's persistent vector holding `elems`.
pub fn make_vector<M: ValueModel>(rt: &mut Runtime<M>, elems: &[u64]) -> u64 {
    if user_phase(rt) {
        build_persistent_vector(rt, elems)
    } else {
        build_pvec(rt, elems)
    }
}

/// Build the current phase's map from a flat `k v k v …` slice.
pub fn make_map<M: ValueModel>(rt: &mut Runtime<M>, kvs: &[u64]) -> u64 {
    if user_phase(rt) {
        build_pam(rt, kvs)
    } else {
        let lst = rt.vec_to_list(kvs);
        record(rt, "Map", vec![lst])
    }
}

/// Build the current phase's set holding `elems`.
pub fn make_set<M: ValueModel>(rt: &mut Runtime<M>, elems: &[u64]) -> u64 {
    if user_phase(rt) {
        // A PersistentHashSet is a wrapper over any map: {elem nil …}. The
        // backing PersistentArrayMap promotes itself to the HAMT on write.
        let nil = rt.encode(Val::Nil);
        let kvs: Vec<u64> = elems.iter().flat_map(|&e| [e, nil]).collect();
        let pam = build_pam(rt, &kvs);
        record(rt, "PersistentHashSet", vec![nil, pam, nil])
    } else {
        let lst = rt.vec_to_list(elems);
        record(rt, "Set", vec![lst])
    }
}

/// `PersistentArrayMap` layout (cljs): fields `[meta cnt arr __hash]`, where
/// `arr` is the flat `k v` array and `cnt` counts ENTRIES.
fn build_pam<M: ValueModel>(rt: &mut Runtime<M>, kvs: &[u64]) -> u64 {
    let nil = rt.encode(Val::Nil);
    let cnt = rt.encode(Val::Int((kvs.len() / 2) as i128));
    let arr = array(rt, kvs);
    record(rt, "PersistentArrayMap", vec![nil, cnt, arr, nil])
}

/// A 32-wide node array, nil-padded (what `pv-fresh-node` / `-make-array 32`
/// produce), holding `children` in its low slots.
fn node_array<M: ValueModel>(rt: &mut Runtime<M>, children: &[u64]) -> u64 {
    let nil = rt.encode(Val::Nil);
    let mut v = vec![nil; 32];
    v[..children.len()].copy_from_slice(children);
    array(rt, &v)
}

/// Split `elems` into the trie body (full 32-wide leaves) and the tail, exactly
/// as a conj-built vector lays out: `tail_off = ((cnt-1) >> 5) << 5` for
/// cnt ≥ 32, else 0.
fn trie_split(elems: &[u64]) -> (usize, &[u64]) {
    let cnt = elems.len();
    let tail_off = if cnt < 32 { 0 } else { ((cnt - 1) >> 5) << 5 };
    (tail_off, &elems[tail_off..])
}

/// Core-phase `PVec` (record: `[cnt shift root tail]`, raw arrays at every
/// trie level). Mirrors `-empty-pvec` / `-pv-conj`'s invariants.
fn build_pvec<M: ValueModel>(rt: &mut Runtime<M>, elems: &[u64]) -> u64 {
    let (tail_off, tail_elems) = trie_split(elems);
    let tail = array(rt, tail_elems);
    // Leaves are always FULL 32-wide arrays (tail_off is a multiple of 32).
    let mut level: Vec<u64> =
        elems[..tail_off].chunks(32).map(|c| array(rt, c)).collect();
    let mut shift = 5i128;
    while level.len() > 32 {
        level = level.chunks(32).map(|c| node_array(rt, c)).collect();
        shift += 5;
    }
    let root = node_array(rt, &level);
    let cnt = rt.encode(Val::Int(elems.len() as i128));
    let shift = rt.encode(Val::Int(shift));
    record(rt, "PVec", vec![cnt, shift, root, tail])
}

/// User-phase `PersistentVector` (fields `[meta cnt shift root tail __hash]`;
/// trie nodes are `VectorNode` records `[edit arr]`; the tail is a raw array).
/// `__hash` is nil = uncached; `-hash` recomputes on demand.
fn build_persistent_vector<M: ValueModel>(rt: &mut Runtime<M>, elems: &[u64]) -> u64 {
    let nil = rt.encode(Val::Nil);
    let (tail_off, tail_elems) = trie_split(elems);
    let tail = array(rt, tail_elems);
    let mk_node = |rt: &mut Runtime<M>, children: &[u64]| -> u64 {
        let arr = node_array(rt, children);
        record(rt, "VectorNode", vec![nil, arr])
    };
    let mut level: Vec<u64> = elems[..tail_off]
        .chunks(32)
        .map(|c| {
            let arr = array(rt, c);
            record(rt, "VectorNode", vec![nil, arr])
        })
        .collect();
    let mut shift = 5i128;
    while level.len() > 32 {
        level = level.chunks(32).map(|c| mk_node(rt, c)).collect();
        shift += 5;
    }
    let root = mk_node(rt, &level);
    let cnt = rt.encode(Val::Int(elems.len() as i128));
    let shift = rt.encode(Val::Int(shift));
    record(rt, "PersistentVector", vec![nil, cnt, shift, root, tail, nil])
}

// ─────────────────────────────────────────────────────────────────────────
// Readers — representation-blind structural access for the expander.
// ─────────────────────────────────────────────────────────────────────────

/// The record's `(tag, fields)` if `v` is a Record, else None.
fn record_parts<M: ValueModel>(rt: &Runtime<M>, v: u64) -> Option<(String, Vec<u64>)> {
    if let Val::Ref(id) = rt.decode(v) {
        if let ObjView::Record { type_id, fields } = rt.view_gc(id) {
            return Some((rt.sym_name(type_id).to_string(), fields.to_vec()));
        }
    }
    None
}

fn is_nil<M: ValueModel>(rt: &Runtime<M>, v: u64) -> bool {
    matches!(rt.decode(v), Val::Nil)
}

fn int_of<M: ValueModel>(rt: &Runtime<M>, v: u64) -> usize {
    match rt.decode(v) {
        Val::Int(n) => n as usize,
        _ => panic!("collection layout: expected an int field"),
    }
}

fn arr_elems<M: ValueModel>(rt: &Runtime<M>, arr: u64) -> Vec<u64> {
    let Val::Ref(id) = rt.decode(arr) else { panic!("collection layout: expected an array") };
    let ObjView::Vector { elems, .. } = rt.view_gc(id) else {
        panic!("collection layout: expected an array")
    };
    elems.to_vec()
}

fn arr_get<M: ValueModel>(rt: &Runtime<M>, arr: u64, i: usize) -> u64 {
    let Val::Ref(id) = rt.decode(arr) else { panic!("trie node: not an array") };
    let ObjView::Vector { elems, .. } = rt.view_gc(id) else { panic!("trie node: not an array") };
    elems[i]
}

/// The elements of any vector representation: `PVec`, `PersistentVector`, or
/// the legacy `Vector` display record (kept only as a print shim for
/// `:arglists` / `-realize`). `None` if `form` is not a vector.
pub fn vector_items<M: ValueModel>(rt: &Runtime<M>, form: u64) -> Option<Vec<u64>> {
    let (tag, fields) = record_parts(rt, form)?;
    match tag.as_str() {
        "Vector" => Some(rt.list_to_vec(*fields.first()?)),
        "PVec" => Some(walk_trie(rt, &fields, 0, false)),
        "PersistentVector" => Some(walk_trie(rt, &fields, 1, true)),
        _ => None,
    }
}

/// Read a trie-vector's logical elements. `base` is the index of `cnt` in the
/// record (`PVec` has no meta field; `PersistentVector` does); `node_records`
/// says whether trie nodes are `VectorNode` records (arr = field 1) or raw
/// arrays. Mirrors cljs `unchecked-array-for`.
fn walk_trie<M: ValueModel>(
    rt: &Runtime<M>,
    fields: &[u64],
    base: usize,
    node_records: bool,
) -> Vec<u64> {
    let cnt = int_of(rt, fields[base]);
    let shift = int_of(rt, fields[base + 1]);
    let root = fields[base + 2];
    let tail = fields[base + 3];
    let node_arr = |rt: &Runtime<M>, node: u64| -> u64 {
        if node_records {
            record_parts(rt, node).expect("VectorNode").1[1]
        } else {
            node
        }
    };
    let tail_off = if cnt < 32 { 0 } else { ((cnt - 1) >> 5) << 5 };
    let mut out = Vec::with_capacity(cnt);
    for i in 0..cnt {
        let arr = if i >= tail_off {
            tail
        } else {
            let mut node = root;
            let mut level = shift;
            while level > 0 {
                node = arr_get(rt, node_arr(rt, node), (i >> level) & 31);
                level -= 5;
            }
            node_arr(rt, node)
        };
        out.push(arr_get(rt, arr, i & 31));
    }
    out
}

/// The flat `k v k v …` entries of any map representation (`Map`,
/// `PersistentArrayMap`, `PersistentHashMap`). `None` if not a map.
pub fn map_entries<M: ValueModel>(rt: &Runtime<M>, form: u64) -> Option<Vec<u64>> {
    let (tag, fields) = record_parts(rt, form)?;
    match tag.as_str() {
        "Map" => Some(rt.list_to_vec(*fields.first()?)),
        "PersistentArrayMap" => Some(arr_elems(rt, fields[2])),
        "PersistentHashMap" => {
            // fields: [meta cnt root has-nil? nil-val __hash]
            let mut out = Vec::new();
            if matches!(rt.decode(fields[3]), Val::Bool(true)) {
                out.push(<M::R as Repr>::enc_nil());
                out.push(fields[4]);
            }
            if !is_nil(rt, fields[2]) {
                phm_node_entries(rt, fields[2], &mut out);
            }
            Some(out)
        }
        _ => None,
    }
}

/// Walk a cljs HAMT node, appending `k v` pairs. Node layouts:
/// `BitmapIndexedNode [edit bitmap arr]` — arr pairs, key nil ⇒ child in val slot;
/// `ArrayNode [edit cnt arr]` — arr of maybe-nil child nodes;
/// `HashCollisionNode [edit hash cnt arr]` — flat kv pairs.
fn phm_node_entries<M: ValueModel>(rt: &Runtime<M>, node: u64, out: &mut Vec<u64>) {
    let Some((tag, fields)) = record_parts(rt, node) else { return };
    match tag.as_str() {
        "BitmapIndexedNode" => {
            let arr = arr_elems(rt, fields[2]);
            for kv in arr.chunks(2) {
                if kv.len() < 2 {
                    break;
                }
                if is_nil(rt, kv[0]) {
                    if !is_nil(rt, kv[1]) {
                        phm_node_entries(rt, kv[1], out);
                    }
                } else {
                    out.push(kv[0]);
                    out.push(kv[1]);
                }
            }
        }
        "ArrayNode" => {
            for child in arr_elems(rt, fields[2]) {
                if !is_nil(rt, child) {
                    phm_node_entries(rt, child, out);
                }
            }
        }
        "HashCollisionNode" => {
            let arr = arr_elems(rt, fields[3]);
            for kv in arr.chunks(2) {
                if kv.len() == 2 && !is_nil(rt, kv[0]) {
                    out.push(kv[0]);
                    out.push(kv[1]);
                }
            }
        }
        other => panic!("unknown HAMT node type: {other}"),
    }
}

/// The elements of any set representation (`Set`, `PersistentHashSet`).
pub fn set_items<M: ValueModel>(rt: &Runtime<M>, form: u64) -> Option<Vec<u64>> {
    let (tag, fields) = record_parts(rt, form)?;
    match tag.as_str() {
        "Set" => Some(rt.list_to_vec(*fields.first()?)),
        "PersistentHashSet" => {
            // fields: [meta hash-map __hash] — the elements are the map's keys.
            let kvs = map_entries(rt, fields[1])?;
            Some(kvs.chunks(2).filter_map(|kv| kv.first().copied()).collect())
        }
        _ => None,
    }
}

/// Is `form` any map representation? (The `& {:keys …}` kwargs and
/// destructuring paths test this.)
pub fn is_map_rep<M: ValueModel>(rt: &Runtime<M>, form: u64) -> bool {
    map_entries(rt, form).is_some()
}
