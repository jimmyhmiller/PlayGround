//! Writes a memscope [`HeapGraph`] as a **JVM HPROF 1.0.2** binary heap dump, so
//! a Rust program's heap opens in **Eclipse MAT** / **VisualVM** — dominator
//! tree, retained sizes, paths-to-GC-roots, leak suspects, class histograms.
//!
//! The mapping (Rust → HPROF):
//! * each live allocation → an HPROF object; its id is the allocation address.
//! * each recovered Rust type → an HPROF class; the class name is the Rust type.
//! * a struct/`Box`/`Rc`/`Arc` allocation → an **instance** whose object-reference
//!   fields are the type's pointer fields (offsets from DWARF layout), filled
//!   from the graph's edges.
//! * a `Vec<ptr>` / `HashMap` allocation → an **object array** of its referents.
//! * a `String` / `Vec<u8>` / opaque allocation → a **byte array** (its bytes are
//!   read from memory, so MAT even shows string contents).
//! * the graph's approximate roots → HPROF GC roots, so MAT keeps them reachable.
//!
//! HPROF is big-endian with a fixed identifier size (we use 8 bytes = 64-bit
//! addresses). Heap records are emitted in length-bounded `HEAP_DUMP_SEGMENT`s.

use std::collections::HashMap;
use std::io::{self, Write};

use memscope_proto::{AllocShape, HeapGraph};
pub use memscope_symbols::MemReader;
use memscope_symbols::LayoutIndex;

/// Per-type layout the writer needs: a type's shallow size and the byte offsets
/// of its pointer fields. Implemented for the DWARF [`LayoutIndex`]; a trait so
/// the writer is testable without a real binary.
pub trait TypeLayout {
    fn size_of(&self, name: &str) -> Option<u64>;
    fn pointer_field_offsets(&self, name: &str) -> Vec<u64>;
}

impl TypeLayout for LayoutIndex {
    fn size_of(&self, name: &str) -> Option<u64> {
        LayoutIndex::size_of(self, name)
    }
    fn pointer_field_offsets(&self, name: &str) -> Vec<u64> {
        self.pointer_fields(name)
            .unwrap_or_default()
            .into_iter()
            .map(|f| f.offset)
            .collect()
    }
}

// --- HPROF constants ---------------------------------------------------------

const ID_SIZE: u32 = 8;

// Top-level record tags.
const TAG_UTF8: u8 = 0x01;
const TAG_LOAD_CLASS: u8 = 0x02;
const TAG_TRACE: u8 = 0x05;
const TAG_HEAP_DUMP_SEGMENT: u8 = 0x1C;
const TAG_HEAP_DUMP_END: u8 = 0x2C;

// Heap-dump sub-record tags.
const SUB_ROOT_UNKNOWN: u8 = 0xFF;
const SUB_CLASS_DUMP: u8 = 0x20;
const SUB_INSTANCE_DUMP: u8 = 0x21;
const SUB_OBJ_ARRAY_DUMP: u8 = 0x22;
const SUB_PRIM_ARRAY_DUMP: u8 = 0x23;

// Basic-type codes.
const T_OBJECT: u8 = 2;
const T_FLOAT: u8 = 6;
const T_DOUBLE: u8 = 7;
const T_BYTE: u8 = 8;
const T_SHORT: u8 = 9;
const T_INT: u8 = 10;
const T_LONG: u8 = 11;

/// Synthetic-identifier base: top bit set, so class/string ids never collide
/// with user-space heap addresses (which have the top bit clear).
const SYNTH_BASE: u64 = 0x8000_0000_0000_0000;

/// Base for the synthetic backing-array ids minted for typed scalar buffers
/// (a `String`/`Vec<u8>` instance owns a primitive array holding its bytes).
/// Bit 62 set: distinct from real addresses (well below) and class/string ids
/// (bit 63).
const CHILD_BASE: u64 = 0x4000_0000_0000_0000;

/// Flush a heap-dump segment once its buffered body exceeds this many bytes.
const SEGMENT_LIMIT: usize = 16 * 1024 * 1024;

/// Summary of what was written.
#[derive(Clone, Copy, Debug, Default)]
pub struct HprofStats {
    pub objects: u64,
    pub classes: u64,
    pub roots: u64,
    pub bytes_written: u64,
}

/// Write `graph` as an HPROF heap dump to `w`. `mem` reads allocation bytes for
/// byte-array leaves (string/buffer contents); `timestamp_ms` stamps the header.
pub fn write_hprof(
    w: &mut impl Write,
    graph: &HeapGraph,
    layout: &dyn TypeLayout,
    mem: &dyn MemReader,
    timestamp_ms: u64,
) -> io::Result<HprofStats> {
    let mut hp = HprofWriter::new(w);
    hp.write_header(timestamp_ms)?;

    // --- plan: group edges by source node (offset -> target address) ---------
    // Built first so node classification is O(1) per node (whether it has any
    // out-edges), not O(edges) — otherwise the whole pass is O(nodes × edges).
    let mut edges_by_node: Vec<Vec<(u64, u64)>> = vec![Vec::new(); graph.nodes.len()];
    for e in &graph.edges {
        if let Some(to) = graph.nodes.get(e.to as usize) {
            edges_by_node[e.from as usize].push((e.offset, to.addr));
        }
    }
    for v in &mut edges_by_node {
        v.sort_by_key(|(off, _)| *off);
    }

    // Classify every node (instance / object-array / byte-array).
    let kinds: Vec<NodeKind> = graph
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| classify(n, !edges_by_node[i].is_empty()))
        .collect();

    // Distinct instance type names -> a representative byte size.
    let mut inst_types: HashMap<&str, u64> = HashMap::new();
    for (i, n) in graph.nodes.iter().enumerate() {
        if matches!(kinds[i], NodeKind::Instance) {
            if let Some(ty) = n.ty.as_deref() {
                let sz = layout.size_of(ty).unwrap_or(n.size);
                inst_types.entry(ty).or_insert(sz);
            }
        }
    }

    // Reserve well-known class ids + their name strings.
    let object_class = hp.intern_class("java.lang.Object");
    let byte_array_class = hp.intern_class("byte[]");

    // A class per Rust instance type, with its pointer fields.
    struct ClassDef {
        class_id: u64,
        inst_size: u64,
        field_offsets: Vec<u64>,
    }
    let mut classes: HashMap<String, ClassDef> = HashMap::new();
    for (ty, &sz) in &inst_types {
        let class_id = hp.intern_class(&jvm_safe_name(ty));
        let mut field_offsets: Vec<u64> = layout.pointer_field_offsets(ty);
        field_offsets.sort_unstable();
        field_offsets.dedup();
        // Reserve a field-name string per offset.
        for off in &field_offsets {
            hp.intern_string(&format!("f@0x{off:x}"));
        }
        classes.insert((*ty).to_string(), ClassDef { class_id, inst_size: sz, field_offsets });
    }

    // A distinct array class per Rust container type (`Vec<T>` / `HashMap<T>`),
    // so object arrays carry their real type instead of a generic `Object[]`.
    let mut array_classes: HashMap<String, u64> = HashMap::new();
    for (i, n) in graph.nodes.iter().enumerate() {
        if matches!(kinds[i], NodeKind::ObjArray) {
            let name = array_class_name(n);
            if !array_classes.contains_key(&name) {
                let id = hp.intern_class(&name);
                array_classes.insert(name, id);
            }
        }
    }

    // A class per typed scalar buffer (`String` / `Vec<u8>` / `Vec<u32>` …): an
    // instance owning a primitive array of its bytes. This is the only way HPROF
    // lets a variable-length buffer carry a Rust name — a primitive array on its
    // own is always `byte[]`/`int[]`. Genuinely untyped buffers stay `byte[]`.
    let data_field = hp.intern_string("data");
    let mut buffer_classes: HashMap<String, u64> = HashMap::new();
    for (i, n) in graph.nodes.iter().enumerate() {
        if matches!(kinds[i], NodeKind::PrimArray) {
            if let Some(name) = buffer_class_name(n) {
                if !buffer_classes.contains_key(&name) {
                    let id = hp.intern_class(&name);
                    buffer_classes.insert(name, id);
                }
            }
        }
    }

    // --- emit string + class records before the heap dump --------------------
    hp.write_strings()?;
    hp.write_load_class(object_class)?;
    hp.write_load_class(byte_array_class)?;
    for c in classes.values() {
        hp.write_load_class(c.class_id)?;
    }
    for &id in array_classes.values() {
        hp.write_load_class(id)?;
    }
    for &id in buffer_classes.values() {
        hp.write_load_class(id)?;
    }
    hp.write_dummy_trace()?;

    // --- heap dump -----------------------------------------------------------
    let mut seg = SegmentWriter::new();

    // Class dumps: Object (hierarchy root), byte[], each instance class (with
    // fields), and each per-container array class.
    seg.class_dump(object_class, 0, 0, &[], &hp);
    seg.class_dump(byte_array_class, object_class, 0, &[], &hp);
    for c in classes.values() {
        let fields: Vec<(u64, u8)> =
            c.field_offsets.iter().map(|&off| (hp.string_id(&format!("f@0x{off:x}")), T_OBJECT)).collect();
        seg.class_dump(c.class_id, object_class, c.inst_size as u32, &fields, &hp);
        seg.maybe_flush(&mut hp)?;
    }
    for &id in array_classes.values() {
        seg.class_dump(id, object_class, 0, &[], &hp);
    }
    // Buffer wrapper classes: one `data` reference to the backing primitive array.
    for &id in buffer_classes.values() {
        seg.class_dump(id, object_class, 24, &[(data_field, T_OBJECT)], &hp);
    }

    let mut stats = HprofStats::default();
    stats.classes =
        classes.len() as u64 + array_classes.len() as u64 + buffer_classes.len() as u64 + 2;
    // Synthetic ids for the backing primitive arrays owned by buffer wrappers.
    let mut next_child = CHILD_BASE;

    // Object records.
    for (i, n) in graph.nodes.iter().enumerate() {
        match kinds[i] {
            NodeKind::Instance => {
                let cdef = n.ty.as_deref().and_then(|t| classes.get(t));
                let Some(cdef) = cdef else { continue };
                // Field values, in declared order: edge target at that offset, or null.
                let mut vals: Vec<u64> = Vec::with_capacity(cdef.field_offsets.len());
                for &off in &cdef.field_offsets {
                    let target = edges_by_node[i].iter().find(|(o, _)| *o == off).map(|(_, t)| *t);
                    vals.push(target.unwrap_or(0));
                }
                seg.instance_dump(n.addr, cdef.class_id, &vals);
            }
            NodeKind::ObjArray => {
                let elems: Vec<u64> = edges_by_node[i].iter().map(|(_, t)| *t).collect();
                let cls = array_classes.get(&array_class_name(n)).copied().unwrap_or(object_class);
                seg.obj_array_dump(n.addr, cls, &elems);
            }
            NodeKind::PrimArray => {
                let bytes = read_bytes(mem, n.addr, n.size);
                let (etype, esize) = prim_elem(n.ty.as_deref());
                match buffer_class_name(n).and_then(|nm| buffer_classes.get(&nm).copied()) {
                    Some(cls) => {
                        // A named instance (`String`/`Vec<u8>`/…) owning a
                        // primitive array of its bytes.
                        let child = next_child;
                        next_child += 1;
                        seg.instance_dump(n.addr, cls, &[child]);
                        seg.prim_array_dump(child, etype, esize, &bytes);
                        stats.objects += 1; // the backing array (wrapper counted below)
                    }
                    None => seg.prim_array_dump(n.addr, T_BYTE, 1, &bytes),
                }
            }
        }
        stats.objects += 1;
        seg.maybe_flush(&mut hp)?;
    }

    // GC roots: keep the graph's approximate roots reachable in MAT.
    for &r in &graph.roots {
        if let Some(n) = graph.nodes.get(r as usize) {
            seg.root_unknown(n.addr);
            stats.roots += 1;
        }
    }

    seg.flush(&mut hp)?;
    hp.write_record(TAG_HEAP_DUMP_END, &[])?;

    stats.bytes_written = hp.bytes_written;
    Ok(stats)
}

// --- node classification -----------------------------------------------------

enum NodeKind {
    Instance,
    ObjArray,
    PrimArray,
}

fn classify(n: &memscope_proto::GraphNode, has_edges: bool) -> NodeKind {
    match n.shape {
        Some(AllocShape::StringBuf) => NodeKind::PrimArray,
        Some(AllocShape::HashTable) => NodeKind::ObjArray,
        Some(AllocShape::Vec) => {
            if has_edges {
                NodeKind::ObjArray
            } else {
                NodeKind::PrimArray
            }
        }
        Some(AllocShape::Boxed) | Some(AllocShape::Rc) | Some(AllocShape::Arc) => {
            if n.ty.is_some() {
                NodeKind::Instance
            } else {
                NodeKind::PrimArray
            }
        }
        _ => {
            if n.ty.is_some() {
                NodeKind::Instance
            } else {
                NodeKind::PrimArray
            }
        }
    }
}

/// Largest byte-array payload we'll serialize. A live allocation should never
/// exceed this; a larger `len` means a corrupt/stale size (e.g. an address freed
/// and reused while dumping a mutating heap), so we cap it rather than loop for
/// billions of iterations or `Vec::with_capacity` ourselves into an OOM abort.
const MAX_PRIM_BYTES: usize = 512 * 1024 * 1024;

/// The HPROF array-class name for an object-array node: its Rust container type,
/// so a `Vec<Box<Particle>>` reads as `Vec<Box<Particle>>` (not `java.lang.Object[]`)
/// and a hash map as `HashMap<…>`.
fn array_class_name(n: &memscope_proto::GraphNode) -> String {
    match (n.shape, n.ty.as_deref()) {
        (Some(AllocShape::Vec), Some(t)) => format!("Vec<{t}>"),
        (Some(AllocShape::Vec), None) => "Vec<?>".to_string(),
        (Some(AllocShape::HashTable), Some(t)) => format!("HashMap<{t}>"),
        (Some(AllocShape::HashTable), None) => "HashMap<?>".to_string(),
        (_, Some(t)) => format!("Slice<{t}>"),
        (_, None) => "Slice<?>".to_string(),
    }
}

/// The Rust container name for a typed scalar buffer (a `PrimArray` node we have
/// a type for), e.g. `String` or `Vec<u32>`. `None` = genuinely untyped bytes,
/// which stay a plain `byte[]`.
fn buffer_class_name(n: &memscope_proto::GraphNode) -> Option<String> {
    match (n.shape, n.ty.as_deref()) {
        (Some(AllocShape::StringBuf), _) => Some("String".to_string()),
        (Some(AllocShape::Vec), Some(t)) => Some(format!("Vec<{t}>")),
        _ => None,
    }
}

/// HPROF primitive element (type code, byte size) for a Rust scalar element type,
/// so the backing array reads as the right kind (`u32` → `int[]`, `f64` →
/// `double[]`, …) rather than always `byte[]`.
fn prim_elem(ty: Option<&str>) -> (u8, usize) {
    match ty {
        Some("u16") | Some("i16") => (T_SHORT, 2),
        Some("u32") | Some("i32") | Some("char") => (T_INT, 4),
        Some("u64") | Some("i64") | Some("usize") | Some("isize") => (T_LONG, 8),
        Some("f32") => (T_FLOAT, 4),
        Some("f64") => (T_DOUBLE, 8),
        _ => (T_BYTE, 1), // u8 / i8 / bool / unknown → bytes
    }
}

/// Avoid emitting a class name that starts with `[`: that's the JVM
/// array-descriptor prefix, which MAT / heapster re-parse — turning a Rust slice
/// `[String]` into a mangled `String][]`. Rewrite Rust slice/array syntax to a
/// readable, non-colliding form.
fn jvm_safe_name(name: &str) -> String {
    if let Some(inner) = name.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        if inner.contains(';') {
            format!("Array<{inner}>")
        } else {
            format!("Slice<{inner}>")
        }
    } else {
        name.to_string()
    }
}

/// Read `len` bytes (clamped to [`MAX_PRIM_BYTES`]) of an allocation's contents.
/// Unreadable bytes (e.g. a since-freed/unmapped region) are left zero rather
/// than faulting — the reader is expected to read *safely*.
fn read_bytes(mem: &dyn MemReader, addr: u64, len: u64) -> Vec<u8> {
    let len = (len as usize).min(MAX_PRIM_BYTES);
    let mut out = vec![0u8; len];
    mem.read_into(addr, &mut out);
    out
}

// --- top-level record writer -------------------------------------------------

struct HprofWriter<'a, W: Write> {
    w: &'a mut W,
    bytes_written: u64,
    strings: Vec<(u64, String)>,        // (string id, value) in insertion order
    string_ids: HashMap<String, u64>,
    class_names: HashMap<u64, u64>,     // class id -> name string id
    next_synth: u64,
}

impl<'a, W: Write> HprofWriter<'a, W> {
    fn new(w: &'a mut W) -> Self {
        HprofWriter {
            w,
            bytes_written: 0,
            strings: Vec::new(),
            string_ids: HashMap::new(),
            class_names: HashMap::new(),
            next_synth: SYNTH_BASE,
        }
    }

    fn synth_id(&mut self) -> u64 {
        let id = self.next_synth;
        self.next_synth += 1;
        id
    }

    /// Intern a string, returning its id (creating a UTF8 record later).
    fn intern_string(&mut self, s: &str) -> u64 {
        if let Some(&id) = self.string_ids.get(s) {
            return id;
        }
        let id = self.synth_id();
        self.string_ids.insert(s.to_string(), id);
        self.strings.push((id, s.to_string()));
        id
    }

    fn string_id(&self, s: &str) -> u64 {
        self.string_ids[s]
    }

    /// A class id IS a synthetic id; its name string is interned alongside.
    fn intern_class(&mut self, name: &str) -> u64 {
        // Class object id and the name string are distinct ids.
        let name_id = self.intern_string(name);
        let class_id = self.synth_id();
        // Remember class -> name string mapping via a parallel map in strings?
        // We only need the name id at LOAD_CLASS time, so stash it.
        self.class_names.insert(class_id, name_id);
        class_id
    }

    fn write_header(&mut self, ts_ms: u64) -> io::Result<()> {
        self.raw(b"JAVA PROFILE 1.0.2\0")?;
        self.raw(&ID_SIZE.to_be_bytes())?;
        self.raw(&((ts_ms >> 32) as u32).to_be_bytes())?;
        self.raw(&((ts_ms & 0xffff_ffff) as u32).to_be_bytes())?;
        Ok(())
    }

    fn write_strings(&mut self) -> io::Result<()> {
        // Take ownership to avoid borrow conflict.
        let strings = std::mem::take(&mut self.strings);
        for (id, s) in &strings {
            let mut body = Vec::with_capacity(8 + s.len());
            body.extend_from_slice(&id.to_be_bytes());
            body.extend_from_slice(s.as_bytes());
            self.write_record(TAG_UTF8, &body)?;
        }
        self.strings = strings;
        Ok(())
    }

    fn write_load_class(&mut self, class_id: u64) -> io::Result<()> {
        static_assert_idsize();
        let name_id = self.class_names[&class_id];
        let mut body = Vec::with_capacity(24);
        body.extend_from_slice(&1u32.to_be_bytes()); // class serial (unused but nonzero)
        body.extend_from_slice(&class_id.to_be_bytes());
        body.extend_from_slice(&1u32.to_be_bytes()); // stack trace serial
        body.extend_from_slice(&name_id.to_be_bytes());
        self.write_record(TAG_LOAD_CLASS, &body)
    }

    /// A single empty stack trace (serial 1) referenced by all heap records.
    fn write_dummy_trace(&mut self) -> io::Result<()> {
        let mut body = Vec::new();
        body.extend_from_slice(&1u32.to_be_bytes()); // trace serial
        body.extend_from_slice(&1u32.to_be_bytes()); // thread serial
        body.extend_from_slice(&0u32.to_be_bytes()); // number of frames
        self.write_record(TAG_TRACE, &body)
    }

    fn write_record(&mut self, tag: u8, body: &[u8]) -> io::Result<()> {
        self.raw(&[tag])?;
        self.raw(&0u32.to_be_bytes())?; // time delta
        self.raw(&(body.len() as u32).to_be_bytes())?;
        self.raw(body)
    }

    fn raw(&mut self, b: &[u8]) -> io::Result<()> {
        self.w.write_all(b)?;
        self.bytes_written += b.len() as u64;
        Ok(())
    }
}

fn static_assert_idsize() {
    debug_assert_eq!(ID_SIZE, 8);
}

// --- heap-dump segment buffer ------------------------------------------------

struct SegmentWriter {
    buf: Vec<u8>,
}

impl SegmentWriter {
    fn new() -> Self {
        SegmentWriter { buf: Vec::with_capacity(SEGMENT_LIMIT) }
    }

    fn id(&mut self, id: u64) {
        self.buf.extend_from_slice(&id.to_be_bytes());
    }
    fn u4(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }
    fn u2(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }
    fn u1(&mut self, v: u8) {
        self.buf.push(v);
    }

    fn class_dump<W: Write>(
        &mut self,
        class_id: u64,
        super_id: u64,
        inst_size: u32,
        fields: &[(u64, u8)],
        _hp: &HprofWriter<W>,
    ) {
        self.u1(SUB_CLASS_DUMP);
        self.id(class_id);
        self.u4(1); // stack trace serial
        self.id(super_id);
        self.id(0); // class loader
        self.id(0); // signers
        self.id(0); // protection domain
        self.id(0); // reserved
        self.id(0); // reserved
        self.u4(inst_size);
        self.u2(0); // constant pool entries
        self.u2(0); // static fields
        self.u2(fields.len() as u16);
        for (name_id, ty) in fields {
            self.id(*name_id);
            self.u1(*ty);
        }
    }

    fn instance_dump(&mut self, obj_id: u64, class_id: u64, field_vals: &[u64]) {
        self.u1(SUB_INSTANCE_DUMP);
        self.id(obj_id);
        self.u4(1); // stack trace serial
        self.id(class_id);
        self.u4((field_vals.len() * ID_SIZE as usize) as u32);
        for v in field_vals {
            self.id(*v);
        }
    }

    fn obj_array_dump(&mut self, obj_id: u64, class_id: u64, elems: &[u64]) {
        self.u1(SUB_OBJ_ARRAY_DUMP);
        self.id(obj_id);
        self.u4(1);
        self.u4(elems.len() as u32);
        self.id(class_id);
        for e in elems {
            self.id(*e);
        }
    }

    /// `bytes` is the raw allocation content; it's reinterpreted as
    /// `bytes.len() / elem_size` elements of `elem_type` (so a `Vec<u32>` buffer
    /// becomes an `int[]`, not 4× as many `byte[]`).
    fn prim_array_dump(&mut self, obj_id: u64, elem_type: u8, elem_size: usize, bytes: &[u8]) {
        let n = bytes.len() / elem_size.max(1);
        let used = n * elem_size.max(1);
        self.u1(SUB_PRIM_ARRAY_DUMP);
        self.id(obj_id);
        self.u4(1);
        self.u4(n as u32);
        self.u1(elem_type);
        self.buf.extend_from_slice(&bytes[..used]);
    }

    fn root_unknown(&mut self, obj_id: u64) {
        self.u1(SUB_ROOT_UNKNOWN);
        self.id(obj_id);
    }

    fn maybe_flush<W: Write>(&mut self, w: &mut HprofWriter<W>) -> io::Result<()> {
        if self.buf.len() >= SEGMENT_LIMIT {
            self.flush(w)?;
        }
        Ok(())
    }

    fn flush<W: Write>(&mut self, w: &mut HprofWriter<W>) -> io::Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        let body = std::mem::take(&mut self.buf);
        w.write_record(TAG_HEAP_DUMP_SEGMENT, &body)?;
        self.buf = Vec::with_capacity(SEGMENT_LIMIT);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memscope_proto::{GraphEdge, GraphNode};
    use std::collections::{HashMap, HashSet};

    /// A stub layout: type -> (size, pointer-field offsets).
    struct StubLayout(HashMap<String, (u64, Vec<u64>)>);
    impl TypeLayout for StubLayout {
        fn size_of(&self, name: &str) -> Option<u64> {
            self.0.get(name).map(|(s, _)| *s)
        }
        fn pointer_field_offsets(&self, name: &str) -> Vec<u64> {
            self.0.get(name).map(|(_, f)| f.clone()).unwrap_or_default()
        }
    }

    /// A reader returning zeros (no real memory in tests).
    struct ZeroReader;
    impl MemReader for ZeroReader {
        fn read_uint(&self, _addr: u64, _size: u64) -> Option<u64> {
            Some(0)
        }
    }

    fn node(addr: u64, size: u64, ty: Option<&str>, shape: Option<AllocShape>) -> GraphNode {
        GraphNode {
            addr,
            size,
            ty: ty.map(String::from),
            shape,
            retained_size: size,
            idom: -1,
            in_degree: 0,
            out_degree: 0,
        }
    }

    /// A small heap: a Vec holding two boxed Particles (one links to the other),
    /// plus an unreferenced String — exercises obj-array, instance, and byte-array.
    fn sample_graph() -> HeapGraph {
        HeapGraph {
            nodes: vec![
                node(0x1000, 16, Some("Particle"), Some(AllocShape::Vec)), // 0: the Vec
                node(0x2000, 16, Some("Particle"), Some(AllocShape::Boxed)), // 1
                node(0x3000, 16, Some("Particle"), Some(AllocShape::Boxed)), // 2
                node(0x5000, 12, Some("u8"), Some(AllocShape::StringBuf)),   // 3: a String
            ],
            edges: vec![
                GraphEdge { from: 0, to: 1, offset: 0 },
                GraphEdge { from: 0, to: 2, offset: 8 },
                GraphEdge { from: 1, to: 2, offset: 8 }, // particle 1 -> particle 2
            ],
            roots: vec![0, 3],
            total_bytes: 60,
            opaque_nodes: 0,
        }
    }

    fn layout() -> StubLayout {
        let mut m = HashMap::new();
        m.insert("Particle".to_string(), (16u64, vec![8u64])); // one pointer field at +8
        StubLayout(m)
    }

    /// A minimal HPROF reader used only to validate structural consistency.
    struct Parsed {
        classes: HashMap<u64, usize>, // class id -> instance field count
        objects: HashSet<u64>,        // all object ids
        refs: Vec<u64>,               // every reference value emitted
        instance_class: Vec<(u64, u64, u32)>, // (obj, class, field-bytes)
        roots: Vec<u64>,
    }

    fn parse(buf: &[u8]) -> Parsed {
        // Header: name\0, u4 idsize, u8 timestamp.
        let nul = buf.iter().position(|&b| b == 0).unwrap();
        let mut p = nul + 1;
        let idsize = u32::from_be_bytes(buf[p..p + 4].try_into().unwrap());
        assert_eq!(idsize, 8, "id size");
        p += 4 + 8;

        let rd_id = |b: &[u8], p: usize| u64::from_be_bytes(b[p..p + 8].try_into().unwrap());
        let rd_u4 = |b: &[u8], p: usize| u32::from_be_bytes(b[p..p + 4].try_into().unwrap());
        let rd_u2 = |b: &[u8], p: usize| u16::from_be_bytes(b[p..p + 2].try_into().unwrap());

        let mut out = Parsed {
            classes: HashMap::new(),
            objects: HashSet::new(),
            refs: Vec::new(),
            instance_class: Vec::new(),
            roots: Vec::new(),
        };

        while p < buf.len() {
            let tag = buf[p];
            let len = rd_u4(buf, p + 5) as usize;
            let body_start = p + 9;
            let body_end = body_start + len;
            if tag == TAG_HEAP_DUMP_SEGMENT {
                let mut q = body_start;
                while q < body_end {
                    let sub = buf[q];
                    q += 1;
                    match sub {
                        SUB_CLASS_DUMP => {
                            let class_id = rd_id(buf, q);
                            q += 8 + 4 + 8 * 6 + 4; // id, trace, 6 ids, instsize
                            let cp = rd_u2(buf, q) as usize;
                            q += 2;
                            assert_eq!(cp, 0);
                            let nstatic = rd_u2(buf, q) as usize;
                            q += 2;
                            assert_eq!(nstatic, 0);
                            let ninst = rd_u2(buf, q) as usize;
                            q += 2;
                            out.classes.insert(class_id, ninst);
                            q += ninst * (8 + 1);
                        }
                        SUB_INSTANCE_DUMP => {
                            let obj = rd_id(buf, q);
                            q += 8 + 4;
                            let class = rd_id(buf, q);
                            q += 8;
                            let nbytes = rd_u4(buf, q);
                            q += 4;
                            out.objects.insert(obj);
                            out.instance_class.push((obj, class, nbytes));
                            // Field values are references (id-sized each).
                            for k in 0..(nbytes as usize / 8) {
                                out.refs.push(rd_id(buf, q + k * 8));
                            }
                            q += nbytes as usize;
                        }
                        SUB_OBJ_ARRAY_DUMP => {
                            let obj = rd_id(buf, q);
                            q += 8 + 4;
                            let nelem = rd_u4(buf, q) as usize;
                            q += 4 + 8; // nelem, array class id
                            out.objects.insert(obj);
                            for k in 0..nelem {
                                out.refs.push(rd_id(buf, q + k * 8));
                            }
                            q += nelem * 8;
                        }
                        SUB_PRIM_ARRAY_DUMP => {
                            let obj = rd_id(buf, q);
                            q += 8 + 4;
                            let nelem = rd_u4(buf, q) as usize;
                            q += 4;
                            let ty = buf[q];
                            q += 1;
                            let esz = match ty {
                                6 | 10 => 4,        // float / int
                                7 | 11 => 8,        // double / long
                                5 | 9 => 2,         // char / short
                                _ => 1,             // byte / boolean
                            };
                            out.objects.insert(obj);
                            q += nelem * esz;
                        }
                        SUB_ROOT_UNKNOWN => {
                            out.roots.push(rd_id(buf, q));
                            q += 8;
                        }
                        other => panic!("unexpected sub-record tag {other:#x}"),
                    }
                }
                assert_eq!(q, body_end, "segment sub-records must consume the body");
            }
            p = body_end;
        }
        out
    }

    #[test]
    fn jvm_safe_names_never_start_with_bracket() {
        // A leading '[' is the JVM array-descriptor prefix; MAT/heapster reparse
        // it (turning `[String]` into `String][]`).
        assert_eq!(jvm_safe_name("[String]"), "Slice<String>");
        assert_eq!(jvm_safe_name("[u8; 4]"), "Array<u8; 4>");
        assert_eq!(jvm_safe_name("Particle"), "Particle");
        assert_eq!(jvm_safe_name("Vec<Box<Particle>>"), "Vec<Box<Particle>>");
        for n in ["[String]", "[u8; 4]", "Particle"] {
            assert!(!jvm_safe_name(n).starts_with('['), "{n} still starts with [");
        }
    }

    #[test]
    fn header_is_valid_hprof() {
        let g = sample_graph();
        let mut buf = Vec::new();
        write_hprof(&mut buf, &g, &layout(), &ZeroReader, 0).unwrap();
        assert!(buf.starts_with(b"JAVA PROFILE 1.0.2\0"));
    }

    #[test]
    fn every_reference_and_root_resolves() {
        let g = sample_graph();
        let mut buf = Vec::new();
        let stats = write_hprof(&mut buf, &g, &layout(), &ZeroReader, 0).unwrap();
        // 4 nodes, but the StringBuf node becomes a `String` wrapper instance + a
        // backing byte[] child = 5 objects.
        assert_eq!(stats.objects, 5);
        assert_eq!(stats.roots, 2);

        let parsed = parse(&buf);
        assert_eq!(parsed.objects.len(), 5, "one object per node + the String's backing array");
        // Every emitted reference points at a real object or is null.
        for r in &parsed.refs {
            assert!(*r == 0 || parsed.objects.contains(r), "dangling reference {r:#x}");
        }
        // Every root is a real object.
        for r in &parsed.roots {
            assert!(parsed.objects.contains(r), "root {r:#x} is not an object");
        }
        // Particle 1 links to Particle 2 (0x3000) via its one field.
        assert!(parsed.refs.contains(&0x3000));
    }

    #[test]
    fn instance_field_bytes_match_class_declaration() {
        let g = sample_graph();
        let mut buf = Vec::new();
        write_hprof(&mut buf, &g, &layout(), &ZeroReader, 0).unwrap();
        let parsed = parse(&buf);
        // Each instance's field-bytes block equals its class's declared field count * 8.
        for (_obj, class, nbytes) in &parsed.instance_class {
            let fields = parsed.classes[class];
            assert_eq!(*nbytes as usize, fields * 8, "instance/class field mismatch");
        }
    }
}
