//! Heap-exploration dump — render the object graph using reflection metadata.
//!
//! Walks every allocated object in the heap's live spaces (via
//! [`Heap::walk_live_objects`]) and renders each one with its source type name,
//! field names, and decoded field values, resolving GC pointers to stable
//! per-dump object ids (`#0`, `#1`, …) so the output reads as a graph rather
//! than a wall of raw addresses.
//!
//! Used by the `GCR_HEAP_DUMP` end-of-run hook and available to host tooling
//! directly (`dump_heap_text(heap)`). Objects whose layout carries no nominal
//! metadata (builtins like arrays/closures) render structurally.

use std::collections::HashMap;
use std::fmt::Write as _;

use crate::gc::field::{read_varlen_bytes, read_varlen_count};
use crate::gc::heap::Heap;
use crate::gc::reflect::{FieldMeta, FieldTy, ScalarKind, TypeKind};
use crate::gc::scan::scan_object;
use crate::gc::type_info::{TypeInfo, VarLenKind};

/// Render the heap's live objects as a human-readable text report: a summary
/// line, a per-type histogram, and one line per object showing its decoded
/// fields. Pointers are shown as `#N` referring to the object's line.
///
/// Quiescence is ENFORCED, not assumed: the whole walk runs under a
/// [`Heap::pause_world`] stop-the-world pause, so no other mutator is allocating
/// or relocating while we read object memory (a live unjoined thread would
/// otherwise make this a data race / use-after-free — the same hazard fixed for
/// the allocation-site profiler). Do not call while already holding `gc_lock`
/// (e.g. from inside a GC callback): `pause_world` re-takes it and would
/// deadlock.
///
/// # Safety
/// All allocated objects must be valid (initialized headers, written varlen
/// counts) — true for any object the mutator has finished allocating.
pub unsafe fn dump_heap_text(heap: &Heap) -> String {
    // Read the live heap under STW so no mutator mutates/relocates mid-walk.
    let _pause = heap.pause_world();
    // Pass 1: collect object pointers in walk order and assign stable ids.
    let mut objs: Vec<(*mut u8, u16)> = Vec::new();
    let mut id_of: HashMap<usize, usize> = HashMap::new();
    unsafe {
        heap.walk_live_objects(&mut |obj, info| {
            id_of.insert(obj as usize, objs.len());
            objs.push((obj, info.type_id));
        });
    }

    let mut out = String::new();
    let _ = writeln!(out, "── gc-rust heap dump: {} objects ──", objs.len());

    // Per-type histogram (by name).
    let mut hist: HashMap<String, usize> = HashMap::new();
    for &(_obj, type_id) in &objs {
        let name = type_name(heap, type_id);
        *hist.entry(name).or_insert(0) += 1;
    }
    if !hist.is_empty() {
        let mut pairs: Vec<(&String, &usize)> = hist.iter().collect();
        pairs.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        let parts: Vec<String> = pairs.iter().map(|(n, c)| format!("{n}×{c}")).collect();
        let _ = writeln!(out, "   by type: {}", parts.join("  "));
    }
    let _ = writeln!(out);

    // Pass 2: render each object.
    for (id, &(obj, type_id)) in objs.iter().enumerate() {
        let rendered = unsafe { render_object(heap, obj, type_id, &id_of) };
        let _ = writeln!(out, "#{id} {rendered}");
    }
    out
}

/// A walked object plus its precomputed size and outgoing edges.
struct Node {
    obj: *mut u8,
    type_id: u16,
    bytes: usize,
    /// Ids of objects this one references (direct fields, interior pointers, and
    /// varlen elements — exactly what the GC traces).
    refs: Vec<usize>,
}

/// Collect the live heap as a graph: every object with its size and outgoing
/// reference edges (resolved to walk-order ids). Edges come from `scan_object`,
/// so they match the GC's own view (including interior pointers in flattened
/// value fields and varlen elements).
///
/// # Safety
/// The heap must be quiescent; all objects valid.
unsafe fn collect_graph(heap: &Heap) -> Vec<Node> {
    let mut objs: Vec<(*mut u8, u16, usize)> = Vec::new();
    let mut id_of: HashMap<usize, usize> = HashMap::new();
    unsafe {
        heap.walk_live_objects(&mut |obj, info| {
            let varlen_len = match info.varlen {
                VarLenKind::None => 0,
                _ => read_varlen_count(obj, info),
            };
            id_of.insert(obj as usize, objs.len());
            objs.push((obj, info.type_id, info.allocation_size(varlen_len)));
        });
    }
    let mut nodes = Vec::with_capacity(objs.len());
    for &(obj, type_id, bytes) in &objs {
        let info = heap.type_info_by_id(type_id);
        let mut refs = Vec::new();
        unsafe {
            scan_object(obj, info, |slot| {
                let p = (slot as *const *mut u8).read() ;
                if !p.is_null() {
                    if let Some(&id) = id_of.get(&(p as usize)) {
                        refs.push(id);
                    }
                }
            });
        }
        nodes.push(Node { obj, type_id, bytes, refs });
    }
    nodes
}

/// Walk up two dominator-tree chains to their common ancestor, comparing by
/// reverse-postorder number (Cooper-Harvey-Kennedy `intersect`).
fn dom_intersect(mut a: usize, mut b: usize, idom: &[usize], rpo_num: &[usize]) -> usize {
    while a != b {
        while rpo_num[a] > rpo_num[b] {
            a = idom[a];
        }
        while rpo_num[b] > rpo_num[a] {
            b = idom[b];
        }
    }
    a
}

/// Exclusive **retained size** per object: the total bytes that become
/// unreachable if that object is removed — i.e. the size of the subtree it
/// dominates. Computed over the snapshot graph with a virtual super-root
/// connected to the proxy roots (in-degree-0 objects, plus an entry into any
/// pure cycle so every node is reachable). Uses the Cooper-Harvey-Kennedy
/// iterative dominator algorithm — simple and fine for exploration-scale heaps.
fn retained_sizes(nodes: &[Node]) -> Vec<usize> {
    let n = nodes.len();
    if n == 0 {
        return vec![];
    }
    let r = n; // virtual root
    let total = n + 1;

    // Proxy roots: in-degree-0 objects, then add any still-unreachable node (a
    // pure cycle has no in-degree-0 entry) until the whole graph is covered.
    let mut indeg = vec![0usize; n];
    for node in nodes {
        for &t in &node.refs {
            indeg[t] += 1;
        }
    }
    let mut roots: Vec<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
    loop {
        let mut seen = vec![false; n];
        let mut stack = roots.clone();
        for &s in &roots {
            seen[s] = true;
        }
        while let Some(v) = stack.pop() {
            for &t in &nodes[v].refs {
                if !seen[t] {
                    seen[t] = true;
                    stack.push(t);
                }
            }
        }
        match (0..n).find(|&i| !seen[i]) {
            Some(u) => roots.push(u),
            None => break,
        }
    }

    // Successor lists including the virtual root, and predecessor lists.
    let mut succ: Vec<Vec<usize>> = Vec::with_capacity(total);
    for node in nodes {
        succ.push(node.refs.clone());
    }
    succ.push(roots.clone()); // succ[r]
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); total];
    for (i, s) in succ.iter().enumerate() {
        for &t in s {
            preds[t].push(i);
        }
    }

    // Postorder DFS from r → reverse-postorder numbering.
    let mut postorder = Vec::with_capacity(total);
    let mut visited = vec![false; total];
    let mut stack: Vec<(usize, usize)> = vec![(r, 0)];
    visited[r] = true;
    while let Some(&(v, ci)) = stack.last() {
        if ci < succ[v].len() {
            stack.last_mut().unwrap().1 += 1;
            let w = succ[v][ci];
            if !visited[w] {
                visited[w] = true;
                stack.push((w, 0));
            }
        } else {
            postorder.push(v);
            stack.pop();
        }
    }
    let rpo: Vec<usize> = postorder.iter().rev().copied().collect();
    let mut rpo_num = vec![usize::MAX; total];
    for (i, &v) in rpo.iter().enumerate() {
        rpo_num[v] = i;
    }

    // Iterative dominators.
    let mut idom = vec![usize::MAX; total];
    idom[r] = r;
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &rpo {
            if b == r {
                continue;
            }
            let mut new_idom = usize::MAX;
            for &p in &preds[b] {
                if idom[p] != usize::MAX {
                    new_idom = if new_idom == usize::MAX {
                        p
                    } else {
                        dom_intersect(p, new_idom, &idom, &rpo_num)
                    };
                }
            }
            if new_idom != usize::MAX && idom[b] != new_idom {
                idom[b] = new_idom;
                changed = true;
            }
        }
    }

    // Subtree sum: process leaves-first (reverse RPO), folding each node's total
    // into its immediate dominator.
    let mut ret = vec![0usize; total];
    for (i, node) in nodes.iter().enumerate() {
        ret[i] = node.bytes;
    }
    for &v in rpo.iter().rev() {
        if v == r {
            continue;
        }
        let d = idom[v];
        if d != usize::MAX && d != v {
            ret[d] += ret[v];
        }
    }
    ret.truncate(n);
    ret
}

/// Render the live heap as a JSON snapshot for external tooling: a summary
/// (object/byte totals + per-type histogram + top reachable roots) and the full
/// object graph (each object's type, size, human render, and outgoing edges).
///
/// "Reachable bytes" for a root is the transitive closure size from that
/// object; roots are objects with no in-edges within the snapshot (the top-level
/// structures). This is a program-end snapshot — there is no live stack root set
/// — so it reflects all currently-allocated objects.
///
/// # Safety
/// The heap must be quiescent; all objects valid.
pub unsafe fn dump_heap_json(heap: &Heap) -> String {
    let nodes = unsafe { collect_graph(heap) };
    let id_of: HashMap<usize, usize> =
        nodes.iter().enumerate().map(|(i, n)| (n.obj as usize, i)).collect();
    let _ = &id_of;

    let total_bytes: usize = nodes.iter().map(|n| n.bytes).sum();

    // Per-type histogram (count + bytes), sorted by bytes desc.
    let mut hist: HashMap<String, (usize, usize)> = HashMap::new();
    for n in &nodes {
        let e = hist.entry(type_name(heap, n.type_id)).or_insert((0, 0));
        e.0 += 1;
        e.1 += n.bytes;
    }
    let mut by_type: Vec<(String, usize, usize)> =
        hist.into_iter().map(|(k, (c, b))| (k, c, b)).collect();
    by_type.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)));

    // In-degree to find roots (objects nothing else points at).
    let mut indeg = vec![0usize; nodes.len()];
    for n in &nodes {
        for &t in &n.refs {
            indeg[t] += 1;
        }
    }
    // Reachable bytes per root (transitive closure).
    let mut roots: Vec<(usize, usize)> = Vec::new(); // (id, reachable_bytes)
    for (i, n) in nodes.iter().enumerate() {
        if indeg[i] == 0 {
            let mut seen = vec![false; nodes.len()];
            let mut stack = vec![i];
            let mut bytes = 0usize;
            while let Some(v) = stack.pop() {
                if seen[v] {
                    continue;
                }
                seen[v] = true;
                bytes += nodes[v].bytes;
                stack.extend(nodes[v].refs.iter().copied());
            }
            let _ = n;
            roots.push((i, bytes));
        }
    }
    roots.sort_by(|a, b| b.1.cmp(&a.1));

    // Exclusive retained size per object (dominator subtree), and the top
    // retainers — the "what's actually holding memory" view.
    let retained = retained_sizes(&nodes);
    let mut top: Vec<(usize, usize)> = retained.iter().copied().enumerate().collect();
    top.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    top.truncate(10);

    // ---- Emit JSON (hand-built; gcrust-rt has no serde) -------------------
    let mut out = String::new();
    out.push_str("{\n  \"summary\": {");
    let _ = write!(out, "\"objects\": {}, \"bytes\": {}, ", nodes.len(), total_bytes);
    out.push_str("\"by_type\": [");
    for (i, (name, count, bytes)) in by_type.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        let _ = write!(out, "{{\"name\": {}, \"count\": {}, \"bytes\": {}}}", json_str(name), count, bytes);
    }
    out.push_str("], \"roots\": [");
    for (i, (id, bytes)) in roots.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        let _ = write!(out, "{{\"id\": {}, \"reachable_bytes\": {}}}", id, bytes);
    }
    out.push_str("], \"top_retainers\": [");
    for (i, (id, bytes)) in top.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        let _ = write!(out, "{{\"id\": {}, \"retained_bytes\": {}}}", id, bytes);
    }
    out.push_str("]},\n  \"objects\": [\n");
    for (id, n) in nodes.iter().enumerate() {
        let render = unsafe { render_object(heap, n.obj, n.type_id, &id_of) };
        let refs: Vec<String> = n.refs.iter().map(|r| r.to_string()).collect();
        let _ = write!(
            out,
            "    {{\"id\": {}, \"type\": {}, \"bytes\": {}, \"retained_bytes\": {}, \"render\": {}, \"refs\": [{}]}}",
            id,
            json_str(&type_name(heap, n.type_id)),
            n.bytes,
            retained[id],
            json_str(&render),
            refs.join(", "),
        );
        out.push_str(if id + 1 < nodes.len() { ",\n" } else { "\n" });
    }
    out.push_str("  ]\n}\n");
    out
}

/// Encode `s` as a JSON string literal (quotes + escapes for `"`, `\`, control
/// chars). Type names (`List<i64>`) and renders (string contents with quotes)
/// can contain characters that must be escaped.
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// The type name for `type_id` (for the histogram), or a synthetic `<type N>`.
fn type_name(heap: &Heap, type_id: u16) -> String {
    match heap.type_meta_by_id(type_id) {
        Some(m) => m.name.clone(),
        None => format!("<type {type_id}>"),
    }
}

/// Render one object's contents (without its `#id` prefix).
///
/// # Safety
/// `obj` must be a valid object of `type_id`.
unsafe fn render_object(
    heap: &Heap,
    obj: *mut u8,
    type_id: u16,
    id_of: &HashMap<usize, usize>,
) -> String {
    let info = heap.type_info_by_id(type_id);
    let Some(meta) = heap.type_meta_by_id(type_id) else {
        return unsafe { render_opaque(&format!("<type {type_id}>"), obj, info) };
    };
    match &meta.kind {
        TypeKind::Struct { fields } => {
            let body = unsafe { render_fields(heap, obj, 0, fields, id_of) };
            format!("{} {{ {} }}", meta.name, body)
        }
        TypeKind::Enum {
            tag_offset,
            variants,
        } => {
            let tag = unsafe { read_u32(obj, *tag_offset as usize) };
            match variants.iter().find(|v| v.tag == tag) {
                Some(v) if v.fields.is_empty() => format!("{}::{}", meta.name, v.name),
                Some(v) => {
                    let body = unsafe { render_fields(heap, obj, 0, &v.fields, id_of) };
                    format!("{}::{}({})", meta.name, v.name, body)
                }
                None => format!("{}::<unknown tag {}>", meta.name, tag),
            }
        }
        TypeKind::Opaque => unsafe { render_opaque(&meta.name, obj, info) },
    }
}

/// Render a comma-separated `name: value` (or just `value` for positional)
/// field list. `base` is the byte offset (within `obj`) of the aggregate these
/// fields belong to — 0 for the object itself, or the value field's offset when
/// rendering a flattened value aggregate (whose field offsets are relative).
///
/// # Safety
/// `obj` must be a valid object whose layout includes all `fields` at `base`.
unsafe fn render_fields(
    heap: &Heap,
    obj: *mut u8,
    base: usize,
    fields: &[FieldMeta],
    id_of: &HashMap<usize, usize>,
) -> String {
    let mut parts = Vec::with_capacity(fields.len());
    for f in fields {
        let val = unsafe { render_field_value(heap, obj, base, f, id_of) };
        // Positional fields ("0", "1", …) render value-only for tuple-ish look.
        if f.name.chars().all(|c| c.is_ascii_digit()) {
            parts.push(val);
        } else {
            parts.push(format!("{}: {}", f.name, val));
        }
    }
    parts.join(", ")
}

/// Decode and format a single field's value at `base + f.offset` within `obj`.
///
/// # Safety
/// `obj` must be valid and `base + f.offset` in bounds for it.
unsafe fn render_field_value(
    heap: &Heap,
    obj: *mut u8,
    base: usize,
    f: &FieldMeta,
    id_of: &HashMap<usize, usize>,
) -> String {
    let foff = base + f.offset as usize;
    let at = unsafe { obj.add(foff) };
    match f.ty {
        FieldTy::Ref(_) => {
            let p = unsafe { (at as *const *mut u8).read_unaligned() };
            if p.is_null() {
                "null".to_string()
            } else if let Some(id) = id_of.get(&(p as usize)) {
                format!("#{id}")
            } else {
                // Points outside the walked spaces (e.g. a permanent/global).
                format!("@{:p}", p)
            }
        }
        FieldTy::Scalar(k) => unsafe { render_scalar(at, k) },
        FieldTy::Value(vid) => match heap.value_meta_by_id(vid) {
            // Recurse into a flattened value aggregate: its field offsets are
            // value-relative, so render them with `base = foff`.
            Some(vm) => match &vm.kind {
                TypeKind::Struct { fields } => {
                    let body = unsafe { render_fields(heap, obj, foff, fields, id_of) };
                    format!("{} {{ {} }}", vm.name, body)
                }
                _ => format!("{} <value>", vm.name),
            },
            None => "<value>".to_string(),
        },
    }
}

/// Read a u32 at byte `offset` within `obj` (the enum tag word).
///
/// # Safety
/// `obj.add(offset)` must point at a readable u32.
unsafe fn read_u32(obj: *mut u8, offset: usize) -> u32 {
    unsafe { (obj.add(offset) as *const u32).read_unaligned() }
}

/// Decode a scalar of kind `k` at `at`.
///
/// # Safety
/// `at` must point at a readable scalar of the right size.
unsafe fn render_scalar(at: *const u8, k: ScalarKind) -> String {
    unsafe {
        match k {
            ScalarKind::I8 => format!("{}", (at as *const i8).read_unaligned()),
            ScalarKind::I16 => format!("{}", (at as *const i16).read_unaligned()),
            ScalarKind::I32 => format!("{}", (at as *const i32).read_unaligned()),
            ScalarKind::I64 => format!("{}", (at as *const i64).read_unaligned()),
            ScalarKind::U8 => format!("{}", at.read_unaligned()),
            ScalarKind::U16 => format!("{}", (at as *const u16).read_unaligned()),
            ScalarKind::U32 => format!("{}", (at as *const u32).read_unaligned()),
            ScalarKind::U64 => format!("{}", (at as *const u64).read_unaligned()),
            ScalarKind::F32 => format!("{}", (at as *const f32).read_unaligned()),
            ScalarKind::F64 => format!("{}", (at as *const f64).read_unaligned()),
            ScalarKind::Bool => format!("{}", at.read_unaligned() != 0),
            ScalarKind::Char => {
                let c = (at as *const u32).read_unaligned();
                char::from_u32(c).map(|c| format!("'{c}'")).unwrap_or_else(|| format!("char({c})"))
            }
            ScalarKind::Ptr => format!("{:p}", (at as *const *const u8).read_unaligned()),
        }
    }
}

/// Render an object with no nominal field metadata (builtins). Strings show
/// their contents; other varlen objects show their element count.
///
/// # Safety
/// `obj` must be a valid object described by `info`.
unsafe fn render_opaque(name: &str, obj: *mut u8, info: &TypeInfo) -> String {
    unsafe {
        match info.varlen {
            VarLenKind::Bytes if name == "String" => {
                let bytes = read_varlen_bytes(obj, info);
                format!("String {:?}", String::from_utf8_lossy(bytes))
            }
            VarLenKind::Bytes => {
                let n = read_varlen_count(obj, info);
                format!("{name} <{n} bytes>")
            }
            VarLenKind::Values => {
                let n = read_varlen_count(obj, info);
                format!("{name} <{n} elements>")
            }
            VarLenKind::None => name.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(bytes: usize, refs: Vec<usize>) -> Node {
        Node { obj: core::ptr::null_mut(), type_id: 0, bytes, refs }
    }

    #[test]
    fn retained_shared_node_is_dominated_by_common_ancestor() {
        // 4=Two(->2,->3); 2->1; 3->1; 1->0. Node 1 is shared by 2 and 3.
        let nodes = vec![
            node(40, vec![]),       // 0 leaf
            node(40, vec![0]),      // 1 (shared) -> 0
            node(40, vec![1]),      // 2 -> 1
            node(40, vec![1]),      // 3 -> 1
            node(32, vec![2, 3]),   // 4 root -> 2,3
        ];
        let r = retained_sizes(&nodes);
        assert_eq!(r[4], 192, "root retains the whole graph");
        assert_eq!(r[1], 80, "shared node retains itself + its exclusive child");
        assert_eq!(r[2], 40, "does not retain the shared node");
        assert_eq!(r[3], 40);
        assert_eq!(r[0], 40);
    }

    #[test]
    fn retained_chain_is_cumulative() {
        let nodes = vec![node(10, vec![1]), node(20, vec![2]), node(30, vec![])];
        let r = retained_sizes(&nodes);
        assert_eq!(r, vec![60, 50, 30]);
    }

    #[test]
    fn retained_pure_cycle_does_not_panic() {
        // 0 <-> 1 with no in-degree-0 entry: a proxy root is injected.
        let nodes = vec![node(10, vec![1]), node(20, vec![0])];
        let r = retained_sizes(&nodes);
        assert_eq!(r[0], 30, "the injected root dominates the whole cycle");
        assert_eq!(r[1], 20);
    }

    #[test]
    fn retained_empty() {
        assert_eq!(retained_sizes(&[]), Vec::<usize>::new());
    }
}
