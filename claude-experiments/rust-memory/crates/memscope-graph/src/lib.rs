//! Reconstructs the **heap reference graph** from a live allocation set plus
//! DWARF type layouts — the thing that turns memscope from an allocation
//! profiler into a heap analyzer (retained sizes, dominators, leak ownership).
//!
//! For each live allocation we know its address, size, recovered element type,
//! and container shape. From the type layout we know the byte offsets of every
//! pointer field. So we read those pointers out of the allocation's bytes and,
//! whenever one lands inside another live allocation, record a directed edge.
//! From the edge set we derive approximate roots, the dominator tree, and each
//! node's retained size.
//!
//! ## What is and isn't walked (v1, honest about limits)
//! * `Box<T>` — one `T` at offset 0.
//! * `Rc<T>` / `Arc<T>` — one `T` after two `usize` ref counts (offset 16).
//! * `Vec<T>` / slices — `size/size_of::<T>()` elements (the backing buffer;
//!   the uninitialized tail past `len` is read but only yields an edge if it
//!   happens to hold a live address, which is filtered out almost always).
//! * `String` / `Vec<u8>` — leaf (no interior pointers).
//! * **HashMap / HashSet** — *opaque* in v1: the hashbrown control-byte + bucket
//!   layout isn't decoded, so edges out of entries are missed. Counted in
//!   `opaque_nodes`.
//! * **Enum variants** — pointer fields inside a `variant_part` are read without
//!   checking the active discriminant; a stale read only produces an edge if it
//!   lands exactly on a live allocation, so false edges are rare but possible.
//!
//! Roots are "not referenced by any tracked allocation" — a heap-only
//! approximation (true roots live on the stack/in statics, which we don't
//! track), but exactly what you want for retained-size/leak analysis.

use std::collections::HashMap;

use memscope_proto::{AllocShape, GraphEdge, GraphNode, HeapGraph};
use memscope_symbols::{LayoutIndex, PtrField};

/// One live allocation to place in the graph.
pub struct NodeInput {
    pub addr: u64,
    pub size: u64,
    pub type_name: Option<String>,
    pub shape: Option<AllocShape>,
}

/// Reads `8` bytes of process memory at `addr`. The graph builder only ever asks
/// for addresses provably inside a live allocation, so an in-process reader can
/// simply dereference. Returns `None` if the read can't be served.
pub trait MemReader {
    fn read_u64(&self, addr: u64) -> Option<u64>;
}

/// An in-process reader: reads our own address space directly. Safe given the
/// builder's bounds guarantee, but still uses unaligned reads.
pub struct InProcessReader;

impl MemReader for InProcessReader {
    #[inline]
    fn read_u64(&self, addr: u64) -> Option<u64> {
        if addr == 0 {
            return None;
        }
        // SAFETY: the builder only requests offsets strictly within a live
        // allocation's bounds; that memory is mapped and owned by this process
        // for the duration of the (point-in-time) walk.
        unsafe { Some((addr as *const u8).cast::<u64>().read_unaligned()) }
    }
}

/// Maximum element slots walked per `Vec`-shaped allocation (backstop against a
/// pathological capacity; real container sizes are far below this).
const MAX_VEC_SLOTS: u64 = 1 << 22;

/// Build the heap reference graph.
pub fn build(nodes: &[NodeInput], layout: &LayoutIndex, mem: &dyn MemReader) -> HeapGraph {
    let n = nodes.len();

    // addr -> node index, and a sorted (start, end, idx) table for interior
    // pointers (a pointer into the middle of an allocation).
    let mut by_addr: HashMap<u64, u32> = HashMap::with_capacity(n);
    let mut ranges: Vec<(u64, u64, u32)> = Vec::with_capacity(n);
    for (i, node) in nodes.iter().enumerate() {
        by_addr.insert(node.addr, i as u32);
        ranges.push((node.addr, node.addr.saturating_add(node.size), i as u32));
    }
    ranges.sort_unstable_by_key(|r| r.0);

    let find = |p: u64| -> Option<u32> {
        if let Some(&i) = by_addr.get(&p) {
            return Some(i);
        }
        // Interior pointer: largest start <= p, check p < end.
        let idx = ranges.partition_point(|r| r.0 <= p);
        if idx == 0 {
            return None;
        }
        let (start, end, i) = ranges[idx - 1];
        if p >= start && p < end {
            Some(i)
        } else {
            None
        }
    };

    // Cache flattened pointer fields per type name.
    let mut field_cache: HashMap<String, Vec<PtrField>> = HashMap::new();
    let mut opaque_nodes = 0u32;

    let mut edges: Vec<GraphEdge> = Vec::new();
    let mut out_degree = vec![0u32; n];
    let mut in_degree = vec![0u32; n];

    for (i, node) in nodes.iter().enumerate() {
        let Some(ty) = node.type_name.as_deref() else {
            opaque_nodes += 1;
            continue;
        };

        // How many element instances live in this allocation, and where.
        let (base, stride, count) = instance_params(node.shape, layout, ty, node.size);
        if count == 0 {
            // Leaf (String/u8) or opaque (HashMap) — no walkable interior.
            if matches!(node.shape, Some(AllocShape::HashTable)) {
                opaque_nodes += 1;
            }
            continue;
        }

        let fields = field_cache.entry(ty.to_string()).or_insert_with(|| {
            layout.pointer_fields(ty).unwrap_or_default()
        });
        if fields.is_empty() {
            continue;
        }

        for inst in 0..count {
            let inst_base = base + inst.saturating_mul(stride);
            for f in fields.iter() {
                let off = inst_base + f.offset;
                if off + 8 > node.size {
                    continue;
                }
                let Some(p) = mem.read_u64(node.addr + off) else {
                    continue;
                };
                if p == 0 {
                    continue;
                }
                if let Some(to) = find(p) {
                    if to as usize != i {
                        edges.push(GraphEdge {
                            from: i as u32,
                            to,
                            offset: off,
                        });
                        out_degree[i] += 1;
                        in_degree[to as usize] += 1;
                    }
                }
            }
        }
    }

    // Successor lists for the dominator computation.
    let mut succ: Vec<Vec<u32>> = vec![Vec::new(); n];
    for e in &edges {
        succ[e.from as usize].push(e.to);
    }

    let roots: Vec<u32> = (0..n as u32)
        .filter(|&i| in_degree[i as usize] == 0)
        .collect();

    let idom = dominators(n, &succ, &roots);

    // Retained size: process nodes children-before-parents (reverse RPO) and
    // fold each node's retained bytes into its immediate dominator.
    let order = reverse_postorder(n, &succ, &roots);
    let mut retained: Vec<u64> = nodes.iter().map(|x| x.size).collect();
    for &node_idx in order.iter().rev() {
        let d = idom[node_idx as usize];
        if d >= 0 {
            retained[d as usize] += retained[node_idx as usize];
        }
    }

    let total_bytes = nodes.iter().map(|x| x.size).sum();

    let graph_nodes = nodes
        .iter()
        .enumerate()
        .map(|(i, node)| GraphNode {
            addr: node.addr,
            size: node.size,
            ty: node.type_name.clone(),
            shape: node.shape,
            retained_size: retained[i],
            idom: idom[i],
            in_degree: in_degree[i],
            out_degree: out_degree[i],
        })
        .collect();

    HeapGraph {
        nodes: graph_nodes,
        edges,
        roots,
        total_bytes,
        opaque_nodes,
    }
}

/// (base offset, stride, instance count) for walking an allocation's elements,
/// derived from the container shape and element size.
fn instance_params(
    shape: Option<AllocShape>,
    layout: &LayoutIndex,
    ty: &str,
    alloc_size: u64,
) -> (u64, u64, u64) {
    let esz = layout.size_of(ty).unwrap_or(0);
    match shape {
        Some(AllocShape::Vec) => {
            if esz == 0 {
                (0, 0, 0)
            } else {
                ((0), esz, (alloc_size / esz).min(MAX_VEC_SLOTS))
            }
        }
        // Rc/Arc inner = { strong: usize, weak: usize, value: T }.
        Some(AllocShape::Rc) | Some(AllocShape::Arc) => (16, esz, 1),
        // u8 buffers have no interior pointers.
        Some(AllocShape::StringBuf) => (0, 0, 0),
        // hashbrown layout not decoded in v1.
        Some(AllocShape::HashTable) => (0, 0, 0),
        // Box and everything else: a single instance at offset 0.
        _ => (0, esz, 1),
    }
}

/// Iterative dominators (Cooper–Harvey–Kennedy). A virtual super-root (node id
/// `n`) links to every `root`; `idom` is computed in node space. Returns the
/// immediate dominator per real node (-1 for the roots and unreachable nodes).
fn dominators(n: usize, succ: &[Vec<u32>], roots: &[u32]) -> Vec<i64> {
    let vroot = n;
    // RPO over the augmented graph (vroot first).
    let rpo = reverse_postorder_with_vroot(n, succ, roots);
    let mut rpo_num = vec![usize::MAX; n + 1];
    for (i, &node) in rpo.iter().enumerate() {
        rpo_num[node as usize] = i;
    }

    // Predecessors in the augmented graph.
    let mut preds: Vec<Vec<u32>> = vec![Vec::new(); n + 1];
    for &r in roots {
        preds[r as usize].push(vroot as u32);
    }
    for (from, ss) in succ.iter().enumerate() {
        for &to in ss {
            preds[to as usize].push(from as u32);
        }
    }

    // doms[node] = immediate dominator (node space); usize::MAX = undefined.
    let mut doms = vec![usize::MAX; n + 1];
    doms[vroot] = vroot;

    // intersect walks up the dom tree by RPO number until the two meet.
    let intersect = |mut a: usize, mut b: usize, doms: &[usize], rpo_num: &[usize]| -> usize {
        while a != b {
            while rpo_num[a] > rpo_num[b] {
                a = doms[a];
            }
            while rpo_num[b] > rpo_num[a] {
                b = doms[b];
            }
        }
        a
    };

    let mut changed = true;
    while changed {
        changed = false;
        for &node in rpo.iter() {
            let node = node as usize;
            if node == vroot {
                continue;
            }
            let mut new_idom = usize::MAX;
            for &p in &preds[node] {
                let p = p as usize;
                if rpo_num[p] == usize::MAX {
                    continue; // predecessor not reachable from vroot
                }
                if doms[p] != usize::MAX {
                    new_idom = if new_idom == usize::MAX {
                        p
                    } else {
                        intersect(p, new_idom, &doms, &rpo_num)
                    };
                }
            }
            if new_idom != usize::MAX && doms[node] != new_idom {
                doms[node] = new_idom;
                changed = true;
            }
        }
    }

    (0..n)
        .map(|i| {
            let d = doms[i];
            if d == usize::MAX || d == vroot {
                -1
            } else {
                d as i64
            }
        })
        .collect()
}

/// Reverse-postorder of the augmented graph (virtual root `n` -> all roots),
/// virtual root first.
fn reverse_postorder_with_vroot(n: usize, succ: &[Vec<u32>], roots: &[u32]) -> Vec<u32> {
    let total = n + 1;
    let vroot = n as u32;
    let mut visited = vec![false; total];
    let mut post: Vec<u32> = Vec::with_capacity(total);
    let mut stack: Vec<(u32, usize)> = vec![(vroot, 0)];
    visited[n] = true;
    while let Some(&mut (node, ref mut ci)) = stack.last_mut() {
        let children: &[u32] = if node as usize == n {
            roots
        } else {
            &succ[node as usize]
        };
        if *ci < children.len() {
            let c = children[*ci];
            *ci += 1;
            if !visited[c as usize] {
                visited[c as usize] = true;
                stack.push((c, 0));
            }
        } else {
            post.push(node);
            stack.pop();
        }
    }
    post.reverse();
    post
}

/// Reverse-postorder of nodes reachable from the virtual super-root (which links
/// to every root). The virtual root itself is omitted.
fn reverse_postorder(n: usize, succ: &[Vec<u32>], roots: &[u32]) -> Vec<u32> {
    let mut visited = vec![false; n];
    let mut post: Vec<u32> = Vec::with_capacity(n);
    // Iterative DFS to avoid stack overflow on deep graphs.
    for &r in roots {
        if visited[r as usize] {
            continue;
        }
        let mut stack: Vec<(u32, usize)> = vec![(r, 0)];
        visited[r as usize] = true;
        while let Some(&mut (node, ref mut ci)) = stack.last_mut() {
            let children = &succ[node as usize];
            if *ci < children.len() {
                let c = children[*ci];
                *ci += 1;
                if !visited[c as usize] {
                    visited[c as usize] = true;
                    stack.push((c, 0));
                }
            } else {
                post.push(node);
                stack.pop();
            }
        }
    }
    post.reverse();
    post
}

#[cfg(test)]
mod tests {
    use super::*;

    fn succ_of(n: usize, edges: &[(u32, u32)]) -> Vec<Vec<u32>> {
        let mut s = vec![Vec::new(); n];
        for &(a, b) in edges {
            s[a as usize].push(b);
        }
        s
    }

    #[test]
    fn dom_chain() {
        let s = succ_of(4, &[(0, 1), (1, 2), (2, 3)]);
        assert_eq!(dominators(4, &s, &[0]), vec![-1, 0, 1, 2]);
    }

    #[test]
    fn dom_diamond() {
        // the join 3 is dominated by 0, not by 1 or 2.
        let s = succ_of(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        assert_eq!(dominators(4, &s, &[0]), vec![-1, 0, 0, 0]);
    }

    #[test]
    fn dom_cycle() {
        // 0 -> 1 -> 2 -> 1 back edge.
        let s = succ_of(3, &[(0, 1), (1, 2), (2, 1)]);
        assert_eq!(dominators(3, &s, &[0]), vec![-1, 0, 1]);
    }

    fn fold_retained(n: usize, edges: &[(u32, u32)], sizes: &[u64]) -> Vec<u64> {
        let s = succ_of(n, edges);
        let roots: Vec<u32> = (0..n as u32)
            .filter(|&i| !edges.iter().any(|&(_, b)| b == i))
            .collect();
        let idom = dominators(n, &s, &roots);
        let order = reverse_postorder(n, &s, &roots);
        let mut retained = sizes.to_vec();
        for &node in order.iter().rev() {
            let d = idom[node as usize];
            if d >= 0 {
                retained[d as usize] += retained[node as usize];
            }
        }
        retained
    }

    #[test]
    fn retained_chain() {
        assert_eq!(fold_retained(4, &[(0, 1), (1, 2), (2, 3)], &[1, 1, 1, 1]), vec![4, 3, 2, 1]);
    }

    #[test]
    fn retained_diamond_join_not_double_counted() {
        let r = fold_retained(4, &[(0, 1), (0, 2), (1, 3), (2, 3)], &[10, 10, 10, 10]);
        assert_eq!(r[0], 40); // root dominates all
        assert_eq!(r[3], 10); // shared join retains only itself
        assert_eq!(r[1], 10);
        assert_eq!(r[2], 10);
    }
}
