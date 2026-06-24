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
//! ## What is walked
//! * `Box<T>` — one `T` at offset 0.
//! * `Rc<T>` / `Arc<T>` — one `T` after two `usize` ref counts (offset 16).
//! * `Vec<T>` / slices — `size/size_of::<T>()` elements (the backing buffer;
//!   the uninitialized tail past `len` is read but only yields an edge if it
//!   happens to hold a live address, which is filtered out almost always).
//! * `String` / `Vec<u8>` — leaf (no interior pointers).
//! * **HashMap / HashSet** — the hashbrown allocation is decoded: bucket count is
//!   recovered from the allocation size (`solve_hashbrown_buckets`), control
//!   bytes are read to find FULL buckets, and each entry `(K, V)` is walked.
//! * **Enums** — walked **discriminant-aware**: the active variant is read from
//!   memory (`LayoutIndex::collect_pointer_offsets`) and only its fields are
//!   followed, so inactive variants never produce false edges. Niche-optimized
//!   `Option<ptr>` works (None vs Some by the niche value).
//!
//! Roots are "not referenced by any tracked allocation" — a heap-only
//! approximation (true roots live on the stack/in statics, which we don't
//! track), but exactly what you want for retained-size/leak analysis.

use std::collections::HashMap;

use memscope_proto::{AllocShape, GraphEdge, GraphNode, HeapGraph};
use memscope_symbols::LayoutIndex;

/// One live allocation to place in the graph.
pub struct NodeInput {
    pub addr: u64,
    pub size: u64,
    pub type_name: Option<String>,
    pub shape: Option<AllocShape>,
}

pub use memscope_symbols::MemReader;

/// An in-process reader: reads our own address space directly. Safe given the
/// builder's bounds guarantee, but still uses unaligned reads.
pub struct InProcessReader;

impl MemReader for InProcessReader {
    #[inline]
    fn read_uint(&self, addr: u64, size: u64) -> Option<u64> {
        if addr == 0 {
            return None;
        }
        // SAFETY: the builder only requests addresses provably inside a live
        // allocation; that memory is mapped and owned by this process for the
        // duration of the (point-in-time) walk.
        unsafe {
            let p = addr as *const u8;
            Some(match size {
                1 => p.read_unaligned() as u64,
                2 => p.cast::<u16>().read_unaligned() as u64,
                4 => p.cast::<u32>().read_unaligned() as u64,
                _ => p.cast::<u64>().read_unaligned(),
            })
        }
    }
}

#[inline]
fn read_ptr(mem: &dyn MemReader, addr: u64) -> Option<u64> {
    mem.read_uint(addr, 8)
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

    let mut opaque_nodes = 0u32;
    let mut edges: Vec<GraphEdge> = Vec::new();
    let mut out_degree = vec![0u32; n];
    let mut in_degree = vec![0u32; n];

    for (i, node) in nodes.iter().enumerate() {
        let Some(ty) = node.type_name.as_deref() else {
            opaque_nodes += 1;
            continue;
        };

        // The byte offsets (relative to node.addr) at which an element instance
        // begins. Most shapes have one or a contiguous array of these; a
        // HashMap reads its control bytes to find the live buckets.
        let inst_bases: Vec<u64> = match instance_bases(node, layout, ty, mem) {
            InstanceBases::List(v) => v,
            InstanceBases::Opaque => {
                opaque_nodes += 1;
                continue;
            }
            InstanceBases::Leaf => continue,
        };

        for inst_base in inst_bases {
            // Reader-aware: resolves enum discriminants so only live variants
            // contribute pointer fields.
            let offsets = layout.collect_pointer_offsets(ty, node.addr + inst_base, mem);
            for rel in offsets {
                let off = inst_base + rel;
                if off + 8 > node.size {
                    continue;
                }
                let Some(p) = read_ptr(mem, node.addr + off) else {
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

/// Candidate hashbrown control-group widths: 16 (x86_64 SSE2 / aarch64 NEON) and
/// 8 (the generic `u64` group std uses on some builds). We try each and keep the
/// one whose computed layout matches the recorded allocation size exactly.
const HASHBROWN_GROUP_WIDTHS: [u64; 2] = [16, 8];

enum InstanceBases {
    /// Element instances begin at these byte offsets (relative to alloc start).
    List(Vec<u64>),
    /// A leaf buffer with no walkable interior (e.g. `String`/`Vec<u8>`).
    Leaf,
    /// Couldn't decode the interior — count this node as opaque.
    Opaque,
}

/// Where each element instance begins inside `node`, given its container shape.
fn instance_bases(
    node: &NodeInput,
    layout: &LayoutIndex,
    ty: &str,
    mem: &dyn MemReader,
) -> InstanceBases {
    let esz = layout.size_of(ty).unwrap_or(0);
    match node.shape {
        Some(AllocShape::Vec) => {
            if esz == 0 {
                InstanceBases::Opaque
            } else {
                let count = (node.size / esz).min(MAX_VEC_SLOTS);
                InstanceBases::List((0..count).map(|i| i * esz).collect())
            }
        }
        // Rc/Arc inner = { strong: usize, weak: usize, value: T }.
        Some(AllocShape::Rc) | Some(AllocShape::Arc) => InstanceBases::List(vec![16]),
        // u8 buffers have no interior pointers.
        Some(AllocShape::StringBuf) => InstanceBases::Leaf,
        Some(AllocShape::HashTable) => match hashbrown_entry_bases(node, layout, ty, esz, mem) {
            Some(bases) => InstanceBases::List(bases),
            None => InstanceBases::Opaque,
        },
        // Box and everything else: a single instance at offset 0.
        _ => InstanceBases::List(vec![0]),
    }
}

/// Decode a hashbrown table allocation: recover the bucket count from the
/// allocation size via hashbrown's layout formula, read the control bytes, and
/// return the byte offset of every FULL bucket's entry.
///
/// hashbrown lays the allocation out as `[entries..][padding][ctrl..]` where the
/// ctrl pointer (`= data_end`) sits at `align_up(entry_size * buckets,
/// ctrl_align)`, and bucket `i` lives at `data_end - (i+1)*entry_size` (entries
/// grow *down* from ctrl). Total size = `data_end + buckets + GROUP_WIDTH`.
fn hashbrown_entry_bases(
    node: &NodeInput,
    layout: &LayoutIndex,
    ty: &str,
    entry_size: u64,
    mem: &dyn MemReader,
) -> Option<Vec<u64>> {
    if entry_size == 0 {
        return None;
    }
    let entry_align = layout.align_of(ty).unwrap_or(entry_size.next_power_of_two().max(1));
    let (buckets, data_end) = solve_hashbrown_buckets(node.size, entry_size, entry_align)?;

    // Read control bytes; a bucket is FULL when the ctrl byte's top bit is 0.
    let mut bases = Vec::new();
    for i in 0..buckets {
        let ctrl = mem.read_uint(node.addr + data_end + i, 1)?;
        if ctrl & 0x80 == 0 {
            bases.push(data_end - (i + 1) * entry_size);
        }
    }
    Some(bases)
}

/// Solve hashbrown's `(group_width, power-of-two bucket count)` from the recorded
/// allocation size: `data_end = align_up(entry_size*buckets, max(align, gw))`,
/// `total = data_end + buckets + gw`. Returns `(buckets, data_end)` for the
/// group width whose layout matches `alloc_size` exactly.
fn solve_hashbrown_buckets(alloc_size: u64, entry_size: u64, entry_align: u64) -> Option<(u64, u64)> {
    if entry_size == 0 {
        return None;
    }
    for &gw in &HASHBROWN_GROUP_WIDTHS {
        let ctrl_align = entry_align.max(gw);
        let mut nb = 1u64;
        while nb <= (1u64 << 31) {
            let de = align_up(entry_size * nb, ctrl_align);
            let total = de + nb + gw;
            if total == alloc_size {
                return Some((nb, de));
            }
            if total > alloc_size {
                break;
            }
            nb <<= 1;
        }
    }
    None
}

#[inline]
fn align_up(x: u64, align: u64) -> u64 {
    debug_assert!(align.is_power_of_two());
    (x + align - 1) & !(align - 1)
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

    #[test]
    fn hashbrown_bucket_solve_real_case() {
        // The live serve HashMap<u64, Session>: entry (u64, Session) = 64 bytes,
        // align 8, allocation 532488 bytes -> 8192 buckets, data_end 524288,
        // using the 8-byte generic group (the +8 tail, not +16).
        assert_eq!(solve_hashbrown_buckets(532488, 64, 8), Some((8192, 524288)));
    }

    #[test]
    fn hashbrown_bucket_solve_group16() {
        // A table whose size only matches with the 16-byte SIMD group.
        // entry_size=8, align=8, buckets=4 -> data_end=align_up(32,16)=32,
        // total = 32 + 4 + 16 = 52.
        assert_eq!(solve_hashbrown_buckets(52, 8, 8), Some((4, 32)));
    }

    #[test]
    fn hashbrown_bucket_solve_no_match() {
        assert_eq!(solve_hashbrown_buckets(999, 64, 8), None);
    }
}
