//! Graph algorithms that produce per-node scalar/label values, used to color
//! nodes in the viewer. Everything here is written to scale: union-find for
//! components, CSR-based iteration for the rest, rayon where it pays off.

use crate::graph::{Graph, NodeId};
use rayon::prelude::*;

/// Weakly-connected components via union-find with path halving + union by size.
/// Returns a component label per node (labels are the representative node id).
pub fn connected_components(g: &Graph) -> Vec<u32> {
    let n = g.num_nodes() as usize;
    let mut parent: Vec<u32> = (0..n as u32).collect();
    let mut size: Vec<u32> = vec![1; n];

    #[inline]
    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            // Path halving.
            let gp = parent[parent[x as usize] as usize];
            parent[x as usize] = gp;
            x = gp;
        }
        x
    }

    for &[a, b] in g.edges() {
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        if ra != rb {
            let (small, big) = if size[ra as usize] < size[rb as usize] {
                (ra, rb)
            } else {
                (rb, ra)
            };
            parent[small as usize] = big;
            size[big as usize] += size[small as usize];
        }
    }

    // Flatten to canonical roots in parallel.
    let parent_snapshot = parent.clone();
    let mut label = vec![0u32; n];
    label.par_iter_mut().enumerate().for_each(|(i, l)| {
        // Follow to root using the (mostly compressed) snapshot.
        let mut x = i as u32;
        while parent_snapshot[x as usize] != x {
            x = parent_snapshot[x as usize];
        }
        *l = x;
    });
    label
}

/// Number of distinct components (helper for stats).
pub fn count_components(labels: &[u32]) -> usize {
    use ahash::AHashSet;
    let mut set = AHashSet::with_capacity(1024);
    for &l in labels {
        set.insert(l);
    }
    set.len()
}

/// BFS distance from `source`. Unreachable nodes get `u32::MAX`.
pub fn bfs_distances(g: &mut Graph, source: NodeId) -> Vec<u32> {
    let n = g.num_nodes() as usize;
    let csr = g.csr();
    let mut dist = vec![u32::MAX; n];
    dist[source as usize] = 0;
    let mut frontier = vec![source];
    let mut d = 0u32;
    while !frontier.is_empty() {
        d += 1;
        let mut next = Vec::new();
        for &u in &frontier {
            for &v in csr.neighbors(u) {
                if dist[v as usize] == u32::MAX {
                    dist[v as usize] = d;
                    next.push(v);
                }
            }
        }
        frontier = next;
    }
    dist
}

/// PageRank (undirected treated as symmetric). `iters` power iterations,
/// `damping` typically 0.85. Returns a normalized rank per node.
pub fn pagerank(g: &mut Graph, iters: u32, damping: f32) -> Vec<f32> {
    let n = g.num_nodes() as usize;
    let csr = g.csr();
    let inv_n = 1.0 / n as f32;
    let mut rank = vec![inv_n; n];
    let mut next = vec![0f32; n];
    let deg: Vec<u32> = (0..n as u32).map(|i| csr.degree(i)).collect();

    for _ in 0..iters {
        let base = (1.0 - damping) * inv_n;
        // Scatter contributions: parallel over target nodes by gathering from
        // neighbors (pull model — race-free).
        next.par_iter_mut().enumerate().for_each(|(i, out)| {
            let mut acc = 0f32;
            for &v in csr.neighbors(i as NodeId) {
                let dv = deg[v as usize];
                if dv > 0 {
                    acc += rank[v as usize] / dv as f32;
                }
            }
            *out = base + damping * acc;
        });
        // Redistribute dangling mass (degree-0 nodes) uniformly to preserve sum.
        let dangling: f32 = (0..n)
            .into_par_iter()
            .map(|i| if deg[i] == 0 { rank[i] } else { 0.0 })
            .sum();
        let add = damping * dangling * inv_n;
        next.par_iter_mut().for_each(|r| *r += add);
        std::mem::swap(&mut rank, &mut next);
    }
    rank
}

/// Greedy graph coloring (largest-degree-first ordering). Returns a color index
/// per node; adjacent nodes get distinct colors. Not optimal (NP-hard) but a
/// good, fast proper coloring — useful both as a visual and as a real result.
pub fn greedy_coloring(g: &mut Graph) -> Vec<u32> {
    let n = g.num_nodes() as usize;
    let csr = g.csr();
    let mut order: Vec<u32> = (0..n as u32).collect();
    // Largest degree first tends to reduce color count.
    order.par_sort_unstable_by_key(|&i| std::cmp::Reverse(csr.degree(i)));

    let mut color = vec![u32::MAX; n];
    let mut forbidden = vec![u32::MAX; 64]; // reused scratch, grown as needed
    for &u in &order {
        // Mark colors used by neighbors.
        for &v in csr.neighbors(u) {
            let c = color[v as usize];
            if c != u32::MAX {
                if (c as usize) >= forbidden.len() {
                    forbidden.resize(c as usize + 1, u32::MAX);
                }
                forbidden[c as usize] = u;
            }
        }
        // First color not forbidden by a neighbor in this pass.
        let mut c = 0u32;
        while (c as usize) < forbidden.len() && forbidden[c as usize] == u {
            c += 1;
        }
        color[u as usize] = c;
    }
    color
}

/// Degree centrality (raw undirected degree per node).
pub fn degree_centrality(g: &Graph) -> Vec<u32> {
    g.degrees()
}

/// Label propagation community detection. Fast, parallel-friendly heuristic:
/// each node adopts the most frequent label among its neighbors. `iters` sweeps.
pub fn label_propagation(g: &mut Graph, iters: u32) -> Vec<u32> {
    let n = g.num_nodes() as usize;
    let csr = g.csr();
    let mut labels: Vec<u32> = (0..n as u32).collect();
    for _ in 0..iters {
        // Synchronous update into a fresh buffer (deterministic, parallel).
        let next: Vec<u32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let nb = csr.neighbors(i as NodeId);
                if nb.is_empty() {
                    return labels[i];
                }
                // Plurality vote via a small hash map (neighbor counts).
                use ahash::AHashMap;
                let mut counts: AHashMap<u32, u32> = AHashMap::with_capacity(nb.len());
                for &v in nb {
                    *counts.entry(labels[v as usize]).or_insert(0) += 1;
                }
                // Argmax; tie-break to the smallest label for determinism.
                let mut best = labels[i];
                let mut best_c = 0u32;
                for (&lab, &c) in &counts {
                    if c > best_c || (c == best_c && lab < best) {
                        best = lab;
                        best_c = c;
                    }
                }
                best
            })
            .collect();
        if next == labels {
            break;
        }
        labels = next;
    }
    labels
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate;

    #[test]
    fn components_two_islands() {
        // 0-1-2  and  3-4
        let g = Graph::new(5, vec![[0, 1], [1, 2], [3, 4]]);
        let labels = connected_components(&g);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_ne!(labels[0], labels[3]);
        assert_eq!(count_components(&labels), 2);
    }

    #[test]
    fn bfs_line() {
        let mut g = Graph::new(4, vec![[0, 1], [1, 2], [2, 3]]);
        let d = bfs_distances(&mut g, 0);
        assert_eq!(d, vec![0, 1, 2, 3]);
    }

    #[test]
    fn coloring_is_proper() {
        let mut g = generate::grid_2d(20, 20);
        let colors = greedy_coloring(&mut g);
        // Verify no edge is monochromatic.
        for &[a, b] in g.edges() {
            assert_ne!(colors[a as usize], colors[b as usize]);
        }
    }

    #[test]
    fn pagerank_sums_to_one() {
        let mut g = generate::barabasi_albert(2000, 3, 1);
        let pr = pagerank(&mut g, 30, 0.85);
        let sum: f32 = pr.iter().sum();
        assert!((sum - 1.0).abs() < 1e-2, "pagerank sum was {sum}");
    }
}
