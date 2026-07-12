//! Core graph representation.
//!
//! Design for scale: the canonical form is a flat, undirected **edge list**
//! (`Vec<[u32; 2]>`), which is exactly what the GPU spring-layout and the edge
//! renderer consume. Algorithms that need neighbor iteration build a **CSR**
//! (compressed sparse row) adjacency on demand; CSR construction is parallelized.
//!
//! Node ids are `u32` (up to ~4.29B nodes — covers the 1B target). Edge indices
//! are `u64` because a billion-node graph with a modest average degree easily
//! exceeds 4B directed entries.

use rayon::prelude::*;

pub type NodeId = u32;

/// An undirected graph stored as a flat edge list, with a lazily-built CSR.
pub struct Graph {
    num_nodes: u64,
    /// Canonical undirected edge list — each edge stored once.
    edges: Vec<[NodeId; 2]>,
    /// Cached CSR adjacency (built on demand). Directed doubling of `edges`.
    csr: Option<Csr>,
}

/// Compressed-sparse-row adjacency. For an undirected graph each edge appears
/// twice (once per endpoint) so neighbor iteration is symmetric.
pub struct Csr {
    /// `offsets[i]..offsets[i+1]` indexes into `targets` for node `i`.
    /// Length `num_nodes + 1`.
    pub offsets: Vec<u64>,
    /// Concatenated neighbor lists. Length = 2 * num_edges (undirected).
    pub targets: Vec<NodeId>,
}

impl Csr {
    #[inline]
    pub fn neighbors(&self, node: NodeId) -> &[NodeId] {
        let s = self.offsets[node as usize] as usize;
        let e = self.offsets[node as usize + 1] as usize;
        &self.targets[s..e]
    }

    #[inline]
    pub fn degree(&self, node: NodeId) -> u32 {
        (self.offsets[node as usize + 1] - self.offsets[node as usize]) as u32
    }
}

impl Graph {
    pub fn new(num_nodes: u64, edges: Vec<[NodeId; 2]>) -> Self {
        Graph { num_nodes, edges, csr: None }
    }

    /// An empty graph with `num_nodes` isolated vertices.
    pub fn empty(num_nodes: u64) -> Self {
        Graph { num_nodes, edges: Vec::new(), csr: None }
    }

    #[inline]
    pub fn num_nodes(&self) -> u64 {
        self.num_nodes
    }

    #[inline]
    pub fn num_edges(&self) -> u64 {
        self.edges.len() as u64
    }

    #[inline]
    pub fn edges(&self) -> &[[NodeId; 2]] {
        &self.edges
    }

    /// Flattened `[src0, dst0, src1, dst1, ...]` view for GPU upload.
    pub fn edges_flat(&self) -> &[u32] {
        bytemuck::cast_slice(&self.edges)
    }

    /// Build (or return cached) CSR adjacency. Parallel counting-sort style
    /// construction: O(N + E) work, scales across cores.
    pub fn csr(&mut self) -> &Csr {
        if self.csr.is_none() {
            self.csr = Some(build_csr(self.num_nodes, &self.edges));
        }
        self.csr.as_ref().unwrap()
    }

    /// Borrow an already-built CSR without mutation, if present.
    pub fn csr_ref(&self) -> Option<&Csr> {
        self.csr.as_ref()
    }

    /// Build a fresh CSR without caching it (useful when a consumer wants to own
    /// adjacency independently of the graph's lifetime, e.g. a layout engine).
    pub fn compute_csr(&self) -> Csr {
        build_csr(self.num_nodes, &self.edges)
    }

    pub fn ensure_csr(&mut self) {
        let _ = self.csr();
    }

    /// Per-node degree (undirected). Cheap even without CSR.
    pub fn degrees(&self) -> Vec<u32> {
        let mut deg = vec![0u32; self.num_nodes as usize];
        // Sequential to avoid atomics; degree counting is memory-bound anyway.
        for &[a, b] in &self.edges {
            deg[a as usize] = deg[a as usize].saturating_add(1);
            deg[b as usize] = deg[b as usize].saturating_add(1);
        }
        deg
    }
}

/// Parallel CSR construction from an undirected edge list.
fn build_csr(num_nodes: u64, edges: &[[NodeId; 2]]) -> Csr {
    let n = num_nodes as usize;

    // 1. Degree histogram (undirected: count both endpoints).
    //    Use per-chunk local histograms then reduce, to stay lock-free.
    let num_threads = rayon::current_num_threads().max(1);
    let chunk = edges.len().div_ceil(num_threads).max(1);
    let partials: Vec<Vec<u64>> = edges
        .par_chunks(chunk)
        .map(|c| {
            let mut local = vec![0u64; n];
            for &[a, b] in c {
                local[a as usize] += 1;
                local[b as usize] += 1;
            }
            local
        })
        .collect();

    let mut degree = vec![0u64; n];
    // Reduce partial histograms in parallel over node ranges.
    degree
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, d)| {
            let mut sum = 0u64;
            for p in &partials {
                sum += p[i];
            }
            *d = sum;
        });

    // 2. Prefix sum -> offsets (length n+1). Sequential scan (fast, O(N)).
    let mut offsets = vec![0u64; n + 1];
    let mut acc = 0u64;
    for i in 0..n {
        offsets[i] = acc;
        acc += degree[i];
    }
    offsets[n] = acc;

    // 3. Scatter targets. Use a per-node cursor initialized to offsets.
    //    Sequential scatter (random writes; parallelizing needs atomics and
    //    rarely helps due to memory bandwidth). Cursor is a moving write head.
    let total = acc as usize;
    let mut targets = vec![0u32; total];
    let mut cursor: Vec<u64> = offsets[..n].to_vec();
    for &[a, b] in edges {
        let ia = cursor[a as usize];
        targets[ia as usize] = b;
        cursor[a as usize] = ia + 1;
        let ib = cursor[b as usize];
        targets[ib as usize] = a;
        cursor[b as usize] = ib + 1;
    }

    Csr { offsets, targets }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_neighbors_symmetric() {
        // Triangle 0-1-2 plus a pendant 2-3.
        let mut g = Graph::new(4, vec![[0, 1], [1, 2], [2, 0], [2, 3]]);
        let csr = g.csr();
        let mut n0: Vec<_> = csr.neighbors(0).to_vec();
        n0.sort();
        assert_eq!(n0, vec![1, 2]);
        let mut n2: Vec<_> = csr.neighbors(2).to_vec();
        n2.sort();
        assert_eq!(n2, vec![0, 1, 3]);
        assert_eq!(csr.degree(2), 3);
        assert_eq!(csr.degree(3), 1);
    }

    #[test]
    fn degrees_match_csr() {
        let mut g = Graph::new(4, vec![[0, 1], [1, 2], [2, 0], [2, 3]]);
        let deg = g.degrees();
        let csr = g.csr();
        for i in 0..4u32 {
            assert_eq!(deg[i as usize], csr.degree(i));
        }
    }
}
