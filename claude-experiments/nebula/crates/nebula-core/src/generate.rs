//! Synthetic graph generators.
//!
//! These exist to stress the pipeline at scale and to produce structure worth
//! visualizing (hubs, communities, spatial clusters). All are designed to be
//! fast — the large ones use per-thread RNGs and parallel edge sampling so we
//! can conjure tens of millions of edges in a fraction of a second.

use crate::graph::{Graph, NodeId};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

/// 2D grid lattice of `w * h` nodes with 4-neighbour connectivity.
pub fn grid_2d(w: u32, h: u32) -> Graph {
    let n = w as u64 * h as u64;
    let mut edges = Vec::with_capacity((2 * n) as usize);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if x + 1 < w {
                edges.push([i, i + 1]);
            }
            if y + 1 < h {
                edges.push([i, i + w]);
            }
        }
    }
    Graph::new(n, edges)
}

/// Erdős–Rényi G(n, m): `m` uniformly-random edges (self-loops rejected,
/// duplicates allowed for speed at scale). Parallel sampling.
pub fn erdos_renyi_m(n: u64, m: u64, seed: u64) -> Graph {
    assert!(n >= 2, "need at least 2 nodes");
    let edges = par_sample_edges(m, seed, move |rng| {
        let a = rng.gen_range(0..n) as NodeId;
        let mut b = rng.gen_range(0..n) as NodeId;
        if a == b {
            b = ((b as u64 + 1) % n) as NodeId;
        }
        [a, b]
    });
    Graph::new(n, edges)
}

/// Barabási–Albert preferential attachment: scale-free graph with hubs.
/// Each new node attaches `m` edges to existing nodes chosen proportionally to
/// degree. Inherently sequential, but O(E) and cache-friendly.
///
/// Uses the classic "target array" trick: a growing array where each node id
/// appears once per incident edge, so a uniform pick is degree-proportional.
pub fn barabasi_albert(n: u64, m: u32, seed: u64) -> Graph {
    let m = m.max(1);
    assert!(n as u32 > m, "n must exceed m");
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    let est_edges = (n.saturating_sub(m as u64)) * m as u64;
    let mut edges: Vec<[NodeId; 2]> = Vec::with_capacity(est_edges as usize);
    // Repeated-node array for degree-proportional sampling.
    let mut repeated: Vec<NodeId> = Vec::with_capacity((est_edges * 2) as usize);

    // Seed core: a small complete-ish graph of the first m nodes.
    for i in 0..m {
        for j in (i + 1)..m {
            edges.push([i, j]);
            repeated.push(i);
            repeated.push(j);
        }
    }
    if repeated.is_empty() {
        // m == 1: seed with node 0 so the first pick has something to hit.
        repeated.push(0);
    }

    for new_node in m..(n as NodeId) {
        // Choose m distinct existing targets by sampling the repeated array.
        let mut chosen: [NodeId; 32] = [0; 32];
        let mm = m as usize;
        let mut count = 0usize;
        // Small m: linear distinctness check is fine (m typically <= 16).
        while count < mm {
            let t = repeated[rng.gen_range(0..repeated.len())];
            if chosen[..count].iter().all(|&c| c != t) {
                chosen[count] = t;
                count += 1;
            }
        }
        for &t in &chosen[..mm] {
            edges.push([new_node, t]);
            repeated.push(new_node);
            repeated.push(t);
        }
    }
    Graph::new(n, edges)
}

/// Stochastic block model: `k` communities of roughly equal size. Intra-community
/// edges are dense, inter-community sparse. Great for community-detection and
/// coloring demos. Parallel sampling; `m` total edges.
pub fn stochastic_blocks(n: u64, k: u32, m: u64, mix: f32, seed: u64) -> Graph {
    assert!(k >= 1);
    let k = k as u64;
    let block = n / k;
    let mix = mix.clamp(0.0, 1.0);
    let edges = par_sample_edges(m, seed, move |rng| {
        let a = rng.gen_range(0..n);
        let b = if rng.gen::<f32>() < mix {
            // Inter-community: any node.
            rng.gen_range(0..n)
        } else {
            // Intra-community: same block as `a`.
            let blk = (a / block).min(k - 1);
            let lo = blk * block;
            let hi = if blk == k - 1 { n } else { lo + block };
            rng.gen_range(lo..hi)
        };
        let b = if a == b { (b + 1) % n } else { b };
        [a as NodeId, b as NodeId]
    });
    Graph::new(n, edges)
}

/// Random geometric graph: scatter nodes in the unit square, connect pairs
/// within `radius`. Produces organic spatial clusters. Uses a uniform grid to
/// keep neighbor search near-linear.
pub fn random_geometric(n: u64, radius: f32, seed: u64) -> Graph {
    let n_us = n as usize;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut pos = Vec::with_capacity(n_us);
    for _ in 0..n_us {
        pos.push([rng.gen::<f32>(), rng.gen::<f32>()]);
    }

    // Grid buckets sized to the connection radius so we only test adjacent cells.
    let cells = (1.0 / radius).floor().max(1.0) as usize;
    let cell_of = |p: [f32; 2]| -> (usize, usize) {
        let cx = ((p[0] * cells as f32) as usize).min(cells - 1);
        let cy = ((p[1] * cells as f32) as usize).min(cells - 1);
        (cx, cy)
    };
    let mut buckets: Vec<Vec<NodeId>> = vec![Vec::new(); cells * cells];
    for (i, &p) in pos.iter().enumerate() {
        let (cx, cy) = cell_of(p);
        buckets[cy * cells + cx].push(i as NodeId);
    }

    let r2 = radius * radius;
    // Parallel over cells: each node checks its own + 8 neighbor cells, keeping
    // only pairs (i < j) to avoid duplicates.
    let edges: Vec<[NodeId; 2]> = (0..cells * cells)
        .into_par_iter()
        .flat_map_iter(|c| {
            let cx = c % cells;
            let cy = c / cells;
            let mut local = Vec::new();
            for &i in &buckets[c] {
                let pi = pos[i as usize];
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let nx = cx as i32 + dx;
                        let ny = cy as i32 + dy;
                        if nx < 0 || ny < 0 || nx >= cells as i32 || ny >= cells as i32 {
                            continue;
                        }
                        let nb = ny as usize * cells + nx as usize;
                        for &j in &buckets[nb] {
                            if j <= i {
                                continue;
                            }
                            let pj = pos[j as usize];
                            let ddx = pi[0] - pj[0];
                            let ddy = pi[1] - pj[1];
                            if ddx * ddx + ddy * ddy <= r2 {
                                local.push([i, j]);
                            }
                        }
                    }
                }
            }
            local
        })
        .collect();

    Graph::new(n, edges)
}

/// Sample `m` edges in parallel using independent per-chunk RNG streams derived
/// from `seed`. `f` maps an RNG to one edge.
fn par_sample_edges<F>(m: u64, seed: u64, f: F) -> Vec<[NodeId; 2]>
where
    F: Fn(&mut Xoshiro256PlusPlus) -> [NodeId; 2] + Sync + Send,
{
    let num_threads = rayon::current_num_threads().max(1);
    let per = (m / num_threads as u64).max(1);
    let ranges: Vec<(u64, u64)> = (0..num_threads as u64)
        .map(|t| {
            let start = t * per;
            let end = if t == num_threads as u64 - 1 { m } else { start + per };
            (start, end)
        })
        .collect();

    ranges
        .into_par_iter()
        .flat_map_iter(|(start, end)| {
            // Distinct stream per chunk via jump-seeded RNG.
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed ^ (start.wrapping_mul(0x9E3779B97F4A7C15)));
            (start..end).map(move |_| f(&mut rng))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_edge_count() {
        let g = grid_2d(4, 3);
        assert_eq!(g.num_nodes(), 12);
        // horizontal: 3*3=9, vertical: 4*2=8 -> 17
        assert_eq!(g.num_edges(), 17);
    }

    #[test]
    fn ba_is_connected_ish() {
        let mut g = barabasi_albert(1000, 3, 42);
        assert_eq!(g.num_nodes(), 1000);
        let deg = g.degrees();
        let max = *deg.iter().max().unwrap();
        // Scale-free: expect at least one strong hub.
        assert!(max > 20, "expected a hub, max degree was {max}");
        let _ = g.csr();
    }

    #[test]
    fn geometric_has_edges() {
        let g = random_geometric(5000, 0.03, 7);
        assert!(g.num_edges() > 0);
    }

    #[test]
    fn blocks_generate() {
        let g = stochastic_blocks(10_000, 5, 50_000, 0.05, 1);
        assert_eq!(g.num_nodes(), 10_000);
        assert!(g.num_edges() > 0);
    }
}
