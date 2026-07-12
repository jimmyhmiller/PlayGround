//! `nebula-layout` — pluggable graph layout algorithms (CPU side).
//!
//! The GPU force-directed layout lives in `nebula-render` (it needs wgpu), but
//! these CPU layouts do the essential jobs of *seeding* initial positions and
//! providing a correct, parallel reference implementation of force-directed
//! layout that runs anywhere.
//!
//! The key abstraction is [`Layout`] for one-shot placement and
//! [`IterativeLayout`] for physics-style layouts that converge over steps.

use nebula_core::{Graph, Pos};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

mod force;
pub use force::ForceDirectedCpu;

/// One-shot placement: fills `pos` with a position for every node.
pub trait Layout: Send + Sync {
    fn name(&self) -> &str;
    fn place(&self, graph: &Graph, pos: &mut [Pos], seed: u64);
}

/// Physics-style layout advanced one step at a time. `place` seeds it.
pub trait IterativeLayout: Layout {
    /// Advance the simulation one step, mutating positions in place.
    /// Returns the total movement this step (a convergence signal).
    fn step(&mut self, graph: &Graph, pos: &mut [Pos]) -> f32;

    /// Reset any internal state (velocities, temperature) for a fresh run.
    fn reset(&mut self);
}

/// Uniform-random placement in a centered square of side `extent`.
pub struct RandomLayout {
    pub extent: f32,
}

impl Default for RandomLayout {
    fn default() -> Self {
        RandomLayout { extent: 1000.0 }
    }
}

impl Layout for RandomLayout {
    fn name(&self) -> &str {
        "random"
    }
    fn place(&self, graph: &Graph, pos: &mut [Pos], seed: u64) {
        let half = self.extent * 0.5;
        // Parallel fill with per-chunk RNG streams.
        pos.par_chunks_mut(65536)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let mut rng =
                    Xoshiro256PlusPlus::seed_from_u64(seed ^ (ci as u64).wrapping_mul(0x9E3779B9));
                for p in chunk {
                    *p = [rng.gen_range(-half..half), rng.gen_range(-half..half)];
                }
            });
        let _ = graph;
    }
}

/// Places nodes evenly on a circle — cheap, and a pleasing starting point.
pub struct CircleLayout {
    pub radius: f32,
}

impl Default for CircleLayout {
    fn default() -> Self {
        CircleLayout { radius: 800.0 }
    }
}

impl Layout for CircleLayout {
    fn name(&self) -> &str {
        "circle"
    }
    fn place(&self, graph: &Graph, pos: &mut [Pos], _seed: u64) {
        let n = graph.num_nodes().max(1) as f32;
        let two_pi = std::f32::consts::TAU;
        pos.par_iter_mut().enumerate().for_each(|(i, p)| {
            let t = two_pi * (i as f32) / n;
            *p = [self.radius * t.cos(), self.radius * t.sin()];
        });
    }
}

/// A square grid — useful for lattice graphs and as a neutral seed.
pub struct GridLayout {
    pub spacing: f32,
}

impl Default for GridLayout {
    fn default() -> Self {
        GridLayout { spacing: 20.0 }
    }
}

impl Layout for GridLayout {
    fn name(&self) -> &str {
        "grid"
    }
    fn place(&self, graph: &Graph, pos: &mut [Pos], _seed: u64) {
        let n = graph.num_nodes();
        let cols = (n as f64).sqrt().ceil() as u64;
        let cols = cols.max(1);
        let half = (cols as f32) * self.spacing * 0.5;
        pos.par_iter_mut().enumerate().for_each(|(i, p)| {
            let i = i as u64;
            let x = (i % cols) as f32 * self.spacing - half;
            let y = (i / cols) as f32 * self.spacing - half;
            *p = [x, y];
        });
    }
}

/// Result of the shared Sugiyama pipeline: a rank (layer) and an aligned `x`
/// coordinate for every real node. Dummy nodes (inserted for long edges) are
/// used internally but not exposed.
struct Layered {
    /// Rank (layer index, 0 = roots) per real node.
    rank: Vec<i32>,
    /// Deepest rank.
    max_rank: i32,
    /// Aligned in-layer x coordinate (unit spacing) per real node.
    x: Vec<f32>,
    /// Fractional position within the node's layer in `[0, 1)` (order index /
    /// layer size, dummies included). Used by the radial layout as an angle so
    /// each ring spreads evenly around the full circle.
    order_frac: Vec<f32>,
}

/// The full Sugiyama pipeline, shared by the layered and radial layouts:
///   1. **Rank assignment** — longest-path (ASAP) seed, then optional tightening
///      by iteratively moving each node to the median of its neighbours within
///      its feasible `[maxparent+1, minchild-1]` window (a safe, edge-length-
///      reducing local search; approximates network-simplex balancing).
///   2. **Dummy nodes** — every edge spanning >1 rank is subdivided so the graph
///      is proper (edges only join adjacent ranks); this straightens long edges
///      and makes crossing counts honest.
///   3. **Ordering** — median heuristic sweeps (down/up) over reals + dummies.
///   4. **Coordinate assignment** — iterative median with a separation constraint
///      (left-to-right and right-to-left passes averaged, which provably keeps
///      the min gap), pulling each node under the median of its neighbours.
fn build_layered(graph: &Graph, tighten: bool, order_sweeps: u32, coord_iters: u32) -> Layered {
    use std::collections::VecDeque;
    let n = graph.num_nodes() as usize;
    if n == 0 {
        return Layered { rank: Vec::new(), max_rank: 0, x: Vec::new(), order_frac: Vec::new() };
    }

    // Directed adjacency (u → v) + parents + in-degree, from real edges.
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut parents: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut indeg = vec![0u32; n];
    for &[u, v] in graph.edges() {
        let (u, v) = (u as usize, v as usize);
        if u >= n || v >= n || u == v {
            continue;
        }
        out[u].push(v as u32);
        parents[v].push(u as u32);
        indeg[v] += 1;
    }

    // 1a. ASAP ranks via Kahn's topological order (longest path from roots).
    let mut rank = vec![0i32; n];
    let mut remaining = indeg.clone();
    let mut queue: VecDeque<usize> = (0..n).filter(|&i| remaining[i] == 0).collect();
    let mut processed = vec![false; n];
    while let Some(u) = queue.pop_front() {
        processed[u] = true;
        let ru = rank[u];
        for &v in &out[u] {
            let v = v as usize;
            if rank[v] < ru + 1 {
                rank[v] = ru + 1;
            }
            remaining[v] -= 1;
            if remaining[v] == 0 {
                queue.push_back(v);
            }
        }
    }
    // Cycle fallback: unprocessed nodes sit just below their deepest placed parent.
    for i in 0..n {
        if !processed[i] {
            let deepest = parents[i]
                .iter()
                .filter(|&&p| processed[p as usize])
                .map(|&p| rank[p as usize])
                .max()
                .unwrap_or(0);
            rank[i] = deepest + 1;
        }
    }

    // 1b. Tighten: move each node to the median neighbour rank, clamped to its
    //     feasible window so every edge stays strictly downward.
    if tighten {
        for _ in 0..6 {
            for v in 0..n {
                if parents[v].is_empty() && out[v].is_empty() {
                    continue;
                }
                let lo = parents[v]
                    .iter()
                    .map(|&p| rank[p as usize] + 1)
                    .max()
                    .unwrap_or(i32::MIN);
                let hi = out[v]
                    .iter()
                    .map(|&c| rank[c as usize] - 1)
                    .min()
                    .unwrap_or(i32::MAX);
                if lo > hi {
                    continue; // over-constrained (cycle); leave as-is
                }
                let mut ns: Vec<i32> = parents[v]
                    .iter()
                    .map(|&p| rank[p as usize])
                    .chain(out[v].iter().map(|&c| rank[c as usize]))
                    .collect();
                ns.sort_unstable();
                let med = ns[ns.len() / 2];
                rank[v] = med.clamp(lo, hi);
            }
        }
    }

    // Normalize ranks to start at 0.
    let min_rank = rank.iter().copied().min().unwrap_or(0);
    for r in &mut rank {
        *r -= min_rank;
    }
    let max_rank = rank.iter().copied().max().unwrap_or(0);

    // 2. Build a proper layered graph with dummy nodes. Node ids: 0..n real,
    //    n.. dummies. `up`/`down` hold adjacent-rank neighbours (incl dummies).
    let mut up: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut down: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut layers: Vec<Vec<u32>> = vec![Vec::new(); (max_rank + 1) as usize];
    for v in 0..n {
        layers[rank[v] as usize].push(v as u32);
    }
    let mut next_id = n as u32;
    for &[u0, v0] in graph.edges() {
        let (u, v) = (u0 as usize, v0 as usize);
        if u >= n || v >= n || u == v {
            continue;
        }
        let (ru, rv) = (rank[u], rank[v]);
        if rv <= ru {
            continue; // back/flat edge: ignore for layout geometry
        }
        if rv - ru == 1 {
            down[u].push(v as u32);
            up[v].push(u as u32);
        } else {
            // Subdivide u → v with a dummy at each intermediate rank.
            let mut prev = u as u32;
            for r in (ru + 1)..rv {
                let d = next_id;
                next_id += 1;
                up.push(Vec::new());
                down.push(Vec::new());
                layers[r as usize].push(d);
                down[prev as usize].push(d);
                up[d as usize].push(prev);
                prev = d;
            }
            down[prev as usize].push(v0);
            up[v].push(prev);
        }
    }
    let total = next_id as usize;

    // 3. Ordering: init to bucket order, then median heuristic sweeps.
    let mut order = vec![0f32; total];
    for l in &layers {
        for (i, &id) in l.iter().enumerate() {
            order[id as usize] = i as f32;
        }
    }
    let median = |ids: &[u32], key: &[f32]| -> Option<f32> {
        if ids.is_empty() {
            return None;
        }
        let mut v: Vec<f32> = ids.iter().map(|&m| key[m as usize]).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let m = v.len();
        Some(if m % 2 == 1 { v[m / 2] } else { (v[m / 2 - 1] + v[m / 2]) * 0.5 })
    };
    for s in 0..order_sweeps {
        let down_dir = s % 2 == 0;
        let range: Vec<usize> = if down_dir {
            (1..layers.len()).collect()
        } else {
            (0..layers.len().saturating_sub(1)).rev().collect()
        };
        for l in range {
            let neigh = if down_dir { &up } else { &down };
            let mut keyed: Vec<(f32, u32)> = layers[l]
                .iter()
                .map(|&id| (median(&neigh[id as usize], &order).unwrap_or(order[id as usize]), id))
                .collect();
            keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            layers[l] = keyed.iter().map(|(_, id)| *id).collect();
            for (i, &id) in layers[l].iter().enumerate() {
                order[id as usize] = i as f32;
            }
        }
    }

    // 4. Coordinate assignment: iterative median with a separation constraint.
    let sep = 1.0f32;
    let mut x = vec![0f32; total];
    for l in &layers {
        for (i, &id) in l.iter().enumerate() {
            x[id as usize] = i as f32 * sep;
        }
    }
    for it in 0..coord_iters {
        let down_dir = it % 2 == 0;
        let range: Vec<usize> = if down_dir {
            (0..layers.len()).collect()
        } else {
            (0..layers.len()).rev().collect()
        };
        for l in range {
            let ln = &layers[l];
            let cnt = ln.len();
            if cnt == 0 {
                continue;
            }
            let neigh = if down_dir { &up } else { &down };
            let desired: Vec<f32> = ln
                .iter()
                .map(|&id| median(&neigh[id as usize], &x).unwrap_or(x[id as usize]))
                .collect();
            // Enforce min separation while keeping the fixed order: a
            // left-to-right pass and a right-to-left pass, averaged. Each pass
            // guarantees consecutive gaps >= sep, so their average does too.
            let mut xl = desired.clone();
            for i in 1..cnt {
                if xl[i] < xl[i - 1] + sep {
                    xl[i] = xl[i - 1] + sep;
                }
            }
            let mut xr = desired.clone();
            for i in (0..cnt - 1).rev() {
                if xr[i] > xr[i + 1] - sep {
                    xr[i] = xr[i + 1] - sep;
                }
            }
            for i in 0..cnt {
                x[ln[i] as usize] = (xl[i] + xr[i]) * 0.5;
            }
        }
    }

    // Fractional in-layer position for the radial layout (real nodes only).
    let mut order_frac = vec![0f32; n];
    for l in &layers {
        let len = l.len().max(1) as f32;
        for &id in l {
            if (id as usize) < n {
                order_frac[id as usize] = order[id as usize] / len;
            }
        }
    }

    let mut xreal = vec![0f32; n];
    xreal.copy_from_slice(&x[..n]);
    Layered { rank, max_rank, x: xreal, order_frac }
}

/// Layered (Sugiyama-style) layout for directed / DAG-like graphs such as task
/// graphs. Treats each stored edge `[u, v]` as `u → v` (the canonical edge list
/// preserves insertion order, so parent→child direction survives). Roots sit at
/// the top; each deeper dependency layer is below and centered.
pub struct LayeredLayout {
    /// World-space height of the whole layout.
    pub target_height: f32,
    /// Width-to-height ratio of the bounding box (keeps wide DAGs viewable).
    pub aspect: f32,
    /// Median ordering sweeps (more = fewer crossings, slower).
    pub sweeps: u32,
}

impl Default for LayeredLayout {
    fn default() -> Self {
        LayeredLayout { target_height: 4000.0, aspect: 1.8, sweeps: 4 }
    }
}

impl Layout for LayeredLayout {
    fn name(&self) -> &str {
        "layered"
    }

    fn place(&self, graph: &Graph, pos: &mut [Pos], _seed: u64) {
        let n = graph.num_nodes() as usize;
        if n == 0 {
            return;
        }
        let lg = build_layered(graph, true, self.sweeps, 8);
        let (mut xmin, mut xmax) = (f32::MAX, f32::MIN);
        for v in 0..n {
            xmin = xmin.min(lg.x[v]);
            xmax = xmax.max(lg.x[v]);
        }
        let cx = (xmin + xmax) * 0.5;
        let xspan = (xmax - xmin).max(1.0);
        let sy = self.target_height / (lg.max_rank.max(1) as f32);
        let sx = (self.target_height * self.aspect) / xspan;
        for v in 0..n {
            pos[v] = [(lg.x[v] - cx) * sx, -(lg.rank[v] as f32) * sy];
        }
    }
}

/// Radial (concentric) DAG layout. Uses the same Sugiyama pipeline, then maps
/// each node to polar coordinates: **radius** grows with rank (root at the
/// center) and **angle** comes from the aligned x coordinate — so subtrees fan
/// out into angular wedges and children sit under their parents. Far better use
/// of screen space than horizontal bands for a wide, single-rooted DAG.
pub struct RadialLayout {
    /// Outer radius (radius of the deepest layer).
    pub radius: f32,
    /// Median ordering sweeps.
    pub sweeps: u32,
}

impl Default for RadialLayout {
    fn default() -> Self {
        RadialLayout { radius: 3000.0, sweeps: 4 }
    }
}

impl Layout for RadialLayout {
    fn name(&self) -> &str {
        "radial"
    }

    fn place(&self, graph: &Graph, pos: &mut [Pos], _seed: u64) {
        let n = graph.num_nodes() as usize;
        if n == 0 {
            return;
        }
        let lg = build_layered(graph, true, self.sweeps, 8);
        let ring = self.radius / (lg.max_rank.max(1) as f32);
        for v in 0..n {
            let ang = std::f32::consts::TAU * lg.order_frac[v];
            let rad = (lg.rank[v] as f32 + 0.4) * ring;
            pos[v] = [rad * ang.cos(), rad * ang.sin()];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nebula_core::generate;

    #[test]
    fn random_layout_fills_within_bounds() {
        let g = generate::grid_2d(10, 10);
        let mut pos = vec![[0.0f32; 2]; g.num_nodes() as usize];
        RandomLayout { extent: 100.0 }.place(&g, &mut pos, 1);
        for p in &pos {
            assert!(p[0].abs() <= 50.0 && p[1].abs() <= 50.0);
        }
    }

    #[test]
    fn circle_layout_on_radius() {
        let g = generate::grid_2d(6, 6);
        let mut pos = vec![[0.0f32; 2]; g.num_nodes() as usize];
        CircleLayout { radius: 10.0 }.place(&g, &mut pos, 0);
        for p in &pos {
            let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert!((r - 10.0).abs() < 1e-3);
        }
    }
}
