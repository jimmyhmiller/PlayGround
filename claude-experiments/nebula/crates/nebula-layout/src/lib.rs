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

/// Layered (Sugiyama-style) layout for directed / DAG-like graphs such as task
/// graphs. Treats each stored edge `[u, v]` as a directed edge `u → v` (the
/// canonical edge list preserves insertion order, so parent→child direction
/// survives even though the graph is otherwise undirected).
///
/// Three classic stages:
///   1. **Layer assignment** by longest path from the roots (in-degree 0),
///      falling back to a BFS layer for nodes trapped in cycles.
///   2. **Crossing reduction** via a few barycenter (median-of-neighbors)
///      ordering sweeps, down then up.
///   3. **Coordinate assignment**: `y` from the layer, `x` from the in-layer
///      order, each layer centered on the origin.
pub struct LayeredLayout {
    /// World-space height of the whole layout (roots at top, leaves at bottom).
    pub target_height: f32,
    /// Width-to-height ratio of the layout's bounding box. Layers are scaled to
    /// this so a very wide/shallow DAG stays viewable instead of a flat band.
    pub aspect: f32,
    /// Barycenter ordering sweeps (more = fewer crossings, slower).
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
        use std::collections::VecDeque;
        let n = graph.num_nodes() as usize;
        if n == 0 {
            return;
        }
        let edges = graph.edges();

        // Directed adjacency (u → v) + in-adjacency (parents) + in-degree.
        let mut out: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut parents: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut indeg = vec![0u32; n];
        for &[u, v] in edges {
            let (u, v) = (u as usize, v as usize);
            if u >= n || v >= n || u == v {
                continue;
            }
            out[u].push(v as u32);
            parents[v].push(u as u32);
            indeg[v] += 1;
        }

        // 1. Longest-path layering via Kahn's topological order.
        let mut layer = vec![0i32; n];
        let mut remaining = indeg.clone();
        let mut queue: VecDeque<usize> =
            (0..n).filter(|&i| remaining[i] == 0).collect();
        let mut processed = vec![false; n];
        while let Some(u) = queue.pop_front() {
            processed[u] = true;
            let lu = layer[u];
            for &v in &out[u] {
                let v = v as usize;
                if layer[v] < lu + 1 {
                    layer[v] = lu + 1;
                }
                remaining[v] -= 1;
                if remaining[v] == 0 {
                    queue.push_back(v);
                }
            }
        }
        // Cycle fallback: nodes never dequeued sit in a cycle. Place each just
        // below its deepest already-placed parent so it still flows downward.
        for i in 0..n {
            if !processed[i] {
                let deepest = parents[i]
                    .iter()
                    .filter(|&&p| processed[p as usize])
                    .map(|&p| layer[p as usize])
                    .max()
                    .unwrap_or(0);
                layer[i] = deepest + 1;
            }
        }

        // Bucket nodes into layers (initial order = node id).
        let max_layer = layer.iter().copied().max().unwrap_or(0) as usize;
        let mut layers: Vec<Vec<u32>> = vec![Vec::new(); max_layer + 1];
        for i in 0..n {
            layers[layer[i] as usize].push(i as u32);
        }

        // In-layer x index per node, initialized from the bucket order.
        let mut order = vec![0f32; n];
        for lnodes in &layers {
            for (i, &nid) in lnodes.iter().enumerate() {
                order[nid as usize] = i as f32;
            }
        }

        // 2. Barycenter crossing reduction: alternate downward (order by parent
        //    average) and upward (order by child average) sweeps.
        let barycenter = |nid: u32, neigh: &[Vec<u32>], order: &[f32]| -> f32 {
            let ns = &neigh[nid as usize];
            if ns.is_empty() {
                order[nid as usize]
            } else {
                ns.iter().map(|&m| order[m as usize]).sum::<f32>() / ns.len() as f32
            }
        };
        for s in 0..self.sweeps {
            let down = s % 2 == 0;
            let range: Vec<usize> = if down {
                (1..layers.len()).collect()
            } else {
                (0..layers.len().saturating_sub(1)).rev().collect()
            };
            for l in range {
                let neigh = if down { &parents } else { &out };
                let mut keyed: Vec<(f32, u32)> = layers[l]
                    .iter()
                    .map(|&nid| (barycenter(nid, neigh, &order), nid))
                    .collect();
                keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                layers[l] = keyed.iter().map(|(_, nid)| *nid).collect();
                for (i, &nid) in layers[l].iter().enumerate() {
                    order[nid as usize] = i as f32;
                }
            }
        }

        // 3. Coordinate assignment. Scale so the whole layout fits a box of the
        //    requested height and aspect: `y` from the layer (roots at top),
        //    `x` from the in-layer order with each layer centered on x=0.
        let widest = layers.iter().map(|l| l.len()).max().unwrap_or(1).max(1);
        let sy = self.target_height / (max_layer.max(1) as f32);
        let sx = (self.target_height * self.aspect) / widest as f32;
        for (l, lnodes) in layers.iter().enumerate() {
            let half = lnodes.len().saturating_sub(1) as f32 * 0.5;
            let y = -(l as f32) * sy;
            for (i, &nid) in lnodes.iter().enumerate() {
                pos[nid as usize] = [(i as f32 - half) * sx, y];
            }
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
