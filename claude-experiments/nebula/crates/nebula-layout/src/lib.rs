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
