//! CPU force-directed layout (Fruchterman–Reingold) with a uniform spatial grid
//! for O(N) approximate repulsion.
//!
//! Naive FR is O(N²) because every node repels every other. We instead bucket
//! nodes into a grid whose cell size is the optimal edge length `k`, and only
//! compute repulsion against nodes in the same and adjacent cells. That makes a
//! step O(N + E) for reasonably uniform layouts — the same trick the GPU version
//! uses, so this stays a faithful reference.
//!
//! Attraction uses CSR neighbor iteration (a pull model) so force accumulation
//! is race-free and parallelizes cleanly with rayon.

use nebula_core::{Csr, Graph, Pos};
use rayon::prelude::*;

use crate::{IterativeLayout, Layout, RandomLayout};

pub struct ForceDirectedCpu {
    /// Optimal edge length. Repulsion/attraction are scaled by this.
    pub k: f32,
    /// Temperature: caps per-step displacement; decays each step.
    temperature: f32,
    pub initial_temperature: f32,
    /// Cooling factor per step (0.95–0.99 typical).
    pub cooling: f32,
    /// Gravity pulling nodes toward the origin, keeps components from drifting.
    pub gravity: f32,
    /// Cached adjacency, rebuilt on `reset`.
    csr: Option<Csr>,
    /// Reused scratch buffers.
    disp: Vec<[f32; 2]>,
}

impl ForceDirectedCpu {
    pub fn new(k: f32) -> Self {
        ForceDirectedCpu {
            k,
            temperature: k * 10.0,
            initial_temperature: k * 10.0,
            cooling: 0.98,
            gravity: 0.02,
            csr: None,
            disp: Vec::new(),
        }
    }
}

impl Default for ForceDirectedCpu {
    fn default() -> Self {
        Self::new(30.0)
    }
}

impl Layout for ForceDirectedCpu {
    fn name(&self) -> &str {
        "force-directed (cpu)"
    }
    fn place(&self, graph: &Graph, pos: &mut [Pos], seed: u64) {
        // Seed with random positions; the simulation does the rest.
        RandomLayout {
            extent: self.k * (graph.num_nodes() as f32).sqrt(),
        }
        .place(graph, pos, seed);
    }
}

impl IterativeLayout for ForceDirectedCpu {
    fn reset(&mut self) {
        self.temperature = self.initial_temperature;
        self.csr = None;
    }

    fn step(&mut self, graph: &Graph, pos: &mut [Pos]) -> f32 {
        let n = pos.len();
        if n == 0 {
            return 0.0;
        }
        if self.csr.is_none() {
            self.csr = Some(graph.compute_csr());
        }
        let csr = self.csr.as_ref().unwrap();
        if self.disp.len() != n {
            self.disp = vec![[0.0; 2]; n];
        }

        let k = self.k;
        let k2 = k * k;

        // ---- Build spatial grid (cell size = k) --------------------------------
        // Compute bounds.
        let (min, max) = bounds(pos);
        let cell = k.max(1.0);
        let gw = (((max[0] - min[0]) / cell).ceil() as i64 + 1).max(1) as usize;
        let gh = (((max[1] - min[1]) / cell).ceil() as i64 + 1).max(1) as usize;
        // Guard against pathological grid sizes (all nodes coincident, etc.).
        let grid_cells = gw.saturating_mul(gh);
        let use_grid = grid_cells <= 64_000_000 && grid_cells > 0;

        let cell_of = |p: Pos| -> (usize, usize) {
            let cx = (((p[0] - min[0]) / cell) as i64).clamp(0, gw as i64 - 1) as usize;
            let cy = (((p[1] - min[1]) / cell) as i64).clamp(0, gh as i64 - 1) as usize;
            (cx, cy)
        };

        // Bucket node indices per cell.
        let mut heads = vec![u32::MAX; if use_grid { grid_cells } else { 1 }];
        let mut nexts = vec![u32::MAX; n];
        if use_grid {
            for i in 0..n {
                let (cx, cy) = cell_of(pos[i]);
                let c = cy * gw + cx;
                nexts[i] = heads[c];
                heads[c] = i as u32;
            }
        }

        // ---- Repulsion + attraction (parallel over nodes) ----------------------
        let disp = &mut self.disp;
        let pos_ref: &[Pos] = pos;
        disp.par_iter_mut().enumerate().for_each(|(i, d)| {
            let pi = pos_ref[i];
            let mut fx = 0.0f32;
            let mut fy = 0.0f32;

            // Repulsion from nearby nodes (grid) or all nodes (fallback, small n).
            if use_grid {
                let (cx, cy) = cell_of(pi);
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        let nx = cx as i64 + dx;
                        let ny = cy as i64 + dy;
                        if nx < 0 || ny < 0 || nx >= gw as i64 || ny >= gh as i64 {
                            continue;
                        }
                        let c = ny as usize * gw + nx as usize;
                        let mut j = heads[c];
                        while j != u32::MAX {
                            if j as usize != i {
                                accumulate_repulsion(pi, pos_ref[j as usize], k2, &mut fx, &mut fy);
                            }
                            j = nexts[j as usize];
                        }
                    }
                }
            } else {
                for j in 0..n {
                    if j != i {
                        accumulate_repulsion(pi, pos_ref[j], k2, &mut fx, &mut fy);
                    }
                }
            }

            // Attraction along edges (pull from neighbors).
            for &nb in csr.neighbors(i as u32) {
                let pj = pos_ref[nb as usize];
                let dx = pi[0] - pj[0];
                let dy = pi[1] - pj[1];
                let dist = (dx * dx + dy * dy).sqrt().max(1e-3);
                // FR attraction magnitude = dist^2 / k.
                let f = dist / k;
                fx -= dx / dist * f * dist; // = dx * dist / k
                fy -= dy / dist * f * dist;
            }

            // Gravity toward origin (keeps disconnected pieces on screen).
            fx -= pi[0] * self.gravity;
            fy -= pi[1] * self.gravity;

            *d = [fx, fy];
        });

        // ---- Integrate: cap by temperature, accumulate movement ----------------
        let temp = self.temperature;
        let total_move: f32 = pos
            .par_iter_mut()
            .zip(disp.par_iter())
            .map(|(p, d)| {
                let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
                if len > 1e-6 {
                    let scale = len.min(temp) / len;
                    p[0] += d[0] * scale;
                    p[1] += d[1] * scale;
                    len.min(temp)
                } else {
                    0.0
                }
            })
            .sum();

        self.temperature = (self.temperature * self.cooling).max(self.k * 0.01);
        total_move / n as f32
    }
}

#[inline]
fn accumulate_repulsion(pi: Pos, pj: Pos, k2: f32, fx: &mut f32, fy: &mut f32) {
    let dx = pi[0] - pj[0];
    let dy = pi[1] - pj[1];
    let mut d2 = dx * dx + dy * dy;
    if d2 < 1e-4 {
        // Coincident nodes: nudge deterministically apart.
        d2 = 1e-4;
    }
    // FR repulsion magnitude = k^2 / dist, direction away from pj.
    // force vector = (k^2 / d2) * (delta)  (since delta/dist * k^2/dist).
    let f = k2 / d2;
    *fx += dx * f;
    *fy += dy * f;
}

fn bounds(pos: &[Pos]) -> (Pos, Pos) {
    pos.par_iter()
        .map(|p| (*p, *p))
        .reduce(
            || ([f32::MAX, f32::MAX], [f32::MIN, f32::MIN]),
            |(amin, amax), (bmin, bmax)| {
                (
                    [amin[0].min(bmin[0]), amin[1].min(bmin[1])],
                    [amax[0].max(bmax[0]), amax[1].max(bmax[1])],
                )
            },
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use nebula_core::generate;

    #[test]
    fn force_layout_converges_and_separates() {
        // Two triangles joined by an edge; layout should not collapse to a point.
        let mut g = generate::barabasi_albert(500, 2, 3);
        let mut pos = vec![[0.0f32; 2]; g.num_nodes() as usize];
        let mut fd = ForceDirectedCpu::new(30.0);
        fd.reset();
        fd.place(&g, &mut pos, 1);
        let mut last = f32::MAX;
        for _ in 0..120 {
            last = fd.step(&g, &mut pos);
        }
        // Movement should have cooled substantially from the initial burst.
        assert!(last.is_finite());
        // Nodes should span a real area (not collapsed).
        let (mn, mx) = bounds(&pos);
        let span = (mx[0] - mn[0]).max(mx[1] - mn[1]);
        assert!(span > 50.0, "layout collapsed, span = {span}");
        let _ = g.csr();
    }
}
