//! Procedurally *generated* world terrain — not just sampled noise.
//!
//! At world-gen we build a heightfield grid and run real terrain processes on it:
//!   1. domain-warped fractal noise for organic base shapes,
//!   2. an **edge falloff** so the world is a continent ringed by ocean,
//!   3. **thermal erosion** passes that relax steep slopes into ridges + talus,
//!   4. **rivers traced by gradient descent** from the highlands, carving their
//!      beds downhill until they reach the sea (tributaries merge into valleys),
//!   5. a precomputed **hillshade** (sun-relief) so the land looks 3-D.
//!
//! All deterministic from a fixed seed, computed once at startup (~tens of ms),
//! then sampled with cheap bilinear lookups — so per-frame rendering is fast.

use crate::util::Rng;
use raylib::prelude::Vector2;
use std::collections::HashSet;

const GW: usize = 256;
const GH: usize = 220;
const ORIGIN_X: f32 = -3000.0;
const ORIGIN_Y: f32 = -3000.0;
const CELL: f32 = 58.0; // world units per grid cell
const BASE: f32 = 1500.0; // noise feature size (world units)
pub const SEA_LEVEL: f32 = 0.33;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Land {
    Sea,
    Sand,
    Grass,
    Hill,
    Mountain,
    Snow,
}

pub struct River {
    pub points: Vec<Vector2>,
    /// Width per point — rivers taper from a narrow source to a wide mouth.
    pub widths: Vec<f32>,
}

pub struct Terrain {
    height: Vec<f32>,
    shade: Vec<f32>,
    rivers: Vec<River>,
}

#[inline]
fn idx(i: usize, j: usize) -> usize {
    j * GW + i
}

impl Terrain {
    pub fn new(seed: u64) -> Terrain {
        let mut height = base_field(seed);
        thermal_erode(&mut height, 16);
        let rivers = trace_rivers(&mut height, seed);
        let shade = hillshade(&height);
        Terrain { height, shade, rivers }
    }

    /// Bilinear height in [0, 1] at a world point (edges clamp to the grid).
    pub fn height(&self, x: f32, y: f32) -> f32 {
        sample(&self.height, x, y)
    }

    /// Hillshade multiplier (~0.55 in shadow to ~1.25 on lit slopes).
    pub fn shade(&self, x: f32, y: f32) -> f32 {
        sample(&self.shade, x, y)
    }

    pub fn land_at(&self, x: f32, y: f32) -> Land {
        classify(self.height(x, y))
    }

    pub fn rivers(&self) -> &[River] {
        &self.rivers
    }

    pub fn river_dist(&self, p: Vector2) -> f32 {
        let mut best = f32::INFINITY;
        for r in &self.rivers {
            for w in r.points.windows(2) {
                best = best.min(pt_seg_dist(p, w[0], w[1]));
            }
        }
        best
    }

    pub fn is_buildable(&self, p: Vector2) -> bool {
        matches!(self.land_at(p.x, p.y), Land::Grass | Land::Hill | Land::Sand)
            && self.river_dist(p) > 60.0
    }
}

pub fn classify(h: f32) -> Land {
    match h {
        x if x < SEA_LEVEL => Land::Sea,
        x if x < 0.37 => Land::Sand,
        x if x < 0.58 => Land::Grass,
        x if x < 0.70 => Land::Hill,
        x if x < 0.84 => Land::Mountain,
        _ => Land::Snow,
    }
}

// ---- grid helpers ------------------------------------------------------------

fn cell_world(i: usize, j: usize) -> (f32, f32) {
    (ORIGIN_X + i as f32 * CELL, ORIGIN_Y + j as f32 * CELL)
}

/// Bilinear sample of a grid at a world point.
fn sample(grid: &[f32], x: f32, y: f32) -> f32 {
    let fx = ((x - ORIGIN_X) / CELL).clamp(0.0, (GW - 1) as f32);
    let fy = ((y - ORIGIN_Y) / CELL).clamp(0.0, (GH - 1) as f32);
    let i0 = fx.floor() as usize;
    let j0 = fy.floor() as usize;
    let i1 = (i0 + 1).min(GW - 1);
    let j1 = (j0 + 1).min(GH - 1);
    let tx = fx - i0 as f32;
    let ty = fy - j0 as f32;
    let a = grid[idx(i0, j0)];
    let b = grid[idx(i1, j0)];
    let c = grid[idx(i0, j1)];
    let d = grid[idx(i1, j1)];
    let top = a + (b - a) * tx;
    let bot = c + (d - c) * tx;
    top + (bot - top) * ty
}

// ---- generation --------------------------------------------------------------

fn base_field(seed: u64) -> Vec<f32> {
    let mut h = vec![0.0f32; GW * GH];
    let margin = 30.0;
    for j in 0..GH {
        for i in 0..GW {
            let (wx, wy) = cell_world(i, j);
            let nx = wx / BASE;
            let ny = wy / BASE;
            // Domain warp the sample point for organic, non-blobby coastlines.
            let wxx = nx + 0.55 * fbm(nx + 5.2, ny + 1.3, seed ^ 0xA1);
            let wyy = ny + 0.55 * fbm(nx + 9.1, ny + 4.7, seed ^ 0xB2);
            let mut v = fbm(wxx, wyy, seed);
            v = ((v - 0.5) * 1.75 + 0.5).clamp(0.0, 1.0); // contrast: bolder relief

            // Edge falloff -> a continent in an ocean.
            let de = (i.min(GW - 1 - i)).min(j.min(GH - 1 - j)) as f32;
            let fall = (de / margin).clamp(0.0, 1.0);
            let fall = fall * fall * (3.0 - 2.0 * fall); // smoothstep
            v = v * (0.25 + 0.75 * fall) - (1.0 - fall) * 0.28;

            h[idx(i, j)] = v.clamp(0.0, 1.0);
        }
    }
    h
}

/// Thermal erosion: shed material from slopes steeper than the talus angle into
/// the lowest neighbour. Relaxes noise into ridgelines and talus slopes.
fn thermal_erode(h: &mut [f32], passes: usize) {
    const TALUS: f32 = 0.018;
    let mut delta = vec![0.0f32; GW * GH];
    for _ in 0..passes {
        for d in delta.iter_mut() {
            *d = 0.0;
        }
        for j in 1..GH - 1 {
            for i in 1..GW - 1 {
                let c = idx(i, j);
                let hc = h[c];
                let mut lowest = c;
                let mut maxd = 0.0;
                for (di, dj) in NEIGH {
                    let n = idx((i as i32 + di) as usize, (j as i32 + dj) as usize);
                    let diff = hc - h[n];
                    if diff > maxd {
                        maxd = diff;
                        lowest = n;
                    }
                }
                if maxd > TALUS {
                    let mv = (maxd - TALUS) * 0.25;
                    delta[c] -= mv;
                    delta[lowest] += mv;
                }
            }
        }
        for k in 0..h.len() {
            h[k] = (h[k] + delta[k]).clamp(0.0, 1.0);
        }
    }
}

const NEIGH: [(i32, i32); 8] =
    [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)];

fn lowest_neighbour(h: &[f32], i: usize, j: usize) -> (usize, f32) {
    let mut best = idx(i, j);
    let mut bh = f32::INFINITY;
    for (di, dj) in NEIGH {
        let ni = i as i32 + di;
        let nj = j as i32 + dj;
        if ni < 0 || nj < 0 || ni >= GW as i32 || nj >= GH as i32 {
            continue;
        }
        let n = idx(ni as usize, nj as usize);
        if h[n] < bh {
            bh = h[n];
            best = n;
        }
    }
    (best, bh)
}

/// Trace rivers downhill from highland sources, carving their beds so later
/// tributaries fall into the same valleys. Width grows with drainage (how many
/// source paths share a cell), so trunks downstream are wide.
fn trace_rivers(h: &mut [f32], seed: u64) -> Vec<River> {
    // Candidate sources: random-ish highland cells, processed highest first.
    let mut rng = Rng(seed ^ 0xC0FFEE | 1);
    let mut sources: Vec<usize> = Vec::new();
    for _ in 0..520 {
        let i = 4 + (rng.next_u64() as usize % (GW - 8));
        let j = 4 + (rng.next_u64() as usize % (GH - 8));
        if h[idx(i, j)] > 0.6 {
            sources.push(idx(i, j));
        }
    }
    sources.sort_by(|&a, &b| h[b].partial_cmp(&h[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut visits = vec![0u32; GW * GH];
    let mut paths: Vec<Vec<usize>> = Vec::new();
    for &src in &sources {
        let mut cur = src;
        let mut path = vec![cur];
        let mut seen: HashSet<usize> = HashSet::new();
        seen.insert(cur);
        for _ in 0..600 {
            if h[cur] < SEA_LEVEL {
                break;
            }
            let (i, j) = (cur % GW, cur / GW);
            let (nb, nh) = lowest_neighbour(h, i, j);
            if nh >= h[cur] || !seen.insert(nb) {
                break; // fell into a basin (lake) or a loop
            }
            cur = nb;
            path.push(cur);
        }
        if path.len() >= 12 {
            for &c in &path {
                visits[c] += 1;
                // Carve the channel a touch so a valley forms around it.
                h[c] = (h[c] - 0.012).max(0.0);
            }
            paths.push(path);
        }
    }

    // Keep the longest few as the visible rivers.
    paths.sort_by_key(|p| std::cmp::Reverse(p.len()));
    paths.truncate(9);
    paths
        .into_iter()
        .map(|path| {
            let mut points: Vec<Vector2> = path
                .iter()
                .map(|&c| {
                    let (wx, wy) = cell_world(c % GW, c / GW);
                    Vector2::new(wx, wy)
                })
                .collect();
            let widths: Vec<f32> = path
                .iter()
                .map(|&c| (12.0 + (visits[c] as f32).sqrt() * 4.5).min(52.0))
                .collect();
            smooth(&mut points);
            River { points, widths }
        })
        .collect()
}

/// One light smoothing pass so river polylines aren't blocky.
fn smooth(pts: &mut [Vector2]) {
    if pts.len() < 3 {
        return;
    }
    let orig = pts.to_vec();
    for i in 1..pts.len() - 1 {
        pts[i] = Vector2::new(
            (orig[i - 1].x + orig[i].x * 2.0 + orig[i + 1].x) * 0.25,
            (orig[i - 1].y + orig[i].y * 2.0 + orig[i + 1].y) * 0.25,
        );
    }
}

/// Hillshade: surface normal dotted with a fixed sun direction.
fn hillshade(h: &[f32]) -> Vec<f32> {
    let mut shade = vec![1.0f32; GW * GH];
    // Sun from the upper-left, fairly high.
    let (sx, sy, sz) = normalize3(-0.6, -0.55, 0.58);
    let relief = 7.0;
    for j in 1..GH - 1 {
        for i in 1..GW - 1 {
            let dzdx = (h[idx(i + 1, j)] - h[idx(i - 1, j)]) * relief;
            let dzdy = (h[idx(i, j + 1)] - h[idx(i, j - 1)]) * relief;
            let (nx, ny, nz) = normalize3(-dzdx, -dzdy, 1.0);
            let dot = (nx * sx + ny * sy + nz * sz).clamp(0.0, 1.0);
            shade[idx(i, j)] = 0.6 + dot * 0.62;
        }
    }
    shade
}

fn normalize3(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let m = (x * x + y * y + z * z).sqrt().max(1e-6);
    (x / m, y / m, z / m)
}

// ---- noise -------------------------------------------------------------------

fn hash2(x: i32, y: i32, seed: u64) -> f32 {
    let mut h = seed
        ^ (x as i64 as u64).wrapping_mul(0x9E3779B97F4A7C15)
        ^ (y as i64 as u64).wrapping_mul(0xC2B2AE3D27D4EB4F);
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51AFD7ED558CCD);
    h ^= h >> 33;
    (h & 0xFFFFFF) as f32 / 0xFFFFFF as f32
}

fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn lerpf(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn vnoise(x: f32, y: f32, seed: u64) -> f32 {
    let xi = x.floor();
    let yi = y.floor();
    let (x0, y0) = (xi as i32, yi as i32);
    let (xf, yf) = (x - xi, y - yi);
    let a = hash2(x0, y0, seed);
    let b = hash2(x0 + 1, y0, seed);
    let c = hash2(x0, y0 + 1, seed);
    let d = hash2(x0 + 1, y0 + 1, seed);
    let u = smoothstep(xf);
    let v = smoothstep(yf);
    lerpf(lerpf(a, b, u), lerpf(c, d, u), v)
}

fn fbm(x: f32, y: f32, seed: u64) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut freq = 1.0;
    let mut norm = 0.0;
    for o in 0..4 {
        sum += amp * vnoise(x * freq, y * freq, seed ^ (o as u64).wrapping_mul(0x1357_9BDF));
        norm += amp;
        freq *= 2.0;
        amp *= 0.5;
    }
    sum / norm
}

fn pt_seg_dist(p: Vector2, a: Vector2, b: Vector2) -> f32 {
    let ab = b - a;
    let ap = p - a;
    let len2 = ab.x * ab.x + ab.y * ab.y;
    let t = if len2 > 0.0 {
        ((ap.x * ab.x + ap.y * ab.y) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let proj = Vector2::new(a.x + ab.x * t, a.y + ab.y * t);
    p.distance_to(proj)
}
