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

use raylib::prelude::Vector2;

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

pub struct Terrain {
    height: Vec<f32>,
    shade: Vec<f32>,
}

#[inline]
fn idx(i: usize, j: usize) -> usize {
    j * GW + i
}

impl Terrain {
    pub fn new(seed: u64) -> Terrain {
        let mut height = base_field(seed);
        thermal_erode(&mut height, 16);
        // Rivers are carved *into* the heightfield below sea level, so they render
        // as real water (same depth shading + foam as the sea), not painted lines.
        carve_rivers(&mut height);
        let shade = hillshade(&height);
        Terrain { height, shade }
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

    /// True where a settlement can stand: dry land (rivers are now Sea cells, so
    /// this also keeps towns out of the water).
    pub fn is_buildable(&self, p: Vector2) -> bool {
        matches!(self.land_at(p.x, p.y), Land::Grass | Land::Hill | Land::Sand)
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

// ---- hydrology: fill depressions -> flow accumulation -> river network -------

/// f32 with a total order, for the priority queue.
#[derive(PartialEq)]
struct OrdF(f32);
impl Eq for OrdF {}
impl Ord for OrdF {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&o.0)
    }
}
impl PartialOrd for OrdF {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}

/// Priority-flood depression filling (Barnes et al.): flood inward from the map
/// border so every cell gets a monotonic downhill path off the map. Pits fill to
/// their spill level (becoming lakes) — so flow never gets stuck.
fn priority_flood(height: &[f32]) -> Vec<f32> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let mut filled = height.to_vec();
    let mut visited = vec![false; GW * GH];
    let mut heap: BinaryHeap<Reverse<(OrdF, usize)>> = BinaryHeap::new();
    for j in 0..GH {
        for i in 0..GW {
            if i == 0 || j == 0 || i == GW - 1 || j == GH - 1 {
                let c = idx(i, j);
                visited[c] = true;
                heap.push(Reverse((OrdF(filled[c]), c)));
            }
        }
    }
    const EPS: f32 = 1e-5;
    while let Some(Reverse((OrdF(h), c))) = heap.pop() {
        let (i, j) = (c % GW, c / GW);
        for (di, dj) in NEIGH {
            let ni = i as i32 + di;
            let nj = j as i32 + dj;
            if ni < 0 || nj < 0 || ni >= GW as i32 || nj >= GH as i32 {
                continue;
            }
            let n = idx(ni as usize, nj as usize);
            if visited[n] {
                continue;
            }
            visited[n] = true;
            if filled[n] < h + EPS {
                filled[n] = h + EPS;
            }
            heap.push(Reverse((OrdF(filled[n]), n)));
        }
    }
    filled
}

/// D8 steepest-descent receiver for each cell (-1 = drains off the map / sea).
fn flow_receivers(filled: &[f32]) -> Vec<i32> {
    let mut recv = vec![-1i32; GW * GH];
    for j in 1..GH - 1 {
        for i in 1..GW - 1 {
            let c = idx(i, j);
            let mut best = -1i32;
            let mut bh = filled[c];
            for (di, dj) in NEIGH {
                let n = idx((i as i32 + di) as usize, (j as i32 + dj) as usize);
                if filled[n] < bh {
                    bh = filled[n];
                    best = n as i32;
                }
            }
            recv[c] = best;
        }
    }
    recv
}

/// Flow accumulation: drainage area through each cell (process high cells first,
/// pushing their area downstream). Trunks near the sea accumulate the most.
fn flow_accumulation(filled: &[f32], recv: &[i32]) -> Vec<f32> {
    let n = GW * GH;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| filled[b].total_cmp(&filled[a]));
    let mut acc = vec![1.0f32; n];
    for &c in &order {
        let r = recv[c];
        if r >= 0 {
            acc[r as usize] += acc[c];
        }
    }
    acc
}

/// Carve a small brush at a river cell so a visible valley forms.
fn carve(h: &mut [f32], i: usize, j: usize, depth: f32) {
    for (di, dj) in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)] {
        let ni = i as i32 + di;
        let nj = j as i32 + dj;
        if ni < 1 || nj < 1 || ni >= GW as i32 - 1 || nj >= GH as i32 - 1 {
            continue;
        }
        let n = idx(ni as usize, nj as usize);
        let amt = if di == 0 && dj == 0 { depth } else { depth * 0.5 };
        h[n] = (h[n] - amt).max(0.0);
    }
}

/// The full river pipeline: fill -> receivers -> accumulation -> carve valleys ->
/// extract the high-drainage cells as a merging, downstream-widening network.
fn carve_rivers(height: &mut [f32]) {
    // Iterative hydraulic carving: each pass concentrates flow into valleys, so
    // parallel rivulets converge into branching trunks (a real drainage network).
    for _ in 0..4 {
        let filled = priority_flood(height);
        let recv = flow_receivers(&filled);
        let acc = flow_accumulation(&filled, &recv);
        for c in 0..GW * GH {
            if acc[c] > 10.0 && height[c] > SEA_LEVEL {
                let depth = (0.005 + 0.004 * acc[c].ln()).min(0.03);
                carve(height, c % GW, c / GW, depth);
            }
        }
    }

    // Final drainage, then stamp each channel down BELOW sea level (width + depth
    // scaled by how much water it carries) so it becomes real water in the terrain.
    let filled = priority_flood(height);
    let recv = flow_receivers(&filled);
    let acc = flow_accumulation(&filled, &recv);

    const RIVER_T: f32 = 22.0;
    let mut depth = vec![0.0f32; GW * GH]; // target depth below sea, max-accumulated
    for c in 0..GW * GH {
        let _ = recv;
        if acc[c] <= RIVER_T {
            continue;
        }
        let s = acc[c].sqrt();
        let d = (0.045 + s * 0.0045).min(0.15); // deeper trunks read darker
        let radius = (0.7 + s * 0.06).clamp(0.7, 3.0); // cells
        stamp_channel(&mut depth, c % GW, c / GW, d, radius);
    }
    for c in 0..GW * GH {
        if depth[c] > 0.0 {
            let target = (SEA_LEVEL - depth[c]).max(0.0);
            if target < height[c] {
                height[c] = target;
            }
        }
    }
}

/// Stamp a tapered channel into the depth field: deepest at the centre, fading to
/// the banks, so rendering gives a smooth water edge (and the shoreline foam).
fn stamp_channel(depth: &mut [f32], ci: usize, cj: usize, d: f32, radius: f32) {
    let r = radius.ceil() as i32;
    for dj in -r..=r {
        for di in -r..=r {
            let ni = ci as i32 + di;
            let nj = cj as i32 + dj;
            if ni < 1 || nj < 1 || ni >= GW as i32 - 1 || nj >= GH as i32 - 1 {
                continue;
            }
            let dist = ((di * di + dj * dj) as f32).sqrt();
            if dist > radius {
                continue;
            }
            let v = d * (1.0 - dist / radius);
            let n = idx(ni as usize, nj as usize);
            if v > depth[n] {
                depth[n] = v;
            }
        }
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

