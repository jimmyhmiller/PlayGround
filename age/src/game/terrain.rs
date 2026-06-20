//! Procedural world terrain: a value-noise heightfield classified into
//! sea / sand / grass / hills / mountains / snow, plus a handful of winding
//! rivers. Deterministic from a fixed world seed, so the landscape is stable
//! across runs and only needs the seed (no stored grid).
//!
//! Rivers are explicit meander polylines drawn as ribbons on top of the ground
//! (so per-cell rendering never has to query them); the cheap `river_dist` query
//! is only used to keep cities off the water and to place bridges.

use crate::util::Rng;
use raylib::prelude::Vector2;

/// World units per base noise feature (bigger = larger continents/ranges).
const BASE: f32 = 1700.0;
/// The big region rivers are generated to span.
const AREA_MIN: f32 = -2500.0;
const AREA_MAX: f32 = 15000.0;

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
    pub width: f32,
}

pub struct Terrain {
    seed: u64,
    rivers: Vec<River>,
}

impl Terrain {
    pub fn new(seed: u64) -> Terrain {
        Terrain { seed, rivers: gen_rivers(seed) }
    }

    /// Height in [0, 1] at a world point (fractal value noise).
    pub fn height(&self, x: f32, y: f32) -> f32 {
        fbm(x / BASE, y / BASE, self.seed)
    }

    pub fn land_at(&self, x: f32, y: f32) -> Land {
        classify(self.height(x, y))
    }

    pub fn rivers(&self) -> &[River] {
        &self.rivers
    }

    /// Distance from `p` to the nearest river centerline (INF if no rivers).
    pub fn river_dist(&self, p: Vector2) -> f32 {
        let mut best = f32::INFINITY;
        for r in &self.rivers {
            for w in r.points.windows(2) {
                best = best.min(pt_seg_dist(p, w[0], w[1]));
            }
        }
        best
    }

    /// Can a settlement sit here? Buildable land, clear of rivers.
    pub fn is_buildable(&self, p: Vector2) -> bool {
        matches!(self.land_at(p.x, p.y), Land::Grass | Land::Hill | Land::Sand)
            && self.river_dist(p) > 60.0
    }
}

pub fn classify(h: f32) -> Land {
    match h {
        x if x < 0.31 => Land::Sea,
        x if x < 0.35 => Land::Sand,
        x if x < 0.60 => Land::Grass,
        x if x < 0.72 => Land::Hill,
        x if x < 0.86 => Land::Mountain,
        _ => Land::Snow,
    }
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

fn smooth(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
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
    let u = smooth(xf);
    let v = smooth(yf);
    lerp(lerp(a, b, u), lerp(c, d, u), v)
}

/// Fractal Brownian motion — 4 octaves of value noise.
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

// ---- rivers ------------------------------------------------------------------

fn gen_rivers(seed: u64) -> Vec<River> {
    let mut top = Rng(seed | 1);
    let n = 3 + (top.next_u64() % 3) as usize; // 3..=5 rivers
    let mut rivers = Vec::new();
    for i in 0..n {
        let mut r = Rng::seeded(&(seed, i, "river"));
        let horizontal = r.next_f32() < 0.5;
        let (start, end) = if horizontal {
            (
                Vector2::new(AREA_MIN, r.range(AREA_MIN, AREA_MAX)),
                Vector2::new(AREA_MAX, r.range(AREA_MIN, AREA_MAX)),
            )
        } else {
            (
                Vector2::new(r.range(AREA_MIN, AREA_MAX), AREA_MIN),
                Vector2::new(r.range(AREA_MIN, AREA_MAX), AREA_MAX),
            )
        };
        let main = end - start;
        let len = main.length().max(1.0);
        let dir = Vector2::new(main.x / len, main.y / len);
        let perp = Vector2::new(-dir.y, dir.x);
        let amp = r.range(350.0, 800.0);
        let waves = r.range(1.5, 3.5);
        let phase = r.range(0.0, 6.2831);
        let steps = 40;
        let mut points = Vec::with_capacity(steps + 1);
        for s in 0..=steps {
            let t = s as f32 / steps as f32;
            let base = start + dir * (len * t);
            let off = (t * waves * 6.2831 + phase).sin() * amp
                + (t * waves * 2.6 * 6.2831 + phase).sin() * amp * 0.28;
            points.push(base + perp * off);
        }
        rivers.push(River { points, width: r.range(28.0, 50.0) });
    }
    rivers
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
