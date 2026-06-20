//! Small dependency-free helpers: wall-clock time, deterministic hashing/RNG,
//! and an RFC-3339 timestamp parser (so we don't pull in `chrono`).

use std::hash::{Hash, Hasher};
use std::time::{SystemTime, UNIX_EPOCH};

/// Current wall-clock time as fractional unix seconds.
pub fn now_unix() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Stable 64-bit hash of any hashable value (uses std's SipHasher, which is
/// stable within a process — good enough for deterministic layout/seeding).
pub fn hash64<T: Hash>(v: &T) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    v.hash(&mut h);
    h.finish()
}

/// A tiny deterministic PRNG (SplitMix64). Seed it from a stable id so a given
/// city/villager always lands in the same place across runs.
#[derive(Clone)]
pub struct Rng(pub u64);

impl Rng {
    pub fn seeded<T: Hash>(v: &T) -> Rng {
        Rng(hash64(v) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Uniform float in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Uniform float in [lo, hi).
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }
}

/// Parse an RFC-3339 / ISO-8601 timestamp like `2026-06-20T20:47:15.159Z`
/// into fractional unix seconds. Returns `None` on anything unexpected.
/// Only the `Z` (UTC) suffix is handled — that's what Claude writes.
pub fn parse_rfc3339(s: &str) -> Option<f64> {
    let b = s.as_bytes();
    if b.len() < 19 {
        return None;
    }
    let num = |a: usize, z: usize| -> Option<i64> { s.get(a..z)?.parse().ok() };
    let year = num(0, 4)?;
    let month = num(5, 7)?;
    let day = num(8, 10)?;
    let hour = num(11, 13)?;
    let min = num(14, 16)?;
    let sec = num(17, 19)?;
    // optional fractional seconds after '.'
    let frac = if b.get(19) == Some(&b'.') {
        let mut i = 20;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        s.get(19..i).and_then(|f| f.parse::<f64>().ok()).unwrap_or(0.0)
    } else {
        0.0
    };
    let days = days_from_civil(year, month, day);
    let secs = days * 86400 + hour * 3600 + min * 60 + sec;
    Some(secs as f64 + frac)
}

/// Days since the unix epoch (1970-01-01) for a proleptic-Gregorian date.
/// Howard Hinnant's algorithm.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = (y - era * 400) as i64; // [0, 399]
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe - 719468
}
