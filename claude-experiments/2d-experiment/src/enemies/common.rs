use bevy::prelude::*;

use crate::{
    PhysicalTranslation, Player, PlayerSpawn, PreviousPhysicalTranslation, Velocity,
    ray_segment_hit,
};

use super::monster::Monster;
use super::sentinel::Sentinel;

/// Simple deterministic-ish random f32 in [lo, hi) seeded from a global counter.
pub fn rand_range(lo: f32, hi: f32) -> f32 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static CTR: AtomicU64 = AtomicU64::new(0);
    let n = CTR.fetch_add(1, Ordering::Relaxed);
    let mut x = n.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    let frac = (x & 0xFFFFFF) as f32 / 0xFFFFFF as f32;
    lo + frac * (hi - lo)
}

pub const ALERT_TIMEOUT: f32 = 5.0;

/// Returns true if the segment from `a` to `b` is blocked by any wall AABB.
pub fn segment_blocked(a: Vec2, b: Vec2, walls: &[(Vec2, Vec2)]) -> bool {
    let dir = b - a;
    if dir.length_squared() < 0.001 {
        return false;
    }
    for &(center, half) in walls {
        let bl = center + Vec2::new(-half.x, -half.y);
        let br = center + Vec2::new(half.x, -half.y);
        let tr = center + Vec2::new(half.x, half.y);
        let tl = center + Vec2::new(-half.x, half.y);
        for (ea, eb) in [(bl, br), (br, tr), (tr, tl), (tl, bl)] {
            if let Some(t) = ray_segment_hit(a, dir, ea, eb) {
                if t > 0.001 && t < 0.999 {
                    return true;
                }
            }
        }
    }
    false
}

/// Find the next waypoint toward `to` using a visibility graph over expanded
/// wall corners. Returns None if no path exists.
pub fn pathfind_next_waypoint(
    from: Vec2,
    to: Vec2,
    walls: &[(Vec2, Vec2)],
    agent_half: Vec2,
) -> Option<Vec2> {
    let expanded: Vec<(Vec2, Vec2)> = walls
        .iter()
        .map(|(c, h)| (*c, *h + agent_half))
        .collect();

    if !segment_blocked(from, to, &expanded) {
        return Some(to);
    }

    let margin = 2.0;
    let mut nodes = vec![from, to];
    for &(center, half) in &expanded {
        let h = half + Vec2::splat(margin);
        nodes.push(center + Vec2::new(-h.x, -h.y));
        nodes.push(center + Vec2::new(h.x, -h.y));
        nodes.push(center + Vec2::new(h.x, h.y));
        nodes.push(center + Vec2::new(-h.x, h.y));
    }

    let valid: Vec<Vec2> = nodes
        .iter()
        .copied()
        .enumerate()
        .filter(|&(i, p)| {
            i < 2
                || !expanded
                    .iter()
                    .any(|(c, h)| (p.x - c.x).abs() < h.x && (p.y - c.y).abs() < h.y)
        })
        .map(|(_, p)| p)
        .collect();

    let n = valid.len();
    let mut dist = vec![f32::INFINITY; n];
    let mut prev = vec![usize::MAX; n];
    let mut visited = vec![false; n];
    dist[0] = 0.0;

    for _ in 0..n {
        let mut u = usize::MAX;
        let mut best = f32::INFINITY;
        for i in 0..n {
            if !visited[i] && dist[i] < best {
                best = dist[i];
                u = i;
            }
        }
        if u == usize::MAX || u == 1 {
            break;
        }
        visited[u] = true;

        for v in 0..n {
            if visited[v] {
                continue;
            }
            if segment_blocked(valid[u], valid[v], &expanded) {
                continue;
            }
            let d = dist[u] + (valid[v] - valid[u]).length();
            if d < dist[v] {
                dist[v] = d;
                prev[v] = u;
            }
        }
    }

    if dist[1].is_infinite() {
        return None;
    }

    let mut step = 1;
    while prev[step] != 0 && prev[step] != usize::MAX {
        step = prev[step];
    }
    Some(valid[step])
}

pub fn monster_attack(
    mut players: Query<
        (&mut PhysicalTranslation, &mut PreviousPhysicalTranslation, &PlayerSpawn, &mut Velocity),
        With<Player>,
    >,
    monsters: Query<(&PhysicalTranslation, &Monster), (Without<Player>, Without<Sentinel>)>,
    sentinels_q: Query<(&PhysicalTranslation, &Sentinel), (Without<Player>, Without<Monster>)>,
) {
    let Ok((mut phys, mut prev, spawn, mut vel)) = players.single_mut() else { return };
    let player_p = phys.0;

    for (monster_pos, monster) in &monsters {
        let dist = (monster_pos.0 - player_p).length();
        if dist < monster.attack_reach {
            vel.0 = Vec2::ZERO;
            phys.0 = spawn.0;
            prev.0 = spawn.0;
            info!("player hit by monster — respawning");
            return;
        }
    }
    for (sentinel_pos, sentinel) in &sentinels_q {
        let dist = (sentinel_pos.0 - player_p).length();
        if dist < sentinel.attack_reach {
            vel.0 = Vec2::ZERO;
            phys.0 = spawn.0;
            prev.0 = spawn.0;
            info!("player hit by sentinel — respawning");
            return;
        }
    }
}
