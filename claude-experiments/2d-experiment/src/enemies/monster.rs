use bevy::prelude::*;
use bevy_inspector_egui::InspectorOptions;
use bevy_inspector_egui::prelude::ReflectInspectorOptions;

use crate::{
    Collider, PhysicalTranslation, Player, PreviousPhysicalTranslation, Velocity,
    WallFootprint, YSorted, resolve_collision,
};

use super::common::*;
use super::sentinel::Sentinel;

// =====================================================================
// Monster
// =====================================================================

#[derive(Component, Reflect, InspectorOptions)]
#[reflect(InspectorOptions)]
pub struct Monster {
    #[inspector(min = 0.0, max = 500.0)]
    pub speed: f32,
    #[inspector(min = 0.0, max = 800.0)]
    pub detection_range: f32,
    #[inspector(min = 0.0, max = 100.0)]
    pub attack_reach: f32,
    #[inspector(min = 0.0, max = 100.0)]
    pub strength: f32,
}

impl Default for Monster {
    fn default() -> Self {
        Self {
            speed: 120.0,
            detection_range: 300.0,
            attack_reach: 28.0,
            strength: 1.0,
        }
    }
}

/// Tracks whether a monster has spotted the player and how long since it lost sight.
#[derive(Component)]
pub struct MonsterAlert {
    pub has_seen: bool,
    pub time_since_los: f32,
    pub notice_accumulator: f32,
    pub notice_threshold: f32,
    pub wander_dir: Vec2,
    pub wander_timer: f32,
}

impl Default for MonsterAlert {
    fn default() -> Self {
        let angle = rand_range(0.0, std::f32::consts::TAU);
        Self {
            has_seen: false,
            time_since_los: 0.0,
            notice_accumulator: 0.0,
            notice_threshold: rand_range(0.3, 1.5),
            wander_dir: Vec2::new(angle.cos(), angle.sin()),
            wander_timer: rand_range(1.0, 3.0),
        }
    }
}

pub fn spawn_monster(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
    pos: Vec2,
    monster: Monster,
) -> Entity {
    let half = Vec2::new(12.0, 12.0);
    commands
        .spawn((
            monster,
            MonsterAlert::default(),
            Mesh2d(meshes.add(Rectangle::new(half.x * 2.0, half.y * 2.0))),
            MeshMaterial2d(materials.add(Color::srgb(0.92, 0.25, 0.25))),
            Transform::from_xyz(pos.x, pos.y, 1.0),
            Collider { half },
            Velocity::default(),
            PhysicalTranslation(pos),
            PreviousPhysicalTranslation(pos),
            YSorted { ground_offset: 0.0 },
            Name::new("Monster"),
        ))
        .id()
}

pub fn monster_ai(
    fixed_time: Res<Time<Fixed>>,
    players: Query<&PhysicalTranslation, With<Player>>,
    walls: Query<(&Transform, &Collider), (With<WallFootprint>, Without<Player>, Without<Monster>)>,
    mut monsters: Query<
        (&Monster, &mut MonsterAlert, &mut PhysicalTranslation, &mut PreviousPhysicalTranslation, &mut Velocity, &Collider),
        (Without<Player>, Without<Sentinel>),
    >,
) {
    let Ok(player_pos) = players.single() else { return };
    let dt = fixed_time.delta_secs();
    let wall_list: Vec<(Vec2, Vec2)> = walls
        .iter()
        .map(|(tf, c)| (tf.translation.truncate(), c.half))
        .collect();

    for (monster, mut alert, mut pos, mut prev, mut vel, col) in &mut monsters {
        prev.0 = pos.0;

        let to_player = player_pos.0 - pos.0;
        let dist = to_player.length();
        let in_range = dist < monster.detection_range && dist > 0.1;
        let has_los = in_range && !segment_blocked(pos.0, player_pos.0, &wall_list);

        // Update alert state
        if has_los {
            if alert.has_seen {
                alert.time_since_los = 0.0;
            } else {
                alert.notice_accumulator += dt;
                if alert.notice_accumulator >= alert.notice_threshold {
                    alert.has_seen = true;
                    alert.time_since_los = 0.0;
                }
            }
        } else {
            alert.notice_accumulator = 0.0;
            if alert.has_seen {
                alert.time_since_los += dt;
                if !in_range || alert.time_since_los > ALERT_TIMEOUT {
                    alert.has_seen = false;
                    alert.notice_threshold = rand_range(0.3, 1.5);
                }
            }
        }

        if has_los && alert.has_seen {
            let dir = to_player.normalize_or_zero();
            vel.0 = dir * monster.speed;
        } else if alert.has_seen && in_range {
            if let Some(wp) = pathfind_next_waypoint(pos.0, player_pos.0, &wall_list, col.half) {
                let dir = (wp - pos.0).normalize_or_zero();
                vel.0 = dir * monster.speed;
            } else {
                vel.0 = Vec2::ZERO;
            }
        } else {
            // Idle — roam randomly
            alert.wander_timer -= dt;
            if alert.wander_timer <= 0.0 {
                if rand_range(0.0, 1.0) < 0.3 {
                    alert.wander_dir = Vec2::ZERO;
                    alert.wander_timer = rand_range(1.0, 3.0);
                } else {
                    let angle = rand_range(0.0, std::f32::consts::TAU);
                    alert.wander_dir = Vec2::new(angle.cos(), angle.sin());
                    alert.wander_timer = rand_range(1.5, 4.0);
                }
            }
            vel.0 = alert.wander_dir * monster.speed * 0.3;
        }

        let move_delta = vel.0 * dt;
        let old_pos = pos.0;
        pos.0 = resolve_collision(pos.0, col.half, move_delta, &wall_list, &mut vel.0);

        if !alert.has_seen && vel.0.length_squared() < 1.0 && alert.wander_dir != Vec2::ZERO && (pos.0 - old_pos).length() < 0.01 {
            let angle = rand_range(0.0, std::f32::consts::TAU);
            alert.wander_dir = Vec2::new(angle.cos(), angle.sin());
            alert.wander_timer = rand_range(1.5, 4.0);
        }
    }
}
