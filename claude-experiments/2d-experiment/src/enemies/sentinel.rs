use bevy::prelude::*;
use bevy_inspector_egui::InspectorOptions;
use bevy_inspector_egui::prelude::ReflectInspectorOptions;

use crate::{
    Collider, PhysicalTranslation, Player, PreviousPhysicalTranslation, Velocity,
    WallFootprint, YSorted, resolve_collision,
};

use super::common::*;
use super::monster::Monster;

// =====================================================================
// Sentinel — enemy with a narrow searchlight
// =====================================================================

#[derive(Component, Reflect, InspectorOptions)]
#[reflect(InspectorOptions)]
pub struct Sentinel {
    #[inspector(min = 0.0, max = 500.0)]
    pub light_range: f32,
    #[inspector(min = 1.0, max = 90.0)]
    pub light_half_angle_deg: f32,
    #[inspector(min = 0.0, max = 4.0)]
    pub light_intensity: f32,
    #[inspector(min = 0.0, max = 500.0)]
    pub speed: f32,
    #[inspector(min = 0.0, max = 100.0)]
    pub attack_reach: f32,
    #[inspector(min = 0.0, max = 180.0)]
    pub sweep_speed: f32,
}

impl Default for Sentinel {
    fn default() -> Self {
        Self {
            light_range: 350.0,
            light_half_angle_deg: 18.0,
            light_intensity: 1.5,
            speed: 60.0,
            attack_reach: 28.0,
            sweep_speed: 40.0,
        }
    }
}

/// Tracks the sentinel's current aim angle and sweep direction.
#[derive(Component)]
pub struct SentinelState {
    pub aim_angle: f32,
    pub sweep_dir: f32,
    pub alerted: bool,
}

impl Default for SentinelState {
    fn default() -> Self {
        let angle = rand_range(0.0, std::f32::consts::TAU);
        Self {
            aim_angle: angle,
            sweep_dir: 1.0,
            alerted: false,
        }
    }
}

pub fn spawn_sentinel(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
    pos: Vec2,
    sentinel: Sentinel,
) -> Entity {
    let half = Vec2::new(14.0, 14.0);
    commands
        .spawn((
            sentinel,
            SentinelState::default(),
            Mesh2d(meshes.add(Rectangle::new(half.x * 2.0, half.y * 2.0))),
            MeshMaterial2d(materials.add(Color::srgb(0.85, 0.65, 0.15))),
            Transform::from_xyz(pos.x, pos.y, 1.0),
            Collider { half },
            Velocity::default(),
            PhysicalTranslation(pos),
            PreviousPhysicalTranslation(pos),
            YSorted { ground_offset: 0.0 },
            Name::new("Sentinel"),
        ))
        .id()
}

pub fn sentinel_ai(
    fixed_time: Res<Time<Fixed>>,
    players: Query<&PhysicalTranslation, With<Player>>,
    walls: Query<(&Transform, &Collider), (With<WallFootprint>, Without<Player>, Without<Monster>, Without<Sentinel>)>,
    mut sentinels: Query<
        (&Sentinel, &mut SentinelState, &mut PhysicalTranslation, &mut PreviousPhysicalTranslation, &mut Velocity, &Collider),
        (Without<Player>, Without<Monster>),
    >,
) {
    let Ok(player_pos) = players.single() else { return };
    let dt = fixed_time.delta_secs();
    let wall_list: Vec<(Vec2, Vec2)> = walls
        .iter()
        .map(|(tf, c)| (tf.translation.truncate(), c.half))
        .collect();

    for (sentinel, mut sstate, mut pos, mut prev, mut vel, col) in &mut sentinels {
        prev.0 = pos.0;

        let to_player = player_pos.0 - pos.0;
        let dist = to_player.length();
        let aim_dir = Vec2::new(sstate.aim_angle.cos(), sstate.aim_angle.sin());

        let in_cone = dist < sentinel.light_range
            && dist > 0.1
            && {
                let dir_to_player = to_player / dist;
                let cos_half = sentinel.light_half_angle_deg.to_radians().cos();
                dir_to_player.dot(aim_dir) > cos_half
            }
            && !segment_blocked(pos.0, player_pos.0, &wall_list);

        if in_cone {
            sstate.alerted = true;
            sstate.aim_angle = to_player.to_angle();
            let dir = to_player.normalize_or_zero();
            vel.0 = dir * sentinel.speed;
        } else if sstate.alerted {
            sstate.alerted = false;
            vel.0 = Vec2::ZERO;
        } else {
            // Idle patrol — sweep light back and forth
            sstate.aim_angle += sentinel.sweep_speed.to_radians() * sstate.sweep_dir * dt;
            if sstate.aim_angle.sin().abs() < 0.01 || rand_range(0.0, 1.0) < 0.002 {
                sstate.sweep_dir = -sstate.sweep_dir;
            }
            vel.0 = Vec2::ZERO;
        }

        let move_delta = vel.0 * dt;
        pos.0 = resolve_collision(pos.0, col.half, move_delta, &wall_list, &mut vel.0);
    }
}
