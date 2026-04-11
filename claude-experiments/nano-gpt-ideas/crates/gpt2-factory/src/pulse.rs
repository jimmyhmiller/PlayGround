//! Motion systems — the factory feels alive when things move.
//!
//! Two independent effects, both looping continuously:
//!
//! 1. **Activation pulse** — a wide translucent disc + attached point light
//!    that rises from below the foundation to above the top of the tower.
//!    The point light physically re-illuminates attention panels and belt
//!    cells as it passes, so you see a wave of "something is happening"
//!    sweep up the whole building. Loops with a brief pause at the ends.
//!
//! 2. **Belt particles** — small glowing spheres scattered through the
//!    belt volume. Each rises at its own speed and wraps. There's always
//!    visible motion even when the pulse is off-screen.
//!
//! Controls (physical keys so they work regardless of keyboard layout):
//!   `P` — pause/unpause the pulse
//!   `R` — reset pulse to the bottom
//!   `[` / `]` — slow down / speed up the pulse

use bevy::prelude::*;

use crate::config::*;

pub struct PulsePlugin;

impl Plugin for PulsePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ActivationPulse {
            y: PULSE_START_Y,
            speed: 16.0,
            enabled: true,
        })
        .add_systems(Startup, (spawn_pulse, spawn_belt_particles))
        .add_systems(
            Update,
            (advance_pulse, advance_belt_particles, pulse_keys),
        );
    }
}

/// Pulse starts this far below the ground slab so it has a visible entry
/// animation rather than popping in at Y=0.
const PULSE_START_Y: f32 = -6.0;

#[derive(Resource)]
pub struct ActivationPulse {
    pub y: f32,
    pub speed: f32,
    pub enabled: bool,
}

#[derive(Component)]
pub struct PulseDisc;

#[derive(Component)]
pub struct PulseLight;

#[derive(Component)]
pub struct BeltParticle {
    pub speed: f32,
}

fn spawn_pulse(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    pulse: Res<ActivationPulse>,
) {
    // Wide thin translucent slab that physically intersects each floor
    // as it rises. Slightly narrower than the tower so you can see it
    // poking out above/below from outside.
    let disc_mesh = meshes.add(Cuboid::new(
        BLOCK_WIDTH * 1.2,
        0.5,
        BLOCK_DEPTH * 1.3,
    ));
    let disc_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.35, 0.7, 1.0, 0.22),
        emissive: LinearRgba::rgb(1.8, 3.0, 5.5),
        alpha_mode: AlphaMode::Blend,
        perceptual_roughness: 1.0,
        metallic: 0.0,
        ..default()
    });
    commands.spawn((
        PbrBundle {
            mesh: disc_mesh,
            material: disc_mat,
            transform: Transform::from_xyz(0.0, pulse.y, 0.0),
            ..default()
        },
        PulseDisc,
    ));

    // The point light is the real star — it's what visibly lights up
    // surrounding cells as the wave sweeps past.
    commands.spawn((
        PointLightBundle {
            point_light: PointLight {
                color: Color::srgb(0.55, 0.8, 1.0),
                intensity: 6_000_000.0,
                range: 55.0,
                radius: 3.5,
                shadows_enabled: false,
                ..default()
            },
            transform: Transform::from_xyz(0.0, pulse.y, 0.0),
            ..default()
        },
        PulseLight,
    ));
}

fn advance_pulse(
    time: Res<Time>,
    mut pulse: ResMut<ActivationPulse>,
    mut discs: Query<&mut Transform, (With<PulseDisc>, Without<PulseLight>)>,
    mut lights: Query<&mut Transform, (With<PulseLight>, Without<PulseDisc>)>,
) {
    if !pulse.enabled {
        return;
    }
    pulse.y += pulse.speed * time.delta_seconds();
    let top = tower_total_height() + 8.0;
    if pulse.y > top {
        pulse.y = PULSE_START_Y;
    }
    for mut tf in discs.iter_mut() {
        tf.translation.y = pulse.y;
    }
    for mut tf in lights.iter_mut() {
        tf.translation.y = pulse.y;
    }
}

// ----------------------------------------------------------------------------
// Belt particles
// ----------------------------------------------------------------------------

const PARTICLES_PER_TOKEN: usize = 18;
const N_PARTICLE_TOKENS: usize = 8;
/// Approximate X extent of the belt grid (mirrors `GRID_X_EXTENT` in belt.rs).
const PARTICLE_X_SPREAD: f32 = 22.0;
/// Approximate Z extent of the belt grid.
const PARTICLE_Z_SPREAD: f32 = 14.0;

fn spawn_belt_particles(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let particle_mesh = meshes.add(Sphere::new(0.14));
    let particle_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.75, 0.85, 1.0),
        emissive: LinearRgba::rgb(2.0, 3.2, 5.0),
        perceptual_roughness: 0.9,
        metallic: 0.0,
        ..default()
    });

    let tower_h = tower_total_height();

    for t in 0..N_PARTICLE_TOKENS {
        let token_z = (t as f32 - (N_PARTICLE_TOKENS as f32 - 1.0) * 0.5)
            * (PARTICLE_Z_SPREAD / N_PARTICLE_TOKENS as f32);
        for p in 0..PARTICLES_PER_TOKEN {
            let phase = p as f32 / PARTICLES_PER_TOKEN as f32;
            let y0 = phase * tower_h;
            // Spread X across the belt width using a coprime stride for
            // pseudo-random distribution that's deterministic at startup.
            let x_slot = ((p * 11 + t * 7) % 24) as f32;
            let x = (x_slot / 24.0 - 0.47) * PARTICLE_X_SPREAD;
            let jitter_z = (p as f32 * 0.73).sin() * 0.35;
            let speed_jitter = ((p * 5 + t * 3) % 7) as f32 * 0.4;
            commands.spawn((
                PbrBundle {
                    mesh: particle_mesh.clone(),
                    material: particle_mat.clone(),
                    transform: Transform::from_xyz(x, y0, token_z + jitter_z),
                    ..default()
                },
                BeltParticle {
                    speed: 9.0 + speed_jitter,
                },
            ));
        }
    }
}

fn advance_belt_particles(
    time: Res<Time>,
    mut q: Query<(&BeltParticle, &mut Transform)>,
) {
    let dt = time.delta_seconds();
    let tower_h = tower_total_height();
    for (p, mut tf) in q.iter_mut() {
        tf.translation.y += p.speed * dt;
        if tf.translation.y > tower_h + 2.0 {
            tf.translation.y = -2.0;
        }
    }
}

// ----------------------------------------------------------------------------
// Controls
// ----------------------------------------------------------------------------

fn pulse_keys(keys: Res<ButtonInput<KeyCode>>, mut pulse: ResMut<ActivationPulse>) {
    if keys.just_pressed(KeyCode::KeyP) {
        pulse.enabled = !pulse.enabled;
    }
    if keys.just_pressed(KeyCode::KeyR) {
        pulse.y = PULSE_START_Y;
    }
    if keys.just_pressed(KeyCode::BracketLeft) {
        pulse.speed = (pulse.speed * 0.8).max(2.0);
    }
    if keys.just_pressed(KeyCode::BracketRight) {
        pulse.speed = (pulse.speed * 1.25).min(80.0);
    }
}
