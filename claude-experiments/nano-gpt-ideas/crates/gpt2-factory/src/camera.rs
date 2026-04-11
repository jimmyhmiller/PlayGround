//! First-person free-fly camera.
//!
//! WASD moves along the ground plane relative to yaw. Space/Ctrl go up/down.
//! Shift sprints. Mouse look is always active while the cursor is grabbed.
//! Click to grab the cursor, Escape to release.

use bevy::prelude::*;
use bevy::input::mouse::MouseMotion;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::core_pipeline::bloom::BloomSettings;
use bevy::window::{CursorGrabMode, PrimaryWindow};

use crate::config;

pub struct FpsCameraPlugin;

impl Plugin for FpsCameraPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_systems(Startup, spawn_camera)
            .add_systems(Update, (mouse_look, movement));
    }
}

#[derive(Component)]
pub struct FpsCamera {
    pub yaw: f32,
    pub pitch: f32,
    pub speed: f32,
    pub sprint_mul: f32,
    pub sensitivity: f32,
}

impl Default for FpsCamera {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: -0.1,
            speed: 12.0,
            sprint_mul: 4.0,
            sensitivity: 0.002,
        }
    }
}

fn spawn_camera(mut commands: Commands) {
    // Start just outside the entry hall looking down the length of the building.
    let start = Vec3::new(
        config::BLOCK_WIDTH * 0.0,
        2.5,
        -(config::HALL_LENGTH * 0.6),
    );
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(start)
                .looking_at(Vec3::new(0.0, 10.0, 20.0), Vec3::Y),
            tonemapping: Tonemapping::TonyMcMapface,
            ..default()
        },
        BloomSettings::NATURAL,
        FpsCamera::default(),
    ));
}

fn mouse_look(
    mut motion: EventReader<MouseMotion>,
    window: Query<&Window, With<PrimaryWindow>>,
    mut q: Query<(&mut FpsCamera, &mut Transform)>,
) {
    let Ok(win) = window.get_single() else { return };
    if win.cursor.grab_mode == CursorGrabMode::None {
        motion.clear();
        return;
    }
    let mut dx = 0.0;
    let mut dy = 0.0;
    for ev in motion.read() {
        dx += ev.delta.x;
        dy += ev.delta.y;
    }
    if dx == 0.0 && dy == 0.0 { return; }
    for (mut cam, mut tf) in q.iter_mut() {
        cam.yaw -= dx * cam.sensitivity;
        cam.pitch = (cam.pitch - dy * cam.sensitivity).clamp(-1.5, 1.5);
        let rot = Quat::from_axis_angle(Vec3::Y, cam.yaw)
            * Quat::from_axis_angle(Vec3::X, cam.pitch);
        tf.rotation = rot;
    }
}

fn movement(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut q: Query<(&FpsCamera, &mut Transform)>,
) {
    let dt = time.delta_seconds();
    for (cam, mut tf) in q.iter_mut() {
        let mut dir = Vec3::ZERO;
        // Forward/right derived from yaw only — keeps WASD on the ground plane.
        let fwd = Vec3::new(-cam.yaw.sin(), 0.0, -cam.yaw.cos());
        let right = Vec3::new(cam.yaw.cos(), 0.0, -cam.yaw.sin());
        if keys.pressed(KeyCode::KeyW) { dir += fwd; }
        if keys.pressed(KeyCode::KeyS) { dir -= fwd; }
        if keys.pressed(KeyCode::KeyD) { dir += right; }
        if keys.pressed(KeyCode::KeyA) { dir -= right; }
        if keys.pressed(KeyCode::Space) { dir += Vec3::Y; }
        if keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight) {
            dir -= Vec3::Y;
        }
        if dir.length_squared() > 0.0 {
            dir = dir.normalize();
            let mut speed = cam.speed;
            if keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) {
                speed *= cam.sprint_mul;
            }
            tf.translation += dir * speed * dt;
        }
    }
}
