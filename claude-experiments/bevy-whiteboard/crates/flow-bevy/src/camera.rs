//! Pan/zoom camera.
//!
//! Bindings (Mac-shaped):
//!   - pan: middle/right mouse drag, OR two-finger scroll (MouseWheel)
//!   - zoom: trackpad pinch (PinchGesture), OR ⌘ + two-finger scroll
//!
//! On macOS the trackpad sends two-finger scroll as `MouseWheel` events
//! and pinch as `PinchGesture`. So `MouseWheel` is bound to pan (the
//! standard "scroll the canvas" gesture) and pinch to zoom — which is
//! what users expect from native apps. ⌘+scroll keeps wheel-zoom for
//! users on a real wheel mouse without a trackpad.

use bevy::input::gestures::PinchGesture;
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::prelude::*;

pub struct CameraPlugin;
impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_camera)
            .add_systems(Update, (pan_camera, pan_wheel, zoom_pinch));
    }
}

#[derive(Component)]
pub struct MainCamera;

fn spawn_camera(mut commands: Commands) {
    commands.spawn((Camera2d, MainCamera));
}

fn pan_camera(
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion_evr: MessageReader<bevy::input::mouse::MouseMotion>,
    mut q: Query<(&mut Transform, &Projection), With<MainCamera>>,
) {
    if !(buttons.pressed(MouseButton::Middle) || buttons.pressed(MouseButton::Right)) {
        motion_evr.clear();
        return;
    }
    let Ok((mut tf, proj)) = q.single_mut() else { return };
    let scale = match proj {
        Projection::Orthographic(o) => o.scale,
        _ => 1.0,
    };
    for ev in motion_evr.read() {
        tf.translation.x -= ev.delta.x * scale;
        tf.translation.y += ev.delta.y * scale;
    }
}

fn pan_wheel(
    keys: Res<ButtonInput<KeyCode>>,
    mut wheel_evr: MessageReader<MouseWheel>,
    mut q: Query<(&mut Transform, &mut Projection), With<MainCamera>>,
) {
    let Ok((mut tf, mut proj)) = q.single_mut() else { return };
    let Projection::Orthographic(ortho) = proj.as_mut() else { return };
    // ⌘+scroll keeps the old wheel-zoom binding for non-trackpad users.
    let cmd_held = keys.pressed(KeyCode::SuperLeft) || keys.pressed(KeyCode::SuperRight);
    for ev in wheel_evr.read() {
        let (px_x, px_y) = match ev.unit {
            MouseScrollUnit::Line => (ev.x * 20.0, ev.y * 20.0),
            MouseScrollUnit::Pixel => (ev.x, ev.y),
        };
        if cmd_held {
            let step = match ev.unit {
                MouseScrollUnit::Line => ev.y * 0.1,
                MouseScrollUnit::Pixel => ev.y * 0.005,
            };
            ortho.scale = (ortho.scale * (1.0 - step)).clamp(0.2, 5.0);
        } else {
            // Natural-scrolling: dragging two fingers down moves the
            // canvas DOWN (i.e. camera up), matching macOS conventions.
            tf.translation.x -= px_x * ortho.scale;
            tf.translation.y += px_y * ortho.scale;
        }
    }
}

fn zoom_pinch(
    mut pinch_evr: MessageReader<PinchGesture>,
    mut q: Query<&mut Projection, With<MainCamera>>,
) {
    let Ok(mut proj) = q.single_mut() else { return };
    let Projection::Orthographic(ortho) = proj.as_mut() else { return };
    for ev in pinch_evr.read() {
        // PinchGesture(delta): positive = pinch out (zoom in → smaller scale).
        ortho.scale = (ortho.scale * (1.0 - ev.0)).clamp(0.2, 5.0);
    }
}

/// Project a screen-space cursor position into world coordinates using
/// the main camera.
pub fn cursor_to_world(
    windows: &Query<&Window>,
    cams: &Query<(&Camera, &GlobalTransform), With<MainCamera>>,
) -> Option<Vec2> {
    let win = windows.single().ok()?;
    let cursor = win.cursor_position()?;
    let (cam, xf) = cams.single().ok()?;
    cam.viewport_to_world_2d(xf, cursor).ok()
}
