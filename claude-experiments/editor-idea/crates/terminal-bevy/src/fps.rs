//! On-screen FPS / frame-time overlay. Toggle with Cmd+Shift+F.
//!
//! When the overlay is on, also forces winit into Continuous update
//! mode so the readings reflect real per-frame cost rather than
//! reactive idle. Without that override the meter would read
//! near-zero whenever the app is idle in reactive mode (the default).
//! When the overlay is off, winit goes back to whatever
//! `maintain_winit_mode_for_animation` wants.
//!
//! Renders a Text2d on the menu overlay layer so it sits above every
//! pane. Repositioned each frame against the primary window so it
//! tracks resizes without an extra event hookup.

use bevy::camera::visibility::RenderLayers;
use bevy::input::keyboard::KeyboardInput;
use bevy::prelude::*;
use bevy::sprite::Anchor;

use crate::{MonoFont, FONT_SIZE, MENU_OVERLAY_LAYER};

const MARGIN: f32 = 8.0;
// Above context_menu's MENU_Z (700) so the meter doesn't get hidden
// behind a context menu, but well inside the Camera2d default depth
// range (±1000). Z=10000 falls outside the frustum and renders as
// invisible — learned that the fun way.
const Z: f32 = 950.0;

pub struct FpsOverlayPlugin;

impl Plugin for FpsOverlayPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FpsOverlayState>().add_systems(
            Update,
            (toggle_overlay, sync_overlay_visibility, update_overlay).chain(),
        );
    }
}

#[derive(Component)]
struct FpsOverlay;

#[derive(Resource, Default)]
struct FpsOverlayState {
    enabled: bool,
    accum_secs: f64,
    accum_frames: u32,
    last_fps: f32,
    last_ms: f32,
    entity: Option<Entity>,
}

fn toggle_overlay(
    mut events: MessageReader<KeyboardInput>,
    mods: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<FpsOverlayState>,
) {
    let cmd = mods.pressed(KeyCode::SuperLeft) || mods.pressed(KeyCode::SuperRight);
    let shift = mods.pressed(KeyCode::ShiftLeft) || mods.pressed(KeyCode::ShiftRight);
    for ev in events.read() {
        if ev.state.is_pressed() && cmd && shift && matches!(ev.key_code, KeyCode::KeyF) {
            state.enabled = !state.enabled;
            state.accum_secs = 0.0;
            state.accum_frames = 0;
        }
    }
}

fn sync_overlay_visibility(
    mut commands: Commands,
    mut state: ResMut<FpsOverlayState>,
    mut settings: ResMut<bevy::winit::WinitSettings>,
    font: Option<Res<MonoFont>>,
) {
    if state.enabled {
        // Pin to Continuous so the meter reflects real per-frame cost,
        // not reactive idle. Runs each frame so
        // `maintain_winit_mode_for_animation` can't flip it back.
        let want = bevy::winit::UpdateMode::Continuous;
        if settings.focused_mode != want {
            settings.focused_mode = want;
        }
        if settings.unfocused_mode != want {
            settings.unfocused_mode = want;
        }
        if state.entity.is_none() {
            let Some(font) = font else { return };
            let e = commands
                .spawn((
                    FpsOverlay,
                    Text2d::new("fps --"),
                    TextFont {
                        font: font.0.clone(),
                        font_size: FONT_SIZE,
                        ..default()
                    },
                    TextColor(Color::srgb(1.0, 1.0, 0.4)),
                    Anchor::TOP_RIGHT,
                    Transform::from_xyz(0.0, 0.0, Z),
                    RenderLayers::layer(MENU_OVERLAY_LAYER),
                ))
                .id();
            state.entity = Some(e);
        }
    } else if let Some(e) = state.entity.take() {
        commands.entity(e).despawn();
    }
}

fn update_overlay(
    time: Res<Time<Real>>,
    windows: Query<&Window>,
    mut state: ResMut<FpsOverlayState>,
    mut q: Query<(&mut Text2d, &mut Transform), With<FpsOverlay>>,
) {
    if !state.enabled {
        return;
    }

    let dt = time.delta_secs_f64();
    state.accum_secs += dt;
    state.accum_frames += 1;

    if state.accum_secs >= 0.25 {
        let avg_dt = state.accum_secs / state.accum_frames as f64;
        state.last_fps = (1.0 / avg_dt) as f32;
        state.last_ms = (avg_dt * 1000.0) as f32;
        state.accum_secs = 0.0;
        state.accum_frames = 0;
    }

    let Ok(window) = windows.single() else { return };
    let win_w = window.width();
    let win_h = window.height();

    let Ok((mut text, mut tx)) = q.single_mut() else {
        return;
    };
    text.0 = format!("{:>5.1} fps  {:>5.2} ms", state.last_fps, state.last_ms);
    tx.translation.x = win_w * 0.5 - MARGIN;
    tx.translation.y = win_h * 0.5 - MARGIN;
}
