//! A full-width red bar pinned to the top of the window whenever this
//! is a debug build (`cfg!(debug_assertions)` is true). Release builds
//! add nothing at all.
//!
//! Exists because a debug build of this Bevy app is dramatically slower
//! than release, and a laggy debug build looks identical to the normal
//! app otherwise. The bar makes the build profile impossible to miss.
//!
//! Renders on the `MENU_OVERLAY_LAYER` (screen space, above every pane),
//! mirroring `fps.rs`: that overlay camera is a 1:1 pixel projection
//! centered at the window origin with y up, so the top edge is at
//! `+win_h/2`. Repositioned/resized each frame against the primary
//! window so it tracks resizes without a separate event hookup.

use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;
use bevy::sprite::Anchor;

use crate::{MonoFont, MENU_OVERLAY_LAYER};

const BAR_H: f32 = 22.0;
// Above panes; same band as the fps meter (Z=950), well inside the
// Camera2d ±1000 depth range. Label sits just in front of the bar.
const BAR_Z: f32 = 960.0;
const TEXT_Z: f32 = 961.0;

pub struct DebugBarPlugin;

impl Plugin for DebugBarPlugin {
    fn build(&self, app: &mut App) {
        // Compiles in every profile; only wired up in debug builds.
        if cfg!(debug_assertions) {
            app.add_systems(Startup, spawn_debug_bar)
                .add_systems(Update, position_debug_bar);
        }
    }
}

#[derive(Component)]
struct DebugBar;

#[derive(Component)]
struct DebugBarLabel;

fn spawn_debug_bar(mut commands: Commands, font: Option<Res<MonoFont>>) {
    commands.spawn((
        DebugBar,
        Sprite {
            color: Color::srgb(0.85, 0.12, 0.12),
            // Real width is set every frame from the window size.
            custom_size: Some(Vec2::new(10.0, BAR_H)),
            ..default()
        },
        Anchor::TOP_CENTER,
        Transform::from_xyz(0.0, 0.0, BAR_Z),
        RenderLayers::layer(MENU_OVERLAY_LAYER),
    ));

    if let Some(font) = font {
        commands.spawn((
            DebugBarLabel,
            Text2d::new("DEBUG BUILD \u{2014} not the release app (expect lag)"),
            TextFont {
                font: font.0.clone(),
                font_size: 13.0,
                ..default()
            },
            TextColor(Color::WHITE),
            Anchor::CENTER,
            Transform::from_xyz(0.0, 0.0, TEXT_Z),
            RenderLayers::layer(MENU_OVERLAY_LAYER),
        ));
    }
}

fn position_debug_bar(
    windows: Query<&Window>,
    mut bar: Query<(&mut Sprite, &mut Transform), (With<DebugBar>, Without<DebugBarLabel>)>,
    mut label: Query<&mut Transform, (With<DebugBarLabel>, Without<DebugBar>)>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let win_w = window.width();
    let top_y = window.height() * 0.5;

    if let Ok((mut sprite, mut tx)) = bar.single_mut() {
        sprite.custom_size = Some(Vec2::new(win_w, BAR_H));
        tx.translation.x = 0.0;
        tx.translation.y = top_y;
    }
    if let Ok(mut tx) = label.single_mut() {
        tx.translation.x = 0.0;
        tx.translation.y = top_y - BAR_H * 0.5;
    }
}
