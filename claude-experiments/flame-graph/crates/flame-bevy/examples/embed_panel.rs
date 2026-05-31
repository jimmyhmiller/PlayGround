//! Embed the flame graph as a 2D sprite panel inside a Bevy app.
//!
//! Run with: `cargo run --example embed_panel -p flame-bevy`
//!
//! Drop a trace file's path as the first arg, or omit to see the synthesized
//! demo profile. Use mouse to pan/zoom, 1-5 to switch tabs, f/m/a as
//! documented in the standalone viewer.

use std::sync::Arc;

use bevy::input::keyboard::KeyCode;
use bevy::prelude::*;
use flame_bevy::flame_core::{ProfileBuilder, TrackKind};
use flame_bevy::{FlameGraph, FlameGraphInput, FlameGraphPlugin};

const PANEL_SIZE: (u32, u32) = (1100, 700);

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "flame-bevy: embed_panel".into(),
                resolution: (1280u32, 800u32).into(),
                ..Default::default()
            }),
            ..Default::default()
        }))
        .add_plugins(FlameGraphPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, reload_on_r)
        .run();
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let image = images.add(FlameGraph::blank_image(PANEL_SIZE.0, PANEL_SIZE.1));

    commands.spawn(Camera2d);

    // Sprite-backed panel. Its top-left in window coords is what
    // FlameGraphInput needs to translate cursor positions correctly. With a
    // Camera2d at origin and a Sprite centered at (0, 0), the panel's
    // top-left is window_size/2 - panel/2.
    let mut input = FlameGraphInput::default();
    // Sprite is centered on the entity's transform; in Bevy's default 2D
    // setup, (0,0) is the window center. Compute panel top-left in window
    // pixel space.
    let win_w = 1280.0;
    let win_h = 800.0;
    input.panel_origin = Vec2::new(
        (win_w - PANEL_SIZE.0 as f32) * 0.5,
        (win_h - PANEL_SIZE.1 as f32) * 0.5,
    );

    let mut flame = FlameGraph::new(image.clone(), PANEL_SIZE);
    flame.set_profile(Arc::new(demo_profile()));

    commands.spawn((flame, input, Sprite::from_image(image)));
}

/// Press `r` to swap in a fresh synth profile (different shape) — quick
/// sanity check that profile-swapping mid-session works in the embed.
fn reload_on_r(
    keys: Res<ButtonInput<KeyCode>>,
    mut q: Query<&mut FlameGraph>,
) {
    if keys.just_pressed(KeyCode::KeyR) {
        for mut flame in &mut q {
            flame.set_profile(Arc::new(demo_profile_2()));
        }
    }
}

fn demo_profile() -> flame_bevy::flame_core::Profile {
    let mut b = ProfileBuilder::new();
    let proc = b.add_process(0, "demo");
    let cat = b.intern_category("demo");
    let names: Vec<_> = (0..8)
        .map(|i| b.intern_string(&format!("frame_{i}")))
        .collect();
    for t in 0..4 {
        let thread = b.add_thread(Some(proc), t as i64, &format!("worker {t}"));
        let track = b.add_track(TrackKind::Thread(thread), &format!("worker {t}"), None);
        let total: u64 = 1_000_000;
        let mut width = total;
        for d in 0..8 {
            let start = (total - width) / 2;
            b.add_complete_slice(
                track,
                d as u16,
                start,
                width,
                names[d as usize],
                cat,
                None,
            );
            width = (width * 7) / 10;
        }
    }
    b.finish()
}

fn demo_profile_2() -> flame_bevy::flame_core::Profile {
    let mut b = ProfileBuilder::new();
    let proc = b.add_process(1, "demo2");
    let cat = b.intern_category("alt");
    let names: Vec<_> = (0..6)
        .map(|i| b.intern_string(&format!("alt_{i}")))
        .collect();
    for t in 0..6 {
        let thread = b.add_thread(Some(proc), t as i64, &format!("alt {t}"));
        let track = b.add_track(TrackKind::Thread(thread), &format!("alt {t}"), None);
        let total: u64 = 2_000_000;
        let mut width = total;
        for d in 0..6 {
            let start = (total - width) / 2;
            b.add_complete_slice(
                track,
                d as u16,
                start,
                width,
                names[d as usize],
                cat,
                None,
            );
            width = (width * 6) / 10;
        }
    }
    b.finish()
}
