//! Cover the whole Bevy window with a flame graph.
//!
//! Run with: `cargo run --example embed_fullscreen -p flame-bevy`
//!
//! The flame graph fills the primary window; when the window is resized, the
//! panel resizes with it.

use std::sync::Arc;

use bevy::prelude::*;
use bevy::window::WindowResized;
use flame_bevy::flame_core::{ProfileBuilder, TrackKind};
use flame_bevy::{FlameGraph, FlameGraphInput, FlameGraphPlugin};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "flame-bevy: embed_fullscreen".into(),
                resolution: (1280u32, 800u32).into(),
                ..Default::default()
            }),
            ..Default::default()
        }))
        .add_plugins(FlameGraphPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, fit_to_window)
        .run();
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    windows: Query<&Window>,
) {
    let win = windows.single().expect("primary window");
    let w = win.physical_width().max(1);
    let h = win.physical_height().max(1);

    let image = images.add(FlameGraph::blank_image(w, h));

    commands.spawn(Camera2d);

    let mut flame = FlameGraph::new(image.clone(), (w, h));
    flame.set_profile(Arc::new(demo_profile()));

    commands.spawn((
        flame,
        FlameGraphInput::default(),
        Sprite::from_image(image),
    ));
}

/// On window resize, resize the source `Image` to match. The plugin's
/// `resize_image_to_panel` system then picks that up and resizes the
/// renderer. We resize via the image (rather than the FlameGraph directly)
/// so a host that also wants to use Bevy's own resizing of UiImage's sees
/// the consistent dimensions.
fn fit_to_window(
    mut resized: MessageReader<WindowResized>,
    flames: Query<&FlameGraph>,
    mut images: ResMut<Assets<Image>>,
) {
    let Some(ev) = resized.read().last() else { return };
    let w = ev.width.max(1.0) as u32;
    let h = ev.height.max(1.0) as u32;
    for flame in &flames {
        let Some(img) = images.get_mut(&flame.image()) else { continue };
        let cur = img.texture_descriptor.size;
        if cur.width == w && cur.height == h {
            continue;
        }
        img.texture_descriptor.size = bevy::render::render_resource::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };
        // Reallocate the data buffer; the plugin will overwrite it on next paint.
        img.data = Some(vec![0; (w * h * 4) as usize]);
    }
}

fn demo_profile() -> flame_bevy::flame_core::Profile {
    let mut b = ProfileBuilder::new();
    let proc = b.add_process(0, "fullscreen-demo");
    let cat = b.intern_category("demo");
    let names: Vec<_> = (0..10)
        .map(|i| b.intern_string(&format!("f{i}")))
        .collect();
    for t in 0..6 {
        let thread = b.add_thread(Some(proc), t as i64, &format!("thread {t}"));
        let track = b.add_track(TrackKind::Thread(thread), &format!("thread {t}"), None);
        let total: u64 = 3_000_000;
        let mut width = total;
        for d in 0..10 {
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
            width = (width * 8) / 10;
        }
    }
    b.finish()
}
