//! Text-rendering throughput comparison: spawn N short labels, then
//! count each up by 0.01 every frame so the text genuinely changes
//! every frame (4000 entities × ~5 chars × 60 fps = 1.2 M glyph
//! updates/sec — the workload that exposes the cosmic-text path).
//!
//! Three modes:
//!
//!   * `text2d`     — `bevy_sprite::Text2d` (cosmic-text pipeline).
//!   * `atlas`      — vendored `flow_bevy::glyph_atlas::GlyphAtlas`
//!                    (Jimmy's terminal atlas: swash-rasterized
//!                    1024² atlas, one `Sprite` per char).
//!   * `image_font` — `bevy_image_font` 0.11's `ImageFontSpriteText`
//!                    using its bundled example pixel font.
//!
//! Usage:
//!     cargo run --release -p flow-bevy --bin text_bench -- \
//!         <text2d|atlas|image_font> [duration_seconds] [num_labels]
//!
//! Duration defaults to 10s, num_labels to 4000.

use bevy::app::AppExit;
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::prelude::TextureAtlas;
use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::window::PresentMode;

use bevy_image_font::atlas_sprites::ImageFontSpriteText;
use bevy_image_font::{ImageFont, ImageFontPlugin, ImageFontText};

use flow_bevy::glyph_atlas::GlyphAtlas;

const FONT_PATH: &str = "/Library/Fonts/SF-Mono-Regular.otf";
const FONT_SIZE: f32 = 11.0;
const LINE_HEIGHT: f32 = 14.0;
const WARMUP_SECS: f64 = 1.0;
const DEFAULT_DURATION_SECS: f64 = 10.0;
const DEFAULT_NUM_LABELS: usize = 4000;
/// Max chars we'll ever render per label. Atlas mode pre-spawns this
/// many child sprites so per-frame updates only mutate components,
/// never spawn or despawn.
const ATLAS_MAX_CHARS: usize = 8;

#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Text2d,
    Atlas,
    ImageFont,
}

impl Mode {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "text2d" => Ok(Self::Text2d),
            "atlas" => Ok(Self::Atlas),
            "image_font" => Ok(Self::ImageFont),
            other => Err(format!(
                "unknown mode {other:?}; expected text2d|atlas|image_font"
            )),
        }
    }
}

#[derive(Resource)]
struct BenchState {
    mode: Mode,
    duration_s: f64,
    num_labels: usize,
    frame_samples: Vec<f64>,
    reported: bool,
}

/// Per-label running counter. Incremented by 0.01 each frame and
/// rendered as `format!("{:.2}", value)`.
#[derive(Component)]
struct LabelValue(f32);

/// On Atlas mode, each label is a parent entity carrying
/// `LabelValue` plus this list of `ATLAS_MAX_CHARS` pre-spawned
/// child sprite entities (in left-to-right order). The update
/// system rewrites the children's `TextureAtlas.index` and
/// visibility per frame.
#[derive(Component)]
struct AtlasLabelChildren([Entity; ATLAS_MAX_CHARS]);

/// Atlas char→slot lookup populated at startup with every glyph
/// our update path could ask for. Letting the update system stay
/// purely component-mutating with no `&mut Atlas`.
#[derive(Resource)]
struct AtlasGlyphTable {
    image: Handle<Image>,
    layout: Handle<bevy::image::TextureAtlasLayout>,
    /// Slot for each ASCII byte. Filled during setup; misses
    /// (chars we never use) stay at slot 0 (tofu).
    slots: [u32; 128],
    cell_w_logical: f32,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mode = Mode::parse(
        &args.next().expect("usage: text_bench <text2d|atlas|image_font> [secs] [n]"),
    )
    .unwrap();
    let duration_s: f64 = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_DURATION_SECS);
    let num_labels: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_NUM_LABELS);

    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: format!("text_bench [{:?}]", mode),
            resolution: (1400u32, 900u32).into(),
            present_mode: PresentMode::AutoNoVsync,
            ..default()
        }),
        ..default()
    }));
    app.add_plugins(FrameTimeDiagnosticsPlugin::default());
    app.add_plugins(ImageFontPlugin);
    app.insert_resource(BenchState {
        mode,
        duration_s,
        num_labels,
        frame_samples: Vec::with_capacity(120_000),
        reported: false,
    });
    app.add_systems(Startup, setup);
    app.add_systems(
        Update,
        (
            tick_label_values,
            (
                update_text2d_labels.run_if(mode_is(Mode::Text2d)),
                update_atlas_labels.run_if(mode_is(Mode::Atlas)),
                update_image_font_labels.run_if(mode_is(Mode::ImageFont)),
            ),
            sample_and_maybe_exit,
        )
            .chain(),
    );
    app.run();
}

fn mode_is(want: Mode) -> impl Fn(Res<BenchState>) -> bool + Copy {
    move |state: Res<BenchState>| state.mode == want
}

fn setup(
    mut commands: Commands,
    state: Res<BenchState>,
    asset_server: Res<AssetServer>,
    mut images: ResMut<Assets<Image>>,
    mut layouts: ResMut<Assets<bevy::image::TextureAtlasLayout>>,
) {
    commands.spawn(Camera2d);

    let n = state.num_labels;
    let cols = (n as f32).sqrt().ceil() as usize;
    let rows = (n + cols - 1) / cols;
    let cell_w = 60.0_f32;
    let cell_h = 18.0_f32;
    let origin_x = -(cols as f32 * cell_w) * 0.5;
    let origin_y = (rows as f32 * cell_h) * 0.5;

    eprintln!(
        "text_bench: mode={:?} labels={} grid={}×{}",
        state.mode, n, cols, rows
    );

    match state.mode {
        Mode::Text2d => {
            for i in 0..n {
                let (col, row) = (i % cols, i / cols);
                let x = origin_x + col as f32 * cell_w;
                let y = origin_y - row as f32 * cell_h;
                commands.spawn((
                    Text2d::new("0.00"),
                    TextColor(Color::WHITE),
                    TextFont { font_size: FONT_SIZE, ..default() },
                    Transform::from_xyz(x, y, 0.0),
                    LabelValue(i as f32 * 0.001),
                ));
            }
        }
        Mode::Atlas => {
            let font_bytes: &'static [u8] = {
                let bytes = std::fs::read(FONT_PATH)
                    .unwrap_or_else(|e| panic!("read {FONT_PATH}: {e}"));
                Box::leak(bytes.into_boxed_slice())
            };
            let cell_w_logical = measure_advance(font_bytes, FONT_SIZE, '0');
            let mut atlas = GlyphAtlas::new(
                font_bytes,
                FONT_SIZE,
                cell_w_logical,
                LINE_HEIGHT,
                &mut images,
                &mut layouts,
            );
            // Pre-load every char we'll need (digits, '.', '-', and
            // letters used in any other label format).
            let mut slots = [0u32; 128];
            for byte in 0..128u8 {
                let ch = byte as char;
                slots[byte as usize] = atlas.lookup_or_insert(ch, &mut images, &mut layouts);
            }
            let table = AtlasGlyphTable {
                image: atlas.image.clone(),
                layout: atlas.layout.clone(),
                slots,
                cell_w_logical,
            };

            for i in 0..n {
                let (col, row) = (i % cols, i / cols);
                let x = origin_x + col as f32 * cell_w;
                let y = origin_y - row as f32 * cell_h;
                let parent = commands
                    .spawn((
                        Transform::from_xyz(x, y, 0.0),
                        Visibility::Inherited,
                        LabelValue(i as f32 * 0.001),
                    ))
                    .id();
                let mut kids = [Entity::PLACEHOLDER; ATLAS_MAX_CHARS];
                for k in 0..ATLAS_MAX_CHARS {
                    let cx = (k as f32) * cell_w_logical;
                    let child = commands
                        .spawn((
                            Sprite {
                                image: table.image.clone(),
                                texture_atlas: Some(TextureAtlas {
                                    layout: table.layout.clone(),
                                    index: 0, // tofu, will be set on first update
                                }),
                                color: Color::WHITE,
                                custom_size: Some(Vec2::new(cell_w_logical, LINE_HEIGHT)),
                                ..default()
                            },
                            Anchor::TOP_LEFT,
                            Transform::from_xyz(cx, 0.0, 0.0),
                            Visibility::Hidden,
                        ))
                        .id();
                    commands.entity(parent).add_child(child);
                    kids[k] = child;
                }
                commands.entity(parent).insert(AtlasLabelChildren(kids));
            }
            commands.insert_resource(atlas);
            commands.insert_resource(table);
        }
        Mode::ImageFont => {
            let font_handle: Handle<ImageFont> = asset_server.load("example_font.image_font.ron");
            for i in 0..n {
                let (col, row) = (i % cols, i / cols);
                let x = origin_x + col as f32 * cell_w;
                let y = origin_y - row as f32 * cell_h;
                commands.spawn((
                    ImageFontSpriteText::default().color(Color::WHITE),
                    ImageFontText::default()
                        .text("0.00")
                        .font(font_handle.clone()),
                    Transform::from_xyz(x, y, 0.0),
                    LabelValue(i as f32 * 0.001),
                ));
            }
        }
    }
}

fn tick_label_values(mut q: Query<&mut LabelValue>) {
    for mut v in q.iter_mut() {
        v.0 += 0.01;
    }
}

fn update_text2d_labels(mut q: Query<(&LabelValue, &mut Text2d)>) {
    let mut buf = String::with_capacity(8);
    for (v, mut text) in q.iter_mut() {
        buf.clear();
        use std::fmt::Write;
        write!(&mut buf, "{:.2}", v.0).ok();
        if text.0 != buf {
            text.0.clear();
            text.0.push_str(&buf);
        }
    }
}

fn update_atlas_labels(
    table: Res<AtlasGlyphTable>,
    parents: Query<(&LabelValue, &AtlasLabelChildren)>,
    mut sprites: Query<(&mut Sprite, &mut Visibility)>,
) {
    let mut buf = String::with_capacity(8);
    for (v, kids) in parents.iter() {
        buf.clear();
        use std::fmt::Write;
        write!(&mut buf, "{:.2}", v.0).ok();
        let bytes = buf.as_bytes();
        for k in 0..ATLAS_MAX_CHARS {
            let Ok((mut sprite, mut vis)) = sprites.get_mut(kids.0[k]) else { continue };
            if k < bytes.len() {
                let slot = table.slots[bytes[k] as usize];
                if let Some(atlas) = sprite.texture_atlas.as_mut() {
                    if atlas.index != slot as usize {
                        atlas.index = slot as usize;
                    }
                }
                if !matches!(*vis, Visibility::Inherited | Visibility::Visible) {
                    *vis = Visibility::Inherited;
                }
            } else if !matches!(*vis, Visibility::Hidden) {
                *vis = Visibility::Hidden;
            }
        }
        let _ = table.cell_w_logical; // referenced for layout, no per-frame use
    }
}

fn update_image_font_labels(mut q: Query<(&LabelValue, &mut ImageFontText)>) {
    let mut buf = String::with_capacity(8);
    for (v, mut text) in q.iter_mut() {
        buf.clear();
        use std::fmt::Write;
        write!(&mut buf, "{:.2}", v.0).ok();
        if text.text != buf {
            text.text = buf.clone();
        }
    }
}

fn sample_and_maybe_exit(
    time: Res<Time<Real>>,
    mut state: ResMut<BenchState>,
    mut exit: bevy::ecs::message::MessageWriter<AppExit>,
) {
    if state.reported {
        return;
    }
    let elapsed = time.elapsed_secs_f64();
    let dt_ms = time.delta_secs_f64() * 1000.0;
    if elapsed < WARMUP_SECS {
        return;
    }
    state.frame_samples.push(dt_ms);
    if elapsed >= WARMUP_SECS + state.duration_s {
        report(&state);
        state.reported = true;
        exit.write(AppExit::Success);
    }
}

fn report(state: &BenchState) {
    let n = state.frame_samples.len();
    if n == 0 {
        eprintln!("text_bench[{:?}]: no frames captured", state.mode);
        return;
    }
    let mut sorted = state.frame_samples.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean: f64 = sorted.iter().sum::<f64>() / n as f64;
    let q = |p: f64| sorted[((n - 1) as f64 * p) as usize];
    let line = |label: &str, ms: f64| {
        println!("  {label:<8}: {ms:7.3} ms  ({:6.1} fps)", 1000.0 / ms);
    };
    println!();
    println!("=== text_bench[{:?}] ===", state.mode);
    println!("  labels  : {}", state.num_labels);
    println!("  frames  : {n}");
    println!("  window  : {:.2}s after {:.2}s warmup", state.duration_s, WARMUP_SECS);
    line("mean", mean);
    line("p50", q(0.50));
    line("p95", q(0.95));
    line("p99", q(0.99));
    line("max", sorted[n - 1]);
}

fn measure_advance(font_bytes: &[u8], font_size: f32, ch: char) -> f32 {
    use swash::FontRef;
    let font = FontRef::from_index(font_bytes, 0).expect("font parse");
    let glyph_id = font.charmap().map(ch);
    if glyph_id == 0 {
        return font_size * 0.6;
    }
    let metrics = font.metrics(&[]).scale(font_size);
    let advance = metrics.average_width;
    if advance > 0.0 { advance } else { font_size * 0.6 }
}
