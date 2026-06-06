//! glaze_gallery — compiles `challenges.glz` and renders every `style`'s shader
//! layer as a tile in a grid, then snapshots. One image, one challenge per tile.
//!
//!   glaze_gallery --out /tmp/gallery.png [--cols 4] [--frames 80]

use std::process::ExitCode;

use bevy::app::AppExit;
use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy::sprite_render::MeshMaterial2d;
use bevy::window::{ExitCondition, WindowPlugin, WindowResolution};

use glaze::{Layer, parse};
use widget_bevy::glaze_material::{GlazeMaterial, GlazeMaterialPlugin, GlazeUniforms, assemble_wgsl};

const SHEET: &str = include_str!("challenges.glz");

const TILE_W: f32 = 220.0;
const TILE_H: f32 = 150.0;
const GAP: f32 = 16.0;
const MARGIN: f32 = 24.0;

#[derive(Resource)]
struct Gallery {
    tiles: Vec<(String, String)>, // (style name, wgsl body)
    cols: usize,
}

#[derive(Resource)]
struct Cfg {
    out: std::path::PathBuf,
    frames: u32,
}

#[derive(Resource, Default)]
struct Frames {
    n: u32,
    fired: bool,
}

fn main() -> ExitCode {
    let mut out = std::path::PathBuf::from("/tmp/glaze_gallery.png");
    let mut cols = 4usize;
    let mut frames = 80u32;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => out = it.next().map(Into::into).unwrap_or(out),
            "--cols" => cols = it.next().and_then(|s| s.parse().ok()).unwrap_or(cols),
            "--frames" => frames = it.next().and_then(|s| s.parse().ok()).unwrap_or(frames),
            _ => {}
        }
    }

    let prog = match parse(SHEET) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("glaze: challenges.glz failed to compile: {e}");
            return ExitCode::from(1);
        }
    };

    // Compile every style's shader layer; report failures loudly.
    let mut tiles = Vec::new();
    for style in &prog.styles {
        match prog.resolve(&style.name, &Default::default(), &[]) {
            Ok(c) => match c.layers.iter().find_map(|l| match l {
                Layer::Shader(s) => Some(s.wgsl_body.clone()),
                _ => None,
            }) {
                Some(body) => tiles.push((style.name.clone(), body)),
                None => eprintln!("glaze: `{}` has no shader layer (skipped)", style.name),
            },
            Err(e) => eprintln!("glaze: `{}` failed: {e}", style.name),
        }
    }
    let n = tiles.len();
    let rows = n.div_ceil(cols);
    eprintln!("glaze_gallery: {n} tiles, {cols}×{rows} grid");
    for (i, (name, _)) in tiles.iter().enumerate() {
        eprintln!("  [{},{}] {}", i / cols, i % cols, name);
    }

    let win_w = (cols as f32 * TILE_W + (cols as f32 - 1.0) * GAP + 2.0 * MARGIN) as u32;
    let win_h = (rows as f32 * TILE_H + (rows as f32 - 1.0) * GAP + 2.0 * MARGIN) as u32;

    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "glaze gallery".into(),
            resolution: WindowResolution::new(win_w, win_h),
            visible: false,
            ..default()
        }),
        exit_condition: ExitCondition::DontExit,
        close_when_requested: true,
        ..default()
    }));
    app.add_plugins(GlazeMaterialPlugin);
    app.insert_resource(ClearColor(Color::srgb(0.03, 0.035, 0.05)));
    app.insert_resource(Gallery { tiles, cols });
    app.insert_resource(Cfg { out, frames });
    app.init_resource::<Frames>();
    app.add_systems(Startup, setup);
    app.add_systems(Update, snapshot_when_ready);
    app.run();
    ExitCode::SUCCESS
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<GlazeMaterial>>,
    mut shaders: ResMut<Assets<Shader>>,
    gallery: Res<Gallery>,
) {
    commands.spawn(Camera2d);
    let mesh = meshes.add(Rectangle::new(TILE_W, TILE_H));

    let cols = gallery.cols as f32;
    let rows = (gallery.tiles.len().div_ceil(gallery.cols)) as f32;
    let grid_w = cols * TILE_W + (cols - 1.0) * GAP;
    let grid_h = rows * TILE_H + (rows - 1.0) * GAP;
    let start_x = -grid_w * 0.5 + TILE_W * 0.5;
    let start_y = grid_h * 0.5 - TILE_H * 0.5;

    for (i, (_name, body)) in gallery.tiles.iter().enumerate() {
        let col = (i % gallery.cols) as f32;
        let row = (i / gallery.cols) as f32;
        let x = start_x + col * (TILE_W + GAP);
        let y = start_y - row * (TILE_H + GAP);

        let shader = shaders.add(Shader::from_wgsl(assemble_wgsl(body), "glaze://gallery.wgsl"));
        let mat = mats.add(GlazeMaterial {
            u: GlazeUniforms {
                size: Vec2::new(TILE_W, TILE_H),
                resolution: Vec2::new(TILE_W, TILE_H),
                ..Default::default()
            },
            fragment: shader,
        });
        commands.spawn((
            bevy::mesh::Mesh2d(mesh.clone()),
            MeshMaterial2d(mat),
            Transform::from_xyz(x, y, 0.0),
        ));
    }
}

fn snapshot_when_ready(
    mut commands: Commands,
    mut frames: ResMut<Frames>,
    cfg: Res<Cfg>,
    mut exit: MessageWriter<AppExit>,
) {
    frames.n += 1;
    if frames.fired {
        if frames.n.saturating_sub(cfg.frames) > 30 {
            exit.write(AppExit::Success);
        }
        return;
    }
    if frames.n < cfg.frames {
        return;
    }
    frames.fired = true;
    eprintln!("glaze_gallery: saving {}", cfg.out.display());
    commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(cfg.out.clone()));
}
