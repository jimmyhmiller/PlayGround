//! glaze_shader_demo — Stage 3 end-to-end: a `.glz` `overlay shader {}` block is
//! compiled by `glaze` (binding-time analysis → WGSL), wrapped in a `Material2d`,
//! and run on a fullscreen quad on the GPU. `time`/`uv` drive a pulsing radial
//! glow; static tokens/constants were folded by the compiler before the shader
//! ever ran.
//!
//! Headless: renders N frames then writes a screenshot and exits.
//!   glaze_shader_demo --out /tmp/glow.png [--size 700x460] [--frames 90]

use std::path::PathBuf;
use std::process::ExitCode;

use bevy::app::AppExit;
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy::shader::ShaderRef;
use bevy::sprite_render::{AlphaMode2d, Material2d, Material2dPlugin, MeshMaterial2d};
use bevy::window::{ExitCondition, WindowPlugin, WindowResolution};

use glaze::{Layer, parse};

// The stylesheet. A spatial, animated shader authored in Glaze.
const SHEET: &str = r#"
    token gold = oklch(0.84 0.13 85)
    style hero {
      overlay shader {
        let d     = length(uv - vec2(0.5, 0.5))   // uv → per-fragment (dynamic)
        let ring  = smoothstep(0.5, 0.0, d)
        let pulse = 0.55 + 0.45*sin(time*2.5)      // time → uniform (dynamic)
        let base  = vec4(0.06, 0.07, 0.10, 1.0)    // folded constant
        emit base + ring * pulse * gold            // gold token → folded constant
      }
    }
"#;

/// Canonical per-frame inputs. Field order/layout matches the WGSL struct below;
/// `encase` (via `ShaderType`) inserts the same std140 padding WGSL does.
#[derive(Clone, Copy, ShaderType)]
struct GlazeUniforms {
    time: f32,
    dt: f32,
    hover: f32,
    focus: f32,
    press: f32,
    mouse: Vec2,
    size: Vec2,
    resolution: Vec2,
}

impl Default for GlazeUniforms {
    fn default() -> Self {
        GlazeUniforms {
            time: 0.0,
            dt: 0.0,
            hover: 1.0,
            focus: 0.0,
            press: 0.0,
            mouse: Vec2::ZERO,
            size: Vec2::splat(1.0),
            resolution: Vec2::splat(1.0),
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct GlazeMaterial {
    #[uniform(0)]
    u: GlazeUniforms,
}

impl Material2d for GlazeMaterial {
    fn fragment_shader() -> ShaderRef {
        // Loaded from the asset dir we populate at startup (see `main`).
        ShaderRef::Path("glow.wgsl".into())
    }
    fn alpha_mode(&self) -> AlphaMode2d {
        AlphaMode2d::Blend
    }
}

/// Wrap the compiler's fragment body in a complete mesh2d fragment shader with
/// the canonical uniform block. This boilerplate is the host's job; the *logic*
/// inside came entirely from the Glaze compiler.
fn assemble_wgsl(body: &str) -> String {
    format!(
        "#import bevy_sprite::mesh2d_vertex_output::VertexOutput\n\
         \n\
         struct GlazeUniforms {{\n\
         \x20   time: f32,\n\x20   dt: f32,\n\x20   hover: f32,\n\x20   focus: f32,\n\
         \x20   press: f32,\n\x20   mouse: vec2<f32>,\n\x20   size: vec2<f32>,\n\x20   resolution: vec2<f32>,\n\
         }};\n\
         @group(#{{MATERIAL_BIND_GROUP}}) @binding(0) var<uniform> u: GlazeUniforms;\n\
         \n\
         @fragment\n\
         fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {{\n{body}}}\n"
    )
}

#[derive(Resource)]
struct Cfg {
    out: PathBuf,
    size: Vec2,
    frames: u32,
}

#[derive(Resource, Default)]
struct Frames {
    n: u32,
    fired: bool,
}

fn main() -> ExitCode {
    // ---- args ----
    let mut out = PathBuf::from("/tmp/glaze_glow.png");
    let mut size = Vec2::new(700.0, 460.0);
    let mut frames = 90u32;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => out = it.next().map(PathBuf::from).unwrap_or(out),
            "--size" => {
                if let Some((w, h)) = it.next().and_then(|s| {
                    let (a, b) = s.split_once('x')?;
                    Some((a.parse().ok()?, b.parse().ok()?))
                }) {
                    size = Vec2::new(w, h);
                }
            }
            "--frames" => frames = it.next().and_then(|s| s.parse().ok()).unwrap_or(frames),
            _ => {}
        }
    }

    // ---- compile the Glaze shader ----
    let prog = match parse(SHEET) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("glaze: stylesheet failed to compile: {e}");
            return ExitCode::from(1);
        }
    };
    let compiled = match prog.resolve("hero", &Default::default(), &[]) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("glaze: resolve failed: {e}");
            return ExitCode::from(1);
        }
    };
    let shader = compiled.layers.iter().find_map(|l| match l {
        Layer::Shader(s) => Some(s),
        _ => None,
    });
    let Some(shader) = shader else {
        eprintln!("glaze: no shader layer in `hero`");
        return ExitCode::from(1);
    };
    let wgsl = assemble_wgsl(&shader.wgsl_body);
    eprintln!("--- generated WGSL ---\n{wgsl}\n--- uniforms used: {:?} ---", shader.used);

    // ---- write the shader where the AssetServer can load it ----
    let asset_dir = std::env::temp_dir().join("glaze_shader_demo_assets");
    if let Err(e) = std::fs::create_dir_all(&asset_dir) {
        eprintln!("glaze: mkdir {}: {e}", asset_dir.display());
        return ExitCode::from(1);
    }
    if let Err(e) = std::fs::write(asset_dir.join("glow.wgsl"), &wgsl) {
        eprintln!("glaze: write shader: {e}");
        return ExitCode::from(1);
    }

    // ---- app ----
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "glaze shader demo".into(),
                    resolution: WindowResolution::new(size.x as u32, size.y as u32),
                    visible: false,
                    ..default()
                }),
                exit_condition: ExitCondition::DontExit,
                close_when_requested: true,
                ..default()
            })
            .set(AssetPlugin {
                file_path: asset_dir.to_string_lossy().into_owned(),
                ..default()
            }),
    );
    app.add_plugins(Material2dPlugin::<GlazeMaterial>::default());
    app.insert_resource(ClearColor(Color::srgb(0.04, 0.05, 0.07)));
    app.insert_resource(Cfg { out, size, frames });
    app.init_resource::<Frames>();
    app.add_systems(Startup, setup);
    app.add_systems(Update, (animate, snapshot_when_ready));
    app.run();
    ExitCode::SUCCESS
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<GlazeMaterial>>,
    cfg: Res<Cfg>,
) {
    commands.spawn(Camera2d);
    let mesh = meshes.add(Rectangle::new(cfg.size.x, cfg.size.y));
    let mat = mats.add(GlazeMaterial {
        u: GlazeUniforms {
            resolution: cfg.size,
            size: cfg.size,
            ..Default::default()
        },
    });
    commands.spawn((bevy::mesh::Mesh2d(mesh), MeshMaterial2d(mat), Transform::default()));
}

fn animate(
    time: Res<Time>,
    mats_q: Query<&MeshMaterial2d<GlazeMaterial>>,
    mut mats: ResMut<Assets<GlazeMaterial>>,
) {
    let t = time.elapsed_secs();
    for handle in &mats_q {
        if let Some(m) = mats.get_mut(&handle.0) {
            m.u.time = t;
            m.u.dt = time.delta_secs();
        }
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
    let out = cfg.out.clone();
    eprintln!("glaze: saving {}", out.display());
    commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(out));
}
