//! Carved-relief UI prototype with an egui debug panel.
//!
//! Press `Tab` to toggle debug mode — that opens the floating egui
//! tweak panel and re-enables keyboard shortcuts:
//!   • W/A/S/D or arrow keys           → pan the camera
//!   • Q / E                           → zoom out / in
//!   • [ / ]                           → camera pitch
//!   • T                               → snap pitch
//!   • R                               → reset
//! Everything else is in the panel.

use bevy::asset::Asset;
use bevy::input::ButtonInput;
use bevy::input::mouse::MouseWheel;
use bevy::mesh::{Mesh, Mesh2d};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;
use bevy::sprite_render::{Material2d, Material2dPlugin, MeshMaterial2d};
use bevy::window::WindowResized;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};

#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct CarvedMaterial {
    /// x = logical width, y = height, z = time, w = device scale
    #[uniform(0)]
    params: Vec4,
    /// xyz = direction toward light (normalized), w = intensity
    #[uniform(1)]
    light0_dir: Vec4,
    /// rgb = light color (multiplied into diffuse), a = unused
    #[uniform(2)]
    light0_col: Vec4,
    #[uniform(3)]
    light1_dir: Vec4,
    #[uniform(4)]
    light1_col: Vec4,
    #[uniform(5)]
    light2_dir: Vec4,
    #[uniform(6)]
    light2_col: Vec4,
    /// x = ambient, y = shadow softness k (higher = sharper),
    /// z = max shadow strength multiplier (0..1), w = unused
    #[uniform(7)]
    shading: Vec4,
    /// xy = pan offset (world px), z = zoom factor, w = tilt (radians)
    #[uniform(8)]
    camera: Vec4,
}

impl Material2d for CarvedMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/carved.wgsl".into()
    }
}

const INITIAL_W: f32 = 1400.0;
const INITIAL_H: f32 = 900.0;

const COLOR_PRESETS: &[(&str, Vec3)] = &[
    ("white",   Vec3::new(1.0, 1.0, 1.0)),
    ("warm",    Vec3::new(1.0, 0.86, 0.70)),
    ("cool",    Vec3::new(0.70, 0.86, 1.0)),
    ("amber",   Vec3::new(1.0, 0.65, 0.30)),
    ("magenta", Vec3::new(1.0, 0.40, 1.0)),
    ("teal",    Vec3::new(0.30, 1.0, 0.85)),
    ("red",     Vec3::new(1.0, 0.30, 0.30)),
    ("green",   Vec3::new(0.40, 1.0, 0.40)),
    ("blue",    Vec3::new(0.45, 0.55, 1.0)),
];

#[derive(Resource)]
struct DebugMode(bool);

#[derive(Clone, Copy)]
struct LightDef {
    dir: Vec3,
    intensity: f32,
    color_idx: usize,
}

impl LightDef {
    fn color(&self) -> Vec3 {
        COLOR_PRESETS[self.color_idx % COLOR_PRESETS.len()].1
    }
    fn color_name(&self) -> &'static str {
        COLOR_PRESETS[self.color_idx % COLOR_PRESETS.len()].0
    }
}

#[derive(Resource)]
struct LightControl {
    lights: [LightDef; 3],
    active: usize,
    ambient: f32,
}

#[derive(Resource)]
struct ShadowControl {
    /// Disc-jitter blur radius in world pixels. 0 = hard shadows.
    blur_radius: f32,
    /// Number of disc samples (1, 4, 8, 16, 32). Higher = smoother but slower.
    n_samples: u32,
    /// Max strength of the shadow term (0..1). Lower = lifted shadows.
    max_strength: f32,
}

#[derive(Resource)]
struct CameraControl {
    pan: Vec2,
    zoom: f32,
    /// Pitch in radians. 0 = top-down, π/2 = horizontal.
    tilt: f32,
}

fn default_lights() -> [LightDef; 3] {
    [
        LightDef {
            dir: Vec3::new(-0.55, -0.55, 0.55),
            intensity: 1.0,
            color_idx: 1, // warm
        },
        LightDef {
            dir: Vec3::new(0.55, -0.40, 0.55),
            intensity: 0.0,
            color_idx: 2, // cool
        },
        LightDef {
            dir: Vec3::new(0.0, 0.55, 0.55),
            intensity: 0.0,
            color_idx: 0, // white
        },
    ]
}

fn main() {
    let asset_path = format!("{}/assets", env!("CARGO_MANIFEST_DIR"));

    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "carved-ui".into(),
                        resolution: (INITIAL_W as u32, INITIAL_H as u32).into(),
                        ..default()
                    }),
                    ..default()
                })
                .set(AssetPlugin {
                    file_path: asset_path,
                    ..default()
                }),
        )
        .add_plugins(Material2dPlugin::<CarvedMaterial>::default())
        .add_plugins(EguiPlugin::default())
        .insert_resource(DebugMode(false))
        .insert_resource(LightControl {
            lights: default_lights(),
            active: 0,
            ambient: 0.28,
        })
        .insert_resource(ShadowControl {
            blur_radius: 4.0,
            n_samples: 8,
            max_strength: 1.0,
        })
        .insert_resource(CameraControl {
            pan: Vec2::ZERO,
            zoom: 1.0,
            tilt: 0.0,
        })
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                toggle_debug,
                debug_inputs,
                fit_quad_to_window,
                sync_uniforms,
            ),
        )
        .add_systems(EguiPrimaryContextPass, debug_panel)
        .run();
}

#[derive(Component)]
struct Backdrop;

fn pack_light(l: &LightDef) -> (Vec4, Vec4) {
    // Normalize on the way out so the resource keeps the user's raw
    // xyz (so the sliders are independent), but the shader gets a
    // unit direction. Fall back to straight-down if the user zeroed
    // everything out.
    let dir = l.dir.try_normalize().unwrap_or(Vec3::Z);
    (dir.extend(l.intensity), l.color().extend(0.0))
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<CarvedMaterial>>,
) {
    commands.spawn(Camera2d);

    let lights = default_lights();
    let (l0d, l0c) = pack_light(&lights[0]);
    let (l1d, l1c) = pack_light(&lights[1]);
    let (l2d, l2c) = pack_light(&lights[2]);

    let mesh = meshes.add(Rectangle::new(1.0, 1.0));
    let material = materials.add(CarvedMaterial {
        params: Vec4::new(INITIAL_W, INITIAL_H, 0.0, 1.0),
        light0_dir: l0d,
        light0_col: l0c,
        light1_dir: l1d,
        light1_col: l1c,
        light2_dir: l2d,
        light2_col: l2c,
        shading: Vec4::new(0.28, 6.0, 1.0, 0.0),
        camera: Vec4::new(0.0, 0.0, 1.0, 0.0),
    });

    commands.spawn((
        Mesh2d(mesh),
        MeshMaterial2d(material),
        Transform::from_scale(Vec3::new(INITIAL_W, INITIAL_H, 1.0)),
        Backdrop,
    ));
}

fn toggle_debug(keys: Res<ButtonInput<KeyCode>>, mut debug: ResMut<DebugMode>) {
    if keys.just_pressed(KeyCode::Tab) {
        debug.0 = !debug.0;
    }
}

fn debug_inputs(
    debug: Res<DebugMode>,
    keys: Res<ButtonInput<KeyCode>>,
    mut wheel: MessageReader<MouseWheel>,
    time: Res<Time>,
    mut light: ResMut<LightControl>,
    mut shadow: ResMut<ShadowControl>,
    mut camera: ResMut<CameraControl>,
) {
    if !debug.0 {
        wheel.clear();
        return;
    }

    // Reset.
    if keys.just_pressed(KeyCode::KeyR) {
        light.lights = default_lights();
        light.active = 0;
        light.ambient = 0.28;
        shadow.blur_radius = 4.0;
        shadow.n_samples = 8;
        shadow.max_strength = 1.0;
        camera.pan = Vec2::ZERO;
        camera.zoom = 1.0;
        camera.tilt = 0.0;
    }

    // Active light selection.
    if keys.just_pressed(KeyCode::Digit1) {
        light.active = 0;
    }
    if keys.just_pressed(KeyCode::Digit2) {
        light.active = 1;
    }
    if keys.just_pressed(KeyCode::Digit3) {
        light.active = 2;
    }

    // Toggle active light.
    if keys.just_pressed(KeyCode::Digit0) {
        let idx = light.active;
        let cur = light.lights[idx].intensity;
        light.lights[idx].intensity = if cur > 0.01 { 0.0 } else { 1.0 };
    }

    // Cycle active light's color.
    if keys.just_pressed(KeyCode::KeyC) {
        let idx = light.active;
        light.lights[idx].color_idx =
            (light.lights[idx].color_idx + 1) % COLOR_PRESETS.len();
    }

    // Discard any mouse wheel events so they don't queue up.
    wheel.clear();

    // Shadow blur radius: F = sharper (smaller), G = softer (larger).
    let dt = time.delta_secs();
    let blur_rate = 60.0; // px / sec
    if keys.pressed(KeyCode::KeyF) {
        shadow.blur_radius = (shadow.blur_radius - blur_rate * dt).max(0.0);
    }
    if keys.pressed(KeyCode::KeyG) {
        shadow.blur_radius = (shadow.blur_radius + blur_rate * dt).min(60.0);
    }
    // Sample count cycle (N): 1 → 4 → 8 → 16 → 32 → 1.
    if keys.just_pressed(KeyCode::KeyN) {
        shadow.n_samples = match shadow.n_samples {
            0 | 1 => 4,
            2..=4 => 8,
            5..=8 => 16,
            9..=16 => 32,
            _ => 1,
        };
    }

    // Camera pan.
    let pan_speed = 480.0 / camera.zoom;
    let mut delta = Vec2::ZERO;
    if keys.pressed(KeyCode::KeyW) || keys.pressed(KeyCode::ArrowUp) {
        delta.y -= 1.0;
    }
    if keys.pressed(KeyCode::KeyS) || keys.pressed(KeyCode::ArrowDown) {
        delta.y += 1.0;
    }
    if keys.pressed(KeyCode::KeyA) || keys.pressed(KeyCode::ArrowLeft) {
        delta.x -= 1.0;
    }
    if keys.pressed(KeyCode::KeyD) || keys.pressed(KeyCode::ArrowRight) {
        delta.x += 1.0;
    }
    if delta != Vec2::ZERO {
        camera.pan += delta.normalize() * pan_speed * dt;
    }

    // Zoom.
    let zoom_rate = 1.6_f32.powf(dt);
    if keys.pressed(KeyCode::KeyE) {
        camera.zoom = (camera.zoom * zoom_rate).min(8.0);
    }
    if keys.pressed(KeyCode::KeyQ) {
        camera.zoom = (camera.zoom / zoom_rate).max(0.2);
    }

    // Camera pitch.
    let tilt_speed = 1.2;
    let tilt_max = 1.4;
    if keys.pressed(KeyCode::BracketLeft) {
        camera.tilt = (camera.tilt - tilt_speed * dt).max(0.0);
    }
    if keys.pressed(KeyCode::BracketRight) {
        camera.tilt = (camera.tilt + tilt_speed * dt).min(tilt_max);
    }
    if keys.just_pressed(KeyCode::KeyT) {
        camera.tilt = if camera.tilt < 0.05 { 0.6 } else { 0.0 };
    }
}

fn sync_uniforms(
    time: Res<Time>,
    windows: Query<&Window>,
    light: Res<LightControl>,
    shadow: Res<ShadowControl>,
    camera: Res<CameraControl>,
    mut materials: ResMut<Assets<CarvedMaterial>>,
) {
    let Ok(win) = windows.single() else { return };
    let scale = win.scale_factor();
    let w = win.physical_width() as f32 / scale;
    let h = win.physical_height() as f32 / scale;
    let (l0d, l0c) = pack_light(&light.lights[0]);
    let (l1d, l1c) = pack_light(&light.lights[1]);
    let (l2d, l2c) = pack_light(&light.lights[2]);
    for (_, mat) in materials.iter_mut() {
        mat.params = Vec4::new(w, h, time.elapsed_secs(), scale);
        mat.light0_dir = l0d;
        mat.light0_col = l0c;
        mat.light1_dir = l1d;
        mat.light1_col = l1c;
        mat.light2_dir = l2d;
        mat.light2_col = l2c;
        mat.shading = Vec4::new(
            light.ambient,
            shadow.blur_radius,
            shadow.max_strength,
            shadow.n_samples as f32,
        );
        mat.camera = Vec4::new(camera.pan.x, camera.pan.y, camera.zoom, camera.tilt);
    }
}

fn fit_quad_to_window(
    mut resized: MessageReader<WindowResized>,
    mut q: Query<&mut Transform, With<Backdrop>>,
) {
    for ev in resized.read() {
        for mut tf in q.iter_mut() {
            tf.scale = Vec3::new(ev.width, ev.height, 1.0);
        }
    }
}

fn debug_panel(
    mut contexts: EguiContexts,
    debug: Res<DebugMode>,
    mut light: ResMut<LightControl>,
    mut shadow: ResMut<ShadowControl>,
    mut camera: ResMut<CameraControl>,
) -> Result {
    if !debug.0 {
        return Ok(());
    }
    let ctx = contexts.ctx_mut()?;

    egui::Window::new("Carved-UI Debug")
        .default_pos([12.0, 12.0])
        .default_width(280.0)
        .show(ctx, |ui| {
            ui.collapsing("Lights", |ui| {
                ui.add(
                    egui::Slider::new(&mut light.ambient, 0.0..=1.0).text("ambient"),
                );
                ui.separator();

                let active = light.active;
                for i in 0..3 {
                    let header = format!(
                        "Light {}{}",
                        i + 1,
                        if i == active { "  (active)" } else { "" },
                    );
                    egui::CollapsingHeader::new(header)
                        .default_open(i == 0 || light.lights[i].intensity > 0.01)
                        .show(ui, |ui| {
                            if ui.button("Make active").clicked() {
                                light.active = i;
                            }
                            let l = &mut light.lights[i];
                            let mut on = l.intensity > 0.01;
                            if ui.checkbox(&mut on, "enabled").changed() {
                                l.intensity = if on { 1.0_f32.max(l.intensity) } else { 0.0 };
                            }
                            ui.add(
                                egui::Slider::new(&mut l.intensity, 0.0..=2.0)
                                    .text("intensity"),
                            );

                            // Direction sliders — independent. The
                            // shader receives a normalized direction;
                            // the raw xyz here just sets which way it
                            // points (magnitude doesn't matter).
                            ui.add(egui::Slider::new(&mut l.dir.x, -2.0..=2.0).text("dir x"));
                            ui.add(egui::Slider::new(&mut l.dir.y, -2.0..=2.0).text("dir y"));
                            ui.add(
                                egui::Slider::new(&mut l.dir.z, 0.0..=2.0)
                                    .text("dir z (elevation)"),
                            );

                            // Color preset dropdown.
                            let cur_name = l.color_name();
                            egui::ComboBox::from_label("color")
                                .selected_text(cur_name)
                                .show_ui(ui, |ui| {
                                    for (idx, (name, _)) in COLOR_PRESETS.iter().enumerate() {
                                        ui.selectable_value(
                                            &mut l.color_idx,
                                            idx,
                                            *name,
                                        );
                                    }
                                });
                        });
                }
            });

            ui.collapsing("Shadows", |ui| {
                ui.add(
                    egui::Slider::new(&mut shadow.blur_radius, 0.0..=80.0)
                        .text("blur radius (px)"),
                );
                let mut n = shadow.n_samples as i32;
                ui.horizontal(|ui| {
                    ui.label("samples:");
                    for &opt in &[1, 4, 8, 16, 32] {
                        ui.selectable_value(&mut n, opt, opt.to_string());
                    }
                });
                shadow.n_samples = n.max(1) as u32;
                ui.add(
                    egui::Slider::new(&mut shadow.max_strength, 0.0..=1.0)
                        .text("max strength"),
                );
            });

            ui.collapsing("Camera", |ui| {
                ui.add(egui::Slider::new(&mut camera.pan.x, -1500.0..=1500.0).text("pan x"));
                ui.add(egui::Slider::new(&mut camera.pan.y, -1500.0..=1500.0).text("pan y"));
                ui.add(egui::Slider::new(&mut camera.zoom, 0.2..=8.0).text("zoom"));
                let mut tilt_deg = camera.tilt.to_degrees();
                if ui
                    .add(egui::Slider::new(&mut tilt_deg, 0.0..=80.0).text("tilt (°)"))
                    .changed()
                {
                    camera.tilt = tilt_deg.to_radians();
                }
            });

            ui.separator();
            ui.horizontal(|ui| {
                if ui.button("Reset all").clicked() {
                    light.lights = default_lights();
                    light.active = 0;
                    light.ambient = 0.28;
                    shadow.blur_radius = 4.0;
                    shadow.n_samples = 8;
                    shadow.max_strength = 1.0;
                    camera.pan = Vec2::ZERO;
                    camera.zoom = 1.0;
                    camera.tilt = 0.0;
                }
                ui.label("Tab to hide");
            });
        });
    Ok(())
}
