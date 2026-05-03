//! Carved-relief UI prototype rebuilt on Bevy's 3D pipeline.
//!
//! Each "paper layer" is a flat 3D mesh at its own z height. Lights
//! are real `DirectionalLight`s with cascaded shadow maps — Bevy
//! does the lighting work the previous version was hand-rolling per
//! fragment. The single fullscreen ray-marched shader is gone.
//!
//! Press Tab to toggle debug mode. Controls:
//!   • W/A/S/D or arrow keys   → pan camera
//!   • Q / E                   → zoom out / in
//!   • [ / ]                   → camera pitch
//!   • T                       → snap pitch
//!   • R                       → reset
//!   • 1 / 2 / 3               → select active light
//!   • 0                       → toggle active light
//!   • C                       → cycle active light's color

use bevy::asset::RenderAssetUsages;
use bevy::input::ButtonInput;
use bevy::input::mouse::MouseWheel;
use bevy::light::{CascadeShadowConfigBuilder, GlobalAmbientLight};
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};

// Scene dimensions in world units (meters). Heights and xy positions
// from the original demo are given in "shader pixels"; PX scales them
// to world units. Keep heights and xy on the same scale so the carved
// look is geometrically honest.
const SCENE_W: f32 = 14.0;
const SCENE_H: f32 = 9.0;
const PX: f32 = 0.01;

const COLOR_PRESETS: &[(&str, Color)] = &[
    ("white",   Color::srgb(1.0,  1.0,  1.0 )),
    ("warm",    Color::srgb(1.0,  0.86, 0.70)),
    ("cool",    Color::srgb(0.70, 0.86, 1.0 )),
    ("amber",   Color::srgb(1.0,  0.65, 0.30)),
    ("magenta", Color::srgb(1.0,  0.40, 1.0 )),
    ("teal",    Color::srgb(0.30, 1.0,  0.85)),
    ("red",     Color::srgb(1.0,  0.30, 0.30)),
    ("green",   Color::srgb(0.40, 1.0,  0.40)),
    ("blue",    Color::srgb(0.45, 0.55, 1.0 )),
];

#[derive(Resource)]
struct DebugMode(bool);

#[derive(Clone, Copy)]
struct LightDef {
    /// Direction TOWARD the light source, in original shader space
    /// (x right, y down, z up). Translated to world space at sync time.
    dir: Vec3,
    intensity: f32,
    color_idx: usize,
}

impl LightDef {
    fn color(&self) -> Color {
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
struct CameraControl {
    pan: Vec2,
    zoom: f32,
    /// Pitch in radians. 0 = top-down, π/2 = horizontal.
    tilt: f32,
}

#[derive(Component)]
struct DirLightIndex(usize);

fn default_lights() -> [LightDef; 3] {
    [
        LightDef { dir: Vec3::new(-0.55, -0.55, 0.55), intensity: 1.0, color_idx: 1 },
        LightDef { dir: Vec3::new( 0.55, -0.40, 0.55), intensity: 0.0, color_idx: 2 },
        LightDef { dir: Vec3::new( 0.0,   0.55, 0.55), intensity: 0.0, color_idx: 0 },
    ]
}

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "carved-ui (3D)".into(),
                    resolution: (1400, 900).into(),
                    ..default()
                }),
                ..default()
            }),
        )
        .add_plugins(EguiPlugin::default())
        .insert_resource(DebugMode(false))
        .insert_resource(LightControl {
            lights: default_lights(),
            active: 0,
            ambient: 0.28,
        })
        .insert_resource(CameraControl {
            pan: Vec2::ZERO,
            zoom: 1.0,
            tilt: 0.0,
        })
        .insert_resource(GlobalAmbientLight {
            color: Color::WHITE,
            brightness: 220.0,
            affects_lightmapped_meshes: true,
        })
        .insert_resource(ClearColor(Color::srgb(0.05, 0.04, 0.07)))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                toggle_debug,
                debug_inputs,
                sync_lights,
                sync_ambient,
                sync_camera,
            ),
        )
        .add_systems(EguiPrimaryContextPass, debug_panel)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Camera — perspective. With tilt=0 it looks straight down the +z
    // axis; tilt rotates it around the scene center. Distance / fov
    // tuned so the 14x9 scene comfortably fills a 1400x900 window.
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.0, 20.0).looking_at(Vec3::ZERO, Vec3::Y),
        Projection::from(PerspectiveProjection {
            fov: 30f32.to_radians(),
            near: 0.1,
            far: 200.0,
            ..default()
        }),
    ));

    // Three directional lights — orientation, color, and intensity are
    // driven every frame by `sync_lights` from the `LightControl`
    // resource. Cascade tuned to the small scene depth (~2 units of
    // z range from background to cream sheet).
    for i in 0..3 {
        commands.spawn((
            DirectionalLight {
                color: Color::WHITE,
                illuminance: 0.0,
                shadows_enabled: false,
                // Default biases assume a much larger scene; with our
                // sub-meter height differences they cause shadows to
                // vanish. Tighten them so layers actually shadow.
                shadow_depth_bias: 0.003,
                shadow_normal_bias: 0.05,
                ..default()
            },
            CascadeShadowConfigBuilder {
                num_cascades: 2,
                first_cascade_far_bound: 8.0,
                maximum_distance: 60.0,
                ..default()
            }
            .build(),
            DirLightIndex(i),
        ));
    }

    // Materials — paper-like: high roughness, no metalness, low
    // reflectance. We share material handles for layers that share a
    // color so we only allocate once per palette entry.
    let mk = |materials: &mut Assets<StandardMaterial>, color: Color| {
        materials.add(StandardMaterial {
            base_color: color,
            perceptual_roughness: 0.95,
            metallic: 0.0,
            reflectance: 0.05,
            ..default()
        })
    };

    let cream_mat  = mk(&mut materials, Color::srgb(0.88, 0.92, 0.94));
    let sky_mat    = mk(&mut materials, Color::srgb(0.18, 0.22, 0.42));
    let star_mat   = mk(&mut materials, Color::srgb(1.00, 1.00, 0.96));
    let moon_mat   = mk(&mut materials, Color::srgb(0.97, 0.97, 0.92));
    let cloud_mat  = mk(&mut materials, Color::srgb(0.78, 0.84, 0.92));
    let hill_back  = mk(&mut materials, Color::srgb(0.82, 0.88, 0.92));
    let hill_mid   = mk(&mut materials, Color::srgb(0.92, 0.95, 0.97));
    let hill_front = mk(&mut materials, Color::srgb(0.98, 0.99, 1.00));
    let tree_a     = mk(&mut materials, Color::srgb(0.20, 0.30, 0.50));
    let tree_b     = mk(&mut materials, Color::srgb(0.30, 0.40, 0.60));

    // Convert original shader coords (centered at (700,450), y down)
    // to Bevy world coords (centered at origin, y up).
    let to_world = |sx: f32, sy: f32| Vec2::new((sx - 700.0) * PX, -(sy - 450.0) * PX);
    let layer_at = |sx: f32, sy: f32, sz: f32| {
        let p = to_world(sx, sy);
        Transform::from_xyz(p.x, p.y, sz * PX)
    };

    let sky_world_center = to_world(700.0, 450.0);
    let sky_world_radius = 380.0 * PX;

    // Cream sheet — full-screen rectangle with a circular hole that
    // exposes the sky disc behind it. Positioned at the highest z so
    // its edge is what casts shadows onto everything below.
    let cream_mesh = meshes.add(rect_with_hole_mesh(
        SCENE_W + 1.0,
        SCENE_H + 1.0,
        sky_world_center,
        sky_world_radius,
    ));
    commands.spawn((
        Mesh3d(cream_mesh),
        MeshMaterial3d(cream_mat),
        Transform::from_xyz(0.0, 0.0, 12.0 * PX),
    ));

    // Sky disc.
    commands.spawn((
        Mesh3d(meshes.add(Circle::new(sky_world_radius))),
        MeshMaterial3d(sky_mat),
        Transform::from_xyz(sky_world_center.x, sky_world_center.y, -65.0 * PX),
    ));

    // Stars.
    let stars = [
        (550.0, 215.0, 2.5), (620.0, 175.0, 2.0), (680.0, 250.0, 3.5),
        (720.0, 180.0, 2.0), (780.0, 230.0, 2.5), (830.0, 290.0, 3.0),
        (950.0, 320.0, 2.0), (880.0, 380.0, 2.5), (750.0, 350.0, 2.0),
        (630.0, 300.0, 2.5), (580.0, 380.0, 2.0), (680.0, 420.0, 2.5),
        (810.0, 450.0, 2.0), (700.0, 280.0, 1.6), (640.0, 240.0, 1.6),
        (770.0, 380.0, 1.8), (940.0, 250.0, 1.8), (870.0, 200.0, 2.0),
        (990.0, 400.0, 2.5), (580.0, 460.0, 1.8), (900.0, 470.0, 2.2),
    ];
    for (sx, sy, r) in stars {
        commands.spawn((
            Mesh3d(meshes.add(Circle::new(r * PX))),
            MeshMaterial3d(star_mat.clone()),
            layer_at(sx, sy, -58.0),
        ));
    }

    // Moon.
    commands.spawn((
        Mesh3d(meshes.add(Circle::new(38.0 * PX))),
        MeshMaterial3d(moon_mat),
        layer_at(900.0, 210.0, -50.0),
    ));

    // Cloud puffs (six overlapping circles).
    for (sx, sy, r) in [
        (425.0, 220.0, 60.0),
        (440.0, 290.0, 75.0),
        (420.0, 360.0, 70.0),
        (450.0, 440.0, 80.0),
        (420.0, 520.0, 65.0),
        (380.0, 380.0, 55.0),
    ] {
        commands.spawn((
            Mesh3d(meshes.add(Circle::new(r * PX))),
            MeshMaterial3d(cloud_mat.clone()),
            layer_at(sx, sy, -44.0),
        ));
    }

    // Snow hills.
    for (sx, sy, r, sz, mat) in [
        (900.0, 1100.0, 600.0, -32.0, hill_back),
        (450.0, 1100.0, 580.0, -20.0, hill_mid),
        (700.0, 1450.0, 950.0,  -8.0, hill_front),
    ] {
        commands.spawn((
            Mesh3d(meshes.add(Circle::new(r * PX))),
            MeshMaterial3d(mat),
            layer_at(sx, sy, sz),
        ));
    }

    // Pine trees — flat triangles. Apex up in world space (y > center)
    // because we flipped y; in shader space that was apex up-screen.
    let trees = [
        (500.0, 600.0, 22.0, 55.0, &tree_b),
        (560.0, 640.0, 25.0, 58.0, &tree_a),
        (620.0, 605.0, 24.0, 56.0, &tree_b),
        (700.0, 660.0, 30.0, 70.0, &tree_a),
        (780.0, 645.0, 25.0, 60.0, &tree_b),
        (850.0, 670.0, 28.0, 65.0, &tree_a),
        (910.0, 650.0, 24.0, 58.0, &tree_b),
        (970.0, 660.0, 22.0, 52.0, &tree_a),
    ];
    for (sx, sy, half_w, h, mat) in trees {
        let hw = half_w * PX;
        let hh = h * PX;
        let mesh = meshes.add(Triangle2d::new(
            Vec2::new(0.0, hh),
            Vec2::new(-hw, -hh),
            Vec2::new(hw, -hh),
        ));
        commands.spawn((
            Mesh3d(mesh),
            MeshMaterial3d((*mat).clone()),
            layer_at(sx, sy, 0.0),
        ));
    }
}

/// Rectangle (centered on origin, in the xy plane) with a circular
/// hole punched through it. Triangulated as a strip of quads between
/// the inner ring (the circle) and an outer ring (each circle vertex
/// projected radially outward to the rectangle's bounds).
fn rect_with_hole_mesh(width: f32, height: f32, hole_center: Vec2, hole_radius: f32) -> Mesh {
    const N: usize = 96;
    let hw = width * 0.5;
    let hh = height * 0.5;

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(N * 2);
    let mut normals:   Vec<[f32; 3]> = Vec::with_capacity(N * 2);
    let mut uvs:       Vec<[f32; 2]> = Vec::with_capacity(N * 2);
    let mut indices:   Vec<u32>      = Vec::with_capacity(N * 6);

    for i in 0..N {
        let theta = i as f32 / N as f32 * std::f32::consts::TAU;
        let dir = Vec2::new(theta.cos(), theta.sin());
        let inner = hole_center + dir * hole_radius;

        // Distance from hole_center along `dir` to the rect boundary.
        let mut t = f32::INFINITY;
        if dir.x >  1e-6 { t = t.min(( hw - hole_center.x) / dir.x); }
        if dir.x < -1e-6 { t = t.min((-hw - hole_center.x) / dir.x); }
        if dir.y >  1e-6 { t = t.min(( hh - hole_center.y) / dir.y); }
        if dir.y < -1e-6 { t = t.min((-hh - hole_center.y) / dir.y); }
        let outer = hole_center + dir * t;

        positions.push([inner.x, inner.y, 0.0]);
        positions.push([outer.x, outer.y, 0.0]);
        normals.push([0.0, 0.0, 1.0]);
        normals.push([0.0, 0.0, 1.0]);
        uvs.push([(inner.x + hw) / width, (inner.y + hh) / height]);
        uvs.push([(outer.x + hw) / width, (outer.y + hh) / height]);
    }

    for i in 0..N as u32 {
        let next = (i + 1) % N as u32;
        let i_in  = i * 2;
        let i_out = i * 2 + 1;
        let n_in  = next * 2;
        let n_out = next * 2 + 1;
        indices.extend_from_slice(&[i_in, i_out, n_out, i_in, n_out, n_in]);
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL,   normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0,     uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh
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
    mut camera: ResMut<CameraControl>,
) {
    if !debug.0 {
        wheel.clear();
        return;
    }
    let dt = time.delta_secs();

    if keys.just_pressed(KeyCode::KeyR) {
        light.lights = default_lights();
        light.active = 0;
        light.ambient = 0.28;
        camera.pan = Vec2::ZERO;
        camera.zoom = 1.0;
        camera.tilt = 0.0;
    }

    if keys.just_pressed(KeyCode::Digit1) { light.active = 0; }
    if keys.just_pressed(KeyCode::Digit2) { light.active = 1; }
    if keys.just_pressed(KeyCode::Digit3) { light.active = 2; }

    if keys.just_pressed(KeyCode::Digit0) {
        let idx = light.active;
        let cur = light.lights[idx].intensity;
        light.lights[idx].intensity = if cur > 0.01 { 0.0 } else { 1.0 };
    }
    if keys.just_pressed(KeyCode::KeyC) {
        let idx = light.active;
        light.lights[idx].color_idx =
            (light.lights[idx].color_idx + 1) % COLOR_PRESETS.len();
    }

    wheel.clear();

    // Pan.
    let pan_speed = 480.0 / camera.zoom;
    let mut delta = Vec2::ZERO;
    if keys.pressed(KeyCode::KeyW) || keys.pressed(KeyCode::ArrowUp)    { delta.y -= 1.0; }
    if keys.pressed(KeyCode::KeyS) || keys.pressed(KeyCode::ArrowDown)  { delta.y += 1.0; }
    if keys.pressed(KeyCode::KeyA) || keys.pressed(KeyCode::ArrowLeft)  { delta.x -= 1.0; }
    if keys.pressed(KeyCode::KeyD) || keys.pressed(KeyCode::ArrowRight) { delta.x += 1.0; }
    if delta != Vec2::ZERO {
        camera.pan += delta.normalize() * pan_speed * dt;
    }

    // Zoom.
    let zoom_rate = 1.6_f32.powf(dt);
    if keys.pressed(KeyCode::KeyE) { camera.zoom = (camera.zoom * zoom_rate).min(8.0); }
    if keys.pressed(KeyCode::KeyQ) { camera.zoom = (camera.zoom / zoom_rate).max(0.2); }

    // Pitch.
    let tilt_speed = 1.2;
    let tilt_max = 1.4;
    if keys.pressed(KeyCode::BracketLeft)  { camera.tilt = (camera.tilt - tilt_speed * dt).max(0.0); }
    if keys.pressed(KeyCode::BracketRight) { camera.tilt = (camera.tilt + tilt_speed * dt).min(tilt_max); }
    if keys.just_pressed(KeyCode::KeyT) {
        camera.tilt = if camera.tilt < 0.05 { 0.6 } else { 0.0 };
    }
}

/// Apply the slider state to the actual `DirectionalLight` entities.
fn sync_lights(
    light_ctrl: Res<LightControl>,
    mut q: Query<(&mut DirectionalLight, &mut Transform, &DirLightIndex)>,
) {
    for (mut dl, mut tf, idx) in q.iter_mut() {
        let l = light_ctrl.lights[idx.0];
        // Shader-space y was screen-down; in world space y is up.
        let dir_to_light = Vec3::new(l.dir.x, -l.dir.y, l.dir.z)
            .try_normalize()
            .unwrap_or(Vec3::Z);
        // Bevy's directional light shines along the entity's forward
        // (-Z). We want it to shine FROM dir_to_light TOWARD the
        // surface, so forward = -dir_to_light.
        let forward = -dir_to_light;
        let up = if forward.y.abs() < 0.99 { Vec3::Y } else { Vec3::X };
        *tf = Transform::IDENTITY.looking_to(forward, up);

        let on = l.intensity > 0.01;
        dl.color = l.color();
        dl.illuminance = l.intensity * 12_000.0;
        dl.shadows_enabled = on;
    }
}

fn sync_ambient(light: Res<LightControl>, mut ambient: ResMut<GlobalAmbientLight>) {
    ambient.brightness = light.ambient * 800.0;
}

fn sync_camera(
    camera_ctrl: Res<CameraControl>,
    mut q: Query<&mut Transform, With<Camera3d>>,
) {
    let Ok(mut tf) = q.single_mut() else { return };
    let target = Vec3::new(camera_ctrl.pan.x * PX, -camera_ctrl.pan.y * PX, 0.0);
    let base_distance = 20.0;
    let distance = base_distance / camera_ctrl.zoom;
    let offset = Quat::from_rotation_x(-camera_ctrl.tilt) * Vec3::new(0.0, 0.0, distance);
    *tf = Transform::from_translation(target + offset).looking_at(target, Vec3::Y);
}

fn debug_panel(
    mut contexts: EguiContexts,
    debug: Res<DebugMode>,
    mut light: ResMut<LightControl>,
    mut camera: ResMut<CameraControl>,
) -> Result {
    if !debug.0 {
        return Ok(());
    }
    let ctx = contexts.ctx_mut()?;

    egui::Window::new("Carved-UI Debug (3D)")
        .default_pos([12.0, 12.0])
        .default_width(280.0)
        .show(ctx, |ui| {
            ui.collapsing("Lights", |ui| {
                ui.add(egui::Slider::new(&mut light.ambient, 0.0..=1.0).text("ambient"));
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
                                egui::Slider::new(&mut l.intensity, 0.0..=2.0).text("intensity"),
                            );
                            ui.add(egui::Slider::new(&mut l.dir.x, -2.0..=2.0).text("dir x"));
                            ui.add(egui::Slider::new(&mut l.dir.y, -2.0..=2.0).text("dir y"));
                            ui.add(
                                egui::Slider::new(&mut l.dir.z, 0.0..=2.0)
                                    .text("dir z (elevation)"),
                            );
                            let cur_name = l.color_name();
                            egui::ComboBox::from_label("color")
                                .selected_text(cur_name)
                                .show_ui(ui, |ui| {
                                    for (idx, (name, _)) in COLOR_PRESETS.iter().enumerate() {
                                        ui.selectable_value(&mut l.color_idx, idx, *name);
                                    }
                                });
                        });
                }
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
                    camera.pan = Vec2::ZERO;
                    camera.zoom = 1.0;
                    camera.tilt = 0.0;
                }
                ui.label("Tab to hide");
            });
        });
    Ok(())
}
