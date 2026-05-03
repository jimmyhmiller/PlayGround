//! Carved letters demo. Loads a TTF, lays out a string of glyphs in
//! world space, and generates a 3D mesh: a paper sheet with each
//! glyph carved as a hole, beveled walls sloping inward as they drop
//! to a colored floor layer below.
//!
//! Run with:
//!     cargo run --bin letters_demo
//!
//! Override the font with `LETTERS_FONT=/path/to/font.ttf`.
//!
//! Controls:
//!   • Tab                       → toggle debug panel
//!   • W/A/S/D / arrow keys      → pan
//!   • Q / E                     → zoom out / in
//!   • [ / ]                     → camera tilt
//!   • R                         → reset

use bevy::asset::RenderAssetUsages;
use bevy::input::ButtonInput;
use bevy::input::mouse::MouseWheel;
use bevy::light::{CascadeShadowConfigBuilder, GlobalAmbientLight};
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use lyon::math::point;
use lyon::path::iterator::PathIterator;
use lyon::path::{Path, PathEvent};
use lyon::tessellation::{
    BuffersBuilder, FillOptions, FillRule, FillTessellator, FillVertex, VertexBuffers,
};
use ttf_parser::{Face, GlyphId, OutlineBuilder};

const FONT_FALLBACKS: &[&str] = &[
    "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
];

const DEFAULT_TEXT: &str = "Hello";

// ─── Resources ────────────────────────────────────────────────────

#[derive(Resource)]
struct DebugMode(bool);

#[derive(Resource)]
struct CameraControl {
    pan: Vec2,
    zoom: f32,
    /// Pitch in radians. 0 = top-down, π/2 = horizontal.
    tilt: f32,
}

#[derive(Resource)]
struct CarveParams {
    /// World-space height of the paper sheet. Top sits at z=thickness,
    /// bottom (and the floor below) sits at z=0.
    thickness: f32,
    /// World-space inward miter inset of the bottom of each bevel
    /// wall. Larger = shallower carve angle.
    bevel_inset: f32,
    /// Bézier flattening tolerance in world units.
    flatten_tol: f32,
    /// Cap on miter length as a multiple of `bevel_inset`. Avoids
    /// long spikes at acute concave corners.
    miter_limit: f32,
    paper_color: Color,
    letter_color: Color,
    text: String,
    /// World-space height of one em (so em_world≈glyph cap height).
    em_world: f32,
    /// Set true to ask the regen system to rebuild meshes next frame.
    dirty: bool,
}

#[derive(Resource)]
struct LightControl {
    /// Direction TOWARD the light (world space, y-up, z-up).
    dir: Vec3,
    intensity: f32,
    ambient: f32,
}

#[derive(Resource)]
struct LoadedFont(Vec<u8>);

#[derive(Resource, Default)]
struct GlyphData {
    /// Bounding rect of the paper, padded outward from the glyph bbox.
    paper_rect: Rect,
    /// One entry per laid-out glyph; each glyph is a list of closed
    /// polylines. The closing edge is implicit — first vertex is NOT
    /// repeated at the end.
    contours: Vec<Vec<Vec<Vec2>>>,
}

#[derive(Component, Clone, Copy)]
enum CarveRole {
    Paper,
    Floor,
}

// ─── App ──────────────────────────────────────────────────────────

fn main() {
    let font_data = load_font();

    App::new()
        .add_plugins(
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "carved letters".into(),
                    resolution: (1400, 900).into(),
                    ..default()
                }),
                ..default()
            }),
        )
        .add_plugins(EguiPlugin::default())
        .insert_resource(DebugMode(false))
        .insert_resource(CameraControl {
            pan: Vec2::ZERO,
            zoom: 1.0,
            tilt: 0.55,
        })
        .insert_resource(CarveParams {
            thickness: 0.16,
            bevel_inset: 0.07,
            flatten_tol: 0.008,
            miter_limit: 2.5,
            paper_color: Color::srgb(0.94, 0.91, 0.84),
            letter_color: Color::srgb(0.55, 0.18, 0.20),
            text: DEFAULT_TEXT.into(),
            em_world: 2.0,
            dirty: true,
        })
        .insert_resource(LightControl {
            dir: Vec3::new(-0.45, 0.6, 0.7).normalize(),
            intensity: 1.2,
            ambient: 0.22,
        })
        .insert_resource(LoadedFont(font_data))
        .insert_resource(GlyphData::default())
        .insert_resource(GlobalAmbientLight {
            color: Color::WHITE,
            brightness: 200.0,
            affects_lightmapped_meshes: true,
        })
        .insert_resource(ClearColor(Color::srgb(0.06, 0.06, 0.08)))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                toggle_debug,
                debug_inputs,
                regen_glyphs_and_meshes,
                sync_lights,
                sync_ambient,
                sync_camera,
            ),
        )
        .add_systems(EguiPrimaryContextPass, debug_panel)
        .run();
}

fn load_font() -> Vec<u8> {
    if let Ok(p) = std::env::var("LETTERS_FONT") {
        return std::fs::read(&p)
            .unwrap_or_else(|e| panic!("failed to read LETTERS_FONT={p}: {e}"));
    }
    for p in FONT_FALLBACKS {
        if let Ok(d) = std::fs::read(p) {
            return d;
        }
    }
    panic!(
        "no font found. Set LETTERS_FONT to a TTF path, or install one of: {:?}",
        FONT_FALLBACKS
    );
}

// ─── Setup ────────────────────────────────────────────────────────

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    params: Res<CarveParams>,
) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, -3.0, 6.0).looking_at(Vec3::ZERO, Vec3::Z),
        Projection::from(PerspectiveProjection {
            fov: 35f32.to_radians(),
            near: 0.05,
            far: 200.0,
            ..default()
        }),
    ));

    commands.spawn((
        DirectionalLight {
            color: Color::WHITE,
            illuminance: 14_000.0,
            shadows_enabled: true,
            shadow_depth_bias: 0.005,
            shadow_normal_bias: 0.4,
            ..default()
        },
        CascadeShadowConfigBuilder {
            num_cascades: 2,
            first_cascade_far_bound: 6.0,
            maximum_distance: 30.0,
            ..default()
        }
        .build(),
        Transform::IDENTITY,
    ));

    let paper_mat = materials.add(StandardMaterial {
        base_color: params.paper_color,
        perceptual_roughness: 0.95,
        metallic: 0.0,
        reflectance: 0.04,
        ..default()
    });
    let letter_mat = materials.add(StandardMaterial {
        base_color: params.letter_color,
        perceptual_roughness: 0.85,
        metallic: 0.0,
        reflectance: 0.04,
        ..default()
    });

    let paper_mesh_h = meshes.add(empty_mesh());
    let floor_mesh_h = meshes.add(empty_mesh());

    commands.spawn((
        Mesh3d(paper_mesh_h),
        MeshMaterial3d(paper_mat),
        Transform::IDENTITY,
        CarveRole::Paper,
    ));
    commands.spawn((
        Mesh3d(floor_mesh_h),
        MeshMaterial3d(letter_mat),
        Transform::IDENTITY,
        CarveRole::Floor,
    ));
}

fn empty_mesh() -> Mesh {
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<[f32; 3]>::new());
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, Vec::<[f32; 3]>::new());
    mesh.insert_indices(Indices::U32(Vec::new()));
    mesh
}

// ─── Glyph parsing ────────────────────────────────────────────────

/// Captures `OutlineBuilder` callbacks into our own op enum so we can
/// build a lyon `Path` afterwards without naming lyon's builder type.
enum Op {
    MoveTo(Vec2),
    LineTo(Vec2),
    QuadTo(Vec2, Vec2),
    CurveTo(Vec2, Vec2, Vec2),
    Close,
}

struct OpCollector {
    ops: Vec<Op>,
    scale: f32,
    offset: Vec2,
}

impl OpCollector {
    fn t(&self, x: f32, y: f32) -> Vec2 {
        Vec2::new(x * self.scale + self.offset.x, y * self.scale + self.offset.y)
    }
}

impl OutlineBuilder for OpCollector {
    fn move_to(&mut self, x: f32, y: f32) {
        self.ops.push(Op::MoveTo(self.t(x, y)));
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.ops.push(Op::LineTo(self.t(x, y)));
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.ops.push(Op::QuadTo(self.t(x1, y1), self.t(x, y)));
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        self.ops
            .push(Op::CurveTo(self.t(x1, y1), self.t(x2, y2), self.t(x, y)));
    }
    fn close(&mut self) {
        self.ops.push(Op::Close);
    }
}

fn ops_to_path(ops: &[Op]) -> Path {
    let mut pb = Path::builder();
    let mut started = false;
    for op in ops {
        match op {
            Op::MoveTo(p) => {
                if started {
                    pb.end(false);
                }
                pb.begin(point(p.x, p.y));
                started = true;
            }
            Op::LineTo(p) => {
                pb.line_to(point(p.x, p.y));
            }
            Op::QuadTo(c, p) => {
                pb.quadratic_bezier_to(point(c.x, c.y), point(p.x, p.y));
            }
            Op::CurveTo(c1, c2, p) => {
                pb.cubic_bezier_to(
                    point(c1.x, c1.y),
                    point(c2.x, c2.y),
                    point(p.x, p.y),
                );
            }
            Op::Close => {
                if started {
                    pb.end(true);
                    started = false;
                }
            }
        }
    }
    if started {
        pb.end(false);
    }
    pb.build()
}

fn outline_glyph_to_path(face: &Face, glyph_id: GlyphId, scale: f32, offset: Vec2) -> Path {
    let mut col = OpCollector {
        ops: Vec::new(),
        scale,
        offset,
    };
    face.outline_glyph(glyph_id, &mut col);
    ops_to_path(&col.ops)
}

fn flatten_path_to_contours(path: &Path, tol: f32) -> Vec<Vec<Vec2>> {
    let mut out = Vec::new();
    let mut current: Vec<Vec2> = Vec::new();
    for evt in path.iter().flattened(tol) {
        match evt {
            PathEvent::Begin { at } => {
                current.clear();
                current.push(Vec2::new(at.x, at.y));
            }
            PathEvent::Line { to, .. } => {
                current.push(Vec2::new(to.x, to.y));
            }
            PathEvent::Quadratic { .. } | PathEvent::Cubic { .. } => {
                // Should not appear after .flattened().
            }
            PathEvent::End { close, .. } => {
                // Drop trailing duplicate of the first vertex if present.
                if let (Some(first), Some(last)) =
                    (current.first().copied(), current.last().copied())
                {
                    if (first - last).length_squared() < 1e-12 && current.len() > 1 {
                        current.pop();
                    }
                }
                if close && current.len() >= 3 {
                    out.push(std::mem::take(&mut current));
                } else {
                    current.clear();
                }
            }
        }
    }
    out
}

fn build_glyph_data(font_data: &[u8], text: &str, em_world: f32, flatten_tol: f32) -> GlyphData {
    let face = Face::parse(font_data, 0).expect("invalid font data");
    let upem = face.units_per_em() as f32;
    let scale = em_world / upem;

    let mut all_contours: Vec<Vec<Vec<Vec2>>> = Vec::new();
    let mut x_cursor = 0.0_f32;
    let mut bounds_min = Vec2::splat(f32::INFINITY);
    let mut bounds_max = Vec2::splat(f32::NEG_INFINITY);

    for ch in text.chars() {
        let Some(glyph_id) = face.glyph_index(ch) else {
            continue;
        };
        let advance = face.glyph_hor_advance(glyph_id).unwrap_or(0) as f32;

        let path = outline_glyph_to_path(&face, glyph_id, scale, Vec2::new(x_cursor, 0.0));
        let contours = flatten_path_to_contours(&path, flatten_tol);
        for contour in &contours {
            for p in contour {
                bounds_min = bounds_min.min(*p);
                bounds_max = bounds_max.max(*p);
            }
        }
        all_contours.push(contours);
        x_cursor += advance * scale;
    }

    if !bounds_min.is_finite() {
        bounds_min = Vec2::ZERO;
        bounds_max = Vec2::new(em_world, em_world);
    }

    // Center text around origin.
    let center = (bounds_min + bounds_max) * 0.5;
    for glyph in all_contours.iter_mut() {
        for contour in glyph.iter_mut() {
            for p in contour.iter_mut() {
                *p -= center;
            }
        }
    }
    bounds_min -= center;
    bounds_max -= center;

    let pad = em_world * 0.4;
    let paper_rect = Rect {
        min: bounds_min - Vec2::splat(pad),
        max: bounds_max + Vec2::splat(pad),
    };

    GlyphData {
        paper_rect,
        contours: all_contours,
    }
}

// ─── Mesh generation ──────────────────────────────────────────────

/// Inward-miter offset of a closed polyline. "Inward" = LEFT of the
/// walking direction (which, for TTF in y-up world space, is into the
/// glyph fill — i.e., into the carved hole — for both outer CCW and
/// inner CW contours).
fn miter_offset_left(contour: &[Vec2], inset: f32, miter_limit: f32) -> Vec<Vec2> {
    let n = contour.len();
    if n < 2 {
        return contour.to_vec();
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let prev = contour[(i + n - 1) % n];
        let curr = contour[i];
        let next = contour[(i + 1) % n];
        let e_in = (curr - prev).try_normalize().unwrap_or(Vec2::X);
        let e_out = (next - curr).try_normalize().unwrap_or(Vec2::X);
        // Left perpendicular of (dx, dy) = (-dy, dx).
        let n_in = Vec2::new(-e_in.y, e_in.x);
        let n_out = Vec2::new(-e_out.y, e_out.x);
        let bisector = (n_in + n_out).try_normalize().unwrap_or(n_in);
        // Miter length = inset / cos(half-turn-angle). Clamp.
        let cos_half = bisector.dot(n_in).max(1.0 / miter_limit.max(1.0));
        let miter_len = inset / cos_half;
        out.push(curr + bisector * miter_len);
    }
    out
}

fn build_paper_mesh(data: &GlyphData, params: &CarveParams) -> Mesh {
    let r = data.paper_rect;
    let pt = params.thickness;

    // Build a path containing the outer rect plus every glyph contour.
    // Lyon's even-odd fill rule lets all the glyph contours act as
    // holes regardless of winding.
    let mut pb = Path::builder();
    pb.begin(point(r.min.x, r.min.y));
    pb.line_to(point(r.max.x, r.min.y));
    pb.line_to(point(r.max.x, r.max.y));
    pb.line_to(point(r.min.x, r.max.y));
    pb.end(true);
    for glyph in &data.contours {
        for contour in glyph {
            if contour.len() < 3 {
                continue;
            }
            pb.begin(point(contour[0].x, contour[0].y));
            for p in &contour[1..] {
                pb.line_to(point(p.x, p.y));
            }
            pb.end(true);
        }
    }
    let path = pb.build();

    let mut buffers: VertexBuffers<[f32; 2], u32> = VertexBuffers::new();
    let mut tess = FillTessellator::new();
    let opts = FillOptions::default()
        .with_fill_rule(FillRule::EvenOdd)
        .with_tolerance(params.flatten_tol);
    tess.tessellate_path(
        &path,
        &opts,
        &mut BuffersBuilder::new(&mut buffers, |v: FillVertex| {
            let p = v.position();
            [p.x, p.y]
        }),
    )
    .expect("tessellation failed");

    let mut positions: Vec<[f32; 3]> = buffers
        .vertices
        .iter()
        .map(|v| [v[0], v[1], pt])
        .collect();
    let mut normals: Vec<[f32; 3]> = vec![[0.0, 0.0, 1.0]; positions.len()];
    let mut indices: Vec<u32> = buffers.indices.clone();

    // Side walls.
    for glyph in &data.contours {
        for contour in glyph {
            if contour.len() < 3 {
                continue;
            }
            let offsets = miter_offset_left(contour, params.bevel_inset, params.miter_limit);
            let n = contour.len();
            for i in 0..n {
                let j = (i + 1) % n;
                let p0 = Vec3::new(contour[i].x, contour[i].y, pt);
                let p1 = Vec3::new(contour[j].x, contour[j].y, pt);
                let o0 = Vec3::new(offsets[i].x, offsets[i].y, 0.0);
                let o1 = Vec3::new(offsets[j].x, offsets[j].y, 0.0);

                // Outward-facing normal (= into the carved hole).
                let normal = (p1 - p0)
                    .cross(o0 - p0)
                    .try_normalize()
                    .unwrap_or(Vec3::Z);

                let base = positions.len() as u32;
                for v in [p0, p1, o1, o0] {
                    positions.push([v.x, v.y, v.z]);
                    normals.push([normal.x, normal.y, normal.z]);
                }
                indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            }
        }
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

fn build_floor_mesh(data: &GlyphData) -> Mesh {
    let r = data.paper_rect;
    let positions: Vec<[f32; 3]> = vec![
        [r.min.x, r.min.y, 0.0],
        [r.max.x, r.min.y, 0.0],
        [r.max.x, r.max.y, 0.0],
        [r.min.x, r.max.y, 0.0],
    ];
    let normals = vec![[0.0, 0.0, 1.0]; 4];
    let indices: Vec<u32> = vec![0, 1, 2, 0, 2, 3];
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

fn regen_glyphs_and_meshes(
    mut params: ResMut<CarveParams>,
    font: Res<LoadedFont>,
    mut data: ResMut<GlyphData>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    q: Query<(&Mesh3d, &MeshMaterial3d<StandardMaterial>, &CarveRole)>,
) {
    if !params.dirty {
        return;
    }
    *data = build_glyph_data(&font.0, &params.text, params.em_world, params.flatten_tol);

    for (mesh3d, mat3d, role) in q.iter() {
        let new_mesh = match role {
            CarveRole::Paper => build_paper_mesh(&data, &params),
            CarveRole::Floor => build_floor_mesh(&data),
        };
        if let Some(m) = meshes.get_mut(&mesh3d.0) {
            *m = new_mesh;
        }
        if let Some(mat) = materials.get_mut(&mat3d.0) {
            mat.base_color = match role {
                CarveRole::Paper => params.paper_color,
                CarveRole::Floor => params.letter_color,
            };
        }
    }
    params.dirty = false;
}

// ─── Input + sync ─────────────────────────────────────────────────

fn toggle_debug(keys: Res<ButtonInput<KeyCode>>, mut debug: ResMut<DebugMode>) {
    if keys.just_pressed(KeyCode::Tab) {
        debug.0 = !debug.0;
    }
}

fn debug_inputs(
    keys: Res<ButtonInput<KeyCode>>,
    mut wheel: MessageReader<MouseWheel>,
    time: Res<Time>,
    mut camera: ResMut<CameraControl>,
    mut params: ResMut<CarveParams>,
) {
    wheel.clear();
    let dt = time.delta_secs();

    if keys.just_pressed(KeyCode::KeyR) {
        camera.pan = Vec2::ZERO;
        camera.zoom = 1.0;
        camera.tilt = 0.55;
        params.thickness = 0.16;
        params.bevel_inset = 0.07;
        params.dirty = true;
    }

    let pan_speed = 4.0 / camera.zoom;
    let mut delta = Vec2::ZERO;
    if keys.pressed(KeyCode::KeyW) || keys.pressed(KeyCode::ArrowUp) {
        delta.y += 1.0;
    }
    if keys.pressed(KeyCode::KeyS) || keys.pressed(KeyCode::ArrowDown) {
        delta.y -= 1.0;
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

    let zoom_rate = 1.6_f32.powf(dt);
    if keys.pressed(KeyCode::KeyE) {
        camera.zoom = (camera.zoom * zoom_rate).min(8.0);
    }
    if keys.pressed(KeyCode::KeyQ) {
        camera.zoom = (camera.zoom / zoom_rate).max(0.2);
    }

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

fn sync_lights(
    light_ctrl: Res<LightControl>,
    mut q: Query<(&mut DirectionalLight, &mut Transform), With<DirectionalLight>>,
) {
    for (mut dl, mut tf) in q.iter_mut() {
        let dir_to_light = light_ctrl.dir.try_normalize().unwrap_or(Vec3::Z);
        let forward = -dir_to_light;
        let up = if forward.z.abs() < 0.99 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        *tf = Transform::IDENTITY.looking_to(forward, up);
        dl.illuminance = light_ctrl.intensity * 12_000.0;
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
    let target = Vec3::new(camera_ctrl.pan.x, camera_ctrl.pan.y, 0.0);
    let base_distance = 7.0;
    let distance = base_distance / camera_ctrl.zoom;
    // Tilt 0 = straight down (camera high above on +Z axis).
    // Tilt π/2 = horizontal (camera to the south on -Y axis).
    let tilt = camera_ctrl.tilt;
    let cam_offset = Vec3::new(0.0, -distance * tilt.sin(), distance * tilt.cos());
    *tf = Transform::from_translation(target + cam_offset).looking_at(target, Vec3::Z);
}

// ─── Debug panel ──────────────────────────────────────────────────

fn debug_panel(
    mut contexts: EguiContexts,
    debug: Res<DebugMode>,
    mut params: ResMut<CarveParams>,
    mut light: ResMut<LightControl>,
    mut camera: ResMut<CameraControl>,
) -> Result {
    if !debug.0 {
        return Ok(());
    }
    let ctx = contexts.ctx_mut()?;

    egui::Window::new("Carved letters")
        .default_pos([12.0, 12.0])
        .default_width(320.0)
        .show(ctx, |ui| {
            ui.collapsing("Text", |ui| {
                let mut text = params.text.clone();
                if ui.text_edit_singleline(&mut text).changed() {
                    params.text = text;
                    params.dirty = true;
                }
                let mut em = params.em_world;
                if ui
                    .add(egui::Slider::new(&mut em, 0.5..=4.0).text("em (world)"))
                    .changed()
                {
                    params.em_world = em;
                    params.dirty = true;
                }
            });

            ui.collapsing("Carve", |ui| {
                let mut t = params.thickness;
                if ui
                    .add(egui::Slider::new(&mut t, 0.02..=0.6).text("thickness"))
                    .changed()
                {
                    params.thickness = t;
                    params.dirty = true;
                }
                let mut b = params.bevel_inset;
                if ui
                    .add(egui::Slider::new(&mut b, 0.0..=0.3).text("bevel inset"))
                    .changed()
                {
                    params.bevel_inset = b;
                    params.dirty = true;
                }
                let mut tol = params.flatten_tol;
                if ui
                    .add(
                        egui::Slider::new(&mut tol, 0.001..=0.05)
                            .logarithmic(true)
                            .text("flatten tolerance"),
                    )
                    .changed()
                {
                    params.flatten_tol = tol;
                    params.dirty = true;
                }
                let mut ml = params.miter_limit;
                if ui
                    .add(egui::Slider::new(&mut ml, 1.0..=8.0).text("miter limit"))
                    .changed()
                {
                    params.miter_limit = ml;
                    params.dirty = true;
                }
            });

            ui.collapsing("Color", |ui| {
                let mut paper = color_to_rgb(params.paper_color);
                if ui.color_edit_button_rgb(&mut paper).changed() {
                    params.paper_color = Color::linear_rgb(paper[0], paper[1], paper[2]);
                    params.dirty = true;
                }
                ui.label("paper");
                let mut letter = color_to_rgb(params.letter_color);
                if ui.color_edit_button_rgb(&mut letter).changed() {
                    params.letter_color = Color::linear_rgb(letter[0], letter[1], letter[2]);
                    params.dirty = true;
                }
                ui.label("letter");
            });

            ui.collapsing("Light", |ui| {
                ui.add(egui::Slider::new(&mut light.intensity, 0.0..=2.5).text("intensity"));
                ui.add(egui::Slider::new(&mut light.ambient, 0.0..=1.0).text("ambient"));
                ui.add(egui::Slider::new(&mut light.dir.x, -1.5..=1.5).text("dir x"));
                ui.add(egui::Slider::new(&mut light.dir.y, -1.5..=1.5).text("dir y"));
                ui.add(egui::Slider::new(&mut light.dir.z, 0.0..=1.5).text("dir z"));
            });

            ui.collapsing("Camera", |ui| {
                ui.add(egui::Slider::new(&mut camera.pan.x, -10.0..=10.0).text("pan x"));
                ui.add(egui::Slider::new(&mut camera.pan.y, -10.0..=10.0).text("pan y"));
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
            ui.label("Tab to hide.");
        });
    Ok(())
}

fn color_to_rgb(c: Color) -> [f32; 3] {
    let lc = c.to_linear();
    [lc.red, lc.green, lc.blue]
}
