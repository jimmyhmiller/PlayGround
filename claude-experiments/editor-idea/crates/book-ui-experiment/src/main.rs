//! Paper-stack editor. A `STACK_W × STACK_H` grid of columns, each
//! column has an integer "top layer index" into a shared palette.
//! Carving tools mutate that integer field; the surface mesh is
//! regenerated from it. Lights and shadows are stock Bevy 3D.
//!
//! Press Tab to toggle the debug panel. Controls:
//!   • Left mouse drag        → apply current tool
//!   • 1 / 2                  → select Dig / Extrude
//!   • W/A/S/D or arrows      → pan camera
//!   • Q / E                  → zoom out / in
//!   • [ / ]                  → camera pitch
//!   • T                      → snap pitch
//!   • R                      → reset camera and stack

use bevy::asset::RenderAssetUsages;
use bevy::input::ButtonInput;
use bevy::input::mouse::MouseWheel;
use bevy::light::{CascadeShadowConfigBuilder, GlobalAmbientLight};
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};

const STACK_W: u16 = 128;
const STACK_H: u16 = 128;
const MAX_LAYERS: u16 = 32;
const CELL_SIZE: f32 = 0.08;
// Default layer geometry. Paper is the actual sheet body; gap is the
// empty space above each sheet. Side faces only render sheet bodies,
// so the gaps show as visible slits between strata. Both are runtime-
// tweakable via the egui panel — these are starting values.
const DEFAULT_PAPER_THICKNESS: f32 = 0.012;
const DEFAULT_GAP_THICKNESS: f32 = 0.004;

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

// ─── Resources ────────────────────────────────────────────────────

#[derive(Resource)]
struct DebugMode(bool);

#[derive(Clone, Copy)]
struct LightDef {
    /// Direction TOWARD the light, in shader-space (y down).
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

#[derive(Resource)]
struct PaperStack {
    width: u16,
    height: u16,
    max_layers: u16,
    cells: Vec<u16>,
    palette: Vec<Color>,
    /// Vertical thickness of the paper sheet itself (sheet body).
    paper_thickness: f32,
    /// Empty space above each sheet, before the next one starts.
    gap_thickness: f32,
}

impl PaperStack {
    fn new(width: u16, height: u16, max_layers: u16, palette: Vec<Color>) -> Self {
        let cells = vec![max_layers; (width as usize) * (height as usize)];
        Self {
            width, height, max_layers, cells, palette,
            paper_thickness: DEFAULT_PAPER_THICKNESS,
            gap_thickness: DEFAULT_GAP_THICKNESS,
        }
    }
    /// Vertical distance from the bottom of one sheet to the bottom
    /// of the next (paper + gap above).
    fn layer_pitch(&self) -> f32 {
        self.paper_thickness + self.gap_thickness
    }
    /// World-z of the top surface of a column with `h` sheets.
    fn cell_top_z(&self, h: u16) -> f32 {
        if h == 0 {
            0.0
        } else {
            (h as f32 - 1.0) * self.layer_pitch() + self.paper_thickness
        }
    }
    /// World-z of the bottom of sheet `layer` (0-indexed).
    fn sheet_bottom_z(&self, layer: u16) -> f32 {
        layer as f32 * self.layer_pitch()
    }
    fn idx(&self, x: u16, y: u16) -> usize {
        (y as usize) * (self.width as usize) + (x as usize)
    }
    fn get(&self, x: u16, y: u16) -> u16 {
        self.cells[self.idx(x, y)]
    }
    fn set(&mut self, x: u16, y: u16, v: u16) {
        let i = self.idx(x, y);
        self.cells[i] = v.min(self.max_layers);
    }
    fn cell_world_xy(&self, x: u16, y: u16) -> Vec2 {
        Vec2::new(
            (x as f32 - self.width  as f32 * 0.5 + 0.5) * CELL_SIZE,
            (y as f32 - self.height as f32 * 0.5 + 0.5) * CELL_SIZE,
        )
    }
    fn world_to_cell(&self, wx: f32, wy: f32) -> Option<(u16, u16)> {
        let cx_f = wx / CELL_SIZE + self.width  as f32 * 0.5;
        let cy_f = wy / CELL_SIZE + self.height as f32 * 0.5;
        let cx = cx_f.floor() as i32;
        let cy = cy_f.floor() as i32;
        if cx < 0 || cy < 0 || cx >= self.width as i32 || cy >= self.height as i32 {
            None
        } else {
            Some((cx as u16, cy as u16))
        }
    }
    fn reset_full(&mut self) {
        self.cells.fill(self.max_layers);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tool {
    Dig,
    Extrude,
}

#[derive(Resource)]
struct Brush {
    tool: Tool,
    /// Brush radius in cell units.
    radius: f32,
    /// Layer count applied per stroke step.
    strength: u16,
}

#[derive(Resource, Default)]
struct HoverCell(Option<(u16, u16)>);

/// The cell where the brush was last applied during the current
/// stroke. Used to apply once per cell as the cursor drags.
#[derive(Resource, Default)]
struct StrokeLastCell(Option<(u16, u16)>);

#[derive(Resource, Default)]
struct StackDirty(bool);

#[derive(Component)]
struct StackMesh;

// ─── Defaults ─────────────────────────────────────────────────────

fn default_lights() -> [LightDef; 3] {
    [
        LightDef { dir: Vec3::new(-0.55, -0.55, 0.55), intensity: 1.0, color_idx: 0 },
        LightDef { dir: Vec3::new( 0.55, -0.40, 0.55), intensity: 0.0, color_idx: 2 },
        LightDef { dir: Vec3::new( 0.0,   0.55, 0.55), intensity: 0.0, color_idx: 1 },
    ]
}

/// Hue-rotated rainbow palette. 32 colors covering most of the wheel,
/// with mid saturation / mid lightness so adjacent layers are clearly
/// distinct. The user can recolor any individual layer at runtime.
fn default_palette(n: u16) -> Vec<Color> {
    (0..n)
        .map(|i| {
            let t = i as f32 / (n.max(2) - 1) as f32;
            let h = (t * 320.0) % 360.0;
            let s = 0.55;
            let l = 0.42 + 0.18 * (1.0 - (t - 0.5).abs() * 2.0);
            Color::hsl(h, s, l)
        })
        .collect()
}

// ─── App ──────────────────────────────────────────────────────────

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "paper-stack editor".into(),
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
            ambient: 0.32,
        })
        .insert_resource(CameraControl {
            pan: Vec2::ZERO,
            zoom: 1.0,
            tilt: 0.55,
        })
        .insert_resource(GlobalAmbientLight {
            color: Color::WHITE,
            brightness: 260.0,
            affects_lightmapped_meshes: true,
        })
        .insert_resource(PaperStack::new(
            STACK_W,
            STACK_H,
            MAX_LAYERS,
            default_palette(MAX_LAYERS),
        ))
        .insert_resource(Brush {
            tool: Tool::Dig,
            radius: 4.0,
            strength: 1,
        })
        .insert_resource(HoverCell::default())
        .insert_resource(StrokeLastCell::default())
        .insert_resource(StackDirty(false))
        .insert_resource(ClearColor(Color::srgb(0.05, 0.04, 0.07)))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                toggle_debug,
                debug_inputs,
                update_hover,
                apply_brush_input,
                regen_stack_mesh,
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
    stack: Res<PaperStack>,
) {
    // Camera — sync_camera sets the actual transform every frame.
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

    // Three directional lights — sync_lights drives their actual
    // orientation/intensity from `LightControl` every frame.
    for i in 0..3 {
        commands.spawn((
            DirectionalLight {
                color: Color::WHITE,
                illuminance: 0.0,
                shadows_enabled: false,
                shadow_depth_bias: 0.005,
                shadow_normal_bias: 0.1,
                ..default()
            },
            CascadeShadowConfigBuilder {
                num_cascades: 2,
                first_cascade_far_bound: 6.0,
                maximum_distance: 40.0,
                ..default()
            }
            .build(),
            DirLightIndex(i),
        ));
    }

    // Single material — vertex colors carry per-layer color, so
    // base_color stays white and the StandardMaterial multiplies it
    // in. High roughness / no metalness for a paper-ish feel.
    let stack_mat = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        perceptual_roughness: 0.92,
        metallic: 0.0,
        reflectance: 0.04,
        ..default()
    });

    let mesh_handle = meshes.add(build_stack_mesh(&stack));
    commands.spawn((
        Mesh3d(mesh_handle),
        MeshMaterial3d(stack_mat),
        Transform::IDENTITY,
        StackMesh,
    ));
}

// ─── Mesh generation ──────────────────────────────────────────────

fn build_stack_mesh(stack: &PaperStack) -> Mesh {
    // Top quads for every cell + side strips wherever a neighbor is
    // shorter. Per-vertex color carries the palette entry; out-of-grid
    // neighbors are treated as height = 0 so the outer faces of the
    // stack show the full strata.
    //
    // Estimate capacity to avoid reallocation churn. Worst case is
    // ~5*W*H quads (top + 4 sides), each with `max_layers` vertical
    // sub-strips. Initial state has no internal sides at all.
    let n_cells = (stack.width as usize) * (stack.height as usize);
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n_cells * 8);
    let mut normals:   Vec<[f32; 3]> = Vec::with_capacity(n_cells * 8);
    let mut colors:    Vec<[f32; 4]> = Vec::with_capacity(n_cells * 8);
    let mut indices:   Vec<u32>      = Vec::with_capacity(n_cells * 12);

    let half_cell = CELL_SIZE * 0.5;
    let pt = stack.paper_thickness;
    let floor_color = color_to_rgba(Color::srgb(0.10, 0.09, 0.07));

    for y in 0..stack.height {
        for x in 0..stack.width {
            let h = stack.get(x, y);
            let center = stack.cell_world_xy(x, y);
            let z_top = stack.cell_top_z(h);

            // Top face. Wound CCW from above so the +Z normal faces
            // the camera.
            let top_color = if h > 0 {
                color_to_rgba(stack.palette[(h - 1) as usize])
            } else {
                floor_color
            };
            push_quad(
                &mut positions,
                &mut normals,
                &mut colors,
                &mut indices,
                [
                    [center.x - half_cell, center.y - half_cell, z_top],
                    [center.x + half_cell, center.y - half_cell, z_top],
                    [center.x + half_cell, center.y + half_cell, z_top],
                    [center.x - half_cell, center.y + half_cell, z_top],
                ],
                [0.0, 0.0, 1.0],
                top_color,
            );

            if h == 0 {
                continue;
            }

            // Side faces. For each cardinal neighbor, if my height
            // exceeds neighbor's, emit one quad per exposed layer
            // colored by that layer's palette entry.
            for &(dx, dy) in &[(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                let neighbor_h = if nx < 0 || ny < 0
                    || nx >= stack.width as i32 || ny >= stack.height as i32
                {
                    0
                } else {
                    stack.get(nx as u16, ny as u16)
                };
                if neighbor_h >= h {
                    continue;
                }

                let (a_off, b_off, normal) = match (dx, dy) {
                    (1, 0)  => ([ half_cell, -half_cell], [ half_cell,  half_cell], [ 1.0, 0.0, 0.0]),
                    (-1, 0) => ([-half_cell,  half_cell], [-half_cell, -half_cell], [-1.0, 0.0, 0.0]),
                    (0, 1)  => ([ half_cell,  half_cell], [-half_cell,  half_cell], [ 0.0, 1.0, 0.0]),
                    (0, -1) => ([-half_cell, -half_cell], [ half_cell, -half_cell], [ 0.0, -1.0, 0.0]),
                    _ => unreachable!(),
                };

                for layer in neighbor_h..h {
                    // Render only the sheet body. The gap above it
                    // (z0 + pt .. z0 + pt + gap) is left empty so the
                    // strata read as discrete sheets.
                    let z0 = stack.sheet_bottom_z(layer);
                    let z1 = z0 + pt;
                    let lc = color_to_rgba(stack.palette[layer as usize]);
                    push_quad(
                        &mut positions,
                        &mut normals,
                        &mut colors,
                        &mut indices,
                        [
                            [center.x + a_off[0], center.y + a_off[1], z0],
                            [center.x + b_off[0], center.y + b_off[1], z0],
                            [center.x + b_off[0], center.y + b_off[1], z1],
                            [center.x + a_off[0], center.y + a_off[1], z1],
                        ],
                        normal,
                        lc,
                    );
                }
            }
        }
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL,   normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR,    colors);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

fn push_quad(
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
    verts: [[f32; 3]; 4],
    normal: [f32; 3],
    color: [f32; 4],
) {
    let v0 = positions.len() as u32;
    for v in verts {
        positions.push(v);
        normals.push(normal);
        colors.push(color);
    }
    indices.extend_from_slice(&[v0, v0 + 1, v0 + 2, v0, v0 + 2, v0 + 3]);
}

fn color_to_rgba(c: Color) -> [f32; 4] {
    let lc = c.to_linear();
    [lc.red, lc.green, lc.blue, lc.alpha]
}

// ─── Input ────────────────────────────────────────────────────────

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
    mut brush: ResMut<Brush>,
    mut stack: ResMut<PaperStack>,
    mut dirty: ResMut<StackDirty>,
    mut light: ResMut<LightControl>,
) {
    wheel.clear();
    let dt = time.delta_secs();

    if keys.just_pressed(KeyCode::KeyR) {
        camera.pan = Vec2::ZERO;
        camera.zoom = 1.0;
        camera.tilt = 0.55;
        light.lights = default_lights();
        light.ambient = 0.32;
        stack.reset_full();
        dirty.0 = true;
    }

    if keys.just_pressed(KeyCode::Digit1) { brush.tool = Tool::Dig; }
    if keys.just_pressed(KeyCode::Digit2) { brush.tool = Tool::Extrude; }

    // Pan.
    let pan_speed = 5.0 / camera.zoom;
    let mut delta = Vec2::ZERO;
    if keys.pressed(KeyCode::KeyW) || keys.pressed(KeyCode::ArrowUp)    { delta.y += 1.0; }
    if keys.pressed(KeyCode::KeyS) || keys.pressed(KeyCode::ArrowDown)  { delta.y -= 1.0; }
    if keys.pressed(KeyCode::KeyA) || keys.pressed(KeyCode::ArrowLeft)  { delta.x -= 1.0; }
    if keys.pressed(KeyCode::KeyD) || keys.pressed(KeyCode::ArrowRight) { delta.x += 1.0; }
    if delta != Vec2::ZERO {
        camera.pan += delta.normalize() * pan_speed * dt;
    }

    let zoom_rate = 1.6_f32.powf(dt);
    if keys.pressed(KeyCode::KeyE) { camera.zoom = (camera.zoom * zoom_rate).min(8.0); }
    if keys.pressed(KeyCode::KeyQ) { camera.zoom = (camera.zoom / zoom_rate).max(0.2); }

    let tilt_speed = 1.2;
    let tilt_max = 1.4;
    if keys.pressed(KeyCode::BracketLeft)  { camera.tilt = (camera.tilt - tilt_speed * dt).max(0.0); }
    if keys.pressed(KeyCode::BracketRight) { camera.tilt = (camera.tilt + tilt_speed * dt).min(tilt_max); }
    if keys.just_pressed(KeyCode::KeyT) {
        camera.tilt = if camera.tilt < 0.05 { 0.6 } else { 0.0 };
    }
}

// ─── Picking ──────────────────────────────────────────────────────

fn update_hover(
    windows: Query<&Window>,
    cameras: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    stack: Res<PaperStack>,
    mut hover: ResMut<HoverCell>,
    mut contexts: EguiContexts,
) {
    // egui owns the cursor when it's over a panel — don't pick the
    // stack underneath it.
    if let Ok(ctx) = contexts.ctx_mut() {
        if ctx.is_pointer_over_area() {
            hover.0 = None;
            return;
        }
    }

    let Ok(window) = windows.single() else { hover.0 = None; return; };
    let Some(cursor) = window.cursor_position() else { hover.0 = None; return; };
    let Ok((camera, cam_tf)) = cameras.single() else { hover.0 = None; return; };
    let Ok(ray) = camera.viewport_to_world(cam_tf, cursor) else {
        hover.0 = None;
        return;
    };

    hover.0 = raymarch_stack(ray, &stack);
}

fn raymarch_stack(ray: Ray3d, stack: &PaperStack) -> Option<(u16, u16)> {
    // Start the march at the first point where the ray could possibly
    // intersect the stack — i.e., where its z drops to the maximum
    // possible top. Saves hundreds of useless steps coming down from
    // the camera position.
    let max_top = stack.cell_top_z(stack.max_layers);
    let dir = ray.direction.as_vec3();
    let t_start = if dir.z < -1e-4 {
        ((max_top - ray.origin.z) / dir.z).max(0.0)
    } else {
        0.0
    };

    let dt = CELL_SIZE * 0.5;
    let max_steps = 2_000;
    let mut t = t_start;
    for _ in 0..max_steps {
        let p = ray.origin + dir * t;
        // Ray dropped below the floor — no chance of a hit anymore.
        if p.z < -stack.paper_thickness {
            return None;
        }
        if let Some((cx, cy)) = stack.world_to_cell(p.x, p.y) {
            let top_z = stack.cell_top_z(stack.get(cx, cy));
            if p.z <= top_z {
                return Some((cx, cy));
            }
        }
        t += dt;
    }
    None
}

// ─── Brush ────────────────────────────────────────────────────────

fn apply_brush_input(
    mouse: Res<ButtonInput<MouseButton>>,
    hover: Res<HoverCell>,
    brush: Res<Brush>,
    mut last: ResMut<StrokeLastCell>,
    mut stack: ResMut<PaperStack>,
    mut dirty: ResMut<StackDirty>,
) {
    if !mouse.pressed(MouseButton::Left) {
        last.0 = None;
        return;
    }

    let Some(cell) = hover.0 else { return; };
    // Apply once per cell during a stroke — the brush hits each cell
    // it passes through, but only once even if the cursor lingers.
    if last.0 == Some(cell) {
        return;
    }
    last.0 = Some(cell);

    let r = brush.radius.ceil() as i32;
    let r2 = brush.radius * brush.radius;
    let (cx, cy) = (cell.0 as i32, cell.1 as i32);
    let mut changed = false;

    for dy in -r..=r {
        for dx in -r..=r {
            let d2 = (dx * dx + dy * dy) as f32;
            if d2 > r2 { continue; }
            let x = cx + dx;
            let y = cy + dy;
            if x < 0 || y < 0 || x >= stack.width as i32 || y >= stack.height as i32 {
                continue;
            }
            let cur = stack.get(x as u16, y as u16);
            let new = match brush.tool {
                Tool::Dig     => cur.saturating_sub(brush.strength),
                Tool::Extrude => (cur + brush.strength).min(stack.max_layers),
            };
            if new != cur {
                stack.set(x as u16, y as u16, new);
                changed = true;
            }
        }
    }
    if changed {
        dirty.0 = true;
    }
}

fn regen_stack_mesh(
    stack: Res<PaperStack>,
    mut dirty: ResMut<StackDirty>,
    mut meshes: ResMut<Assets<Mesh>>,
    q: Query<&Mesh3d, With<StackMesh>>,
) {
    if !dirty.0 {
        return;
    }
    let Ok(handle) = q.single() else { return };
    if let Some(mesh) = meshes.get_mut(&handle.0) {
        *mesh = build_stack_mesh(&stack);
    }
    dirty.0 = false;
}

// ─── Sync (resources → entities) ──────────────────────────────────

fn sync_lights(
    light_ctrl: Res<LightControl>,
    mut q: Query<(&mut DirectionalLight, &mut Transform, &DirLightIndex)>,
) {
    for (mut dl, mut tf, idx) in q.iter_mut() {
        let l = light_ctrl.lights[idx.0];
        // Shader-space y was screen-down; world y is up.
        let dir_to_light = Vec3::new(l.dir.x, -l.dir.y, l.dir.z)
            .try_normalize()
            .unwrap_or(Vec3::Z);
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
    let target = Vec3::new(camera_ctrl.pan.x, camera_ctrl.pan.y, 0.0);
    let base_distance = 14.0;
    let distance = base_distance / camera_ctrl.zoom;
    let offset = Quat::from_rotation_x(-camera_ctrl.tilt) * Vec3::new(0.0, 0.0, distance);
    *tf = Transform::from_translation(target + offset).looking_at(target, Vec3::Y);
}

// ─── egui panel ───────────────────────────────────────────────────

fn debug_panel(
    mut contexts: EguiContexts,
    debug: Res<DebugMode>,
    mut light: ResMut<LightControl>,
    mut camera: ResMut<CameraControl>,
    mut brush: ResMut<Brush>,
    mut stack: ResMut<PaperStack>,
    mut dirty: ResMut<StackDirty>,
) -> Result {
    if !debug.0 {
        return Ok(());
    }
    let ctx = contexts.ctx_mut()?;

    egui::Window::new("Paper-stack editor")
        .default_pos([12.0, 12.0])
        .default_width(300.0)
        .show(ctx, |ui| {
            ui.collapsing("Tool", |ui| {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut brush.tool, Tool::Dig,     "Dig (1)");
                    ui.selectable_value(&mut brush.tool, Tool::Extrude, "Extrude (2)");
                });
                ui.add(egui::Slider::new(&mut brush.radius, 0.5..=20.0).text("radius (cells)"));
                let mut s = brush.strength as i32;
                if ui.add(egui::Slider::new(&mut s, 1..=8).text("strength (layers)")).changed() {
                    brush.strength = s.max(1) as u16;
                }
                if ui.button("Reset stack").clicked() {
                    stack.reset_full();
                    dirty.0 = true;
                }
            });

            ui.collapsing("Layer geometry", |ui| {
                if ui
                    .add(
                        egui::Slider::new(&mut stack.paper_thickness, 0.002..=0.06)
                            .text("paper thickness"),
                    )
                    .changed()
                {
                    dirty.0 = true;
                }
                if ui
                    .add(
                        egui::Slider::new(&mut stack.gap_thickness, 0.0..=0.03)
                            .text("gap thickness"),
                    )
                    .changed()
                {
                    dirty.0 = true;
                }
                ui.label(format!(
                    "stack height: {:.3}",
                    stack.cell_top_z(stack.max_layers)
                ));
            });

            ui.collapsing("Palette", |ui| {
                let n = stack.palette.len();
                egui::ScrollArea::vertical().max_height(220.0).show(ui, |ui| {
                    for i in 0..n {
                        ui.horizontal(|ui| {
                            ui.label(format!("{i:>2}"));
                            let lc = stack.palette[i].to_linear();
                            let mut rgb = [lc.red, lc.green, lc.blue];
                            if ui.color_edit_button_rgb(&mut rgb).changed() {
                                stack.palette[i] = Color::linear_rgb(rgb[0], rgb[1], rgb[2]);
                                dirty.0 = true;
                            }
                        });
                    }
                });
                if ui.button("Reset palette").clicked() {
                    stack.palette = default_palette(stack.max_layers);
                    dirty.0 = true;
                }
            });

            ui.collapsing("Lights", |ui| {
                ui.add(egui::Slider::new(&mut light.ambient, 0.0..=1.0).text("ambient"));
                ui.separator();
                for i in 0..3 {
                    egui::CollapsingHeader::new(format!("Light {}", i + 1))
                        .default_open(light.lights[i].intensity > 0.01)
                        .show(ui, |ui| {
                            let l = &mut light.lights[i];
                            let mut on = l.intensity > 0.01;
                            if ui.checkbox(&mut on, "enabled").changed() {
                                l.intensity = if on { 1.0_f32.max(l.intensity) } else { 0.0 };
                            }
                            ui.add(egui::Slider::new(&mut l.intensity, 0.0..=2.0).text("intensity"));
                            ui.add(egui::Slider::new(&mut l.dir.x, -2.0..=2.0).text("dir x"));
                            ui.add(egui::Slider::new(&mut l.dir.y, -2.0..=2.0).text("dir y"));
                            ui.add(egui::Slider::new(&mut l.dir.z, 0.0..=2.0).text("dir z (elev)"));
                            let cur = l.color_name();
                            egui::ComboBox::from_label("color")
                                .selected_text(cur)
                                .show_ui(ui, |ui| {
                                    for (idx, (name, _)) in COLOR_PRESETS.iter().enumerate() {
                                        ui.selectable_value(&mut l.color_idx, idx, *name);
                                    }
                                });
                        });
                }
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
            ui.label("Tab to hide. Drag with left mouse to carve.");
        });
    Ok(())
}
